import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from dataset import EUROCDataset
from learning import run_piecewise_gt_bias_baseline
from evalutation import AOE, ATE
from dataset import pload


def method_name_for_update_hz(update_hz: float) -> str:
    suffix = f"{int(update_hz)}hz" if float(update_hz).is_integer() else f"{update_hz:g}hz"
    return f"Euroc_piecewise_gt_bias_{suffix}"


def dir_name_for_update_hz(update_hz: float) -> str:
    suffix = f"{int(update_hz)}hz" if float(update_hz).is_integer() else f"{update_hz:g}hz"
    return f"piecewise_gt_bias_{suffix}"


def evaluate_piecewise_baseline(outputdir: str, update_hz: float, seqs: list[str]):
    base = os.path.join(outputdir, dir_name_for_update_hz(update_hz))
    vals = []
    for seq in seqs:
        d = pload(base, seq + ".p")
        X_gt = d["X_gt"]
        X_piecewise = d["X_piecewise"]
        aoe = AOE(X_gt[:, :3, :3], X_piecewise[:, :3, :3]).item()
        ate_vel = ATE(X_gt[:, :3, 3], X_piecewise[:, :3, 3]).item()
        ate_pos = ATE(X_gt[:, :3, 4], X_piecewise[:, :3, 4]).item()
        vals.append((seq, aoe, ate_vel, ate_pos))
    n = len(vals)
    return {
        "done": f"{n}/5",
        "aoe": sum(v[1] for v in vals) / n,
        "ate_vel": sum(v[2] for v in vals) / n,
        "ate_pos": sum(v[3] for v in vals) / n,
        "per_seq": vals,
    }


def update_evaluation_comparison(project_root: str, method: str, summary: dict):
    path = os.path.join(project_root, "results", "evaluation_comparison.txt")
    with open(path, "r") as f:
        lines = f.read().splitlines()

    rows = lines[2:]
    filtered_rows = [row for row in rows if row.split()[0] != method]
    parsed_rows = []
    for row in filtered_rows:
        parts = row.split()
        parsed_rows.append((parts[0], parts[1], float(parts[2]), float(parts[3]), float(parts[4])))
    parsed_rows.append((method, summary["done"], summary["aoe"], summary["ate_vel"], summary["ate_pos"]))
    parsed_rows.sort(key=lambda row: row[4])

    method_width = max(len("method"), max(len(row[0]) for row in parsed_rows))
    header = f"{'method':{method_width}s} {'done':>6s} {'AOE':>10s} {'ATE vel':>10s} {'ATE pos':>12s}"
    divider = "-" * len(header)
    formatted_rows = [
        f"{name:{method_width}s} {done:>6s} {aoe:10.3f} {ate_vel:10.3f} {ate_pos:12.3f}"
        for name, done, aoe, ate_vel, ate_pos in parsed_rows
    ]

    with open(path, "w") as f:
        f.write("\n".join([header, divider] + formatted_rows) + "\n")


def main():
    parser = argparse.ArgumentParser("Run piecewise gt-bias baseline on Euroc test sequences")
    parser.add_argument("--outputdir", type=str, default="results/baselines/Euroc_piecewise_gt_bias")
    parser.add_argument("--update_hz", type=float, default=10.0)
    parser.add_argument("--test_device", type=str, default="cpu")
    parser.add_argument("--test_recompute", action="store_true")
    args = parser.parse_args()

    test_device = args.test_device
    if test_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"test_device={test_device} was requested, but CUDA is not available.")

    test_seqs = [
        'MH_02_easy',
        'MH_04_difficult',
        'V2_02_medium',
        'V1_03_difficult',
        'V1_01_easy'
    ]
    dataset_parameters = {
        'dataset_name': 'EUROC',
        'data_dir': 'data/EUROC',
        'train_seqs': [],
        'val_seqs': [],
        'test_seqs': test_seqs,
        'dt': 0.005,
        'percent_for_val': 0.2,
        'loss_window': 16,
        'batch_size': 1000,
        'sg_window_bw': 1,
        'sg_order_bw': 0,
        'sg_window_ba': 1,
        'sg_order_ba': 0,
        'test_gt_source': 'state_groundtruth_estimate0',
    }
    dataset_test = EUROCDataset(
        **dataset_parameters,
        data_cache_dir='data/EUROC_cache_test_gt',
        mode='test'
    )

    run_piecewise_gt_bias_baseline(
        dataset_test,
        output_dir=args.outputdir,
        update_hz=args.update_hz,
        device=test_device,
        recompute=args.test_recompute,
    )

    summary = evaluate_piecewise_baseline(args.outputdir, args.update_hz, test_seqs)
    method = method_name_for_update_hz(args.update_hz)
    update_evaluation_comparison(PROJECT_ROOT, method, summary)
    print(
        f"{method}: AOE {summary['aoe']:.3f}, "
        f"ATE vel {summary['ate_vel']:.3f}, "
        f"ATE pos {summary['ate_pos']:.3f}"
    )


if __name__ == "__main__":
    main()
