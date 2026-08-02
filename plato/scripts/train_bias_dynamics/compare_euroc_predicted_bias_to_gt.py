#!/usr/bin/env python3
"""Compare EuRoC predicted bias trajectories against GT-differenced bias labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_ROOT = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DEBIAS_ROOT))
sys.path.insert(0, str(DEBIAS_ROOT / "BiasDy"))

from bias_trend_utils import as_numpy, collect_bias_trend_metrics, write_bias_trend_outputs  # noqa: E402
from dataset import EUROCDataset  # noqa: E402
from learning import pload  # noqa: E402


DEFAULT_TEST_SEQS = [
    "MH_02_easy",
    "MH_04_difficult",
    "V2_02_medium",
    "V1_03_difficult",
    "V1_01_easy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEBIAS_ROOT / "data" / "EUROC")
    parser.add_argument(
        "--test-cache-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "EUROC_test_gt",
    )
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--loss-window", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--sg-window-bw", type=int, default=101)
    parser.add_argument("--sg-order-bw", type=int, default=3)
    parser.add_argument("--sg-window-ba", type=int, default=31)
    parser.add_argument("--sg-order-ba", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-plot-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.result_dir / "bias_trend_gt_compare")
    dataset = EUROCDataset(
        dataset_name="EUROC",
        data_dir=str(args.data_root),
        train_seqs=[],
        val_seqs=[],
        test_seqs=list(args.test_seqs),
        mode="test",
        percent_for_val=0.2,
        batch_size=1,
        loss_window=args.loss_window,
        dt=args.dt,
        sg_window_bw=args.sg_window_bw,
        sg_order_bw=args.sg_order_bw,
        sg_window_ba=args.sg_window_ba,
        sg_order_ba=args.sg_order_ba,
        trainval_gt_root=None,
        test_gt_root=None,
        trainval_gt_source="state_groundtruth_estimate0",
        test_gt_source="state_groundtruth_estimate0",
        trainval_supervision_timing="imu",
        data_cache_dir=str(args.test_cache_dir),
        recompute=False,
    )

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "result_dir": str(args.result_dir),
        "test_seqs": list(args.test_seqs),
        "sequences": {},
    }
    for seq in args.test_seqs:
        result_path = args.result_dir / "Sequnces-results" / seq / "FullTra_result.p"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        result = pload(str(result_path))
        sol_pred = as_numpy(result["sol_pred"])
        t = as_numpy(result["t_gt"]).reshape(-1)
        bw_pred = sol_pred[:, 4:7]
        ba_pred = sol_pred[:, 13:16]
        seq_index = dataset.find_sequence_index(seq)
        bw_gt = as_numpy(dataset.get_w_bias(seq_index)[0])
        ba_gt = as_numpy(dataset.get_a_bias(seq_index)[0])
        n = min(t.shape[0], bw_pred.shape[0], ba_pred.shape[0], bw_gt.shape[0], ba_gt.shape[0])
        t = t[:n]
        bw_pred = bw_pred[:n]
        ba_pred = ba_pred[:n]
        bw_gt = bw_gt[:n]
        ba_gt = ba_gt[:n]

        seq_rows, seq_metrics = collect_bias_trend_metrics(
            seq,
            t,
            bw_pred,
            ba_pred,
            bw_gt,
            ba_gt,
            output_dir,
            reference_label="GT-differenced",
            max_plot_s=args.max_plot_s,
        )
        rows.extend(seq_rows)
        summary["sequences"][seq] = seq_metrics

    write_bias_trend_outputs(output_dir, rows, summary)


if __name__ == "__main__":
    main()
