#!/usr/bin/env python3
"""Create a EuRoC cache variant with train/val bias labels set to train-global constants.

Debias_IMU short-window training uses `w_bias[0]` and `a_bias[0]` as the ODE
initial bias state.  This utility keeps poses and IMU untouched, but replaces
train/validation bias-label arrays with one train-only global statistic so the
short-window initial condition matches a deployment-style global prior.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import torch


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_SEQS = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]
DEFAULT_TEST_SEQS = [
    "MH_02_easy",
    "MH_04_difficult",
    "V2_02_medium",
    "V1_03_difficult",
    "V1_01_easy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "EUROC_test_gt",
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "EUROC_train_global_median_y0",
    )
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--exclude-train-seq", nargs="*", default=[])
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--stat", choices=["mean", "median"], default="median")
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, payload) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def compute_train_bias(source_cache: Path, train_seqs: list[str], stat: str) -> tuple[torch.Tensor, torch.Tensor]:
    bw_values = []
    ba_values = []
    for seq in train_seqs:
        data = load_pickle(source_cache / f"{seq}.p")
        bw_values.append(data["w_bias"].float())
        ba_values.append(data["a_bias"].float())
    bw = torch.cat(bw_values, dim=0)
    ba = torch.cat(ba_values, dim=0)
    if stat == "mean":
        return bw.mean(dim=0), ba.mean(dim=0)
    return bw.median(dim=0).values, ba.median(dim=0).values


def main() -> None:
    args = parse_args()
    train_seqs = [seq for seq in args.train_seqs if seq not in set(args.exclude_train_seq)]
    all_seqs = []
    for seq in [*train_seqs, *args.test_seqs]:
        if seq not in all_seqs:
            all_seqs.append(seq)

    bw_const, ba_const = compute_train_bias(args.source_cache, train_seqs, args.stat)
    args.output_cache.mkdir(parents=True, exist_ok=True)

    for seq in all_seqs:
        src_data = args.source_cache / f"{seq}.p"
        src_gt = args.source_cache / f"{seq}_gt.p"
        dst_data = args.output_cache / f"{seq}.p"
        dst_gt = args.output_cache / f"{seq}_gt.p"
        data = load_pickle(src_data)
        if seq in train_seqs:
            data["w_bias"] = bw_const.to(dtype=data["w_bias"].dtype).expand_as(data["w_bias"]).clone()
            data["a_bias"] = ba_const.to(dtype=data["a_bias"].dtype).expand_as(data["a_bias"]).clone()
        write_pickle(dst_data, data)
        shutil.copy2(src_gt, dst_gt)

    summary = {
        "source_cache": str(args.source_cache),
        "output_cache": str(args.output_cache),
        "train_seqs": train_seqs,
        "test_seqs": list(args.test_seqs),
        "stat": args.stat,
        "bw_const": [float(x) for x in bw_const.tolist()],
        "ba_const": [float(x) for x in ba_const.tolist()],
        "note": "Only train/validation sequence bias arrays are replaced; IMU and GT poses are unchanged.",
    }
    write_pickle(args.output_cache / "train_global_bias_summary.p", summary)
    (args.output_cache / "train_global_bias_summary.json").write_text(
        __import__("json").dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
