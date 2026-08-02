#!/usr/bin/env python3
"""Create EuRoC test OpenVINS SE23-MSE vs NLL summary tables."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


SEQUENCES = [
    "MH_02_easy",
    "MH_04_difficult",
    "V1_01_easy",
    "V1_03_difficult",
    "V2_02_medium",
]
METHODS = ["gt_mse", "gt_nll", "vo_mse", "vo_nll"]
LABEL = {
    "gt_mse": "GT SE23-MSE",
    "gt_nll": "GT NLL",
    "vo_mse": "VO SE23-MSE",
    "vo_nll": "VO NLL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results/summary.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/amao/workspace/formalPLATO/Results"),
    )
    parser.add_argument("--stem", default="euroc_test_openvins_se23mse_vs_nll")
    return parser.parse_args()


def f6(value: str | float) -> str:
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    rows = []
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in METHODS and row["metric_status"] == "ok":
                rows.append(row)
    by_key = {(row["sequence"], row["method"]): row for row in rows}
    missing = [(seq, method) for seq in SEQUENCES for method in METHODS if (seq, method) not in by_key]
    if missing:
        raise RuntimeError(f"Missing OpenVINS rows: {missing}")

    method_summary = []
    for method in METHODS:
        vals = [by_key[(seq, method)] for seq in SEQUENCES]
        method_summary.append(
            {
                "method": LABEL[method],
                "sequences": len(vals),
                "mean_pos_rmse_m": statistics.fmean(float(v["ate_rmse_pos_m"]) for v in vals),
                "mean_rot_rmse_deg": statistics.fmean(float(v["ate_rmse_ori_deg"]) for v in vals),
                "mean_pos_mean_m": statistics.fmean(float(v["ate_mean_pos_m"]) for v in vals),
                "mean_rot_mean_deg": statistics.fmean(float(v["ate_mean_ori_deg"]) for v in vals),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_root / f"{args.stem}.csv"
    out_md = args.output_root / f"{args.stem}.md"

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence",
                "supervision",
                "se23mse_pos_rmse_m",
                "se23mse_rot_rmse_deg",
                "nll_pos_rmse_m",
                "nll_rot_rmse_deg",
            ]
        )
        for supervision, prefix in [("GT", "gt"), ("VO", "vo")]:
            for seq in SEQUENCES:
                mse = by_key[(seq, f"{prefix}_mse")]
                nll = by_key[(seq, f"{prefix}_nll")]
                writer.writerow(
                    [
                        seq,
                        supervision,
                        f6(mse["ate_rmse_pos_m"]),
                        f6(mse["ate_rmse_ori_deg"]),
                        f6(nll["ate_rmse_pos_m"]),
                        f6(nll["ate_rmse_ori_deg"]),
                    ]
                )

    lines = [
        "# EuRoC Test OpenVINS: SE23-MSE vs NLL",
        "",
        f"Source: `{args.input}`.",
        "Metric: posyaw-aligned ATE; position RMSE in meters, rotation RMSE in degrees; no scale alignment.",
        "",
        "## Method Average",
        "",
        "| method | sequences | mean pos RMSE (m) | mean rot RMSE (deg) | mean pos mean (m) | mean rot mean (deg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        lines.append(
            f"| {row['method']} | {row['sequences']} | "
            f"{row['mean_pos_rmse_m']:.6f} | {row['mean_rot_rmse_deg']:.6f} | "
            f"{row['mean_pos_mean_m']:.6f} | {row['mean_rot_mean_deg']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Sequence Comparison",
            "",
            "| supervision | sequence | SE23-MSE pos RMSE (m) | SE23-MSE rot RMSE (deg) | NLL pos RMSE (m) | NLL rot RMSE (deg) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for supervision, prefix in [("GT", "gt"), ("VO", "vo")]:
        for seq in SEQUENCES:
            mse = by_key[(seq, f"{prefix}_mse")]
            nll = by_key[(seq, f"{prefix}_nll")]
            lines.append(
                f"| {supervision} | {seq} | {f6(mse['ate_rmse_pos_m'])} | "
                f"{f6(mse['ate_rmse_ori_deg'])} | {f6(nll['ate_rmse_pos_m'])} | "
                f"{f6(nll['ate_rmse_ori_deg'])} |"
            )
    lines.extend(["", f"CSV: `{out_csv}`"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_md)
    print(out_csv)


if __name__ == "__main__":
    main()
