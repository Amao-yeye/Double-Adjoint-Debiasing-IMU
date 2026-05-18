#!/usr/bin/env python3
"""Summarize PLATO corldata Debias_IMU network sweep logs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


LINE_RE = re.compile(
    r"stage=(?P<stage>Gyro|Acc)\s+epoch=(?P<epoch>\d+)\s+"
    r"train_loss=(?P<train>[-+0-9.eE]+)\s+val_loss=(?P<val>[-+0-9.eE]+)"
)
DRIVER_RE = re.compile(r"^===\s+(?P<name>[^:]+):\s+(?P<network>[^=]+?)\s+===$")


@dataclass
class ValPoint:
    stage: str
    epoch: int
    train_loss: float
    val_loss: float


def parse_log(path: Path) -> list[ValPoint]:
    points: list[ValPoint] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        points.append(
            ValPoint(
                stage=match.group("stage"),
                epoch=int(match.group("epoch")),
                train_loss=float(match.group("train")),
                val_loss=float(match.group("val")),
            )
        )
    return points


def infer_case_name(path: Path) -> str:
    name = path.stem
    prefix = "corldata_gyro_sweep_"
    suffix = "_e601_bal_lw16_b1000"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def summarize(logs: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for log in sorted(logs):
        points = parse_log(log)
        case = infer_case_name(log)
        stages = sorted({point.stage for point in points})
        for stage in stages:
            stage_points = [point for point in points if point.stage == stage]
            best = min(stage_points, key=lambda point: point.val_loss)
            last = stage_points[-1]
            rows.append(
                {
                    "case": case,
                    "stage": stage,
                    "best_epoch": best.epoch,
                    "best_val_loss": best.val_loss,
                    "best_train_loss": best.train_loss,
                    "last_epoch": last.epoch,
                    "last_val_loss": last.val_loss,
                    "num_val_points": len(stage_points),
                    "completed_600": last.epoch >= 600,
                    "log": str(log),
                }
            )
        if not points:
            rows.append(
                {
                    "case": case,
                    "stage": "none",
                    "best_epoch": "",
                    "best_val_loss": "",
                    "best_train_loss": "",
                    "last_epoch": "",
                    "last_val_loss": "",
                    "num_val_points": 0,
                    "completed_600": False,
                    "log": str(log),
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "stage",
        "best_epoch",
        "best_val_loss",
        "best_train_loss",
        "last_epoch",
        "last_val_loss",
        "num_val_points",
        "completed_600",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# corldata Bias Dynamics Network Sweep",
        "",
        "| case | stage | best epoch | best val loss | last epoch | last val loss | completed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best_val = row["best_val_loss"]
        last_val = row["last_val_loss"]
        best_val_s = f"{best_val:.6g}" if isinstance(best_val, float) else str(best_val)
        last_val_s = f"{last_val:.6g}" if isinstance(last_val, float) else str(last_val)
        lines.append(
            f"| {row['case']} | {row['stage']} | {row['best_epoch']} | {best_val_s} | "
            f"{row['last_epoch']} | {last_val_s} | {row['completed_600']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-glob",
        default="results/debias_imu/logs/corldata_gyro_sweep_*_e601_bal_lw16_b1000.log",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/debias_imu/corldata_network_sweep_summary"),
    )
    args = parser.parse_args()

    logs = [
        path
        for path in Path(".").glob(args.log_glob)
        if path.is_file() and "_driver_" not in path.name
    ]
    rows = summarize(logs)
    write_csv(rows, args.output_prefix.with_suffix(".csv"))
    write_markdown(rows, args.output_prefix.with_suffix(".md"))
    print(f"wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
