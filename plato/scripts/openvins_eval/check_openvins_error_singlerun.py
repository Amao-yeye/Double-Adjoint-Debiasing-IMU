#!/usr/bin/env python3
"""Cross-check OpenVINS result summaries with ov_eval error_singlerun."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


PLATO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=PLATO_ROOT / "external/OpenVINS/src/open_vins/ov_data/euroc_mav")
    parser.add_argument("--ros-setup", type=Path, default=Path("/opt/ros/noetic/setup.bash"))
    parser.add_argument("--openvins-setup", type=Path, default=PLATO_ROOT / "external/OpenVINS/devel/setup.bash")
    parser.add_argument("--alignment", default="posyaw")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def run_error_singlerun(args: argparse.Namespace, gt: Path, est: Path) -> tuple[float, float, str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/plato_mpl"
    env["ROS_HOME"] = "/tmp/plato_ros_home"
    command = (
        f"source '{args.ros_setup}' && "
        f"source '{args.openvins_setup}' && "
        f"rosrun ov_eval error_singlerun {args.alignment} '{gt}' '{est}'"
    )
    proc = subprocess.run(["bash", "-lc", command], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    match = re.search(r"rmse_ori = ([0-9.]+) \| rmse_pos = ([0-9.]+)", proc.stdout)
    if not match:
        raise RuntimeError(f"error_singlerun failed for {est}:\n{proc.stdout}")
    return float(match.group(1)), float(match.group(2)), proc.stdout


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric_status") != "ok":
                continue
            out_dir = Path(row["out_dir"])
            gt = args.gt_root / f"{row['sequence']}.txt"
            odom = out_dir / "odomimu.txt"
            official_ori, official_pos, _ = run_error_singlerun(args, gt, odom)
            py_ori = float(row["ate_rmse_ori_deg"])
            py_pos = float(row["ate_rmse_pos_m"])
            rows.append(
                {
                    "method": row["method"],
                    "sequence": row["sequence"],
                    "python_pos_rmse_m": f"{py_pos:.6f}",
                    "official_pos_rmse_m": f"{official_pos:.6f}",
                    "abs_diff_pos_m": f"{abs(py_pos - official_pos):.6f}",
                    "python_rot_rmse_deg": f"{py_ori:.6f}",
                    "official_rot_rmse_deg": f"{official_ori:.6f}",
                    "abs_diff_rot_deg": f"{abs(py_ori - official_ori):.6f}",
                }
            )

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OpenVINS error_singlerun Cross-Check",
        "",
        f"Summary source: `{args.summary}`",
        f"Alignment: `{args.alignment}`",
        "",
        "| method | sequence | Python pos | Official pos | pos diff | Python rot | Official rot | rot diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['sequence']} | {row['python_pos_rmse_m']} | "
            f"{row['official_pos_rmse_m']} | {row['abs_diff_pos_m']} | "
            f"{row['python_rot_rmse_deg']} | {row['official_rot_rmse_deg']} | {row['abs_diff_rot_deg']} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_csv)
    print(args.output_md)


if __name__ == "__main__":
    main()
