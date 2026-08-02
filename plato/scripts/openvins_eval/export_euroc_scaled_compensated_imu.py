#!/usr/bin/env python3
"""Export EuRoC corrected IMU CSVs with scaled learned-bias compensation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = PLATO_ROOT / "external/Debias_IMU/rss_debiasing_restart/Debias_IMU/data/EUROC"
DEFAULT_SOURCE_ROOT = Path("/home/amao/workspace/formalPLATO/Results/openvins_imu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-method", default="vo_mse")
    parser.add_argument("--sequences", nargs="+", default=["MH_04_difficult"])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--component", choices=["both", "gyro", "accel"], default="both")
    return parser.parse_args()


def read_imu_csv(path: Path) -> tuple[list[str], dict[int, np.ndarray]]:
    rows: dict[int, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            stamp = int(row[0])
            rows[stamp] = np.asarray([float(v) for v in row[1:7]], dtype=float)
    return header, rows


def read_bias_csv(path: Path) -> tuple[list[str], list[int], np.ndarray]:
    stamps: list[int] = []
    values: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            stamps.append(int(row[0]))
            values.append([float(v) for v in row[1:7]])
    if not stamps:
        raise RuntimeError(f"No bias rows in {path}")
    return header, stamps, np.asarray(values, dtype=float)


def method_name(base_method: str, component: str, gamma: float) -> str:
    return f"{base_method}_{component}_g{int(round(gamma * 100)):03d}"


def correction_for_component(bias: np.ndarray, gamma: float, component: str) -> np.ndarray:
    correction = np.zeros_like(bias)
    if component in ("both", "gyro"):
        correction[:, :3] = gamma * bias[:, :3]
    if component in ("both", "accel"):
        correction[:, 3:] = gamma * bias[:, 3:]
    return correction


def write_imu_csv(path: Path, header: list[str], stamps: list[int], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for stamp, row in zip(stamps, values):
            writer.writerow([stamp] + [f"{v:.10f}" for v in row])


def write_bias_csv(path: Path, header: list[str], stamps: list[int], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for stamp, row in zip(stamps, values):
            writer.writerow([stamp] + [f"{v:.10f}" for v in row])


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "description": "Scaled learned-bias EuRoC IMU exports for OpenVINS diagnostics.",
        "source_root": str(args.source_root),
        "raw_root": str(args.raw_root),
        "base_method": args.base_method,
        "component": args.component,
        "methods": {},
    }
    methods = manifest["methods"]
    assert isinstance(methods, dict)

    for sequence in args.sequences:
        raw_header, raw_rows = read_imu_csv(args.raw_root / sequence / "mav0/imu0/data.csv")
        bias_header, stamps, bias = read_bias_csv(args.source_root / args.base_method / sequence / "bias.csv")
        missing = [stamp for stamp in stamps if stamp not in raw_rows]
        if missing:
            raise RuntimeError(f"{sequence}: {len(missing)} bias timestamps are missing from raw IMU, first={missing[0]}")
        raw = np.asarray([raw_rows[stamp] for stamp in stamps], dtype=float)

        for gamma in args.gammas:
            method = method_name(args.base_method, args.component, gamma)
            correction = correction_for_component(bias, gamma, args.component)
            corrected = raw - correction
            out_imu = args.output_root / method / sequence / "mav0/imu0/data.csv"
            out_bias = args.output_root / method / sequence / "bias.csv"
            write_imu_csv(out_imu, raw_header, stamps, corrected)
            write_bias_csv(out_bias, bias_header, stamps, correction)
            methods.setdefault(
                method,
                {
                    "description": f"{args.base_method} {args.component} scaled compensation, gamma={gamma:g}",
                    "base_method": args.base_method,
                    "component": args.component,
                    "gamma": gamma,
                    "sequences": {},
                },
            )
            method_entry = methods[method]
            assert isinstance(method_entry, dict)
            seqs = method_entry["sequences"]
            assert isinstance(seqs, dict)
            seqs[sequence] = {
                "imu_csv": str(out_imu),
                "bias_csv": str(out_bias),
                "samples": len(stamps),
                "timestamp_start_ns": stamps[0],
                "timestamp_end_ns": stamps[-1],
            }

    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_root / "manifest.json")


if __name__ == "__main__":
    main()
