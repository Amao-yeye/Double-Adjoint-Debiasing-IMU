#!/usr/bin/env python3
"""Export network-debiased IMU as EuRoC/OpenVINS-compatible CSV files."""

import argparse
import os
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np


EUROC_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
    "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)


def pload(path: Path) -> dict:
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            return pickle.load(f)


def _available_sequences(results_dir: Path) -> list[str]:
    seq_root = results_dir / "Sequnces-results"
    if not seq_root.exists():
        raise FileNotFoundError(f"Missing result directory: {seq_root}")
    return sorted(
        p.name for p in seq_root.iterdir()
        if p.is_dir() and (p / "FullTra_result.p").exists()
    )


def _read_raw_imu_csv(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw IMU CSV: {path}")
    rows = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    if rows.ndim == 1:
        rows = rows[None, :]
    timestamps = rows[:, 0].astype(np.float64)
    return rows[:, 0].tolist(), timestamps


def _timestamps_from_raw(raw_csv: Path, timestamp_start: float, count: int) -> np.ndarray:
    raw_timestamp_strings, raw_timestamps_float = _read_raw_imu_csv(raw_csv)
    start_idx = int(np.argmin(np.abs(raw_timestamps_float - float(timestamp_start))))
    end_idx = start_idx + count
    if end_idx > len(raw_timestamp_strings):
        raise ValueError(
            f"{raw_csv} does not have enough IMU rows from matched start index "
            f"{start_idx}: need {count}, have {len(raw_timestamp_strings) - start_idx}"
        )
    return np.asarray(raw_timestamp_strings[start_idx:end_idx], dtype=object)


def _timestamps_from_result(result: dict, count: int) -> np.ndarray:
    if "t_gt" not in result or "timestamp_start" not in result:
        raise KeyError(
            "FullTra_result.p must contain t_gt and timestamp_start when raw_data_dir is not provided"
        )
    rel_t = result["t_gt"].detach().cpu().numpy().astype(np.float64)
    timestamps = np.rint(float(result["timestamp_start"]) + rel_t * 1e9).astype(np.int64)
    if timestamps.shape[0] != count:
        raise ValueError(f"timestamp count {timestamps.shape[0]} does not match IMU count {count}")
    return timestamps.astype(str).astype(object)


def _prepare_sequence_tree(raw_seq_dir: Path | None, out_seq_dir: Path, mode: str) -> None:
    if mode == "none" or raw_seq_dir is None:
        (out_seq_dir / "mav0" / "imu0").mkdir(parents=True, exist_ok=True)
        return
    if mode == "copy-metadata":
        for rel in ("mav0/body.yaml", "mav0/imu0/sensor.yaml"):
            src = raw_seq_dir / rel
            dst = out_seq_dir / rel
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        (out_seq_dir / "mav0" / "imu0").mkdir(parents=True, exist_ok=True)
        return
    if mode == "symlink":
        out_seq_dir.mkdir(parents=True, exist_ok=True)
        skipped_imu_csv = Path("mav0") / "imu0" / "data.csv"
        for root, dirs, files in os.walk(raw_seq_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(raw_seq_dir)
            out_root = out_seq_dir / rel_root
            out_root.mkdir(parents=True, exist_ok=True)
            for directory in dirs:
                (out_root / directory).mkdir(exist_ok=True)
            for filename in files:
                rel_file = rel_root / filename
                if rel_file == skipped_imu_csv:
                    continue
                src = root_path / filename
                dst = out_seq_dir / rel_file
                if dst.exists() or dst.is_symlink():
                    continue
                os.symlink(src.resolve(), dst)
        return
    raise ValueError(f"Unsupported tree mode: {mode}")


def export_sequence(
    results_dir: Path,
    output_dir: Path,
    sequence: str,
    raw_data_dir: Path | None,
    tree_mode: str,
) -> Path:
    result_path = results_dir / "Sequnces-results" / sequence / "FullTra_result.p"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result pickle: {result_path}")
    result = pload(result_path)
    net_imu = result["net_imu"].detach().cpu().numpy()
    if net_imu.ndim != 2 or net_imu.shape[1] != 6:
        raise ValueError(f"net_imu in {result_path} should have shape [N, 6], got {net_imu.shape}")

    raw_seq_dir = raw_data_dir / sequence if raw_data_dir is not None else None
    out_seq_dir = output_dir / sequence
    _prepare_sequence_tree(raw_seq_dir, out_seq_dir, tree_mode)

    raw_csv = raw_seq_dir / "mav0" / "imu0" / "data.csv" if raw_seq_dir is not None else None
    if raw_csv is not None and raw_csv.exists():
        timestamps = _timestamps_from_raw(raw_csv, result["timestamp_start"], net_imu.shape[0])
    else:
        timestamps = _timestamps_from_result(result, net_imu.shape[0])

    out_imu_dir = out_seq_dir / "mav0" / "imu0"
    out_imu_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_imu_dir / "data.csv"

    data = np.column_stack([timestamps, net_imu.astype(np.float64)])
    np.savetxt(
        out_csv,
        data,
        delimiter=",",
        header=EUROC_IMU_HEADER,
        comments="",
        fmt=["%s"] + ["%.18e"] * 6,
    )
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Debias_IMU FullTra_result.p outputs into OpenVINS/EuRoC "
            "mav0/imu0/data.csv files using net_imu = raw_imu - predicted_bias."
        )
    )
    parser.add_argument("--results_dir", required=True, help="Experiment output directory containing Sequnces-results.")
    parser.add_argument("--output_dir", required=True, help="Output EuRoC-style dataset root.")
    parser.add_argument("--raw_data_dir", default=None, help="Original EuRoC/TUM-style dataset root used to preserve timestamps and metadata.")
    parser.add_argument("--sequences", nargs="*", default=None, help="Sequences to export. Defaults to all FullTra_result.p sequences.")
    parser.add_argument(
        "--tree_mode",
        choices=["copy-metadata", "symlink", "none"],
        default="copy-metadata",
        help=(
            "How much of the original sequence tree to expose in output. "
            "Use symlink when OpenVINS should read the same camera files from the original dataset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    raw_data_dir = Path(args.raw_data_dir) if args.raw_data_dir else None
    sequences = args.sequences or _available_sequences(results_dir)

    for sequence in sequences:
        out_csv = export_sequence(results_dir, output_dir, sequence, raw_data_dir, args.tree_mode)
        print(f"{sequence}: wrote {out_csv}")


if __name__ == "__main__":
    main()
