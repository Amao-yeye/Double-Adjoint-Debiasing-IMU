#!/usr/bin/env python3
"""Export corldata IMU streams for OpenVINS corrected-IMU trials.

The learned-corrected stream is the original high-rate IMU with the Debias_IMU
predicted gyro/accel biases subtracted. Images are not touched here; bag
creation reads the existing exported corldata image folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
OPENVINS_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
    "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            return pickle.load(f)


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def read_timestamped_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    values: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"No rows found in {path}")
        rows = [] if any(ch.isalpha() for ch in ",".join(first)) else [first]
        rows.extend(reader)
        for row in rows:
            if not row:
                continue
            timestamps.append(int(row[0].strip()))
            values.append([float(x) for x in row[1:7]])
    if not timestamps:
        raise ValueError(f"No IMU rows found in {path}")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def read_image_timestamps(path: Path) -> np.ndarray:
    values = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"No image timestamps found in {path}")
    return np.asarray(values, dtype=np.int64)


def interp_columns(x_src: np.ndarray, values: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    out = np.empty((x_dst.shape[0], values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(x_dst, x_src, values[:, col])
    return out


def write_openvins_imu_csv(path: Path, timestamps_ns: np.ndarray, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([timestamps_ns.astype(object), values.astype(np.float64)])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=OPENVINS_IMU_HEADER,
        comments="",
        fmt=["%s"] + ["%.18e"] * 6,
    )


def write_bias_csv(path: Path, timestamps_ns: np.ndarray, bias: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_ns",
                "bg_x_rad_s",
                "bg_y_rad_s",
                "bg_z_rad_s",
                "ba_x_m_s2",
                "ba_y_m_s2",
                "ba_z_m_s2",
            ]
        )
        for row in np.column_stack([timestamps_ns, bias]):
            writer.writerow([str(int(row[0]))] + [f"{x:.18e}" for x in row[1:]])


def summarize_imu(timestamps_ns: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    dt_s = np.diff(timestamps_ns.astype(np.float64)) * 1e-9
    return {
        "rows": int(values.shape[0]),
        "start_ns": int(timestamps_ns[0]),
        "end_ns": int(timestamps_ns[-1]),
        "duration_s": float((timestamps_ns[-1] - timestamps_ns[0]) * 1e-9),
        "median_dt_s": float(np.median(dt_s)) if dt_s.size else 0.0,
        "min_dt_s": float(np.min(dt_s)) if dt_s.size else 0.0,
        "max_dt_s": float(np.max(dt_s)) if dt_s.size else 0.0,
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
    }


def export_sequence(args: argparse.Namespace, sequence: str) -> Path:
    source_sequence_dir = args.pseudo_gt_root / sequence / "sequence"
    source_imu = source_sequence_dir / "imu.txt"
    image_timestamps_path = source_sequence_dir / "timestamps.txt"
    result_path = args.results_dir / "Sequnces-results" / sequence / "FullTra_result.p"
    constant_path = args.results_dir / "constant_bias" / f"{sequence}.p"
    for path in [source_imu, image_timestamps_path, result_path, constant_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    result = load_pickle(result_path)
    constant = load_pickle(constant_path)
    sol_pred = as_numpy(result["sol_pred"]).astype(np.float64)
    t_pred_rel = as_numpy(result["t_gt"]).astype(np.float64)
    timestamp_start = float(as_numpy(result["timestamp_start"]))
    pred_t_ns = np.rint(timestamp_start + t_pred_rel * 1e9).astype(np.int64)
    pred_x = (pred_t_ns.astype(np.float64) - float(pred_t_ns[0])) * 1e-9

    bg_pred = sol_pred[:, 4:7]
    ba_pred = sol_pred[:, 13:16]
    bias_pred = np.column_stack([bg_pred, ba_pred])

    raw_t_ns_all, raw_values_all = read_timestamped_imu_csv(source_imu)
    raw_mask = (raw_t_ns_all >= pred_t_ns[0]) & (raw_t_ns_all <= pred_t_ns[-1])
    if int(raw_mask.sum()) < 2:
        raise ValueError(f"No useful IMU overlap for {sequence}")
    raw_t_ns = raw_t_ns_all[raw_mask]
    raw_values = raw_values_all[raw_mask]
    raw_x = (raw_t_ns.astype(np.float64) - float(pred_t_ns[0])) * 1e-9
    bias_interp = interp_columns(pred_x, bias_pred, raw_x)

    bw_const = as_numpy(constant["bw_const"]).astype(np.float64).reshape(3)
    ba_const = as_numpy(constant["ba_const"]).astype(np.float64).reshape(3)
    const_bias = np.concatenate([bw_const, ba_const])

    method_values = {
        "raw": raw_values,
        "constant": raw_values - const_bias.reshape(1, 6),
        "network": raw_values - bias_interp,
        "network_gyro_only": raw_values - np.column_stack([bias_interp[:, :3], np.zeros_like(bias_interp[:, 3:6])]),
        "network_accel_only": raw_values - np.column_stack([np.zeros_like(bias_interp[:, :3]), bias_interp[:, 3:6]]),
    }
    output_dir = args.output_dir / sequence
    method_paths: dict[str, str] = {}
    for method in args.methods:
        out_csv = output_dir / "methods" / method / "mav0" / "imu0" / "data.csv"
        write_openvins_imu_csv(out_csv, raw_t_ns, method_values[method])
        method_paths[method] = str(out_csv)

    write_bias_csv(output_dir / "biases" / "network_bias_prediction_grid.csv", pred_t_ns, bias_pred)
    write_bias_csv(output_dir / "biases" / "network_bias_interpolated_to_raw_imu.csv", raw_t_ns, bias_interp)
    (output_dir / "biases" / "constant_bias.json").write_text(
        json.dumps({"bw_const_rad_s": bw_const.tolist(), "ba_const_m_s2": ba_const.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )

    imu_raw_saved = as_numpy(result["imu_raw"]).astype(np.float64)
    net_imu_saved = as_numpy(result["net_imu"]).astype(np.float64)
    raw_interp_at_pred = interp_columns(raw_x, raw_values, pred_x)
    network_at_pred = raw_interp_at_pred - bias_pred
    image_timestamps = read_image_timestamps(image_timestamps_path)
    image_inside_bias_span = (image_timestamps >= raw_t_ns[0]) & (image_timestamps <= raw_t_ns[-1])

    validation = {
        "sequence": sequence,
        "results_dir": str(args.results_dir),
        "source_sequence_dir": str(source_sequence_dir),
        "source_imu": str(source_imu),
        "fulltra_result": str(result_path),
        "constant_bias_result": str(constant_path),
        "methods": method_paths,
        "prediction_grid": {
            "rows": int(pred_t_ns.shape[0]),
            "start_ns": int(pred_t_ns[0]),
            "end_ns": int(pred_t_ns[-1]),
            "median_dt_s": float(np.median(np.diff(pred_t_ns)) * 1e-9),
        },
        "raw_imu_export": summarize_imu(raw_t_ns, raw_values),
        "image_timestamps": {
            "path": str(image_timestamps_path),
            "rows": int(image_timestamps.shape[0]),
            "start_ns": int(image_timestamps[0]),
            "end_ns": int(image_timestamps[-1]),
            "inside_bias_span": int(image_inside_bias_span.sum()),
            "outside_bias_span": int((~image_inside_bias_span).sum()),
        },
        "checks": {
            "raw_sequence_interp_vs_saved_imu_max_abs": float(np.max(np.abs(raw_interp_at_pred - imu_raw_saved))),
            "network_from_raw_interp_vs_saved_net_imu_max_abs": float(np.max(np.abs(network_at_pred - net_imu_saved))),
            "raw_export_exactly_preserves_source_overlap": bool(
                np.array_equal(raw_t_ns, raw_t_ns_all[raw_mask])
                and np.allclose(raw_values, raw_values_all[raw_mask], rtol=0.0, atol=0.0)
            ),
            "strictly_increasing_raw_imu_timestamps": bool(np.all(np.diff(raw_t_ns) > 0)),
            "strictly_increasing_prediction_timestamps": bool(np.all(np.diff(pred_t_ns) > 0)),
        },
        "bias_stats": {
            "network_bg_mean_rad_s": bg_pred.mean(axis=0).tolist(),
            "network_ba_mean_m_s2": ba_pred.mean(axis=0).tolist(),
            "network_bg_std_rad_s": bg_pred.std(axis=0).tolist(),
            "network_ba_std_m_s2": ba_pred.std(axis=0).tolist(),
            "constant_bias": const_bias.tolist(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export corldata corrected IMU CSVs for OpenVINS.")
    parser.add_argument("--sequence", action="append", dest="sequences", default=[])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "debias_imu"
        / "corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue",
    )
    parser.add_argument(
        "--pseudo-gt-root",
        type=Path,
        default=PLATO_ROOT / "results" / "pseudo_gt" / "corldata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "interval_nll_corrected_inputs",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["raw", "constant", "network", "network_gyro_only", "network_accel_only"],
        default=["raw", "constant", "network"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = args.sequences or ["S13", "S15"]
    for sequence in sequences:
        out_dir = export_sequence(args, sequence)
        print(f"Wrote {sequence}: {out_dir}")


if __name__ == "__main__":
    main()
