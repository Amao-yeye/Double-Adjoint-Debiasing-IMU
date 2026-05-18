#!/usr/bin/env python3
"""Export LIASuw IMU variants for the OpenVINS debiased-IMU route.

This follows the Debiasing 6-DOF IMU paper's downstream route: the learned
bias trajectory is evaluated by replacing OpenVINS' input IMU stream with
raw-minus-predicted-bias measurements. OpenVINS itself remains unchanged.
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
from scipy.spatial.transform import Rotation, Slerp


PLATO_ROOT = Path(__file__).resolve().parents[3]

OPENVINS_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
    "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)

GT_ASL_HEADER = (
    "#time(ns),px,py,pz,qw,qx,qy,qz,vx,vy,vz,bwx,bwy,bwz,bax,bay,baz"
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


def first_line_is_header(path: Path) -> bool:
    first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    return any(ch.isalpha() for ch in first) or first.startswith("#")


def read_numeric_csv(path: Path) -> np.ndarray:
    skip_header = 1 if first_line_is_header(path) else 0
    data = np.genfromtxt(path, delimiter=",", skip_header=skip_header)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(f"{path} should have at least 7 columns, got {data.shape}")
    return data.astype(np.float64)


def read_timestamped_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    skip_header = first_line_is_header(path)
    timestamps: list[int] = []
    values: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            timestamps.append(int(row[0].strip()))
            values.append([float(x) for x in row[1:7]])
    if not timestamps:
        raise ValueError(f"No IMU rows found in {path}")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def read_image_timestamps(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            values.append(int(stripped))
    if not values:
        raise ValueError(f"No image timestamps found in {path}")
    return np.asarray(values, dtype=np.int64)


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


def write_bias_csv(path: Path, timestamps_ns: np.ndarray, bg: np.ndarray, ba: np.ndarray) -> None:
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
        for row in np.column_stack([timestamps_ns, bg, ba]):
            writer.writerow([f"{int(row[0])}"] + [f"{x:.18e}" for x in row[1:]])


def write_gt_asl_csv(
    body_gt_csv: Path,
    output_csv: Path,
    target_timestamps_ns: np.ndarray | None = None,
    clip_target_times: bool = False,
    bias: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    gt = read_numeric_csv(body_gt_csv)
    if gt.shape[1] < 8:
        raise ValueError(f"{body_gt_csv} should contain timestamp,p,q columns")
    t_s = gt[:, 0]
    p = gt[:, 1:4]
    q_wxyz = gt[:, 4:8]
    velocity = np.gradient(p, t_s, axis=0, edge_order=1)
    if target_timestamps_ns is None:
        out_t_s = t_s
        timestamps_ns = np.rint(t_s * 1e9).astype(np.int64)
        out_p = p
        out_q_wxyz = q_wxyz
        out_velocity = velocity
    else:
        target_timestamps_ns = np.asarray(target_timestamps_ns, dtype=np.int64)
        target_t_s = target_timestamps_ns.astype(np.float64) * 1e-9
        if clip_target_times:
            valid = (target_t_s >= t_s[0] - 0.10) & (target_t_s <= t_s[-1] + 0.10)
        else:
            valid = (target_t_s >= t_s[0]) & (target_t_s <= t_s[-1])
        if not np.any(valid):
            raise ValueError("No target GT timestamps lie inside the body GT span")
        timestamps_ns = target_timestamps_ns[valid]
        out_t_s = timestamps_ns.astype(np.float64) * 1e-9
        interp_t_s = np.clip(out_t_s, t_s[0], t_s[-1])
        out_p = interp_columns(t_s, p, interp_t_s)
        q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
        slerp = Slerp(t_s, Rotation.from_quat(q_xyzw))
        out_q_xyzw = slerp(interp_t_s).as_quat()
        out_q_wxyz = out_q_xyzw[:, [3, 0, 1, 2]]
        out_velocity = interp_columns(t_s, velocity, interp_t_s)
    if bias is None:
        bias_columns = np.zeros((timestamps_ns.shape[0], 6), dtype=np.float64)
    else:
        bias = np.asarray(bias, dtype=np.float64).reshape(6)
        bias_columns = np.broadcast_to(bias.reshape(1, 6), (timestamps_ns.shape[0], 6))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([timestamps_ns.astype(object), out_p, out_q_wxyz, out_velocity, bias_columns])
    np.savetxt(
        output_csv,
        data,
        delimiter=",",
        header=GT_ASL_HEADER,
        comments="",
        fmt=["%s"] + ["%.18e"] * 16,
    )
    dt = np.diff(t_s)
    return {
        "path": str(output_csv),
        "rows": int(timestamps_ns.shape[0]),
        "start_s": float(t_s[0]),
        "end_s": float(t_s[-1]),
        "output_start_s": float(out_t_s[0]),
        "output_end_s": float(out_t_s[-1]),
        "median_dt_s": float(np.median(dt)) if dt.size else 0.0,
        "target_timestamps": "source_gt" if target_timestamps_ns is None else "image_frames",
        "bias_source": "zeros" if bias is None else "constant_bias",
    }


def interp_columns(x_src: np.ndarray, values: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    out = np.empty((x_dst.shape[0], values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(x_dst, x_src, values[:, col])
    return out


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


def export_sequence(args: argparse.Namespace) -> Path:
    sequence = args.sequence
    results_dir = args.results_dir
    pseudo_gt_root = args.pseudo_gt_root
    source_sequence_dir = pseudo_gt_root / sequence / "sequence"
    source_imu = source_sequence_dir / "imu.txt"
    image_timestamps_path = source_sequence_dir / "timestamps.txt"
    body_gt_csv = pseudo_gt_root / sequence / "debias_imu_body_gt_200hz" / "body_gt_world.csv"
    result_path = results_dir / "Sequnces-results" / sequence / "FullTra_result.p"
    constant_path = results_dir / "constant_bias" / f"{sequence}.p"

    for path in [source_imu, image_timestamps_path, body_gt_csv, result_path, constant_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    result = load_pickle(result_path)
    constant = load_pickle(constant_path)

    sol_pred = as_numpy(result["sol_pred"]).astype(np.float64)
    imu_raw_pred = as_numpy(result["imu_raw"]).astype(np.float64)
    net_imu_saved = as_numpy(result["net_imu"]).astype(np.float64)
    t_pred_rel = as_numpy(result["t_gt"]).astype(np.float64)
    timestamp_start = float(as_numpy(result["timestamp_start"]))
    pred_t_ns = np.rint(timestamp_start + t_pred_rel * 1e9).astype(np.int64)
    pred_x = (pred_t_ns.astype(np.float64) - float(pred_t_ns[0])) * 1e-9

    bg_pred = sol_pred[:, 4:7]
    ba_pred = sol_pred[:, 13:16]
    bias_pred = np.column_stack([bg_pred, ba_pred])
    reconstructed_net = imu_raw_pred - bias_pred
    net_imu_check = np.max(np.abs(reconstructed_net - net_imu_saved))

    raw_t_ns_all, raw_values_all = read_timestamped_imu_csv(source_imu)
    raw_mask = (raw_t_ns_all >= pred_t_ns[0]) & (raw_t_ns_all <= pred_t_ns[-1])
    if raw_mask.sum() < 2:
        raise ValueError("No useful overlap between source raw IMU and predicted bias trajectory.")
    raw_t_ns = raw_t_ns_all[raw_mask]
    raw_values = raw_values_all[raw_mask]
    raw_x = (raw_t_ns.astype(np.float64) - float(pred_t_ns[0])) * 1e-9

    image_timestamps = read_image_timestamps(image_timestamps_path)
    image_inside_bias_span = (image_timestamps >= pred_t_ns[0]) & (image_timestamps <= pred_t_ns[-1])

    bias_interp = interp_columns(pred_x, bias_pred, raw_x)
    network_values = raw_values - bias_interp

    bw_const = as_numpy(constant["bw_const"]).astype(np.float64).reshape(3)
    ba_const = as_numpy(constant["ba_const"]).astype(np.float64).reshape(3)
    const_bias = np.concatenate([bw_const, ba_const])
    constant_values = raw_values - const_bias.reshape(1, 6)

    raw_interp_at_pred = interp_columns(raw_x, raw_values, pred_x)
    network_at_pred_from_raw = raw_interp_at_pred - bias_pred
    raw_interp_diff = raw_interp_at_pred - imu_raw_pred
    network_interp_diff = network_at_pred_from_raw - net_imu_saved

    output_dir = args.output_dir / sequence
    method_values = {
        "raw": raw_values,
        "constant": constant_values,
        "network": network_values,
    }
    method_paths: dict[str, str] = {}
    for method, values in method_values.items():
        if method not in args.methods:
            continue
        out_csv = output_dir / "methods" / method / "mav0" / "imu0" / "data.csv"
        write_openvins_imu_csv(out_csv, raw_t_ns, values)
        method_paths[method] = str(out_csv)

    write_bias_csv(output_dir / "biases" / "network_bias_prediction_grid.csv", pred_t_ns, bg_pred, ba_pred)
    write_bias_csv(
        output_dir / "biases" / "network_bias_interpolated_to_raw_imu.csv",
        raw_t_ns,
        bias_interp[:, :3],
        bias_interp[:, 3:6],
    )
    (output_dir / "biases" / "constant_bias.json").write_text(
        json.dumps(
            {
                "bw_const_rad_s": bw_const.tolist(),
                "ba_const_m_s2": ba_const.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    gt_summary = write_gt_asl_csv(body_gt_csv, output_dir / "groundtruth" / "body_gt_openvins_asl.csv")
    gt_image_time_summary = write_gt_asl_csv(
        body_gt_csv,
        output_dir / "groundtruth" / "body_gt_openvins_asl_image_times.csv",
        image_timestamps[image_inside_bias_span],
    )
    image_timestamps_in_bag = image_timestamps[image_timestamps <= raw_t_ns[-1]]
    gt_init_summary = write_gt_asl_csv(
        body_gt_csv,
        output_dir / "groundtruth" / "body_gt_openvins_asl_gtinit.csv",
        image_timestamps_in_bag - 1_000,
        clip_target_times=True,
    )
    gt_init_raw_constbias_summary = write_gt_asl_csv(
        body_gt_csv,
        output_dir / "groundtruth" / "body_gt_openvins_asl_gtinit_raw_constbias.csv",
        image_timestamps_in_bag - 1_000,
        clip_target_times=True,
        bias=const_bias,
    )

    validation = {
        "sequence": sequence,
        "results_dir": str(results_dir),
        "source_sequence_dir": str(source_sequence_dir),
        "source_imu": str(source_imu),
        "body_gt_csv": str(body_gt_csv),
        "fulltra_result": str(result_path),
        "constant_bias_result": str(constant_path),
        "methods": method_paths,
        "bias_files": {
            "network_prediction_grid": str(output_dir / "biases" / "network_bias_prediction_grid.csv"),
            "network_interpolated_to_raw": str(output_dir / "biases" / "network_bias_interpolated_to_raw_imu.csv"),
            "constant": str(output_dir / "biases" / "constant_bias.json"),
        },
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
        "groundtruth": gt_summary,
        "groundtruth_image_times": gt_image_time_summary,
        "groundtruth_gtinit": gt_init_summary,
        "groundtruth_gtinit_raw_constbias": gt_init_raw_constbias_summary,
        "checks": {
            "net_imu_equals_imu_raw_minus_predicted_bias_max_abs": float(net_imu_check),
            "raw_sequence_interp_vs_saved_200hz_imu_max_abs": float(np.max(np.abs(raw_interp_diff))),
            "network_from_raw_interp_vs_saved_net_imu_max_abs": float(np.max(np.abs(network_interp_diff))),
            "raw_export_rows_equal_original_overlap_rows": bool(
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
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    write_readme(output_dir, validation)
    return output_dir


def write_readme(output_dir: Path, validation: dict[str, Any]) -> None:
    checks = validation["checks"]
    methods = validation["methods"]
    lines = [
        "# LIASuw OpenVINS Debiased-IMU Export",
        "",
        "This directory implements the paper-style OpenVINS route: OpenVINS remains unchanged,",
        "and only its input IMU stream is replaced by raw-minus-bias measurements.",
        "",
        "## Methods",
    ]
    for name, path in methods.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- `net_imu = imu_raw - predicted_bias` max abs diff: "
            f"{checks['net_imu_equals_imu_raw_minus_predicted_bias_max_abs']:.6e}",
            f"- high-rate raw IMU interpolation vs saved 200 Hz IMU max abs diff: "
            f"{checks['raw_sequence_interp_vs_saved_200hz_imu_max_abs']:.6e}",
            f"- high-rate network IMU interpolation vs saved 200 Hz net_imu max abs diff: "
            f"{checks['network_from_raw_interp_vs_saved_net_imu_max_abs']:.6e}",
            f"- raw export exactly preserves original overlap rows: "
            f"{checks['raw_export_rows_equal_original_overlap_rows']}",
            f"- raw IMU timestamps strictly increase: "
            f"{checks['strictly_increasing_raw_imu_timestamps']}",
            "",
            "Full details are in `validation.json`.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LIASuw raw/constant/network IMU streams for OpenVINS."
    )
    parser.add_argument("--sequence", default="0205s14_80")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "debias_imu"
        / "LIASuw_phys_delta_vo_pseudogt_framefixed_e1800_w16pose_b64",
    )
    parser.add_argument(
        "--pseudo-gt-root",
        type=Path,
        default=PLATO_ROOT / "results" / "pseudo_gt" / "LIASuw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "LIASuw"
        / "phys_delta_vo_framefixed_clean_imu",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["raw", "constant", "network"],
        default=["raw", "constant", "network"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = export_sequence(args)
    print(f"Wrote LIASuw OpenVINS IMU export: {out_dir}")
    print(f"Validation: {out_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
