#!/usr/bin/env python3
"""Estimate per-sequence corldata Body_0-to-IMU extrinsics from VO references.

The model is:

    T_M_Body0(t) * T_Body0_IMU ~= T_M_V * T_V_IMU_ref(t)

where T_M_Body0 comes from motion capture Body_0.tum and T_V_IMU_ref is the
raw ORB-SLAM3 left-camera trajectory right-multiplied by the physical
camera-to-IMU transform from the corldata stereo-inertial config.

This is a diagnostic calibration route. It records one fitted SE(3) extrinsic
per sequence and does not modify training GT exports.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
ORB_EXPORT_DIR = Path(__file__).resolve().parent
TOOLS_ESTIMATE = PLATO_ROOT / "tools" / "estimate_mocap_body_gt.py"
DEFAULT_DATA_ROOT = Path("/home/amao/datasets/corldata/0511_data")
DEFAULT_PSEUDO_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "corldata"
DEFAULT_CONFIG = PLATO_ROOT / "plato" / "configs" / "orbslam3" / "corldata_davis_stereo_inertial.yaml"
DEFAULT_OUTPUT_ROOT = DEFAULT_PSEUDO_ROOT / "body0_imu_extrinsics_vo_ref"

sys.path.insert(0, str(ORB_EXPORT_DIR))
import run_liasuw_orbslam3_trial as trial  # noqa: E402

spec = importlib.util.spec_from_file_location("plato_estimate_mocap_body_gt", TOOLS_ESTIMATE)
estimate_tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = estimate_tool
assert spec.loader is not None
spec.loader.exec_module(estimate_tool)
estimate_tool.require_scipy()
Rotation = estimate_tool.Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequences", nargs="+", help="Defaults to all corldata sequence dirs with alignment config.")
    parser.add_argument("--max-time-diff", type=float, default=0.03)
    parser.add_argument("--min-associations", type=int, default=100)
    parser.add_argument("--max-calib-samples", type=int, default=2500)
    parser.add_argument("--robust-loss", default="soft_l1")
    parser.add_argument("--loss-f-scale", type=float, default=0.05)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--write-vo-imu-ref", action="store_true", help="Also write the VO IMU-reference TUM trajectory.")
    return parser.parse_args()


def sequence_sort_key(name: str) -> tuple[str, int | str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    if not match:
        return name, name
    return match.group(1), int(match.group(2))


def discover_sequences(pseudo_root: Path) -> list[str]:
    sequences: list[str] = []
    for seq_dir in sorted(pseudo_root.glob("S*"), key=lambda p: sequence_sort_key(p.name)):
        if (seq_dir / "alignment_config" / "vo_gt_alignment.json").exists():
            sequences.append(seq_dir.name)
    if not sequences:
        raise FileNotFoundError(f"No corldata sequence alignment configs found under {pseudo_root}")
    return sequences


def load_opencv_matrix(path: Path, key: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"{re.escape(key)}:\s*!!opencv-matrix\b.*?data:\s*\[([^\]]+)\]", re.S)
    match = pattern.search(text)
    if match is None:
        raise KeyError(f"{key} not found in {path}")
    values = [float(token) for token in re.split(r"[\s,]+", match.group(1).strip()) if token]
    if len(values) != 16:
        raise ValueError(f"{path}:{key} has {len(values)} values, expected 16")
    return np.asarray(values, dtype=float).reshape(4, 4)


def inv_se3(pose: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = pose[:3, :3].T
    out[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return out


def se3(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


def pose_matrix(row: dict[str, Any]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = trial.quat_xyzw_to_rot(row["q"])
    pose[:3, 3] = np.asarray(row["p"], dtype=float)
    return pose


def matrix_to_row(timestamp: float, pose: np.ndarray) -> dict[str, Any]:
    return {
        "t": float(timestamp),
        "p": pose[:3, 3].copy(),
        "q": trial.rot_to_quat_xyzw(pose[:3, :3]),
    }


def raw_vo_path(seq_dir: Path) -> Path:
    candidates = [
        seq_dir / "raw" / "estimate_f_seconds.tum",
        seq_dir / "raw" / "vo_camera_traj_seconds.tum",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{seq_dir}: missing raw VO seconds trajectory")


def load_time_offset(seq_dir: Path) -> float:
    path = seq_dir / "alignment_config" / "vo_gt_alignment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if "time_offset_s" in data:
        return float(data["time_offset_s"])
    search = data.get("time_offset_search", {})
    if "time_offset_s" in search:
        return float(search["time_offset_s"])
    raise KeyError(f"{path} does not record time_offset_s")


def make_vo_imu_reference(camera_rows: list[dict[str, Any]], T_C_I: np.ndarray) -> list[dict[str, Any]]:
    return [matrix_to_row(float(row["t"]), pose_matrix(row) @ T_C_I) for row in camera_rows]


def downsample_pairs(pairs: list[tuple[int, int, float, float]], max_samples: int) -> list[tuple[int, int, float, float]]:
    if max_samples <= 0 or len(pairs) <= max_samples:
        return pairs
    indices = np.unique(np.round(np.linspace(0, len(pairs) - 1, max_samples)).astype(int))
    return [pairs[int(idx)] for idx in indices]


def calibration_data_from_pairs(
    sequence: str,
    gt_rows: list[dict[str, Any]],
    vo_imu_rows: list[dict[str, Any]],
    pairs: list[tuple[int, int, float, float]],
) -> Any:
    times = np.asarray([gt_rows[gt_idx]["t"] for gt_idx, _, _, _ in pairs], dtype=float)
    mocap_p = np.asarray([gt_rows[gt_idx]["p"] for gt_idx, _, _, _ in pairs], dtype=float)
    mocap_q = np.asarray([gt_rows[gt_idx]["q"] for gt_idx, _, _, _ in pairs], dtype=float)
    vio_p = np.asarray([vo_imu_rows[est_idx]["p"] for _, est_idx, _, _ in pairs], dtype=float)
    vio_q = np.asarray([vo_imu_rows[est_idx]["q"] for _, est_idx, _, _ in pairs], dtype=float)
    mocap = estimate_tool.PoseTrajectory(f"{sequence}_T_M_Body0", times, mocap_p, Rotation.from_quat(mocap_q))
    vio = estimate_tool.PoseTrajectory(f"{sequence}_T_V_IMU_ref", times.copy(), vio_p, Rotation.from_quat(vio_q))
    return estimate_tool.CalibrationData(times=times, mocap=mocap, vio=vio)


def closed_form_initial_guess(data: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit = trial.fit_world_sensor_se3(
        data.mocap.p,
        data.vio.p,
        data.mocap.rot.as_matrix(),
        data.vio.rot.as_matrix(),
    )
    T_IMU_Body0 = se3(fit["sensor_rot"], fit["sensor_trans"])
    T_Body0_IMU = inv_se3(T_IMU_Body0)
    T_mocapWorld_voWorld = se3(fit["world_rot"], fit["world_trans"])
    record = {
        "method": "trial.fit_world_sensor_se3",
        "model": "T_M_Body0 ~= T_M_V * T_V_IMU_ref * T_IMU_Body0; inverted for T_Body0_IMU",
        "T_Body0_IMU_initial": T_Body0_IMU.tolist(),
        "T_IMU_Body0_initial": T_IMU_Body0.tolist(),
        "T_mocapWorld_voWorld_initial": T_mocapWorld_voWorld.tolist(),
        "translation_lstsq_rank": fit["translation_lstsq_rank"],
        "translation_lstsq_condition": fit["translation_lstsq_condition"],
        "sensor_frame_residual_mean_deg": float(np.mean(fit["sensor_frame_residuals_deg"])),
        "sensor_frame_residual_rmse_deg": float(np.sqrt(np.mean(np.asarray(fit["sensor_frame_residuals_deg"]) ** 2))),
    }
    return T_Body0_IMU, T_mocapWorld_voWorld, record


def estimate_transforms_with_initial(
    data: Any,
    T_Body0_IMU_init: np.ndarray,
    T_mocapWorld_voWorld_init: np.ndarray,
    time_offset: float,
    robust_loss: str,
    f_scale: float,
    max_nfev: int,
) -> Any:
    def fun(x: np.ndarray) -> np.ndarray:
        T_Body0_IMU = estimate_tool.se3_exp(x[:6]) @ T_Body0_IMU_init
        T_mocapWorld_voWorld = estimate_tool.se3_exp(x[6:]) @ T_mocapWorld_voWorld_init
        return estimate_tool.residual_vector_for_transforms(data, T_Body0_IMU, T_mocapWorld_voWorld)

    result = estimate_tool.least_squares(
        fun,
        np.zeros(12, dtype=float),
        loss=robust_loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    T_Body0_IMU = estimate_tool.se3_exp(result.x[:6]) @ T_Body0_IMU_init
    T_mocapWorld_voWorld = estimate_tool.se3_exp(result.x[6:]) @ T_mocapWorld_voWorld_init
    stats = estimate_tool.compute_alignment_stats(data, T_Body0_IMU, T_mocapWorld_voWorld)
    return estimate_tool.CalibrationResult(
        T_MB=T_Body0_IMU,
        T_WO=T_mocapWorld_voWorld,
        time_offset=time_offset,
        result=result,
        stats=stats,
        data=data,
    )


def rotation_angle_deg(Rm: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(Rm) - 1.0) * 0.5, -1.0, 1.0))))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_matrix_yaml(path: Path, key: str, matrix: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["%YAML:1.0", f"# {metadata.get('description', '')}"]
    for meta_key, value in metadata.items():
        lines.append(f"# {meta_key}: {value}")
    flat = ", ".join(f"{float(value):.12g}" for value in matrix.reshape(-1))
    lines.extend(
        [
            f"{key}: !!opencv-matrix",
            "  rows: 4",
            "  cols: 4",
            "  dt: d",
            f"  data: [{flat}]",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tum(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            p = np.asarray(row["p"], dtype=float)
            q = np.asarray(row["q"], dtype=float)
            handle.write(
                f"{float(row['t']):.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def estimate_sequence(args: argparse.Namespace, sequence: str, T_I_C: np.ndarray, T_C_I: np.ndarray) -> dict[str, Any]:
    seq_dir = args.pseudo_root / sequence
    gt_path = args.data_root / "gt" / sequence / "Body_0.tum"
    camera_path = raw_vo_path(seq_dir)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gt_rows = trial.load_tum(gt_path, timestamp_scale="sec")
    camera_rows = trial.load_tum(camera_path, timestamp_scale="sec")
    vo_imu_rows = make_vo_imu_reference(camera_rows, T_C_I)
    time_offset = load_time_offset(seq_dir)
    pairs_all = trial.associate_by_time(gt_rows, vo_imu_rows, time_offset, args.max_time_diff)
    if len(pairs_all) < args.min_associations:
        raise RuntimeError(f"{sequence}: only {len(pairs_all)} associations, need {args.min_associations}")
    pairs = downsample_pairs(pairs_all, args.max_calib_samples)
    data = calibration_data_from_pairs(sequence, gt_rows, vo_imu_rows, pairs)
    T_Body0_IMU_init, T_mocapWorld_voWorld_init, initial_record = closed_form_initial_guess(data)
    result = estimate_transforms_with_initial(
        data,
        T_Body0_IMU_init,
        T_mocapWorld_voWorld_init,
        time_offset,
        args.robust_loss,
        args.loss_f_scale,
        args.max_nfev,
    )

    T_Body0_IMU = result.T_MB
    T_IMU_Body0 = inv_se3(T_Body0_IMU)
    out_dir = args.output_root / sequence
    per_seq_alignment_dir = seq_dir / "alignment_config"
    if args.write_vo_imu_ref:
        write_tum(out_dir / "vo_imu_reference.tum", vo_imu_rows)

    record: dict[str, Any] = {
        "dataset": "corldata",
        "sequence": sequence,
        "purpose": "per-sequence diagnostic Body_0-to-IMU extrinsic estimated from raw ORB-SLAM3 VO reference",
        "warning": (
            "This is fitted per sequence from VO and mocap. Treat it as a diagnostic extrinsic until "
            "cross-sequence stability is confirmed; do not assume it is a globally calibrated rig extrinsic."
        ),
        "model": "T_M_Body0(t) * T_Body0_IMU ~= T_M_V * T_V_IMU_ref(t)",
        "convention": "T_A_B maps coordinates from frame B into frame A; p_A_B is the origin of B expressed in A.",
        "source_mocap_body0": str(gt_path),
        "source_vo_camera": str(camera_path),
        "source_camera_imu_config": str(args.config),
        "time_offset_s": time_offset,
        "time_offset_sign": "association uses t_gt ~= t_vo + time_offset_s",
        "max_time_difference_s": args.max_time_diff,
        "associated_pose_count": len(pairs_all),
        "calibration_sample_count": len(pairs),
        "T_I_C_left_camera_to_imu_body": T_I_C.tolist(),
        "T_C_I_imu_body_to_left_camera": T_C_I.tolist(),
        "T_Body0_IMU": T_Body0_IMU.tolist(),
        "T_IMU_Body0": T_IMU_Body0.tolist(),
        "p_Body0_IMU_m": T_Body0_IMU[:3, 3].tolist(),
        "p_Body0_IMU_norm_m": float(np.linalg.norm(T_Body0_IMU[:3, 3])),
        "T_Body0_IMU_rotation_angle_deg": rotation_angle_deg(T_Body0_IMU[:3, :3]),
        "T_mocapWorld_voWorld": result.T_WO.tolist(),
        "initial_guess": initial_record,
        "optimization": {
            "robust_loss": args.robust_loss,
            "loss_f_scale": args.loss_f_scale,
            "max_nfev": args.max_nfev,
            "success": bool(result.result.success),
            "status": int(result.result.status),
            "message": str(result.result.message),
            "cost": float(result.result.cost),
            "optimality": float(result.result.optimality),
            "nfev": int(result.result.nfev),
        },
        "alignment_stats": result.stats,
        "outputs": {
            "aggregate_record": str(out_dir / "body0_imu_extrinsic_vo_ref.json"),
            "sequence_alignment_record": str(per_seq_alignment_dir / "body0_imu_extrinsic_vo_ref.json"),
            "T_Body0_IMU_yaml": str(out_dir / "T_Body0_IMU.yaml"),
        },
    }
    write_json(out_dir / "body0_imu_extrinsic_vo_ref.json", record)
    write_json(per_seq_alignment_dir / "body0_imu_extrinsic_vo_ref.json", record)
    write_matrix_yaml(
        out_dir / "T_Body0_IMU.yaml",
        "T_Body0_IMU",
        T_Body0_IMU,
        {
            "description": "Diagnostic corldata Body_0-to-IMU extrinsic",
            "sequence": sequence,
            "model": record["model"],
        },
    )
    return {
        "sequence": sequence,
        "status": "ok",
        "time_offset_s": time_offset,
        "associated_pose_count": len(pairs_all),
        "calibration_sample_count": len(pairs),
        "p_Body0_IMU_x_m": float(T_Body0_IMU[0, 3]),
        "p_Body0_IMU_y_m": float(T_Body0_IMU[1, 3]),
        "p_Body0_IMU_z_m": float(T_Body0_IMU[2, 3]),
        "p_Body0_IMU_norm_m": float(np.linalg.norm(T_Body0_IMU[:3, 3])),
        "T_Body0_IMU_rotation_angle_deg": rotation_angle_deg(T_Body0_IMU[:3, :3]),
        "translation_rmse_m": result.stats["translation_rmse_m"],
        "rotation_rmse_deg": result.stats["rotation_rmse_deg"],
        "se3_residual_rmse": result.stats["se3_residual_rmse"],
        "record": str(out_dir / "body0_imu_extrinsic_vo_ref.json"),
        "sequence_record": str(per_seq_alignment_dir / "body0_imu_extrinsic_vo_ref.json"),
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# corldata Body_0-to-IMU Extrinsics From VO References",
        "",
        "Model: `T_M_Body0(t) * T_Body0_IMU ~= T_M_V * T_V_IMU_ref(t)`.",
        "",
        "| sequence | associations | p_Body0_IMU [m] | norm m | rot angle deg | trans RMSE m | rot RMSE deg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ok_rows:
        lines.append(
            "| {seq} | {assoc} | ({x:.4f}, {y:.4f}, {z:.4f}) | {norm:.4f} | {angle:.3f} | {tr:.5f} | {rr:.3f} |".format(
                seq=row["sequence"],
                assoc=int(row["associated_pose_count"]),
                x=float(row["p_Body0_IMU_x_m"]),
                y=float(row["p_Body0_IMU_y_m"]),
                z=float(row["p_Body0_IMU_z_m"]),
                norm=float(row["p_Body0_IMU_norm_m"]),
                angle=float(row["T_Body0_IMU_rotation_angle_deg"]),
                tr=float(row["translation_rmse_m"]),
                rr=float(row["rotation_rmse_deg"]),
            )
        )
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        lines.extend(["", "Failed sequences:"])
        for row in failed:
            lines.append(f"- {row['sequence']}: {row.get('error', 'unknown error')}")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.pseudo_root = args.pseudo_root.resolve()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()

    sequences = args.sequences or discover_sequences(args.pseudo_root)
    T_I_C = load_opencv_matrix(args.config, "IMU.T_b_c1")
    T_C_I = inv_se3(T_I_C)

    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        print(f"[extrinsic] {sequence}", flush=True)
        try:
            rows.append(estimate_sequence(args, sequence, T_I_C, T_C_I))
        except Exception as exc:  # noqa: BLE001
            rows.append({"sequence": sequence, "status": "failed", "error": str(exc)})
            print(f"[extrinsic] {sequence} failed: {exc}", file=sys.stderr, flush=True)
    write_summary(args.output_root, rows)
    print(json.dumps({"output_root": str(args.output_root), "rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
