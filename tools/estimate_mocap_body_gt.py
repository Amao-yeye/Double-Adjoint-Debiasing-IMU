#!/usr/bin/env python3
"""Estimate mocap-marker to IMU/body extrinsic and export body GT.

Core model:
    T_WM(t) * T_MB ~= T_WO * T_OB_vio(t)
    T_WB_gt(t) = T_WM(t) * T_MB
    T_OB_gt(t) = inv(T_WB_gt(t0)) * T_WB_gt(t)

Conventions:
    T_AB maps points from frame B to frame A:
        p_A = R_AB p_B + p_AB

Inputs:
    mocap pose: T_WM(t), marker rigid body M in mocap/world frame W
    VIO pose:   T_OB_vio(t), IMU/body B in VIO initialized world frame O
    IMU csv:    timestamps used to limit overlap and, optionally, resample final body GT

This script uses only SE(3). It does not estimate scale.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

least_squares = None
Rotation = None
Slerp = None


def require_scipy() -> None:
    global least_squares, Rotation, Slerp
    if least_squares is not None and Rotation is not None and Slerp is not None:
        return
    try:
        from scipy.optimize import least_squares as scipy_least_squares
        from scipy.spatial.transform import Rotation as scipy_rotation
        from scipy.spatial.transform import Slerp as scipy_slerp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required for SE(3) optimization and quaternion SLERP. "
            "Install scipy in the Python environment used to run this tool."
        ) from exc
    least_squares = scipy_least_squares
    Rotation = scipy_rotation
    Slerp = scipy_slerp


@dataclass
class PoseTrajectory:
    name: str
    t: np.ndarray
    p: np.ndarray
    rot: Rotation

    def __len__(self) -> int:
        return int(self.t.shape[0])


@dataclass
class CalibrationData:
    times: np.ndarray
    mocap: PoseTrajectory
    vio: PoseTrajectory


@dataclass
class CalibrationResult:
    T_MB: np.ndarray
    T_WO: np.ndarray
    time_offset: float
    result: object
    stats: Dict[str, float]
    data: CalibrationData


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    theta2 = theta * theta
    return (
        np.eye(3)
        + ((1.0 - math.cos(theta)) / theta2) * K
        + ((theta - math.sin(theta)) / (theta2 * theta)) * (K @ K)
    )


def so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    theta2 = theta * theta
    coeff = (1.0 / theta2) - ((1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta)))
    return np.eye(3) - 0.5 * K + coeff * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential map. xi order is [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho = xi[:3]
    phi = xi[3:]
    Rm = Rotation.from_rotvec(phi).as_matrix()
    J = so3_left_jacobian(phi)
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = J @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) logarithm map. Output order is [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]."""
    Rm = T[:3, :3]
    t = T[:3, 3]
    phi = Rotation.from_matrix(Rm).as_rotvec()
    rho = so3_left_jacobian_inv(phi) @ t
    return np.concatenate([rho, phi])


def pose_to_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = T[:3, 3].copy()
    q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    return p, q_wxyz


def invert_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    Rm = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def compose_T(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def make_T(position: np.ndarray, rot: Rotation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = position
    return T


def canonical_name(name: str) -> str:
    name = name.strip().lstrip("#").strip().lower()
    name = re.sub(r"\[[^\]]*\]", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def split_fields(line: str) -> List[str]:
    line = line.strip()
    if "," in line:
        return [item.strip() for item in line.split(",") if item.strip()]
    return [item for item in re.split(r"\s+", line) if item]


def is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def read_table(path: Path) -> Tuple[Optional[List[str]], np.ndarray]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                candidate = line.lstrip("#").strip()
                lower = candidate.lower()
                if not rows and ("timestamp" in lower or lower.startswith("time")):
                    line = candidate
                else:
                    continue
            fields = split_fields(line)
            if fields:
                rows.append(fields)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    has_header = any(not is_float_token(token) for token in rows[0])
    if has_header:
        header = [canonical_name(item) for item in rows[0]]
        data_rows = rows[1:]
    else:
        header = None
        data_rows = rows

    if not data_rows:
        raise ValueError(f"No numeric data rows found in {path}")

    width = len(data_rows[0])
    numeric: List[List[float]] = []
    for row_idx, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) < width:
            raise ValueError(f"Row {row_idx} in {path} has too few columns")
        try:
            values = [float(item) for item in row[:width]]
        except ValueError as exc:
            raise ValueError(f"Failed to parse numeric row {row_idx} in {path}: {row}") from exc
        if not np.all(np.isfinite(values)):
            continue
        numeric.append(values)

    if not numeric:
        raise ValueError(f"No finite numeric rows found in {path}")
    return header, np.asarray(numeric, dtype=float)


def find_column(header: Sequence[str], aliases: Sequence[str], label: str) -> int:
    alias_set = {canonical_name(item) for item in aliases}
    for idx, name in enumerate(header):
        if name in alias_set:
            return idx
    raise ValueError(f"Could not find {label} column. Header columns: {list(header)}")


def pose_column_indices(header: Optional[List[str]], quat_order: str, path: Path) -> Tuple[int, List[int], List[int]]:
    if header is None:
        return 0, [1, 2, 3], [4, 5, 6, 7]

    t_idx = find_column(
        header,
        ["timestamp", "time", "t", "ts", "timestampns", "timens", "body0timestamp"],
        "timestamp",
    )
    px = find_column(
        header,
        ["px", "positionx", "posx", "x", "tx", "p0", "prsrx", "pwbx", "body0posx", "body0posxmm"],
        "position x",
    )
    py = find_column(
        header,
        ["py", "positiony", "posy", "y", "ty", "p1", "prsry", "pwby", "body0posy", "body0posymm"],
        "position y",
    )
    pz = find_column(
        header,
        ["pz", "positionz", "posz", "z", "tz", "p2", "prsrz", "pwbz", "body0posz", "body0poszmm"],
        "position z",
    )

    q_aliases = {
        "x": ["qx", "quatx", "quaternionx", "orientationx", "ox", "qrsx", "qwbx", "body0quax"],
        "y": ["qy", "quaty", "quaterniony", "orientationy", "oy", "qrsy", "qwby", "body0quay"],
        "z": ["qz", "quatz", "quaternionz", "orientationz", "oz", "qrsz", "qwbz", "body0quaz"],
        "w": [
            "qw",
            "qr",
            "quatw",
            "quatr",
            "quaternionw",
            "quaternionr",
            "orientationw",
            "ow",
            "qrsw",
            "qwbw",
            "body0quar",
        ],
    }
    try:
        qx = find_column(header, q_aliases["x"], "quaternion x")
        qy = find_column(header, q_aliases["y"], "quaternion y")
        qz = find_column(header, q_aliases["z"], "quaternion z")
        qw = find_column(header, q_aliases["w"], "quaternion w")
        return t_idx, [px, py, pz], [qx, qy, qz, qw] if quat_order == "xyzw" else [qw, qx, qy, qz]
    except ValueError:
        generic = []
        for aliases in [
            ["q0", "quat0", "quaternion0"],
            ["q1", "quat1", "quaternion1"],
            ["q2", "quat2", "quaternion2"],
            ["q3", "quat3", "quaternion3"],
        ]:
            generic.append(find_column(header, aliases, "generic quaternion component"))
        return t_idx, [px, py, pz], generic


def time_column_index(header: Optional[List[str]]) -> int:
    if header is None:
        return 0
    return find_column(header, ["timestamp", "time", "t", "ts", "timestampns", "timens", "body0timestamp"], "timestamp")


def position_scale_from_unit(header: Optional[List[str]], p_idx: Sequence[int], unit: str, name: str) -> float:
    if unit == "m":
        return 1.0
    if unit == "mm":
        return 1e-3
    if unit != "auto":
        raise ValueError(f"Unsupported position unit for {name}: {unit}")
    if header is not None:
        selected = [header[i] for i in p_idx]
        if any(col.endswith("mm") or "mm" in col for col in selected):
            print(f"{name}: auto-detected position unit mm; converting positions to meters.")
            return 1e-3
    return 1.0


def unit_scale(unit: str) -> float:
    if unit == "s":
        return 1.0
    if unit == "ms":
        return 1e-3
    if unit == "ns":
        return 1e-9
    raise ValueError(f"Unsupported timestamp unit: {unit}")


def ensure_strictly_increasing(t: np.ndarray, name: str) -> None:
    if len(t) < 2:
        raise ValueError(f"{name} has fewer than two timestamps")
    diffs = np.diff(t)
    if not np.all(diffs > 0):
        bad = int(np.where(diffs <= 0)[0][0])
        raise ValueError(f"{name} timestamps must be strictly increasing; first bad index {bad}")


def check_quaternions(quats_xyzw: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(quats_xyzw, axis=1)
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    max_dev = float(np.max(np.abs(norms - 1.0)))
    if min_norm < 1e-8:
        raise ValueError(f"{name} contains a near-zero quaternion")
    if max_dev > 1e-3:
        print(
            f"WARNING: {name} quaternion norm max deviation is {max_dev:.3e}; "
            "quaternions will be normalized.",
            file=sys.stderr,
        )
    quats_xyzw = quats_xyzw / norms[:, None]
    print(f"{name}: quaternion norm range before normalization [{min_norm:.9f}, {max_norm:.9f}]")
    return quats_xyzw


def check_rotation_determinants(rot: Rotation, name: str) -> None:
    dets = np.linalg.det(rot.as_matrix())
    max_dev = float(np.max(np.abs(dets - 1.0)))
    if max_dev > 1e-6:
        print(f"WARNING: {name} rotation determinant max deviation is {max_dev:.3e}", file=sys.stderr)


def read_pose_trajectory(
    path: Path,
    timestamp_unit: str,
    quat_order: str,
    name: str,
    position_unit: str = "m",
) -> PoseTrajectory:
    header, data = read_table(path)
    t_idx, p_idx, q_idx = pose_column_indices(header, quat_order, path)
    if data.shape[1] <= max([t_idx] + p_idx + q_idx):
        raise ValueError(f"{path} does not contain enough columns for pose parsing")

    t = data[:, t_idx] * unit_scale(timestamp_unit)
    p_scale = position_scale_from_unit(header, p_idx, position_unit, name)
    p = data[:, p_idx] * p_scale
    q_raw = data[:, q_idx]
    if quat_order == "xyzw":
        q_xyzw = q_raw
    else:
        q_xyzw = q_raw[:, [1, 2, 3, 0]]
    q_xyzw = check_quaternions(q_xyzw, name)
    ensure_strictly_increasing(t, name)
    rot = Rotation.from_quat(q_xyzw)
    check_rotation_determinants(rot, name)
    return PoseTrajectory(name=name, t=t, p=p, rot=rot)


def read_imu_timestamps(path: Path, timestamp_unit: str) -> np.ndarray:
    header, data = read_table(path)
    idx = time_column_index(header)
    if data.shape[1] <= idx:
        raise ValueError(f"{path} does not contain timestamp column {idx}")
    t = data[:, idx] * unit_scale(timestamp_unit)
    ensure_strictly_increasing(t, "imu")
    return t


def interpolate_trajectory(traj: PoseTrajectory, query_t: np.ndarray, name: Optional[str] = None) -> PoseTrajectory:
    if len(query_t) == 0:
        raise ValueError("No query timestamps for interpolation")
    if query_t[0] < traj.t[0] - 1e-12 or query_t[-1] > traj.t[-1] + 1e-12:
        raise ValueError(
            f"Interpolation requested outside {traj.name} range: "
            f"[{query_t[0]:.9f}, {query_t[-1]:.9f}] vs [{traj.t[0]:.9f}, {traj.t[-1]:.9f}]"
        )
    p_interp = np.column_stack([np.interp(query_t, traj.t, traj.p[:, axis]) for axis in range(3)])
    origin = traj.t[0]
    slerp = Slerp(traj.t - origin, traj.rot)
    rot_interp = slerp(query_t - origin)
    return PoseTrajectory(name=name or f"{traj.name}_interp", t=query_t.copy(), p=p_interp, rot=rot_interp)


def subset_trajectory(traj: PoseTrajectory, mask: np.ndarray, name: Optional[str] = None) -> PoseTrajectory:
    return PoseTrajectory(name=name or traj.name, t=traj.t[mask], p=traj.p[mask], rot=traj.rot[mask])


def downsample_indices(n: int, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n, dtype=int)
    return np.unique(np.round(np.linspace(0, n - 1, max_samples)).astype(int))


def sync_for_calibration(
    mocap: PoseTrajectory,
    vio: PoseTrajectory,
    time_offset: float,
    min_samples: int,
    max_samples: int,
) -> CalibrationData:
    mocap_corrected = PoseTrajectory(
        name=f"{mocap.name}_time_corrected",
        t=mocap.t + time_offset,
        p=mocap.p,
        rot=mocap.rot,
    )
    t_start = max(float(mocap_corrected.t[0]), float(vio.t[0]))
    t_end = min(float(mocap_corrected.t[-1]), float(vio.t[-1]))
    if not t_start < t_end:
        raise ValueError(
            f"No valid overlap after time_offset={time_offset:.9f}: "
            f"mocap [{mocap_corrected.t[0]:.9f}, {mocap_corrected.t[-1]:.9f}], "
            f"vio [{vio.t[0]:.9f}, {vio.t[-1]:.9f}]"
        )

    mask = (vio.t >= t_start) & (vio.t <= t_end)
    idx = np.where(mask)[0]
    if len(idx) < min_samples:
        raise ValueError(f"Only {len(idx)} synchronized samples; need at least {min_samples}")
    idx = idx[downsample_indices(len(idx), max_samples)]
    times = vio.t[idx]
    mocap_interp = interpolate_trajectory(mocap_corrected, times, name="mocap_interp_T_WM")
    vio_sub = PoseTrajectory(name="vio_T_OB", t=times.copy(), p=vio.p[idx], rot=vio.rot[idx])
    return CalibrationData(times=times, mocap=mocap_interp, vio=vio_sub)


def initial_T_WO(data: CalibrationData, T_MB_init: np.ndarray) -> np.ndarray:
    T_WM0 = make_T(data.mocap.p[0], data.mocap.rot[0])
    T_OB0 = make_T(data.vio.p[0], data.vio.rot[0])
    return T_WM0 @ T_MB_init @ invert_T(T_OB0)


def residual_vector_for_transforms(data: CalibrationData, T_MB: np.ndarray, T_WO: np.ndarray) -> np.ndarray:
    R_MB = T_MB[:3, :3]
    p_MB = T_MB[:3, 3]
    R_WO = T_WO[:3, :3]
    p_WO = T_WO[:3, 3]
    R_WM_all = data.mocap.rot.as_matrix()
    R_OB_all = data.vio.rot.as_matrix()

    residuals = np.empty((len(data.times), 6), dtype=float)
    for i in range(len(data.times)):
        R_left = R_WM_all[i] @ R_MB
        p_left = R_WM_all[i] @ p_MB + data.mocap.p[i]
        R_right = R_WO @ R_OB_all[i]
        p_right = R_WO @ data.vio.p[i] + p_WO

        E = np.eye(4)
        E[:3, :3] = R_left.T @ R_right
        E[:3, 3] = R_left.T @ (p_right - p_left)
        residuals[i] = se3_log(E)
    return residuals.reshape(-1)


def estimate_transforms(
    data: CalibrationData,
    time_offset: float,
    robust_loss: str,
    f_scale: float,
    max_nfev: int,
    verbose: int = 0,
) -> CalibrationResult:
    T_MB_init = np.eye(4)
    T_WO_init = initial_T_WO(data, T_MB_init)

    def fun(x: np.ndarray) -> np.ndarray:
        T_MB = se3_exp(x[:6]) @ T_MB_init
        T_WO = se3_exp(x[6:]) @ T_WO_init
        return residual_vector_for_transforms(data, T_MB, T_WO)

    result = least_squares(
        fun,
        np.zeros(12, dtype=float),
        loss=robust_loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        verbose=verbose,
    )
    T_MB = se3_exp(result.x[:6]) @ T_MB_init
    T_WO = se3_exp(result.x[6:]) @ T_WO_init
    stats = compute_alignment_stats(data, T_MB, T_WO)
    return CalibrationResult(T_MB=T_MB, T_WO=T_WO, time_offset=time_offset, result=result, stats=stats, data=data)


def compute_alignment_stats(data: CalibrationData, T_MB: np.ndarray, T_WO: np.ndarray) -> Dict[str, float]:
    body_gt = transform_by_right_extrinsic(data.mocap, T_MB, name="T_WM_T_MB")
    vio_aligned = transform_by_left(data.vio, T_WO, name="T_WO_T_OB")
    rel = relative_errors(body_gt, vio_aligned)
    trans = rel["translation_error_m"]
    rot = rel["rotation_error_deg"]
    se3_res = residual_vector_for_transforms(data, T_MB, T_WO)
    return {
        "se3_residual_rmse": float(np.sqrt(np.mean(se3_res * se3_res))),
        "translation_rmse_m": rmse(trans),
        "rotation_rmse_deg": rmse(rot),
        "translation_median_m": float(np.median(trans)),
        "rotation_median_deg": float(np.median(rot)),
        "translation_max_m": float(np.max(trans)),
        "rotation_max_deg": float(np.max(rot)),
    }


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def transform_by_right_extrinsic(traj: PoseTrajectory, T_right: np.ndarray, name: str) -> PoseTrajectory:
    R_right = Rotation.from_matrix(T_right[:3, :3])
    p_right = T_right[:3, 3]
    rot_new = traj.rot * R_right
    p_new = traj.rot.apply(p_right) + traj.p
    return PoseTrajectory(name=name, t=traj.t.copy(), p=p_new, rot=rot_new)


def transform_by_left(traj: PoseTrajectory, T_left: np.ndarray, name: str) -> PoseTrajectory:
    R_left = Rotation.from_matrix(T_left[:3, :3])
    p_left = T_left[:3, 3]
    rot_new = R_left * traj.rot
    p_new = R_left.apply(traj.p) + p_left
    return PoseTrajectory(name=name, t=traj.t.copy(), p=p_new, rot=rot_new)


def relative_errors(a: PoseTrajectory, b: PoseTrajectory) -> Dict[str, np.ndarray]:
    if len(a) != len(b):
        raise ValueError("Trajectory lengths differ for relative error calculation")
    trans = np.linalg.norm(b.p - a.p, axis=1)
    rel_rot = a.rot.inv() * b.rot
    rot_deg = rel_rot.magnitude() * 180.0 / math.pi
    return {"translation_error_m": trans, "rotation_error_deg": rot_deg}


def estimate_or_use_time_offset(args, mocap: PoseTrajectory, vio: PoseTrajectory) -> CalibrationResult:
    if not args.estimate_time_offset:
        data = sync_for_calibration(mocap, vio, args.time_offset, args.min_samples, args.max_calib_samples)
        print_sync_summary(data, args.time_offset)
        return estimate_transforms(data, args.time_offset, args.robust_loss, args.loss_f_scale, args.max_nfev)

    offsets = np.arange(
        args.time_offset + args.time_offset_search_min,
        args.time_offset + args.time_offset_search_max + 0.5 * args.time_offset_search_step,
        args.time_offset_search_step,
    )
    best: Optional[CalibrationResult] = None
    print(
        "Searching time_offset over {} candidates: [{:.6f}, {:.6f}] step {:.6f}".format(
            len(offsets), float(offsets[0]), float(offsets[-1]), args.time_offset_search_step
        )
    )
    for offset in offsets:
        try:
            data = sync_for_calibration(mocap, vio, float(offset), args.min_samples, args.max_calib_samples)
            candidate = estimate_transforms(
                data,
                float(offset),
                args.robust_loss,
                args.loss_f_scale,
                args.time_offset_grid_max_nfev,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  offset {offset:.6f}: skipped ({exc})")
            continue
        score = candidate.stats["se3_residual_rmse"]
        print(
            "  offset {:+.6f}: se3_rmse={:.6f}, trans_rmse={:.6f} m, rot_rmse={:.3f} deg".format(
                offset,
                score,
                candidate.stats["translation_rmse_m"],
                candidate.stats["rotation_rmse_deg"],
            )
        )
        if best is None or score < best.stats["se3_residual_rmse"]:
            best = candidate

    if best is None:
        raise RuntimeError("No valid time_offset candidate produced enough synchronized samples")

    print("Best grid-search time_offset: {:+.9f}".format(best.time_offset))
    data = sync_for_calibration(mocap, vio, best.time_offset, args.min_samples, args.max_calib_samples)
    print_sync_summary(data, best.time_offset)
    return estimate_transforms(data, best.time_offset, args.robust_loss, args.loss_f_scale, args.max_nfev)


def print_sync_summary(data: CalibrationData, time_offset: float) -> None:
    print("Calibration synchronization:")
    print(f"  time_offset: {time_offset:+.9f} s")
    print(f"  samples used: {len(data.times)}")
    print(f"  first timestamp: {data.times[0]:.9f} s")
    print(f"  last timestamp:  {data.times[-1]:.9f} s")
    print(f"  duration:        {data.times[-1] - data.times[0]:.9f} s")


def valid_query_times(source: PoseTrajectory, query_t: np.ndarray) -> np.ndarray:
    return query_t[(query_t >= source.t[0]) & (query_t <= source.t[-1])]


def estimate_sample_rate(t: np.ndarray) -> Dict[str, float]:
    if len(t) < 2:
        return {"median_hz": 0.0, "mean_hz": 0.0, "min_dt_s": 0.0, "max_dt_s": 0.0}
    dt = np.diff(t)
    return {
        "median_hz": float(1.0 / np.median(dt)),
        "mean_hz": float(1.0 / np.mean(dt)),
        "min_dt_s": float(np.min(dt)),
        "max_dt_s": float(np.max(dt)),
    }


def format_rate(stats: Dict[str, float]) -> str:
    return (
        f"median {stats['median_hz']:.3f} Hz, mean {stats['mean_hz']:.3f} Hz, "
        f"dt range [{stats['min_dt_s']:.9f}, {stats['max_dt_s']:.9f}] s"
    )


def output_times_for_body_gt(args, body_world_at_mocap: PoseTrajectory, imu_t: np.ndarray) -> np.ndarray:
    t_start = float(body_world_at_mocap.t[0])
    t_end = float(body_world_at_mocap.t[-1])
    if len(imu_t) > 0:
        t_start = max(t_start, float(imu_t[0]))
        t_end = min(t_end, float(imu_t[-1]))
    if not t_start < t_end:
        raise ValueError(
            "No valid overlap between final body GT and IMU timestamps: "
            f"body [{body_world_at_mocap.t[0]:.9f}, {body_world_at_mocap.t[-1]:.9f}], "
            f"imu [{imu_t[0]:.9f}, {imu_t[-1]:.9f}]"
        )

    if args.output_timestamps == "imu":
        return valid_query_times(body_world_at_mocap, imu_t)
    if args.output_timestamps == "mocap":
        mask = (body_world_at_mocap.t >= t_start) & (body_world_at_mocap.t <= t_end)
        return body_world_at_mocap.t[mask]
    if args.output_timestamps == "rate":
        if args.output_rate_hz <= 0.0:
            raise ValueError("--output_rate_hz must be positive when --output_timestamps rate is used")
        dt = 1.0 / args.output_rate_hz
        count = int(math.floor((t_end - t_start) / dt)) + 1
        return t_start + np.arange(count, dtype=float) * dt
    raise ValueError(f"Unsupported output timestamp mode: {args.output_timestamps}")


def write_pose_csv(path: Path, traj: PoseTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"])
        q_xyzw = traj.rot.as_quat()
        q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
        for t, p, q in zip(traj.t, traj.p, q_wxyz):
            writer.writerow(
                [
                    f"{t:.9f}",
                    f"{p[0]:.12f}",
                    f"{p[1]:.12f}",
                    f"{p[2]:.12f}",
                    f"{q[0]:.12f}",
                    f"{q[1]:.12f}",
                    f"{q[2]:.12f}",
                    f"{q[3]:.12f}",
                ]
            )


def write_debias_imu_gt_csv(path: Path, traj: PoseTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "#timestamp [ns]",
                "p_RS_R_x [m]",
                "p_RS_R_y [m]",
                "p_RS_R_z [m]",
                "q_RS_w []",
                "q_RS_x []",
                "q_RS_y []",
                "q_RS_z []",
            ]
        )
        q_xyzw = traj.rot.as_quat()
        q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
        for t, p, q in zip(traj.t, traj.p, q_wxyz):
            timestamp_ns = int(round(float(t) * 1e9))
            writer.writerow(
                [
                    str(timestamp_ns),
                    f"{p[0]:.12f}",
                    f"{p[1]:.12f}",
                    f"{p[2]:.12f}",
                    f"{q[0]:.12f}",
                    f"{q[1]:.12f}",
                    f"{q[2]:.12f}",
                    f"{q[3]:.12f}",
                ]
            )


def normalize_first_frame(body_world: PoseTrajectory) -> Tuple[PoseTrajectory, Dict[str, float]]:
    T0 = make_T(body_world.p[0], body_world.rot[0])
    T0_inv = invert_T(T0)
    normalized = transform_by_left(body_world, T0_inv, name="body_gt_firstframe")
    first_T = make_T(normalized.p[0], normalized.rot[0])
    first_rot_deg = Rotation.from_matrix(first_T[:3, :3]).magnitude() * 180.0 / math.pi
    checks = {
        "first_translation_norm_m": float(np.linalg.norm(first_T[:3, 3])),
        "first_rotation_angle_deg": float(first_rot_deg),
    }
    return normalized, checks


def export_body_gt(args, mocap: PoseTrajectory, T_MB: np.ndarray, time_offset: float, imu_t: np.ndarray):
    mocap_corrected = PoseTrajectory(
        name="mocap_corrected_T_WM",
        t=mocap.t + time_offset,
        p=mocap.p,
        rot=mocap.rot,
    )
    body_world_at_mocap = transform_by_right_extrinsic(mocap_corrected, T_MB, name="body_gt_world_at_mocap")
    output_t = output_times_for_body_gt(args, body_world_at_mocap, imu_t)
    if len(output_t) < args.min_samples:
        raise ValueError(
            f"Only {len(output_t)} output timestamps overlap final body GT range; need at least {args.min_samples}"
        )
    if args.output_timestamps == "mocap":
        mask = np.isin(body_world_at_mocap.t, output_t)
        body_world = subset_trajectory(body_world_at_mocap, mask, name="body_gt_world")
    else:
        body_world = interpolate_trajectory(body_world_at_mocap, output_t, name="body_gt_world")
    write_pose_csv(args.output_dir / "body_gt_world.csv", body_world)
    if args.write_debias_imu_csv:
        write_debias_imu_gt_csv(args.output_dir / "debias_imu" / "data.csv", body_world)
    output_rate_stats = estimate_sample_rate(body_world.t)
    print(f"Body GT output timestamp mode: {args.output_timestamps}")
    print(f"Body GT output sample rate: {format_rate(output_rate_stats)}")

    body_first = None
    first_checks = None
    if args.normalize_first_frame:
        body_first, first_checks = normalize_first_frame(body_world)
        write_pose_csv(args.output_dir / "body_gt_firstframe.csv", body_first)
        if args.write_debias_imu_csv:
            write_debias_imu_gt_csv(args.output_dir / "debias_imu_firstframe" / "data.csv", body_first)
        if first_checks["first_translation_norm_m"] > 1e-8 or first_checks["first_rotation_angle_deg"] > 1e-6:
            print(f"WARNING: first-frame normalization check is not exactly identity: {first_checks}", file=sys.stderr)

    return body_world, body_first, first_checks, output_rate_stats


def plot_alignment(output_dir: Path, result: CalibrationResult) -> Dict[str, Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    body_gt = transform_by_right_extrinsic(result.data.mocap, result.T_MB, name="T_WM_T_MB")
    vio_aligned = transform_by_left(result.data.vio, result.T_WO, name="T_WO_T_OB")
    err = relative_errors(body_gt, vio_aligned)
    t_rel = result.data.times - result.data.times[0]
    paths: Dict[str, Path] = {}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(body_gt.p[:, 0], body_gt.p[:, 1], body_gt.p[:, 2], label="T_WM * T_MB")
    ax.plot(vio_aligned.p[:, 0], vio_aligned.p[:, 1], vio_aligned.p[:, 2], label="T_WO * T_OB_vio")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()
    paths["alignment_position_3d"] = plot_dir / "alignment_position_3d.png"
    fig.savefig(paths["alignment_position_3d"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ["x", "y", "z"]
    for axis in range(3):
        axes[axis].plot(t_rel, body_gt.p[:, axis], label="T_WM * T_MB")
        axes[axis].plot(t_rel, vio_aligned.p[:, axis], label="T_WO * T_OB_vio", linestyle="--")
        axes[axis].set_ylabel(f"{labels[axis]} [m]")
        axes[axis].grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("time since first calibration sample [s]")
    fig.tight_layout()
    paths["alignment_position_xyz"] = plot_dir / "alignment_position_xyz.png"
    fig.savefig(paths["alignment_position_xyz"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_rel, err["translation_error_m"])
    axes[0].set_ylabel("translation error [m]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_rel, err["rotation_error_deg"])
    axes[1].set_ylabel("rotation error [deg]")
    axes[1].set_xlabel("time since first calibration sample [s]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    paths["alignment_error"] = plot_dir / "alignment_error.png"
    fig.savefig(paths["alignment_error"], dpi=180)
    plt.close(fig)
    return paths


def plot_body_gt(output_dir: Path, body_world: PoseTrajectory, body_first: Optional[PoseTrajectory]) -> Dict[str, Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    def plot_xyz(traj: PoseTrajectory, filename: str, title: str) -> Path:
        t_rel = traj.t - traj.t[0]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for axis, label in enumerate(["x", "y", "z"]):
            axes[axis].plot(t_rel, traj.p[:, axis])
            axes[axis].set_ylabel(f"{label} [m]")
            axes[axis].grid(True, alpha=0.3)
        axes[0].set_title(title)
        axes[-1].set_xlabel("time since first output sample [s]")
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    paths["body_gt_world_position"] = plot_xyz(
        body_world, "body_gt_world_position.png", "Body GT in mocap/world frame W"
    )
    if body_first is not None:
        paths["body_gt_firstframe_position"] = plot_xyz(
            body_first, "body_gt_firstframe_position.png", "Body GT normalized to first frame"
        )
    return paths


def yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    if value is None:
        return "null"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def yaml_lines(value, indent: int = 0) -> List[str]:
    pad = " " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple, np.ndarray)):
                lines.append(f"{pad}{key}:")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, np.ndarray):
        return yaml_lines(value.tolist(), indent)
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{pad}[]"]
        if all(not isinstance(item, (dict, list, tuple, np.ndarray)) for item in value):
            return [f"{pad}[{', '.join(yaml_scalar(item) for item in value)}]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list, tuple, np.ndarray)):
                lines.append(f"{pad}-")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
        return lines
    return [f"{pad}{yaml_scalar(value)}"]


def write_calibration_yaml(
    path: Path,
    result: CalibrationResult,
    body_world: PoseTrajectory,
    first_checks: Optional[Dict[str, float]],
    args,
    output_rate_stats: Dict[str, float],
) -> None:
    R_WB0 = body_world.rot[0].as_matrix()
    data = {
        "model": "T_WM * T_MB ~= T_WO * T_OB_vio",
        "T_MB": result.T_MB,
        "T_WO": result.T_WO,
        "time_offset": result.time_offset,
        "residuals": result.stats,
        "samples": {
            "number_of_samples_used_for_calibration": int(len(result.data.times)),
            "first_timestamp_used": float(result.data.times[0]),
            "last_timestamp_used": float(result.data.times[-1]),
        },
        "body_gt_world": {
            "file": "body_gt_world.csv",
            "debias_imu_file": "debias_imu/data.csv" if args.write_debias_imu_csv else None,
            "meaning": "T_WB_gt with W equal to the mocap/world frame.",
            "timestamp_mode": args.output_timestamps,
            "output_rate_hz_requested": args.output_rate_hz if args.output_timestamps == "rate" else None,
            "detected_output_rate": output_rate_stats,
            "number_of_samples": int(len(body_world)),
            "first_timestamp": float(body_world.t[0]),
            "last_timestamp": float(body_world.t[-1]),
        },
        "first_frame_normalization": {
            "enabled": first_checks is not None,
            "file": "body_gt_firstframe.csv" if first_checks is not None else None,
            "debias_imu_file": "debias_imu_firstframe/data.csv"
            if first_checks is not None and args.write_debias_imu_csv
            else None,
            "checks": first_checks,
            "R_OW_for_gravity_if_normalized": R_WB0.T if first_checks is not None else None,
        },
        "gravity_reminder": (
            "If IMU preintegration runs in the first-frame world O, transform gravity as "
            "g_O = R_OW * g_W, where R_OW = R_WB(t0)^T. If mocap world z is up, "
            "usually g_W = [0, 0, -9.81]^T."
        ),
    }
    path.write_text("\n".join(yaml_lines(data)) + "\n", encoding="utf-8")


def print_stats(result: CalibrationResult, args) -> None:
    stats = result.stats
    print("Final residual statistics:")
    print(f"  translation RMSE:       {stats['translation_rmse_m']:.6f} m")
    print(f"  rotation RMSE:          {stats['rotation_rmse_deg']:.6f} deg")
    print(f"  median translation err: {stats['translation_median_m']:.6f} m")
    print(f"  median rotation err:    {stats['rotation_median_deg']:.6f} deg")
    print(f"  max translation err:    {stats['translation_max_m']:.6f} m")
    print(f"  max rotation err:       {stats['rotation_max_deg']:.6f} deg")
    if (
        stats["translation_rmse_m"] > args.warning_translation_rmse
        or stats["rotation_rmse_deg"] > args.warning_rotation_rmse_deg
    ):
        print(
            "WARNING: calibration may be poor due to VIO drift, wrong time offset, "
            "wrong quaternion order, or wrong frame convention.",
            file=sys.stderr,
        )


def check_transform(T: np.ndarray, name: str) -> None:
    det = float(np.linalg.det(T[:3, :3]))
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f"{name} has invalid rotation determinant: {det}")


def print_matrix(name: str, T: np.ndarray) -> None:
    print(f"{name}:")
    for row in T:
        print("  " + " ".join(f"{value: .9f}" for value in row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate T_MB from mocap marker pose and VIO body trajectory, then export body GT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python tools/estimate_mocap_body_gt.py \\
      --mocap_csv data/mocap/rigid_body_pose.csv \\
      --vio_csv data/vio/body_trajectory.csv \\
      --imu_csv data/imu/data.csv \\
      --output_dir output/body_gt \\
      --timestamp_unit ns \\
      --mocap_quat_order xyzw \\
      --vio_quat_order wxyz \\
      --output_timestamps rate \\
      --output_rate_hz 120 \\
      --write_debias_imu_csv \\
      --normalize_first_frame
""",
    )
    parser.add_argument("--mocap_csv", required=True, type=Path)
    parser.add_argument("--vio_csv", required=True, type=Path)
    parser.add_argument("--imu_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--timestamp_unit", choices=["s", "ms", "ns"], default="s")
    parser.add_argument(
        "--mocap_timestamp_unit",
        choices=["s", "ms", "ns"],
        default=None,
        help="Override timestamp unit for --mocap_csv. Defaults to --timestamp_unit.",
    )
    parser.add_argument(
        "--vio_timestamp_unit",
        choices=["s", "ms", "ns"],
        default=None,
        help="Override timestamp unit for --vio_csv. Defaults to --timestamp_unit.",
    )
    parser.add_argument(
        "--imu_timestamp_unit",
        choices=["s", "ms", "ns"],
        default=None,
        help="Override timestamp unit for --imu_csv. Defaults to --timestamp_unit.",
    )
    parser.add_argument("--mocap_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--vio_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument(
        "--mocap_position_unit",
        choices=["auto", "m", "mm"],
        default="auto",
        help="Position unit for mocap poses. auto detects LIASuw Body_0_Pos_*(mm) headers.",
    )
    parser.add_argument(
        "--vio_position_unit",
        choices=["auto", "m", "mm"],
        default="m",
        help="Position unit for VIO trajectory poses.",
    )
    parser.add_argument(
        "--output_timestamps",
        choices=["imu", "mocap", "rate"],
        default="imu",
        help=(
            "Timestamp grid for body_gt_world.csv. imu preserves the original requested behavior; "
            "mocap keeps mocap/GT timestamps in the IMU overlap; rate resamples to --output_rate_hz."
        ),
    )
    parser.add_argument(
        "--output_rate_hz",
        type=float,
        default=120.0,
        help="Fixed output rate used when --output_timestamps rate is selected.",
    )
    parser.add_argument(
        "--time_offset",
        type=float,
        default=0.0,
        help="Apply t_mocap_corrected = t_mocap + time_offset before synchronization.",
    )
    parser.add_argument("--estimate_time_offset", action="store_true")
    parser.add_argument(
        "--time_offset_search_min",
        type=float,
        default=-0.1,
        help="Grid-search lower bound relative to --time_offset when --estimate_time_offset is set.",
    )
    parser.add_argument(
        "--time_offset_search_max",
        type=float,
        default=0.1,
        help="Grid-search upper bound relative to --time_offset when --estimate_time_offset is set.",
    )
    parser.add_argument("--time_offset_search_step", type=float, default=0.005)
    parser.add_argument("--normalize_first_frame", action="store_true")
    parser.add_argument(
        "--write_debias_imu_csv",
        action="store_true",
        help="Also write Debias_IMU/TUM-VI-style data.csv files with ns timestamps and q_w q_x q_y q_z.",
    )
    parser.add_argument(
        "--robust_loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="huber",
    )
    parser.add_argument("--loss_f_scale", type=float, default=1.0)
    parser.add_argument("--max_nfev", type=int, default=200)
    parser.add_argument("--time_offset_grid_max_nfev", type=int, default=80)
    parser.add_argument("--max_calib_samples", type=int, default=2000)
    parser.add_argument("--min_samples", type=int, default=20)
    parser.add_argument("--warning_translation_rmse", type=float, default=0.10)
    parser.add_argument("--warning_rotation_rmse_deg", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_scipy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mocap_timestamp_unit = args.mocap_timestamp_unit or args.timestamp_unit
    vio_timestamp_unit = args.vio_timestamp_unit or args.timestamp_unit
    imu_timestamp_unit = args.imu_timestamp_unit or args.timestamp_unit

    for path in [args.mocap_csv, args.vio_csv, args.imu_csv]:
        if not path.exists():
            raise FileNotFoundError(path)

    mocap = read_pose_trajectory(
        args.mocap_csv,
        mocap_timestamp_unit,
        args.mocap_quat_order,
        "mocap",
        position_unit=args.mocap_position_unit,
    )
    vio = read_pose_trajectory(
        args.vio_csv,
        vio_timestamp_unit,
        args.vio_quat_order,
        "vio",
        position_unit=args.vio_position_unit,
    )
    imu_t = read_imu_timestamps(args.imu_csv, imu_timestamp_unit)
    print(f"Loaded mocap poses: {len(mocap)} [{mocap.t[0]:.9f}, {mocap.t[-1]:.9f}] s")
    print(f"  mocap sample rate: {format_rate(estimate_sample_rate(mocap.t))}")
    print(f"Loaded VIO poses:   {len(vio)} [{vio.t[0]:.9f}, {vio.t[-1]:.9f}] s")
    print(f"  VIO sample rate:   {format_rate(estimate_sample_rate(vio.t))}")
    print(f"Loaded IMU stamps:  {len(imu_t)} [{imu_t[0]:.9f}, {imu_t[-1]:.9f}] s")
    print(f"  IMU sample rate:   {format_rate(estimate_sample_rate(imu_t))}")

    result = estimate_or_use_time_offset(args, mocap, vio)
    check_transform(result.T_MB, "T_MB")
    check_transform(result.T_WO, "T_WO")

    body_world, body_first, first_checks, output_rate_stats = export_body_gt(
        args, mocap, result.T_MB, result.time_offset, imu_t
    )
    alignment_plots = plot_alignment(args.output_dir, result)
    body_plots = plot_body_gt(args.output_dir, body_world, body_first)
    write_calibration_yaml(
        args.output_dir / "calibration_result.yaml",
        result,
        body_world,
        first_checks,
        args,
        output_rate_stats,
    )

    print_matrix("Estimated T_MB (B to M, IMU/body frame into mocap marker frame)", result.T_MB)
    print_matrix("Estimated T_WO (O to W, VIO world into mocap world)", result.T_WO)
    print_stats(result, args)
    print("Output files:")
    print(f"  {args.output_dir / 'body_gt_world.csv'}")
    if args.normalize_first_frame:
        print(f"  {args.output_dir / 'body_gt_firstframe.csv'}")
    if args.write_debias_imu_csv:
        print(f"  {args.output_dir / 'debias_imu' / 'data.csv'}")
        if args.normalize_first_frame:
            print(f"  {args.output_dir / 'debias_imu_firstframe' / 'data.csv'}")
    print(f"  {args.output_dir / 'calibration_result.yaml'}")
    for path in {**alignment_plots, **body_plots}.values():
        print(f"  {path}")
    print("")
    print("Meaning:")
    print("  T_MB maps IMU/body-frame coordinates into the mocap rigid-body marker frame M.")
    print(
        "  body_gt_world.csv stores T_WB_gt with W equal to mocap/world; "
        f"timestamp mode is {args.output_timestamps}."
    )
    print("  body_gt_firstframe.csv stores inv(T_WB_gt(t0)) * T_WB_gt(t), if requested.")
    print("  If preintegration uses the first-frame world O, transform gravity as g_O = R_OW * g_W,")
    print("  where R_OW = R_WB(t0)^T. If mocap z is up, usually g_W = [0, 0, -9.81]^T.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
