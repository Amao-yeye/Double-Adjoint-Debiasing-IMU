#!/usr/bin/env python3
"""Estimate LIASuw per-sequence gravity direction for Debias_IMU supervision.

The fitted model matches the accelerometer pseudo-bias construction used by
run_liasuw_debias_ad_pipeline.py:

    a_m = R_WB^T (a_W - g_W) + b_a

where g_W is the gravity acceleration vector in the supervision/world frame.
For a z-up world this is close to [0, 0, -9.81].
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PLATO_ROOT / "results" / "debias_imu" / "LIASuw_training_candidates_200hz"
DEFAULT_OUTPUT = PLATO_ROOT / "results" / "debias_imu" / "LIASuw_gravity_calibration" / "gravity_calibration.json"
DEFAULT_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def read_numeric_csv(path: Path) -> np.ndarray:
    first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    skip = 1 if first.startswith("#") or any(ch.isalpha() for ch in first) else 0
    data = np.genfromtxt(path, delimiter=",", skip_header=skip)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data.astype(np.float64)


def selected_sequences(data_root: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    selected_path = data_root / "selected_sequences.txt"
    if selected_path.exists():
        return [line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sequences = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and not path.name.endswith("_firstframe") and (path / "mav0" / "imu0" / "data.csv").exists():
            sequences.append(path.name)
    return sequences


def make_savgol_window(n: int, requested: int, order: int) -> int:
    window = min(requested, n if n % 2 else n - 1)
    min_window = order + 2 if order % 2 else order + 3
    window = max(window, min_window)
    if window % 2 == 0:
        window += 1
    if window > n:
        raise ValueError(f"Not enough samples ({n}) for Savitzky-Golay order={order}")
    return window


def interp_pose(gt: np.ndarray, query_t_s: np.ndarray) -> tuple[np.ndarray, Rotation]:
    t_gt = gt[:, 0] * 1e-9
    unique_t, unique_idx = np.unique(t_gt, return_index=True)
    gt = gt[unique_idx]
    if query_t_s[0] < unique_t[0] or query_t_s[-1] > unique_t[-1]:
        raise ValueError("Query time is outside GT time range.")
    p = np.column_stack([np.interp(query_t_s, unique_t, gt[:, i]) for i in range(1, 4)])
    rot = Rotation.from_quat(gt[:, [5, 6, 7, 4]])
    return p, Slerp(unique_t, rot)(query_t_s)


def prepare_accel_system(
    sequence_dir: Path,
    dt: float,
    sg_window: int,
    sg_order: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    path_imu = sequence_dir / "mav0" / "imu0" / "data.csv"
    path_gt = sequence_dir / "mav0" / "mocap0" / "data.csv"
    imu = read_numeric_csv(path_imu)
    gt = read_numeric_csv(path_gt)

    t0_ns = max(imu[0, 0], gt[0, 0])
    t1_ns = min(imu[-1, 0], gt[-1, 0])
    imu = imu[(imu[:, 0] >= t0_ns) & (imu[:, 0] <= t1_ns)]
    if imu.shape[0] < 4:
        raise ValueError(f"{sequence_dir.name} has too few overlapping IMU samples.")

    imu_t_rel = imu[:, 0] * 1e-9
    imu_t_rel = imu_t_rel - imu_t_rel[0]
    ts = dt * np.arange(imu.shape[0], dtype=np.float64)
    ts = ts[ts <= imu_t_rel[-1]]
    query_t_s = imu[0, 0] * 1e-9 + ts
    query_t_s = query_t_s[(query_t_s >= gt[0, 0] * 1e-9) & (query_t_s <= gt[-1, 0] * 1e-9)]
    ts = query_t_s - query_t_s[0]

    imu_resampled = np.zeros((query_t_s.shape[0], 6), dtype=np.float64)
    for axis in range(6):
        imu_resampled[:, axis] = np.interp(ts, imu_t_rel, imu[:, 1 + axis])

    p_gt, rot_gt = interp_pose(gt, query_t_s)
    window = make_savgol_window(p_gt.shape[0], sg_window, sg_order)
    v_gt = savgol_filter(p_gt, window_length=window, polyorder=sg_order, deriv=1, axis=0, mode="interp") / dt
    a_gt = (v_gt[1:] - v_gt[:-1]) / dt

    rot_used = rot_gt[:-1]
    acc_meas = imu_resampled[:-1, 3:6]
    rt_mats = rot_used.inv().as_matrix()
    rt_a = rot_used.inv().apply(a_gt)
    lhs = acc_meas - rt_a
    metadata = {
        "samples": int(lhs.shape[0]),
        "duration_s": float(query_t_s[-1] - query_t_s[0]),
        "dt_s": float(dt),
        "sg_window": int(window),
        "sg_order": int(sg_order),
        "start_time_s": float(query_t_s[0]),
        "end_time_s": float(query_t_s[-1]),
    }
    return lhs, rt_mats, metadata


def rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def constant_bias_for_gravity(lhs: np.ndarray, rt_mats: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    rt_g = np.einsum("nij,j->ni", rt_mats, gravity)
    return np.mean(lhs + rt_g, axis=0)


def residual_for_gravity(lhs: np.ndarray, rt_mats: np.ndarray, gravity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bias = constant_bias_for_gravity(lhs, rt_mats, gravity)
    residual = lhs + np.einsum("nij,j->ni", rt_mats, gravity) - bias
    return residual, bias


def normalized(vec: np.ndarray, magnitude: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return DEFAULT_GRAVITY / np.linalg.norm(DEFAULT_GRAVITY) * magnitude
    return vec / norm * magnitude


def fit_gravity(lhs: np.ndarray, rt_mats: np.ndarray, gravity_mag: float, default_gravity: np.ndarray) -> dict[str, object]:
    default_residual, default_ba = residual_for_gravity(lhs, rt_mats, default_gravity)

    a_mat = np.concatenate([-rt_mats, np.broadcast_to(np.eye(3), rt_mats.shape)], axis=2).reshape(-1, 6)
    x_lsq, *_ = np.linalg.lstsq(a_mat, lhs.reshape(-1), rcond=None)
    lsq_gravity = x_lsq[:3]

    starts = [
        default_gravity,
        lsq_gravity,
        np.array([gravity_mag, 0.0, 0.0]),
        np.array([-gravity_mag, 0.0, 0.0]),
        np.array([0.0, gravity_mag, 0.0]),
        np.array([0.0, -gravity_mag, 0.0]),
        np.array([0.0, 0.0, gravity_mag]),
        np.array([0.0, 0.0, -gravity_mag]),
    ]

    def objective(g: np.ndarray) -> float:
        residual, _ = residual_for_gravity(lhs, rt_mats, g)
        return float(np.mean(np.sum(residual * residual, axis=1)))

    constraint = {"type": "eq", "fun": lambda g: float(np.dot(g, g) - gravity_mag * gravity_mag)}
    best_result = None
    for start in starts:
        start = normalized(np.asarray(start, dtype=np.float64), gravity_mag)
        result = minimize(
            objective,
            start,
            method="SLSQP",
            constraints=constraint,
            options={"maxiter": 300, "ftol": 1e-12, "disp": False},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError("Gravity optimization failed to run.")
    gravity = normalized(np.asarray(best_result.x, dtype=np.float64), gravity_mag)
    opt_residual, opt_ba = residual_for_gravity(lhs, rt_mats, gravity)
    cos_tilt = float(np.dot(gravity, default_gravity) / (np.linalg.norm(gravity) * np.linalg.norm(default_gravity)))
    cos_tilt = min(1.0, max(-1.0, cos_tilt))
    return {
        "gravity_accel_world_m_s2": gravity.tolist(),
        "gravity_specific_world_m_s2": (-gravity).tolist(),
        "ba_const_m_s2": opt_ba.tolist(),
        "default_gravity_accel_world_m_s2": default_gravity.tolist(),
        "default_ba_const_m_s2": default_ba.tolist(),
        "unconstrained_lsq_gravity_accel_world_m_s2": lsq_gravity.tolist(),
        "unconstrained_lsq_gravity_norm_m_s2": float(np.linalg.norm(lsq_gravity)),
        "default_accel_residual_rmse_m_s2": rmse(default_residual),
        "optimized_accel_residual_rmse_m_s2": rmse(opt_residual),
        "improvement_ratio": float(rmse(opt_residual) / max(rmse(default_residual), 1e-12)),
        "tilt_from_default_deg": float(math.degrees(math.acos(cos_tilt))),
        "optimizer_success": bool(best_result.success),
        "optimizer_message": str(best_result.message),
    }


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# LIASuw Route-A Gravity Calibration",
        "",
        f"- Data root: `{payload['metadata']['data_root']}`",
        f"- Gravity norm: `{payload['metadata']['gravity_mag_m_s2']:.6f} m/s^2`",
        f"- Default gravity acceleration: `{payload['metadata']['default_gravity_accel_world_m_s2']}`",
        "",
        "| sequence | samples | default residual | optimized residual | tilt deg | optimized g_W | ba const | default ba const |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for seq, item in payload["sequences"].items():
        meta = item["metadata"]
        lines.append(
            "| {seq} | {samples} | {default:.4f} | {opt:.4f} | {tilt:.2f} | `{g}` | `{ba}` | `{default_ba}` |".format(
                seq=seq,
                samples=meta["samples"],
                default=item["default_accel_residual_rmse_m_s2"],
                opt=item["optimized_accel_residual_rmse_m_s2"],
                tilt=item["tilt_from_default_deg"],
                g=[round(float(x), 5) for x in item["gravity_accel_world_m_s2"]],
                ba=[round(float(x), 5) for x in item["ba_const_m_s2"]],
                default_ba=[round(float(x), 5) for x in item["default_ba_const_m_s2"]],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--gravity-mag", type=float, default=9.81)
    parser.add_argument("--sg-window", type=int, default=21)
    parser.add_argument("--sg-order", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = selected_sequences(args.data_root, args.sequences)
    if not sequences:
        raise RuntimeError(f"No sequences found under {args.data_root}")

    default_gravity = normalized(DEFAULT_GRAVITY, args.gravity_mag)
    payload: dict[str, object] = {
        "metadata": {
            "data_root": str(args.data_root),
            "dt_s": args.dt,
            "gravity_mag_m_s2": args.gravity_mag,
            "default_gravity_accel_world_m_s2": default_gravity.tolist(),
            "model": "a_m = R_WB^T (a_W - g_W) + b_a",
        },
        "sequences": {},
    }

    for sequence in sequences:
        sequence_dir = args.data_root / sequence
        print(f"Estimating {sequence} ...", flush=True)
        lhs, rt_mats, metadata = prepare_accel_system(sequence_dir, args.dt, args.sg_window, args.sg_order)
        result = fit_gravity(lhs, rt_mats, args.gravity_mag, default_gravity)
        result["metadata"] = metadata
        payload["sequences"][sequence] = result
        print(
            "  default_residual={:.4f}, optimized_residual={:.4f}, tilt={:.2f} deg, ba={}".format(
                result["default_accel_residual_rmse_m_s2"],
                result["optimized_accel_residual_rmse_m_s2"],
                result["tilt_from_default_deg"],
                [round(float(x), 4) for x in result["ba_const_m_s2"]],
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    write_markdown(md_path, payload)
    print(f"Wrote {args.output}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
