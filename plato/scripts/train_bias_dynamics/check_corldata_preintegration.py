#!/usr/bin/env python3
"""Check corldata IMU/body trajectory consistency with high-rate IMU propagation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_training_pseudogt"
DEFAULT_PSEUDO_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "corldata"
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "results" / "debias_imu" / "corldata_preintegration_checks"
DEFAULT_SEQUENCES = ["S00", "S02", "S03", "S07"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--windows", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument(
        "--pose-source",
        choices=["vo_tum", "gt_tum", "mocap_csv"],
        default="vo_tum",
        help=(
            "Trajectory source for the pose supervision. vo_tum preserves the original "
            "VO pseudo-GT check, gt_tum reads debias_imu_body_gt_1000hz_gt/body_gt_world_imu_frame.tum, "
            "and mocap_csv reads <data-root>/<seq>/mav0/mocap0/data.csv."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, 0.0, -9.7803])
    parser.add_argument("--top-k", type=int, default=3, help="Number of high-displacement windows checked per sequence/window size.")
    parser.add_argument("--max-pose-dt-s", type=float, default=0.04)
    parser.add_argument("--max-speed-m-s", type=float, default=1.5)
    return parser.parse_args()


def hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=float)


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = hat(w)
    if theta < 1e-12:
        return np.eye(3) + W
    return np.eye(3) + math.sin(theta) / theta * W + (1.0 - math.cos(theta)) / (theta * theta) * (W @ W)


def so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.array([0.5 * (R[2, 1] - R[1, 2]), 0.5 * (R[0, 2] - R[2, 0]), 0.5 * (R[1, 0] - R[0, 1])])
    return theta / (2.0 * math.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0].astype(float)
    p = data[:, 1:4].astype(float)
    R = Rotation.from_quat(data[:, 4:8]).as_matrix()
    return t, p, R


def load_mocap_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0].astype(np.float64) * 1e-9
    p = data[:, 1:4].astype(float)
    q_wxyz = data[:, 4:8].astype(float)
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    R = Rotation.from_quat(q_xyzw).as_matrix()
    return t, p, R


def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0].astype(np.float64) * 1e-9
    u = data[:, 1:7].astype(float)
    return t, u


def odd_window(n: int, target: int, order: int) -> int:
    if n <= order + 2:
        return max(3, order + 2 + ((order + 2) % 2 == 0))
    window = min(target, n if n % 2 else n - 1)
    min_window = order + 2 if (order + 2) % 2 else order + 3
    return max(window, min_window)


def smooth(data: np.ndarray, target_window: int, order: int = 3) -> np.ndarray:
    if data.shape[0] < 9:
        return data.copy()
    window = odd_window(data.shape[0], target_window, order)
    return savgol_filter(data, window, order, axis=0, mode="interp")


def interp_columns(t_src: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    out = np.empty((t_query.shape[0], values.shape[1]), dtype=float)
    for axis in range(values.shape[1]):
        out[:, axis] = np.interp(t_query, t_src, values[:, axis])
    return out


def compute_pose_rates_and_biases(t_pose: np.ndarray, p: np.ndarray, R: np.ndarray, t_imu: np.ndarray, u_imu: np.ndarray, gravity: np.ndarray):
    p_smooth = smooth(p, 21, 3)
    v = np.gradient(p_smooth, t_pose, axis=0, edge_order=2)
    a_world = np.gradient(v, t_pose, axis=0, edge_order=2)
    imu_at_pose = interp_columns(t_imu, u_imu, t_pose)

    w_body = np.zeros((len(t_pose), 3), dtype=float)
    for idx in range(len(t_pose) - 1):
        dt = t_pose[idx + 1] - t_pose[idx]
        w_body[idx] = so3_log(R[idx].T @ R[idx + 1]) / dt
    w_body[-1] = w_body[-2]

    bg = imu_at_pose[:, :3] - w_body
    a_world_minus_g = a_world - gravity.reshape(1, 3)
    ba = imu_at_pose[:, 3:6] - np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), a_world_minus_g)
    bg = smooth(bg, 15, 3)
    ba = smooth(ba, 15, 3)
    return v, bg, ba


def choose_windows(t: np.ndarray, p: np.ndarray, v: np.ndarray, window: int, top_k: int, max_pose_dt: float, max_speed: float) -> list[int]:
    if len(p) < window:
        return []
    disp = np.linalg.norm(p[window - 1 :] - p[: len(p) - window + 1], axis=1)
    order = np.argsort(disp)[::-1]
    chosen: list[int] = []
    min_sep = max(1, window // 2)
    for idx in order:
        start = int(idx)
        stop = start + window
        if np.max(np.diff(t[start:stop])) > max_pose_dt:
            continue
        if np.max(np.linalg.norm(v[start:stop], axis=1)) > max_speed:
            continue
        if all(abs(start - other) >= min_sep for other in chosen):
            chosen.append(start)
        if len(chosen) >= top_k:
            break
    return sorted(chosen)


def make_grid(t_imu: np.ndarray, t_pose_window: np.ndarray) -> np.ndarray:
    t0 = t_pose_window[0]
    t1 = t_pose_window[-1]
    imu_inside = t_imu[(t_imu > t0) & (t_imu < t1)]
    grid = np.concatenate(([t0], imu_inside, t_pose_window[1:-1], [t1]))
    grid = np.unique(np.round(grid, 12))
    grid.sort()
    return grid


def propagate_on_grid(
    grid: np.ndarray,
    t_pose_window: np.ndarray,
    R0: np.ndarray,
    v0: np.ndarray,
    p0: np.ndarray,
    t_imu: np.ndarray,
    u_imu: np.ndarray,
    t_bias: np.ndarray,
    bg: np.ndarray,
    ba: np.ndarray,
    gravity: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_grid = interp_columns(t_imu, u_imu, grid)
    if mode == "raw":
        bg_grid = np.zeros((len(grid), 3), dtype=float)
        ba_grid = np.zeros((len(grid), 3), dtype=float)
    elif mode == "constant":
        bg0 = interp_columns(t_bias, bg, np.array([grid[0]]))[0]
        ba0 = interp_columns(t_bias, ba, np.array([grid[0]]))[0]
        bg_grid = np.repeat(bg0.reshape(1, 3), len(grid), axis=0)
        ba_grid = np.repeat(ba0.reshape(1, 3), len(grid), axis=0)
    elif mode == "pseudo_bias":
        bg_grid = interp_columns(t_bias, bg, grid)
        ba_grid = interp_columns(t_bias, ba, grid)
    else:
        raise ValueError(mode)

    R_states = [R0.copy()]
    v_states = [v0.copy()]
    p_states = [p0.copy()]
    for idx in range(len(grid) - 1):
        dt = grid[idx + 1] - grid[idx]
        omega = u_grid[idx, :3] - bg_grid[idx]
        acc_body = u_grid[idx, 3:6] - ba_grid[idx]
        acc_world = R_states[-1] @ acc_body + gravity
        R_next = R_states[-1] @ so3_exp(omega * dt)
        v_next = v_states[-1] + acc_world * dt
        p_next = p_states[-1] + v_states[-1] * dt + 0.5 * acc_world * dt * dt
        R_states.append(R_next)
        v_states.append(v_next)
        p_states.append(p_next)

    R_grid = np.stack(R_states, axis=0)
    v_grid = np.stack(v_states, axis=0)
    p_grid = np.stack(p_states, axis=0)
    pose_indices = np.searchsorted(grid, np.round(t_pose_window, 12))
    return R_grid[pose_indices], v_grid[pose_indices], p_grid[pose_indices]


def metrics(R_gt: np.ndarray, v_gt: np.ndarray, p_gt: np.ndarray, R_est: np.ndarray, v_est: np.ndarray, p_est: np.ndarray) -> dict[str, float]:
    pos = np.linalg.norm(p_est - p_gt, axis=1)
    vel = np.linalg.norm(v_est - v_gt, axis=1)
    rot = np.array([np.linalg.norm(so3_log(R_gt[i].T @ R_est[i])) for i in range(len(R_gt))], dtype=float) * 180.0 / math.pi
    return {
        "position_rmse_m": float(np.sqrt(np.mean(pos * pos))),
        "position_max_m": float(np.max(pos)),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(vel * vel))),
        "velocity_max_m_s": float(np.max(vel)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rot * rot))),
        "rotation_max_deg": float(np.max(rot)),
    }


def evaluate_sequence(args: argparse.Namespace, sequence: str) -> list[dict[str, object]]:
    imu_path = args.data_root / sequence / "mav0" / "imu0" / "data.csv"
    if not imu_path.exists():
        raise FileNotFoundError(imu_path)
    if args.pose_source == "vo_tum":
        pose_path = args.pseudo_root / sequence / "debias_imu_body_gt_200hz" / "vo_imu_traj.tum"
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)
        t_pose, p, R = load_tum(pose_path)
    elif args.pose_source == "gt_tum":
        pose_path = args.pseudo_root / sequence / "debias_imu_body_gt_1000hz_gt" / "body_gt_world_imu_frame.tum"
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)
        t_pose, p, R = load_tum(pose_path)
    elif args.pose_source == "mocap_csv":
        pose_path = args.data_root / sequence / "mav0" / "mocap0" / "data.csv"
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)
        t_pose, p, R = load_mocap_csv(pose_path)
    else:
        raise ValueError(args.pose_source)
    t_imu, u_imu = load_imu_csv(imu_path)
    gravity = np.asarray(args.gravity, dtype=float)
    v, bg, ba = compute_pose_rates_and_biases(t_pose, p, R, t_imu, u_imu, gravity)

    rows: list[dict[str, object]] = []
    for window in args.windows:
        for start in choose_windows(t_pose, p, v, window, args.top_k, args.max_pose_dt_s, args.max_speed_m_s):
            stop = start + window
            t_window = t_pose[start:stop]
            grid = make_grid(t_imu, t_window)
            base = {
                "sequence": sequence,
                "pose_source": args.pose_source,
                "window": window,
                "start_index": start,
                "start_time_s": float(t_window[0]),
                "duration_s": float(t_window[-1] - t_window[0]),
                "imu_steps": int(len(grid) - 1),
            }
            for mode in ["raw", "constant", "pseudo_bias"]:
                R_est, v_est, p_est = propagate_on_grid(
                    grid,
                    t_window,
                    R[start],
                    v[start],
                    p[start],
                    t_imu,
                    u_imu,
                    t_pose,
                    bg,
                    ba,
                    gravity,
                    mode,
                )
                row = dict(base)
                row["mode"] = mode
                row.update(metrics(R[start:stop], v[start:stop], p[start:stop], R_est, v_est, p_est))
                rows.append(row)
    return rows


def write_outputs(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["sequence"]), int(row["window"]), str(row["mode"])), []).append(row)
    lines = [
        "# corldata IMU Preintegration Consistency Check",
        "",
        "Each row aggregates the checked high-displacement windows. `pseudo_bias` uses biases differenced from the selected reference trajectory; `constant` holds the first pseudo-bias fixed; `raw` uses no bias correction.",
        f"Pose source: `{rows[0]['pose_source']}`.",
        "",
        "| sequence | window poses | mode | windows | pos RMSE mean/max m | rot RMSE mean/max deg | vel RMSE mean m/s |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for key in sorted(grouped):
        seq, window, mode = key
        vals = grouped[key]
        pos = np.array([float(row["position_rmse_m"]) for row in vals])
        rot = np.array([float(row["rotation_rmse_deg"]) for row in vals])
        vel = np.array([float(row["velocity_rmse_m_s"]) for row in vals])
        lines.append(
            f"| {seq} | {window} | {mode} | {len(vals)} | {pos.mean():.6f}/{pos.max():.6f} | "
            f"{rot.mean():.4f}/{rot.max():.4f} | {vel.mean():.6f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.pseudo_root = args.pseudo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    rows: list[dict[str, object]] = []
    for sequence in args.sequences:
        print(f"[preintegrate] {sequence}", flush=True)
        rows.extend(evaluate_sequence(args, sequence))
    if not rows:
        raise RuntimeError("No preintegration windows were evaluated.")
    write_outputs(args.output_dir, rows)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
