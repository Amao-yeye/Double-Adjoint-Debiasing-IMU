import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


TRAIN_SEQS = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]


@dataclass
class Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quats_wxyz: np.ndarray
    velocities: np.ndarray
    bw: np.ndarray
    ba: np.ndarray


def load_euroc_csv(path: str) -> Trajectory:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    return Trajectory(
        timestamps=data[:, 0].astype(np.int64),
        positions=data[:, 1:4].astype(np.float64),
        quats_wxyz=data[:, 4:8].astype(np.float64),
        velocities=data[:, 8:11].astype(np.float64),
        bw=data[:, 11:14].astype(np.float64),
        ba=data[:, 14:17].astype(np.float64),
    )


def nearest_neighbor_pairs(ts_est: np.ndarray, ts_gt: np.ndarray, max_diff_ns: int) -> list[tuple[int, int, int]]:
    pairs = []
    for est_idx, t_est in enumerate(ts_est):
        pos = np.searchsorted(ts_gt, t_est)
        candidates = []
        if pos < len(ts_gt):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue
        gt_idx = min(candidates, key=lambda idx: abs(int(ts_gt[idx]) - int(t_est)))
        dt = abs(int(ts_gt[gt_idx]) - int(t_est))
        if dt <= max_diff_ns:
            pairs.append((est_idx, gt_idx, dt))
    return pairs


def umeyama_se3(points_src: np.ndarray, points_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_src.shape != points_dst.shape or points_src.shape[1] != 3:
        raise ValueError("Expected matched Nx3 source and destination points for SE(3) alignment.")

    mu_src = points_src.mean(axis=0)
    mu_dst = points_dst.mean(axis=0)
    src_centered = points_src - mu_src
    dst_centered = points_dst - mu_dst

    cov = dst_centered.T @ src_centered / points_src.shape[0]
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R_align = U @ S @ Vt
    t_align = mu_dst - R_align @ mu_src
    return R_align, t_align


def pose_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = position
    return T


def origin_anchor_se3(gt: Trajectory, est: Trajectory, pairs: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    est_idx0, gt_idx0, _ = pairs[0]
    T_vo0 = pose_matrix(est.positions[est_idx0], est.quats_wxyz[est_idx0])
    T_gt0 = pose_matrix(gt.positions[gt_idx0], gt.quats_wxyz[gt_idx0])
    T_align = T_gt0 @ np.linalg.inv(T_vo0)
    return T_align[:3, :3], T_align[:3, 3]


def geodesic_deg(rot_err: R) -> np.ndarray:
    return np.degrees(rot_err.magnitude())


def evaluate_alignment(gt: Trajectory, est: Trajectory, pairs: list[tuple[int, int, int]], R_align: np.ndarray, t_align: np.ndarray) -> dict:
    est_idx = np.array([p[0] for p in pairs], dtype=int)
    gt_idx = np.array([p[1] for p in pairs], dtype=int)
    dt_ms = np.array([p[2] for p in pairs], dtype=np.float64) / 1e6

    p_est = est.positions[est_idx]
    p_gt = gt.positions[gt_idx]
    p_est_aligned = (R_align @ p_est.T).T + t_align
    pos_err = np.linalg.norm(p_est_aligned - p_gt, axis=1)

    R_est = R.from_quat(est.quats_wxyz[est_idx][:, [1, 2, 3, 0]])
    R_gt = R.from_quat(gt.quats_wxyz[gt_idx][:, [1, 2, 3, 0]])
    R_est_aligned = R.from_matrix(R_align) * R_est
    rot_err = geodesic_deg(R_est_aligned * R_gt.inv())

    v_est = est.velocities[est_idx]
    v_gt = gt.velocities[gt_idx]
    v_est_aligned = (R_align @ v_est.T).T
    vel_err = np.linalg.norm(v_est_aligned - v_gt, axis=1)

    return {
        "matched_pairs": int(len(pairs)),
        "time_offset_mean_ms": float(dt_ms.mean()) if len(dt_ms) else 0.0,
        "time_offset_max_ms": float(dt_ms.max()) if len(dt_ms) else 0.0,
        "position_rmse_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "position_median_m": float(np.median(pos_err)),
        "position_max_m": float(np.max(pos_err)),
        "orientation_rmse_deg": float(np.sqrt(np.mean(rot_err ** 2))),
        "orientation_median_deg": float(np.median(rot_err)),
        "orientation_max_deg": float(np.max(rot_err)),
        "velocity_rmse_mps": float(np.sqrt(np.mean(vel_err ** 2))),
        "velocity_median_mps": float(np.median(vel_err)),
        "velocity_max_mps": float(np.max(vel_err)),
    }


def apply_se3_to_trajectory(vo: Trajectory, R_align: np.ndarray, t_align: np.ndarray) -> Trajectory:
    pos_new = (R_align @ vo.positions.T).T + t_align
    rot_new = R.from_matrix(R_align) * R.from_quat(vo.quats_wxyz[:, [1, 2, 3, 0]])
    quat_xyzw = rot_new.as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    vel_new = (R_align @ vo.velocities.T).T
    return Trajectory(
        timestamps=vo.timestamps.copy(),
        positions=pos_new,
        quats_wxyz=quat_wxyz,
        velocities=vel_new,
        bw=vo.bw.copy(),
        ba=vo.ba.copy(),
    )


def save_euroc_csv(path: str, traj: Trajectory) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = [
        "#timestamp",
        "p_RS_R_x [m]",
        "p_RS_R_y [m]",
        "p_RS_R_z [m]",
        "q_RS_w []",
        "q_RS_x []",
        "q_RS_y []",
        "q_RS_z []",
        "v_RS_R_x [m s^-1]",
        "v_RS_R_y [m s^-1]",
        "v_RS_R_z [m s^-1]",
        "b_w_RS_S_x [rad s^-1]",
        "b_w_RS_S_y [rad s^-1]",
        "b_w_RS_S_z [rad s^-1]",
        "b_a_RS_S_x [m s^-2]",
        "b_a_RS_S_y [m s^-2]",
        "b_a_RS_S_z [m s^-2]",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(traj.timestamps)):
            row = [
                int(traj.timestamps[i]),
                *traj.positions[i].tolist(),
                *traj.quats_wxyz[i].tolist(),
                *traj.velocities[i].tolist(),
                *traj.bw[i].tolist(),
                *traj.ba[i].tolist(),
            ]
            writer.writerow([f"{x:.9f}" if isinstance(x, float) else x for x in row])


def save_transform(seq_dir: str, R_align: np.ndarray, t_align: np.ndarray, metrics: dict) -> None:
    T = np.eye(4)
    T[:3, :3] = R_align
    T[:3, 3] = t_align
    np.savetxt(os.path.join(seq_dir, "se3_transform.txt"), T, fmt="%.12f")

    rot = R.from_matrix(R_align).as_quat()  # xyzw
    with open(os.path.join(seq_dir, "se3_transform.json"), "w") as f:
        json.dump(
            {
                "x": float(t_align[0]),
                "y": float(t_align[1]),
                "z": float(t_align[2]),
                "qx": float(rot[0]),
                "qy": float(rot[1]),
                "qz": float(rot[2]),
                "qw": float(rot[3]),
            },
            f,
            indent=2,
        )

    with open(os.path.join(seq_dir, "alignment_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser("Prepare EuRoC VO pseudo GT with SE(3) alignment to GT")
    parser.add_argument("--data_root", type=str, default="data/EUROC")
    parser.add_argument("--source_dir", type=str, default="stereo_estimate_aligned")
    parser.add_argument("--output_dir", type=str, default="stereo_estimate_aligned_se3")
    parser.add_argument("--gt_dir", type=str, default="state_groundtruth_estimate0")
    parser.add_argument("--max_diff_ms", type=float, default=20.0)
    parser.add_argument("--method", type=str, default="origin_anchor", choices=["origin_anchor", "umeyama_se3"])
    parser.add_argument("--seq", nargs="*", default=TRAIN_SEQS)
    args = parser.parse_args()

    max_diff_ns = int(args.max_diff_ms * 1e6)
    summary_rows = []

    for seq in args.seq:
        gt_path = os.path.join(args.data_root, seq, "mav0", args.gt_dir, "data.csv")
        vo_path = os.path.join(args.data_root, seq, "mav0", args.source_dir, "data.csv")
        out_dir = os.path.join(args.data_root, seq, "mav0", args.output_dir)

        gt = load_euroc_csv(gt_path)
        vo = load_euroc_csv(vo_path)
        pairs = nearest_neighbor_pairs(vo.timestamps, gt.timestamps, max_diff_ns)
        if len(pairs) < 3:
            raise RuntimeError(f"{seq}: not enough timestamp matches ({len(pairs)}) for SE(3) alignment.")

        est_idx = np.array([p[0] for p in pairs], dtype=int)
        gt_idx = np.array([p[1] for p in pairs], dtype=int)
        if args.method == "umeyama_se3":
            R_align, t_align = umeyama_se3(vo.positions[est_idx], gt.positions[gt_idx])
        else:
            R_align, t_align = origin_anchor_se3(gt, vo, pairs)

        aligned = apply_se3_to_trajectory(vo, R_align, t_align)
        metrics = evaluate_alignment(gt, vo, pairs, R_align, t_align)
        metrics["method"] = args.method
        metrics["anchor_est_timestamp"] = int(vo.timestamps[pairs[0][0]])
        metrics["anchor_gt_timestamp"] = int(gt.timestamps[pairs[0][1]])

        save_euroc_csv(os.path.join(out_dir, "data.csv"), aligned)
        save_transform(out_dir, R_align, t_align, metrics)

        summary_rows.append({"sequence": seq, **metrics})
        print(
            f"{seq:16} matched={metrics['matched_pairs']:4d} "
            f"pos_rmse={metrics['position_rmse_m']:.4f} m "
            f"rot_rmse={metrics['orientation_rmse_deg']:.4f} deg "
            f"vel_rmse={metrics['velocity_rmse_mps']:.4f} m/s"
        )

    summary_path = os.path.join(args.data_root, f"{args.output_dir}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
