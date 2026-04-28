import argparse
import os
from itertools import permutations, product

import numpy as np


TRAIN_SEQS = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", skip_header=1)


def normalize_quat(q):
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def angle_deg(R):
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def all_axis_aligned_rotations():
    rotations = []
    eye = np.eye(3)
    for perm in permutations(range(3)):
        P = eye[:, perm]
        for signs in product([-1, 1], repeat=3):
            S = np.diag(signs)
            R = P @ S
            det = round(np.linalg.det(R))
            if det == 1:
                rotations.append(R)
    unique = []
    for R in rotations:
        if not any(np.allclose(R, U) for U in unique):
            unique.append(R)
    return unique


def rotation_name(R):
    axis_names = ["x", "y", "z"]
    cols = []
    for col in range(3):
        idx = int(np.argmax(np.abs(R[:, col])))
        sign = int(np.sign(R[idx, col]))
        cols.append(f"{'+' if sign > 0 else '-'}{axis_names[idx]}")
    return "[" + ", ".join(cols) + "]"


def build_common_arrays(base_dir, seq):
    gt = load_csv(os.path.join(base_dir, seq, "mav0", "state_groundtruth_estimate0", "data.csv"))
    vo = load_csv(os.path.join(base_dir, seq, "mav0", "stereo_estimate_aligned", "data.csv"))

    gt_map = {int(row[0]): row[1:] for row in gt}
    vo_map = {int(row[0]): row[1:] for row in vo}
    common = sorted(set(gt_map).intersection(vo_map))
    if not common:
        raise RuntimeError(f"No common timestamps for {seq}")

    gt_arr = np.array([gt_map[t] for t in common])
    vo_arr = np.array([vo_map[t] for t in common])

    p_gt = gt_arr[:, 0:3]
    p_vo = vo_arr[:, 0:3]
    q_gt = normalize_quat(gt_arr[:, 3:7])
    q_vo = normalize_quat(vo_arr[:, 3:7])
    v_gt = gt_arr[:, 7:10]
    v_vo = vo_arr[:, 7:10]
    return p_gt, p_vo, q_gt, q_vo, v_gt, v_vo, common


def summarize_errors(p_gt, p_vo, q_gt, q_vo, v_gt, v_vo, R_fix):
    pos_err = np.linalg.norm(p_gt - p_vo, axis=1)
    vel_err = np.linalg.norm(v_gt - v_vo, axis=1)

    rot_err = []
    for qg, qv in zip(q_gt, q_vo):
        R_gt = quat_to_rotmat(qg)
        R_vo = quat_to_rotmat(qv)
        rot_err.append(angle_deg(R_gt @ (R_fix @ R_vo).T))
    rot_err = np.array(rot_err)

    return {
        "pos_rmse": float(np.sqrt(np.mean(pos_err**2))),
        "pos_mean": float(np.mean(pos_err)),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot_err**2))),
        "rot_mean_deg": float(np.mean(rot_err)),
        "vel_rmse": float(np.sqrt(np.mean(vel_err**2))),
        "vel_mean": float(np.mean(vel_err)),
    }


def main():
    parser = argparse.ArgumentParser("Compare EuRoC GT and VO trajectories under fixed frame rotations")
    parser.add_argument("--data_dir", type=str, default="data/EUROC")
    parser.add_argument("--seq", type=str, nargs="*", default=TRAIN_SEQS)
    args = parser.parse_args()

    rotations = all_axis_aligned_rotations()

    print("Trying 24 axis-aligned frame rotations for VO orientation")
    print()

    for seq in args.seq:
        p_gt, p_vo, q_gt, q_vo, v_gt, v_vo, common = build_common_arrays(args.data_dir, seq)

        baseline = summarize_errors(p_gt, p_vo, q_gt, q_vo, v_gt, v_vo, np.eye(3))
        best = None
        for R_fix in rotations:
            metrics = summarize_errors(p_gt, p_vo, q_gt, q_vo, v_gt, v_vo, R_fix)
            item = (metrics["rot_mean_deg"], rotation_name(R_fix), metrics)
            if best is None or item[0] < best[0]:
                best = item

        print(seq)
        print(
            f"  raw      rot_mean={baseline['rot_mean_deg']:.3f} deg, "
            f"rot_rmse={baseline['rot_rmse_deg']:.3f} deg, "
            f"pos_rmse={baseline['pos_rmse']:.3f} m, vel_rmse={baseline['vel_rmse']:.3f} m/s"
        )
        print(
            f"  best fix {best[1]} -> rot_mean={best[2]['rot_mean_deg']:.3f} deg, "
            f"rot_rmse={best[2]['rot_rmse_deg']:.3f} deg"
        )
        print(f"  common timestamps: {len(common)}")
        print()


if __name__ == "__main__":
    main()
