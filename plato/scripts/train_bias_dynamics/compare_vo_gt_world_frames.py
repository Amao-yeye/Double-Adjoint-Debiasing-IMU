#!/usr/bin/env python3
"""Compare LIASuw VO-IMU pseudo-GT world frames against mocap/body GT frames."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PSEUDO_GT_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "LIASuw"
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "results" / "debias_imu" / "vo_gt_frame_checks"
DEFAULT_SEQUENCES = [
    "0912s06_80",
    "0912s07_30",
    "0912s10_40",
    "0203s18_80",
    "0203s19_80",
    "0203s22_80",
    "0205s14_80",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-gt-root", type=Path, default=DEFAULT_PSEUDO_GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    return parser.parse_args()


def read_tum_xyzw(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1:4], Rotation.from_quat(data[:, 4:8])


def read_body_gt_wxyz(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    times = []
    positions = []
    quats_xyzw = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["timestamp"]))
            positions.append([float(row["p_x"]), float(row["p_y"]), float(row["p_z"])])
            quats_xyzw.append([float(row["q_x"]), float(row["q_y"]), float(row["q_z"]), float(row["q_w"])])
    return np.asarray(times), np.asarray(positions), Rotation.from_quat(np.asarray(quats_xyzw))


def interpolate_gt(
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    r_gt: Rotation,
    t_query: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    p_interp = np.column_stack([np.interp(t_query, t_gt, p_gt[:, axis]) for axis in range(3)])
    r_interp = Slerp(t_gt, r_gt)(t_query)
    return p_interp, r_interp


def rotation_rmse_deg(rotations: Rotation) -> float:
    angles = rotations.magnitude()
    return float(np.sqrt(np.mean(np.square(angles))) * 180.0 / math.pi)


def translation_rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=1))))


def load_calibration_rotation(path: Path) -> Rotation | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    matrix = payload.get("T_WO")
    if matrix is None:
        return None
    return Rotation.from_matrix(np.asarray(matrix, dtype=float)[:3, :3])


def compare_sequence(pseudo_gt_root: Path, sequence: str) -> dict[str, object]:
    body_dir = pseudo_gt_root / sequence / "debias_imu_body_gt_200hz"
    t_vo, p_vo, r_vo = read_tum_xyzw(body_dir / "vo_imu_traj.tum")
    t_gt, p_gt, r_gt = read_body_gt_wxyz(body_dir / "body_gt_world.csv")

    mask = (t_vo >= t_gt[0]) & (t_vo <= t_gt[-1])
    t_vo = t_vo[mask]
    p_vo = p_vo[mask]
    r_vo = Rotation.from_quat(r_vo.as_quat()[mask])
    p_gt_at_vo, r_gt_at_vo = interpolate_gt(t_gt, p_gt, r_gt, t_vo)

    direct_rot_err = r_vo.inv() * r_gt_at_vo
    frame_rot_samples = r_gt_at_vo * r_vo.inv()
    frame_rot = frame_rot_samples.mean()
    aligned_rot_err = (frame_rot * r_vo).inv() * r_gt_at_vo
    t_frame_samples = p_gt_at_vo - frame_rot.apply(p_vo)
    t_frame = np.mean(t_frame_samples, axis=0)
    aligned_pos_err = frame_rot.apply(p_vo) + t_frame - p_gt_at_vo

    gravity_gt = np.array([0.0, 0.0, -9.81])
    gravity_vo = frame_rot.inv().apply(gravity_gt)
    gravity_default = np.array([0.0, 0.0, -9.81])
    gravity_tilt_deg = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(gravity_vo, gravity_default)
                    / (np.linalg.norm(gravity_vo) * np.linalg.norm(gravity_default)),
                    -1.0,
                    1.0,
                )
            )
        )
    )

    calib_rot = load_calibration_rotation(body_dir / "calibration_result.yaml")
    calib_delta_deg = None
    if calib_rot is not None:
        calib_delta_deg = float((calib_rot.inv() * frame_rot).magnitude() * 180.0 / math.pi)

    return {
        "sequence": sequence,
        "num_vo_poses": int(t_vo.shape[0]),
        "duration_s": float(t_vo[-1] - t_vo[0]),
        "direct_vo_to_gt_rot_rmse_deg": rotation_rmse_deg(direct_rot_err),
        "R_Wm_Wv_angle_deg": float(frame_rot.magnitude() * 180.0 / math.pi),
        "aligned_rot_residual_rmse_deg": rotation_rmse_deg(aligned_rot_err),
        "aligned_pos_residual_rmse_m": translation_rmse(aligned_pos_err),
        "gravity_in_vo_x": float(gravity_vo[0]),
        "gravity_in_vo_y": float(gravity_vo[1]),
        "gravity_in_vo_z": float(gravity_vo[2]),
        "gravity_vo_vs_default_tilt_deg": float(gravity_tilt_deg),
        "calibration_T_WO_delta_deg": calib_delta_deg,
    }


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    csv_path = output_dir / "vo_gt_world_frame_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "vo_gt_world_frame_comparison.md"
    headers = [
        "sequence",
        "R_Wm_Wv_angle_deg",
        "gravity_vo_vs_default_tilt_deg",
        "gravity_in_vo",
        "aligned_rot_residual_rmse_deg",
        "aligned_pos_residual_rmse_m",
    ]
    lines = [
        "# VO World Frame vs GT World Frame",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        gravity = f"[{row['gravity_in_vo_x']:.3f}, {row['gravity_in_vo_y']:.3f}, {row['gravity_in_vo_z']:.3f}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sequence"]),
                    f"{row['R_Wm_Wv_angle_deg']:.3f}",
                    f"{row['gravity_vo_vs_default_tilt_deg']:.3f}",
                    gravity,
                    f"{row['aligned_rot_residual_rmse_deg']:.3f}",
                    f"{row['aligned_pos_residual_rmse_m']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`R_Wm_Wv` maps VO world vectors into mocap/GT world. If training directly",
            "uses VO poses as GT, Debias_IMU's default gravity `[0, 0, -9.81]` is only",
            "correct when `gravity_vo_vs_default_tilt_deg` is near zero.",
            "",
            f"CSV: `{csv_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [compare_sequence(args.pseudo_gt_root, sequence) for sequence in args.sequences]
    write_outputs(rows, args.output_dir)
    for row in rows:
        print(
            f"{row['sequence']}: "
            f"R_Wm_Wv_angle={row['R_Wm_Wv_angle_deg']:.3f} deg, "
            f"gravity_tilt={row['gravity_vo_vs_default_tilt_deg']:.3f} deg, "
            f"g_vo=[{row['gravity_in_vo_x']:.3f}, {row['gravity_in_vo_y']:.3f}, {row['gravity_in_vo_z']:.3f}], "
            f"rot_residual={row['aligned_rot_residual_rmse_deg']:.3f} deg, "
            f"pos_residual={row['aligned_pos_residual_rmse_m']:.4f} m"
        )
    print(f"Wrote {args.output_dir / 'vo_gt_world_frame_comparison.md'}")


if __name__ == "__main__":
    main()
