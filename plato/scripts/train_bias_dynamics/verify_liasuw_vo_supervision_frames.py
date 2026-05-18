#!/usr/bin/env python3
"""Verify cached LIASuw VO supervision poses are in the mocap/GT world frame."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_ROOT = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU"
BIASDY_ROOT = DEBIAS_ROOT / "BiasDy"
for path in (str(DEBIAS_ROOT), str(BIASDY_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset import pload  # noqa: E402


DEFAULT_CACHE_DIR = PLATO_ROOT / "results" / "debias_imu" / "cache" / "LIASuw_vo_supervision_frame_fixed_smoke"
DEFAULT_PSEUDO_GT_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "LIASuw"
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "results" / "debias_imu" / "vo_gt_frame_checks"
DEFAULT_SEQUENCES = [
    "0912s06_80",
    "0912s07_30",
    "0912s10_40",
    "0203s18_80",
    "0203s19_80",
    "0203s22_80",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pseudo-gt-root", type=Path, default=DEFAULT_PSEUDO_GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--expected-world-frame", default="gt")
    parser.add_argument("--max-rot-rmse-deg", type=float, default=5.0)
    parser.add_argument("--max-pos-rmse-m", type=float, default=0.08)
    return parser.parse_args()


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


def interpolate_reference(
    t_ref: np.ndarray,
    p_ref: np.ndarray,
    r_ref: Rotation,
    t_query: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    p_interp = np.column_stack([np.interp(t_query, t_ref, p_ref[:, axis]) for axis in range(3)])
    r_interp = Slerp(t_ref, r_ref)(t_query)
    return p_interp, r_interp


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def verify_sequence(args: argparse.Namespace, sequence: str) -> dict[str, object]:
    imudict = pload(str(args.cache_dir), sequence + ".p")
    gtdict = pload(str(args.cache_dir), sequence + "_gt.p")
    world_frame = imudict.get("vo_supervision_world_frame")
    if world_frame != args.expected_world_frame:
        raise AssertionError(f"{sequence}: cached world frame is {world_frame!r}, expected {args.expected_world_frame!r}")

    t_abs = float(imudict["t_start"]) / 1e9 + gtdict["t_gt"].cpu().numpy().astype(np.float64)
    p_cached = gtdict["p_gt"].cpu().numpy().astype(np.float64)
    r_cached = Rotation.from_matrix(gtdict["R_gt"].cpu().numpy().astype(np.float64))

    body_path = args.pseudo_gt_root / sequence / "debias_imu_body_gt_200hz" / "body_gt_world.csv"
    t_ref, p_ref, r_ref = read_body_gt_wxyz(body_path)
    p_ref_at_cache, r_ref_at_cache = interpolate_reference(t_ref, p_ref, r_ref, t_abs)
    p_ref_at_cache = p_ref_at_cache - p_ref_at_cache[0]

    pos_err = np.linalg.norm(p_cached - p_ref_at_cache, axis=1)
    rot_err_deg = (r_cached.inv() * r_ref_at_cache).magnitude() * 180.0 / np.pi
    pos_rmse = rmse(pos_err)
    rot_rmse = rmse(rot_err_deg)
    if pos_rmse > args.max_pos_rmse_m:
        raise AssertionError(f"{sequence}: position RMSE {pos_rmse:.6f} exceeds {args.max_pos_rmse_m}")
    if rot_rmse > args.max_rot_rmse_deg:
        raise AssertionError(f"{sequence}: rotation RMSE {rot_rmse:.6f} exceeds {args.max_rot_rmse_deg}")

    return {
        "sequence": sequence,
        "world_frame": world_frame,
        "num_cached_poses": int(t_abs.shape[0]),
        "pos_rmse_m": pos_rmse,
        "pos_max_m": float(np.max(pos_err)),
        "rot_rmse_deg": rot_rmse,
        "rot_max_deg": float(np.max(rot_err_deg)),
        "transform_note": imudict.get("T_supervision_source_note", ""),
    }


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vo_supervision_cache_frame_check.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "vo_supervision_cache_frame_check.md"
    lines = [
        "# VO Supervision Cache Frame Check",
        "",
        "| sequence | world_frame | poses | pos_rmse_m | rot_rmse_deg |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sequence']} | {row['world_frame']} | {row['num_cached_poses']} | "
            f"{row['pos_rmse_m']:.4f} | {row['rot_rmse_deg']:.3f} |"
        )
    lines.extend(["", f"CSV: `{csv_path}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [verify_sequence(args, sequence) for sequence in args.sequences]
    write_outputs(rows, args.output_dir)
    for row in rows:
        print(
            f"{row['sequence']}: frame={row['world_frame']}, poses={row['num_cached_poses']}, "
            f"pos_rmse={row['pos_rmse_m']:.4f} m, rot_rmse={row['rot_rmse_deg']:.3f} deg"
        )
    print(f"Wrote {args.output_dir / 'vo_supervision_cache_frame_check.md'}")


if __name__ == "__main__":
    main()
