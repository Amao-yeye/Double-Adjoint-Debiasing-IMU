#!/usr/bin/env python3
"""Verify that LIASuw Debias_IMU supervision uses world-frame IMU/body GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AGGREGATE_ROOT = PLATO_ROOT / "results" / "debias_imu" / "LIASuw_training_candidates_200hz"
DEFAULT_PSEUDO_GT_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "LIASuw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--pseudo-gt-root", type=Path, default=DEFAULT_PSEUDO_GT_ROOT)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--pose-atol", type=float, default=1e-10)
    return parser.parse_args()


def read_sequences(root: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    selected = root / "selected_sequences.txt"
    if selected.exists():
        return [line.strip() for line in selected.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.endswith("_firstframe"))


def load_body_world(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    timestamps_ns = np.round(data["timestamp"] * 1e9).astype(np.int64)
    pose = np.column_stack(
        [
            data["p_x"],
            data["p_y"],
            data["p_z"],
            data["q_w"],
            data["q_x"],
            data["q_y"],
            data["q_z"],
        ]
    )
    return timestamps_ns, pose


def load_tumvi_mocap(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0].astype(np.int64), data[:, 1:8]


def main() -> int:
    args = parse_args()
    sequences = read_sequences(args.aggregate_root, args.sequences)
    failures: list[str] = []
    print("Checking LIASuw Debias_IMU supervision frame files")
    for seq in sequences:
        body_path = args.pseudo_gt_root / seq / "debias_imu_body_gt_200hz" / "body_gt_world.csv"
        firstframe_path = args.pseudo_gt_root / seq / "debias_imu_body_gt_200hz" / "body_gt_firstframe.csv"
        mocap_path = args.aggregate_root / seq / "mav0" / "mocap0" / "data.csv"

        body_t_ns, body_pose = load_body_world(body_path)
        mocap_t_ns, mocap_pose = load_tumvi_mocap(mocap_path)
        n = min(len(body_t_ns), len(mocap_t_ns))
        max_dt_ns = int(np.max(np.abs(body_t_ns[:n] - mocap_t_ns[:n])))
        max_pose = float(np.max(np.abs(body_pose[:n] - mocap_pose[:n])))
        ok = n == len(body_t_ns) == len(mocap_t_ns) and max_dt_ns == 0 and max_pose <= args.pose_atol

        firstframe_note = "firstframe=n/a"
        if firstframe_path.exists():
            _, first_pose = load_body_world(firstframe_path)
            firstframe_note = (
                f"firstframe |p0|={np.linalg.norm(first_pose[0, :3]):.3e}, "
                f"q0={first_pose[0, 3]:.6f},{first_pose[0, 4]:.6f},{first_pose[0, 5]:.6f},{first_pose[0, 6]:.6f}"
            )

        status = "OK" if ok else "FAIL"
        print(
            f"{status} {seq:12s} rows={n:5d} "
            f"max_dt_ns={max_dt_ns:3d} max_pose_abs_diff={max_pose:.3e}; {firstframe_note}"
        )
        if not ok:
            failures.append(seq)

    if failures:
        print("Failed sequences: " + ", ".join(failures))
        return 1
    print("All checked aggregate mocap0 files exactly match body_gt_world.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
