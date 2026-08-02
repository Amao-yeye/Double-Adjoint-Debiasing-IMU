"""Trajectory containers and basic trajectory transforms.

Frame convention: `T_AB` maps a point expressed in frame B into frame A:
`p_A = R_AB p_B + p_AB`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class PoseTrajectory:
    """Timestamped SE(3) pose trajectory with positions and SciPy rotations."""

    name: str
    t: np.ndarray
    p: np.ndarray
    rot: Rotation

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def copy(self, name: str | None = None) -> "PoseTrajectory":
        return PoseTrajectory(
            name=self.name if name is None else name,
            t=self.t.copy(),
            p=self.p.copy(),
            rot=Rotation.from_quat(self.rot.as_quat()),
        )


def transform_by_left(traj: PoseTrajectory, T_left: np.ndarray, name: str) -> PoseTrajectory:
    """Apply `T_left` to every pose: `T_out = T_left * T_traj`."""

    R_left = Rotation.from_matrix(T_left[:3, :3])
    p_left = T_left[:3, 3]
    return PoseTrajectory(
        name=name,
        t=traj.t.copy(),
        p=R_left.apply(traj.p) + p_left,
        rot=R_left * traj.rot,
    )


def transform_by_right(traj: PoseTrajectory, T_right: np.ndarray, name: str) -> PoseTrajectory:
    """Apply `T_right` to every pose: `T_out = T_traj * T_right`."""

    R_right = Rotation.from_matrix(T_right[:3, :3])
    p_right = T_right[:3, 3]
    return PoseTrajectory(
        name=name,
        t=traj.t.copy(),
        p=traj.rot.apply(p_right) + traj.p,
        rot=traj.rot * R_right,
    )


def matrices_from_trajectory(traj: PoseTrajectory) -> np.ndarray:
    mats = np.repeat(np.eye(4)[None, :, :], len(traj), axis=0)
    mats[:, :3, :3] = traj.rot.as_matrix()
    mats[:, :3, 3] = traj.p
    return mats


def trajectory_from_matrices(name: str, timestamps: np.ndarray, mats: np.ndarray) -> PoseTrajectory:
    return PoseTrajectory(
        name=name,
        t=np.asarray(timestamps, dtype=float).copy(),
        p=mats[:, :3, 3].copy(),
        rot=Rotation.from_matrix(mats[:, :3, :3]),
    )

