"""TUM/EuRoC-style pose trajectory IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from plato.lib.geometry.trajectory import PoseTrajectory
from plato.lib.io.tables import ensure_increasing, find_column, read_table, timestamp_unit_scale


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Near-zero quaternion encountered")
    return q / norm


def read_pose_trajectory(
    path: Path,
    name: str,
    timestamp_unit: str = "auto",
    quat_order: str = "xyzw",
    position_unit: str = "auto",
) -> PoseTrajectory:
    """Read a pose trajectory with columns `t p q`.

    Headerless rows use TUM order: `t tx ty tz qx qy qz qw`.
    Headered rows can use common timestamp/position/quaternion aliases.
    """

    header, data = read_table(Path(path))
    if header is None:
        t_idx = 0
        p_idx = [1, 2, 3]
        q_idx = [4, 5, 6, 7]
    else:
        t_idx = find_column(header, ["timestamp", "time", "t", "ts", "body0timestamp"], "timestamp")
        p_idx = [
            find_column(header, ["tx", "px", "x", "positionx", "posx", "body0posx", "body0posxmm"], "position x"),
            find_column(header, ["ty", "py", "y", "positiony", "posy", "body0posy", "body0posymm"], "position y"),
            find_column(header, ["tz", "pz", "z", "positionz", "posz", "body0posz", "body0poszmm"], "position z"),
        ]
        aliases = {
            "x": ["qx", "quatx", "quaternionx", "body0quax"],
            "y": ["qy", "quaty", "quaterniony", "body0quay"],
            "z": ["qz", "quatz", "quaternionz", "body0quaz"],
            "w": ["qw", "qr", "quatr", "quaternionw", "body0quar"],
        }
        if quat_order == "xyzw":
            q_idx = [
                find_column(header, aliases["x"], "quaternion x"),
                find_column(header, aliases["y"], "quaternion y"),
                find_column(header, aliases["z"], "quaternion z"),
                find_column(header, aliases["w"], "quaternion w"),
            ]
        elif quat_order == "wxyz":
            q_idx = [
                find_column(header, aliases["w"], "quaternion w"),
                find_column(header, aliases["x"], "quaternion x"),
                find_column(header, aliases["y"], "quaternion y"),
                find_column(header, aliases["z"], "quaternion z"),
            ]
        else:
            raise ValueError(f"Unsupported quaternion order: {quat_order}")

    scale, _unit = timestamp_unit_scale(data[:, t_idx], timestamp_unit)
    t = data[:, t_idx] * scale
    p_scale = 1.0
    if position_unit == "mm":
        p_scale = 1e-3
    elif position_unit == "auto" and header is not None:
        if any("mm" in header[idx] for idx in p_idx):
            p_scale = 1e-3
    elif position_unit != "auto" and position_unit != "m":
        raise ValueError(f"Unsupported position unit: {position_unit}")

    p = data[:, p_idx] * p_scale
    q_raw = data[:, q_idx]
    if quat_order == "wxyz":
        q_raw = q_raw[:, [1, 2, 3, 0]]
    q_xyzw = normalize_quat_xyzw(q_raw)

    order = np.argsort(t)
    t = t[order]
    p = p[order]
    q_xyzw = q_xyzw[order]
    keep = np.concatenate([[True], np.diff(t) > 0.0])
    t = t[keep]
    p = p[keep]
    q_xyzw = q_xyzw[keep]
    ensure_increasing(t, name)
    return PoseTrajectory(name=name, t=t, p=p, rot=Rotation.from_quat(q_xyzw))


def write_tum_xyzw(path: Path, traj: PoseTrajectory) -> None:
    """Write TUM order: `t tx ty tz qx qy qz qw`."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = traj.rot.as_quat()
    with path.open("w", encoding="utf-8") as handle:
        for t, p, quat in zip(traj.t, traj.p, q):
            handle.write(
                "{:.9f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(
                    float(t), p[0], p[1], p[2], quat[0], quat[1], quat[2], quat[3]
                )
            )
