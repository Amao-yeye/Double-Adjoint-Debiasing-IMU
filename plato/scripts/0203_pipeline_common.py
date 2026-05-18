#!/usr/bin/env python3
"""0203 LIASuw stereo VO evaluation and IMU-groundtruth generation.

Frame convention used throughout this module:

    T_AB maps a point expressed in frame B into frame A:
        p_A = R_AB p_B + p_AB

Important trajectory meanings:

    ORB-SLAM3 stereo SaveTrajectoryEuRoC writes T_Wv_C, the left camera pose
    in the ORB/VO world. It already writes Twc, not Tcw.

    LIASuw mocap mc.csv is parsed as T_Wm_M, the rigid mocap marker pose in
    the mocap world.

    The final exported IMU GT is:
        T_Wv_I_gt(t) = T_Wv_Wm * T_Wm_M(t_mocap) * T_M_I

Timestamp convention:

    time_offset_s is defined by t_mocap = t_vo + time_offset_s when
    synchronizing VO/IMU-clock trajectories with mocap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "plato_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation, Slerp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by environment
    raise ModuleNotFoundError(
        "scipy is required for 0203 trajectory interpolation and SE(3) "
        "optimization. On this machine use /home/amao/anaconda3/bin/python "
        "for these scripts; bag export is invoked separately with /usr/bin/python3."
    ) from exc


PLATO_ROOT = Path(__file__).resolve().parents[2]
ORB_ROOT = PLATO_ROOT / "external" / "ORB_SLAM3"
DEFAULT_DATASET_ROOT = Path(os.environ.get("PLATO_DATASET_ROOT", "/home/amao/datasets"))
DEFAULT_STEREO_CONFIG = PLATO_ROOT / "plato/configs/orbslam3/LIASuw_image_nf2000_far8_depth30.yaml"
DEFAULT_OUTPUT_ROOT = PLATO_ROOT / "results"


@dataclass
class PoseTrajectory:
    name: str
    t: np.ndarray
    p: np.ndarray
    rot: Rotation

    def __len__(self) -> int:
        return int(self.t.shape[0])


@dataclass
class SequenceInfo:
    sequence: str
    split: str
    bag_path: Path
    mocap_path: Optional[Path]
    extrinsic_path: Optional[Path]
    skip_reason: str = ""


@dataclass
class CameraImuExtrinsic:
    path: Path
    convention: str
    T_C_I: np.ndarray
    source_key: str
    note: str


@dataclass
class EvalResult:
    sequence: str
    selected: bool
    status: str
    metrics: Dict[str, float]
    result_path: Path


@dataclass
class CalibrationResult:
    T_Wv_Wm: np.ndarray
    T_M_I: np.ndarray
    initial_stats: Dict[str, float]
    final_stats: Dict[str, float]
    optimizer: object
    sample_count: int


def normalize_sequence_name(value: str) -> str:
    text = value.strip().lower().replace("_", "").replace("-", "")
    text = text.replace("0203", "")
    match = re.search(r"s?0*(\d+)", text)
    if not match:
        raise ValueError(f"Could not parse sequence name: {value}")
    return f"0203s{int(match.group(1)):02d}"


def sequence_number(sequence: str) -> str:
    match = re.search(r"(\d+)$", sequence)
    if not match:
        raise ValueError(f"Could not parse sequence number from {sequence}")
    return f"{int(match.group(1)):02d}"


def liasuw_root(dataset_root: Path) -> Path:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if dataset_root.name == "LIASuw":
        return dataset_root
    return dataset_root / "LIASuw"


def split_dir_name(split: str) -> str:
    normalized = split.lower()
    mapping = {
        "training": "LIASuw_Training",
        "train": "LIASuw_Training",
        "testing": "LIASuw_Testing",
        "test": "LIASuw_Testing",
        "unusable": "LIASuw_Unusable",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported LIASuw split: {split}")
    return mapping[normalized]


def discover_sequences(dataset_root: Path, splits: Sequence[str]) -> List[SequenceInfo]:
    root = liasuw_root(dataset_root)
    sequences: List[SequenceInfo] = []
    for split in splits:
        split_dir = root / split_dir_name(split)
        bag_dir = split_dir / "0203"
        if not bag_dir.exists():
            continue
        for bag_path in sorted(bag_dir.glob("s*.bag")):
            sequence = normalize_sequence_name(bag_path.stem)
            num = sequence_number(sequence)
            mocap_candidates = [
                split_dir / "0203" / "GT" / num / "mc.csv",
                root / "0203" / "未处理" / f"s{num}" / "mc.csv",
            ]
            mocap_path = next((path for path in mocap_candidates if path.exists()), None)
            extrinsic_candidates = [split_dir / "esio.yaml", root / "esio.yaml"]
            extrinsic_path = next((path for path in extrinsic_candidates if path.exists()), None)
            skip_reasons: List[str] = []
            if mocap_path is None:
                skip_reasons.append("missing mocap mc.csv")
            if extrinsic_path is None:
                skip_reasons.append("missing IMU-camera extrinsic config")
            sequences.append(
                SequenceInfo(
                    sequence=sequence,
                    split=split_dir.name,
                    bag_path=bag_path,
                    mocap_path=mocap_path,
                    extrinsic_path=extrinsic_path,
                    skip_reason="; ".join(skip_reasons),
                )
            )
    return sequences


def filter_sequences(sequences: Sequence[SequenceInfo], requested: Sequence[str]) -> List[SequenceInfo]:
    if not requested:
        return list(sequences)
    wanted = {normalize_sequence_name(item) for value in requested for item in value.split(",") if item.strip()}
    return [info for info in sequences if info.sequence in wanted]


def parse_default_time_offset(dataset_root: Path) -> float:
    order_path = liasuw_root(dataset_root) / "0203" / "GT" / "order.txt"
    if not order_path.exists():
        return 0.0
    text = order_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"--t_offset\s+([-+]?\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    return float(match.group(1))


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def existing_file(path: Path, overwrite: bool) -> bool:
    return path.exists() and not overwrite


def run_command(
    cmd: Sequence[object],
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    process_env = os.environ.copy()
    process_env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "plato_matplotlib"))
    if env:
        process_env.update(env)
    result = subprocess.run(
        [str(item) for item in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=process_env,
    )
    if log_path:
        safe_mkdir(log_path.parent)
        log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code {}: {}\nLog: {}".format(
                result.returncode,
                " ".join(str(item) for item in cmd),
                log_path,
            )
        )


def orbslam_runtime_env(orb_root: Path) -> Dict[str, str]:
    lib_dirs = [
        orb_root / "lib",
        orb_root / "Thirdparty" / "DBoW2" / "lib",
        orb_root / "Thirdparty" / "g2o" / "lib",
        orb_root / "Thirdparty" / "PangolinInstall" / "lib",
    ]
    parts = [str(path) for path in lib_dirs if path.exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current:
        parts.append(current)
    return {"LD_LIBRARY_PATH": ":".join(parts)}


def timestamp_unit_scale(values: np.ndarray, requested: str = "auto") -> Tuple[float, str]:
    if requested != "auto":
        mapping = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
        return mapping[requested], requested
    median_abs = float(np.median(np.abs(values)))
    if median_abs > 1e17:
        return 1e-9, "ns"
    if median_abs > 1e14:
        return 1e-6, "us"
    if median_abs > 1e11:
        return 1e-3, "ms"
    return 1.0, "s"


def read_table(path: Path) -> Tuple[Optional[List[str]], np.ndarray]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                candidate = line.lstrip("#").strip()
                if not rows and ("timestamp" in candidate.lower() or candidate.lower().startswith("time")):
                    line = candidate
                else:
                    continue
            fields = [item.strip() for item in re.split(r",|\s+", line) if item.strip()]
            if fields:
                rows.append(fields)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    has_header = any(not is_float_token(item) for item in rows[0])
    header = [canonical_name(item) for item in rows[0]] if has_header else None
    data_rows = rows[1:] if has_header else rows
    width = len(data_rows[0])
    numeric: List[List[float]] = []
    for row in data_rows:
        if len(row) < width:
            continue
        try:
            values = [float(item) for item in row[:width]]
        except ValueError:
            continue
        if np.all(np.isfinite(values)):
            numeric.append(values)
    if not numeric:
        raise ValueError(f"No numeric rows found in {path}")
    return header, np.asarray(numeric, dtype=float)


def is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def canonical_name(name: str) -> str:
    name = name.strip().lstrip("#").strip().lower()
    name = re.sub(r"\[[^\]]*\]", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def find_column(header: Sequence[str], aliases: Sequence[str], label: str) -> int:
    alias_set = {canonical_name(item) for item in aliases}
    for idx, name in enumerate(header):
        if name in alias_set:
            return idx
    raise ValueError(f"Could not find {label} column in header {list(header)}")


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Near-zero quaternion encountered")
    return q / norm


def ensure_increasing(timestamps: np.ndarray, name: str) -> None:
    if len(timestamps) < 2:
        raise ValueError(f"{name} contains fewer than two timestamps")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError(f"{name} timestamps are not strictly increasing")


def read_pose_trajectory(
    path: Path,
    name: str,
    timestamp_unit: str = "auto",
    quat_order: str = "xyzw",
    position_unit: str = "auto",
) -> PoseTrajectory:
    header, data = read_table(path)
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
        else:
            q_idx = [
                find_column(header, aliases["w"], "quaternion w"),
                find_column(header, aliases["x"], "quaternion x"),
                find_column(header, aliases["y"], "quaternion y"),
                find_column(header, aliases["z"], "quaternion z"),
            ]
    scale, _unit = timestamp_unit_scale(data[:, t_idx], timestamp_unit)
    t = data[:, t_idx] * scale
    p_scale = 1.0
    if position_unit == "mm":
        p_scale = 1e-3
    elif position_unit == "auto" and header is not None:
        if any("mm" in header[idx] for idx in p_idx):
            p_scale = 1e-3
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
    safe_mkdir(path.parent)
    q = traj.rot.as_quat()
    with path.open("w", encoding="utf-8") as handle:
        for t, p, quat in zip(traj.t, traj.p, q):
            handle.write(
                "{:.9f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(
                    float(t), p[0], p[1], p[2], quat[0], quat[1], quat[2], quat[3]
                )
            )


def write_matrix_yaml(path: Path, name: str, matrix: np.ndarray, extra: Optional[Dict[str, object]] = None) -> None:
    data: Dict[str, object] = {"convention": "T_AB maps points from B to A", name: matrix}
    if extra:
        data.update(extra)
    safe_mkdir(path.parent)
    path.write_text("\n".join(yaml_lines(data)) + "\n", encoding="utf-8")


def yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def yaml_lines(value: object, indent: int = 0) -> List[str]:
    pad = " " * indent
    if isinstance(value, np.ndarray):
        return yaml_lines(value.tolist(), indent)
    if isinstance(value, dict):
        lines: List[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple, np.ndarray)):
                lines.append(f"{pad}{key}:")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{pad}[]"]
        if all(not isinstance(item, (dict, list, tuple, np.ndarray)) for item in value):
            return [f"{pad}[{', '.join(yaml_scalar(item) for item in value)}]"]
        lines = []
        for item in value:
            lines.append(f"{pad}-")
            lines.extend(yaml_lines(item, indent + 2))
        return lines
    return [f"{pad}{yaml_scalar(value)}"]


def make_T(position: np.ndarray, rot: Rotation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    Rm = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def transform_by_left(traj: PoseTrajectory, T_left: np.ndarray, name: str) -> PoseTrajectory:
    R_left = Rotation.from_matrix(T_left[:3, :3])
    p_left = T_left[:3, 3]
    return PoseTrajectory(
        name=name,
        t=traj.t.copy(),
        p=R_left.apply(traj.p) + p_left,
        rot=R_left * traj.rot,
    )


def transform_by_right(traj: PoseTrajectory, T_right: np.ndarray, name: str) -> PoseTrajectory:
    R_right = Rotation.from_matrix(T_right[:3, :3])
    p_right = T_right[:3, 3]
    return PoseTrajectory(
        name=name,
        t=traj.t.copy(),
        p=traj.rot.apply(p_right) + traj.p,
        rot=traj.rot * R_right,
    )


def matrices_from_traj(traj: PoseTrajectory) -> np.ndarray:
    mats = np.repeat(np.eye(4)[None, :, :], len(traj), axis=0)
    mats[:, :3, :3] = traj.rot.as_matrix()
    mats[:, :3, 3] = traj.p
    return mats


def traj_from_matrices(name: str, timestamps: np.ndarray, mats: np.ndarray) -> PoseTrajectory:
    return PoseTrajectory(
        name=name,
        t=np.asarray(timestamps, dtype=float).copy(),
        p=mats[:, :3, 3].copy(),
        rot=Rotation.from_matrix(mats[:, :3, :3]),
    )


def interpolate_trajectory(traj: PoseTrajectory, query_t: np.ndarray, name: str) -> PoseTrajectory:
    query_t = np.asarray(query_t, dtype=float)
    if len(query_t) == 0:
        raise ValueError("No timestamps requested for interpolation")
    if query_t[0] < traj.t[0] - 1e-9 or query_t[-1] > traj.t[-1] + 1e-9:
        raise ValueError(
            f"Interpolation outside {traj.name} range: "
            f"query [{query_t[0]:.9f}, {query_t[-1]:.9f}], "
            f"source [{traj.t[0]:.9f}, {traj.t[-1]:.9f}]"
        )
    p = np.column_stack([np.interp(query_t, traj.t, traj.p[:, axis]) for axis in range(3)])
    origin = traj.t[0]
    rot = Slerp(traj.t - origin, traj.rot)(query_t - origin)
    return PoseTrajectory(name=name, t=query_t.copy(), p=p, rot=rot)


def trajectory_overlap_mask(vo_t: np.ndarray, mocap: PoseTrajectory, time_offset: float) -> np.ndarray:
    query = vo_t + time_offset
    return (query >= mocap.t[0]) & (query <= mocap.t[-1])


def synchronized_vo_mocap(
    vo: PoseTrajectory,
    mocap: PoseTrajectory,
    time_offset: float,
    max_samples: int,
    min_samples: int,
) -> Tuple[PoseTrajectory, PoseTrajectory]:
    mask = trajectory_overlap_mask(vo.t, mocap, time_offset)
    indices = np.where(mask)[0]
    if len(indices) < min_samples:
        raise ValueError(f"Only {len(indices)} synchronized samples; need {min_samples}")
    if max_samples > 0 and len(indices) > max_samples:
        indices = np.unique(np.round(np.linspace(indices[0], indices[-1], max_samples)).astype(int))
        indices = indices[trajectory_overlap_mask(vo.t[indices], mocap, time_offset)]
    vo_sub = PoseTrajectory(name=f"{vo.name}_sync", t=vo.t[indices], p=vo.p[indices], rot=vo.rot[indices])
    mocap_interp = interpolate_trajectory(mocap, vo_sub.t + time_offset, name=f"{mocap.name}_at_vo_plus_offset")
    mocap_interp.t = vo_sub.t.copy()
    return vo_sub, mocap_interp


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = src_centered.T @ dst_centered / src.shape[0]
    u_mat, _, vh_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vh_mat.T @ u_mat.T) < 0:
        correction[2, 2] = -1.0
    Rm = vh_mat.T @ correction @ u_mat.T
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = dst_mean - Rm @ src_mean
    return T


def sim3_scale_diagnostic(src: np.ndarray, dst: np.ndarray) -> float:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = src_centered.T @ dst_centered / src.shape[0]
    u_mat, singular_values, vh_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vh_mat.T @ u_mat.T) < 0:
        correction[2, 2] = -1.0
    variance = np.sum(src_centered * src_centered) / src.shape[0]
    return float(np.sum(singular_values * np.diag(correction)) / variance)


def mean_rotation(rotations: Rotation) -> Rotation:
    matrix = np.mean(rotations.as_matrix(), axis=0)
    u_mat, _, vh_mat = np.linalg.svd(matrix)
    Rm = u_mat @ vh_mat
    if np.linalg.det(Rm) < 0:
        u_mat[:, -1] *= -1.0
        Rm = u_mat @ vh_mat
    return Rotation.from_matrix(Rm)


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def rotation_errors_deg(a: PoseTrajectory, b: PoseTrajectory) -> np.ndarray:
    return (a.rot.inv() * b.rot).magnitude() * 180.0 / math.pi


def relative_errors(a: PoseTrajectory, b: PoseTrajectory) -> Dict[str, np.ndarray]:
    return {
        "translation_error_m": np.linalg.norm(b.p - a.p, axis=1),
        "rotation_error_deg": rotation_errors_deg(a, b),
    }


def rpy_deg(rot: Rotation) -> np.ndarray:
    return rot.as_euler("xyz", degrees=True)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * K + (K @ K) / 6.0
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
        return np.eye(3) - 0.5 * K + (K @ K) / 12.0
    theta2 = theta * theta
    coeff = (1.0 / theta2) - ((1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta)))
    return np.eye(3) - 0.5 * K + coeff * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho = xi[:3]
    phi = xi[3:]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(phi).as_matrix()
    T[:3, 3] = so3_left_jacobian(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    phi = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    rho = so3_left_jacobian_inv(phi) @ T[:3, 3]
    return np.concatenate([rho, phi])


def parse_opencv_matrix_after_key(text: str, key: str) -> Optional[np.ndarray]:
    key_re = re.escape(key)
    pattern = re.compile(
        key_re
        + r".*?rows:\s*(\d+).*?cols:\s*(\d+).*?data:\s*\[([^\]]+)\]",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    rows = int(match.group(1))
    cols = int(match.group(2))
    values = [float(item) for item in re.split(r"[,\s]+", match.group(3).strip()) if item]
    return np.asarray(values, dtype=float).reshape(rows, cols)


def parse_assignment_matrix(text: str, label: str) -> Optional[np.ndarray]:
    label_match = re.search(re.escape(label), text)
    if not label_match:
        return None
    start = text.find("[[", label_match.end())
    if start < 0:
        return None
    end = text.find("]]", start)
    if end < 0:
        return None
    block = text[start : end + 2]
    rows: List[List[float]] = []
    for row_text in re.findall(r"\[([^\[\]]+)\]", block):
        row = [float(item) for item in row_text.replace(",", " ").split()]
        if row:
            rows.append(row)
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def load_camera_imu_extrinsic(path: Path) -> CameraImuExtrinsic:
    text = path.read_text(encoding="utf-8", errors="replace")

    # ORB-SLAM3 stereo-inertial config convention: IMU.T_b_c1 is T_I_C.
    T_b_c1 = parse_opencv_matrix_after_key(text, "IMU.T_b_c1")
    if T_b_c1 is not None:
        T_I_C = T_b_c1
        return CameraImuExtrinsic(
            path=path,
            convention="T_I_C",
            T_C_I=invert_T(T_I_C),
            source_key="IMU.T_b_c1",
            note="ORB-SLAM3 uses IMU.T_b_c1 as left camera frame to body/IMU frame.",
        )

    T_C_I = parse_assignment_matrix(text, "T_ci")
    T_I_C = parse_assignment_matrix(text, "T_ic")
    if T_C_I is not None:
        return CameraImuExtrinsic(
            path=path,
            convention="T_C_I",
            T_C_I=T_C_I,
            source_key="T_ci",
            note="Kalibr-style T_ci is imu0 to cam0, i.e. T_C_I.",
        )
    if T_I_C is not None:
        return CameraImuExtrinsic(
            path=path,
            convention="T_I_C",
            T_C_I=invert_T(T_I_C),
            source_key="T_ic",
            note="Kalibr-style T_ic is cam0 to imu0, so it is inverted before right-multiplying T_W_C.",
        )

    R_event = parse_opencv_matrix_after_key(text, "extrinsicRotation_event")
    t_event = parse_opencv_matrix_after_key(text, "extrinsicTranslation_event")
    if R_event is not None and t_event is not None:
        T = np.eye(4)
        T[:3, :3] = R_event
        T[:3, 3] = t_event.reshape(3)
        flag_match = re.search(r"T_event_imu:\s*([01])", text)
        flag = int(flag_match.group(1)) if flag_match else 0
        if flag == 1:
            return CameraImuExtrinsic(
                path=path,
                convention="T_C_I",
                T_C_I=T,
                source_key="extrinsicRotation_event/extrinsicTranslation_event with T_event_imu=1",
                note="Config flag says the provided transform is IMU to event/camera.",
            )
        return CameraImuExtrinsic(
            path=path,
            convention="T_I_C",
            T_C_I=invert_T(T),
            source_key="extrinsicRotation_event/extrinsicTranslation_event with T_event_imu=0",
            note=(
                "esio.yaml comments define imu^R_event and imu^T_event, i.e. T_I_C; "
                "the script inverts it before computing T_W_I = T_W_C * T_C_I."
            ),
        )

    raise ValueError(f"Could not identify an IMU-camera extrinsic convention in {path}")


def count_timestamps(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))


def truncate_sequence(sequence_dir: Path, duration_s: float) -> Optional[Dict[str, object]]:
    if duration_s <= 0.0:
        return None
    timestamps_path = sequence_dir / "timestamps.txt"
    timestamps = [int(line.strip().split()[0]) for line in timestamps_path.read_text().splitlines() if line.strip()]
    if len(timestamps) < 2:
        raise ValueError(f"Not enough timestamps in {timestamps_path}")
    first = timestamps[0]
    max_ts = first + int(duration_s * 1e9)
    kept = [ts for ts in timestamps if ts <= max_ts]
    if len(kept) < 3:
        raise ValueError(f"Duration {duration_s} keeps only {len(kept)} stereo frames")
    keep_names = {f"{ts}.png" for ts in kept}
    for image_dir in [sequence_dir / "left", sequence_dir / "right"]:
        for path in image_dir.glob("*.png"):
            if path.name not in keep_names:
                path.unlink()
    timestamps_path.write_text("".join(f"{ts}\n" for ts in kept), encoding="utf-8")
    imu_path = sequence_dir / "imu.txt"
    kept_imu = 0
    if imu_path.exists():
        rows = []
        for line in imu_path.read_text().splitlines():
            if not line.strip():
                continue
            ts = int(line.split(",", 1)[0])
            if first <= ts <= kept[-1]:
                rows.append(line)
        imu_path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
        kept_imu = len(rows)
    return {
        "requested_duration_s": duration_s,
        "kept_stereo_pairs": len(kept),
        "start_timestamp_ns": first,
        "end_timestamp_ns": kept[-1],
        "actual_duration_s": (kept[-1] - first) * 1e-9,
        "kept_imu_measurements": kept_imu,
    }


def run_stereo_vo_for_sequence(info: SequenceInfo, args: argparse.Namespace) -> Dict[str, object]:
    seq_dir = args.vo_output_dir / info.sequence
    raw_dir = seq_dir / "raw"
    logs_dir = seq_dir / "logs"
    sequence_dir = seq_dir / "sequence"
    safe_mkdir(raw_dir)
    safe_mkdir(logs_dir)
    raw_estimate = raw_dir / "vo_camera_traj.txt"
    raw_keyframes = raw_dir / "vo_camera_keyframes.txt"
    raw_seconds = raw_dir / "vo_camera_traj_seconds.txt"
    metadata_path = seq_dir / "run_metadata.json"

    if info.skip_reason:
        status = {"sequence": info.sequence, "status": "skipped", "reason": info.skip_reason}
        metadata_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    export_script = args.orb_root / "tools" / "export_liasuw_bag.py"
    executable = args.orb_root / "Examples" / "Stereo" / "stereo_tum_vi_batch"
    vocabulary = args.orb_root / "Vocabulary" / "ORBvoc.txt"
    for required in [export_script, executable, vocabulary, args.config_path]:
        if not required.exists():
            raise FileNotFoundError(required)

    if not sequence_dir.exists() or args.overwrite:
        prepare_dir(sequence_dir, args.overwrite)
        export_cmd: List[object] = [args.export_python, export_script, info.bag_path, sequence_dir]
        if args.max_frames > 0:
            export_cmd += ["--max-frames", args.max_frames]
        run_command(export_cmd, log_path=logs_dir / "export.log")
        truncation = truncate_sequence(sequence_dir, args.duration_s)
    else:
        truncation = None

    if existing_file(raw_estimate, args.overwrite):
        status = "reused_existing"
    else:
        run_name = f"plato0203_{info.sequence}_{os.getpid()}_{int(time.time())}"
        generated_f = args.orb_root / f"f_{run_name}.txt"
        generated_kf = args.orb_root / f"kf_{run_name}.txt"
        run_command(
            [
                executable,
                vocabulary,
                args.config_path,
                sequence_dir / "left",
                sequence_dir / "right",
                sequence_dir / "timestamps.txt",
                run_name,
            ],
            cwd=args.orb_root,
            log_path=logs_dir / "orbslam3.log",
            env=orbslam_runtime_env(args.orb_root),
        )
        if not generated_f.exists():
            raise FileNotFoundError(f"ORB-SLAM3 did not write {generated_f}")
        shutil.move(str(generated_f), raw_estimate)
        if generated_kf.exists():
            shutil.move(str(generated_kf), raw_keyframes)
        status = "ran"

    traj = read_pose_trajectory(raw_estimate, "vo_camera_raw", timestamp_unit="auto", quat_order="xyzw", position_unit="m")
    # ORB-SLAM3 SaveTrajectoryEuRoC stores Twc for stereo: T_Wv_C, not T_C_W.
    write_tum_xyzw(raw_seconds, traj)
    input_frames = count_timestamps(sequence_dir / "timestamps.txt")
    duration = float(traj.t[-1] - traj.t[0]) if len(traj) >= 2 else 0.0
    tracking_success_rate = float(len(traj) / input_frames) if input_frames else 0.0
    metadata = {
        "sequence": info.sequence,
        "split": info.split,
        "status": status,
        "bag_path": str(info.bag_path),
        "mocap_path": str(info.mocap_path),
        "extrinsic_path": str(info.extrinsic_path),
        "mode": "ORB-SLAM3 stereo visual odometry only; no IMU factors are used.",
        "trajectory_convention": "vo_camera_traj_seconds.txt stores T_Wv_C (left camera in VO world).",
        "config_path": str(args.config_path),
        "num_input_frames": input_frames,
        "num_estimated_frames": len(traj),
        "tracking_success_rate": tracking_success_rate,
        "duration_s": duration,
        "truncation": truncation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def evaluate_time_offset_candidate(
    vo: PoseTrajectory,
    mocap: PoseTrajectory,
    offset: float,
    max_samples: int,
    min_samples: int,
) -> Tuple[float, int]:
    vo_sync, mocap_sync = synchronized_vo_mocap(vo, mocap, offset, max_samples, min_samples)
    T_Wv_Wm = umeyama_se3(mocap_sync.p, vo_sync.p)
    mocap_aligned = transform_by_left(mocap_sync, T_Wv_Wm, "mocap_aligned")
    return rmse(np.linalg.norm(vo_sync.p - mocap_aligned.p, axis=1)), len(vo_sync)


def choose_time_offset(vo: PoseTrajectory, mocap: PoseTrajectory, args: argparse.Namespace) -> Tuple[float, List[Dict[str, float]]]:
    base = args.time_offset_s
    if not args.estimate_time_offset:
        return base, []
    offsets = np.arange(
        base + args.time_offset_search_min,
        base + args.time_offset_search_max + 0.5 * args.time_offset_search_step,
        args.time_offset_search_step,
    )
    rows: List[Dict[str, float]] = []
    best_offset = base
    best_score = float("inf")
    for offset in offsets:
        try:
            score, count = evaluate_time_offset_candidate(vo, mocap, float(offset), args.max_eval_samples, args.min_sync_samples)
        except Exception:
            continue
        rows.append({"time_offset_s": float(offset), "position_rmse_m": score, "samples": float(count)})
        if score < best_score:
            best_score = score
            best_offset = float(offset)
    if not rows:
        raise ValueError("No valid time-offset candidate produced synchronized samples")
    return best_offset, rows


def evaluate_sequence(info: SequenceInfo, args: argparse.Namespace) -> Dict[str, object]:
    eval_dir = args.eval_output_dir / info.sequence
    safe_mkdir(eval_dir)
    fig_dir = eval_dir / "figures"
    safe_mkdir(fig_dir)
    metrics_path = eval_dir / "metrics.json"
    if existing_file(metrics_path, args.overwrite):
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if info.skip_reason:
        metrics = {"sequence": info.sequence, "status": "skipped", "reason": info.skip_reason}
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    vo_path = args.vo_output_dir / info.sequence / "raw" / "vo_camera_traj_seconds.txt"
    if not vo_path.exists():
        raise FileNotFoundError(vo_path)
    vo = read_pose_trajectory(vo_path, "vo_T_Wv_C", timestamp_unit="s", quat_order="xyzw", position_unit="m")
    mocap = read_pose_trajectory(info.mocap_path, "mocap_T_Wm_M", timestamp_unit="auto", quat_order="xyzw", position_unit="auto")
    time_offset, offset_rows = choose_time_offset(vo, mocap, args)
    vo_sync, mocap_sync = synchronized_vo_mocap(vo, mocap, time_offset, args.max_eval_samples, args.min_sync_samples)
    T_Wv_Wm_eval = umeyama_se3(mocap_sync.p, vo_sync.p)
    mocap_aligned = transform_by_left(mocap_sync, T_Wv_Wm_eval, "mocap_marker_aligned_to_vo_world")

    # Evaluation rotation uses one constant right-side frame rotation. This keeps
    # scale fixed and estimates only the unknown marker-vs-left-camera frame
    # rotation needed for a meaningful orientation RMSE before the full IMU
    # extrinsic calibration stage.
    R_M_C_samples = mocap_aligned.rot.inv() * vo_sync.rot
    R_M_C_eval = mean_rotation(R_M_C_samples)
    T_M_C_eval = np.eye(4)
    T_M_C_eval[:3, :3] = R_M_C_eval.as_matrix()
    mocap_as_camera = transform_by_right(mocap_aligned, T_M_C_eval, "mocap_aligned_with_eval_camera_rotation")

    trans_err = np.linalg.norm(vo_sync.p - mocap_aligned.p, axis=1)
    rot_err = rotation_errors_deg(mocap_as_camera, vo_sync)
    tracking_meta = read_json_if_exists(args.vo_output_dir / info.sequence / "run_metadata.json")
    input_frames = int(tracking_meta.get("num_input_frames", count_timestamps(args.vo_output_dir / info.sequence / "sequence" / "timestamps.txt")))
    tracking_success_rate = float(tracking_meta.get("tracking_success_rate", len(vo) / input_frames if input_frames else 0.0))
    duration = float(tracking_meta.get("duration_s", vo.t[-1] - vo.t[0]))
    sim3_scale = sim3_scale_diagnostic(mocap_sync.p, vo_sync.p)

    write_tum_xyzw(eval_dir / "vo_camera_associated.txt", vo_sync)
    write_tum_xyzw(eval_dir / "mocap_marker_aligned_to_vo_world.txt", mocap_aligned)
    write_tum_xyzw(eval_dir / "mocap_marker_aligned_as_camera.txt", mocap_as_camera)
    write_eval_csv(eval_dir / "aligned_poses.csv", vo_sync, mocap_aligned, mocap_as_camera, trans_err, rot_err)
    if offset_rows:
        write_dict_csv(eval_dir / "time_offset_search.csv", offset_rows)
    make_comparison_plots(fig_dir, vo_sync, mocap_as_camera, "Stereo VO camera", "Mocap marker aligned", trans_err, rot_err)

    metrics = {
        "sequence": info.sequence,
        "split": info.split,
        "status": "evaluated",
        "num_frames": int(len(vo)),
        "num_associated_frames": int(len(vo_sync)),
        "duration_s": duration,
        "tracking_success_rate": tracking_success_rate,
        "time_offset_s": time_offset,
        "time_offset_convention": "t_mocap = t_vo + time_offset_s",
        "ate_rmse_m": rmse(trans_err),
        "ate_mean_m": float(np.mean(trans_err)),
        "ate_max_m": float(np.max(trans_err)),
        "rot_rmse_deg": rmse(rot_err),
        "rot_mean_deg": float(np.mean(rot_err)),
        "rot_max_deg": float(np.max(rot_err)),
        "alignment": "SE(3) world alignment, no scale; rotation RMSE also estimates one constant marker-to-camera right rotation",
        "sim3_scale_diagnostic_not_applied": sim3_scale,
        "result_path": str(eval_dir),
        "vo_path": str(vo_path),
        "mocap_path": str(info.mocap_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def read_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_eval_csv(
    path: Path,
    vo: PoseTrajectory,
    mocap_pos_aligned: PoseTrajectory,
    mocap_rot_aligned: PoseTrajectory,
    trans_err: np.ndarray,
    rot_err: np.ndarray,
) -> None:
    safe_mkdir(path.parent)
    vo_rpy = rpy_deg(vo.rot)
    gt_rpy = rpy_deg(mocap_rot_aligned.rot)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "vo_x",
                "vo_y",
                "vo_z",
                "vo_roll_deg",
                "vo_pitch_deg",
                "vo_yaw_deg",
                "gt_x",
                "gt_y",
                "gt_z",
                "gt_roll_deg",
                "gt_pitch_deg",
                "gt_yaw_deg",
                "translation_error_m",
                "rotation_error_deg",
            ]
        )
        for i in range(len(vo)):
            writer.writerow(
                [
                    f"{vo.t[i]:.9f}",
                    *[f"{value:.12f}" for value in vo.p[i]],
                    *[f"{value:.9f}" for value in vo_rpy[i]],
                    *[f"{value:.12f}" for value in mocap_pos_aligned.p[i]],
                    *[f"{value:.9f}" for value in gt_rpy[i]],
                    f"{trans_err[i]:.12f}",
                    f"{rot_err[i]:.9f}",
                ]
            )


def write_dict_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    safe_mkdir(path.parent)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_comparison_plots(
    fig_dir: Path,
    estimate: PoseTrajectory,
    reference: PoseTrajectory,
    estimate_label: str,
    reference_label: str,
    trans_err: Optional[np.ndarray] = None,
    rot_err: Optional[np.ndarray] = None,
) -> None:
    safe_mkdir(fig_dir)
    t = estimate.t - estimate.t[0]
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for axis in range(3):
        axes[axis].plot(t, estimate.p[:, axis], label=estimate_label, linewidth=1.1)
        axes[axis].plot(t, reference.p[:, axis], label=reference_label, linestyle="--", linewidth=1.0)
        axes[axis].set_ylabel(f"{labels[axis]} [m]")
        axes[axis].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(fig_dir / "position_xyz_comparison.png", dpi=180)
    plt.close(fig)

    est_rpy = rpy_deg(estimate.rot)
    ref_rpy = rpy_deg(reference.rot)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for axis, label in enumerate(["roll", "pitch", "yaw"]):
        axes[axis].plot(t, est_rpy[:, axis], label=estimate_label, linewidth=1.1)
        axes[axis].plot(t, ref_rpy[:, axis], label=reference_label, linestyle="--", linewidth=1.0)
        axes[axis].set_ylabel(f"{label} [deg]")
        axes[axis].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(fig_dir / "orientation_rpy_comparison.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(reference.p[:, 0], reference.p[:, 1], reference.p[:, 2], label=reference_label)
    ax.plot(estimate.p[:, 0], estimate.p[:, 1], estimate.p[:, 2], label=estimate_label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "trajectory_3d_comparison.png", dpi=180)
    plt.close(fig)

    if trans_err is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, trans_err, linewidth=1.1)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("translation residual [m]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "residual_translation_norm.png", dpi=180)
        plt.close(fig)
    if rot_err is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, rot_err, linewidth=1.1)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("rotation residual [deg]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "residual_rotation_angle.png", dpi=180)
        plt.close(fig)


def convert_vo_camera_to_imu(info: SequenceInfo, args: argparse.Namespace) -> Dict[str, object]:
    out_dir = args.imu_gt_output_dir / info.sequence
    safe_mkdir(out_dir)
    out_path = out_dir / "vo_imu_traj.txt"
    meta_path = out_dir / "vo_imu_metadata.json"
    if existing_file(out_path, args.overwrite):
        return read_json_if_exists(meta_path)
    if info.skip_reason:
        status = {"sequence": info.sequence, "status": "skipped", "reason": info.skip_reason}
        meta_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status
    vo_path = args.vo_output_dir / info.sequence / "raw" / "vo_camera_traj_seconds.txt"
    vo = read_pose_trajectory(vo_path, "vo_T_Wv_C", timestamp_unit="s", quat_order="xyzw")
    extrinsic = load_camera_imu_extrinsic(info.extrinsic_path)
    # If the calibration file is T_C_I, use it directly:
    #     T_Wv_I = T_Wv_C * T_C_I
    # If the calibration file is T_I_C, load_camera_imu_extrinsic already
    # inverted it to T_C_I before this right multiplication.
    vo_imu = transform_by_right(vo, extrinsic.T_C_I, "vo_T_Wv_I")
    write_tum_xyzw(out_path, vo_imu)
    write_matrix_yaml(
        out_dir / "T_C_I_used.yaml",
        "T_C_I",
        extrinsic.T_C_I,
        {
            "source_file": str(extrinsic.path),
            "source_key": extrinsic.source_key,
            "source_convention": extrinsic.convention,
            "note": extrinsic.note,
        },
    )
    metadata = {
        "sequence": info.sequence,
        "status": "converted",
        "vo_camera_path": str(vo_path),
        "vo_imu_path": str(out_path),
        "extrinsic_path": str(extrinsic.path),
        "extrinsic_source_key": extrinsic.source_key,
        "extrinsic_source_convention": extrinsic.convention,
        "used_formula": "T_Wv_I = T_Wv_C * T_C_I; if source was T_I_C, T_C_I = inv(T_I_C).",
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def residual_vector(data_vo: PoseTrajectory, data_mocap: PoseTrajectory, T_Wv_Wm: np.ndarray, T_M_I: np.ndarray) -> np.ndarray:
    vo_mats = matrices_from_traj(data_vo)
    mocap_mats = matrices_from_traj(data_mocap)
    residuals = np.empty((len(data_vo), 6), dtype=float)
    for i in range(len(data_vo)):
        predicted = T_Wv_Wm @ mocap_mats[i] @ T_M_I
        error = invert_T(vo_mats[i]) @ predicted
        residuals[i] = se3_log(error)
    return residuals.reshape(-1)


def residual_stats(data_vo: PoseTrajectory, data_mocap: PoseTrajectory, T_Wv_Wm: np.ndarray, T_M_I: np.ndarray) -> Dict[str, float]:
    vo_mats = matrices_from_traj(data_vo)
    mocap_mats = matrices_from_traj(data_mocap)
    trans: List[float] = []
    rot: List[float] = []
    se3 = residual_vector(data_vo, data_mocap, T_Wv_Wm, T_M_I)
    for i in range(len(data_vo)):
        predicted = T_Wv_Wm @ mocap_mats[i] @ T_M_I
        error = invert_T(vo_mats[i]) @ predicted
        trans.append(float(np.linalg.norm(error[:3, 3])))
        rot.append(float(Rotation.from_matrix(error[:3, :3]).magnitude() * 180.0 / math.pi))
    return {
        "se3_residual_rmse": rmse(se3),
        "translation_rmse_m": rmse(np.asarray(trans)),
        "translation_mean_m": float(np.mean(trans)),
        "translation_max_m": float(np.max(trans)),
        "rotation_rmse_deg": rmse(np.asarray(rot)),
        "rotation_mean_deg": float(np.mean(rot)),
        "rotation_max_deg": float(np.max(rot)),
    }


def estimate_marker_to_imu(info: SequenceInfo, args: argparse.Namespace) -> CalibrationResult:
    out_dir = args.imu_gt_output_dir / info.sequence
    safe_mkdir(out_dir)
    report_path = out_dir / "calibration_report.txt"
    if existing_file(report_path, args.overwrite):
        payload = read_json_if_exists(out_dir / "calibration_result.json")
        return CalibrationResult(
            T_Wv_Wm=np.asarray(payload["T_Wv_Wm"], dtype=float),
            T_M_I=np.asarray(payload["T_M_I"], dtype=float),
            initial_stats=payload["initial_stats"],
            final_stats=payload["final_stats"],
            optimizer=None,
            sample_count=int(payload["sample_count"]),
        )
    vo_imu = read_pose_trajectory(out_dir / "vo_imu_traj.txt", "vo_T_Wv_I", timestamp_unit="s", quat_order="xyzw")
    mocap = read_pose_trajectory(info.mocap_path, "mocap_T_Wm_M", timestamp_unit="auto", quat_order="xyzw", position_unit="auto")
    eval_metrics = read_json_if_exists(args.eval_output_dir / info.sequence / "metrics.json")
    time_offset = float(eval_metrics.get("time_offset_s", args.time_offset_s))
    vo_sync, mocap_sync = synchronized_vo_mocap(vo_imu, mocap, time_offset, args.max_calib_samples, args.min_sync_samples)

    T_Wv_Wm_init = umeyama_se3(mocap_sync.p, vo_sync.p)
    T_M_I_init = np.eye(4)
    initial_stats = residual_stats(vo_sync, mocap_sync, T_Wv_Wm_init, T_M_I_init)

    def fun(x: np.ndarray) -> np.ndarray:
        T_Wv_Wm = se3_exp(x[:6]) @ T_Wv_Wm_init
        T_M_I = se3_exp(x[6:]) @ T_M_I_init
        return residual_vector(vo_sync, mocap_sync, T_Wv_Wm, T_M_I)

    result = least_squares(
        fun,
        np.zeros(12, dtype=float),
        loss=args.robust_loss,
        f_scale=args.loss_f_scale,
        max_nfev=args.max_nfev,
    )
    T_Wv_Wm = se3_exp(result.x[:6]) @ T_Wv_Wm_init
    T_M_I = se3_exp(result.x[6:]) @ T_M_I_init
    final_stats = residual_stats(vo_sync, mocap_sync, T_Wv_Wm, T_M_I)

    write_matrix_yaml(
        out_dir / "T_Wv_Wm.yaml",
        "T_Wv_Wm",
        T_Wv_Wm,
        {"meaning": "mocap world Wm to VO world Wv", "time_offset_s": time_offset},
    )
    write_matrix_yaml(
        out_dir / "T_M_I.yaml",
        "T_M_I",
        T_M_I,
        {"meaning": "IMU frame I to mocap marker frame M"},
    )
    payload = {
        "sequence": info.sequence,
        "time_offset_s": time_offset,
        "sample_count": int(len(vo_sync)),
        "model": "min sum || log(inv(T_Wv_I_vo) * T_Wv_Wm * T_Wm_M * T_M_I) ||^2",
        "T_Wv_Wm": T_Wv_Wm.tolist(),
        "T_M_I": T_M_I.tolist(),
        "initial_stats": initial_stats,
        "final_stats": final_stats,
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
        },
    }
    (out_dir / "calibration_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = [
        f"Sequence: {info.sequence}",
        "Convention: T_AB maps points from B to A.",
        "Optimization model:",
        "  minimize sum || log(inv(T_Wv_I_vo(t)) * T_Wv_Wm * T_Wm_M(t) * T_M_I) ||^2",
        f"Time offset convention: t_mocap = t_vo + {time_offset:.9f}",
        f"Samples used: {len(vo_sync)}",
        f"Robust loss: {args.robust_loss}, f_scale={args.loss_f_scale}",
        "",
        "Initial residuals:",
        json.dumps(initial_stats, indent=2, sort_keys=True),
        "",
        "Final residuals:",
        json.dumps(final_stats, indent=2, sort_keys=True),
        "",
        "T_M_I (IMU frame I to mocap marker frame M):",
        np.array2string(T_M_I, precision=9, suppress_small=False),
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return CalibrationResult(T_Wv_Wm, T_M_I, initial_stats, final_stats, result, len(vo_sync))


def generate_imu_gt(info: SequenceInfo, calibration: CalibrationResult, args: argparse.Namespace) -> Dict[str, object]:
    out_dir = args.imu_gt_output_dir / info.sequence
    safe_mkdir(out_dir)
    gt_path = out_dir / "gt_imu_traj.txt"
    first_path = out_dir / "gt_imu_traj_first_frame_identity.txt"
    metrics_path = out_dir / "imu_gt_metrics.json"
    if existing_file(gt_path, args.overwrite) and existing_file(first_path, args.overwrite):
        return read_json_if_exists(metrics_path)
    vo_imu = read_pose_trajectory(out_dir / "vo_imu_traj.txt", "vo_T_Wv_I", timestamp_unit="s", quat_order="xyzw")
    mocap = read_pose_trajectory(info.mocap_path, "mocap_T_Wm_M", timestamp_unit="auto", quat_order="xyzw", position_unit="auto")
    eval_metrics = read_json_if_exists(args.eval_output_dir / info.sequence / "metrics.json")
    time_offset = float(eval_metrics.get("time_offset_s", args.time_offset_s))

    if args.gt_timestamp_mode == "vo":
        mask = trajectory_overlap_mask(vo_imu.t, mocap, time_offset)
        output_t = vo_imu.t[mask]
        mocap_interp = interpolate_trajectory(mocap, output_t + time_offset, "mocap_for_gt_on_vo_times")
        mocap_interp.t = output_t.copy()
    else:
        corrected_t = mocap.t - time_offset
        mask = (corrected_t >= vo_imu.t[0]) & (corrected_t <= vo_imu.t[-1])
        if int(np.sum(mask)) < args.min_sync_samples:
            raise ValueError(f"Only {int(np.sum(mask))} mocap output samples overlap VO range")
        mocap_interp = PoseTrajectory(
            name="mocap_for_gt_on_mocap_times",
            t=corrected_t[mask],
            p=mocap.p[mask],
            rot=mocap.rot[mask],
        )
    gt_mats = matrices_from_traj(mocap_interp)
    for i in range(len(mocap_interp)):
        gt_mats[i] = calibration.T_Wv_Wm @ gt_mats[i] @ calibration.T_M_I
    gt = traj_from_matrices("gt_T_Wv_I", mocap_interp.t, gt_mats)
    write_tum_xyzw(gt_path, gt)

    first_inv = invert_T(make_T(gt.p[0], gt.rot[0]))
    gt_first = transform_by_left(gt, first_inv, "gt_T_I0_It")
    write_tum_xyzw(first_path, gt_first)

    vo_sync, gt_on_vo = sync_gt_to_vo(vo_imu, gt)
    err = relative_errors(vo_sync, gt_on_vo)
    fig_dir = out_dir / "figures"
    make_comparison_plots(
        fig_dir,
        vo_sync,
        gt_on_vo,
        "VO-derived IMU",
        "Converted mocap IMU GT",
        err["translation_error_m"],
        err["rotation_error_deg"],
    )
    metrics = {
        "sequence": info.sequence,
        "status": "generated",
        "time_offset_s": time_offset,
        "gt_timestamp_mode": args.gt_timestamp_mode,
        "gt_samples": int(len(gt)),
        "vo_comparison_samples": int(len(vo_sync)),
        "translation_rmse_m": rmse(err["translation_error_m"]),
        "translation_mean_m": float(np.mean(err["translation_error_m"])),
        "translation_max_m": float(np.max(err["translation_error_m"])),
        "rotation_rmse_deg": rmse(err["rotation_error_deg"]),
        "rotation_mean_deg": float(np.mean(err["rotation_error_deg"])),
        "rotation_max_deg": float(np.max(err["rotation_error_deg"])),
        "gt_imu_traj": str(gt_path),
        "gt_imu_traj_first_frame_identity": str(first_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def sync_gt_to_vo(vo_imu: PoseTrajectory, gt: PoseTrajectory) -> Tuple[PoseTrajectory, PoseTrajectory]:
    mask = (vo_imu.t >= gt.t[0]) & (vo_imu.t <= gt.t[-1])
    indices = np.where(mask)[0]
    if len(indices) < 3:
        raise ValueError("Not enough VO samples overlap GT for validation plots")
    vo_sync = PoseTrajectory("vo_T_Wv_I_sync", vo_imu.t[indices], vo_imu.p[indices], vo_imu.rot[indices])
    gt_interp = interpolate_trajectory(gt, vo_sync.t, "gt_T_Wv_I_at_vo")
    return vo_sync, gt_interp


def select_sequences(eval_results: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    candidates = []
    for item in eval_results:
        if item.get("status") != "evaluated":
            continue
        selected = (
            float(item.get("ate_rmse_m", float("inf"))) <= args.max_ate_rmse_m
            and float(item.get("rot_rmse_deg", float("inf"))) <= args.max_rot_rmse_deg
            and float(item.get("tracking_success_rate", 0.0)) >= args.min_tracking_success_rate
        )
        item["selected"] = selected
        item["selection_status"] = "selected" if selected else "not_selected"
        candidates.append(item)
    candidates.sort(
        key=lambda row: (
            float(row.get("ate_rmse_m", float("inf"))),
            float(row.get("rot_rmse_deg", float("inf"))),
            -float(row.get("tracking_success_rate", 0.0)),
        )
    )
    return [str(item["sequence"]) for item in candidates if item.get("selected")]


def write_summary(eval_results: Sequence[Dict[str, object]], args: argparse.Namespace, selected: Sequence[str]) -> None:
    safe_mkdir(args.eval_output_dir)
    selected_set = set(selected)
    rows: List[Dict[str, object]] = []
    for item in eval_results:
        sequence = str(item.get("sequence", ""))
        status = item.get("status", "unknown")
        selection_status = "selected" if sequence in selected_set else "not_selected"
        if status != "evaluated":
            selection_status = "skipped"
        rows.append(
            {
                "sequence": sequence,
                "split": item.get("split", ""),
                "num_frames": item.get("num_frames", 0),
                "duration": item.get("duration_s", ""),
                "tracking_success_rate": item.get("tracking_success_rate", ""),
                "ate_rmse_m": item.get("ate_rmse_m", ""),
                "rot_rmse_deg": item.get("rot_rmse_deg", ""),
                "time_offset_s": item.get("time_offset_s", ""),
                "selection_status": selection_status,
                "result_path": item.get("result_path", ""),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["ate_rmse_m"]) if row["ate_rmse_m"] != "" else float("inf"),
            float(row["rot_rmse_deg"]) if row["rot_rmse_deg"] != "" else float("inf"),
        )
    )
    write_dict_csv(args.eval_output_dir / "summary.csv", rows)
    safe_mkdir(args.imu_gt_output_dir)
    write_dict_csv(args.imu_gt_output_dir / "summary.csv", rows)
    (args.imu_gt_output_dir / "selected_sequences.txt").write_text(
        "".join(f"{sequence}\n" for sequence in selected), encoding="utf-8"
    )


def write_report(
    sequences: Sequence[SequenceInfo],
    eval_results: Sequence[Dict[str, object]],
    selected: Sequence[str],
    imu_metrics: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    safe_mkdir(args.imu_gt_output_dir)
    eval_by_seq = {str(item.get("sequence")): item for item in eval_results}
    selected_set = set(selected)
    lines = [
        "# 0203 Stereo VO and IMU GT Report",
        "",
        f"Dataset root: `{args.dataset_root}`",
        f"ORB-SLAM3 mode: stereo visual odometry only, no IMU factors.",
        f"Stereo config: `{args.config_path}`",
        f"Time offset convention: `t_mocap = t_vo + time_offset_s`.",
        f"Selection thresholds: ATE <= {args.max_ate_rmse_m} m, rotation <= {args.max_rot_rmse_deg} deg, tracking >= {args.min_tracking_success_rate}.",
        "",
        "## Sequence Summary",
        "",
        "| sequence | split | tracking | ATE RMSE m | rot RMSE deg | selected | notes |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for info in sequences:
        item = eval_by_seq.get(info.sequence, {})
        notes = info.skip_reason or item.get("reason", "")
        lines.append(
            "| {seq} | {split} | {track} | {ate} | {rot} | {sel} | {notes} |".format(
                seq=info.sequence,
                split=info.split,
                track=format_float(item.get("tracking_success_rate")),
                ate=format_float(item.get("ate_rmse_m")),
                rot=format_float(item.get("rot_rmse_deg")),
                sel="yes" if info.sequence in selected_set else "no",
                notes=notes,
            )
        )
    lines.extend(["", "## IMU-Camera Extrinsics", ""])
    for info in sequences:
        if info.sequence not in selected_set:
            continue
        meta = read_json_if_exists(args.imu_gt_output_dir / info.sequence / "vo_imu_metadata.json")
        lines.extend(
            [
                f"### {info.sequence}",
                f"- Extrinsic file: `{meta.get('extrinsic_path', info.extrinsic_path)}`",
                f"- Source key: `{meta.get('extrinsic_source_key', '')}`",
                f"- Source convention: `{meta.get('extrinsic_source_convention', '')}`",
                f"- Formula: `{meta.get('used_formula', '')}`",
            ]
        )
    lines.extend(["", "## Marker-to-IMU Calibration", ""])
    for sequence in selected:
        cal = read_json_if_exists(args.imu_gt_output_dir / sequence / "calibration_result.json")
        metrics = imu_metrics.get(sequence, {})
        lines.extend(
            [
                f"### {sequence}",
                f"- T_M_I file: `{args.imu_gt_output_dir / sequence / 'T_M_I.yaml'}`",
                f"- Calibration residual translation RMSE: {format_float(cal.get('final_stats', {}).get('translation_rmse_m'))} m",
                f"- Calibration residual rotation RMSE: {format_float(cal.get('final_stats', {}).get('rotation_rmse_deg'))} deg",
                f"- Converted IMU GT vs VO-IMU translation RMSE: {format_float(metrics.get('translation_rmse_m'))} m",
                f"- Converted IMU GT vs VO-IMU rotation RMSE: {format_float(metrics.get('rotation_rmse_deg'))} deg",
            ]
        )
        if cal.get("T_M_I"):
            lines.append("")
            lines.append("```text")
            lines.append(np.array2string(np.asarray(cal["T_M_I"]), precision=6, suppress_small=False))
            lines.append("```")
    lines.extend(
        [
            "",
            "## Recommended Training Sequences",
            "",
            "\n".join(f"- `{sequence}`" for sequence in selected) if selected else "No sequence satisfied the configured thresholds.",
            "",
            "## Required Outputs",
            "",
            f"- Summary CSV: `{args.eval_output_dir / 'summary.csv'}`",
            f"- Selected sequence list: `{args.imu_gt_output_dir / 'selected_sequences.txt'}`",
        ]
    )
    (args.imu_gt_output_dir / "0203_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: object) -> str:
    try:
        if value is None or value == "":
            return ""
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def run_pipeline(args: argparse.Namespace) -> None:
    safe_mkdir(args.vo_output_dir)
    safe_mkdir(args.eval_output_dir)
    safe_mkdir(args.imu_gt_output_dir)
    all_sequences = discover_sequences(args.dataset_root, args.splits)
    sequences = filter_sequences(all_sequences, args.sequence)
    if not sequences:
        raise RuntimeError("No 0203 sequences matched the requested split/sequence filters")

    if args.stage in {"all", "scan"}:
        scan_rows = [
            {
                "sequence": info.sequence,
                "split": info.split,
                "bag_path": str(info.bag_path),
                "mocap_path": str(info.mocap_path or ""),
                "extrinsic_path": str(info.extrinsic_path or ""),
                "skip_reason": info.skip_reason,
            }
            for info in sequences
        ]
        write_dict_csv(args.eval_output_dir / "discovered_sequences.csv", scan_rows)
        if args.stage == "scan":
            print(json.dumps({"discovered": scan_rows}, indent=2, ensure_ascii=False))
            return

    if args.stage in {"all", "stereo_vo"}:
        for info in sequences:
            print(f"[stereo_vo] {info.sequence}")
            run_stereo_vo_for_sequence(info, args)
        if args.stage == "stereo_vo":
            return

    eval_results: List[Dict[str, object]] = []
    if args.stage in {"all", "evaluate", "convert", "calibrate", "imu_gt", "report"}:
        for info in sequences:
            print(f"[evaluate] {info.sequence}")
            eval_results.append(evaluate_sequence(info, args))
        selected = select_sequences(eval_results, args)
        write_summary(eval_results, args, selected)
        if args.stage == "evaluate":
            write_report(sequences, eval_results, selected, {}, args)
            return
    else:
        eval_results = [read_json_if_exists(path) for path in sorted(args.eval_output_dir.glob("*/metrics.json"))]
        selected = [line.strip() for line in (args.imu_gt_output_dir / "selected_sequences.txt").read_text().splitlines() if line.strip()] if (args.imu_gt_output_dir / "selected_sequences.txt").exists() else []

    selected_infos = [info for info in sequences if info.sequence in set(selected)]
    if args.stage in {"all", "convert", "calibrate", "imu_gt"}:
        for info in selected_infos:
            print(f"[convert] {info.sequence}")
            convert_vo_camera_to_imu(info, args)
        if args.stage == "convert":
            return

    calibrations: Dict[str, CalibrationResult] = {}
    if args.stage in {"all", "calibrate", "imu_gt"}:
        for info in selected_infos:
            print(f"[calibrate] {info.sequence}")
            calibrations[info.sequence] = estimate_marker_to_imu(info, args)
        if args.stage == "calibrate":
            return

    imu_metrics: Dict[str, Dict[str, object]] = {}
    if args.stage in {"all", "imu_gt"}:
        for info in selected_infos:
            print(f"[imu_gt] {info.sequence}")
            calibration = calibrations.get(info.sequence)
            if calibration is None:
                calibration = estimate_marker_to_imu(info, args)
            imu_metrics[info.sequence] = generate_imu_gt(info, calibration, args)
        if args.stage == "imu_gt":
            write_report(sequences, eval_results, selected, imu_metrics, args)
            return

    if args.stage in {"all", "report"}:
        for sequence in selected:
            imu_metrics[sequence] = read_json_if_exists(args.imu_gt_output_dir / sequence / "imu_gt_metrics.json")
        write_report(sequences, eval_results, selected, imu_metrics, args)


def build_parser(default_stage: str = "all") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PLATO 0203 stereo-VO to IMU-GT pipeline.")
    parser.add_argument("--stage", choices=["all", "scan", "stereo_vo", "evaluate", "convert", "calibrate", "imu_gt", "report"], default=default_stage)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--splits", nargs="+", default=["Training", "Testing"], help="LIASuw splits to scan: Training Testing Unusable")
    parser.add_argument("--sequence", nargs="*", default=[], help="Optional sequence filters, e.g. 0203S08 s09")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vo-output-dir", type=Path, default=None)
    parser.add_argument("--eval-output-dir", type=Path, default=None)
    parser.add_argument("--imu-gt-output-dir", type=Path, default=None)
    parser.add_argument("--orb-root", type=Path, default=ORB_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_STEREO_CONFIG)
    parser.add_argument("--export-python", default="/usr/bin/python3")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional smoke-test frame cap before ORB-SLAM3.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional first-N-seconds truncation after bag export.")
    parser.add_argument("--time-offset-s", type=float, default=None, help="Association convention: t_mocap = t_vo + offset.")
    parser.add_argument("--estimate-time-offset", action="store_true", default=True)
    parser.add_argument("--no-estimate-time-offset", dest="estimate_time_offset", action="store_false")
    parser.add_argument("--time-offset-search-min", type=float, default=-0.20)
    parser.add_argument("--time-offset-search-max", type=float, default=0.20)
    parser.add_argument("--time-offset-search-step", type=float, default=0.02)
    parser.add_argument("--min-sync-samples", type=int, default=20)
    parser.add_argument("--max-eval-samples", type=int, default=2500)
    parser.add_argument("--max-calib-samples", type=int, default=2000)
    parser.add_argument("--max-ate-rmse-m", type=float, default=0.05)
    parser.add_argument("--max-rot-rmse-deg", type=float, default=4.0)
    parser.add_argument("--min-tracking-success-rate", type=float, default=0.90)
    parser.add_argument("--robust-loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="huber")
    parser.add_argument("--loss-f-scale", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--gt-timestamp-mode", choices=["mocap", "vo"], default="mocap")
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.dataset_root = Path(args.dataset_root).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()
    args.vo_output_dir = Path(args.vo_output_dir).expanduser().resolve() if args.vo_output_dir else args.output_root / "0203_stereo_vo"
    args.eval_output_dir = Path(args.eval_output_dir).expanduser().resolve() if args.eval_output_dir else args.output_root / "0203_eval"
    args.imu_gt_output_dir = Path(args.imu_gt_output_dir).expanduser().resolve() if args.imu_gt_output_dir else args.output_root / "0203_imu_gt"
    args.orb_root = Path(args.orb_root).expanduser().resolve()
    args.config_path = Path(args.config_path).expanduser().resolve()
    if args.time_offset_s is None:
        args.time_offset_s = parse_default_time_offset(args.dataset_root)
    return args


def main(default_stage: str = "all") -> None:
    parser = build_parser(default_stage)
    args = finalize_args(parser.parse_args())
    run_pipeline(args)
