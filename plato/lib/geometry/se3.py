"""Small SE(3)/SO(3) helpers shared by PLATO scripts."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(position: np.ndarray, rot: Rotation) -> np.ndarray:
    """Create a 4x4 transform from translation and SciPy rotation."""

    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""

    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    Rm = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Fit the no-scale SE(3) transform mapping `src` points to `dst`."""

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both have shape (N, 3)")
    if src.shape[0] < 3:
        raise ValueError("At least three points are required for SE(3) alignment")
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
    """Return the Umeyama Sim(3) scale diagnostic without applying it."""

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both have shape (N, 3)")
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
    """Project the arithmetic mean of rotation matrices back onto SO(3)."""

    matrix = np.mean(rotations.as_matrix(), axis=0)
    u_mat, _, vh_mat = np.linalg.svd(matrix)
    Rm = u_mat @ vh_mat
    if np.linalg.det(Rm) < 0:
        u_mat[:, -1] *= -1.0
        Rm = u_mat @ vh_mat
    return Rotation.from_matrix(Rm)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
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
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * K + (K @ K) / 12.0
    theta2 = theta * theta
    coeff = (1.0 / theta2) - ((1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta)))
    return np.eye(3) - 0.5 * K + coeff * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential with tangent order `[rho, phi]`."""

    xi = np.asarray(xi, dtype=float).reshape(6)
    rho = xi[:3]
    phi = xi[3:]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(phi).as_matrix()
    T[:3, 3] = so3_left_jacobian(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) logarithm with tangent order `[rho, phi]`."""

    T = np.asarray(T, dtype=float)
    phi = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    rho = so3_left_jacobian_inv(phi) @ T[:3, 3]
    return np.concatenate([rho, phi])


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))
