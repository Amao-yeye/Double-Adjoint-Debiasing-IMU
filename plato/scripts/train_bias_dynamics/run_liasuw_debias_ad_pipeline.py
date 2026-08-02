#!/usr/bin/env python3
"""Run the Debias_IMU AD training/test pipeline on LIASuw body GT.

This is PLATO glue code.  It intentionally leaves external/Debias_IMU untouched
and reuses its model, loss, training, and test functions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_ROOT = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU"
BIASDY_ROOT = DEBIAS_ROOT / "BiasDy"
MPL_CACHE_DIR = PLATO_ROOT / "results" / "debias_imu" / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

for path in (str(DEBIAS_ROOT), str(BIASDY_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

import Interpolation  # noqa: E402
import lie_algebra as Lie  # noqa: E402
import learning as debias_learning  # noqa: E402
from covariance import paper_invariant_error_matrices  # noqa: E402
from dataset import BaseDataset, pdump, pload  # noqa: E402
from learning import (  # noqa: E402
    Acc_train_SE23,
    Gyro_train,
    format_training_profile,
    plot_bias,
    simple_visualization,
)
from m2_bias_networks import (  # noqa: E402
    ACC_FBU_RESIDUAL_NAME,
    ACC_BASELINE_NO_UDOT_NAME,
    ACC_CONSERVATIVE_RESIDUAL_NAME,
    ACC_M2_NAME,
    ACC_PHYS_DELTA_NAME,
    ACC_PHYS_EQ_NAME,
    ACC_PHYS_RATE_NAME,
    ACC_SLOW_RESIDUAL_NAME,
    GYRO_FBU_RESIDUAL_NAME,
    GYRO_BASELINE_NO_UDOT_NAME,
    GYRO_CONSERVATIVE_RESIDUAL_NAME,
    GYRO_M2_NAME,
    GYRO_PHYS_DELTA_NAME,
    GYRO_PHYS_EQ_NAME,
    GYRO_PHYS_RATE_NAME,
    GYRO_SLOW_RESIDUAL_NAME,
    GYRO_SLOW_RESIDUAL_TINY_NAME,
    ACC_SLOW_RESIDUAL_TINY_NAME,
    ba_func_choose_with_m2,
    bw_func_choose_with_m2,
)
from network_for_BiadDy import (  # noqa: E402
    ba_func_choose as debias_ba_func_choose,
    bw_func_choose as debias_bw_func_choose,
)


DEFAULT_TRAIN_SEQS = [
    "0912s06_80",
    "0912s07_30",
    "0912s10_40",
    "0203s18_80",
    "0203s19_80",
    "0203s22_80",
]
DEFAULT_TEST_SEQS = ["0205s14_80"]
MOCAP_200HZ_SUPERVISION = "mocap_200hz"
MOCAP_NONINTERP_SUPERVISION = "mocap_noninterp"
VO_IMU_NONINTERP_SUPERVISION = "vo_imu_noninterp"
DEFAULT_GRAVITY_ACCEL_WORLD = (0.0, 0.0, -9.81)
GRAVITY_ALIGNMENT_ROUTE = "world_align_to_default_gravity"


def bw_func_choose(network_type: str, device: str = "cuda"):
    return bw_func_choose_with_m2(network_type, device=device, fallback=debias_bw_func_choose)


def ba_func_choose(network_type: str, device: str = "cuda"):
    return ba_func_choose_with_m2(network_type, device=device, fallback=debias_ba_func_choose)


debias_learning.bw_func_choose = bw_func_choose
debias_learning.ba_func_choose = ba_func_choose

DELTA_IMU_NETWORKS = {GYRO_PHYS_DELTA_NAME, ACC_PHYS_DELTA_NAME}
NO_DERIV_NETWORKS = {
    GYRO_M2_NAME,
    ACC_M2_NAME,
    GYRO_PHYS_EQ_NAME,
    ACC_PHYS_EQ_NAME,
    GYRO_PHYS_RATE_NAME,
    ACC_PHYS_RATE_NAME,
    GYRO_PHYS_DELTA_NAME,
    ACC_PHYS_DELTA_NAME,
    GYRO_FBU_RESIDUAL_NAME,
    ACC_FBU_RESIDUAL_NAME,
    GYRO_SLOW_RESIDUAL_NAME,
    ACC_SLOW_RESIDUAL_NAME,
    GYRO_SLOW_RESIDUAL_TINY_NAME,
    ACC_SLOW_RESIDUAL_TINY_NAME,
    GYRO_CONSERVATIVE_RESIDUAL_NAME,
    ACC_CONSERVATIVE_RESIDUAL_NAME,
    GYRO_BASELINE_NO_UDOT_NAME,
    ACC_BASELINE_NO_UDOT_NAME,
}


class SplineDeltaIMU:
    """Discrete no-derivative IMU change feature: u(t) - u(t - dt)."""

    def __init__(self, spline: Interpolation.CubicHermiteSpline, dt: float):
        self.spline = spline
        self.dt = float(dt)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        dt = torch.as_tensor(self.dt, dtype=t.dtype, device=t.device)
        t_min = self.spline._times[0].to(dtype=t.dtype, device=t.device)
        t_prev = torch.maximum(t - dt, t_min)
        return self.spline.evaluate(t) - self.spline.evaluate(t_prev)


def _uses_delta_imu(*network_names: str | None) -> bool:
    return any(name in DELTA_IMU_NETWORKS for name in network_names if name is not None)


def _requires_spline_derivative(network_name: str) -> bool:
    return network_name not in NO_DERIV_NETWORKS


def _set_model_imu_functions(model, spline: Interpolation.CubicHermiteSpline, dt: float, use_delta_imu: bool) -> None:
    model.set_u_func(spline.evaluate)
    model.set_u_dot_func(SplineDeltaIMU(spline, dt) if use_delta_imu else spline.derivative)


def _mask_flag(dataset: BaseDataset) -> bool:
    return {"EUROC": False, "TUM": True, "Fetch": False}[dataset.dataset_name]


def _rotation_rmse_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> float:
    rotvec = Lie.SO3log(R_pred.transpose(-1, -2) @ R_gt)
    return float(torch.sqrt(rotvec.square().sum(dim=-1).mean()).item() * 180.0 / np.pi)


def _vector_rmse(vec_pred: torch.Tensor, vec_gt: torch.Tensor) -> float:
    return float(torch.sqrt((vec_pred - vec_gt).square().sum(dim=-1).mean()).item())


def _select_initial_bias(
    bias: torch.Tensor,
    t_bias: torch.Tensor,
    mode: str,
    window_s: float,
    device: str,
) -> torch.Tensor:
    bias = bias.to(device)
    t_bias = t_bias.to(device)
    if mode == "first":
        return bias[0]
    if window_s <= 0:
        raise ValueError("--initial-bias-window-s must be positive when initial bias mode is not 'first'.")
    mask = (t_bias >= t_bias[0]) & (t_bias <= t_bias[0] + float(window_s))
    if mask.ndim > 1:
        mask = mask.squeeze(-1)
    values = bias[mask]
    if values.numel() == 0:
        values = bias[:1]
    if mode == "mean_window":
        return values.mean(dim=0)
    if mode == "median_window":
        return values.median(dim=0).values
    raise ValueError(f"Unsupported initial bias mode: {mode}")


def _resolve_initial_bias(
    bias: torch.Tensor,
    t_bias: torch.Tensor,
    mode: str,
    window_s: float,
    device: str,
    fixed_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if fixed_bias is not None:
        return fixed_bias.to(device=device, dtype=bias.dtype)
    return _select_initial_bias(bias, t_bias, mode, window_s, device)


def _apply_train_initial_bias_modes(
    y0_batch: torch.Tensor,
    train_initial_bw_mode: str = "label",
    train_initial_ba_mode: str = "label",
) -> torch.Tensor:
    if train_initial_bw_mode not in {"label", "zero"}:
        raise ValueError(f"Unsupported train initial gyro-bias mode: {train_initial_bw_mode}")
    if train_initial_ba_mode not in {"label", "zero"}:
        raise ValueError(f"Unsupported train initial accel-bias mode: {train_initial_ba_mode}")
    if train_initial_bw_mode == "zero":
        y0_batch[..., 4:7] = 0.0
    if train_initial_ba_mode == "zero":
        y0_batch[..., 13:16] = 0.0
    return y0_batch


def _weighted_se23_loss(
    X_pred: torch.Tensor,
    X_gt: torch.Tensor,
    rot_weight: float,
    vel_weight: float,
    pos_weight: float,
) -> torch.Tensor:
    xi = Lie.SEn3log(Lie.SEn3inverse(X_gt) @ X_pred)
    weights = torch.tensor(
        [rot_weight] * 3 + [vel_weight] * 3 + [pos_weight] * 3,
        dtype=xi.dtype,
        device=xi.device,
    )
    denom = weights.sum().clamp_min(torch.finfo(xi.dtype).eps)
    return (xi.square() * weights).sum() / (xi.numel() / xi.shape[-1] * denom)


def _bias_rate_regularization(solution: torch.Tensor, t_odeint: torch.Tensor, bias_slice: slice) -> torch.Tensor:
    if solution.shape[0] < 2:
        return torch.zeros((), dtype=solution.dtype, device=solution.device)
    dt = (t_odeint[1:] - t_odeint[:-1]).to(dtype=solution.dtype, device=solution.device)
    dt = dt.reshape((-1,) + (1,) * (solution.ndim - 1)).clamp_min(torch.finfo(solution.dtype).eps)
    bias_rate = (solution[1:, ..., bias_slice] - solution[:-1, ..., bias_slice]) / dt
    return bias_rate.square().mean()


def _nll_optimize_q(args: argparse.Namespace) -> bool:
    return bool(args.nll_optimize_q or args.nll_covariance_strategy in {"optimize_q", "q_only"})


def _nll_q_only(args: argparse.Namespace) -> bool:
    return args.nll_covariance_strategy == "q_only"


def _load_ba_model_initialization(ba_model: torch.nn.Module, checkpoint_path: str | Path | None, device: str) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("func_ba_model", checkpoint)
    ba_model.load_state_dict(state)
    print(f"Loaded Acc initialization from {checkpoint_path}", flush=True)


def _load_bw_model_initialization(bw_model: torch.nn.Module, checkpoint_path: str | Path | None, device: str) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("func_bw_model", checkpoint)
    bw_model.load_state_dict(state)
    print(f"Loaded Gyro initialization from {checkpoint_path}", flush=True)


def _select_nll_supervision_states(
    solution_aug: torch.Tensor,
    R_sol: torch.Tensor,
    X_gt_batch: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if mode == "per_state":
        return solution_aug, R_sol, X_gt_batch, 1
    if mode == "endpoint":
        return solution_aug[-1:], R_sol[-1:], X_gt_batch[-1:], 0
    raise ValueError(f"Unsupported NLL supervision mode: {mode}")


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(torch.finfo(value.dtype).tiny)
    return torch.log(torch.expm1(value).clamp_min(torch.finfo(value.dtype).tiny))


class IntervalNoiseVariances(torch.nn.Module):
    """Learn scalar IMU white-noise variances for interval preintegration factors."""

    def __init__(
        self,
        sigma_g_nominal: float,
        sigma_a_nominal: float,
        sigma_g_floor2: float,
        sigma_a_floor2: float,
        device: str,
    ):
        super().__init__()
        self.sigma_g_floor2 = float(sigma_g_floor2)
        self.sigma_a_floor2 = float(sigma_a_floor2)
        raw_g = self._make_raw_sigma2(sigma_g_nominal, self.sigma_g_floor2, device)
        raw_a = self._make_raw_sigma2(sigma_a_nominal, self.sigma_a_floor2, device)
        self.raw_sigma_g2 = torch.nn.Parameter(raw_g)
        self.raw_sigma_a2 = torch.nn.Parameter(raw_a)

    @staticmethod
    def _make_raw_sigma2(nominal_sigma: float, floor2: float, device: str) -> torch.Tensor:
        nominal2 = max(float(nominal_sigma) ** 2, float(floor2) + 1e-20)
        softplus_value = torch.tensor(nominal2 - float(floor2), dtype=torch.float32, device=device)
        return _softplus_inverse(softplus_value)

    def variances(self) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_g_floor2 = torch.as_tensor(
            self.sigma_g_floor2,
            dtype=self.raw_sigma_g2.dtype,
            device=self.raw_sigma_g2.device,
        )
        sigma_a_floor2 = torch.as_tensor(
            self.sigma_a_floor2,
            dtype=self.raw_sigma_a2.dtype,
            device=self.raw_sigma_a2.device,
        )
        sigma_g2 = sigma_g_floor2 + torch.nn.functional.softplus(self.raw_sigma_g2)
        sigma_a2 = sigma_a_floor2 + torch.nn.functional.softplus(self.raw_sigma_a2)
        return sigma_g2, sigma_a2

    def sigmas(self) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_g2, sigma_a2 = self.variances()
        return torch.sqrt(sigma_g2), torch.sqrt(sigma_a2)

    def prior(
        self,
        lambda_g: float,
        lambda_a: float,
        sigma_g_nominal: float,
        sigma_a_nominal: float,
    ) -> torch.Tensor:
        sigma_g, sigma_a = self.sigmas()
        loss = torch.zeros((), dtype=sigma_g.dtype, device=sigma_g.device)
        if lambda_g > 0:
            nominal_g = torch.as_tensor(sigma_g_nominal, dtype=sigma_g.dtype, device=sigma_g.device)
            loss = loss + float(lambda_g) * (torch.log(sigma_g) - torch.log(nominal_g)).square()
        if lambda_a > 0:
            nominal_a = torch.as_tensor(sigma_a_nominal, dtype=sigma_a.dtype, device=sigma_a.device)
            loss = loss + float(lambda_a) * (torch.log(sigma_a) - torch.log(nominal_a)).square()
        return loss


def _load_interval_noise_initialization(
    noise_model: IntervalNoiseVariances,
    checkpoint_path: str | Path | None,
    device: str,
) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    raw_g = checkpoint.get("raw_sigma_g2", checkpoint.get("interval_raw_sigma_g2"))
    raw_a = checkpoint.get("raw_sigma_a2", checkpoint.get("interval_raw_sigma_a2"))
    if raw_g is None or raw_a is None:
        print(f"No interval noise parameters found in {checkpoint_path}; using nominal initialization.", flush=True)
        return
    with torch.no_grad():
        noise_model.raw_sigma_g2.copy_(raw_g.to(device=device, dtype=noise_model.raw_sigma_g2.dtype))
        noise_model.raw_sigma_a2.copy_(raw_a.to(device=device, dtype=noise_model.raw_sigma_a2.dtype))
    sigma_g, sigma_a = noise_model.sigmas()
    print(
        f"Loaded interval noise initialization from {checkpoint_path}: "
        f"sigma_g={sigma_g.item():.8g}, sigma_a={sigma_a.item():.8g}",
        flush=True,
    )


def _interval_residual_mask(
    rot_weight: float,
    vel_weight: float,
    pos_weight: float,
    device: torch.device | str,
) -> torch.Tensor:
    mask = torch.tensor(
        [rot_weight > 0] * 3 + [vel_weight > 0] * 3 + [pos_weight > 0] * 3,
        dtype=torch.bool,
        device=device,
    )
    if not bool(mask.any()):
        raise ValueError("At least one of acc-loss-rot/vel/pos weights must be positive.")
    return mask


def _interval_preintegration_linearization(
    R_delta: torch.Tensor,
    dv_delta: torch.Tensor,
    dp_delta: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-order invariant interval covariance linearization.

    The covariance state is [delta_theta, delta_v, delta_p]. G_a and G_g are
    continuous-time unit-noise injection matrices; the caller applies * dt.
    """

    batch_shape = R_delta.shape[:-2]
    dtype = R_delta.dtype
    device = R_delta.device
    eye3 = torch.eye(3, dtype=dtype, device=device).expand(batch_shape + (3, 3))
    phi = torch.eye(9, dtype=dtype, device=device).expand(batch_shape + (9, 9)).clone()
    phi[..., 6:9, 3:6] = eye3 * dt.to(dtype=dtype, device=device)

    g_a = torch.zeros(batch_shape + (9, 3), dtype=dtype, device=device)
    g_g = torch.zeros_like(g_a)
    g_a[..., 3:6, :] = -R_delta
    g_g[..., 0:3, :] = -R_delta
    g_g[..., 3:6, :] = -(Lie.so3hat(dv_delta) @ R_delta)
    g_g[..., 6:9, :] = -(Lie.so3hat(dp_delta) @ R_delta)
    return phi, g_g, g_a


def _interval_preintegration_residual_nll(
    solution: torch.Tensor,
    R_sol: torch.Tensor,
    X_gt_batch: torch.Tensor,
    t_gt_batch: torch.Tensor,
    t_odeint: torch.Tensor,
    eval_indices: torch.Tensor,
    noise_model: IntervalNoiseVariances,
    gravity: torch.Tensor,
    residual_mask: torch.Tensor | None,
    target_cov_floor: float,
    jitter: float,
    detach_covariance_theta: bool,
    covariance_gradient_mode: str = "analytic_sensitivity",
    debug_covariance_gradients: bool = False,
    return_residuals: bool = False,
) -> tuple[torch.Tensor, dict]:
    if X_gt_batch.shape[0] < 2:
        raise ValueError("Interval preintegration NLL requires at least two supervised states per window.")
    if covariance_gradient_mode not in {"analytic_sensitivity", "ad"}:
        raise ValueError(f"Unsupported covariance_gradient_mode: {covariance_gradient_mode}")

    dtype = solution.dtype
    device = solution.device
    if solution.shape[0] == X_gt_batch.shape[0]:
        solution_eval = solution
        R_eval = R_sol
        supervision_indices = torch.arange(X_gt_batch.shape[0], dtype=torch.long, device=device)
    else:
        supervision_indices = eval_indices.to(device=device, dtype=torch.long)
        solution_eval = solution[supervision_indices]
        R_eval = R_sol[supervision_indices]

    R_gt = X_gt_batch[..., :3, :3]
    v_gt = X_gt_batch[..., :3, 3]
    p_gt = X_gt_batch[..., :3, 4]
    v_hat = solution_eval[..., 7:10]
    p_hat = solution_eval[..., 10:13]
    gravity = gravity.to(dtype=dtype, device=device)

    dt_interval = (t_gt_batch[1:] - t_gt_batch[:-1]).to(dtype=dtype, device=device).unsqueeze(-1)
    delta_R_hat = R_eval[:-1].transpose(-1, -2) @ R_eval[1:]
    delta_R_star = R_gt[:-1].transpose(-1, -2) @ R_gt[1:]
    e_R = Lie.SO3log(delta_R_hat.transpose(-1, -2) @ delta_R_star)

    dv_hat = (R_eval[:-1].transpose(-1, -2) @ (v_hat[1:] - v_hat[:-1] - gravity * dt_interval).unsqueeze(-1)).squeeze(-1)
    dv_star = (R_gt[:-1].transpose(-1, -2) @ (v_gt[1:] - v_gt[:-1] - gravity * dt_interval).unsqueeze(-1)).squeeze(-1)
    dp_hat = (
        R_eval[:-1].transpose(-1, -2)
        @ (
            p_hat[1:]
            - p_hat[:-1]
            - v_hat[:-1] * dt_interval
            - 0.5 * gravity * dt_interval.square()
        ).unsqueeze(-1)
    ).squeeze(-1)
    dp_star = (
        R_gt[:-1].transpose(-1, -2)
        @ (
            p_gt[1:]
            - p_gt[:-1]
            - v_gt[:-1] * dt_interval
            - 0.5 * gravity * dt_interval.square()
        ).unsqueeze(-1)
    ).squeeze(-1)
    residual_all = torch.cat((e_R, dv_star - dv_hat, dp_star - dp_hat), dim=-1)

    alpha_g, alpha_a = noise_model.variances()
    alpha_g = alpha_g.to(dtype=dtype)
    alpha_a = alpha_a.to(dtype=dtype)
    eye9 = torch.eye(9, dtype=dtype, device=device)
    supervision_indices_cpu = supervision_indices.detach().cpu().long()
    residual_chunks = []
    covariance_chunks = []
    sensitivity_g_chunks = []
    sensitivity_a_chunks = []

    for interval_idx in range(X_gt_batch.shape[0] - 1):
        start_idx = int(supervision_indices_cpu[interval_idx].item())
        end_idx = int(supervision_indices_cpu[interval_idx + 1].item())
        if end_idx <= start_idx:
            continue

        batch_shape = solution.shape[1:-1]
        D_g = torch.zeros(batch_shape + (9, 9), dtype=dtype, device=device)
        D_a = torch.zeros_like(D_g)
        P_ad = torch.zeros_like(D_g)
        R_start = R_sol[start_idx]
        v_start = solution[start_idx, ..., 7:10]
        p_start = solution[start_idx, ..., 10:13]

        for imu_idx in range(start_idx, end_idx):
            dt_k = (t_odeint[imu_idx + 1] - t_odeint[imu_idx]).to(dtype=dtype, device=device)
            tau = (t_odeint[imu_idx] - t_odeint[start_idx]).to(dtype=dtype, device=device)
            R_delta = R_start.transpose(-1, -2) @ R_sol[imu_idx]
            dv_delta = (
                R_start.transpose(-1, -2)
                @ (solution[imu_idx, ..., 7:10] - v_start - gravity * tau).unsqueeze(-1)
            ).squeeze(-1)
            dp_delta = (
                R_start.transpose(-1, -2)
                @ (
                    solution[imu_idx, ..., 10:13]
                    - p_start
                    - v_start * tau
                    - 0.5 * gravity * tau.square()
                ).unsqueeze(-1)
            ).squeeze(-1)
            if detach_covariance_theta:
                R_delta = R_delta.detach()
                dv_delta = dv_delta.detach()
                dp_delta = dp_delta.detach()

            phi, G_g, G_a = _interval_preintegration_linearization(R_delta, dv_delta, dp_delta, dt_k)
            Q_g_unit = (G_g @ G_g.transpose(-1, -2)) * dt_k
            Q_a_unit = (G_a @ G_a.transpose(-1, -2)) * dt_k
            D_g = phi @ D_g @ phi.transpose(-1, -2) + Q_g_unit
            D_a = phi @ D_a @ phi.transpose(-1, -2) + Q_a_unit
            if covariance_gradient_mode == "ad":
                P_ad = phi @ P_ad @ phi.transpose(-1, -2) + alpha_g * Q_g_unit + alpha_a * Q_a_unit

        if covariance_gradient_mode == "analytic_sensitivity":
            P_interval = alpha_g * D_g + alpha_a * D_a
        else:
            P_interval = P_ad
        S = P_interval + (float(target_cov_floor) + float(jitter)) * eye9
        S = 0.5 * (S + S.transpose(-1, -2))
        residual_chunks.append(residual_all[interval_idx])
        covariance_chunks.append(S)
        if debug_covariance_gradients:
            sensitivity_g_chunks.append(D_g)
            sensitivity_a_chunks.append(D_a)

    if not residual_chunks:
        raise ValueError("No valid interval preintegration factors were built for this batch.")

    residual_full = torch.stack(residual_chunks, dim=0)
    residual = residual_full
    S = torch.stack(covariance_chunks, dim=0)
    D_g_full = torch.stack(sensitivity_g_chunks, dim=0) if debug_covariance_gradients else None
    D_a_full = torch.stack(sensitivity_a_chunks, dim=0) if debug_covariance_gradients else None
    if residual_mask is not None:
        idx = torch.nonzero(residual_mask.to(device=device), as_tuple=False).squeeze(-1)
        residual = residual.index_select(-1, idx)
        S = S.index_select(-1, idx).index_select(-2, idx)
        if debug_covariance_gradients:
            D_g_full = D_g_full.index_select(-1, idx).index_select(-2, idx)
            D_a_full = D_a_full.index_select(-1, idx).index_select(-2, idx)

    flat_residual = residual.reshape(-1, residual.shape[-1])
    flat_S = S.reshape(-1, S.shape[-2], S.shape[-1])
    chol = torch.linalg.cholesky(flat_S)
    solved = torch.cholesky_solve(flat_residual.unsqueeze(-1), chol).squeeze(-1)
    mahal = (flat_residual * solved).sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    loss = 0.5 * (mahal + logdet).mean()

    info = {
        "num_intervals": len(residual_chunks),
        "num_factors": flat_residual.shape[0],
        "quad_mean": mahal.detach().mean(),
        "logdet_mean": logdet.detach().mean(),
        "min_chol_diag": torch.diagonal(chol, dim1=-2, dim2=-1).detach().amin(),
    }
    if debug_covariance_gradients:
        flat_D_g = D_g_full.reshape(-1, D_g_full.shape[-2], D_g_full.shape[-1])
        flat_D_a = D_a_full.reshape(-1, D_a_full.shape[-2], D_a_full.shape[-1])
        solve_D_g = torch.cholesky_solve(flat_D_g, chol)
        solve_D_a = torch.cholesky_solve(flat_D_a, chol)
        trace_g = torch.diagonal(solve_D_g, dim1=-2, dim2=-1).sum(dim=-1)
        trace_a = torch.diagonal(solve_D_a, dim1=-2, dim2=-1).sum(dim=-1)
        Dg_y = (flat_D_g @ solved.unsqueeze(-1)).squeeze(-1)
        Da_y = (flat_D_a @ solved.unsqueeze(-1)).squeeze(-1)
        quad_g = (solved * Dg_y).sum(dim=-1)
        quad_a = (solved * Da_y).sum(dim=-1)
        grad_alpha_g = 0.5 * (trace_g - quad_g).mean()
        grad_alpha_a = 0.5 * (trace_a - quad_a).mean()
        info.update(
            {
                "grad_alpha_g_closed": grad_alpha_g.detach(),
                "grad_alpha_a_closed": grad_alpha_a.detach(),
                "grad_raw_sigma_g2_closed": (
                    grad_alpha_g * torch.sigmoid(noise_model.raw_sigma_g2.to(dtype=dtype))
                ).detach(),
                "grad_raw_sigma_a2_closed": (
                    grad_alpha_a * torch.sigmoid(noise_model.raw_sigma_a2.to(dtype=dtype))
                ).detach(),
            }
        )
    if return_residuals:
        info["residual"] = residual_full.detach()
    return loss, info


def _select_trajectory_supervision_indices(
    num_states: int,
    mode: str,
    stride: int,
    max_states: int,
    device: torch.device | str,
) -> torch.Tensor:
    if num_states < 2:
        raise ValueError("Trajectory-state NLL requires at least two supervised states per window.")
    if mode == "endpoint":
        return torch.tensor([num_states - 1], dtype=torch.long, device=device)
    if mode != "sparse":
        raise ValueError(f"Unsupported trajectory supervision mode: {mode}")
    if stride <= 0:
        raise ValueError("--trajectory-supervision-stride must be positive.")

    indices = list(range(1, num_states, int(stride)))
    if indices[-1] != num_states - 1:
        indices.append(num_states - 1)
    if max_states > 0 and len(indices) > max_states:
        positions = np.linspace(0, len(indices) - 1, int(max_states))
        indices = [indices[int(round(pos))] for pos in positions]
        deduped = []
        for idx in indices:
            if not deduped or idx != deduped[-1]:
                deduped.append(idx)
        indices = deduped
        if indices[-1] != num_states - 1:
            indices[-1] = num_states - 1
    return torch.tensor(indices, dtype=torch.long, device=device)


def _trajectory_state_residual_nll(
    solution: torch.Tensor,
    R_sol: torch.Tensor,
    X_gt_batch: torch.Tensor,
    t_odeint: torch.Tensor,
    eval_indices: torch.Tensor,
    noise_model: IntervalNoiseVariances,
    gravity: torch.Tensor,
    residual_mask: torch.Tensor | None,
    target_cov_floor: float,
    jitter: float,
    detach_covariance_theta: bool,
    supervision_mode: str,
    supervision_stride: int,
    max_supervised_states: int,
    return_residuals: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Trajectory-state NLL with a joint covariance over supervised timestamps.

    Residual convention follows the existing SE23 loss:
        xi_i = Log(inv(X_gt_i) @ X_pred_i)

    The covariance is propagated from the window start and keeps cross-time
    covariance blocks between all selected supervised states. The initial state
    is omitted because the covariance is zero by construction.
    """

    dtype = solution.dtype
    device = solution.device
    if solution.shape[0] == X_gt_batch.shape[0]:
        solution_eval = solution
        R_eval = R_sol
        supervision_indices = torch.arange(X_gt_batch.shape[0], dtype=torch.long, device=device)
    else:
        supervision_indices = eval_indices.to(device=device, dtype=torch.long)
        solution_eval = solution[supervision_indices]
        R_eval = R_sol[supervision_indices]

    selected_gt_indices = _select_trajectory_supervision_indices(
        X_gt_batch.shape[0],
        supervision_mode,
        supervision_stride,
        max_supervised_states,
        device,
    )
    selected_ode_indices = supervision_indices.index_select(0, selected_gt_indices)
    selected_ode_indices_cpu = selected_ode_indices.detach().cpu().long().tolist()

    X_pred_all = Lie.SEn3fromSO3Rn(R_eval, solution_eval[..., 7:13])
    residual_full = Lie.SEn3log(Lie.SEn3inverse(X_gt_batch) @ X_pred_all).index_select(0, selected_gt_indices)

    alpha_g, alpha_a = noise_model.variances()
    alpha_g = alpha_g.to(dtype=dtype, device=device)
    alpha_a = alpha_a.to(dtype=dtype, device=device)
    noise_cov = torch.diag(torch.cat([alpha_a.repeat(3), alpha_g.repeat(3)])).to(dtype=dtype, device=device)

    batch_shape = solution.shape[1:-1]
    P = torch.zeros(batch_shape + (9, 9), dtype=dtype, device=device)
    cross_to_selected = torch.empty((0,) + batch_shape + (9, 9), dtype=dtype, device=device)
    lower_blocks: list[torch.Tensor] = []
    diag_blocks: list[torch.Tensor] = []
    gravity = gravity.to(dtype=dtype, device=device)

    next_selected = 0
    for imu_idx in range(solution.shape[0] - 1):
        dt_k = (t_odeint[imu_idx + 1] - t_odeint[imu_idx]).to(dtype=dtype, device=device)
        R_cov = R_sol[imu_idx]
        v_cov = solution[imu_idx, ..., 7:10]
        p_cov = solution[imu_idx, ..., 10:13]
        if detach_covariance_theta:
            R_cov = R_cov.detach()
            v_cov = v_cov.detach()
            p_cov = p_cov.detach()
        F, G = paper_invariant_error_matrices(R_cov, v_cov, p_cov, dt_k, gravity=gravity)
        Q = G @ noise_cov @ G.transpose(-1, -2)
        if cross_to_selected.shape[0] > 0:
            cross_to_selected = F.unsqueeze(0) @ cross_to_selected
        P = F @ P @ F.transpose(-1, -2) + Q
        P = 0.5 * (P + P.transpose(-1, -2))

        if next_selected < len(selected_ode_indices_cpu) and imu_idx + 1 == selected_ode_indices_cpu[next_selected]:
            lower_blocks.append(cross_to_selected.clone())
            diag_blocks.append(P.clone())
            cross_to_selected = torch.cat([cross_to_selected, P.unsqueeze(0)], dim=0)
            next_selected += 1

    if next_selected != len(selected_ode_indices_cpu):
        raise ValueError(
            "Failed to collect all trajectory-state covariance blocks: "
            f"collected={next_selected}, expected={len(selected_ode_indices_cpu)}"
        )

    num_supervised = len(diag_blocks)
    full_dim = 9 * num_supervised
    Sigma = torch.zeros(batch_shape + (full_dim, full_dim), dtype=dtype, device=device)
    for row in range(num_supervised):
        row_slice = slice(9 * row, 9 * (row + 1))
        Sigma[..., row_slice, row_slice] = diag_blocks[row]
        for col in range(row):
            col_slice = slice(9 * col, 9 * (col + 1))
            block = lower_blocks[row][col]
            Sigma[..., row_slice, col_slice] = block
            Sigma[..., col_slice, row_slice] = block.transpose(-1, -2)

    residual = residual_full
    if residual_mask is not None:
        idx = torch.nonzero(residual_mask.to(device=device), as_tuple=False).squeeze(-1)
        residual = residual.index_select(-1, idx)
        block_indices = torch.cat([idx + 9 * block for block in range(num_supervised)], dim=0)
        Sigma = Sigma.index_select(-1, block_indices).index_select(-2, block_indices)

    residual_dim = residual.shape[-1]
    residual_batch = residual.permute(*range(1, residual.ndim - 1), 0, residual.ndim - 1).reshape(
        batch_shape + (num_supervised * residual_dim,)
    )
    eye = torch.eye(Sigma.shape[-1], dtype=dtype, device=device)
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2)) + (float(target_cov_floor) + float(jitter)) * eye

    flat_residual = residual_batch.reshape(-1, residual_batch.shape[-1])
    flat_Sigma = Sigma.reshape(-1, Sigma.shape[-2], Sigma.shape[-1])
    chol = torch.linalg.cholesky(flat_Sigma)
    solved = torch.cholesky_solve(flat_residual.unsqueeze(-1), chol).squeeze(-1)
    mahal = (flat_residual * solved).sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    loss = 0.5 * (mahal + logdet).mean()

    info = {
        "num_supervised_states": num_supervised,
        "num_factors": flat_residual.shape[0],
        "quad_mean": mahal.detach().mean(),
        "logdet_mean": logdet.detach().mean(),
        "min_chol_diag": torch.diagonal(chol, dim1=-2, dim2=-1).detach().amin(),
    }
    if return_residuals:
        info["residual"] = residual_full.detach()
    return loss, info


def _format_interval_residual_metrics(residual: torch.Tensor, prefix: str = "interval") -> str:
    flat = residual.reshape(-1, residual.shape[-1])
    rot_rmse = torch.sqrt(flat[:, :3].square().sum(dim=-1).mean()).item() * 180.0 / np.pi
    vel_rmse = torch.sqrt(flat[:, 3:6].square().sum(dim=-1).mean()).item() if flat.shape[-1] >= 6 else float("nan")
    pos_rmse = torch.sqrt(flat[:, 6:9].square().sum(dim=-1).mean()).item() if flat.shape[-1] >= 9 else float("nan")
    return (
        f"val_{prefix}_rot_rmse_deg={rot_rmse:.6g} "
        f"val_{prefix}_vel_rmse_mps={vel_rmse:.6g} "
        f"val_{prefix}_pos_rmse_m={pos_rmse:.6g}"
    )


def _print_gyro_val_metrics(epoch: int, loss_total: float, R_chunks: list[torch.Tensor], R_gt_chunks: list[torch.Tensor]) -> None:
    R_pred = torch.cat(R_chunks, dim=1)
    R_gt = torch.cat(R_gt_chunks, dim=1)
    unscaled_sum = loss_total / 1e6
    print(
        f"epoch: {epoch}, val_loss_unscaled_sum: {unscaled_sum:.8g}, "
        f"val_rot_rmse_deg: {_rotation_rmse_deg(R_pred, R_gt):.6g}",
        flush=True,
    )


def _format_se23_rmse_metrics(
    R_pred: torch.Tensor,
    solution: torch.Tensor,
    X_gt: torch.Tensor,
) -> str:
    return (
        f"val_rot_rmse_deg={_rotation_rmse_deg(R_pred, X_gt[..., :3, :3]):.6g} "
        f"val_vel_rmse_mps={_vector_rmse(solution[..., 7:10], X_gt[..., :3, 3]):.6g} "
        f"val_pos_rmse_m={_vector_rmse(solution[..., 10:13], X_gt[..., :3, 4]):.6g}"
    )


def _print_se23_val_metrics(
    epoch: int,
    loss_total: float,
    R_chunks: list[torch.Tensor],
    solution_chunks: list[torch.Tensor],
    X_gt_chunks: list[torch.Tensor],
) -> None:
    R_pred = torch.cat(R_chunks, dim=1)
    solution = torch.cat(solution_chunks, dim=1)
    X_gt = torch.cat(X_gt_chunks, dim=1)
    X_pred = Lie.SEn3fromSO3Rn(R_pred, solution[..., 7:13])
    xi = Lie.SEn3log(Lie.SEn3inverse(X_gt) @ X_pred)
    unscaled_sum = loss_total / 1e6
    unscaled_avg = float(xi.square().mean().item())
    print(
        f"epoch: {epoch}, val_loss_unscaled_sum: {unscaled_sum:.8g}, "
        f"val_loss_unscaled_avg: {unscaled_avg:.8g}, "
        f"{_format_se23_rmse_metrics(R_pred, solution, X_gt)}",
        flush=True,
    )


def Gyro_train_with_readable_metrics(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    outpur_dir: str,
    bw_func_name: str,
    integral_method="euler",
    device="cuda",
    lr=0.005,
    weight_decay=1e-6,
    epoch=1801,
    lr_step_size: int = 300,
    lr_gamma: float = 0.5,
    train_batch_strategy: str = "random_sequence",
    val_freq: int = 100,
    save_freq: int = 100,
    bw_init_model_path: str | Path | None = None,
    gyro_bias_rate_regularization_weight: float = 0.0,
    train_initial_bw_mode: str = "label",
    train_initial_ba_mode: str = "label",
):
    num_iterations = epoch
    torch.manual_seed(3407)
    mask_flag = _mask_flag(dataset_train)
    debias_learning.validate_train_batch_strategy(train_batch_strategy)

    model_save_path = os.path.join(outpur_dir, "Gyro_weights")
    os.makedirs(model_save_path, exist_ok=True)
    log_writer_path = os.path.join(outpur_dir, "Gyro_logs")

    bw_func = bw_func_choose(bw_func_name, device)
    _load_bw_model_initialization(bw_func, bw_init_model_path, device)
    model = debias_learning.VFSE23vbiasBase(biasfunc_w=bw_func, device=device)
    best_loss = float("inf")
    writter = debias_learning.SummaryWriter(log_writer_path)
    model = model.to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(True)
    model.freeze_func_ba()
    use_delta_imu = _uses_delta_imu(bw_func_name)

    debias_learning.write_parameters("Gyro", dataset_train, outpur_dir, bw_func_name, lr, weight_decay, epoch, train_batch_strategy)

    criterion = debias_learning.SO3log_loss
    epoch_times = []
    gpu_memory = []
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    loss_total = float("nan")
    torch.save(
        {
            "func_bw_model": model.biasfunc_w.state_dict(),
            "VF_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": -1,
            "valloss": loss_total,
            "checkpoint_type": "initial",
        },
        model_save_path + "/model_initial.pt",
    )
    for epoch_idx in range(num_iterations):
        epoch_start = debias_learning.start_epoch_profile(device)
        optimizer.zero_grad()
        model.train()

        train_losses = []
        batch_sizes = []
        for index, batch_size in debias_learning.get_train_batches(dataset_train, train_batch_strategy):
            (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_train.get_data_for_biasDy(
                index, device, mask_flag=mask_flag, batch_size=batch_size
            )
            y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
            y0_batch = _apply_train_initial_bias_modes(
                y0_batch,
                train_initial_bw_mode,
                train_initial_ba_mode,
            )
            spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
            _set_model_imu_functions(model, spline, dataset_train.dt, use_delta_imu)
            model.set_R0(R0_batch)
            solution, R_sol = debias_learning.odeint_SO3(
                model, y0_batch, R0_batch, t_odeint, rtol=1e-7, atol=1e-9, method=integral_method
            )
            if R_sol.shape[0] != X_gt_batch.shape[0]:
                R_sol = R_sol[eval_indices]
            loss = 1e6 * criterion(R_sol, X_gt_batch[..., :3, :3])
            if gyro_bias_rate_regularization_weight > 0:
                loss = loss + float(gyro_bias_rate_regularization_weight) * _bias_rate_regularization(
                    solution,
                    t_odeint,
                    slice(4, 7),
                )
            train_losses.append(loss)
            batch_sizes.append(batch_size)

        train_loss = debias_learning.combine_train_losses(train_losses, batch_sizes)
        train_loss.backward()
        optimizer.step()

        writter.add_scalar("train_loss", train_loss.item(), epoch_idx)
        writter.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_idx)
        scheduler.step()

        if epoch_idx % val_freq == 0:
            with torch.no_grad():
                model.eval()
                loss_total = 0.0
                for index in range(dataset_val.length()):
                    (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_val.get_data_for_biasDy(
                        index, device, mask_flag=mask_flag
                    )
                    y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
                    y0_batch = _apply_train_initial_bias_modes(
                        y0_batch,
                        train_initial_bw_mode,
                        train_initial_ba_mode,
                    )
                    spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
                    _set_model_imu_functions(model, spline, dataset_val.dt, use_delta_imu)
                    model.set_R0(R0_batch)
                    _, R_sol = debias_learning.odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method)
                    if R_sol.shape[0] != X_gt_batch.shape[0]:
                        R_sol = R_sol[eval_indices]
                    loss = 1e6 * torch.nn.MSELoss()(R_sol, X_gt_batch[..., :3, :3])
                    loss_total += loss.item()
                writter.add_scalar("val_loss", loss_total, epoch_idx)
                print(
                    f"stage=Gyro epoch={epoch_idx} train_loss={train_loss.item():.6g} val_loss={loss_total:.6g}",
                    flush=True,
                )

            if loss_total < best_loss:
                best_loss = loss_total
                torch.save(
                    {
                        "func_bw_model": model.biasfunc_w.state_dict(),
                        "VF_model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch_idx,
                        "best_loss": best_loss,
                    },
                    model_save_path + "/best_val_model.pt",
                )

        if epoch_idx % save_freq == 0:
            torch.save(
                {
                    "func_bw_model": model.biasfunc_w.state_dict(),
                    "VF_model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch_idx,
                    "valloss": loss_total,
                },
                model_save_path + "/model_epoch_" + str(epoch_idx) + ".pt",
            )
        epoch_time, peak_memory_mb = debias_learning.finish_epoch_profile(device, epoch_start)
        epoch_times.append(epoch_time)
        gpu_memory.append(peak_memory_mb)
    writter.close()
    return debias_learning.write_stage_profile(outpur_dir, "Gyro", epoch_times, gpu_memory)


def Acc_train_SE23_with_readable_metrics(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    outpur_dir: str,
    bw_model,
    ba_model_name: str,
    integral_method="euler",
    device="cuda",
    lr=0.005,
    weight_decay=1e-6,
    epoch=1801,
    lr_step_size: int = 300,
    lr_gamma: float = 0.5,
    train_batch_strategy: str = "random_sequence",
    val_freq: int = 100,
    save_freq: int = 100,
    bw_model_name: str | None = None,
    acc_loss_rot_weight: float = 1.0,
    acc_loss_vel_weight: float = 1.0,
    acc_loss_pos_weight: float = 1.0,
    ba_init_model_path: str | Path | None = None,
    acc_bias_rate_regularization_weight: float = 0.0,
    train_initial_bw_mode: str = "label",
    train_initial_ba_mode: str = "label",
):
    num_iterations = epoch
    torch.manual_seed(3407)
    mask_flag = _mask_flag(dataset_train)
    debias_learning.validate_train_batch_strategy(train_batch_strategy)

    model_save_path = os.path.join(outpur_dir, "Acc_weights")
    os.makedirs(model_save_path, exist_ok=True)
    log_writer_path = os.path.join(outpur_dir, "Acc_logs")

    ba_model = ba_func_choose(ba_model_name, device)
    _load_ba_model_initialization(ba_model, ba_init_model_path, device)
    model = debias_learning.VFSE23vbiasBase(biasfunc_w=bw_model, biasfunc_a=ba_model, device=device)
    best_loss = float("inf")
    writter = debias_learning.SummaryWriter(log_writer_path)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.freeze_func_bw()
    use_delta_imu = _uses_delta_imu(bw_model_name, ba_model_name)

    debias_learning.write_parameters("Acc", dataset_train, outpur_dir, ba_model_name, lr, weight_decay, epoch, train_batch_strategy)

    epoch_times = []
    gpu_memory = []
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    loss_total = float("nan")
    torch.save(
        {
            "func_ba_model": model.biasfunc_a.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": -1,
            "valloss": loss_total,
            "checkpoint_type": "initial",
        },
        model_save_path + "/model_initial.pt",
    )
    for epoch_idx in range(num_iterations):
        epoch_start = debias_learning.start_epoch_profile(device)
        optimizer.zero_grad()
        model.train()

        train_losses = []
        batch_sizes = []
        for index, batch_size in debias_learning.get_train_batches(dataset_train, train_batch_strategy):
            (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_train.get_data_for_biasDy(
                index, device, mask_flag=mask_flag, batch_size=batch_size
            )
            y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
            y0_batch = _apply_train_initial_bias_modes(
                y0_batch,
                train_initial_bw_mode,
                train_initial_ba_mode,
            )
            spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
            _set_model_imu_functions(model, spline, dataset_train.dt, use_delta_imu)
            model.set_R0(R0_batch)
            solution, R_sol = debias_learning.odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method)
            solution_full = solution
            solution, R_sol = debias_learning.select_supervision_states(solution, R_sol, X_gt_batch, eval_indices)

            X_pred = Lie.SEn3fromSO3Rn(R_sol, solution[..., 7:13])
            loss = (
                1e6
                * _weighted_se23_loss(
                    X_pred,
                    X_gt_batch,
                    acc_loss_rot_weight,
                    acc_loss_vel_weight,
                    acc_loss_pos_weight,
                )
            )
            if acc_bias_rate_regularization_weight > 0:
                loss = loss + float(acc_bias_rate_regularization_weight) * _bias_rate_regularization(
                    solution_full,
                    t_odeint,
                    slice(13, 16),
                )
            train_losses.append(loss)
            batch_sizes.append(batch_size)

        train_loss = debias_learning.combine_train_losses(train_losses, batch_sizes)
        train_loss.backward()
        optimizer.step()

        writter.add_scalar("train_loss", train_loss.item(), epoch_idx)
        writter.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_idx)
        scheduler.step()

        if epoch_idx % val_freq == 0:
            with torch.no_grad():
                model.eval()
                loss_total = 0.0
                for index in range(dataset_val.length()):
                    (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_val.get_data_for_biasDy(
                        index, device, mask_flag=mask_flag
                    )
                    y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
                    y0_batch = _apply_train_initial_bias_modes(
                        y0_batch,
                        train_initial_bw_mode,
                        train_initial_ba_mode,
                    )
                    spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
                    _set_model_imu_functions(model, spline, dataset_val.dt, use_delta_imu)
                    model.set_R0(R0_batch)
                    solution, R_sol = debias_learning.odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method)
                    solution, R_sol = debias_learning.select_supervision_states(solution, R_sol, X_gt_batch, eval_indices)
                    X_pred = Lie.SEn3fromSO3Rn(R_sol, solution[..., 7:13])
                    loss = 1e6 * _weighted_se23_loss(
                        X_pred,
                        X_gt_batch,
                        acc_loss_rot_weight,
                        acc_loss_vel_weight,
                        acc_loss_pos_weight,
                    )
                    loss_total += loss.item()
                writter.add_scalar("val_loss", loss_total, epoch_idx)
                print(
                    f"stage=Acc epoch={epoch_idx} train_loss={train_loss.item():.6g} val_loss={loss_total:.6g}",
                    flush=True,
                )

            if loss_total < best_loss:
                best_loss = loss_total
                torch.save(
                    {
                        "func_ba_model": model.biasfunc_a.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch_idx,
                        "best_loss": best_loss,
                    },
                    model_save_path + "/best_val_model.pt",
                )

        if epoch_idx % save_freq == 0:
            torch.save(
                {
                    "func_ba_model": model.biasfunc_a.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch_idx,
                    "valloss": loss_total,
                },
                model_save_path + "/model_epoch_" + str(epoch_idx) + ".pt",
            )
        epoch_time, peak_memory_mb = debias_learning.finish_epoch_profile(device, epoch_start)
        epoch_times.append(epoch_time)
        gpu_memory.append(peak_memory_mb)
    writter.close()
    return debias_learning.write_stage_profile(outpur_dir, "Acc", epoch_times, gpu_memory)


def Acc_train_SE23_NLL_with_readable_metrics(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    outpur_dir: str,
    bw_model,
    ba_model_name: str,
    integral_method="euler",
    device="cuda",
    lr=0.005,
    weight_decay=1e-6,
    epoch=1801,
    lr_step_size: int = 300,
    lr_gamma: float = 0.5,
    train_batch_strategy: str = "random_sequence",
    val_freq: int = 100,
    save_freq: int = 100,
    bw_model_name: str | None = None,
    nll_sigma_a: float = 2.0e-3,
    nll_sigma_g: float = 1.6968e-4,
    nll_w_rot: float = 1.0e-3,
    nll_w_vel: float = 2.0e-2,
    nll_w_pos: float = 1.0e-2,
    nll_optimize_q: bool = False,
    nll_q_block_epochs: int = 100,
    nll_lr_q: float = 1.0e-2,
    nll_sigma_a_min: float = 1.0e-4,
    nll_sigma_a_max: float = 5.0e-2,
    nll_sigma_g_min: float = 1.0e-5,
    nll_sigma_g_max: float = 5.0e-3,
    nll_supervision_mode: str = "per_state",
    ba_init_model_path: str | Path | None = None,
    nll_q_only: bool = False,
    nll_train_gyro: bool = False,
    nll_mean_anchor_weight: float = 0.0,
    nll_mean_anchor_mode: str = "endpoint",
    train_initial_bw_mode: str = "label",
    train_initial_ba_mode: str = "label",
):
    """Acc training with per-state SE_2(3) Gaussian NLL and covariance propagation."""
    num_iterations = epoch
    torch.manual_seed(3407)
    mask_flag = _mask_flag(dataset_train)
    debias_learning.validate_train_batch_strategy(train_batch_strategy)

    model_save_path = os.path.join(outpur_dir, "Acc_weights")
    os.makedirs(model_save_path, exist_ok=True)
    gyro_model_save_path = os.path.join(outpur_dir, "Gyro_weights")
    if nll_train_gyro:
        os.makedirs(gyro_model_save_path, exist_ok=True)
    log_writer_path = os.path.join(outpur_dir, "Acc_logs")

    ba_model = ba_func_choose(ba_model_name, device)
    _load_ba_model_initialization(ba_model, ba_init_model_path, device)
    model = debias_learning.VFSE23vbiasBase(biasfunc_w=bw_model, biasfunc_a=ba_model, device=device)
    best_loss = float("inf")
    writter = debias_learning.SummaryWriter(log_writer_path)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    if nll_train_gyro:
        model.unfreeze_func_bw()
    else:
        model.freeze_func_bw()
    model.unfreeze_func_ba()
    anchor_model = None
    if nll_mean_anchor_weight > 0:
        anchor_model = debias_learning.VFSE23vbiasBase(
            biasfunc_w=copy.deepcopy(bw_model),
            biasfunc_a=ba_func_choose(ba_model_name, device),
            device=device,
        )
        _load_ba_model_initialization(anchor_model.biasfunc_a, ba_init_model_path, device)
        anchor_model.set_bw_zero(False)
        anchor_model.set_ba_zero(False)
        anchor_model.freeze_func_bw()
        anchor_model.freeze_func_ba()
        anchor_model.eval()
    use_delta_imu = _uses_delta_imu(bw_model_name, ba_model_name)

    debias_learning.write_parameters("Acc", dataset_train, outpur_dir, ba_model_name, lr, weight_decay, epoch, train_batch_strategy)

    fixed_noise_cov, measurement_cov = debias_learning.make_fixed_nll_covariances(
        device=device,
        sigma_a=nll_sigma_a,
        sigma_g=nll_sigma_g,
        w_rot=nll_w_rot,
        w_vel=nll_w_vel,
        w_pos=nll_w_pos,
    )
    log_sigma_a = torch.nn.Parameter(torch.log(torch.tensor(nll_sigma_a, device=device)))
    log_sigma_g = torch.nn.Parameter(torch.log(torch.tensor(nll_sigma_g, device=device)))

    epoch_times = []
    gpu_memory = []
    net_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer_net = torch.optim.Adam(net_params, lr=lr, weight_decay=weight_decay)
    scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=lr_step_size, gamma=lr_gamma)
    optimizer_q = torch.optim.Adam([log_sigma_a, log_sigma_g], lr=nll_lr_q)
    loss_total = float("nan")

    for epoch_idx in range(num_iterations):
        optimize_q = nll_optimize_q and (
            nll_q_only or ((epoch_idx // nll_q_block_epochs) % 2 == 1)
        )
        for param in net_params:
            param.requires_grad = not optimize_q
        log_sigma_a.requires_grad_(optimize_q)
        log_sigma_g.requires_grad_(optimize_q)

        epoch_start = debias_learning.start_epoch_profile(device)
        optimizer_net.zero_grad()
        optimizer_q.zero_grad()
        model.train()

        if nll_optimize_q:
            noise_cov, sigma_a_cur, sigma_g_cur = debias_learning.make_bounded_nll_noise_covariance(
                log_sigma_a,
                log_sigma_g,
                nll_sigma_a_min,
                nll_sigma_a_max,
                nll_sigma_g_min,
                nll_sigma_g_max,
            )
        else:
            noise_cov = fixed_noise_cov
            sigma_a_cur = torch.tensor(nll_sigma_a, device=device)
            sigma_g_cur = torch.tensor(nll_sigma_g, device=device)
        joint_model = debias_learning.make_joint_nominal_covariance_ode(model, noise_cov)

        train_losses = []
        batch_sizes = []
        for index, batch_size in debias_learning.get_train_batches(dataset_train, train_batch_strategy):
            (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_train.get_data_for_biasDy(
                index, device, mask_flag=mask_flag, batch_size=batch_size
            )
            y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
            y0_batch = _apply_train_initial_bias_modes(
                y0_batch,
                train_initial_bw_mode,
                train_initial_ba_mode,
            )
            y0_aug = debias_learning.augment_state_with_covariance(y0_batch)
            spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
            _set_model_imu_functions(joint_model, spline, dataset_train.dt, use_delta_imu)
            joint_model.set_R0(R0_batch)
            solution_aug, R_sol = debias_learning.odeint_SO3(joint_model, y0_aug, R0_batch, t_odeint, method=integral_method)
            solution_aug, R_sol = debias_learning.select_supervision_states(solution_aug, R_sol, X_gt_batch, eval_indices)
            solution_aug_for_anchor = solution_aug
            R_sol_for_anchor = R_sol
            solution_aug, R_sol, X_gt_batch_nll, skip_initial = _select_nll_supervision_states(
                solution_aug,
                R_sol,
                X_gt_batch,
                nll_supervision_mode,
            )
            solution, P = debias_learning.split_state_with_covariance(solution_aug)

            X_pred = Lie.SEn3fromSO3Rn(R_sol, solution[..., 7:13])
            loss = debias_learning.SE23_nll_loss(
                X_pred,
                X_gt_batch_nll,
                P,
                measurement_cov=measurement_cov,
                skip_initial=skip_initial,
            )
            if anchor_model is not None:
                solution_aug_anchor, R_anchor, _, _ = _select_nll_supervision_states(
                    solution_aug_for_anchor,
                    R_sol_for_anchor,
                    X_gt_batch,
                    nll_mean_anchor_mode,
                )
                solution_anchor_cur, _ = debias_learning.split_state_with_covariance(solution_aug_anchor)
                X_anchor_cur = Lie.SEn3fromSO3Rn(R_anchor, solution_anchor_cur[..., 7:13])
                with torch.no_grad():
                    _set_model_imu_functions(anchor_model, spline, dataset_train.dt, use_delta_imu)
                    anchor_model.set_R0(R0_batch)
                    solution_ref, R_ref = debias_learning.odeint_SO3(
                        anchor_model,
                        y0_batch,
                        R0_batch,
                        t_odeint,
                        method=integral_method,
                    )
                    solution_ref, R_ref = debias_learning.select_supervision_states(
                        solution_ref,
                        R_ref,
                        X_gt_batch,
                        eval_indices,
                    )
                    solution_ref, R_ref, _, _ = _select_nll_supervision_states(
                        solution_ref,
                        R_ref,
                        X_gt_batch,
                        nll_mean_anchor_mode,
                    )
                    X_anchor_ref = Lie.SEn3fromSO3Rn(R_ref, solution_ref[..., 7:13])
                anchor_loss = 1e6 * _weighted_se23_loss(X_anchor_cur, X_anchor_ref, 1.0, 1.0, 1.0)
                loss = loss + nll_mean_anchor_weight * anchor_loss
            train_losses.append(loss)
            batch_sizes.append(batch_size)

        train_loss = debias_learning.combine_train_losses(train_losses, batch_sizes)
        train_loss.backward()
        if optimize_q:
            optimizer_q.step()
        else:
            optimizer_net.step()

        writter.add_scalar("train_loss", train_loss.item(), epoch_idx)
        writter.add_scalar("lr", optimizer_q.param_groups[0]["lr"] if optimize_q else optimizer_net.param_groups[0]["lr"], epoch_idx)
        writter.add_scalar("nll_sigma_a", sigma_a_cur.item(), epoch_idx)
        writter.add_scalar("nll_sigma_g", sigma_g_cur.item(), epoch_idx)
        writter.add_scalar("nll_optimize_q", float(optimize_q), epoch_idx)
        if not optimize_q:
            scheduler_net.step()

        if epoch_idx % val_freq == 0:
            with torch.no_grad():
                model.eval()
                loss_total = 0.0
                nll_loss_total = 0.0
                anchor_loss_total = 0.0
                val_r_chunks = []
                val_solution_chunks = []
                val_gt_chunks = []
                if nll_optimize_q:
                    noise_cov_eval, sigma_a_eval, sigma_g_eval = debias_learning.make_bounded_nll_noise_covariance(
                        log_sigma_a,
                        log_sigma_g,
                        nll_sigma_a_min,
                        nll_sigma_a_max,
                        nll_sigma_g_min,
                        nll_sigma_g_max,
                    )
                else:
                    noise_cov_eval = fixed_noise_cov
                    sigma_a_eval = sigma_a_cur
                    sigma_g_eval = sigma_g_cur
                joint_model = debias_learning.make_joint_nominal_covariance_ode(model, noise_cov_eval)
                for index in range(dataset_val.length()):
                    (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_val.get_data_for_biasDy(
                        index, device, mask_flag=mask_flag
                    )
                    y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
                    y0_batch = _apply_train_initial_bias_modes(
                        y0_batch,
                        train_initial_bw_mode,
                        train_initial_ba_mode,
                    )
                    y0_aug = debias_learning.augment_state_with_covariance(y0_batch)
                    spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
                    _set_model_imu_functions(joint_model, spline, dataset_val.dt, use_delta_imu)
                    joint_model.set_R0(R0_batch)
                    solution_aug, R_sol = debias_learning.odeint_SO3(joint_model, y0_aug, R0_batch, t_odeint, method=integral_method)
                    solution_aug, R_sol = debias_learning.select_supervision_states(solution_aug, R_sol, X_gt_batch, eval_indices)
                    solution_aug_for_anchor = solution_aug
                    R_sol_for_anchor = R_sol
                    solution_aug, R_sol, X_gt_batch_nll, skip_initial = _select_nll_supervision_states(
                        solution_aug,
                        R_sol,
                        X_gt_batch,
                        nll_supervision_mode,
                    )
                    solution, P = debias_learning.split_state_with_covariance(solution_aug)
                    X_pred = Lie.SEn3fromSO3Rn(R_sol, solution[..., 7:13])
                    loss = debias_learning.SE23_nll_loss(
                        X_pred,
                        X_gt_batch_nll,
                        P,
                        measurement_cov=measurement_cov,
                        skip_initial=skip_initial,
                    )
                    nll_loss_total += loss.item()
                    if anchor_model is not None:
                        solution_aug_anchor, R_anchor, _, _ = _select_nll_supervision_states(
                            solution_aug_for_anchor,
                            R_sol_for_anchor,
                            X_gt_batch,
                            nll_mean_anchor_mode,
                        )
                        solution_anchor_cur, _ = debias_learning.split_state_with_covariance(solution_aug_anchor)
                        X_anchor_cur = Lie.SEn3fromSO3Rn(R_anchor, solution_anchor_cur[..., 7:13])
                        _set_model_imu_functions(anchor_model, spline, dataset_val.dt, use_delta_imu)
                        anchor_model.set_R0(R0_batch)
                        solution_ref, R_ref = debias_learning.odeint_SO3(
                            anchor_model,
                            y0_batch,
                            R0_batch,
                            t_odeint,
                            method=integral_method,
                        )
                        solution_ref, R_ref = debias_learning.select_supervision_states(
                            solution_ref,
                            R_ref,
                            X_gt_batch,
                            eval_indices,
                        )
                        solution_ref, R_ref, _, _ = _select_nll_supervision_states(
                            solution_ref,
                            R_ref,
                            X_gt_batch,
                            nll_mean_anchor_mode,
                        )
                        X_anchor_ref = Lie.SEn3fromSO3Rn(R_ref, solution_ref[..., 7:13])
                        anchor_loss = 1e6 * _weighted_se23_loss(X_anchor_cur, X_anchor_ref, 1.0, 1.0, 1.0)
                        anchor_loss_total += anchor_loss.item()
                        loss = loss + nll_mean_anchor_weight * anchor_loss
                    loss_total += loss.item()
                    val_r_chunks.append(R_sol.detach())
                    val_solution_chunks.append(solution.detach())
                    val_gt_chunks.append(X_gt_batch_nll.detach())
                writter.add_scalar("val_loss", loss_total, epoch_idx)
                R_val = torch.cat(val_r_chunks, dim=1)
                solution_val = torch.cat(val_solution_chunks, dim=1)
                X_gt_val = torch.cat(val_gt_chunks, dim=1)
                print(
                    f"stage=AccNLL epoch={epoch_idx} train_loss={train_loss.item():.6g} "
                    f"val_loss={loss_total:.6g} mode={nll_supervision_mode} optimize_q={optimize_q} "
                    f"sigma_a={sigma_a_eval.item():.6g} sigma_g={sigma_g_eval.item():.6g} "
                    f"val_nll={nll_loss_total:.6g} val_anchor={anchor_loss_total:.6g} "
                    f"{_format_se23_rmse_metrics(R_val, solution_val, X_gt_val)}",
                    flush=True,
                )

            if loss_total < best_loss:
                best_loss = loss_total
                best_checkpoint = {
                    "func_ba_model": model.biasfunc_a.state_dict(),
                    "optimizer": optimizer_net.state_dict(),
                    "scheduler": scheduler_net.state_dict(),
                    "optimizer_q": optimizer_q.state_dict(),
                    "log_sigma_a": log_sigma_a.detach(),
                    "log_sigma_g": log_sigma_g.detach(),
                    "epoch": epoch_idx,
                    "best_loss": best_loss,
                        "nll_train_gyro": nll_train_gyro,
                        "nll_mean_anchor_weight": nll_mean_anchor_weight,
                        "nll_mean_anchor_mode": nll_mean_anchor_mode,
                    }
                if nll_train_gyro:
                    best_checkpoint["func_bw_model"] = model.biasfunc_w.state_dict()
                torch.save(best_checkpoint, model_save_path + "/best_val_model.pt")
                if nll_train_gyro:
                    torch.save(
                        {
                            "func_bw_model": model.biasfunc_w.state_dict(),
                            "optimizer": optimizer_net.state_dict(),
                            "scheduler": scheduler_net.state_dict(),
                            "optimizer_q": optimizer_q.state_dict(),
                            "log_sigma_a": log_sigma_a.detach(),
                            "log_sigma_g": log_sigma_g.detach(),
                            "epoch": epoch_idx,
                            "best_loss": best_loss,
                            "nll_joint_acc_checkpoint": model_save_path + "/best_val_model.pt",
                        },
                        gyro_model_save_path + "/best_val_model.pt",
                    )

        if epoch_idx % save_freq == 0:
            epoch_checkpoint = {
                "func_ba_model": model.biasfunc_a.state_dict(),
                "optimizer": optimizer_net.state_dict(),
                "scheduler": scheduler_net.state_dict(),
                "optimizer_q": optimizer_q.state_dict(),
                "log_sigma_a": log_sigma_a.detach(),
                "log_sigma_g": log_sigma_g.detach(),
                "epoch": epoch_idx,
                "valloss": loss_total,
                "nll_train_gyro": nll_train_gyro,
                "nll_mean_anchor_weight": nll_mean_anchor_weight,
                "nll_mean_anchor_mode": nll_mean_anchor_mode,
            }
            if nll_train_gyro:
                epoch_checkpoint["func_bw_model"] = model.biasfunc_w.state_dict()
            torch.save(epoch_checkpoint, model_save_path + "/model_epoch_" + str(epoch_idx) + ".pt")
            if nll_train_gyro:
                torch.save(
                    {
                        "func_bw_model": model.biasfunc_w.state_dict(),
                        "optimizer": optimizer_net.state_dict(),
                        "scheduler": scheduler_net.state_dict(),
                        "optimizer_q": optimizer_q.state_dict(),
                        "log_sigma_a": log_sigma_a.detach(),
                        "log_sigma_g": log_sigma_g.detach(),
                        "epoch": epoch_idx,
                        "valloss": loss_total,
                        "nll_joint_acc_checkpoint": model_save_path + "/model_epoch_" + str(epoch_idx) + ".pt",
                    },
                    gyro_model_save_path + "/model_epoch_" + str(epoch_idx) + ".pt",
                )
        epoch_time, peak_memory_mb = debias_learning.finish_epoch_profile(device, epoch_start)
        epoch_times.append(epoch_time)
        gpu_memory.append(peak_memory_mb)

    writter.close()
    return debias_learning.write_stage_profile(outpur_dir, "Acc", epoch_times, gpu_memory)


def Acc_train_interval_preintegration_NLL_with_readable_metrics(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    outpur_dir: str,
    bw_model,
    ba_model_name: str,
    integral_method="euler",
    device="cuda",
    lr=0.005,
    weight_decay=1e-6,
    epoch=1801,
    lr_step_size: int = 300,
    lr_gamma: float = 0.5,
    train_batch_strategy: str = "random_sequence",
    val_freq: int = 100,
    save_freq: int = 100,
    bw_model_name: str | None = None,
    acc_loss_rot_weight: float = 1.0,
    acc_loss_vel_weight: float = 1.0,
    acc_loss_pos_weight: float = 1.0,
    ba_init_model_path: str | Path | None = None,
    interval_nll_mode: str = "nll_cov_only",
    interval_sigma_g_nominal: float = 1.6968e-4,
    interval_sigma_a_nominal: float = 2.0e-3,
    interval_sigma_g_floor2: float = 1.0e-12,
    interval_sigma_a_floor2: float = 1.0e-10,
    interval_target_cov_floor: float = 1.0e-9,
    interval_jitter: float = 1.0e-8,
    interval_prior_lambda_g: float = 0.0,
    interval_prior_lambda_a: float = 0.0,
    interval_lr_sigma: float = 1.0e-2,
    interval_detach_covariance_theta: bool = True,
    interval_covariance_gradient_mode: str = "analytic_sensitivity",
    interval_debug_covariance_gradients: bool = False,
    interval_noise_init_model_path: str | Path | None = None,
    nll_train_gyro: bool = False,
    nll_family: str = "interval",
    trajectory_supervision_mode: str = "sparse",
    trajectory_supervision_stride: int = 1,
    trajectory_max_supervised_states: int = 0,
    train_initial_bw_mode: str = "label",
    train_initial_ba_mode: str = "label",
):
    """Acc training with preintegration NLL variants.

    nll_family="interval" keeps the previous independent interval-factor loss.
    nll_family="trajectory_state" builds one stacked trajectory residual per
    window and a joint covariance over all selected supervised timestamps.
    """

    if interval_nll_mode == "mse_only":
        return Acc_train_SE23_with_readable_metrics(
            dataset_train,
            dataset_val,
            outpur_dir,
            bw_model,
            ba_model_name,
            integral_method=integral_method,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            epoch=epoch,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma,
            train_batch_strategy=train_batch_strategy,
            val_freq=val_freq,
            save_freq=save_freq,
            bw_model_name=bw_model_name,
            acc_loss_rot_weight=acc_loss_rot_weight,
            acc_loss_vel_weight=acc_loss_vel_weight,
            acc_loss_pos_weight=acc_loss_pos_weight,
            ba_init_model_path=ba_init_model_path,
            train_initial_bw_mode=train_initial_bw_mode,
            train_initial_ba_mode=train_initial_ba_mode,
        )
    if nll_family not in {"interval", "trajectory_state"}:
        raise ValueError(f"Unsupported nll_family: {nll_family}")
    if interval_nll_mode not in {"nll_cov_only", "nll_joint"}:
        raise ValueError(f"Unsupported interval_nll_mode: {interval_nll_mode}")
    if interval_covariance_gradient_mode not in {"analytic_sensitivity", "ad"}:
        raise ValueError(f"Unsupported interval_covariance_gradient_mode: {interval_covariance_gradient_mode}")
    if nll_family == "trajectory_state" and interval_covariance_gradient_mode != "analytic_sensitivity":
        raise ValueError("Trajectory-state NLL currently uses scalar-noise AD and does not support interval covariance modes.")

    num_iterations = epoch
    torch.manual_seed(3407)
    mask_flag = _mask_flag(dataset_train)
    debias_learning.validate_train_batch_strategy(train_batch_strategy)

    model_save_path = os.path.join(outpur_dir, "Acc_weights")
    os.makedirs(model_save_path, exist_ok=True)
    gyro_model_save_path = os.path.join(outpur_dir, "Gyro_weights")
    if nll_train_gyro and interval_nll_mode == "nll_joint":
        os.makedirs(gyro_model_save_path, exist_ok=True)
    log_writer_path = os.path.join(outpur_dir, "Acc_logs")

    ba_model = ba_func_choose(ba_model_name, device)
    _load_ba_model_initialization(ba_model, ba_init_model_path, device)
    model = debias_learning.VFSE23vbiasBase(biasfunc_w=bw_model, biasfunc_a=ba_model, device=device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    for param in model.parameters():
        param.requires_grad = False
    if interval_nll_mode == "nll_joint":
        model.unfreeze_func_ba()
        if nll_train_gyro:
            model.unfreeze_func_bw()

    noise_model = IntervalNoiseVariances(
        sigma_g_nominal=interval_sigma_g_nominal,
        sigma_a_nominal=interval_sigma_a_nominal,
        sigma_g_floor2=interval_sigma_g_floor2,
        sigma_a_floor2=interval_sigma_a_floor2,
        device=device,
    )
    _load_interval_noise_initialization(
        noise_model,
        interval_noise_init_model_path if interval_noise_init_model_path is not None else ba_init_model_path,
        device,
    )
    residual_mask = _interval_residual_mask(
        acc_loss_rot_weight,
        acc_loss_vel_weight,
        acc_loss_pos_weight,
        device,
    )
    use_delta_imu = _uses_delta_imu(bw_model_name, ba_model_name)
    gravity = model.g_const.to(device)

    debias_learning.write_parameters("Acc", dataset_train, outpur_dir, ba_model_name, lr, weight_decay, epoch, train_batch_strategy)
    writter = debias_learning.SummaryWriter(log_writer_path)
    best_loss = float("inf")

    net_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer_net = None
    scheduler_net = None
    if net_params:
        optimizer_net = torch.optim.Adam(net_params, lr=lr, weight_decay=weight_decay)
        scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=lr_step_size, gamma=lr_gamma)
    optimizer_sigma = torch.optim.Adam(noise_model.parameters(), lr=interval_lr_sigma)

    epoch_times = []
    gpu_memory = []
    loss_total = float("nan")

    def add_prior(loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior = noise_model.prior(
            interval_prior_lambda_g,
            interval_prior_lambda_a,
            interval_sigma_g_nominal,
            interval_sigma_a_nominal,
        )
        return loss + prior, prior

    for epoch_idx in range(num_iterations):
        epoch_start = debias_learning.start_epoch_profile(device)
        if optimizer_net is not None:
            optimizer_net.zero_grad()
        optimizer_sigma.zero_grad()
        model.train()
        noise_model.train()

        train_losses = []
        batch_sizes = []
        for index, batch_size in debias_learning.get_train_batches(dataset_train, train_batch_strategy):
            (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_train.get_data_for_biasDy(
                index, device, mask_flag=mask_flag, batch_size=batch_size
            )
            y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
            y0_batch = _apply_train_initial_bias_modes(
                y0_batch,
                train_initial_bw_mode,
                train_initial_ba_mode,
            )
            spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
            _set_model_imu_functions(model, spline, dataset_train.dt, use_delta_imu)
            model.set_R0(R0_batch)
            solution, R_sol = debias_learning.odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method)
            if nll_family == "trajectory_state":
                loss, _ = _trajectory_state_residual_nll(
                    solution,
                    R_sol,
                    X_gt_batch,
                    t_odeint,
                    eval_indices,
                    noise_model,
                    gravity,
                    residual_mask,
                    interval_target_cov_floor,
                    interval_jitter,
                    interval_detach_covariance_theta,
                    trajectory_supervision_mode,
                    trajectory_supervision_stride,
                    trajectory_max_supervised_states,
                )
            else:
                loss, _ = _interval_preintegration_residual_nll(
                    solution,
                    R_sol,
                    X_gt_batch,
                    t_gt_batch,
                    t_odeint,
                    eval_indices,
                    noise_model,
                    gravity,
                    residual_mask,
                    interval_target_cov_floor,
                    interval_jitter,
                    interval_detach_covariance_theta,
                    interval_covariance_gradient_mode,
                    False,
                )
            loss, _ = add_prior(loss)
            train_losses.append(loss)
            batch_sizes.append(batch_size)

        train_loss = debias_learning.combine_train_losses(train_losses, batch_sizes)
        train_loss.backward()
        if optimizer_net is not None:
            optimizer_net.step()
        optimizer_sigma.step()

        sigma_g, sigma_a = noise_model.sigmas()
        writter.add_scalar("train_loss", train_loss.item(), epoch_idx)
        writter.add_scalar("lr", optimizer_net.param_groups[0]["lr"] if optimizer_net is not None else 0.0, epoch_idx)
        writter.add_scalar("interval_lr_sigma", optimizer_sigma.param_groups[0]["lr"], epoch_idx)
        writter.add_scalar("interval_sigma_g", sigma_g.item(), epoch_idx)
        writter.add_scalar("interval_sigma_a", sigma_a.item(), epoch_idx)
        if scheduler_net is not None:
            scheduler_net.step()

        if epoch_idx % val_freq == 0:
            with torch.no_grad():
                model.eval()
                noise_model.eval()
                loss_total = 0.0
                nll_loss_total = 0.0
                prior_loss_total = 0.0
                interval_factors = 0
                supervised_states = 0
                quad_total = 0.0
                logdet_total = 0.0
                min_chol_diag = float("inf")
                residual_chunks = []
                debug_grad_totals = {
                    "grad_alpha_g_closed": 0.0,
                    "grad_alpha_a_closed": 0.0,
                    "grad_raw_sigma_g2_closed": 0.0,
                    "grad_raw_sigma_a2_closed": 0.0,
                }
                for index in range(dataset_val.length()):
                    (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_val.get_data_for_biasDy(
                        index, device, mask_flag=mask_flag
                    )
                    y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
                    y0_batch = _apply_train_initial_bias_modes(
                        y0_batch,
                        train_initial_bw_mode,
                        train_initial_ba_mode,
                    )
                    spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
                    _set_model_imu_functions(model, spline, dataset_val.dt, use_delta_imu)
                    model.set_R0(R0_batch)
                    solution, R_sol = debias_learning.odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method)
                    if nll_family == "trajectory_state":
                        nll_loss, info = _trajectory_state_residual_nll(
                            solution,
                            R_sol,
                            X_gt_batch,
                            t_odeint,
                            eval_indices,
                            noise_model,
                            gravity,
                            residual_mask,
                            interval_target_cov_floor,
                            interval_jitter,
                            interval_detach_covariance_theta,
                            trajectory_supervision_mode,
                            trajectory_supervision_stride,
                            trajectory_max_supervised_states,
                            return_residuals=True,
                        )
                    else:
                        nll_loss, info = _interval_preintegration_residual_nll(
                            solution,
                            R_sol,
                            X_gt_batch,
                            t_gt_batch,
                            t_odeint,
                            eval_indices,
                            noise_model,
                            gravity,
                            residual_mask,
                            interval_target_cov_floor,
                            interval_jitter,
                            interval_detach_covariance_theta,
                            interval_covariance_gradient_mode,
                            interval_debug_covariance_gradients,
                            return_residuals=True,
                        )
                    loss, prior = add_prior(nll_loss)
                    loss_total += loss.item()
                    nll_loss_total += nll_loss.item()
                    prior_loss_total += prior.item()
                    interval_factors += int(info["num_factors"])
                    supervised_states += int(info.get("num_supervised_states", info.get("num_intervals", 0)))
                    if "quad_mean" in info:
                        quad_total += float(info["quad_mean"].item()) * int(info["num_factors"])
                    if "logdet_mean" in info:
                        logdet_total += float(info["logdet_mean"].item()) * int(info["num_factors"])
                    if "min_chol_diag" in info:
                        min_chol_diag = min(min_chol_diag, float(info["min_chol_diag"].item()))
                    residual_chunks.append(info["residual"])
                    if interval_debug_covariance_gradients:
                        for name in debug_grad_totals:
                            debug_grad_totals[name] += float(info[name].item())

                writter.add_scalar("val_loss", loss_total, epoch_idx)
                sigma_g, sigma_a = noise_model.sigmas()
                residual_val = torch.cat([chunk.reshape(-1, chunk.shape[-1]) for chunk in residual_chunks], dim=0)
                quad_mean_text = ""
                if interval_factors > 0 and (quad_total != 0.0 or logdet_total != 0.0):
                    quad_mean = quad_total / interval_factors
                    logdet_mean = logdet_total / interval_factors
                    writter.add_scalar("val_nll_quad_mean", quad_mean, epoch_idx)
                    writter.add_scalar("val_nll_logdet_mean", logdet_mean, epoch_idx)
                    quad_mean_text = f"val_quad_mean={quad_mean:.6g} val_logdet_mean={logdet_mean:.6g} "
                min_chol_text = ""
                if np.isfinite(min_chol_diag):
                    writter.add_scalar("val_min_chol_diag", min_chol_diag, epoch_idx)
                    min_chol_text = f"val_min_chol_diag={min_chol_diag:.6g} "
                debug_grad_text = ""
                if interval_debug_covariance_gradients and residual_chunks:
                    for name, value in debug_grad_totals.items():
                        writter.add_scalar(f"val_{name}", value, epoch_idx)
                    debug_grad_text = (
                        f"grad_alpha_g_closed={debug_grad_totals['grad_alpha_g_closed']:.6g} "
                        f"grad_alpha_a_closed={debug_grad_totals['grad_alpha_a_closed']:.6g} "
                        f"grad_raw_sigma_g2_closed={debug_grad_totals['grad_raw_sigma_g2_closed']:.6g} "
                        f"grad_raw_sigma_a2_closed={debug_grad_totals['grad_raw_sigma_a2_closed']:.6g} "
                    )
                print(
                    f"stage={'AccTrajectoryStateNLL' if nll_family == 'trajectory_state' else 'AccIntervalNLL'} "
                    f"epoch={epoch_idx} train_loss={train_loss.item():.6g} "
                    f"val_loss={loss_total:.6g} family={nll_family} mode={interval_nll_mode} train_gyro={nll_train_gyro} "
                    f"sigma_a={sigma_a.item():.8g} sigma_g={sigma_g.item():.8g} "
                    f"val_preintegration_nll={nll_loss_total:.6g} val_prior={prior_loss_total:.6g} "
                    f"val_factors={interval_factors} val_supervised_states={supervised_states} "
                    f"{quad_mean_text}{min_chol_text}"
                    f"detach_cov_theta={interval_detach_covariance_theta} "
                    f"cov_grad_mode={interval_covariance_gradient_mode} "
                    f"{debug_grad_text}"
                    f"{_format_interval_residual_metrics(residual_val, 'trajectory' if nll_family == 'trajectory_state' else 'interval')}",
                    flush=True,
                )

            if loss_total < best_loss:
                best_loss = loss_total
                best_checkpoint = {
                    "func_ba_model": model.biasfunc_a.state_dict(),
                    "optimizer": optimizer_net.state_dict() if optimizer_net is not None else None,
                    "scheduler": scheduler_net.state_dict() if scheduler_net is not None else None,
                    "optimizer_sigma": optimizer_sigma.state_dict(),
                    "raw_sigma_g2": noise_model.raw_sigma_g2.detach(),
                    "raw_sigma_a2": noise_model.raw_sigma_a2.detach(),
                    "interval_raw_sigma_g2": noise_model.raw_sigma_g2.detach(),
                    "interval_raw_sigma_a2": noise_model.raw_sigma_a2.detach(),
                    "interval_sigma_g": sigma_g.detach(),
                    "interval_sigma_a": sigma_a.detach(),
                    "epoch": epoch_idx,
                    "best_loss": best_loss,
                    "interval_nll_mode": interval_nll_mode,
                    "interval_detach_covariance_theta": interval_detach_covariance_theta,
                    "interval_covariance_gradient_mode": interval_covariance_gradient_mode,
                    "interval_debug_covariance_gradients": interval_debug_covariance_gradients,
                    "nll_train_gyro": nll_train_gyro,
                    "nll_family": nll_family,
                    "trajectory_supervision_mode": trajectory_supervision_mode,
                    "trajectory_supervision_stride": trajectory_supervision_stride,
                    "trajectory_max_supervised_states": trajectory_max_supervised_states,
                }
                if nll_train_gyro and interval_nll_mode == "nll_joint":
                    best_checkpoint["func_bw_model"] = model.biasfunc_w.state_dict()
                torch.save(best_checkpoint, model_save_path + "/best_val_model.pt")
                if nll_train_gyro and interval_nll_mode == "nll_joint":
                    torch.save(
                        {
                            "func_bw_model": model.biasfunc_w.state_dict(),
                            "optimizer": optimizer_net.state_dict() if optimizer_net is not None else None,
                            "scheduler": scheduler_net.state_dict() if scheduler_net is not None else None,
                            "optimizer_sigma": optimizer_sigma.state_dict(),
                            "raw_sigma_g2": noise_model.raw_sigma_g2.detach(),
                            "raw_sigma_a2": noise_model.raw_sigma_a2.detach(),
                            "epoch": epoch_idx,
                            "best_loss": best_loss,
                            "interval_joint_acc_checkpoint": model_save_path + "/best_val_model.pt",
                        },
                        gyro_model_save_path + "/best_val_model.pt",
                    )

        if epoch_idx % save_freq == 0:
            epoch_checkpoint = {
                "func_ba_model": model.biasfunc_a.state_dict(),
                "optimizer": optimizer_net.state_dict() if optimizer_net is not None else None,
                "scheduler": scheduler_net.state_dict() if scheduler_net is not None else None,
                "optimizer_sigma": optimizer_sigma.state_dict(),
                "raw_sigma_g2": noise_model.raw_sigma_g2.detach(),
                "raw_sigma_a2": noise_model.raw_sigma_a2.detach(),
                "interval_raw_sigma_g2": noise_model.raw_sigma_g2.detach(),
                "interval_raw_sigma_a2": noise_model.raw_sigma_a2.detach(),
                "interval_sigma_g": sigma_g.detach(),
                "interval_sigma_a": sigma_a.detach(),
                "epoch": epoch_idx,
                "valloss": loss_total,
                "interval_nll_mode": interval_nll_mode,
                "interval_detach_covariance_theta": interval_detach_covariance_theta,
                "interval_covariance_gradient_mode": interval_covariance_gradient_mode,
                "interval_debug_covariance_gradients": interval_debug_covariance_gradients,
                "nll_train_gyro": nll_train_gyro,
                "nll_family": nll_family,
                "trajectory_supervision_mode": trajectory_supervision_mode,
                "trajectory_supervision_stride": trajectory_supervision_stride,
                "trajectory_max_supervised_states": trajectory_max_supervised_states,
            }
            if nll_train_gyro and interval_nll_mode == "nll_joint":
                epoch_checkpoint["func_bw_model"] = model.biasfunc_w.state_dict()
            torch.save(epoch_checkpoint, model_save_path + "/model_epoch_" + str(epoch_idx) + ".pt")
            if nll_train_gyro and interval_nll_mode == "nll_joint":
                torch.save(
                    {
                        "func_bw_model": model.biasfunc_w.state_dict(),
                        "optimizer": optimizer_net.state_dict() if optimizer_net is not None else None,
                        "scheduler": scheduler_net.state_dict() if scheduler_net is not None else None,
                        "optimizer_sigma": optimizer_sigma.state_dict(),
                        "raw_sigma_g2": noise_model.raw_sigma_g2.detach(),
                        "raw_sigma_a2": noise_model.raw_sigma_a2.detach(),
                        "epoch": epoch_idx,
                        "valloss": loss_total,
                        "interval_joint_acc_checkpoint": model_save_path + "/model_epoch_" + str(epoch_idx) + ".pt",
                    },
                    gyro_model_save_path + "/model_epoch_" + str(epoch_idx) + ".pt",
                )

        epoch_time, peak_memory_mb = debias_learning.finish_epoch_profile(device, epoch_start)
        epoch_times.append(epoch_time)
        gpu_memory.append(peak_memory_mb)

    writter.close()
    return debias_learning.write_stage_profile(outpur_dir, "Acc", epoch_times, gpu_memory)


def _rotate(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (R @ x.unsqueeze(-1)).squeeze(-1)


def _calculate_raw_imu(dataset: BaseDataset, index: int, device: str, path: Path) -> None:
    if path.exists():
        return
    print("No results using raw data, calculate it now!")
    _t_imu, u_imu, X_gt, t_gt = dataset.get_full_trajectory(index)
    u_imu = u_imu.to(device)
    X_gt = X_gt.to(device)
    t_gt = t_gt.to(device)
    X_raw = torch.zeros_like(X_gt)
    X_raw[0] = X_gt[0]
    g_const = torch.tensor([0, 0, -9.81], dtype=X_gt.dtype, device=device)
    for i in range(1, X_gt.shape[0]):
        dt = t_gt[i] - t_gt[i - 1]
        X_raw[i, :3, :3] = X_raw[i - 1, :3, :3] @ Lie.SO3exp(u_imu[i - 1, :3] * dt)
        X_raw[i, :3, 3] = X_raw[i - 1, :3, 3] + (_rotate(X_raw[i - 1, :3, :3], u_imu[i - 1, 3:]) + g_const) * dt
        X_raw[i, :3, 4] = X_raw[i - 1, :3, 4] + X_raw[i - 1, :3, 3] * dt
    pdump({"X_raw": X_raw.cpu()}, str(path))


def _calculate_constant_bias(
    dataset: BaseDataset,
    index: int,
    device: str,
    path: Path,
    initial_bias_mode: str = "first",
    initial_bias_window_s: float = 1.0,
    fixed_bw0: torch.Tensor | None = None,
    fixed_ba0: torch.Tensor | None = None,
) -> None:
    if path.exists():
        return
    print("No results using constant bias, calculate it now!")
    _t_imu, u_imu, X_gt, t_gt = dataset.get_full_trajectory(index)
    bias_w_gt, t_w_bias = dataset.get_w_bias(index)
    bias_a_gt, t_a_bias = dataset.get_a_bias(index)
    u_imu = u_imu.to(device)
    X_gt = X_gt.to(device)
    t_gt = t_gt.to(device)
    bw_const = _resolve_initial_bias(
        bias_w_gt,
        t_w_bias,
        initial_bias_mode,
        initial_bias_window_s,
        device,
        fixed_bias=fixed_bw0,
    )
    ba_const = _resolve_initial_bias(
        bias_a_gt,
        t_a_bias,
        initial_bias_mode,
        initial_bias_window_s,
        device,
        fixed_bias=fixed_ba0,
    )
    X_const = torch.zeros_like(X_gt)
    X_const[0] = X_gt[0]
    g_const = torch.tensor([0, 0, -9.81], dtype=X_gt.dtype, device=device)
    for i in range(1, X_gt.shape[0]):
        dt = t_gt[i] - t_gt[i - 1]
        w_corr = u_imu[i - 1, :3] - bw_const
        a_corr = u_imu[i - 1, 3:] - ba_const
        X_const[i, :3, :3] = X_const[i - 1, :3, :3] @ Lie.SO3exp(w_corr * dt)
        X_const[i, :3, 3] = X_const[i - 1, :3, 3] + (_rotate(X_const[i - 1, :3, :3], a_corr) + g_const) * dt
        X_const[i, :3, 4] = X_const[i - 1, :3, 4] + X_const[i - 1, :3, 3] * dt
    pdump(
        {
            "X_const": X_const.cpu(),
            "bw_const": bw_const.cpu(),
            "ba_const": ba_const.cpu(),
            "initial_bias_mode": initial_bias_mode,
            "initial_bias_window_s": float(initial_bias_window_s),
        },
        str(path),
    )


def Test_with_configured_aux(
    dataset_test: BaseDataset,
    output_dir: str,
    bw_model,
    ba_model,
    integral_method: str = "euler",
    device: str = "cpu",
    recompute: bool = False,
    use_delta_imu: bool = False,
    initial_bias_mode: str = "first",
    initial_bias_window_s: float = 1.0,
    fixed_bw0: torch.Tensor | None = None,
    fixed_ba0: torch.Tensor | None = None,
) -> None:
    model = debias_learning.VFSE23vbiasBase(biasfunc_w=bw_model, biasfunc_a=ba_model, device=device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.eval()

    output_path = Path(output_dir)
    for index in range(dataset_test.length()):
        seq_name = dataset_test.sequences[index]
        seq_dir = output_path / "Sequnces-results" / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        path_tmp = seq_dir / "FullTra_result.p"
        if path_tmp.exists() and not recompute:
            continue
        print("Testing trajectory: ", seq_name)
        coeff, time_tensor = dataset_test.get_coeff(index)
        spline = Interpolation.CubicHermiteSpline(time_tensor, coeff, device)
        _t_imu, u_imu, X_gt, t_gt = dataset_test.get_full_trajectory(index, device=device)
        R0 = X_gt[0, :3, :3].clone()
        bias_w_gt, t_w_bias = dataset_test.get_w_bias(index)
        bias_a_gt, t_a_bias = dataset_test.get_a_bias(index)
        bw0 = _resolve_initial_bias(
            bias_w_gt,
            t_w_bias,
            initial_bias_mode,
            initial_bias_window_s,
            device,
            fixed_bias=fixed_bw0,
        )
        ba0 = _resolve_initial_bias(
            bias_a_gt,
            t_a_bias,
            initial_bias_mode,
            initial_bias_window_s,
            device,
            fixed_bias=fixed_ba0,
        )
        vec3zeros = torch.zeros(3, dtype=X_gt.dtype, device=device)
        y0 = torch.cat([vec3zeros, t_gt[0].unsqueeze(-1), bw0, X_gt[0, :3, 3], X_gt[0, :3, 4], ba0], dim=-1).unsqueeze(0)

        _set_model_imu_functions(model, spline, dataset_test.dt, use_delta_imu)
        model.set_R0(R0)
        sol_pred, R_pred = debias_learning.odeint_SO3(model, y0, R0, t_gt, method=integral_method)
        sol_pred = sol_pred.squeeze(1)
        R_pred = R_pred.squeeze(1)
        net_imu = u_imu - torch.cat([sol_pred[:, 4:7], sol_pred[:, 13:16]], dim=-1)
        pdump(
            {
                "sol_pred": sol_pred.cpu(),
                "R_pred": R_pred.cpu(),
                "X_gt": X_gt.cpu(),
                "t_gt": t_gt.cpu(),
                "net_imu": net_imu.cpu(),
                "imu_raw": u_imu.cpu(),
                "timestamp_start": dataset_test.get_start_timestamp(index),
                "initial_bias_mode": initial_bias_mode,
                "initial_bias_window_s": float(initial_bias_window_s),
                "bw0": bw0.cpu(),
                "ba0": ba0.cpu(),
                "fixed_initial_bias": fixed_bw0 is not None or fixed_ba0 is not None,
            },
            str(path_tmp),
        )

    for index in range(dataset_test.length()):
        seq_name = dataset_test.sequences[index]
        raw_path = output_path / "raw_int" / f"{seq_name}.p"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _calculate_raw_imu(dataset_test, index, device, raw_path)
        const_path = output_path / "constant_bias" / f"{seq_name}.p"
        const_path.parent.mkdir(parents=True, exist_ok=True)
        _calculate_constant_bias(
            dataset_test,
            index,
            device,
            const_path,
            initial_bias_mode=initial_bias_mode,
            initial_bias_window_s=initial_bias_window_s,
            fixed_bw0=fixed_bw0,
            fixed_ba0=fixed_ba0,
        )


class LIASuwDebiasDataset(BaseDataset):
    """TUM-VI-style loader for LIASuw IMU/body-GT CSVs.

    The generated LIASuw Debias_IMU tree has `mav0/imu0/data.csv` and
    `mav0/mocap0/data.csv`, matching the TUM-VI layout.  The external
    TUMDataset scans every directory in the root, which is inconvenient because
    PLATO also exports `_firstframe` diagnostic folders.  This loader processes
    only the train/val/test sequence lists passed by the runner.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        data_cache_dir: str,
        train_seqs: list[str],
        val_seqs: list[str],
        test_seqs: list[str],
        mode: str,
        percent_for_val: float,
        batch_size: int,
        loss_window: int,
        dt: float,
        sg_window_bw: int,
        sg_order_bw: int,
        sg_window_ba: int,
        sg_order_ba: int,
        trainval_supervision_source: str = MOCAP_200HZ_SUPERVISION,
        pseudo_gt_root: str | Path | None = None,
        vo_supervision_world_frame: str = "gt",
        vo_gap_tolerance_s: float = 0.015,
        gravity_calibration_json: str | Path | None = None,
        gravity_accel_world: tuple[float, float, float] = DEFAULT_GRAVITY_ACCEL_WORLD,
        source_position_sg_window: int = 0,
        source_position_sg_order: int = 3,
        recompute: bool = False,
    ):
        super().__init__(
            dataset_name,
            data_cache_dir,
            train_seqs,
            val_seqs,
            test_seqs,
            mode,
            percent_for_val,
            batch_size,
            loss_window,
            dt,
        )
        self._sg_window_bw = sg_window_bw
        self._sg_order_bw = sg_order_bw
        self._sg_window_ba = sg_window_ba
        self._sg_order_ba = sg_order_ba
        self._trainval_supervision_source = trainval_supervision_source
        self._pseudo_gt_root = Path(pseudo_gt_root) if pseudo_gt_root is not None else None
        self._vo_supervision_world_frame = vo_supervision_world_frame
        self._vo_gap_tolerance_s = vo_gap_tolerance_s
        self._gravity_default_accel_world = np.asarray(gravity_accel_world, dtype=np.float64)
        self._source_position_sg_window = int(source_position_sg_window)
        self._source_position_sg_order = int(source_position_sg_order)
        if self._source_position_sg_window < 0:
            raise ValueError("source_position_sg_window must be non-negative.")
        if self._source_position_sg_order < 0:
            raise ValueError("source_position_sg_order must be non-negative.")
        if self._gravity_default_accel_world.shape != (3,):
            raise ValueError(f"gravity_accel_world must have shape (3,), got {self._gravity_default_accel_world.shape}")
        self._gravity_calibration_path = Path(gravity_calibration_json) if gravity_calibration_json else None
        self._gravity_calibration = self._load_gravity_calibration(self._gravity_calibration_path)
        self._trainval_supervision_timing = (
            "source"
            if trainval_supervision_source in {MOCAP_NONINTERP_SUPERVISION, VO_IMU_NONINTERP_SUPERVISION}
            else "imu"
        )
        Path(self.data_cache_dir).mkdir(parents=True, exist_ok=True)
        self.read_data(data_dir, recompute=recompute)

    @staticmethod
    def _load_gravity_calibration(path: Path | None) -> dict:
        if path is None:
            return {}
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        return payload

    def _all_requested_sequences(self) -> list[str]:
        sequences: list[str] = []
        for seq_list in self._sequences_dict.values():
            for seq in seq_list:
                if seq not in sequences:
                    sequences.append(seq)
        return sequences

    def _sequence_supervision_source(self, sequence: str) -> str:
        seq_mode = self.find_sequence_mode(sequence)
        if seq_mode == "test":
            return MOCAP_200HZ_SUPERVISION
        return self._trainval_supervision_source

    def _gravity_record(self, sequence: str) -> dict | None:
        if not self._gravity_calibration:
            return None
        records = self._gravity_calibration.get("sequences", self._gravity_calibration)
        if not isinstance(records, dict):
            raise ValueError("gravity calibration JSON must contain a 'sequences' object or be keyed by sequence.")
        record = records.get(sequence)
        if record is None:
            return None
        if not isinstance(record, dict):
            raise ValueError(f"Gravity calibration for {sequence} must be a JSON object.")
        return record

    def _gravity_accel_world(self, sequence: str) -> np.ndarray:
        record = self._gravity_record(sequence)
        if record is None:
            return self._gravity_default_accel_world.copy()
        for key in ("gravity_accel_world_m_s2", "gravity_accel_world_mps2", "gravity_accel_world"):
            if key in record:
                gravity = np.asarray(record[key], dtype=np.float64)
                break
        else:
            if "gravity_specific_world_m_s2" in record:
                gravity = -np.asarray(record["gravity_specific_world_m_s2"], dtype=np.float64)
            else:
                raise ValueError(f"Gravity calibration for {sequence} has no gravity_accel_world field.")
        if gravity.shape != (3,):
            raise ValueError(f"Gravity calibration for {sequence} must have shape (3,), got {gravity.shape}")
        return gravity

    def _gravity_cache_key(self, sequence: str) -> dict:
        record = self._gravity_record(sequence)
        source = str(self._gravity_calibration_path) if record is not None else "default"
        gravity = self._gravity_accel_world(sequence)
        return {
            "source": source,
            "route": GRAVITY_ALIGNMENT_ROUTE,
            "gravity_accel_world_m_s2": [round(float(x), 12) for x in gravity],
        }

    def _source_derivative_cache_key(self) -> dict | None:
        if self._source_position_sg_window <= 0:
            return None
        return {
            "method": "savgol_position_derivative",
            "window": int(self._source_position_sg_window),
            "order": int(self._source_position_sg_order),
        }

    def _effective_source_sg_window(self, num_samples: int) -> int | None:
        if self._source_position_sg_window <= 0:
            return None
        max_window = int(num_samples) if int(num_samples) % 2 == 1 else int(num_samples) - 1
        min_window = int(self._source_position_sg_order) + 2
        if min_window % 2 == 0:
            min_window += 1
        window = int(self._source_position_sg_window)
        if window % 2 == 0:
            window += 1
        window = min(window, max_window)
        if window < min_window:
            return None
        return window

    def _source_position_derivatives(
        self,
        p_gt: np.ndarray,
        t_gt: np.ndarray,
        source_dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        window = self._effective_source_sg_window(p_gt.shape[0])
        if window is None:
            v_gt = np.gradient(p_gt, t_gt, axis=0, edge_order=2)
            a_world = np.gradient(v_gt, t_gt, axis=0, edge_order=2)
            return v_gt, a_world
        v_gt = self.apply_sg_smoother(
            p_gt,
            window,
            self._source_position_sg_order,
            deriv=1,
        ) / source_dt
        a_world = self.apply_sg_smoother(
            p_gt,
            window,
            self._source_position_sg_order,
            deriv=2,
        ) / (source_dt * source_dt)
        return v_gt, a_world

    def _gravity_alignment_matrix(self, sequence: str) -> np.ndarray:
        source = self._gravity_accel_world(sequence)
        target = np.asarray(DEFAULT_GRAVITY_ACCEL_WORLD, dtype=np.float64)
        source_norm = np.linalg.norm(source)
        target_norm = np.linalg.norm(target)
        if source_norm < 1e-12 or target_norm < 1e-12:
            raise ValueError(f"Invalid gravity vector for {sequence}: source={source}, target={target}")
        source_dir = source / source_norm
        target_dir = target / target_norm
        if np.linalg.norm(source_dir - target_dir) < 1e-12:
            return np.eye(3, dtype=np.float64)
        rotation, _ = Rotation.align_vectors(target_dir.reshape(1, 3), source_dir.reshape(1, 3))
        return rotation.as_matrix()

    def _quality_window_path(self, data_dir: str | Path, sequence: str) -> Path:
        return Path(data_dir) / sequence / "mav0" / "quality" / f"valid_window_starts_{self._loss_window}.txt"

    def _quality_stats_path(self, data_dir: str | Path, sequence: str) -> Path:
        return Path(data_dir) / sequence / "mav0" / "quality" / f"window_stats_{self._loss_window}.json"

    def _quality_cache_key(self, data_dir: str | Path, sequence: str) -> dict | None:
        path = self._quality_window_path(data_dir, sequence)
        if not path.exists():
            return None
        stats_path = self._quality_stats_path(data_dir, sequence)
        return {
            "path": str(path),
            "stats_path": str(stats_path) if stats_path.exists() else None,
            "loss_window": int(self._loss_window),
            "timing": "imu",
        }

    def _load_quality_valid_starts(self, data_dir: str | Path, sequence: str) -> tuple[torch.Tensor | None, dict | None]:
        key = self._quality_cache_key(data_dir, sequence)
        if key is None:
            return None, None
        path = Path(key["path"])
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            starts = np.empty((0,), dtype=np.int64)
        else:
            starts = np.loadtxt(path, dtype=np.int64)
            starts = np.atleast_1d(starts).astype(np.int64)
        return torch.from_numpy(starts).long(), key

    def _cache_sources_match(self, sequences: list[str], data_dir: str | Path) -> bool:
        for sequence in sequences:
            expected = self._sequence_supervision_source(sequence)
            try:
                imudict = pload(self.data_cache_dir, sequence + ".p")
            except Exception:
                return False
            actual = imudict.get("supervision_source", MOCAP_200HZ_SUPERVISION)
            if actual != expected:
                return False
            if expected == VO_IMU_NONINTERP_SUPERVISION:
                actual_frame = imudict.get("vo_supervision_world_frame")
                if actual_frame != self._vo_supervision_world_frame:
                    return False
            expected_gravity = self._gravity_cache_key(sequence)
            actual_gravity = imudict.get("gravity_calibration")
            if actual_gravity is None and expected_gravity["source"] == "default":
                pass
            elif actual_gravity != expected_gravity:
                return False
            if expected in {MOCAP_NONINTERP_SUPERVISION, VO_IMU_NONINTERP_SUPERVISION}:
                expected_derivative = self._source_derivative_cache_key()
                actual_derivative = imudict.get("source_derivative_filter")
                if actual_derivative != expected_derivative:
                    return False
            expected_quality = self._quality_cache_key(data_dir, sequence)
            actual_quality = imudict.get("quality_window_filter")
            if expected_quality is None:
                if actual_quality is not None:
                    return False
            elif actual_quality != expected_quality:
                return False
        return True

    def _vo_pose_path(self, sequence: str) -> Path:
        if self._pseudo_gt_root is None:
            raise ValueError("--pseudo-gt-root is required when trainval supervision source is VO.")
        return self._pseudo_gt_root / sequence / "debias_imu_body_gt_200hz" / "vo_imu_traj.tum"

    def _vo_world_transform(self, sequence: str) -> tuple[np.ndarray, str]:
        if self._vo_supervision_world_frame == "vo":
            return np.eye(4), "identity: keep T_Wv_I in VO world"
        if self._vo_supervision_world_frame != "gt":
            raise ValueError(f"Unsupported VO supervision world frame: {self._vo_supervision_world_frame}")

        calibration_path = self._pseudo_gt_root / sequence / "debias_imu_body_gt_200hz" / "calibration_result.yaml"
        if not calibration_path.exists():
            raise FileNotFoundError(calibration_path)
        with calibration_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        transform = np.asarray(payload["T_WO"], dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"{calibration_path} T_WO has shape {transform.shape}, expected (4, 4)")
        return transform, f"{calibration_path}: T_WO/T_Wm_Wv applied to T_Wv_I"

    def _valid_source_starts(self, t_gt: np.ndarray, source_dt: float) -> np.ndarray:
        max_start = t_gt.shape[0] - self._loss_window
        if max_start < 0:
            raise ValueError(f"VO supervision has fewer poses ({t_gt.shape[0]}) than loss_window={self._loss_window}")
        dt_gt = np.diff(t_gt)
        valid = []
        for start in range(max_start + 1):
            window_dt = dt_gt[start : start + self._loss_window - 1]
            if np.all(window_dt > 0.0) and np.all(np.abs(window_dt - source_dt) <= self._vo_gap_tolerance_s):
                valid.append(start)
        if not valid:
            raise ValueError("No valid non-gap VO supervision windows were found.")
        return np.asarray(valid, dtype=np.int64)

    def _valid_mocap_source_starts(
        self,
        t_gt: np.ndarray,
        imu_t: np.ndarray,
        min_dt_s: float = 0.003,
        max_gap_s: float = 0.03,
        max_imu_gap_s: float = 0.02,
    ) -> np.ndarray:
        max_start = t_gt.shape[0] - self._loss_window
        if max_start < 0:
            raise ValueError(f"Mocap supervision has fewer poses ({t_gt.shape[0]}) than loss_window={self._loss_window}")
        dt_gt = np.diff(t_gt)
        dt_imu = np.diff(imu_t)
        valid = []
        for start in range(max_start + 1):
            stop = start + self._loss_window
            window_dt = dt_gt[start : stop - 1]
            if not (np.all(window_dt >= min_dt_s) and np.all(window_dt <= max_gap_s)):
                continue
            imu_start = np.searchsorted(imu_t, t_gt[start], side="left")
            imu_stop = np.searchsorted(imu_t, t_gt[stop - 1], side="right")
            if imu_stop - imu_start < 2:
                continue
            if np.any(dt_imu[max(0, imu_start - 1) : max(0, imu_stop - 1)] > max_imu_gap_s):
                continue
            valid.append(start)
        if not valid:
            raise ValueError("No valid non-interpolated mocap supervision windows were found.")
        return np.asarray(valid, dtype=np.int64)

    def _cache_mocap_noninterp_sequence(self, data_dir: str, sequence: str) -> None:
        path_imu = Path(data_dir) / sequence / "mav0" / "imu0" / "data.csv"
        path_gt = Path(data_dir) / sequence / "mav0" / "mocap0" / "data.csv"
        if not path_imu.exists():
            raise FileNotFoundError(path_imu)
        if not path_gt.exists():
            raise FileNotFoundError(path_gt)

        imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
        gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)
        if imu.ndim == 1:
            imu = imu.reshape(1, -1)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)

        imu_t_s_all = imu[:, 0] / 1e9
        gt_t_s_all = gt[:, 0] / 1e9
        t0_s = max(gt_t_s_all[0], imu_t_s_all[0])
        t_end_s = min(gt_t_s_all[-1], imu_t_s_all[-1])
        idx0_imu = np.searchsorted(imu_t_s_all, t0_s)
        idx_end_imu = np.searchsorted(imu_t_s_all, t_end_s, "right")
        idx0_gt = np.searchsorted(gt_t_s_all, t0_s)
        idx_end_gt = np.searchsorted(gt_t_s_all, t_end_s, "right")
        imu = imu[idx0_imu:idx_end_imu]
        gt = gt[idx0_gt:idx_end_gt]
        if imu.shape[0] < 4 or gt.shape[0] < max(4, self._loss_window):
            raise ValueError(f"Sequence {sequence} is too short after IMU/mocap overlap cropping.")

        timestamp_start = imu[0, 0]
        imu_ori = imu.copy()
        imu_t_rel = imu_ori[:, 0] / 1e9
        imu_t_rel = imu_t_rel - imu_t_rel[0]
        ts = self.dt * np.arange(imu_ori.shape[0])
        imu_resampled = np.zeros_like(imu_ori)
        imu_resampled[:, 0] = ts
        for axis in range(6):
            imu_resampled[:, 1 + axis] = np.interp(ts, imu_t_rel, imu_ori[:, 1 + axis])

        t_gt = (gt[:, 0] - timestamp_start) / 1e9
        valid_gt = (t_gt >= ts[0]) & (t_gt <= ts[-1])
        gt = gt[valid_gt]
        t_gt = t_gt[valid_gt]
        if gt.shape[0] < max(4, self._loss_window):
            raise ValueError(f"Sequence {sequence} has too few sparse mocap poses inside the IMU span.")

        p_gt = gt[:, 1:4]
        q_gt_np = gt[:, [4, 5, 6, 7]]
        q_gt = torch.from_numpy(q_gt_np).double()
        q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        rot_gt = Lie.quat_to_SO3(q_gt).cpu()

        source_dt = float(np.median(np.diff(t_gt)))
        v_gt, a_world = self._source_position_derivatives(p_gt, t_gt, source_dt)
        imu_at_gt = np.zeros((t_gt.shape[0], 6), dtype=np.float64)
        for axis in range(6):
            imu_at_gt[:, axis] = np.interp(t_gt, ts, imu_resampled[:, 1 + axis])

        imu_gt_t = torch.from_numpy(imu_at_gt).double()
        dt_gt_t = torch.from_numpy(np.diff(t_gt)).double().unsqueeze(-1)
        w_body = Lie.SO3log(rot_gt[:-1].transpose(-1, -2) @ rot_gt[1:]) / dt_gt_t
        w_bias = imu_gt_t[:-1, :3] - w_body
        w_bias = torch.cat((w_bias, w_bias[-1].unsqueeze(0)), dim=0)
        w_bias = torch.from_numpy(
            self.apply_sg_smoother(w_bias.cpu().numpy(), self._sg_window_bw, self._sg_order_bw)
        )

        g_const = torch.tensor(DEFAULT_GRAVITY_ACCEL_WORLD, dtype=torch.double)
        a_world_minus_g = torch.from_numpy(a_world).double() - g_const.unsqueeze(0)
        a_bias = imu_gt_t[:, 3:6] - (rot_gt.transpose(-1, -2) @ a_world_minus_g.unsqueeze(-1)).squeeze(-1)
        a_bias = torch.from_numpy(
            self.apply_sg_smoother(a_bias.cpu().numpy(), self._sg_window_ba, self._sg_order_ba)
        )

        imu_t = torch.from_numpy(imu_resampled[:, 1:]).double()
        ts_t = torch.from_numpy(ts).double()
        coeff = Interpolation.get_cubic_hermite_coeffs(ts_t, imu_t)
        imu_coeff = tuple(x.float() for x in coeff)
        precomputed_valid_path = Path(data_dir) / sequence / "mav0" / "quality" / f"valid_source_starts_{self._loss_window}.txt"
        if precomputed_valid_path.exists() and precomputed_valid_path.read_text(encoding="utf-8").strip():
            valid_starts = np.atleast_1d(np.loadtxt(precomputed_valid_path, dtype=np.int64)).astype(np.int64)
        else:
            valid_starts = self._valid_mocap_source_starts(t_gt, imu_t_rel)
        valid_starts = valid_starts[(valid_starts >= 0) & (valid_starts <= t_gt.shape[0] - self._loss_window)]
        if valid_starts.size == 0:
            raise ValueError(f"No valid non-interpolated mocap windows remain for {sequence} after overlap cropping.")

        gravity_cache_key = self._gravity_cache_key(sequence)
        source_derivative_key = self._source_derivative_cache_key()
        print(
            f"  supervision_source={MOCAP_NONINTERP_SUPERVISION}, "
            f"mocap_poses={t_gt.shape[0]}, source_dt_median={source_dt:.6f}s, "
            f"window_span_median={(self._loss_window - 1) * source_dt:.6f}s, "
            f"valid_windows={valid_starts.shape[0]}",
            flush=True,
        )
        pdump(
            {
                "t_imu": ts_t.float(),
                "u_imu": imu_t.float(),
                "w_bias": w_bias.float(),
                "a_bias": a_bias.float(),
                "t_bias": torch.from_numpy(t_gt).float(),
                "coeff": imu_coeff,
                "t_start": timestamp_start,
                "mask_missing": torch.zeros_like(ts_t).float(),
                "source_dt": source_dt,
                "source_valid_starts": torch.from_numpy(valid_starts).long(),
                "supervision_source": MOCAP_NONINTERP_SUPERVISION,
                "R_gravity_alignment": np.eye(3, dtype=np.float64).tolist(),
                "gravity_calibration": gravity_cache_key,
                "source_derivative_filter": source_derivative_key,
            },
            self.data_cache_dir,
            sequence + ".p",
        )
        pdump(
            {
                "t_gt": torch.from_numpy(t_gt).float(),
                "q_gt": q_gt.float(),
                "R_gt": rot_gt.float(),
                "v_gt": torch.from_numpy(v_gt).float(),
                "p_gt": torch.from_numpy(p_gt).float(),
                "supervision_source": MOCAP_NONINTERP_SUPERVISION,
                "R_gravity_alignment": np.eye(3, dtype=np.float64).tolist(),
                "gravity_calibration": gravity_cache_key,
                "source_derivative_filter": source_derivative_key,
            },
            self.data_cache_dir,
            sequence + "_gt.p",
        )

    def _cache_vo_noninterp_sequence(self, data_dir: str, sequence: str) -> None:
        path_imu = Path(data_dir) / sequence / "mav0" / "imu0" / "data.csv"
        path_vo = self._vo_pose_path(sequence)
        if not path_imu.exists():
            raise FileNotFoundError(path_imu)
        if not path_vo.exists():
            raise FileNotFoundError(path_vo)

        imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
        vo = np.loadtxt(path_vo)
        if imu.ndim == 1:
            imu = imu.reshape(1, -1)
        if vo.ndim == 1:
            vo = vo.reshape(1, -1)

        imu_t_s_all = imu[:, 0] / 1e9
        vo_t_s_all = vo[:, 0]
        t0_s = max(vo_t_s_all[0], imu_t_s_all[0])
        t_end_s = min(vo_t_s_all[-1], imu_t_s_all[-1])
        idx0_imu = np.searchsorted(imu_t_s_all, t0_s)
        idx_end_imu = np.searchsorted(imu_t_s_all, t_end_s, "right")
        idx0_vo = np.searchsorted(vo_t_s_all, t0_s)
        idx_end_vo = np.searchsorted(vo_t_s_all, t_end_s, "right")
        imu = imu[idx0_imu:idx_end_imu]
        vo = vo[idx0_vo:idx_end_vo]
        if imu.shape[0] < 4 or vo.shape[0] < max(4, self._loss_window):
            raise ValueError(f"Sequence {sequence} is too short after IMU/VO overlap cropping.")

        timestamp_start = imu[0, 0]
        imu_ori = imu.copy()
        imu_t_rel = imu_ori[:, 0] / 1e9
        imu_t_rel = imu_t_rel - imu_t_rel[0]
        ts = self.dt * np.arange(imu_ori.shape[0])
        imu_resampled = np.zeros_like(imu_ori)
        imu_resampled[:, 0] = ts
        for axis in range(6):
            imu_resampled[:, 1 + axis] = np.interp(ts, imu_t_rel, imu_ori[:, 1 + axis])

        t_gt = vo[:, 0] - imu_ori[0, 0] / 1e9
        valid_gt = (t_gt >= ts[0]) & (t_gt <= ts[-1])
        vo = vo[valid_gt]
        t_gt = t_gt[valid_gt]
        if vo.shape[0] < max(4, self._loss_window):
            raise ValueError(f"Sequence {sequence} has too few VO poses inside the resampled IMU span.")

        T_supervision_source, transform_note = self._vo_world_transform(sequence)
        R_supervision_source = T_supervision_source[:3, :3]
        p_supervision_source = T_supervision_source[:3, 3]
        p_gt = (R_supervision_source @ vo[:, 1:4].T).T + p_supervision_source
        gravity_cache_key = self._gravity_cache_key(sequence)
        R_gravity_align = self._gravity_alignment_matrix(sequence)
        p_gt = (R_gravity_align @ p_gt.T).T
        p_gt = p_gt - p_gt[0]
        r_vo = Rotation.from_quat(vo[:, 4:8])
        r_supervision = Rotation.from_matrix(R_gravity_align) * Rotation.from_matrix(R_supervision_source) * r_vo
        q_xyzw = r_supervision.as_quat()
        q_gt_np = np.column_stack((q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]))
        q_gt = torch.from_numpy(q_gt_np).double()
        q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        rot_gt = Lie.quat_to_SO3(q_gt).cpu()

        source_dt = float(np.median(np.diff(t_gt)))
        v_gt, a_world = self._source_position_derivatives(p_gt, t_gt, source_dt)
        imu_at_gt = np.zeros((t_gt.shape[0], 6), dtype=np.float64)
        for axis in range(6):
            imu_at_gt[:, axis] = np.interp(t_gt, ts, imu_resampled[:, 1 + axis])

        imu_gt_t = torch.from_numpy(imu_at_gt).double()
        dt_gt_t = torch.from_numpy(np.diff(t_gt)).double().unsqueeze(-1)
        w_body = Lie.SO3log(rot_gt[:-1].transpose(-1, -2) @ rot_gt[1:]) / dt_gt_t
        w_bias = imu_gt_t[:-1, :3] - w_body
        w_bias = torch.cat((w_bias, w_bias[-1].unsqueeze(0)), dim=0)
        w_bias = torch.from_numpy(
            self.apply_sg_smoother(w_bias.cpu().numpy(), self._sg_window_bw, self._sg_order_bw)
        )

        g_const = torch.tensor(DEFAULT_GRAVITY_ACCEL_WORLD, dtype=torch.double)
        a_world_minus_g = torch.from_numpy(a_world).double() - g_const.unsqueeze(0)
        a_bias = imu_gt_t[:, 3:6] - (rot_gt.transpose(-1, -2) @ a_world_minus_g.unsqueeze(-1)).squeeze(-1)
        a_bias = torch.from_numpy(
            self.apply_sg_smoother(a_bias.cpu().numpy(), self._sg_window_ba, self._sg_order_ba)
        )

        imu_t = torch.from_numpy(imu_resampled[:, 1:]).double()
        ts_t = torch.from_numpy(ts).double()
        coeff = Interpolation.get_cubic_hermite_coeffs(ts_t, imu_t)
        imu_coeff = tuple(x.float() for x in coeff)
        valid_starts = self._valid_source_starts(t_gt, source_dt)
        source_derivative_key = self._source_derivative_cache_key()

        print(
            f"  supervision_source={VO_IMU_NONINTERP_SUPERVISION}, "
            f"world_frame={self._vo_supervision_world_frame}, "
            f"vo_poses={t_gt.shape[0]}, source_dt={source_dt:.6f}s, "
            f"window_span={(self._loss_window - 1) * source_dt:.6f}s, "
            f"valid_windows={valid_starts.shape[0]}, "
            f"gravity_accel_world={gravity_cache_key['gravity_accel_world_m_s2']}, "
            f"gravity_route={GRAVITY_ALIGNMENT_ROUTE}",
            flush=True,
        )
        pdump(
            {
                "t_imu": ts_t.float(),
                "u_imu": imu_t.float(),
                "w_bias": w_bias.float(),
                "a_bias": a_bias.float(),
                "t_bias": torch.from_numpy(t_gt).float(),
                "coeff": imu_coeff,
                "t_start": timestamp_start,
                "mask_missing": torch.zeros_like(ts_t).float(),
                "source_dt": source_dt,
                "source_valid_starts": torch.from_numpy(valid_starts).long(),
                "supervision_source": VO_IMU_NONINTERP_SUPERVISION,
                "vo_supervision_world_frame": self._vo_supervision_world_frame,
                "T_supervision_source": T_supervision_source.tolist(),
                "T_supervision_source_note": transform_note,
                "R_gravity_alignment": R_gravity_align.tolist(),
                "vo_pose_path": str(path_vo),
                "gravity_calibration": gravity_cache_key,
                "source_derivative_filter": source_derivative_key,
            },
            self.data_cache_dir,
            sequence + ".p",
        )
        pdump(
            {
                "t_gt": torch.from_numpy(t_gt).float(),
                "q_gt": q_gt.float(),
                "R_gt": rot_gt.float(),
                "v_gt": torch.from_numpy(v_gt).float(),
                "p_gt": torch.from_numpy(p_gt).float(),
                "supervision_source": VO_IMU_NONINTERP_SUPERVISION,
                "vo_supervision_world_frame": self._vo_supervision_world_frame,
                "gravity_calibration": gravity_cache_key,
                "source_derivative_filter": source_derivative_key,
            },
            self.data_cache_dir,
            sequence + "_gt.p",
        )

    def read_data(self, data_dir: str, recompute: bool = False):
        sequences = self._all_requested_sequences()
        required = []
        for seq in sequences:
            required.append(Path(self.data_cache_dir) / f"{seq}.p")
            required.append(Path(self.data_cache_dir) / f"{seq}_gt.p")
        if required and all(path.exists() for path in required) and self._cache_sources_match(sequences, data_dir) and not recompute:
            return

        print("Start read_data for LIASuw Debias_IMU tree")
        for sequence in sequences:
            print(f"\nSequence name: {sequence}")
            supervision_source = self._sequence_supervision_source(sequence)
            if supervision_source == MOCAP_NONINTERP_SUPERVISION:
                self._cache_mocap_noninterp_sequence(data_dir, sequence)
                continue
            if supervision_source == VO_IMU_NONINTERP_SUPERVISION:
                self._cache_vo_noninterp_sequence(data_dir, sequence)
                continue
            if supervision_source != MOCAP_200HZ_SUPERVISION:
                raise ValueError(f"Unsupported LIASuw supervision source: {supervision_source}")

            path_imu = Path(data_dir) / sequence / "mav0" / "imu0" / "data.csv"
            path_gt = Path(data_dir) / sequence / "mav0" / "mocap0" / "data.csv"
            if not path_imu.exists():
                raise FileNotFoundError(path_imu)
            if not path_gt.exists():
                raise FileNotFoundError(path_gt)

            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)
            if imu.ndim == 1:
                imu = imu.reshape(1, -1)
            if gt.ndim == 1:
                gt = gt.reshape(1, -1)

            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)
            timestamp_start = imu[idx0_imu, 0]
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, "right")
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, "right")
            imu = imu[idx0_imu:idx_end_imu]
            gt = gt[idx0_gt:idx_end_gt]

            seq_mode = self.find_sequence_mode(sequence)
            if seq_mode in ("train", "val"):
                ts = self.dt * np.arange(imu.shape[0])
                tmp = imu[:, 0] / 1e9
                tmp = tmp - tmp[0]
                imu_ori = imu.copy()
                imu = np.zeros_like(imu_ori)
                imu[:, 0] = ts
                for i in range(6):
                    imu[:, 1 + i] = np.interp(ts, tmp, imu_ori[:, 1 + i])
                t_gt = (gt[:, 0] - imu_ori[0, 0]) / 1e9
                gt = self.interpolate(gt, t_gt, ts)
            elif seq_mode == "test":
                t_start = imu[0, 0]
                ts = (imu[:, 0] - t_start) / 1e9
                t_gt = (gt[:, 0] - t_start) / 1e9
                gt = self.interpolate(gt, t_gt, ts)
            else:
                raise ValueError(f"Unsupported sequence mode: {seq_mode}")

            p_gt = gt[:, 1:4]
            R_gravity_align = self._gravity_alignment_matrix(sequence)
            p_gt = (R_gravity_align @ p_gt.T).T
            p_gt = p_gt - p_gt[0]
            r_gt = Rotation.from_matrix(R_gravity_align) * Rotation.from_quat(gt[:, [5, 6, 7, 4]])
            q_xyzw = r_gt.as_quat()
            q_gt_np = np.column_stack((q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]))
            q_gt = torch.from_numpy(q_gt_np).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            rot_gt = Lie.quat_to_SO3(q_gt).cpu()
            p_gt_t = torch.from_numpy(p_gt).double()
            v_gt = self.apply_sg_smoother(p_gt, 21, 5, deriv=1) / self.dt
            v_gt_t = torch.from_numpy(v_gt).double()
            imu_t = torch.from_numpy(imu[:, 1:]).double()
            ts_t = torch.from_numpy(ts).double()

            dt = (ts_t[1:] - ts_t[:-1]).unsqueeze(-1)
            w_bias = imu_t[:-1, :3] - Lie.SO3log(rot_gt[:-1].transpose(-1, -2) @ rot_gt[1:]) / dt
            w_bias = torch.cat((w_bias, w_bias[-1].unsqueeze(0)), dim=0)
            w_bias = torch.from_numpy(
                self.apply_sg_smoother(w_bias.cpu().numpy(), self._sg_window_bw, self._sg_order_bw)
            )

            gravity_cache_key = self._gravity_cache_key(sequence)
            print(
                f"  gravity_accel_world={gravity_cache_key['gravity_accel_world_m_s2']}, "
                f"gravity_route={GRAVITY_ALIGNMENT_ROUTE}",
                flush=True,
            )
            g_const = torch.tensor(DEFAULT_GRAVITY_ACCEL_WORLD, dtype=torch.double)
            a_world = (v_gt_t[1:] - v_gt_t[:-1]) / dt - g_const.unsqueeze(0)
            a_bias = imu_t[:-1, 3:6] - (rot_gt[:-1].transpose(-1, -2) @ a_world.unsqueeze(-1)).squeeze(-1)
            a_bias = torch.cat((a_bias, a_bias[-1].unsqueeze(0)), dim=0)
            a_bias = torch.from_numpy(
                self.apply_sg_smoother(a_bias.cpu().numpy(), self._sg_window_ba, self._sg_order_ba)
            )

            coeff = Interpolation.get_cubic_hermite_coeffs(ts_t, imu_t)
            imu_coeff = tuple(x.float() for x in coeff)

            tmp_idx = np.searchsorted(t_gt, ts)
            tmp_idx = np.clip(tmp_idx, 0, t_gt.shape[0] - 1)
            diff_t = ts - t_gt[tmp_idx]
            mask_missing = torch.zeros_like(ts_t)
            mask_missing[np.abs(diff_t) > 0.01] = 1
            quality_valid_starts, quality_cache_key = self._load_quality_valid_starts(data_dir, sequence)
            quality_payload = {}
            if quality_valid_starts is not None:
                max_start = int(ts_t.shape[0] - self._loss_window)
                quality_valid_starts = quality_valid_starts[
                    (quality_valid_starts >= 0) & (quality_valid_starts <= max_start)
                ]
                quality_payload = {
                    "quality_valid_starts": quality_valid_starts,
                    "quality_window_filter": quality_cache_key,
                }
                print(
                    f"  quality_window_filter={quality_cache_key['path']}, "
                    f"valid_windows={int(quality_valid_starts.numel())}",
                    flush=True,
                )

            pdump(
                {
                    "t_imu": ts_t.float(),
                    "u_imu": imu_t.float(),
                    "w_bias": w_bias.float(),
                    "a_bias": a_bias.float(),
                    "coeff": imu_coeff,
                    "t_start": timestamp_start,
                    "mask_missing": mask_missing.float(),
                    "supervision_source": MOCAP_200HZ_SUPERVISION,
                    "R_gravity_alignment": R_gravity_align.tolist(),
                    "gravity_calibration": gravity_cache_key,
                    **quality_payload,
                },
                self.data_cache_dir,
                sequence + ".p",
            )
            pdump(
                {
                    "t_gt": ts_t.float(),
                    "q_gt": q_gt.float(),
                    "R_gt": rot_gt.float(),
                    "v_gt": v_gt_t.float(),
                    "p_gt": p_gt_t.float(),
                    "supervision_source": MOCAP_200HZ_SUPERVISION,
                    "R_gravity_alignment": R_gravity_align.tolist(),
                    "gravity_calibration": gravity_cache_key,
                },
                self.data_cache_dir,
                sequence + "_gt.p",
            )

    def get_data_for_biasDy(self, i: int, device="cpu", mask_flag=False, batch_size: int | None = None):
        if self._trainval_supervision_timing != "source":
            imudict = self.load_imu(i)
            if imudict.get("quality_valid_starts") is None:
                return super().get_data_for_biasDy(i, device=device, mask_flag=mask_flag, batch_size=batch_size)
            return self._get_data_for_imu_quality_windows(
                i,
                imudict,
                device=device,
                batch_size=batch_size,
            )

        batch_size = self._batch_size if batch_size is None else batch_size
        imudict = self.load_imu(i)
        gtdict = self.load_gt(i)
        coeff = imudict["coeff"]
        spline_time = imudict["t_imu"]
        bias_w = imudict["w_bias"]
        bias_a = imudict["a_bias"]
        valid_starts = imudict.get("source_valid_starts")
        if valid_starts is None:
            raise ValueError("Source-timing supervision cache is missing source_valid_starts.")

        n_gt = gtdict["R_gt"].shape[0]
        index_train_max = round(n_gt * (1 - self._percent_for_val))
        if self._mode == "train":
            candidates = valid_starts[valid_starts <= index_train_max - self._loss_window]
            if candidates.numel() == 0:
                raise ValueError("No valid VO source windows are available for training split.")
            choice = torch.randint(0, candidates.numel(), (batch_size,))
            s = candidates[choice]
        elif self._mode == "val":
            candidates = valid_starts[valid_starts >= index_train_max]
            if candidates.numel() == 0:
                raise ValueError("No valid VO source windows are available for validation split.")
            s = candidates[:: self._loss_window]
        else:
            raise ValueError(
                "get_data_for_biasDy is only for training or validation; use get_full_trajectory for full sequences."
            )

        R_gt_batch = torch.stack([gtdict["R_gt"][s + offset] for offset in range(self._loss_window)], dim=0)
        v_gt_batch = torch.stack([gtdict["v_gt"][s + offset] for offset in range(self._loss_window)], dim=0)
        p_gt_batch = torch.stack([gtdict["p_gt"][s + offset] for offset in range(self._loss_window)], dim=0)
        X_gt_batch = Lie.SEn3fromSO3Rn(R_gt_batch, torch.cat((v_gt_batch, p_gt_batch), dim=-1))
        t_gt_batch = torch.stack([gtdict["t_gt"][s + offset] for offset in range(self._loss_window)], dim=0)
        bw_gt_batch = torch.stack([bias_w[s + offset] for offset in range(self._loss_window)], dim=0)
        ba_gt_batch = torch.stack([bias_a[s + offset] for offset in range(self._loss_window)], dim=0)

        t_eval = torch.median(t_gt_batch - t_gt_batch[0:1], dim=1).values
        n_steps = torch.ceil(t_eval[-1] / self.dt).long().item()
        t_odeint = torch.arange(0, n_steps + 1, dtype=t_eval.dtype) * self.dt
        t_odeint[-1] = t_eval[-1]
        eval_indices = torch.searchsorted(t_odeint, t_eval)
        t_odeint[eval_indices] = t_eval

        a, b, c, d = coeff
        coeff = (a.to(device), b.to(device), c.to(device), d.to(device))
        bwa_gt_batch = torch.cat((bw_gt_batch, ba_gt_batch), dim=-1)
        return (
            t_gt_batch.to(device),
            X_gt_batch.to(device),
            bwa_gt_batch.to(device),
        ), t_odeint.to(device), coeff, spline_time.to(device), eval_indices.to(device)

    def _get_data_for_imu_quality_windows(
        self,
        i: int,
        imudict: dict,
        device: str = "cpu",
        batch_size: int | None = None,
    ):
        batch_size = self._batch_size if batch_size is None else batch_size
        gtdict = self.load_gt(i)
        coeff = imudict["coeff"]
        spline_time = imudict["t_imu"]
        bias_w = imudict["w_bias"]
        bias_a = imudict["a_bias"]
        valid_starts = imudict["quality_valid_starts"].long()
        n_max = imudict["u_imu"].shape[0]
        index_train_max = round(n_max * (1 - self._percent_for_val))
        if self._mode == "train":
            candidates = valid_starts[valid_starts <= index_train_max - self._loss_window]
            if candidates.numel() == 0:
                raise ValueError(f"No valid quality-filtered windows are available for training sequence {self.sequences[i]}.")
            choice = torch.randint(0, candidates.numel(), (batch_size,))
            s = candidates[choice]
        elif self._mode == "val":
            candidates = valid_starts[(valid_starts >= index_train_max) & (valid_starts <= n_max - self._loss_window)]
            if candidates.numel() == 0:
                raise ValueError(f"No valid quality-filtered windows are available for validation sequence {self.sequences[i]}.")
            s = candidates[:: self._loss_window]
        else:
            raise ValueError(
                "get_data_for_biasDy is only for training or validation; use get_full_trajectory for full sequences."
            )

        R_gt_batch = torch.stack([gtdict["R_gt"][s + offset] for offset in range(self._loss_window)], dim=0)
        v_gt_batch = torch.stack([gtdict["v_gt"][s + offset] for offset in range(self._loss_window)], dim=0)
        p_gt_batch = torch.stack([gtdict["p_gt"][s + offset] for offset in range(self._loss_window)], dim=0)
        X_gt_batch = Lie.SEn3fromSO3Rn(R_gt_batch, torch.cat((v_gt_batch, p_gt_batch), dim=-1))
        t_gt_batch = torch.stack([gtdict["t_gt"][s + offset] for offset in range(self._loss_window)], dim=0)
        bw_gt_batch = torch.stack([bias_w[s + offset] for offset in range(self._loss_window)], dim=0)
        ba_gt_batch = torch.stack([bias_a[s + offset] for offset in range(self._loss_window)], dim=0)
        t_odeint = torch.arange(0, self._loss_window, dtype=t_gt_batch.dtype) * self.dt

        a, b, c, d = coeff
        coeff = (a.to(device), b.to(device), c.to(device), d.to(device))
        bwa_gt_batch = torch.cat((bw_gt_batch, ba_gt_batch), dim=-1)
        return (
            t_gt_batch.to(device),
            X_gt_batch.to(device),
            bwa_gt_batch.to(device),
        ), t_odeint.to(device), coeff, spline_time.to(device), s.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LIASuw Debias_IMU AD pipeline")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "LIASuw_training_candidates_200hz",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "LIASuw_training_candidates_200hz_ad",
    )
    parser.add_argument(
        "--pseudo-gt-root",
        type=Path,
        default=PLATO_ROOT / "results" / "pseudo_gt" / "LIASuw",
        help="Root containing per-sequence Debias_IMU body-frame VO pseudo-GT exports.",
    )
    parser.add_argument(
        "--outputdir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "LIASuw_AD_pipeline",
    )
    parser.add_argument(
        "--trainval-supervision-source",
        choices=[MOCAP_200HZ_SUPERVISION, MOCAP_NONINTERP_SUPERVISION, VO_IMU_NONINTERP_SUPERVISION],
        default=MOCAP_200HZ_SUPERVISION,
        help="Use dense mocap/body GT, sparse non-interpolated mocap/body GT, or non-interpolated stereo VO poses for train/val supervision.",
    )
    parser.add_argument(
        "--vo-supervision-world-frame",
        choices=["gt", "vo"],
        default="gt",
        help=(
            "World frame used when --trainval-supervision-source=vo_imu_noninterp. "
            "'gt' applies each sequence's T_WO/T_Wm_Wv from calibration_result.yaml so the IMU/body "
            "pose is supervised in mocap/GT world with Debias_IMU's default gravity. "
            "'vo' preserves the old raw VO world behavior and requires a matching gravity transform."
        ),
    )
    parser.add_argument(
        "--vo-gap-tolerance-s",
        type=float,
        default=0.015,
        help="Reject source-timing VO loss windows containing pose-to-pose gaps larger than nominal dt plus this tolerance.",
    )
    parser.add_argument(
        "--source-position-sg-window",
        type=int,
        default=0,
        help=(
            "For source-timing mocap/VO supervision, optionally derive v/a labels from a Savitzky-Golay "
            "filter on position instead of raw finite differences. 0 keeps the legacy gradient path."
        ),
    )
    parser.add_argument(
        "--source-position-sg-order",
        type=int,
        default=3,
        help="Polynomial order for --source-position-sg-window.",
    )
    parser.add_argument(
        "--gravity-calibration-json",
        type=Path,
        default=None,
        help=(
            "Optional Route-A gravity calibration JSON. When provided, each sequence's pseudo accel bias "
            "uses its calibrated gravity acceleration vector in the supervision world frame."
        ),
    )
    parser.add_argument(
        "--gravity-accel-world",
        type=float,
        nargs=3,
        default=DEFAULT_GRAVITY_ACCEL_WORLD,
        metavar=("GX", "GY", "GZ"),
        help="Fallback gravity acceleration vector in the supervision world frame. Default preserves old behavior.",
    )
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--val-seqs", nargs="+", default=None)
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--integral_method", type=str, default="euler")
    parser.add_argument(
        "--network_type_Gyro",
        type=str,
        default="bw_func_net",
        help=(
            f"Gyro bias dynamics network. Use {GYRO_M2_NAME} for the PLATO M2 no-derivative "
            f"ablation, {GYRO_PHYS_EQ_NAME} for the physics-informed equilibrium ablation, "
            f"{GYRO_PHYS_RATE_NAME} for the bounded physics-informed rate ablation, "
            f"{GYRO_PHYS_DELTA_NAME} for the no-derivative finite-difference ablation, "
            f"{GYRO_FBU_RESIDUAL_NAME} for the pure f(b,u) residual MLP, "
            f"{GYRO_SLOW_RESIDUAL_NAME} for the slow rate-limited residual ablation, "
            f"{GYRO_CONSERVATIVE_RESIDUAL_NAME} for the shared conservative residual model, "
            f"or {GYRO_BASELINE_NO_UDOT_NAME} for the original-style network without u_dot."
        ),
    )
    parser.add_argument(
        "--network_type_Acc",
        type=str,
        default="ba_func_net",
        help=(
            f"Accel bias dynamics network. Use {ACC_M2_NAME} for the PLATO M2 no-derivative "
            f"ablation, {ACC_PHYS_EQ_NAME} for the physics-informed equilibrium ablation, "
            f"{ACC_PHYS_RATE_NAME} for the bounded physics-informed rate ablation, "
            f"{ACC_PHYS_DELTA_NAME} for the no-derivative finite-difference ablation, "
            f"{ACC_FBU_RESIDUAL_NAME} for the pure f(b,u) residual MLP, "
            f"{ACC_SLOW_RESIDUAL_NAME} for the slow rate-limited residual ablation, "
            f"{ACC_CONSERVATIVE_RESIDUAL_NAME} for the shared conservative residual model, "
            f"or {ACC_BASELINE_NO_UDOT_NAME} for the original-style network without u_dot."
        ),
    )
    parser.add_argument("--loss_window", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr_Gyro", type=float, default=5e-3)
    parser.add_argument("--weight_decay_Gyro", type=float, default=1e-6)
    parser.add_argument("--gyro_lr_step_size", type=int, default=300)
    parser.add_argument("--gyro_lr_gamma", type=float, default=0.5)
    parser.add_argument("--lr_Acc", type=float, default=5e-3)
    parser.add_argument("--weight_decay_Acc", type=float, default=1e-6)
    parser.add_argument("--acc_lr_step_size", type=int, default=300)
    parser.add_argument("--acc_lr_gamma", type=float, default=0.5)
    parser.add_argument("--acc-loss-rot-weight", type=float, default=1.0)
    parser.add_argument("--acc-loss-vel-weight", type=float, default=1.0)
    parser.add_argument("--acc-loss-pos-weight", type=float, default=1.0)
    parser.add_argument(
        "--acc-loss-type",
        choices=["se23", "nll", "interval_nll", "trajectory_nll"],
        default="se23",
        help=(
            "Accelerometer-stage supervision loss. 'nll' is the per-state/endpoint SE_2(3) "
            "trajectory likelihood; 'interval_nll' is an interval preintegration-factor likelihood; "
            "'trajectory_nll' is the stacked trajectory-state likelihood with cross-time covariance."
        ),
    )
    parser.add_argument(
        "--nll-covariance-strategy",
        choices=["fixed", "optimize_q", "q_only"],
        default="fixed",
        help=(
            "For --acc-loss-type=nll: keep process covariance fixed, alternate network/Q optimization, "
            "or freeze the network and optimize only the IMU noise covariance."
        ),
    )
    parser.add_argument("--nll-optimize-q", "--nll_optimize_q", dest="nll_optimize_q", action="store_true")
    parser.add_argument("--nll-sigma-a", "--nll_sigma_a", dest="nll_sigma_a", type=float, default=2.0e-3)
    parser.add_argument("--nll-sigma-g", "--nll_sigma_g", dest="nll_sigma_g", type=float, default=1.6968e-4)
    parser.add_argument("--nll-w-rot", "--nll_w_rot", dest="nll_w_rot", type=float, default=1.0e-3)
    parser.add_argument("--nll-w-vel", "--nll_w_vel", dest="nll_w_vel", type=float, default=2.0e-2)
    parser.add_argument("--nll-w-pos", "--nll_w_pos", dest="nll_w_pos", type=float, default=1.0e-2)
    parser.add_argument("--nll-q-block-epochs", "--nll_q_block_epochs", dest="nll_q_block_epochs", type=int, default=100)
    parser.add_argument("--nll-lr-q", "--nll_lr_q", dest="nll_lr_q", type=float, default=1.0e-2)
    parser.add_argument("--nll-sigma-a-min", "--nll_sigma_a_min", dest="nll_sigma_a_min", type=float, default=1.0e-4)
    parser.add_argument("--nll-sigma-a-max", "--nll_sigma_a_max", dest="nll_sigma_a_max", type=float, default=5.0e-2)
    parser.add_argument("--nll-sigma-g-min", "--nll_sigma_g_min", dest="nll_sigma_g_min", type=float, default=1.0e-5)
    parser.add_argument("--nll-sigma-g-max", "--nll_sigma_g_max", dest="nll_sigma_g_max", type=float, default=5.0e-3)
    parser.add_argument(
        "--nll-supervision-mode",
        "--nll_supervision_mode",
        dest="nll_supervision_mode",
        choices=["per_state", "endpoint"],
        default="per_state",
        help=(
            "For --acc-loss-type=nll: 'per_state' applies NLL independently to every supervised state; "
            "'endpoint' applies the likelihood only to the final supervised state using the covariance "
            "propagated from the window start."
        ),
    )
    parser.add_argument(
        "--nll-train-gyro",
        "--nll_train_gyro",
        dest="nll_train_gyro",
        action="store_true",
        help=(
            "For --acc-loss-type=nll, interval_nll, or trajectory_nll: jointly fine-tune the loaded gyro bias dynamics "
            "with the accel bias dynamics. Default keeps the previous fixed-gyro behavior."
        ),
    )
    parser.add_argument(
        "--nll-mean-anchor-weight",
        "--nll_mean_anchor_weight",
        dest="nll_mean_anchor_weight",
        type=float,
        default=0.0,
        help="Weight for an optional SE23 anchor to the initialized mean model during NLL fine-tuning.",
    )
    parser.add_argument(
        "--nll-mean-anchor-mode",
        "--nll_mean_anchor_mode",
        dest="nll_mean_anchor_mode",
        choices=["per_state", "endpoint"],
        default="endpoint",
        help="States used by the optional initialized-mean anchor during NLL fine-tuning.",
    )
    parser.add_argument(
        "--interval-nll-mode",
        "--interval_nll_mode",
        dest="interval_nll_mode",
        choices=["mse_only", "nll_cov_only", "nll_joint"],
        default="nll_cov_only",
        help=(
            "For --acc-loss-type=interval_nll: mse_only delegates to the existing SE23 MSE Acc trainer; "
            "nll_cov_only freezes NODE/theta and learns only scalar IMU noise variances; "
            "nll_joint trains theta plus scalar IMU noise variances."
        ),
    )
    parser.add_argument("--interval-sigma-g-nominal", "--interval_sigma_g_nominal", dest="interval_sigma_g_nominal", type=float, default=1.6968e-4)
    parser.add_argument("--interval-sigma-a-nominal", "--interval_sigma_a_nominal", dest="interval_sigma_a_nominal", type=float, default=2.0e-3)
    parser.add_argument("--interval-sigma-g-floor2", "--interval_sigma_g_floor2", dest="interval_sigma_g_floor2", type=float, default=1.0e-12)
    parser.add_argument("--interval-sigma-a-floor2", "--interval_sigma_a_floor2", dest="interval_sigma_a_floor2", type=float, default=1.0e-10)
    parser.add_argument("--interval-target-cov-floor", "--interval_target_cov_floor", dest="interval_target_cov_floor", type=float, default=1.0e-9)
    parser.add_argument("--interval-jitter", "--interval_jitter", dest="interval_jitter", type=float, default=1.0e-8)
    parser.add_argument("--interval-prior-lambda-g", "--interval_prior_lambda_g", dest="interval_prior_lambda_g", type=float, default=0.0)
    parser.add_argument("--interval-prior-lambda-a", "--interval_prior_lambda_a", dest="interval_prior_lambda_a", type=float, default=0.0)
    parser.add_argument("--interval-lr-sigma", "--interval_lr_sigma", dest="interval_lr_sigma", type=float, default=1.0e-2)
    parser.add_argument(
        "--interval-detach-covariance-theta",
        "--interval_detach_covariance_theta",
        dest="interval_detach_covariance_theta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach Phi/G covariance propagation from theta while keeping interval residual gradients to theta.",
    )
    parser.add_argument(
        "--interval-covariance-gradient-mode",
        "--interval_covariance_gradient_mode",
        dest="interval_covariance_gradient_mode",
        choices=["analytic_sensitivity", "ad"],
        default="analytic_sensitivity",
        help=(
            "For interval NLL: analytic_sensitivity propagates unit D_g/D_a sensitivities and builds "
            "S=alpha_g*D_g+alpha_a*D_a; ad propagates the alpha-scaled covariance recurrence directly."
        ),
    )
    parser.add_argument(
        "--interval-debug-covariance-gradients",
        "--interval_debug_covariance_gradients",
        dest="interval_debug_covariance_gradients",
        action="store_true",
        help="Print closed-form scalar NLL gradients for alpha_g/alpha_a and raw variance parameters during validation.",
    )
    parser.add_argument(
        "--interval-noise-init-model-path",
        "--interval_noise_init_model_path",
        dest="interval_noise_init_model_path",
        type=Path,
        default=None,
        help="Optional checkpoint containing raw_sigma_g2/raw_sigma_a2 for interval NLL stage initialization.",
    )
    parser.add_argument(
        "--trajectory-supervision-mode",
        "--trajectory_supervision_mode",
        dest="trajectory_supervision_mode",
        choices=["sparse", "endpoint"],
        default="sparse",
        help="For trajectory NLL: select sparse intermediate states or the endpoint-only baseline.",
    )
    parser.add_argument(
        "--trajectory-supervision-stride",
        "--trajectory_supervision_stride",
        dest="trajectory_supervision_stride",
        type=int,
        default=1,
        help="For trajectory NLL sparse mode: use every K supervised states after the initial state.",
    )
    parser.add_argument(
        "--trajectory-max-supervised-states",
        "--trajectory_max_supervised_states",
        dest="trajectory_max_supervised_states",
        type=int,
        default=0,
        help="For trajectory NLL sparse mode: optional cap on supervised states per window; 0 means no cap.",
    )
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--train_batch_strategy", choices=["random_sequence", "balanced_sequences"], default="random_sequence")
    parser.add_argument("--percent_for_val", type=float, default=0.2)
    parser.add_argument("--sg_window_bw", type=int, default=15)
    parser.add_argument("--sg_order_bw", type=int, default=3)
    parser.add_argument("--sg_window_ba", type=int, default=15)
    parser.add_argument("--sg_order_ba", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-device", type=str, default="cpu")
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--test-recompute", action="store_true")
    parser.add_argument("--only-test", action="store_true")
    parser.add_argument(
        "--stop-after-gyro",
        action="store_true",
        help="Train gyro bias dynamics and exit before Acc training/testing. Useful for network screening.",
    )
    parser.add_argument(
        "--skip-gyro-train",
        action="store_true",
        help="Skip gyro training and load --bw_model_path or the resolved gyro checkpoint before Acc training.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train requested stages but do not run full-sequence testing/plot summaries.",
    )
    parser.add_argument("--bw_model_path", type=Path, default=None)
    parser.add_argument("--ba_model_path", type=Path, default=None)
    parser.add_argument(
        "--ba-init-model-path",
        "--ba_init_model_path",
        dest="ba_init_model_path",
        type=Path,
        default=None,
        help="Optional Acc checkpoint used to initialize NLL/SE23 Acc training before optimization.",
    )
    parser.add_argument("--use-best-val", action="store_true")
    parser.add_argument(
        "--initial-bias-mode",
        choices=["first", "mean_window", "median_window"],
        default="first",
        help="Initial test bias policy. Default matches Debias_IMU: use the first bias label.",
    )
    parser.add_argument(
        "--initial-bias-window-s",
        type=float,
        default=1.0,
        help="Window length used by mean_window/median_window initial bias policies.",
    )
    parser.add_argument(
        "--train-initial-bw-mode",
        choices=["label", "zero"],
        default="label",
        help="Initial gyro-bias state used inside training/validation ODE windows.",
    )
    parser.add_argument(
        "--train-initial-ba-mode",
        choices=["label", "zero"],
        default="label",
        help="Initial accel-bias state used inside training/validation ODE windows.",
    )
    parser.add_argument(
        "--readable-val-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print validation RMSE metrics alongside the original scaled Debias_IMU losses.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def make_dataset_parameters(args: argparse.Namespace) -> dict:
    val_seqs = list(args.val_seqs) if args.val_seqs is not None else list(args.train_seqs)
    return {
        "dataset_name": "TUM",
        "data_dir": str(args.data_root),
        "train_seqs": list(args.train_seqs),
        "val_seqs": val_seqs,
        "test_seqs": list(args.test_seqs),
        "percent_for_val": args.percent_for_val,
        "batch_size": args.batch_size,
        "loss_window": args.loss_window,
        "dt": args.dt,
        "sg_window_bw": args.sg_window_bw,
        "sg_order_bw": args.sg_order_bw,
        "sg_window_ba": args.sg_window_ba,
        "sg_order_ba": args.sg_order_ba,
        "trainval_supervision_source": args.trainval_supervision_source,
        "pseudo_gt_root": args.pseudo_gt_root,
        "vo_supervision_world_frame": args.vo_supervision_world_frame,
        "vo_gap_tolerance_s": args.vo_gap_tolerance_s,
        "gravity_calibration_json": args.gravity_calibration_json,
        "gravity_accel_world": tuple(args.gravity_accel_world),
        "source_position_sg_window": args.source_position_sg_window,
        "source_position_sg_order": args.source_position_sg_order,
    }


def resolve_checkpoint(
    outputdir: Path,
    stage_dir: str,
    explicit_path: Path | None,
    epoch: int,
    save_freq: int,
    use_best_val: bool,
) -> Path:
    if explicit_path is not None:
        return explicit_path
    if use_best_val:
        return outputdir / stage_dir / "best_val_model.pt"
    last_saved_epoch = max(((epoch - 1) // save_freq) * save_freq, 0)
    return outputdir / stage_dir / f"model_epoch_{last_saved_epoch}.pt"


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={args.device} was requested, but CUDA is not available.")
    if args.test_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"test-device={args.test_device} was requested, but CUDA is not available.")
    if not args.only_test and args.epoch <= 0:
        raise ValueError("--epoch must be positive unless --only-test is used")
    if args.stop_after_gyro and args.skip_gyro_train:
        raise ValueError("--stop-after-gyro cannot be combined with --skip-gyro-train")
    if args.only_test and (args.stop_after_gyro or args.skip_gyro_train or args.skip_test):
        raise ValueError("--only-test cannot be combined with train-stage control flags")
    if args.acc_loss_type == "nll" and args.nll_q_block_epochs <= 0:
        raise ValueError("--nll-q-block-epochs must be positive")
    if args.acc_loss_type == "nll" and args.nll_supervision_mode != "per_state" and not args.readable_val_metrics:
        raise ValueError("NLL endpoint supervision is implemented in the PLATO readable-metrics trainer.")
    if args.acc_loss_type in {"interval_nll", "trajectory_nll"} and not args.readable_val_metrics:
        raise ValueError("PLATO preintegration NLL variants are implemented in the readable-metrics trainer.")
    if args.acc_loss_type in {"interval_nll", "trajectory_nll"} and args.interval_nll_mode != "nll_joint" and args.nll_train_gyro:
        raise ValueError("--nll-train-gyro is only meaningful with --interval-nll-mode=nll_joint.")
    if args.acc_loss_type == "trajectory_nll" and args.interval_covariance_gradient_mode != "analytic_sensitivity":
        raise ValueError("--acc-loss-type=trajectory_nll currently requires --interval-covariance-gradient-mode=analytic_sensitivity.")
    if args.ba_init_model_path is not None and not args.readable_val_metrics:
        raise ValueError("--ba-init-model-path requires the PLATO readable-metrics trainer.")
    use_delta_imu = _uses_delta_imu(args.network_type_Gyro, args.network_type_Acc)
    if use_delta_imu and not args.readable_val_metrics:
        raise ValueError("Finite-difference IMU ablations require --readable-val-metrics so train/test use the same aux input.")
    if _uses_delta_imu(args.network_type_Acc) and _requires_spline_derivative(args.network_type_Gyro):
        raise ValueError(
            "Accel finite-difference ablation cannot be combined with a gyro model that requires spline derivatives."
        )

    args.outputdir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    params = make_dataset_parameters(args)

    print("LIASuw Debias_IMU AD pipeline")
    print(f"train_seqs={args.train_seqs}")
    print(f"val_seqs={args.val_seqs if args.val_seqs is not None else args.train_seqs}")
    print(f"test_seqs={args.test_seqs}")
    print(f"trainval_supervision_source={args.trainval_supervision_source}")
    print(f"gravity_accel_world_default={tuple(args.gravity_accel_world)}")
    print(
        f"source_position_sg_window={args.source_position_sg_window}, "
        f"source_position_sg_order={args.source_position_sg_order}"
    )
    if args.gravity_calibration_json is not None:
        print(f"gravity_calibration_json={args.gravity_calibration_json}")
    ode_grid_window_s = (args.loss_window - 1) * args.dt
    if args.trainval_supervision_source == VO_IMU_NONINTERP_SUPERVISION:
        print(f"pseudo_gt_root={args.pseudo_gt_root}")
        print(f"vo_supervision_world_frame={args.vo_supervision_world_frame}")
        print(
            f"source_timing_window={args.loss_window} VO poses; physical span is read from VO timestamps, "
            f"not assumed from IMU dt. ode_grid_window_s={ode_grid_window_s:.6g} from dt={args.dt}"
        )
    elif args.trainval_supervision_source == MOCAP_NONINTERP_SUPERVISION:
        print(
            f"source_timing_window={args.loss_window} mocap poses; physical span is read from mocap timestamps, "
            f"not assumed from IMU dt. ode_grid_window_s={ode_grid_window_s:.6g} from dt={args.dt}"
        )
    print(f"network_type_Gyro={args.network_type_Gyro}, network_type_Acc={args.network_type_Acc}")
    print(f"device={args.device}, test_device={args.test_device}, epoch={args.epoch}, loss_window={args.loss_window}")
    if args.trainval_supervision_source == MOCAP_200HZ_SUPERVISION:
        print(f"nominal_imu_window_s={ode_grid_window_s:.6g} from loss_window={args.loss_window}, dt={args.dt}")
    print(f"acc_loss_type={args.acc_loss_type}")
    print(
        f"train_initial_bw_mode={args.train_initial_bw_mode}, "
        f"train_initial_ba_mode={args.train_initial_ba_mode}"
    )
    if args.acc_loss_type == "nll":
        print(
            "nll_covariance "
            f"strategy={args.nll_covariance_strategy}, optimize_q={_nll_optimize_q(args)}, "
            f"q_only={_nll_q_only(args)}, train_gyro={args.nll_train_gyro}, mode={args.nll_supervision_mode}, "
            f"sigma_a={args.nll_sigma_a}, sigma_g={args.nll_sigma_g}, "
            f"w_rot={args.nll_w_rot}, w_vel={args.nll_w_vel}, w_pos={args.nll_w_pos}, "
            f"mean_anchor_weight={args.nll_mean_anchor_weight}, mean_anchor_mode={args.nll_mean_anchor_mode}, "
            f"sigma_a_bounds=({args.nll_sigma_a_min}, {args.nll_sigma_a_max}), "
            f"sigma_g_bounds=({args.nll_sigma_g_min}, {args.nll_sigma_g_max})"
        )
    if args.acc_loss_type in {"interval_nll", "trajectory_nll"}:
        print(
            f"{args.acc_loss_type} "
            f"mode={args.interval_nll_mode}, train_gyro={args.nll_train_gyro}, "
            f"sigma_a_nominal={args.interval_sigma_a_nominal}, sigma_g_nominal={args.interval_sigma_g_nominal}, "
            f"sigma_a_floor2={args.interval_sigma_a_floor2}, sigma_g_floor2={args.interval_sigma_g_floor2}, "
            f"target_cov_floor={args.interval_target_cov_floor}, jitter={args.interval_jitter}, "
            f"prior_lambda_a={args.interval_prior_lambda_a}, prior_lambda_g={args.interval_prior_lambda_g}, "
            f"lr_sigma={args.interval_lr_sigma}, detach_covariance_theta={args.interval_detach_covariance_theta}, "
            f"covariance_gradient_mode={args.interval_covariance_gradient_mode}, "
            f"debug_covariance_gradients={args.interval_debug_covariance_gradients}, "
            f"noise_init={args.interval_noise_init_model_path}, "
            f"trajectory_mode={args.trajectory_supervision_mode}, "
            f"trajectory_stride={args.trajectory_supervision_stride}, "
            f"trajectory_max_states={args.trajectory_max_supervised_states}"
        )
    if args.stop_after_gyro or args.skip_gyro_train or args.skip_test:
        print(
            f"stage_control stop_after_gyro={args.stop_after_gyro}, "
            f"skip_gyro_train={args.skip_gyro_train}, skip_test={args.skip_test}"
        )

    gyro_ckpt = resolve_checkpoint(
        args.outputdir,
        "Gyro_weights",
        args.bw_model_path,
        args.epoch,
        args.save_freq,
        args.use_best_val,
    )
    acc_ckpt = resolve_checkpoint(
        args.outputdir,
        "Acc_weights",
        args.ba_model_path,
        args.epoch,
        args.save_freq,
        args.use_best_val,
    )

    if not args.only_test:
        dataset_train = LIASuwDebiasDataset(
            **params,
            data_cache_dir=str(args.cache_dir),
            mode="train",
            recompute=args.recompute_cache,
        )
        dataset_val = LIASuwDebiasDataset(
            **params,
            data_cache_dir=str(args.cache_dir),
            mode="val",
            recompute=False,
        )
        t_start = time.time()
        gyro_train_fn = Gyro_train_with_readable_metrics if args.readable_val_metrics else Gyro_train
        if args.acc_loss_type == "nll":
            acc_train_fn = Acc_train_SE23_NLL_with_readable_metrics if args.readable_val_metrics else debias_learning.Acc_train_SE23_NLL
        elif args.acc_loss_type in {"interval_nll", "trajectory_nll"}:
            acc_train_fn = Acc_train_interval_preintegration_NLL_with_readable_metrics
        else:
            acc_train_fn = Acc_train_SE23_with_readable_metrics if args.readable_val_metrics else Acc_train_SE23
        gyro_profile = None
        acc_profile = None
        if not args.skip_gyro_train:
            gyro_profile = gyro_train_fn(
                dataset_train,
                dataset_val,
                str(args.outputdir),
                args.network_type_Gyro,
                args.integral_method,
                args.device,
                args.lr_Gyro,
                args.weight_decay_Gyro,
                epoch=args.epoch,
                lr_step_size=args.gyro_lr_step_size,
                lr_gamma=args.gyro_lr_gamma,
                train_batch_strategy=args.train_batch_strategy,
                val_freq=args.val_freq,
                save_freq=args.save_freq,
                **(
                    {
                        "train_initial_bw_mode": args.train_initial_bw_mode,
                        "train_initial_ba_mode": args.train_initial_ba_mode,
                    }
                    if args.readable_val_metrics
                    else {}
                ),
            )

        if args.stop_after_gyro:
            elapsed = time.time() - t_start
            with open(args.outputdir / "time_memory_record.txt", "w", encoding="utf-8") as f:
                f.write(
                    format_training_profile(
                        elapsed,
                        gyro_profile,
                        None,
                        extra={
                            "dataset": "LIASuw",
                            "trainval_supervision_source": args.trainval_supervision_source,
                            "pseudo_gt_root": str(args.pseudo_gt_root),
                            "vo_supervision_world_frame": args.vo_supervision_world_frame,
                            "vo_gap_tolerance_s": args.vo_gap_tolerance_s,
                            "source_position_sg_window": args.source_position_sg_window,
                            "source_position_sg_order": args.source_position_sg_order,
                            "train_seqs": list(args.train_seqs),
                            "test_seqs": list(args.test_seqs),
                            "loss_window": args.loss_window,
                            "ode_grid_window_s": (args.loss_window - 1) * args.dt,
                            "nominal_imu_window_s": (args.loss_window - 1) * args.dt
                            if args.trainval_supervision_source == MOCAP_200HZ_SUPERVISION
                            else None,
                            "source_timing_window_note": (
                                "loss_window counts source poses; physical span is read from source timestamps"
                                if args.trainval_supervision_source
                                in {MOCAP_NONINTERP_SUPERVISION, VO_IMU_NONINTERP_SUPERVISION}
                                else None
                            ),
                            "batch_size": args.batch_size,
                            "network_type_Gyro": args.network_type_Gyro,
                            "train_initial_bw_mode": args.train_initial_bw_mode,
                            "train_initial_ba_mode": args.train_initial_ba_mode,
                            "stage_control": "stop_after_gyro",
                            "gyro_checkpoint": str(gyro_ckpt),
                        },
                    )
                )
            return

        bw_func = bw_func_choose(args.network_type_Gyro, device=args.device)
        bw_func.load_state_dict(torch.load(gyro_ckpt, map_location=args.device, weights_only=True)["func_bw_model"])
        acc_kwargs = {}
        if args.acc_loss_type == "nll":
            acc_kwargs.update(
                {
                    "nll_sigma_a": args.nll_sigma_a,
                    "nll_sigma_g": args.nll_sigma_g,
                    "nll_w_rot": args.nll_w_rot,
                    "nll_w_vel": args.nll_w_vel,
                    "nll_w_pos": args.nll_w_pos,
                    "nll_optimize_q": _nll_optimize_q(args),
                    "nll_q_block_epochs": args.nll_q_block_epochs,
                    "nll_lr_q": args.nll_lr_q,
                    "nll_sigma_a_min": args.nll_sigma_a_min,
                    "nll_sigma_a_max": args.nll_sigma_a_max,
                    "nll_sigma_g_min": args.nll_sigma_g_min,
                    "nll_sigma_g_max": args.nll_sigma_g_max,
                    "nll_supervision_mode": args.nll_supervision_mode,
                    "ba_init_model_path": args.ba_init_model_path,
                    "nll_q_only": _nll_q_only(args),
                    "nll_train_gyro": args.nll_train_gyro,
                    "nll_mean_anchor_weight": args.nll_mean_anchor_weight,
                    "nll_mean_anchor_mode": args.nll_mean_anchor_mode,
                }
            )
        elif args.acc_loss_type in {"interval_nll", "trajectory_nll"}:
            acc_kwargs.update(
                {
                    "interval_nll_mode": args.interval_nll_mode,
                    "interval_sigma_g_nominal": args.interval_sigma_g_nominal,
                    "interval_sigma_a_nominal": args.interval_sigma_a_nominal,
                    "interval_sigma_g_floor2": args.interval_sigma_g_floor2,
                    "interval_sigma_a_floor2": args.interval_sigma_a_floor2,
                    "interval_target_cov_floor": args.interval_target_cov_floor,
                    "interval_jitter": args.interval_jitter,
                    "interval_prior_lambda_g": args.interval_prior_lambda_g,
                    "interval_prior_lambda_a": args.interval_prior_lambda_a,
                    "interval_lr_sigma": args.interval_lr_sigma,
                    "interval_detach_covariance_theta": args.interval_detach_covariance_theta,
                    "interval_covariance_gradient_mode": args.interval_covariance_gradient_mode,
                    "interval_debug_covariance_gradients": args.interval_debug_covariance_gradients,
                    "interval_noise_init_model_path": args.interval_noise_init_model_path,
                    "nll_train_gyro": args.nll_train_gyro,
                    "ba_init_model_path": args.ba_init_model_path,
                    "nll_family": "trajectory_state" if args.acc_loss_type == "trajectory_nll" else "interval",
                    "trajectory_supervision_mode": args.trajectory_supervision_mode,
                    "trajectory_supervision_stride": args.trajectory_supervision_stride,
                    "trajectory_max_supervised_states": args.trajectory_max_supervised_states,
                }
            )
        if args.readable_val_metrics:
            acc_kwargs["bw_model_name"] = args.network_type_Gyro
            acc_kwargs["train_initial_bw_mode"] = args.train_initial_bw_mode
            acc_kwargs["train_initial_ba_mode"] = args.train_initial_ba_mode
            if args.ba_init_model_path is not None:
                acc_kwargs["ba_init_model_path"] = args.ba_init_model_path
            if args.acc_loss_type in {"se23", "interval_nll", "trajectory_nll"}:
                acc_kwargs["acc_loss_rot_weight"] = args.acc_loss_rot_weight
                acc_kwargs["acc_loss_vel_weight"] = args.acc_loss_vel_weight
                acc_kwargs["acc_loss_pos_weight"] = args.acc_loss_pos_weight
            elif args.acc_loss_type == "nll":
                acc_kwargs["nll_train_gyro"] = args.nll_train_gyro
                acc_kwargs["nll_mean_anchor_weight"] = args.nll_mean_anchor_weight
                acc_kwargs["nll_mean_anchor_mode"] = args.nll_mean_anchor_mode
        acc_profile = acc_train_fn(
            dataset_train,
            dataset_val,
            str(args.outputdir),
            bw_func,
            args.network_type_Acc,
            args.integral_method,
            args.device,
            args.lr_Acc,
            args.weight_decay_Acc,
            epoch=args.epoch,
            lr_step_size=args.acc_lr_step_size,
            lr_gamma=args.acc_lr_gamma,
            train_batch_strategy=args.train_batch_strategy,
            val_freq=args.val_freq,
            save_freq=args.save_freq,
            **acc_kwargs,
        )
        if args.acc_loss_type in {"nll", "interval_nll", "trajectory_nll"} and args.nll_train_gyro:
            gyro_ckpt = resolve_checkpoint(
                args.outputdir,
                "Gyro_weights",
                None,
                args.epoch,
                args.save_freq,
                args.use_best_val,
            )

        elapsed = time.time() - t_start
        with open(args.outputdir / "time_memory_record.txt", "w", encoding="utf-8") as f:
            f.write(
                format_training_profile(
                    elapsed,
                    gyro_profile,
                    acc_profile,
                    extra={
                        "dataset": "LIASuw",
                        "trainval_supervision_source": args.trainval_supervision_source,
                        "test_supervision_source": MOCAP_200HZ_SUPERVISION,
                        "pseudo_gt_root": str(args.pseudo_gt_root),
                        "vo_supervision_world_frame": args.vo_supervision_world_frame,
                        "vo_gap_tolerance_s": args.vo_gap_tolerance_s,
                        "source_position_sg_window": args.source_position_sg_window,
                        "source_position_sg_order": args.source_position_sg_order,
                        "gravity_accel_world_default": tuple(args.gravity_accel_world),
                        "gravity_calibration_json": str(args.gravity_calibration_json)
                        if args.gravity_calibration_json is not None
                        else None,
                        "train_seqs": list(args.train_seqs),
                        "test_seqs": list(args.test_seqs),
                        "loss_window": args.loss_window,
                        "ode_grid_window_s": (args.loss_window - 1) * args.dt,
                        "nominal_imu_window_s": (args.loss_window - 1) * args.dt
                        if args.trainval_supervision_source == MOCAP_200HZ_SUPERVISION
                        else None,
                        "source_timing_window_note": (
                            "loss_window counts source poses; physical span is read from source timestamps"
                            if args.trainval_supervision_source
                            in {MOCAP_NONINTERP_SUPERVISION, VO_IMU_NONINTERP_SUPERVISION}
                            else None
                        ),
                        "batch_size": args.batch_size,
                        "integral_method": args.integral_method,
                        "network_type_Gyro": args.network_type_Gyro,
                        "network_type_Acc": args.network_type_Acc,
                        "acc_loss_type": args.acc_loss_type,
                        "initial_bias_mode": args.initial_bias_mode,
                        "initial_bias_window_s": args.initial_bias_window_s,
                        "train_initial_bw_mode": args.train_initial_bw_mode,
                        "train_initial_ba_mode": args.train_initial_ba_mode,
                        "acc_loss_rot_weight": args.acc_loss_rot_weight,
                        "acc_loss_vel_weight": args.acc_loss_vel_weight,
                        "acc_loss_pos_weight": args.acc_loss_pos_weight,
                        "nll_covariance_strategy": args.nll_covariance_strategy,
                        "nll_optimize_q": _nll_optimize_q(args),
                        "nll_sigma_a": args.nll_sigma_a,
                        "nll_sigma_g": args.nll_sigma_g,
                        "nll_w_rot": args.nll_w_rot,
                        "nll_w_vel": args.nll_w_vel,
                        "nll_w_pos": args.nll_w_pos,
                        "nll_q_block_epochs": args.nll_q_block_epochs,
                        "nll_lr_q": args.nll_lr_q,
                        "nll_sigma_a_min": args.nll_sigma_a_min,
                        "nll_sigma_a_max": args.nll_sigma_a_max,
                        "nll_sigma_g_min": args.nll_sigma_g_min,
                        "nll_sigma_g_max": args.nll_sigma_g_max,
                        "nll_supervision_mode": args.nll_supervision_mode,
                        "ba_init_model_path": str(args.ba_init_model_path) if args.ba_init_model_path is not None else None,
                        "nll_q_only": _nll_q_only(args),
                        "nll_train_gyro": args.nll_train_gyro,
                        "nll_mean_anchor_weight": args.nll_mean_anchor_weight,
                        "nll_mean_anchor_mode": args.nll_mean_anchor_mode,
                        "interval_nll_mode": args.interval_nll_mode,
                        "interval_sigma_g_nominal": args.interval_sigma_g_nominal,
                        "interval_sigma_a_nominal": args.interval_sigma_a_nominal,
                        "interval_sigma_g_floor2": args.interval_sigma_g_floor2,
                        "interval_sigma_a_floor2": args.interval_sigma_a_floor2,
                        "interval_target_cov_floor": args.interval_target_cov_floor,
                        "interval_jitter": args.interval_jitter,
                        "interval_prior_lambda_g": args.interval_prior_lambda_g,
                        "interval_prior_lambda_a": args.interval_prior_lambda_a,
                        "interval_lr_sigma": args.interval_lr_sigma,
                        "interval_detach_covariance_theta": args.interval_detach_covariance_theta,
                        "interval_covariance_gradient_mode": args.interval_covariance_gradient_mode,
                        "interval_debug_covariance_gradients": args.interval_debug_covariance_gradients,
                        "interval_noise_init_model_path": str(args.interval_noise_init_model_path)
                        if args.interval_noise_init_model_path is not None
                        else None,
                        "trajectory_supervision_mode": args.trajectory_supervision_mode,
                        "trajectory_supervision_stride": args.trajectory_supervision_stride,
                        "trajectory_max_supervised_states": args.trajectory_max_supervised_states,
                        "readable_val_metrics": args.readable_val_metrics,
                        "device": args.device,
                        "test_device": args.test_device,
                        "stage_control": "skip_test" if args.skip_test else "all",
                        "gyro_checkpoint": str(gyro_ckpt),
                        "acc_checkpoint": str(acc_ckpt),
                    },
                )
            )

        if args.skip_test:
            return

    dataset_test = LIASuwDebiasDataset(
        **params,
        data_cache_dir=str(args.cache_dir),
        mode="test",
        recompute=False,
    )

    bw_func = bw_func_choose(args.network_type_Gyro, device=args.test_device)
    bw_func.load_state_dict(torch.load(gyro_ckpt, map_location=args.test_device, weights_only=True)["func_bw_model"])
    ba_func = ba_func_choose(args.network_type_Acc, device=args.test_device)
    ba_func.load_state_dict(torch.load(acc_ckpt, map_location=args.test_device, weights_only=True)["func_ba_model"])
    with torch.no_grad():
        Test_with_configured_aux(
            dataset_test,
            str(args.outputdir),
            bw_func,
            ba_func,
            args.integral_method,
            device=args.test_device,
            recompute=args.test_recompute,
            use_delta_imu=use_delta_imu,
            initial_bias_mode=args.initial_bias_mode,
            initial_bias_window_s=args.initial_bias_window_s,
        )

    result_root = args.outputdir / "Sequnces-results"
    lines = []
    for seq_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        path_tmp = seq_dir / "FullTra_result.p"
        if args.skip_plots:
            continue
        aoe, ate_vel, ate_pos = simple_visualization(str(path_tmp), return_pos=True)
        lines.append(f"sequence: {seq_dir.name:20} AOE: {aoe:.3f} ATE vel: {ate_vel:.3f} ATE pos: {ate_pos:.3f}")
        plot_bias(str(path_tmp), dataset_test)
    if lines:
        with open(args.outputdir / "evaluation_summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")


if __name__ == "__main__":
    main()
