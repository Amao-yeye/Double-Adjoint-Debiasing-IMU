"""PLATO-owned M2 bias dynamics networks for Debias_IMU ablations.

These modules intentionally live outside external/Debias_IMU.  They preserve
the Debias_IMU bias-function call signature, but ignore spline derivatives so
the dynamics depend only on current bias and raw IMU measurements.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import lie_algebra as Lie


GYRO_M2_NAME = "bw_func_m2_no_deriv"
ACC_M2_NAME = "ba_func_m2_no_deriv"
GYRO_PHYS_EQ_NAME = "bw_func_phys_eq"
ACC_PHYS_EQ_NAME = "ba_func_phys_eq"
GYRO_PHYS_RATE_NAME = "bw_func_phys_rate"
ACC_PHYS_RATE_NAME = "ba_func_phys_rate"
GYRO_PHYS_DELTA_NAME = "bw_func_phys_delta"
ACC_PHYS_DELTA_NAME = "ba_func_phys_delta"
GYRO_FBU_RESIDUAL_NAME = "bw_func_fbu_residual"
ACC_FBU_RESIDUAL_NAME = "ba_func_fbu_residual"
GYRO_SLOW_RESIDUAL_NAME = "bw_func_slow_residual"
ACC_SLOW_RESIDUAL_NAME = "ba_func_slow_residual"
GYRO_BASELINE_NO_UDOT_NAME = "bw_func_net_no_udot"
ACC_BASELINE_NO_UDOT_NAME = "ba_func_net_no_udot"


class _M2BiasFuncNet(nn.Module):
    """dot b = -Lambda * b + residual_scale * NN(b, omega_m, a_m)."""

    def __init__(
        self,
        bias_slice: slice,
        bias_scale: float,
        hidden_dim: int,
        lambda_init: float = -20.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.bias_slice = bias_slice
        self.bias_scale = float(bias_scale)
        self.residual_scale = float(residual_scale)
        self.raw_lambda = nn.Parameter(torch.full((3,), float(lambda_init)))
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(module.bias, 0.0)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.weight, 0.0)
            nn.init.constant_(final.bias, 0.0)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        del imu_meas_dot, R0
        bias = y[..., self.bias_slice]
        omega = imu_meas[..., :3]
        accel = imu_meas[..., 3:6]
        features = torch.cat(
            [
                bias / self.bias_scale,
                omega,
                accel / 9.81,
            ],
            dim=-1,
        )
        lam = torch.nn.functional.softplus(self.raw_lambda).to(dtype=bias.dtype, device=bias.device)
        return -lam * bias + self.residual_scale * self.net(features)


class M2GyroFuncNet(_M2BiasFuncNet):
    def __init__(self):
        super().__init__(slice(4, 7), bias_scale=0.1, hidden_dim=128)


class M2AccFuncNet(_M2BiasFuncNet):
    def __init__(self):
        super().__init__(slice(13, 16), bias_scale=1.0, hidden_dim=128)


class BaselineNoUDotGyroFuncNet(nn.Module):
    """Original-style gyro network using [b_g, omega_m] and no IMU derivative."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 6),
        )
        self.net2 = nn.Sequential(
            nn.Linear(6, 512),
            nn.Tanh(),
            nn.Linear(512, 6),
        )
        self.linear2 = nn.Linear(6, 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        del imu_meas_dot, R0
        x = torch.cat([y[..., 4:7], imu_meas[..., :3]], dim=-1)
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)


class BaselineNoUDotAccFuncNet(nn.Module):
    """Original-style accel network using [b_a, accel_m] and no IMU derivative."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 6),
        )
        self.net2 = nn.Sequential(
            nn.Linear(6, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 6),
        )
        self.linear2 = nn.Linear(6, 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        del imu_meas_dot, R0
        x = torch.cat([y[..., 13:16], imu_meas[..., 3:6]], dim=-1)
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)


class _FbuResidualBiasFuncNet(nn.Module):
    """Pure f(b, u) residual MLP with bounded bias-rate output.

    This is a conservative middle point between the unconstrained M2 MLP and
    the heavily structured physics-rate model.  It uses only the current bias
    and current IMU measurement, and starts near constant-bias behavior.
    """

    def __init__(
        self,
        bias_slice: slice,
        bias_scale: float,
        hidden_dim: int,
        rate_bound: float,
    ):
        super().__init__()
        self.bias_slice = bias_slice
        self.bias_scale = float(bias_scale)
        self.rate_bound = float(rate_bound)
        feature_dim = 11
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.linear_out = nn.Linear(feature_dim, 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        del imu_meas_dot, R0
        bias = y[..., self.bias_slice] / self.bias_scale
        omega = imu_meas[..., :3] / 5.0
        accel = imu_meas[..., 3:6] / 9.81
        features = torch.cat(
            [
                bias,
                omega,
                accel,
                torch.linalg.norm(omega, dim=-1, keepdim=True),
                torch.linalg.norm(accel, dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        hidden = self.net(features) + features
        hidden = self.net2(hidden) + hidden
        return self.rate_bound * torch.tanh(self.linear_out(hidden))


class FbuResidualGyroFuncNet(_FbuResidualBiasFuncNet):
    def __init__(self):
        super().__init__(slice(4, 7), bias_scale=0.1, hidden_dim=128, rate_bound=0.25)


class FbuResidualAccFuncNet(_FbuResidualBiasFuncNet):
    def __init__(self):
        super().__init__(slice(13, 16), bias_scale=1.0, hidden_dim=128, rate_bound=2.5)


class _SlowResidualBiasFuncNet(nn.Module):
    """Rate-limited f(b, u, R) bias dynamics for corldata ablations.

    The model starts exactly at constant-bias behavior and can only learn slow
    bias-rate corrections:
        dot b = -lambda * (b - b_center) + rate_bound * tanh(linear(phi) + NN(phi)).

    It intentionally avoids IMU derivatives because the current corldata
    analysis points to good local frame consistency but weak full-trajectory
    generalization, especially for accelerometer bias.
    """

    def __init__(
        self,
        bias_slice: slice,
        bias_scale: float,
        hidden_dim: int,
        rate_bound: float,
        lambda_init: float = -9.0,
        use_acc_bias: bool = True,
    ):
        super().__init__()
        self.bias_slice = bias_slice
        self.bias_scale = float(bias_scale)
        self.rate_bound = float(rate_bound)
        self.use_acc_bias = bool(use_acc_bias)
        self.raw_lambda = nn.Parameter(torch.full((3,), float(lambda_init)))
        self.bias_center = nn.Parameter(torch.zeros(3))
        self.feature_dim = 31
        self.linear_phys = nn.Linear(self.feature_dim, 3)
        self.residual = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.linear_phys.weight, 0.0)
        nn.init.constant_(self.linear_phys.bias, 0.0)
        for module in self.residual:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(module.bias, 0.0)
        final = self.residual[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.weight, 0.0)
            nn.init.constant_(final.bias, 0.0)

    def _features(self, y: torch.Tensor, imu_meas: torch.Tensor, R0: torch.Tensor) -> torch.Tensor:
        bias = y[..., self.bias_slice]
        bg = y[..., 4:7]
        ba = y[..., 13:16] if self.use_acc_bias else torch.zeros_like(y[..., 13:16])
        omega_c = imu_meas[..., :3] - bg
        f_c = imu_meas[..., 3:6] - ba
        g_body = _PhysEqBiasFuncNet._body_gravity(y, R0)
        a_dyn_body = f_c + g_body

        bias_n = bias / self.bias_scale
        omega_n = omega_c / 5.0
        f_n = f_c / 9.81
        a_dyn_n = a_dyn_body / 9.81
        g_body_n = g_body / 9.81
        features = [
            bias_n,
            omega_n,
            f_n,
            a_dyn_n,
            g_body_n,
            torch.linalg.norm(omega_n, dim=-1, keepdim=True),
            torch.linalg.norm(f_n, dim=-1, keepdim=True),
            torch.linalg.norm(a_dyn_n, dim=-1, keepdim=True),
            torch.linalg.norm(g_body_n, dim=-1, keepdim=True),
            omega_n * omega_n,
            f_n * f_n,
            torch.cross(omega_n, f_n, dim=-1),
            torch.cross(omega_n, g_body_n, dim=-1),
        ]
        return torch.cat(features, dim=-1)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        del imu_meas_dot
        bias = y[..., self.bias_slice]
        features = self._features(y, imu_meas, R0)
        lam = torch.nn.functional.softplus(self.raw_lambda).to(dtype=y.dtype, device=y.device)
        center = self.bias_center.to(dtype=y.dtype, device=y.device)
        rate_logits = self.linear_phys(features) + self.residual(features)
        return -lam * (bias - center) + self.rate_bound * torch.tanh(rate_logits)


class SlowResidualGyroFuncNet(_SlowResidualBiasFuncNet):
    def __init__(self):
        super().__init__(
            slice(4, 7),
            bias_scale=0.1,
            hidden_dim=96,
            rate_bound=0.03,
            use_acc_bias=False,
        )


class SlowResidualAccFuncNet(_SlowResidualBiasFuncNet):
    def __init__(self):
        super().__init__(
            slice(13, 16),
            bias_scale=1.0,
            hidden_dim=96,
            rate_bound=0.15,
            use_acc_bias=True,
        )


class _PhysEqBiasFuncNet(nn.Module):
    """Physics-informed stable equilibrium bias dynamics.

    The structure is:
        dot b = lambda * (b_eq(features) - b) + residual_scale * tanh(MLP(features))

    All equilibrium heads and the residual final layer are zero-initialized, so
    the initial model is close to constant bias while still allowing gradients
    into the structured terms through a small positive lambda.
    """

    def __init__(
        self,
        bias_slice: slice,
        bias_scale: float,
        hidden_dim: int,
        lambda_init: float = -5.0,
        residual_scale: float = 0.05,
        use_acc_bias: bool = True,
        include_acc_terms: bool = True,
    ):
        super().__init__()
        self.bias_slice = bias_slice
        self.bias_scale = float(bias_scale)
        self.residual_scale = float(residual_scale)
        self.use_acc_bias = bool(use_acc_bias)
        self.include_acc_terms = bool(include_acc_terms)
        self.raw_lambda = nn.Parameter(torch.full((3,), float(lambda_init)))
        self.eq_bias = nn.Parameter(torch.zeros(3))
        self.eq_omega = nn.Linear(3, 3, bias=False)
        self.eq_adyn = nn.Linear(3, 3, bias=False)
        self.eq_gbody = nn.Linear(3, 3, bias=False)
        if self.include_acc_terms:
            self.eq_omega_sq = nn.Linear(3, 3, bias=False)
            self.eq_cross = nn.Linear(3, 3, bias=False)
        else:
            self.eq_omega_sq = None
            self.eq_cross = None

        residual_dim = 21 if self.include_acc_terms else 15
        self.residual = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.eq_omega, self.eq_adyn, self.eq_gbody, self.eq_omega_sq, self.eq_cross):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.0)
        for module in self.residual:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(module.bias, 0.0)
        final = self.residual[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.weight, 0.0)
            nn.init.constant_(final.bias, 0.0)

    @staticmethod
    def _body_gravity(y: torch.Tensor, R0: torch.Tensor) -> torch.Tensor:
        R0 = R0.to(device=y.device, dtype=y.dtype)
        R = R0 @ Lie.SO3exp(y[..., :3])
        gravity = torch.tensor([0.0, 0.0, -9.81], device=y.device, dtype=y.dtype)
        gravity = gravity.expand(*y.shape[:-1], 3)
        return (R.transpose(-1, -2) @ gravity.unsqueeze(-1)).squeeze(-1)

    def _features(self, y: torch.Tensor, imu_meas: torch.Tensor, R0: torch.Tensor):
        bias = y[..., self.bias_slice]
        bg = y[..., 4:7]
        ba = y[..., 13:16] if self.use_acc_bias else torch.zeros_like(y[..., 13:16])
        omega_c = imu_meas[..., :3] - bg
        f_c = imu_meas[..., 3:6] - ba
        g_body = self._body_gravity(y, R0)
        a_dyn_body = f_c + g_body

        omega_n = omega_c / 5.0
        a_dyn_n = a_dyn_body / 9.81
        g_body_n = g_body / 9.81
        bias_n = bias / self.bias_scale
        omega_sq_n = omega_n * omega_n
        cross_n = torch.cross(omega_n, a_dyn_n, dim=-1)

        residual_parts = [
            bias_n,
            omega_n,
            a_dyn_n,
            g_body_n,
            torch.linalg.norm(omega_n, dim=-1, keepdim=True),
            torch.linalg.norm(a_dyn_n, dim=-1, keepdim=True),
            torch.linalg.norm(g_body_n, dim=-1, keepdim=True),
        ]
        if self.include_acc_terms:
            residual_parts.extend([omega_sq_n, cross_n])
        residual_features = torch.cat(residual_parts, dim=-1)
        return bias, omega_n, a_dyn_n, g_body_n, omega_sq_n, cross_n, residual_features

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        del imu_meas_dot
        bias, omega_n, a_dyn_n, g_body_n, omega_sq_n, cross_n, residual_features = self._features(y, imu_meas, R0)
        b_eq = self.eq_bias.to(device=y.device, dtype=y.dtype)
        b_eq = b_eq + self.eq_omega(omega_n) + self.eq_adyn(a_dyn_n) + self.eq_gbody(g_body_n)
        if self.include_acc_terms:
            b_eq = b_eq + self.eq_omega_sq(omega_sq_n) + self.eq_cross(cross_n)
        lam = torch.nn.functional.softplus(self.raw_lambda).to(dtype=y.dtype, device=y.device)
        residual = self.residual_scale * torch.tanh(self.residual(residual_features))
        return lam * (b_eq - bias) + residual


class PhysEqGyroFuncNet(_PhysEqBiasFuncNet):
    def __init__(self):
        super().__init__(
            slice(4, 7),
            bias_scale=0.1,
            hidden_dim=96,
            residual_scale=0.03,
            use_acc_bias=False,
            include_acc_terms=False,
        )


class PhysEqAccFuncNet(_PhysEqBiasFuncNet):
    def __init__(self):
        super().__init__(
            slice(13, 16),
            bias_scale=1.0,
            hidden_dim=96,
            residual_scale=0.3,
            use_acc_bias=True,
            include_acc_terms=True,
        )


class _PhysRateBiasFuncNet(nn.Module):
    """Physics-informed bounded bias-rate dynamics.

    dot b = rate_bound * tanh(linear_phys(phi) + NN(phi))

    phi contains only current bias, corrected angular velocity, specific force,
    body-frame gravity, and low-order inertial invariants.  The structured
    linear layer and the residual final layer are zero-initialized, so the
    model starts exactly at constant bias while preserving direct gradients
    through the physics features.
    """

    def __init__(
        self,
        bias_slice: slice,
        bias_scale: float,
        hidden_dim: int,
        rate_bound: float,
        use_acc_bias: bool = True,
        use_delta_features: bool = False,
    ):
        super().__init__()
        self.bias_slice = bias_slice
        self.bias_scale = float(bias_scale)
        self.rate_bound = float(rate_bound)
        self.use_acc_bias = bool(use_acc_bias)
        self.use_delta_features = bool(use_delta_features)
        self.feature_dim = 54 if self.use_delta_features else 31
        self.linear_phys = nn.Linear(self.feature_dim, 3)
        self.residual = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.linear_phys.weight, 0.0)
        nn.init.constant_(self.linear_phys.bias, 0.0)
        for module in self.residual:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(module.bias, 0.0)
        final = self.residual[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.weight, 0.0)
            nn.init.constant_(final.bias, 0.0)

    def _features(
        self,
        y: torch.Tensor,
        imu_meas: torch.Tensor,
        imu_aux: torch.Tensor | None,
        R0: torch.Tensor,
    ) -> torch.Tensor:
        bias = y[..., self.bias_slice]
        bg = y[..., 4:7]
        ba = y[..., 13:16] if self.use_acc_bias else torch.zeros_like(y[..., 13:16])
        omega_c = imu_meas[..., :3] - bg
        f_c = imu_meas[..., 3:6] - ba
        g_body = _PhysEqBiasFuncNet._body_gravity(y, R0)
        a_dyn_body = f_c + g_body

        bias_n = bias / self.bias_scale
        omega_n = omega_c / 5.0
        f_n = f_c / 9.81
        a_dyn_n = a_dyn_body / 9.81
        g_body_n = g_body / 9.81
        features = [
            bias_n,
            omega_n,
            f_n,
            a_dyn_n,
            g_body_n,
            torch.linalg.norm(omega_n, dim=-1, keepdim=True),
            torch.linalg.norm(f_n, dim=-1, keepdim=True),
            torch.linalg.norm(a_dyn_n, dim=-1, keepdim=True),
            torch.linalg.norm(g_body_n, dim=-1, keepdim=True),
            omega_n * omega_n,
            f_n * f_n,
            torch.cross(omega_n, f_n, dim=-1),
            torch.cross(omega_n, g_body_n, dim=-1),
        ]
        if self.use_delta_features:
            if imu_aux is None:
                imu_aux = torch.zeros_like(imu_meas)
            delta_omega_n = imu_aux[..., :3] / 0.1
            delta_f_n = imu_aux[..., 3:6] / 1.0
            features.extend(
                [
                    delta_omega_n,
                    delta_f_n,
                    torch.linalg.norm(delta_omega_n, dim=-1, keepdim=True),
                    torch.linalg.norm(delta_f_n, dim=-1, keepdim=True),
                    delta_omega_n * delta_omega_n,
                    delta_f_n * delta_f_n,
                    torch.cross(delta_omega_n, delta_f_n, dim=-1),
                    torch.cross(omega_n, delta_f_n, dim=-1),
                    torch.cross(delta_omega_n, f_n, dim=-1),
                ]
            )
        return torch.cat(features, dim=-1)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        features = self._features(y, imu_meas, imu_meas_dot, R0)
        rate_logits = self.linear_phys(features) + self.residual(features)
        return self.rate_bound * torch.tanh(rate_logits)


class PhysRateGyroFuncNet(_PhysRateBiasFuncNet):
    def __init__(self):
        super().__init__(
            slice(4, 7),
            bias_scale=0.1,
            hidden_dim=128,
            rate_bound=0.08,
            use_acc_bias=False,
        )


class PhysRateAccFuncNet(_PhysRateBiasFuncNet):
    def __init__(self):
        super().__init__(
            slice(13, 16),
            bias_scale=1.0,
            hidden_dim=128,
            rate_bound=0.8,
            use_acc_bias=True,
        )


class PhysDeltaGyroFuncNet(_PhysRateBiasFuncNet):
    def __init__(self):
        super().__init__(
            slice(4, 7),
            bias_scale=0.1,
            hidden_dim=128,
            rate_bound=0.08,
            use_acc_bias=False,
            use_delta_features=True,
        )


class PhysDeltaAccFuncNet(_PhysRateBiasFuncNet):
    def __init__(self):
        super().__init__(
            slice(13, 16),
            bias_scale=1.0,
            hidden_dim=128,
            rate_bound=0.8,
            use_acc_bias=True,
            use_delta_features=True,
        )


def bw_func_choose_with_m2(network_type: str, device: str, fallback: Callable):
    if network_type == GYRO_BASELINE_NO_UDOT_NAME:
        return BaselineNoUDotGyroFuncNet().to(device)
    if network_type == GYRO_M2_NAME:
        return M2GyroFuncNet().to(device)
    if network_type == GYRO_PHYS_EQ_NAME:
        return PhysEqGyroFuncNet().to(device)
    if network_type == GYRO_PHYS_RATE_NAME:
        return PhysRateGyroFuncNet().to(device)
    if network_type == GYRO_PHYS_DELTA_NAME:
        return PhysDeltaGyroFuncNet().to(device)
    if network_type == GYRO_FBU_RESIDUAL_NAME:
        return FbuResidualGyroFuncNet().to(device)
    if network_type == GYRO_SLOW_RESIDUAL_NAME:
        return SlowResidualGyroFuncNet().to(device)
    return fallback(network_type, device=device)


def ba_func_choose_with_m2(network_type: str, device: str, fallback: Callable):
    if network_type == ACC_BASELINE_NO_UDOT_NAME:
        return BaselineNoUDotAccFuncNet().to(device)
    if network_type == ACC_M2_NAME:
        return M2AccFuncNet().to(device)
    if network_type == ACC_PHYS_EQ_NAME:
        return PhysEqAccFuncNet().to(device)
    if network_type == ACC_PHYS_RATE_NAME:
        return PhysRateAccFuncNet().to(device)
    if network_type == ACC_PHYS_DELTA_NAME:
        return PhysDeltaAccFuncNet().to(device)
    if network_type == ACC_FBU_RESIDUAL_NAME:
        return FbuResidualAccFuncNet().to(device)
    if network_type == ACC_SLOW_RESIDUAL_NAME:
        return SlowResidualAccFuncNet().to(device)
    return fallback(network_type, device=device)
