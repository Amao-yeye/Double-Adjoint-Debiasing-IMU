import torch

import lie_algebra as Lie


def _batch_eye(batch_shape, size, device, dtype):
    return torch.eye(size, device=device, dtype=dtype).expand(*batch_shape, size, size).clone()


def paper_invariant_error_matrices(R, v, p, dt, gravity=None):
    """Build the right-invariant error transition used in target_CoRL26.pdf.

    State error order is [theta, v, p]. Noise order is [n_a, n_g].
    Shapes:
        R: (..., 3, 3)
        v: (..., 3)
        p: (..., 3)
        dt: scalar or broadcastable to (...)

    Returns:
        F: (..., 9, 9)
        G: (..., 9, 6)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R should have shape (..., 3, 3), got {tuple(R.shape)}")
    if v.shape[-1:] != (3,) or p.shape[-1:] != (3,):
        raise ValueError("v and p should have shape (..., 3)")

    batch_shape = R.shape[:-2]
    device = R.device
    dtype = R.dtype

    v = v.to(device=device, dtype=dtype)
    p = p.to(device=device, dtype=dtype)
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    dt = torch.broadcast_to(dt, batch_shape)
    dt_mat = dt[..., None, None]

    if gravity is None:
        gravity = torch.tensor([0.0, 0.0, -9.81], device=device, dtype=dtype)
    else:
        gravity = torch.as_tensor(gravity, device=device, dtype=dtype)

    I3 = _batch_eye(batch_shape, 3, device, dtype)
    F = torch.zeros(*batch_shape, 9, 9, device=device, dtype=dtype)
    g_hat = Lie.so3hat(gravity).expand(*batch_shape, 3, 3)

    F[..., 0:3, 0:3] = I3
    F[..., 3:6, 0:3] = g_hat * dt_mat
    F[..., 3:6, 3:6] = I3
    F[..., 6:9, 0:3] = 0.5 * g_hat * dt_mat.square()
    F[..., 6:9, 3:6] = I3 * dt_mat
    F[..., 6:9, 6:9] = I3

    G = torch.zeros(*batch_shape, 9, 6, device=device, dtype=dtype)
    G[..., 0:3, 3:6] = R
    G[..., 3:6, 0:3] = R
    G[..., 3:6, 3:6] = Lie.so3hat(v) @ R
    G[..., 6:9, 3:6] = Lie.so3hat(p) @ R
    G = -dt_mat * G

    return F, G


def propagate_paper_invariant_covariance_step(P, R, v, p, dt, noise_cov, gravity=None):
    """Propagate one covariance step using right-invariant error dynamics.

    Implements:
        P_next = F P F^T + G Sigma_n G^T

    State error order is [theta, v, p]. Noise order is [n_a, n_g].
    """
    F, G = paper_invariant_error_matrices(R, v, p, dt, gravity=gravity)
    noise_cov = torch.as_tensor(noise_cov, device=R.device, dtype=R.dtype)
    return F @ P @ F.transpose(-1, -2) + G @ noise_cov @ G.transpose(-1, -2)


def propagate_paper_invariant_covariance(R, v, p, dt, noise_cov, P0=None, gravity=None):
    """Propagate covariance along a nominal trajectory.

    Args:
        R: (T, ..., 3, 3) nominal rotations.
        v: (T, ..., 3) nominal velocities.
        p: (T, ..., 3) nominal positions.
        dt: scalar or length T-1 tensor of transition durations.
        noise_cov: (..., 6, 6) or (T-1, ..., 6, 6), ordered as [n_a, n_g].
        P0: optional (..., 9, 9). Defaults to zeros.
        gravity: optional gravity vector. Defaults to [0, 0, -9.81].

    Returns:
        P: (T, ..., 9, 9) propagated covariance sequence.
    """
    if R.ndim < 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R should have shape (T, ..., 3, 3), got {tuple(R.shape)}")
    if R.shape[0] < 1:
        raise ValueError("R should contain at least one state")

    T = R.shape[0]
    batch_shape = R.shape[1:-2]
    device = R.device
    dtype = R.dtype

    if P0 is None:
        P0 = torch.zeros(*batch_shape, 9, 9, device=device, dtype=dtype)
    else:
        P0 = torch.as_tensor(P0, device=device, dtype=dtype)

    P = torch.empty(T, *batch_shape, 9, 9, device=device, dtype=dtype)
    P[0] = P0

    noise_cov = torch.as_tensor(noise_cov, device=device, dtype=dtype)
    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)

    for k in range(T - 1):
        dt_k = dt_tensor[k] if dt_tensor.ndim > 0 and dt_tensor.shape[0] == T - 1 else dt_tensor
        noise_cov_k = noise_cov[k] if noise_cov.ndim >= 3 and noise_cov.shape[0] == T - 1 else noise_cov
        P[k + 1] = propagate_paper_invariant_covariance_step(
            P[k],
            R[k],
            v[k],
            p[k],
            dt_k,
            noise_cov_k,
            gravity=gravity,
        )

    return P


def paper_invariant_continuous_matrices(R, v, p, gravity=None):
    """Build continuous right-invariant error matrices from target_CoRL26.pdf.

    Continuous dynamics:
        xi_dot = A xi + G_t n_t

    State error order is [theta, v, p]. Noise order is [n_a, n_g].
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R should have shape (..., 3, 3), got {tuple(R.shape)}")
    if v.shape[-1:] != (3,) or p.shape[-1:] != (3,):
        raise ValueError("v and p should have shape (..., 3)")

    batch_shape = R.shape[:-2]
    device = R.device
    dtype = R.dtype

    if gravity is None:
        gravity = torch.tensor([0.0, 0.0, -9.81], device=device, dtype=dtype)
    else:
        gravity = torch.as_tensor(gravity, device=device, dtype=dtype)

    I3 = _batch_eye(batch_shape, 3, device, dtype)
    g_hat = Lie.so3hat(gravity).expand(*batch_shape, 3, 3)

    A = torch.zeros(*batch_shape, 9, 9, device=device, dtype=dtype)
    A[..., 3:6, 0:3] = g_hat
    A[..., 6:9, 3:6] = I3

    G = torch.zeros(*batch_shape, 9, 6, device=device, dtype=dtype)
    G[..., 0:3, 3:6] = R
    G[..., 3:6, 0:3] = R
    G[..., 3:6, 3:6] = Lie.so3hat(v.to(device=device, dtype=dtype)) @ R
    G[..., 6:9, 3:6] = Lie.so3hat(p.to(device=device, dtype=dtype)) @ R
    G = -G

    return A, G


def augment_state_with_covariance(y, P0=None):
    """Append a flattened 9x9 invariant covariance to a nominal ODE state."""
    if y.shape[-1] != 16:
        raise ValueError(f"nominal state y should have last dimension 16, got {y.shape[-1]}")

    batch_shape = y.shape[:-1]
    if P0 is None:
        P0 = torch.zeros(*batch_shape, 9, 9, device=y.device, dtype=y.dtype)
    else:
        P0 = torch.as_tensor(P0, device=y.device, dtype=y.dtype)
    return torch.cat([y, P0.reshape(*batch_shape, 81)], dim=-1)


def split_state_with_covariance(y_aug):
    """Split augmented state into nominal state and 9x9 invariant covariance."""
    if y_aug.shape[-1] != 97:
        raise ValueError(f"augmented state should have last dimension 97, got {y_aug.shape[-1]}")
    y = y_aug[..., :16]
    P = y_aug[..., 16:].reshape(*y_aug.shape[:-1], 9, 9)
    return y, P


class JointNominalInvariantCovarianceODE(torch.nn.Module):
    """Joint ODE for nominal state and paper right-invariant covariance.

    The wrapped nominal model must be compatible with odeint_SO3 and use the
    standard state order [xi, t, bw, v, p, ba]. The augmented state order is:
        [xi, t, bw, v, p, ba, vec(P)]
    where P is a 9x9 covariance for invariant error [theta, v, p].
    """

    def __init__(self, nominal_model, noise_cov, gravity=None):
        super().__init__()
        self.nominal_model = nominal_model
        if isinstance(noise_cov, torch.nn.Parameter):
            self.noise_cov = noise_cov
        elif isinstance(noise_cov, torch.Tensor) and noise_cov.requires_grad:
            self.noise_cov = noise_cov
        else:
            self.register_buffer("noise_cov", torch.as_tensor(noise_cov).clone())
        if gravity is None:
            self.gravity = None
        else:
            self.register_buffer("gravity", torch.as_tensor(gravity).clone())

    def set_u_func(self, u_func):
        self.nominal_model.set_u_func(u_func)

    def set_u_dot_func(self, u_dot_func):
        self.nominal_model.set_u_dot_func(u_dot_func)

    def set_R0(self, R0):
        self.nominal_model.set_R0(R0)

    @property
    def R0(self):
        return self.nominal_model.R0

    def callback_change_chart(self, R0):
        self.nominal_model.callback_change_chart(R0)

    def __call__(self, t, y_aug):
        y, P = split_state_with_covariance(y_aug)

        y_dot = self.nominal_model(t, y)
        R = self.nominal_model.R0.to(device=y.device, dtype=y.dtype) @ Lie.SO3exp(y[..., :3])
        v = y[..., 7:10]
        p = y[..., 10:13]
        gravity = None if self.gravity is None else self.gravity.to(device=y.device, dtype=y.dtype)
        A, G = paper_invariant_continuous_matrices(R, v, p, gravity=gravity)

        noise_cov = self.noise_cov.to(device=y.device, dtype=y.dtype)
        P_dot = A @ P + P @ A.transpose(-1, -2) + G @ noise_cov @ G.transpose(-1, -2)
        return torch.cat([y_dot, P_dot.reshape(*y.shape[:-1], 81)], dim=-1)


def make_joint_nominal_covariance_ode(nominal_model, noise_cov, gravity=None):
    """Create an odeint_SO3-compatible joint nominal/covariance ODE."""
    return JointNominalInvariantCovarianceODE(nominal_model, noise_cov, gravity=gravity)
