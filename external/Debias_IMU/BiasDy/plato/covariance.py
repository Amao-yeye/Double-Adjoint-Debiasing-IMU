from __future__ import annotations

import torch

try:
    import lie_algebra as Lie
except ImportError:  # Support `python -m BiasDy.plato...` from Debias_IMU root.
    from BiasDy import lie_algebra as Lie


def se23_phi(
    dt: torch.Tensor | float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> torch.Tensor:
    """SE_2(3) invariant-error transition matrix for constant gravity dynamics."""
    dt = _as_tensor(dt, device=device, dtype=dtype)
    device = dt.device
    dtype = dt.dtype
    batch_shape = dt.shape

    eye3 = _eye(3, batch_shape, device, dtype)
    phi = torch.zeros(*batch_shape, 9, 9, device=device, dtype=dtype)
    phi[..., 0:3, 0:3] = eye3
    phi[..., 3:6, 3:6] = eye3
    phi[..., 6:9, 6:9] = eye3

    g = torch.tensor(gravity, device=device, dtype=dtype).expand(*batch_shape, 3)
    g_hat = Lie.so3hat(g)
    dt_mat = dt.unsqueeze(-1).unsqueeze(-1)
    phi[..., 3:6, 0:3] = g_hat * dt_mat
    phi[..., 6:9, 0:3] = 0.5 * g_hat * dt_mat.square()
    phi[..., 6:9, 3:6] = eye3 * dt_mat
    return phi


def se23_Gd(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Discrete IMU noise injection matrix.

    The convention is per-sample discrete noise:
    Q_d,k = G_d,k Q_n G_d,k^T, where Q_n = diag(sigma_a^2 I, sigma_g^2 I).
    Columns are ordered [n_a, n_g], rows [phi, rho_v, rho_p].
    """
    device = R.device
    dtype = R.dtype
    dt = _as_tensor(dt, device=device, dtype=dtype)
    batch_shape = torch.broadcast_shapes(R.shape[:-2], v.shape[:-1], p.shape[:-1], dt.shape)

    R = R.expand(*batch_shape, 3, 3)
    v = v.expand(*batch_shape, 3)
    p = p.expand(*batch_shape, 3)
    dt = dt.expand(batch_shape)

    zeros = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
    G = torch.zeros(*batch_shape, 9, 6, device=device, dtype=dtype)
    G[..., 0:3, 0:3] = zeros
    G[..., 0:3, 3:6] = R
    G[..., 3:6, 0:3] = R
    G[..., 3:6, 3:6] = torch.matmul(Lie.so3hat(v), R)
    G[..., 6:9, 0:3] = zeros
    G[..., 6:9, 3:6] = torch.matmul(Lie.so3hat(p), R)
    return -dt.unsqueeze(-1).unsqueeze(-1) * G


def preintegrate_interval_covariance(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    start_idx: int,
    end_idx: int,
    Qn: torch.Tensor,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate covariance for one supervision interval [start_idx, end_idx]."""
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")

    batch_shape = R.shape[1:-2]
    device = R.device
    dtype = R.dtype
    P = torch.zeros(*batch_shape, 9, 9, device=device, dtype=dtype)
    Phi_total = _eye(9, batch_shape, device, dtype)
    Qn = Qn.to(device=device, dtype=dtype)

    for step in range(start_idx, end_idx):
        dt = _step_dt(t, step).to(device=device, dtype=dtype)
        Phi_step = se23_phi(dt, device=device, dtype=dtype)
        Gd_step = se23_Gd(R[step], v[step], p[step], dt)
        Qd_step = torch.matmul(torch.matmul(Gd_step, Qn), Gd_step.transpose(-1, -2))
        P = torch.matmul(torch.matmul(Phi_step, P), Phi_step.transpose(-1, -2)) + Qd_step
        Phi_total = torch.matmul(Phi_step, Phi_total)

    return Phi_total, add_jitter(symmetrize(P), jitter)


def preintegrate_supervision_covariances(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    supervised_indices: torch.Tensor,
    Qn: torch.Tensor,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Phi_i^s and Q_i^s for all adjacent supervised states."""
    supervised_indices = supervised_indices.to(device=R.device, dtype=torch.long)
    if supervised_indices.dim() != 1 or supervised_indices.numel() < 2:
        raise ValueError("supervised_indices must be a 1D tensor with at least two entries")

    Phi_list = []
    Q_list = []
    for i in range(supervised_indices.numel() - 1):
        start_idx = int(supervised_indices[i].item())
        end_idx = int(supervised_indices[i + 1].item())
        Phi_i, Q_i = preintegrate_interval_covariance(
            R, v, p, t, start_idx, end_idx, Qn, jitter=jitter
        )
        Phi_list.append(Phi_i)
        Q_list.append(Q_i)
    return torch.stack(Phi_list, dim=0), torch.stack(Q_list, dim=0)


def build_se23_precision(
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
    prior_std: float = 1e-4,
    jitter: float = 1e-8,
) -> torch.Tensor:
    """Assemble dense storage for the block-tridiagonal SE_2(3) precision."""
    if isinstance(Phi_list, list):
        Phi_list = torch.stack(Phi_list, dim=0)
    if isinstance(Q_list, list):
        Q_list = torch.stack(Q_list, dim=0)
    if Phi_list.shape[0] != Q_list.shape[0]:
        raise ValueError("Phi_list and Q_list must have the same interval count")

    M = Phi_list.shape[0]
    dx = 9
    batch_shape = Phi_list.shape[1:-2]
    device = Phi_list.device
    dtype = Phi_list.dtype
    total_dim = (M + 1) * dx
    Lambda = torch.zeros(*batch_shape, total_dim, total_dim, device=device, dtype=dtype)
    prior_inv = (1.0 / (prior_std * prior_std)) * _eye(dx, batch_shape, device, dtype)
    Lambda[..., 0:dx, 0:dx] = Lambda[..., 0:dx, 0:dx] + prior_inv

    for i in range(M):
        Phi_i = Phi_list[i]
        Q_i = add_jitter(symmetrize(Q_list[i]), jitter)
        Q_inv = cholesky_inverse_spd(Q_i, jitter=jitter)
        i0 = i * dx
        i1 = (i + 1) * dx
        block_ii = torch.matmul(torch.matmul(Phi_i.transpose(-1, -2), Q_inv), Phi_i)
        block_i_next = -torch.matmul(Phi_i.transpose(-1, -2), Q_inv)
        block_next_i = -torch.matmul(Q_inv, Phi_i)

        Lambda[..., i0:i0 + dx, i0:i0 + dx] = (
            Lambda[..., i0:i0 + dx, i0:i0 + dx] + block_ii
        )
        Lambda[..., i0:i0 + dx, i1:i1 + dx] = (
            Lambda[..., i0:i0 + dx, i1:i1 + dx] + block_i_next
        )
        Lambda[..., i1:i1 + dx, i0:i0 + dx] = (
            Lambda[..., i1:i1 + dx, i0:i0 + dx] + block_next_i
        )
        Lambda[..., i1:i1 + dx, i1:i1 + dx] = (
            Lambda[..., i1:i1 + dx, i1:i1 + dx] + Q_inv
        )

    return add_jitter(symmetrize(Lambda), jitter)


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def add_jitter(matrix: torch.Tensor, jitter: float) -> torch.Tensor:
    if jitter <= 0:
        return matrix
    eye = _eye(matrix.shape[-1], matrix.shape[:-2], matrix.device, matrix.dtype)
    return matrix + jitter * eye


def safe_cholesky(matrix: torch.Tensor, jitter: float = 1e-8, max_tries: int = 5) -> torch.Tensor:
    matrix = symmetrize(matrix)
    eye = _eye(matrix.shape[-1], matrix.shape[:-2], matrix.device, matrix.dtype)
    current_jitter = jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(matrix + current_jitter * eye)
        except RuntimeError:
            current_jitter *= 10.0
    return torch.linalg.cholesky(matrix + current_jitter * eye)


def cholesky_inverse_spd(matrix: torch.Tensor, jitter: float = 1e-8) -> torch.Tensor:
    chol = safe_cholesky(matrix, jitter=jitter)
    return torch.cholesky_inverse(chol)


def logdet_spd(matrix: torch.Tensor, jitter: float = 1e-8) -> torch.Tensor:
    chol = safe_cholesky(matrix, jitter=jitter)
    return 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(dim=-1)


def _step_dt(t: torch.Tensor, step: int) -> torch.Tensor:
    return t[step + 1] - t[step]


def _as_tensor(
    value: torch.Tensor | float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if device is None:
            device = value.device
        if dtype is None:
            dtype = value.dtype
        return value.to(device=device, dtype=dtype)
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.tensor(value, device=device, dtype=dtype)


def _eye(
    n: int,
    batch_shape: torch.Size | tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.eye(n, device=device, dtype=dtype).expand(*batch_shape, n, n)
