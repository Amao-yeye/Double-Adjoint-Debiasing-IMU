from __future__ import annotations

import os
import sys

import torch

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)

try:
    import lie_algebra as Lie
except ImportError:
    from BiasDy import lie_algebra as Lie

try:
    from plato.covariance import (
        build_se23_precision,
        preintegrate_supervision_covariances,
        se23_Gd,
        se23_phi,
        safe_cholesky,
    )
    from plato.nll import (
        PlatoNoiseParams,
        full_state_nees,
        full_state_nll,
        pose_nees,
        pose_nll,
        sparse_full_state_nees,
        sparse_full_state_nll,
        sparse_pose_nees,
        sparse_pose_nll,
    )
except ImportError:
    from .covariance import (
        build_se23_precision,
        preintegrate_supervision_covariances,
        se23_Gd,
        se23_phi,
        safe_cholesky,
    )
    from .nll import (
        PlatoNoiseParams,
        full_state_nees,
        full_state_nll,
        pose_nees,
        pose_nll,
        sparse_full_state_nees,
        sparse_full_state_nll,
        sparse_pose_nees,
        sparse_pose_nll,
    )


def _make_test_trajectory(T: int = 8, B: int = 2):
    torch.manual_seed(19)
    t = 0.005 * torch.arange(T, dtype=torch.float64)
    R = Lie.SO3exp(0.01 * torch.randn(T, B, 3, dtype=torch.float64))
    v = torch.randn(T, B, 3, dtype=torch.float64)
    p = torch.randn(T, B, 3, dtype=torch.float64)
    return R, v, p, t


def test_phi_and_gd_shapes():
    R, v, p, t = _make_test_trajectory()
    Phi = se23_phi(t[1] - t[0], device=R.device, dtype=R.dtype)
    Gd = se23_Gd(R[0], v[0], p[0], t[1] - t[0])
    assert Phi.shape == (9, 9)
    assert Gd.shape == (2, 9, 6)


def test_precision_spd_and_nll_finite():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    supervised_indices = torch.tensor([0, 2, 5, 7])
    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn(), jitter=1e-8
    )
    Lambda = build_se23_precision(Phi_list, Q_list, prior_std=1e-3, jitter=1e-8)
    safe_cholesky(Q_list, jitter=1e-8)
    safe_cholesky(Lambda, jitter=1e-8)

    residual_full = 1e-3 * torch.randn(4, 2, 9, dtype=R.dtype)
    residual_pose = residual_full[..., [0, 1, 2, 6, 7, 8]]
    loss_full = full_state_nll(residual_full, Lambda, reduction="mean")
    loss_full_batch_first = full_state_nll(
        residual_full.transpose(0, 1), Lambda, reduction="mean", time_first=False
    )
    loss_pose = pose_nll(residual_pose, Lambda, reduction="mean")
    assert torch.isfinite(loss_full)
    assert torch.isfinite(loss_full_batch_first)
    assert torch.isfinite(loss_pose)


def test_nees_matches_quadratic_nll_terms():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    supervised_indices = torch.tensor([0, 2, 5, 7])
    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn(), jitter=1e-8
    )
    Lambda = build_se23_precision(Phi_list, Q_list, prior_std=1e-3, jitter=1e-8)

    residual_full = 1e-3 * torch.randn(4, 2, 9, dtype=R.dtype)
    residual_pose = residual_full[..., [0, 1, 2, 6, 7, 8]]

    full_quad = full_state_nll(
        residual_full,
        Lambda,
        reduction="none",
        time_first=True,
        include_logdet=False,
    )
    full_metric = full_state_nees(
        residual_full,
        Lambda,
        reduction="none",
        time_first=True,
    )
    pose_quad = pose_nll(
        residual_pose,
        Lambda,
        reduction="none",
        time_first=True,
        include_logdet=False,
    )
    pose_metric = pose_nees(
        residual_pose,
        Lambda,
        reduction="none",
        time_first=True,
    )
    sparse_full_quad = sparse_full_state_nll(
        residual_full,
        Phi_list,
        Q_list,
        jitter=1e-8,
        reduction="none",
        time_first=True,
        include_logdet=False,
    )
    sparse_full_metric = sparse_full_state_nees(
        residual_full,
        Phi_list,
        Q_list,
        jitter=1e-8,
        reduction="none",
        time_first=True,
    )
    sparse_pose_quad = sparse_pose_nll(
        residual_pose,
        Phi_list,
        Q_list,
        jitter=1e-8,
        reduction="none",
        time_first=True,
        include_logdet=False,
    )
    sparse_pose_metric = sparse_pose_nees(
        residual_pose,
        Phi_list,
        Q_list,
        jitter=1e-8,
        reduction="none",
        time_first=True,
    )

    assert torch.allclose(full_metric, 2.0 * full_quad, atol=1e-8, rtol=1e-8)
    assert torch.allclose(pose_metric, 2.0 * pose_quad, atol=1e-8, rtol=1e-8)
    assert torch.allclose(sparse_full_metric, 2.0 * sparse_full_quad, atol=1e-8, rtol=1e-8)
    assert torch.allclose(sparse_pose_metric, 2.0 * sparse_pose_quad, atol=1e-8, rtol=1e-8)


def test_nll_gradients():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    residual = torch.nn.Parameter(1e-3 * torch.randn(4, 2, 9, dtype=R.dtype))
    supervised_indices = torch.tensor([0, 2, 5, 7])

    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn(), jitter=1e-8
    )
    Lambda = build_se23_precision(Phi_list, Q_list, prior_std=1e-3, jitter=1e-8)
    loss = full_state_nll(residual, Lambda, reduction="mean")
    loss.backward()

    assert residual.grad is not None
    assert noise.log_sigma_a.grad is not None
    assert noise.log_sigma_g.grad is not None


def test_sparse_full_state_nll_finite_and_gradients():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    residual = torch.nn.Parameter(1e-3 * torch.randn(4, 2, 9, dtype=R.dtype))
    supervised_indices = torch.tensor([0, 2, 5, 7])

    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn(), jitter=1e-8
    )
    loss = sparse_full_state_nll(
        residual,
        Phi_list,
        Q_list,
        jitter=1e-8,
        reduction="mean",
        time_first=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert residual.grad is not None
    assert noise.log_sigma_a.grad is not None
    assert noise.log_sigma_g.grad is not None


def test_sparse_pose_nll_matches_dense_woodbury():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    residual = torch.nn.Parameter(1e-3 * torch.randn(4, 2, 6, dtype=R.dtype))
    supervised_indices = torch.tensor([0, 2, 5, 7])

    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn(), jitter=1e-8
    )
    Lambda = build_se23_precision(Phi_list, Q_list, prior_std=1e-3, jitter=1e-8)
    dense = pose_nll(
        residual,
        Lambda,
        meas_sigma_rot=1e-2,
        meas_sigma_pos=5e-2,
        jitter=1e-8,
        reduction="mean",
        time_first=True,
    )
    sparse = sparse_pose_nll(
        residual,
        Phi_list,
        Q_list,
        meas_sigma_rot=1e-2,
        meas_sigma_pos=5e-2,
        prior_std=1e-3,
        jitter=1e-8,
        reduction="mean",
        time_first=True,
    )
    sparse.backward()

    assert torch.isfinite(sparse)
    assert torch.allclose(sparse, dense, atol=1e-4, rtol=1e-4)
    assert residual.grad is not None
    assert noise.log_sigma_a.grad is not None
    assert noise.log_sigma_g.grad is not None


def main():
    test_phi_and_gd_shapes()
    test_precision_spd_and_nll_finite()
    test_nees_matches_quadratic_nll_terms()
    test_nll_gradients()
    test_sparse_full_state_nll_finite_and_gradients()
    test_sparse_pose_nll_matches_dense_woodbury()
    print("PLATO NLL smoke tests passed")


if __name__ == "__main__":
    main()
