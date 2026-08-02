from __future__ import annotations

import os
import sys

import torch

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)

try:
    import Interpolation
    import lie_algebra as Lie
except ImportError:
    from BiasDy import Interpolation
    from BiasDy import lie_algebra as Lie

try:
    from plato.supervision import (
        PoseTrajectory,
        align_vo_first_gt,
        build_supervised_indices,
        get_gt_se23_supervision_batch,
        get_vo_pose_supervision_batch,
        interpolate_pose,
        resolve_supervision_stride,
    )
except ImportError:
    from .supervision import (
        PoseTrajectory,
        align_vo_first_gt,
        build_supervised_indices,
        get_gt_se23_supervision_batch,
        get_vo_pose_supervision_batch,
        interpolate_pose,
        resolve_supervision_stride,
    )


class FakeEurocDataset:
    def __init__(self, mode: str = "train", batch_size: int = 5):
        self.dataset_name = "EUROC"
        self._mode = mode
        self._batch_size = batch_size
        self._loss_window = 16
        self._percent_for_val = 0.2
        self.dt = 0.005
        self.sequences = ["MH_fake"]
        T = 160
        t = self.dt * torch.arange(T, dtype=torch.float32)
        phi = torch.stack([0.01 * t, -0.02 * t, 0.005 * t], dim=-1)
        R = Lie.SO3exp(phi)
        v = torch.stack([0.1 + 0.01 * t, -0.2 + 0.02 * t, 0.05 * torch.ones_like(t)], dim=-1)
        p = torch.stack([0.1 * t, -0.2 * t, 0.05 * t], dim=-1)
        u = torch.stack([
            0.01 * torch.ones_like(t),
            -0.02 * torch.ones_like(t),
            0.005 * torch.ones_like(t),
            torch.ones_like(t),
            0.5 * torch.ones_like(t),
            -0.1 * torch.ones_like(t),
        ], dim=-1)
        coeff = tuple(c.float() for c in Interpolation.get_cubic_hermite_coeffs(t, u))
        self._imu = {
            "t_imu": t,
            "u_imu": u,
            "w_bias": torch.zeros(T, 3),
            "a_bias": torch.zeros(T, 3),
            "coeff": coeff,
            "t_start": 1.0e18,
        }
        self._gt = {
            "t_gt": t,
            "R_gt": R,
            "v_gt": v,
            "p_gt": p,
        }

    def load_imu(self, _):
        return self._imu

    def load_gt(self, _):
        return self._gt

    def get_start_timestamp(self, _):
        return self._imu["t_start"]


def test_gt_batch_shapes():
    dataset = FakeEurocDataset(mode="train", batch_size=4)
    batch = get_gt_se23_supervision_batch(dataset, 0, window_steps=20, batch_size=4)
    assert batch.mode == "gt_se23"
    assert batch.X_state.shape == (20, 4, 5, 5)
    assert batch.X_ref.shape == (20, 4, 5, 5)
    assert batch.supervised_indices.shape == (20,)


def test_vo_sampling_shapes():
    dataset = FakeEurocDataset(mode="train", batch_size=3)
    gt = dataset.load_gt(0)
    vo_traj = PoseTrajectory(t=gt["t_gt"][::10], R=gt["R_gt"][::10], p=gt["p_gt"][::10])
    batch = get_vo_pose_supervision_batch(
        dataset,
        0,
        vo_traj,
        window_steps=6,
        sup_stride=10,
        batch_size=3,
    )
    assert batch.mode == "vo_pose"
    assert batch.X_state.shape == (51, 3, 5, 5)
    assert batch.R_ref.shape[-2:] == (3, 3)
    assert batch.p_ref.shape[-1] == 3
    assert batch.supervised_indices[-1].item() == 50


def test_vo_alignment_and_interpolation():
    dataset = FakeEurocDataset(mode="train", batch_size=2)
    gt = dataset.load_gt(0)
    R_offset = Lie.SO3exp(torch.tensor([0.1, -0.05, 0.02]))
    p_offset = torch.tensor([1.0, -0.2, 0.4])
    R_vo = torch.matmul(R_offset.transpose(-1, -2).unsqueeze(0), gt["R_gt"])
    p_vo = torch.matmul(
        R_offset.transpose(-1, -2),
        (gt["p_gt"] - p_offset).unsqueeze(-1),
    ).squeeze(-1)
    vo_traj = PoseTrajectory(t=gt["t_gt"], R=R_vo, p=p_vo)
    aligned = align_vo_first_gt(vo_traj, PoseTrajectory(t=gt["t_gt"], R=gt["R_gt"], p=gt["p_gt"]))
    R_eval, p_eval = interpolate_pose(gt["t_gt"][5:10], aligned)
    assert (p_eval - gt["p_gt"][5:10]).abs().max() < 1e-5
    assert Lie.SO3log(torch.matmul(gt["R_gt"][5:10].transpose(-1, -2), R_eval)).abs().max() < 1e-5


def test_stride_helpers():
    assert resolve_supervision_stride(0.005, sup_rate_hz=5.0) == 40
    assert build_supervised_indices(400, 40)[-1].item() == 399


def main():
    test_gt_batch_shapes()
    test_vo_sampling_shapes()
    test_vo_alignment_and_interpolation()
    test_stride_helpers()
    print("PLATO supervision smoke tests passed")


if __name__ == "__main__":
    main()
