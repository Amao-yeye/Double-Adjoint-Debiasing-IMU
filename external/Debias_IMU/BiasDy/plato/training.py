from __future__ import annotations

import os
import sys
import json
from dataclasses import asdict, dataclass, replace

import torch
from torch.utils.tensorboard import SummaryWriter

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBIAS_ROOT = os.path.abspath(os.path.join(BIASDY_DIR, ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)
if DEBIAS_ROOT not in sys.path:
    sys.path.insert(0, DEBIAS_ROOT)

import Interpolation
import lie_algebra as Lie
from dataset import BaseDataset
from network_for_BiadDy import VFSE23vbiasBase, ba_func_choose, bw_func_choose
from SO3diffeq import odeint_SO3

from plato.analytic_gradient import adjoint_bias_gradient, invariant_transition_matrices, weighted_se23_mse_state_gradient
from plato.bias_adjoint import accumulate_ba_euler_adjoint_gradients
from plato.covariance import build_se23_precision, preintegrate_supervision_covariances
from plato.gyro_adjoint import accumulate_bg_euler_adjoint_gradients
from plato.losses import (
    pose_block_mse,
    pose_global_residual,
    pose_mse_loss,
    pose_right_residual,
    relative_se23_residual,
    relative_pose_residual,
    se23_block_mse,
    se23_global_residual,
    se23_gt_right_residual,
    se23_log_residual,
    se23_mse_loss,
    se23_pose_residual,
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
from plato.so3_analytic_gradient import so3_gyro_bias_gradient
from plato.supervision import (
    PoseTrajectory,
    get_gt_se23_supervision_batch,
    get_vo_pose_supervision_batch,
    get_vo_se23_supervision_batch,
)


@dataclass
class PlatoTrainConfig:
    supervision: str
    window_steps: int
    sup_stride: int | None
    sup_rate_hz: float | None
    integral_method: str
    batch_size: int
    outputdir: str
    network_type_Gyro: str
    network_type_Acc: str
    meas_sigma_rot_gt: float
    meas_sigma_vel_gt: float
    meas_sigma_pos_gt: float
    meas_sigma_rot_vo: float
    meas_sigma_pos_vo: float
    init_sigma_a: float
    init_sigma_g: float
    plato_noise_isotropic: bool
    nll_min_sigma_a: float | None
    nll_max_sigma_a: float | None
    nll_min_sigma_g: float | None
    nll_max_sigma_g: float | None
    cov_jitter: float
    prior_std: float
    val_freq: int
    save_freq: int
    device: str
    train_sequence_strategy: str = "random"
    mse_rot_weight: float = 1.0
    mse_vel_weight: float = 1.0
    mse_pos_weight: float = 1.0
    mse_acc_rot_weight: float = 0.0
    mse_acc_vel_weight: float = 1.0
    mse_acc_pos_weight: float = 1.0
    mse_gyro_full_state: bool = False
    gyro_train_with_acc: bool = False
    mse_residual_mode: str = "log"
    gyro_multi_window_steps: tuple[int, ...] = ()
    gyro_multi_window_weights: tuple[float, ...] = ()
    nll_gt_mode: str = "woodbury"
    nll_gt_residual_mode: str = "right"
    nll_vo_mode: str = "woodbury"
    nll_mse_anchor_weight: float = 0.0
    vo_residual_mode: str = "absolute"
    vo_init_mode: str = "gt"
    gradient_mode: str = "autograd"
    analytic_grad_scale: float = 1.0
    analytic_grad_clip_norm: float = 0.0
    analytic_residual_jacobian: str = "bch"
    train_seqs: tuple[str, ...] = ()
    val_seqs: tuple[str, ...] = ()
    test_seqs: tuple[str, ...] = ()


def train_mse_gyro(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    lr: float,
    weight_decay: float,
    epoch_count: int,
    gyro_checkpoint: str | None = None,
    acc_checkpoint: str | None = None,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "Gyro_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "Gyro_logs"))
    _reset_peak_memory(device)

    if gyro_checkpoint is None:
        bw_func = bw_func_choose(config.network_type_Gyro, device)
    else:
        bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    if config.gyro_train_with_acc:
        if acc_checkpoint is None:
            raise ValueError("acc_checkpoint is required when gyro_train_with_acc=True")
        ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
        model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
        model.set_ba_zero(False)
        model.freeze_func_ba()
    else:
        model = VFSE23vbiasBase(biasfunc_w=bw_func, device=device).to(device)
        model.set_ba_zero(True)
    model.set_bw_zero(False)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    best_loss = float("inf")
    last_loss_total = float("inf")

    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        if config.gradient_mode == "autograd":
            loss, metrics, analytic_grad_norm = _train_gyro_mse_step(
                model,
                dataset_train,
                config,
                vo_train,
                step=epoch,
                writer=writer,
            )
        elif config.gradient_mode == "double_adjoint":
            loss, metrics, analytic_grad_norm = _train_gyro_mse_double_adjoint_step(
                model,
                dataset_train,
                config,
                vo_train,
                step=epoch,
                writer=writer,
            )
        else:
            raise ValueError("gradient_mode must be autograd or double_adjoint")
        optimizer.step()
        scheduler.step()

        _write_metrics(writer, "train", metrics, loss, epoch)
        if analytic_grad_norm is not None:
            writer.add_scalar("train/analytic_grad_bg_norm", analytic_grad_norm.item(), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % config.val_freq == 0:
            last_loss_total, val_window_losses = _validate_gyro_mse(
                model, dataset_val, config, vo_by_seq=vo_val
            )
            writer.add_scalar("val_loss", last_loss_total, epoch)
            for window_steps, window_loss in val_window_losses.items():
                writer.add_scalar(f"val_w{window_steps}/loss", window_loss, epoch)
            print(f"epoch: {epoch}, train_loss: {loss.item()}, val_loss: {last_loss_total}")
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "mse_gyro",
                    epoch,
                    last_loss_total,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "mse_gyro",
                epoch,
                last_loss_total,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "mse_gyro", device)
    final_epoch = max(0, epoch_count - 1)
    final_path = os.path.join(model_save_path, f"model_epoch_{final_epoch}.pt")
    if epoch_count > 0 and not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "mse_gyro",
            final_epoch,
            last_loss_total,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return final_path


def _train_gyro_mse_step(
    model,
    dataset_train: BaseDataset,
    config: PlatoTrainConfig,
    vo_train: dict[str, PoseTrajectory] | None,
    step: int,
    writer: SummaryWriter,
):
    window_steps, window_weights = _gyro_window_schedule(config)
    total_loss = None
    total_metrics: dict[str, torch.Tensor] = {}
    total_weight = 0.0

    for window, weight in zip(window_steps, window_weights):
        window_config = replace(config, window_steps=window)
        batch = _sample_batch(dataset_train, window_config, vo_train, step=step)
        solution, R_sol = _integrate_batch(model, dataset_train, batch, window_config)
        loss, metrics = _mse_loss_for_stage(batch, solution, R_sol, window_config, stage="gyro")
        scaled_loss = loss * weight
        scaled_loss.backward()

        detached_loss = scaled_loss.detach()
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        total_weight += float(weight)
        for name, value in metrics.items():
            weighted_value = value.detach() * float(weight)
            total_metrics[name] = total_metrics.get(name, torch.zeros_like(weighted_value)) + weighted_value

        if len(window_steps) > 1:
            _write_metrics(writer, f"train_w{window}", metrics, loss.detach(), step)

    if total_loss is None:
        raise ValueError("No gyro supervision windows configured")
    averaged_metrics = {
        name: value / max(total_weight, 1e-12)
        for name, value in total_metrics.items()
    }
    return total_loss, averaged_metrics, None


def _train_gyro_mse_double_adjoint_step(
    model,
    dataset_train: BaseDataset,
    config: PlatoTrainConfig,
    vo_train: dict[str, PoseTrajectory] | None,
    step: int,
    writer: SummaryWriter,
):
    window_steps, window_weights = _gyro_window_schedule(config)
    total_loss = None
    total_metrics: dict[str, torch.Tensor] = {}
    total_weight = 0.0
    grad_norm_sq = torch.zeros((), device=config.device)

    for window, weight in zip(window_steps, window_weights):
        window_config = replace(config, window_steps=window)
        batch = _sample_batch(dataset_train, window_config, vo_train, step=step)
        loss, metrics, grad_norm = _train_gyro_mse_double_adjoint_window(
            model,
            dataset_train,
            batch,
            window_config,
            weight=float(weight),
        )

        detached_loss = loss.detach()
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        total_weight += float(weight)
        for name, value in metrics.items():
            weighted_value = value.detach() * float(weight)
            total_metrics[name] = total_metrics.get(name, torch.zeros_like(weighted_value)) + weighted_value
        grad_norm_sq = grad_norm_sq + grad_norm.square()

        if len(window_steps) > 1:
            _write_metrics(writer, f"train_w{window}", metrics, loss.detach(), step)
            writer.add_scalar(f"train_w{window}/analytic_grad_bg_norm", grad_norm.item(), step)

    if total_loss is None:
        raise ValueError("No gyro supervision windows configured")
    averaged_metrics = {
        name: value / max(total_weight, 1e-12)
        for name, value in total_metrics.items()
    }
    return total_loss, averaged_metrics, grad_norm_sq.sqrt().detach()


def _train_gyro_mse_double_adjoint_window(
    model,
    dataset_train: BaseDataset,
    batch,
    config: PlatoTrainConfig,
    weight: float = 1.0,
):
    if config.supervision != "gt_se23":
        raise NotImplementedError("gyro double_adjoint currently supports only gt_se23 supervision")
    if config.integral_method != "euler":
        raise NotImplementedError("gyro double_adjoint currently supports only integral_method='euler'")
    if config.mse_residual_mode != "right":
        raise NotImplementedError("gyro double_adjoint requires mse_residual_mode='right'")

    sup = batch.supervised_indices
    dense_sup = torch.arange(batch.t_odeint.shape[0], device=sup.device, dtype=sup.dtype)
    if sup.shape != dense_sup.shape or not torch.equal(sup, dense_sup):
        raise NotImplementedError("gyro double_adjoint currently requires dense GT supervision over the whole window")

    with torch.no_grad():
        solution, R_sol = _integrate_batch(model, dataset_train, batch, config)
        loss_unweighted, metrics = _mse_loss_for_stage(batch, solution, R_sol, config, stage="gyro")

    R_ref = batch.X_ref[..., :3, :3]
    analytic = so3_gyro_bias_gradient(
        R_sol.detach(),
        R_ref.detach(),
        t=batch.t_odeint.detach(),
        rot_weight=config.mse_rot_weight,
        residual_jacobian=config.analytic_residual_jacobian,
        invariant_error="right",
    )
    grad_bg = (
        1e6
        * float(weight)
        * float(config.analytic_grad_scale)
        * analytic.bias_gradient
    )
    if config.analytic_grad_clip_norm > 0:
        grad_norm = grad_bg.norm()
        if grad_norm > config.analytic_grad_clip_norm:
            grad_bg = grad_bg * (config.analytic_grad_clip_norm / (grad_norm + 1e-12))

    accumulate_bg_euler_adjoint_gradients(
        model,
        solution,
        batch.t_odeint,
        grad_bg,
        jump_indices=torch.arange(grad_bg.shape[0], device=grad_bg.device),
    )
    return (float(weight) * loss_unweighted).detach(), metrics, grad_bg.norm().detach()


def train_mse_acc(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    gyro_checkpoint: str,
    lr: float,
    weight_decay: float,
    epoch_count: int,
    acc_checkpoint: str | None = None,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "Acc_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "Acc_logs"))
    _reset_peak_memory(device)

    bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    if acc_checkpoint is None:
        ba_func = ba_func_choose(config.network_type_Acc, device)
    else:
        ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.freeze_func_bw()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    best_loss = float("inf")
    last_loss_total = float("inf")

    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        batch = _sample_batch(dataset_train, config, vo_train, step=epoch)
        if config.gradient_mode == "autograd":
            solution, R_sol = _integrate_batch(model, dataset_train, batch, config)
            loss, metrics = _mse_loss_for_stage(batch, solution, R_sol, config, stage="acc")
            loss.backward()
            analytic_grad_norm = None
        elif config.gradient_mode == "double_adjoint":
            loss, metrics, analytic_grad_norm = _train_acc_mse_double_adjoint_step(
                model,
                dataset_train,
                batch,
                config,
            )
        else:
            raise ValueError("gradient_mode must be autograd or double_adjoint")
        optimizer.step()
        scheduler.step()

        _write_metrics(writer, "train", metrics, loss, epoch)
        if analytic_grad_norm is not None:
            writer.add_scalar("train/analytic_grad_ba_norm", analytic_grad_norm.item(), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % config.val_freq == 0:
            last_loss_total = _validate_mse(model, dataset_val, config, stage="acc", vo_by_seq=vo_val)
            writer.add_scalar("val_loss", last_loss_total, epoch)
            print(f"epoch: {epoch}, train_loss: {loss.item()}, val_loss: {last_loss_total}")
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "mse_acc",
                    epoch,
                    last_loss_total,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "mse_acc",
                epoch,
                last_loss_total,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "mse_acc", device)
    final_epoch = max(0, epoch_count - 1)
    final_path = os.path.join(model_save_path, f"model_epoch_{final_epoch}.pt")
    if epoch_count > 0 and not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "mse_acc",
            final_epoch,
            last_loss_total,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return final_path


def _train_acc_mse_double_adjoint_step(
    model,
    dataset_train: BaseDataset,
    batch,
    config: PlatoTrainConfig,
):
    if config.supervision != "gt_se23":
        raise NotImplementedError("double_adjoint currently supports only gt_se23 supervision")
    if config.integral_method != "euler":
        raise NotImplementedError("double_adjoint currently supports only integral_method='euler'")
    if config.mse_residual_mode != "right":
        raise NotImplementedError("double_adjoint requires mse_residual_mode='right'")

    sup = batch.supervised_indices
    dense_sup = torch.arange(batch.t_odeint.shape[0], device=sup.device, dtype=sup.dtype)
    if sup.shape != dense_sup.shape or not torch.equal(sup, dense_sup):
        raise NotImplementedError("double_adjoint currently requires dense GT supervision over the whole window")

    with torch.no_grad():
        solution, R_sol = _integrate_batch(model, dataset_train, batch, config)

    residual = se23_gt_right_residual(batch.X_ref, R_sol, solution)
    blocks = _mse_blocks_for_stage("acc", full_state=True, config=config)
    rot_weight, vel_weight, pos_weight = _mse_weights_for_stage("acc", config)
    loss = 1e6 * se23_mse_loss(
        residual,
        rot_weight=rot_weight,
        vel_weight=vel_weight,
        pos_weight=pos_weight,
        blocks=blocks,
    )
    metrics = se23_block_mse(residual)

    state_grad = weighted_se23_mse_state_gradient(
        residual.detach(),
        rot_weight=rot_weight,
        vel_weight=vel_weight,
        pos_weight=pos_weight,
        blocks=blocks,
        residual_jacobian=config.analytic_residual_jacobian,
    )
    F, G = invariant_transition_matrices(
        R_sol.detach(),
        solution[..., 7:10].detach(),
        solution[..., 10:13].detach(),
        t=batch.t_odeint.detach(),
    )
    analytic = adjoint_bias_gradient(state_grad, F, G, detach=True)
    # adjoint_bias_gradient returns [ba, bg], matching se23_Gd columns [n_a, n_g].
    grad_ba = 1e6 * float(config.analytic_grad_scale) * analytic.bias_gradient[..., 0:3]
    if config.analytic_grad_clip_norm > 0:
        grad_norm = grad_ba.norm()
        if grad_norm > config.analytic_grad_clip_norm:
            grad_ba = grad_ba * (config.analytic_grad_clip_norm / (grad_norm + 1e-12))

    accumulate_ba_euler_adjoint_gradients(
        model,
        solution,
        batch.t_odeint,
        grad_ba,
        jump_indices=torch.arange(grad_ba.shape[0], device=grad_ba.device),
    )
    return loss.detach(), metrics, grad_ba.norm().detach()


def train_mse_joint(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    gyro_checkpoint: str,
    acc_checkpoint: str,
    lr: float,
    weight_decay: float,
    epoch_count: int,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "Joint_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "Joint_logs"))
    _reset_peak_memory(device)

    bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.unfreeze_func_bw()
    model.unfreeze_func_ba()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    best_loss = float("inf")
    last_loss_total = float("inf")

    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        batch = _sample_batch(dataset_train, config, vo_train, step=epoch)
        solution, R_sol = _integrate_batch(model, dataset_train, batch, config)
        loss, metrics = _mse_loss_for_stage(batch, solution, R_sol, config, stage="joint")
        loss.backward()
        optimizer.step()
        scheduler.step()

        _write_metrics(writer, "train", metrics, loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % config.val_freq == 0:
            last_loss_total = _validate_mse(
                model, dataset_val, config, stage="joint", vo_by_seq=vo_val
            )
            writer.add_scalar("val_loss", last_loss_total, epoch)
            print(f"epoch: {epoch}, train_loss: {loss.item()}, val_loss: {last_loss_total}")
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "mse_joint",
                    epoch,
                    last_loss_total,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "mse_joint",
                epoch,
                last_loss_total,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "mse_joint", device)
    final_epoch = max(0, epoch_count - 1)
    final_path = os.path.join(model_save_path, f"model_epoch_{final_epoch}.pt")
    if epoch_count > 0 and not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "mse_joint",
            final_epoch,
            last_loss_total,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return final_path


def train_nll(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    gyro_checkpoint: str,
    acc_checkpoint: str,
    lr: float,
    weight_decay: float,
    epoch_count: int,
    nll_freeze_gyro: bool = False,
    nll_freeze_acc: bool = False,
    noise_checkpoint: str | None = None,
    nll_freeze_covariance: bool = False,
    nll_covariance_only: bool = False,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "NLL_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "NLL_logs"))
    _reset_peak_memory(device)

    bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    if nll_covariance_only:
        for param in model.parameters():
            param.requires_grad = False
    elif nll_freeze_gyro:
        model.freeze_func_bw()
    else:
        model.unfreeze_func_bw()
    if not nll_covariance_only:
        if nll_freeze_acc:
            model.freeze_func_ba()
        else:
            model.unfreeze_func_ba()

    noise_params = PlatoNoiseParams(
        init_sigma_a=config.init_sigma_a,
        init_sigma_g=config.init_sigma_g,
        isotropic=config.plato_noise_isotropic,
        device=device,
    )
    if noise_checkpoint is not None:
        checkpoint = torch.load(noise_checkpoint, map_location=device, weights_only=False)
        if "noise_params" not in checkpoint:
            raise KeyError(f"Checkpoint has no noise_params: {noise_checkpoint}")
        noise_params.load_state_dict(checkpoint["noise_params"])
    noise_params.clamp_sigmas(
        min_sigma_a=config.nll_min_sigma_a,
        max_sigma_a=config.nll_max_sigma_a,
        min_sigma_g=config.nll_min_sigma_g,
        max_sigma_g=config.nll_max_sigma_g,
    )
    if nll_freeze_covariance:
        for param in noise_params.parameters():
            param.requires_grad = False
    if nll_covariance_only and nll_freeze_covariance:
        raise ValueError("nll_covariance_only cannot be combined with nll_freeze_covariance")
    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    trainable += list(filter(lambda p: p.requires_grad, noise_params.parameters()))
    if not trainable:
        raise ValueError("No trainable parameters selected for NLL training")
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    last_loss_total = float("inf")

    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        batch = _sample_batch(dataset_train, config, vo_train, step=epoch)
        if config.gradient_mode == "autograd":
            solution, R_sol = _integrate_batch(model, dataset_train, batch, config)
            loss, metrics = _nll_loss_for_batch(batch, solution, R_sol, config, noise_params)
            loss.backward()
            analytic_grad_norms = {}
        elif config.gradient_mode == "double_adjoint":
            loss, metrics, analytic_grad_norms = _train_nll_double_adjoint_step(
                model,
                dataset_train,
                batch,
                config,
                noise_params,
            )
        else:
            raise ValueError("gradient_mode must be autograd or double_adjoint")
        optimizer.step()
        noise_params.clamp_sigmas(
            min_sigma_a=config.nll_min_sigma_a,
            max_sigma_a=config.nll_max_sigma_a,
            min_sigma_g=config.nll_min_sigma_g,
            max_sigma_g=config.nll_max_sigma_g,
        )

        _write_metrics(writer, "train", metrics, loss, epoch)
        for name, value in analytic_grad_norms.items():
            writer.add_scalar(f"train/{name}", value.item(), epoch)
        writer.add_scalar("train/log_sigma_a", noise_params.log_sigma_a.mean().item(), epoch)
        writer.add_scalar("train/log_sigma_g", noise_params.log_sigma_g.mean().item(), epoch)
        sigma_a, sigma_g = noise_params.sigmas()
        writer.add_scalar("train/sigma_a", sigma_a.mean().item(), epoch)
        writer.add_scalar("train/sigma_g", sigma_g.mean().item(), epoch)

        if epoch % config.val_freq == 0:
            last_loss_total, val_metrics = _validate_nll(
                model, noise_params, dataset_val, config, vo_by_seq=vo_val
            )
            writer.add_scalar("val_loss", last_loss_total, epoch)
            _write_scalar_metrics(writer, "val", val_metrics, epoch)
            print(f"epoch: {epoch}, train_nll: {loss.item()}, val_loss: {last_loss_total}")
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "nll",
                    epoch,
                    last_loss_total,
                    model=model,
                    noise_params=noise_params,
                    optimizer=optimizer,
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "nll",
                epoch,
                last_loss_total,
                model=model,
                noise_params=noise_params,
                optimizer=optimizer,
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "nll", device)
    final_epoch = max(0, epoch_count - 1)
    final_path = os.path.join(model_save_path, f"model_epoch_{final_epoch}.pt")
    if epoch_count > 0 and not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "nll",
            final_epoch,
            last_loss_total,
            model=model,
            noise_params=noise_params,
            optimizer=optimizer,
        )
    return final_path


def _train_nll_double_adjoint_step(
    model,
    dataset_train: BaseDataset,
    batch,
    config: PlatoTrainConfig,
    noise_params: PlatoNoiseParams,
):
    if config.supervision != "gt_se23":
        raise NotImplementedError("NLL double_adjoint currently supports only gt_se23 supervision")
    if config.integral_method != "euler":
        raise NotImplementedError("NLL double_adjoint currently supports only integral_method='euler'")

    with torch.no_grad():
        solution, R_sol = _integrate_batch(model, dataset_train, batch, config)

    delta = torch.zeros(
        *solution.shape[:-1],
        9,
        device=solution.device,
        dtype=solution.dtype,
        requires_grad=True,
    )
    R_perturbed, solution_perturbed = _apply_se23_left_perturbation(
        R_sol.detach(),
        solution.detach(),
        delta,
    )
    loss, metrics = _nll_loss_for_batch(
        batch,
        solution_perturbed,
        R_perturbed,
        config,
        noise_params,
    )
    loss.backward()
    if delta.grad is None:
        raise RuntimeError("NLL double_adjoint failed to produce a state gradient")

    state_grad = delta.grad.detach()
    train_ba = _has_trainable_parameters(model.biasfunc_a)
    train_bg = _has_trainable_parameters(model.biasfunc_w)
    grad_norms = _accumulate_se23_bias_adjoint_gradients(
        model,
        solution,
        R_sol,
        batch.t_odeint,
        state_grad,
        config,
        train_ba=train_ba,
        train_bg=train_bg,
    )
    metrics = _detach_metrics(metrics)
    return loss.detach(), metrics, grad_norms


def _apply_se23_left_perturbation(
    R_pred: torch.Tensor,
    solution: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_pred = Lie.SEn3fromSO3Rn(
        R_pred,
        torch.cat([solution[..., 7:10], solution[..., 10:13]], dim=-1),
    )
    X_perturbed = torch.matmul(Lie.SEn3exp(delta), X_pred)
    solution_perturbed = solution.clone()
    solution_perturbed[..., 7:10] = X_perturbed[..., :3, 3]
    solution_perturbed[..., 10:13] = X_perturbed[..., :3, 4]
    return X_perturbed[..., :3, :3], solution_perturbed


def _accumulate_se23_bias_adjoint_gradients(
    model,
    solution: torch.Tensor,
    R_sol: torch.Tensor,
    t_odeint: torch.Tensor,
    state_grad: torch.Tensor,
    config: PlatoTrainConfig,
    train_ba: bool,
    train_bg: bool,
) -> dict[str, torch.Tensor]:
    F, G = invariant_transition_matrices(
        R_sol.detach(),
        solution[..., 7:10].detach(),
        solution[..., 10:13].detach(),
        t=t_odeint.detach(),
    )
    analytic = adjoint_bias_gradient(state_grad, F, G, detach=True)
    grad_norms: dict[str, torch.Tensor] = {}

    if train_ba:
        grad_ba = float(config.analytic_grad_scale) * analytic.bias_gradient[..., 0:3]
        grad_ba = _clip_adjoint_jump_gradient(grad_ba, config)
        accumulate_ba_euler_adjoint_gradients(
            model,
            solution,
            t_odeint,
            grad_ba,
            jump_indices=torch.arange(grad_ba.shape[0], device=grad_ba.device),
        )
        grad_norms["analytic_grad_ba_norm"] = grad_ba.norm().detach()

    if train_bg:
        grad_bg = float(config.analytic_grad_scale) * analytic.bias_gradient[..., 3:6]
        grad_bg = _clip_adjoint_jump_gradient(grad_bg, config)
        accumulate_bg_euler_adjoint_gradients(
            model,
            solution,
            t_odeint,
            grad_bg,
            jump_indices=torch.arange(grad_bg.shape[0], device=grad_bg.device),
        )
        grad_norms["analytic_grad_bg_norm"] = grad_bg.norm().detach()

    return grad_norms


def _clip_adjoint_jump_gradient(gradient: torch.Tensor, config: PlatoTrainConfig) -> torch.Tensor:
    if config.analytic_grad_clip_norm <= 0:
        return gradient
    grad_norm = gradient.norm()
    if grad_norm > config.analytic_grad_clip_norm:
        return gradient * (config.analytic_grad_clip_norm / (grad_norm + 1e-12))
    return gradient


def _has_trainable_parameters(module) -> bool:
    return module is not None and any(parameter.requires_grad for parameter in module.parameters())


def _detach_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in metrics.items()}


def _add_nees_metrics(metrics: dict[str, torch.Tensor], nees: torch.Tensor, dof: int) -> None:
    dof_tensor = torch.as_tensor(float(dof), device=nees.device, dtype=nees.dtype)
    metrics["nees"] = nees.detach()
    metrics["nees_dof"] = dof_tensor
    metrics["nees_per_dof"] = (nees.detach() / dof_tensor).detach()


def _reset_peak_memory(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _write_peak_memory_summary(outputdir: str, stage: str, device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    path = os.path.join(outputdir, "memory_summary.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {}
    summary[stage] = {
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024 ** 2,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def load_bw_func(network_type: str, checkpoint_path: str, device: str):
    bw_func = bw_func_choose(network_type, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bw_func.load_state_dict(checkpoint["func_bw_model"])
    return bw_func


def load_ba_func(network_type: str, checkpoint_path: str, device: str):
    ba_func = ba_func_choose(network_type, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ba_func.load_state_dict(checkpoint["func_ba_model"])
    return ba_func


def _sample_batch(
    dataset: BaseDataset,
    config: PlatoTrainConfig,
    vo_by_seq: dict[str, PoseTrajectory] | None,
    step: int | None = None,
):
    if dataset._mode == "train":
        if config.train_sequence_strategy == "cycle":
            index = int(step or 0) % dataset.length()
        elif config.train_sequence_strategy == "random":
            index = torch.randint(0, dataset.length(), (1,)).item()
        else:
            raise ValueError("train_sequence_strategy must be random or cycle")
    else:
        index = 0
    return _get_batch_for_index(dataset, index, config, vo_by_seq)


def _get_batch_for_index(
    dataset: BaseDataset,
    index: int,
    config: PlatoTrainConfig,
    vo_by_seq: dict[str, PoseTrajectory] | None,
):
    if config.supervision == "gt_se23":
        mask_flag = {"EUROC": False, "TUM": True, "Fetch": False, "AquaLux": True}.get(dataset.dataset_name, False)
        return get_gt_se23_supervision_batch(
            dataset,
            index,
            device=config.device,
            window_steps=config.window_steps,
            batch_size=config.batch_size,
            mask_flag=mask_flag,
        )
    if config.supervision in ("vo_pose", "vo_se23"):
        mask_flag = False
        if vo_by_seq is None:
            raise ValueError(f"vo_by_seq is required for {config.supervision} supervision")
        seq = dataset.sequences[index]
        batch_fn = (
            get_vo_se23_supervision_batch
            if config.supervision == "vo_se23"
            else get_vo_pose_supervision_batch
        )
        return batch_fn(
            dataset,
            index,
            vo_by_seq[seq],
            device=config.device,
            window_steps=config.window_steps,
            sup_stride=config.sup_stride,
            sup_rate_hz=config.sup_rate_hz,
            batch_size=config.batch_size,
            mask_flag=mask_flag,
            vo_init_mode=config.vo_init_mode,
        )
    raise ValueError("supervision must be one of: gt_se23, vo_pose, vo_se23")


def _integrate_batch(model, dataset: BaseDataset, batch, config: PlatoTrainConfig):
    y0_batch, R0_batch = dataset.construct_init_forodeint(
        batch.t_state, batch.X_state, batch.bwa_state
    )
    spline = Interpolation.CubicHermiteSpline(batch.spline_time, batch.coeff, device=config.device)
    model.set_u_func(spline.evaluate)
    model.set_u_dot_func(spline.derivative)
    model.set_R0(R0_batch)
    if hasattr(model, "set_imu_window_context"):
        model.set_imu_window_context(
            float(dataset.dt),
            t_min=batch.spline_time[0],
            t_max=batch.spline_time[-1],
        )
    if model.biasfunc_a is not None and hasattr(model.biasfunc_a, "set_initial_ba"):
        model.biasfunc_a.set_initial_ba(y0_batch[..., 13:16])
    return odeint_SO3(
        model,
        y0_batch,
        R0_batch,
        batch.t_odeint,
        rtol=1e-7,
        atol=1e-9,
        method=config.integral_method,
    )


def _mse_loss_for_stage(batch, solution, R_sol, config: PlatoTrainConfig, stage: str):
    sup = batch.supervised_indices
    if config.supervision in ("gt_se23", "vo_se23"):
        residual = _se23_residual_for_config(batch, solution, R_sol, config)
        blocks = _mse_blocks_for_stage(stage, full_state=True, config=config)
        rot_weight, vel_weight, pos_weight = _mse_weights_for_stage(stage, config)
        loss = 1e6 * se23_mse_loss(
            residual,
            rot_weight=rot_weight,
            vel_weight=vel_weight,
            pos_weight=pos_weight,
            blocks=blocks,
        )
        metrics = se23_block_mse(residual)
        return loss, metrics

    residual = _pose_residual_for_config(batch, solution, R_sol, config)
    blocks = _mse_blocks_for_stage(stage, full_state=False, config=config)
    rot_weight, _, pos_weight = _mse_weights_for_stage(stage, config)
    loss = 1e6 * pose_mse_loss(
        residual,
        rot_weight=rot_weight,
        pos_weight=pos_weight,
        blocks=blocks,
    )
    metrics = pose_block_mse(residual)
    return loss, metrics


def _nll_loss_for_batch(batch, solution, R_sol, config: PlatoTrainConfig, noise_params):
    sup = batch.supervised_indices
    Phi_list, Q_list = preintegrate_supervision_covariances(
        R_sol,
        solution[..., 7:10],
        solution[..., 10:13],
        batch.t_odeint,
        sup,
        noise_params.Qn(),
        jitter=config.cov_jitter,
    )
    if config.supervision == "gt_se23":
        if config.nll_gt_mode == "sparse":
            residual = _sparse_gt_nll_residual(batch, solution, R_sol, config)
            loss = sparse_full_state_nll(
                residual,
                Phi_list,
                Q_list,
                prior_std=config.prior_std,
                include_prior=False,
                include_logdet=True,
                jitter=config.cov_jitter,
                reduction="mean",
                time_first=True,
            )
            nees = sparse_full_state_nees(
                residual,
                Phi_list,
                Q_list,
                prior_std=config.prior_std,
                include_prior=False,
                jitter=config.cov_jitter,
                reduction="mean",
                time_first=True,
            )
            nees_dof = 9 * int(Phi_list.shape[0])
        elif config.nll_gt_mode == "woodbury":
            X_ref = _se23_ref_at_supervision(batch)
            residual = se23_log_residual(X_ref, R_sol[sup], solution[sup])
            Lambda = build_se23_precision(
                Phi_list, Q_list, prior_std=config.prior_std, jitter=config.cov_jitter
            )
            loss = full_state_nll(
                residual,
                Lambda,
                meas_sigma_rot=config.meas_sigma_rot_gt,
                meas_sigma_vel=config.meas_sigma_vel_gt,
                meas_sigma_pos=config.meas_sigma_pos_gt,
                jitter=config.cov_jitter,
                reduction="mean",
                time_first=True,
            )
            nees = full_state_nees(
                residual,
                Lambda,
                meas_sigma_rot=config.meas_sigma_rot_gt,
                meas_sigma_vel=config.meas_sigma_vel_gt,
                meas_sigma_pos=config.meas_sigma_pos_gt,
                jitter=config.cov_jitter,
                reduction="mean",
                time_first=True,
            )
            nees_dof = 9 * int(residual.shape[0])
        else:
            raise ValueError("nll_gt_mode must be woodbury or sparse")
        metrics = se23_block_mse(residual)
        metrics["nll"] = loss.detach()
        _add_nees_metrics(metrics, nees, nees_dof)
        if config.nll_mse_anchor_weight > 0:
            anchor_residual = _se23_residual_for_config(batch, solution, R_sol, config)
            anchor = 1e6 * se23_mse_loss(
                anchor_residual,
                rot_weight=config.mse_rot_weight,
                vel_weight=config.mse_vel_weight,
                pos_weight=config.mse_pos_weight,
                blocks=("rot", "vel", "pos"),
            )
            metrics["mse_anchor"] = anchor.detach()
            loss = loss + config.nll_mse_anchor_weight * anchor
        return loss, metrics

    residual = _pose_residual_for_config(batch, solution, R_sol, config)
    if config.vo_residual_mode == "relative":
        if Phi_list.shape[0] < 2:
            raise ValueError("relative VO NLL needs at least three supervised states")
        P0 = Q_list[0]
        Phi_for_nll = Phi_list[1:]
        Q_for_nll = Q_list[1:]
    else:
        P0 = None
        Phi_for_nll = Phi_list
        Q_for_nll = Q_list
    if config.nll_vo_mode == "sparse":
        loss = sparse_pose_nll(
            residual,
            Phi_for_nll,
            Q_for_nll,
            meas_sigma_rot=config.meas_sigma_rot_vo,
            meas_sigma_pos=config.meas_sigma_pos_vo,
            prior_std=config.prior_std,
            P0=P0,
            include_logdet=True,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
        nees = sparse_pose_nees(
            residual,
            Phi_for_nll,
            Q_for_nll,
            meas_sigma_rot=config.meas_sigma_rot_vo,
            meas_sigma_pos=config.meas_sigma_pos_vo,
            prior_std=config.prior_std,
            P0=P0,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
    elif config.nll_vo_mode == "woodbury":
        Lambda = build_se23_precision(
            Phi_for_nll, Q_for_nll, prior_std=config.prior_std, jitter=config.cov_jitter
        )
        loss = pose_nll(
            residual,
            Lambda,
            meas_sigma_rot=config.meas_sigma_rot_vo,
            meas_sigma_pos=config.meas_sigma_pos_vo,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
        nees = pose_nees(
            residual,
            Lambda,
            meas_sigma_rot=config.meas_sigma_rot_vo,
            meas_sigma_pos=config.meas_sigma_pos_vo,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
    else:
        raise ValueError("nll_vo_mode must be woodbury or sparse")
    metrics = pose_block_mse(residual)
    metrics["nll"] = loss.detach()
    _add_nees_metrics(metrics, nees, 6 * int(residual.shape[0]))
    if config.nll_mse_anchor_weight > 0:
        anchor_residual = _pose_residual_for_config(batch, solution, R_sol, config)
        anchor = 1e6 * pose_mse_loss(
            anchor_residual,
            rot_weight=config.mse_rot_weight,
            pos_weight=config.mse_pos_weight,
            blocks=("rot", "pos"),
        )
        metrics["mse_anchor"] = anchor.detach()
        loss = loss + config.nll_mse_anchor_weight * anchor
    return loss, metrics


def _sparse_gt_nll_residual(batch, solution, R_sol, config: PlatoTrainConfig):
    sup = batch.supervised_indices
    X_ref = _se23_ref_at_supervision(batch)
    if config.nll_gt_residual_mode == "right":
        return se23_gt_right_residual(X_ref, R_sol[sup], solution[sup])
    if config.nll_gt_residual_mode == "log":
        return se23_log_residual(X_ref, R_sol[sup], solution[sup])
    if config.nll_gt_residual_mode == "hybrid_global":
        right_residual = se23_gt_right_residual(X_ref, R_sol[sup], solution[sup])
        global_residual = se23_global_residual(X_ref, R_sol[sup], solution[sup])
        return torch.cat((right_residual[..., :3], -global_residual[..., 3:]), dim=-1)
    raise ValueError("nll_gt_residual_mode must be right, log, or hybrid_global")


def _se23_residual_for_config(batch, solution, R_sol, config: PlatoTrainConfig):
    sup = batch.supervised_indices
    X_ref = _se23_ref_at_supervision(batch)
    if config.supervision == "vo_se23" and config.vo_residual_mode == "relative":
        return relative_se23_residual(X_ref, R_sol[sup], solution[sup], mode=config.mse_residual_mode)
    if config.supervision == "vo_se23" and config.vo_residual_mode != "absolute":
        raise ValueError("vo_residual_mode must be absolute or relative")
    if config.mse_residual_mode == "log":
        return se23_log_residual(X_ref, R_sol[sup], solution[sup])
    if config.mse_residual_mode == "right":
        return se23_gt_right_residual(X_ref, R_sol[sup], solution[sup])
    if config.mse_residual_mode == "global":
        return se23_global_residual(X_ref, R_sol[sup], solution[sup])
    if config.mse_residual_mode == "hybrid":
        log_residual = se23_log_residual(X_ref, R_sol[sup], solution[sup])
        global_residual = se23_global_residual(X_ref, R_sol[sup], solution[sup])
        return torch.cat((log_residual[..., :3], global_residual[..., 3:]), dim=-1)
    raise ValueError("mse_residual_mode must be log, right, global, or hybrid")


def _se23_ref_at_supervision(batch):
    sup = batch.supervised_indices
    if batch.X_ref is None:
        raise ValueError("SE23 supervision requires X_ref")
    if batch.X_ref.shape[0] == sup.shape[0]:
        return batch.X_ref
    return batch.X_ref[sup]


def _pose_residual_for_config(batch, solution, R_sol, config: PlatoTrainConfig):
    sup = batch.supervised_indices
    if config.vo_residual_mode == "relative":
        return relative_pose_residual(
            batch.R_ref,
            batch.p_ref,
            R_sol[sup],
            solution[sup, ..., 10:13],
            mode=config.mse_residual_mode,
        )
    if config.vo_residual_mode != "absolute":
        raise ValueError("vo_residual_mode must be absolute or relative")
    if config.mse_residual_mode == "log":
        return se23_pose_residual(
            batch.R_ref, batch.p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
    if config.mse_residual_mode == "right":
        return pose_right_residual(
            batch.R_ref, batch.p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
    if config.mse_residual_mode == "global":
        return pose_global_residual(
            batch.R_ref, batch.p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
    if config.mse_residual_mode == "hybrid":
        log_residual = se23_pose_residual(
            batch.R_ref, batch.p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
        global_residual = pose_global_residual(
            batch.R_ref, batch.p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
        return torch.cat((log_residual[..., :3], global_residual[..., 3:]), dim=-1)
    raise ValueError("mse_residual_mode must be log, global, or hybrid")


def _mse_blocks_for_stage(stage: str, full_state: bool, config: PlatoTrainConfig) -> tuple[str, ...]:
    if stage == "gyro":
        if not config.mse_gyro_full_state:
            return ("rot",)
        block_weights = (
            ("rot", config.mse_rot_weight),
            ("vel", config.mse_vel_weight),
            ("pos", config.mse_pos_weight),
        ) if full_state else (
            ("rot", config.mse_rot_weight),
            ("pos", config.mse_pos_weight),
        )
        blocks = tuple(name for name, weight in block_weights if weight != 0)
        if not blocks:
            raise ValueError("At least one gyro SE23 block weight must be nonzero")
        return blocks
    if stage == "acc":
        block_weights = (
            ("rot", config.mse_acc_rot_weight),
            ("vel", config.mse_acc_vel_weight),
            ("pos", config.mse_acc_pos_weight),
        ) if full_state else (
            ("rot", config.mse_acc_rot_weight),
            ("pos", config.mse_acc_pos_weight),
        )
        blocks = tuple(name for name, weight in block_weights if weight != 0)
        if not blocks:
            raise ValueError("At least one acc SE23 block weight must be nonzero")
        return blocks
    if stage == "joint":
        return ("rot", "vel", "pos") if full_state else ("rot", "pos")
    raise ValueError("stage must be gyro, acc, or joint")


def _mse_weights_for_stage(stage: str, config: PlatoTrainConfig) -> tuple[float, float, float]:
    if stage == "acc":
        return config.mse_acc_rot_weight, config.mse_acc_vel_weight, config.mse_acc_pos_weight
    return config.mse_rot_weight, config.mse_vel_weight, config.mse_pos_weight


def _gyro_window_schedule(config: PlatoTrainConfig) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if config.gyro_multi_window_steps:
        window_steps = tuple(int(window) for window in config.gyro_multi_window_steps)
        if any(window < 2 for window in window_steps):
            raise ValueError("All gyro multi-window steps must be at least 2")
        if config.gyro_multi_window_weights:
            window_weights = tuple(float(weight) for weight in config.gyro_multi_window_weights)
            if len(window_weights) != len(window_steps):
                raise ValueError("--gyro_multi_window_weights must match --gyro_multi_window_steps")
        else:
            window_weights = tuple(1.0 for _ in window_steps)
    else:
        window_steps = (int(config.window_steps),)
        window_weights = (1.0,)
    if any(weight <= 0 for weight in window_weights):
        raise ValueError("All gyro multi-window weights must be positive")
    return window_steps, window_weights


def _validate_gyro_mse(model, dataset_val: BaseDataset, config: PlatoTrainConfig, vo_by_seq):
    window_steps, window_weights = _gyro_window_schedule(config)
    total_loss = 0.0
    window_losses: dict[int, float] = {}
    for window, weight in zip(window_steps, window_weights):
        window_config = replace(config, window_steps=window)
        loss = _validate_mse(model, dataset_val, window_config, stage="gyro", vo_by_seq=vo_by_seq)
        window_losses[window] = loss
        total_loss += float(weight) * loss
    return total_loss, window_losses


def _validate_mse(model, dataset_val: BaseDataset, config: PlatoTrainConfig, stage: str, vo_by_seq):
    model.eval()
    loss_total = 0.0
    with torch.no_grad():
        for index in range(dataset_val.length()):
            batch = _get_batch_for_index(dataset_val, index, config, vo_by_seq)
            solution, R_sol = _integrate_batch(model, dataset_val, batch, config)
            loss, _ = _mse_loss_for_stage(batch, solution, R_sol, config, stage=stage)
            loss_total += loss.item()
    return loss_total


def _validate_nll(model, noise_params, dataset_val: BaseDataset, config: PlatoTrainConfig, vo_by_seq):
    model.eval()
    loss_total = 0.0
    metric_totals: dict[str, float] = {}
    metric_count = 0
    with torch.no_grad():
        for index in range(dataset_val.length()):
            batch = _get_batch_for_index(dataset_val, index, config, vo_by_seq)
            solution, R_sol = _integrate_batch(model, dataset_val, batch, config)
            loss, metrics = _nll_loss_for_batch(batch, solution, R_sol, config, noise_params)
            loss_total += loss.item()
            for name, value in metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + _scalar_value(value)
            metric_count += 1
    denom = max(metric_count, 1)
    avg_metrics = {
        name: torch.tensor(total / denom, dtype=torch.float64)
        for name, total in metric_totals.items()
    }
    return loss_total, avg_metrics


def _write_metrics(writer, prefix: str, metrics: dict[str, torch.Tensor], loss, epoch: int):
    writer.add_scalar(f"{prefix}/loss", _scalar_value(loss), epoch)
    _write_scalar_metrics(writer, prefix, metrics, epoch)


def _write_scalar_metrics(writer, prefix: str, metrics: dict[str, torch.Tensor], epoch: int):
    tags = {
        "rot": "se23_mse_rot",
        "vel": "se23_mse_vel",
        "pos": "se23_mse_pos",
        "nll": "nll",
        "nees": "nees",
        "nees_dof": "nees_dof",
        "nees_per_dof": "nees_per_dof",
        "mse_anchor": "mse_anchor",
    }
    for name, tag in tags.items():
        if name in metrics:
            writer.add_scalar(f"{prefix}/{tag}", _scalar_value(metrics[name]), epoch)


def _scalar_value(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _save_checkpoint(
    path: str,
    config: PlatoTrainConfig,
    code_stage: str,
    epoch: int,
    val_loss: float,
    model,
    optimizer,
    scheduler=None,
    noise_params=None,
):
    checkpoint = {
        "metadata": _checkpoint_metadata(config, code_stage),
        "VF_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "valloss": val_loss,
    }
    if model.biasfunc_w is not None:
        checkpoint["func_bw_model"] = model.biasfunc_w.state_dict()
    if model.biasfunc_a is not None:
        checkpoint["func_ba_model"] = model.biasfunc_a.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if noise_params is not None:
        checkpoint["noise_params"] = noise_params.state_dict()
        sigma_a, sigma_g = noise_params.sigmas()
        checkpoint["learned_imu_covariance"] = {
            "sigma_a": sigma_a.detach().cpu(),
            "sigma_g": sigma_g.detach().cpu(),
            "Qn": noise_params.Qn().detach().cpu(),
        }
    torch.save(checkpoint, path)


def _checkpoint_metadata(config: PlatoTrainConfig, code_stage: str) -> dict:
    metadata = asdict(config)
    metadata["code_stage"] = code_stage
    metadata["measurement_covariance"] = {
        "gt": {
            "rot": config.meas_sigma_rot_gt,
            "vel": config.meas_sigma_vel_gt,
            "pos": config.meas_sigma_pos_gt,
        },
        "vo": {
            "rot": config.meas_sigma_rot_vo,
            "pos": config.meas_sigma_pos_vo,
        },
    }
    return metadata
