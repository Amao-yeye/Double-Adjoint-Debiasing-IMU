#!/usr/bin/env python3
"""Initialize no-u_dot EuRoC bias dynamics from differenced bias labels.

This is an initialization stage only.  The trajectory objective remains the
Debias_IMU/RSS rollout loss run afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_BIASDY = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU" / "BiasDy"
sys.path.insert(0, str(DEBIAS_BIASDY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import pload  # noqa: E402
from run_liasuw_debias_ad_pipeline import ba_func_choose, bw_func_choose  # noqa: E402


DEFAULT_TRAIN_SEQS = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--network-type-gyro", default="bw_func_net_no_udot")
    parser.add_argument("--network-type-acc", default="ba_func_net_no_udot")
    parser.add_argument("--diff-stride", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--gyro-steps", type=int, default=2000)
    parser.add_argument("--acc-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gyro-rate-clip", type=float, default=0.2)
    parser.add_argument("--acc-rate-clip", type=float, default=3.0)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_rate_dataset(cache_dir: Path, seqs: list[str], bias_key: str, diff_stride: int):
    biases = []
    imus = []
    targets = []
    for seq in seqs:
        data = pload(str(cache_dir / f"{seq}.p"))
        bias = data[bias_key].float()
        imu = data["u_imu"].float()
        t = data.get("t_bias", data["t_imu"]).float()
        stride = int(diff_stride)
        if bias.shape[0] <= 2 * stride:
            raise ValueError(f"{seq} is too short for diff_stride={stride}")
        center = slice(stride, -stride)
        rate = (bias[2 * stride :] - bias[: -2 * stride]) / (t[2 * stride :] - t[: -2 * stride]).unsqueeze(-1)
        biases.append(bias[center])
        imus.append(imu[center])
        targets.append(rate)
    return torch.cat(biases, dim=0), torch.cat(imus, dim=0), torch.cat(targets, dim=0)


def _make_y(batch_bias: torch.Tensor, bias_slice: slice) -> torch.Tensor:
    y = torch.zeros((batch_bias.shape[0], 16), dtype=batch_bias.dtype, device=batch_bias.device)
    y[:, bias_slice] = batch_bias
    return y


def _train_one(
    *,
    model: torch.nn.Module,
    bias: torch.Tensor,
    imu: torch.Tensor,
    target: torch.Tensor,
    bias_slice: slice,
    output_path: Path,
    state_key: str,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    rate_clip: float,
    log_freq: int,
    save_freq: int,
    device: str,
) -> list[dict[str, float]]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.SmoothL1Loss(beta=max(rate_clip * 0.05, 1e-6))
    bias = bias.to(device)
    imu = imu.to(device)
    target = target.clamp(min=-rate_clip, max=rate_clip).to(device)
    R0 = torch.eye(3, dtype=bias.dtype, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for step in range(steps + 1):
        index = torch.randint(0, bias.shape[0], (batch_size,), device=device)
        batch_bias = bias[index]
        batch_imu = imu[index]
        batch_target = target[index]
        y = _make_y(batch_bias, bias_slice)
        optimizer.zero_grad()
        pred = model(y, batch_imu, None, R0[: batch_bias.shape[0]])
        loss = criterion(pred, batch_target)
        loss.backward()
        optimizer.step()
        if step % log_freq == 0 or step == steps:
            with torch.no_grad():
                mae = (pred - batch_target).abs().mean().item()
                pred_abs = pred.abs().mean().item()
            row = {"step": step, "loss": float(loss.item()), "mae": mae, "pred_abs_mean": pred_abs}
            history.append(row)
            print(
                f"{state_key} step={step} loss={row['loss']:.6g} "
                f"mae={mae:.6g} pred_abs_mean={pred_abs:.6g}",
                flush=True,
            )
        if step % save_freq == 0 or step == steps:
            torch.save(
                {
                    state_key: model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": step,
                    "loss": float(loss.item()),
                    "checkpoint_type": "bias_rate_pretrain",
                },
                output_path.parent / f"model_epoch_{step}.pt",
            )
    torch.save(
        {
            state_key: model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": steps,
            "loss": float(loss.item()),
            "checkpoint_type": "bias_rate_pretrain",
        },
        output_path,
    )
    return history


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={args.device} requested but CUDA is unavailable")
    args.outputdir.mkdir(parents=True, exist_ok=True)
    (args.outputdir / "resolved_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"bias-rate init diff_stride={args.diff_stride}, "
        f"gyro_clip={args.gyro_rate_clip}, acc_clip={args.acc_rate_clip}, device={args.device}",
        flush=True,
    )

    bw_bias, bw_imu, bw_target = _load_rate_dataset(args.cache_dir, args.train_seqs, "w_bias", args.diff_stride)
    ba_bias, ba_imu, ba_target = _load_rate_dataset(args.cache_dir, args.train_seqs, "a_bias", args.diff_stride)

    gyro = bw_func_choose(args.network_type_gyro, args.device)
    gyro_history = _train_one(
        model=gyro,
        bias=bw_bias,
        imu=bw_imu,
        target=bw_target,
        bias_slice=slice(4, 7),
        output_path=args.outputdir / "Gyro_weights" / "bias_rate_init.pt",
        state_key="func_bw_model",
        steps=args.gyro_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rate_clip=args.gyro_rate_clip,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        device=args.device,
    )

    acc = ba_func_choose(args.network_type_acc, args.device)
    acc_history = _train_one(
        model=acc,
        bias=ba_bias,
        imu=ba_imu,
        target=ba_target,
        bias_slice=slice(13, 16),
        output_path=args.outputdir / "Acc_weights" / "bias_rate_init.pt",
        state_key="func_ba_model",
        steps=args.acc_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rate_clip=args.acc_rate_clip,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        device=args.device,
    )

    (args.outputdir / "history.json").write_text(
        json.dumps({"gyro": gyro_history, "acc": acc_history}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
