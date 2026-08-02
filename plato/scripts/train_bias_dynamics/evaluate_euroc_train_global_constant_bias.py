#!/usr/bin/env python3
"""Evaluate EuRoC pure-IMU rollout with train-only global constant biases.

This is a diagnostic for the conservative residual bias-dynamics path.  It
uses bias labels from the training cache only to estimate one global constant
gyro and accelerometer bias, then rolls out the held-out test sequences without
using any test-sequence bias labels for initialization.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

import sys


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_BIASDY = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU" / "BiasDy"
if str(DEBIAS_BIASDY) not in sys.path:
    sys.path.insert(0, str(DEBIAS_BIASDY))

import lie_algebra as Lie  # noqa: E402
from evalutation import AOE, ATE  # noqa: E402


DEFAULT_TRAIN_SEQS = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]
DEFAULT_TEST_SEQS = [
    "MH_02_easy",
    "MH_04_difficult",
    "V2_02_medium",
    "V1_03_difficult",
    "V1_01_easy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "EUROC_test_gt",
        help="Cache containing EuRoC *.p and *_gt.p files.",
    )
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--exclude-train-seq", nargs="*", default=[])
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--stat", choices=["mean", "median"], default="median")
    parser.add_argument(
        "--rollout-mode",
        choices=[
            "train_global_constant",
            "first_to_train_global_exp",
            "imu_window_gyro_train_global_acc",
            "imu_window_static_gyro_acc",
        ],
        default="train_global_constant",
    )
    parser.add_argument("--lambda-g", type=float, default=0.0)
    parser.add_argument("--lambda-a", type=float, default=0.0)
    parser.add_argument("--imu-window-s", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "EUROC_train_global_constant_bias_summary.json",
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def rotate(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (R @ x.unsqueeze(-1)).squeeze(-1)


def bias_stat(cache_dir: Path, sequences: list[str], stat: str) -> tuple[torch.Tensor, torch.Tensor]:
    bw_values = []
    ba_values = []
    for seq in sequences:
        data = load_pickle(cache_dir / f"{seq}.p")
        bw_values.append(data["w_bias"].float())
        ba_values.append(data["a_bias"].float())
    bw = torch.cat(bw_values, dim=0)
    ba = torch.cat(ba_values, dim=0)
    if stat == "mean":
        return bw.mean(dim=0), ba.mean(dim=0)
    return bw.median(dim=0).values, ba.median(dim=0).values


def rollout_constant(
    cache_dir: Path,
    sequence: str,
    bw_const: torch.Tensor,
    ba_const: torch.Tensor,
    rollout_mode: str,
    lambda_g: float,
    lambda_a: float,
    imu_window_s: float,
) -> dict[str, float]:
    imu = load_pickle(cache_dir / f"{sequence}.p")
    gt = load_pickle(cache_dir / f"{sequence}_gt.p")
    u_imu = imu["u_imu"].float()
    t_gt = gt["t_gt"].float()
    R_gt = gt["R_gt"].float()
    v_gt = gt["v_gt"].float()
    p_gt = gt["p_gt"].float()
    X_gt = Lie.SEn3fromSO3Rn(R_gt, torch.cat((v_gt, p_gt), dim=-1))
    bw_start = imu["w_bias"][0].float()
    ba_start = imu["a_bias"][0].float()
    t_imu = imu["t_imu"].float()
    imu_mask = (t_imu >= t_imu[0]) & (t_imu <= t_imu[0] + float(imu_window_s))
    if not bool(imu_mask.any()):
        imu_mask = torch.zeros_like(t_imu, dtype=torch.bool)
        imu_mask[0] = True
    u_window = u_imu[imu_mask]
    bw_imu_window = u_window[:, :3].mean(dim=0)
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
    ba_static_window = u_window[:, 3:6].mean(dim=0) - (R_gt[0].T @ (-gravity))

    R_pred = torch.zeros_like(R_gt)
    v_pred = torch.zeros_like(v_gt)
    p_pred = torch.zeros_like(p_gt)
    R_pred[0] = R_gt[0]
    v_pred[0] = v_gt[0]
    p_pred[0] = p_gt[0]
    for i in range(1, t_gt.shape[0]):
        dt = t_gt[i] - t_gt[i - 1]
        if rollout_mode == "train_global_constant":
            bw_i = bw_const
            ba_i = ba_const
        elif rollout_mode == "imu_window_gyro_train_global_acc":
            bw_i = bw_imu_window
            ba_i = ba_const
        elif rollout_mode == "imu_window_static_gyro_acc":
            bw_i = bw_imu_window
            ba_i = ba_static_window
        else:
            t_prev = t_gt[i - 1]
            bw_i = bw_const + (bw_start - bw_const) * torch.exp(-torch.tensor(lambda_g, dtype=t_gt.dtype) * t_prev)
            ba_i = ba_const + (ba_start - ba_const) * torch.exp(-torch.tensor(lambda_a, dtype=t_gt.dtype) * t_prev)
        w_corr = u_imu[i - 1, :3] - bw_i
        a_corr = u_imu[i - 1, 3:6] - ba_i
        R_pred[i] = R_pred[i - 1] @ Lie.SO3exp(w_corr * dt)
        v_pred[i] = v_pred[i - 1] + (rotate(R_pred[i - 1], a_corr) + gravity) * dt
        p_pred[i] = p_pred[i - 1] + v_pred[i - 1] * dt

    return {
        "AOE_deg": float(AOE(X_gt[:, :3, :3], R_pred).item()),
        "ATE_vel": float(ATE(X_gt[:, :3, 3], v_pred).item()),
        "ATE_pos_m": float(ATE(X_gt[:, :3, 4], p_pred).item()),
    }


def main() -> None:
    args = parse_args()
    train_seqs = [seq for seq in args.train_seqs if seq not in set(args.exclude_train_seq)]
    bw_const, ba_const = bias_stat(args.cache_dir, train_seqs, args.stat)
    rows = {}
    for seq in args.test_seqs:
        rows[seq] = rollout_constant(
            args.cache_dir,
            seq,
            bw_const,
            ba_const,
            args.rollout_mode,
            args.lambda_g,
            args.lambda_a,
            args.imu_window_s,
        )
    mean = {
        key: float(torch.tensor([rows[seq][key] for seq in args.test_seqs], dtype=torch.float64).mean().item())
        for key in ["AOE_deg", "ATE_vel", "ATE_pos_m"]
    }
    summary = {
        "cache_dir": str(args.cache_dir),
        "train_seqs": train_seqs,
        "test_seqs": list(args.test_seqs),
        "stat": args.stat,
        "rollout_mode": args.rollout_mode,
        "lambda_g": float(args.lambda_g),
        "lambda_a": float(args.lambda_a),
        "imu_window_s": float(args.imu_window_s),
        "bw_const": [float(x) for x in bw_const.tolist()],
        "ba_const": [float(x) for x in ba_const.tolist()],
        "sequences": rows,
        "mean": mean,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
