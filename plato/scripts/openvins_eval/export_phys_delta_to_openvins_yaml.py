#!/usr/bin/env python3
"""Export PLATO phys_delta Debias_IMU checkpoints to an OpenVINS-readable YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _flatten_param(state_dict: dict[str, torch.Tensor], key: str, shape: tuple[int, ...]) -> list[float]:
    tensor = state_dict[key].detach().cpu()
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{key} has shape {tuple(tensor.shape)}, expected {shape}")
    return [float(v) for v in tensor.reshape(-1).tolist()]


def _write_scalar(f, key: str, value) -> None:
    if isinstance(value, str):
        f.write(f'{key}: "{value}"\n')
    else:
        f.write(f"{key}: {value}\n")


def _write_vector(f, key: str, values: list[float], per_line: int = 6) -> None:
    f.write(f"{key}: [\n")
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        suffix = "," if start + per_line < len(values) else " ]"
        f.write("  " + ", ".join(f"{v:.17g}" for v in chunk) + suffix + "\n")


def _load_model_state(path: Path, key: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if key not in checkpoint:
        raise KeyError(f"{path} does not contain {key}; keys={sorted(checkpoint.keys())}")
    return checkpoint[key]


def _write_phys_delta_model(f, prefix: str, state_dict: dict[str, torch.Tensor], bias_scale: float, rate_bound: float, use_acc_bias: int) -> None:
    feature_dim = 54
    hidden_dim = 128
    _write_scalar(f, f"{prefix}_feature_dim", feature_dim)
    _write_scalar(f, f"{prefix}_hidden_dim", hidden_dim)
    _write_scalar(f, f"{prefix}_bias_scale", bias_scale)
    _write_scalar(f, f"{prefix}_rate_bound", rate_bound)
    _write_scalar(f, f"{prefix}_use_acc_bias", use_acc_bias)
    _write_vector(f, f"{prefix}_linear_phys_weight", _flatten_param(state_dict, "linear_phys.weight", (3, feature_dim)))
    _write_vector(f, f"{prefix}_linear_phys_bias", _flatten_param(state_dict, "linear_phys.bias", (3,)))
    _write_vector(f, f"{prefix}_residual0_weight", _flatten_param(state_dict, "residual.0.weight", (hidden_dim, feature_dim)))
    _write_vector(f, f"{prefix}_residual0_bias", _flatten_param(state_dict, "residual.0.bias", (hidden_dim,)))
    _write_vector(f, f"{prefix}_residual2_weight", _flatten_param(state_dict, "residual.2.weight", (hidden_dim, hidden_dim)))
    _write_vector(f, f"{prefix}_residual2_bias", _flatten_param(state_dict, "residual.2.bias", (hidden_dim,)))
    _write_vector(f, f"{prefix}_residual4_weight", _flatten_param(state_dict, "residual.4.weight", (3, hidden_dim)))
    _write_vector(f, f"{prefix}_residual4_bias", _flatten_param(state_dict, "residual.4.bias", (3,)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gyro-checkpoint",
        type=Path,
        default=Path("results/debias_imu/LIASuw_phys_delta_e1800_w16_b64/Gyro_weights/best_val_model.pt"),
    )
    parser.add_argument(
        "--accel-checkpoint",
        type=Path,
        default=Path("results/debias_imu/LIASuw_phys_delta_e1800_w16_b64/Acc_weights/best_val_model.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/debias_imu/LIASuw_phys_delta_e1800_w16_b64/openvins/phys_delta_openvins.yaml"),
    )
    args = parser.parse_args()

    gyro_state = _load_model_state(args.gyro_checkpoint, "func_bw_model")
    accel_state = _load_model_state(args.accel_checkpoint, "func_ba_model")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("%YAML:1.0\n---\n")
        _write_scalar(f, "model_type", "phys_delta")
        _write_scalar(f, "source_gyro_checkpoint", str(args.gyro_checkpoint))
        _write_scalar(f, "source_accel_checkpoint", str(args.accel_checkpoint))
        _write_phys_delta_model(f, "gyro", gyro_state, bias_scale=0.1, rate_bound=0.08, use_acc_bias=0)
        _write_phys_delta_model(f, "accel", accel_state, bias_scale=1.0, rate_bound=0.8, use_acc_bias=1)
    print(args.output)


if __name__ == "__main__":
    main()
