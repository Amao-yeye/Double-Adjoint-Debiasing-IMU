#!/usr/bin/env python3
"""Plot EuRoC before/after learned-bias preintegration errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_ROOT = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU"
sys.path.insert(0, str(DEBIAS_ROOT))
sys.path.insert(0, str(DEBIAS_ROOT / "BiasDy"))

import lie_algebra as Lie  # noqa: E402
from learning import pload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "debias_imu"
        / "EUROC_tiny_acc_fixed_gyro_vo_body_w64_sg_ad_e301_b1000_noV203",
    )
    parser.add_argument("--sequence", default="MH_02_easy")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=40.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "debias_imu"
        / "figures"
        / "euroc_mh02_preintegration_before_after",
    )
    return parser.parse_args()


def rotation_error_deg(R_gt: torch.Tensor, R_pred: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(Lie.SO3log(R_gt @ R_pred.transpose(-1, -2)), dim=-1) * 180.0 / torch.pi


def rmse(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.square())).item())


def main() -> None:
    args = parse_args()
    model_path = args.result_dir / "Sequnces-results" / args.sequence / "FullTra_result.p"
    raw_path = args.result_dir / "raw_int" / f"{args.sequence}.p"
    const_path = args.result_dir / "constant_bias" / f"{args.sequence}.p"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not const_path.exists():
        raise FileNotFoundError(const_path)

    model = pload(str(model_path))
    raw = pload(str(raw_path))
    const = pload(str(const_path))

    X_gt = model["X_gt"]
    t = model["t_gt"].reshape(-1)
    X_raw = raw["X_raw"]
    X_const = const["X_const"]
    R_learned = model["R_pred"]
    p_learned = model["sol_pred"][:, 10:13]

    t_rel = t - t[0]
    mask = (t_rel >= args.start_s) & (t_rel <= args.start_s + args.duration_s)
    if int(mask.sum()) < 2:
        raise ValueError("Selected time window has fewer than two samples.")

    t_plot = t_rel[mask] - args.start_s
    p_gt = X_gt[mask, :3, 4]
    R_gt = X_gt[mask, :3, :3]
    series = {
        "Raw IMU": (X_raw[mask, :3, :3], X_raw[mask, :3, 4]),
        "Constant bias": (X_const[mask, :3, :3], X_const[mask, :3, 4]),
        "Learned": (R_learned[mask], p_learned[mask]),
    }

    pos_err = {}
    rot_err = {}
    summary = {}
    for name, (R_pred, p_pred) in series.items():
        pos = torch.linalg.norm(p_pred - p_gt, dim=-1)
        rot = rotation_error_deg(R_gt, R_pred)
        pos_err[name] = pos
        rot_err[name] = rot
        summary[name] = {
            "pos_rmse_m": rmse(pos),
            "rot_rmse_deg": rmse(rot),
            "pos_final_m": float(pos[-1].item()),
            "rot_final_deg": float(rot[-1].item()),
        }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "Raw IMU": "#b23a48",
        "Constant bias": "#5b6c8f",
        "Learned": "#138a72",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)

    xy0 = p_gt[0]
    axes[0].plot((p_gt[:, 0] - xy0[0]).numpy(), (p_gt[:, 1] - xy0[1]).numpy(), color="black", lw=2.0, label="GT")
    for name, (_, p_pred) in series.items():
        if name == "Raw IMU":
            continue
        axes[0].plot(
            (p_pred[:, 0] - xy0[0]).detach().cpu().numpy(),
            (p_pred[:, 1] - xy0[1]).detach().cpu().numpy(),
            color=colors[name],
            lw=1.6,
            label=name,
        )
    axes[0].set_title("Trajectory segment")
    axes[0].set_xlabel("x offset (m)")
    axes[0].set_ylabel("y offset (m)")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)

    for name, err in pos_err.items():
        axes[1].plot(t_plot.numpy(), err.numpy(), color=colors[name], lw=1.8, label=name)
    axes[1].set_title("Position error")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("error (m)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)

    for name, err in rot_err.items():
        axes[2].plot(t_plot.numpy(), err.numpy(), color=colors[name], lw=1.8, label=name)
    axes[2].set_title("Rotation error")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("error (deg)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.25)

    handles = []
    labels = []
    for ax in (axes[0], axes[2]):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))

    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    json_path = args.output_prefix.with_suffix(".json")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    json_path.write_text(
        json.dumps(
            {
                "result_dir": str(args.result_dir),
                "sequence": args.sequence,
                "start_s": args.start_s,
                "duration_s": args.duration_s,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
