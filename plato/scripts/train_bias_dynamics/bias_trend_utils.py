"""Shared bias trend metrics and plotting helpers for PLATO Debias_IMU runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def axis_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = pred - ref
    pred_centered = pred - pred.mean()
    ref_centered = ref - ref.mean()
    centered_diff = pred_centered - ref_centered
    pred_std = pred_centered.std()
    ref_std = ref_centered.std()
    if pred_std < 1.0e-12 or ref_std < 1.0e-12:
        corr = float("nan")
    else:
        corr = float(np.mean(pred_centered * ref_centered) / (pred_std * ref_std))
    return {
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "centered_rmse": float(np.sqrt(np.mean(centered_diff * centered_diff))),
        "corr": corr,
        "pred_mean": float(pred.mean()),
        "ref_mean": float(ref.mean()),
        "pred_std": float(pred.std()),
        "ref_std": float(ref.std()),
    }


def vector_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = pred - ref
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    centered_diff = pred_centered - ref_centered
    return {
        "rmse_norm": float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))),
        "mae_norm": float(np.mean(np.linalg.norm(diff, axis=1))),
        "centered_rmse_norm": float(np.sqrt(np.mean(np.sum(centered_diff * centered_diff, axis=1)))),
    }


def plot_bias_trend(
    t: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
    title: str,
    ylabel: str,
    reference_label: str,
    path: Path,
    max_plot_s: float,
) -> None:
    mask = np.ones_like(t, dtype=bool)
    if max_plot_s > 0:
        mask &= t <= t[0] + max_plot_s
    axes_names = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.0), sharex=True, constrained_layout=True)
    for axis, ax in enumerate(axes):
        ax.plot(t[mask] - t[0], ref[mask, axis], color="black", lw=1.5, label=reference_label)
        ax.plot(t[mask] - t[0], pred[mask, axis], color="#1677a3", lw=1.25, label="predicted")
        ax.set_ylabel(f"{ylabel}-{axes_names[axis]}")
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def collect_bias_trend_metrics(
    sequence: str,
    t: np.ndarray,
    bw_pred: np.ndarray,
    ba_pred: np.ndarray,
    bw_ref: np.ndarray,
    ba_ref: np.ndarray,
    output_dir: Path,
    reference_label: str,
    max_plot_s: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    seq_metrics: dict[str, object] = {
        "bw": vector_metrics(bw_pred, bw_ref),
        "ba": vector_metrics(ba_pred, ba_ref),
    }
    for kind, pred, ref in [("bw", bw_pred, bw_ref), ("ba", ba_pred, ba_ref)]:
        for axis, axis_name in enumerate(["x", "y", "z"]):
            metrics = axis_metrics(pred[:, axis], ref[:, axis])
            seq_metrics[f"{kind}_{axis_name}"] = metrics
            rows.append({"sequence": sequence, "kind": kind, "axis": axis_name, **metrics})
        plot_bias_trend(
            t,
            pred,
            ref,
            title=f"{sequence} {kind}: predicted vs reference bias",
            ylabel=kind,
            reference_label=reference_label,
            path=output_dir / sequence / f"{kind}_trend.png",
            max_plot_s=max_plot_s,
        )
    return rows, seq_metrics


def write_bias_trend_outputs(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bias_trend_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "kind",
                "axis",
                "rmse",
                "mae",
                "centered_rmse",
                "corr",
                "pred_mean",
                "ref_mean",
                "pred_std",
                "ref_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "bias_trend_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
