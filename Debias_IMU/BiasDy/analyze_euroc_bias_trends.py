import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import pload


EUROC_SEQUENCES = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]


def moving_average(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    if x.ndim != 2 or x.shape[-1] != 3:
        raise ValueError(f"bias tensor should have shape (N, 3), got {tuple(x.shape)}")
    if x.shape[0] < window:
        return x
    weight = torch.ones(3, 1, window, dtype=x.dtype, device=x.device) / window
    padded = torch.nn.functional.pad(
        x.transpose(0, 1).unsqueeze(0),
        (window // 2, window - 1 - window // 2),
        mode="replicate",
    )
    return torch.nn.functional.conv1d(padded, weight, groups=3).squeeze(0).transpose(0, 1)


def load_bias(cache_dir: str, sequence: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = pload(cache_dir, sequence + ".p")
    t = data.get("t_bias", data["t_imu"]).detach().cpu().float()
    bw = data["w_bias"].detach().cpu().float()
    ba = data["a_bias"].detach().cpu().float()
    n = min(t.shape[0], bw.shape[0], ba.shape[0])
    return t[:n], bw[:n], ba[:n]


def plot_sequence(sequence: str, t: torch.Tensor, bw: torch.Tensor, ba: torch.Tensor, out_dir: str, window: int) -> None:
    t_np = t.numpy()
    bw_np = bw.numpy()
    ba_np = ba.numpy()
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for dim in range(3):
        axes[0].plot(t_np, bw_np[:, dim], label=f"bw_{labels[dim]}", linewidth=1.2)
        axes[1].plot(t_np, ba_np[:, dim], label=f"ba_{labels[dim]}", linewidth=1.2)
    axes[0].set_ylabel("gyro bias (rad/s)")
    axes[1].set_ylabel("acc bias (m/s^2)")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title(f"{sequence} bias trends, moving average window={window}")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{sequence}_bias_trend_w{window}.png"), dpi=180)
    plt.close(fig)


def summarize(sequence: str, t: torch.Tensor, bw: torch.Tensor, ba: torch.Tensor) -> dict[str, float | str | int]:
    duration = (t[-1] - t[0]).item() if t.numel() > 1 else 0.0
    row: dict[str, float | str | int] = {
        "sequence": sequence,
        "num_samples": int(t.shape[0]),
        "duration_s": duration,
    }
    for name, tensor in (("bw", bw), ("ba", ba)):
        for dim, axis in enumerate(("x", "y", "z")):
            values = tensor[:, dim]
            row[f"{name}_{axis}_mean"] = values.mean().item()
            row[f"{name}_{axis}_std"] = values.std(unbiased=False).item()
            row[f"{name}_{axis}_min"] = values.min().item()
            row[f"{name}_{axis}_max"] = values.max().item()
            row[f"{name}_{axis}_range"] = (values.max() - values.min()).item()
        norm = tensor.norm(dim=-1)
        row[f"{name}_norm_mean"] = norm.mean().item()
        row[f"{name}_norm_std"] = norm.std(unbiased=False).item()
        row[f"{name}_norm_max"] = norm.max().item()
    return row


def write_summary(rows: list[dict[str, float | str | int]], out_dir: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "bias_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_sequences(raw: str) -> list[str]:
    if raw == "all":
        return EUROC_SEQUENCES
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser("Analyze EuRoC bias trends estimated from GT finite differences.")
    parser.add_argument("--cache_dir", type=str, default="data/EUROC_cache_gt")
    parser.add_argument("--output_dir", type=str, default="results/analysis/euroc_bias_trends")
    parser.add_argument("--sequences", type=str, default="all", help="Comma-separated sequence names or 'all'.")
    parser.add_argument("--window", type=int, default=10, help="Moving-average window in samples.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for sequence in parse_sequences(args.sequences):
        path = os.path.join(args.cache_dir, sequence + ".p")
        if not os.path.exists(path):
            print(f"skip missing sequence cache: {path}")
            continue
        t, bw, ba = load_bias(args.cache_dir, sequence)
        bw_smooth = moving_average(bw, args.window)
        ba_smooth = moving_average(ba, args.window)
        plot_sequence(sequence, t, bw_smooth, ba_smooth, args.output_dir, args.window)
        rows.append(summarize(sequence, t, bw_smooth, ba_smooth))
        print(f"{sequence}: N={t.shape[0]}, duration={(t[-1] - t[0]).item():.2f}s")

    write_summary(rows, args.output_dir)
    print(f"wrote plots and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
