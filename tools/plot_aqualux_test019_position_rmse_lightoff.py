#!/usr/bin/env python3
"""Plot Test_019 cumulative position RMSE with light-off intervals."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = find_repo_root()
DEFAULT_RESULT_DIR = (
    REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "vo_se3_mse_acc_nogate" / "Test_019"
)
DEFAULT_CSV = DEFAULT_RESULT_DIR / "test019_raw_ours_firstframe_time_rmse_lightoff.csv"
DEFAULT_SEGMENTS = (
    REPO_ROOT
    / "Results"
    / "AquaLux"
    / "openvins_test"
    / "vo_se3_mse_acc_nogate"
    / "hybrid_imu"
    / "Test_019"
    / "segments.csv"
)
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "test019_raw_ours_position_rmse_lightoff_only.png"
DEFAULT_PDF = DEFAULT_RESULT_DIR / "test019_raw_ours_position_rmse_lightoff_only.pdf"
DEFAULT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
DEFAULT_TIME0 = 1779620742.494100094


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--time0", type=float, default=DEFAULT_TIME0)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--legend-font-size", type=float, default=18.0)
    return parser.parse_args()


def setup_fonts(font_path: Path) -> None:
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_prop = fm.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def read_rmse_csv(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            time_s = float(row["time_s"])
            rmse = float(row["position_cumulative_rmse_m"])
            series.setdefault(method, []).append((time_s, rmse))
    if not series:
        raise ValueError(f"No RMSE rows found in {path}")
    return {
        method: (
            np.asarray([item[0] for item in values], dtype=np.float64),
            np.asarray([item[1] for item in values], dtype=np.float64),
        )
        for method, values in series.items()
    }


def read_lightoff_segments(path: Path, time0: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status", "ok") != "ok":
                continue
            start = float(row["start"]) - time0
            end = float(row["end"]) - time0
            intervals.append((start, end))
    return intervals


def display_label(method: str) -> str:
    if method == "Raw OpenVINS":
        return "Raw"
    if method.startswith("VO-MSE") or method.startswith("VO-SE3-MSE"):
        return "VO-NLL"
    return method


def plot(args: argparse.Namespace) -> tuple[Path, Path | None]:
    setup_fonts(args.font)
    series = read_rmse_csv(args.csv)
    intervals = read_lightoff_segments(args.segments, args.time0)

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 1.2,
            "grid.alpha": 1.0,
            "legend.frameon": True,
        }
    )

    fig, ax = plt.subplots(figsize=(14.4, 5.5), constrained_layout=True)

    span_color = "#c9c9c9"
    for start, end in intervals:
        ax.axvspan(start, end, color=span_color, alpha=0.45, linewidth=0, zorder=0)

    lines = []
    for method in sorted(series.keys(), key=lambda value: 0 if value == "Raw OpenVINS" else 1):
        times, rmse = series[method]
        if method == "Raw OpenVINS":
            (line,) = ax.plot(
                times,
                rmse,
                color="#6b7280",
                linewidth=3.8,
                linestyle="--",
                label=display_label(method),
                zorder=3,
            )
        else:
            (line,) = ax.plot(
                times,
                rmse,
                color="#1d4ed8",
                linewidth=4.0,
                label=display_label(method),
                zorder=4,
            )
        lines.append(line)

    lightoff_patch = mpatches.Patch(
        facecolor=span_color,
        edgecolor=span_color,
        alpha=0.45,
        linewidth=0.0,
        label="Light off",
    )
    legend = ax.legend(
        handles=[*lines, lightoff_patch],
        loc="upper left",
        ncol=3,
        fontsize=args.legend_font_size,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#d1d5db",
        handlelength=2.3,
        columnspacing=1.6,
        borderpad=0.35,
    )
    legend.set_zorder(20)
    legend.get_frame().set_linewidth(1.0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative RMSE error (m)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", width=1.4, length=7, pad=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    pdf = None
    if args.pdf is not None:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf)
        pdf = args.pdf
    plt.close(fig)
    return args.output, pdf


def main() -> int:
    args = parse_args()
    output, pdf = plot(args)
    print(f"wrote {output}")
    if pdf is not None:
        print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
