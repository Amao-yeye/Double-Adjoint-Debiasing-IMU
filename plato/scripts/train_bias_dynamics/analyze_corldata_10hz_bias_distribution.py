#!/usr/bin/env python3
"""Analyze 10 Hz GT-differenced corldata IMU bias distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from scipy.stats import wasserstein_distance


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_1000hz"
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_1000hz_bias_10hz"
DEFAULT_TRAIN_SEQS = ["S00", "S01", "S02", "S03", "S05", "S07"]
DEFAULT_TEST_SEQS = ["S13", "S15", "S20", "S21", "S22", "S23"]
BIAS_DIMS = ["bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z"]
UNITS = {
    "bg_x": "rad/s",
    "bg_y": "rad/s",
    "bg_z": "rad/s",
    "ba_x": "m/s^2",
    "ba_y": "m/s^2",
    "ba_z": "m/s^2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--bin-duration-s", type=float, default=0.1, help="Bias is assumed constant in each bin.")
    parser.add_argument("--quality-window-size", type=int, default=128)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, 0.0, -9.81])
    parser.add_argument("--position-sg-window", type=int, default=201)
    parser.add_argument("--position-sg-order", type=int, default=3)
    parser.add_argument("--bias-sg-window", type=int, default=101)
    parser.add_argument("--bias-sg-order", type=int, default=3)
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Use all 10 Hz bins instead of requiring valid_window_starts_<N>.txt.",
    )
    return parser.parse_args()


def odd_window(n: int, target: int, order: int) -> int:
    if n <= order + 2:
        return max(3, order + 2 + ((order + 2) % 2 == 0))
    window = min(target, n if n % 2 else n - 1)
    min_window = order + 2 if (order + 2) % 2 else order + 3
    return max(window, min_window)


def smooth(data: np.ndarray, target_window: int, order: int) -> np.ndarray:
    if data.shape[0] < max(9, order + 3):
        return data.copy()
    window = odd_window(data.shape[0], target_window, order)
    return savgol_filter(data, window, order, axis=0, mode="interp")


def load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_sequence(data_root: Path, sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Rotation]:
    imu_path = data_root / sequence / "mav0" / "imu0" / "data.csv"
    mocap_path = data_root / sequence / "mav0" / "mocap0" / "data.csv"
    if not imu_path.exists():
        raise FileNotFoundError(imu_path)
    if not mocap_path.exists():
        raise FileNotFoundError(mocap_path)

    imu = load_csv(imu_path)
    mocap = load_csv(mocap_path)
    if imu.shape[0] != mocap.shape[0]:
        raise ValueError(f"{sequence}: IMU rows {imu.shape[0]} != mocap rows {mocap.shape[0]}")
    if not np.array_equal(imu[:, 0].astype(np.int64), mocap[:, 0].astype(np.int64)):
        max_dt_ns = np.max(np.abs(imu[:, 0].astype(np.int64) - mocap[:, 0].astype(np.int64)))
        raise ValueError(f"{sequence}: IMU/mocap timestamps do not match, max_dt_ns={max_dt_ns}")

    t = imu[:, 0].astype(np.float64) * 1e-9
    u = imu[:, 1:7].astype(np.float64)
    p = mocap[:, 1:4].astype(np.float64)
    q_wxyz = mocap[:, 4:8].astype(np.float64)
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(q_xyzw)
    return t, u, p, rot


def quality_valid_starts(data_root: Path, sequence: str, window_size: int) -> np.ndarray | None:
    path = data_root / sequence / "mav0" / "quality" / f"valid_window_starts_{window_size}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return np.empty((0,), dtype=np.int64)
    return np.atleast_1d(np.loadtxt(path, dtype=np.int64)).astype(np.int64)


def compute_sample_bias(
    t_abs: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    rot: Rotation,
    gravity: np.ndarray,
    position_sg_window: int,
    position_sg_order: int,
    bias_sg_window: int,
    bias_sg_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = t_abs - t_abs[0]
    p_smooth = smooth(p, position_sg_window, position_sg_order)
    v = np.gradient(p_smooth, t, axis=0, edge_order=2)
    a_world = np.gradient(v, t, axis=0, edge_order=2)

    rel = rot[:-1].inv() * rot[1:]
    dt = np.diff(t).reshape(-1, 1)
    w_body = np.empty((t.shape[0], 3), dtype=np.float64)
    w_body[:-1] = rel.as_rotvec() / dt
    w_body[-1] = w_body[-2]

    bg = u[:, :3] - w_body
    specific_force = rot.inv().apply(a_world - gravity.reshape(1, 3))
    ba = u[:, 3:6] - specific_force
    bias = np.column_stack((bg, ba))
    bias = smooth(bias, bias_sg_window, bias_sg_order)
    return t, bias, v


def bin_bias(
    sequence: str,
    split: str,
    t_rel: np.ndarray,
    bias: np.ndarray,
    speed: np.ndarray,
    valid_starts: np.ndarray | None,
    bin_duration_s: float,
) -> list[dict[str, float | int | str]]:
    median_dt = float(np.median(np.diff(t_rel)))
    bin_samples = int(round(bin_duration_s / median_dt))
    if bin_samples < 2:
        raise ValueError(f"{sequence}: bin_duration_s={bin_duration_s} gives bin_samples={bin_samples}")
    starts = np.arange(0, t_rel.shape[0] - bin_samples + 1, bin_samples, dtype=np.int64)
    if valid_starts is not None:
        starts = starts[np.isin(starts, valid_starts)]

    rows: list[dict[str, float | int | str]] = []
    for bin_index, start in enumerate(starts):
        stop = int(start + bin_samples)
        window_bias = bias[start:stop]
        row: dict[str, float | int | str] = {
            "sequence": sequence,
            "split": split,
            "bin_index": bin_index,
            "start_index": int(start),
            "stop_index": int(stop),
            "t_start_s": float(t_rel[start]),
            "t_end_s": float(t_rel[stop - 1]),
            "duration_s": float(t_rel[stop - 1] - t_rel[start]),
            "samples": int(bin_samples),
            "speed_mean_m_s": float(np.linalg.norm(speed[start:stop], axis=1).mean()),
        }
        for idx, dim in enumerate(BIAS_DIMS):
            row[dim] = float(window_bias[:, idx].mean())
            row[f"{dim}_std_in_bin"] = float(window_bias[:, idx].std())
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_to_arrays(rows: Iterable[dict[str, float | int | str]], dims: list[str]) -> dict[str, np.ndarray]:
    collected = {dim: [] for dim in dims}
    for row in rows:
        for dim in dims:
            collected[dim].append(float(row[dim]))
    return {dim: np.asarray(values, dtype=np.float64) for dim, values in collected.items()}


def stats_for(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def write_summary_tables(output_dir: Path, rows: list[dict[str, float | int | str]]) -> tuple[list[dict], list[dict], list[dict]]:
    by_split: list[dict] = []
    by_sequence: list[dict] = []
    similarity: list[dict] = []

    for split in ["train", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        arrays = rows_to_arrays(split_rows, BIAS_DIMS)
        for dim in BIAS_DIMS:
            stat = stats_for(arrays[dim])
            by_split.append({"split": split, "dimension": dim, "unit": UNITS[dim], **stat})

    for sequence in sorted({str(row["sequence"]) for row in rows}):
        seq_rows = [row for row in rows if row["sequence"] == sequence]
        split = str(seq_rows[0]["split"])
        arrays = rows_to_arrays(seq_rows, BIAS_DIMS)
        for dim in BIAS_DIMS:
            stat = stats_for(arrays[dim])
            by_sequence.append({"sequence": sequence, "split": split, "dimension": dim, "unit": UNITS[dim], **stat})

    train_arrays = rows_to_arrays([row for row in rows if row["split"] == "train"], BIAS_DIMS)
    test_arrays = rows_to_arrays([row for row in rows if row["split"] == "test"], BIAS_DIMS)
    for dim in BIAS_DIMS:
        train = train_arrays[dim]
        test = test_arrays[dim]
        train_stat = stats_for(train)
        test_stat = stats_for(test)
        train_std = float(train_stat["std"])
        p05 = float(train_stat["p05"])
        p95 = float(train_stat["p95"])
        train_min = float(train_stat["min"])
        train_max = float(train_stat["max"])
        similarity.append(
            {
                "dimension": dim,
                "unit": UNITS[dim],
                "train_mean": float(train_stat["mean"]),
                "test_mean": float(test_stat["mean"]),
                "mean_delta_test_minus_train": float(test_stat["mean"] - train_stat["mean"]),
                "abs_mean_delta_over_train_std": float(abs(test_stat["mean"] - train_stat["mean"]) / train_std)
                if train_std > 1e-12
                else float("nan"),
                "train_std": train_std,
                "test_std": float(test_stat["std"]),
                "train_p05": p05,
                "train_p95": p95,
                "test_fraction_inside_train_p05_p95": float(np.mean((test >= p05) & (test <= p95))),
                "test_fraction_inside_train_min_max": float(np.mean((test >= train_min) & (test <= train_max))),
                "wasserstein": float(wasserstein_distance(train, test)),
            }
        )

    write_rows(output_dir / "summary_by_split.csv", by_split)
    write_rows(output_dir / "summary_by_sequence.csv", by_sequence)
    write_rows(output_dir / "distribution_similarity.csv", similarity)
    (output_dir / "summary_by_split.json").write_text(json.dumps(by_split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary_by_sequence.json").write_text(
        json.dumps(by_sequence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "distribution_similarity.json").write_text(
        json.dumps(similarity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return by_split, by_sequence, similarity


def plot_sequence_timeseries(output_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    seq_dir = output_dir / "plots" / "sequence_bias"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for sequence in sorted({str(row["sequence"]) for row in rows}):
        seq_rows = [row for row in rows if row["sequence"] == sequence]
        t = np.asarray([float(row["t_start_s"]) for row in seq_rows])
        fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True)
        for axis, dim in zip(axes, BIAS_DIMS):
            axis.plot(t, [float(row[dim]) for row in seq_rows], linewidth=0.9)
            axis.set_ylabel(f"{dim}\n{UNITS[dim]}")
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel("time from sequence start (s)")
        fig.suptitle(f"{sequence} 10 Hz GT-differenced bias")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(seq_dir / f"{sequence}_bias_10hz.png", dpi=160)
        plt.close(fig)


def plot_train_test_distributions(output_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    train_arrays = rows_to_arrays(train_rows, BIAS_DIMS)
    test_arrays = rows_to_arrays(test_rows, BIAS_DIMS)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for axis, dim in zip(axes.ravel(), BIAS_DIMS):
        train = train_arrays[dim]
        test = test_arrays[dim]
        combined = np.concatenate((train, test))
        lo, hi = np.percentile(combined, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
            lo, hi = float(combined.min()), float(combined.max())
        bins = np.linspace(lo, hi, 70)
        axis.hist(train, bins=bins, density=True, alpha=0.55, label="train")
        axis.hist(test, bins=bins, density=True, alpha=0.55, label="test")
        axis.set_title(f"{dim} ({UNITS[dim]})")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Train/Test 10 Hz GT-differenced bias distributions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_dir / "train_test_bias_hist.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for axis, dim in zip(axes.ravel(), BIAS_DIMS):
        axis.boxplot([train_arrays[dim], test_arrays[dim]], labels=["train", "test"], showfliers=False)
        axis.set_title(f"{dim} ({UNITS[dim]})")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Train/Test bias distribution boxplots, outliers hidden")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_dir / "train_test_bias_boxplot.png", dpi=170)
    plt.close(fig)


def write_analysis_md(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    by_split: list[dict],
    similarity: list[dict],
    train_seqs: list[str],
    test_seqs: list[str],
    args: argparse.Namespace,
) -> None:
    counts_by_seq = {seq: sum(1 for row in rows if row["sequence"] == seq) for seq in train_seqs + test_seqs}
    train_count = sum(1 for row in rows if row["split"] == "train")
    test_count = sum(1 for row in rows if row["split"] == "test")
    ranked_shift = sorted(
        similarity,
        key=lambda row: float(row["abs_mean_delta_over_train_std"])
        if np.isfinite(float(row["abs_mean_delta_over_train_std"]))
        else -1.0,
        reverse=True,
    )
    low_overlap = [
        row
        for row in similarity
        if float(row["test_fraction_inside_train_p05_p95"]) < 0.6
        or float(row["test_fraction_inside_train_min_max"]) < 0.95
    ]

    lines = [
        "# corldata 10 Hz GT-Differenced Bias Analysis",
        "",
        f"Data root: `{args.data_root}`",
        f"Train sequences: `{', '.join(train_seqs)}`",
        f"Test sequences: `{', '.join(test_seqs)}`",
        f"Bin duration: `{args.bin_duration_s:.3f} s`",
        f"Quality filter: `valid_window_starts_{args.quality_window_size}.txt`"
        if not args.no_quality_filter
        else "Quality filter: disabled",
        "",
        "## Counts",
        "",
        f"- Train bins: `{train_count}`",
        f"- Test bins: `{test_count}`",
        "",
        "| sequence | split | bins |",
        "|---|---|---:|",
    ]
    for seq in train_seqs:
        lines.append(f"| {seq} | train | {counts_by_seq.get(seq, 0)} |")
    for seq in test_seqs:
        lines.append(f"| {seq} | test | {counts_by_seq.get(seq, 0)} |")

    lines.extend(
        [
            "",
            "## Train/Test Similarity",
            "",
            "| dimension | unit | train mean | train std | test mean | test std | mean shift / train std | test inside train p05-p95 | test inside train min-max |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in similarity:
        lines.append(
            "| {dim} | {unit} | {tm:.6g} | {ts:.6g} | {em:.6g} | {es:.6g} | {shift:.3f} | {p:.3f} | {mm:.3f} |".format(
                dim=row["dimension"],
                unit=row["unit"],
                tm=float(row["train_mean"]),
                ts=float(row["train_std"]),
                em=float(row["test_mean"]),
                es=float(row["test_std"]),
                shift=float(row["abs_mean_delta_over_train_std"]),
                p=float(row["test_fraction_inside_train_p05_p95"]),
                mm=float(row["test_fraction_inside_train_min_max"]),
            )
        )

    lines.extend(["", "## Automated Interpretation", ""])
    if ranked_shift:
        top = ranked_shift[:3]
        lines.append(
            "- Largest normalized mean shifts: "
            + ", ".join(
                f"{row['dimension']}={float(row['abs_mean_delta_over_train_std']):.2f} train-std"
                for row in top
            )
            + "."
        )
    if low_overlap:
        lines.append(
            "- Low train/test overlap flags: "
            + ", ".join(
                f"{row['dimension']} (p05-p95 fraction={float(row['test_fraction_inside_train_p05_p95']):.2f}, "
                f"min-max fraction={float(row['test_fraction_inside_train_min_max']):.2f})"
                for row in low_overlap
            )
            + "."
        )
    else:
        lines.append("- No dimension has severe train/test support mismatch by the default overlap thresholds.")

    lines.extend(
        [
            "- These labels are diagnostics derived from GT and IMU. Test-set labels must not be used for baseline training or test-specific tuning.",
            "- A fair baseline check is valid only if the supervision frame is first verified by preintegration and the train/test split remains unchanged.",
            "",
            "## Outputs",
            "",
            "- `bias_10hz.csv`: per-bin six-axis bias labels.",
            "- `summary_by_split.csv`: aggregate train/test statistics.",
            "- `summary_by_sequence.csv`: per-sequence statistics.",
            "- `distribution_similarity.csv`: train/test distribution comparison metrics.",
            "- `plots/train_test_bias_hist.png`: train/test distribution overlays.",
            "- `plots/train_test_bias_boxplot.png`: train/test boxplots.",
            "- `plots/sequence_bias/*.png`: per-sequence six-axis bias time series.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gravity = np.asarray(args.gravity, dtype=np.float64)

    all_rows: list[dict[str, float | int | str]] = []
    split_by_seq = {seq: "train" for seq in args.train_seqs}
    split_by_seq.update({seq: "test" for seq in args.test_seqs})
    for sequence in args.train_seqs + args.test_seqs:
        print(f"[bias10hz] {sequence}", flush=True)
        t, u, p, rot = load_sequence(args.data_root, sequence)
        t_rel, sample_bias, v = compute_sample_bias(
            t,
            u,
            p,
            rot,
            gravity,
            args.position_sg_window,
            args.position_sg_order,
            args.bias_sg_window,
            args.bias_sg_order,
        )
        valid_starts = None
        if not args.no_quality_filter:
            valid_starts = quality_valid_starts(args.data_root, sequence, args.quality_window_size)
            if valid_starts is None:
                raise FileNotFoundError(
                    args.data_root
                    / sequence
                    / "mav0"
                    / "quality"
                    / f"valid_window_starts_{args.quality_window_size}.txt"
                )
        rows = bin_bias(
            sequence,
            split_by_seq[sequence],
            t_rel,
            sample_bias,
            v,
            valid_starts,
            args.bin_duration_s,
        )
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No 10 Hz bias bins were generated.")
    write_rows(args.output_dir / "bias_10hz.csv", all_rows)
    by_split, _, similarity = write_summary_tables(args.output_dir, all_rows)
    plot_sequence_timeseries(args.output_dir, all_rows)
    plot_train_test_distributions(args.output_dir, all_rows)
    write_analysis_md(args.output_dir, all_rows, by_split, similarity, args.train_seqs, args.test_seqs, args)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(all_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
