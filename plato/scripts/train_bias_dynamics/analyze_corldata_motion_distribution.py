#!/usr/bin/env python3
"""Analyze corldata GT-differenced motion distributions.

The Debias_IMU bias dynamics receive IMU measurements ``u`` as input.  This
diagnostic separates the motion component implied by the aligned GT from the raw
IMU measurement so we can compare whether train/test sequences have similar
angular-velocity and acceleration regimes.
"""

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
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_1000hz_motion_10hz"
DEFAULT_LIGHT_CSV = PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "light_segments" / "light_segments.csv"
DEFAULT_TRAIN_SEQS = ["S00", "S01", "S02", "S03", "S05", "S07"]
DEFAULT_TEST_SEQS = ["S13", "S15", "S20", "S21", "S22", "S23"]
MOTION_DIMS = ["omega_gt_norm_mean", "acc_dyn_gt_norm_mean", "speed_mean_m_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--light-segments-csv", type=Path, default=DEFAULT_LIGHT_CSV)
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--sequences", nargs="+", default=None, help="Default: all sequence folders in data root.")
    parser.add_argument("--bin-duration-s", type=float, default=0.1)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, 0.0, -9.81])
    parser.add_argument("--position-sg-window", type=int, default=201)
    parser.add_argument("--position-sg-order", type=int, default=3)
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


def discover_sequences(data_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and (path / "mav0" / "imu0" / "data.csv").exists()
    )


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
    imu_t = imu[:, 0].astype(np.int64)
    mocap_t = mocap[:, 0].astype(np.int64)
    if not np.array_equal(imu_t, mocap_t):
        max_dt_ns = int(np.max(np.abs(imu_t - mocap_t)))
        raise ValueError(f"{sequence}: IMU/mocap timestamps do not match, max_dt_ns={max_dt_ns}")

    t = imu_t.astype(np.float64) * 1e-9
    u = imu[:, 1:7].astype(np.float64)
    p = mocap[:, 1:4].astype(np.float64)
    q_wxyz = mocap[:, 4:8].astype(np.float64)
    rot = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])
    return t, u, p, rot


def compute_gt_motion(
    t_abs: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    rot: Rotation,
    gravity: np.ndarray,
    position_sg_window: int,
    position_sg_order: int,
) -> dict[str, np.ndarray]:
    t = t_abs - t_abs[0]
    p_smooth = smooth(p, position_sg_window, position_sg_order)
    v_world = np.gradient(p_smooth, t, axis=0, edge_order=2)
    a_world = np.gradient(v_world, t, axis=0, edge_order=2)

    rel = rot[:-1].inv() * rot[1:]
    dt = np.diff(t).reshape(-1, 1)
    omega_gt = np.empty((t.shape[0], 3), dtype=np.float64)
    omega_gt[:-1] = rel.as_rotvec() / dt
    omega_gt[-1] = omega_gt[-2]

    acc_dyn_gt = rot.inv().apply(a_world)
    acc_specific_gt = rot.inv().apply(a_world - gravity.reshape(1, 3))
    return {
        "t": t,
        "omega_meas": u[:, :3],
        "acc_meas": u[:, 3:6],
        "omega_gt": omega_gt,
        "acc_dyn_gt": acc_dyn_gt,
        "acc_specific_gt": acc_specific_gt,
        "speed": np.linalg.norm(v_world, axis=1),
    }


def read_light_segments(path: Path) -> dict[str, list[tuple[str, float, float]]]:
    segments: dict[str, list[tuple[str, float, float]]] = {}
    if not path.exists():
        return segments
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            segments.setdefault(row["sequence"], []).append(
                (row["state"], float(row["start_rel_s"]), float(row["end_rel_s"]))
            )
    return segments


def light_state_at(segments: list[tuple[str, float, float]], t_mid: float) -> str:
    for state, start, end in segments:
        if start <= t_mid <= end:
            return state
    return "unknown"


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def bin_sequence(
    sequence: str,
    split: str,
    motion: dict[str, np.ndarray],
    light_segments: list[tuple[str, float, float]],
    bin_duration_s: float,
) -> list[dict[str, float | int | str]]:
    t = motion["t"]
    median_dt = float(np.median(np.diff(t)))
    bin_samples = int(round(bin_duration_s / median_dt))
    if bin_samples < 2:
        raise ValueError(f"{sequence}: bin_duration_s={bin_duration_s} gives bin_samples={bin_samples}")
    starts = np.arange(0, t.shape[0] - bin_samples + 1, bin_samples, dtype=np.int64)

    rows: list[dict[str, float | int | str]] = []
    for bin_index, start in enumerate(starts):
        stop = int(start + bin_samples)
        t_start = float(t[start])
        t_end = float(t[stop - 1])
        t_mid = 0.5 * (t_start + t_end)
        row: dict[str, float | int | str] = {
            "sequence": sequence,
            "split": split,
            "light_state": light_state_at(light_segments, t_mid),
            "bin_index": int(bin_index),
            "start_index": int(start),
            "stop_index": int(stop),
            "t_start_s": t_start,
            "t_end_s": t_end,
            "t_mid_s": t_mid,
            "duration_s": t_end - t_start,
            "samples": int(bin_samples),
        }

        for name in ("omega_meas", "acc_meas", "omega_gt", "acc_dyn_gt", "acc_specific_gt"):
            values = motion[name][start:stop]
            norms = np.linalg.norm(values, axis=1)
            row[f"{name}_norm_mean"] = float(norms.mean())
            row[f"{name}_norm_rms"] = rms(norms)
            row[f"{name}_norm_p95"] = float(np.percentile(norms, 95))
            for axis, suffix in enumerate(("x", "y", "z")):
                row[f"{name}_{suffix}_mean"] = float(values[:, axis].mean())
                row[f"{name}_{suffix}_std"] = float(values[:, axis].std())

        speed = motion["speed"][start:stop]
        row["speed_mean_m_s"] = float(speed.mean())
        row["speed_p95_m_s"] = float(np.percentile(speed, 95))
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


def finite_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def group_rows(rows: Iterable[dict[str, float | int | str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, float | int | str]]]:
    groups: dict[tuple[str, ...], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    return groups


def summarize_groups(
    rows: list[dict[str, float | int | str]],
    group_keys: tuple[str, ...],
    dims: list[str],
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for key, grouped in sorted(group_rows(rows, *group_keys).items()):
        row: dict[str, float | int | str] = {name: value for name, value in zip(group_keys, key)}
        row["bins"] = len(grouped)
        row["duration_s"] = float(sum(float(item["duration_s"]) for item in grouped))
        for dim in dims:
            values = np.asarray([float(item[dim]) for item in grouped], dtype=np.float64)
            for stat_name, stat_value in finite_stats(values).items():
                if stat_name == "count":
                    continue
                row[f"{dim}_{stat_name}"] = stat_value
        out.append(row)
    return out


def values_for(rows: list[dict[str, float | int | str]], dim: str) -> np.ndarray:
    return np.asarray([float(row[dim]) for row in rows], dtype=np.float64)


def norm_wasserstein(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b) / max(scale, 1e-9))


def sequence_similarity(rows: list[dict[str, float | int | str]], train_seqs: set[str]) -> tuple[list[dict], list[dict]]:
    train_rows = [row for row in rows if str(row["sequence"]) in train_seqs]
    scales = {
        dim: float(np.std(values_for(train_rows, dim), ddof=1))
        for dim in MOTION_DIMS
    }
    by_sequence = group_rows(rows, "sequence")

    to_train: list[dict] = []
    for (sequence,), seq_rows in by_sequence.items():
        row: dict[str, float | int | str] = {"sequence": sequence, "bins": len(seq_rows)}
        distances = []
        for dim in MOTION_DIMS:
            dist = norm_wasserstein(values_for(seq_rows, dim), values_for(train_rows, dim), scales[dim])
            row[f"{dim}_wdist_to_train_norm"] = dist
            distances.append(dist)
        row["motion_distance_to_train"] = float(np.nanmean(distances))
        to_train.append(row)
    to_train.sort(key=lambda item: (float(item["motion_distance_to_train"]), str(item["sequence"])))

    pairwise: list[dict] = []
    sequences = sorted(by_sequence)
    for (seq_a,) in sequences:
        for (seq_b,) in sequences:
            if seq_a >= seq_b:
                continue
            distances = []
            for dim in MOTION_DIMS:
                pooled = np.concatenate((values_for(by_sequence[(seq_a,)], dim), values_for(by_sequence[(seq_b,)], dim)))
                scale = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 1.0
                distances.append(norm_wasserstein(values_for(by_sequence[(seq_a,)], dim), values_for(by_sequence[(seq_b,)], dim), scale))
            pairwise.append({"sequence_a": seq_a, "sequence_b": seq_b, "motion_distance": float(np.nanmean(distances))})
    pairwise.sort(key=lambda item: float(item["motion_distance"]))
    return to_train, pairwise


def light_similarity(rows: list[dict[str, float | int | str]]) -> list[dict]:
    out: list[dict] = []
    for (sequence,), seq_rows in sorted(group_rows(rows, "sequence").items()):
        on_rows = [row for row in seq_rows if row["light_state"] == "light_on"]
        off_rows = [row for row in seq_rows if row["light_state"] == "light_off"]
        if not on_rows or not off_rows:
            continue
        row: dict[str, float | int | str] = {"sequence": sequence, "light_on_bins": len(on_rows), "light_off_bins": len(off_rows)}
        distances = []
        for dim in MOTION_DIMS:
            pooled = np.concatenate((values_for(on_rows, dim), values_for(off_rows, dim)))
            scale = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 1.0
            dist = norm_wasserstein(values_for(on_rows, dim), values_for(off_rows, dim), scale)
            row[f"{dim}_on_mean"] = float(values_for(on_rows, dim).mean())
            row[f"{dim}_off_mean"] = float(values_for(off_rows, dim).mean())
            row[f"{dim}_wdist_on_off_norm"] = dist
            distances.append(dist)
        row["motion_distance_light_on_off"] = float(np.nanmean(distances))
        out.append(row)
    out.sort(key=lambda item: float(item["motion_distance_light_on_off"]))
    return out


def plot_motion(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sequences = sorted({str(row["sequence"]) for row in rows})

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for ax, dim, ylabel in zip(
        axes,
        MOTION_DIMS,
        ["GT omega norm (rad/s)", "GT dynamic acc norm (m/s^2)", "GT speed (m/s)"],
    ):
        data = [
            [float(row[dim]) for row in rows if row["sequence"] == seq]
            for seq in sequences
        ]
        ax.boxplot(data, tick_labels=sequences, showfliers=False)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("sequence")
    fig.tight_layout()
    fig.savefig(plot_dir / "sequence_motion_boxplot.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"train": "C0", "test": "C3", "other": "C2"}
    for ax, dim, title in zip(axes, MOTION_DIMS, ["omega", "dynamic acc", "speed"]):
        for split in ("train", "test", "other"):
            values = [float(row[dim]) for row in rows if row["split"] == split]
            if values:
                ax.hist(values, bins=60, alpha=0.45, density=True, color=colors[split], label=split)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "train_test_motion_hist.png", dpi=180)
    plt.close(fig)


def write_analysis(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    summary_by_sequence: list[dict[str, float | int | str]],
    to_train: list[dict[str, float | int | str]],
    pairwise: list[dict[str, float | int | str]],
    light_sim: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    missing_requested: list[str],
) -> None:
    sequence_rows = {str(row["sequence"]): row for row in summary_by_sequence}
    top_to_train = [row for row in to_train if row["sequence"] in args.test_seqs][:8]
    closest_pairs = pairwise[:12]

    lines = [
        "# corldata GT-Differenced Motion Analysis",
        "",
        f"Data root: `{args.data_root}`",
        f"Bin duration: `{args.bin_duration_s:.3f} s`",
        f"Train sequences: `{', '.join(args.train_seqs)}`",
        f"Test sequences: `{', '.join(args.test_seqs)}`",
        "",
        "This analysis differentiates the aligned GT trajectory to estimate body-frame angular velocity and acceleration, then compares 10 Hz motion bins. `acc_dyn_gt` is translational acceleration in body frame; `acc_specific_gt` is the ideal accelerometer specific force including gravity.",
        "",
    ]
    if missing_requested:
        lines += [
            "## Missing Requested Sequences",
            "",
            "- " + ", ".join(f"`{seq}`" for seq in missing_requested) + " were requested but not present in the aggregate data root.",
            "",
        ]

    lines += [
        "## Sequence Motion Summary",
        "",
        "| sequence | split | bins | omega mean | omega p95 | dyn acc mean | dyn acc p95 | speed mean | speed p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seq in sorted(sequence_rows):
        row = sequence_rows[seq]
        lines.append(
            f"| {seq} | {row['split']} | {int(row['bins'])} | "
            f"{float(row['omega_gt_norm_mean_mean']):.4f} | {float(row['omega_gt_norm_mean_p95']):.4f} | "
            f"{float(row['acc_dyn_gt_norm_mean_mean']):.4f} | {float(row['acc_dyn_gt_norm_mean_p95']):.4f} | "
            f"{float(row['speed_mean_m_s_mean']):.4f} | {float(row['speed_mean_m_s_p95']):.4f} |"
        )
    lines += ["", "## Test Sequences Closest To Training Pool", "", "| sequence | distance to train | omega dist | acc dist | speed dist |", "|---|---:|---:|---:|---:|"]
    for row in top_to_train:
        lines.append(
            f"| {row['sequence']} | {float(row['motion_distance_to_train']):.3f} | "
            f"{float(row['omega_gt_norm_mean_wdist_to_train_norm']):.3f} | "
            f"{float(row['acc_dyn_gt_norm_mean_wdist_to_train_norm']):.3f} | "
            f"{float(row['speed_mean_m_s_wdist_to_train_norm']):.3f} |"
        )

    lines += ["", "## Closest Sequence Pairs", "", "| sequence A | sequence B | motion distance |", "|---|---|---:|"]
    for row in closest_pairs:
        lines.append(f"| {row['sequence_a']} | {row['sequence_b']} | {float(row['motion_distance']):.3f} |")

    lines += [
        "",
        "## Light-On / Light-Off Motion Similarity",
        "",
        "| sequence | distance on/off | on bins | off bins | omega on/off | dyn acc on/off | speed on/off |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in light_sim:
        lines.append(
            f"| {row['sequence']} | {float(row['motion_distance_light_on_off']):.3f} | "
            f"{int(row['light_on_bins'])} | {int(row['light_off_bins'])} | "
            f"{float(row['omega_gt_norm_mean_on_mean']):.3f}/{float(row['omega_gt_norm_mean_off_mean']):.3f} | "
            f"{float(row['acc_dyn_gt_norm_mean_on_mean']):.3f}/{float(row['acc_dyn_gt_norm_mean_off_mean']):.3f} | "
            f"{float(row['speed_mean_m_s_on_mean']):.3f}/{float(row['speed_mean_m_s_off_mean']):.3f} |"
        )

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- Smaller normalized Wasserstein distances mean closer motion distributions; values below about `0.5` are usually close enough to treat as similar for this diagnostic, while values above `1.0` indicate a clear regime shift.",
        "- This measures motion similarity, not label quality. Low-quality pseudo-GT sequences such as previously flagged `S22` can still look motion-similar but should remain low-confidence for supervision.",
        "- Using testing-sequence light-on VO as training supervision is a transductive/semi-supervised setting. It should be reported separately from the strict baseline because it uses part of the test sequence at training/adaptation time.",
        "",
        "## Outputs",
        "",
        "- `motion_10hz.csv`: per-bin GT-differenced motion and raw IMU summaries.",
        "- `summary_by_sequence.csv`: per-sequence motion statistics.",
        "- `sequence_similarity_to_train.csv`: per-sequence distance to the training motion pool.",
        "- `pairwise_sequence_similarity.csv`: pairwise sequence motion distances.",
        "- `light_state_summary.csv` and `light_on_off_similarity.csv`: light-state motion summaries for processed testing sequences.",
        "- `plots/sequence_motion_boxplot.png` and `plots/train_test_motion_hist.png`: quick visual diagnostics.",
        "",
    ]
    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_sequences = discover_sequences(args.data_root)
    sequences = args.sequences if args.sequences is not None else all_sequences
    missing_requested = [seq for seq in sequences if seq not in all_sequences]
    sequences = [seq for seq in sequences if seq in all_sequences]

    train_seqs = set(args.train_seqs)
    test_seqs = set(args.test_seqs)
    light_segments = read_light_segments(args.light_segments_csv)
    gravity = np.asarray(args.gravity, dtype=np.float64)

    rows: list[dict[str, float | int | str]] = []
    for sequence in sequences:
        split = "train" if sequence in train_seqs else "test" if sequence in test_seqs else "other"
        t, u, p, rot = load_sequence(args.data_root, sequence)
        motion = compute_gt_motion(
            t,
            u,
            p,
            rot,
            gravity,
            args.position_sg_window,
            args.position_sg_order,
        )
        rows.extend(
            bin_sequence(
                sequence,
                split,
                motion,
                light_segments.get(sequence, []),
                args.bin_duration_s,
            )
        )

    write_rows(args.output_dir / "motion_10hz.csv", rows)
    summary_by_sequence = summarize_groups(rows, ("sequence", "split"), MOTION_DIMS)
    summary_by_light = summarize_groups(rows, ("sequence", "light_state"), MOTION_DIMS)
    write_rows(args.output_dir / "summary_by_sequence.csv", summary_by_sequence)
    write_rows(args.output_dir / "light_state_summary.csv", summary_by_light)
    to_train, pairwise = sequence_similarity(rows, train_seqs)
    light_sim = light_similarity(rows)
    write_rows(args.output_dir / "sequence_similarity_to_train.csv", to_train)
    write_rows(args.output_dir / "pairwise_sequence_similarity.csv", pairwise)
    write_rows(args.output_dir / "light_on_off_similarity.csv", light_sim)
    plot_motion(rows, args.output_dir)
    write_analysis(args.output_dir, rows, summary_by_sequence, to_train, pairwise, light_sim, args, missing_requested)

    metadata = {
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "sequences": sequences,
        "missing_requested": missing_requested,
        "train_seqs": args.train_seqs,
        "test_seqs": args.test_seqs,
        "bin_duration_s": args.bin_duration_s,
        "gravity": args.gravity,
        "rows": len(rows),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
