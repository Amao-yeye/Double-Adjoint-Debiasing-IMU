#!/usr/bin/env python3
"""Evaluate pure IMU preintegration over natural corldata light-off intervals.

This isolates the learned bias trajectory from OpenVINS. For each detected
light-off interval, the script initializes from the GT state at the interval
start, preintegrates raw IMU with different bias corrections, and compares the
result to GT over the interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PLATO_ROOT
    / "results"
    / "debias_imu"
    / "corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue"
)
DEFAULT_LIGHT_SEGMENTS = PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "light_segments" / "light_segments.csv"
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "light_off_preintegration"


@dataclass(frozen=True)
class Segment:
    index: int
    state: str
    start: float
    end: float


def hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=np.float64)


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = hat(w)
    if theta < 1.0e-12:
        return np.eye(3) + W
    return np.eye(3) + math.sin(theta) / theta * W + (1.0 - math.cos(theta)) / (theta * theta) * (W @ W)


def so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1.0e-12:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * math.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            return pickle.load(f)


def load_segments(path: Path, sequence: str) -> list[Segment]:
    segments: list[Segment] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if row["sequence"] != sequence:
                continue
            segments.append(
                Segment(
                    index=idx,
                    state=row["state"],
                    start=float(row["start_gt_time_s"]),
                    end=float(row["end_gt_time_s"]),
                )
            )
    if not segments:
        raise ValueError(f"No light segments for {sequence} in {path}")
    return segments


def propagate(
    t: np.ndarray,
    imu_raw: np.ndarray,
    bias: np.ndarray,
    R0: np.ndarray,
    v0: np.ndarray,
    p0: np.ndarray,
    gravity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_states = np.empty((t.shape[0], 3, 3), dtype=np.float64)
    v_states = np.empty((t.shape[0], 3), dtype=np.float64)
    p_states = np.empty((t.shape[0], 3), dtype=np.float64)
    R_states[0] = R0
    v_states[0] = v0
    p_states[0] = p0
    for i in range(t.shape[0] - 1):
        dt = float(t[i + 1] - t[i])
        omega = imu_raw[i, :3] - bias[i, :3]
        acc_body = imu_raw[i, 3:6] - bias[i, 3:6]
        acc_world = R_states[i] @ acc_body + gravity
        R_states[i + 1] = R_states[i] @ so3_exp(omega * dt)
        v_states[i + 1] = v_states[i] + acc_world * dt
        p_states[i + 1] = p_states[i] + v_states[i] * dt
    return R_states, v_states, p_states


def build_method_bias(method: str, learned_bias: np.ndarray) -> np.ndarray:
    bias = np.zeros_like(learned_bias)
    if method == "raw":
        return bias
    if method == "learned":
        return learned_bias.copy()
    if method == "learned_gyro_only":
        bias[:, :3] = learned_bias[:, :3]
        return bias
    if method == "learned_accel_only":
        bias[:, 3:6] = learned_bias[:, 3:6]
        return bias
    if method == "learned_const_start":
        return np.repeat(learned_bias[0:1], learned_bias.shape[0], axis=0)
    if method == "learned_delta_accel":
        bias[:, 3:6] = learned_bias[:, 3:6] - learned_bias[0:1, 3:6]
        return bias
    if method == "neg_learned_delta_accel":
        bias[:, 3:6] = -(learned_bias[:, 3:6] - learned_bias[0:1, 3:6])
        return bias
    raise ValueError(f"Unknown method: {method}")


def metric_summary(
    R_gt: np.ndarray,
    v_gt: np.ndarray,
    p_gt: np.ndarray,
    R_est: np.ndarray,
    v_est: np.ndarray,
    p_est: np.ndarray,
) -> dict[str, float]:
    pos = np.linalg.norm(p_est - p_gt, axis=1)
    vel = np.linalg.norm(v_est - v_gt, axis=1)
    rot = np.array([np.linalg.norm(so3_log(R_gt[i].T @ R_est[i])) for i in range(R_gt.shape[0])]) * 180.0 / math.pi
    return {
        "pos_rmse_m": float(np.sqrt(np.mean(pos * pos))),
        "pos_mean_m": float(np.mean(pos)),
        "pos_max_m": float(np.max(pos)),
        "pos_end_m": float(pos[-1]),
        "vel_rmse_mps": float(np.sqrt(np.mean(vel * vel))),
        "vel_end_mps": float(vel[-1]),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot * rot))),
        "rot_max_deg": float(np.max(rot)),
        "rot_end_deg": float(rot[-1]),
    }


def evaluate_sequence(args: argparse.Namespace, sequence: str) -> list[dict[str, Any]]:
    result_path = args.results_dir / "Sequnces-results" / sequence / "FullTra_result.p"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    data = load_pickle(result_path)
    sol_pred = as_numpy(data["sol_pred"]).astype(np.float64)
    X_gt = as_numpy(data["X_gt"]).astype(np.float64)
    t_rel = as_numpy(data["t_gt"]).astype(np.float64)
    imu_raw = as_numpy(data["imu_raw"]).astype(np.float64)
    timestamp_start_s = float(as_numpy(data["timestamp_start"])) * 1.0e-9
    t_abs = timestamp_start_s + t_rel
    learned_bias = np.column_stack([sol_pred[:, 4:7], sol_pred[:, 13:16]])
    R_gt = X_gt[:, :3, :3]
    v_gt = X_gt[:, :3, 3]
    p_gt = X_gt[:, :3, 4]
    gravity = np.asarray(args.gravity, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for segment in load_segments(args.light_segments, sequence):
        if segment.state != "light_off":
            continue
        mask = (t_abs >= segment.start) & (t_abs <= segment.end)
        indices = np.flatnonzero(mask)
        if indices.shape[0] < 2:
            continue
        start = int(indices[0])
        stop = int(indices[-1]) + 1
        t_seg = t_abs[start:stop]
        base = {
            "sequence": sequence,
            "segment_index": segment.index,
            "segment_start_s": segment.start,
            "segment_end_s": segment.end,
            "start_sample_s": float(t_seg[0]),
            "end_sample_s": float(t_seg[-1]),
            "duration_s": float(t_seg[-1] - t_seg[0]),
            "samples": int(t_seg.shape[0]),
        }
        for method in args.methods:
            bias = build_method_bias(method, learned_bias[start:stop])
            R_est, v_est, p_est = propagate(
                t_seg,
                imu_raw[start:stop],
                bias,
                R_gt[start],
                v_gt[start],
                p_gt[start],
                gravity,
            )
            row = dict(base)
            row["method"] = method
            row.update(metric_summary(R_gt[start:stop], v_gt[start:stop], p_gt[start:stop], R_est, v_est, p_est))
            rows.append(row)
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["sequence"], row["method"]), []).append(row)
    lines = [
        "# corldata Light-Off Pure IMU Preintegration",
        "",
        "Each light-off interval is initialized from GT state at interval start. OpenVINS is not used.",
        "",
        "| sequence | method | intervals | pos RMSE mean m | pos end mean/max m | vel RMSE mean m/s | rot RMSE mean deg | rot end mean/max deg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(grouped):
        seq, method = key
        vals = grouped[key]
        pos_rmse = np.array([float(v["pos_rmse_m"]) for v in vals])
        pos_end = np.array([float(v["pos_end_m"]) for v in vals])
        vel_rmse = np.array([float(v["vel_rmse_mps"]) for v in vals])
        rot_rmse = np.array([float(v["rot_rmse_deg"]) for v in vals])
        rot_end = np.array([float(v["rot_end_deg"]) for v in vals])
        lines.append(
            f"| {seq} | {method} | {len(vals)} | {pos_rmse.mean():.3f} | "
            f"{pos_end.mean():.3f}/{pos_end.max():.3f} | {vel_rmse.mean():.3f} | "
            f"{rot_rmse.mean():.3f} | {rot_end.mean():.3f}/{rot_end.max():.3f} |"
        )
    lines.extend(
        [
            "",
            "Per-interval endpoint details:",
            "",
            "| sequence | interval | duration s | method | pos end m | vel end m/s | rot end deg |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['sequence']} | {row['segment_index']} | {row['duration_s']:.3f} | {row['method']} | "
            f"{row['pos_end_m']:.3f} | {row['vel_end_mps']:.3f} | {row['rot_end_deg']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", dest="sequences", default=[])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--light-segments", type=Path, default=DEFAULT_LIGHT_SEGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, 0.0, -9.81])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "raw",
            "learned",
            "learned_gyro_only",
            "learned_accel_only",
            "learned_const_start",
            "learned_delta_accel",
            "neg_learned_delta_accel",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = args.sequences or ["S13", "S15"]
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        print(f"[light-off-preintegration] {sequence}", flush=True)
        rows.extend(evaluate_sequence(args, sequence))
    if not rows:
        raise RuntimeError("No light-off intervals evaluated")
    write_outputs(rows, args.output_dir)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
