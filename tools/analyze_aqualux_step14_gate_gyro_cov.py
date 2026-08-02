#!/usr/bin/env python3
"""Step 14 diagnostics for AquaLux OpenVINS bias compensation.

Outputs:
- per-sequence gate features
- gyro/acc bias convention and sign diagnostics
- light-off pure preintegration covariance audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
BIASDY_DIR = REPO_ROOT / "external" / "Debias_IMU" / "BiasDy"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(BIASDY_DIR) not in sys.path:
    sys.path.insert(0, str(BIASDY_DIR))

import lie_algebra as Lie  # noqa: E402
from dataset import pload  # noqa: E402
from evaluate_aqualux_openvins_compensate1 import (  # noqa: E402
    DEFAULT_BASELINE_ROOT,
    DEFAULT_RAW_ROOT,
    bias_derivative,
    choose_device,
    load_models,
    read_openvins_estimate,
    read_weak_visual_segments,
)
from run_aqualux_openvins_raw import TEST_SEQUENCES  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "baseline" / "step14_gate_gyro_cov"
DEFAULT_CACHE_DIR = REPO_ROOT / "Results" / "AquaLux" / "preintegration" / "rss_baseline_unseen_corrected" / "cache"
DEFAULT_RAW_METRICS = REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "raw" / "metrics.csv"
DEFAULT_ACC_GATE_METRICS = (
    REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "baseline" / "segment_delta_acc_covgate" / "metrics.csv"
)
DEFAULT_MANIFEST = REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "configs" / "raw" / "config_manifest.json"
GRAVITY = np.array([0.0, 0.0, -9.7879], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-openvins-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--gyro-checkpoint", type=Path, default=None)
    parser.add_argument("--acc-checkpoint", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-metrics", type=Path, default=DEFAULT_RAW_METRICS)
    parser.add_argument("--acc-gate-metrics", type=Path, default=DEFAULT_ACC_GATE_METRICS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sequences", nargs="+", default=list(TEST_SEQUENCES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--msckf-update-threshold", type=float, default=5e-4)
    parser.add_argument("--min-weak-duration", type=float, default=0.5)
    parser.add_argument("--merge-gap", type=float, default=0.2)
    parser.add_argument("--gate-max-position-cov-trace", type=float, default=1.0)
    parser.add_argument("--max-segments-per-seq", type=int, default=0, help="0 means all segments.")
    return parser.parse_args()


def read_csv_by_sequence(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sequence"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def rotation_error_deg(r_gt: np.ndarray, r_pred: np.ndarray) -> float:
    err = Rotation.from_matrix(r_gt.T @ r_pred).magnitude()
    return float(err * 180.0 / math.pi)


def covariance_features(odom_path: Path) -> dict[str, float | int]:
    pr: list[float] = []
    pt: list[float] = []
    with odom_path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 20:
                continue
            values = [float(x) for x in parts[:20]]
            pr.append(values[8] + values[11] + values[13])
            pt.append(values[14] + values[17] + values[19])
    if not pr:
        return {
            "odom_rows": 0,
            "raw_max_rotation_cov_trace": math.nan,
            "raw_mean_rotation_cov_trace": math.nan,
            "raw_max_position_cov_trace": math.nan,
            "raw_mean_position_cov_trace": math.nan,
        }
    return {
        "odom_rows": len(pr),
        "raw_max_rotation_cov_trace": float(np.max(pr)),
        "raw_mean_rotation_cov_trace": float(np.mean(pr)),
        "raw_max_position_cov_trace": float(np.max(pt)),
        "raw_mean_position_cov_trace": float(np.mean(pt)),
    }


def load_cache(cache_dir: Path, sequence: str) -> dict[str, np.ndarray | float]:
    imu = pload(str(cache_dir), f"{sequence}.p")
    gt = pload(str(cache_dir), f"{sequence}_gt.p")
    t_rel = to_np(imu["t_imu"]).astype(np.float64)
    t_abs = float(imu["t_start"]) + t_rel
    return {
        "t": t_abs,
        "u": to_np(imu["u_imu"]).astype(np.float64),
        "w_bias": to_np(imu["w_bias"]).astype(np.float64),
        "a_bias": to_np(imu["a_bias"]).astype(np.float64),
        "R": to_np(gt["R_gt"]).astype(np.float64),
        "v": to_np(gt["v_gt"]).astype(np.float64),
        "p": to_np(gt["p_gt"]).astype(np.float64),
    }


def to_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def nearest_index(stamps: np.ndarray, stamp: float) -> int:
    idx = int(np.searchsorted(stamps, stamp))
    if idx <= 0:
        return 0
    if idx >= len(stamps):
        return len(stamps) - 1
    return idx if abs(stamps[idx] - stamp) < abs(stamps[idx - 1] - stamp) else idx - 1


def segment_indices(stamps: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.where((stamps >= start) & (stamps <= end))[0]


def predict_biases(
    stamps: np.ndarray,
    imu: np.ndarray,
    bg0: np.ndarray,
    ba0: np.ndarray,
    bw_model,
    ba_model,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(stamps) < 2:
        return np.tile(bg0, (len(stamps), 1)), np.tile(ba0, (len(stamps), 1))
    imu_dot = np.gradient(imu, stamps, axis=0, edge_order=1)
    bg = bg0.astype(np.float64).copy()
    ba = ba0.astype(np.float64).copy()
    bg_hist = np.zeros((len(stamps), 3), dtype=np.float64)
    ba_hist = np.zeros((len(stamps), 3), dtype=np.float64)
    for i in range(len(stamps)):
        bg_hist[i] = bg
        ba_hist[i] = ba
        if i + 1 >= len(stamps):
            break
        y = np.zeros(16, dtype=np.float64)
        y[3] = stamps[i] - stamps[0]
        y[4:7] = bg
        y[13:16] = ba
        bg_dot, ba_dot = bias_derivative(bw_model, ba_model, y, imu[i], imu_dot[i], device)
        dt = float(stamps[i + 1] - stamps[i])
        if dt > 0:
            bg = bg + bg_dot * dt
            ba = ba + ba_dot * dt
    return bg_hist, ba_hist


def integrate_with_covariance(
    stamps: np.ndarray,
    imu: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    p0: np.ndarray,
    bg: np.ndarray,
    ba: np.ndarray,
    noise: dict[str, float],
) -> dict[str, object]:
    r = Rotation.from_matrix(r0)
    v = v0.astype(np.float64).copy()
    p = p0.astype(np.float64).copy()
    P = np.zeros((15, 15), dtype=np.float64)
    min_eig = 0.0
    max_trace_pose = 0.0
    for i in range(len(stamps) - 1):
        dt = float(stamps[i + 1] - stamps[i])
        if dt <= 0:
            continue
        bg_i = bg[i] if bg.ndim == 2 else bg
        ba_i = ba[i] if ba.ndim == 2 else ba
        w_corr = imu[i, :3] - bg_i
        a_corr = imu[i, 3:6] - ba_i
        Rm = r.as_matrix()

        F = np.zeros((15, 15), dtype=np.float64)
        F[0:3, 0:3] = -skew(w_corr)
        F[0:3, 9:12] = -np.eye(3)
        F[3:6, 0:3] = -Rm @ skew(a_corr)
        F[3:6, 12:15] = -Rm
        F[6:9, 3:6] = np.eye(3)
        Phi = np.eye(15) + F * dt
        G = np.zeros((15, 12), dtype=np.float64)
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = -Rm
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        Qc = np.diag(
            [
                noise["sigma_w"] ** 2,
                noise["sigma_w"] ** 2,
                noise["sigma_w"] ** 2,
                noise["sigma_a"] ** 2,
                noise["sigma_a"] ** 2,
                noise["sigma_a"] ** 2,
                noise["sigma_wb"] ** 2,
                noise["sigma_wb"] ** 2,
                noise["sigma_wb"] ** 2,
                noise["sigma_ab"] ** 2,
                noise["sigma_ab"] ** 2,
                noise["sigma_ab"] ** 2,
            ]
        )
        P = Phi @ P @ Phi.T + G @ Qc @ G.T * dt
        P = 0.5 * (P + P.T)
        eig = np.linalg.eigvalsh(P)
        min_eig = min(min_eig, float(np.min(eig)))
        max_trace_pose = max(max_trace_pose, float(np.trace(P[:9, :9])))

        # Nominal state propagation uses the same gravity convention as AquaLux preintegration.
        p = p + v * dt
        v = v + (Rm @ a_corr + GRAVITY) * dt
        r = r * Rotation.from_rotvec(w_corr * dt)

    return {
        "R": r.as_matrix(),
        "v": v,
        "p": p,
        "P": P,
        "min_cov_eig": min_eig,
        "max_pose_cov_trace": max_trace_pose,
        "final_pose_cov_trace": float(np.trace(P[:9, :9])),
        "final_rot_cov_trace": float(np.trace(P[:3, :3])),
        "final_vel_cov_trace": float(np.trace(P[3:6, 3:6])),
        "final_pos_cov_trace": float(np.trace(P[6:9, 6:9])),
    }


def read_noise(manifest_path: Path) -> dict[str, float]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    imu = manifest["imu_noise"]
    return {
        "sigma_w": float(imu["gyroscope_noise_density"]),
        "sigma_wb": float(imu["gyroscope_random_walk"]),
        "sigma_a": float(imu["accelerometer_noise_density"]),
        "sigma_ab": float(imu["accelerometer_random_walk"]),
    }


def analyze_sequence(
    sequence: str,
    args: argparse.Namespace,
    raw_metrics: dict[str, dict[str, str]],
    acc_metrics: dict[str, dict[str, str]],
    bw_model,
    ba_model,
    device: str,
    noise: dict[str, float],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    raw_dir = args.raw_openvins_root / sequence
    cov = covariance_features(raw_dir / "odomimu.txt")
    segments = read_weak_visual_segments(
        raw_dir / "timing.txt",
        args.msckf_update_threshold,
        args.min_weak_duration,
        args.merge_gap,
    )
    if args.max_segments_per_seq > 0:
        segments = segments[: args.max_segments_per_seq]
    cache = load_cache(args.cache_dir, sequence)
    est = read_openvins_estimate(raw_dir / "estimate.txt")
    stamps = cache["t"]
    total_duration = float(stamps[-1] - stamps[0])
    weak_duration = float(sum(end - start for start, end, _ in segments))
    raw_row = raw_metrics[sequence]
    acc_row = acc_metrics.get(sequence, {})
    gate_pass = float(cov["raw_max_position_cov_trace"]) > args.gate_max_position_cov_trace
    gate_row: dict[str, object] = {
        "sequence": sequence,
        "raw_position_rmse_m": float(raw_row["ate_rmse_pos_m"]),
        "raw_rotation_rmse_deg": float(raw_row["ate_rmse_rot_deg"]),
        "acc_covgate_position_rmse_m": float(acc_row["ate_rmse_pos_m"]) if acc_row else math.nan,
        "acc_covgate_rotation_rmse_deg": float(acc_row["ate_rmse_rot_deg"]) if acc_row else math.nan,
        **cov,
        "weak_segments": len(segments),
        "weak_duration_s": weak_duration,
        "weak_fraction": weak_duration / total_duration if total_duration > 0 else math.nan,
        "acc_delta_gate_pass": gate_pass,
        "gyro_delta_gate_pass": False,
        "gate_rule": f"raw_max_position_cov_trace>{args.gate_max_position_cov_trace:g}",
        "gyro_gate_reason": "disabled_until_bias_delta_sign_and_OpenVINS_stability_are_fixed",
    }

    bias_rows: list[dict[str, object]] = []
    cov_rows: list[dict[str, object]] = []
    for seg_index, (start, end, weak_count) in enumerate(segments):
        idx = segment_indices(stamps, start, end)
        if len(idx) < 5:
            continue
        seed_idx = int(np.searchsorted(est["t"], start, side="right") - 1)
        if seed_idx < 0:
            continue
        cache_seed_idx = idx[0]
        bg_seed_ov = est["bg"][seed_idx]
        ba_seed_ov = est["ba"][seed_idx]
        bg_gt = cache["w_bias"][idx]
        ba_gt = cache["a_bias"][idx]
        imu = cache["u"][idx]
        seg_t = stamps[idx]
        bg_pred, ba_pred = predict_biases(seg_t, imu, bg_seed_ov, ba_seed_ov, bw_model, ba_model, device)

        bg_gt_start = cache["w_bias"][cache_seed_idx]
        ba_gt_start = cache["a_bias"][cache_seed_idx]
        bg_pred_delta = bg_pred - bg_seed_ov[None, :]
        ba_pred_delta = ba_pred - ba_seed_ov[None, :]
        bg_gt_delta = bg_gt - bg_gt_start[None, :]
        ba_gt_delta = ba_gt - ba_gt_start[None, :]
        gyro_delta_rmse = rmse_vec(bg_pred_delta - bg_gt_delta)
        gyro_delta_flip_rmse = rmse_vec(-bg_pred_delta - bg_gt_delta)
        acc_delta_rmse = rmse_vec(ba_pred_delta - ba_gt_delta)
        acc_delta_flip_rmse = rmse_vec(-ba_pred_delta - ba_gt_delta)
        bias_rows.append(
            {
                "sequence": sequence,
                "segment_index": seg_index,
                "start": start,
                "end": end,
                "duration_s": end - start,
                "weak_update_rows": weak_count,
                "samples": len(idx),
                "ov_seed_bg_norm": norm(bg_seed_ov),
                "gt_start_bg_norm": norm(bg_gt_start),
                "ov_minus_gt_bg_norm": norm(bg_seed_ov - bg_gt_start),
                "ov_seed_ba_norm": norm(ba_seed_ov),
                "gt_start_ba_norm": norm(ba_gt_start),
                "ov_minus_gt_ba_norm": norm(ba_seed_ov - ba_gt_start),
                "gyro_delta_rmse_same": gyro_delta_rmse,
                "gyro_delta_rmse_signflip": gyro_delta_flip_rmse,
                "gyro_signflip_better": gyro_delta_flip_rmse < gyro_delta_rmse,
                "acc_delta_rmse_same": acc_delta_rmse,
                "acc_delta_rmse_signflip": acc_delta_flip_rmse,
                "acc_signflip_better": acc_delta_flip_rmse < acc_delta_rmse,
                "mean_pred_gyro_delta_norm": float(np.mean(np.linalg.norm(bg_pred_delta, axis=1))),
                "mean_gt_gyro_delta_norm": float(np.mean(np.linalg.norm(bg_gt_delta, axis=1))),
                "mean_pred_acc_delta_norm": float(np.mean(np.linalg.norm(ba_pred_delta, axis=1))),
                "mean_gt_acc_delta_norm": float(np.mean(np.linalg.norm(ba_gt_delta, axis=1))),
            }
        )

        r0 = cache["R"][idx[0]]
        v0 = cache["v"][idx[0]]
        p0 = cache["p"][idx[0]]
        r_end = cache["R"][idx[-1]]
        v_end = cache["v"][idx[-1]]
        p_end = cache["p"][idx[-1]]
        variants = {
            "ideal_gt_bias": (bg_gt, ba_gt),
            "ov_seed_constant": (bg_seed_ov, ba_seed_ov),
            "model_gyro_only": (bg_pred, np.tile(ba_seed_ov, (len(idx), 1))),
            "model_acc_only": (np.tile(bg_seed_ov, (len(idx), 1)), ba_pred),
            "model_both": (bg_pred, ba_pred),
        }
        for variant, (bg_use, ba_use) in variants.items():
            out = integrate_with_covariance(seg_t, imu, r0, v0, p0, bg_use, ba_use, noise)
            pos_err = norm(out["p"] - p_end)
            vel_err = norm(out["v"] - v_end)
            rot_err = rotation_error_deg(r_end, out["R"])
            cov_rows.append(
                {
                    "sequence": sequence,
                    "segment_index": seg_index,
                    "variant": variant,
                    "duration_s": end - start,
                    "samples": len(idx),
                    "position_error_m": pos_err,
                    "velocity_error_mps": vel_err,
                    "rotation_error_deg": rot_err,
                    "final_pos_cov_trace": out["final_pos_cov_trace"],
                    "final_rot_cov_trace": out["final_rot_cov_trace"],
                    "final_vel_cov_trace": out["final_vel_cov_trace"],
                    "final_pose_cov_trace": out["final_pose_cov_trace"],
                    "max_pose_cov_trace": out["max_pose_cov_trace"],
                    "min_cov_eig": out["min_cov_eig"],
                    "psd_ok": out["min_cov_eig"] >= -1e-12,
                    "pos_error_over_3sigma_trace": pos_err / (3.0 * math.sqrt(max(out["final_pos_cov_trace"], 1e-18))),
                    "rot_error_rad_over_3sigma_trace": (rot_err * math.pi / 180.0)
                    / (3.0 * math.sqrt(max(out["final_rot_cov_trace"], 1e-18))),
                }
            )
    return gate_row, bias_rows, cov_rows


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def rmse_vec(v: np.ndarray) -> float:
    if len(v) == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.sum(v * v, axis=1))))


def summarize(rows: list[dict[str, object]], key: str, selector: tuple[str, object] | None = None) -> float:
    vals = []
    for row in rows:
        if selector is not None and row.get(selector[0]) != selector[1]:
            continue
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            vals.append(float(value))
    return float(np.mean(vals)) if vals else math.nan


def write_gate_yaml(path: Path, gate_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Fixed Step14 diagnostic gates for later AquaLux OpenVINS methods",
        "rule:",
        "  acc_delta:",
        "    enabled_when: raw_max_position_cov_trace > 1.0",
        "    correction_mode: delta",
        "    components: acc",
        "  gyro_delta:",
        "    enabled: false",
        "    reason: disabled_until_bias_delta_sign_and_OpenVINS_stability_are_fixed",
        "sequences:",
    ]
    for row in gate_rows:
        lines.extend(
            [
                f"  {row['sequence']}:",
                f"    acc_delta_gate_pass: {str(bool(row['acc_delta_gate_pass'])).lower()}",
                f"    gyro_delta_gate_pass: {str(bool(row['gyro_delta_gate_pass'])).lower()}",
                f"    raw_max_position_cov_trace: {float(row['raw_max_position_cov_trace']):.12g}",
                f"    weak_fraction: {float(row['weak_fraction']):.12g}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    gate_rows: list[dict[str, object]],
    bias_rows: list[dict[str, object]],
    cov_rows: list[dict[str, object]],
    noise: dict[str, float],
) -> None:
    ideal_pos = summarize(cov_rows, "position_error_m", ("variant", "ideal_gt_bias"))
    ideal_rot = summarize(cov_rows, "rotation_error_deg", ("variant", "ideal_gt_bias"))
    seed_pos = summarize(cov_rows, "position_error_m", ("variant", "ov_seed_constant"))
    seed_rot = summarize(cov_rows, "rotation_error_deg", ("variant", "ov_seed_constant"))
    gyro_rmse = summarize(bias_rows, "gyro_delta_rmse_same")
    gyro_flip = summarize(bias_rows, "gyro_delta_rmse_signflip")
    acc_rmse = summarize(bias_rows, "acc_delta_rmse_same")
    acc_flip = summarize(bias_rows, "acc_delta_rmse_signflip")
    psd_bad = sum(1 for row in cov_rows if not bool(row.get("psd_ok")))
    lines = [
        "# Step 14 Gate, Gyro, and Covariance Diagnostics",
        "",
        "## Scope",
        "",
        "- Dataset: AquaLux Test sequences.",
        "- Purpose: diagnose gyro compensation, fixed gates, and pure preintegration covariance.",
        "- Gravity used in preintegration: `[0, 0, -9.7879]`.",
        "",
        "## IMU Noise",
        "",
        "```text",
        f"gyroscope_noise_density = {noise['sigma_w']}",
        f"gyroscope_random_walk = {noise['sigma_wb']}",
        f"accelerometer_noise_density = {noise['sigma_a']}",
        f"accelerometer_random_walk = {noise['sigma_ab']}",
        "```",
        "",
        "## Fixed Diagnostic Gate",
        "",
        f"- Acc delta gate: `raw_max_position_cov_trace > {args.gate_max_position_cov_trace:g}`.",
        "- Gyro delta gate: disabled for now.",
        "- Reason: direct gyro-delta OpenVINS injection already caused severe rotation divergence; bias diagnostics below are used before re-enabling it.",
        "",
        "| sequence | raw pos | raw rot | max Pt trace | weak frac | acc gate | gyro gate |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in gate_rows:
        lines.append(
            f"| `{row['sequence']}` | {float(row['raw_position_rmse_m']):.6g} | "
            f"{float(row['raw_rotation_rmse_deg']):.6g} | {float(row['raw_max_position_cov_trace']):.6g} | "
            f"{float(row['weak_fraction']):.6g} | {row['acc_delta_gate_pass']} | {row['gyro_delta_gate_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Bias Convention Diagnostics",
            "",
            f"- Mean gyro delta RMSE, same sign: `{gyro_rmse:.6g}`.",
            f"- Mean gyro delta RMSE, sign flipped: `{gyro_flip:.6g}`.",
            f"- Mean acc delta RMSE, same sign: `{acc_rmse:.6g}`.",
            f"- Mean acc delta RMSE, sign flipped: `{acc_flip:.6g}`.",
            "",
            "Interpretation:",
            "",
            "- The diagnostics compare model-predicted bias delta from OpenVINS seed bias against GT-derived bias delta from the AquaLux cache.",
            "- If sign-flip RMSE is consistently lower, the injection sign should be revisited before OpenVINS trials.",
            "- If OpenVINS seed bias differs strongly from GT-derived bias, the model seed is outside the training supervision convention.",
            "",
            "## Pure Preintegration and Covariance",
            "",
            f"- Mean ideal-GT-bias position error over weak intervals: `{ideal_pos:.6g} m`.",
            f"- Mean ideal-GT-bias rotation error over weak intervals: `{ideal_rot:.6g} deg`.",
            f"- Mean OpenVINS-seed-constant position error: `{seed_pos:.6g} m`.",
            f"- Mean OpenVINS-seed-constant rotation error: `{seed_rot:.6g} deg`.",
            f"- Covariance PSD failures: `{psd_bad}` out of `{len(cov_rows)}` propagated covariance rows.",
            "",
            "Interpretation:",
            "",
            "- Low ideal-GT-bias error supports the current coordinate-frame and gravity convention.",
            "- Large OpenVINS-seed-constant error indicates that bias seed quality and model prediction dominate pure inertial odometry.",
            "- The covariance propagation is structurally PSD in this audit when `psd_bad = 0`; it still does not include learned-model uncertainty.",
            "",
            "## Artifacts",
            "",
            f"- Gate table: `{args.output_root / 'gate_table.csv'}`",
            f"- Fixed gate YAML: `{args.output_root / 'gate_table.yaml'}`",
            f"- Bias diagnostics: `{args.output_root / 'gyro_diagnostics.csv'}`",
            f"- Covariance metrics: `{args.output_root / 'preintegration_covariance_metrics.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    bw_model, ba_model, gyro_ckpt, acc_ckpt = load_models(args, device)
    raw_metrics = read_csv_by_sequence(args.raw_metrics)
    acc_metrics = read_csv_by_sequence(args.acc_gate_metrics) if args.acc_gate_metrics.exists() else {}
    noise = read_noise(args.manifest)
    gate_rows: list[dict[str, object]] = []
    bias_rows: list[dict[str, object]] = []
    cov_rows: list[dict[str, object]] = []

    for index, sequence in enumerate(args.sequences, start=1):
        print(f"[{index}/{len(args.sequences)}] Step14 diagnostics/{sequence}", flush=True)
        gate_row, seq_bias_rows, seq_cov_rows = analyze_sequence(
            sequence,
            args,
            raw_metrics,
            acc_metrics,
            bw_model,
            ba_model,
            device,
            noise,
        )
        gate_rows.append(gate_row)
        bias_rows.extend(seq_bias_rows)
        cov_rows.extend(seq_cov_rows)
        write_csv(args.output_root / "gate_table.csv", gate_rows)
        write_csv(args.output_root / "gyro_diagnostics.csv", bias_rows)
        write_csv(args.output_root / "preintegration_covariance_metrics.csv", cov_rows)

    write_gate_yaml(args.output_root / "gate_table.yaml", gate_rows)
    write_report(args.output_root / "preintegration_covariance_report.md", args, gate_rows, bias_rows, cov_rows, noise)
    # Keep a short top-level summary symlink-equivalent file for users browsing the directory.
    (args.output_root / "summary.md").write_text(
        (args.output_root / "preintegration_covariance_report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"wrote {args.output_root / 'summary.md'}")
    print(f"checkpoints: gyro={gyro_ckpt} acc={acc_ckpt}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    raise SystemExit(main())
