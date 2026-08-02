#!/usr/bin/env python3
"""Evaluate AquaLux OpenVINS baseline compensate1 as an offline stitch prototype.

The prototype keeps the raw OpenVINS trajectory outside weak visual-update
intervals. Inside each interval it seeds the BiasDy baseline with the last
OpenVINS state/bias and performs pure IMU propagation with learned bias
dynamics. This tests the requested handoff idea without patching OpenVINS C++.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
BIASDY_DIR = REPO_ROOT / "external" / "Debias_IMU" / "BiasDy"
if str(BIASDY_DIR) not in sys.path:
    sys.path.insert(0, str(BIASDY_DIR))

import lie_algebra as Lie  # noqa: E402
from network_for_BiadDy import ba_func_choose, bw_func_choose  # noqa: E402

from run_aqualux_openvins_raw import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EVO_HOME,
    DEFAULT_EVO_USER_SITE,
    DEFAULT_MPLCONFIGDIR,
    DEFAULT_RESULTS_ROOT,
    TEST_SEQUENCES,
    count_timestamp_matches,
    parse_evo_stats,
)

DEFAULT_BASELINE_ROOT = REPO_ROOT / "Results" / "AquaLux" / "preintegration" / "rss_baseline_unseen_corrected"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "baseline" / "compensate1"
DEFAULT_RAW_ROOT = DEFAULT_RESULTS_ROOT / "raw"
DEFAULT_GRAVITY = np.array([0.0, 0.0, -9.7879], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-openvins-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--gyro-checkpoint", type=Path, default=None)
    parser.add_argument("--acc-checkpoint", type=Path, default=None)
    parser.add_argument("--sequences", nargs="+", default=list(TEST_SEQUENCES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--msckf-update-threshold", type=float, default=5e-4)
    parser.add_argument("--min-weak-duration", type=float, default=0.5)
    parser.add_argument("--merge-gap", type=float, default=0.2)
    parser.add_argument("--t-max-diff", type=float, default=0.02)
    parser.add_argument("--evo-home", type=Path, default=DEFAULT_EVO_HOME)
    parser.add_argument("--mplconfigdir", type=Path, default=DEFAULT_MPLCONFIGDIR)
    parser.add_argument("--evo-user-site", type=Path, default=DEFAULT_EVO_USER_SITE)
    return parser.parse_args()


def choose_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {name}, but CUDA is not available")
    return name


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])
    if not rows:
        raise ValueError(f"No TUM rows in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def write_tum(path: Path, stamps: np.ndarray, positions: np.ndarray, quats_xyzw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for stamp, p, q in zip(stamps, positions, quats_xyzw):
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            handle.write(
                f"{stamp:.9f} {p[0]:.12g} {p[1]:.12g} {p[2]:.12g} "
                f"{q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g}\n"
            )


def read_openvins_estimate(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 17:
                continue
            rows.append([float(x) for x in parts[:17]])
    if not rows:
        raise ValueError(f"No OpenVINS estimate rows in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return {
        "t": data[:, 0],
        "q": data[:, 1:5],
        "p": data[:, 5:8],
        "v": data[:, 8:11],
        "bg": data[:, 11:14],
        "ba": data[:, 14:17],
    }


def read_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if rows.ndim == 0:
        rows = rows.reshape(1)
    required = ("timestamp", "wx", "wy", "wz", "ax", "ay", "az")
    if rows.dtype.names is None or any(name not in rows.dtype.names for name in required):
        raise ValueError(f"{path}: missing required IMU columns")
    stamps = np.asarray(rows["timestamp"], dtype=np.float64)
    imu = np.column_stack([rows[name] for name in required[1:]]).astype(np.float64)
    order = np.argsort(stamps)
    stamps = stamps[order]
    imu = imu[order]
    keep = np.r_[True, np.diff(stamps) > 0]
    return stamps[keep], imu[keep]


def read_weak_visual_segments(
    timing_path: Path,
    threshold: float,
    min_duration: float,
    merge_gap: float,
) -> list[tuple[float, float, int]]:
    rows: list[tuple[float, float]] = []
    with timing_path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(",")
            if len(parts) < 4:
                continue
            rows.append((float(parts[0]), float(parts[3])))
    if not rows:
        return []
    stamps = np.asarray([r[0] for r in rows], dtype=np.float64)
    msckf = np.asarray([r[1] for r in rows], dtype=np.float64)
    weak = msckf < threshold
    raw_segments: list[tuple[float, float, int]] = []
    i = 0
    while i < len(weak):
        if not weak[i]:
            i += 1
            continue
        j = i
        while j < len(weak) and weak[j]:
            j += 1
        duration = float(stamps[j - 1] - stamps[i]) if j > i else 0.0
        if duration >= min_duration:
            raw_segments.append((float(stamps[i]), float(stamps[j - 1]), int(j - i)))
        i = j
    if not raw_segments:
        return []
    merged: list[tuple[float, float, int]] = [raw_segments[0]]
    for start, end, count in raw_segments[1:]:
        prev_start, prev_end, prev_count = merged[-1]
        if start - prev_end <= merge_gap:
            merged[-1] = (prev_start, end, prev_count + count)
        else:
            merged.append((start, end, count))
    return merged


def interp_imu(stamps: np.ndarray, imu: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, stamps, imu[:, axis]) for axis in range(6)])


def load_models(args: argparse.Namespace, device: str):
    gyro_ckpt = args.gyro_checkpoint or args.baseline_root / "Gyro_weights" / "model_epoch_1800.pt"
    acc_ckpt = args.acc_checkpoint or args.baseline_root / "Acc_weights" / "model_epoch_1800.pt"
    bw = bw_func_choose("bw_func_net", device=device)
    ba = ba_func_choose("ba_func_net", device=device)
    bw.load_state_dict(torch.load(gyro_ckpt, map_location=device)["func_bw_model"])
    ba.load_state_dict(torch.load(acc_ckpt, map_location=device)["func_ba_model"])
    bw.eval()
    ba.eval()
    return bw, ba, gyro_ckpt, acc_ckpt


@torch.no_grad()
def bias_derivative(
    bw_model,
    ba_model,
    y_np: np.ndarray,
    imu_np: np.ndarray,
    imu_dot_np: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = torch.from_numpy(y_np.astype(np.float32)).to(device).unsqueeze(0)
    imu = torch.from_numpy(imu_np.astype(np.float32)).to(device).unsqueeze(0)
    imu_dot = torch.from_numpy(imu_dot_np.astype(np.float32)).to(device).unsqueeze(0)
    r0 = torch.eye(3, dtype=torch.float32, device=device)
    bg_dot = bw_model(y, imu, imu_dot, r0).detach().cpu().numpy()[0].astype(np.float64)
    ba_dot = ba_model(y, imu, imu_dot, r0).detach().cpu().numpy()[0].astype(np.float64)
    return bg_dot, ba_dot


def propagate_segment(
    output_times: np.ndarray,
    seed: dict[str, np.ndarray | float],
    imu_stamps: np.ndarray,
    imu_values: np.ndarray,
    bw_model,
    ba_model,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t0 = float(seed["t"])
    query = np.unique(np.r_[t0, output_times[output_times > t0]])
    if len(query) < 2:
        return output_times, np.empty((0, 3)), np.empty((0, 4)), np.empty((0, 6))

    imu_query = interp_imu(imu_stamps, imu_values, query)
    imu_dot = np.gradient(imu_query, query, axis=0, edge_order=1)

    rot = Rotation.from_quat(np.asarray(seed["q"], dtype=np.float64))
    p = np.asarray(seed["p"], dtype=np.float64).copy()
    v = np.asarray(seed["v"], dtype=np.float64).copy()
    bg = np.asarray(seed["bg"], dtype=np.float64).copy()
    ba = np.asarray(seed["ba"], dtype=np.float64).copy()

    pred_by_time: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        query[0]: (p.copy(), rot.as_quat(), np.r_[bg, ba].copy())
    }
    for i in range(len(query) - 1):
        dt = float(query[i + 1] - query[i])
        if dt <= 0:
            continue
        y = np.zeros(16, dtype=np.float64)
        y[3] = query[i] - t0
        y[4:7] = bg
        y[7:10] = v
        y[10:13] = p
        y[13:16] = ba
        bg_dot, ba_dot = bias_derivative(bw_model, ba_model, y, imu_query[i], imu_dot[i], device)

        corrected_w = imu_query[i, :3] - bg
        corrected_a = imu_query[i, 3:6] - ba
        acc_world = rot.apply(corrected_a) + DEFAULT_GRAVITY
        p = p + v * dt
        v = v + acc_world * dt
        rot = rot * Rotation.from_rotvec(corrected_w * dt)
        bg = bg + bg_dot * dt
        ba = ba + ba_dot * dt
        pred_by_time[query[i + 1]] = (p.copy(), rot.as_quat(), np.r_[bg, ba].copy())

    positions: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    kept_times: list[float] = []
    for stamp in output_times:
        key = query[np.searchsorted(query, stamp)]
        if not math.isclose(float(key), float(stamp), abs_tol=5e-7):
            continue
        p_i, q_i, b_i = pred_by_time[float(key)]
        kept_times.append(float(stamp))
        positions.append(p_i)
        quats.append(q_i)
        biases.append(b_i)
    return (
        np.asarray(kept_times, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(quats, dtype=np.float64),
        np.asarray(biases, dtype=np.float64),
    )


def make_hybrid_sequence(
    sequence: str,
    args: argparse.Namespace,
    bw_model,
    ba_model,
    device: str,
) -> dict[str, object]:
    raw_dir = args.raw_openvins_root / sequence
    out_dir = args.output_root / sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_tum = raw_dir / "estimate_tum.tum"
    if not raw_tum.exists():
        raw_tum = raw_dir / "odomimu.txt"
    raw_t, raw_p, raw_q = read_tum(raw_tum)
    est = read_openvins_estimate(raw_dir / "estimate.txt")
    imu_t, imu_u = read_imu_csv(args.data_root / "0524" / sequence / "imu_diagnostics" / "imu_left.csv")
    gt_tum = args.data_root / "0524" / sequence / "ATO" / "gt_imu_gravity_aligned.tum"

    segments = read_weak_visual_segments(
        raw_dir / "timing.txt",
        args.msckf_update_threshold,
        args.min_weak_duration,
        args.merge_gap,
    )

    hybrid_p = raw_p.copy()
    hybrid_q = raw_q.copy()
    segment_rows: list[dict[str, object]] = []

    for seg_index, (start, end, weak_count) in enumerate(segments):
        replace_idx = np.where((raw_t >= start) & (raw_t <= end))[0]
        seed_idx = int(np.searchsorted(est["t"], start, side="right") - 1)
        if len(replace_idx) < 2 or seed_idx < 0:
            segment_rows.append(
                {
                    "sequence": sequence,
                    "segment_index": seg_index,
                    "start": start,
                    "end": end,
                    "weak_update_rows": weak_count,
                    "replaced_odom_rows": len(replace_idx),
                    "status": "skipped_no_seed_or_rows",
                }
            )
            continue
        seed = {key: est[key][seed_idx] for key in ("t", "q", "p", "v", "bg", "ba")}
        kept_t, pred_p, pred_q, pred_bias = propagate_segment(
            raw_t[replace_idx],
            seed,
            imu_t,
            imu_u,
            bw_model,
            ba_model,
            device,
        )
        if len(kept_t) == 0:
            status = "skipped_no_propagation"
        else:
            by_time = {float(t): i for i, t in enumerate(kept_t)}
            replaced = 0
            for idx in replace_idx:
                hit = by_time.get(float(raw_t[idx]))
                if hit is None:
                    continue
                hybrid_p[idx] = pred_p[hit]
                hybrid_q[idx] = pred_q[hit]
                replaced += 1
            status = "ok" if replaced > 0 else "skipped_no_matching_output_time"
        segment_rows.append(
            {
                "sequence": sequence,
                "segment_index": seg_index,
                "start": start,
                "end": end,
                "duration_s": end - start,
                "weak_update_rows": weak_count,
                "seed_time": float(seed["t"]),
                "seed_bg_x": float(seed["bg"][0]),
                "seed_bg_y": float(seed["bg"][1]),
                "seed_bg_z": float(seed["bg"][2]),
                "seed_ba_x": float(seed["ba"][0]),
                "seed_ba_y": float(seed["ba"][1]),
                "seed_ba_z": float(seed["ba"][2]),
                "replaced_odom_rows": int(len(replace_idx)),
                "propagated_rows": int(len(kept_t)),
                "status": status,
            }
        )

    hybrid_tum = out_dir / "hybrid_estimate.tum"
    write_tum(hybrid_tum, raw_t, hybrid_p, hybrid_q)
    write_csv(out_dir / "segments.csv", segment_rows)
    run_config = {
        "sequence": sequence,
        "raw_tum": str(raw_tum),
        "hybrid_tum": str(hybrid_tum),
        "gt_tum": str(gt_tum),
        "timing": str(raw_dir / "timing.txt"),
        "msckf_update_threshold": args.msckf_update_threshold,
        "min_weak_duration": args.min_weak_duration,
        "merge_gap": args.merge_gap,
        "gravity": DEFAULT_GRAVITY.tolist(),
        "prototype_note": "Offline stitch prototype; not a C++ OpenVINS state/covariance handoff.",
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    pos = run_evo(args, gt_tum, hybrid_tum, "trans_part", out_dir / "evo_ape_position.zip", out_dir / "evo_ape_position.txt")
    rot = run_evo(args, gt_tum, hybrid_tum, "angle_deg", out_dir / "evo_ape_rotation.zip", out_dir / "evo_ape_rotation.txt")
    metric_status = "ok" if pos["status"] == "ok" and rot["status"] == "ok" else "failed"
    return {
        "method": "compensate1",
        "sequence": sequence,
        "metric_status": metric_status,
        "segments": len(segments),
        "segments_ok": sum(1 for row in segment_rows if row.get("status") == "ok"),
        "replaced_odom_rows": sum(int(row.get("replaced_odom_rows", 0)) for row in segment_rows if row.get("status") == "ok"),
        "hybrid_rows": len(raw_t),
        "matched_poses_nearest": count_timestamp_matches(gt_tum, hybrid_tum, args.t_max_diff),
        "ate_rmse_pos_m": pos.get("rmse"),
        "ate_rmse_rot_deg": rot.get("rmse"),
        "evo_position_status": pos["status"],
        "evo_rotation_status": rot["status"],
        "hybrid_tum": str(hybrid_tum),
        "out_dir": str(out_dir),
    }


def evo_env(args: argparse.Namespace) -> dict[str, str]:
    args.evo_home.mkdir(parents=True, exist_ok=True)
    args.mplconfigdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(args.evo_home)
    env["MPLCONFIGDIR"] = str(args.mplconfigdir)
    if args.evo_user_site.exists():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(args.evo_user_site) if not existing else str(args.evo_user_site) + os.pathsep + existing
    return env


def run_evo(
    args: argparse.Namespace,
    gt_tum: Path,
    estimate_tum: Path,
    pose_relation: str,
    save_path: Path,
    log_path: Path,
) -> dict[str, object]:
    if save_path.exists():
        save_path.unlink()
    cmd = [
        "evo_ape",
        "tum",
        str(gt_tum),
        str(estimate_tum),
        "--align",
        "--pose_relation",
        pose_relation,
        "--t_max_diff",
        f"{args.t_max_diff:.9g}",
        "--save_results",
        str(save_path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=evo_env(args))
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stats = parse_evo_stats(proc.stdout)
    return {
        "status": "ok" if proc.returncode == 0 and "rmse" in stats else "failed",
        "returncode": int(proc.returncode),
        "rmse": stats.get("rmse"),
        "log": str(log_path),
        "result_zip": str(save_path) if save_path.exists() else "",
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
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


def write_summary(args: argparse.Namespace, rows: list[dict[str, object]], gyro_ckpt: Path, acc_ckpt: Path) -> None:
    ok_rows = [row for row in rows if row.get("metric_status") == "ok"]
    pos_values = [float(row["ate_rmse_pos_m"]) for row in ok_rows if row.get("ate_rmse_pos_m") is not None]
    rot_values = [float(row["ate_rmse_rot_deg"]) for row in ok_rows if row.get("ate_rmse_rot_deg") is not None]
    pos_mean = float(np.mean(pos_values)) if pos_values else math.nan
    rot_mean = float(np.mean(rot_values)) if rot_values else math.nan
    lines = [
        "# AquaLux OpenVINS Baseline Compensate1 Summary",
        "",
        "Method: offline stitch prototype for the requested online handoff path.",
        "",
        "- Outside weak visual-update intervals, the trajectory is raw OpenVINS `odomimu`.",
        "- Inside weak intervals, propagation is pure IMU with BiasDy bias dynamics seeded from the last OpenVINS `bg/ba`.",
        "- After each interval, the trajectory returns to the original OpenVINS trajectory.",
        "- This is not yet a C++ OpenVINS state/covariance handoff.",
        "",
        "Weak interval detection:",
        "",
        "```text",
        f"msckf_update_time < {args.msckf_update_threshold:.12g}",
        f"min_weak_duration = {args.min_weak_duration:.12g} s",
        f"merge_gap = {args.merge_gap:.12g} s",
        "```",
        "",
        "Checkpoints:",
        "",
        f"- Gyro: `{gyro_ckpt}`",
        f"- Acc: `{acc_ckpt}`",
        "",
        f"Average over ok sequences: position RMSE `{pos_mean:.6g} m`, rotation RMSE `{rot_mean:.6g} deg`.",
        "",
        "| sequence | metric | segments | ok segments | replaced rows | position RMSE (m) | rotation RMSE (deg) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pos = row.get("ate_rmse_pos_m")
        rot = row.get("ate_rmse_rot_deg")
        lines.append(
            f"| {row.get('sequence')} | {row.get('metric_status')} | {row.get('segments')} | "
            f"{row.get('segments_ok')} | {row.get('replaced_odom_rows')} | "
            f"{float(pos):.6g} | {float(rot):.6g} |"
            if pos is not None and rot is not None
            else f"| {row.get('sequence')} | {row.get('metric_status')} | {row.get('segments')} | "
            f"{row.get('segments_ok')} | {row.get('replaced_odom_rows')} | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            f"- Metrics CSV: `{args.output_root / 'metrics.csv'}`",
            f"- Per-sequence folders: `{args.output_root}/Test_*`",
        ]
    )
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    bw_model, ba_model, gyro_ckpt, acc_ckpt = load_models(args, device)
    rows: list[dict[str, object]] = []
    for index, sequence in enumerate(args.sequences, start=1):
        print(f"[{index}/{len(args.sequences)}] compensate1/{sequence}", flush=True)
        row = make_hybrid_sequence(sequence, args, bw_model, ba_model, device)
        rows.append(row)
        write_csv(args.output_root / "metrics.csv", rows)
        (args.output_root / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_summary(args, rows, gyro_ckpt, acc_ckpt)
        print(
            f"  metric={row['metric_status']} pos={row.get('ate_rmse_pos_m')} "
            f"rot={row.get('ate_rmse_rot_deg')} segments={row.get('segments_ok')}/{row.get('segments')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
