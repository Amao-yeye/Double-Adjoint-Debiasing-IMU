#!/usr/bin/env python3
"""Evaluate an existing corldata ORB-SLAM3 stereo-inertial trajectory with evo.

The ORB-SLAM3 IMU_STEREO EuRoC trajectory writer exports the IMU/body pose.
This evaluator therefore fits a per-sequence diagnostic transform from the
mocap Body_0 frame to the estimated ORB body frame, converts GT into that fitted
sensor frame, then lets evo perform the final SE(3) no-scale world alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_liasuw_orbslam3_trial as trial  # noqa: E402


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("/home/amao/datasets/corldata/0511_data")
DEFAULT_EVO_APE = Path("/home/amao/anaconda3/bin/evo_ape")
DEFAULT_EVO_TRAJ = Path("/home/amao/anaconda3/bin/evo_traj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequence", default="S02")
    parser.add_argument(
        "--estimate",
        type=Path,
        default=DEFAULT_PLATO_ROOT
        / "results"
        / "pseudo_gt"
        / "corldata"
        / "S02"
        / "stereo_inertial_vis"
        / "f_corldata_s02_stereo_inertial_vis.txt",
    )
    parser.add_argument("--output-name", default="stereo_inertial_eval")
    parser.add_argument("--evo-ape", type=Path, default=DEFAULT_EVO_APE)
    parser.add_argument("--evo-traj", type=Path, default=DEFAULT_EVO_TRAJ)
    parser.add_argument("--max-time-diff", type=float, default=0.03)
    parser.add_argument("--offset-center", type=float, default=None)
    parser.add_argument("--offset-radius", type=float, default=0.2)
    parser.add_argument("--coarse-step", type=float, default=0.01)
    parser.add_argument("--fine-radius", type=float, default=0.03)
    parser.add_argument("--fine-step", type=float, default=0.001)
    parser.add_argument("--min-associations", type=int, default=100)
    return parser.parse_args()


def se3(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


def inv_se3(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = pose[:3, :3].T
    inv[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inv


def rmse(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values * values))))


def rotation_angle_deg(rot: np.ndarray) -> float:
    value = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(value))


def frange(start: float, stop: float, step: float) -> list[float]:
    count = int(math.floor((stop - start) / step + 0.5)) + 1
    return [round(start + idx * step, 9) for idx in range(count)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_zip_stats(path: Path) -> dict[str, float]:
    with ZipFile(path, "r") as archive:
        with archive.open("stats.json") as handle:
            return json.loads(handle.read().decode("utf-8"))


def run_command(command: list[object], cwd: Path, log_path: Path) -> None:
    result = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(map(str, command))}\n{log_path}")


def existing_offset(seq_dir: Path) -> float | None:
    for path in [
        seq_dir / "alignment_config" / "vo_gt_alignment.json",
        seq_dir / "evo_gt_aligned" / "metrics.json",
    ]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "time_offset_s" in data:
            return float(data["time_offset_s"])
        search = data.get("time_offset_search")
        if isinstance(search, dict) and "time_offset_s" in search:
            return float(search["time_offset_s"])
    return None


def position_rmse_for_offset(gt_rows: list[dict], est_rows: list[dict], offset: float, max_diff: float):
    try:
        pairs = trial.associate_by_time(gt_rows, est_rows, offset, max_diff)
    except ValueError:
        return [], math.nan
    gt_positions = np.array([gt_rows[gt_idx]["p"] for gt_idx, _, _, _ in pairs], dtype=float)
    est_positions = np.array([est_rows[est_idx]["p"] for _, est_idx, _, _ in pairs], dtype=float)
    align_rot, align_trans = trial.umeyama_se3(est_positions, gt_positions)
    aligned = (align_rot @ est_positions.T).T + align_trans
    errors = np.linalg.norm(aligned - gt_positions, axis=1)
    return pairs, rmse(errors)


def estimate_time_offset(args: argparse.Namespace, seq_dir: Path, gt_rows: list[dict], est_rows: list[dict], out_dir: Path):
    center = args.offset_center
    if center is None:
        center = existing_offset(seq_dir)
    if center is None:
        center = 1.0

    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    def scan(stage: str, offsets: list[float]) -> None:
        nonlocal best
        for offset in offsets:
            pairs, score = position_rmse_for_offset(gt_rows, est_rows, offset, args.max_time_diff)
            row = {
                "stage": stage,
                "time_offset_s": offset,
                "associated_pose_count": len(pairs),
                "position_rmse_m": score,
            }
            rows.append(row)
            if len(pairs) < args.min_associations or math.isnan(score):
                continue
            if best is None or score < float(best["position_rmse_m"]):
                best = dict(row)

    scan("coarse", frange(center - args.offset_radius, center + args.offset_radius, args.coarse_step))
    if best is None:
        raise RuntimeError("no valid time offset found")
    coarse_best = float(best["time_offset_s"])
    scan("fine", frange(coarse_best - args.fine_radius, coarse_best + args.fine_radius, args.fine_step))
    assert best is not None
    best["initial_center_time_offset_s"] = center
    best["coarse_best_time_offset_s"] = coarse_best
    best["max_time_difference_s"] = args.max_time_diff
    write_csv(out_dir / "time_offset_scan.csv", rows)
    return best


def fit_sensor_to_body(gt_rows: list[dict], est_rows: list[dict], offset: float, max_diff: float):
    pairs = trial.associate_by_time(gt_rows, est_rows, offset, max_diff)
    gt_positions = np.array([gt_rows[gt_idx]["p"] for gt_idx, _, _, _ in pairs], dtype=float)
    est_positions = np.array([est_rows[est_idx]["p"] for _, est_idx, _, _ in pairs], dtype=float)
    gt_rotations = np.array([trial.quat_xyzw_to_rot(gt_rows[gt_idx]["q"]) for gt_idx, _, _, _ in pairs], dtype=float)
    est_rotations = np.array([trial.quat_xyzw_to_rot(est_rows[est_idx]["q"]) for _, est_idx, _, _ in pairs], dtype=float)
    fit = trial.fit_world_sensor_se3(gt_positions, est_positions, gt_rotations, est_rotations)
    t_e_b = se3(fit["sensor_rot"], fit["sensor_trans"])
    t_b_e = inv_se3(t_e_b)
    t_m_v = se3(fit["world_rot"], fit["world_trans"])
    return fit, t_e_b, t_b_e, t_m_v, pairs


def transform_gt_to_sensor(gt_rows: list[dict], t_b_e: np.ndarray) -> list[dict[str, object]]:
    converted = []
    for row in gt_rows:
        t_m_b = se3(trial.quat_xyzw_to_rot(row["q"]), row["p"])
        t_m_e = t_m_b @ t_b_e
        converted.append({"t": float(row["t"]), "p": t_m_e[:3, 3], "q": trial.rot_to_quat_xyzw(t_m_e[:3, :3])})
    return converted


def aligned_metrics(out_dir: Path, aligned_est_path: Path, aligned_ref_path: Path) -> dict[str, object]:
    est_rows = trial.load_tum(aligned_est_path, timestamp_scale="sec")
    ref_rows = trial.load_tum(aligned_ref_path, timestamp_scale="sec")
    pairs = trial.associate_by_time(ref_rows, est_rows, 0.0, 0.05)
    rows = []
    pos_errors = []
    rot_errors = []
    for ref_idx, est_idx, time_diff, _ in pairs:
        ref = ref_rows[ref_idx]
        est = est_rows[est_idx]
        ref_rot = trial.quat_xyzw_to_rot(ref["q"])
        est_rot = trial.quat_xyzw_to_rot(est["q"])
        pos_error = float(np.linalg.norm(est["p"] - ref["p"]))
        rot_error = rotation_angle_deg(est_rot.T @ ref_rot)
        pos_errors.append(pos_error)
        rot_errors.append(rot_error)
        rows.append(
            {
                "timestamp": ref["t"],
                "est_x": est["p"][0],
                "est_y": est["p"][1],
                "est_z": est["p"][2],
                "gt_x": ref["p"][0],
                "gt_y": ref["p"][1],
                "gt_z": ref["p"][2],
                "position_error_m": pos_error,
                "orientation_error_deg": rot_error,
                "time_difference_s": time_diff,
            }
        )
    write_csv(out_dir / "aligned_poses.csv", rows)
    pos = np.array(pos_errors, dtype=float)
    rot = np.array(rot_errors, dtype=float)
    return {
        "associated_pose_count": len(rows),
        "position_rmse_m": rmse(pos),
        "position_mean_m": float(np.mean(pos)),
        "position_median_m": float(np.median(pos)),
        "position_p90_m": float(np.percentile(pos, 90)),
        "position_p99_m": float(np.percentile(pos, 99)),
        "position_max_m": float(np.max(pos)),
        "orientation_rmse_deg": rmse(rot),
        "orientation_mean_deg": float(np.mean(rot)),
        "orientation_p90_deg": float(np.percentile(rot, 90)),
        "orientation_p99_deg": float(np.percentile(rot, 99)),
        "orientation_max_deg": float(np.max(rot)),
    }


def run_evo(args: argparse.Namespace, out_dir: Path, ref_path: Path, est_path: Path, offset: float) -> dict[str, object]:
    config_path = out_dir / "evo_headless_config.json"
    config_path.write_text('{"plot_backend": "Agg"}\n', encoding="utf-8")

    traj_log = out_dir / "evo_traj_body_aligned.txt"
    run_command(
        [
            args.evo_traj,
            "tum",
            est_path.resolve(),
            "--ref",
            ref_path.resolve(),
            "--sync",
            "--t_offset",
            offset,
            "--t_max_diff",
            args.max_time_diff,
            "--align",
            "--plot_mode",
            "xyz",
            "--save_plot",
            out_dir / "evo_traj_body_aligned.png",
            "--save_as_tum",
            "--no_warnings",
            "-c",
            config_path,
        ],
        cwd=out_dir,
        log_path=traj_log,
    )

    saved_est = out_dir / est_path.name
    saved_ref = out_dir / ref_path.name
    aligned_est = out_dir / "estimate_evo_aligned.tum"
    aligned_ref = out_dir / "groundtruth_body_frame_evo.tum"
    if not saved_est.exists() or not saved_ref.exists():
        raise FileNotFoundError("evo_traj did not save aligned TUM files")
    saved_est.replace(aligned_est)
    saved_ref.replace(aligned_ref)

    trans_zip = out_dir / "evo_ape_trans_body_aligned.zip"
    run_command(
        [
            args.evo_ape,
            "tum",
            ref_path.resolve(),
            est_path.resolve(),
            "-r",
            "trans_part",
            "--align",
            "--t_offset",
            offset,
            "--t_max_diff",
            args.max_time_diff,
            "--save_results",
            trans_zip,
            "--save_plot",
            out_dir / "evo_ape_trans_body_aligned.png",
            "--plot_mode",
            "xyz",
            "--no_warnings",
            "-c",
            config_path,
        ],
        cwd=out_dir,
        log_path=out_dir / "evo_ape_trans_body_aligned.txt",
    )

    rot_zip = out_dir / "evo_ape_rot_body_aligned.zip"
    run_command(
        [
            args.evo_ape,
            "tum",
            ref_path.resolve(),
            est_path.resolve(),
            "-r",
            "angle_deg",
            "--align",
            "--t_offset",
            offset,
            "--t_max_diff",
            args.max_time_diff,
            "--save_results",
            rot_zip,
            "--save_plot",
            out_dir / "evo_ape_rot_body_aligned.png",
            "--plot_mode",
            "xyz",
            "--no_warnings",
            "-c",
            config_path,
        ],
        cwd=out_dir,
        log_path=out_dir / "evo_ape_rot_body_aligned.txt",
    )

    return {
        "aligned_estimate_tum": str(aligned_est),
        "aligned_groundtruth_tum": str(aligned_ref),
        "ape_trans_stats": read_zip_stats(trans_zip),
        "ape_rot_stats": read_zip_stats(rot_zip),
    }


def main() -> None:
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.data_root = args.data_root.resolve()
    args.estimate = args.estimate.resolve()
    args.evo_ape = args.evo_ape.resolve()
    args.evo_traj = args.evo_traj.resolve()

    seq_dir = args.plato_root / "results" / "pseudo_gt" / "corldata" / args.sequence
    gt_body = args.data_root / "gt" / args.sequence / "Body_0.tum"
    out_dir = seq_dir / args.output_name
    inputs_dir = out_dir / "inputs"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = trial.load_tum(gt_body, timestamp_scale="sec")
    est_rows = trial.load_tum(args.estimate, timestamp_scale="auto")
    est_seconds = inputs_dir / "estimate_body_seconds.tum"
    trial.write_tum(est_seconds, est_rows)

    best_offset = estimate_time_offset(args, seq_dir, gt_rows, est_rows, out_dir)
    offset = float(best_offset["time_offset_s"])
    fit, t_e_b, t_b_e, t_m_v, pairs = fit_sensor_to_body(gt_rows, est_rows, offset, args.max_time_diff)
    gt_sensor_rows = transform_gt_to_sensor(gt_rows, t_b_e)
    gt_sensor = inputs_dir / "groundtruth_body_frame.tum"
    trial.write_tum(gt_sensor, gt_sensor_rows)

    evo_status = run_evo(args, out_dir, gt_sensor, est_seconds, offset)
    csv_metrics = aligned_metrics(out_dir, Path(evo_status["aligned_estimate_tum"]), Path(evo_status["aligned_groundtruth_tum"]))
    metrics = {
        "dataset": "corldata",
        "sequence": args.sequence,
        "mode": "ORB-SLAM3 stereo-inertial; saved trajectory is IMU/body pose",
        "estimate": str(args.estimate),
        "time_offset_s": offset,
        "max_time_difference_s": args.max_time_diff,
        "estimated_pose_count": len(est_rows),
        "associated_pose_count": csv_metrics["associated_pose_count"],
        "time_offset_search": best_offset,
        "position_rmse_m": csv_metrics["position_rmse_m"],
        "position_mean_m": csv_metrics["position_mean_m"],
        "position_median_m": csv_metrics["position_median_m"],
        "position_p90_m": csv_metrics["position_p90_m"],
        "position_p99_m": csv_metrics["position_p99_m"],
        "position_max_m": csv_metrics["position_max_m"],
        "orientation_rmse_deg": csv_metrics["orientation_rmse_deg"],
        "orientation_mean_deg": csv_metrics["orientation_mean_deg"],
        "orientation_p90_deg": csv_metrics["orientation_p90_deg"],
        "orientation_p99_deg": csv_metrics["orientation_p99_deg"],
        "orientation_max_deg": csv_metrics["orientation_max_deg"],
        "evo": evo_status,
        "frame_fit": {
            "model": "T_M_B_gt ~= T_M_V_fit * T_V_E_est * T_E_B_fit; groundtruth_estimated_body_frame = T_M_B_gt * inv(T_E_B_fit)",
            "warning": "T_E_B_fit is fitted from this same sequence; it is not a calibrated global mocap Body_0 to IMU transform.",
            "T_E_B_fit_estimated_body_to_mocap_body": t_e_b.tolist(),
            "T_B_E_fit_mocap_body_to_estimated_body": t_b_e.tolist(),
            "T_M_V_fit_world_alignment_diagnostic": t_m_v.tolist(),
            "estimated_body_to_mocap_body_translation_norm_m": float(np.linalg.norm(fit["sensor_trans"])),
            "estimated_body_to_mocap_body_rotation_angle_deg": rotation_angle_deg(fit["sensor_rot"]),
            "associated_pose_count_for_fit": len(pairs),
            "translation_lstsq_rank": fit["translation_lstsq_rank"],
            "translation_lstsq_condition": fit["translation_lstsq_condition"],
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    report = [
        f"# corldata {args.sequence} ORB-SLAM3 Stereo-Inertial evo Evaluation",
        "",
        "The estimate is ORB-SLAM3 `IMU_STEREO` EuRoC output, i.e. IMU/body pose. GT is converted into the fitted estimated-body frame, then evo applies final SE(3) no-scale alignment.",
        "",
        f"- time offset: `{offset:.6f} s`",
        f"- associated poses: `{csv_metrics['associated_pose_count']}/{len(est_rows)}`",
        f"- position RMSE / p90 / p99 / max: `{csv_metrics['position_rmse_m']:.6f}` / `{csv_metrics['position_p90_m']:.6f}` / `{csv_metrics['position_p99_m']:.6f}` / `{csv_metrics['position_max_m']:.6f}` m",
        f"- orientation RMSE / p90 / p99 / max: `{csv_metrics['orientation_rmse_deg']:.3f}` / `{csv_metrics['orientation_p90_deg']:.3f}` / `{csv_metrics['orientation_p99_deg']:.3f}` / `{csv_metrics['orientation_max_deg']:.3f}` deg",
        f"- fitted estimated-body to mocap-body translation norm: `{metrics['frame_fit']['estimated_body_to_mocap_body_translation_norm_m']:.4f} m`",
        "",
        "Outputs:",
        "",
        f"- `{metrics_path}`",
        f"- `{out_dir / 'estimate_evo_aligned.tum'}`",
        f"- `{out_dir / 'groundtruth_body_frame_evo.tum'}`",
        f"- `{out_dir / 'evo_ape_trans_body_aligned.png'}`",
        f"- `{out_dir / 'evo_ape_rot_body_aligned.png'}`",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(metrics_path), "report": str(report_path), **metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
