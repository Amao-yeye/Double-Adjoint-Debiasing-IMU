#!/usr/bin/env python3
"""Run the corldata pseudo-GT VO evaluation workflow with evo alignment.

Workflow:
1. Generate ORB-SLAM3 stereo VO if raw outputs are missing.
2. Remove each sequence's old evo/ directory.
3. Estimate time offset from VO/GT positions.
4. Fit a per-sequence diagnostic camera-to-Body_0 right SE(3).
5. Convert GT Body_0 poses into the fitted camera frame.
6. Run evo_traj/evo_ape with --align and save evo's aligned trajectory.
7. Write aligned CSV/metrics for draggable HTML generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

import numpy as np

import run_liasuw_orbslam3_trial as trial


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("/home/amao/datasets/corldata/0511_data")
DEFAULT_CONFIG = DEFAULT_PLATO_ROOT / "plato" / "configs" / "orbslam3" / "corldata_davis_liasuw_0205_baseline.yaml"
DEFAULT_EVO_APE = Path("/home/amao/anaconda3/bin/evo_ape")
DEFAULT_EVO_TRAJ = Path("/home/amao/anaconda3/bin/evo_traj")
DEFAULT_BAG_PYTHON = Path("/usr/bin/python3")
DEFAULT_SEQUENCES = ["S00", "S01", "S02", "S03", "S04", "S05"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--orb-root", type=Path, default=trial.DEFAULT_ORB_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--evo-ape", type=Path, default=DEFAULT_EVO_APE)
    parser.add_argument("--evo-traj", type=Path, default=DEFAULT_EVO_TRAJ)
    parser.add_argument("--bag-python", type=Path, default=DEFAULT_BAG_PYTHON)
    parser.add_argument(
        "--max-stereo-dt",
        type=float,
        default=0.02,
        help="Maximum left/right DAVIS timestamp delta for bag export, in seconds.",
    )
    parser.add_argument("--max-time-diff", type=float, default=0.03)
    parser.add_argument("--offset-radius", type=float, default=0.35)
    parser.add_argument("--coarse-step", type=float, default=0.01)
    parser.add_argument("--fine-radius", type=float, default=0.03)
    parser.add_argument("--fine-step", type=float, default=0.001)
    parser.add_argument("--min-associations", type=int, default=100)
    parser.add_argument("--output-name", default="evo_gt_aligned")
    parser.add_argument("--summary-prefix", help="Aggregate output prefix. Defaults to '<first>_<last>_<output-name>'.")
    parser.add_argument("--force-raw", action="store_true", help="rerun bag export and ORB-SLAM3 even if raw outputs exist")
    parser.add_argument("--skip-orb", action="store_true", help="only evaluate sequences that already have raw ORB-SLAM3 outputs")
    return parser.parse_args()


def seq_root(plato_root: Path, sequence: str) -> Path:
    return plato_root / "results" / "pseudo_gt" / "corldata" / sequence


def raw_paths(seq_dir: Path) -> dict[str, Path]:
    return {
        "sequence": seq_dir / "sequence",
        "raw": seq_dir / "raw",
        "left": seq_dir / "sequence" / "left",
        "right": seq_dir / "sequence" / "right",
        "timestamps": seq_dir / "sequence" / "timestamps.txt",
        "estimate": seq_dir / "raw" / "estimate_f.txt",
        "keyframes": seq_dir / "raw" / "estimate_kf.txt",
        "estimate_seconds": seq_dir / "raw" / "estimate_f_seconds.tum",
        "export_log": seq_dir / "raw" / "export.log",
        "orb_log": seq_dir / "raw" / "orbslam3.log",
    }


def raw_outputs_ready(seq_dir: Path) -> bool:
    paths = raw_paths(seq_dir)
    required = [
        paths["estimate"],
        paths["keyframes"],
        paths["estimate_seconds"],
        paths["timestamps"],
        paths["left"],
        paths["right"],
    ]
    return all(path.exists() for path in required)


def ensure_raw_outputs(args: argparse.Namespace, sequence: str, seq_dir: Path) -> dict[str, object]:
    paths = raw_paths(seq_dir)
    if raw_outputs_ready(seq_dir) and not args.force_raw:
        est_count = len(trial.load_tum(paths["estimate_seconds"], timestamp_scale="sec"))
        timestamp_count = trial.count_timestamps(paths["timestamps"])
        return {
            "ran_orbslam3": False,
            "raw_outputs_reused": True,
            "input_stereo_pairs": timestamp_count,
            "estimated_pose_count_raw": est_count,
            "tracking_success_rate_raw": est_count / timestamp_count if timestamp_count else math.nan,
            "export_log": str(paths["export_log"]),
            "orbslam_log": str(paths["orb_log"]),
        }
    if args.skip_orb:
        raise FileNotFoundError(f"{sequence}: raw ORB-SLAM3 outputs are missing and --skip-orb was set")

    shutil.rmtree(paths["sequence"], ignore_errors=True)
    shutil.rmtree(paths["raw"], ignore_errors=True)
    paths["sequence"].mkdir(parents=True, exist_ok=True)
    paths["raw"].mkdir(parents=True, exist_ok=True)

    bag = args.data_root / f"{sequence}.bag"
    if not bag.exists():
        raise FileNotFoundError(bag)
    run_name = f"plato_corldata_{sequence}_stereo_vo"
    trial.run_command(
        [
            args.bag_python,
            args.orb_root / "tools" / "export_liasuw_bag.py",
            bag,
            paths["sequence"],
            "--max-stereo-dt",
            args.max_stereo_dt,
        ],
        log_path=paths["export_log"],
    )

    executable = args.orb_root / "Examples" / "Stereo" / "stereo_tum_vi_batch"
    vocabulary = args.orb_root / "Vocabulary" / "ORBvoc.txt"
    generated_estimate = args.orb_root / f"f_{run_name}.txt"
    generated_keyframes = args.orb_root / f"kf_{run_name}.txt"
    for generated in [generated_estimate, generated_keyframes]:
        if generated.exists():
            generated.unlink()

    trial.run_command(
        [
            executable,
            vocabulary,
            args.config,
            paths["left"],
            paths["right"],
            paths["timestamps"],
            run_name,
        ],
        cwd=args.orb_root,
        log_path=paths["orb_log"],
        env=trial.orbslam_runtime_env(args.orb_root),
    )

    if not generated_estimate.exists() or not generated_keyframes.exists():
        raise FileNotFoundError(f"{sequence}: ORB-SLAM3 did not produce f_/kf_ trajectory files")
    shutil.move(str(generated_estimate), paths["estimate"])
    shutil.move(str(generated_keyframes), paths["keyframes"])
    trial.write_tum(paths["estimate_seconds"], trial.load_tum(paths["estimate"], timestamp_scale="auto"))

    est_count = len(trial.load_tum(paths["estimate_seconds"], timestamp_scale="sec"))
    timestamp_count = trial.count_timestamps(paths["timestamps"])
    return {
        "ran_orbslam3": True,
        "raw_outputs_reused": False,
        "input_stereo_pairs": timestamp_count,
        "estimated_pose_count_raw": est_count,
        "tracking_success_rate_raw": est_count / timestamp_count if timestamp_count else math.nan,
        "export_log": str(paths["export_log"]),
        "orbslam_log": str(paths["orb_log"]),
    }


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
        for row in rows:
            writer.writerow(row)


def rmse(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values * values))))


def rotation_angle_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))))


def frange(start: float, stop: float, step: float) -> list[float]:
    count = int(math.floor((stop - start) / step + 0.5)) + 1
    return [round(start + idx * step, 9) for idx in range(count)]


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


def existing_offset(seq_dir: Path) -> float | None:
    candidates = [
        seq_dir / "alignment_config" / "vo_gt_alignment.json",
        seq_dir / "evo_gt_aligned" / "metrics.json",
        seq_dir / "metrics.json",
    ]
    for metrics_path in candidates:
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "time_offset_s" in metrics:
                return float(metrics["time_offset_s"])
            search = metrics.get("time_offset_search")
            if isinstance(search, dict) and "time_offset_s" in search:
                return float(search["time_offset_s"])
        except Exception:
            continue
    return None


def estimate_time_offset(args: argparse.Namespace, sequence: str, gt_rows: list[dict], est_rows: list[dict], out_dir: Path) -> dict[str, object]:
    center = existing_offset(seq_root(args.plato_root, sequence))
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
        raise RuntimeError(f"{sequence}: no valid time offset found")
    coarse_best = float(best["time_offset_s"])
    scan("fine", frange(coarse_best - args.fine_radius, coarse_best + args.fine_radius, args.fine_step))
    assert best is not None
    best["initial_center_time_offset_s"] = center
    best["coarse_best_time_offset_s"] = coarse_best
    best["offset_radius_s"] = args.offset_radius
    best["coarse_step_s"] = args.coarse_step
    best["fine_radius_s"] = args.fine_radius
    best["fine_step_s"] = args.fine_step

    write_csv(out_dir / "time_offset_scan.csv", rows)
    return best


def fit_frame_config(
    sequence: str,
    gt_rows: list[dict],
    est_rows: list[dict],
    offset: float,
    max_diff: float,
) -> tuple[dict[str, object], list[tuple[int, int, float, float]]]:
    pairs = trial.associate_by_time(gt_rows, est_rows, offset, max_diff)
    gt_positions = np.array([gt_rows[gt_idx]["p"] for gt_idx, _, _, _ in pairs], dtype=float)
    est_positions = np.array([est_rows[est_idx]["p"] for _, est_idx, _, _ in pairs], dtype=float)
    gt_rotations = np.array([trial.quat_xyzw_to_rot(gt_rows[gt_idx]["q"]) for gt_idx, _, _, _ in pairs], dtype=float)
    est_rotations = np.array([trial.quat_xyzw_to_rot(est_rows[est_idx]["q"]) for _, est_idx, _, _ in pairs], dtype=float)
    fit = trial.fit_world_sensor_se3(gt_positions, est_positions, gt_rotations, est_rotations)

    t_c_b = se3(fit["sensor_rot"], fit["sensor_trans"])
    t_b_c = inv_se3(t_c_b)
    t_m_v = se3(fit["world_rot"], fit["world_trans"])

    config = {
        "dataset": "corldata",
        "sequence": sequence,
        "purpose": "per-sequence diagnostic frame conversion for evo evaluation",
        "warning": "T_camera_to_body_fit is fitted from this same sequence; it is not a calibrated global rig extrinsic.",
        "model": "T_M_B_gt ~= T_M_V_fit * T_V_C_vo * T_C_B_fit; groundtruth_camera_frame = T_M_B_gt * inv(T_C_B_fit)",
        "time_offset_s": offset,
        "max_time_difference_s": max_diff,
        "associated_pose_count": len(pairs),
        "T_C_B_fit_camera_to_body": t_c_b.tolist(),
        "T_B_C_fit_body_to_camera": t_b_c.tolist(),
        "T_M_V_fit_world_alignment_diagnostic": t_m_v.tolist(),
        "camera_to_body_translation_norm_m": float(np.linalg.norm(fit["sensor_trans"])),
        "camera_to_body_rotation_angle_deg": rotation_angle_deg(fit["sensor_rot"]),
        "translation_lstsq_rank": fit["translation_lstsq_rank"],
        "translation_lstsq_condition": fit["translation_lstsq_condition"],
    }
    return config, pairs


def transform_gt_to_camera(gt_rows: list[dict], t_b_c: np.ndarray) -> list[dict[str, object]]:
    converted = []
    for row in gt_rows:
        t_m_b = se3(trial.quat_xyzw_to_rot(row["q"]), row["p"])
        t_m_c = t_m_b @ t_b_c
        converted.append(
            {
                "t": float(row["t"]),
                "p": t_m_c[:3, 3],
                "q": trial.rot_to_quat_xyzw(t_m_c[:3, :3]),
            }
        )
    return converted


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


def read_zip_stats(path: Path) -> dict[str, float]:
    with ZipFile(path, "r") as archive:
        with archive.open("stats.json") as handle:
            return json.loads(handle.read().decode("utf-8"))


def run_evo(args: argparse.Namespace, seq_dir: Path, gt_camera_path: Path, est_path: Path, offset: float) -> dict[str, object]:
    evo_dir = seq_dir / "evo"
    if evo_dir.exists():
        shutil.rmtree(evo_dir)
    evo_dir.mkdir(parents=True, exist_ok=True)
    config_path = evo_dir / "evo_headless_config.json"
    config_path.write_text('{"plot_backend": "Agg"}\n', encoding="utf-8")

    for stale in ["estimate_f_seconds.tum", "groundtruth_camera_frame.tum", "estimate_evo_aligned.tum", "groundtruth_camera_frame_evo.tum"]:
        path = evo_dir / stale
        if path.exists():
            path.unlink()

    traj_log = evo_dir / "evo_traj_gt_camera_aligned.txt"
    run_command(
        [
            args.evo_traj,
            "tum",
            est_path.resolve(),
            "--ref",
            gt_camera_path.resolve(),
            "--sync",
            "--t_offset",
            offset,
            "--t_max_diff",
            args.max_time_diff,
            "--align",
            "--plot_mode",
            "xyz",
            "--save_plot",
            evo_dir / "evo_traj_gt_camera_aligned.png",
            "--save_as_tum",
            "--no_warnings",
            "-c",
            config_path,
        ],
        cwd=evo_dir,
        log_path=traj_log,
    )

    evo_est = evo_dir / "estimate_f_seconds.tum"
    evo_ref = evo_dir / "groundtruth_camera_frame.tum"
    aligned_est = evo_dir / "estimate_evo_aligned.tum"
    aligned_ref = evo_dir / "groundtruth_camera_frame_evo.tum"
    if not evo_est.exists() or not evo_ref.exists():
        raise FileNotFoundError(f"evo_traj did not save expected TUM files in {evo_dir}")
    evo_est.replace(aligned_est)
    evo_ref.replace(aligned_ref)

    trans_log = evo_dir / "evo_ape_trans_gt_camera_aligned.txt"
    trans_zip = evo_dir / "evo_ape_trans_gt_camera_aligned.zip"
    run_command(
        [
            args.evo_ape,
            "tum",
            gt_camera_path.resolve(),
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
            evo_dir / "evo_ape_trans_gt_camera_aligned.png",
            "--no_warnings",
            "--plot_mode",
            "xyz",
            "-c",
            config_path,
        ],
        cwd=evo_dir,
        log_path=trans_log,
    )

    rot_log = evo_dir / "evo_ape_rot_gt_camera_aligned.txt"
    rot_zip = evo_dir / "evo_ape_rot_gt_camera_aligned.zip"
    run_command(
        [
            args.evo_ape,
            "tum",
            gt_camera_path.resolve(),
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
            evo_dir / "evo_ape_rot_gt_camera_aligned.png",
            "--no_warnings",
            "--plot_mode",
            "xyz",
            "-c",
            config_path,
        ],
        cwd=evo_dir,
        log_path=rot_log,
    )

    return {
        "evo_dir": str(evo_dir),
        "aligned_estimate_tum": str(aligned_est),
        "aligned_groundtruth_tum": str(aligned_ref),
        "traj_log": str(traj_log),
        "ape_trans_log": str(trans_log),
        "ape_rot_log": str(rot_log),
        "ape_trans_stats": read_zip_stats(trans_zip),
        "ape_rot_stats": read_zip_stats(rot_zip),
    }


def aligned_csv_from_evo(out_dir: Path, aligned_est_path: Path, aligned_ref_path: Path) -> dict[str, float]:
    est_rows = trial.load_tum(aligned_est_path, timestamp_scale="sec")
    ref_rows = trial.load_tum(aligned_ref_path, timestamp_scale="sec")
    pairs = trial.associate_by_time(ref_rows, est_rows, 0.0, 0.05)

    rows = []
    position_errors = []
    orientation_errors = []
    rpy_errors = []
    for ref_idx, est_idx, time_diff, _ in pairs:
        gt = ref_rows[ref_idx]
        est = est_rows[est_idx]
        gt_rot = trial.quat_xyzw_to_rot(gt["q"])
        est_rot = trial.quat_xyzw_to_rot(est["q"])
        gt_rpy = trial.rot_to_rpy_zyx(gt_rot)
        est_rpy = trial.rot_to_rpy_zyx(est_rot)
        rpy_error = np.array([trial.wrap_angle(est_rpy[idx] - gt_rpy[idx]) for idx in range(3)], dtype=float)
        pos_error = float(np.linalg.norm(est["p"] - gt["p"]))
        ori_error = rotation_angle_deg(est_rot.T @ gt_rot)
        position_errors.append(pos_error)
        orientation_errors.append(ori_error)
        rpy_errors.append(np.degrees(rpy_error))
        rows.append(
            {
                "timestamp": float(gt["t"]),
                "est_x": est["p"][0],
                "est_y": est["p"][1],
                "est_z": est["p"][2],
                "est_roll": est_rpy[0],
                "est_pitch": est_rpy[1],
                "est_yaw": est_rpy[2],
                "gt_x": gt["p"][0],
                "gt_y": gt["p"][1],
                "gt_z": gt["p"][2],
                "gt_roll": gt_rpy[0],
                "gt_pitch": gt_rpy[1],
                "gt_yaw": gt_rpy[2],
                "position_error_m": pos_error,
                "orientation_error_deg": ori_error,
                "time_difference_s": time_diff,
            }
        )

    write_csv(out_dir / "aligned_poses.csv", rows)
    pos = np.array(position_errors, dtype=float)
    ori = np.array(orientation_errors, dtype=float)
    rpy = np.array(rpy_errors, dtype=float)
    return {
        "associated_pose_count": len(rows),
        "position_rmse_m": rmse(pos),
        "position_mean_m": float(np.mean(pos)),
        "position_median_m": float(np.median(pos)),
        "position_p90_m": float(np.percentile(pos, 90)),
        "position_p99_m": float(np.percentile(pos, 99)),
        "position_max_m": float(np.max(pos)),
        "orientation_rmse_deg": rmse(ori),
        "orientation_mean_deg": float(np.mean(ori)),
        "orientation_max_deg": float(np.max(ori)),
        "rpy_rmse_deg": {
            "roll": rmse(rpy[:, 0]),
            "pitch": rmse(rpy[:, 1]),
            "yaw": rmse(rpy[:, 2]),
        },
    }


def evaluate_sequence(args: argparse.Namespace, sequence: str) -> dict[str, object]:
    seq_dir = seq_root(args.plato_root, sequence)
    raw_status = ensure_raw_outputs(args, sequence, seq_dir)
    raw_est = seq_dir / "raw" / "estimate_f.txt"
    raw_est_seconds = seq_dir / "raw" / "estimate_f_seconds.tum"
    gt_body = args.data_root / "gt" / sequence / "Body_0.tum"
    if not raw_est.exists():
        raise FileNotFoundError(raw_est)
    if not raw_est_seconds.exists():
        trial.write_tum(raw_est_seconds, trial.load_tum(raw_est, timestamp_scale="auto"))

    out_dir = seq_dir / args.output_name
    config_dir = seq_dir / "alignment_config"
    gt_aligned_dir = seq_dir / "gt_aligned"
    for path in [out_dir, config_dir, gt_aligned_dir]:
        path.mkdir(parents=True, exist_ok=True)

    gt_rows = trial.load_tum(gt_body, timestamp_scale="sec")
    est_rows = trial.load_tum(raw_est_seconds, timestamp_scale="sec")

    best_offset = estimate_time_offset(args, sequence, gt_rows, est_rows, config_dir)
    offset = float(best_offset["time_offset_s"])
    frame_config, pairs = fit_frame_config(sequence, gt_rows, est_rows, offset, args.max_time_diff)
    frame_config["time_offset_search"] = best_offset
    config_path = config_dir / "vo_gt_alignment.json"
    config_path.write_text(json.dumps(frame_config, indent=2, sort_keys=True), encoding="utf-8")

    t_b_c = np.array(frame_config["T_B_C_fit_body_to_camera"], dtype=float)
    gt_camera_rows = transform_gt_to_camera(gt_rows, t_b_c)
    gt_camera_path = gt_aligned_dir / "groundtruth_camera_frame.tum"
    trial.write_tum(gt_camera_path, gt_camera_rows)

    evo_status = run_evo(args, seq_dir, gt_camera_path, raw_est_seconds, offset)
    aligned_est_path = Path(evo_status["aligned_estimate_tum"])
    aligned_ref_path = Path(evo_status["aligned_groundtruth_tum"])
    shutil.copy2(aligned_est_path, out_dir / "estimate_evo_aligned.tum")
    shutil.copy2(aligned_ref_path, out_dir / "groundtruth_camera_frame_evo.tum")
    shutil.copy2(gt_camera_path, out_dir / "groundtruth_camera_frame_full.tum")
    csv_metrics = aligned_csv_from_evo(out_dir, out_dir / "estimate_evo_aligned.tum", out_dir / "groundtruth_camera_frame_evo.tum")

    metrics = {
        "dataset": "corldata",
        "sequence": sequence,
        "mode": "ORB-SLAM3 stereo visual-only evaluated against GT converted to fitted camera frame; evo --align world alignment",
        "alignment": "T_gt_camera = T_gt_body * T_B_C_fit; final world alignment and aligned estimate trajectory are produced by evo_traj/evo_ape --align without scale correction",
        "time_offset_s": offset,
        "max_time_difference_s": args.max_time_diff,
        "raw_vo": raw_status,
        "estimated_pose_count": len(est_rows),
        "associated_pose_count": csv_metrics["associated_pose_count"],
        "position_rmse_m": csv_metrics["position_rmse_m"],
        "position_mean_m": csv_metrics["position_mean_m"],
        "position_median_m": csv_metrics["position_median_m"],
        "position_p90_m": csv_metrics["position_p90_m"],
        "position_p99_m": csv_metrics["position_p99_m"],
        "position_max_m": csv_metrics["position_max_m"],
        "orientation_rmse_deg": csv_metrics["orientation_rmse_deg"],
        "orientation_mean_deg": csv_metrics["orientation_mean_deg"],
        "orientation_max_deg": csv_metrics["orientation_max_deg"],
        "rpy_rmse_deg": csv_metrics["rpy_rmse_deg"],
        "alignment_config": str(config_path),
        "gt_camera_frame_tum": str(gt_camera_path),
        "evo": evo_status,
        "frame_fit": {
            "camera_to_body_translation_norm_m": frame_config["camera_to_body_translation_norm_m"],
            "camera_to_body_rotation_angle_deg": frame_config["camera_to_body_rotation_angle_deg"],
            "translation_lstsq_condition": frame_config["translation_lstsq_condition"],
            "associated_pose_count_for_fit": len(pairs),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def write_aggregate(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, str]:
    root = args.plato_root / "results" / "pseudo_gt" / "corldata"
    if args.summary_prefix:
        prefix = args.summary_prefix
    else:
        prefix = f"{args.sequences[0]}_{args.sequences[-1]}_{args.output_name}"
    csv_path = root / f"{prefix}_summary.csv"
    json_path = root / f"{prefix}_summary.json"
    report_path = root / f"{prefix}_report.md"

    flat_rows = []
    for row in rows:
        frame = row["frame_fit"]
        evo_trans = row["evo"]["ape_trans_stats"]
        evo_rot = row["evo"]["ape_rot_stats"]
        flat_rows.append(
            {
                "sequence": row["sequence"],
                "time_offset_s": row["time_offset_s"],
                "ran_orbslam3": row["raw_vo"]["ran_orbslam3"],
                "input_stereo_pairs": row["raw_vo"]["input_stereo_pairs"],
                "raw_tracking_success_rate": row["raw_vo"]["tracking_success_rate_raw"],
                "associated_pose_count": row["associated_pose_count"],
                "estimated_pose_count": row["estimated_pose_count"],
                "position_rmse_m": row["position_rmse_m"],
                "position_p90_m": row["position_p90_m"],
                "position_p99_m": row["position_p99_m"],
                "position_max_m": row["position_max_m"],
                "orientation_rmse_deg": row["orientation_rmse_deg"],
                "orientation_max_deg": row["orientation_max_deg"],
                "evo_trans_rmse_m": evo_trans["rmse"],
                "evo_rot_rmse_deg": evo_rot["rmse"],
                "camera_to_body_translation_norm_m": frame["camera_to_body_translation_norm_m"],
                "camera_to_body_rotation_angle_deg": frame["camera_to_body_rotation_angle_deg"],
                "frame_fit_condition": frame["translation_lstsq_condition"],
                "orbslam_log": row["raw_vo"]["orbslam_log"],
            }
        )
    write_csv(csv_path, flat_rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# corldata {args.sequences[0]}-{args.sequences[-1]} evo GT-Aligned Pseudo-GT Evaluation",
        "",
        "Workflow: estimate time offset and per-sequence diagnostic camera/body frame transform, convert mocap Body_0 GT to the fitted camera frame, then call evo with `--align` and no scale correction. The draggable HTML reads evo's saved aligned trajectory, not a PLATO-side world alignment.",
        "",
        "| seq | raw VO | offset s | assoc/est | pos RMSE / p90 / p99 / max m | rot RMSE / max deg | fitted right t norm m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in flat_rows:
        lines.append(
            "| {sequence} | {raw_vo} | {offset:.3f} | {assoc}/{est} | {pos:.5f} / {p90:.5f} / {p99:.5f} / {pmax:.5f} | {ori:.3f} / {orimax:.3f} | {tnorm:.4f} |".format(
                sequence=row["sequence"],
                raw_vo="run" if row["ran_orbslam3"] else "reuse",
                offset=row["time_offset_s"],
                assoc=row["associated_pose_count"],
                est=row["estimated_pose_count"],
                pos=row["position_rmse_m"],
                p90=row["position_p90_m"],
                p99=row["position_p99_m"],
                pmax=row["position_max_m"],
                ori=row["orientation_rmse_deg"],
                orimax=row["orientation_max_deg"],
                tnorm=row["camera_to_body_translation_norm_m"],
            )
        )
    lines.extend(
        [
            "",
            "VO generation:",
            "",
            "- If `raw/estimate_f_seconds.tum` is missing or `--force-raw` is passed, the workflow exports DAVIS stereo images with `external/ORB_SLAM3/tools/export_liasuw_bag.py` and runs ORB-SLAM3 `Examples/Stereo/stereo_tum_vi_batch`.",
            "- ORB-SLAM3 logs are saved under `results/pseudo_gt/corldata/<seq>/raw/orbslam3.log`.",
            "",
            "Per-sequence outputs:",
            "",
            "- `results/pseudo_gt/corldata/<seq>/raw/estimate_f_seconds.tum`",
            "- `results/pseudo_gt/corldata/<seq>/alignment_config/vo_gt_alignment.json`",
            "- `results/pseudo_gt/corldata/<seq>/gt_aligned/groundtruth_camera_frame.tum`",
            "- `results/pseudo_gt/corldata/<seq>/evo/estimate_evo_aligned.tum`",
            "- `results/pseudo_gt/corldata/<seq>/evo/groundtruth_camera_frame_evo.tum`",
            "- `results/pseudo_gt/corldata/<seq>/evo_gt_aligned/aligned_poses.csv`",
            "- `results/pseudo_gt/corldata/<seq>/evo_gt_aligned/metrics.json`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "report": str(report_path)}


def main() -> None:
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.orb_root = args.orb_root.resolve()
    args.data_root = args.data_root.resolve()
    args.config = args.config.resolve()
    args.evo_ape = args.evo_ape.resolve()
    args.evo_traj = args.evo_traj.resolve()
    args.bag_python = args.bag_python.resolve()
    required = [
        args.data_root,
        args.evo_ape,
        args.evo_traj,
        args.config,
        args.bag_python,
        args.orb_root / "Examples" / "Stereo" / "stereo_tum_vi_batch",
        args.orb_root / "Vocabulary" / "ORBvoc.txt",
        args.orb_root / "tools" / "export_liasuw_bag.py",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    rows = []
    for sequence in args.sequences:
        print(f"[{sequence}] ensuring VO, estimating offset/frame, and running evo aligned evaluation", flush=True)
        rows.append(evaluate_sequence(args, sequence))
    outputs = write_aggregate(args, rows)
    outputs["sequence_count"] = len(rows)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
