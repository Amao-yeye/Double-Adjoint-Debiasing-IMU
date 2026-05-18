#!/usr/bin/env python3
"""Run and evaluate corldata S00-S05 ORB-SLAM3 stereo trials.

This is PLATO glue code. It keeps generated outputs under results/ and reuses
the existing ORB-SLAM3 single-trial evaluator for no-scale world alignment
and right-multiplied fixed-frame SE(3) compensation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

import run_liasuw_orbslam3_trial as trial


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("/home/amao/datasets/corldata/0511_data")
DEFAULT_CONFIG = DEFAULT_PLATO_ROOT / "plato" / "configs" / "orbslam3" / "corldata_davis_liasuw_0205_baseline.yaml"
DEFAULT_EVO_APE = Path("/home/amao/anaconda3/bin/evo_ape")
DEFAULT_EVO_TRAJ = Path("/home/amao/anaconda3/bin/evo_traj")
DEFAULT_BAG_PYTHON = Path("/usr/bin/python3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--orb-root", type=Path, default=trial.DEFAULT_ORB_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sequences", nargs="+", default=["S00", "S01", "S02", "S03", "S04", "S05"])
    parser.add_argument("--max-time-diff", type=float, default=0.03)
    parser.add_argument(
        "--max-stereo-dt",
        type=float,
        default=0.02,
        help="Maximum left/right DAVIS timestamp delta for bag export, in seconds.",
    )
    parser.add_argument("--coarse-min-offset", type=float, default=-3.0)
    parser.add_argument("--coarse-max-offset", type=float, default=3.0)
    parser.add_argument("--coarse-step", type=float, default=0.01)
    parser.add_argument("--fine-radius", type=float, default=0.06)
    parser.add_argument("--fine-step", type=float, default=0.001)
    parser.add_argument("--min-associations", type=int, default=100)
    parser.add_argument("--evo-ape", type=Path, default=DEFAULT_EVO_APE)
    parser.add_argument("--evo-traj", type=Path, default=DEFAULT_EVO_TRAJ)
    parser.add_argument("--bag-python", type=Path, default=DEFAULT_BAG_PYTHON)
    parser.add_argument("--force-raw", action="store_true", help="rerun bag export and ORB-SLAM3 even if raw outputs exist")
    parser.add_argument("--skip-orb", action="store_true", help="only evaluate sequences that already have raw ORB-SLAM3 outputs")
    parser.add_argument("--skip-evo", action="store_true", help="skip evo plots/zips and only write PLATO metrics")
    return parser.parse_args()


def result_paths(root: Path, sequence: str) -> dict[str, Path]:
    result_dir = root / "results" / "pseudo_gt" / "corldata" / sequence
    paths = {
        "result": result_dir,
        "sequence": result_dir / "sequence",
        "raw": result_dir / "raw",
        "aligned": result_dir / "aligned",
        "plots": result_dir / "plots",
        "evo": result_dir / "evo",
        "metrics": result_dir / "metrics.json",
        "report": result_dir / "report.md",
        "tuning_log": result_dir / "tuning_log.md",
    }
    paths["raw_estimate"] = paths["raw"] / "estimate_f.txt"
    paths["raw_keyframes"] = paths["raw"] / "estimate_kf.txt"
    paths["raw_estimate_seconds"] = paths["raw"] / "estimate_f_seconds.tum"
    paths["raw_groundtruth"] = paths["raw"] / "groundtruth.tum"
    paths["aligned_csv"] = paths["aligned"] / "aligned_poses.csv"
    paths["aligned_estimate_tum"] = paths["aligned"] / "estimate_aligned_se3.tum"
    paths["associated_groundtruth_tum"] = paths["aligned"] / "groundtruth_associated.tum"
    paths["frame_alignment"] = paths["aligned"] / "camera_to_gt_frame_rotation.json"
    paths["time_offset_scan"] = paths["aligned"] / "time_offset_scan.csv"
    return paths


def raw_outputs_ready(paths: dict[str, Path]) -> bool:
    required = [
        paths["raw_estimate"],
        paths["raw_keyframes"],
        paths["sequence"] / "timestamps.txt",
        paths["sequence"] / "left",
        paths["sequence"] / "right",
    ]
    return all(path.exists() for path in required)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def ensure_raw_outputs(args: argparse.Namespace, sequence: str, paths: dict[str, Path]) -> None:
    if raw_outputs_ready(paths) and not args.force_raw:
        return
    if args.skip_orb:
        raise FileNotFoundError(f"{sequence}: raw ORB-SLAM3 outputs are missing and --skip-orb was set")

    remove_tree(paths["sequence"])
    remove_tree(paths["raw"])
    paths["sequence"].mkdir(parents=True, exist_ok=True)
    paths["raw"].mkdir(parents=True, exist_ok=True)

    bag = args.data_root / f"{sequence}.bag"
    run_name = f"plato_corldata_{sequence}_stereo_vo"
    export_log = paths["raw"] / "export.log"
    orb_log = paths["raw"] / "orbslam3.log"

    trial.run_command(
        [
            args.bag_python,
            args.orb_root / "tools" / "export_liasuw_bag.py",
            bag,
            paths["sequence"],
            "--max-stereo-dt",
            args.max_stereo_dt,
        ],
        log_path=export_log,
    )

    executable = args.orb_root / "Examples" / "Stereo" / "stereo_tum_vi_batch"
    vocabulary = args.orb_root / "Vocabulary" / "ORBvoc.txt"
    generated_f = args.orb_root / f"f_{run_name}.txt"
    generated_kf = args.orb_root / f"kf_{run_name}.txt"
    for generated in [generated_f, generated_kf]:
        if generated.exists():
            generated.unlink()

    trial.run_command(
        [
            executable,
            vocabulary,
            args.config,
            paths["sequence"] / "left",
            paths["sequence"] / "right",
            paths["sequence"] / "timestamps.txt",
            run_name,
        ],
        cwd=args.orb_root,
        log_path=orb_log,
        env=trial.orbslam_runtime_env(args.orb_root),
    )

    if not generated_f.exists() or not generated_kf.exists():
        raise FileNotFoundError(f"{sequence}: ORB-SLAM3 did not produce f_/kf_ trajectory files")
    shutil.move(str(generated_f), paths["raw_estimate"])
    shutil.move(str(generated_kf), paths["raw_keyframes"])


def frange(start: float, stop: float, step: float) -> np.ndarray:
    count = int(math.floor((stop - start) / step + 0.5)) + 1
    return start + np.arange(count, dtype=float) * step


def position_rmse_for_offset(gt_rows: list[dict], est_rows: list[dict], offset: float, max_diff: float):
    pairs = trial.associate_by_time(gt_rows, est_rows, offset, max_diff)
    if not pairs:
        return pairs, math.nan, math.nan
    gt_positions = np.array([gt_rows[gt_idx]["p"] for gt_idx, _, _, _ in pairs], dtype=float)
    est_positions = np.array([est_rows[est_idx]["p"] for _, est_idx, _, _ in pairs], dtype=float)
    gt_rotations = np.array([trial.quat_xyzw_to_rot(gt_rows[gt_idx]["q"]) for gt_idx, _, _, _ in pairs], dtype=float)
    est_rotations = np.array([trial.quat_xyzw_to_rot(est_rows[est_idx]["q"]) for _, est_idx, _, _ in pairs], dtype=float)
    pose_alignment = trial.fit_world_sensor_se3(gt_positions, est_positions, gt_rotations, est_rotations)
    align_rot = pose_alignment["world_rot"]
    align_trans = pose_alignment["world_trans"]
    sensor_trans = pose_alignment["sensor_trans"]
    lever_arm_positions = est_positions + np.einsum("nij,j->ni", est_rotations, sensor_trans)
    aligned = (align_rot @ lever_arm_positions.T).T + align_trans
    errors = np.linalg.norm(aligned - gt_positions, axis=1)
    return pairs, trial.rmse(errors), trial.diagnostic_sim3_scale(est_positions, gt_positions)


def scan_time_offset(args: argparse.Namespace, sequence: str, paths: dict[str, Path]) -> dict[str, object]:
    gt_path = args.data_root / "gt" / sequence / "Body_0.tum"
    gt_rows = trial.load_tum(gt_path, timestamp_scale="sec")
    est_rows = trial.load_tum(paths["raw_estimate"], timestamp_scale="auto")

    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    def scan(stage: str, offsets: np.ndarray) -> None:
        nonlocal best
        for offset in offsets:
            offset = float(round(offset, 9))
            pairs, position_rmse, sim3_scale = position_rmse_for_offset(
                gt_rows, est_rows, offset, args.max_time_diff
            )
            associated = len(pairs)
            row = {
                "stage": stage,
                "time_offset_s": offset,
                "associated_pose_count": associated,
                "position_rmse_m": position_rmse,
                "sim3_scale_diagnostic": sim3_scale,
            }
            rows.append(row)
            if associated < args.min_associations or math.isnan(position_rmse):
                continue
            if best is None or position_rmse < float(best["position_rmse_m"]):
                best = dict(row)

    coarse_offsets = frange(args.coarse_min_offset, args.coarse_max_offset, args.coarse_step)
    scan("coarse", coarse_offsets)
    if best is None:
        raise RuntimeError(f"{sequence}: no valid time offset found")

    coarse_best = float(best["time_offset_s"])
    fine_offsets = frange(coarse_best - args.fine_radius, coarse_best + args.fine_radius, args.fine_step)
    scan("fine", fine_offsets)
    assert best is not None

    paths["aligned"].mkdir(parents=True, exist_ok=True)
    with paths["time_offset_scan"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "time_offset_s",
                "associated_pose_count",
                "position_rmse_m",
                "sim3_scale_diagnostic",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    best["coarse_best_time_offset_s"] = coarse_best
    best["scan_csv"] = str(paths["time_offset_scan"])
    best["searched_min_offset_s"] = args.coarse_min_offset
    best["searched_max_offset_s"] = args.coarse_max_offset
    best["coarse_step_s"] = args.coarse_step
    best["fine_radius_s"] = args.fine_radius
    best["fine_step_s"] = args.fine_step
    return best


def make_trial_args(args: argparse.Namespace, sequence: str, best_offset: float) -> SimpleNamespace:
    return SimpleNamespace(
        plato_root=args.plato_root,
        orb_root=args.orb_root,
        dataset="corldata",
        sequence=sequence,
        bag=args.data_root / f"{sequence}.bag",
        groundtruth=args.data_root / "gt" / sequence / "Body_0.tum",
        config=args.config,
        mode="stereo",
        time_offset=best_offset,
        max_time_diff=args.max_time_diff,
        run_name=f"plato_corldata_{sequence}_stereo_vo",
        executable=args.orb_root / "Examples" / "Stereo" / "stereo_tum_vi_batch",
        vocabulary=args.orb_root / "Vocabulary" / "ORBvoc.txt",
        sequence_dir=args.plato_root / "results" / "pseudo_gt" / "corldata" / sequence / "sequence",
        truncation_info=None,
    )


def evaluate_sequence(args: argparse.Namespace, sequence: str, paths: dict[str, Path], best_offset: float) -> dict:
    for key in ["aligned", "plots", "evo"]:
        remove_tree(paths[key])
        paths[key].mkdir(parents=True, exist_ok=True)
    for key in ["metrics", "report", "tuning_log", "frame_alignment"]:
        if paths[key].exists():
            paths[key].unlink()
    paths["time_offset_scan"].parent.mkdir(parents=True, exist_ok=True)

    metrics = trial.evaluate_and_plot(make_trial_args(args, sequence, best_offset), paths)
    return metrics


def write_evo_config(path: Path) -> None:
    path.write_text('{"plot_backend": "Agg"}\n', encoding="utf-8")


def run_evo_command(command: list[object], log_path: Path) -> None:
    trial.run_command(command, log_path=log_path, env={"MPLBACKEND": "Agg"})


def run_evo_outputs(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, object]:
    if args.skip_evo:
        return {"available": False, "reason": "skipped by --skip-evo"}
    if not args.evo_ape.exists():
        return {"available": False, "reason": f"evo_ape executable not found: {args.evo_ape}"}
    if not args.evo_traj.exists():
        return {"available": False, "reason": f"evo_traj executable not found: {args.evo_traj}"}

    evo_dir = paths["evo"]
    evo_dir.mkdir(parents=True, exist_ok=True)
    config = evo_dir / "evo_headless_config.json"
    write_evo_config(config)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    offset = metrics["time_offset_s"]
    max_diff = metrics["max_time_difference_s"]

    raw_common = [
        args.evo_ape,
        "tum",
        paths["raw_groundtruth"],
        paths["raw_estimate_seconds"],
        "--align",
        "--t_offset",
        offset,
        "--t_max_diff",
        max_diff,
        "--no_warnings",
        "--plot_mode",
        "xyz",
        "-c",
        config,
    ]
    frame_common = [
        args.evo_ape,
        "tum",
        paths["associated_groundtruth_tum"],
        paths["aligned_estimate_tum"],
        "--t_max_diff",
        "0.001",
        "--no_warnings",
        "--plot_mode",
        "xyz",
        "-c",
        config,
    ]
    commands = {
        "ape_trans_se3": raw_common
        + [
            "-r",
            "trans_part",
            "--save_results",
            evo_dir / "evo_ape_trans_se3.zip",
            "--save_plot",
            evo_dir / "evo_ape_trans_se3.png",
        ],
        "ape_rot_deg_se3_direct": raw_common
        + [
            "-r",
            "angle_deg",
            "--save_results",
            evo_dir / "evo_ape_rot_deg_se3_direct.zip",
            "--save_plot",
            evo_dir / "evo_ape_rot_deg_se3_direct.png",
        ],
        "ape_trans_frame_compensated": frame_common
        + [
            "-r",
            "trans_part",
            "--save_results",
            evo_dir / "evo_ape_trans_frame_compensated.zip",
            "--save_plot",
            evo_dir / "evo_ape_trans_frame_compensated.png",
        ],
        "ape_rot_deg_frame_compensated": frame_common
        + [
            "-r",
            "angle_deg",
            "--save_results",
            evo_dir / "evo_ape_rot_deg_frame_compensated.zip",
            "--save_plot",
            evo_dir / "evo_ape_rot_deg_frame_compensated.png",
        ],
        "traj_se3": [
            args.evo_traj,
            "tum",
            paths["raw_estimate_seconds"],
            "--ref",
            paths["raw_groundtruth"],
            "--sync",
            "--t_offset",
            offset,
            "--t_max_diff",
            max_diff,
            "--align",
            "--plot_mode",
            "xyz",
            "--save_plot",
            evo_dir / "evo_traj_se3.png",
            "-c",
            config,
        ],
    }
    outputs: dict[str, object] = {}
    for name, command in commands.items():
        log_path = evo_dir / f"{name}.txt"
        run_evo_command(command, log_path)
        outputs[name] = str(log_path)

    return {"available": True, "outputs": outputs}


def read_zip_stats(path: Path) -> dict[str, float]:
    with ZipFile(path, "r") as archive:
        with archive.open("stats.json") as handle:
            return json.loads(handle.read().decode("utf-8"))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def count_log_occurrences(path: Path, needle: str) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.count(needle)


def orb_log_path(paths: dict[str, Path]) -> Path:
    raw_log = paths["raw"] / "orbslam3.log"
    if raw_log.exists():
        return raw_log
    legacy_log = paths["result"] / "logs" / "orbslam3.log"
    return legacy_log


def sequence_summary(sequence: str, paths: dict[str, Path], best: dict[str, object], evo_status: dict[str, object]) -> dict[str, object]:
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    frame = metrics["orientation_frame_alignment"]
    raw_timestamps = trial.count_timestamps(paths["sequence"] / "timestamps.txt")
    raw_estimated = count_lines(paths["raw_estimate"])
    log_path = orb_log_path(paths)
    row = {
        "sequence": sequence,
        "time_offset_s": metrics["time_offset_s"],
        "coarse_best_time_offset_s": best["coarse_best_time_offset_s"],
        "associated_pose_count": metrics["associated_pose_count"],
        "estimated_pose_count": metrics["estimated_pose_count"],
        "input_stereo_pairs": raw_timestamps,
        "tracking_success_rate": raw_estimated / raw_timestamps if raw_timestamps else math.nan,
        "tracking_drop_count_proxy": metrics["tracking_drop_count_proxy"],
        "position_rmse_m": metrics["position_rmse_m"],
        "position_mean_m": metrics["position_mean_m"],
        "position_median_m": metrics["position_median_m"],
        "position_max_m": metrics["position_max_m"],
        "orientation_rmse_deg_frame_compensated": metrics["orientation_rmse_deg"],
        "orientation_mean_deg_frame_compensated": metrics["orientation_mean_deg"],
        "orientation_max_deg_frame_compensated": metrics["orientation_max_deg"],
        "orientation_rmse_deg_direct_evo": metrics["orientation_without_sensor_frame_alignment_rmse_deg"],
        "frame_rotation_angle_deg": frame["angle_deg"],
        "frame_translation_norm_m": frame.get("translation_norm_m", math.nan),
        "frame_translation_lstsq_condition": frame.get("translation_lstsq_condition", math.nan),
        "frame_rotation_residual_rmse_deg": frame["residual_rmse_deg"],
        "rpy_rmse_roll_deg": metrics["rpy_rmse_deg"]["roll"],
        "rpy_rmse_pitch_deg": metrics["rpy_rmse_deg"]["pitch"],
        "rpy_rmse_yaw_deg": metrics["rpy_rmse_deg"]["yaw"],
        "sim3_scale_diagnostic": metrics["sim3_scale_diagnostic"],
        "loop_detected_count": count_log_occurrences(log_path, "*Loop detected"),
        "tracking_failure_log_count": count_log_occurrences(log_path, "Fail to track local map"),
        "relocalization_log_count": count_log_occurrences(log_path, "Relocalized"),
        "orbslam_log": str(log_path),
        "time_offset_scan_csv": str(paths["time_offset_scan"]),
        "metrics_json": str(paths["metrics"]),
        "report_md": str(paths["report"]),
        "evo_dir": str(paths["evo"]),
        "evo_available": evo_status.get("available", False),
    }
    trans_zip = paths["evo"] / "evo_ape_trans_se3.zip"
    rot_zip = paths["evo"] / "evo_ape_rot_deg_frame_compensated.zip"
    if trans_zip.exists():
        stats = read_zip_stats(trans_zip)
        row["evo_trans_rmse_m"] = stats["rmse"]
    if rot_zip.exists():
        stats = read_zip_stats(rot_zip)
        row["evo_frame_compensated_rot_rmse_deg"] = stats["rmse"]
    return row


def write_csv_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def fmt(value: object, precision: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.{precision}f}"
    return str(value)


def write_markdown_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# corldata S00-S05 ORB-SLAM3 Stereo Evaluation",
        "",
        "Settings:",
        "",
        "- ORB-SLAM3 mode: stereo visual-only.",
        "- Image topics: `/davis_left/image_raw` and `/davis_right/image_raw`.",
        "- Config: `plato/configs/orbslam3/corldata_davis_liasuw_0205_baseline.yaml`.",
        "- Time offset: automatically selected per sequence by minimizing the final compensated position RMSE.",
        "- Position metric: SE(3) no-scale world alignment plus a per-sequence right-multiplied constant sensor-to-`Body_0` SE(3) transform.",
        "- Angle metric in the main table: the same fixed-frame SE(3) compensation; direct evo rotation APE is kept separately.",
        "- Direct evo rotation APE is recorded separately because it is dominated by the fixed DAVIS/IMU-to-mocap `Body_0` frame mismatch.",
        "",
        "| seq | offset s | assoc/est | pos RMSE m | pos max m | angle RMSE deg | angle max deg | frame t norm m | direct rot RMSE deg | sim3 scale diag | loop det |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {sequence} | {offset} | {assoc}/{est} | {pos_rmse} | {pos_max} | {ori_rmse} | {ori_max} | {frame_t} | {ori_direct} | {scale} | {loop} |".format(
                sequence=row["sequence"],
                offset=fmt(row["time_offset_s"], 3),
                assoc=row["associated_pose_count"],
                est=row["estimated_pose_count"],
                pos_rmse=fmt(row["position_rmse_m"], 5),
                pos_max=fmt(row["position_max_m"], 5),
                ori_rmse=fmt(row["orientation_rmse_deg_frame_compensated"], 3),
                ori_max=fmt(row["orientation_max_deg_frame_compensated"], 3),
                frame_t=fmt(row["frame_translation_norm_m"], 4),
                ori_direct=fmt(row["orientation_rmse_deg_direct_evo"], 2),
                scale=fmt(row["sim3_scale_diagnostic"], 4),
                loop=row["loop_detected_count"],
            )
        )

    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- CSV summary: `results/pseudo_gt/corldata/S00_S05_evaluation_summary.csv`",
            "- Per-sequence metrics: `results/pseudo_gt/corldata/<seq>/metrics.json`",
            "- Per-sequence time-offset scans: `results/pseudo_gt/corldata/<seq>/aligned/time_offset_scan.csv`",
            "- Per-sequence evo plots/zips: `results/pseudo_gt/corldata/<seq>/evo/`",
            "",
            "Notes:",
            "",
            "- ORB-SLAM3 loop closing is enabled in these runs; `loop det` counts `*Loop detected` messages in each ORB-SLAM3 log.",
            "- The Sim(3) scale diagnostic is reported only as a diagnostic and is not applied to the SE(3) position metric.",
            "- Do not use these as accepted pseudo-groundtruth until the time offset and camera/IMU/Body_0 frame conventions are confirmed across the dataset.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.orb_root = args.orb_root.resolve()
    args.data_root = args.data_root.resolve()
    args.config = args.config.resolve()

    for path in [
        args.orb_root / "Examples" / "Stereo" / "stereo_tum_vi_batch",
        args.orb_root / "Vocabulary" / "ORBvoc.txt",
        args.config,
        args.bag_python,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    rows: list[dict[str, object]] = []
    for sequence in args.sequences:
        paths = result_paths(args.plato_root, sequence)
        print(f"[{sequence}] ensuring raw ORB-SLAM3 outputs", flush=True)
        ensure_raw_outputs(args, sequence, paths)

        print(f"[{sequence}] scanning time offset", flush=True)
        best = scan_time_offset(args, sequence, paths)
        best_offset = float(best["time_offset_s"])
        print(
            f"[{sequence}] best offset {best_offset:.3f}s, scan RMSE {float(best['position_rmse_m']):.5f}m",
            flush=True,
        )
        scan_csv_text = paths["time_offset_scan"].read_text(encoding="utf-8")

        print(f"[{sequence}] evaluating and plotting", flush=True)
        metrics = evaluate_sequence(args, sequence, paths, best_offset)
        # evaluate_sequence recreates aligned/, so restore the just-computed scan CSV.
        paths["time_offset_scan"].write_text(scan_csv_text, encoding="utf-8")

        print(f"[{sequence}] running evo", flush=True)
        evo_status = run_evo_outputs(args, paths)
        (paths["evo"] / "evo_status.json").write_text(json.dumps(evo_status, indent=2), encoding="utf-8")
        rows.append(sequence_summary(sequence, paths, best, evo_status))

    root = args.plato_root / "results" / "pseudo_gt" / "corldata"
    csv_summary = root / "S00_S05_evaluation_summary.csv"
    md_summary = root / "S00_S05_evaluation_report.md"
    json_summary = root / "S00_S05_evaluation_summary.json"
    write_csv_summary(csv_summary, rows)
    write_markdown_summary(md_summary, rows)
    json_summary.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"csv": str(csv_summary), "report": str(md_summary), "json": str(json_summary)}, indent=2))


if __name__ == "__main__":
    main()
