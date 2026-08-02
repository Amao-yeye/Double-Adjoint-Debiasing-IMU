#!/usr/bin/env python3
"""Run OpenVINS on EuRoC bags while replacing only the IMU stream.

The camera streams come from the original EuRoC rosbag. The IMU stream is
loaded from each method's EuRoC-format compensated IMU CSV via the PLATO
`_path_imu_csv_override` extension in `ros1_serial_msckf`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMU_ROOT = Path("/home/amao/workspace/formalPLATO/Results/openvins_imu")
DEFAULT_BAGS_ROOT = Path("/home/amao/datasets/EuRoC/EurocRosbags")
DEFAULT_CONFIG_ROOT = PLATO_ROOT / "external" / "OpenVINS" / "src" / "open_vins" / "config" / "euroc_mav"
DEFAULT_GT_ROOT = PLATO_ROOT / "external" / "OpenVINS" / "src" / "open_vins" / "ov_data" / "euroc_mav"
DEFAULT_ROS_SETUP = Path("/opt/ros/noetic/setup.bash")
DEFAULT_OPENVINS_SETUP = PLATO_ROOT / "external" / "OpenVINS" / "devel" / "setup.bash"

METHODS = ("gt_mse", "gt_nll", "vo_mse", "vo_nll")
SEQUENCES = ("MH_02_easy", "MH_04_difficult", "V1_01_easy", "V1_03_difficult", "V2_02_medium")
BAG_STARTS = {
    "MH_02_easy": 35.0,
    "MH_04_difficult": 10.0,
    "V1_01_easy": 0.0,
    "V1_03_difficult": 0.0,
    "V2_02_medium": 0.0,
}

EUROC_DEFAULT_NOISE = {
    "accelerometer_noise_density": 2.0000e-3,
    "accelerometer_random_walk": 3.0000e-3,
    "gyroscope_noise_density": 1.6968e-4,
    "gyroscope_random_walk": 1.9393e-5,
}


@dataclass(frozen=True)
class RunSpec:
    method: str
    sequence: str
    imu_csv: Path
    bag: Path
    gt: Path
    config: Path
    out_dir: Path
    bag_start: float
    bag_duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu-root", type=Path, default=DEFAULT_IMU_ROOT)
    parser.add_argument("--output-name", default="openvins_results")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to <imu-root>/<output-name>.",
    )
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--ros-setup", type=Path, default=DEFAULT_ROS_SETUP)
    parser.add_argument("--openvins-setup", type=Path, default=DEFAULT_OPENVINS_SETUP)
    parser.add_argument("--ros-home", type=Path, default=Path("/tmp/plato_ros_home"))
    parser.add_argument("--ros-log-dir", type=Path, default=Path("/tmp/plato_ros_log"))
    parser.add_argument("--ros-master-uri", default="http://localhost:11311")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES), choices=SEQUENCES)
    parser.add_argument("--bag-duration", type=float, default=-1.0, help="Use -1 for full sequence.")
    parser.add_argument("--bag-start", type=float, default=None, help="Override the per-sequence bag start time.")
    parser.add_argument("--alignment", default="posyaw", choices=("posyaw", "posyawsingle", "se3", "se3single", "sim3", "none"))
    parser.add_argument(
        "--mse-noise-profile",
        default="default",
        choices=("default", "matching_nll"),
        help="For MSE methods, optionally use the matching NLL OpenVINS white-noise density.",
    )
    parser.add_argument(
        "--mse-noise-sigma-a",
        type=float,
        default=None,
        help="For MSE methods, override accelerometer per-sample sigma and convert to OpenVINS density.",
    )
    parser.add_argument(
        "--mse-noise-sigma-g",
        type=float,
        default=None,
        help="For MSE methods, override gyroscope per-sample sigma and convert to OpenVINS density.",
    )
    parser.add_argument(
        "--force-noise-density-a",
        type=float,
        default=None,
        help="Override accelerometer white-noise density for all selected methods.",
    )
    parser.add_argument(
        "--force-noise-density-g",
        type=float,
        default=None,
        help="Override gyroscope white-noise density for all selected methods.",
    )
    parser.add_argument(
        "--imu-rate-hz",
        type=float,
        default=200.0,
        help="IMU rate used when converting per-sample sigma to white-noise density.",
    )
    parser.add_argument(
        "--estimator-overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Patch scalar entries in estimator_config.yaml for diagnostic sweeps.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shell_quote(path: Path | str) -> str:
    return "'" + str(path).replace("'", "'\"'\"'") + "'"


def make_env_prefix(args: argparse.Namespace) -> str:
    return (
        f"source {shell_quote(args.ros_setup)} && "
        f"source {shell_quote(args.openvins_setup)} && "
        f"export ROS_HOME={shell_quote(args.ros_home)} && "
        f"export ROS_LOG_DIR={shell_quote(args.ros_log_dir)} && "
        f"export ROS_MASTER_URI={shell_quote(args.ros_master_uri)}"
    )


def run_bash(command: str, env_prefix: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", f"{env_prefix} && {command}"], text=True, **kwargs)


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def read_noise_suggestion(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.exists():
        return values
    with path.open() as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in EUROC_DEFAULT_NOISE:
                values[key] = float(value.strip())
    return values


def patch_yaml_scalar(text: str, key: str, value: float) -> str:
    pattern = re.compile(rf"(^\s*{re.escape(key)}:\s*)([^\s#]+)", re.MULTILINE)
    replacement = rf"\g<1>{value:.16e}"
    text, n = pattern.subn(replacement, text)
    if n != 1:
        raise ValueError(f"Expected exactly one YAML key {key}, found {n}")
    return text


def patch_yaml_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(^\s*{re.escape(key)}:\s*)([^\s#]+)", re.MULTILINE)
    replacement = rf"\g<1>{value}"
    text, n = pattern.subn(replacement, text)
    if n != 1:
        raise ValueError(f"Expected exactly one YAML key {key}, found {n}")
    return text


def prepare_method_configs(args: argparse.Namespace, out_root: Path) -> dict[str, Path]:
    config_paths: dict[str, Path] = {}
    for method in args.methods:
        method_config_dir = out_root / "configs" / method
        method_config_dir.mkdir(parents=True, exist_ok=True)
        for name in ("estimator_config.yaml", "kalibr_imucam_chain.yaml", "kalibr_imu_chain.yaml"):
            shutil.copy2(args.config_root / name, method_config_dir / name)

        estimator_yaml = method_config_dir / "estimator_config.yaml"
        estimator_text = estimator_yaml.read_text()
        for item in args.estimator_overrides:
            if "=" not in item:
                raise ValueError(f"Estimator override must be KEY=VALUE, got {item!r}")
            key, value = item.split("=", 1)
            estimator_text = patch_yaml_value(estimator_text, key.strip(), value.strip())
        estimator_yaml.write_text(estimator_text)

        noise = dict(EUROC_DEFAULT_NOISE)
        noise_source = "euroc_default"
        if args.mse_noise_profile == "matching_nll" and method in ("gt_mse", "vo_mse"):
            matching_nll = method.replace("_mse", "_nll")
            noise.update(read_noise_suggestion(args.imu_root / matching_nll / "openvins_noise_suggestion.yaml"))
            noise_source = f"{matching_nll}_suggestion"
        if method in ("gt_mse", "vo_mse") and (args.mse_noise_sigma_a is not None or args.mse_noise_sigma_g is not None):
            dt_s = 1.0 / float(args.imu_rate_hz)
            if args.mse_noise_sigma_a is not None:
                noise["accelerometer_noise_density"] = float(args.mse_noise_sigma_a) * float(np.sqrt(dt_s))
            if args.mse_noise_sigma_g is not None:
                noise["gyroscope_noise_density"] = float(args.mse_noise_sigma_g) * float(np.sqrt(dt_s))
            noise_source = f"mse_sigma_override_rate_{args.imu_rate_hz:g}hz"
        if method.endswith("_nll"):
            noise.update(read_noise_suggestion(args.imu_root / method / "openvins_noise_suggestion.yaml"))
            noise_source = f"{method}_suggestion"
        if args.force_noise_density_a is not None:
            noise["accelerometer_noise_density"] = float(args.force_noise_density_a)
            noise_source += "_force_density"
        if args.force_noise_density_g is not None:
            noise["gyroscope_noise_density"] = float(args.force_noise_density_g)
            noise_source += "_force_density"

        imu_yaml = method_config_dir / "kalibr_imu_chain.yaml"
        text = imu_yaml.read_text()
        for key, value in noise.items():
            text = patch_yaml_scalar(text, key, value)
        imu_yaml.write_text(text)

        (method_config_dir / "noise_used.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "noise_source": noise_source,
                    "estimator_overrides": args.estimator_overrides,
                    **noise,
                },
                indent=2,
            )
            + "\n"
        )
        config_paths[method] = method_config_dir / "estimator_config.yaml"
    return config_paths


def validate_inputs(args: argparse.Namespace) -> dict:
    manifest_path = args.imu_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for method in args.methods:
        if method not in manifest["methods"]:
            raise KeyError(f"Method {method} is not in {manifest_path}")
        for seq in args.sequences:
            imu_csv = args.imu_root / method / seq / "mav0" / "imu0" / "data.csv"
            bag = args.bags_root / f"{seq}.bag"
            gt = args.gt_root / f"{seq}.txt"
            for path in (imu_csv, bag, gt):
                if not path.exists():
                    raise FileNotFoundError(f"Missing required input: {path}")
    return manifest


def ensure_ros_master(args: argparse.Namespace, env_prefix: str, out_root: Path) -> subprocess.Popen | None:
    args.ros_home.mkdir(parents=True, exist_ok=True)
    args.ros_log_dir.mkdir(parents=True, exist_ok=True)
    probe = run_bash("rosparam list", env_prefix, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        return None

    log_path = out_root / "roscore.log"
    log = log_path.open("w")
    proc = subprocess.Popen(["bash", "-lc", f"{env_prefix} && roscore"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(40):
        time.sleep(0.5)
        probe = run_bash("rosparam list", env_prefix, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode == 0:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"roscore exited early; see {log_path}")
    raise RuntimeError(f"ROS master is not reachable at {args.ros_master_uri}; see {log_path}")


def cleanup_ros_namespace(env_prefix: str, name: str) -> None:
    run_bash(f"rosparam delete /{name}", env_prefix, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def run_openvins(spec: RunSpec, args: argparse.Namespace, env_prefix: str) -> dict:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    estimate = spec.out_dir / "estimate.txt"
    odom = spec.out_dir / "odomimu.txt"
    console = spec.out_dir / "openvins_console.log"
    recorder_log = spec.out_dir / "pose_to_file_console.log"
    timing = spec.out_dir / "timing.txt"

    node_name = f"ros1_serial_msckf_{spec.method}_{spec.sequence}"
    rec_name = f"pose_to_file_{spec.method}_{spec.sequence}"
    cleanup_ros_namespace(env_prefix, node_name)
    cleanup_ros_namespace(env_prefix, rec_name)

    if args.skip_existing and count_data_rows(estimate) > 0 and count_data_rows(odom) > 0:
        return {
            "method": spec.method,
            "sequence": spec.sequence,
            "status": "skipped",
            "returncode": 0,
            "estimate_rows": count_data_rows(estimate),
            "odom_rows": count_data_rows(odom),
            "runtime_s": 0.0,
            "out_dir": str(spec.out_dir),
        }

    run_meta = {
        "method": spec.method,
        "sequence": spec.sequence,
        "imu_csv": str(spec.imu_csv),
        "bag": str(spec.bag),
        "groundtruth": str(spec.gt),
        "config": str(spec.config),
        "bag_start": spec.bag_start,
        "bag_duration": spec.bag_duration,
        "node_name": node_name,
        "alignment": args.alignment,
    }
    (spec.out_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    recorder_cmd = (
        f"rosrun ov_eval pose_to_file __name:={rec_name} "
        f"_topic:=/{node_name}/odomimu _topic_type:=Odometry _output:={shell_quote(odom)}"
    )
    recorder = None
    start_time = time.monotonic()
    with recorder_log.open("w") as rec_log:
        recorder = subprocess.Popen(["bash", "-lc", f"{env_prefix} && {recorder_cmd}"], stdout=rec_log, stderr=subprocess.STDOUT)
    time.sleep(1.0)

    openvins_cmd = (
        f"rosrun ov_msckf ros1_serial_msckf {shell_quote(spec.config)} "
        f"__name:={node_name} "
        f"_path_bag:={shell_quote(spec.bag)} "
        f"_bag_start:={spec.bag_start:.6f} "
        f"_bag_durr:={spec.bag_duration:.6f} "
        f"_path_imu_csv_override:={shell_quote(spec.imu_csv)} "
        f"_imu_csv_frame_id:=imu4 "
        f"_save_total_state:=true "
        f"_filepath_est:={shell_quote(estimate)} "
        f"_filepath_std:={shell_quote(spec.out_dir / 'std.txt')} "
        f"_record_timing_information:=true "
        f"_record_timing_filepath:={shell_quote(timing)}"
    )
    with console.open("w") as log:
        proc = subprocess.run(["bash", "-lc", f"{env_prefix} && {openvins_cmd}"], stdout=log, stderr=subprocess.STDOUT)

    time.sleep(1.0)
    stop_process(recorder)
    cleanup_ros_namespace(env_prefix, node_name)
    cleanup_ros_namespace(env_prefix, rec_name)

    estimate_rows = count_data_rows(estimate)
    odom_rows = count_data_rows(odom)
    returncode = int(proc.returncode)
    ok_return = returncode == 0 or (returncode in (134, -6) and estimate_rows > 0)
    status = "ok" if ok_return and estimate_rows > 0 else "failed"

    result = {
        "method": spec.method,
        "sequence": spec.sequence,
        "status": status,
        "returncode": returncode,
        "estimate_rows": estimate_rows,
        "odom_rows": odom_rows,
        "runtime_s": round(time.monotonic() - start_time, 3),
        "out_dir": str(spec.out_dir),
    }
    return result


def load_pose_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            vals = [float(x) for x in stripped.split()]
            if len(vals) >= 8:
                rows.append(vals[:8])
    if not rows:
        raise ValueError(f"No pose rows in {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def associate_poses(
    t_est: np.ndarray,
    p_est: np.ndarray,
    q_est: np.ndarray,
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    q_gt: np.ndarray,
    max_dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.searchsorted(t_gt, t_est)
    idx0 = np.clip(idx - 1, 0, len(t_gt) - 1)
    idx1 = np.clip(idx, 0, len(t_gt) - 1)
    dt0 = np.abs(t_gt[idx0] - t_est)
    dt1 = np.abs(t_gt[idx1] - t_est)
    best = np.where(dt1 < dt0, idx1, idx0)
    best_dt = np.minimum(dt0, dt1)
    mask = best_dt <= max_dt
    if int(mask.sum()) < 3:
        raise ValueError(f"Only {int(mask.sum())} estimate/GT timestamp matches within {max_dt}s")
    return t_est[mask], p_est[mask], q_est[mask], t_gt[best[mask]], p_gt[best[mask]], q_gt[best[mask]]


def jpl_quat_to_itoG(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    r_g_to_i = np.empty((n, 3, 3), dtype=float)
    r_g_to_i[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    r_g_to_i[:, 0, 1] = 2 * x * y + 2 * w * z
    r_g_to_i[:, 0, 2] = 2 * x * z - 2 * w * y
    r_g_to_i[:, 1, 0] = 2 * x * y - 2 * w * z
    r_g_to_i[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    r_g_to_i[:, 1, 2] = 2 * y * z + 2 * w * x
    r_g_to_i[:, 2, 0] = 2 * x * z + 2 * w * y
    r_g_to_i[:, 2, 1] = 2 * y * z - 2 * w * x
    r_g_to_i[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return np.transpose(r_g_to_i, (0, 2, 1))


def rot_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def align_est_to_gt(
    p_est: np.ndarray,
    p_gt: np.ndarray,
    r_est: np.ndarray,
    r_gt: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if method == "none":
        return np.eye(3), np.zeros(3), 1.0

    if method == "posyawsingle":
        c_r = r_est[0] @ r_gt[0].T
        theta = float(np.arctan2(c_r[0, 1] - c_r[1, 0], c_r[0, 0] + c_r[1, 1]))
        r = rot_z(theta)
        t = p_gt[0] - r @ p_est[0]
        return r, t, 1.0

    src_mean = p_est.mean(axis=0)
    dst_mean = p_gt.mean(axis=0)
    src = p_est - src_mean
    dst = p_gt - dst_mean

    if method == "posyaw":
        a = float(np.sum(src[:, 0] * dst[:, 1] - src[:, 1] * dst[:, 0]))
        b = float(np.sum(src[:, 0] * dst[:, 0] + src[:, 1] * dst[:, 1]))
        r = rot_z(float(np.arctan2(a, b)))
        t = dst_mean - r @ src_mean
        return r, t, 1.0

    if method == "se3single":
        r = r_gt[0] @ r_est[0].T
        t = p_gt[0] - r @ p_est[0]
        return r, t, 1.0

    if method in ("se3", "sim3"):
        h = src.T @ dst
        u, svals, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1.0
            r = vt.T @ u.T
        scale = 1.0
        if method == "sim3":
            denom = float(np.sum(src * src))
            scale = float(np.sum(svals) / denom) if denom > 0 else 1.0
        t = dst_mean - scale * (r @ src_mean)
        return r, t, scale

    raise ValueError(f"Unsupported alignment method: {method}")


def metric_summary(values: np.ndarray, suffix: str) -> dict[str, float]:
    return {
        f"ate_rmse_{suffix}": float(np.sqrt(np.mean(values * values))),
        f"ate_mean_{suffix}": float(np.mean(values)),
        f"ate_median_{suffix}": float(np.median(values)),
        f"ate_p95_{suffix}": float(np.percentile(values, 95)),
        f"ate_max_{suffix}": float(np.max(values)),
    }


def fast_ate_metrics(est_path: Path, gt_path: Path, alignment: str) -> dict[str, float | int | str]:
    t_est, p_est, q_est = load_pose_file(est_path)
    t_gt, p_gt, q_gt = load_pose_file(gt_path)
    t_est, p_est, q_est, _, p_gt, q_gt = associate_poses(t_est, p_est, q_est, t_gt, p_gt, q_gt)
    r_est = jpl_quat_to_itoG(q_est)
    r_gt = jpl_quat_to_itoG(q_gt)
    r_align, t_align, scale = align_est_to_gt(p_est, p_gt, r_est, r_gt, alignment)
    p_aligned = scale * (p_est @ r_align.T) + t_align
    r_aligned = np.einsum("ij,njk->nik", r_align, r_est)

    pos_err = np.linalg.norm(p_aligned - p_gt, axis=1)
    rel = np.einsum("nij,njk->nik", np.transpose(r_aligned, (0, 2, 1)), r_gt)
    trace = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    ori_err = np.degrees(np.arccos(trace))

    metrics: dict[str, float | int | str] = {
        "metric_status": "ok",
        "metric_returncode": 0,
        "matched_poses": int(len(pos_err)),
        "alignment_scale": float(scale),
    }
    metrics.update(metric_summary(pos_err, "pos_m"))
    metrics.update(metric_summary(ori_err, "ori_deg"))
    return metrics


def evaluate_run(spec: RunSpec, args: argparse.Namespace, env_prefix: str) -> dict:
    odom = spec.out_dir / "odomimu.txt"
    metrics_path = spec.out_dir / f"metrics_{args.alignment}.txt"
    if count_data_rows(odom) == 0:
        return {"metric_status": "no_odom"}
    try:
        metrics = fast_ate_metrics(odom, spec.gt, args.alignment)
    except Exception as exc:
        metrics_path.write_text(f"metric_status: failed\nerror: {exc}\n")
        return {"metric_status": "failed", "metric_returncode": -1, "metric_error": str(exc)}

    lines = [
        f"alignment: {args.alignment}",
        f"matched_poses: {metrics['matched_poses']}",
        f"alignment_scale: {metrics['alignment_scale']:.12g}",
        "Absolute Trajectory Error",
        f"rmse_ori = {metrics['ate_rmse_ori_deg']:.6g} | rmse_pos = {metrics['ate_rmse_pos_m']:.6g}",
        f"mean_ori = {metrics['ate_mean_ori_deg']:.6g} | mean_pos = {metrics['ate_mean_pos_m']:.6g}",
        f"max_ori  = {metrics['ate_max_ori_deg']:.6g} | max_pos  = {metrics['ate_max_pos_m']:.6g}",
    ]
    metrics_path.write_text("\n".join(lines) + "\n")
    return metrics


def write_summaries(out_root: Path, rows: list[dict]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=True) + "\n")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    lines = ["# EuRoC compensated IMU OpenVINS summary", ""]
    lines.append("| method | ok/total | mean ATE pos RMSE (m) | mean ATE ori RMSE (deg) |")
    lines.append("|---|---:|---:|---:|")
    for method, method_rows in grouped.items():
        ok = [r for r in method_rows if r.get("status") in ("ok", "skipped") and r.get("metric_status") == "ok"]
        pos = [float(r["ate_rmse_pos_m"]) for r in ok if "ate_rmse_pos_m" in r]
        ori = [float(r["ate_rmse_ori_deg"]) for r in ok if "ate_rmse_ori_deg" in r]
        pos_mean = sum(pos) / len(pos) if pos else float("nan")
        ori_mean = sum(ori) / len(ori) if ori else float("nan")
        lines.append(f"| {method} | {len(ok)}/{len(method_rows)} | {pos_mean:.6g} | {ori_mean:.6g} |")
    lines.append("")
    lines.append("Per-run outputs contain `estimate.txt`, `std.txt`, `timing.txt`, `odomimu.txt`, OpenVINS logs, and metrics logs.")
    (out_root / "summary.md").write_text("\n".join(lines) + "\n")


def make_specs(args: argparse.Namespace, config_paths: dict[str, Path], out_root: Path) -> Iterable[RunSpec]:
    for method in args.methods:
        for seq in args.sequences:
            yield RunSpec(
                method=method,
                sequence=seq,
                imu_csv=args.imu_root / method / seq / "mav0" / "imu0" / "data.csv",
                bag=args.bags_root / f"{seq}.bag",
                gt=args.gt_root / f"{seq}.txt",
                config=config_paths[method],
                out_dir=out_root / method / seq,
                bag_start=args.bag_start if args.bag_start is not None else BAG_STARTS[seq],
                bag_duration=args.bag_duration,
            )


def main() -> int:
    args = parse_args()
    out_root = args.output_root if args.output_root is not None else args.imu_root / args.output_name
    manifest = validate_inputs(args)
    if args.dry_run:
        for method in args.methods:
            for seq in args.sequences:
                print(
                    f"{method}/{seq}: "
                    f"bag={args.bags_root / f'{seq}.bag'} "
                    f"imu={args.imu_root / method / seq / 'mav0' / 'imu0' / 'data.csv'} "
                    f"gt={args.gt_root / f'{seq}.txt'}"
                )
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "input_manifest_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    config_paths = prepare_method_configs(args, out_root)
    env_prefix = make_env_prefix(args)

    roscore = ensure_ros_master(args, env_prefix, out_root)
    rows: list[dict] = []
    try:
        specs = list(make_specs(args, config_paths, out_root))
        for idx, spec in enumerate(specs, start=1):
            print(f"[{idx}/{len(specs)}] {spec.method}/{spec.sequence}", flush=True)
            row = run_openvins(spec, args, env_prefix)
            if row["status"] in ("ok", "skipped"):
                row.update(evaluate_run(spec, args, env_prefix))
            else:
                row["metric_status"] = "not_run"
            rows.append(row)
            write_summaries(out_root, rows)
            print(
                f"  status={row['status']} est={row['estimate_rows']} odom={row['odom_rows']} "
                f"ate_pos={row.get('ate_rmse_pos_m', 'nan')}",
                flush=True,
            )
    finally:
        stop_process(roscore)

    write_summaries(out_root, rows)
    failed = [r for r in rows if r.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
