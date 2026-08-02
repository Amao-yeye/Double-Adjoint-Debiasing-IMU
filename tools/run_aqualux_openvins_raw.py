#!/usr/bin/env python3
"""Run AquaLux Test sequences through OpenVINS with raw IMU input.

The OpenVINS camera stream is read from each AquaLux rosbag. The IMU stream is
replaced by the raw ATO-side IMU diagnostic CSV through the patched
`path_imu_csv_override` route in ros1_serial_msckf.
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
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("AQUALUX_DATA_ROOT", REPO_ROOT / "datasets" / "AquaLux" / "0524_data")
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "AquaLux" / "openvins_test"
DEFAULT_OPENVINS_ROOT = Path(os.environ.get("OPENVINS_ROOT", REPO_ROOT / "external" / "OpenVINS"))
DEFAULT_ROS_SETUP = Path("/opt/ros/noetic/setup.bash")
DEFAULT_ROS_HOME = Path("/tmp/formalplato_ros_home")
DEFAULT_ROS_LOG_DIR = Path("/tmp/formalplato_ros_log")
DEFAULT_EVO_HOME = Path("/tmp/formalplato_evo_home")
DEFAULT_MPLCONFIGDIR = Path("/tmp/formalplato_mpl")
DEFAULT_EVO_USER_SITE = Path(os.environ["EVO_USER_SITE"]) if "EVO_USER_SITE" in os.environ else None

TEST_SEQUENCES = tuple(f"Test_{idx:03d}" for idx in range(1, 20, 2))

CAM0_INTRINSICS = [312.0206439606905, 312.4734720152093, 159.57147431980286, 125.55510681344391]
CAM0_DISTORTION = [-0.30624346790624646, 0.1425146717217006, -0.0013100988384794235, -0.0047694614401734324]
CAM1_INTRINSICS = [310.8584570180142, 311.4594070713915, 190.26035791110112, 131.3065531064017]
CAM1_DISTORTION = [-0.30328331620524857, 0.1270139887276728, -0.0013512660200573788, 0.001013934053373898]

T_IMU_CAM0 = [
    [0.99982062, -0.00531549, -0.01817911, 0.00126654],
    [0.00546713, 0.99995059, 0.00830192, -0.00295328],
    [0.01813409, -0.00839982, 0.99980028, 0.00206342],
    [0.0, 0.0, 0.0, 1.0],
]
T_IMU_CAM1 = [
    [0.99970707, -0.00501381, 0.02367771, 0.19905791],
    [0.00485213, 0.99996455, 0.00688113, -0.00339201],
    [-0.02371137, -0.00676423, 0.99969596, 0.0016675],
    [0.0, 0.0, 0.0, 1.0],
]

AQUALUX_IMU_NOISE = {
    "accelerometer_noise_density": 0.0436073667065,
    "accelerometer_random_walk": 0.000691276415008,
    "gyroscope_noise_density": 0.00363608382743,
    "gyroscope_random_walk": 4.54949206016e-05,
}

OPENVINS_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
    "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    sequence: str
    bag: Path
    raw_imu_csv: Path
    openvins_imu_csv: Path
    gt_tum: Path
    config: Path
    out_dir: Path
    bag_start: float
    bag_duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--openvins-root", type=Path, default=DEFAULT_OPENVINS_ROOT)
    parser.add_argument("--ros-setup", type=Path, default=DEFAULT_ROS_SETUP)
    parser.add_argument("--ros-home", type=Path, default=DEFAULT_ROS_HOME)
    parser.add_argument("--ros-log-dir", type=Path, default=DEFAULT_ROS_LOG_DIR)
    parser.add_argument("--ros-master-uri", default="http://localhost:11311")
    parser.add_argument("--evo-home", type=Path, default=DEFAULT_EVO_HOME)
    parser.add_argument("--mplconfigdir", type=Path, default=DEFAULT_MPLCONFIGDIR)
    parser.add_argument("--evo-user-site", type=Path, default=DEFAULT_EVO_USER_SITE)
    parser.add_argument("--method", default="raw")
    parser.add_argument("--sequences", nargs="+", default=list(TEST_SEQUENCES))
    parser.add_argument(
        "--source-imu-root",
        type=Path,
        default=None,
        help="Optional per-sequence IMU CSV root. Uses <root>/<sequence>/<source-imu-filename> instead of raw AquaLux IMU.",
    )
    parser.add_argument("--source-imu-filename", default="corrected_imu.csv")
    parser.add_argument("--bag-start", type=float, default=0.0)
    parser.add_argument("--bag-duration", type=float, default=-1.0, help="Use -1 for the full bag.")
    parser.add_argument("--t-max-diff", type=float, default=0.02)
    parser.add_argument("--init-imu-thresh", type=float, default=0.05)
    parser.add_argument("--init-max-features", type=int, default=48)
    parser.add_argument("--init-dyn-min-deg", type=float, default=2.0)
    parser.add_argument("--num-pts", type=int, default=300)
    parser.add_argument("--fast-threshold", type=int, default=5)
    parser.add_argument("--min-px-dist", type=int, default=5)
    parser.add_argument("--accelerometer-noise-density", type=float, default=None)
    parser.add_argument("--gyroscope-noise-density", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shell_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def make_env_prefix(args: argparse.Namespace) -> str:
    openvins_setup = args.openvins_root / "devel" / "setup.bash"
    return (
        f"source {shell_quote(args.ros_setup)} && "
        f"source {shell_quote(openvins_setup)} && "
        f"export ROS_HOME={shell_quote(args.ros_home)} && "
        f"export ROS_LOG_DIR={shell_quote(args.ros_log_dir)} && "
        f"export ROS_MASTER_URI={shell_quote(args.ros_master_uri)}"
    )


def run_bash(command: str, env_prefix: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", f"{env_prefix} && {command}"], text=True, **kwargs)


def ensure_ros_master(args: argparse.Namespace, env_prefix: str) -> subprocess.Popen | None:
    args.ros_home.mkdir(parents=True, exist_ok=True)
    args.ros_log_dir.mkdir(parents=True, exist_ok=True)
    probe = run_bash("rosparam list", env_prefix, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        return None

    log_path = args.results_root / "roscore.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(["bash", "-lc", f"{env_prefix} && roscore"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
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


def timestamp_seconds_to_ns(text: str) -> int:
    value = Decimal(text.strip()) * Decimal(1_000_000_000)
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def export_openvins_imu_csv(raw_csv: Path, output_csv: Path) -> dict[str, float | int | str]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, list[float]]] = []
    with raw_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 7 or header[0].strip().lower() != "timestamp":
            raise ValueError(f"Unexpected AquaLux IMU header in {raw_csv}")
        for row in reader:
            if len(row) < 7:
                continue
            timestamp_ns = timestamp_seconds_to_ns(row[0])
            values = [float(x) for x in row[1:7]]
            rows.append((timestamp_ns, values))

    if not rows:
        raise ValueError(f"No IMU rows found in {raw_csv}")
    timestamps = [item[0] for item in rows]
    non_increasing = sum(1 for prev, cur in zip(timestamps, timestamps[1:]) if cur <= prev)
    if non_increasing:
        raise ValueError(f"Non-increasing IMU timestamps in {raw_csv}: {non_increasing}")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OPENVINS_IMU_HEADER.split(","))
        for timestamp_ns, values in rows:
            writer.writerow([str(timestamp_ns), *[f"{value:.18e}" for value in values]])

    duration_s = (timestamps[-1] - timestamps[0]) * 1e-9
    return {
        "raw_imu_csv": str(raw_csv),
        "openvins_imu_csv": str(output_csv),
        "imu_rows": len(rows),
        "imu_start_ns": timestamps[0],
        "imu_end_ns": timestamps[-1],
        "imu_duration_s": duration_s,
    }


def format_matrix(matrix: list[list[float]], indent: str = "    ") -> str:
    lines = []
    for row in matrix:
        values = ", ".join(f"{value:.12g}" for value in row)
        lines.append(f"{indent}- [{values}]")
    return "\n".join(lines)


def format_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.16g}" for value in values) + "]"


def imu_noise_from_args(args: argparse.Namespace) -> dict[str, float]:
    noise = dict(AQUALUX_IMU_NOISE)
    if args.accelerometer_noise_density is not None:
        noise["accelerometer_noise_density"] = float(args.accelerometer_noise_density)
    if args.gyroscope_noise_density is not None:
        noise["gyroscope_noise_density"] = float(args.gyroscope_noise_density)
    return noise


def write_openvins_configs(config_dir: Path, args: argparse.Namespace) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    estimator = config_dir / "estimator_config.yaml"
    imu_chain = config_dir / "kalibr_imu_chain.yaml"
    imucam_chain = config_dir / "kalibr_imucam_chain.yaml"
    imu_noise = imu_noise_from_args(args)

    estimator.write_text(
        f"""%YAML:1.0

verbosity: "INFO"

use_fej: true
integration: "rk4"
use_stereo: true
max_cameras: 2

calib_cam_extrinsics: false
calib_cam_intrinsics: false
calib_cam_timeoffset: false
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false

max_clones: 11
max_slam: 50
max_slam_in_update: 25
max_msckf_in_update: 40
dt_slam_delay: 1

gravity_mag: 9.7879

feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"

try_zupt: false
zupt_chi2_multipler: 0
zupt_max_velocity: 0.1
zupt_noise_multiplier: 10
zupt_max_disparity: 0.5
zupt_only_at_beginning: false

init_window_time: 2.0
init_imu_thresh: {args.init_imu_thresh:.12g}
init_max_disparity: 10.0
init_max_features: {int(args.init_max_features)}

init_dyn_use: true
init_dyn_mle_opt_calib: false
init_dyn_mle_max_iter: 50
init_dyn_mle_max_time: 0.05
init_dyn_mle_max_threads: 6
init_dyn_num_pose: 6
init_dyn_min_deg: {args.init_dyn_min_deg:.12g}
init_dyn_inflation_ori: 10
init_dyn_inflation_vel: 100
init_dyn_inflation_bg: 10
init_dyn_inflation_ba: 100
init_dyn_min_rec_cond: 1e-20
init_dyn_bias_g: [0.0, 0.0, 0.0]
init_dyn_bias_a: [0.0, 0.0, 0.0]

record_timing_information: false
record_timing_filepath: "/tmp/traj_timing.txt"
save_total_state: false
filepath_est: "/tmp/ov_estimate.txt"
filepath_std: "/tmp/ov_estimate_std.txt"
filepath_gt: "/tmp/ov_groundtruth.txt"

use_klt: true
num_pts: {int(args.num_pts)}
fast_threshold: {int(args.fast_threshold)}
grid_x: 5
grid_y: 5
min_px_dist: {int(args.min_px_dist)}
knn_ratio: 0.70
track_frequency: 38.0
downsample_cameras: false
num_opencv_threads: 4
histogram_method: "HISTOGRAM"

use_aruco: false
num_aruco: 1024
downsize_aruco: true

up_msckf_sigma_px: 1
up_msckf_chi2_multipler: 1
up_slam_sigma_px: 1
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1
up_aruco_chi2_multipler: 1

use_mask: false

relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
""",
        encoding="utf-8",
    )

    imu_chain.write_text(
        f"""%YAML:1.0

imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: {imu_noise["accelerometer_noise_density"]:.16g}
  accelerometer_random_walk: {imu_noise["accelerometer_random_walk"]:.16g}
  gyroscope_noise_density: {imu_noise["gyroscope_noise_density"]:.16g}
  gyroscope_random_walk: {imu_noise["gyroscope_random_walk"]:.16g}
  rostopic: /davis_left/imu
  time_offset: 0.0
  update_rate: 1000.0
  model: "kalibr"
  Tw:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoGYRO:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Ta:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoACC:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Tg:
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )

    imucam_chain.write_text(
        f"""%YAML:1.0

cam0:
  T_imu_cam:
{format_matrix(T_IMU_CAM0)}
  cam_overlaps: [1]
  camera_model: pinhole
  distortion_coeffs: {format_list(CAM0_DISTORTION)}
  distortion_model: radtan
  intrinsics: {format_list(CAM0_INTRINSICS)}
  resolution: [346, 260]
  rostopic: /davis_left/image_raw
  timeshift_cam_imu: 0.00777437
cam1:
  T_imu_cam:
{format_matrix(T_IMU_CAM1)}
  cam_overlaps: [0]
  camera_model: pinhole
  distortion_coeffs: {format_list(CAM1_DISTORTION)}
  distortion_model: radtan
  intrinsics: {format_list(CAM1_INTRINSICS)}
  resolution: [346, 260]
  rostopic: /davis_right/image_raw
  timeshift_cam_imu: 0.00777437
""",
        encoding="utf-8",
    )

    manifest = {
        "source": {
            "cam0": str(args.data_root.parent / "configuration" / "cam0.yaml"),
            "cam1": str(args.data_root.parent / "configuration" / "cam1.yaml"),
            "imu_camera_extrinsics": str(args.data_root.parent / "configuration" / "uw_robot.yaml"),
        },
        "fixed_calibration": {
            "calib_cam_extrinsics": False,
            "calib_cam_intrinsics": False,
            "calib_cam_timeoffset": False,
            "timeshift_cam_imu": 0.00777437,
        },
        "gravity_mag": 9.7879,
        "imu_noise": imu_noise,
        "initialization": {
            "init_imu_thresh": args.init_imu_thresh,
            "init_max_features": args.init_max_features,
            "init_dyn_min_deg": args.init_dyn_min_deg,
        },
        "feature_tracking": {
            "num_pts": args.num_pts,
            "fast_threshold": args.fast_threshold,
            "min_px_dist": args.min_px_dist,
        },
        "topics": {
            "cam0": "/davis_left/image_raw",
            "cam1": "/davis_right/image_raw",
            "imu": "/davis_left/imu",
        },
    }
    (config_dir / "config_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    text = estimator.read_text(encoding="utf-8")
    if "calib_cam_extrinsics: false" not in text:
        raise RuntimeError(f"Generated config does not disable extrinsic estimation: {estimator}")
    return estimator


def data_sequence_dir(data_root: Path, sequence: str) -> Path:
    return data_root / "0524" / sequence


def make_specs(args: argparse.Namespace, config_path: Path) -> Iterable[RunSpec]:
    run_root = args.results_root / args.method
    for sequence in args.sequences:
        seq_dir = data_sequence_dir(args.data_root, sequence)
        source_imu_csv = (
            args.source_imu_root / sequence / args.source_imu_filename
            if args.source_imu_root is not None
            else seq_dir / "imu_diagnostics" / "imu_left.csv"
        )
        yield RunSpec(
            method=args.method,
            sequence=sequence,
            bag=args.data_root / f"{sequence}.bag",
            raw_imu_csv=source_imu_csv,
            openvins_imu_csv=run_root / sequence / "imu_override.csv",
            gt_tum=seq_dir / "ATO" / "gt_imu_gravity_aligned.tum",
            config=config_path,
            out_dir=run_root / sequence,
            bag_start=args.bag_start,
            bag_duration=args.bag_duration,
        )


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        args.ros_setup,
        args.openvins_root / "devel" / "setup.bash",
        args.openvins_root / "src" / "open_vins" / "ov_msckf" / "src" / "ros1_serial_msckf.cpp",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for sequence in args.sequences:
        seq_dir = data_sequence_dir(args.data_root, sequence)
        source_imu_csv = (
            args.source_imu_root / sequence / args.source_imu_filename
            if args.source_imu_root is not None
            else seq_dir / "imu_diagnostics" / "imu_left.csv"
        )
        required_seq = [
            args.data_root / f"{sequence}.bag",
            source_imu_csv,
            seq_dir / "ATO" / "gt_imu_gravity_aligned.tum",
        ]
        for path in required_seq:
            if not path.exists():
                raise FileNotFoundError(path)


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def run_openvins(spec: RunSpec, args: argparse.Namespace, env_prefix: str) -> dict[str, object]:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    estimate = spec.out_dir / "estimate.txt"
    std = spec.out_dir / "std.txt"
    odom = spec.out_dir / "odomimu.txt"
    timing = spec.out_dir / "timing.txt"
    console = spec.out_dir / "openvins_console.log"
    recorder_log = spec.out_dir / "pose_to_file_console.log"

    node_name = f"openvins_{spec.method}_{spec.sequence}"
    recorder_name = f"pose_to_file_{spec.method}_{spec.sequence}"
    cleanup_ros_namespace(env_prefix, node_name)
    cleanup_ros_namespace(env_prefix, recorder_name)

    if args.skip_existing and count_data_rows(odom) > 0 and count_data_rows(estimate) > 0:
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

    imu_meta = export_openvins_imu_csv(spec.raw_imu_csv, spec.openvins_imu_csv)
    gt_copy = spec.out_dir / "gt_imu_gravity_aligned.tum"
    shutil.copy2(spec.gt_tum, gt_copy)

    run_meta = {
        "method": spec.method,
        "sequence": spec.sequence,
        "bag": str(spec.bag),
        "raw_imu_csv": str(spec.raw_imu_csv),
        "source_imu_csv": str(spec.raw_imu_csv),
        "openvins_imu_csv": str(spec.openvins_imu_csv),
        "groundtruth_tum": str(spec.gt_tum),
        "groundtruth_copy": str(gt_copy),
        "config": str(spec.config),
        "bag_start": spec.bag_start,
        "bag_duration": spec.bag_duration,
        "node_name": node_name,
        "recorder_name": recorder_name,
        "openvins_calibration_locked": {
            "calib_cam_extrinsics": False,
            "calib_cam_intrinsics": False,
            "calib_cam_timeoffset": False,
        },
        **imu_meta,
    }
    (spec.out_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    recorder_cmd = (
        f"rosrun ov_eval pose_to_file __name:={recorder_name} "
        f"_topic:=/{node_name}/odomimu _topic_type:=Odometry _output:={shell_quote(odom)}"
    )
    start_time = time.monotonic()
    recorder: subprocess.Popen | None = None
    with recorder_log.open("w", encoding="utf-8") as rec_log:
        recorder = subprocess.Popen(["bash", "-lc", f"{env_prefix} && {recorder_cmd}"], stdout=rec_log, stderr=subprocess.STDOUT)
    time.sleep(1.0)

    openvins_cmd = (
        f"rosrun ov_msckf ros1_serial_msckf {shell_quote(spec.config)} "
        f"__name:={node_name} "
        f"_path_bag:={shell_quote(spec.bag)} "
        f"_bag_start:={spec.bag_start:.6f} "
        f"_bag_durr:={spec.bag_duration:.6f} "
        f"_path_imu_csv_override:={shell_quote(spec.openvins_imu_csv)} "
        f"_imu_csv_frame_id:=davis_left_imu "
        f"_topic_imu:=/davis_left/imu "
        f"_topic_camera0:=/davis_left/image_raw "
        f"_topic_camera1:=/davis_right/image_raw "
        f"_use_stereo:=true "
        f"_max_cameras:=2 "
        f"_save_total_state:=true "
        f"_filepath_est:={shell_quote(estimate)} "
        f"_filepath_std:={shell_quote(std)} "
        f"_record_timing_information:=true "
        f"_record_timing_filepath:={shell_quote(timing)}"
    )
    with console.open("w", encoding="utf-8") as log:
        proc = subprocess.run(["bash", "-lc", f"{env_prefix} && {openvins_cmd}"], stdout=log, stderr=subprocess.STDOUT)

    time.sleep(1.0)
    stop_process(recorder)
    cleanup_ros_namespace(env_prefix, node_name)
    cleanup_ros_namespace(env_prefix, recorder_name)

    estimate_rows = count_data_rows(estimate)
    odom_rows = count_data_rows(odom)
    returncode = int(proc.returncode)
    ok_return = returncode == 0 or (returncode in (134, -6) and odom_rows > 0)
    status = "ok" if ok_return and odom_rows > 0 else "failed"
    return {
        "method": spec.method,
        "sequence": spec.sequence,
        "status": status,
        "returncode": returncode,
        "estimate_rows": estimate_rows,
        "odom_rows": odom_rows,
        "runtime_s": round(time.monotonic() - start_time, 3),
        "out_dir": str(spec.out_dir),
        "openvins_console": str(console),
        "pose_to_file_console": str(recorder_log),
    }


def clean_openvins_tum(odom: Path, output_tum: Path) -> dict[str, int | str]:
    output_tum.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_read = 0
    dropped_non_increasing = 0
    dropped_malformed = 0
    last_t = float("-inf")
    with odom.open(errors="replace") as src, output_tum.open("w", encoding="utf-8") as dst:
        dst.write("# timestamp tx ty tz qx qy qz qw\n")
        for line in src:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                dropped_malformed += 1
                continue
            try:
                values = [float(x) for x in parts[:8]]
            except ValueError:
                dropped_malformed += 1
                continue
            rows_read += 1
            timestamp = values[0]
            if timestamp <= last_t:
                dropped_non_increasing += 1
                continue
            q_norm = sum(q * q for q in values[4:8]) ** 0.5
            if q_norm <= 0.0:
                dropped_malformed += 1
                continue
            values[4:8] = [q / q_norm for q in values[4:8]]
            dst.write(
                f"{values[0]:.9f} {values[1]:.12g} {values[2]:.12g} {values[3]:.12g} "
                f"{values[4]:.12g} {values[5]:.12g} {values[6]:.12g} {values[7]:.12g}\n"
            )
            last_t = timestamp
            rows_written += 1
    return {
        "estimate_tum": str(output_tum),
        "estimate_tum_rows": rows_written,
        "estimate_rows_read": rows_read,
        "estimate_non_increasing_dropped": dropped_non_increasing,
        "estimate_malformed_dropped": dropped_malformed,
    }


def read_timestamps_tum(path: Path) -> list[float]:
    timestamps: list[float] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 8:
                timestamps.append(float(parts[0]))
    return timestamps


def count_timestamp_matches(reference: Path, estimate: Path, max_dt: float) -> int:
    ref = read_timestamps_tum(reference)
    est = read_timestamps_tum(estimate)
    if not ref or not est:
        return 0
    count = 0
    j = 0
    for t in est:
        while j + 1 < len(ref) and abs(ref[j + 1] - t) <= abs(ref[j] - t):
            j += 1
        if abs(ref[j] - t) <= max_dt:
            count += 1
    return count


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_evo_stats(stdout: str) -> dict[str, float]:
    clean = ANSI_RE.sub("", stdout)
    stats: dict[str, float] = {}
    for line in clean.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[0] in {"max", "mean", "median", "min", "rmse", "sse", "std"}:
            try:
                stats[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return stats


def make_evo_env(args: argparse.Namespace) -> dict[str, str]:
    args.evo_home.mkdir(parents=True, exist_ok=True)
    args.mplconfigdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(args.evo_home)
    env["MPLCONFIGDIR"] = str(args.mplconfigdir)
    if args.evo_user_site is not None and args.evo_user_site.exists():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(args.evo_user_site) if not existing else str(args.evo_user_site) + os.pathsep + existing
    return env


def run_evo_ape(
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
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=make_evo_env(args))
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stats = parse_evo_stats(proc.stdout)
    status = "ok" if proc.returncode == 0 and "rmse" in stats else "failed"
    return {
        "status": status,
        "returncode": int(proc.returncode),
        "rmse": stats.get("rmse"),
        "mean": stats.get("mean"),
        "median": stats.get("median"),
        "max": stats.get("max"),
        "log": str(log_path),
        "result_zip": str(save_path) if save_path.exists() else "",
    }


def evaluate_run(spec: RunSpec, args: argparse.Namespace) -> dict[str, object]:
    odom = spec.out_dir / "odomimu.txt"
    if count_data_rows(odom) == 0:
        return {
            "metric_status": "no_odom",
            "evo_position_status": "not_run",
            "evo_rotation_status": "not_run",
        }
    estimate_tum = spec.out_dir / "estimate_tum.tum"
    clean_meta = clean_openvins_tum(odom, estimate_tum)
    if int(clean_meta["estimate_tum_rows"]) < 3:
        return {
            **clean_meta,
            "metric_status": "too_few_estimates",
            "evo_position_status": "not_run",
            "evo_rotation_status": "not_run",
        }
    matched = count_timestamp_matches(spec.gt_tum, estimate_tum, args.t_max_diff)
    pos = run_evo_ape(
        args,
        spec.gt_tum,
        estimate_tum,
        "trans_part",
        spec.out_dir / "evo_ape_position.zip",
        spec.out_dir / "evo_ape_position.txt",
    )
    rot = run_evo_ape(
        args,
        spec.gt_tum,
        estimate_tum,
        "angle_deg",
        spec.out_dir / "evo_ape_rotation.zip",
        spec.out_dir / "evo_ape_rotation.txt",
    )
    metric_status = "ok" if pos["status"] == "ok" and rot["status"] == "ok" else "failed"
    return {
        **clean_meta,
        "metric_status": metric_status,
        "matched_poses_nearest": matched,
        "evo_position_status": pos["status"],
        "evo_position_returncode": pos["returncode"],
        "ate_rmse_pos_m": pos["rmse"],
        "ate_mean_pos_m": pos["mean"],
        "ate_median_pos_m": pos["median"],
        "ate_max_pos_m": pos["max"],
        "evo_position_log": pos["log"],
        "evo_position_zip": pos["result_zip"],
        "evo_rotation_status": rot["status"],
        "evo_rotation_returncode": rot["returncode"],
        "ate_rmse_rot_deg": rot["rmse"],
        "ate_mean_rot_deg": rot["mean"],
        "ate_median_rot_deg": rot["median"],
        "ate_max_rot_deg": rot["max"],
        "evo_rotation_log": rot["log"],
        "evo_rotation_zip": rot["result_zip"],
    }


def value_to_csv(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value_to_csv(row.get(key)) for key in keys})


def write_summaries(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    run_root = args.results_root / args.method
    write_csv(run_root / "metrics.csv", rows)
    (run_root / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.results_root / "summary.csv", rows)

    ok_rows = [row for row in rows if row.get("status") in {"ok", "skipped"} and row.get("metric_status") == "ok"]
    pos_values = [float(row["ate_rmse_pos_m"]) for row in ok_rows if row.get("ate_rmse_pos_m") is not None]
    rot_values = [float(row["ate_rmse_rot_deg"]) for row in ok_rows if row.get("ate_rmse_rot_deg") is not None]
    pos_mean = sum(pos_values) / len(pos_values) if pos_values else None
    rot_mean = sum(rot_values) / len(rot_values) if rot_values else None

    lines = [
        "# AquaLux OpenVINS Test Summary",
        "",
        "Evaluation: `evo_ape tum --align`, no scale correction, `t_max_diff=0.02`.",
        "Position uses `pose_relation=trans_part`; rotation uses `pose_relation=angle_deg`.",
        "",
        "Fixed OpenVINS calibration for all methods:",
        "",
        "```text",
        "calib_cam_extrinsics: false",
        "calib_cam_intrinsics: false",
        "calib_cam_timeoffset: false",
        "```",
        "",
        "AquaLux raw initialization/feature settings:",
        "",
        "```text",
        f"init_imu_thresh: {args.init_imu_thresh:.12g}",
        f"init_max_features: {int(args.init_max_features)}",
        f"init_dyn_min_deg: {args.init_dyn_min_deg:.12g}",
        f"num_pts: {int(args.num_pts)}",
        f"fast_threshold: {int(args.fast_threshold)}",
        f"min_px_dist: {int(args.min_px_dist)}",
        "```",
        "",
        "| method | status | ok/total | mean position RMSE (m) | mean rotation RMSE (deg) |",
        "|---|---|---:|---:|---:|",
        (
            f"| {args.method} | complete/in-progress | {len(ok_rows)}/{len(rows)} | "
            f"{pos_mean:.6g} | {rot_mean:.6g} |"
            if pos_mean is not None and rot_mean is not None
            else f"| {args.method} | complete/in-progress | {len(ok_rows)}/{len(rows)} | n/a | n/a |"
        ),
        "| rss_baseline | pending | 0/0 | n/a | n/a |",
        "| ours | pending | 0/0 | n/a | n/a |",
        "",
        "## Raw Per Sequence",
        "",
        "| sequence | run status | metric status | odom rows | matched poses | position RMSE (m) | rotation RMSE (deg) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        pos = row.get("ate_rmse_pos_m")
        rot = row.get("ate_rmse_rot_deg")
        pos_text = f"{float(pos):.6g}" if pos is not None and pos != "" else "n/a"
        rot_text = f"{float(rot):.6g}" if rot is not None and rot != "" else "n/a"
        lines.append(
            f"| {row.get('sequence')} | {row.get('status')} | {row.get('metric_status')} | "
            f"{row.get('odom_rows', 0)} | {row.get('matched_poses_nearest', 0)} | {pos_text} | {rot_text} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            f"- Raw run root: `{run_root}`",
            f"- Raw metrics CSV: `{run_root / 'metrics.csv'}`",
            f"- Generated OpenVINS config: `{args.results_root / 'configs' / args.method / 'estimator_config.yaml'}`",
            "- Per-sequence folders contain `imu_override.csv`, `estimate.txt`, `std.txt`,",
            "  `timing.txt`, `odomimu.txt`, `estimate_tum.tum`, evo logs/zips, and run logs.",
        ]
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.results_root = args.results_root.resolve()
    validate_inputs(args)

    config_dir = args.results_root / "configs" / args.method
    config_path = write_openvins_configs(config_dir, args)
    specs = list(make_specs(args, config_path))
    if args.dry_run:
        for spec in specs:
            print(
                f"{spec.sequence}: bag={spec.bag} raw_imu={spec.raw_imu_csv} "
                f"gt={spec.gt_tum} out={spec.out_dir}"
            )
        print(f"config={config_path}")
        return 0

    env_prefix = make_env_prefix(args)
    roscore = None if args.eval_only else ensure_ros_master(args, env_prefix)
    rows: list[dict[str, object]] = []
    try:
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] {spec.method}/{spec.sequence}", flush=True)
            if args.eval_only:
                row = {
                    "method": spec.method,
                    "sequence": spec.sequence,
                    "status": "existing",
                    "returncode": 0,
                    "estimate_rows": count_data_rows(spec.out_dir / "estimate.txt"),
                    "odom_rows": count_data_rows(spec.out_dir / "odomimu.txt"),
                    "runtime_s": 0.0,
                    "out_dir": str(spec.out_dir),
                }
            else:
                row = run_openvins(spec, args, env_prefix)
            if row["status"] in {"ok", "skipped", "existing"}:
                row.update(evaluate_run(spec, args))
            else:
                row["metric_status"] = "not_run"
            rows.append(row)
            write_summaries(args, rows)
            print(
                f"  status={row.get('status')} metric={row.get('metric_status')} "
                f"odom={row.get('odom_rows')} pos_rmse={row.get('ate_rmse_pos_m', 'n/a')} "
                f"rot_rmse={row.get('ate_rmse_rot_deg', 'n/a')}",
                flush=True,
            )
    finally:
        stop_process(roscore)

    write_summaries(args, rows)
    failed = [row for row in rows if row.get("status") == "failed" or row.get("metric_status") in {"failed", "not_run"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
