#!/usr/bin/env python3
"""Export LIASuw mocap-derived IMU/body GT for Debias_IMU.

Frame convention:

    T_AB maps points from frame B into frame A.

For each selected PLATO LIASuw candidate:

    1. Read ORB-SLAM3 stereo-only VO as T_Wv_C, the left-camera pose.
    2. Read the session camera-IMU calibration and convert to T_Wv_I:
           T_Wv_I = T_Wv_C * T_C_I
       If the calibration source stores T_I_C, it is inverted first.
    3. Estimate the fixed mocap-marker to IMU/body transform using:
           T_Wm_M(t) * T_M_I ~= T_Wm_Wv * T_Wv_I(t)
       This is delegated to tools/estimate_mocap_body_gt.py, whose output
       model is T_WM * T_MB ~= T_WO * T_OB_vio.
    4. Export body GT at a fixed rate, by default 200 Hz, and resample the
       original LIASuw IMU to the exact same timestamp grid.

The default aggregate output follows the TUM-VI directory layout expected by
Debias_IMU's TUMDataset:

    results/debias_imu/LIASuw_training_candidates_200hz/<sequence>/mav0/imu0/data.csv
    results/debias_imu/LIASuw_training_candidates_200hz/<sequence>/mav0/mocap0/data.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[2]
COMMON_PATH = Path(__file__).resolve().with_name("0203_pipeline_common.py")
TOOLS_ESTIMATE = PLATO_ROOT / "tools" / "estimate_mocap_body_gt.py"
PSEUDO_GT_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "LIASuw"
DEFAULT_AGGREGATE_ROOT = PLATO_ROOT / "results" / "debias_imu" / "LIASuw_training_candidates_200hz"
DATASET_ROOT = Path("/home/amao/datasets/LIASuw")
EXTRINSIC_0912 = DATASET_ROOT / "LIASuw_Training" / "0912event" / "Calibration files" / "cam-imu.yaml"
EXTRINSIC_0203_0205 = DATASET_ROOT / "LIASuw_Training" / "esio.yaml"


spec = importlib.util.spec_from_file_location("plato_0203_pipeline_common", COMMON_PATH)
common = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = common
assert spec.loader is not None
spec.loader.exec_module(common)


@dataclass(frozen=True)
class Candidate:
    sequence: str
    status: str
    mocap_csv: Path
    extrinsic_yaml: Path
    association_time_offset_s: float
    note: str

    @property
    def estimate_tool_time_offset_s(self) -> float:
        # Existing PLATO evaluations use t_mocap = t_vo + offset. The body-GT
        # estimator applies t_mocap_corrected = t_mocap + offset, so the sign
        # is reversed when moving mocap timestamps into VO/IMU time.
        return -self.association_time_offset_s


CANDIDATES: Dict[str, Candidate] = {
    "0912s06_80": Candidate(
        sequence="0912s06_80",
        status="usable",
        mocap_csv=DATASET_ROOT / "LIASuw_Training" / "0912event" / "data" / "GT" / "06" / "mc.csv",
        extrinsic_yaml=EXTRINSIC_0912,
        association_time_offset_s=-1.505,
        note="User-confirmed first-80-second 0912 S06 stereo-only window.",
    ),
    "0912s07_30": Candidate(
        sequence="0912s07_30",
        status="usable",
        mocap_csv=DATASET_ROOT / "LIASuw_Training" / "0912event" / "data" / "GT" / "07" / "mc.csv",
        extrinsic_yaml=EXTRINSIC_0912,
        association_time_offset_s=-1.51,
        note="User-confirmed first-30-second 0912 S07 stereo-only local window.",
    ),
    "0912s10_40": Candidate(
        sequence="0912s10_40",
        status="usable",
        mocap_csv=DATASET_ROOT / "LIASuw_Training" / "0912event" / "data" / "GT" / "10" / "mc.csv",
        extrinsic_yaml=EXTRINSIC_0912,
        association_time_offset_s=-1.56,
        note="User-confirmed first-40-second 0912 S10 stereo-only local window.",
    ),
    "0203s18_80": Candidate(
        sequence="0203s18_80",
        status="usable_candidate",
        mocap_csv=DATASET_ROOT / "LIASuw_Training" / "0203" / "GT" / "18" / "mc.csv",
        extrinsic_yaml=EXTRINSIC_0203_0205,
        association_time_offset_s=2.2,
        note="Usable-quality 0203 S18 candidate from current pass.",
    ),
    "0203s19_80": Candidate(
        sequence="0203s19_80",
        status="usable_candidate",
        mocap_csv=DATASET_ROOT / "LIASuw_Training" / "0203" / "GT" / "19" / "mc.csv",
        extrinsic_yaml=EXTRINSIC_0203_0205,
        association_time_offset_s=2.2,
        note="Usable-quality 0203 S19 candidate from current pass.",
    ),
    "0203s22_80": Candidate(
        sequence="0203s22_80",
        status="usable_candidate",
        mocap_csv=DATASET_ROOT / "LIASuw_Training" / "0203" / "GT" / "22" / "mc.csv",
        extrinsic_yaml=EXTRINSIC_0203_0205,
        association_time_offset_s=2.2,
        note="Usable-quality 0203 S22 candidate from current pass.",
    ),
    "0205s14_80": Candidate(
        sequence="0205s14_80",
        status="usable_candidate",
        mocap_csv=DATASET_ROOT / "LIASuw_Training" / "0205" / "0205gt" / "GT" / "14" / "mc.csv",
        extrinsic_yaml=EXTRINSIC_0203_0205,
        association_time_offset_s=0.34,
        note="Usable-quality 0205 S14 candidate from current pass.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        nargs="*",
        default=["all"],
        help="Candidate sequence names. Default: all current PLATO usable/candidate windows.",
    )
    parser.add_argument("--output-rate-hz", type=float, default=200.0)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--time-offset-search-radius", type=float, default=0.05)
    parser.add_argument("--time-offset-search-step", type=float, default=0.005)
    parser.add_argument("--max-calib-samples", type=int, default=2000)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--grid-max-nfev", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-estimation", action="store_true", help="Reuse existing 200 Hz GT files and only refresh dataset trees/summaries.")
    return parser.parse_args()


def selected_candidates(names: Sequence[str]) -> List[Candidate]:
    expanded: List[str] = []
    for value in names:
        expanded.extend(item.strip() for item in value.split(",") if item.strip())
    if not expanded or "all" in expanded:
        return list(CANDIDATES.values())
    missing = [name for name in expanded if name not in CANDIDATES]
    if missing:
        raise ValueError(f"Unknown candidate(s): {', '.join(missing)}")
    return [CANDIDATES[name] for name in expanded]


def ensure_inputs(candidate: Candidate) -> None:
    result_dir = PSEUDO_GT_ROOT / candidate.sequence
    paths = [
        result_dir / "raw" / "estimate_f_seconds.tum",
        result_dir / "sequence" / "imu.txt",
        candidate.mocap_csv,
        candidate.extrinsic_yaml,
        TOOLS_ESTIMATE,
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)


def convert_vo_camera_to_imu(candidate: Candidate, body_dir: Path, overwrite: bool) -> Dict[str, object]:
    vo_camera_path = PSEUDO_GT_ROOT / candidate.sequence / "raw" / "estimate_f_seconds.tum"
    vo_imu_path = body_dir / "vo_imu_traj.tum"
    metadata_path = body_dir / "vo_imu_metadata.json"
    if vo_imu_path.exists() and metadata_path.exists() and not overwrite:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    body_dir.mkdir(parents=True, exist_ok=True)
    vo_camera = common.read_pose_trajectory(
        vo_camera_path,
        "vo_T_Wv_C",
        timestamp_unit="s",
        quat_order="xyzw",
        position_unit="m",
    )
    extrinsic = common.load_camera_imu_extrinsic(candidate.extrinsic_yaml)
    vo_imu = common.transform_by_right(vo_camera, extrinsic.T_C_I, "vo_T_Wv_I")
    common.write_tum_xyzw(vo_imu_path, vo_imu)
    common.write_matrix_yaml(
        body_dir / "T_C_I_used.yaml",
        "T_C_I",
        extrinsic.T_C_I,
        {
            "source_file": str(extrinsic.path),
            "source_key": extrinsic.source_key,
            "source_convention": extrinsic.convention,
            "note": extrinsic.note,
            "formula": "T_Wv_I = T_Wv_C * T_C_I",
        },
    )
    metadata = {
        "sequence": candidate.sequence,
        "vo_camera_path": str(vo_camera_path),
        "vo_imu_path": str(vo_imu_path),
        "extrinsic_path": str(candidate.extrinsic_yaml),
        "extrinsic_source_key": extrinsic.source_key,
        "extrinsic_source_convention": extrinsic.convention,
        "used_formula": "T_Wv_I = T_Wv_C * T_C_I; if source was T_I_C, T_C_I = inv(T_I_C).",
        "num_vo_poses": int(len(vo_imu)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_body_gt_estimator(candidate: Candidate, body_dir: Path, args: argparse.Namespace) -> None:
    if args.skip_estimation:
        return
    if args.overwrite and body_dir.exists():
        for relative in [
            "body_gt_world.csv",
            "body_gt_firstframe.csv",
            "calibration_result.yaml",
            "debias_imu/data.csv",
            "debias_imu_firstframe/data.csv",
        ]:
            path = body_dir / relative
            if path.exists():
                path.unlink()

    cmd: List[object] = [
        sys.executable,
        TOOLS_ESTIMATE,
        "--mocap_csv",
        candidate.mocap_csv,
        "--vio_csv",
        body_dir / "vo_imu_traj.tum",
        "--imu_csv",
        PSEUDO_GT_ROOT / candidate.sequence / "sequence" / "imu.txt",
        "--output_dir",
        body_dir,
        "--timestamp_unit",
        "s",
        "--imu_timestamp_unit",
        "ns",
        "--mocap_quat_order",
        "xyzw",
        "--vio_quat_order",
        "xyzw",
        "--time_offset",
        candidate.estimate_tool_time_offset_s,
        "--estimate_time_offset",
        "--time_offset_search_min",
        -args.time_offset_search_radius,
        "--time_offset_search_max",
        args.time_offset_search_radius,
        "--time_offset_search_step",
        args.time_offset_search_step,
        "--output_timestamps",
        "rate",
        "--output_rate_hz",
        args.output_rate_hz,
        "--write_debias_imu_csv",
        "--normalize_first_frame",
        "--max_calib_samples",
        args.max_calib_samples,
        "--min_samples",
        args.min_samples,
        "--max_nfev",
        args.max_nfev,
        "--time_offset_grid_max_nfev",
        args.grid_max_nfev,
    ]
    log_path = body_dir / "estimate_mocap_body_gt.log"
    result = subprocess.run(
        [str(item) for item in cmd],
        cwd=str(PLATO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Body-GT estimation failed for {candidate.sequence}; see {log_path}")
    (body_dir / "generation_command.txt").write_text(" ".join(str(item) for item in cmd) + "\n", encoding="utf-8")


def read_numeric_csv(path: Path, skip_header: bool = False) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1 if skip_header else 0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        raise ValueError(f"No data rows in {path}")
    return data


def read_debias_gt_timestamps_ns(path: Path) -> np.ndarray:
    rows = read_numeric_csv(path, skip_header=True)
    return rows[:, 0].astype(np.int64)


def write_resampled_imu_csv(source_imu_path: Path, query_t_ns: np.ndarray, output_path: Path) -> Dict[str, float]:
    imu = read_numeric_csv(source_imu_path, skip_header=False)
    if imu.shape[1] < 7:
        raise ValueError(f"IMU file must have at least 7 columns: {source_imu_path}")
    source_t_s = imu[:, 0].astype(np.float64) * 1e-9
    query_t_s = query_t_ns.astype(np.float64) * 1e-9
    if query_t_s[0] < source_t_s[0] - 1e-9 or query_t_s[-1] > source_t_s[-1] + 1e-9:
        raise ValueError(
            f"Requested IMU interpolation outside source range for {source_imu_path}: "
            f"query [{query_t_s[0]:.9f}, {query_t_s[-1]:.9f}], "
            f"source [{source_t_s[0]:.9f}, {source_t_s[-1]:.9f}]"
        )
    resampled = np.empty((query_t_ns.shape[0], 7), dtype=np.float64)
    resampled[:, 0] = query_t_ns
    for col in range(1, 7):
        resampled[:, col] = np.interp(query_t_s, source_t_s, imu[:, col])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
                "linear_acceleration_x",
                "linear_acceleration_y",
                "linear_acceleration_z",
            ]
        )
        for row in resampled:
            writer.writerow(
                [
                    str(int(round(row[0]))),
                    f"{row[1]:.12f}",
                    f"{row[2]:.12f}",
                    f"{row[3]:.12f}",
                    f"{row[4]:.12f}",
                    f"{row[5]:.12f}",
                    f"{row[6]:.12f}",
                ]
            )

    dt = np.diff(query_t_s)
    return {
        "samples": int(query_t_ns.shape[0]),
        "median_hz": float(1.0 / np.median(dt)) if len(dt) else 0.0,
        "mean_hz": float(1.0 / np.mean(dt)) if len(dt) else 0.0,
        "first_timestamp_ns": int(query_t_ns[0]),
        "last_timestamp_ns": int(query_t_ns[-1]),
    }


def copy_text_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_debias_dataset_tree(candidate: Candidate, body_dir: Path, aggregate_root: Path, firstframe: bool) -> Dict[str, object]:
    source_gt = body_dir / ("debias_imu_firstframe" if firstframe else "debias_imu") / "data.csv"
    if not source_gt.exists():
        raise FileNotFoundError(source_gt)
    suffix = "firstframe" if firstframe else "world"
    per_sequence_root = body_dir / f"debias_tumvi_{suffix}" / "mav0"
    aggregate_sequence_root = aggregate_root / (f"{candidate.sequence}_{suffix}" if firstframe else candidate.sequence) / "mav0"
    query_t_ns = read_debias_gt_timestamps_ns(source_gt)
    source_imu = PSEUDO_GT_ROOT / candidate.sequence / "sequence" / "imu.txt"
    outputs: Dict[str, object] = {}
    for root in [per_sequence_root, aggregate_sequence_root]:
        imu_out = root / "imu0" / "data.csv"
        mocap_out = root / "mocap0" / "data.csv"
        imu_stats = write_resampled_imu_csv(source_imu, query_t_ns, imu_out)
        copy_text_file(source_gt, mocap_out)
        outputs[str(root.parent)] = {
            "imu_csv": str(imu_out),
            "mocap_csv": str(mocap_out),
            "imu_stats": imu_stats,
        }
    return outputs


def parse_yaml_scalar(text: str, key: str) -> Optional[float]:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([-+0-9.eE]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_calibration_summary(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    return {
        "time_offset_s": parse_yaml_scalar(text, "time_offset"),
        "translation_rmse_m": parse_yaml_scalar(text, "translation_rmse_m"),
        "rotation_rmse_deg": parse_yaml_scalar(text, "rotation_rmse_deg"),
        "translation_max_m": parse_yaml_scalar(text, "translation_max_m"),
        "rotation_max_deg": parse_yaml_scalar(text, "rotation_max_deg"),
        "number_of_samples": parse_yaml_scalar(text, "number_of_samples"),
        "first_timestamp": parse_yaml_scalar(text, "first_timestamp"),
        "last_timestamp": parse_yaml_scalar(text, "last_timestamp"),
    }


def write_sequence_readme(
    candidate: Candidate,
    body_dir: Path,
    metadata: Dict[str, object],
    summary: Dict[str, object],
    dataset_outputs: Dict[str, object],
    output_rate_hz: float,
) -> None:
    lines = [
        f"# {candidate.sequence} Debias_IMU Body GT",
        "",
        "This directory is generated from LIASuw mocap GT and the accepted stereo-only VO result.",
        "",
        "## Frame Conversion",
        "",
        "```text",
        "T_Wv_C: ORB-SLAM3 stereo-only left-camera pose in VO world",
        "T_C_I: IMU/body frame to left-camera frame from camera-IMU calibration",
        "T_Wv_I = T_Wv_C * T_C_I",
        "T_Wm_M * T_M_I ~= T_Wm_Wv * T_Wv_I",
        "T_Wm_I_gt = T_Wm_M * T_M_I",
        "```",
        "",
        "The exported default GT is in mocap/world coordinates with the pose frame converted to IMU/body.",
        "The first-frame-normalized copy is provided only for experiments that also rotate gravity consistently.",
        "",
        "## Sources",
        "",
        f"- Candidate status: `{candidate.status}`",
        f"- Mocap marker pose: `{candidate.mocap_csv}`",
        f"- VO camera trajectory: `{metadata.get('vo_camera_path')}`",
        f"- VO-to-IMU trajectory: `{metadata.get('vo_imu_path')}`",
        f"- Camera-IMU extrinsic: `{metadata.get('extrinsic_path')}`",
        f"- Extrinsic source key: `{metadata.get('extrinsic_source_key')}`",
        f"- Extrinsic source convention: `{metadata.get('extrinsic_source_convention')}`",
        f"- Formula: `{metadata.get('used_formula')}`",
        f"- PLATO association offset used as initial guess: `{candidate.association_time_offset_s:.6f} s`, convention `t_mocap = t_vo + offset`.",
        f"- Body-GT estimator initial offset: `{candidate.estimate_tool_time_offset_s:.6f} s`, convention `t_mocap_corrected = t_mocap + offset`.",
        "",
        "## Outputs",
        "",
        f"- `body_gt_world.csv`: seconds timestamp, pose `T_Wm_I_gt` in mocap/world frame.",
        f"- `debias_imu/data.csv`: Debias_IMU/TUM-VI-style 200 Hz mocap0 GT.",
        f"- `debias_tumvi_world/mav0/imu0/data.csv`: original LIASuw IMU resampled to the exact GT timestamps.",
        f"- `debias_tumvi_world/mav0/mocap0/data.csv`: same as `debias_imu/data.csv`.",
        f"- `debias_imu_firstframe/data.csv` and `debias_tumvi_firstframe/`: first-frame-normalized variant.",
        f"- `calibration_result.yaml`: estimated `T_M_I`/`T_MB`, `T_Wm_Wv`/`T_WO`, residuals, and gravity reminder.",
        "",
        "## Summary",
        "",
        f"- Requested output rate: `{output_rate_hz:.3f} Hz`",
        f"- Estimated body-GT time offset: `{summary.get('time_offset_s')}` s",
        f"- Calibration translation RMSE: `{summary.get('translation_rmse_m')}` m",
        f"- Calibration rotation RMSE: `{summary.get('rotation_rmse_deg')}` deg",
        f"- Output samples: `{summary.get('number_of_samples')}`",
        "",
        "## Debias_IMU Dataset Trees",
        "",
    ]
    for root, payload in dataset_outputs.items():
        lines.append(f"- `{root}`")
        lines.append(f"  - IMU: `{payload['imu_csv']}`")
        lines.append(f"  - GT: `{payload['mocap_csv']}`")
    body_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_files(rows: Sequence[Dict[str, object]], aggregate_root: Path) -> None:
    aggregate_root.mkdir(parents=True, exist_ok=True)
    csv_path = aggregate_root / "summary.csv"
    md_path = aggregate_root / "summary.md"
    fields = [
        "sequence",
        "status",
        "association_time_offset_s",
        "estimator_time_offset_s",
        "translation_rmse_m",
        "rotation_rmse_deg",
        "translation_max_m",
        "rotation_max_deg",
        "samples",
        "dataset_sequence_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# LIASuw Debias_IMU 200 Hz Body GT Summary",
        "",
        f"Aggregate dataset root: `{aggregate_root}`",
        "",
        "| sequence | status | estimator offset s | trans RMSE m | rot RMSE deg | samples | dataset dir |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {sequence} | {status} | {offset} | {trans} | {rot} | {samples} | `{path}` |".format(
                sequence=row.get("sequence", ""),
                status=row.get("status", ""),
                offset=format_optional(row.get("estimator_time_offset_s")),
                trans=format_optional(row.get("translation_rmse_m")),
                rot=format_optional(row.get("rotation_rmse_deg")),
                samples=int(row.get("samples") or 0),
                path=row.get("dataset_sequence_dir", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: object) -> str:
    try:
        if value is None or value == "":
            return ""
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def export_candidate(candidate: Candidate, args: argparse.Namespace) -> Dict[str, object]:
    ensure_inputs(candidate)
    body_dir = PSEUDO_GT_ROOT / candidate.sequence / f"debias_imu_body_gt_{int(args.output_rate_hz)}hz"
    body_dir.mkdir(parents=True, exist_ok=True)
    metadata = convert_vo_camera_to_imu(candidate, body_dir, args.overwrite)
    run_body_gt_estimator(candidate, body_dir, args)

    calibration_path = body_dir / "calibration_result.yaml"
    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)
    summary = parse_calibration_summary(calibration_path)
    world_outputs = build_debias_dataset_tree(candidate, body_dir, args.aggregate_root, firstframe=False)
    first_outputs = build_debias_dataset_tree(candidate, body_dir, args.aggregate_root, firstframe=True)
    all_outputs = {**world_outputs, **first_outputs}
    write_sequence_readme(candidate, body_dir, metadata, summary, all_outputs, args.output_rate_hz)

    dataset_sequence_dir = args.aggregate_root / candidate.sequence
    return {
        "sequence": candidate.sequence,
        "status": candidate.status,
        "association_time_offset_s": candidate.association_time_offset_s,
        "estimator_time_offset_s": summary.get("time_offset_s"),
        "translation_rmse_m": summary.get("translation_rmse_m"),
        "rotation_rmse_deg": summary.get("rotation_rmse_deg"),
        "translation_max_m": summary.get("translation_max_m"),
        "rotation_max_deg": summary.get("rotation_max_deg"),
        "samples": int(summary.get("number_of_samples") or 0),
        "body_gt_dir": str(body_dir),
        "dataset_sequence_dir": str(dataset_sequence_dir),
        "note": candidate.note,
    }


def main() -> None:
    args = parse_args()
    args.aggregate_root = Path(args.aggregate_root).expanduser().resolve()
    if args.output_rate_hz <= 0 or not math.isfinite(args.output_rate_hz):
        raise ValueError("--output-rate-hz must be a finite positive value")
    candidates = selected_candidates(args.sequence)
    rows = []
    for candidate in candidates:
        print(f"[export] {candidate.sequence}", flush=True)
        rows.append(export_candidate(candidate, args))
    write_summary_files(rows, args.aggregate_root)
    selected_path = args.aggregate_root / "selected_sequences.txt"
    selected_path.write_text("".join(f"{row['sequence']}\n" for row in rows), encoding="utf-8")
    print(json.dumps({"aggregate_root": str(args.aggregate_root), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
