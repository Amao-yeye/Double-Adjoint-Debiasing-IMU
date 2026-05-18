#!/usr/bin/env python3
"""Export gated residual IMU corrections for corldata OpenVINS trials.

This diagnostic correction is intentionally conservative:

* use the raw-repacked OpenVINS bias estimate near the end of the first
  light-on interval as the reference bias;
* apply only the learned residual bias relative to that reference;
* gate the correction to detected light-off intervals with a short ramp;
* clip the residual per axis before subtracting it from the raw IMU.

No ground-truth state is used to construct the correction.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
OPENVINS_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
    "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)


@dataclass(frozen=True)
class LightSegment:
    state: str
    start: float
    end: float


def read_openvins_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    values: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].lstrip().startswith("#") or row[0] == "timestamp_ns":
                continue
            timestamps.append(int(row[0]))
            values.append([float(v) for v in row[1:7]])
    if not timestamps:
        raise ValueError(f"No IMU rows found in {path}")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def read_bias_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    values: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(int(row["timestamp_ns"]))
            values.append(
                [
                    float(row["bg_x_rad_s"]),
                    float(row["bg_y_rad_s"]),
                    float(row["bg_z_rad_s"]),
                    float(row["ba_x_m_s2"]),
                    float(row["ba_y_m_s2"]),
                    float(row["ba_z_m_s2"]),
                ]
            )
    if not timestamps:
        raise ValueError(f"No bias rows found in {path}")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def read_openvins_bias_estimates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) >= 17:
                rows.append(vals[:17])
    if not rows:
        raise ValueError(f"No OpenVINS estimate rows found in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 11:17]


def openvins_bias_reference_at(
    estimate_t_s: np.ndarray,
    estimate_bias: np.ndarray,
    target_t_s: float,
    *,
    side: str = "left",
) -> tuple[int, float, np.ndarray]:
    if side == "left":
        idx = int(np.searchsorted(estimate_t_s, target_t_s, side="right") - 1)
    elif side == "nearest":
        idx = int(np.searchsorted(estimate_t_s, target_t_s, side="left"))
        if idx >= estimate_t_s.shape[0]:
            idx = estimate_t_s.shape[0] - 1
        if idx > 0 and abs(estimate_t_s[idx - 1] - target_t_s) <= abs(estimate_t_s[idx] - target_t_s):
            idx -= 1
    else:
        raise ValueError(f"Unsupported side: {side}")
    idx = min(max(idx, 0), estimate_t_s.shape[0] - 1)
    return idx, float(estimate_t_s[idx]), estimate_bias[idx].copy()


def read_first_light_on_bias_reference(
    path: Path,
    segments: list[LightSegment],
    estimate_t_s: np.ndarray,
    estimate_bias: np.ndarray,
) -> dict[str, Any]:

    first_on = next((seg for seg in segments if seg.state == "light_on"), None)
    if first_on is None:
        raise ValueError("No light_on segment available for bias reference")
    in_first_on = (estimate_t_s >= first_on.start) & (estimate_t_s <= first_on.end)
    if int(in_first_on.sum()) < 1:
        raise ValueError(f"No OpenVINS estimate row inside first light-on interval for {path}")

    candidate_indices = np.flatnonzero(in_first_on)
    idx = int(candidate_indices[-1])
    return {
        "source": str(path),
        "timestamp_s": float(estimate_t_s[idx]),
        "first_light_on_start_s": float(first_on.start),
        "first_light_on_end_s": float(first_on.end),
        "bg_ref_rad_s": estimate_bias[idx, :3].tolist(),
        "ba_ref_m_s2": estimate_bias[idx, 3:6].tolist(),
        "estimate_rows": int(estimate_t_s.shape[0]),
        "first_light_on_rows": int(in_first_on.sum()),
    }


def read_light_segments(path: Path, sequence: str) -> list[LightSegment]:
    segments: list[LightSegment] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sequence"] != sequence:
                continue
            segments.append(
                LightSegment(
                    state=row["state"],
                    start=float(row["start_gt_time_s"]),
                    end=float(row["end_gt_time_s"]),
                )
            )
    if not segments:
        raise ValueError(f"No light segments for {sequence} in {path}")
    return segments


def build_light_off_gate(
    times_s: np.ndarray,
    segments: list[LightSegment],
    ramp_s: float,
    tail_s: float = 0.0,
) -> np.ndarray:
    gate = np.zeros(times_s.shape, dtype=np.float64)
    ramp = max(float(ramp_s), 1.0e-6)
    tail = max(float(tail_s), 0.0)
    for seg in segments:
        if seg.state != "light_off":
            continue
        inside = (times_s >= seg.start) & (times_s <= seg.end)
        if np.any(inside):
            up = (times_s[inside] - seg.start) / ramp
            if tail > 0.0:
                gate[inside] = np.maximum(gate[inside], np.clip(np.minimum(up, 1.0), 0.0, 1.0))
            else:
                down = (seg.end - times_s[inside]) / ramp
                gate[inside] = np.maximum(
                    gate[inside], np.clip(np.minimum(np.minimum(up, down), 1.0), 0.0, 1.0)
                )
        if tail <= 0.0:
            continue
        tail_inside = (times_s > seg.end) & (times_s <= seg.end + tail)
        if np.any(tail_inside):
            down = (seg.end + tail - times_s[tail_inside]) / tail
            gate[tail_inside] = np.maximum(gate[tail_inside], np.clip(down, 0.0, 1.0))
    return gate


def build_duration_aware_gamma_g(
    times_s: np.ndarray,
    segments: list[LightSegment],
    short_gamma: float,
    long_gamma: float,
    threshold_s: float,
    tail_s: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    gamma_by_row = np.zeros(times_s.shape, dtype=np.float64)
    tail = max(float(tail_s), 0.0)
    records: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if seg.state != "light_off":
            continue
        duration_s = float(seg.end - seg.start)
        gamma = float(short_gamma if duration_s < threshold_s else long_gamma)
        inside = (times_s >= seg.start) & (times_s <= seg.end + tail)
        gamma_by_row[inside] = gamma
        records.append(
            {
                "segment_index": idx,
                "segment_start_s": seg.start,
                "segment_end_s": seg.end,
                "duration_s": duration_s,
                "threshold_s": float(threshold_s),
                "gamma_g": gamma,
                "tail_s": tail,
                "rows": int(inside.sum()),
            }
        )
    return gamma_by_row, records


def build_reference_bias_by_row(
    times_s: np.ndarray,
    segments: list[LightSegment],
    estimate_t_s: np.ndarray,
    estimate_bias: np.ndarray,
    first_reference: dict[str, Any],
    reference_mode: str,
    tail_s: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    first_ref = np.asarray(first_reference["bg_ref_rad_s"] + first_reference["ba_ref_m_s2"], dtype=np.float64)
    reference_by_row = np.tile(first_ref.reshape(1, 6), (times_s.shape[0], 1))
    records: list[dict[str, Any]] = []
    if reference_mode == "first_light_on":
        return reference_by_row, records
    if reference_mode != "per_light_off_start":
        raise ValueError(f"Unsupported reference mode: {reference_mode}")

    tail = max(float(tail_s), 0.0)
    for idx, seg in enumerate(segments):
        if seg.state != "light_off":
            continue
        est_idx, est_t, ref_bias = openvins_bias_reference_at(estimate_t_s, estimate_bias, seg.start, side="left")
        inside = (times_s >= seg.start) & (times_s <= seg.end + tail)
        reference_by_row[inside] = ref_bias.reshape(1, 6)
        records.append(
            {
                "segment_index": idx,
                "segment_start_s": seg.start,
                "segment_end_s": seg.end,
                "estimate_index": est_idx,
                "estimate_timestamp_s": est_t,
                "estimate_lag_s": float(seg.start - est_t),
                "bg_ref_rad_s": ref_bias[:3].tolist(),
                "ba_ref_m_s2": ref_bias[3:6].tolist(),
                "tail_s": tail,
                "rows": int(inside.sum()),
            }
        )
    return reference_by_row, records


def build_learned_delta_reference_by_row(
    times_s: np.ndarray,
    learned_bias: np.ndarray,
    segments: list[LightSegment],
    tail_s: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    reference_by_row = learned_bias.copy()
    tail = max(float(tail_s), 0.0)
    records: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if seg.state != "light_off":
            continue
        inside = (times_s >= seg.start) & (times_s <= seg.end + tail)
        if not np.any(inside):
            records.append(
                {
                    "segment_index": idx,
                    "segment_start_s": seg.start,
                    "segment_end_s": seg.end,
                    "rows": 0,
                    "note": "no raw IMU rows inside segment",
                }
            )
            continue
        start_idx = int(np.flatnonzero(inside)[0])
        start_bias = learned_bias[start_idx].copy()
        reference_by_row[inside] = start_bias.reshape(1, 6)
        records.append(
            {
                "segment_index": idx,
                "segment_start_s": seg.start,
                "segment_end_s": seg.end,
                "start_row_index": start_idx,
                "start_timestamp_s": float(times_s[start_idx]),
                "start_lag_s": float(times_s[start_idx] - seg.start),
                "start_bg_rad_s": start_bias[:3].tolist(),
                "start_ba_m_s2": start_bias[3:6].tolist(),
                "tail_s": tail,
                "rows": int(inside.sum()),
            }
        )
    return reference_by_row, records


def write_openvins_imu_csv(path: Path, timestamps_ns: np.ndarray, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([timestamps_ns.astype(object), values.astype(np.float64)])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=OPENVINS_IMU_HEADER,
        comments="",
        fmt=["%s"] + ["%.18e"] * 6,
    )


def method_token(prefix: str, value: float, scale: int = 1000, width: int = 3) -> str:
    return f"{prefix}{int(round(value * scale)):0{width}d}"


def summarize_delta(gate: np.ndarray, delta: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    active = gate > 0.0
    return {
        "active_rows": int(active.sum()),
        "max_gate": float(gate.max()) if gate.size else 0.0,
        "mean_gate_active": float(gate[active].mean()) if np.any(active) else 0.0,
        "delta_abs_max": np.max(np.abs(delta), axis=0).tolist(),
        "delta_rms_active": np.sqrt(np.mean(delta[active] * delta[active], axis=0)).tolist()
        if np.any(active)
        else [0.0] * 6,
        "learned_residual_abs_p95": np.percentile(np.abs(residual), 95, axis=0).tolist(),
        "learned_residual_abs_max": np.max(np.abs(residual), axis=0).tolist(),
    }


def export_sequence(args: argparse.Namespace, sequence: str) -> Path:
    input_seq = args.input_root / sequence
    raw_path = input_seq / "methods" / "raw" / "mav0" / "imu0" / "data.csv"
    bias_path = input_seq / "biases" / "network_bias_interpolated_to_raw_imu.csv"
    estimate_path = args.raw_openvins_run_root / sequence / "estimate.txt"
    for path in [raw_path, bias_path, estimate_path, args.light_segments]:
        if not path.exists():
            raise FileNotFoundError(path)

    raw_t_ns, raw_values = read_openvins_imu_csv(raw_path)
    bias_t_ns, learned_bias = read_bias_csv(bias_path)
    if not np.array_equal(raw_t_ns, bias_t_ns):
        if raw_t_ns.shape != bias_t_ns.shape:
            raise ValueError(f"Bias timestamp count does not match raw IMU timestamps for {sequence}")
        max_time_diff_ns = int(np.max(np.abs(raw_t_ns - bias_t_ns)))
        if max_time_diff_ns > 1000:
            raise ValueError(
                f"Bias timestamps differ from raw IMU timestamps by up to {max_time_diff_ns} ns for {sequence}"
            )
    else:
        max_time_diff_ns = 0

    segments = read_light_segments(args.light_segments, sequence)
    estimate_t_s, estimate_bias = read_openvins_bias_estimates(estimate_path)
    ref = read_first_light_on_bias_reference(estimate_path, segments, estimate_t_s, estimate_bias)
    t_s = raw_t_ns.astype(np.float64) * 1.0e-9
    gate = build_light_off_gate(t_s, segments, args.ramp_s, tail_s=args.gate_tail_s)
    if args.residual_mode == "openvins_reference":
        reference_by_row, interval_references = build_reference_bias_by_row(
            t_s,
            segments,
            estimate_t_s,
            estimate_bias,
            ref,
            args.reference_mode,
            tail_s=args.gate_tail_s,
        )
    elif args.residual_mode == "learned_delta":
        reference_by_row, interval_references = build_learned_delta_reference_by_row(
            t_s, learned_bias, segments, tail_s=args.gate_tail_s
        )
    else:
        raise ValueError(f"Unsupported residual mode: {args.residual_mode}")
    residual = learned_bias - reference_by_row

    output_seq = args.output_root / sequence
    method_summaries: dict[str, Any] = {}
    method_paths: dict[str, str] = {}
    clipped_bg = np.clip(residual[:, :3], -args.gyro_clip_rad_s, args.gyro_clip_rad_s)
    clipped_ba = np.clip(residual[:, 3:6], -args.accel_clip_m_s2, args.accel_clip_m_s2)
    if not args.skip_fixed_gamma:
        for gamma_g in args.gamma_g:
            for gamma_a in args.gamma_a:
                method = "_".join(
                    [
                        "gated",
                        method_token("g", gamma_g),
                        method_token("a", gamma_a),
                        method_token("cg", args.gyro_clip_rad_s, scale=1000, width=3),
                        method_token("ca", args.accel_clip_m_s2, scale=1000, width=3),
                        method_token("r", args.ramp_s, scale=10, width=2),
                    ]
                    + (
                        [method_token("tail", args.gate_tail_s, scale=10, width=2)]
                        if args.gate_tail_s > 0.0
                        else []
                    )
                )
                scaled = np.column_stack([gamma_g * clipped_bg, gamma_a * clipped_ba])
                delta = gate[:, None] * scaled
                corrected = raw_values - delta
                out_csv = output_seq / "methods" / method / "mav0" / "imu0" / "data.csv"
                write_openvins_imu_csv(out_csv, raw_t_ns, corrected)
                method_paths[method] = str(out_csv)
                method_summaries[method] = {
                    "gamma_g": float(gamma_g),
                    "gamma_a": float(gamma_a),
                    "gyro_clip_rad_s": float(args.gyro_clip_rad_s),
                    "accel_clip_m_s2": float(args.accel_clip_m_s2),
                    "ramp_s": float(args.ramp_s),
                    "gate_tail_s": float(args.gate_tail_s),
                    **summarize_delta(gate, delta, residual),
                }

    duration_gamma_records: list[dict[str, Any]] = []
    if args.duration_aware_gamma_g is not None:
        short_gamma, long_gamma = args.duration_aware_gamma_g
        gamma_g_by_row, duration_gamma_records = build_duration_aware_gamma_g(
            t_s,
            segments,
            short_gamma=float(short_gamma),
            long_gamma=float(long_gamma),
            threshold_s=float(args.duration_threshold_s),
            tail_s=args.gate_tail_s,
        )
        method = "_".join(
            [
                "gated",
                method_token("gdur_s", short_gamma),
                method_token("l", long_gamma),
                method_token("t", args.duration_threshold_s, scale=10, width=3),
                method_token("a", args.duration_aware_gamma_a),
                method_token("cg", args.gyro_clip_rad_s, scale=1000, width=3),
                method_token("ca", args.accel_clip_m_s2, scale=1000, width=3),
                method_token("r", args.ramp_s, scale=10, width=2),
            ]
            + (
                [method_token("tail", args.gate_tail_s, scale=10, width=2)]
                if args.gate_tail_s > 0.0
                else []
            )
        )
        scaled = np.column_stack([gamma_g_by_row[:, None] * clipped_bg, args.duration_aware_gamma_a * clipped_ba])
        delta = gate[:, None] * scaled
        corrected = raw_values - delta
        out_csv = output_seq / "methods" / method / "mav0" / "imu0" / "data.csv"
        write_openvins_imu_csv(out_csv, raw_t_ns, corrected)
        method_paths[method] = str(out_csv)
        method_summaries[method] = {
            "duration_aware_gamma_g": {
                "short_gamma": float(short_gamma),
                "long_gamma": float(long_gamma),
                "threshold_s": float(args.duration_threshold_s),
                "segment_records": duration_gamma_records,
            },
            "gamma_a": float(args.duration_aware_gamma_a),
            "gyro_clip_rad_s": float(args.gyro_clip_rad_s),
            "accel_clip_m_s2": float(args.accel_clip_m_s2),
            "ramp_s": float(args.ramp_s),
            "gate_tail_s": float(args.gate_tail_s),
            **summarize_delta(gate, delta, residual),
        }

    validation = {
        "sequence": sequence,
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "raw_imu": str(raw_path),
        "learned_bias": str(bias_path),
        "reference_mode": args.reference_mode,
        "residual_mode": args.residual_mode,
        "raw_openvins_first_light_on_bias_reference": ref,
        "interval_references": interval_references,
        "light_segments": [
            {"state": seg.state, "start_s": seg.start, "end_s": seg.end, "duration_s": seg.end - seg.start}
            for seg in segments
        ],
        "rows": int(raw_t_ns.shape[0]),
        "max_bias_csv_timestamp_diff_ns": int(max_time_diff_ns),
        "start_s": float(t_s[0]),
        "end_s": float(t_s[-1]),
        "gate_active_rows": int(np.sum(gate > 0.0)),
        "gate_tail_s": float(args.gate_tail_s),
        "duration_aware_gamma_records": duration_gamma_records,
        "methods": method_paths,
        "method_summaries": method_summaries,
    }
    output_seq.mkdir(parents=True, exist_ok=True)
    (output_seq / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return output_seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export gated residual corrected corldata IMU streams.")
    parser.add_argument("--sequence", action="append", dest="sequences", default=[])
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "corldata"
        / "natural"
        / "interval_nll_corrected_inputs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "corldata"
        / "natural"
        / "gated_residual_inputs",
    )
    parser.add_argument(
        "--raw-openvins-run-root",
        type=Path,
        default=PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "repacked_raw",
    )
    parser.add_argument(
        "--light-segments",
        type=Path,
        default=PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "light_segments" / "light_segments.csv",
    )
    parser.add_argument("--gamma-g", nargs="+", type=float, default=[0.0])
    parser.add_argument("--gamma-a", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument(
        "--skip-fixed-gamma",
        action="store_true",
        help="Only export the duration-aware method when --duration-aware-gamma-g is set.",
    )
    parser.add_argument(
        "--duration-aware-gamma-g",
        nargs=2,
        type=float,
        metavar=("SHORT", "LONG"),
        default=None,
        help="Export one method using SHORT gamma for light-off intervals shorter than the threshold, else LONG.",
    )
    parser.add_argument("--duration-aware-gamma-a", type=float, default=0.0)
    parser.add_argument("--duration-threshold-s", type=float, default=10.0)
    parser.add_argument("--gyro-clip-rad-s", type=float, default=0.003)
    parser.add_argument("--accel-clip-m-s2", type=float, default=0.05)
    parser.add_argument("--ramp-s", type=float, default=1.0)
    parser.add_argument(
        "--gate-tail-s",
        type=float,
        default=0.0,
        help=(
            "If positive, keep full correction through the light-off end and ramp it down over this many "
            "seconds after light returns. The default preserves the original symmetric in-dark ramp-down."
        ),
    )
    parser.add_argument(
        "--reference-mode",
        choices=["first_light_on", "per_light_off_start"],
        default="first_light_on",
        help="Bias reference used to convert learned bias into an OpenVINS residual correction.",
    )
    parser.add_argument(
        "--residual-mode",
        choices=["openvins_reference", "learned_delta"],
        default="openvins_reference",
        help=(
            "openvins_reference subtracts an OpenVINS bias reference from learned absolute bias; "
            "learned_delta uses only learned bias change from each light-off start."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = args.sequences or ["S13", "S15"]
    for sequence in sequences:
        output = export_sequence(args, sequence)
        print(f"Wrote {sequence}: {output}")


if __name__ == "__main__":
    main()
