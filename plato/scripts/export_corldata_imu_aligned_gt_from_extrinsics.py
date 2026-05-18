#!/usr/bin/env python3
"""Export corldata IMU-aligned GT using recorded Body_0-to-IMU extrinsics.

This script applies each sequence's recorded `T_Body0_IMU` to mocap
`Body_0.tum`, resamples the resulting IMU/body GT to a fixed-rate grid, aligns
the world gravity direction to -Z, and writes a Debias_IMU-style tree.

It creates a derived diagnostic dataset and does not overwrite
`corldata_gt_body_1000hz`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

import export_corldata_debias_body_gt as base


PLATO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path("/home/amao/datasets/corldata")
DEFAULT_PSEUDO_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "corldata"
DEFAULT_EXTRINSICS_ROOT = DEFAULT_PSEUDO_ROOT / "body0_imu_extrinsics_vo_ref"
DEFAULT_AGGREGATE_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_1000hz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--extrinsics-root", type=Path, default=DEFAULT_EXTRINSICS_ROOT)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--sequences", nargs="+", help="Defaults to sequences present in --extrinsics-root.")
    parser.add_argument("--output-rate-hz", type=float, default=1000.0)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--gravity-magnitude", type=float, default=9.81)
    parser.add_argument("--max-gt-gap-s", type=float, default=0.03)
    parser.add_argument("--max-imu-gap-s", type=float, default=0.02)
    parser.add_argument("--gap-pad-s", type=float, default=0.01)
    parser.add_argument("--max-speed-m-s", type=float, default=3.0)
    parser.add_argument("--max-accel-m-s2", type=float, default=80.0)
    parser.add_argument("--max-angular-speed-rad-s", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sequence_sort_key(path: Path) -> tuple[str, int | str]:
    name = path.name
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(digits) if digits else name


def discover_sequences(extrinsics_root: Path) -> list[str]:
    sequences = [
        path.name
        for path in sorted(extrinsics_root.glob("S*"), key=sequence_sort_key)
        if (path / "body0_imu_extrinsic_vo_ref.json").exists()
    ]
    if not sequences:
        raise FileNotFoundError(f"No per-sequence extrinsic records found under {extrinsics_root}")
    return sequences


def load_extrinsic(extrinsics_root: Path, sequence: str, pseudo_root: Path) -> tuple[np.ndarray, dict[str, Any], Path]:
    candidates = [
        extrinsics_root / sequence / "body0_imu_extrinsic_vo_ref.json",
        pseudo_root / sequence / "alignment_config" / "body0_imu_extrinsic_vo_ref.json",
    ]
    for path in candidates:
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            T = np.asarray(record["T_Body0_IMU"], dtype=float)
            if T.shape != (4, 4):
                raise ValueError(f"{path}: T_Body0_IMU shape is {T.shape}, expected (4, 4)")
            return T, record, path
    raise FileNotFoundError(f"{sequence}: no body0_imu_extrinsic_vo_ref.json found")


def apply_imu_alignment(
    gt_interp: base.PoseSeries,
    imu_interp: np.ndarray,
    T_body0_imu: np.ndarray,
    gravity_magnitude: float,
) -> tuple[np.ndarray, Rotation, np.ndarray, np.ndarray]:
    R_w_body0 = gt_interp.rot.as_matrix()
    p_w_body0 = gt_interp.p
    R_body0_imu = T_body0_imu[:3, :3]
    p_body0_imu = T_body0_imu[:3, 3]

    R_w_imu = R_w_body0 @ R_body0_imu
    p_w_imu = p_w_body0 + np.einsum("nij,j->ni", R_w_body0, p_body0_imu)

    g_source = base.estimate_gravity_world(gt_interp.t, p_w_imu, R_w_imu, imu_interp[:, 3:6])
    target_g = np.array([0.0, 0.0, -float(gravity_magnitude)], dtype=np.float64)
    R_gravity = base.rotation_align_vector(g_source, target_g)
    R_wg_imu = R_gravity @ R_w_imu
    p_wg_imu = (R_gravity @ p_w_imu.T).T
    p_wg_imu = p_wg_imu - p_wg_imu[0]
    return p_wg_imu, Rotation.from_matrix(R_wg_imu), R_gravity, g_source


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "selected_sequences.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['sequence']}\n")
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# corldata IMU-aligned-gt Export",
        "",
        f"Aggregate root: `{root}`",
        "",
        "| sequence | rows | duration s | p_Body0_IMU norm m | valid windows | dropped windows | gravity before align |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {seq} | {rows} | {dur:.3f} | {lever:.4f} | {valid} | {dropped} | ({gx:.3f}, {gy:.3f}, {gz:.3f}) |".format(
                seq=row["sequence"],
                rows=int(row["rows"]),
                dur=float(row["duration_s"]),
                lever=float(row["p_Body0_IMU_norm_m"]),
                valid=int(row["valid_window_count"]),
                dropped=int(row["dropped_window_count"]),
                gx=float(row["gravity_before_alignment_x"]),
                gy=float(row["gravity_before_alignment_y"]),
                gz=float(row["gravity_before_alignment_z"]),
            )
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    gt_path = args.dataset_root / "0511_data" / "gt" / sequence / "Body_0.tum"
    imu_path = args.pseudo_root / sequence / "sequence" / "imu.txt"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not imu_path.exists():
        raise FileNotFoundError(imu_path)

    T_body0_imu, extrinsic_record, extrinsic_path = load_extrinsic(args.extrinsics_root, sequence, args.pseudo_root)
    gt = base.load_tum_xyzw(gt_path)
    imu_t, imu_u = base.load_imu_txt(imu_path)
    overlap_start = max(float(gt.t[0]), float(imu_t[0]))
    overlap_end = min(float(gt.t[-1]), float(imu_t[-1]))
    query_t, query_ns, dt_ns = base.make_query_grid(overlap_start, overlap_end, args.output_rate_hz)
    gt_interp = base.interpolate_gt(gt, query_t)
    imu_interp = base.interp_columns(imu_t, imu_u, query_t)
    p_imu, rot_imu, R_gravity, g_source = apply_imu_alignment(
        gt_interp,
        imu_interp,
        T_body0_imu,
        args.gravity_magnitude,
    )

    valid_starts, quality_stats, reason_masks = base.compute_quality_windows(
        query_t,
        p_imu,
        rot_imu,
        imu_interp,
        gt.t,
        imu_t,
        args,
    )

    per_seq_dir = args.pseudo_root / sequence / "IMU-aligned-gt"
    aggregate_mav0 = args.aggregate_root / sequence / "mav0"
    if args.overwrite:
        for path in (per_seq_dir, args.aggregate_root / sequence):
            if path.exists():
                shutil.rmtree(path)
    per_seq_dir.mkdir(parents=True, exist_ok=True)
    aggregate_mav0.mkdir(parents=True, exist_ok=True)

    for root in (aggregate_mav0, per_seq_dir / "debias_tumvi_source" / "mav0"):
        base.write_imu_csv(root / "imu0" / "data.csv", query_ns, imu_interp)
        base.write_mocap_csv(root / "mocap0" / "data.csv", query_ns, p_imu, rot_imu)
        quality_dir = root / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(quality_dir / f"valid_window_starts_{args.window_size}.txt", valid_starts, fmt="%d")
        (quality_dir / "window_quality.json").write_text(
            json.dumps(quality_stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    base.write_tum(per_seq_dir / "IMU-aligned-gt.tum", query_t, p_imu, rot_imu)
    base.write_mocap_csv(per_seq_dir / "IMU-aligned-gt.csv", query_ns, p_imu, rot_imu)
    base.write_sanity_plots(per_seq_dir, sequence, query_t, p_imu, reason_masks)

    metadata = {
        "dataset": "corldata",
        "sequence": sequence,
        "trajectory_name": "IMU-aligned-gt",
        "source_gt": str(gt_path),
        "source_imu": str(imu_path),
        "source_extrinsic_record": str(extrinsic_path),
        "target_supervision": "T_W_I, IMU/body pose in gravity-aligned world",
        "raw_gt_frame": "T_W_Body0 from mocap Body_0.tum",
        "alignment_model": "T_W_IMU = T_W_Body0 * T_Body0_IMU",
        "T_Body0_IMU": T_body0_imu.tolist(),
        "p_Body0_IMU_m": T_body0_imu[:3, 3].tolist(),
        "p_Body0_IMU_norm_m": float(np.linalg.norm(T_body0_imu[:3, 3])),
        "extrinsic_warning": extrinsic_record.get("warning"),
        "gravity_alignment": {
            "route": "rotate world so median(a_world - R_WI * acc_imu) aligns to default -Z gravity",
            "gravity_before_alignment_m_s2": g_source.tolist(),
            "gravity_target_world_m_s2": [0.0, 0.0, -float(args.gravity_magnitude)],
            "R_gravity_alignment": R_gravity.tolist(),
        },
        "timestamp_policy": {
            "output_rate_hz": float(args.output_rate_hz),
            "dt_ns": int(dt_ns),
            "overlap_start_s": float(overlap_start),
            "overlap_end_s": float(overlap_end),
        },
        "quality": quality_stats,
        "outputs": {
            "per_sequence_dir": str(per_seq_dir),
            "aggregate_sequence_dir": str(args.aggregate_root / sequence),
            "tum": str(per_seq_dir / "IMU-aligned-gt.tum"),
            "csv": str(per_seq_dir / "IMU-aligned-gt.csv"),
        },
    }
    (per_seq_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "sequence": sequence,
        "rows": int(len(query_t)),
        "duration_s": float(query_t[-1] - query_t[0]),
        "output_rate_hz": float(args.output_rate_hz),
        "p_Body0_IMU_norm_m": float(np.linalg.norm(T_body0_imu[:3, 3])),
        "raw_window_count": int(quality_stats["raw_window_count"]),
        "valid_window_count": int(quality_stats["valid_window_count"]),
        "dropped_window_count": int(quality_stats["dropped_window_count"]),
        "gravity_before_alignment_x": float(g_source[0]),
        "gravity_before_alignment_y": float(g_source[1]),
        "gravity_before_alignment_z": float(g_source[2]),
        "per_sequence_dir": str(per_seq_dir),
        "aggregate_sequence_dir": str(args.aggregate_root / sequence),
    }


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.pseudo_root = args.pseudo_root.resolve()
    args.extrinsics_root = args.extrinsics_root.resolve()
    args.aggregate_root = args.aggregate_root.resolve()
    sequences = args.sequences or discover_sequences(args.extrinsics_root)
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        print(f"[IMU-aligned-gt] {sequence}", flush=True)
        rows.append(export_sequence(args, sequence))
    write_summary(args.aggregate_root, rows)
    print(json.dumps({"aggregate_root": str(args.aggregate_root), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
