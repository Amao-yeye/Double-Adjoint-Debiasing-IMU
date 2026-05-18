#!/usr/bin/env python3
"""Export corldata IMU-aligned GT at original mocap timestamps.

This is the sparse counterpart of
`plato/scripts/export_corldata_imu_aligned_gt_from_extrinsics.py`: it applies
the recorded per-sequence `T_Body0_IMU` and the same gravity alignment, but it
does not resample GT poses to the 1000 Hz IMU grid.  The resulting tree keeps
raw IMU samples under `mav0/imu0/data.csv` and sparse IMU-frame mocap poses
under `mav0/mocap0/data.csv`, so training can integrate IMU to GT endpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


PLATO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = PLATO_ROOT / "plato" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import export_corldata_debias_body_gt as base  # noqa: E402
from export_corldata_imu_aligned_gt_from_extrinsics import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_EXTRINSICS_ROOT,
    DEFAULT_PSEUDO_ROOT,
    apply_imu_alignment,
    discover_sequences,
    load_extrinsic,
)


DEFAULT_AGGREGATE_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_sparse_gt"
DEFAULT_DENSE_QUALITY_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_1000hz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--extrinsics-root", type=Path, default=DEFAULT_EXTRINSICS_ROOT)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--dense-quality-root", type=Path, default=DEFAULT_DENSE_QUALITY_ROOT)
    parser.add_argument("--sequences", nargs="+", help="Defaults to sequences present in --extrinsics-root.")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--gravity-rate-hz", type=float, default=1000.0)
    parser.add_argument("--gravity-magnitude", type=float, default=9.81)
    parser.add_argument("--min-gt-dt-s", type=float, default=0.003)
    parser.add_argument("--max-gt-gap-s", type=float, default=0.03)
    parser.add_argument("--max-imu-gap-s", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_imu_txt_with_ns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(f"{path} must have timestamp plus six IMU columns")
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    keep = np.concatenate(([True], np.diff(data[:, 0]) > 0.0))
    data = data[keep]
    ns = np.rint(data[:, 0]).astype(np.int64)
    t = ns.astype(np.float64) * 1e-9
    return ns, t, data[:, 1:7].astype(np.float64)


def transform_body0_to_imu(
    gt: base.PoseSeries,
    T_body0_imu: np.ndarray,
    R_gravity: np.ndarray,
    origin: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    R_w_body0 = gt.rot.as_matrix()
    p_w_body0 = gt.p
    R_body0_imu = T_body0_imu[:3, :3]
    p_body0_imu = T_body0_imu[:3, 3]
    R_w_imu = R_w_body0 @ R_body0_imu
    p_w_imu = p_w_body0 + np.einsum("nij,j->ni", R_w_body0, p_body0_imu)
    p_wg_imu = (R_gravity @ p_w_imu.T).T - origin.reshape(1, 3)
    R_wg_imu = R_gravity @ R_w_imu
    return p_wg_imu, Rotation.from_matrix(R_wg_imu)


def estimate_or_load_gravity(
    args: argparse.Namespace,
    sequence: str,
    gt: base.PoseSeries,
    imu_t: np.ndarray,
    imu_u: np.ndarray,
    T_body0_imu: np.ndarray,
    overlap_start: float,
    overlap_end: float,
) -> tuple[np.ndarray, np.ndarray, list[float], str]:
    metadata_path = args.pseudo_root / sequence / "IMU-aligned-gt" / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        R_gravity = np.asarray(metadata["gravity_alignment"]["R_gravity_alignment"], dtype=np.float64)
        g_source = np.asarray(metadata["gravity_alignment"]["gravity_before_alignment_m_s2"], dtype=np.float64)
        source = str(metadata_path)
    else:
        query_t, _, _ = base.make_query_grid(overlap_start, overlap_end, args.gravity_rate_hz)
        gt_interp = base.interpolate_gt(gt, query_t)
        imu_interp = base.interp_columns(imu_t, imu_u, query_t)
        _, _, R_gravity, g_source = apply_imu_alignment(
            gt_interp,
            imu_interp,
            T_body0_imu,
            args.gravity_magnitude,
        )
        source = "estimated_from_sparse_export_inputs"

    query_t, _, _ = base.make_query_grid(overlap_start, overlap_end, args.gravity_rate_hz)
    gt_origin = base.interpolate_gt(gt, np.asarray([query_t[0]], dtype=np.float64))
    origin, _ = transform_body0_to_imu(
        gt_origin,
        T_body0_imu,
        R_gravity,
        np.zeros(3, dtype=np.float64),
    )
    return R_gravity, origin[0], g_source.tolist(), source


def load_dense_quality_mask(dense_root: Path, sequence: str, window_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    imu_path = dense_root / sequence / "mav0" / "imu0" / "data.csv"
    valid_path = dense_root / sequence / "mav0" / "quality" / f"valid_window_starts_{window_size}.txt"
    if not imu_path.exists() or not valid_path.exists():
        return None
    imu = np.genfromtxt(imu_path, delimiter=",", skip_header=1)
    if imu.ndim == 1:
        imu = imu.reshape(1, -1)
    t = imu[:, 0].astype(np.float64) * 1e-9
    text = valid_path.read_text(encoding="utf-8").strip()
    valid_starts = np.empty((0,), dtype=np.int64) if not text else np.atleast_1d(np.loadtxt(valid_path, dtype=np.int64))
    mask = np.zeros(max(0, len(t) - window_size + 1), dtype=bool)
    valid_starts = valid_starts[(valid_starts >= 0) & (valid_starts < len(mask))]
    mask[valid_starts] = True
    return t, mask


def compute_valid_sparse_starts(
    gt_t: np.ndarray,
    imu_t: np.ndarray,
    args: argparse.Namespace,
    dense_quality: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    dt_gt = np.diff(gt_t)
    dt_imu = np.diff(imu_t)
    raw_count = max(0, len(gt_t) - args.window_size + 1)
    valid: list[int] = []
    drop_counts = {
        "gt_tiny_dt": 0,
        "gt_gap": 0,
        "imu_gap": 0,
        "dense_quality": 0,
    }
    for start in range(raw_count):
        stop = start + args.window_size
        window_dt = dt_gt[start : stop - 1]
        dropped = False
        if np.any(window_dt < args.min_gt_dt_s):
            drop_counts["gt_tiny_dt"] += 1
            dropped = True
        if np.any(window_dt > args.max_gt_gap_s):
            drop_counts["gt_gap"] += 1
            dropped = True
        imu_start = np.searchsorted(imu_t, gt_t[start], side="left")
        imu_stop = np.searchsorted(imu_t, gt_t[stop - 1], side="right")
        if imu_stop - imu_start < 2 or np.any(dt_imu[max(0, imu_start - 1) : max(0, imu_stop - 1)] > args.max_imu_gap_s):
            drop_counts["imu_gap"] += 1
            dropped = True
        if dense_quality is not None:
            dense_t, dense_valid = dense_quality
            dense_start = int(np.searchsorted(dense_t, gt_t[start], side="left"))
            dense_stop = int(np.searchsorted(dense_t, gt_t[stop - 1], side="right") - args.window_size)
            dense_stop = min(dense_stop, len(dense_valid) - 1)
            if dense_start > dense_stop or not np.all(dense_valid[dense_start : dense_stop + 1]):
                drop_counts["dense_quality"] += 1
                dropped = True
        if not dropped:
            valid.append(start)
    stats = {
        "window_size": int(args.window_size),
        "raw_window_count": int(raw_count),
        "valid_window_count": int(len(valid)),
        "dropped_window_count": int(raw_count - len(valid)),
        "drop_reason_statistics": {key: int(value) for key, value in drop_counts.items() if value > 0},
        "min_gt_dt_s": float(args.min_gt_dt_s),
        "max_gt_gap_s": float(args.max_gt_gap_s),
        "max_imu_gap_s": float(args.max_imu_gap_s),
        "used_dense_quality_root": str(args.dense_quality_root) if dense_quality is not None else None,
    }
    return np.asarray(valid, dtype=np.int64), stats


def export_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    gt_path = args.dataset_root / "0511_data" / "gt" / sequence / "Body_0.tum"
    imu_path = args.pseudo_root / sequence / "sequence" / "imu.txt"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not imu_path.exists():
        raise FileNotFoundError(imu_path)

    T_body0_imu, extrinsic_record, extrinsic_path = load_extrinsic(args.extrinsics_root, sequence, args.pseudo_root)
    gt = base.load_tum_xyzw(gt_path)
    imu_ns, imu_t, imu_u = load_imu_txt_with_ns(imu_path)
    overlap_start = max(float(gt.t[0]), float(imu_t[0]))
    overlap_end = min(float(gt.t[-1]), float(imu_t[-1]))
    keep_gt = (gt.t >= overlap_start) & (gt.t <= overlap_end)
    keep_imu = (imu_t >= overlap_start) & (imu_t <= overlap_end)
    gt_sparse = base.PoseSeries(
        t=gt.t[keep_gt],
        p=gt.p[keep_gt],
        rot=Rotation.from_quat(gt.rot.as_quat()[keep_gt]),
        quat_norm_max_error=gt.quat_norm_max_error,
        duplicate_or_nonmonotonic_count=gt.duplicate_or_nonmonotonic_count,
    )
    imu_ns = imu_ns[keep_imu]
    imu_t = imu_t[keep_imu]
    imu_u = imu_u[keep_imu]

    R_gravity, origin, gravity_before, gravity_source = estimate_or_load_gravity(
        args,
        sequence,
        gt,
        imu_t,
        imu_u,
        T_body0_imu,
        overlap_start,
        overlap_end,
    )
    p_imu, rot_imu = transform_body0_to_imu(gt_sparse, T_body0_imu, R_gravity, origin)
    gt_ns = np.rint(gt_sparse.t * 1e9).astype(np.int64)
    dense_quality = load_dense_quality_mask(args.dense_quality_root, sequence, args.window_size)
    valid_starts, quality_stats = compute_valid_sparse_starts(gt_sparse.t, imu_t, args, dense_quality)

    per_seq_dir = args.pseudo_root / sequence / "IMU-aligned-gt-sparse"
    aggregate_mav0 = args.aggregate_root / sequence / "mav0"
    if args.overwrite:
        for path in (per_seq_dir, args.aggregate_root / sequence):
            if path.exists():
                shutil.rmtree(path)
    per_seq_dir.mkdir(parents=True, exist_ok=True)
    aggregate_mav0.mkdir(parents=True, exist_ok=True)

    for root in (aggregate_mav0, per_seq_dir / "debias_tumvi_source" / "mav0"):
        base.write_imu_csv(root / "imu0" / "data.csv", imu_ns, imu_u)
        base.write_mocap_csv(root / "mocap0" / "data.csv", gt_ns, p_imu, rot_imu)
        quality_dir = root / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(quality_dir / f"valid_source_starts_{args.window_size}.txt", valid_starts, fmt="%d")
        (quality_dir / "source_window_quality.json").write_text(
            json.dumps(quality_stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    base.write_tum(per_seq_dir / "IMU-aligned-gt-sparse.tum", gt_sparse.t, p_imu, rot_imu)
    base.write_mocap_csv(per_seq_dir / "IMU-aligned-gt-sparse.csv", gt_ns, p_imu, rot_imu)

    metadata = {
        "dataset": "corldata",
        "sequence": sequence,
        "trajectory_name": "IMU-aligned-gt-sparse",
        "source_gt": str(gt_path),
        "source_imu": str(imu_path),
        "source_extrinsic_record": str(extrinsic_path),
        "target_supervision": "T_W_I at original mocap timestamps",
        "raw_gt_frame": "T_W_Body0 from mocap Body_0.tum",
        "alignment_model": "T_W_IMU = T_W_Body0 * T_Body0_IMU",
        "T_Body0_IMU": T_body0_imu.tolist(),
        "p_Body0_IMU_m": T_body0_imu[:3, 3].tolist(),
        "p_Body0_IMU_norm_m": float(np.linalg.norm(T_body0_imu[:3, 3])),
        "extrinsic_warning": extrinsic_record.get("warning"),
        "gravity_alignment": {
            "route": "reuse dense IMU-aligned export gravity alignment when available",
            "source": gravity_source,
            "gravity_before_alignment_m_s2": gravity_before,
            "gravity_target_world_m_s2": [0.0, 0.0, -float(args.gravity_magnitude)],
            "R_gravity_alignment": R_gravity.tolist(),
            "origin_world_after_gravity_m": origin.tolist(),
        },
        "timestamp_policy": {
            "gt": "original mocap Body_0.tum timestamps after overlap crop",
            "imu": "raw DAVIS IMU timestamps after overlap crop",
            "overlap_start_s": float(overlap_start),
            "overlap_end_s": float(overlap_end),
        },
        "quality": quality_stats,
        "outputs": {
            "per_sequence_dir": str(per_seq_dir),
            "aggregate_sequence_dir": str(args.aggregate_root / sequence),
            "tum": str(per_seq_dir / "IMU-aligned-gt-sparse.tum"),
            "csv": str(per_seq_dir / "IMU-aligned-gt-sparse.csv"),
        },
    }
    (per_seq_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "sequence": sequence,
        "imu_rows": int(len(imu_ns)),
        "gt_rows": int(len(gt_ns)),
        "duration_s": float(overlap_end - overlap_start),
        "source_dt_median_s": float(np.median(np.diff(gt_sparse.t))),
        "valid_window_count": int(len(valid_starts)),
        "raw_window_count": int(quality_stats["raw_window_count"]),
        "dropped_window_count": int(quality_stats["dropped_window_count"]),
        "p_Body0_IMU_norm_m": float(np.linalg.norm(T_body0_imu[:3, 3])),
        "aggregate_sequence_dir": str(args.aggregate_root / sequence),
    }


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "selected_sequences.txt").write_text(
        "".join(f"{row['sequence']}\n" for row in rows),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# corldata Sparse IMU-aligned GT Export",
        "",
        f"Aggregate root: `{root}`",
        "",
        "| sequence | IMU rows | GT rows | duration s | median GT dt s | valid windows | dropped windows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sequence} | {imu_rows} | {gt_rows} | {duration_s:.3f} | {source_dt_median_s:.6f} | "
            "{valid_window_count} | {dropped_window_count} |".format(**row)
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.pseudo_root = args.pseudo_root.resolve()
    args.extrinsics_root = args.extrinsics_root.resolve()
    args.aggregate_root = args.aggregate_root.resolve()
    args.dense_quality_root = args.dense_quality_root.resolve()
    sequences = args.sequences or discover_sequences(args.extrinsics_root)
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        print(f"[sparse IMU-aligned-gt] {sequence}", flush=True)
        rows.append(export_sequence(args, sequence))
    write_summary(args.aggregate_root, rows)
    print(json.dumps({"aggregate_root": str(args.aggregate_root), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
