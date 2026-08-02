#!/usr/bin/env python3
"""Export EuRoC ground truth sampled at another trajectory's timestamps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_ROOT = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU"

DEFAULT_SEQUENCES = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
]

EUROC_HEADER = (
    "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
    "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], "
    "v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], "
    "b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], "
    "b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEBIAS_ROOT / "data" / "EUROC",
        help="EuRoC dataset root containing <seq>/mav0/state_groundtruth_estimate0/data.csv.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEBIAS_ROOT / "dataset" / "Euroc_VO",
        help="Root containing <seq>/data.csv whose timestamps define the output sampling.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "Euroc_GT_at_VO_timestamps",
    )
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    return parser.parse_args()


def read_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def interp_columns(gt: np.ndarray, ref_ts_ns: np.ndarray, columns: list[int]) -> np.ndarray:
    t0 = gt[0, 0]
    gt_t = (gt[:, 0] - t0) * 1.0e-9
    ref_t = (ref_ts_ns - t0) * 1.0e-9
    out = np.zeros((ref_ts_ns.shape[0], len(columns)), dtype=np.float64)
    for out_idx, col in enumerate(columns):
        out[:, out_idx] = np.interp(ref_t, gt_t, gt[:, col])
    return out


def interp_quat_wxyz(gt: np.ndarray, ref_ts_ns: np.ndarray) -> np.ndarray:
    t0 = gt[0, 0]
    gt_t = (gt[:, 0] - t0) * 1.0e-9
    ref_t = (ref_ts_ns - t0) * 1.0e-9
    quat_xyzw = gt[:, [5, 6, 7, 4]]
    rotations = Rotation.from_quat(quat_xyzw)
    interp = Slerp(gt_t, rotations)(ref_t)
    xyzw = interp.as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def write_euroc_csv(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(EUROC_HEADER + "\n")
        for row in rows:
            handle.write(
                f"{int(round(row[0]))},"
                + ",".join(f"{value:.9f}" for value in row[1:])
                + "\n"
            )


def export_sequence(gt_root: Path, reference_root: Path, output_root: Path, sequence: str) -> dict[str, object]:
    gt_path = gt_root / sequence / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    ref_path = reference_root / sequence / "data.csv"
    gt = read_csv(gt_path)
    ref = read_csv(ref_path)
    ref_ts = ref[:, 0].astype(np.int64)
    valid = (ref_ts >= int(gt[0, 0])) & (ref_ts <= int(gt[-1, 0]))
    ref_ts = ref_ts[valid]
    unique_ts, unique_indices = np.unique(ref_ts, return_index=True)
    if unique_ts.shape[0] != ref_ts.shape[0]:
        ref_ts = ref_ts[np.sort(unique_indices)]
    if ref_ts.size < 2:
        raise ValueError(f"{sequence}: fewer than two reference timestamps overlap GT.")

    rows = np.zeros((ref_ts.shape[0], 17), dtype=np.float64)
    rows[:, 0] = ref_ts.astype(np.float64)
    rows[:, 1:4] = interp_columns(gt, ref_ts, [1, 2, 3])
    rows[:, 4:8] = interp_quat_wxyz(gt, ref_ts)
    rows[:, 8:17] = interp_columns(gt, ref_ts, list(range(8, 17)))

    out_path = output_root / sequence / "data.csv"
    write_euroc_csv(out_path, rows)
    dt = np.diff(ref_ts.astype(np.float64)) * 1.0e-9
    return {
        "sequence": sequence,
        "rows": int(rows.shape[0]),
        "path": str(out_path),
        "duration_s": float((ref_ts[-1] - ref_ts[0]) * 1.0e-9),
        "median_dt_s": float(np.median(dt)),
        "dropped_duplicate_timestamps": int(valid.sum() - ref_ts.shape[0]),
    }


def main() -> None:
    args = parse_args()
    summaries = [
        export_sequence(args.gt_root, args.reference_root, args.output_root, sequence)
        for sequence in args.sequences
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("sequence,rows,duration_s,median_dt_s,dropped_duplicate_timestamps,path\n")
        for item in summaries:
            handle.write(
                f"{item['sequence']},{item['rows']},{item['duration_s']:.9f},"
                f"{item['median_dt_s']:.9f},{item['dropped_duplicate_timestamps']},{item['path']}\n"
            )
    for item in summaries:
        print(
            f"{item['sequence']}: rows={item['rows']} "
            f"duration={item['duration_s']:.3f}s median_dt={item['median_dt_s']:.6f}s"
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
