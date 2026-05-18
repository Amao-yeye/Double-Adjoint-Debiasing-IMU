#!/usr/bin/env python3
"""Summarize natural-light corldata OpenVINS runs.

The estimator trajectory is aligned to the IMU-aligned GT with a rigid SE(3)
transform and no scale correction. Two alignment scopes are reported:

* all: fit on all matched states.
* first_light_on: fit only on the first detected light-on interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "raw"
DEFAULT_GT_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_1000hz"
DEFAULT_LIGHT_SEGMENTS = PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "light_segments" / "light_segments.csv"


@dataclass(frozen=True)
class Segment:
    state: str
    start: float
    end: float


def load_openvins_estimate(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) >= 8:
                rows.append(vals[:8])
    if not rows:
        raise ValueError(f"No estimate rows in {path}")
    arr = np.asarray(rows, dtype=float)
    t = arr[:, 0]
    q_jpl_xyzw = arr[:, 1:5]
    p = arr[:, 5:8]
    # OpenVINS stores q_GtoI as JPL x,y,z,w. SciPy's Hamilton xyzw matrix has
    # the opposite cross-product sign, so implement the OpenVINS formula and
    # transpose to get R_ItoG for pose comparison.
    r_g_to_i = np.stack([jpl_quat_to_rot(q) for q in q_jpl_xyzw], axis=0)
    r_i_to_g = np.transpose(r_g_to_i, (0, 2, 1))
    return t, p, r_i_to_g


def jpl_quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    qv = q[:3]
    qw = q[3]
    qx = np.array(
        [
            [0.0, -qv[2], qv[1]],
            [qv[2], 0.0, -qv[0]],
            [-qv[1], qv[0], 0.0],
        ],
        dtype=float,
    )
    return (2.0 * qw * qw - 1.0) * np.eye(3) - 2.0 * qw * qx + 2.0 * np.outer(qv, qv)


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0].astype(np.float64) * 1e-9
    p = data[:, 1:4].astype(float)
    q_wxyz = data[:, 4:8].astype(float)
    r = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    return t, p, r


def interp_gt(t_est: np.ndarray, t_gt: np.ndarray, p_gt: np.ndarray, r_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (t_est >= t_gt[0]) & (t_est <= t_gt[-1])
    t = t_est[mask]
    p = np.column_stack([np.interp(t, t_gt, p_gt[:, i]) for i in range(3)])
    rotations = Rotation.from_matrix(r_gt)
    gt_interp = rotations[np.searchsorted(t_gt, t, side="left").clip(0, len(t_gt) - 1)]
    # Nearest-neighbor orientation is enough for the coarse OpenVINS diagnostic;
    # position metrics are the primary result here.
    return mask, p, gt_interp.as_matrix()


def fit_rigid_no_scale(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_mean - r @ src_mean
    return r, t


def metric_summary(err: np.ndarray) -> dict[str, float | int]:
    err = np.asarray(err, dtype=float)
    if err.size == 0:
        return {
            "count": 0,
            "rmse_m": float("nan"),
            "mean_m": float("nan"),
            "median_m": float("nan"),
            "p90_m": float("nan"),
            "p95_m": float("nan"),
            "p99_m": float("nan"),
            "max_m": float("nan"),
        }
    return {
        "count": int(err.size),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "mean_m": float(np.mean(err)),
        "median_m": float(np.median(err)),
        "p90_m": float(np.percentile(err, 90)),
        "p95_m": float(np.percentile(err, 95)),
        "p99_m": float(np.percentile(err, 99)),
        "max_m": float(np.max(err)),
    }


def rot_error_deg(r_est: np.ndarray, r_gt: np.ndarray) -> np.ndarray:
    rel = np.transpose(r_est, (0, 2, 1)) @ r_gt
    return Rotation.from_matrix(rel).magnitude() * 180.0 / math.pi


def load_segments(path: Path, seq: str) -> list[Segment]:
    segments: list[Segment] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sequence"] != seq:
                continue
            segments.append(
                Segment(
                    state=row["state"],
                    start=float(row["start_gt_time_s"]),
                    end=float(row["end_gt_time_s"]),
                )
            )
    if not segments:
        raise ValueError(f"No light segments for {seq} in {path}")
    return segments


def classify_times(t: np.ndarray, segments: list[Segment]) -> np.ndarray:
    states = np.full(t.shape, "unknown", dtype=object)
    for seg in segments:
        states[(t >= seg.start) & (t <= seg.end)] = seg.state
    return states


def first_light_on_mask(t: np.ndarray, segments: list[Segment]) -> np.ndarray:
    for seg in segments:
        if seg.state == "light_on":
            mask = (t >= seg.start) & (t <= seg.end)
            if int(mask.sum()) >= 5:
                return mask
    raise ValueError("No usable first light-on interval")


def diagnostic_state_masks(t: np.ndarray, states: np.ndarray, segments: list[Segment]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {
        "all": np.ones_like(t, dtype=bool),
        "light_on": states == "light_on",
        "light_off": states == "light_off",
        "unknown": states == "unknown",
    }

    seen_light_off = False
    post_dark_light_on = np.zeros_like(t, dtype=bool)
    reentry_3s = np.zeros_like(t, dtype=bool)
    light_on_segments = [seg for seg in segments if seg.state == "light_on"]
    for seg in segments:
        if seg.state == "light_off":
            seen_light_off = True
            continue
        if seg.state != "light_on" or not seen_light_off:
            continue
        post_dark_light_on |= (t >= seg.start) & (t <= seg.end)
        reentry_3s |= (t >= seg.start) & (t <= min(seg.end, seg.start + 3.0))

    if light_on_segments:
        last = light_on_segments[-1]
        masks["last_light_on"] = (t >= last.start) & (t <= last.end)
    masks["post_dark_light_on"] = post_dark_light_on
    masks["reentry_3s"] = reentry_3s
    return masks


def evaluate_sequence(seq: str, run_root: Path, gt_root: Path, light_csv: Path) -> dict:
    est_path = run_root / seq / "estimate.txt"
    gt_path = gt_root / seq / "mav0" / "mocap0" / "data.csv"
    t_est, p_est, r_est = load_openvins_estimate(est_path)
    t_gt, p_gt, r_gt = load_gt(gt_path)
    valid, p_gt_i, r_gt_i = interp_gt(t_est, t_gt, p_gt, r_gt)
    t = t_est[valid]
    p_est = p_est[valid]
    r_est = r_est[valid]
    segments = load_segments(light_csv, seq)
    states = classify_times(t, segments)

    fit_masks = {
        "all": np.ones_like(t, dtype=bool),
        "first_light_on": first_light_on_mask(t, segments),
    }

    alignments: dict[str, dict] = {}
    for name, fit_mask in fit_masks.items():
        r_align, t_align = fit_rigid_no_scale(p_est[fit_mask], p_gt_i[fit_mask])
        p_aligned = (r_align @ p_est.T).T + t_align
        r_aligned = r_align[None, :, :] @ r_est
        pos_err = np.linalg.norm(p_aligned - p_gt_i, axis=1)
        ori_err = rot_error_deg(r_aligned, r_gt_i)

        by_state: dict[str, dict] = {}
        state_masks = diagnostic_state_masks(t, states, segments)
        for state, mask in state_masks.items():
            if int(mask.sum()) == 0:
                continue
            pos_metrics = metric_summary(pos_err[mask])
            ori_metrics = metric_summary(ori_err[mask])
            by_state[state] = {
                **pos_metrics,
                "rot_rmse_deg": ori_metrics["rmse_m"],
                "rot_mean_deg": ori_metrics["mean_m"],
                "rot_p95_deg": ori_metrics["p95_m"],
                "rot_max_deg": ori_metrics["max_m"],
            }

        by_interval: list[dict] = []
        for idx, seg in enumerate(segments):
            mask = (t >= seg.start) & (t <= seg.end)
            if int(mask.sum()) == 0:
                continue
            pos_metrics = metric_summary(pos_err[mask])
            ori_metrics = metric_summary(ori_err[mask])
            by_interval.append(
                {
                    "index": idx,
                    "state": seg.state,
                    "start_s": seg.start,
                    "end_s": seg.end,
                    **pos_metrics,
                    "rot_rmse_deg": ori_metrics["rmse_m"],
                    "rot_mean_deg": ori_metrics["mean_m"],
                    "rot_p95_deg": ori_metrics["p95_m"],
                    "rot_max_deg": ori_metrics["max_m"],
                    "end_position_error_m": float(pos_err[mask][-1]),
                }
            )

        alignments[name] = {
            "fit_count": int(fit_mask.sum()),
            "states": by_state,
            "intervals": by_interval,
            "final_position_error_m": float(pos_err[-1]),
            "final_timestamp_s": float(t[-1]),
        }

    return {
        "sequence": seq,
        "estimate_rows": int(t_est.size),
        "matched_rows": int(t.size),
        "matched_duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "first_estimate_time_s": float(t[0]),
        "last_estimate_time_s": float(t[-1]),
        "alignments": alignments,
    }


def format_metric(m: dict) -> str:
    return (
        f"{m['count']} | {m['rmse_m']:.3f} | {m['mean_m']:.3f} | "
        f"{m['p95_m']:.3f} | {m['max_m']:.3f} | {m['rot_rmse_deg']:.2f}"
    )


def write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# corldata OpenVINS Natural-Light Summary",
        "",
        "OpenVINS was run on natural-light corldata bags. No artificial image dropout was applied.",
        "Position alignment is SE(3), no scale correction, against `IMU-aligned-gt`.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['sequence']}",
                "",
                f"- estimate rows: {result['estimate_rows']}",
                f"- matched rows: {result['matched_rows']}",
                f"- matched duration: {result['matched_duration_s']:.3f} s",
                "",
            ]
        )
        for align_name in ["all", "first_light_on"]:
            align = result["alignments"][align_name]
            lines.extend(
                [
                    f"### alignment: {align_name}",
                    "",
                    f"- fit rows: {align['fit_count']}",
                    f"- final position error: {align['final_position_error_m']:.3f} m",
                    "",
                    "| segment | count | pos RMSE m | pos mean m | pos p95 m | pos max m | rot RMSE deg |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for state in [
                "all",
                "light_on",
                "light_off",
                "post_dark_light_on",
                "last_light_on",
                "reentry_3s",
                "unknown",
            ]:
                if state not in align["states"]:
                    continue
                lines.append(f"| {state} | {format_metric(align['states'][state])} |")
            lines.append("")
            lines.extend(
                [
                    "| interval | state | count | pos RMSE m | pos mean m | pos p95 m | pos max m | end err m |",
                    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for item in align["intervals"]:
                lines.append(
                    f"| {item['index']} | {item['state']} | {item['count']} | "
                    f"{item['rmse_m']:.3f} | {item['mean_m']:.3f} | {item['p95_m']:.3f} | "
                    f"{item['max_m']:.3f} | {item['end_position_error_m']:.3f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "alignment",
                "segment",
                "count",
                "pos_rmse_m",
                "pos_mean_m",
                "pos_p95_m",
                "pos_max_m",
                "rot_rmse_deg",
                "final_position_error_m",
            ]
        )
        for result in results:
            for align_name, align in result["alignments"].items():
                for state, metrics in align["states"].items():
                    writer.writerow(
                        [
                            result["sequence"],
                            align_name,
                            state,
                            metrics["count"],
                            metrics["rmse_m"],
                            metrics["mean_m"],
                            metrics["p95_m"],
                            metrics["max_m"],
                            metrics["rot_rmse_deg"],
                            align["final_position_error_m"],
                        ]
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", action="append", dest="seqs", default=[])
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--light-segments", type=Path, default=DEFAULT_LIGHT_SEGMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()

    seqs = args.seqs or ["S13", "S15"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_sequence(seq, args.run_root, args.gt_root, args.light_segments) for seq in seqs]
    write_markdown(results, args.out_dir / "summary.md")
    write_csv(results, args.out_dir / "summary.csv")
    (args.out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(args.out_dir / "summary.md")


if __name__ == "__main__":
    main()
