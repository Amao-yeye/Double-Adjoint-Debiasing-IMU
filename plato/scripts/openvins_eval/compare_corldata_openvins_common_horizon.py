#!/usr/bin/env python3
"""Compare corldata OpenVINS runs at a shared per-sequence horizon."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_SCRIPT = PLATO_ROOT / "plato" / "scripts" / "openvins_eval" / "summarize_corldata_openvins_natural.py"
DEFAULT_GT_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_1000hz"
DEFAULT_LIGHT_SEGMENTS = PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "light_segments" / "light_segments.csv"


def load_summary_module() -> Any:
    spec = importlib.util.spec_from_file_location("corldata_openvins_summary", SUMMARY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SUMMARY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    return label, Path(path)


def metric_summary(err: np.ndarray) -> dict[str, float | int]:
    err = np.asarray(err, dtype=float)
    if err.size == 0:
        return {"count": 0, "rmse_m": float("nan"), "mean_m": float("nan"), "p95_m": float("nan"), "max_m": float("nan")}
    return {
        "count": int(err.size),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "mean_m": float(np.mean(err)),
        "p95_m": float(np.percentile(err, 95)),
        "max_m": float(np.max(err)),
    }


def load_timeseries(
    summary: Any,
    seq: str,
    run_root: Path,
    gt_root: Path,
    light_csv: Path,
    alignment: str,
) -> dict[str, Any]:
    est_path = run_root / seq / "estimate.txt"
    gt_path = gt_root / seq / "mav0" / "mocap0" / "data.csv"
    if not est_path.exists():
        raise FileNotFoundError(est_path)
    t_est, p_est, r_est = summary.load_openvins_estimate(est_path)
    t_gt, p_gt, r_gt = summary.load_gt(gt_path)
    valid, p_gt_i, r_gt_i = summary.interp_gt(t_est, t_gt, p_gt, r_gt)
    t = t_est[valid]
    p_est = p_est[valid]
    r_est = r_est[valid]
    segments = summary.load_segments(light_csv, seq)
    states = summary.classify_times(t, segments)

    if alignment == "first_light_on":
        fit_mask = summary.first_light_on_mask(t, segments)
    elif alignment == "all":
        fit_mask = np.ones_like(t, dtype=bool)
    else:
        raise ValueError(f"Unsupported alignment: {alignment}")

    r_align, t_align = summary.fit_rigid_no_scale(p_est[fit_mask], p_gt_i[fit_mask])
    p_aligned = (r_align @ p_est.T).T + t_align
    r_aligned = r_align[None, :, :] @ r_est
    pos_err = np.linalg.norm(p_aligned - p_gt_i, axis=1)
    rot_err = summary.rot_error_deg(r_aligned, r_gt_i)
    masks = summary.diagnostic_state_masks(t, states, segments)
    return {
        "time_s": t,
        "pos_error_m": pos_err,
        "rot_error_deg": rot_err,
        "masks": masks,
        "first_time_s": float(t[0]),
        "last_time_s": float(t[-1]),
        "rows": int(t.size),
    }


def row_at_common_horizon(label: str, seq: str, ts: dict[str, Any], common_time_s: float) -> dict[str, Any]:
    t = ts["time_s"]
    pos_err = ts["pos_error_m"]
    idx = int(np.argmin(np.abs(t - common_time_s)))
    prefix = t <= common_time_s + 1.0e-6
    row: dict[str, Any] = {
        "method": label,
        "sequence": seq,
        "rows": ts["rows"],
        "own_duration_s": ts["last_time_s"] - ts["first_time_s"],
        "own_final_time_s": ts["last_time_s"],
        "own_final_error_m": float(pos_err[-1]),
        "common_time_s": float(common_time_s),
        "common_duration_s": float(common_time_s - ts["first_time_s"]),
        "common_nearest_time_s": float(t[idx]),
        "common_final_error_m": float(pos_err[idx]),
    }
    for name in ["all", "light_off", "post_dark_light_on", "last_light_on", "reentry_3s"]:
        mask = prefix & ts["masks"].get(name, np.zeros_like(t, dtype=bool))
        metrics = metric_summary(pos_err[mask])
        row[f"common_{name}_count"] = metrics["count"]
        row[f"common_{name}_rmse_m"] = metrics["rmse_m"]
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "sequence",
        "rows",
        "own_duration_s",
        "own_final_time_s",
        "own_final_error_m",
        "common_time_s",
        "common_duration_s",
        "common_nearest_time_s",
        "common_final_error_m",
        "common_all_count",
        "common_all_rmse_m",
        "common_light_off_count",
        "common_light_off_rmse_m",
        "common_post_dark_light_on_count",
        "common_post_dark_light_on_rmse_m",
        "common_last_light_on_count",
        "common_last_light_on_rmse_m",
        "common_reentry_3s_count",
        "common_reentry_3s_rmse_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path, alignment: str) -> None:
    lines = [
        "# corldata OpenVINS Common-Horizon Comparison",
        "",
        f"Alignment scope: `{alignment}`. Each sequence is evaluated at the earliest final timestamp among compared runs.",
        "",
        "| method | seq | own dur s | own final m | common dur s | common final m | common all RMSE m | common light-off RMSE m | common post-dark light-on RMSE m | common last light-on RMSE m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['sequence']} | {row['own_duration_s']:.3f} | "
            f"{row['own_final_error_m']:.3f} | {row['common_duration_s']:.3f} | "
            f"{row['common_final_error_m']:.3f} | {row['common_all_rmse_m']:.3f} | "
            f"{row['common_light_off_rmse_m']:.3f} | {row['common_post_dark_light_on_rmse_m']:.3f} | "
            f"{row['common_last_light_on_rmse_m']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare corldata OpenVINS runs at a common horizon.")
    parser.add_argument("--seq", action="append", dest="seqs", default=[])
    parser.add_argument("--run", action="append", type=parse_run, default=[], help="LABEL=run-root. May be repeated.")
    parser.add_argument("--alignment", default="first_light_on", choices=["all", "first_light_on"])
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--light-segments", type=Path, default=DEFAULT_LIGHT_SEGMENTS)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "common_horizon",
    )
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("At least one --run LABEL=PATH is required")

    summary = load_summary_module()
    seqs = args.seqs or ["S13", "S15"]
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for label, run_root in args.run:
        for seq in seqs:
            loaded[(label, seq)] = load_timeseries(summary, seq, run_root, args.gt_root, args.light_segments, args.alignment)

    rows: list[dict[str, Any]] = []
    for seq in seqs:
        common_time_s = min(loaded[(label, seq)]["last_time_s"] for label, _ in args.run)
        for label, _ in args.run:
            rows.append(row_at_common_horizon(label, seq, loaded[(label, seq)], common_time_s))

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_prefix.with_suffix(".csv"))
    write_markdown(rows, args.out_prefix.with_suffix(".md"), args.alignment)
    args.out_prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(args.out_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
