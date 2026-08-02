#!/usr/bin/env python3
"""Compare corldata OpenVINS full-trajectory position and rotation.

Position is aligned with an SE(3), no-scale transform fitted on the selected
alignment window. Rotation is evaluated with a separate fixed-frame orientation
compensation fitted on the same window, avoiding the unsafe corldata convention
of reusing the position-alignment rotation for orientations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


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


def parse_entry(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("Entry must be METHOD:SEQ=RUN_ROOT_OR_SEQUENCE_DIR")
    left, path = value.split("=", 1)
    method, seq = left.split(":", 1)
    method = method.strip()
    seq = seq.strip()
    if not method or not seq:
        raise argparse.ArgumentTypeError("Method and sequence cannot be empty")
    return method, seq, Path(path)


def estimate_path(path: Path, seq: str) -> Path:
    direct = path / "estimate.txt"
    if direct.exists():
        return direct
    nested = path / seq / "estimate.txt"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Missing estimate.txt for {seq}: tried {direct} and {nested}")


def fit_fixed_orientation(r_est: np.ndarray, r_gt: np.ndarray) -> np.ndarray:
    moment = np.zeros((3, 3), dtype=np.float64)
    for est_i, gt_i in zip(r_est, r_gt):
        moment += gt_i @ est_i.T
    u, _, vt = np.linalg.svd(moment)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def metric_summary(values: np.ndarray, suffix: str) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{suffix}_count": 0,
            f"{suffix}_rmse": float("nan"),
            f"{suffix}_mean": float("nan"),
            f"{suffix}_p95": float("nan"),
            f"{suffix}_max": float("nan"),
        }
    return {
        f"{suffix}_count": int(values.size),
        f"{suffix}_rmse": rmse(values),
        f"{suffix}_mean": float(np.mean(values)),
        f"{suffix}_p95": float(np.percentile(values, 95)),
        f"{suffix}_max": float(np.max(values)),
    }


def evaluate_entry(
    summary: Any,
    method: str,
    seq: str,
    path: Path,
    gt_root: Path,
    light_csv: Path,
    alignment: str,
) -> dict[str, Any]:
    t_est, p_est, r_est = summary.load_openvins_estimate(estimate_path(path, seq))
    t_gt, p_gt, r_gt = summary.load_gt(gt_root / seq / "mav0" / "mocap0" / "data.csv")
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

    r_pos, t_pos = summary.fit_rigid_no_scale(p_est[fit_mask], p_gt_i[fit_mask])
    p_aligned = (r_pos @ p_est.T).T + t_pos
    pos_error_m = np.linalg.norm(p_aligned - p_gt_i, axis=1)

    r_fix = fit_fixed_orientation(r_est[fit_mask], r_gt_i[fit_mask])
    r_aligned = r_fix[None, :, :] @ r_est
    rel = np.transpose(r_aligned, (0, 2, 1)) @ r_gt_i
    rot_error_deg = Rotation.from_matrix(rel).magnitude() * 180.0 / math.pi

    masks = summary.diagnostic_state_masks(t, states, segments)
    row: dict[str, Any] = {
        "method": method,
        "sequence": seq,
        "rows": int(t.size),
        "duration_s": float(t[-1] - t[0]),
        "start_time_s": float(t[0]),
        "end_time_s": float(t[-1]),
        "fit_count": int(fit_mask.sum()),
        "position_rmse_m": rmse(pos_error_m),
        "position_final_m": float(pos_error_m[-1]),
        "position_p95_m": float(np.percentile(pos_error_m, 95)),
        "rotation_rmse_deg": rmse(rot_error_deg),
        "rotation_final_deg": float(rot_error_deg[-1]),
        "rotation_p95_deg": float(np.percentile(rot_error_deg, 95)),
    }
    for mask_name in ["light_off", "post_dark_light_on", "last_light_on"]:
        mask = masks.get(mask_name, np.zeros_like(t, dtype=bool))
        row.update(metric_summary(pos_error_m[mask], f"{mask_name}_position_m"))
        row.update(metric_summary(rot_error_deg[mask], f"{mask_name}_rotation_deg"))
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        out.append(
            {
                "method": method,
                "sequence": "AVG",
                "rows": sum(int(row["rows"]) for row in selected),
                "duration_s": float(np.mean([float(row["duration_s"]) for row in selected])),
                "position_rmse_m": float(np.mean([float(row["position_rmse_m"]) for row in selected])),
                "position_final_m": float(np.mean([float(row["position_final_m"]) for row in selected])),
                "rotation_rmse_deg": float(np.mean([float(row["rotation_rmse_deg"]) for row in selected])),
                "rotation_final_deg": float(np.mean([float(row["rotation_final_deg"]) for row in selected])),
            }
        )
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "method",
        "sequence",
        "rows",
        "duration_s",
        "position_rmse_m",
        "position_final_m",
        "position_p95_m",
        "rotation_rmse_deg",
        "rotation_final_deg",
        "rotation_p95_deg",
    ]
    fields = preferred + [field for field in fields if field not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def write_markdown(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], path: Path, alignment: str) -> None:
    lines = [
        "# corldata OpenVINS Full-Trajectory Comparison",
        "",
        f"Alignment scope: `{alignment}`. Position uses SE(3) no-scale alignment. Rotation uses a separate fixed-frame orientation fit on the same alignment window.",
        "",
        "## Per Sequence",
        "",
        "| method | seq | dur s | pos RMSE m | pos final m | rot RMSE deg | rot final deg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['sequence']} | {fmt(row['duration_s'])} | "
            f"{fmt(row['position_rmse_m'])} | {fmt(row['position_final_m'])} | "
            f"{fmt(row['rotation_rmse_deg'])} | {fmt(row['rotation_final_deg'])} |"
        )
    lines.extend(
        [
            "",
            "## Mean",
            "",
            "| method | pos RMSE m | pos final m | rot RMSE deg | rot final deg |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['method']} | {fmt(row['position_rmse_m'])} | {fmt(row['position_final_m'])} | "
            f"{fmt(row['rotation_rmse_deg'])} | {fmt(row['rotation_final_deg'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full-trajectory corldata OpenVINS runs.")
    parser.add_argument("--entry", action="append", type=parse_entry, default=[], help="METHOD:SEQ=RUN_ROOT_OR_SEQUENCE_DIR")
    parser.add_argument("--alignment", default="first_light_on", choices=["all", "first_light_on"])
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--light-segments", type=Path, default=DEFAULT_LIGHT_SEGMENTS)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "full_trajectory_comparison",
    )
    args = parser.parse_args()
    if not args.entry:
        raise SystemExit("At least one --entry METHOD:SEQ=PATH is required")

    summary = load_summary_module()
    rows = [
        evaluate_entry(summary, method, seq, path, args.gt_root, args.light_segments, args.alignment)
        for method, seq, path in args.entry
    ]
    aggregates = aggregate_rows(rows)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows + aggregates, args.out_prefix.with_suffix(".csv"))
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "aggregates": aggregates}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(rows, aggregates, args.out_prefix.with_suffix(".md"), args.alignment)
    print(args.out_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
