#!/usr/bin/env python3
"""Write AquaLux OpenVINS baseline comparison summaries."""

from __future__ import annotations

import csv
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENVINS_ROOT = REPO_ROOT / "Results" / "AquaLux" / "openvins_test"
BASELINE_ROOT = OPENVINS_ROOT / "baseline"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("metric_status") == "ok" and row.get("ate_rmse_pos_m")]


def mean_metric(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(row[key]) for row in ok_rows(rows)]
    return float(sum(vals) / len(vals)) if vals else math.nan


def fmt(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def row_by_seq(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["sequence"]: row for row in rows}


def method_summary(name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    ok = ok_rows(rows)
    return {
        "method": name,
        "ok": len(ok),
        "total": len(rows),
        "mean_pos": mean_metric(rows, "ate_rmse_pos_m"),
        "mean_rot": mean_metric(rows, "ate_rmse_rot_deg"),
    }


def comparison_lines(
    raw: list[dict[str, str]],
    comp1: list[dict[str, str]],
    comp2: list[dict[str, str]],
    gated: list[dict[str, str]] | None = None,
) -> list[str]:
    summaries = [
        method_summary("raw", raw),
        *([method_summary("baseline_segment_delta_acc_covgate", gated)] if gated is not None else []),
        method_summary("baseline_compensate1", comp1),
        method_summary("baseline_compensate2", comp2),
    ]
    lines = [
        "| method | ok/total | mean position RMSE (m) | mean rotation RMSE (deg) | note |",
        "|---|---:|---:|---:|---|",
    ]
    notes = {
        "raw": "reference OpenVINS run",
        "baseline_segment_delta_acc_covgate": "recommended; gated acc-delta compensation",
        "baseline_compensate1": "more reliable; no meaningful improvement over raw",
        "baseline_compensate2": "unstable; two sequences produce no odom",
    }
    for item in summaries:
        lines.append(
            f"| `{item['method']}` | {item['ok']}/{item['total']} | "
            f"{fmt(item['mean_pos'])} | {fmt(item['mean_rot'])} | {notes[item['method']]} |"
        )
    return lines


def per_sequence_lines(
    raw: list[dict[str, str]],
    comp1: list[dict[str, str]],
    comp2: list[dict[str, str]],
    gated: list[dict[str, str]] | None = None,
) -> list[str]:
    raw_map = row_by_seq(raw)
    comp1_map = row_by_seq(comp1)
    comp2_map = row_by_seq(comp2)
    gated_map = row_by_seq(gated) if gated is not None else {}
    seqs = sorted(raw_map)
    lines = [
        "| sequence | raw pos | raw rot | gated pos | gated rot | comp1 pos | comp1 rot | comp2 status | comp2 pos | comp2 rot |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for seq in seqs:
        r = raw_map[seq]
        g = gated_map.get(seq, {})
        c1 = comp1_map.get(seq, {})
        c2 = comp2_map.get(seq, {})
        lines.append(
            f"| `{seq}` | {fmt(r.get('ate_rmse_pos_m'))} | {fmt(r.get('ate_rmse_rot_deg'))} | "
            f"{fmt(g.get('ate_rmse_pos_m'))} | {fmt(g.get('ate_rmse_rot_deg'))} | "
            f"{fmt(c1.get('ate_rmse_pos_m'))} | {fmt(c1.get('ate_rmse_rot_deg'))} | "
            f"{c2.get('metric_status') or c2.get('status') or 'missing'} | "
            f"{fmt(c2.get('ate_rmse_pos_m'))} | {fmt(c2.get('ate_rmse_rot_deg'))} |"
        )
    return lines


def write_compensate2_summary(comp2: list[dict[str, str]]) -> None:
    root = BASELINE_ROOT / "compensate2"
    lines = [
        "# AquaLux OpenVINS Baseline Compensate2 Summary",
        "",
        "Method: offline IMU compensation, then OpenVINS reads the compensated IMU CSV.",
        "",
        "- Bias dynamic checkpoint: `Results/AquaLux/preintegration/rss_baseline_unseen_corrected`.",
        "- Initial bias: first 2 s GT-difference mean.",
        "- IMU rate: native AquaLux rate, about `998 Hz`.",
        "- OpenVINS extrinsic estimation remains disabled.",
        "",
        f"Average over ok sequences: position RMSE `{fmt(mean_metric(comp2, 'ate_rmse_pos_m'))} m`, "
        f"rotation RMSE `{fmt(mean_metric(comp2, 'ate_rmse_rot_deg'))} deg`.",
        "",
        "Reliability assessment: rejected as the preferred path. It produced no odom for "
        "`Test_009` and `Test_015`, and severe rotation/position failures on `Test_003` and `Test_013`.",
        "",
        "| sequence | run status | metric status | odom rows | position RMSE (m) | rotation RMSE (deg) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in comp2:
        lines.append(
            f"| `{row['sequence']}` | {row.get('status', 'n/a')} | {row.get('metric_status', 'n/a')} | "
            f"{row.get('odom_rows', 0)} | {fmt(row.get('ate_rmse_pos_m'))} | {fmt(row.get('ate_rmse_rot_deg'))} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            f"- OpenVINS metrics: `{root / 'metrics.csv'}`",
            f"- BiasDyn mean-2s IMU export: `{root / 'biasdyn_mean2s'}`",
            f"- Per-sequence OpenVINS folders: `{root}/Test_*`",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gated_summary(gated: list[dict[str, str]]) -> None:
    root = BASELINE_ROOT / "segment_delta_acc_covgate"
    lines = [
        "# AquaLux OpenVINS Baseline Segment-Delta Acc Covgate Summary",
        "",
        "Method: run raw OpenVINS first, compensate only light-off IMU acceleration delta when raw OpenVINS position covariance is high, then rerun OpenVINS from the beginning.",
        "",
        "- Correction mode: `delta`, avoiding double subtraction of the OpenVINS internal bias estimate.",
        "- Components: `acc` only. Gyro delta caused severe rotation divergence in sanity tests.",
        "- Gate: compensate a sequence only when raw OpenVINS max position covariance trace is greater than `1.0`.",
        "- In this test, only `Test_009` passed the gate.",
        "",
        f"Average over ok sequences: position RMSE `{fmt(mean_metric(gated, 'ate_rmse_pos_m'))} m`, "
        f"rotation RMSE `{fmt(mean_metric(gated, 'ate_rmse_rot_deg'))} deg`.",
        "",
        "| sequence | metric status | position RMSE (m) | rotation RMSE (deg) |",
        "|---|---|---:|---:|",
    ]
    for row in gated:
        lines.append(
            f"| `{row['sequence']}` | {row.get('metric_status', 'n/a')} | "
            f"{fmt(row.get('ate_rmse_pos_m'))} | {fmt(row.get('ate_rmse_rot_deg'))} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            f"- OpenVINS metrics: `{root / 'metrics.csv'}`",
            f"- Hybrid IMU summary: `{root / 'hybrid_imu' / 'summary.md'}`",
            f"- Per-sequence OpenVINS folders: `{root}/Test_*`",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_baseline_summary(
    raw: list[dict[str, str]],
    comp1: list[dict[str, str]],
    comp2: list[dict[str, str]],
    gated: list[dict[str, str]] | None = None,
) -> None:
    lines = [
        "# AquaLux OpenVINS Baseline Summary",
        "",
        "Evaluation: `evo_ape tum --align`, position uses `trans_part`, rotation uses `angle_deg`.",
        "",
        "Decision:",
        "",
        "- `segment_delta_acc_covgate` is the recommended baseline variant from the tested set: it is conservative and shows aggregate improvement over raw OpenVINS.",
        "- `compensate1` is diagnostic only because it switches back to raw OpenVINS after weak intervals.",
        "- `compensate2` is not reliable for AquaLux OpenVINS in this configuration.",
        "",
        "## Method Comparison",
        "",
        *comparison_lines(raw, comp1, comp2, gated),
        "",
        "## Per Sequence",
        "",
        *per_sequence_lines(raw, comp1, comp2, gated),
        "",
        "Artifacts:",
        "",
        f"- Raw: `{OPENVINS_ROOT / 'raw'}`",
        f"- Segment delta acc covgate: `{BASELINE_ROOT / 'segment_delta_acc_covgate'}`",
        f"- Compensate1: `{BASELINE_ROOT / 'compensate1'}`",
        f"- Compensate2: `{BASELINE_ROOT / 'compensate2'}`",
    ]
    (BASELINE_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_openvins_summary(
    raw: list[dict[str, str]],
    comp1: list[dict[str, str]],
    comp2: list[dict[str, str]],
    gated: list[dict[str, str]] | None = None,
) -> None:
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
        "## Method Comparison",
        "",
        *comparison_lines(raw, comp1, comp2, gated),
        "",
        "## Baseline Decision",
        "",
        "`baseline_segment_delta_acc_covgate` is the recommended baseline variant from the tested set. "
        "It only applies acc-delta compensation when raw OpenVINS reports high position covariance, so normal sequences remain raw-like while `Test_009` improves substantially. "
        "`baseline_compensate1` is diagnostic only because it switches back to raw OpenVINS after each weak interval. "
        "`baseline_compensate2` is rejected because it fails or diverges on multiple Test sequences.",
        "",
        "## Per Sequence",
        "",
        *per_sequence_lines(raw, comp1, comp2, gated),
        "",
        "Artifacts:",
        "",
        f"- Raw metrics: `{OPENVINS_ROOT / 'raw' / 'metrics.csv'}`",
        f"- Baseline summary: `{BASELINE_ROOT / 'summary.md'}`",
    ]
    (OPENVINS_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    raw = read_rows(OPENVINS_ROOT / "raw" / "metrics.csv")
    comp1 = read_rows(BASELINE_ROOT / "compensate1" / "metrics.csv")
    comp2 = read_rows(BASELINE_ROOT / "compensate2" / "metrics.csv")
    gated_path = BASELINE_ROOT / "segment_delta_acc_covgate" / "metrics.csv"
    gated = read_rows(gated_path) if gated_path.exists() else None
    write_compensate2_summary(comp2)
    if gated is not None:
        write_gated_summary(gated)
    write_baseline_summary(raw, comp1, comp2, gated)
    write_openvins_summary(raw, comp1, comp2, gated)
    print("wrote AquaLux OpenVINS baseline summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
