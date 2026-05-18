#!/usr/bin/env python3
"""Compare summarized corldata OpenVINS natural-light runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PLATO_ROOT = Path(__file__).resolve().parents[3]


def load_summary(path: Path) -> list[dict[str, Any]]:
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    return label, Path(path)


def row_from_result(method: str, result: dict[str, Any], alignment: str) -> dict[str, Any]:
    align = result["alignments"][alignment]
    light_off = align["states"].get("light_off", {})
    light_off_intervals = [item for item in align["intervals"] if item["state"] == "light_off"]
    first_off_end = light_off_intervals[0]["end_position_error_m"] if len(light_off_intervals) >= 1 else float("nan")
    second_off_end = light_off_intervals[1]["end_position_error_m"] if len(light_off_intervals) >= 2 else float("nan")
    return {
        "method": method,
        "sequence": result["sequence"],
        "estimate_rows": result["estimate_rows"],
        "duration_s": result["matched_duration_s"],
        "all_rmse_m": align["states"]["all"]["rmse_m"],
        "light_on_rmse_m": align["states"].get("light_on", {}).get("rmse_m", float("nan")),
        "light_off_rmse_m": light_off.get("rmse_m", float("nan")),
        "final_position_error_m": align["final_position_error_m"],
        "first_off_end_error_m": first_off_end,
        "second_off_end_error_m": second_off_end,
        "rot_rmse_deg": align["states"]["all"]["rot_rmse_deg"],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "sequence",
        "estimate_rows",
        "duration_s",
        "all_rmse_m",
        "light_on_rmse_m",
        "light_off_rmse_m",
        "final_position_error_m",
        "first_off_end_error_m",
        "second_off_end_error_m",
        "rot_rmse_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path, alignment: str) -> None:
    lines = [
        "# corldata OpenVINS Natural-Light Comparison",
        "",
        f"Alignment scope: `{alignment}`. Position alignment is SE(3), no scale correction.",
        "",
        "| method | seq | rows | dur s | all RMSE m | light-on RMSE m | light-off RMSE m | final err m | off1 end m | off2 end m | rot RMSE deg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['sequence']} | {row['estimate_rows']} | "
            f"{row['duration_s']:.3f} | {row['all_rmse_m']:.3f} | {row['light_on_rmse_m']:.3f} | "
            f"{row['light_off_rmse_m']:.3f} | {row['final_position_error_m']:.3f} | "
            f"{row['first_off_end_error_m']:.3f} | {row['second_off_end_error_m']:.3f} | "
            f"{row['rot_rmse_deg']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare corldata OpenVINS summary.json files.")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        default=[],
        help="LABEL=summary-dir-or-summary-json. May be repeated.",
    )
    parser.add_argument("--alignment", default="first_light_on", choices=["all", "first_light_on"])
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "comparison",
    )
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("At least one --run LABEL=PATH is required")

    rows: list[dict[str, Any]] = []
    for label, path in args.run:
        for result in load_summary(path):
            rows.append(row_from_result(label, result, args.alignment))

    write_csv(rows, args.out_prefix.with_suffix(".csv"))
    write_markdown(rows, args.out_prefix.with_suffix(".md"), args.alignment)
    print(args.out_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
