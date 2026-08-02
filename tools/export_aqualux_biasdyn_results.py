#!/usr/bin/env python3
"""Export AquaLux BiasDy FullTra_result files to CSV/TUM artifacts.

This is useful when the expensive BiasDy test trajectory files already exist
and we only need per-sequence corrected IMU files or metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
BIASDY_DIR = REPO_ROOT / "external" / "Debias_IMU" / "BiasDy"
if str(BIASDY_DIR) not in sys.path:
    sys.path.insert(0, str(BIASDY_DIR))

from aqualux_dataset import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_GRAVITY_NORM,
    AquaLuxDataset,
    estimate_native_dt,
    sorted_aqualux_sequences,
)
from mainAquaLux import (  # noqa: E402
    aggregate_rows,
    export_preintegration_results,
    write_method_summary,
    write_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--test-seq", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputdir = args.outputdir.resolve()
    run_config_path = outputdir / "run_config.json"
    config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}

    test_seqs = args.test_seq or config.get("test_seqs") or sorted_aqualux_sequences(args.data_dir, "Test_")
    train_seqs = config.get("train_seqs") or sorted_aqualux_sequences(args.data_dir, "Train_")
    cache_dir = args.cache_dir or Path(config.get("cache_dir", outputdir / "cache"))
    dt = float(config.get("effective_dt") or estimate_native_dt(args.data_dir, sorted(set(train_seqs + test_seqs))))

    dataset_args = {
        "dataset_name": "AquaLux",
        "data_dir": args.data_dir,
        "data_cache_dir": str(cache_dir),
        "train_seqs": train_seqs,
        "val_seqs": train_seqs,
        "test_seqs": test_seqs,
        "dt": dt,
        "percent_for_val": float(config.get("percent_for_val", 0.2)),
        "loss_window": int(config.get("loss_window", 64)),
        "batch_size": int(config.get("batch_size", 1000)),
        "sg_window_bw": int(config.get("sg_window_bw", 1)),
        "sg_order_bw": int(config.get("sg_order_bw", 0)),
        "sg_window_ba": int(config.get("sg_window_ba", 1)),
        "sg_order_ba": int(config.get("sg_order_ba", 0)),
        "gravity_norm": float(config.get("gravity_norm", DEFAULT_GRAVITY_NORM)),
        "gt_valid_max_dt": float(config.get("gt_valid_max_dt", 0.02)),
        "velocity_smooth_sec": float(config.get("velocity_smooth_sec", 0.12)),
    }
    dataset_test = AquaLuxDataset(**dataset_args, mode="test", recompute=False)
    rows = export_preintegration_results(dataset_test, outputdir)
    write_metrics_csv(outputdir / "metrics.csv", rows)
    aggregate = aggregate_rows(rows)

    summary_args = SimpleNamespace(
        loss_window=dataset_args["loss_window"],
        gravity_norm=dataset_args["gravity_norm"],
        gt_valid_max_dt=dataset_args["gt_valid_max_dt"],
    )
    write_method_summary(outputdir / "summary.md", rows, aggregate, summary_args, dt)
    print(
        "exported "
        f"{len(rows)} sequences: position_rmse_m={aggregate['position_rmse_m']:.6f}, "
        f"rotation_rmse_deg={aggregate['rotation_rmse_deg']:.6f}"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    raise SystemExit(main())
