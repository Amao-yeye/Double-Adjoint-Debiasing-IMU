#!/usr/bin/env python3
"""Compare corldata predicted bias trajectories against cached reference labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PLATO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bias_trend_utils import as_numpy, collect_bias_trend_metrics, write_bias_trend_outputs  # noqa: E402
from run_liasuw_debias_ad_pipeline import (  # noqa: E402
    DEFAULT_GRAVITY_ACCEL_WORLD,
    MOCAP_200HZ_SUPERVISION,
    LIASuwDebiasDataset,
)
from learning import pload  # noqa: E402


DEFAULT_TEST_SEQS = ["S13", "S15"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "corldata_IMU_aligned_gt_vo_ref_1000hz",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "corldata_imu_aligned_gt_slow_residual_w128_e1801_1000hz",
    )
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--loss-window", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sg-window-bw", type=int, default=101)
    parser.add_argument("--sg-order-bw", type=int, default=3)
    parser.add_argument("--sg-window-ba", type=int, default=101)
    parser.add_argument("--sg-order-ba", type=int, default=3)
    parser.add_argument("--trainval-supervision-source", default=MOCAP_200HZ_SUPERVISION)
    parser.add_argument("--pseudo-gt-root", type=Path, default=None)
    parser.add_argument("--vo-supervision-world-frame", default="body")
    parser.add_argument("--vo-gap-tolerance-s", type=float, default=0.025)
    parser.add_argument("--gravity-accel-world", nargs=3, type=float, default=list(DEFAULT_GRAVITY_ACCEL_WORLD))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-plot-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.result_dir / "bias_trend_reference_compare")
    dataset = LIASuwDebiasDataset(
        dataset_name="TUM",
        data_dir=str(args.data_root),
        train_seqs=[],
        val_seqs=[],
        test_seqs=list(args.test_seqs),
        mode="test",
        percent_for_val=0.2,
        batch_size=1,
        loss_window=args.loss_window,
        dt=args.dt,
        sg_window_bw=args.sg_window_bw,
        sg_order_bw=args.sg_order_bw,
        sg_window_ba=args.sg_window_ba,
        sg_order_ba=args.sg_order_ba,
        trainval_supervision_source=args.trainval_supervision_source,
        pseudo_gt_root=args.pseudo_gt_root,
        vo_supervision_world_frame=args.vo_supervision_world_frame,
        vo_gap_tolerance_s=args.vo_gap_tolerance_s,
        gravity_calibration_json=None,
        gravity_accel_world=tuple(args.gravity_accel_world),
        data_cache_dir=str(args.cache_dir),
        recompute=False,
    )

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "result_dir": str(args.result_dir),
        "data_root": str(args.data_root),
        "cache_dir": str(args.cache_dir),
        "reference_label": "cached pseudo-reference bias",
        "test_seqs": list(args.test_seqs),
        "sequences": {},
    }
    for seq in args.test_seqs:
        result_path = args.result_dir / "Sequnces-results" / seq / "FullTra_result.p"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        result = pload(str(result_path))
        sol_pred = as_numpy(result["sol_pred"])
        t = as_numpy(result["t_gt"]).reshape(-1)
        bw_pred = sol_pred[:, 4:7]
        ba_pred = sol_pred[:, 13:16]
        seq_index = dataset.find_sequence_index(seq)
        bw_ref = as_numpy(dataset.get_w_bias(seq_index)[0])
        ba_ref = as_numpy(dataset.get_a_bias(seq_index)[0])
        n = min(t.shape[0], bw_pred.shape[0], ba_pred.shape[0], bw_ref.shape[0], ba_ref.shape[0])
        seq_rows, seq_metrics = collect_bias_trend_metrics(
            seq,
            t[:n],
            bw_pred[:n],
            ba_pred[:n],
            bw_ref[:n],
            ba_ref[:n],
            output_dir,
            reference_label="pseudo-reference",
            max_plot_s=args.max_plot_s,
        )
        rows.extend(seq_rows)
        summary["sequences"][seq] = seq_metrics

    write_bias_trend_outputs(output_dir, rows, summary)


if __name__ == "__main__":
    main()
