#!/usr/bin/env python3
"""Test corldata with learned gyro bias dynamics and constant accel bias."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_ROOT = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU"
BIASDY_ROOT = DEBIAS_ROOT / "BiasDy"
MPL_CACHE_DIR = PLATO_ROOT / "results" / "debias_imu" / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
for path in (str(DEBIAS_ROOT), str(BIASDY_ROOT), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset import pdump  # noqa: E402,F401
from learning import plot_bias, simple_visualization  # noqa: E402
from run_liasuw_debias_ad_pipeline import (  # noqa: E402
    LIASuwDebiasDataset,
    VO_IMU_NONINTERP_SUPERVISION,
    Test_with_configured_aux,
    bw_func_choose,
    make_dataset_parameters,
)


class ConstantAccelBiasRate(torch.nn.Module):
    """Keep the initial accelerometer bias fixed by returning dot b_a = 0."""

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        del imu_meas, imu_meas_dot, R0
        return torch.zeros_like(y[..., 13:16])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("results/debias_imu/corldata_pseudogt_refs_train_s00s02s03s07_test_s13s15s20s21s22s23"),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--pseudo-gt-root", type=Path, default=Path("results/pseudo_gt/corldata"))
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--bw-model-path", type=Path, required=True)
    parser.add_argument("--network_type_Gyro", default="bw_func_m2_no_deriv")
    parser.add_argument("--train-seqs", nargs="+", default=["S00", "S02", "S03", "S07"])
    parser.add_argument("--test-seqs", nargs="+", default=["S13", "S15", "S20", "S21", "S23"])
    parser.add_argument("--trainval-supervision-source", default=VO_IMU_NONINTERP_SUPERVISION)
    parser.add_argument("--vo-supervision-world-frame", default="vo")
    parser.add_argument("--vo-gap-tolerance-s", type=float, default=0.015)
    parser.add_argument("--gravity-calibration-json", type=Path, default=None)
    parser.add_argument("--gravity-accel-world", type=float, nargs=3, default=(0.0, 0.0, -9.7803))
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--loss_window", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--percent_for_val", type=float, default=0.2)
    parser.add_argument("--sg_window_bw", type=int, default=1)
    parser.add_argument("--sg_order_bw", type=int, default=0)
    parser.add_argument("--sg_window_ba", type=int, default=1)
    parser.add_argument("--sg_order_ba", type=int, default=0)
    parser.add_argument("--test-device", default="cpu")
    parser.add_argument("--test-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outputdir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    params = make_dataset_parameters(args)
    dataset_test = LIASuwDebiasDataset(
        **params,
        data_cache_dir=str(args.cache_dir),
        mode="test",
        recompute=False,
    )

    bw_func = bw_func_choose(args.network_type_Gyro, device=args.test_device)
    ckpt = torch.load(args.bw_model_path, map_location=args.test_device, weights_only=True)
    bw_func.load_state_dict(ckpt["func_bw_model"])
    ba_func = ConstantAccelBiasRate().to(args.test_device)

    with torch.no_grad():
        Test_with_configured_aux(
            dataset_test,
            str(args.outputdir),
            bw_func,
            ba_func,
            integral_method="euler",
            device=args.test_device,
            recompute=args.test_recompute,
            use_delta_imu=False,
        )

    lines = []
    result_root = args.outputdir / "Sequnces-results"
    for seq_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        path_tmp = seq_dir / "FullTra_result.p"
        aoe, ate_vel, ate_pos = simple_visualization(str(path_tmp), return_pos=True)
        lines.append(f"sequence: {seq_dir.name:20} AOE: {aoe:.3f} ATE vel: {ate_vel:.3f} ATE pos: {ate_pos:.3f}")
        plot_bias(str(path_tmp), dataset_test)
    if lines:
        (args.outputdir / "evaluation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
