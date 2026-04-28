import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from dataset import EUROCDataset
from learning import run_constant_bias_baseline


def main():
    parser = argparse.ArgumentParser("Run constant-bias baseline on Euroc test sequences")
    parser.add_argument("--outputdir", type=str, default="results/baselines/Euroc_constant_bias")
    parser.add_argument("--test_device", type=str, default="cpu")
    parser.add_argument("--test_recompute", action="store_true")
    args = parser.parse_args()

    test_device = args.test_device
    if test_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"test_device={test_device} was requested, but CUDA is not available.")

    dataset_parameters = {
        'dataset_name': 'EUROC',
        'data_dir': 'data/EUROC',
        'train_seqs': [],
        'val_seqs': [],
        'test_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy'
        ],
        'dt': 0.005,
        'percent_for_val': 0.2,
        'loss_window': 16,
        'batch_size': 1000,
        'sg_window_bw': 1,
        'sg_order_bw': 0,
        'sg_window_ba': 1,
        'sg_order_ba': 0,
        'test_gt_source': 'state_groundtruth_estimate0',
    }
    dataset_test = EUROCDataset(
        **dataset_parameters,
        data_cache_dir='data/EUROC_cache_test_gt',
        mode='test'
    )

    run_constant_bias_baseline(
        dataset_test,
        output_dir=args.outputdir,
        device=test_device,
        recompute=args.test_recompute,
    )


if __name__ == "__main__":
    main()
