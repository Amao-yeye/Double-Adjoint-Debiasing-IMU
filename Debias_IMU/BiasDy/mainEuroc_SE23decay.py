import torch
import os
import argparse
import time

import Interpolation as Interpolation
from network_for_BiadDy import bw_func_choose, ba_func_choose
from dataset import EUROCDataset
from learning import Gyro_train, Acc_train_SE23decay, Test, simple_visualization, plot_bias

def main():
    parser = argparse.ArgumentParser('Learning Bias Dynamics with SE23 Decay Loss')
    parser.add_argument('--integral_method', type=str, default='euler')
    parser.add_argument('--outputdir', type=str, default="results/Euroc_SE23decay")
    parser.add_argument('--network_type_Gyro', type=str, default='bw_func_net')
    parser.add_argument('--network_type_Acc', type=str, default='ba_func_net')
    parser.add_argument('--bw_model_path', type=str, default=None)
    parser.add_argument('--loss_window', type=int, default=16)
    parser.add_argument('--bw_model_weights', type=str, default=None)
    parser.add_argument('--lr_Gyro', type=float, default=5e-3)
    parser.add_argument('--weight_decay_Gyro', type=float, default=1e-6)
    parser.add_argument('--lr_Acc', type=float, default=5e-3)
    parser.add_argument('--weight_decay_Acc', type=float, default=1e-6)
    parser.add_argument('--epoch', type=int, default=1801)
    parser.add_argument('--decay_lambda', type=float, default=1.0)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--test_device', type=str, default='cpu')
    parser.add_argument('--test_recompute', action='store_true')
    args = parser.parse_args()

    integral_method = args.integral_method
    outputdir = args.outputdir
    network_type_Gyro = args.network_type_Gyro
    network_type_Acc = args.network_type_Acc
    lr_Gyro = args.lr_Gyro
    weight_decay_Gyro = args.weight_decay_Gyro
    lr_Acc = args.lr_Acc
    weight_decay_Acc = args.weight_decay_Acc
    loss_window = args.loss_window
    decay_lambda = args.decay_lambda
    device = "cuda"
    test_device = args.test_device
    epoch = args.epoch
    test_recompute = args.test_recompute

    if test_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"test_device={test_device} was requested, but CUDA is not available.")

    if args.only_test:
        print(f"Euroc testing only with SE_2(3) decay loss (lambda={decay_lambda}, window={loss_window}, device={test_device})...")
    else:
        print(f"Euroc training with SE_2(3) decay loss (lambda={decay_lambda}, window={loss_window})...")

    dataset_parameters = {
        'dataset_name': 'EUROC',
        'data_dir': 'data/EUROC',
        'data_cache_dir': 'data/EUROC',
        'train_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
        ],
        'val_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
        ],
        'test_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy'
        ],
        'dt': 0.005,
        'percent_for_val': 0.2,
        'loss_window': loss_window,
        'batch_size': 1000,
        'sg_window_bw': 1,
        'sg_order_bw': 0,
        'sg_window_ba': 1,
        'sg_order_ba': 0,
        }
    dataset_test = EUROCDataset(**dataset_parameters, mode='test')
    bw_model_path = os.path.join(outputdir, "Gyro_weights", "model_epoch_1800.pt")
    ba_model_path = os.path.join(outputdir, "Acc_weights", "model_epoch_1800.pt")

    if not args.only_test:
        dataset_train = EUROCDataset(**dataset_parameters, mode='train', recompute=False)
        dataset_val = EUROCDataset(**dataset_parameters, mode='val')

        t_start = time.time()

        ###### Stage 1: Gyro training (identical to baseline) ######
        Gyro_train(dataset_train, dataset_val, outputdir, network_type_Gyro, integral_method, device, lr_Gyro, weight_decay_Gyro, epoch=epoch)

        ###### Stage 2: Acc training with SE_2(3) decay loss ######
        bw_func = bw_func_choose(network_type_Gyro, device=device)
        bw_func.load_state_dict(torch.load(bw_model_path, map_location=device, weights_only=True)['func_bw_model'])
        gpu_m = Acc_train_SE23decay(dataset_train, dataset_val, outputdir, bw_func, network_type_Acc, integral_method, device, lr_Acc, weight_decay_Acc, epoch=epoch, decay_lambda=decay_lambda)

        t_end = time.time()
        print("Time for training: ", t_end - t_start)
        ave_gpu_m = torch.tensor(gpu_m).mean()
        with open(os.path.join(outputdir, "time_memory_record.txt"), "w") as f:
            f.write(f"Time for training: {t_end - t_start}\n")
            f.write(f"Max GPU Memory Allocated: {ave_gpu_m:.2f} MB\n")
            f.write(f"decay_lambda: {decay_lambda}\n")
            f.write(f"loss_window: {loss_window}\n")

    ###### testing ######
    bw_func = bw_func_choose(network_type_Gyro, device=test_device)
    bw_func.load_state_dict(torch.load(bw_model_path, map_location=test_device, weights_only=True)['func_bw_model'])
    ba_func = ba_func_choose(network_type_Acc, device=test_device)
    ba_func.load_state_dict(torch.load(ba_model_path, map_location=test_device, weights_only=True)['func_ba_model'])
    with torch.no_grad():
        Test(dataset_test, outputdir, bw_func, ba_func, integral_method, device=test_device, recompute=test_recompute)

    ##### simple evaluation and plot #####
    seq = [f for f in os.listdir(os.path.join(outputdir, "Sequnces-results")) if os.path.isdir(os.path.join(outputdir, "Sequnces-results", f))]
    evaluation_lines = []
    for s in seq:
        path_tmp = os.path.join(outputdir, "Sequnces-results", s, "FullTra_result.p")
        aoe, ate_vel, ate_pos = simple_visualization(path_tmp, return_pos=True)
        evaluation_lines.append(f"sequence: {s:20} AOE: {aoe:.3f} ATE vel: {ate_vel:.3f} ATE pos: {ate_pos:.3f}")
        plot_bias(path_tmp, dataset_test)
    with open(os.path.join(outputdir, "evaluation_summary.txt"), "w") as f:
        f.write("\n".join(evaluation_lines))
        f.write("\n")


if __name__ == "__main__":
    main()
