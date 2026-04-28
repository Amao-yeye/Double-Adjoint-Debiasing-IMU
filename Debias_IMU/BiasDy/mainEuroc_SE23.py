import torch
import os
import argparse
import time
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import Interpolation as Interpolation
from network_for_BiadDy import bw_func_choose, ba_func_choose
from dataset import EUROCDataset
from learning import Gyro_train, Gyro_train_Analytic, Acc_train_SE23, Acc_train_SE23_Analytic, Acc_train_SE23_NLL, Test, simple_visualization, plot_bias

def main():
    parser = argparse.ArgumentParser('Learning Bias Dynamics with SE23 Loss')
    parser.add_argument('--integral_method', type=str, default='euler')
    parser.add_argument('--outputdir', type=str, default="results/Euroc_SE23")
    parser.add_argument('--train_supervision', type=str, default='gt', choices=['gt', 'vo'])
    parser.add_argument('--supervision_timing', type=str, default=None, choices=['imu', 'source'], help=argparse.SUPPRESS)
    parser.add_argument('--exclude_train_seq', nargs='*', default=[])
    parser.add_argument('--network_type_Gyro', type=str, default='bw_func_net')
    parser.add_argument('--network_type_Acc', type=str, default='ba_func_net')
    parser.add_argument('--gyro_loss', type=str, default='ad', choices=['ad', 'analytic'])
    parser.add_argument('--acc_loss', type=str, default='se23', choices=['se23', 'nll', 'analytic'])
    parser.add_argument('-analytic', '--analytic', action='store_true', help='Use analytic outer gradients for accelerometer training.')
    parser.add_argument('--bw_model_path', type=str, default=None)
    parser.add_argument('--ba_model_path', type=str, default=None)
    parser.add_argument('--loss_window', type=int, default=16)
    parser.add_argument('--bw_model_weights', type=str, default=None)
    parser.add_argument('--lr_Gyro', type=float, default=5e-3)
    parser.add_argument('--weight_decay_Gyro', type=float, default=1e-6)
    parser.add_argument('--gyro_lr_step_size', type=int, default=300)
    parser.add_argument('--gyro_lr_gamma', type=float, default=0.5)
    parser.add_argument('--lr_Acc', type=float, default=5e-3)
    parser.add_argument('--weight_decay_Acc', type=float, default=1e-6)
    parser.add_argument('--acc_lr_step_size', type=int, default=300)
    parser.add_argument('--acc_lr_gamma', type=float, default=0.5)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--nll_sigma_a', type=float, default=2.0e-3)
    parser.add_argument('--nll_sigma_g', type=float, default=1.6968e-4)
    parser.add_argument('--nll_w_rot', type=float, default=1.0e-3)
    parser.add_argument('--nll_w_vel', type=float, default=2.0e-2)
    parser.add_argument('--nll_w_pos', type=float, default=1.0e-2)
    parser.add_argument('--nll_optimize_q', action='store_true')
    parser.add_argument('--nll_q_block_epochs', type=int, default=100)
    parser.add_argument('--nll_lr_q', type=float, default=1.0e-2)
    parser.add_argument('--nll_sigma_a_min', type=float, default=1.0e-4)
    parser.add_argument('--nll_sigma_a_max', type=float, default=5.0e-2)
    parser.add_argument('--nll_sigma_g_min', type=float, default=1.0e-5)
    parser.add_argument('--nll_sigma_g_max', type=float, default=5.0e-3)
    parser.add_argument('--analytic_grad_sign', type=float, default=1.0, choices=[-1.0, 1.0], help='Sign applied to analytic dL/db before backpropagating through bias NODE.')
    parser.add_argument('--analytic_grad_scale', type=float, default=1.0, help='Additional scale applied to analytic dL/db.')
    parser.add_argument('--analytic_grad_clip_norm', type=float, default=1.0, help='Clip each analytic bias-gradient tensor to this norm; <=0 disables clipping.')
    parser.add_argument('--analytic_residual_jacobian', type=str, default='left', choices=['identity', 'left', 'right'], help='Jacobian convention for mapping dL/dr through the log residual.')
    parser.add_argument('--analytic_invariant_error', type=str, default='right', choices=['right', 'left'], help='Invariant residual convention used by analytic acc training.')
    parser.add_argument('--gyro_analytic_grad_sign', type=float, default=1.0, choices=[-1.0, 1.0], help='Sign applied to analytic gyro dL/db before the gyro bias NODE adjoint.')
    parser.add_argument('--gyro_analytic_grad_scale', type=float, default=1.0, help='Additional scale applied to analytic gyro dL/db.')
    parser.add_argument('--gyro_analytic_grad_clip_norm', type=float, default=0.0, help='Clip each analytic gyro bias-gradient tensor to this norm; <=0 disables clipping.')
    parser.add_argument('--gyro_analytic_residual_jacobian', type=str, default='left', choices=['identity', 'left', 'right'], help='Jacobian convention for SO(3) log residual.')
    parser.add_argument('--gyro_analytic_invariant_error', type=str, default='right', choices=['right', 'left'], help='Invariant residual convention used by analytic gyro training.')
    parser.add_argument('--inner_vjp', type=str, default='local_ad', choices=['local_ad'], help='Inner bias-dynamics VJP mode for analytic training.')
    parser.add_argument('--train_batch_strategy', type=str, default='random_sequence', choices=['random_sequence', 'balanced_sequences'])
    parser.add_argument('--gyro_train_batch_strategy', type=str, default=None, choices=['random_sequence', 'balanced_sequences'])
    parser.add_argument('--acc_train_batch_strategy', type=str, default=None, choices=['random_sequence', 'balanced_sequences'])
    parser.add_argument('--epoch', type=int, default=1801)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_device', type=str, default='cpu')
    parser.add_argument('--test_recompute', action='store_true')
    parser.add_argument('--use_best_val', action='store_true')
    parser.add_argument('--gyro_checkpoint', type=str, default=None, choices=['last', 'best'])
    parser.add_argument('--acc_checkpoint', type=str, default=None, choices=['last', 'best'])
    args = parser.parse_args()

    ## read the arguments
    integral_method = args.integral_method
    outputdir = args.outputdir
    train_supervision = args.train_supervision
    if args.supervision_timing not in (None, "imu"):
        print(f"Warning: --supervision_timing={args.supervision_timing} is deprecated and ignored; dense IMU-timing supervision is always used.")
    supervision_timing = "imu"
    exclude_train_seq = set(args.exclude_train_seq)
    network_type_Gyro = args.network_type_Gyro
    network_type_Acc = args.network_type_Acc
    gyro_loss = args.gyro_loss
    acc_loss = "analytic" if args.analytic else args.acc_loss
    lr_Gyro = args.lr_Gyro
    weight_decay_Gyro = args.weight_decay_Gyro
    gyro_lr_step_size = args.gyro_lr_step_size
    gyro_lr_gamma = args.gyro_lr_gamma
    lr_Acc = args.lr_Acc
    weight_decay_Acc = args.weight_decay_Acc
    acc_lr_step_size = args.acc_lr_step_size
    acc_lr_gamma = args.acc_lr_gamma
    val_freq = args.val_freq
    save_freq = args.save_freq
    inner_vjp = args.inner_vjp
    train_batch_strategy = args.train_batch_strategy
    gyro_train_batch_strategy = args.gyro_train_batch_strategy or train_batch_strategy
    acc_train_batch_strategy = args.acc_train_batch_strategy or train_batch_strategy
    loss_window = args.loss_window
    device = args.device
    test_device = args.test_device
    epoch = args.epoch
    test_recompute = args.test_recompute
    use_best_val = args.use_best_val
    gyro_checkpoint = args.gyro_checkpoint
    acc_checkpoint = args.acc_checkpoint

    if test_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"test_device={test_device} was requested, but CUDA is not available.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device} was requested, but CUDA is not available.")

    if args.only_test:
        print(f"Euroc testing only with SE_2(3) loss... (window={loss_window}, device={test_device}, train_supervision={train_supervision}, supervision_timing={supervision_timing}, exclude_train_seq={sorted(exclude_train_seq)})")
    else:
        print(f"Euroc training and testing with SE_2(3) loss... (gyro_loss={gyro_loss}, acc_loss={acc_loss}, inner_vjp={inner_vjp}, train_supervision={train_supervision}, supervision_timing={supervision_timing}, gyro_train_batch_strategy={gyro_train_batch_strategy}, acc_train_batch_strategy={acc_train_batch_strategy}, exclude_train_seq={sorted(exclude_train_seq)})")

    trainval_gt_root = 'dataset/Euroc_VO' if train_supervision == 'vo' else None
    trainval_gt_source = 'stereo_estimate_aligned_body' if train_supervision == 'vo' else 'state_groundtruth_estimate0'
    trainval_cache_base = 'data/EUROC_cache_vo_body' if train_supervision == 'vo' else 'data/EUROC_cache_gt'
    trainval_cache_dir = trainval_cache_base if supervision_timing == 'imu' else f'{trainval_cache_base}_source'
    test_cache_dir = 'data/EUROC_cache_test_gt'
    train_seqs = [
        'MH_01_easy',
        'MH_03_medium',
        'MH_05_difficult',
        'V1_02_medium',
        'V2_01_easy',
        'V2_03_difficult'
    ]
    if train_supervision == 'vo':
        exclude_train_seq.add('V2_03_difficult')
    train_seqs = [seq for seq in train_seqs if seq not in exclude_train_seq]

    dataset_parameters = {
        'dataset_name': 'EUROC',
        'data_dir': 'data/EUROC',
        'train_seqs': train_seqs,
        'val_seqs': train_seqs,
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
        'trainval_gt_root': trainval_gt_root,
        'trainval_gt_source': trainval_gt_source,
        'trainval_supervision_timing': supervision_timing,
        'test_gt_source': 'state_groundtruth_estimate0',
        }
    dataset_test = EUROCDataset(**dataset_parameters, data_cache_dir=test_cache_dir, mode='test')
    if gyro_checkpoint is None:
        gyro_checkpoint = "best" if use_best_val else "last"
    if acc_checkpoint is None:
        acc_checkpoint = "best" if use_best_val else "last"

    last_saved_epoch = max(((epoch - 1) // save_freq) * save_freq, 0) if not args.only_test else 1800
    gyro_ckpt_name = "best_val_model.pt" if gyro_checkpoint == "best" else f"model_epoch_{last_saved_epoch}.pt"
    acc_ckpt_name = "best_val_model.pt" if acc_checkpoint == "best" else f"model_epoch_{last_saved_epoch}.pt"
    bw_model_path = args.bw_model_path or os.path.join(outputdir, "Gyro_weights", gyro_ckpt_name)
    ba_model_path = args.ba_model_path or os.path.join(outputdir, "Acc_weights", acc_ckpt_name)

    if not args.only_test:
        dataset_train = EUROCDataset(**dataset_parameters, data_cache_dir=trainval_cache_dir, mode='train', recompute=False)
        dataset_val = EUROCDataset(**dataset_parameters, data_cache_dir=trainval_cache_dir, mode='val')

        t_start = time.time()

        ###### Stage 1: Gyro training ######
        gyro_train_fn = Gyro_train_Analytic if gyro_loss == "analytic" else Gyro_train
        gyro_train_kwargs = {}
        if gyro_loss == "analytic":
            gyro_train_kwargs = {
                "analytic_grad_sign": args.gyro_analytic_grad_sign,
                "analytic_grad_scale": args.gyro_analytic_grad_scale,
                "analytic_grad_clip_norm": args.gyro_analytic_grad_clip_norm,
                "analytic_residual_jacobian": args.gyro_analytic_residual_jacobian,
                "analytic_invariant_error": args.gyro_analytic_invariant_error,
                "inner_vjp": inner_vjp,
            }
        gyro_train_fn(
            dataset_train,
            dataset_val,
            outputdir,
            network_type_Gyro,
            integral_method,
            device,
            lr_Gyro,
            weight_decay_Gyro,
            epoch=epoch,
            lr_step_size=gyro_lr_step_size,
            lr_gamma=gyro_lr_gamma,
            train_batch_strategy=gyro_train_batch_strategy,
            val_freq=val_freq,
            save_freq=save_freq,
            **gyro_train_kwargs,
        )

        ###### Stage 2: Acc training with SE_2(3) loss ######
        bw_func = bw_func_choose(network_type_Gyro, device=device)
        bw_func.load_state_dict(torch.load(bw_model_path, map_location=device, weights_only=True)['func_bw_model'])
        if acc_loss == "nll":
            acc_train_fn = Acc_train_SE23_NLL
        elif acc_loss == "analytic":
            acc_train_fn = Acc_train_SE23_Analytic
        else:
            acc_train_fn = Acc_train_SE23
        acc_train_kwargs = {}
        if acc_loss == "nll":
            acc_train_kwargs = {
                "nll_sigma_a": args.nll_sigma_a,
                "nll_sigma_g": args.nll_sigma_g,
                "nll_w_rot": args.nll_w_rot,
                "nll_w_vel": args.nll_w_vel,
                "nll_w_pos": args.nll_w_pos,
                "nll_optimize_q": args.nll_optimize_q,
                "nll_q_block_epochs": args.nll_q_block_epochs,
                "nll_lr_q": args.nll_lr_q,
                "nll_sigma_a_min": args.nll_sigma_a_min,
                "nll_sigma_a_max": args.nll_sigma_a_max,
                "nll_sigma_g_min": args.nll_sigma_g_min,
                "nll_sigma_g_max": args.nll_sigma_g_max,
            }
        elif acc_loss == "analytic":
            acc_train_kwargs = {
                "analytic_grad_sign": args.analytic_grad_sign,
                "analytic_grad_scale": args.analytic_grad_scale,
                "analytic_grad_clip_norm": args.analytic_grad_clip_norm,
                "analytic_residual_jacobian": args.analytic_residual_jacobian,
                "analytic_invariant_error": args.analytic_invariant_error,
                "inner_vjp": inner_vjp,
            }
        gpu_m = acc_train_fn(
            dataset_train,
            dataset_val,
            outputdir,
            bw_func,
            network_type_Acc,
            integral_method,
            device,
            lr_Acc,
            weight_decay_Acc,
            epoch=epoch,
            lr_step_size=acc_lr_step_size,
            lr_gamma=acc_lr_gamma,
            train_batch_strategy=acc_train_batch_strategy,
            val_freq=val_freq,
            save_freq=save_freq,
            **acc_train_kwargs,
        )

        t_end = time.time()
        print("Time for training: ", t_end - t_start)
        ave_gpu_m = torch.tensor(gpu_m).mean()
        with open(os.path.join(outputdir, "time_memory_record.txt"), "w") as f:
            f.write(f"Time for training: {t_end - t_start}\n")
            f.write(f"Max GPU Memory Allocated: {ave_gpu_m:.2f} MB\n")
            f.write(f"trainval_gt_root: {trainval_gt_root}\n")
            f.write(f"train_supervision: {train_supervision}\n")
            f.write(f"supervision_timing: {supervision_timing}\n")
            f.write(f"acc_loss: {acc_loss}\n")
            f.write(f"gyro_loss: {gyro_loss}\n")
            f.write(f"train_batch_strategy: {train_batch_strategy}\n")
            f.write(f"gyro_train_batch_strategy: {gyro_train_batch_strategy}\n")
            f.write(f"acc_train_batch_strategy: {acc_train_batch_strategy}\n")
            f.write(f"trainval_gt_source: {trainval_gt_source}\n")
            f.write(f"exclude_train_seq: {sorted(exclude_train_seq)}\n")
            f.write(f"lr_Gyro: {lr_Gyro}\n")
            f.write(f"weight_decay_Gyro: {weight_decay_Gyro}\n")
            f.write(f"gyro_lr_step_size: {gyro_lr_step_size}\n")
            f.write(f"gyro_lr_gamma: {gyro_lr_gamma}\n")
            f.write(f"lr_Acc: {lr_Acc}\n")
            f.write(f"weight_decay_Acc: {weight_decay_Acc}\n")
            f.write(f"acc_lr_step_size: {acc_lr_step_size}\n")
            f.write(f"acc_lr_gamma: {acc_lr_gamma}\n")
            f.write(f"val_freq: {val_freq}\n")
            f.write(f"save_freq: {save_freq}\n")
            f.write(f"inner_vjp: {inner_vjp}\n")
            f.write(f"nll_sigma_a: {args.nll_sigma_a}\n")
            f.write(f"nll_sigma_g: {args.nll_sigma_g}\n")
            f.write(f"nll_w_rot: {args.nll_w_rot}\n")
            f.write(f"nll_w_vel: {args.nll_w_vel}\n")
            f.write(f"nll_w_pos: {args.nll_w_pos}\n")
            f.write(f"nll_optimize_q: {args.nll_optimize_q}\n")
            f.write(f"nll_q_block_epochs: {args.nll_q_block_epochs}\n")
            f.write(f"nll_lr_q: {args.nll_lr_q}\n")
            f.write(f"nll_sigma_a_min: {args.nll_sigma_a_min}\n")
            f.write(f"nll_sigma_a_max: {args.nll_sigma_a_max}\n")
            f.write(f"nll_sigma_g_min: {args.nll_sigma_g_min}\n")
            f.write(f"nll_sigma_g_max: {args.nll_sigma_g_max}\n")
            f.write(f"analytic_grad_sign: {args.analytic_grad_sign}\n")
            f.write(f"analytic_grad_scale: {args.analytic_grad_scale}\n")
            f.write(f"analytic_grad_clip_norm: {args.analytic_grad_clip_norm}\n")
            f.write(f"analytic_residual_jacobian: {args.analytic_residual_jacobian}\n")
            f.write(f"analytic_invariant_error: {args.analytic_invariant_error}\n")
            f.write(f"gyro_analytic_grad_sign: {args.gyro_analytic_grad_sign}\n")
            f.write(f"gyro_analytic_grad_scale: {args.gyro_analytic_grad_scale}\n")
            f.write(f"gyro_analytic_grad_clip_norm: {args.gyro_analytic_grad_clip_norm}\n")
            f.write(f"gyro_analytic_residual_jacobian: {args.gyro_analytic_residual_jacobian}\n")
            f.write(f"gyro_analytic_invariant_error: {args.gyro_analytic_invariant_error}\n")
            f.write(f"bw_model_path: {bw_model_path}\n")
            f.write(f"ba_model_path: {ba_model_path}\n")

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
