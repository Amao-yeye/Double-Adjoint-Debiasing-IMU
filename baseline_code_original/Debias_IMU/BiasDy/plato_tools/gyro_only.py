import os

import torch

import Interpolation
from SO3diffeq import odeint_SO3
from dataset import EUROCDataset, pdump, pload
from evalutation import AOE
from network_for_BiadDy import VFSE23vbiasBase, bw_func_net
from plato_tools.gyro_history import set_gyro_history_func_for_sequence


def format_gyro_eval_line(seq_name: str, datadic: dict) -> str:
    return (
        f"sequence: {seq_name:20} "
        f"gyro RMSE: {datadic['gyro_rmse_rad_s']:.6f} rad/s "
        f"({datadic['gyro_rmse_deg_s']:.4f} deg/s), "
        f"bias RMSE: {datadic['gyro_bias_rmse_rad_s']:.6f} rad/s, "
        f"AOE: {datadic['gyro_aoe_deg']:.3f} deg"
    )


def Gyro_Test(
    dataset_test: EUROCDataset,
    output_dir: str,
    bw_model: bw_func_net,
    integral_method="euler",
    device="cuda",
    recompute=False,
    gyro_history_k=0,
):
    if gyro_history_k < 0:
        raise ValueError("gyro_history_k should be non-negative")
    if gyro_history_k > 0 and not getattr(bw_model, "uses_gyro_history", False):
        raise ValueError("gyro_history_k requires a gyro history network")
    if getattr(bw_model, "uses_gyro_history", False) and gyro_history_k <= 0:
        raise ValueError("gyro history networks require --gyro_history_k > 0")

    model = VFSE23vbiasBase(biasfunc_w=bw_model.to(device), device=device)
    model.set_bw_zero(False)
    model.set_ba_zero(True)
    model.freeze_func_ba()
    model.eval()

    result_dir = os.path.join(output_dir, "Gyro-results")
    os.makedirs(result_dir, exist_ok=True)
    summary_lines = []

    for index in range(dataset_test.length()):
        seq_name = dataset_test.sequences[index]
        path_tmp = os.path.join(result_dir, seq_name + ".p")
        if not recompute and os.path.exists(path_tmp):
            datadic = pload(path_tmp)
            summary_lines.append(format_gyro_eval_line(seq_name, datadic))
            continue

        print("Testing gyro trajectory: ", seq_name)
        coeff, time = dataset_test.get_coeff(index)
        spline = Interpolation.CubicHermiteSpline(time, coeff, device)
        _, u_imu, X_gt, t_gt = dataset_test.get_full_trajectory(index, device=device)
        bias_w_gt, _ = dataset_test.get_w_bias(index)
        bias_w_gt = bias_w_gt.to(device)

        R0 = X_gt[0, :3, :3].clone()
        bw0 = bias_w_gt[0]
        vec3zeros = torch.zeros(3, device=device)
        y0 = torch.cat(
            [
                vec3zeros,
                t_gt[0].unsqueeze(-1),
                bw0,
                X_gt[0, :3, 3],
                X_gt[0, :3, 4],
                vec3zeros,
            ],
            dim=-1,
        ).unsqueeze(0)

        model.set_u_func(spline.evaluate)
        model.set_u_dot_func(spline.derivative)
        if gyro_history_k > 0:
            set_gyro_history_func_for_sequence(model, dataset_test, index, gyro_history_k, device)
        model.set_R0(R0)
        sol_pred, R_pred = odeint_SO3(model, y0, R0, t_gt, method=integral_method)
        sol_pred = sol_pred.squeeze(1)
        R_pred = R_pred.squeeze(1)

        n_eval = min(u_imu.shape[0], sol_pred.shape[0], bias_w_gt.shape[0])
        bg_pred = sol_pred[:n_eval, 4:7]
        bg_gt = bias_w_gt[:n_eval]
        omega_hat = u_imu[:n_eval, :3] - bg_pred
        omega_gt = u_imu[:n_eval, :3] - bg_gt
        gyro_rmse = torch.sqrt(torch.mean((omega_hat - omega_gt).square()))
        gyro_bias_rmse = torch.sqrt(torch.mean((bg_pred - bg_gt).square()))
        gyro_rmse_deg = gyro_rmse * 180.0 / torch.pi
        aoe = AOE(X_gt[:n_eval, :3, :3], R_pred[:n_eval])

        datadic = {
            "sol_pred": sol_pred.cpu(),
            "R_pred": R_pred.cpu(),
            "X_gt": X_gt.cpu(),
            "t_gt": t_gt.cpu(),
            "omega_hat": omega_hat.cpu(),
            "omega_gt": omega_gt.cpu(),
            "bg_pred": bg_pred.cpu(),
            "bg_gt": bg_gt.cpu(),
            "imu_raw": u_imu.cpu(),
            "gyro_rmse_rad_s": float(gyro_rmse.detach().cpu()),
            "gyro_rmse_deg_s": float(gyro_rmse_deg.detach().cpu()),
            "gyro_bias_rmse_rad_s": float(gyro_bias_rmse.detach().cpu()),
            "gyro_aoe_deg": float(aoe.detach().cpu()),
            "timestamp_start": dataset_test.get_start_timestamp(index),
        }
        pdump(datadic, path_tmp)
        summary_lines.append(format_gyro_eval_line(seq_name, datadic))

    summary_path = os.path.join(output_dir, "gyro_evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
        f.write("\n")
    print("\n".join(summary_lines))
    print(f"Gyro evaluation saved: {summary_path}")
    return summary_lines
