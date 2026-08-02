#!/usr/bin/env python3
"""Benchmark EuRoC full-AD vs analytic outer + inner adjoint training memory.

This script intentionally uses the same right-invariant MSE objective for both
methods. The production AD trainers use the historical left-invariant losses,
while the current analytic adjoint formulas are validated for the paper's
right-invariant error dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEBIAS_ROOT = PLATO_ROOT / "external" / "Debias_IMU" / "rss_debiasing_restart" / "Debias_IMU"
BIASDY_ROOT = DEBIAS_ROOT / "BiasDy"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(SCRIPT_DIR), str(BIASDY_ROOT), str(DEBIAS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import Interpolation  # noqa: E402
import lie_algebra as Lie  # noqa: E402
from analytic_gradient import se23_invariant_residual, se23_mse_bias_gradient  # noqa: E402
from bias_adjoint import accumulate_ba_euler_adjoint_gradients  # noqa: E402
from dataset import EUROCDataset  # noqa: E402
from gyro_adjoint import accumulate_bg_euler_adjoint_gradients  # noqa: E402
from learning import select_supervision_states  # noqa: E402
from network_for_BiadDy import VFSE23vbiasBase  # noqa: E402
from run_liasuw_debias_ad_pipeline import ba_func_choose, bw_func_choose  # noqa: E402
from so3_analytic_gradient import so3_gyro_bias_gradient, so3_invariant_residual  # noqa: E402
from SO3diffeq import odeint_SO3  # noqa: E402


TRAIN_SEQS = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5, help="Measured optimizer steps per case.")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--network-type-gyro", type=str, default="bw_func_net_no_udot")
    parser.add_argument("--network-type-acc", type=str, default="ba_func_net_no_udot")
    parser.add_argument("--lr", type=float, default=5.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "double_adjoint_memory_benchmark",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "double_adjoint_memory_benchmark",
    )
    return parser.parse_args()


def cuda_enabled(device: str) -> bool:
    return torch.cuda.is_available() and str(device).startswith("cuda")


def make_dataset(window: int, batch_size: int, dt: float, cache_root: Path) -> EUROCDataset:
    cache_dir = cache_root / f"w{window}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "dataset_name": "EUROC",
        "data_dir": str(DEBIAS_ROOT / "data" / "EUROC"),
        "train_seqs": TRAIN_SEQS,
        "val_seqs": TRAIN_SEQS,
        "test_seqs": ["MH_02_easy"],
        "dt": dt,
        "percent_for_val": 0.2,
        "loss_window": window,
        "batch_size": batch_size,
        "sg_window_bw": 1,
        "sg_order_bw": 0,
        "sg_window_ba": 1,
        "sg_order_ba": 0,
        "trainval_gt_root": None,
        "trainval_gt_source": "state_groundtruth_estimate0",
        "trainval_supervision_timing": "imu",
        "test_gt_source": "state_groundtruth_estimate0",
    }
    return EUROCDataset(**params, data_cache_dir=str(cache_dir), mode="train")


def configure_model_controls(model: VFSE23vbiasBase, batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (t_gt_batch, x_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = batch
    y0_batch, r0_batch = EUROCDataset.construct_init_forodeint(t_gt_batch, x_gt_batch, bwa_gt_batch)
    spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
    model.set_u_func(spline.evaluate)
    model.set_u_dot_func(spline.derivative)
    model.set_R0(r0_batch)
    return y0_batch, r0_batch, t_odeint, eval_indices


def build_gyro_model(args: argparse.Namespace, device: torch.device) -> VFSE23vbiasBase:
    torch.manual_seed(args.seed)
    bw_func = bw_func_choose(args.network_type_gyro, device=str(device))
    model = VFSE23vbiasBase(biasfunc_w=bw_func, device=str(device))
    model.to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(True)
    model.freeze_func_ba()
    return model


def build_acc_model(args: argparse.Namespace, device: torch.device) -> VFSE23vbiasBase:
    torch.manual_seed(args.seed)
    bw_func = bw_func_choose(args.network_type_gyro, device=str(device))
    ba_func = ba_func_choose(args.network_type_acc, device=str(device))
    model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=str(device))
    model.to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.freeze_func_bw()
    return model


def measured_start(device: str) -> float:
    if cuda_enabled(device):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def measured_finish(device: str, start: float) -> tuple[float, float]:
    if cuda_enabled(device):
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_mb = 0.0
    return time.perf_counter() - start, peak_mb


def make_case_schedule(dataset_len: int, total_steps: int, seed: int) -> list[tuple[int, int]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    seq_indices = torch.randint(0, dataset_len, (total_steps,), generator=gen).tolist()
    batch_seeds = torch.randint(0, 2**31 - 1, (total_steps,), generator=gen).tolist()
    return [(int(seq), int(batch_seed)) for seq, batch_seed in zip(seq_indices, batch_seeds)]


def get_batch(dataset: EUROCDataset, sequence_index: int, batch_seed: int, device: torch.device):
    torch.manual_seed(batch_seed)
    return dataset.get_data_for_biasDy(sequence_index, device=device, mask_flag=False, batch_size=dataset._batch_size)


def train_gyro_step_ad(model, optimizer, batch, device: torch.device) -> float:
    optimizer.zero_grad(set_to_none=True)
    y0_batch, r0_batch, t_odeint, eval_indices = configure_model_controls(model, batch, device)
    _, r_full = odeint_SO3(model, y0_batch, r0_batch, t_odeint, method="euler")
    r_sol = r_full if r_full.shape[0] == batch[0][1].shape[0] else r_full[eval_indices]
    r_gt = batch[0][1][..., :3, :3]
    loss = 1.0e6 * so3_invariant_residual(r_sol, r_gt, invariant_error="right").square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def train_gyro_step_adjoint(model, optimizer, batch, device: torch.device) -> float:
    optimizer.zero_grad(set_to_none=True)
    y0_batch, r0_batch, t_odeint, eval_indices = configure_model_controls(model, batch, device)
    with torch.no_grad():
        solution_full, r_full = odeint_SO3(model, y0_batch, r0_batch, t_odeint, method="euler")
    solution, r_sol = select_supervision_states(solution_full, r_full, batch[0][1], eval_indices)
    r_gt = batch[0][1][..., :3, :3]
    loss = 1.0e6 * so3_invariant_residual(r_sol, r_gt, invariant_error="right").square().mean()
    analytic = so3_gyro_bias_gradient(
        r_sol.detach(),
        r_gt.detach(),
        dt=(batch[0][0][1:] - batch[0][0][:-1]).detach(),
        reduction="mean",
        detach=True,
        residual_jacobian="left",
        invariant_error="right",
    )
    grad_bg = 1.0e6 * analytic.bias_gradient
    jump_indices = torch.arange(grad_bg.shape[0], device=device) if solution_full.shape[0] == r_gt.shape[0] else eval_indices[: grad_bg.shape[0]].to(device=device)
    accumulate_bg_euler_adjoint_gradients(model, solution_full, t_odeint, grad_bg, jump_indices=jump_indices)
    optimizer.step()
    return float(loss.detach().cpu())


def train_acc_step_ad(model, optimizer, batch, device: torch.device) -> float:
    optimizer.zero_grad(set_to_none=True)
    y0_batch, r0_batch, t_odeint, eval_indices = configure_model_controls(model, batch, device)
    solution_full, r_full = odeint_SO3(model, y0_batch, r0_batch, t_odeint, method="euler")
    solution, r_sol = select_supervision_states(solution_full, r_full, batch[0][1], eval_indices)
    x_pred = Lie.SEn3fromSO3Rn(r_sol, solution[..., 7:13])
    loss = 1.0e6 * se23_invariant_residual(x_pred, batch[0][1], invariant_error="right").square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def train_acc_step_adjoint(model, optimizer, batch, device: torch.device) -> float:
    optimizer.zero_grad(set_to_none=True)
    y0_batch, r0_batch, t_odeint, eval_indices = configure_model_controls(model, batch, device)
    with torch.no_grad():
        solution_full, r_full = odeint_SO3(model, y0_batch, r0_batch, t_odeint, method="euler")
    solution, r_sol = select_supervision_states(solution_full, r_full, batch[0][1], eval_indices)
    x_pred = Lie.SEn3fromSO3Rn(r_sol, solution[..., 7:13])
    loss = 1.0e6 * se23_invariant_residual(x_pred, batch[0][1], invariant_error="right").square().mean()
    analytic = se23_mse_bias_gradient(
        x_pred.detach(),
        batch[0][1].detach(),
        r_sol.detach(),
        solution[..., 7:10].detach(),
        solution[..., 10:13].detach(),
        dt=(batch[0][0][1:] - batch[0][0][:-1]).detach(),
        reduction="mean",
        detach=True,
        residual_jacobian="left",
        invariant_error="right",
    )
    grad_ba = 1.0e6 * analytic.bias_gradient[..., :3]
    jump_indices = torch.arange(grad_ba.shape[0], device=device) if solution_full.shape[0] == batch[0][1].shape[0] else eval_indices[: grad_ba.shape[0]].to(device=device)
    accumulate_ba_euler_adjoint_gradients(model, solution_full, t_odeint, grad_ba, jump_indices=jump_indices)
    optimizer.step()
    return float(loss.detach().cpu())


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "median": math.nan, "max": math.nan, "min": math.nan}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
        "min": min(values),
    }


def run_case(args: argparse.Namespace, dataset: EUROCDataset, window: int, stage: str, method: str, schedule: list[tuple[int, int]]) -> tuple[list[dict], dict]:
    device = torch.device(args.device)
    if stage == "gyro":
        model = build_gyro_model(args, device)
        step_fn = train_gyro_step_ad if method == "ad" else train_gyro_step_adjoint
    elif stage == "acc":
        model = build_acc_model(args, device)
        step_fn = train_acc_step_ad if method == "ad" else train_acc_step_adjoint
    else:
        raise ValueError(f"Unknown stage: {stage}")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    rows = []
    failure = None
    for step_idx, (sequence_index, batch_seed) in enumerate(schedule):
        measured = step_idx >= args.warmup_steps
        try:
            start = measured_start(args.device)
            batch = get_batch(dataset, sequence_index, batch_seed, device)
            loss = step_fn(model, optimizer, batch, device)
            elapsed_s, peak_mb = measured_finish(args.device, start)
            if measured:
                rows.append(
                    {
                        "window": window,
                        "stage": stage,
                        "method": method,
                        "step": step_idx - args.warmup_steps,
                        "sequence_index": sequence_index,
                        "time_s": elapsed_s,
                        "peak_gpu_memory_mb": peak_mb,
                        "loss": loss,
                        "status": "ok",
                    }
                )
        except RuntimeError as exc:
            failure = str(exc).splitlines()[0]
            if cuda_enabled(args.device):
                torch.cuda.empty_cache()
            rows.append(
                {
                    "window": window,
                    "stage": stage,
                    "method": method,
                    "step": max(step_idx - args.warmup_steps, 0),
                    "sequence_index": sequence_index,
                    "time_s": math.nan,
                    "peak_gpu_memory_mb": math.nan,
                    "loss": math.nan,
                    "status": f"failed: {failure}",
                }
            )
            break

    times = [float(row["time_s"]) for row in rows if row["status"] == "ok"]
    mems = [float(row["peak_gpu_memory_mb"]) for row in rows if row["status"] == "ok"]
    time_summary = summarize(times)
    mem_summary = summarize(mems)
    summary = {
        "window": window,
        "stage": stage,
        "method": method,
        "num_measured_steps": len(times),
        "mean_time_s": time_summary["mean"],
        "median_time_s": time_summary["median"],
        "max_time_s": time_summary["max"],
        "mean_peak_gpu_memory_mb": mem_summary["mean"],
        "max_peak_gpu_memory_mb": mem_summary["max"],
        "status": "ok" if failure is None else f"failed: {failure}",
    }

    del model, optimizer
    if cuda_enabled(args.device):
        torch.cuda.empty_cache()
    return rows, summary


def write_markdown(path: Path, summaries: list[dict], args: argparse.Namespace) -> None:
    by_key = {(row["window"], row["stage"], row["method"]): row for row in summaries}
    lines = [
        "# EuRoC Double-Adjoint Memory Benchmark",
        "",
        "Objective for both methods: `1e6 * mean(Log(right-invariant error)^2)`.",
        "Analytic outer gradients use `invariant_error=right`, `residual_jacobian=left`, `inner_vjp=local_ad`.",
        f"Batch size: `{args.batch_size}`, measured steps per case: `{args.steps}`, warmup steps: `{args.warmup_steps}`, device: `{args.device}`.",
        "",
        "| W | Stage | AD time s | Adj time s | Time ratio adj/AD | AD max MB | Adj max MB | Memory ratio adj/AD | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for window in args.windows:
        for stage in ("gyro", "acc"):
            ad = by_key.get((window, stage, "ad"))
            adj = by_key.get((window, stage, "adjoint"))
            if not ad or not adj:
                continue
            ad_time = ad["mean_time_s"]
            adj_time = adj["mean_time_s"]
            ad_mem = ad["max_peak_gpu_memory_mb"]
            adj_mem = adj["max_peak_gpu_memory_mb"]
            time_ratio = adj_time / ad_time if ad_time and not math.isnan(ad_time) else math.nan
            mem_ratio = adj_mem / ad_mem if ad_mem and not math.isnan(ad_mem) else math.nan
            status = f"{ad['status']}; {adj['status']}"
            lines.append(
                f"| {window} | {stage} | {ad_time:.4f} | {adj_time:.4f} | {time_ratio:.3f} | "
                f"{ad_mem:.1f} | {adj_mem:.1f} | {mem_ratio:.3f} | {status} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={args.device} requested but CUDA is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark_config.json").write_text(
        json.dumps(vars(args) | {"output_dir": str(args.output_dir), "cache_root": str(args.cache_root)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    all_rows: list[dict] = []
    summaries: list[dict] = []
    total_steps = args.steps + args.warmup_steps
    for window in args.windows:
        print(f"\n=== W={window} dataset/cache ===", flush=True)
        dataset = make_dataset(window, args.batch_size, args.dt, args.cache_root)
        schedule = make_case_schedule(dataset.length(), total_steps, args.seed + window)
        for stage in ("gyro", "acc"):
            for method in ("ad", "adjoint"):
                print(f"Running W={window} stage={stage} method={method}", flush=True)
                rows, summary = run_case(args, dataset, window, stage, method, schedule)
                all_rows.extend(rows)
                summaries.append(summary)
                print(
                    f"  status={summary['status']} mean_time={summary['mean_time_s']:.4f}s "
                    f"max_mem={summary['max_peak_gpu_memory_mb']:.1f}MB",
                    flush=True,
                )
        del dataset
        if cuda_enabled(args.device):
            torch.cuda.empty_cache()

    detail_csv = args.output_dir / "step_details.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window",
                "stage",
                "method",
                "step",
                "sequence_index",
                "time_s",
                "peak_gpu_memory_mb",
                "loss",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window",
                "stage",
                "method",
                "num_measured_steps",
                "mean_time_s",
                "median_time_s",
                "max_time_s",
                "mean_peak_gpu_memory_mb",
                "max_peak_gpu_memory_mb",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    write_markdown(args.output_dir / "summary.md", summaries, args)
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
