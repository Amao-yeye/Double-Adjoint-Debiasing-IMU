"""
Extract and compare test results from IMU debiasing experiments.
Loads FullTra_result.p files and computes AOE, ATE (velocity), ATE (position) metrics.
"""
import os
import sys
import pickle
import torch

# Add BiasDy to path so we can import lie_algebra and evalutation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BiasDy'))

import lie_algebra as Lie
from evalutation import AOE, ATE


def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict


EXPERIMENTS = [
    "Euroc_baseline_w16",
    "Euroc_SE23_w16",
    "Euroc_SE23decay_w16",
    "Euroc_baseline_w32",
    "Euroc_SE23_w32",
    "Euroc_SE23decay_w32",
    "Euroc_SE23decay_w64",
]

SEQUENCES = [
    "MH_02_easy",
    "MH_04_difficult",
    "V1_01_easy",
    "V1_03_difficult",
    "V2_02_medium",
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def extract_metrics(exp_name, seq_name):
    """Load a FullTra_result.p file and compute AOE, ATE_vel, ATE_pos."""
    path = os.path.join(RESULTS_DIR, exp_name, "Sequnces-results", seq_name, "FullTra_result.p")
    if not os.path.exists(path):
        return None

    data = pload(path)

    X_gt = data['X_gt']        # ground truth SE(3) matrices
    R_pred = data['R_pred']    # predicted rotations
    sol_pred = data['sol_pred']  # order: [xi(3), t(1), bw(3), v(3), p(3), ba(3)]

    with torch.no_grad():
        aoe = AOE(X_gt[:, :3, :3], R_pred).item()           # degrees
        ate_vel = ATE(X_gt[:, :3, 3], sol_pred[:, 7:10]).item()   # m/s
        ate_pos = ATE(X_gt[:, :3, 4], sol_pred[:, 10:13]).item()  # m

    return {'AOE': aoe, 'ATE_vel': ate_vel, 'ATE_pos': ate_pos}


def main():
    # Collect all results
    all_results = {}  # {exp: {seq: {metric: value}}}

    for exp in EXPERIMENTS:
        all_results[exp] = {}
        for seq in SEQUENCES:
            metrics = extract_metrics(exp, seq)
            if metrics is not None:
                all_results[exp][seq] = metrics
            else:
                all_results[exp][seq] = None

    # -------------------------------------------------------------------------
    # Table 1: Per-sequence detail for each metric
    # -------------------------------------------------------------------------
    for metric_name, metric_key, unit in [
        ("Absolute Orientation Error (AOE)", "AOE", "deg"),
        ("Absolute Trajectory Error - Velocity (ATE_vel)", "ATE_vel", "m/s"),
        ("Absolute Trajectory Error - Position (ATE_pos)", "ATE_pos", "m"),
    ]:
        print("=" * 120)
        print(f"  {metric_name}  [{unit}]")
        print("=" * 120)

        # Header
        header = f"{'Experiment':<28}"
        for seq in SEQUENCES:
            header += f"  {seq:>16}"
        header += f"  {'MEAN':>10}"
        print(header)
        print("-" * 120)

        for exp in EXPERIMENTS:
            row = f"{exp:<28}"
            values = []
            for seq in SEQUENCES:
                m = all_results[exp].get(seq)
                if m is not None:
                    val = m[metric_key]
                    values.append(val)
                    row += f"  {val:>16.3f}"
                else:
                    row += f"  {'N/A':>16}"
            if values:
                avg = sum(values) / len(values)
                row += f"  {avg:>10.3f}"
            else:
                row += f"  {'N/A':>10}"
            print(row)

        print()

    # -------------------------------------------------------------------------
    # Table 2: Summary comparison (averaged across sequences)
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("  SUMMARY: Mean metrics across all test sequences")
    print("=" * 90)
    header = f"{'Experiment':<28}  {'AOE (deg)':>12}  {'ATE vel (m/s)':>14}  {'ATE pos (m)':>14}  {'#seq':>5}"
    print(header)
    print("-" * 90)

    for exp in EXPERIMENTS:
        aoe_vals, vel_vals, pos_vals = [], [], []
        for seq in SEQUENCES:
            m = all_results[exp].get(seq)
            if m is not None:
                aoe_vals.append(m['AOE'])
                vel_vals.append(m['ATE_vel'])
                pos_vals.append(m['ATE_pos'])

        n = len(aoe_vals)
        if n > 0:
            row = f"{exp:<28}  {sum(aoe_vals)/n:>12.3f}  {sum(vel_vals)/n:>14.3f}  {sum(pos_vals)/n:>14.3f}  {n:>5}"
        else:
            row = f"{exp:<28}  {'N/A':>12}  {'N/A':>14}  {'N/A':>14}  {0:>5}"
        print(row)

    print()

    # -------------------------------------------------------------------------
    # Table 3: Grouped by window size - method comparison
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("  METHOD COMPARISON BY WINDOW SIZE")
    print("=" * 90)

    # Group experiments by window size
    window_groups = {}
    for exp in EXPERIMENTS:
        # Extract window size (e.g., w16 -> 16)
        w = exp.split('_w')[-1]
        method = exp.replace(f'_w{w}', '')
        if w not in window_groups:
            window_groups[w] = []
        window_groups[w].append((method, exp))

    for w_size in sorted(window_groups.keys(), key=int):
        print(f"\n  Window size: w{w_size}")
        print(f"  {'Method':<28}  {'AOE (deg)':>12}  {'ATE vel (m/s)':>14}  {'ATE pos (m)':>14}")
        print(f"  {'-'*74}")
        for method_name, exp in window_groups[w_size]:
            aoe_vals, vel_vals, pos_vals = [], [], []
            for seq in SEQUENCES:
                m = all_results[exp].get(seq)
                if m is not None:
                    aoe_vals.append(m['AOE'])
                    vel_vals.append(m['ATE_vel'])
                    pos_vals.append(m['ATE_pos'])
            n = len(aoe_vals)
            if n > 0:
                print(f"  {method_name:<28}  {sum(aoe_vals)/n:>12.3f}  {sum(vel_vals)/n:>14.3f}  {sum(pos_vals)/n:>14.3f}")

    print()

    # -------------------------------------------------------------------------
    # Table 4: Best method per sequence
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("  BEST METHOD PER SEQUENCE (lowest error)")
    print("=" * 90)
    for metric_name, metric_key in [("AOE", "AOE"), ("ATE_vel", "ATE_vel"), ("ATE_pos", "ATE_pos")]:
        print(f"\n  Metric: {metric_name}")
        print(f"  {'Sequence':<20}  {'Best Method':<28}  {'Value':>12}")
        print(f"  {'-'*66}")
        for seq in SEQUENCES:
            best_exp = None
            best_val = float('inf')
            for exp in EXPERIMENTS:
                m = all_results[exp].get(seq)
                if m is not None and m[metric_key] < best_val:
                    best_val = m[metric_key]
                    best_exp = exp
            if best_exp is not None:
                print(f"  {seq:<20}  {best_exp:<28}  {best_val:>12.3f}")

    print()


if __name__ == "__main__":
    main()
