from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "Results" / "openvins_imu"
DEBIAS_RESULTS = REPO_ROOT / "external" / "Debias_IMU" / "results"
EUROC_DATA_ROOT = REPO_ROOT / "external" / "Debias_IMU" / "data" / "EUROC"

TEST_SEQUENCES = [
    "MH_02_easy",
    "MH_04_difficult",
    "V2_02_medium",
    "V1_03_difficult",
    "V1_01_easy",
]

METHODS = {
    "gt_mse": {
        "description": "Fair no-gate GT-SE23-MSE: BG damped_145e3 + BA rate_tanh_1e2.",
        "result_dir": DEBIAS_RESULTS / "Euroc_PLATO_bg_damped145e3_ba_rate_tanh1e2_oldV1P10_eval_recheck",
        "gyro_checkpoint": DEBIAS_RESULTS / "Euroc_baseline_w64" / "Gyro_weights" / "model_epoch_1800.pt",
        "acc_checkpoint": DEBIAS_RESULTS
        / "Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000"
        / "Acc_weights"
        / "best_val_model.pt",
        "network_type_Gyro": "bw_func_net_damped_145e3",
        "network_type_Acc": "ba_func_net_rate_tanh_1e2",
    },
    "gt_nll": {
        "description": "Fair no-gate sparse GT NLL from no-gate GT-MSE, epoch 40.",
        "result_dir": DEBIAS_RESULTS
        / "Euroc_PLATO_sparse_gt_nll_logres_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_freezeA_eval_e40",
        "gyro_checkpoint": DEBIAS_RESULTS
        / "Euroc_PLATO_sparse_gt_nll_logres_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_freezeA_save5"
        / "NLL_weights"
        / "model_epoch_40.pt",
        "acc_checkpoint": DEBIAS_RESULTS
        / "Euroc_PLATO_sparse_gt_nll_logres_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_freezeA_save5"
        / "NLL_weights"
        / "model_epoch_40.pt",
        "network_type_Gyro": "bw_func_net_damped_145e3",
        "network_type_Acc": "ba_func_net_rate_tanh_1e2",
    },
    "vo_mse": {
        "description": "VO32 relative MSE result, no-gate BA, better than baseline.",
        "result_dir": DEBIAS_RESULTS / "Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_lr1e5_e11_eval_best",
        "gyro_checkpoint": DEBIAS_RESULTS / "Euroc_baseline_w64" / "Gyro_weights" / "model_epoch_1800.pt",
        "acc_checkpoint": DEBIAS_RESULTS
        / "Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_lr1e5_e11"
        / "Acc_weights"
        / "best_val_model.pt",
        "network_type_Gyro": "bw_func_net_damped_145e3",
        "network_type_Acc": "ba_func_net_rate_tanh_1e2",
    },
    "vo_nll": {
        "description": "VO32 sparse NLL epoch 0, all metrics better than VO32-MSE.",
        "result_dir": DEBIAS_RESULTS / "Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_eval_e0",
        "gyro_checkpoint": DEBIAS_RESULTS
        / "Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_e51"
        / "NLL_weights"
        / "model_epoch_0.pt",
        "acc_checkpoint": DEBIAS_RESULTS
        / "Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_e51"
        / "NLL_weights"
        / "model_epoch_0.pt",
        "network_type_Gyro": "bw_func_net_damped_145e3",
        "network_type_Acc": "ba_func_net_rate_tanh_1e2",
    },
}


IMU_HEADER = [
    "#timestamp [ns]",
    "w_RS_S_x [rad s^-1]",
    "w_RS_S_y [rad s^-1]",
    "w_RS_S_z [rad s^-1]",
    "a_RS_S_x [m s^-2]",
    "a_RS_S_y [m s^-2]",
    "a_RS_S_z [m s^-2]",
]

BIAS_HEADER = [
    "#timestamp [ns]",
    "b_g_x [rad s^-1]",
    "b_g_y [rad s^-1]",
    "b_g_z [rad s^-1]",
    "b_a_x [m s^-2]",
    "b_a_y [m s^-2]",
    "b_a_z [m s^-2]",
]


def main() -> None:
    manifest = {
        "output_root": str(RESULTS_ROOT),
        "timestamp_rule": (
            "Use exact timestamps from external/Debias_IMU/data/EUROC/{seq}/mav0/imu0/data.csv "
            "over the cached FullTra_result interval. Fall back to round(timestamp_start_ns + "
            "t_gt_seconds * 1e9) only if the original timestamp is unavailable."
        ),
        "imu_data_columns": IMU_HEADER,
        "bias_columns": BIAS_HEADER,
        "methods": {},
    }

    for method_name, method in METHODS.items():
        method_manifest = {
            "description": method["description"],
            "source_result_dir": str(method["result_dir"]),
            "gyro_checkpoint": str(method["gyro_checkpoint"]),
            "acc_checkpoint": str(method["acc_checkpoint"]),
            "network_type_Gyro": method["network_type_Gyro"],
            "network_type_Acc": method["network_type_Acc"],
            "sequences": {},
        }
        for sequence in TEST_SEQUENCES:
            source_path = method["result_dir"] / "Sequnces-results" / sequence / "FullTra_result.p"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            data = load_pickle(source_path)
            export_info = export_sequence(method_name, sequence, data)
            method_manifest["sequences"][sequence] = export_info
        manifest["methods"][method_name] = method_manifest

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_ROOT / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def export_sequence(method_name: str, sequence: str, data: dict) -> dict:
    required = ["t_gt", "imu_raw", "net_imu", "timestamp_start"]
    for key in required:
        if key not in data:
            raise KeyError(f"{key} missing from FullTra_result for {method_name}/{sequence}")

    t = to_numpy(data["t_gt"]).astype(np.float64)
    imu_raw = to_numpy(data["imu_raw"]).astype(np.float64)
    imu_corrected = to_numpy(data["net_imu"]).astype(np.float64)
    bias = imu_raw - imu_corrected
    timestamp_start = float(np.asarray(data["timestamp_start"]).item())
    timestamps_ns, timestamp_source = sequence_timestamps(sequence, timestamp_start, t, imu_corrected.shape[0])

    if imu_corrected.shape != imu_raw.shape or imu_corrected.shape[-1] != 6:
        raise ValueError(f"Unexpected IMU shape for {method_name}/{sequence}: {imu_corrected.shape}")
    if timestamps_ns.shape[0] != imu_corrected.shape[0]:
        raise ValueError(f"Timestamp/IMU length mismatch for {method_name}/{sequence}")
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError(f"Non-increasing timestamps for {method_name}/{sequence}")

    seq_root = RESULTS_ROOT / method_name / sequence
    imu_dir = seq_root / "mav0" / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)
    data_csv = imu_dir / "data.csv"
    bias_csv = seq_root / "bias.csv"

    write_matrix_csv(data_csv, IMU_HEADER, timestamps_ns, imu_corrected)
    write_matrix_csv(bias_csv, BIAS_HEADER, timestamps_ns, bias)

    return {
        "imu_csv": str(data_csv),
        "bias_csv": str(bias_csv),
        "samples": int(imu_corrected.shape[0]),
        "timestamp_start_ns": int(timestamps_ns[0]),
        "timestamp_end_ns": int(timestamps_ns[-1]),
        "timestamp_source": timestamp_source,
        "duration_s": float(t[-1] - t[0]),
        "mean_abs_bg": [float(x) for x in np.mean(np.abs(bias[:, :3]), axis=0)],
        "mean_abs_ba": [float(x) for x in np.mean(np.abs(bias[:, 3:]), axis=0)],
    }


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sequence_timestamps(
    sequence: str,
    timestamp_start: float,
    t: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, str]:
    raw_csv = EUROC_DATA_ROOT / sequence / "mav0" / "imu0" / "data.csv"
    if raw_csv.exists():
        raw_timestamps = read_timestamp_column(raw_csv)
        start_ns = int(round(timestamp_start))
        matches = np.flatnonzero(raw_timestamps == start_ns)
        if matches.size:
            start_index = int(matches[0])
            end_index = start_index + samples
            if end_index <= raw_timestamps.shape[0]:
                return raw_timestamps[start_index:end_index].copy(), "original_euroc_imu_csv"
    reconstructed = np.rint(timestamp_start + t * 1e9).astype(np.int64)
    return reconstructed, "reconstructed_from_cache_time"


def read_timestamp_column(path: Path) -> np.ndarray:
    timestamps = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            timestamps.append(int(stripped.split(",", 1)[0]))
    return np.asarray(timestamps, dtype=np.int64)


def write_matrix_csv(path: Path, header: list[str], timestamps_ns: np.ndarray, values: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for timestamp_ns, row in zip(timestamps_ns, values):
            writer.writerow([str(int(timestamp_ns)), *[f"{float(x):.10f}" for x in row]])


if __name__ == "__main__":
    main()
