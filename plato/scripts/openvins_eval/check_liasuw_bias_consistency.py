#!/usr/bin/env python3
"""Check whether OpenVINS and Debias_IMU bias values are consistent.

The check uses one raw IMU stream, one GT initial pose/velocity, and several
bias sources.  It propagates a short inertial trajectory with the same
OpenVINS-compatible convention used by the PLATO body GT:

    R_WB[k+1] = R_WB[k] Exp((w_raw - b_g) dt)
    a_W       = R_WB[k] (a_raw - b_a) - [0, 0, g]

The GT quaternion in the exported LIASuw files is interpreted as R_WB, matching
the Debias_IMU pseudo-bias construction in run_liasuw_debias_ad_pipeline.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PLATO_ROOT = Path(__file__).resolve().parents[3]


def read_numeric_csv(path: Path, delimiter: str = ",") -> np.ndarray:
    first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    skip = 1 if any(ch.isalpha() for ch in first) or first.startswith("#") else 0
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data.astype(np.float64)


def read_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    values: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader)
        if not first[0].startswith("#"):
            timestamps.append(int(first[0]))
            values.append([float(x) for x in first[1:7]])
        for row in reader:
            if not row:
                continue
            timestamps.append(int(row[0]))
            values.append([float(x) for x in row[1:7]])
    return np.asarray(timestamps, dtype=np.float64) * 1e-9, np.asarray(values, dtype=np.float64)


def read_bias_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = read_numeric_csv(path)
    return data[:, 0] * 1e-9, data[:, 1:7]


def read_openvins_estimate(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(x) for x in stripped.split()])
    data = np.asarray(rows, dtype=np.float64)
    return {
        "t": data[:, 0],
        "bg": data[:, 11:14],
        "ba": data[:, 14:17],
    }


def interp_columns(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_dst, t_src, values[:, col]) for col in range(values.shape[1])])


class GTTrajectory:
    def __init__(self, path: Path, gravity_mag: float):
        data = read_numeric_csv(path)
        self.t = data[:, 0] * 1e-9
        self.p = data[:, 1:4]
        self.q_wxyz = data[:, 4:8]
        self.v = data[:, 8:11]
        self.rot = Rotation.from_quat(self.q_wxyz[:, [1, 2, 3, 0]])
        self.slerp = Slerp(self.t, self.rot)
        self.a = np.gradient(self.v, self.t, axis=0, edge_order=1)
        self.g_world = np.array([0.0, 0.0, gravity_mag], dtype=np.float64)

    def position(self, t_query: np.ndarray) -> np.ndarray:
        return interp_columns(self.t, self.p, t_query)

    def velocity(self, t_query: np.ndarray) -> np.ndarray:
        return interp_columns(self.t, self.v, t_query)

    def acceleration(self, t_query: np.ndarray) -> np.ndarray:
        return interp_columns(self.t, self.a, t_query)

    def rotation(self, t_query: np.ndarray) -> Rotation:
        return self.slerp(t_query)

    def expected_specific_force_body(self, t_query: np.ndarray) -> np.ndarray:
        rot = self.rotation(t_query)
        a_plus_g = self.acceleration(t_query) + self.g_world.reshape(1, 3)
        return rot.inv().apply(a_plus_g)

    def expected_body_rate(self, t_query: np.ndarray) -> np.ndarray:
        rot = self.rotation(t_query)
        out = np.zeros((t_query.shape[0], 3), dtype=np.float64)
        if t_query.shape[0] < 2:
            return out
        dt = np.diff(t_query)
        delta = rot[:-1].inv() * rot[1:]
        out[:-1] = delta.as_rotvec() / dt[:, None]
        out[-1] = out[-2]
        return out


def rmse_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def rotation_error_deg(pred: Rotation, gt: Rotation) -> np.ndarray:
    return (pred * gt.inv()).magnitude() * 180.0 / math.pi


def make_time_grid(raw_t: np.ndarray, start_s: float, duration_s: float) -> np.ndarray:
    end_s = start_s + duration_s
    mask = (raw_t >= start_s) & (raw_t <= end_s)
    middle = raw_t[mask]
    grid = np.concatenate([[start_s], middle, [end_s]])
    return np.unique(grid)


def propagate(
    t: np.ndarray,
    imu: np.ndarray,
    bg: np.ndarray,
    ba: np.ndarray,
    gt: GTTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Rotation]:
    rot = gt.rotation(np.array([t[0]]))
    p = gt.position(np.array([t[0]]))[0].copy()
    v = gt.velocity(np.array([t[0]]))[0].copy()

    pred_p = [p.copy()]
    pred_v = [v.copy()]
    pred_rot_mats = [rot.as_matrix()[0]]

    for i in range(t.shape[0] - 1):
        dt = float(t[i + 1] - t[i])
        w_corr = imu[i, :3] - bg[i]
        a_corr = imu[i, 3:6] - ba[i]
        a_world = rot.apply(a_corr.reshape(1, 3))[0] - gt.g_world
        p = p + v * dt + 0.5 * a_world * dt * dt
        v = v + a_world * dt
        rot = rot * Rotation.from_rotvec(w_corr * dt)
        pred_p.append(p.copy())
        pred_v.append(v.copy())
        pred_rot_mats.append(rot.as_matrix()[0])

    return np.asarray(pred_p), np.asarray(pred_v), t, Rotation.from_matrix(np.asarray(pred_rot_mats))


def summarize_mode(
    name: str,
    t: np.ndarray,
    imu: np.ndarray,
    bg: np.ndarray,
    ba: np.ndarray,
    gt: GTTrajectory,
) -> dict[str, object]:
    pred_p, pred_v, pred_t, pred_rot = propagate(t, imu, bg, ba, gt)
    gt_p = gt.position(pred_t)
    gt_v = gt.velocity(pred_t)
    gt_rot = gt.rotation(pred_t)

    expected_w = gt.expected_body_rate(t)
    expected_f = gt.expected_specific_force_body(t)
    gyro_res = imu[:, :3] - bg - expected_w
    accel_res = imu[:, 3:6] - ba - expected_f

    pos_err = np.linalg.norm(pred_p - gt_p, axis=1)
    vel_err = np.linalg.norm(pred_v - gt_v, axis=1)
    rot_err = rotation_error_deg(pred_rot, gt_rot)
    return {
        "method": name,
        "samples": int(t.shape[0]),
        "duration_s": float(t[-1] - t[0]),
        "gyro_specific_rmse_rad_s": rmse_norm(gyro_res),
        "accel_specific_rmse_m_s2": rmse_norm(accel_res),
        "position_rmse_m": float(np.sqrt(np.mean(pos_err * pos_err))),
        "position_final_m": float(pos_err[-1]),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(vel_err * vel_err))),
        "velocity_final_m_s": float(vel_err[-1]),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rot_err * rot_err))),
        "rotation_final_deg": float(rot_err[-1]),
        "mean_bg": bg.mean(axis=0).tolist(),
        "mean_ba": ba.mean(axis=0).tolist(),
        "first_bg": bg[0].tolist(),
        "first_ba": ba[0].tolist(),
    }


def write_outputs(path: Path, summaries: list[dict[str, object]], metadata: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "bias_consistency_summary.json").write_text(
        json.dumps({"metadata": metadata, "summaries": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# LIASuw Bias Consistency Check",
        "",
        f"- Sequence dir: `{metadata['sequence_dir']}`",
        f"- Start time: `{metadata['start_s']:.6f}` s",
        f"- Gravity vector used in world frame: `[0, 0, {metadata['gravity_mag']}]`",
        "",
        "The GT quaternion is interpreted as `R_WB`. Propagation uses `a_raw - b_a` and `w_raw - b_g` with the same initial GT pose and velocity for every row.",
        "",
        "| method | dur s | acc residual m/s^2 | gyro residual rad/s | pos RMSE m | final pos m | rot RMSE deg | final rot deg | vel RMSE m/s | first ba | mean ba |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in summaries:
        lines.append(
            "| {method} | {duration_s:.3f} | {accel_specific_rmse_m_s2:.4f} | {gyro_specific_rmse_rad_s:.5f} | "
            "{position_rmse_m:.4f} | {position_final_m:.4f} | {rotation_rmse_deg:.3f} | {rotation_final_deg:.3f} | "
            "{velocity_rmse_m_s:.4f} | `{first_ba}` | `{mean_ba}` |".format(
                **{
                    **item,
                    "first_ba": [round(float(x), 4) for x in item["first_ba"]],
                    "mean_ba": [round(float(x), 4) for x in item["mean_ba"]],
                }
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `pseudo_gt_from_gt` is the direct raw-minus-GT kinematic bias over the same samples. It is a sanity upper bound because it uses GT at every sample.",
            "- If a bias source is consistent with this GT/IMU propagation model, both its accelerometer residual and inertial rollout error should be close to `pseudo_gt_from_gt`.",
            "- A small OpenVINS initializer bias is not necessarily wrong for OpenVINS' self-initialized filter frame, but if used directly with the exported mocap/body GT frame it is not the same quantity as the Debias_IMU pseudo-bias.",
        ]
    )
    (path / "bias_consistency_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "LIASuw"
        / "phys_delta_vo_framefixed_clean_imu"
        / "0205s14_80",
    )
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--gravity-mag", type=float, default=9.7803)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir
    raw_t, raw_imu = read_imu_csv(sequence_dir / "methods" / "raw" / "mav0" / "imu0" / "data.csv")
    gt = GTTrajectory(sequence_dir / "groundtruth" / "body_gt_openvins_asl.csv", args.gravity_mag)
    net_t, net_bias = read_bias_csv(sequence_dir / "biases" / "network_bias_interpolated_to_raw_imu.csv")
    const = json.loads((sequence_dir / "biases" / "constant_bias.json").read_text(encoding="utf-8"))
    const_bias = np.asarray(const["bw_const_rad_s"] + const["ba_const_m_s2"], dtype=np.float64)
    ov_est = read_openvins_estimate(sequence_dir / "runs" / "nogt_dyn_raw_baseline" / "estimate.txt")
    ov_init_bias = np.concatenate([ov_est["bg"][0], ov_est["ba"][0]])
    start_s = float(ov_est["t"][0])

    t = make_time_grid(raw_t, start_s, args.duration_s)
    imu = interp_columns(raw_t, raw_imu, t)
    net = interp_columns(net_t, net_bias, t)
    expected_w = gt.expected_body_rate(t)
    expected_f = gt.expected_specific_force_body(t)
    pseudo_gt = np.column_stack([imu[:, :3] - expected_w, imu[:, 3:6] - expected_f])

    modes: dict[str, np.ndarray] = {
        "zero_bias": np.zeros((t.shape[0], 6), dtype=np.float64),
        "openvins_init_constant": np.broadcast_to(ov_init_bias.reshape(1, 6), (t.shape[0], 6)),
        "debias_constant": np.broadcast_to(const_bias.reshape(1, 6), (t.shape[0], 6)),
        "debias_network": net,
        "pseudo_gt_from_gt": pseudo_gt,
    }
    summaries = [
        summarize_mode(name, t, imu, bias[:, :3], bias[:, 3:6], gt)
        for name, bias in modes.items()
    ]

    output_dir = args.output_dir or sequence_dir / "runs" / f"bias_consistency_{int(round(args.duration_s))}s"
    metadata = {
        "sequence_dir": str(sequence_dir),
        "start_s": start_s,
        "duration_s": args.duration_s,
        "gravity_mag": args.gravity_mag,
        "raw_imu": str(sequence_dir / "methods" / "raw" / "mav0" / "imu0" / "data.csv"),
        "gt": str(sequence_dir / "groundtruth" / "body_gt_openvins_asl.csv"),
        "openvins_init_bias_bg_ba": ov_init_bias.tolist(),
        "constant_bias_bg_ba": const_bias.tolist(),
    }
    write_outputs(output_dir, summaries, metadata)
    print(output_dir / "bias_consistency_summary.md")
    for item in summaries:
        print(
            f"{item['method']}: acc_res={item['accel_specific_rmse_m_s2']:.4f} "
            f"pos_rmse={item['position_rmse_m']:.4f} rot_rmse={item['rotation_rmse_deg']:.3f} "
            f"vel_rmse={item['velocity_rmse_m_s']:.4f}"
        )


if __name__ == "__main__":
    main()
