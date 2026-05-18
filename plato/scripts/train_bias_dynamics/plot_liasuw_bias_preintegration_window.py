#!/usr/bin/env python3
"""Plot a 32-sample LIASuw pseudo-bias preintegration check as draggable HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = PLATO_ROOT / "results" / "debias_imu" / "cache" / "LIASuw_AD_e1800_w16_b64"
DEFAULT_DATA_ROOT = PLATO_ROOT / "results" / "debias_imu" / "LIASuw_training_candidates_200hz"
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "results" / "debias_imu" / "preintegration_checks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["csv", "cache"], default="csv")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sequence", default="0912s10_40")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument(
        "--start-index",
        default="auto",
        help="Window start index, or 'auto' to choose the 32-sample window with largest GT displacement.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, 0.0, -9.81])
    parser.add_argument(
        "--position-update",
        choices=["euler", "second_order"],
        default="euler",
        help="Discrete position update for the estimated trajectory.",
    )
    return parser.parse_args()


def tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            return pickle.load(f)


def load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def hat(w: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ],
        dtype=float,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = hat(w)
    if theta < 1e-10:
        return np.eye(3) + W
    return np.eye(3) + math.sin(theta) / theta * W + (1.0 - math.cos(theta)) / (theta * theta) * (W @ W)


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-10:
        return np.array(
            [
                0.5 * (R[2, 1] - R[1, 2]),
                0.5 * (R[0, 2] - R[2, 0]),
                0.5 * (R[1, 0] - R[0, 1]),
            ]
        )
    return theta / (2.0 * math.sin(theta)) * np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )


def choose_start_index(p_gt: np.ndarray, window: int) -> int:
    if p_gt.shape[0] < window:
        raise ValueError(f"trajectory length {p_gt.shape[0]} is shorter than window {window}")
    displacements = np.linalg.norm(p_gt[window - 1 :] - p_gt[: p_gt.shape[0] - window + 1], axis=1)
    return int(np.argmax(displacements))


def moving_average_reflect(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data.copy()
    if window % 2 == 0:
        raise ValueError("moving average window should be odd")
    pad = window // 2
    padded = np.pad(data, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(data.shape[1])])


def savgol_columns(data: np.ndarray, window: int, order: int, deriv: int = 0) -> np.ndarray:
    if window <= 1:
        return data.copy()
    if data.shape[0] < window:
        return moving_average_reflect(data, min(data.shape[0] // 2 * 2 + 1, window))
    return np.column_stack([savgol_filter(data[:, i], window, order, deriv=deriv) for i in range(data.shape[1])])


def load_from_cache(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    imu_dict = load_pickle(args.cache_dir / f"{args.sequence}.p")
    gt_dict = load_pickle(args.cache_dir / f"{args.sequence}_gt.p")
    t = tensor_to_numpy(imu_dict["t_imu"]).astype(float)
    u_imu = tensor_to_numpy(imu_dict["u_imu"]).astype(float)
    bg = tensor_to_numpy(imu_dict["w_bias"]).astype(float)
    ba = tensor_to_numpy(imu_dict["a_bias"]).astype(float)
    R_gt = tensor_to_numpy(gt_dict["R_gt"]).astype(float)
    v_gt = tensor_to_numpy(gt_dict["v_gt"]).astype(float)
    p_gt = tensor_to_numpy(gt_dict["p_gt"]).astype(float)
    return t, u_imu, bg, ba, R_gt, v_gt, p_gt, f"cache:{args.cache_dir}"


def load_from_csv(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    seq_root = args.data_root / args.sequence / "mav0"
    imu_csv = seq_root / "imu0" / "data.csv"
    gt_csv = seq_root / "mocap0" / "data.csv"
    imu = load_csv(imu_csv)
    gt = load_csv(gt_csv)
    if imu.shape[0] != gt.shape[0] or np.max(np.abs(imu[:, 0].astype(np.int64) - gt[:, 0].astype(np.int64))) != 0:
        raise ValueError(f"IMU and GT timestamps are not exactly aligned for {args.sequence}")

    t = (imu[:, 0] - imu[0, 0]) / 1e9
    u_imu = imu[:, 1:7].astype(float)
    p_gt = gt[:, 1:4].astype(float)
    p_gt = p_gt - p_gt[0]
    R_gt = np.stack([quat_wxyz_to_R(row) for row in gt[:, 4:8]], axis=0)

    # Match the Debias_IMU TUM loader's velocity and pseudo-bias construction,
    # but read the raw aggregate CSV files directly instead of loading cache.
    dt_median = float(np.median(np.diff(t)))
    v_gt = savgol_columns(p_gt, 21, 5, deriv=1) / dt_median

    dt = np.diff(t)[:, None]
    bg = np.zeros_like(u_imu[:, :3])
    ba = np.zeros_like(u_imu[:, 3:6])
    for i in range(len(t) - 1):
        bg[i] = u_imu[i, :3] - so3_log(R_gt[i].T @ R_gt[i + 1]) / dt[i, 0]
        acc_world_minus_g = (v_gt[i + 1] - v_gt[i]) / dt[i, 0] - np.asarray(args.gravity, dtype=float)
        ba[i] = u_imu[i, 3:6] - R_gt[i].T @ acc_world_minus_g
    bg[-1] = bg[-2]
    ba[-1] = ba[-2]
    bg = savgol_columns(bg, 15, 3, deriv=0)
    ba = savgol_columns(ba, 15, 3, deriv=0)
    return t, u_imu, bg, ba, R_gt, v_gt, p_gt, f"csv:{seq_root}"


def propagate_window(
    t: np.ndarray,
    u_imu: np.ndarray,
    bg: np.ndarray,
    ba: np.ndarray,
    R_gt: np.ndarray,
    v_gt: np.ndarray,
    p_gt: np.ndarray,
    start: int,
    window: int,
    gravity: np.ndarray,
    position_update: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_est = np.zeros((window, 3, 3))
    v_est = np.zeros((window, 3))
    p_est = np.zeros((window, 3))
    R_est[0] = R_gt[start]
    v_est[0] = v_gt[start]
    p_est[0] = p_gt[start]

    for local_idx in range(window - 1):
        k = start + local_idx
        dt = float(t[k + 1] - t[k])
        omega = u_imu[k, :3] - bg[k]
        acc_body = u_imu[k, 3:6] - ba[k]
        acc_world = R_est[local_idx] @ acc_body + gravity

        R_est[local_idx + 1] = R_est[local_idx] @ so3_exp(omega * dt)
        v_est[local_idx + 1] = v_est[local_idx] + acc_world * dt
        if position_update == "second_order":
            p_est[local_idx + 1] = p_est[local_idx] + v_est[local_idx] * dt + 0.5 * acc_world * dt * dt
        else:
            p_est[local_idx + 1] = p_est[local_idx] + v_est[local_idx] * dt
    return R_est, v_est, p_est


def metrics(R_gt: np.ndarray, v_gt: np.ndarray, p_gt: np.ndarray, R_est: np.ndarray, v_est: np.ndarray, p_est: np.ndarray) -> dict:
    pos_err = p_est - p_gt
    vel_err = v_est - v_gt
    rot_err = np.array([np.linalg.norm(so3_log(R_gt[i].T @ R_est[i])) for i in range(R_gt.shape[0])])
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(pos_err * pos_err, axis=1)))),
        "position_max_m": float(np.max(np.linalg.norm(pos_err, axis=1))),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(np.sum(vel_err * vel_err, axis=1)))),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rot_err * rot_err)) * 180.0 / math.pi),
        "rotation_max_deg": float(np.max(rot_err) * 180.0 / math.pi),
    }


def make_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    title = html.escape(payload["title"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; font-family: Arial, sans-serif; background: #111318; color: #eef1f6; }}
    #plot {{ width: 100vw; height: 100vh; display: block; cursor: grab; }}
    #plot:active {{ cursor: grabbing; }}
    #panel {{ position: fixed; left: 16px; top: 14px; max-width: min(560px, calc(100vw - 32px)); padding: 12px 14px; background: rgba(17, 19, 24, 0.82); border: 1px solid rgba(255,255,255,0.18); border-radius: 8px; line-height: 1.38; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
    #panel h1 {{ margin: 0 0 8px; font-size: 16px; font-weight: 700; }}
    #panel .row {{ display: flex; gap: 12px; flex-wrap: wrap; font-size: 12px; color: #c8ced8; }}
    #panel .legend {{ margin-top: 9px; display: flex; gap: 14px; font-size: 13px; }}
    .swatch {{ width: 22px; height: 3px; display: inline-block; vertical-align: middle; margin-right: 6px; }}
    .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; vertical-align: middle; margin-right: 6px; }}
    #charts {{ position: fixed; left: 16px; bottom: 14px; width: min(740px, calc(100vw - 32px)); padding: 10px 12px 12px; background: rgba(17, 19, 24, 0.84); border: 1px solid rgba(255,255,255,0.18); border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
    .chart-row {{ display: grid; grid-template-columns: 118px 1fr; gap: 10px; align-items: center; }}
    .chart-row + .chart-row {{ margin-top: 8px; }}
    .chart-title {{ font-size: 12px; color: #dbe1ec; line-height: 1.25; }}
    .chart-title span {{ display: block; color: #aeb6c4; font-size: 11px; margin-top: 2px; }}
    .bias-chart {{ width: 100%; height: 116px; display: block; }}
    .axis-legend {{ margin-top: 8px; display: flex; gap: 14px; font-size: 12px; color: #c8ced8; }}
    #hint {{ position: fixed; right: 14px; bottom: 12px; color: #aeb6c4; font-size: 12px; background: rgba(17, 19, 24, 0.72); padding: 7px 9px; border-radius: 8px; }}
  </style>
</head>
<body>
<canvas id="plot"></canvas>
<div id="panel">
  <h1>{title}</h1>
  <div class="row" id="meta"></div>
  <div class="legend">
    <span><span class="swatch" style="background:#57d68d"></span>GT position</span>
    <span><span class="swatch" style="background:#ff6b6b"></span>preintegrated position</span>
    <span><span class="swatch" style="background:#f2b84b"></span>constant bias position</span>
    <span><span class="dot" style="background:#eef1f6"></span>sample points</span>
  </div>
</div>
<div id="charts">
  <div class="chart-row">
    <div class="chart-title">gyro bias `b_g`<span>rad/s</span></div>
    <canvas class="bias-chart" id="bgChart"></canvas>
  </div>
  <div class="chart-row">
    <div class="chart-title">accel bias `b_a`<span>m/s^2</span></div>
    <canvas class="bias-chart" id="baChart"></canvas>
  </div>
  <div class="axis-legend">
    <span><span class="swatch" style="background:#ff6b6b"></span>x</span>
    <span><span class="swatch" style="background:#57d68d"></span>y</span>
    <span><span class="swatch" style="background:#70a7ff"></span>z</span>
    <span>solid: pseudo bias · dashed: constant start bias</span>
  </div>
</div>
<div id="hint">drag: rotate · wheel: zoom · double click: reset</div>
<script>
const DATA = {data_json};
const canvas = document.getElementById('plot');
const ctx = canvas.getContext('2d');
const AXIS_COLORS = ['#ff6b6b', '#57d68d', '#70a7ff'];
let yaw = -0.65, pitch = 0.48, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0;

document.getElementById('meta').innerHTML = [
  `seq ${{DATA.sequence}}`,
  `start ${{DATA.start_index}}`,
  `N ${{DATA.window}}`,
  `dt median ${{DATA.dt_median_s.toFixed(6)}} s`,
  `pos RMSE ${{DATA.metrics.position_rmse_m.toExponential(3)}} m`,
  `const pos RMSE ${{DATA.constant_metrics.position_rmse_m.toExponential(3)}} m`,
  `rot RMSE ${{DATA.metrics.rotation_rmse_deg.toExponential(3)}} deg`
].join(' · ');

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
  drawBiasCharts();
}}

function allPoints() {{
  return DATA.gt.concat(DATA.est).concat(DATA.const_est);
}}

function bounds() {{
  const pts = allPoints();
  const mn = [Infinity, Infinity, Infinity];
  const mx = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) {{
    for (let i = 0; i < 3; i++) {{ mn[i] = Math.min(mn[i], p[i]); mx[i] = Math.max(mx[i], p[i]); }}
  }}
  const center = [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];
  const span = Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 1e-4);
  return {{center, span}};
}}

function project(p, b) {{
  let x = p[0] - b.center[0], y = p[1] - b.center[1], z = p[2] - b.center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = cp * y - sp * z1;
  const scale = zoom * 0.72 * Math.min(window.innerWidth, window.innerHeight) / b.span;
  return [window.innerWidth / 2 + x1 * scale, window.innerHeight / 2 - y1 * scale];
}}

function drawPolyline(points, color, width, b) {{
  ctx.beginPath();
  points.forEach((p, i) => {{
    const q = project(p, b);
    if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
  }});
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}}

function drawPoint(p, color, radius, b) {{
  const q = project(p, b);
  ctx.beginPath();
  ctx.arc(q[0], q[1], radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.lineWidth = 1;
  ctx.stroke();
}}

function drawSamplePoints(points, color, radius, b) {{
  for (const p of points) {{
    drawPoint(p, color, radius, b);
  }}
}}

function drawAxes(b) {{
  const origin = b.center;
  const len = b.span * 0.42;
  const axes = [
    [[origin[0], origin[1], origin[2]], [origin[0] + len, origin[1], origin[2]], '#e36a6a', 'x'],
    [[origin[0], origin[1], origin[2]], [origin[0], origin[1] + len, origin[2]], '#6fd17c', 'y'],
    [[origin[0], origin[1], origin[2]], [origin[0], origin[1], origin[2] + len], '#70a7ff', 'z']
  ];
  ctx.font = '12px Arial';
  for (const [a, c, color, label] of axes) {{
    drawPolyline([a, c], color, 1.5, b);
    const q = project(c, b);
    ctx.fillStyle = color;
    ctx.fillText(label, q[0] + 5, q[1] - 5);
  }}
}}

function draw() {{
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  const grad = ctx.createLinearGradient(0, 0, 0, window.innerHeight);
  grad.addColorStop(0, '#151923');
  grad.addColorStop(1, '#0c0e12');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
  const b = bounds();
  drawAxes(b);
  drawPolyline(DATA.gt, '#57d68d', 3.2, b);
  drawPolyline(DATA.est, '#ff6b6b', 2.4, b);
  drawPolyline(DATA.const_est, '#f2b84b', 2.2, b);
  drawSamplePoints(DATA.gt, '#57d68d', 2.4, b);
  drawSamplePoints(DATA.est, '#ff6b6b', 2.2, b);
  drawSamplePoints(DATA.const_est, '#f2b84b', 2.0, b);
  drawPoint(DATA.gt[0], '#ffffff', 5, b);
  drawPoint(DATA.gt[DATA.gt.length - 1], '#57d68d', 5, b);
  drawPoint(DATA.est[DATA.est.length - 1], '#ff6b6b', 5, b);
  drawPoint(DATA.const_est[DATA.const_est.length - 1], '#f2b84b', 5, b);
}}

function formatTick(value) {{
  const av = Math.abs(value);
  if (av > 999 || (av > 0 && av < 0.01)) return value.toExponential(1);
  return value.toFixed(av >= 10 ? 1 : 3);
}}

function drawBiasChart(canvasId, values, label) {{
  const c = document.getElementById(canvasId);
  const cctx = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.max(1, Math.floor(rect.width * dpr));
  c.height = Math.max(1, Math.floor(rect.height * dpr));
  cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  const margin = {{left: 50, right: 10, top: 12, bottom: 28}};
  const xs = DATA.time_rel;
  const constantValues = label === 'b_g' ? DATA.bg_const : DATA.ba_const;
  let minY = Infinity, maxY = -Infinity;
  for (const row of values) {{
    for (const y of row) {{ minY = Math.min(minY, y); maxY = Math.max(maxY, y); }}
  }}
  for (const y of constantValues) {{
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }}
  if (minY === maxY) {{ minY -= 1; maxY += 1; }}
  const pad = Math.max((maxY - minY) * 0.12, 1e-9);
  minY -= pad; maxY += pad;
  const minX = xs[0], maxX = xs[xs.length - 1];
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  const xScale = x => margin.left + (x - minX) / Math.max(maxX - minX, 1e-9) * plotW;
  const yScale = y => margin.top + (maxY - y) / Math.max(maxY - minY, 1e-9) * plotH;

  cctx.clearRect(0, 0, w, h);
  cctx.fillStyle = 'rgba(255,255,255,0.035)';
  cctx.fillRect(0, 0, w, h);
  cctx.strokeStyle = 'rgba(255,255,255,0.18)';
  cctx.lineWidth = 1;
  cctx.strokeRect(margin.left, margin.top, plotW, plotH);
  cctx.fillStyle = '#9ea8b8';
  cctx.font = '11px Arial';
  cctx.textAlign = 'right';
  cctx.fillText(formatTick(maxY), margin.left - 6, margin.top + 8);
  cctx.fillText(formatTick(minY), margin.left - 6, margin.top + plotH);
  cctx.textAlign = 'left';
  cctx.fillText('0s', margin.left, h - 8);
  cctx.textAlign = 'right';
  cctx.fillText(`${{(maxX - minX).toFixed(3)}}s`, margin.left + plotW, h - 8);
  cctx.textAlign = 'left';
  cctx.fillText(label, margin.left + 4, margin.top + 12);

  for (let axis = 0; axis < 3; axis++) {{
    cctx.beginPath();
    for (let i = 0; i < values.length; i++) {{
      const x = xScale(xs[i]);
      const y = yScale(values[i][axis]);
      if (i === 0) cctx.moveTo(x, y); else cctx.lineTo(x, y);
    }}
    cctx.strokeStyle = AXIS_COLORS[axis];
    cctx.lineWidth = 1.8;
    cctx.stroke();
    for (let i = 0; i < values.length; i++) {{
      const x = xScale(xs[i]);
      const y = yScale(values[i][axis]);
      cctx.beginPath();
      cctx.arc(x, y, 1.8, 0, Math.PI * 2);
      cctx.fillStyle = AXIS_COLORS[axis];
      cctx.fill();
    }}
  }}
  cctx.save();
  cctx.setLineDash([6, 4]);
  cctx.lineWidth = 1.1;
  cctx.globalAlpha = 0.72;
  for (let axis = 0; axis < 3; axis++) {{
    const y = yScale(constantValues[axis]);
    cctx.beginPath();
    cctx.moveTo(margin.left, y);
    cctx.lineTo(margin.left + plotW, y);
    cctx.strokeStyle = AXIS_COLORS[axis];
    cctx.stroke();
  }}
  cctx.restore();
}}

function drawBiasCharts() {{
  drawBiasChart('bgChart', DATA.bg, 'b_g');
  drawBiasChart('baChart', DATA.ba, 'b_a');
}}

canvas.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  yaw += dx * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
  draw();
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom *= Math.exp(-e.deltaY * 0.001);
  zoom = Math.max(0.2, Math.min(20, zoom));
  draw();
}}, {{passive: false}});
canvas.addEventListener('dblclick', () => {{ yaw = -0.65; pitch = 0.48; zoom = 1.0; draw(); }});
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    if args.source == "cache":
        t, u_imu, bg, ba, R_gt, v_gt, p_gt, source_description = load_from_cache(args)
    else:
        t, u_imu, bg, ba, R_gt, v_gt, p_gt, source_description = load_from_csv(args)

    n = min(len(t), len(u_imu), len(bg), len(ba), len(R_gt), len(v_gt), len(p_gt))
    t, u_imu, bg, ba, R_gt, v_gt, p_gt = t[:n], u_imu[:n], bg[:n], ba[:n], R_gt[:n], v_gt[:n], p_gt[:n]
    if args.window < 2:
        raise ValueError("--window must be at least 2")
    if args.window > n:
        raise ValueError(f"--window {args.window} exceeds trajectory length {n}")

    if args.start_index == "auto":
        start = choose_start_index(p_gt, args.window)
    else:
        start = int(args.start_index)
    if start < 0 or start + args.window > n:
        raise ValueError(f"start index {start} with window {args.window} is outside trajectory length {n}")

    gravity = np.asarray(args.gravity, dtype=float)
    R_est, v_est, p_est = propagate_window(
        t,
        u_imu,
        bg,
        ba,
        R_gt,
        v_gt,
        p_gt,
        start,
        args.window,
        gravity,
        args.position_update,
    )
    gt_slice = slice(start, start + args.window)
    metric_values = metrics(R_gt[gt_slice], v_gt[gt_slice], p_gt[gt_slice], R_est, v_est, p_est)
    bg_constant = np.repeat(bg[start : start + 1], n, axis=0)
    ba_constant = np.repeat(ba[start : start + 1], n, axis=0)
    R_const, v_const, p_const = propagate_window(
        t,
        u_imu,
        bg_constant,
        ba_constant,
        R_gt,
        v_gt,
        p_gt,
        start,
        args.window,
        gravity,
        args.position_update,
    )
    constant_metric_values = metrics(
        R_gt[gt_slice], v_gt[gt_slice], p_gt[gt_slice], R_const, v_const, p_const
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.sequence}_start{start}_n{args.window}_{args.source}_{args.position_update}.html"
    payload = {
        "title": f"LIASuw bias-GT preintegration check: {args.sequence}",
        "sequence": args.sequence,
        "start_index": start,
        "window": args.window,
        "position_update": args.position_update,
        "source": args.source,
        "source_description": source_description,
        "dt_median_s": float(np.median(np.diff(t[gt_slice]))),
        "t0_s": float(t[start]),
        "metrics": metric_values,
        "constant_metrics": constant_metric_values,
        "time_rel": (t[gt_slice] - t[start]).tolist(),
        "bg": bg[gt_slice].tolist(),
        "ba": ba[gt_slice].tolist(),
        "bg_const": bg[start].tolist(),
        "ba_const": ba[start].tolist(),
        "gt": p_gt[gt_slice].tolist(),
        "est": p_est.tolist(),
        "const_est": p_const.tolist(),
    }
    output_path.write_text(make_html(payload), encoding="utf-8")

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"sequence: {args.sequence}")
    print(f"source: {source_description}")
    print(f"start_index: {start}")
    print(f"window: {args.window}")
    print(f"position_update: {args.position_update}")
    print("pseudo_bias_metrics:")
    for key, value in metric_values.items():
        print(f"  {key}: {value:.12g}")
    print("constant_bias_metrics:")
    for key, value in constant_metric_values.items():
        print(f"  {key}: {value:.12g}")
    print(f"html: {output_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
