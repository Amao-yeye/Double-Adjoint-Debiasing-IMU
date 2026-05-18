#!/usr/bin/env python3
"""Write draggable 3D and orthographic HTML trajectory views for corldata."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEQUENCES = None
DEFAULT_VARIANTS = ["evo_gt_aligned"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES, help="Sequences to render. Defaults to all sequences with available variant outputs.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="Trajectory variants to render. Use 'aligned' for the original associated evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "interactive_3d",
    )
    return parser.parse_args()


def seq_root(plato_root: Path, sequence: str) -> Path:
    return plato_root / "results" / "pseudo_gt" / "corldata" / sequence


def discover_sequences(plato_root: Path, variants: list[str]) -> list[str]:
    root = plato_root / "results" / "pseudo_gt" / "corldata"
    sequences = []
    if not root.exists():
        return sequences
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith("S"):
            continue
        if any(variant_paths(plato_root, path.name, variant)[0].exists() for variant in variants):
            sequences.append(path.name)
    return sequences


def variant_paths(plato_root: Path, sequence: str, variant: str) -> tuple[Path, Path]:
    root = seq_root(plato_root, sequence)
    if variant == "aligned":
        return root / "aligned" / "aligned_poses.csv", root / "metrics.json"
    return root / variant / "aligned_poses.csv", root / variant / "metrics.json"


def load_aligned_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    if len(rows) < 2:
        raise RuntimeError(f"{path} has too few rows")
    return rows


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def median_dt(times: list[float]) -> float:
    deltas = sorted(
        times[idx] - times[idx - 1]
        for idx in range(1, len(times))
        if times[idx] > times[idx - 1]
    )
    if not deltas:
        return 0.05
    return deltas[len(deltas) // 2]


def make_payload(sequence: str, variant: str, rows: list[dict[str, float]], metrics: dict[str, object]) -> dict[str, object]:
    gt = [[row["gt_x"], row["gt_y"], row["gt_z"]] for row in rows]
    est = [[row["est_x"], row["est_y"], row["est_z"]] for row in rows]
    times = [row["timestamp"] for row in rows]
    rel_times = [timestamp - times[0] for timestamp in times]
    position_errors = [row["position_error_m"] for row in rows]
    orientation_errors = [row["orientation_error_deg"] for row in rows]
    all_points = gt + est
    bounds = {
        "min": [min(point[axis] for point in all_points) for axis in range(3)],
        "max": [max(point[axis] for point in all_points) for axis in range(3)],
    }
    center = [(bounds["min"][axis] + bounds["max"][axis]) * 0.5 for axis in range(3)]
    radius = max(
        math.sqrt(sum((point[axis] - center[axis]) ** 2 for axis in range(3)))
        for point in all_points
    )
    if radius <= 0.0:
        radius = 1.0

    summary = {
        "pose_count": len(rows),
        "time_span_s": rel_times[-1] - rel_times[0],
        "position_rmse_m": float(metrics.get("position_rmse_m", rmse(position_errors))),
        "position_max_m": float(metrics.get("position_max_m", max(position_errors))),
        "orientation_rmse_deg": float(metrics.get("orientation_rmse_deg", rmse(orientation_errors))),
        "orientation_max_deg": float(metrics.get("orientation_max_deg", max(orientation_errors))),
        "time_offset_s": metrics.get("time_offset_s"),
        "alignment": metrics.get("alignment", "SE(3) no-scale alignment"),
    }
    if "associated_pose_count_after_mask" in metrics:
        summary["associated_after_mask"] = metrics["associated_pose_count_after_mask"]
        summary["associated_original"] = metrics.get("associated_pose_count_original")
        summary["masked_associated_pose_count"] = metrics.get("masked_associated_pose_count")
        summary["mask_sources"] = metrics.get("mask_sources")

    return {
        "sequence": sequence,
        "variant": variant,
        "title": f"corldata {sequence} {variant} 3D trajectory",
        "gt": gt,
        "est": est,
        "times": rel_times,
        "positionErrors": position_errors,
        "orientationErrors": orientation_errors,
        "bounds": bounds,
        "center": center,
        "radius": radius,
        "breakGapS": max(0.2, median_dt(times) * 5.0),
        "summary": summary,
    }


def make_html(payload: dict[str, object]) -> str:
    title = html.escape(str(payload["title"]))
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
html, body {{
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #101216;
  color: #edf1f7;
  font-family: Arial, Helvetica, sans-serif;
}}
#view {{
  display: block;
  width: 100vw;
  height: 100vh;
  cursor: grab;
  touch-action: none;
}}
#view.dragging {{ cursor: grabbing; }}
#panel {{
  position: fixed;
  left: 16px;
  top: 16px;
  max-width: min(420px, calc(100vw - 32px));
  background: rgba(18, 22, 29, 0.86);
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 8px;
  padding: 14px 16px;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.32);
  backdrop-filter: blur(8px);
}}
h1 {{
  margin: 0 0 8px;
  font-size: 17px;
  line-height: 1.25;
  letter-spacing: 0;
}}
.meta {{
  display: grid;
  grid-template-columns: auto auto;
  gap: 4px 14px;
  margin: 8px 0 10px;
  font-size: 12px;
  color: #cad2df;
}}
.legend {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
  font-size: 12px;
  margin: 8px 0 10px;
}}
.swatch {{
  display: inline-block;
  width: 22px;
  height: 3px;
  margin-right: 6px;
  vertical-align: middle;
  border-radius: 999px;
}}
.gt {{ background: #4ea1ff; }}
.est {{ background: #ff8a24; }}
.err {{ background: #f15c5c; }}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}}
button, label {{
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.08);
  color: #edf1f7;
  font-size: 12px;
  line-height: 1;
}}
button {{
  padding: 8px 10px;
  cursor: pointer;
}}
label {{
  padding: 7px 9px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
button:hover, label:hover {{ background: rgba(255, 255, 255, 0.14); }}
input {{ margin: 0; }}
#hint {{
  position: fixed;
  right: 16px;
  bottom: 14px;
  color: rgba(237, 241, 247, 0.66);
  font-size: 12px;
  text-align: right;
  user-select: none;
}}
#projectionPanel {{
  position: fixed;
  right: 16px;
  top: 16px;
  width: min(360px, calc(100vw - 468px));
  min-width: 300px;
  display: grid;
  gap: 10px;
}}
.projection {{
  background: rgba(18, 22, 29, 0.86);
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(8px);
}}
.projectionTitle {{
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 10px;
  font-size: 12px;
  color: #dbe4f2;
  border-bottom: 1px solid rgba(255, 255, 255, 0.10);
}}
.projectionTitle span:last-child {{
  color: #96a3b7;
}}
.projection canvas {{
  display: block;
  width: 100%;
  height: 152px;
}}
@media (max-width: 980px) {{
  #projectionPanel {{
    left: 8px;
    right: 8px;
    top: auto;
    bottom: 8px;
    width: auto;
    min-width: 0;
    grid-template-columns: repeat(3, minmax(190px, 1fr));
    overflow-x: auto;
  }}
  .projection canvas {{ height: 126px; }}
  #hint {{ display: none; }}
}}
</style>
</head>
<body>
<canvas id="view"></canvas>
<section id="panel">
  <h1 id="title"></h1>
  <div class="legend">
    <span><span class="swatch gt"></span>Ground truth</span>
    <span><span class="swatch est"></span>Estimated VO</span>
    <span><span class="swatch err"></span>Start/end markers</span>
  </div>
  <div class="meta" id="meta"></div>
  <div class="controls">
    <button id="reset">Reset</button>
    <button id="top">Top XY</button>
    <button id="side">Side</button>
    <label><input id="showGt" type="checkbox" checked>GT</label>
    <label><input id="showEst" type="checkbox" checked>VO</label>
    <label><input id="showGrid" type="checkbox" checked>Grid</label>
  </div>
</section>
<section id="projectionPanel" aria-label="orthographic trajectory projections">
  <div class="projection">
    <div class="projectionTitle"><span>XY view</span><span>X/Y in m</span></div>
    <canvas id="xyView" data-plane="xy"></canvas>
  </div>
  <div class="projection">
    <div class="projectionTitle"><span>YZ view</span><span>Y/Z in m</span></div>
    <canvas id="yzView" data-plane="yz"></canvas>
  </div>
  <div class="projection">
    <div class="projectionTitle"><span>XZ view</span><span>X/Z in m</span></div>
    <canvas id="xzView" data-plane="xz"></canvas>
  </div>
</section>
<div id="hint">Drag to rotate view<br>Wheel to zoom</div>
<script>
const DATA = {data_json};
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const meta = document.getElementById("meta");
const projectionCanvases = Array.from(document.querySelectorAll(".projection canvas"));
const PROJECTION_PLANES = {{
  xy: {{ axes: [0, 1], labels: ["X", "Y"] }},
  yz: {{ axes: [1, 2], labels: ["Y", "Z"] }},
  xz: {{ axes: [0, 2], labels: ["X", "Z"] }}
}};
const state = {{
  yaw: -0.68,
  pitch: 0.78,
  zoom: 1.0,
  showGt: true,
  showEst: true,
  showGrid: true
}};
let dragging = false;
let lastX = 0;
let lastY = 0;
let dpr = 1;

document.getElementById("title").textContent = DATA.title;
function fmt(value, digits) {{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}}
function addMeta(label, value) {{
  const k = document.createElement("div");
  const v = document.createElement("div");
  k.textContent = label;
  v.textContent = value;
  meta.appendChild(k);
  meta.appendChild(v);
}}
addMeta("poses", String(DATA.summary.pose_count));
addMeta("span", fmt(DATA.summary.time_span_s, 2) + " s");
addMeta("pos RMSE", fmt(DATA.summary.position_rmse_m, 4) + " m");
addMeta("pos max", fmt(DATA.summary.position_max_m, 4) + " m");
addMeta("ang RMSE", fmt(DATA.summary.orientation_rmse_deg, 3) + " deg");
addMeta("offset", fmt(DATA.summary.time_offset_s, 3) + " s");
if (DATA.summary.masked_associated_pose_count !== undefined) {{
  addMeta("masked poses", String(DATA.summary.masked_associated_pose_count));
}}

function resize() {{
  dpr = window.devicePixelRatio || 1;
  const width = Math.floor(window.innerWidth * dpr);
  const height = Math.floor(window.innerHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
  }}
  for (const projectionCanvas of projectionCanvases) {{
    const rect = projectionCanvas.getBoundingClientRect();
    const pWidth = Math.max(1, Math.floor(rect.width * dpr));
    const pHeight = Math.max(1, Math.floor(rect.height * dpr));
    if (projectionCanvas.width !== pWidth || projectionCanvas.height !== pHeight) {{
      projectionCanvas.width = pWidth;
      projectionCanvas.height = pHeight;
    }}
  }}
  render();
}}

function normalize(p) {{
  return [
    (p[0] - DATA.center[0]) / DATA.radius,
    (p[1] - DATA.center[1]) / DATA.radius,
    (p[2] - DATA.center[2]) / DATA.radius
  ];
}}

function rotatePoint(p) {{
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const x1 = cy * p[0] - sy * p[1];
  const y1 = sy * p[0] + cy * p[1];
  const z1 = p[2];
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  return [x1, y2, z2];
}}

function projectRaw(p) {{
  const r = rotatePoint(normalize(p));
  const scale = Math.min(canvas.width, canvas.height) * 0.42 * state.zoom;
  return {{
    x: canvas.width * 0.5 + r[0] * scale,
    y: canvas.height * 0.54 - r[2] * scale,
    depth: r[1]
  }};
}}

function drawLine(a, b, color, width, dash) {{
  const pa = projectRaw(a);
  const pb = projectRaw(b);
  ctx.strokeStyle = color;
  ctx.lineWidth = width * dpr;
  ctx.setLineDash(dash.map(v => v * dpr));
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
  ctx.setLineDash([]);
}}

function drawPolyline(points, times, color, width, dash) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = width * dpr;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash(dash.map(v => v * dpr));
  ctx.beginPath();
  let open = false;
  for (let i = 0; i < points.length; i++) {{
    const p = projectRaw(points[i]);
    if (i === 0 || times[i] - times[i - 1] > DATA.breakGapS) {{
      if (open) ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      open = true;
    }} else {{
      ctx.lineTo(p.x, p.y);
    }}
  }}
  if (open) ctx.stroke();
  ctx.setLineDash([]);
}}

function drawMarker(point, color, radius, label) {{
  const p = projectRaw(point);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius * dpr, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = `${{12 * dpr}}px Arial`;
  ctx.fillText(label, p.x + 7 * dpr, p.y - 7 * dpr);
}}

function drawLabel(point, text, color) {{
  const p = projectRaw(point);
  ctx.fillStyle = color;
  ctx.font = `${{12 * dpr}}px Arial`;
  ctx.fillText(text, p.x + 5 * dpr, p.y - 5 * dpr);
}}

function drawGrid() {{
  const min = DATA.bounds.min;
  const max = DATA.bounds.max;
  const z = min[2];
  const steps = 6;
  ctx.globalAlpha = 0.34;
  for (let i = 0; i <= steps; i++) {{
    const tx = min[0] + (max[0] - min[0]) * i / steps;
    const ty = min[1] + (max[1] - min[1]) * i / steps;
    drawLine([tx, min[1], z], [tx, max[1], z], "#7b8492", 0.7, []);
    drawLine([min[0], ty, z], [max[0], ty, z], "#7b8492", 0.7, []);
  }}
  ctx.globalAlpha = 1;
  drawLine([min[0], min[1], min[2]], [max[0], min[1], min[2]], "#e15454", 1.4, []);
  drawLine([min[0], min[1], min[2]], [min[0], max[1], min[2]], "#54c06a", 1.4, []);
  drawLine([min[0], min[1], min[2]], [min[0], min[1], max[2]], "#60a5fa", 1.4, []);
  drawLabel([max[0], min[1], min[2]], "X", "#ff8a8a");
  drawLabel([min[0], max[1], min[2]], "Y", "#8bf09d");
  drawLabel([min[0], min[1], max[2]], "Z", "#9bc5ff");
}}

function allTrajectoryPoints() {{
  return DATA.gt.concat(DATA.est);
}}

function niceStep(rawStep) {{
  if (!Number.isFinite(rawStep) || rawStep <= 0) return 0.1;
  const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / power;
  let factor = 10;
  if (normalized <= 1) factor = 1;
  else if (normalized <= 2) factor = 2;
  else if (normalized <= 5) factor = 5;
  return factor * power;
}}

function formatMeters(value) {{
  const absValue = Math.abs(value);
  if (absValue >= 10) return value.toFixed(0);
  if (absValue >= 1) return value.toFixed(1);
  return value.toFixed(2);
}}

function makeTicks(minValue, maxValue, step) {{
  const ticks = [];
  const start = Math.ceil(minValue / step) * step;
  for (let value = start; value <= maxValue + step * 0.25; value += step) {{
    ticks.push(Number(value.toFixed(8)));
    if (ticks.length > 24) break;
  }}
  return ticks;
}}

function projectionLayout(projectionCanvas, planeName) {{
  const plane = PROJECTION_PLANES[planeName];
  const ax = plane.axes[0];
  const ay = plane.axes[1];
  let minX = DATA.bounds.min[ax];
  let maxX = DATA.bounds.max[ax];
  let minY = DATA.bounds.min[ay];
  let maxY = DATA.bounds.max[ay];
  let rangeX = Math.max(maxX - minX, 0.1);
  let rangeY = Math.max(maxY - minY, 0.1);
  const padX = Math.max(rangeX * 0.06, 0.05);
  const padY = Math.max(rangeY * 0.06, 0.05);
  minX -= padX;
  maxX += padX;
  minY -= padY;
  maxY += padY;
  rangeX = maxX - minX;
  rangeY = maxY - minY;

  const margin = {{
    left: 46 * dpr,
    right: 14 * dpr,
    top: 16 * dpr,
    bottom: 34 * dpr
  }};
  const plotW = Math.max(1, projectionCanvas.width - margin.left - margin.right);
  const plotH = Math.max(1, projectionCanvas.height - margin.top - margin.bottom);
  const scale = Math.min(plotW / rangeX, plotH / rangeY);
  const usedW = rangeX * scale;
  const usedH = rangeY * scale;
  const ox = margin.left + (plotW - usedW) * 0.5;
  const oy = margin.top + (plotH - usedH) * 0.5;
  const step = niceStep(Math.max(rangeX, rangeY) / 5.0);

  return {{
    plane,
    ax,
    ay,
    minX,
    maxX,
    minY,
    maxY,
    rangeX,
    rangeY,
    scale,
    usedW,
    usedH,
    ox,
    oy,
    step
  }};
}}

function mapProjection(point, layout) {{
  return {{
    x: layout.ox + (point[layout.ax] - layout.minX) * layout.scale,
    y: layout.oy + layout.usedH - (point[layout.ay] - layout.minY) * layout.scale
  }};
}}

function drawProjectionText(pctx, text, x, y, color, size, align, baseline) {{
  pctx.fillStyle = color;
  pctx.font = String(size * dpr) + "px Arial";
  pctx.textAlign = align;
  pctx.textBaseline = baseline;
  pctx.fillText(text, x, y);
}}

function drawProjectionAxes(pctx, projectionCanvas, layout, planeName) {{
  pctx.fillStyle = "#11151d";
  pctx.fillRect(0, 0, projectionCanvas.width, projectionCanvas.height);
  pctx.fillStyle = "#141a23";
  pctx.fillRect(layout.ox, layout.oy, layout.usedW, layout.usedH);

  const xTicks = makeTicks(layout.minX, layout.maxX, layout.step);
  const yTicks = makeTicks(layout.minY, layout.maxY, layout.step);
  pctx.lineWidth = 0.65 * dpr;
  pctx.strokeStyle = "rgba(151, 164, 184, 0.24)";
  for (const tick of xTicks) {{
    const x = layout.ox + (tick - layout.minX) * layout.scale;
    pctx.beginPath();
    pctx.moveTo(x, layout.oy);
    pctx.lineTo(x, layout.oy + layout.usedH);
    pctx.stroke();
  }}
  for (const tick of yTicks) {{
    const y = layout.oy + layout.usedH - (tick - layout.minY) * layout.scale;
    pctx.beginPath();
    pctx.moveTo(layout.ox, y);
    pctx.lineTo(layout.ox + layout.usedW, y);
    pctx.stroke();
  }}

  pctx.strokeStyle = "rgba(226, 233, 244, 0.62)";
  pctx.lineWidth = 1.1 * dpr;
  pctx.strokeRect(layout.ox, layout.oy, layout.usedW, layout.usedH);

  pctx.strokeStyle = "rgba(226, 233, 244, 0.72)";
  pctx.lineWidth = 1 * dpr;
  for (const tick of xTicks) {{
    const x = layout.ox + (tick - layout.minX) * layout.scale;
    pctx.beginPath();
    pctx.moveTo(x, layout.oy + layout.usedH);
    pctx.lineTo(x, layout.oy + layout.usedH + 4 * dpr);
    pctx.stroke();
    drawProjectionText(pctx, formatMeters(tick), x, projectionCanvas.height - 18 * dpr, "#b8c4d6", 10, "center", "middle");
  }}
  for (const tick of yTicks) {{
    const y = layout.oy + layout.usedH - (tick - layout.minY) * layout.scale;
    pctx.beginPath();
    pctx.moveTo(layout.ox - 4 * dpr, y);
    pctx.lineTo(layout.ox, y);
    pctx.stroke();
    drawProjectionText(pctx, formatMeters(tick), layout.ox - 8 * dpr, y, "#b8c4d6", 10, "right", "middle");
  }}

  drawProjectionText(pctx, layout.plane.labels[0] + " (m)", layout.ox + layout.usedW * 0.5, projectionCanvas.height - 5 * dpr, "#dbe4f2", 11, "center", "bottom");
  drawProjectionText(pctx, layout.plane.labels[1] + " (m)", 8 * dpr, layout.oy + layout.usedH * 0.5, "#dbe4f2", 11, "left", "middle");
  drawProjectionText(pctx, planeName.toUpperCase(), layout.ox + 8 * dpr, layout.oy + 8 * dpr, "#edf1f7", 12, "left", "top");

  const scaleCandidates = [0.1, 0.2, 0.5, 1.0];
  const target = Math.max(layout.rangeX, layout.rangeY) * 0.28;
  let barM = scaleCandidates[0];
  for (const candidate of scaleCandidates) {{
    if (candidate <= target) barM = candidate;
  }}
  const barW = barM * layout.scale;
  const bx0 = layout.ox + layout.usedW - barW - 12 * dpr;
  const by = layout.oy + layout.usedH - 13 * dpr;
  pctx.strokeStyle = "#f5f7fb";
  pctx.lineWidth = 2 * dpr;
  pctx.beginPath();
  pctx.moveTo(bx0, by);
  pctx.lineTo(bx0 + barW, by);
  pctx.moveTo(bx0, by - 4 * dpr);
  pctx.lineTo(bx0, by + 4 * dpr);
  pctx.moveTo(bx0 + barW, by - 4 * dpr);
  pctx.lineTo(bx0 + barW, by + 4 * dpr);
  pctx.stroke();
  drawProjectionText(pctx, formatMeters(barM) + " m", bx0 + barW * 0.5, by - 8 * dpr, "#f5f7fb", 11, "center", "bottom");
}}

function drawProjectedPolyline(pctx, points, times, layout, color, width, dash) {{
  pctx.strokeStyle = color;
  pctx.lineWidth = width * dpr;
  pctx.lineJoin = "round";
  pctx.lineCap = "round";
  pctx.setLineDash(dash.map(v => v * dpr));
  pctx.beginPath();
  let open = false;
  for (let i = 0; i < points.length; i++) {{
    const p = mapProjection(points[i], layout);
    if (i === 0 || times[i] - times[i - 1] > DATA.breakGapS) {{
      if (open) pctx.stroke();
      pctx.beginPath();
      pctx.moveTo(p.x, p.y);
      open = true;
    }} else {{
      pctx.lineTo(p.x, p.y);
    }}
  }}
  if (open) pctx.stroke();
  pctx.setLineDash([]);
}}

function drawProjectedMarker(pctx, point, layout, color, label) {{
  const p = mapProjection(point, layout);
  pctx.fillStyle = color;
  pctx.beginPath();
  pctx.arc(p.x, p.y, 3.6 * dpr, 0, Math.PI * 2);
  pctx.fill();
  drawProjectionText(pctx, label, p.x + 6 * dpr, p.y - 6 * dpr, color, 10, "left", "bottom");
}}

function renderProjection(projectionCanvas) {{
  const pctx = projectionCanvas.getContext("2d");
  const planeName = projectionCanvas.dataset.plane;
  const layout = projectionLayout(projectionCanvas, planeName);
  drawProjectionAxes(pctx, projectionCanvas, layout, planeName);
  if (state.showGt) drawProjectedPolyline(pctx, DATA.gt, DATA.times, layout, "#4ea1ff", 1.7, [7, 5]);
  if (state.showEst) drawProjectedPolyline(pctx, DATA.est, DATA.times, layout, "#ff8a24", 1.8, []);
  drawProjectedMarker(pctx, DATA.gt[0], layout, "#f15c5c", "start");
  drawProjectedMarker(pctx, DATA.gt[DATA.gt.length - 1], layout, "#f15c5c", "end");
}}

function renderProjections() {{
  for (const projectionCanvas of projectionCanvases) {{
    renderProjection(projectionCanvas);
  }}
}}

function render() {{
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const grd = ctx.createLinearGradient(0, 0, 0, canvas.height);
  grd.addColorStop(0, "#12161d");
  grd.addColorStop(1, "#0d0f13");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (state.showGrid) drawGrid();
  if (state.showGt) drawPolyline(DATA.gt, DATA.times, "#4ea1ff", 2.0, [8, 6]);
  if (state.showEst) drawPolyline(DATA.est, DATA.times, "#ff8a24", 2.0, []);

  drawMarker(DATA.gt[0], "#f15c5c", 4, "start");
  drawMarker(DATA.gt[DATA.gt.length - 1], "#f15c5c", 4, "end");
  renderProjections();
}}

canvas.addEventListener("pointerdown", event => {{
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", event => {{
  if (!dragging) return;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  state.yaw += dx * 0.008;
  state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.008));
  render();
}});
canvas.addEventListener("pointerup", event => {{
  dragging = false;
  canvas.classList.remove("dragging");
  canvas.releasePointerCapture(event.pointerId);
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  state.zoom = Math.max(0.25, Math.min(8.0, state.zoom * Math.exp(-event.deltaY * 0.001)));
  render();
}}, {{ passive: false }});

document.getElementById("reset").addEventListener("click", () => {{
  state.yaw = -0.68;
  state.pitch = 0.78;
  state.zoom = 1.0;
  render();
}});
document.getElementById("top").addEventListener("click", () => {{
  state.yaw = 0.0;
  state.pitch = 0.0;
  state.zoom = 1.0;
  render();
}});
document.getElementById("side").addEventListener("click", () => {{
  state.yaw = -1.57;
  state.pitch = 1.05;
  state.zoom = 1.0;
  render();
}});
document.getElementById("showGt").addEventListener("change", event => {{
  state.showGt = event.target.checked;
  render();
}});
document.getElementById("showEst").addEventListener("change", event => {{
  state.showEst = event.target.checked;
  render();
}});
document.getElementById("showGrid").addEventListener("change", event => {{
  state.showGrid = event.target.checked;
  render();
}});
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""


def make_index(generated: list[dict[str, object]], output_dir: Path) -> str:
    rows = []
    for item in generated:
        summary = item["summary"]
        rel = html.escape(str(item["path"].relative_to(output_dir)))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['sequence']))}</td>"
            f"<td>{html.escape(str(item['variant']))}</td>"
            f"<td><a href=\"{rel}\">open</a></td>"
            f"<td>{summary['pose_count']}</td>"
            f"<td>{summary['position_rmse_m']:.5f}</td>"
            f"<td>{summary['position_max_m']:.5f}</td>"
            f"<td>{summary['orientation_rmse_deg']:.3f}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>corldata interactive 3D trajectory index</title>
<style>
body {{ margin: 0; padding: 28px; background: #101216; color: #edf1f7; font-family: Arial, Helvetica, sans-serif; }}
h1 {{ font-size: 22px; margin: 0 0 16px; }}
p {{ color: #cad2df; max-width: 900px; line-height: 1.45; }}
table {{ border-collapse: collapse; width: min(100%, 980px); background: #151922; border: 1px solid rgba(255,255,255,0.14); }}
th, td {{ padding: 9px 11px; border-bottom: 1px solid rgba(255,255,255,0.10); text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
th {{ color: #aeb9c9; font-weight: 600; }}
a {{ color: #7cb7ff; }}
</style>
</head>
<body>
<h1>corldata interactive 3D trajectory views</h1>
<p>Each page shows Ground truth and SE(3)-aligned ORB-SLAM3 estimated VO in the same coordinate frame. Drag inside the page to rotate the view and use the mouse wheel to zoom.</p>
<table>
<thead><tr><th>sequence</th><th>variant</th><th>html</th><th>poses</th><th>pos RMSE m</th><th>pos max m</th><th>angle RMSE deg</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.sequences is None:
        args.sequences = discover_sequences(args.plato_root, args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, object]] = []
    for sequence in args.sequences:
        for variant in args.variants:
            csv_path, metrics_path = variant_paths(args.plato_root, sequence, variant)
            if not csv_path.exists() or not metrics_path.exists():
                print(f"skip missing {sequence} {variant}: {csv_path}")
                continue
            rows = load_aligned_rows(csv_path)
            metrics = load_metrics(metrics_path)
            payload = make_payload(sequence, variant, rows, metrics)
            output_path = args.output_dir / f"{sequence}_{variant}_3d.html"
            output_path.write_text(make_html(payload), encoding="utf-8")
            generated.append(
                {
                    "sequence": sequence,
                    "variant": variant,
                    "path": output_path,
                    "summary": payload["summary"],
                }
            )

    index_path = args.output_dir / "index.html"
    index_path.write_text(make_index(generated, args.output_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "index": str(index_path),
                "html_count": len(generated),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
