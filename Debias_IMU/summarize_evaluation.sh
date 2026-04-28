#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH=".:BiasDy:${PYTHONPATH:-}"

python - <<'PY'
import os
from dataset import pload
from evalutation import AOE, ATE

root = "results"
seqs = ["MH_02_easy", "MH_04_difficult", "V2_02_medium", "V1_03_difficult", "V1_01_easy"]

def display_name(cfg):
    if "_vo_" in cfg and "_vo_body_" not in cfg:
        return cfg.replace("_vo_", "_vo_body_", 1)
    return cfg

rows = []
for cfg in sorted(d for d in os.listdir(root) if d.startswith("Euroc_")):
    cfg_dir = os.path.join(root, cfg)
    if not os.path.isdir(cfg_dir):
        continue

    vals = []
    for seq in seqs:
        path = os.path.join(cfg_dir, "Sequnces-results", seq, "FullTra_result.p")
        if not os.path.exists(path):
            continue

        data = pload(path)
        X_gt = data["X_gt"]
        R_pred = data["R_pred"]
        sol_pred = data["sol_pred"]

        aoe = AOE(X_gt[:, :3, :3], R_pred).item()
        ate_vel = ATE(X_gt[:, :3, 3], sol_pred[:, 7:10]).item()
        ate_pos = ATE(X_gt[:, :3, 4], sol_pred[:, 10:13]).item()
        vals.append((aoe, ate_vel, ate_pos))

    if vals:
        n = len(vals)
        rows.append((
            display_name(cfg),
            f"{n}/5",
            sum(x[0] for x in vals) / n,
            sum(x[1] for x in vals) / n,
            sum(x[2] for x in vals) / n,
        ))

rows.sort(key=lambda row: row[4])

method_width = max([len("method")] + [len(row[0]) for row in rows])
header = f"{'method':{method_width}s} {'done':>6s} {'AOE':>10s} {'ATE vel':>10s} {'ATE pos':>12s}"
lines = [header, "-" * len(header)]
for cfg, done, aoe, ate_vel, ate_pos in rows:
    lines.append(f"{cfg:{method_width}s} {done:>6s} {aoe:10.3f} {ate_vel:10.3f} {ate_pos:12.3f}")

text = "\n".join(lines)
print(text)

out = os.path.join(root, "evaluation_comparison.txt")
with open(out, "w") as f:
    f.write(text + "\n")
print(f"\nSaved to: {out}")
PY
