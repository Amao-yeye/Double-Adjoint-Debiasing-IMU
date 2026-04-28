#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/workspace_Amao/rss_debiasing_restart/Debias_IMU

export PYTHONPATH=".:BiasDy:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp

python BiasDy/mainEuroc_SE23decay.py \
  --only_test \
  --outputdir results/Euroc_SE23decay_w64 \
  --loss_window 64 \
  --decay_lambda 1.0 \
  --test_device cuda \
  > results/Euroc_SE23decay_w64/test_only_gpu.log 2>&1

python BiasDy/mainEuroc_SE23.py \
  --only_test \
  --outputdir results/Euroc_SE23_w64 \
  --loss_window 64 \
  --test_device cuda \
  > results/Euroc_SE23_w64/test_only_gpu.log 2>&1

python BiasDy/mainEuroc.py \
  --only_test \
  --outputdir results/Euroc_baseline_w64 \
  --loss_window 64 \
  --test_device cuda \
  > results/Euroc_baseline_w64/test_only_gpu.log 2>&1
