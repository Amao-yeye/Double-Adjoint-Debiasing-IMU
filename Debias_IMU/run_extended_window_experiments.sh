#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/workspace_Amao/rss_debiasing_restart/Debias_IMU

export PYTHONPATH=".:BiasDy:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp

EPOCHS=1801
TEST_DEVICE="${TEST_DEVICE:-cuda}"
DECAY_LAMBDA="${DECAY_LAMBDA:-1.5}"
WINDOWS=(96 128 160)

run_experiment() {
  local name="$1"
  shift
  local outputdir="$1"
  shift
  local logfile="${outputdir}/train_test.log"

  mkdir -p "$outputdir"
  : > "$logfile"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name}" | tee -a "$logfile"
  printf 'Command:' >> "$logfile"
  printf ' %q' "$@" >> "$logfile"
  printf '\n' >> "$logfile"
  "$@" >> "$logfile" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${name}" | tee -a "$logfile"
}

for window in "${WINDOWS[@]}"; do
  run_experiment "Euroc_baseline_w${window}" "results/Euroc_baseline_w${window}" \
    python BiasDy/mainEuroc.py \
      --outputdir "results/Euroc_baseline_w${window}" \
      --loss_window "$window" \
      --epoch "$EPOCHS" \
      --test_device "$TEST_DEVICE" \
      --skip_axb

  run_experiment "Euroc_SE23_w${window}" "results/Euroc_SE23_w${window}" \
    python BiasDy/mainEuroc_SE23.py \
      --outputdir "results/Euroc_SE23_w${window}" \
      --loss_window "$window" \
      --epoch "$EPOCHS" \
      --test_device "$TEST_DEVICE"

  run_experiment "Euroc_SE23decay_lam1p5_w${window}" "results/Euroc_SE23decay_lam1p5_w${window}" \
    python BiasDy/mainEuroc_SE23decay.py \
      --outputdir "results/Euroc_SE23decay_lam1p5_w${window}" \
      --loss_window "$window" \
      --decay_lambda "$DECAY_LAMBDA" \
      --epoch "$EPOCHS" \
      --test_device "$TEST_DEVICE"
done

bash /home/ubuntu/workspace_Amao/rss_debiasing_restart/Debias_IMU/summarize_evaluation.sh
