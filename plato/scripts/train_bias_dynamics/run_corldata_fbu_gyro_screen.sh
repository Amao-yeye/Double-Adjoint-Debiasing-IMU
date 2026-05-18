#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/amao/workspace/PLATO"
PY="/home/amao/anaconda3/envs/debiasimu/bin/python"
DATA_ROOT="results/debias_imu/corldata_pseudogt_refs_train_s00s02s03s07_test_s13s15s20s21s22s23"
CACHE_DIR="results/debias_imu/cache/corldata_network_screen_refs_lw16_b1000_sg1"
LOG_DIR="results/debias_imu/logs"

mkdir -p "${ROOT}/${LOG_DIR}"
cd "${ROOT}"

common_args=(
  --data-root "${DATA_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --pseudo-gt-root results/pseudo_gt/corldata
  --train-seqs S00 S02 S03 S07
  --test-seqs S13 S15 S20 S21 S23
  --trainval-supervision-source vo_imu_noninterp
  --vo-supervision-world-frame vo
  --gravity-accel-world 0 0 -9.7803
  --dt 0.005
  --loss_window 16
  --batch-size 1000
  --epoch 601
  --val_freq 50
  --save_freq 50
  --train_batch_strategy random_sequence
  --sg_window_bw 1
  --sg_order_bw 0
  --sg_window_ba 1
  --sg_order_ba 0
  --device cuda
  --test-device cpu
  --use-best-val
  --stop-after-gyro
)

run_case() {
  local name="$1"
  local gyro_net="$2"
  local acc_net="$3"
  local outdir="results/debias_imu/corldata_gyro_screen_${name}_e601_rand_lw16_b1000"
  local log="${LOG_DIR}/corldata_gyro_screen_${name}_e601_rand_lw16_b1000.log"
  echo "=== ${name}: ${gyro_net} ==="
  CUDA_VISIBLE_DEVICES=0 "${PY}" plato/scripts/train_bias_dynamics/run_liasuw_debias_ad_pipeline.py \
    "${common_args[@]}" \
    --outputdir "${outdir}" \
    --network_type_Gyro "${gyro_net}" \
    --network_type_Acc "${acc_net}" \
    > "${log}" 2>&1
}

run_case "fbu_residual" "bw_func_fbu_residual" "ba_func_fbu_residual"
run_case "phys_eq" "bw_func_phys_eq" "ba_func_phys_eq"
