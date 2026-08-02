#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAIN="external/Debias_IMU/BiasDy/mainAquaLuxPLATO.py"
ROOT="Results/AquaLux/preintegration/ours_vo_w64pose_nll_alt_bgba_cov_e300"
DATA_DIR="${AQUALUX_DATA_ROOT:-datasets/AquaLux/0524_data}"
CACHE_DIR="Results/AquaLux/preintegration/ours_gt_w64_se23_mse/cache"

BG_INIT="Results/AquaLux/preintegration/checkpoints/vo_se3_mse_bg.pt"
BA_INIT="Results/AquaLux/preintegration/checkpoints/vo_se3_mse_ba.pt"

COMMON_ARGS=(
  --data_dir "$DATA_DIR"
  --data_cache_dir "$CACHE_DIR"
  --network_type_Gyro bw_func_net
  --network_type_Acc ba_func_net
  --device cuda
  --test_device cuda
  --plato_stage nll
  --plato_supervision vo_pose
  --plato_window_steps 64
  --loss_window 64
  --batch_size 16
  --train_sequence_strategy cycle
  --val_freq 5
  --save_freq 25
  --vo_align none
  --vo_init_mode vo_zero_bias
  --vo_residual_mode relative
  --mse_residual_mode right
  --nll_vo_mode sparse
  --nll_mse_anchor_weight 0.001
  --nll_min_sigma_a 1e-4
  --nll_max_sigma_a 1.0
  --nll_min_sigma_g 1e-5
  --nll_max_sigma_g 0.1
  --cov_jitter 1e-8
  --prior_std 1e-4
  --integral_method euler
)

mkdir -p "$ROOT"
MANIFEST="$ROOT/phase_manifest.tsv"
printf "phase\tkind\tepochs\tlr\tresume_checkpoint\tfinal_checkpoint\n" > "$MANIFEST"

resume_gyro="$BG_INIT"
resume_acc="$BA_INIT"
resume_noise=""

run_phase() {
  local phase="$1"
  local kind="$2"
  local epochs="$3"
  local lr="$4"
  shift 4

  local outdir="$ROOT/$phase"
  mkdir -p "$outdir"

  local cmd=(
    "$PYTHON_BIN" "$MAIN"
    "${COMMON_ARGS[@]}"
    --outputdir "$outdir"
    --epoch_nll "$epochs"
    --lr_NLL "$lr"
    --weight_decay_NLL 1e-6
    --resume_gyro "$resume_gyro"
    --resume_acc "$resume_acc"
  )
  if [[ -n "$resume_noise" ]]; then
    cmd+=(--resume_nll_noise "$resume_noise")
  fi
  cmd+=("$@")

  printf "[%s] starting %s\n" "$(date -Is)" "$phase" | tee "$outdir/phase_status.log"
  PYTHONUNBUFFERED=1 "${cmd[@]}" > "$outdir/run_stdout.log" 2>&1

  local final="$outdir/NLL_weights/model_epoch_$((epochs - 1)).pt"
  test -f "$final"
  printf "[%s] finished %s final=%s\n" "$(date -Is)" "$phase" "$final" | tee -a "$outdir/phase_status.log"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$phase" "$kind" "$epochs" "$lr" "$resume_gyro" "$final" >> "$MANIFEST"

  resume_gyro="$final"
  resume_acc="$final"
  resume_noise="$final"
}

for cycle in 01 02 03 04 05 06; do
  run_phase "cycle${cycle}_cov_e25" "covariance" 25 5e-5 --nll_covariance_only
  run_phase "cycle${cycle}_bgba_e25" "bgba" 25 1e-7 --nll_freeze_covariance
done

printf "%s\n" "$resume_gyro" > "$ROOT/final_checkpoint.txt"
