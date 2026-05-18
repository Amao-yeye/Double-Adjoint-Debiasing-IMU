#!/usr/bin/env bash
set -euo pipefail

PLATO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SEQUENCE_DIR="${PLATO_ROOT}/results/openvins_eval/LIASuw/phys_delta_vo_framefixed_clean_imu/0205s14_80"
CONFIG="${PLATO_ROOT}/plato/configs/openvins/liasuw_0205/estimator_config.yaml"
GT="${SEQUENCE_DIR}/groundtruth/body_gt_openvins_asl_gtinit.csv"
METHOD=""
DROPOUT_START=""
DROPOUT_END=""
RUN_SUFFIX=""

usage() {
  cat <<'USAGE'
Usage:
  run_openvins_dropout_trial.sh --method constant|network --dropout-start S --dropout-end S

Options:
  --sequence-dir PATH   OpenVINS eval sequence dir.
  --config PATH         OpenVINS estimator config.
  --gt PATH             ASL groundtruth file for GT initialization.
  --run-suffix SUFFIX   Optional suffix for the output run directory.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="$2"
      shift 2
      ;;
    --dropout-start)
      DROPOUT_START="$2"
      shift 2
      ;;
    --dropout-end)
      DROPOUT_END="$2"
      shift 2
      ;;
    --sequence-dir)
      SEQUENCE_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --gt)
      GT="$2"
      shift 2
      ;;
    --run-suffix)
      RUN_SUFFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${METHOD}" || -z "${DROPOUT_START}" || -z "${DROPOUT_END}" ]]; then
  usage >&2
  exit 2
fi

if [[ "${METHOD}" != "constant" && "${METHOD}" != "network" ]]; then
  echo "--method must be 'constant' or 'network'" >&2
  exit 2
fi

TAG="dropout_${DROPOUT_START}_${DROPOUT_END}"
BAG="${SEQUENCE_DIR}/bags_${TAG}/${METHOD}.bag"
OUT_DIR="${SEQUENCE_DIR}/runs/${TAG}_${METHOD}${RUN_SUFFIX}"

if [[ ! -f "${BAG}" ]]; then
  echo "Missing bag: ${BAG}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${GT}" ]]; then
  echo "Missing GT: ${GT}" >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
source "${PLATO_ROOT}/external/OpenVINS/devel/setup.bash"
export ROS_HOME="${ROS_HOME:-/tmp/plato_ros_home}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

mkdir -p "${OUT_DIR}"

rosrun ov_eval pose_to_file \
  _topic:=/ros1_serial_msckf/odomimu \
  _topic_type:=Odometry \
  _output:="${OUT_DIR}/odomimu.txt" \
  >"${OUT_DIR}/pose_to_file_console.log" 2>&1 &
RECORDER_PID=$!

cleanup() {
  if kill -0 "${RECORDER_PID}" 2>/dev/null; then
    kill "${RECORDER_PID}" 2>/dev/null || true
    wait "${RECORDER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

sleep 1

set +e
rosrun ov_msckf ros1_serial_msckf "${CONFIG}" \
  _path_bag:="${BAG}" \
  _path_gt:="${GT}" \
  _filepath_est:="${OUT_DIR}/estimate.txt" \
  _filepath_std:="${OUT_DIR}/std.txt" \
  >"${OUT_DIR}/openvins_console.log" 2>&1
STATUS=$?
set -e

if [[ "${STATUS}" -ne 0 ]]; then
  if [[ "${STATUS}" -eq 134 && -s "${OUT_DIR}/odomimu.txt" ]]; then
    echo "OpenVINS returned 134 after writing outputs; see ${OUT_DIR}/openvins_console.log"
  else
    echo "OpenVINS failed with status ${STATUS}; see ${OUT_DIR}/openvins_console.log" >&2
    exit "${STATUS}"
  fi
fi

echo "Wrote ${OUT_DIR}"
