#!/usr/bin/env bash
set -euo pipefail

PLATO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${PLATO_ROOT}/plato/configs/openvins/corldata_0511/estimator_config_dynamic.yaml"
DATASET_ROOT="/home/amao/datasets/corldata/0511_data"
SEQ=""
OUT_DIR=""
BAG=""

usage() {
  cat <<'USAGE'
Usage:
  run_corldata_openvins_natural_trial.sh --seq S13

Options:
  --seq SEQ        corldata sequence name, e.g. S13.
  --bag PATH       Optional explicit bag path.
  --config PATH    Optional OpenVINS estimator config.
  --out-dir PATH   Optional output directory.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seq)
      SEQ="$2"
      shift 2
      ;;
    --bag)
      BAG="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
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

if [[ -z "${SEQ}" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "${BAG}" ]]; then
  BAG="${DATASET_ROOT}/${SEQ}.bag"
fi
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${PLATO_ROOT}/results/openvins_eval/corldata/natural/raw/${SEQ}"
fi

if [[ ! -f "${BAG}" ]]; then
  echo "Missing bag: ${BAG}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
source "${PLATO_ROOT}/external/OpenVINS/devel/setup.bash"
export ROS_HOME="${ROS_HOME:-/tmp/plato_ros_home}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/plato_ros_log}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

mkdir -p "${OUT_DIR}" "${ROS_HOME}" "${ROS_LOG_DIR}"

ROSCORE_PID=""
if ! rosparam list >/dev/null 2>&1; then
  roscore >"${OUT_DIR}/roscore.log" 2>&1 &
  ROSCORE_PID=$!
  for _ in {1..30}; do
    if rosparam list >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

cleanup() {
  if [[ -n "${ROSCORE_PID}" ]] && kill -0 "${ROSCORE_PID}" 2>/dev/null; then
    kill "${ROSCORE_PID}" 2>/dev/null || true
    wait "${ROSCORE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! rosparam list >/dev/null 2>&1; then
  echo "ROS master is not reachable at ${ROS_MASTER_URI}" >&2
  exit 1
fi

set +e
rosrun ov_msckf ros1_serial_msckf "${CONFIG}" \
  _path_bag:="${BAG}" \
  _save_total_state:=true \
  _filepath_est:="${OUT_DIR}/estimate.txt" \
  _filepath_std:="${OUT_DIR}/std.txt" \
  _record_timing_information:=true \
  _record_timing_filepath:="${OUT_DIR}/timing.txt" \
  >"${OUT_DIR}/openvins_console.log" 2>&1
STATUS=$?
set -e

if [[ "${STATUS}" -ne 0 ]]; then
  if [[ "${STATUS}" -eq 134 && -s "${OUT_DIR}/estimate.txt" ]]; then
    echo "OpenVINS returned 134 after writing outputs; see ${OUT_DIR}/openvins_console.log"
  else
    echo "OpenVINS failed with status ${STATUS}; see ${OUT_DIR}/openvins_console.log" >&2
    exit "${STATUS}"
  fi
fi

echo "Wrote ${OUT_DIR}"
