#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${OPENVINS_ROOT:-$REPO_ROOT/external/OpenVINS}"
SOURCE_DIR="$WORKSPACE/src/open_vins"
UPSTREAM_URL="https://github.com/rpng/open_vins.git"
UPSTREAM_COMMIT="69488123ed9362dd44b6f28e7f4680abbff1442b"
PATCH_FILE="$REPO_ROOT/patches/external/openvins_local_changes_2026-05-18.patch"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"

test -f "$ROS_SETUP" || {
  printf 'ROS setup not found: %s\n' "$ROS_SETUP" >&2
  exit 1
}
command -v git >/dev/null
command -v catkin >/dev/null || {
  printf 'catkin-tools is required. Install it for ROS Noetic first.\n' >&2
  exit 1
}

mkdir -p "$WORKSPACE/src"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone "$UPSTREAM_URL" "$SOURCE_DIR"
fi

git -C "$SOURCE_DIR" fetch origin
git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_COMMIT"
if git -C "$SOURCE_DIR" apply --check "$PATCH_FILE"; then
  git -C "$SOURCE_DIR" apply "$PATCH_FILE"
elif ! git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE"; then
  printf 'OpenVINS worktree is neither clean nor already patched. Inspect %s\n' "$SOURCE_DIR" >&2
  exit 1
fi

source "$ROS_SETUP"
catkin config --workspace "$WORKSPACE" --extend /opt/ros/noetic
catkin build --workspace "$WORKSPACE"

printf 'OpenVINS ready at %s\n' "$WORKSPACE"

