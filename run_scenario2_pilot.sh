#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-scenario2_pilot_output}"

COMMON_ARGS=(
  --agent_cfg expert.yaml
  --mode eval
  --eval_mode render
  --num_scenario 1
  --save_camera_frames
  --camera_fps 10
  --camera_only
  --progress_interval 20
)

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/run.py \
  "${COMMON_ARGS[@]}" \
  --scenario_cfg standard_eval_scenario2_pilot.yaml \
  --output_dir "${OUTPUT_DIR}"
