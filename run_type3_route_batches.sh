#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python"
OUTPUT_BASE="/media/ubuntu/TOSHIBA EXT1/diy/type3_295_route_batches"

COMMON_ARGS=(
  --agent_cfg expert.yaml
  --mode eval
  --eval_mode render
  --save_camera_frames
  --camera_fps 10
  --camera_only
  --progress_interval 20
)

run_batch() {
  local split_name="$1"
  local scenario_cfg="$2"
  local output_dir="$3"

  echo "============================================================"
  echo "Running split: ${split_name}"
  echo "Scenario config: ${scenario_cfg}"
  echo "Output dir: ${output_dir}"
  echo "============================================================"

  mkdir -p "${output_dir}"

  "${PYTHON_BIN}" scripts/run.py \
    "${COMMON_ARGS[@]}" \
    --scenario_cfg "${scenario_cfg}" \
    --output_dir "${output_dir}"
}

cd "${ROOT_DIR}"

echo "This script runs one batch yaml at a time."
echo "Edit the SCENARIO_CFG and OUTPUT_DIR below before each batch run if needed."

SCENARIO_CFG="${1:-standard_eval_scenario3_train_295.yaml}"
OUTPUT_DIR="${2:-${OUTPUT_BASE}/train_batch_manual}"
SPLIT_NAME="${3:-manual}"

run_batch "${SPLIT_NAME}" "${SCENARIO_CFG}" "${OUTPUT_DIR}"
