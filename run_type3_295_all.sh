#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="/media/ubuntu/TOSHIBA EXT1/diy/type3_295"
PYTHON_BIN="python"

COMMON_ARGS=(
  --agent_cfg expert.yaml
  --mode eval
  --eval_mode render
  --save_camera_frames
  --camera_fps 10
  --camera_only
  --progress_interval 20
)

run_split() {
  local split_name="$1"
  local scenario_cfg="$2"
  local output_dir="${OUTPUT_BASE}/${split_name}"

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

run_split train standard_eval_scenario3_train_295.yaml
run_split val standard_eval_scenario3_val_295.yaml
run_split test standard_eval_scenario3_test_295.yaml

echo "All splits finished."
