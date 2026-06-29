#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Update these paths for your Ubuntu machine before running.
CARLA_ROOT="${CARLA_ROOT:-/home/ubuntu/baiming/carla/CARLA_0.9.13}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_BASE="${OUTPUT_BASE:-/media/ubuntu/TOSHIBA EXT1/diy/type3_295_route_batches}"
CARLA_WAIT_SECONDS="${CARLA_WAIT_SECONDS:-35}"
CARLA_PORT="${CARLA_PORT:-2000}"
MAX_RETRIES_PER_BATCH="${MAX_RETRIES_PER_BATCH:-2}"
NUM_SCENARIO="${NUM_SCENARIO:-1}"
CAMERA_FPS="${CAMERA_FPS:-5}"

COMMON_ARGS=(
  --agent_cfg expert.yaml
  --mode eval
  --eval_mode render
  --num_scenario "${NUM_SCENARIO}"
  --save_camera_frames
  --camera_fps "${CAMERA_FPS}"
  --camera_only
  --progress_interval 20
)

start_carla() {
  local log_dir="${ROOT_DIR}/logs"
  mkdir -p "${log_dir}"
  local log_file="${log_dir}/carla_$(date +%Y%m%d_%H%M%S).log"

  echo "Starting CARLA..."
  (
    cd "${CARLA_ROOT}"
    nohup ./CarlaUE4.sh -quality-level=Low -RenderOffScreen -nosound > "${log_file}" 2>&1 &
  )

  echo "Waiting ${CARLA_WAIT_SECONDS}s for CARLA to boot..."
  sleep "${CARLA_WAIT_SECONDS}"
  echo "CARLA log: ${log_file}"
}

stop_carla() {
  echo "Stopping CARLA..."
  pkill -f CarlaUE4-Linux-Shipping || true
  pkill -f CarlaUE4.sh || true
  sleep 5
}

run_one_batch() {
  local scenario_cfg="$1"
  local output_dir="$2"

  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" scripts/run.py \
    "${COMMON_ARGS[@]}" \
    --scenario_cfg "${scenario_cfg}" \
    --output_dir "${output_dir}"
}

main() {
  if [ "$#" -lt 3 ]; then
    cat <<'EOF'
Usage:
  ./run_batches_with_carla_restart.sh <split_name> <output_subdir> <scenario_yaml_1> [scenario_yaml_2 ...]
  ./run_batches_with_carla_restart.sh <split_name> <output_subdir> --yaml-dir <dir> [--prefix <prefix>]

Example:
  ./run_batches_with_carla_restart.sh train train \
    standard_eval_scenario3_train_batch_00.yaml \
    standard_eval_scenario3_train_batch_01.yaml

  ./run_batches_with_carla_restart.sh train train \
    --yaml-dir /home/ubuntu/baiming/FREA/frea/scenario/config/batches \
    --prefix standard_eval_scenario3_train_batch_
EOF
    exit 1
  fi

  local split_name="$1"
  shift
  local output_subdir="$1"
  shift
  local scenario_cfgs=()

  if [ "${1:-}" = "--yaml-dir" ]; then
    if [ "$#" -lt 2 ]; then
      echo "Missing directory after --yaml-dir"
      exit 1
    fi
    local yaml_dir="$2"
    shift 2
    local prefix=""
    if [ "${1:-}" = "--prefix" ]; then
      if [ "$#" -lt 2 ]; then
        echo "Missing prefix after --prefix"
        exit 1
      fi
      prefix="$2"
      shift 2
    fi

    if [ ! -d "${yaml_dir}" ]; then
      echo "YAML directory not found: ${yaml_dir}"
      exit 1
    fi

    if [ -n "${prefix}" ]; then
      while IFS= read -r file; do
        scenario_cfgs+=("${file}")
      done < <(find "${yaml_dir}" -maxdepth 1 -type f -name "${prefix}*.yaml" | sort)
    else
      while IFS= read -r file; do
        scenario_cfgs+=("${file}")
      done < <(find "${yaml_dir}" -maxdepth 1 -type f -name "*.yaml" | sort)
    fi
  else
    scenario_cfgs=("$@")
  fi

  if [ "${#scenario_cfgs[@]}" -eq 0 ]; then
    echo "No scenario yaml files found to run."
    exit 1
  fi

  cd "${ROOT_DIR}"

  for scenario_cfg in "${scenario_cfgs[@]}"; do
    local batch_name
    batch_name="$(basename "${scenario_cfg}" .yaml)"
    local output_dir="${OUTPUT_BASE}/${output_subdir}/${batch_name}"
    local attempt=0
    local success=0

    while [ "${attempt}" -le "${MAX_RETRIES_PER_BATCH}" ]; do
      echo "============================================================"
      echo "Split: ${split_name}"
      echo "Batch yaml: ${scenario_cfg}"
      echo "Attempt: $((attempt + 1))/$((MAX_RETRIES_PER_BATCH + 1))"
      echo "Output dir: ${output_dir}"
      echo "============================================================"

      stop_carla
      start_carla

      if run_one_batch "${scenario_cfg}" "${output_dir}"; then
        success=1
        echo "Batch succeeded: ${scenario_cfg}"
        break
      fi

      echo "Batch failed: ${scenario_cfg}"
      attempt=$((attempt + 1))
    done

    stop_carla

    if [ "${success}" -ne 1 ]; then
      echo "Batch permanently failed after retries: ${scenario_cfg}"
      exit 1
    fi
  done

  echo "All requested batches finished."
}

main "$@"
