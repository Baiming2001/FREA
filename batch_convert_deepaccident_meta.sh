#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <deepaccident_root_dir> [split_type]"
  echo "Example: $0 /media/ubuntu/TOSHIBA\\ EXT1/deep/train/DeepAccident_train_01 auto"
  exit 1
fi

ROOT_DIR="$1"
SPLIT_TYPE="${2:-auto}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: directory does not exist: $ROOT_DIR"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/convert_deepaccident_meta_to_json.py"

if [[ ! -f "$CONVERTER" ]]; then
  echo "Error: converter script not found: $CONVERTER"
  exit 1
fi

converted_count=0

for scenario_dir in "$ROOT_DIR"/*; do
  if [[ ! -d "$scenario_dir" ]]; then
    continue
  fi

  meta_dir="$scenario_dir/meta"
  output_dir="$scenario_dir/meta_data"

  if [[ -d "$meta_dir" ]]; then
    echo "Converting: $meta_dir -> $output_dir"
    python "$CONVERTER" \
      --input "$meta_dir" \
      --output "$output_dir" \
      --split-type "$SPLIT_TYPE"
    converted_count=$((converted_count + 1))
  fi
done

echo "Finished. Converted metadata folders: $converted_count"
