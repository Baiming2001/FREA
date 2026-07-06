#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./images_to_video.sh -i INPUT_DIR [-o OUTPUT_FILE] [-r FRAMERATE] [-e EXTENSIONS] [--overwrite]

Options:
  -i, --input        Input image directory (required)
  -o, --output       Output video file name or absolute path (default: output.mp4)
  -r, --framerate    Frame rate in FPS (default: 25)
  -e, --extensions   Comma-separated extensions (default: png,jpg,jpeg,bmp)
      --overwrite    Overwrite existing output file
  -h, --help         Show this help

Example:
  ./images_to_video.sh -i /data/frames -o result.mp4 -r 30 --overwrite
EOF
}

input_dir=""
output_file="output.mp4"
framerate="25"
extensions="png,jpg,jpeg,bmp"
overwrite="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      input_dir="${2:-}"
      shift 2
      ;;
    -o|--output)
      output_file="${2:-}"
      shift 2
      ;;
    -r|--framerate)
      framerate="${2:-}"
      shift 2
      ;;
    -e|--extensions)
      extensions="${2:-}"
      shift 2
      ;;
    --overwrite)
      overwrite="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$input_dir" ]]; then
  echo "Input directory is required." >&2
  usage
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not installed or not available in PATH." >&2
  exit 1
fi

if [[ ! -d "$input_dir" ]]; then
  echo "Input directory does not exist: $input_dir" >&2
  exit 1
fi

if ! [[ "$framerate" =~ ^[0-9]+$ ]] || [[ "$framerate" -le 0 ]]; then
  echo "Framerate must be a positive integer." >&2
  exit 1
fi

input_dir="$(realpath "$input_dir")"

if [[ "$output_file" = /* ]]; then
  output_path="$output_file"
else
  output_path="$input_dir/$output_file"
fi

mkdir -p "$(dirname "$output_path")"

if [[ -e "$output_path" && "$overwrite" != "true" ]]; then
  echo "Output file already exists: $output_path" >&2
  echo "Use --overwrite to replace it." >&2
  exit 1
fi

list_file="$(mktemp)"
trap 'rm -f "$list_file"' EXIT

IFS=',' read -r -a ext_array <<< "$extensions"

found_any="false"
for ext in "${ext_array[@]}"; do
  while IFS= read -r -d '' file; do
    printf "file '%s'\n" "${file//\'/\'\\\'\'}" >> "$list_file"
    found_any="true"
  done < <(find "$input_dir" -maxdepth 1 -type f \( -iname "*.${ext}" \) -print0)
done

if [[ "$found_any" != "true" ]]; then
  echo "No supported images were found in: $input_dir" >&2
  exit 1
fi

sort -u "$list_file" -o "$list_file"

echo "Input directory: $input_dir"
echo "Output file: $output_path"
echo "Frame rate: $framerate FPS"

if [[ "$overwrite" == "true" ]]; then
  ffmpeg -y \
    -r "$framerate" \
    -f concat \
    -safe 0 \
    -i "$list_file" \
    -vf "fps=$framerate,format=yuv420p" \
    -c:v libx264 \
    "$output_path"
else
  ffmpeg -n \
    -r "$framerate" \
    -f concat \
    -safe 0 \
    -i "$list_file" \
    -vf "fps=$framerate,format=yuv420p" \
    -c:v libx264 \
    "$output_path"
fi

echo "Video generation completed."
