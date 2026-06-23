from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^(?P<map>Town\d+)_type(?P<type_id>\d+)_subtype(?P<subtype_id>\d+)_scenario(?P<scenario_number>\d+)\.txt$"
)


def split_camel_case(text: str) -> list[str]:
    return re.findall(r"[A-Z][a-z0-9]*", text)


def infer_weather_label(environment_label: str) -> str:
    tokens = split_camel_case(environment_label)
    if any("Rain" in token for token in tokens):
        return "rainy"
    if any("Wet" in token for token in tokens):
        return "wet"
    if any("Cloud" in token for token in tokens):
        return "cloudy"
    if any("Clear" in token for token in tokens):
        return "clear"
    return "unknown"


def infer_time_of_day_label(environment_label: str) -> str:
    tokens = split_camel_case(environment_label)
    if any("Night" in token for token in tokens):
        return "night"
    if any("Sunset" in token or "Dusk" in token for token in tokens):
        return "sunset"
    if any("Noon" in token or "Day" in token or "Morning" in token for token in tokens):
        return "noon"
    return "unknown"


def infer_accident_type(meta_path: Path, split_type: str, has_collision: bool) -> str:
    if split_type == "normal":
        return "normal"
    if split_type == "accident":
        return "B"

    path_parts = {part.lower() for part in meta_path.parts}
    if "normal" in path_parts or any("normal" in part for part in path_parts):
        return "normal"
    if "accident" in path_parts or any("accident" in part for part in path_parts):
        return "B"
    return "B" if has_collision else "normal"


def parse_filename(meta_path: Path) -> dict[str, object]:
    match = FILENAME_PATTERN.match(meta_path.name)
    if not match:
        raise ValueError(f"Unsupported DeepAccident meta filename: {meta_path.name}")

    return {
        "map": match.group("map"),
        "scenario_type_id": int(match.group("type_id")),
        "scenario_subtype_id": int(match.group("subtype_id")),
        "scenario_number": int(match.group("scenario_number")),
        "scene_name": meta_path.stem,
    }


def parse_key_value_lines(lines: list[str]) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip().lower().replace(" ", "_")] = value.strip()
    return data


def parse_first_line(first_line: str) -> dict[str, object]:
    fields = first_line.split()
    parsed = {
        "environment_label": fields[0] if len(fields) > 0 else "",
        "other_actor_id": int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else None,
        "other_actor_type": fields[2] if len(fields) > 2 else None,
        "ego_actor_id": int(fields[3]) if len(fields) > 3 and fields[3].isdigit() else None,
        "ego_actor_type": fields[4] if len(fields) > 4 else None,
        "distance": float(fields[5]) if len(fields) > 5 else None,
        "relative_position": fields[6:8] if len(fields) > 7 else fields[6:] if len(fields) > 6 else [],
        "frame_index": int(fields[8]) if len(fields) > 8 and fields[8].isdigit() else None,
    }
    return parsed


def parse_agents_id(raw_value: str) -> list[int]:
    ids = []
    for token in raw_value.split():
        if token.isdigit():
            ids.append(int(token))
    return ids


def convert_meta_file(meta_path: Path, output_path: Path, split_type: str) -> None:
    lines = [line.strip() for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Meta file is empty: {meta_path}")

    filename_info = parse_filename(meta_path)
    first_line_info = parse_first_line(lines[0])
    kv_info = parse_key_value_lines(lines[1:])

    colliding_agents_raw = kv_info.get("colliding_agents", "")
    colliding_agents = colliding_agents_raw.split() if colliding_agents_raw else []
    agents_id = parse_agents_id(kv_info.get("agents_id", ""))
    environment_label = first_line_info["environment_label"]
    weather_label = infer_weather_label(environment_label)
    time_of_day_label = infer_time_of_day_label(environment_label)
    accident_type = infer_accident_type(meta_path, split_type, has_collision=bool(colliding_agents))

    actors = {
        "ego": {
            "id": first_line_info["ego_actor_id"],
            "type_id": first_line_info["ego_actor_type"],
        },
        "leading": None,
        "other": {
            "id": first_line_info["other_actor_id"],
            "type_id": first_line_info["other_actor_type"],
        },
    }

    metadata = {
        "scenario_id": None,
        "data_id": None,
        "map": filename_info["map"],
        "camera_fps": None,
        "views": [],
        "actors": actors,
        "parameters": None,
        "weather": None,
        "weather_label": weather_label,
        "time_of_day_label": time_of_day_label,
        "scenario_type_id": filename_info["scenario_type_id"],
        "scenario_subtype_id": filename_info["scenario_subtype_id"],
        "scenario_number": filename_info["scenario_number"],
        "scene_name": filename_info["scene_name"],
        "accident_type": accident_type,
        "source_dataset": "DeepAccident",
        "deepaccident": {
            "environment_label": environment_label,
            "distance": first_line_info["distance"],
            "relative_position": first_line_info["relative_position"],
            "frame_index": first_line_info["frame_index"],
            "colliding_agents": colliding_agents,
            "agents_id": agents_id,
            "road_type": kv_info.get("road_type"),
            "another_vehicle_spawn_side": kv_info.get("another_vehicle_spawn_side"),
            "ego_vehicle_direction": kv_info.get("ego_vehicle_direction"),
            "other_vehicle_direction": kv_info.get("other_vehicle_direction"),
            "raw_lines": lines,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_meta_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.txt"))


def build_output_path(meta_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root
    relative_path = meta_path.relative_to(input_root)
    return output_root / relative_path.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepAccident metadata txt files into the current FREA-style meta.json structure."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A DeepAccident metadata txt file or a directory containing metadata txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output json file path for single-file mode, or output root directory for batch mode.",
    )
    parser.add_argument(
        "--split-type",
        choices=["auto", "normal", "accident"],
        default="auto",
        help="How to assign accident_type. auto infers from path first, then from collision info.",
    )
    args = parser.parse_args()

    meta_files = iter_meta_files(args.input)
    if not meta_files:
        raise FileNotFoundError(f"No txt metadata files found under: {args.input}")

    for meta_file in meta_files:
        output_path = build_output_path(meta_file, args.input, args.output)
        convert_meta_file(meta_file, output_path, args.split_type)
        print(f"Converted: {meta_file} -> {output_path}")


if __name__ == "__main__":
    main()
