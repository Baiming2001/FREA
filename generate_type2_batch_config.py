#!/usr/bin/env python
import argparse
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_ALLOWED_ROUTE_KEYS = [
    "1:9",
    "5:4",
    "5:5",
    "5:6",
    "5:8",
    "5:10",
]
DEFAULT_SPLIT_WEIGHTS = {
    "train": 7.0,
    "val": 1.5,
    "test": 1.5,
}
DEFAULT_OUTCOME_WEIGHTS = {
    "collision": 7.0,
    "near_miss": 3.0,
}
DEFAULT_WEATHER_WEIGHTS = {
    "clear": 4.0,
    "rainy": 4.0,
    "cloudy": 1.0,
    "wet": 1.0,
}
DEFAULT_TIME_WEIGHTS = {
    "noon": 3.0,
    "sunset": 1.0,
    "night": 1.0,
}
DEFAULT_LEADING_VEHICLE_MODELS = [
    "vehicle.tesla.model3",
    "vehicle.audi.tt",
    "vehicle.mercedes.coupe",
    "vehicle.audi.etron",
]
DEFAULT_EGO_VEHICLE_MODELS = [
    "vehicle.lincoln.mkz_2017",
    "vehicle.tesla.model3",
    "vehicle.audi.etron",
    "vehicle.mercedes.coupe",
]
DEFAULT_OTHER_VEHICLE_MODELS = [
    "vehicle.audi.tt",
    "vehicle.lincoln.mkz_2017",
    "vehicle.tesla.model3",
]
DISALLOWED_VEHICLE_MODEL_KEYWORDS = (
    "firetruck",
    "ambulance",
    "truck",
    "bus",
    "van",
    "sprinter",
    "patrol",
    "carlacola",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced type-2 parking dataset JSON from "
            "scan_type2_roadside_routes.py results, with optional split JSON/YAML outputs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the route scan JSON produced by scan_type2_roadside_routes.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output combined scenario_type JSON path.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root used to infer scenario_type relative paths for YAML generation.",
    )
    parser.add_argument(
        "--scenario-type-root",
        type=Path,
        default=None,
        help=(
            "Root directory pointed to by scenario_type_dir in YAML files. "
            "Defaults to <root-dir>/frea/scenario/config/scenario_type."
        ),
    )
    parser.add_argument(
        "--split-json-dir",
        type=Path,
        default=None,
        help="Optional directory to write split-specific JSON files.",
    )
    parser.add_argument(
        "--base-yaml",
        type=Path,
        default=None,
        help="Optional base scenario YAML template used to emit YAML configs.",
    )
    parser.add_argument(
        "--yaml-output",
        type=Path,
        default=None,
        help="Optional YAML path for the combined JSON output.",
    )
    parser.add_argument(
        "--split-yaml-dir",
        type=Path,
        default=None,
        help="Optional directory to write train/val/test YAML files.",
    )
    parser.add_argument(
        "--yaml-prefix",
        type=str,
        default=None,
        help="Optional prefix for split YAML filenames. Defaults to the output JSON stem.",
    )
    parser.add_argument(
        "--scenario-type-prefix",
        type=str,
        default=None,
        help=(
            "Optional prefix prepended to JSON filenames when writing YAML scenario_type values, "
            "for cases where JSONs live in a subdirectory under scenario_type_dir."
        ),
    )
    parser.add_argument(
        "--towns",
        nargs="+",
        default=None,
        help="Optional subset of towns to keep.",
    )
    parser.add_argument(
        "--source-scenarios",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of source scenario ids to keep.",
    )
    parser.add_argument(
        "--allowed-routes",
        nargs="+",
        default=DEFAULT_ALLOWED_ROUTE_KEYS,
        help=(
            "Allowed source routes in scenario_id:route_id form. "
            "Defaults to the latest approved type-2 route set."
        ),
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="Optional cap on the number of distinct parking routes kept after filtering.",
    )
    parser.add_argument(
        "--total-scenes",
        type=int,
        default=75,
        help="Total number of scenes to generate.",
    )
    parser.add_argument(
        "--split-weights",
        nargs="+",
        default=["train=7", "val=1.5", "test=1.5"],
        help="Dataset split weights, e.g. train=7 val=1.5 test=1.5.",
    )
    parser.add_argument(
        "--outcome-weights",
        nargs="+",
        default=["collision=7", "near_miss=3"],
        help="Target outcome weights, e.g. collision=7 near_miss=3.",
    )
    parser.add_argument(
        "--weather-weights",
        nargs="+",
        default=["clear=4", "rainy=4", "cloudy=1", "wet=1"],
        help="Weather roulette weights, e.g. clear=4 rainy=4 cloudy=1 wet=1.",
    )
    parser.add_argument(
        "--time-weights",
        nargs="+",
        default=["noon=3", "sunset=1", "night=1"],
        help="Time-of-day roulette weights, e.g. noon=3 sunset=1 night=1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible route remainder and attribute assignment.",
    )
    parser.add_argument(
        "--start-data-id",
        type=int,
        default=0,
        help="Starting data_id and scenario_number offset.",
    )
    parser.add_argument(
        "--scenario-type-id",
        type=int,
        default=2,
        help="Custom scenario type id written into parameters.",
    )
    parser.add_argument(
        "--scenario-subtype-id",
        type=int,
        default=1,
        help="Custom scenario subtype id written into parameters.",
    )
    parser.add_argument(
        "--min-route-progress",
        type=float,
        default=0.0,
        help="Optionally drop candidates whose leading route progress ratio is below this threshold.",
    )
    parser.add_argument(
        "--dedupe-geometry",
        action="store_true",
        help="Keep only one route per identical parking/driving geometry footprint.",
    )
    parser.add_argument(
        "--leading-vehicle-models",
        nargs="+",
        default=DEFAULT_LEADING_VEHICLE_MODELS,
        help="Candidate CARLA blueprint ids used to randomize the leading vehicle model.",
    )
    parser.add_argument(
        "--ego-vehicle-models",
        nargs="+",
        default=DEFAULT_EGO_VEHICLE_MODELS,
        help="Candidate CARLA blueprint ids used to randomize the ego vehicle model.",
    )
    parser.add_argument(
        "--other-vehicle-models",
        nargs="+",
        default=DEFAULT_OTHER_VEHICLE_MODELS,
        help="Candidate CARLA blueprint ids used to randomize the other vehicle model.",
    )
    parser.add_argument(
        "--pilot-paired-outcomes",
        action="store_true",
        help=(
            "Generate exactly one normal and one collision scene per selected route. "
            "Useful for pilot configs."
        ),
    )
    return parser.parse_args()


def parse_weight_map(entries):
    result = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid weight entry: {entry}. Expected label=value.")
        label, value_text = entry.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid empty label in weight entry: {entry}")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in weight entry: {entry}") from exc
        if value < 0:
            raise ValueError(f"Weight must be non-negative: {entry}")
        result[label] = value
    if not result:
        raise ValueError("At least one weight entry is required.")
    if sum(result.values()) <= 0:
        raise ValueError("Weight sum must be positive.")
    return result


def parse_route_keys(entries):
    route_keys = set()
    for entry in entries:
        normalized = entry.strip().replace("/", ":")
        if ":" not in normalized:
            raise ValueError(
                f"Invalid route key: {entry}. Expected scenario_id:route_id, e.g. 5:10."
            )
        scenario_text, route_text = normalized.split(":", 1)
        try:
            route_keys.add((int(scenario_text), int(route_text)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid route key: {entry}. Expected integer scenario_id and route_id."
            ) from exc
    if not route_keys:
        raise ValueError("At least one allowed route must be provided.")
    return route_keys


def route_priority(route_entry):
    candidate = route_entry["best_candidate"]
    lane_priority = 0 if candidate["lane_type"] == "parking" else 1
    progress_priority = abs(candidate["route_progress_ratio"] - 0.35)
    return (
        lane_priority,
        progress_priority,
        candidate["lateral_distance_m"],
        route_entry["town"],
        route_entry["source_scenario_id"],
        route_entry["route_id"],
    )


def candidate_geometry_key(route_entry):
    candidate = route_entry["best_candidate"]
    driving = candidate["driving_transform"]
    roadside = candidate["roadside_transform"]
    return (
        route_entry["town"],
        round(float(driving["x"]), 3),
        round(float(driving["y"]), 3),
        round(float(driving.get("yaw", 0.0)), 3),
        round(float(roadside["x"]), 3),
        round(float(roadside["y"]), 3),
        round(float(roadside.get("yaw", 0.0)), 3),
    )


def compute_integer_quotas(total_count, weights):
    labels = list(weights.keys())
    total_weight = sum(weights.values())
    raw_counts = {
        label: (total_count * weights[label] / total_weight) for label in labels
    }
    quotas = {label: int(raw_counts[label]) for label in labels}
    assigned = sum(quotas.values())
    remaining = total_count - assigned
    remainders = sorted(
        labels,
        key=lambda label: (raw_counts[label] - quotas[label], weights[label], label),
        reverse=True,
    )
    for index in range(remaining):
        quotas[remainders[index]] += 1
    return quotas


def weighted_choice(rng, weight_map):
    labels = list(weight_map.keys())
    weights = list(weight_map.values())
    return rng.choices(labels, weights=weights, k=1)[0]


def choose_from_candidates(rng, values, label):
    candidates = [
        value.strip()
        for value in values
        if str(value).strip()
        and not any(keyword in value.strip().lower() for keyword in DISALLOWED_VEHICLE_MODEL_KEYWORDS)
    ]
    if not candidates:
        raise ValueError(f"At least one candidate is required for {label}.")
    return rng.choice(candidates)


def build_balanced_label_pool(total_count, weights, rng):
    quotas = compute_integer_quotas(total_count, weights)
    labels = []
    for label, count in quotas.items():
        labels.extend([label] * count)
    rng.shuffle(labels)
    return labels, quotas


def filter_routes(args, data):
    allowed_route_keys = parse_route_keys(args.allowed_routes)
    routes = data.get("routes", [])

    filtered_routes = []
    for route_entry in routes:
        candidate = route_entry.get("best_candidate")
        if not route_entry.get("has_candidate") or candidate is None:
            continue
        if candidate.get("lane_type") != "parking":
            continue
        if float(candidate.get("route_progress_ratio", 0.0)) < args.min_route_progress:
            continue
        if args.towns is not None and route_entry["town"] not in args.towns:
            continue
        if (
            args.source_scenarios is not None
            and route_entry["source_scenario_id"] not in args.source_scenarios
        ):
            continue
        route_key = (route_entry["source_scenario_id"], route_entry["route_id"])
        if route_key not in allowed_route_keys:
            continue
        filtered_routes.append(route_entry)

    filtered_routes.sort(key=route_priority)

    if args.dedupe_geometry:
        unique_routes = []
        seen_geometry = set()
        for route_entry in filtered_routes:
            geometry_key = candidate_geometry_key(route_entry)
            if geometry_key in seen_geometry:
                continue
            seen_geometry.add(geometry_key)
            unique_routes.append(route_entry)
        filtered_routes = unique_routes

    if args.max_routes is not None:
        filtered_routes = filtered_routes[: args.max_routes]

    found_route_keys = {
        (route_entry["source_scenario_id"], route_entry["route_id"])
        for route_entry in filtered_routes
    }
    missing_route_keys = sorted(allowed_route_keys - found_route_keys)
    if missing_route_keys:
        missing_text = ", ".join(f"{scenario_id}:{route_id}" for scenario_id, route_id in missing_route_keys)
        raise ValueError(
            f"Missing required allowed routes after filtering: {missing_text}. "
            "Please check the scan JSON or your filters."
        )

    if not filtered_routes:
        raise ValueError("No usable parking routes remained after filtering.")

    return filtered_routes


def allocate_route_counts(route_entries, total_scenes, rng):
    if total_scenes < len(route_entries):
        raise ValueError(
            f"total-scenes={total_scenes} is smaller than the number of usable routes={len(route_entries)}."
        )

    counts = {
        (route_entry["source_scenario_id"], route_entry["route_id"]): total_scenes // len(route_entries)
        for route_entry in route_entries
    }
    remainder = total_scenes % len(route_entries)
    shuffled_route_entries = route_entries[:]
    rng.shuffle(shuffled_route_entries)
    for route_entry in shuffled_route_entries[:remainder]:
        route_key = (route_entry["source_scenario_id"], route_entry["route_id"])
        counts[route_key] += 1
    return counts


def allocate_route_outcomes(route_entries, route_counts, outcome_pool, rng, paired_outcomes=False):
    route_outcomes = {}
    if paired_outcomes:
        for route_entry in route_entries:
            route_key = (route_entry["source_scenario_id"], route_entry["route_id"])
            route_count = route_counts[route_key]
            if route_count != 2:
                raise ValueError(
                    "When --pilot-paired-outcomes is enabled, each selected route must receive exactly 2 scenes. "
                    f"Route {route_key[0]}:{route_key[1]} received {route_count}."
                )
            route_outcomes[route_key] = ["normal", "collision"]
        return route_outcomes

    shuffled_outcomes = list(outcome_pool)
    rng.shuffle(shuffled_outcomes)
    cursor = 0
    for route_entry in route_entries:
        route_key = (route_entry["source_scenario_id"], route_entry["route_id"])
        route_count = route_counts[route_key]
        route_outcomes[route_key] = shuffled_outcomes[cursor: cursor + route_count]
        cursor += route_count
    return route_outcomes


def build_parameters(
    args,
    route_entry,
    outcome,
    split_name,
    weather_label,
    time_of_day_label,
    scenario_number,
    leading_vehicle_model,
    ego_vehicle_model,
    other_vehicle_model,
):
    candidate = route_entry["best_candidate"]
    return {
        "scenario_type_id": args.scenario_type_id,
        "scenario_subtype_id": args.scenario_subtype_id,
        "scenario_number": scenario_number,
        "target_outcome": outcome,
        "split_name": split_name,
        "weather_label": weather_label,
        "time_of_day_label": time_of_day_label,
        "source_town": route_entry["town"],
        "source_scenario_id": route_entry["source_scenario_id"],
        "source_route_file": route_entry["route_file"],
        "ego_vehicle_model": ego_vehicle_model,
        "leading_vehicle_model": leading_vehicle_model,
        "other_vehicle_model": other_vehicle_model,
        "leading_spawn_mode": "parking",
        "leading_spawn_side": candidate["lane_side"],
        "leading_lane_type": candidate["lane_type"],
        "leading_route_progress_ratio": candidate["route_progress_ratio"],
        "leading_lateral_distance_m": candidate["lateral_distance_m"],
        "leading_driving_anchor_transform": candidate["driving_transform"],
        "leading_roadside_transform": candidate["roadside_transform"],
    }


def build_entries(args, route_entries):
    split_weights = parse_weight_map(args.split_weights)
    outcome_weights = parse_weight_map(args.outcome_weights)
    weather_weights = parse_weight_map(args.weather_weights)
    time_weights = parse_weight_map(args.time_weights)
    rng = random.Random(args.seed)

    route_counts = allocate_route_counts(route_entries, args.total_scenes, rng)
    split_pool, split_quotas = build_balanced_label_pool(args.total_scenes, split_weights, rng)
    outcome_pool, outcome_quotas = build_balanced_label_pool(args.total_scenes, outcome_weights, rng)
    route_outcomes = allocate_route_outcomes(
        route_entries,
        route_counts,
        outcome_pool,
        rng,
        paired_outcomes=args.pilot_paired_outcomes,
    )

    entries = []
    data_id = args.start_data_id
    scenario_number = args.start_data_id + 1

    for route_entry in route_entries:
        route_key = (route_entry["source_scenario_id"], route_entry["route_id"])
        for outcome in route_outcomes[route_key]:
            split_name = split_pool.pop()
            weather_label = weighted_choice(rng, weather_weights)
            time_of_day_label = weighted_choice(rng, time_weights)
            leading_vehicle_model = choose_from_candidates(
                rng, args.leading_vehicle_models, "leading vehicle models"
            )
            ego_vehicle_model = choose_from_candidates(
                rng, args.ego_vehicle_models, "ego vehicle models"
            )
            other_vehicle_model = choose_from_candidates(
                rng, args.other_vehicle_models, "other vehicle models"
            )
            entries.append(
                {
                    "data_id": data_id,
                    "scenario_id": route_entry["source_scenario_id"],
                    "route_id": route_entry["route_id"],
                    "risk_level": None,
                    "parameters": build_parameters(
                        args,
                        route_entry,
                        outcome,
                        split_name,
                        weather_label,
                        time_of_day_label,
                        scenario_number,
                        leading_vehicle_model,
                        ego_vehicle_model,
                        other_vehicle_model,
                    ),
                }
            )
            data_id += 1
            scenario_number += 1

    return entries, route_counts, split_quotas, outcome_quotas


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def replace_scenario_type(base_yaml_text, scenario_type_value):
    replaced = False
    output_lines = []
    for line in base_yaml_text.splitlines():
        if line.startswith("scenario_type:"):
            output_lines.append(f"scenario_type: '{scenario_type_value}'")
            replaced = True
        else:
            output_lines.append(line)

    if not replaced:
        raise ValueError("Base yaml does not contain a scenario_type field")

    return "\n".join(output_lines) + "\n"


def resolve_scenario_type_value(json_path, args):
    if args.scenario_type_prefix:
        prefix = args.scenario_type_prefix.strip("/\\")
        return f"{prefix}/{json_path.name}" if prefix else json_path.name

    scenario_type_root = (
        args.scenario_type_root.resolve()
        if args.scenario_type_root is not None
        else (args.root_dir.resolve() / "frea" / "scenario" / "config" / "scenario_type")
    )
    try:
        relative_path = json_path.resolve().relative_to(scenario_type_root)
        return relative_path.as_posix()
    except ValueError:
        return json_path.name


def write_yaml(base_yaml_text, json_path, yaml_path, args):
    scenario_type_value = resolve_scenario_type_value(json_path, args)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        replace_scenario_type(base_yaml_text, scenario_type_value),
        encoding="utf-8",
    )


def summarize_entries(entries):
    split_counter = Counter()
    outcome_counter = Counter()
    weather_counter = Counter()
    time_counter = Counter()

    for entry in entries:
        parameters = entry["parameters"]
        split_counter[parameters["split_name"]] += 1
        outcome_counter[parameters["target_outcome"]] += 1
        weather_counter[parameters["weather_label"]] += 1
        time_counter[parameters["time_of_day_label"]] += 1

    return split_counter, outcome_counter, weather_counter, time_counter


def main():
    args = parse_args()
    if args.base_yaml is None and (args.yaml_output is not None or args.split_yaml_dir is not None):
        raise ValueError("--base-yaml is required when writing YAML outputs.")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    route_entries = filter_routes(args, data)
    entries, route_counts, split_quotas, outcome_quotas = build_entries(args, route_entries)

    write_json(args.output, entries)

    split_output_paths = {}
    if args.split_json_dir is not None:
        split_groups = {}
        for entry in entries:
            split_name = entry["parameters"]["split_name"]
            split_groups.setdefault(split_name, []).append(entry)

        for split_name, split_entries in sorted(split_groups.items()):
            split_path = args.split_json_dir / f"{args.output.stem}_{split_name}.json"
            write_json(split_path, split_entries)
            split_output_paths[split_name] = split_path

    if args.base_yaml is not None:
        base_yaml_text = args.base_yaml.read_text(encoding="utf-8")
        if args.yaml_output is not None:
            write_yaml(base_yaml_text, args.output, args.yaml_output, args)
        if args.split_yaml_dir is not None:
            yaml_prefix = args.yaml_prefix or args.output.stem
            if not split_output_paths:
                raise ValueError("--split-yaml-dir requires --split-json-dir so each split has a JSON file.")
            for split_name, split_json_path in sorted(split_output_paths.items()):
                split_yaml_path = args.split_yaml_dir / f"{yaml_prefix}_{split_name}.yaml"
                write_yaml(base_yaml_text, split_json_path, split_yaml_path, args)

    split_counter, outcome_counter, weather_counter, time_counter = summarize_entries(entries)

    print(f"Generated {len(entries)} type-2 scenes -> {args.output}")
    print(f"Split quotas target: {dict(split_quotas)}")
    print(f"Split counts actual: {dict(split_counter)}")
    print(f"Outcome quotas target: {dict(outcome_quotas)}")
    print(f"Outcome counts actual: {dict(outcome_counter)}")
    print(f"Weather counts actual: {dict(weather_counter)}")
    print(f"Time counts actual: {dict(time_counter)}")
    print("Route allocation:")
    for route_entry in route_entries:
        route_key = (route_entry["source_scenario_id"], route_entry["route_id"])
        candidate = route_entry["best_candidate"]
        print(
            "  "
            f"town={route_entry['town']} "
            f"scenario={route_entry['source_scenario_id']} "
            f"route={route_entry['route_id']} "
            f"count={route_counts[route_key]} "
            f"progress={candidate['route_progress_ratio']} "
            f"lateral={candidate['lateral_distance_m']}"
        )
    if split_output_paths:
        for split_name, split_path in sorted(split_output_paths.items()):
            print(f"Split JSON: {split_name} -> {split_path}")
    if args.yaml_output is not None:
        print(f"Combined YAML -> {args.yaml_output}")
    if args.split_yaml_dir is not None:
        print(f"Split YAML directory -> {args.split_yaml_dir}")


if __name__ == "__main__":
    main()
