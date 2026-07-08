#!/usr/bin/env python
import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


DEFAULT_ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ROUTE_POOL = [
    "1:9",
    "5:4",
    "5:5",
    "5:6",
    "5:8",
    "5:10",
    "3:5",
    "3:9",
    "3:4",
]
DEFAULT_WEATHER_WEIGHTS = {
    "clear": 4,
    "rainy": 4,
    "cloudy": 1,
    "wet": 1,
}
DEFAULT_TIME_WEIGHTS = {
    "noon": 3,
    "sunset": 1,
    "night": 1,
}
DEFAULT_SPLIT_WEIGHTS = {
    "train": 7.0,
    "val": 1.5,
    "test": 1.5,
}
DEFAULT_OUTCOME_WEIGHTS = {
    "collision": 7.0,
    "normal": 3.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a batch scenario_type JSON for FREA type-1 loss-of-control scenes."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Repository root that contains frea/scenario/scenario_data/route.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--route-pool",
        nargs="+",
        default=DEFAULT_ROUTE_POOL,
        help="Selected route pool as scenario_id:route_id pairs.",
    )
    parser.add_argument(
        "--total-scenes",
        type=int,
        default=130,
        help="Total number of scenes to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible generation.",
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
        default=1,
        help="Metadata scenario type id written into parameters.",
    )
    parser.add_argument(
        "--scenario-subtype-id",
        type=int,
        default=1,
        help="Metadata scenario subtype id written into parameters.",
    )
    parser.add_argument(
        "--route-start-max-fraction",
        type=float,
        default=0.4,
        help="Maximum randomized ego route start fraction.",
    )
    parser.add_argument(
        "--ego-loss-trigger-min-seconds",
        type=float,
        default=5.8,
        help="Minimum loss-of-control trigger time for collision scenes.",
    )
    parser.add_argument(
        "--ego-loss-trigger-max-seconds",
        type=float,
        default=6.2,
        help="Maximum loss-of-control trigger time for collision scenes.",
    )
    parser.add_argument(
        "--weather-weights",
        nargs="+",
        default=None,
        help="Optional weather weights, e.g. clear=4 rainy=4 cloudy=1 wet=1.",
    )
    parser.add_argument(
        "--time-weights",
        nargs="+",
        default=None,
        help="Optional time-of-day weights, e.g. noon=3 sunset=1 night=1.",
    )
    parser.add_argument(
        "--split-weights",
        nargs="+",
        default=None,
        help="Optional split weights, e.g. train=7 val=1.5 test=1.5.",
    )
    parser.add_argument(
        "--outcome-weights",
        nargs="+",
        default=None,
        help="Optional outcome weights, e.g. collision=7 normal=3 or collision=7 near_miss=3.",
    )
    parser.add_argument(
        "--route-fixed-outcomes",
        nargs="+",
        default=None,
        help="Optional outcomes that must appear once per route before filling remaining scenes, e.g. collision normal.",
    )
    return parser.parse_args()


def parse_weight_map(entries, default_map):
    if not entries:
        return dict(default_map)

    parsed = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid weight entry: {entry}. Expected key=value")
        key, value_text = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty key in weight entry: {entry}")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in weight entry: {entry}") from exc
        if value < 0:
            raise ValueError(f"Weight must be non-negative: {entry}")
        parsed[key] = value
    return parsed


def parse_route_pool(route_pool_entries):
    parsed = []
    for entry in route_pool_entries:
        if ":" not in entry:
            raise ValueError(f"Invalid route pool entry: {entry}. Expected scenario_id:route_id")
        scenario_text, route_text = entry.split(":", 1)
        try:
            scenario_id = int(scenario_text)
            route_id = int(route_text)
        except ValueError as exc:
            raise ValueError(f"Invalid route ids in route pool entry: {entry}") from exc
        parsed.append((scenario_id, route_id))
    return parsed


def build_exact_quota(total_count, weight_map, key_order):
    if total_count < 0:
        raise ValueError("total_count must be non-negative")

    positive_items = [(key, float(weight_map.get(key, 0.0))) for key in key_order]
    positive_items = [(key, weight) for key, weight in positive_items if weight > 0]
    if not positive_items:
        raise ValueError("At least one positive weight is required")

    total_weight = sum(weight for _, weight in positive_items)
    raw_counts = []
    assigned = 0
    for order_index, (key, weight) in enumerate(positive_items):
        raw = total_count * weight / total_weight
        count = int(raw)
        raw_counts.append((key, count, raw - count, order_index))
        assigned += count

    remaining = total_count - assigned
    raw_counts.sort(key=lambda item: (-item[2], item[3]))
    for index in range(remaining):
        key, count, fraction, order_index = raw_counts[index]
        raw_counts[index] = (key, count + 1, fraction, order_index)

    raw_counts.sort(key=lambda item: item[3])
    return {key: count for key, count, _, _ in raw_counts}


def allocate_route_counts(total_scenes, route_pool):
    route_count = len(route_pool)
    if route_count == 0:
        raise ValueError("Route pool must not be empty")

    base = total_scenes // route_count
    remainder = total_scenes % route_count
    allocation = {}
    for index, route_key in enumerate(route_pool):
        allocation[route_key] = base + (1 if index < remainder else 0)
    return allocation


def weighted_choice(rng, weight_map):
    labels = list(weight_map.keys())
    weights = list(weight_map.values())
    return rng.choices(labels, weights=weights, k=1)[0]


def build_route_fixed_instances(route_pool, fixed_outcomes):
    instances = []
    for route_key in route_pool:
        for outcome_label in fixed_outcomes:
            instances.append((route_key, outcome_label))
    return instances


def discover_route_lookup(root_dir, route_pool):
    route_lookup = {}
    route_dir_root = root_dir / "frea" / "scenario" / "scenario_data" / "route"

    for scenario_id, route_id in route_pool:
        route_file = (
            route_dir_root
            / f"scenario_{scenario_id:02d}_routes"
            / f"scenario_{scenario_id:02d}_route_{route_id:02d}.xml"
        )
        if not route_file.exists():
            raise FileNotFoundError(f"Route file not found: {route_file}")

        tree = ET.parse(route_file)
        route_node = tree.getroot().find("route")
        if route_node is None:
            raise ValueError(f"Route file missing <route> node: {route_file}")

        town = route_node.attrib.get("town")
        if town is None:
            raise ValueError(f"Route file missing route town attribute: {route_file}")

        route_lookup[(scenario_id, route_id)] = {
            "town": town,
            "route_file": str(route_file.relative_to(root_dir)).replace("\\", "/"),
            "source_scenario_id": scenario_id,
            "route_id": route_id,
        }

    return route_lookup


def build_type1_parameters(args, route_entry, scenario_number, split_name, outcome_label, weather_label, time_label, rng):
    other_lane_side = rng.choice(["left", "right"])
    other_spawn_mode = "adjacent_rear"
    if outcome_label != "collision" and rng.random() < 0.25:
        other_spawn_mode = "same_lane_rear"

    if other_spawn_mode == "same_lane_rear":
        other_distance_back_m = round(rng.uniform(10.0, 15.0), 2)
    else:
        other_distance_back_m = round(rng.uniform(8.0, 12.0), 2)
    other_target_speed_mps = round(rng.uniform(7.8, 8.8), 2)
    other_speed_variation_mps = round(rng.uniform(0.05, 0.12), 2)
    other_follow_speed_offset_mps = round(rng.uniform(0.2, 0.5), 2)
    route_start_ratio = round(rng.uniform(0.0, args.route_start_max_fraction), 4)
    ego_loss_direction = other_lane_side
    ego_loss_trigger_seconds = round(
        rng.uniform(args.ego_loss_trigger_min_seconds, args.ego_loss_trigger_max_seconds),
        2,
    )
    ego_loss_duration_seconds = round(rng.uniform(2.3, 2.8), 2)
    ego_loss_ramp_seconds = round(rng.uniform(0.8, 1.1), 2)
    ego_loss_steer_magnitude = round(rng.uniform(0.58, 0.72), 2)

    if outcome_label != "collision":
        ego_loss_trigger_seconds = 6.0
        ego_loss_duration_seconds = 2.5
        ego_loss_ramp_seconds = 0.9
        ego_loss_steer_magnitude = 0.65

    return {
        "scenario_type_id": args.scenario_type_id,
        "scenario_subtype_id": args.scenario_subtype_id,
        "scenario_number": scenario_number,
        "target_outcome": outcome_label,
        "weather_label": weather_label,
        "time_of_day_label": time_label,
        "split_name": split_name,
        "source_town": route_entry["town"],
        "source_scenario_id": route_entry["source_scenario_id"],
        "source_route_file": route_entry["route_file"],
        "route_start_ratio": route_start_ratio,
        "route_start_min_remaining_points": 35,
        "scene_total_seconds": 10.0,
        "other_spawn_mode": other_spawn_mode,
        "other_lane_side": other_lane_side,
        "other_distance_back_m": other_distance_back_m,
        "other_target_speed_mps": other_target_speed_mps,
        "other_speed_variation_mps": other_speed_variation_mps,
        "other_follow_speed_offset_mps": other_follow_speed_offset_mps,
        "other_lookahead_distance_m": 10.0,
        "ego_loss_direction": ego_loss_direction,
        "ego_loss_trigger_seconds": ego_loss_trigger_seconds,
        "ego_loss_duration_seconds": ego_loss_duration_seconds,
        "ego_loss_ramp_seconds": ego_loss_ramp_seconds,
        "ego_loss_steer_magnitude": ego_loss_steer_magnitude,
        "ego_loss_min_throttle": 0.25,
        "ego_loss_max_brake": 0.0,
    }


def main():
    args = parse_args()
    if args.total_scenes <= 0:
        raise ValueError("--total-scenes must be positive")
    if args.route_start_max_fraction < 0.0:
        raise ValueError("--route-start-max-fraction must be non-negative")
    if args.ego_loss_trigger_max_seconds < args.ego_loss_trigger_min_seconds:
        raise ValueError("--ego-loss-trigger-max-seconds must be >= --ego-loss-trigger-min-seconds")

    weather_weights = parse_weight_map(args.weather_weights, DEFAULT_WEATHER_WEIGHTS)
    time_weights = parse_weight_map(args.time_weights, DEFAULT_TIME_WEIGHTS)
    split_weights = parse_weight_map(args.split_weights, DEFAULT_SPLIT_WEIGHTS)
    outcome_weights = parse_weight_map(args.outcome_weights, DEFAULT_OUTCOME_WEIGHTS)
    route_pool = parse_route_pool(args.route_pool)
    route_lookup = discover_route_lookup(args.root_dir, route_pool)
    fixed_outcomes = list(args.route_fixed_outcomes or [])

    if fixed_outcomes:
        missing_fixed_outcomes = [label for label in fixed_outcomes if label not in outcome_weights]
        if missing_fixed_outcomes:
            raise ValueError(
                "All --route-fixed-outcomes must also exist in --outcome-weights. "
                f"Missing: {missing_fixed_outcomes}"
            )

    fixed_route_instances = build_route_fixed_instances(route_pool, fixed_outcomes)
    if len(fixed_route_instances) > args.total_scenes:
        raise ValueError(
            "--total-scenes is smaller than the number of required fixed route/outcome pairs: "
            f"{len(fixed_route_instances)}"
        )

    remaining_scene_count = args.total_scenes - len(fixed_route_instances)
    route_allocation = allocate_route_counts(remaining_scene_count, route_pool)
    split_quota = build_exact_quota(args.total_scenes, split_weights, ["train", "val", "test"])
    outcome_order = list(outcome_weights.keys())
    outcome_quota = build_exact_quota(remaining_scene_count, outcome_weights, outcome_order)

    split_labels = []
    for split_name in ["train", "val", "test"]:
        split_labels.extend([split_name] * split_quota.get(split_name, 0))

    outcome_labels = []
    for outcome_name in outcome_order:
        outcome_labels.extend([outcome_name] * outcome_quota.get(outcome_name, 0))

    rng = random.Random(args.seed)
    rng.shuffle(split_labels)
    rng.shuffle(outcome_labels)

    route_instances = list(fixed_route_instances)
    for route_key in route_pool:
        route_instances.extend([route_key] * route_allocation[route_key])
    dynamic_route_instances = route_instances[len(fixed_route_instances):]
    rng.shuffle(dynamic_route_instances)
    route_instances = fixed_route_instances + dynamic_route_instances

    entries = []
    for offset, route_key in enumerate(route_instances):
        split_name = split_labels[offset]
        fixed_outcome_index = offset if offset < len(fixed_route_instances) else None
        if fixed_outcome_index is not None:
            route_key, outcome_label = fixed_route_instances[fixed_outcome_index]
        else:
            outcome_label = outcome_labels[offset - len(fixed_route_instances)]
        route_entry = route_lookup[route_key]
        weather_label = weighted_choice(rng, weather_weights)
        time_label = weighted_choice(rng, time_weights)
        data_id = args.start_data_id + offset
        scenario_number = data_id + 1

        entries.append(
            {
                "data_id": data_id,
                "scenario_id": route_entry["source_scenario_id"],
                "route_id": route_entry["route_id"],
                "risk_level": None,
                "parameters": build_type1_parameters(
                    args,
                    route_entry,
                    scenario_number,
                    split_name,
                    outcome_label,
                    weather_label,
                    time_label,
                    rng,
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(entries, output_file, indent=2)

    route_usage = Counter((entry["scenario_id"], entry["route_id"]) for entry in entries)
    split_usage = Counter(entry["parameters"]["split_name"] for entry in entries)
    outcome_usage = Counter(entry["parameters"]["target_outcome"] for entry in entries)
    weather_usage = Counter(entry["parameters"]["weather_label"] for entry in entries)
    time_usage = Counter(entry["parameters"]["time_of_day_label"] for entry in entries)

    print(f"Generated {len(entries)} entries -> {args.output}")
    print("Route allocation:")
    for scenario_id, route_id in route_pool:
        town = route_lookup[(scenario_id, route_id)]["town"]
        print(f"  town={town} scenario={scenario_id} route={route_id} -> {route_usage[(scenario_id, route_id)]}")
    print(f"Split counts: {dict(split_usage)}")
    print(f"Outcome counts: {dict(outcome_usage)}")
    print(f"Weather counts: {dict(weather_usage)}")
    print(f"Time counts: {dict(time_usage)}")


if __name__ == "__main__":
    main()
