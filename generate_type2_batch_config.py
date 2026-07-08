#!/usr/bin/env python
import argparse
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_ROUTE_POOL = [
    "1:9",
    "5:4",
    "5:5",
    "5:6",
    "5:8",
    "5:10",
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
    "near_miss": 3.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a batch scenario_type JSON for FREA type-2 parking scenes."
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
        default=75,
        help="Total number of scenes to generate.",
    )
    parser.add_argument(
        "--towns",
        nargs="+",
        default=None,
        help="Optional subset of towns to keep.",
    )
    parser.add_argument(
        "--min-route-progress",
        type=float,
        default=0.05,
        help="Drop parking candidates whose route progress ratio is below this threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible allocation.",
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
        help="Metadata scenario type id written into parameters.",
    )
    parser.add_argument(
        "--scenario-subtype-id",
        type=int,
        default=1,
        help="Metadata scenario subtype id written into parameters.",
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
        help="Optional outcome weights, e.g. collision=7 near_miss=3.",
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


def select_candidate(route_entry, min_route_progress):
    candidates = route_entry.get("candidates", [])
    if not candidates and route_entry.get("best_candidate") is not None:
        candidates = [route_entry["best_candidate"]]

    parking_candidates = []
    for candidate in candidates:
        if candidate.get("lane_type") != "parking":
            continue
        if float(candidate.get("route_progress_ratio", 0.0)) < min_route_progress:
            continue
        parking_candidates.append(candidate)

    if not parking_candidates:
        return None
    parking_candidates.sort(
        key=lambda candidate: (
            abs(candidate["route_progress_ratio"] - 0.35),
            candidate["lateral_distance_m"],
        )
    )
    return parking_candidates[0]


def build_route_lookup(scan_data, min_route_progress, selected_towns):
    route_lookup = {}
    for route_entry in scan_data.get("routes", []):
        if selected_towns is not None and route_entry["town"] not in selected_towns:
            continue
        if not route_entry.get("has_candidate"):
            continue
        candidate = select_candidate(route_entry, min_route_progress)
        if candidate is None:
            continue
        route_lookup[(route_entry["source_scenario_id"], route_entry["route_id"])] = {
            **route_entry,
            "selected_candidate": candidate,
        }
    return route_lookup


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


def weighted_choice(rng, weight_map):
    labels = list(weight_map.keys())
    weights = list(weight_map.values())
    return rng.choices(labels, weights=weights, k=1)[0]


def build_parameters(args, route_entry, scenario_number, split_name, outcome_label, weather_label, time_label):
    candidate = route_entry["selected_candidate"]
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
        "leading_spawn_mode": "parking",
        "leading_spawn_side": candidate["lane_side"],
        "leading_lane_type": candidate["lane_type"],
        "leading_route_progress_ratio": candidate["route_progress_ratio"],
        "leading_lateral_distance_m": candidate["lateral_distance_m"],
        "leading_driving_anchor_transform": candidate["driving_transform"],
        "leading_roadside_transform": candidate["roadside_transform"],
    }


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


def main():
    args = parse_args()
    if args.total_scenes <= 0:
        raise ValueError("--total-scenes must be positive")

    weather_weights = parse_weight_map(args.weather_weights, DEFAULT_WEATHER_WEIGHTS)
    time_weights = parse_weight_map(args.time_weights, DEFAULT_TIME_WEIGHTS)
    split_weights = parse_weight_map(args.split_weights, DEFAULT_SPLIT_WEIGHTS)
    outcome_weights = parse_weight_map(args.outcome_weights, DEFAULT_OUTCOME_WEIGHTS)
    route_pool = parse_route_pool(args.route_pool)
    selected_towns = set(args.towns) if args.towns is not None else None

    scan_data = json.loads(args.input.read_text(encoding="utf-8"))
    route_lookup = build_route_lookup(scan_data, args.min_route_progress, selected_towns)

    missing_routes = [route_key for route_key in route_pool if route_key not in route_lookup]
    if missing_routes:
        missing_text = ", ".join(f"{scenario_id}:{route_id}" for scenario_id, route_id in missing_routes)
        raise ValueError(f"Selected routes are missing usable parking candidates: {missing_text}")

    route_allocation = allocate_route_counts(args.total_scenes, route_pool)
    split_quota = build_exact_quota(args.total_scenes, split_weights, ["train", "val", "test"])
    outcome_quota = build_exact_quota(args.total_scenes, outcome_weights, ["collision", "near_miss"])

    split_labels = []
    for split_name in ["train", "val", "test"]:
        split_labels.extend([split_name] * split_quota.get(split_name, 0))

    outcome_labels = []
    for outcome_name in ["collision", "near_miss"]:
        outcome_labels.extend([outcome_name] * outcome_quota.get(outcome_name, 0))

    rng = random.Random(args.seed)
    rng.shuffle(split_labels)
    rng.shuffle(outcome_labels)

    route_instances = []
    for route_key in route_pool:
        route_instances.extend([route_key] * route_allocation[route_key])
    rng.shuffle(route_instances)

    entries = []
    for offset, route_key in enumerate(route_instances):
        route_entry = route_lookup[route_key]
        split_name = split_labels[offset]
        outcome_label = outcome_labels[offset]
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
                "parameters": build_parameters(
                    args,
                    route_entry,
                    scenario_number,
                    split_name,
                    outcome_label,
                    weather_label,
                    time_label,
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
        print(f"  scenario={scenario_id} route={route_id} -> {route_usage[(scenario_id, route_id)]}")
    print(f"Split counts: {dict(split_usage)}")
    print(f"Outcome counts: {dict(outcome_usage)}")
    print(f"Weather counts: {dict(weather_usage)}")
    print(f"Time counts: {dict(time_usage)}")


if __name__ == "__main__":
    main()
