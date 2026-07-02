#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate custom type-2 pilot test cases from parking-only route scan results."
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
        help="Output scenario_type JSON path.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=["normal", "collision"],
        choices=["normal", "collision"],
        help="Outcome labels to generate for each selected parking route.",
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
        "--max-routes",
        type=int,
        default=None,
        help="Optional cap on the number of distinct parking routes to keep after ranking.",
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
        "--split-name",
        type=str,
        default="pilot",
        help="Split label written into parameters.",
    )
    return parser.parse_args()


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


def build_parameters(args, route_entry, outcome, scenario_number):
    candidate = route_entry["best_candidate"]
    return {
        "scenario_type_id": args.scenario_type_id,
        "scenario_subtype_id": args.scenario_subtype_id,
        "scenario_number": scenario_number,
        "target_outcome": outcome,
        "split_name": args.split_name,
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


def main():
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    routes = data.get("routes", [])

    filtered_routes = []
    for route_entry in routes:
        candidate = route_entry.get("best_candidate")
        if not route_entry.get("has_candidate") or candidate is None:
            continue
        if candidate.get("lane_type") != "parking":
            continue
        if args.towns is not None and route_entry["town"] not in args.towns:
            continue
        if args.source_scenarios is not None and route_entry["source_scenario_id"] not in args.source_scenarios:
            continue
        filtered_routes.append(route_entry)

    filtered_routes.sort(key=route_priority)
    if args.max_routes is not None:
        filtered_routes = filtered_routes[: args.max_routes]

    entries = []
    data_id = args.start_data_id
    scenario_number = args.start_data_id + 1
    for route_entry in filtered_routes:
        for outcome in args.outcomes:
            entries.append(
                {
                    "data_id": data_id,
                    "scenario_id": route_entry["source_scenario_id"],
                    "route_id": route_entry["route_id"],
                    "risk_level": None,
                    "parameters": build_parameters(args, route_entry, outcome, scenario_number),
                }
            )
            data_id += 1
            scenario_number += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"Generated {len(entries)} test cases from {len(filtered_routes)} parking routes -> {args.output}")
    for route_entry in filtered_routes:
        candidate = route_entry["best_candidate"]
        print(
            "  "
            f"town={route_entry['town']} "
            f"scenario={route_entry['source_scenario_id']} "
            f"route={route_entry['route_id']} "
            f"progress={candidate['route_progress_ratio']} "
            f"lateral={candidate['lateral_distance_m']}"
        )


if __name__ == "__main__":
    main()
