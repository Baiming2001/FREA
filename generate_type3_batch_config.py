#!/usr/bin/env python
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06"]
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
DEFAULT_OUTCOME_WEIGHTS = {
    "collision": 7,
    "near_miss": 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a batch scenario_type JSON for FREA type-3 scenes."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "FREA",
        help="Repository root that contains frea/scenario/...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=3,
        help="Scenario id to scan routes for.",
    )
    parser.add_argument(
        "--towns",
        nargs="+",
        default=DEFAULT_TOWNS,
        help="Town list to include, e.g. Town01 Town02 ...",
    )
    parser.add_argument(
        "--per-town",
        type=int,
        default=15,
        help="Number of scenes to generate per town.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--scenario-type-id",
        type=int,
        default=3,
        help="Metadata scenario type id written into parameters.",
    )
    parser.add_argument(
        "--scenario-subtype-id",
        type=int,
        default=1,
        help="Metadata scenario subtype id written into parameters.",
    )
    return parser.parse_args()


def weighted_choice(rng, weight_dict):
    labels = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return rng.choices(labels, weights=weights, k=1)[0]


def discover_routes(root_dir, scenario_id):
    route_dir = root_dir / "frea" / "scenario" / "scenario_data" / "route" / f"scenario_{scenario_id:02d}_routes"
    if not route_dir.exists():
        raise FileNotFoundError(f"Route directory not found: {route_dir}")

    routes_by_town = defaultdict(list)
    for route_file in sorted(route_dir.glob(f"scenario_{scenario_id:02d}_route_*.xml")):
        route_id = int(route_file.stem.split("_")[-1])
        tree = ET.parse(route_file)
        route_node = tree.getroot().find("route")
        if route_node is None:
            continue
        town = route_node.attrib.get("town")
        if town is None:
            continue
        routes_by_town[town].append(route_id)

    for town, route_ids in routes_by_town.items():
        route_ids.sort()
    return routes_by_town


def generate_entries(args, routes_by_town):
    rng = random.Random(args.seed)
    entries = []
    data_id = 0
    available_towns = sorted(routes_by_town.keys())

    for town in args.towns:
        route_ids = routes_by_town.get(town, [])
        if not route_ids:
            raise ValueError(
                f"No routes discovered for {town} under scenario_{args.scenario_id:02d}_routes. "
                f"Available towns: {available_towns}"
            )

        shuffled_routes = route_ids[:]
        rng.shuffle(shuffled_routes)
        for index in range(args.per_town):
            route_id = shuffled_routes[index % len(shuffled_routes)]
            weather_label = weighted_choice(rng, DEFAULT_WEATHER_WEIGHTS)
            time_of_day_label = weighted_choice(rng, DEFAULT_TIME_WEIGHTS)
            target_outcome = weighted_choice(rng, DEFAULT_OUTCOME_WEIGHTS)

            entries.append({
                "data_id": data_id,
                "scenario_id": args.scenario_id,
                "route_id": route_id,
                "risk_level": None,
                "parameters": {
                    "scenario_type_id": args.scenario_type_id,
                    "scenario_subtype_id": args.scenario_subtype_id,
                    "scenario_number": data_id + 1,
                    "target_outcome": target_outcome,
                    "weather_label": weather_label,
                    "time_of_day_label": time_of_day_label,
                },
            })
            data_id += 1

    return entries


def main():
    args = parse_args()
    routes_by_town = discover_routes(args.root_dir, args.scenario_id)
    entries = generate_entries(args, routes_by_town)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2)

    print(f"Generated {len(entries)} entries -> {args.output}")
    for town in args.towns:
        route_count = len(routes_by_town.get(town, []))
        print(f"{town}: {args.per_town} scenes using {route_count} available routes")


if __name__ == "__main__":
    main()
