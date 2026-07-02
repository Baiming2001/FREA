#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan FREA route files and find route waypoints suitable for the redefined "
            "type-2 scenario where the leading car starts from a roadside parking position."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root that contains frea/scenario/...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for structured scan results.",
    )
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=2,
        help="Scenario route directory to scan, e.g. 2 -> scenario_02_routes.",
    )
    parser.add_argument(
        "--towns",
        nargs="+",
        default=["Town01", "Town02", "Town03", "Town04", "Town05"],
        help="Town list to include.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="CARLA host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA port.",
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=60.0,
        help="CARLA client timeout in seconds.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=10,
        help="Sample every N interpolated route waypoints while scanning.",
    )
    parser.add_argument(
        "--max-route-progress",
        type=float,
        default=0.7,
        help="Only scan candidate leading-car start points within this route progress ratio.",
    )
    parser.add_argument(
        "--min-route-progress",
        type=float,
        default=0.05,
        help="Skip the earliest route points to avoid starts too close to the route origin.",
    )
    parser.add_argument(
        "--max-candidates-per-route",
        type=int,
        default=8,
        help="How many candidate points to keep per route in the JSON output.",
    )
    return parser.parse_args()


def ensure_repo_on_path(root_dir: Path):
    repo_str = str(root_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def direction_dot(waypoint_a, waypoint_b):
    forward_a = waypoint_a.transform.get_forward_vector()
    forward_b = waypoint_b.transform.get_forward_vector()
    return (
        forward_a.x * forward_b.x
        + forward_a.y * forward_b.y
        + forward_a.z * forward_b.z
    )


def is_same_direction(reference_waypoint, candidate_waypoint, min_dot=0.35):
    if reference_waypoint is None or candidate_waypoint is None:
        return False
    return direction_dot(reference_waypoint, candidate_waypoint) >= min_dot


def iter_outer_lanes(reference_waypoint, lane_side, max_hops=6):
    current = reference_waypoint
    for hop in range(1, max_hops + 1):
        current = current.get_right_lane() if lane_side == "right" else current.get_left_lane()
        if current is None:
            return
        yield hop, current


def classify_lane_type(carla_module, waypoint):
    if waypoint.lane_type == carla_module.LaneType.Parking:
        return "parking"
    if waypoint.lane_type == carla_module.LaneType.Shoulder:
        return "shoulder"
    if waypoint.lane_type == carla_module.LaneType.Driving:
        return "driving"
    return str(waypoint.lane_type)


def find_roadside_candidate(carla_module, driving_waypoint, lane_side):
    best_candidate = None
    best_rank = None
    best_score = None

    for hop, candidate in iter_outer_lanes(driving_waypoint, lane_side):
        lane_label = classify_lane_type(carla_module, candidate)
        if lane_label not in {"parking", "shoulder"}:
            continue
        if not is_same_direction(driving_waypoint, candidate):
            continue

        lateral_distance = candidate.transform.location.distance(driving_waypoint.transform.location)
        lane_rank = 0 if lane_label == "parking" else 1
        score = lateral_distance + hop * 0.5

        if best_candidate is None or (lane_rank, score) < (best_rank, best_score):
            best_candidate = candidate
            best_rank = lane_rank
            best_score = score

    return best_candidate


def build_candidate_record(carla_module, driving_waypoint, roadside_waypoint, lane_side, route_index, route_length):
    driving_loc = driving_waypoint.transform.location
    roadside_loc = roadside_waypoint.transform.location
    return {
        "route_index": route_index,
        "route_progress_ratio": round(route_index / max(1, route_length - 1), 4),
        "lane_side": lane_side,
        "lane_type": classify_lane_type(carla_module, roadside_waypoint),
        "lane_id": roadside_waypoint.lane_id,
        "road_id": roadside_waypoint.road_id,
        "lateral_distance_m": round(driving_loc.distance(roadside_loc), 3),
        "driving_transform": {
            "x": round(driving_loc.x, 3),
            "y": round(driving_loc.y, 3),
            "z": round(driving_loc.z, 3),
            "yaw": round(driving_waypoint.transform.rotation.yaw, 3),
        },
        "roadside_transform": {
            "x": round(roadside_loc.x, 3),
            "y": round(roadside_loc.y, 3),
            "z": round(roadside_loc.z, 3),
            "yaw": round(roadside_waypoint.transform.rotation.yaw, 3),
        },
    }


def route_candidate_priority(candidate):
    lane_priority = 0 if candidate["lane_type"] == "parking" else 1
    progress_target = abs(candidate["route_progress_ratio"] - 0.35)
    return (lane_priority, progress_target, candidate["lateral_distance_m"])


def discover_route_files(root_dir, scenario_id, towns):
    route_dir = root_dir / "frea" / "scenario" / "scenario_data" / "route" / f"scenario_{scenario_id:02d}_routes"
    if not route_dir.exists():
        raise FileNotFoundError(f"Route directory not found: {route_dir}")

    from frea.scenario.tools.route_parser import RouteParser

    route_records = []
    for route_file in sorted(route_dir.glob(f"scenario_{scenario_id:02d}_route_*.xml")):
        parsed_routes = RouteParser.parse_routes_file(
            str(route_file),
            str(root_dir / "frea" / "scenario" / "scenario_data" / "route" / "scenarios" / f"scenario_{scenario_id:02d}.json")
        )
        if not parsed_routes:
            continue

        route_config = parsed_routes[0]
        if route_config.town not in towns:
            continue

        route_records.append(
            {
                "town": route_config.town,
                "route_id": int(route_file.stem.split("_")[-1]),
                "route_file": route_file,
                "config": route_config,
            }
        )

    if not route_records:
        raise ValueError(
            f"No route files matched towns {towns} under scenario_{scenario_id:02d}_routes"
        )

    return route_records


def scan_route(world, route_config, args, carla_module):
    from frea.scenario.tools.route_manipulation import interpolate_trajectory

    _, dense_route = interpolate_trajectory(world, route_config.trajectory, 5.0)
    if not dense_route:
        return {
            "route_length": 0,
            "sampled_points": 0,
            "candidate_count": 0,
            "has_candidate": False,
            "best_candidate": None,
            "candidates": [],
        }

    total_points = len(dense_route)
    start_index = min(total_points - 1, max(0, int(math.floor(args.min_route_progress * total_points))))
    end_index = min(total_points, max(start_index + 1, int(math.ceil(args.max_route_progress * total_points))))

    candidates = []
    sampled_points = 0
    carla_map = world.get_map()

    for route_index in range(start_index, end_index, max(1, args.sample_step)):
        route_transform, _ = dense_route[route_index]
        driving_waypoint = carla_map.get_waypoint(
            route_transform.location,
            project_to_road=True,
            lane_type=carla_module.LaneType.Driving,
        )
        if driving_waypoint is None:
            continue

        sampled_points += 1
        for lane_side in ("right", "left"):
            roadside_waypoint = find_roadside_candidate(carla_module, driving_waypoint, lane_side)
            if roadside_waypoint is None:
                continue

            candidates.append(
                build_candidate_record(
                    carla_module,
                    driving_waypoint,
                    roadside_waypoint,
                    lane_side,
                    route_index,
                    total_points,
                )
            )

    candidates.sort(key=route_candidate_priority)
    limited_candidates = candidates[: max(1, args.max_candidates_per_route)]
    best_candidate = limited_candidates[0] if limited_candidates else None
    return {
        "route_length": total_points,
        "sampled_points": sampled_points,
        "candidate_count": len(candidates),
        "has_candidate": bool(best_candidate),
        "best_candidate": best_candidate,
        "candidates": limited_candidates,
    }


def main():
    args = parse_args()
    root_dir = args.root_dir.resolve()
    ensure_repo_on_path(root_dir)

    import carla

    route_records = discover_route_files(root_dir, args.scenario_id, set(args.towns))
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.client_timeout)

    output = {
        "scenario_id": args.scenario_id,
        "towns": args.towns,
        "scan_parameters": {
            "sample_step": args.sample_step,
            "min_route_progress": args.min_route_progress,
            "max_route_progress": args.max_route_progress,
            "max_candidates_per_route": args.max_candidates_per_route,
        },
        "towns_summary": {},
        "routes": [],
    }

    route_records.sort(key=lambda item: (item["town"], item["route_id"]))
    current_town = None
    world = None
    town_usable_count = {}
    town_total_count = {}

    for route_record in route_records:
        town = route_record["town"]
        route_id = route_record["route_id"]
        if town != current_town:
            print(f"Loading world: {town}")
            world = client.load_world(town)
            current_town = town

        print(f"Scanning {town} route {route_id:02d}")
        scan_result = scan_route(world, route_record["config"], args, carla)
        town_total_count[town] = town_total_count.get(town, 0) + 1
        if scan_result["has_candidate"]:
            town_usable_count[town] = town_usable_count.get(town, 0) + 1

        output["routes"].append(
            {
                "town": town,
                "route_id": route_id,
                "route_file": str(route_record["route_file"]),
                **scan_result,
            }
        )

    for town in args.towns:
        output["towns_summary"][town] = {
            "num_routes": town_total_count.get(town, 0),
            "num_routes_with_candidates": town_usable_count.get(town, 0),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote scan results -> {args.output}")


if __name__ == "__main__":
    main()
