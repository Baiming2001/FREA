#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：route_scenario.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import time
import copy
import numpy as np
import carla

from frea.gym_carla.envs.utils import get_locations_nearby_spawn_points, calculate_abs_velocity, get_relative_info, get_relative_route_info, \
    get_relative_waypoint_info
from frea.scenario.scenario_manager.timer import GameTime
from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

from frea.scenario.tools.route_manipulation import interpolate_trajectory
from frea.scenario.tools.scenario_utils import (
    get_valid_spawn_points,
    convert_transform_to_location
)
from frea.scenario.scenario_definition.adv_behavior_scenario import AdvBehaviorSingle
from frea.scenario.scenario_definition.atomic_criteria import (
    Status,
    CollisionTest,
    DrivenDistanceTest,
    AverageVelocityTest,
    OffRoadTest,
    KeepLaneTest,
    InRouteTest,
    RouteCompletionTest,
    StuckDetectorTest,
    RunningRedLightTest,
    RunningStopTest,
)


class RouteScenario():
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, ego_id, max_running_step, env_params, mode, logger):
        self.world = world
        self.logger = logger
        self.config = config
        self.ego_id = ego_id
        self.max_running_step = max_running_step
        self.mode = mode
        self.timeout = 60
        self.ego_max_driven_distance = 200
        self.fixed_delta_seconds = env_params.get('fixed_delta_seconds', 0.1)
        self.traffic_intensity = env_params['traffic_intensity']
        self.search_radius = env_params['search_radius']
        self.goal_point_radius = env_params['goal_point_radius']
        self.post_collision_hold_steps = max(1, int(0.5 / self.fixed_delta_seconds))
        self.post_collision_steps = 0

        # create the route and ego's position (the start point of the route)
        self.route, self.ego_vehicle, self.gps_route = self._update_route_and_ego(timeout=self.timeout)
        self.global_route_waypoints, self.global_route_lane_road_ids = self._global_route_to_waypoints()
        self.next_intersection_loc = CarlaDataProvider.get_next_intersection_location(self.route[0][0].location)
        self.unactivated_actors = []
        self.CBVs_nearby_vehicles = {}
        self.special_actors = {}
        self.criteria = self._create_criteria()
        self.scenario_instance = AdvBehaviorSingle(self.world, self.ego_vehicle, env_params)  # create the scenario instance

    def _get_custom_scenario_type_id(self):
        parameters = getattr(self.config, 'parameters', None) or {}
        scenario_type_id = parameters.get('scenario_type_id')
        if scenario_type_id is None:
            return self.config.scenario_id
        try:
            return int(scenario_type_id)
        except (TypeError, ValueError):
            return self.config.scenario_id

    def _is_custom_scenario_type(self, scenario_type_id):
        return self._get_custom_scenario_type_id() == scenario_type_id

    def _get_scenario_parameters(self):
        parameters = copy.deepcopy(self.config.parameters) if self.config.parameters is not None else {}
        if self._is_custom_scenario_type(2):
            scenario_description = getattr(self.config, 'scenario_description', None) or {}
            trigger_position = scenario_description.get('trigger_position') or {}
            scenario2_defaults = {
                'target_outcome': 'normal',
                'scenario_type_id': 2,
                'leading_vehicle_model': 'vehicle.tesla.model3',
                'other_vehicle_model': 'vehicle.audi.tt',
                'other_actor_source': 'left',
                'leading_spawn_mode': 'template',
                'leading_release_distance_m': 18.0,
                'leading_target_speed_mps': 7.0,
                'leading_post_merge_speed_mps': 8.0,
                'leading_lookahead_distance_m': 10.0,
                'leading_min_travel_distance_m': 14.0,
                'other_distance_back_m': 10.0,
                'other_target_speed_mps': 8.0,
                'other_speed_variation_mps': 0.1,
                'other_min_follow_distance_m': 8.0,
                'other_follow_speed_offset_mps': 0.5,
                'other_far_follow_extra_mps': 2.5,
                'other_far_follow_distance_m': 14.0,
                'other_close_speed_penalty_mps': 0.1,
                'other_start_boost_mps': 6.0,
                'other_start_boost_speed_threshold_mps': 7.0,
                'other_lane_side': 'right',
                'ego_clear_distance_m': 22.0,
                'scene_end_after_stop_seconds': 0.5,
                'ego_reaction_delay_seconds': 0.0,
                'ego_min_throttle_during_delay': 0.0,
                'trigger_position_x': trigger_position.get('x'),
                'trigger_position_y': trigger_position.get('y'),
                'trigger_position_z': trigger_position.get('z'),
            }
            outcome_profiles = {
                'collision': {
                    'leading_release_distance_m': 5.5,
                    'leading_target_speed_mps': 7.5,
                    'leading_post_merge_speed_mps': 4.5,
                    'leading_lookahead_distance_m': 6.0,
                    'leading_min_travel_distance_m': 8.0,
                    'other_target_speed_mps': 9.0,
                    'other_speed_variation_mps': 0.03,
                    'other_follow_speed_offset_mps': 0.8,
                    'other_far_follow_extra_mps': 3.0,
                    'other_far_follow_distance_m': 16.0,
                    'other_close_speed_penalty_mps': 0.05,
                    'other_start_boost_mps': 7.0,
                    'other_start_boost_speed_threshold_mps': 8.0,
                    'ego_clear_distance_m': 18.0,
                    'ego_reaction_delay_seconds': 3.0,
                    'ego_min_throttle_during_delay': 0.4,
                    'ego_force_no_brake_distance_m': 18.0,
                    'ego_max_brake_during_collision_window': 0.02,
                },
                'normal': {
                    'leading_release_distance_m': 24.0,
                    'leading_target_speed_mps': 8.0,
                    'leading_post_merge_speed_mps': 9.0,
                    'leading_min_travel_distance_m': 16.0,
                    'other_target_speed_mps': 7.5,
                    'other_speed_variation_mps': 0.1,
                    'other_follow_speed_offset_mps': 0.0,
                    'other_far_follow_extra_mps': 1.0,
                    'other_far_follow_distance_m': 14.0,
                    'other_close_speed_penalty_mps': 0.5,
                    'other_start_boost_mps': 4.0,
                    'other_start_boost_speed_threshold_mps': 5.0,
                    'ego_clear_distance_m': 22.0,
                },
            }
            target_outcome = str(parameters.get('target_outcome', scenario2_defaults['target_outcome'])).lower()
            scenario2_defaults.update(outcome_profiles.get(target_outcome, {}))
            scenario2_defaults.update(parameters)
            scenario2_defaults['target_outcome'] = target_outcome
            self.config.parameters = copy.deepcopy(scenario2_defaults)
            return scenario2_defaults
        if self._is_custom_scenario_type(3):
            scenario3_defaults = {
                'target_outcome': 'near_miss',
                'leading_distance_m': 12.0,
                'other_distance_back_m': 8.0,
                'other_lane_side': 'left',
                'leading_target_speed_mps': 10.0,
                'other_target_speed_mps': 10.0,
                'leading_brake_after_seconds': 3.0,
                'leading_brake_duration_seconds': 2.0,
                'leading_post_brake_speed_mps': 0.0,
                'other_speed_variation_mps': 0.3,
                'scene_end_after_stop_seconds': 1.0,
                'ego_reaction_delay_seconds': 0.0,
                'ego_min_throttle_during_delay': 0.0,
                'leading_lookahead_distance_m': 14.0,
                'other_lookahead_distance_m': 10.0,
            }
            outcome_profiles = {
                'collision': {
                    'leading_distance_m': 6.0,
                    'leading_target_speed_mps': 12.0,
                    'other_target_speed_mps': 11.0,
                    'leading_brake_after_seconds': 6.0,
                    'leading_brake_duration_seconds': 1.0,
                    'leading_post_brake_speed_mps': 0.0,
                    'other_speed_variation_mps': 0.15,
                    'ego_reaction_delay_seconds': 1.2,
                    'ego_min_throttle_during_delay': 0.35,
                    'leading_lookahead_distance_m': 14.0,
                    'other_lookahead_distance_m': 10.0,
                },
                'near_miss': {
                    'leading_distance_m': 14.0,
                    'leading_target_speed_mps': 10.0,
                    'other_target_speed_mps': 10.0,
                    'leading_brake_after_seconds': 3.0,
                    'leading_brake_duration_seconds': 2.0,
                    'leading_post_brake_speed_mps': 1.5,
                    'other_speed_variation_mps': 0.3,
                    'ego_reaction_delay_seconds': 0.0,
                    'ego_min_throttle_during_delay': 0.0,
                    'leading_lookahead_distance_m': 14.0,
                    'other_lookahead_distance_m': 10.0,
                },
                'normal': {
                    'leading_distance_m': 14.0,
                    'leading_target_speed_mps': 10.0,
                    'other_target_speed_mps': 10.0,
                    'leading_brake_after_seconds': 3.0,
                    'leading_brake_duration_seconds': 2.0,
                    'leading_post_brake_speed_mps': 1.5,
                    'other_speed_variation_mps': 0.3,
                    'ego_reaction_delay_seconds': 0.0,
                    'ego_min_throttle_during_delay': 0.0,
                    'leading_lookahead_distance_m': 14.0,
                    'other_lookahead_distance_m': 10.0,
                },
            }
            target_outcome = str(parameters.get('target_outcome', scenario3_defaults['target_outcome'])).lower()
            scenario3_defaults.update(outcome_profiles.get(target_outcome, {}))
            scenario3_defaults.update(parameters)
            scenario3_defaults['target_outcome'] = target_outcome
            self.config.parameters = copy.deepcopy(scenario3_defaults)
            return scenario3_defaults
        return parameters

    def _build_transform_from_parameter(self, key_name):
        transform_dict = self.config.parameters.get(key_name) if self.config.parameters is not None else None
        if transform_dict is None:
            return None
        return carla.Transform(
            location=carla.Location(
                x=float(transform_dict['x']),
                y=float(transform_dict['y']),
                z=float(transform_dict.get('z', 0.0))
            ),
            rotation=carla.Rotation(
                pitch=float(transform_dict.get('pitch', 0.0)),
                yaw=float(transform_dict.get('yaw', 0.0)),
                roll=float(transform_dict.get('roll', 0.0))
            )
        )

    def _shift_waypoint(self, waypoint, distance, forward=True):
        if waypoint is None:
            return None
        candidates = waypoint.next(distance) if forward else waypoint.previous(distance)
        return candidates[0] if candidates else waypoint

    def _is_same_direction_lane(self, reference_waypoint, candidate_waypoint, min_dot=0.5):
        if reference_waypoint is None or candidate_waypoint is None:
            return False

        reference_forward = reference_waypoint.transform.get_forward_vector()
        candidate_forward = candidate_waypoint.transform.get_forward_vector()
        direction_dot = (
            reference_forward.x * candidate_forward.x
            + reference_forward.y * candidate_forward.y
            + reference_forward.z * candidate_forward.z
        )
        return direction_dot >= min_dot

    def _slice_dense_route_for_route_start(self, gps_route, route, route_start_ratio, min_remaining_points=20):
        if route is None or len(route) <= min_remaining_points:
            return gps_route, route

        clamped_ratio = max(0.0, min(0.7, float(route_start_ratio)))
        max_start_index = max(0, int((len(route) - min_remaining_points) * clamped_ratio))
        if max_start_index <= 0:
            return gps_route, route

        sliced_route = route[max_start_index:]
        sliced_gps_route = gps_route[max_start_index:] if gps_route is not None else gps_route
        if len(sliced_route) < min_remaining_points:
            return gps_route, route
        return sliced_gps_route, sliced_route

    def _get_adjacent_driving_lane(self, waypoint, lane_side):
        candidate = waypoint.get_left_lane() if lane_side == 'left' else waypoint.get_right_lane()
        if candidate is None or candidate.lane_type != carla.LaneType.Driving or not self._is_same_direction_lane(waypoint, candidate):
            fallback = waypoint.get_right_lane() if lane_side == 'left' else waypoint.get_left_lane()
            if (
                fallback is not None
                and fallback.lane_type == carla.LaneType.Driving
                and self._is_same_direction_lane(waypoint, fallback)
            ):
                return fallback
            return None
        return candidate

    def _find_nearby_adjacent_lane(self, waypoint, lane_side, search_distances=None):
        if waypoint is None:
            return None, None

        if search_distances is None:
            search_distances = [0.0, 5.0, 10.0, 15.0, 20.0]

        for distance in search_distances:
            candidate_waypoints = []
            if distance == 0.0:
                candidate_waypoints.append(waypoint)
            else:
                candidate_waypoints.append(self._shift_waypoint(waypoint, distance, forward=True))
                candidate_waypoints.append(self._shift_waypoint(waypoint, distance, forward=False))

            for candidate_waypoint in candidate_waypoints:
                adjacent_lane = self._get_adjacent_driving_lane(candidate_waypoint, lane_side)
                if adjacent_lane is not None:
                    return adjacent_lane, candidate_waypoint

        return None, None

    def _find_rear_adjacent_lane(self, waypoint, lane_side, search_distances=None):
        if waypoint is None:
            return None, None

        if search_distances is None:
            search_distances = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]

        for distance in search_distances:
            candidate_waypoint = waypoint if distance == 0.0 else self._shift_waypoint(waypoint, distance, forward=False)
            adjacent_lane = self._get_adjacent_driving_lane(candidate_waypoint, lane_side)
            if adjacent_lane is not None:
                return adjacent_lane, candidate_waypoint

        return None, None

    def _is_waypoint_behind_reference(self, reference_waypoint, candidate_waypoint, min_back_distance=1.0):
        if reference_waypoint is None or candidate_waypoint is None:
            return False

        reference_location = reference_waypoint.transform.location
        candidate_location = candidate_waypoint.transform.location
        forward_vector = reference_waypoint.transform.get_forward_vector()
        relative_x = candidate_location.x - reference_location.x
        relative_y = candidate_location.y - reference_location.y
        longitudinal_projection = relative_x * forward_vector.x + relative_y * forward_vector.y
        return longitudinal_projection <= -abs(min_back_distance)

    def _project_to_rear_adjacent_waypoint(self, reference_waypoint, lane_side, back_distance):
        if reference_waypoint is None:
            return None

        reference_transform = reference_waypoint.transform
        reference_location = reference_transform.location
        forward_vector = reference_transform.get_forward_vector()
        right_vector = reference_transform.get_right_vector()

        lane_width = getattr(reference_waypoint, 'lane_width', 3.5) or 3.5
        lateral_sign = -1.0 if lane_side == 'left' else 1.0
        lateral_offset = lane_width * lateral_sign

        target_location = carla.Location(
            x=reference_location.x - forward_vector.x * back_distance + right_vector.x * lateral_offset,
            y=reference_location.y - forward_vector.y * back_distance + right_vector.y * lateral_offset,
            z=reference_location.z
        )
        return self.world.get_map().get_waypoint(
            target_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

    def _get_same_lane_rear_waypoint(self, reference_waypoint, back_distance):
        if reference_waypoint is None:
            return None
        return self._shift_waypoint(reference_waypoint, back_distance, forward=False)

    def _resolve_scenario3_other_waypoint(self, ego_start_waypoint, scenario_params):
        adjacent_lane, _ = self._find_rear_adjacent_lane(
            ego_start_waypoint,
            scenario_params['other_lane_side']
        )
        if adjacent_lane is not None:
            other_waypoint = self._shift_waypoint(adjacent_lane, scenario_params['other_distance_back_m'], forward=False)
            if self._is_waypoint_behind_reference(ego_start_waypoint, other_waypoint):
                return other_waypoint

            fallback_distances = [scenario_params['other_distance_back_m'] + extra for extra in (5.0, 10.0, 15.0, 20.0)]
            for fallback_distance in fallback_distances:
                fallback_waypoint = self._shift_waypoint(adjacent_lane, fallback_distance, forward=False)
                if self._is_waypoint_behind_reference(ego_start_waypoint, fallback_waypoint):
                    return fallback_waypoint

            projected_waypoint = self._project_to_rear_adjacent_waypoint(
                ego_start_waypoint,
                scenario_params['other_lane_side'],
                scenario_params['other_distance_back_m']
            )
            if projected_waypoint is not None and self._is_waypoint_behind_reference(ego_start_waypoint, projected_waypoint):
                return projected_waypoint

        same_lane_waypoint = self._get_same_lane_rear_waypoint(ego_start_waypoint, scenario_params['other_distance_back_m'])
        if same_lane_waypoint is not None and self._is_waypoint_behind_reference(ego_start_waypoint, same_lane_waypoint, min_back_distance=0.5):
            return same_lane_waypoint

        extended_same_lane = self._get_same_lane_rear_waypoint(ego_start_waypoint, scenario_params['other_distance_back_m'] + 5.0)
        if extended_same_lane is not None and self._is_waypoint_behind_reference(ego_start_waypoint, extended_same_lane, min_back_distance=0.5):
            return extended_same_lane

        raise RuntimeError('Failed to place Scenario 3 other vehicle behind ego vehicle')

    def _spawn_special_actor(self, role_name, transform, vehicle_model):
        actor = CarlaDataProvider.request_new_actor(
            vehicle_model,
            transform,
            rolename=role_name,
            autopilot=False
        )
        actor.set_autopilot(False, CarlaDataProvider.get_traffic_manager_port())
        self.special_actors[role_name] = actor
        CarlaDataProvider.set_special_actor(self.ego_vehicle, role_name, actor)
        return actor

    def _choose_scenario2_actor_transform(self, scenario_params):
        scenario_description = getattr(self.config, 'scenario_description', None) or {}
        other_actors = scenario_description.get('other_actors') or {}
        source_preference = []
        preferred_source = str(scenario_params.get('other_actor_source', 'left')).lower()
        if preferred_source:
            source_preference.append(preferred_source)
        for fallback_source in ('left', 'right', 'front'):
            if fallback_source not in source_preference:
                source_preference.append(fallback_source)

        for source_name in source_preference:
            actor_configs = other_actors.get(source_name) or []
            if actor_configs:
                return source_name, actor_configs[0]
        return None, None

    def _initialize_scenario2_actors(self):
        scenario_params = self._get_scenario_parameters()
        roadside_transform = self._build_transform_from_parameter('leading_roadside_transform')
        driving_anchor_transform = self._build_transform_from_parameter('leading_driving_anchor_transform')
        if roadside_transform is not None and driving_anchor_transform is not None:
            anchor_waypoint = self.world.get_map().get_waypoint(
                driving_anchor_transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if anchor_waypoint is None:
                raise RuntimeError('Failed to resolve custom Scenario 2 leading driving anchor waypoint')

            ego_start_waypoint = self.world.get_map().get_waypoint(
                self.route[0][0].location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if ego_start_waypoint is None:
                raise RuntimeError('Failed to resolve custom Scenario 2 ego start waypoint')

            other_waypoint = self._resolve_scenario3_other_waypoint(ego_start_waypoint, scenario_params)
            self._spawn_special_actor('leading', roadside_transform, scenario_params['leading_vehicle_model'])
            self._spawn_special_actor('other', other_waypoint.transform, scenario_params['other_vehicle_model'])
            special_actor_routes = {
                'leading': self._build_special_actor_route(anchor_waypoint),
                'other': self._build_special_actor_route(other_waypoint),
            }
            self.scenario_instance.set_special_actors(self.special_actors, scenario_params, special_actor_routes)
            return

        source_name, actor_config = self._choose_scenario2_actor_transform(scenario_params)
        if actor_config is None:
            self.logger.log('>> Scenario 2 has no matched other_actor template, skip scripted crossing actor', color='yellow')
            return

        actor_transform = carla.Transform(
            location=carla.Location(
                x=float(actor_config['x']),
                y=float(actor_config['y']),
                z=float(actor_config['z'])
            ),
            rotation=carla.Rotation(yaw=float(actor_config['yaw']))
        )
        self._spawn_special_actor('other', actor_transform, scenario_params['other_vehicle_model'])
        scenario_params['other_actor_source'] = source_name
        self.scenario_instance.set_special_actors(self.special_actors, scenario_params, {})

    def _build_special_actor_route(self, actor_waypoint, min_remaining_points=20):
        if actor_waypoint is None or not self.route:
            return []

        actor_location = actor_waypoint.transform.location
        actor_forward = actor_waypoint.transform.get_forward_vector()
        best_index = None
        best_distance = float('inf')

        for index, (route_transform, _) in enumerate(self.route):
            route_location = route_transform.location
            relative_x = route_location.x - actor_location.x
            relative_y = route_location.y - actor_location.y
            longitudinal_projection = relative_x * actor_forward.x + relative_y * actor_forward.y

            # Keep the actor on its forward path instead of snapping to points behind it.
            if longitudinal_projection < -2.0:
                continue

            distance = route_location.distance(actor_location)
            if distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index is None:
            return []

        remaining_route = [route_transform for route_transform, _ in self.route[best_index:]]
        if len(remaining_route) < min_remaining_points:
            return []
        return remaining_route

    def _initialize_scenario3_actors(self):
        scenario_params = self._get_scenario_parameters()
        carla_map = self.world.get_map()
        ego_start_waypoint = carla_map.get_waypoint(
            self.route[0][0].location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if ego_start_waypoint is None:
            raise RuntimeError('Failed to get ego start waypoint for Scenario 3')

        leading_waypoint = self._shift_waypoint(ego_start_waypoint, scenario_params['leading_distance_m'], forward=True)
        other_waypoint = self._resolve_scenario3_other_waypoint(ego_start_waypoint, scenario_params)

        self._spawn_special_actor('leading', leading_waypoint.transform, 'vehicle.tesla.model3')
        self._spawn_special_actor('other', other_waypoint.transform, 'vehicle.audi.tt')

        special_actor_routes = {
            'leading': self._build_special_actor_route(leading_waypoint),
        }
        self.scenario_instance.set_special_actors(self.special_actors, scenario_params, special_actor_routes)

    def _global_route_to_waypoints(self):
        waypoints_list = []
        waypoint_lane_road_ids = set()
        carla_map = self.world.get_map()
        for node in self.route:
            loc = node[0].location
            waypoint = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            waypoints_list.append(waypoint)
            waypoint_lane_road_ids.add((waypoint.lane_id, waypoint.road_id))
        return waypoints_list, waypoint_lane_road_ids

    def _update_route_and_ego(self, timeout=None):
        ego_vehicle = None
        route = None
        gps_route = None
        scenario_id = self.config.scenario_id
        if scenario_id == 0:
            vehicle_spawn_points = get_valid_spawn_points(self.world)
            for random_transform in vehicle_spawn_points:
                gps_route, route = interpolate_trajectory(self.world, [random_transform])
                ego_vehicle = self._spawn_ego_vehicle(route[0][0], self.config.auto_ego)
                if ego_vehicle is not None:
                    break
        else:
            route_start_ratio = 0.0
            if self.config.parameters is not None:
                route_start_ratio = self.config.parameters.get('route_start_ratio', 0.0)
            gps_route, route = interpolate_trajectory(self.world, self.config.trajectory)
            gps_route, route = self._slice_dense_route_for_route_start(gps_route, route, route_start_ratio)
            ego_vehicle = self._spawn_ego_vehicle(route[0][0], self.config.auto_ego)

        CarlaDataProvider.set_ego_vehicle_route(ego_vehicle, convert_transform_to_location(route))
        CarlaDataProvider.set_scenario_config(self.config)

        # Timeout of a scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle, gps_route

    def _estimate_route_timeout(self, route):
        route_length = 0.0  # in meters
        min_length = 100.0
        SECONDS_GIVEN_PER_METERS = 1

        if len(route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _spawn_ego_vehicle(self, elevate_transform, autopilot=False):
        role_name = 'ego_vehicle' + str(self.ego_id)

        success = False
        ego_vehicle = None
        while not success:
            try:
                ego_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.lincoln.mkz_2017',
                    elevate_transform,
                    rolename=role_name, 
                    autopilot=autopilot
                )
                ego_vehicle.set_autopilot(autopilot, CarlaDataProvider.get_traffic_manager_port())
                success = True
            except RuntimeError:
                print("WARNING: Failed to spawn the ego vehicle, try to modify the z position of the spawn point")
                elevate_transform.location.z += 0.1
        return ego_vehicle

    def get_location_nearby_spawn_points(self):
        start_location = self.route[0][0].location
        end_location = self.route[-1][0].location
        locations_list = [start_location, self.next_intersection_loc, end_location]
        radius_list = [20, 50, 30]
        closest_dis = 5

        spawn_points = get_locations_nearby_spawn_points(
            locations_list, radius_list, closest_dis, self.global_route_lane_road_ids, self.traffic_intensity
        )
        amount = len(spawn_points)
        return amount, spawn_points

    def initialize_actors(self):
        if self._is_custom_scenario_type(2):
            self._initialize_scenario2_actors()
        if self._is_custom_scenario_type(3):
            self._initialize_scenario3_actors()

        amount, spawn_points = self.get_location_nearby_spawn_points()
        # don't activate all the actors when initialization
        new_actors = CarlaDataProvider.request_new_batch_actors(
            model='vehicle.*',
            amount=amount,
            spawn_points=spawn_points,
            autopilot=False,
            random_location=False,
            rolename='background'
        )
        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")
        self.logger.log(f'>> successfully spawning {len(new_actors)} Autopilot vehicles', color='green')
        self.unactivated_actors.extend(new_actors)
        CarlaDataProvider.set_scenario_actors(self.ego_vehicle, new_actors)

    def activate_background_actors(self, activate_threshold=50):
        ego_location = CarlaDataProvider.get_location(self.ego_vehicle)
        unactivated_actors = list(self.unactivated_actors)  # for safer remove
        for actor in unactivated_actors:
            if CarlaDataProvider.get_location(actor).distance(ego_location) < activate_threshold:
                actor.set_autopilot(True)
                self.unactivated_actors.remove(actor)

    def get_running_status(self, running_record):
        running_status = {
            'current_game_time': GameTime.get_time(),
            'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.yaw / 180 * np.pi
        }

        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        ego_stop = False
        ego_truncated = False
        ego_collision = False
        collision_actor_id = None
        # collision with other objects
        if running_status['collision'][0] == Status.FAILURE:
            ego_collision = True
            collision_actor_id = running_status['collision'][1]
            special_actor_ids = {actor.id for actor in self.special_actors.values() if actor is not None}
            if (
                collision_actor_id is not None
                and collision_actor_id not in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle)
                and collision_actor_id not in special_actor_ids
            ):
                ego_stop = True
                self.logger.log(f'>> Scenario stops due to collision with normal object', color='yellow')

        # out of the road detection
        if running_status['off_road'] == Status.FAILURE:
            ego_stop = True
            self.logger.log('>> Scenario stops due to off road', color='yellow')

        # route completed
        if running_status['route_complete'] == 100:
            ego_stop = True
            self.logger.log('>> Scenario stops due to route completion', color='yellow')

        # stuck
        if self.mode == 'train_scenario' and running_status['stuck'] == Status.FAILURE:
            ego_stop = True
            ego_truncated = True
            self.logger.log('>> Scenario stops due to stuck', color='yellow')

        # stop at a max step
        if len(running_record) >= self.max_running_step: 
            ego_stop = True
            ego_truncated = True
            self.logger.log('>> Scenario stops due to max steps', color='yellow')

        custom_scenario_type = self._get_custom_scenario_type_id()
        if custom_scenario_type in (2, 3) and ego_collision:
            self.post_collision_steps += 1
            if self.post_collision_steps >= self.post_collision_hold_steps:
                ego_stop = True
                self.logger.log('>> Scenario stops 0.5s after collision', color='yellow')
        elif custom_scenario_type in (2, 3):
            self.post_collision_steps = 0

        if custom_scenario_type in (2, 3) and self.scenario_instance.should_terminate_episode():
            ego_stop = True
            self.logger.log('>> Scenario stops because scripted actors completed the intended interaction', color='yellow')

        if running_status['current_game_time'] >= self.timeout:
            ego_stop = True
            ego_truncated = True
            self.logger.log('>> Scenario stops due to timeout', color='yellow')

        return running_status, ego_stop, ego_collision, ego_truncated

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        # the criteria needed both in training and evaluating
        criteria['route_complete'] = RouteCompletionTest(self.ego_vehicle, route=route)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicle, optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicle, terminate_on_failure=False)  # don't terminate on failure
        if self.mode == 'eval':
            # extra criteria for evaluating
            criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicle, distance_success=1e4, distance_acceptable=1e4, optional=True)
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicle, route=route, offroad_max=30)
            criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicle, optional=True)  # need sensor
            criteria['stuck'] = StuckDetectorTest(actor=self.ego_vehicle, len_thresh=120, speed_thresh=0.05, terminate_on_failure=True)
        elif self.mode == 'train_scenario':
            criteria['stuck'] = StuckDetectorTest(actor=self.ego_vehicle, len_thresh=120, speed_thresh=0.05, terminate_on_failure=True)

        return criteria

    def update_info(self, vehicle_count=3, goal_waypoint=None):
        '''
            scenario agent state:
            first row is CBV's relative state [x, y, bbox_x, bbox_y, yaw, forward speed]
            second row is ego's relative state [x, y, bbox_x, bbox_y, yaw, forward speed]
            rest row are other bv's relative state [x, y, bbox_x, bbox_y, yaw, forward speed]
        '''
        CBVs_obs = {}
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle).items():
            actors_info = []
            # the basic information about the CBV (center vehicle)
            CBV_transform = CarlaDataProvider.get_transform(CBV)
            CBV_matrix = np.array(CBV_transform.get_matrix())
            CBV_yaw = CBV_transform.rotation.yaw / 180 * np.pi
            # the relative CBV info
            CBV_info = get_relative_info(actor=CBV, center_yaw=CBV_yaw, center_matrix=CBV_matrix)
            actors_info.append(CBV_info)
            # the relative ego info
            ego_info = get_relative_info(actor=self.ego_vehicle, center_yaw=CBV_yaw, center_matrix=CBV_matrix)
            actors_info.append(ego_info)

            for actor in self.CBVs_nearby_vehicles[CBV_id]:
                if actor.id == self.ego_vehicle.id:
                    continue  # except the ego actor
                elif len(actors_info) < vehicle_count:
                    actor_info = get_relative_info(actor=actor, center_yaw=CBV_yaw, center_matrix=CBV_matrix)
                    actors_info.append(actor_info)
                else:
                    # avoiding too many nearby vehicles
                    break
            while len(actors_info) < vehicle_count:  # if no enough nearby vehicles, padding with 0
                actors_info.append([0] * len(CBV_info))

            # goal information
            if goal_waypoint is not None:
                route_info = get_relative_waypoint_info(goal_waypoint, center_yaw=CBV_yaw, center_matrix=CBV_matrix, radius=self.goal_point_radius)
                actors_info.append(route_info)

            CBVs_obs[CBV_id] = np.array(actors_info, dtype=np.float32)
        return {
            'CBVs_obs': CBVs_obs  # the controlled bv on the first line, while the rest bvs are sorted in ascending order
        }

    def update_ego_info(self, ego_nearby_vehicles, desired_nearby_vehicle=3, waypoints=None):
        '''
            safety network input state:
            all the rows are other bv's relative state
        '''
        infos = []
        # the basic information about the ego (center vehicle)
        ego_transform = CarlaDataProvider.get_transform(self.ego_vehicle)
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_yaw = ego_transform.rotation.yaw / 180 * np.pi
        ego_extent = self.ego_vehicle.bounding_box.extent
        # the relative CBV info
        ego_info = get_relative_info(actor=self.ego_vehicle, center_yaw=ego_yaw, center_matrix=ego_matrix)
        infos.append(ego_info)
        for actor in ego_nearby_vehicles:
            if len(infos) < desired_nearby_vehicle:
                actor_info = get_relative_info(actor=actor, center_yaw=ego_yaw, center_matrix=ego_matrix)
                infos.append(actor_info)
            else:
                break
        while len(infos) < desired_nearby_vehicle:  # if no enough nearby vehicles, padding with 0
            infos.append([0] * len(ego_info))

        # route information
        if waypoints is not None:
            route_info = get_relative_route_info(waypoints, center_yaw=ego_yaw, center_matrix=ego_matrix, center_extent=ego_extent)
            infos.append(route_info)

        # get the info of the ego vehicle and the other actors
        infos = np.array(infos, dtype=np.float32)

        return {
            'ego_obs': infos
        }

    def clean_up(self):
        # stop criterion and destroy sensors
        for _, criterion in self.criteria.items():
            criterion.terminate()
        time.sleep(0.1)

        self.scenario_instance.clean_up()  # nothing need to clean

        # clean background vehicle (the vehicle will be destroyed in CarlaDataProvider)
        self.unactivated_actors = []


