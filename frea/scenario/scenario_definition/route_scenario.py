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
        self.traffic_intensity = env_params['traffic_intensity']
        self.search_radius = env_params['search_radius']
        self.goal_point_radius = env_params['goal_point_radius']

        # create the route and ego's position (the start point of the route)
        self.route, self.ego_vehicle, self.gps_route = self._update_route_and_ego(timeout=self.timeout)
        self.global_route_waypoints, self.global_route_lane_road_ids = self._global_route_to_waypoints()
        self.next_intersection_loc = CarlaDataProvider.get_next_intersection_location(self.route[0][0].location)
        self.unactivated_actors = []
        self.CBVs_nearby_vehicles = {}
        self.special_actors = {}
        self.criteria = self._create_criteria()
        self.scenario_instance = AdvBehaviorSingle(self.world, self.ego_vehicle, env_params)  # create the scenario instance

    def _get_scenario_parameters(self):
        parameters = copy.deepcopy(self.config.parameters) if self.config.parameters is not None else {}
        if self.config.scenario_id == 3:
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
                'scene_end_after_stop_seconds': 1.0
            }
            outcome_profiles = {
                'collision': {
                    'leading_distance_m': 6.0,
                    'leading_target_speed_mps': 12.0,
                    'other_target_speed_mps': 11.0,
                    'leading_brake_after_seconds': 1.5,
                    'leading_brake_duration_seconds': 1.0,
                    'leading_post_brake_speed_mps': 0.0,
                    'other_speed_variation_mps': 0.15,
                },
                'near_miss': {
                    'leading_distance_m': 14.0,
                    'leading_target_speed_mps': 10.0,
                    'other_target_speed_mps': 10.0,
                    'leading_brake_after_seconds': 3.0,
                    'leading_brake_duration_seconds': 2.0,
                    'leading_post_brake_speed_mps': 1.5,
                    'other_speed_variation_mps': 0.3,
                },
                'normal': {
                    'leading_distance_m': 14.0,
                    'leading_target_speed_mps': 10.0,
                    'other_target_speed_mps': 10.0,
                    'leading_brake_after_seconds': 3.0,
                    'leading_brake_duration_seconds': 2.0,
                    'leading_post_brake_speed_mps': 1.5,
                    'other_speed_variation_mps': 0.3,
                },
            }
            target_outcome = str(parameters.get('target_outcome', scenario3_defaults['target_outcome'])).lower()
            scenario3_defaults.update(outcome_profiles.get(target_outcome, {}))
            scenario3_defaults.update(parameters)
            scenario3_defaults['target_outcome'] = target_outcome
            return scenario3_defaults
        return parameters

    def _shift_waypoint(self, waypoint, distance, forward=True):
        if waypoint is None:
            return None
        candidates = waypoint.next(distance) if forward else waypoint.previous(distance)
        return candidates[0] if candidates else waypoint

    def _get_adjacent_driving_lane(self, waypoint, lane_side):
        candidate = waypoint.get_left_lane() if lane_side == 'left' else waypoint.get_right_lane()
        if candidate is None or candidate.lane_type != carla.LaneType.Driving:
            fallback = waypoint.get_right_lane() if lane_side == 'left' else waypoint.get_left_lane()
            if fallback is not None and fallback.lane_type == carla.LaneType.Driving:
                return fallback
            return None
        return candidate

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
        adjacent_lane = self._get_adjacent_driving_lane(ego_start_waypoint, scenario_params['other_lane_side'])
        if adjacent_lane is None:
            raise RuntimeError('Failed to find an adjacent driving lane for Scenario 3 other vehicle')
        other_waypoint = self._shift_waypoint(adjacent_lane, scenario_params['other_distance_back_m'], forward=False)

        self._spawn_special_actor('leading', leading_waypoint.transform, 'vehicle.tesla.model3')
        self._spawn_special_actor('other', other_waypoint.transform, 'vehicle.audi.tt')
        self.scenario_instance.set_special_actors(self.special_actors, scenario_params)

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
            gps_route, route = interpolate_trajectory(self.world, self.config.trajectory)
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
        if self.config.scenario_id == 3:
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
        # collision with other objects
        if running_status['collision'][0] == Status.FAILURE:
            ego_collision = True
            if running_status['collision'][1] is not None and running_status['collision'][1] not in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle):
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

        if self.config.scenario_id == 3 and self.scenario_instance.should_terminate_episode():
            ego_stop = True
            self.logger.log('>> Scenario stops because leading and ego vehicles have both settled after braking', color='yellow')

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


