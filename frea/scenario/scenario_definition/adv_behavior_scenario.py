#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：adv_behavior_scenario.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import carla

import numpy as np
from frea.gym_carla.envs.utils import calculate_abs_velocity
from frea.scenario.tools.scenario_operation import ScenarioOperation
from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from frea.scenario.scenario_definition.basic_scenario import BasicScenario


class AdvBehaviorSingle(BasicScenario):
    """
        This class holds everything required for a scenario, in which an other vehicle takes priority from the ego vehicle, 
        by running a red traffic light (while the ego vehicle has green).
    """

    def __init__(self, world, ego_vehicle, env_params, timeout=100):
        super(AdvBehaviorSingle, self).__init__("AdvBehaviorSingle", None, world)
        self.timeout = timeout
        self.ego_vehicle = ego_vehicle
        self._map = CarlaDataProvider.get_map()
        self.signalized_junction = env_params['signalized_junction']
        if self.signalized_junction:
            self.last_ego_waypoint = self._map.get_waypoint(self.ego_vehicle.get_location())
            self.traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
            if self.traffic_light is None:
                print(">> No traffic light for the given location of the ego vehicle found")
            else:
                self.traffic_light.set_state(carla.TrafficLightState.Green)
                self.traffic_light.set_green_time(self.timeout)
        else:
            # set all the traffic light to green
            CarlaDataProvider.set_all_traffic_light(traffic_light_state=carla.TrafficLightState.Green, timeout=self.timeout)

        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]
        self.fixed_delta_seconds = env_params.get('fixed_delta_seconds', 0.1)
        self.scenario_operation = ScenarioOperation(fixed_delta_seconds=self.fixed_delta_seconds)
        self.special_actors = {}
        self.special_actor_indices = {}
        self.special_actor_routes = {}
        self.special_actor_route_indices = {}
        self.scripted_parameters = {}
        self.script_step = 0
        self.stop_hold_steps = 0
        self.should_terminate = False
        self.script_state = {}

    def set_special_actors(self, special_actors, scripted_parameters=None, special_actor_routes=None):
        self.special_actors = special_actors or {}
        self.scripted_parameters = scripted_parameters or {}
        self.special_actor_routes = special_actor_routes or {}
        actor_list = [actor for actor in self.special_actors.values() if actor is not None]
        self.scenario_operation.other_actors = actor_list
        self.scenario_operation.vehicle_controller = {}
        self.scenario_operation._init_vehicle_controller()
        self.special_actor_indices = {}
        self.special_actor_route_indices = {}
        self.script_step = 0
        self.stop_hold_steps = 0
        self.should_terminate = False
        self.script_state = {}
        for index, (role_name, actor) in enumerate(self.special_actors.items()):
            if actor is not None:
                self.special_actor_indices[role_name] = index
                self.special_actor_route_indices[role_name] = 0

    def _build_location_from_parameters(self, prefix):
        x = self.scripted_parameters.get(f'{prefix}_x')
        y = self.scripted_parameters.get(f'{prefix}_y')
        z = self.scripted_parameters.get(f'{prefix}_z')
        if x is None or y is None:
            return None
        return carla.Location(x=float(x), y=float(y), z=float(z or 0.0))

    def _get_transform_parameter_location(self, key_name):
        transform_dict = self.scripted_parameters.get(key_name)
        if transform_dict is None:
            return None
        return carla.Location(
            x=float(transform_dict['x']),
            y=float(transform_dict['y']),
            z=float(transform_dict.get('z', 0.0))
        )

    def _follow_route_with_pid(self, role_name, target_speed, lookahead_steps=8, reach_threshold_m=4.0):
        actor = self.special_actors.get(role_name)
        actor_index = self.special_actor_indices.get(role_name)
        route = self.special_actor_routes.get(role_name) or []
        if actor is None or actor_index is None or not route:
            return False

        actor_location = CarlaDataProvider.get_location(actor)
        route_index = self.special_actor_route_indices.get(role_name, 0)

        while route_index < len(route) - 1:
            distance_to_current = actor_location.distance(route[route_index].location)
            if distance_to_current > reach_threshold_m:
                break
            route_index += 1

        self.special_actor_route_indices[role_name] = route_index
        target_index = min(route_index + lookahead_steps, len(route) - 1)
        self.scenario_operation.drive_to_target_followlane(actor_index, route[target_index], target_speed)
        return True

    def _follow_lane_with_pid(self, role_name, target_speed, lookahead_distance=8.0):
        if self._follow_route_with_pid(role_name, target_speed):
            return

        actor = self.special_actors.get(role_name)
        actor_index = self.special_actor_indices.get(role_name)
        if actor is None or actor_index is None:
            return

        waypoint = self._map.get_waypoint(
            CarlaDataProvider.get_location(actor),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return

        next_waypoints = waypoint.next(lookahead_distance)
        if not next_waypoints:
            return

        self.scenario_operation.drive_to_target_followlane(actor_index, next_waypoints[0].transform, target_speed)

    def _get_speed_with_variation(self, base_speed, variation_amplitude):
        variation = variation_amplitude * np.sin(self.script_step * 0.25)
        return max(0.0, base_speed + variation)

    def _update_scripted_special_actors(self):
        if not self.special_actors:
            return

        if int(self.scripted_parameters.get('scenario_type_id', -1)) == 2:
            self._update_scenario2_special_actors()
            self.script_step += 1
            return

        leading_speed = self.scripted_parameters.get('leading_target_speed_mps', 7.0)
        other_base_speed = self.scripted_parameters.get('other_target_speed_mps', 7.0)
        brake_after_seconds = self.scripted_parameters.get('leading_brake_after_seconds', 3.0)
        brake_duration_seconds = self.scripted_parameters.get('leading_brake_duration_seconds', 2.0)
        post_brake_speed = self.scripted_parameters.get('leading_post_brake_speed_mps', 0.0)
        other_speed_variation = self.scripted_parameters.get('other_speed_variation_mps', 0.3)
        scene_end_after_stop_seconds = self.scripted_parameters.get('scene_end_after_stop_seconds', 1.0)

        brake_start_step = int(brake_after_seconds / self.fixed_delta_seconds)
        brake_end_step = brake_start_step + int(brake_duration_seconds / self.fixed_delta_seconds)
        hold_steps_needed = max(1, int(scene_end_after_stop_seconds / self.fixed_delta_seconds))

        leading_lookahead_distance = self.scripted_parameters.get('leading_lookahead_distance_m', 14.0)
        other_lookahead_distance = self.scripted_parameters.get('other_lookahead_distance_m', 10.0)

        if self.special_actors.get('leading') is not None:
            if self.script_step < brake_start_step:
                self._follow_lane_with_pid('leading', leading_speed, lookahead_distance=leading_lookahead_distance)
            elif self.script_step < brake_end_step:
                self.scenario_operation.brake(self.special_actors['leading'])
            elif post_brake_speed > 0.0:
                self._follow_lane_with_pid('leading', post_brake_speed, lookahead_distance=leading_lookahead_distance)
            else:
                self.scenario_operation.brake(self.special_actors['leading'])

        if self.special_actors.get('other') is not None:
            ego_speed = calculate_abs_velocity(CarlaDataProvider.get_velocity(self.ego_vehicle))
            # Keep the adjacent vehicle from lagging behind the ego during cruising.
            target_other_reference_speed = max(other_base_speed, ego_speed + 0.5)
            target_other_speed = self._get_speed_with_variation(
                target_other_reference_speed,
                other_speed_variation
            )
            self._follow_lane_with_pid('other', target_other_speed, lookahead_distance=other_lookahead_distance)

        if self.script_step >= brake_end_step:
            leading_actor = self.special_actors.get('leading')
            leading_speed_now = calculate_abs_velocity(CarlaDataProvider.get_velocity(leading_actor)) if leading_actor is not None else 0.0
            ego_speed_now = calculate_abs_velocity(CarlaDataProvider.get_velocity(self.ego_vehicle))
            if leading_speed_now < 0.3 and ego_speed_now < 0.5:
                self.stop_hold_steps += 1
            else:
                self.stop_hold_steps = 0
            self.should_terminate = self.stop_hold_steps >= hold_steps_needed
        else:
            self.should_terminate = False

        self.script_step += 1

    def _update_scenario2_special_actors(self):
        leading_actor = self.special_actors.get('leading')
        if leading_actor is not None:
            anchor_location = self._get_transform_parameter_location('leading_driving_anchor_transform')
            if anchor_location is None:
                self.should_terminate = False
                return

            ego_location = CarlaDataProvider.get_location(self.ego_vehicle)
            ego_speed = calculate_abs_velocity(CarlaDataProvider.get_velocity(self.ego_vehicle))
            leading_location = CarlaDataProvider.get_location(leading_actor)

            release_distance = float(self.scripted_parameters.get('leading_release_distance_m', 18.0))
            leading_speed = float(self.scripted_parameters.get('leading_target_speed_mps', 7.0))
            leading_post_merge_speed = float(self.scripted_parameters.get('leading_post_merge_speed_mps', leading_speed))
            leading_lookahead_distance = float(self.scripted_parameters.get('leading_lookahead_distance_m', 10.0))
            leading_min_travel_distance = float(self.scripted_parameters.get('leading_min_travel_distance_m', 14.0))
            ego_clear_distance = float(self.scripted_parameters.get('ego_clear_distance_m', 22.0))
            other_base_speed = float(self.scripted_parameters.get('other_target_speed_mps', 8.0))
            other_speed_variation = float(self.scripted_parameters.get('other_speed_variation_mps', 0.1))
            other_min_follow_distance = float(self.scripted_parameters.get('other_min_follow_distance_m', 8.0))
            other_match_speed_gain = float(self.scripted_parameters.get('other_match_speed_gain', 0.6))
            other_close_speed_penalty = float(self.scripted_parameters.get('other_close_speed_penalty_mps', 0.6))
            other_start_boost = float(self.scripted_parameters.get('other_start_boost_mps', 2.5))
            other_start_boost_speed_threshold = float(self.scripted_parameters.get('other_start_boost_speed_threshold_mps', 3.0))
            scene_end_after_stop_seconds = float(self.scripted_parameters.get('scene_end_after_stop_seconds', 0.5))
            hold_steps_needed = max(1, int(scene_end_after_stop_seconds / self.fixed_delta_seconds))

            anchor_distance = ego_location.distance(anchor_location)
            initial_leading_location = self.script_state.setdefault(
                'scenario2_leading_initial_location',
                carla.Location(x=leading_location.x, y=leading_location.y, z=leading_location.z)
            )
            initial_ego_location = self.script_state.setdefault(
                'scenario2_ego_initial_location',
                carla.Location(x=ego_location.x, y=ego_location.y, z=ego_location.z)
            )
            leading_travel_distance = leading_location.distance(initial_leading_location)
            ego_travel_distance = ego_location.distance(initial_ego_location)

            released = self.script_state.get('scenario2_leading_released', False)
            if not released and anchor_distance <= release_distance:
                released = True
                self.script_state['scenario2_leading_released'] = True
                self.script_state['scenario2_release_step'] = self.script_step

            if released:
                target_speed = leading_post_merge_speed if leading_travel_distance >= leading_min_travel_distance else leading_speed
                self._follow_lane_with_pid('leading', target_speed, lookahead_distance=leading_lookahead_distance)
            else:
                self.scenario_operation.brake(leading_actor)

            other_actor = self.special_actors.get('other')
            if other_actor is not None:
                other_location = CarlaDataProvider.get_location(other_actor)
                other_distance_to_ego = other_location.distance(ego_location)
                other_speed = calculate_abs_velocity(CarlaDataProvider.get_velocity(other_actor))
                speed_gap = ego_speed - other_base_speed
                matched_speed = other_base_speed + max(0.0, speed_gap) * other_match_speed_gain
                if other_distance_to_ego <= other_min_follow_distance:
                    target_other_speed = max(0.0, ego_speed - other_close_speed_penalty)
                else:
                    target_other_reference_speed = min(max(other_base_speed, matched_speed), ego_speed + 0.5)
                    if other_speed < other_start_boost_speed_threshold and ego_speed > other_speed:
                        target_other_reference_speed = max(
                            target_other_reference_speed,
                            min(ego_speed + 1.0, other_speed + other_start_boost)
                        )
                    target_other_speed = self._get_speed_with_variation(
                        target_other_reference_speed,
                        other_speed_variation
                    )
                self._follow_lane_with_pid('other', target_other_speed, lookahead_distance=10.0)

            if released and leading_travel_distance >= leading_min_travel_distance and ego_travel_distance >= ego_clear_distance:
                self.stop_hold_steps += 1
            else:
                self.stop_hold_steps = 0
            self.should_terminate = self.stop_hold_steps >= hold_steps_needed
            return

        other_actor = self.special_actors.get('other')
        if other_actor is None:
            self.should_terminate = False
            return

        trigger_location = self._build_location_from_parameters('trigger_position')
        if trigger_location is None:
            target_speed = self.scripted_parameters.get('other_target_speed_mps', 8.0)
            lookahead_distance = self.scripted_parameters.get('other_lookahead_distance_m', 12.0)
            self._follow_lane_with_pid('other', target_speed, lookahead_distance=lookahead_distance)
            self.should_terminate = False
            return

        release_distance = float(self.scripted_parameters.get('other_release_distance_m', 20.0))
        clear_distance = float(self.scripted_parameters.get('other_clear_distance_m', 22.0))
        ego_clear_distance = float(self.scripted_parameters.get('ego_clear_distance_m', 18.0))
        other_min_travel_distance = float(self.scripted_parameters.get('other_min_travel_distance_m', 18.0))
        base_speed = float(self.scripted_parameters.get('other_target_speed_mps', 8.0))
        speed_variation = float(self.scripted_parameters.get('other_speed_variation_mps', 0.15))
        lookahead_distance = float(self.scripted_parameters.get('other_lookahead_distance_m', 12.0))

        ego_location = CarlaDataProvider.get_location(self.ego_vehicle)
        other_location = CarlaDataProvider.get_location(other_actor)
        ego_distance_to_trigger = ego_location.distance(trigger_location)
        other_distance_to_trigger = other_location.distance(trigger_location)
        initial_other_location = self.script_state.setdefault('scenario2_other_initial_location', carla.Location(
            x=other_location.x,
            y=other_location.y,
            z=other_location.z
        ))
        initial_ego_location = self.script_state.setdefault('scenario2_ego_initial_location', carla.Location(
            x=ego_location.x,
            y=ego_location.y,
            z=ego_location.z
        ))
        other_travel_distance = other_location.distance(initial_other_location)
        ego_travel_distance = ego_location.distance(initial_ego_location)

        released = self.script_state.get('scenario2_other_released', False)
        if not released and ego_distance_to_trigger <= release_distance:
            released = True
            self.script_state['scenario2_other_released'] = True
            self.script_state['scenario2_trigger_min_distance_after_release'] = other_distance_to_trigger

        if released:
            min_distance_after_release = self.script_state.get('scenario2_trigger_min_distance_after_release', other_distance_to_trigger)
            min_distance_after_release = min(min_distance_after_release, other_distance_to_trigger)
            self.script_state['scenario2_trigger_min_distance_after_release'] = min_distance_after_release

            target_speed = self._get_speed_with_variation(base_speed, speed_variation)
            self._follow_lane_with_pid('other', target_speed, lookahead_distance=lookahead_distance)
            crossed_trigger = (
                min_distance_after_release <= max(4.0, lookahead_distance * 0.5)
                and other_distance_to_trigger >= clear_distance
            )
            ego_cleared_scene = (
                ego_distance_to_trigger >= ego_clear_distance
                or ego_travel_distance >= ego_clear_distance
            )
        else:
            self.scenario_operation.brake(other_actor)
            crossed_trigger = False
            ego_cleared_scene = False

        self.should_terminate = (
            released
            and crossed_trigger
            and other_travel_distance >= other_min_travel_distance
            and ego_cleared_scene
        )

    def should_terminate_episode(self):
        return getattr(self, 'should_terminate', False)

    def convert_actions(self, scenario_actions):
        acc = scenario_actions[0]  # continuous action: acc
        steer = scenario_actions[1]  # continuous action: steering

        # normalize and clip the action
        acc = acc * self.acc_max
        steer = steer * self.steering_max
        acc = max(min(self.acc_max, acc), -self.acc_max)
        steer = max(min(self.steering_max, steer), -self.steering_max)

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
            reverse = False
        else:
            # throttle = 0
            # brake = np.clip(-acc / 8, 0, 1)
            # reverse = False
            # enable driving back
            reverse = True
            throttle = -np.clip(acc / 3, -1, 0)
            brake = 0

        # apply ego control
        act = carla.VehicleControl(reverse=reverse, throttle=float(throttle), steer=float(steer), brake=float(brake))
        return act

    def update_traffic_light(self):
        ego_waypoint = self._map.get_waypoint(CarlaDataProvider.get_location(self.ego_vehicle))
        if not ego_waypoint.is_junction and self.last_ego_waypoint.is_junction:  # last tick the ego is in the junction, but the current step is out
            traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle)
            # if the ego's next traffic light is not None and has changed, then set the next traffic light to green
            if traffic_light is not None and traffic_light != self.traffic_light:
                self.traffic_light = traffic_light
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.set_green_time(self.timeout)
        self.last_ego_waypoint = ego_waypoint

    def update_behavior(self, scenario_actions):
        if scenario_actions is not None:
            # apply scenario action for each CBV
            for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle).items():
                scenario_action = scenario_actions[CBV_id]
                act = self.convert_actions(scenario_action)
                CBV.apply_control(act)  # apply the control of the CBV on the next tick

        self._update_scripted_special_actors()

        if self.signalized_junction:  # if the signal controls the junction, the traffic need to be updated
            self.update_traffic_light()

    def clean_up(self):
        pass
