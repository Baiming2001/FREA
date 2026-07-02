#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：carla_env.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import copy
import json
import os
import queue
import time
import threading
import weakref

import numpy as np
import pygame
from skimage.transform import resize
import gym
import carla
from PIL import Image

from frea.gym_carla.envs.route_planner import RoutePlanner
from frea.gym_carla.envs.misc import (
    display_to_rgb,
    rgb_to_display_surface,
    get_lane_dis,
    get_pos,
)
from frea.agent.agent_utils.explainability_utils import get_masked_viz_3rd_person
from frea.gym_carla.envs.utils import get_CBV_candidates, get_nearby_vehicles, find_closest_vehicle, \
    update_goal_CBV_dis, get_CBV_ego_reward, calculate_abs_velocity, \
    process_ego_action, get_ego_min_dis, check_interaction, check_CBV_BV_stuck, draw_trajectory, get_records, get_closest_CBV_flag, get_min_distance_across_bboxes
from frea.scenario.scenario_definition.route_scenario import RouteScenario
from frea.scenario.scenario_manager.scenario_manager import ScenarioManager
from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from frea.agent.agent_utils.visualization import draw_route


class CarlaEnv(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator. 
    """

    def __init__(self, env_params, birdeye_render=None, display=None, world=None,
                 use_feasibility=None, agent_state_encoder=None, logger=None):
        assert world is not None, "the world passed into CarlaEnv is None"

        self.config = None
        self.world = world
        self.display = display
        self.logger = logger
        self.birdeye_render = birdeye_render

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.time_step = 0
        self.env_id = None
        self.ego_vehicle = None
        self.ego_nearby_vehicles = []
        self.env_params = env_params
        self.auto_ego = env_params['auto_ego']
        self.enable_sem = env_params['enable_sem']
        self.save_camera_frames = env_params.get('save_camera_frames', False)
        self.camera_only = env_params.get('camera_only', False)
        self.camera_fps = env_params.get('camera_fps', 10)
        default_camera_width = 1600
        default_camera_height = 900
        default_camera_quality = 90
        self.camera_export_width = env_params.get('camera_export_width', default_camera_width)
        self.camera_export_height = env_params.get('camera_export_height', default_camera_height)
        self.camera_export_format = str(env_params.get('camera_export_format', 'jpg')).lower()
        self.camera_export_quality = env_params.get('camera_export_quality', default_camera_quality)
        self.camera_export_optimize = env_params.get('camera_export_optimize', False)
        self.save_actor_lidar_frames = env_params.get('save_actor_lidar_frames', not self.camera_only)
        self.camera_export_queue_size = int(env_params.get('camera_export_queue_size', 1024))
        self.fixed_delta_seconds = env_params.get('fixed_delta_seconds', 0.1)
        self.capture_stride = max(1, round((1 / self.fixed_delta_seconds) / self.camera_fps))
        self.output_dir = env_params.get('output_dir')
        self.ego_agent_learnable = env_params['ego_agent_learnable']
        self.scenario_agent_learnable = env_params['scenario_agent_learnable']
        self.mode = env_params['mode']
        self.eval_mode = env_params['eval_mode']

        self.lidar_sensor = None
        self.camera_sensor = None
        self.sem_sensor = None
        self.CBVs_collision_sensor = {}
        self.lidar_data = None
        self.lidar_height = 2.1

        self.CBVs = {}
        self.CBVs_nearby_vehicles = {}
        self.gps_route = None
        self.route = None
        self.CBVs_collision = {}
        self.ego_collide = False
        self.scene_had_ego_collision = False
        self.front_camera_frame_idx = 0
        self.actor_camera_sensors = {}
        self.actor_camera_imgs = {}
        self.camera_actor_ids = {}
        self.actor_lidar_sensors = {}
        self.actor_lidar_points = {}
        self.lidar_actor_ids = {}
        self.export_queue = None
        self.export_worker = None
        self.export_stop_token = object()
        self.export_worker_error = None
        self.metadata_actor_roles = ['ego', 'leading', 'other']
        self.camera_actor_roles = ['ego', 'other']
        self.draw_debug_overlays = env_params.get('draw_debug_overlays', False)
        self.search_radius = env_params['search_radius']
        self.agent_obs_type = env_params['agent_obs_type']
        self.agent_state_encoder = agent_state_encoder

        # for feasibility
        self.use_feasibility = use_feasibility

        # for CBV
        self.CBVs_select_method = env_params['CBV_selection']
        self.goal_point_radius = env_params['goal_point_radius']

        # scenario manager
        self.scenario_manager = ScenarioManager(env_params, self.logger)

        # for birdeye view and front view visualization
        self.ego_agent_learnable = env_params['ego_agent_learnable']
        self.viz_route = env_params['viz_route']
        self.display_size = env_params['display_size']
        self.obs_range = env_params['obs_range']
        self.d_behind = env_params['d_behind']
        self.disable_lidar = env_params['disable_lidar']

        # for env wrapper
        self.max_past_step = env_params['max_past_step']
        self.max_episode_step = env_params['max_episode_step']
        self.max_waypt = env_params['max_waypt']
        self.lidar_bin = env_params['lidar_bin']
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']
        self.acc_range = env_params['continuous_accel_range']
        self.steering_range = env_params['continuous_steer_range']

        # for scenario
        self.ROOT_DIR = env_params['ROOT_DIR']
        self.warm_up_steps = env_params['warm_up_steps']

        self.obs_size = int(self.obs_range / self.lidar_bin)
        self._ensure_export_worker()

    def _ensure_export_worker(self):
        if not self.save_camera_frames:
            return
        if self.export_queue is None:
            self.export_queue = queue.Queue(maxsize=self.camera_export_queue_size)
        if self.export_worker is not None and self.export_worker.is_alive():
            return

        self.export_worker_error = None
        self.export_worker = threading.Thread(
            target=self._export_worker_loop,
            name=f'carla-export-env-{id(self)}',
            daemon=True
        )
        self.export_worker.start()

    def _export_worker_loop(self):
        while True:
            job = self.export_queue.get()
            try:
                if job is self.export_stop_token:
                    return

                job_type = job['type']
                if job_type == 'image':
                    image = Image.fromarray(job['array'])
                    if job['frame_ext'] in ('jpg', 'jpeg'):
                        image.save(
                            job['path'],
                            format='JPEG',
                            quality=int(job['quality']),
                            optimize=bool(job['optimize'])
                        )
                    else:
                        image.save(job['path'])
                elif job_type == 'npy':
                    np.save(job['path'], job['array'])
                else:
                    raise ValueError(f"Unsupported export job type: {job_type}")
            except Exception as exc:
                self.export_worker_error = exc
                if self.logger is not None:
                    self.logger.log(f'>> Camera export worker error: {exc}', color='red')
            finally:
                self.export_queue.task_done()

    def _queue_export_job(self, job):
        self._ensure_export_worker()
        if self.export_worker_error is not None:
            raise RuntimeError(f'Camera export worker failed: {self.export_worker_error}')
        self.export_queue.put(job)

    def _flush_export_jobs(self):
        if self.export_queue is None:
            return
        self.export_queue.join()
        if self.export_worker_error is not None:
            raise RuntimeError(f'Camera export worker failed: {self.export_worker_error}')

    def _shutdown_export_worker(self):
        if self.export_queue is None:
            return
        self._flush_export_jobs()
        self.export_queue.put(self.export_stop_token)
        if self.export_worker is not None:
            self.export_worker.join(timeout=10.0)
        self.export_worker = None
        self.export_queue = None

    def _apply_collision_target_ego_override(self, throttle, steer, brake):
        if self.config is None or self.config.parameters is None:
            return throttle, steer, brake

        scenario_type_id = self.config.parameters.get('scenario_type_id', self.config.scenario_id)
        try:
            scenario_type_id = int(scenario_type_id)
        except (TypeError, ValueError):
            scenario_type_id = self.config.scenario_id
        if scenario_type_id not in (2, 3):
            return throttle, steer, brake

        parameters = self.config.parameters
        target_outcome = str(parameters.get('target_outcome', '')).lower()
        if target_outcome != 'collision':
            return throttle, steer, brake

        if scenario_type_id == 3:
            trigger_after_seconds = float(parameters.get('leading_brake_after_seconds', 0.0))
        else:
            trigger_after_seconds = 0.0
        ego_reaction_delay_seconds = float(parameters.get('ego_reaction_delay_seconds', 0.0))
        ego_min_throttle_during_delay = float(parameters.get('ego_min_throttle_during_delay', 0.0))

        if ego_reaction_delay_seconds <= 0.0:
            return throttle, steer, brake

        trigger_start_step = int(trigger_after_seconds / self.fixed_delta_seconds)
        reaction_resume_step = trigger_start_step + int(ego_reaction_delay_seconds / self.fixed_delta_seconds)
        if trigger_start_step <= self.time_step < reaction_resume_step:
            throttle = max(float(throttle), ego_min_throttle_during_delay)
            brake = 0.0
        return throttle, steer, brake

    def _create_sensors(self):
        if self.eval_mode == 'render':
            if not self.disable_lidar:
                # lidar sensor
                self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
                self.lidar_bp = CarlaDataProvider._blueprint_library.find('sensor.lidar.ray_cast')
                self.lidar_bp.set_attribute('channels', '16')
                self.lidar_bp.set_attribute('range', '1000')

            # camera sensor
            self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
            self.BGR_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
            # self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))  # for ego view
            # self.camera_trans = carla.Transform(carla.Location(x=-2., y=0., z=12.),carla.Rotation(pitch=-50.0))  # for third-person view
            self.camera_trans = carla.Transform(carla.Location(x=12., y=0., z=20.),carla.Rotation(pitch=-90.0))  # god view
            self.camera_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
            self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
            self.camera_bp.set_attribute('fov', '110')
            # Set the time in seconds between sensor captures
            self.camera_bp.set_attribute('sensor_tick', '0.02')

            # actor-view camera sensors for saving multi-view image sequences
            self.front_camera_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.rgb')
            self.front_camera_bp.set_attribute('image_size_x', str(self.camera_export_width))
            self.front_camera_bp.set_attribute('image_size_y', str(self.camera_export_height))
            self.front_camera_bp.set_attribute('fov', '110')
            self.front_camera_bp.set_attribute('sensor_tick', str(self.fixed_delta_seconds))
            self.camera_view_transforms = {
                'front': carla.Transform(carla.Location(x=0.8, z=1.7), carla.Rotation(yaw=0.0)),
                'front_left': carla.Transform(carla.Location(x=0.8, y=-0.35, z=1.7), carla.Rotation(yaw=-45.0)),
                'front_right': carla.Transform(carla.Location(x=0.8, y=0.35, z=1.7), carla.Rotation(yaw=45.0))
            }
            self.actor_camera_imgs = {
                role_name: {
                    view_name: np.zeros((self.camera_export_height, self.camera_export_width, 3), dtype=np.uint8)
                    for view_name in self.camera_view_transforms
                }
                for role_name in self.camera_actor_roles
            }
            self.actor_lidar_bp = CarlaDataProvider._blueprint_library.find('sensor.lidar.ray_cast')
            self.actor_lidar_bp.set_attribute('channels', '32')
            self.actor_lidar_bp.set_attribute('range', '70')
            self.actor_lidar_bp.set_attribute('upper_fov', '10')
            self.actor_lidar_bp.set_attribute('lower_fov', '-20')
            self.actor_lidar_bp.set_attribute('rotation_frequency', '10')
            self.actor_lidar_bp.set_attribute('points_per_second', '560000')
            self.actor_lidar_bp.set_attribute('sensor_tick', str(self.fixed_delta_seconds))
            self.actor_lidar_trans = carla.Transform(carla.Location(x=0.0, z=2.1))
            self.actor_lidar_points = {
                role_name: np.zeros((0, 4), dtype=np.float32)
                for role_name in self.camera_actor_roles
            }

            # sem camera sensor
            if self.enable_sem:
                self.sem_img = np.zeros((self.obs_size, self.obs_size, 2), dtype=np.uint8)
                # self.sem_trans = carla.Transform(carla.Location(x=-2., y=0, z=12.), carla.Rotation(pitch=-50.0))  # for third-person view
                self.sem_trans = carla.Transform(carla.Location(x=12., y=0, z=20.), carla.Rotation(pitch=-90.0))  # god view
                self.sem_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.semantic_segmentation')
                self.sem_bp.set_attribute('image_size_x', str(self.obs_size))
                self.sem_bp.set_attribute('image_size_y', str(self.obs_size))
                self.sem_bp.set_attribute('fov', '110')
                # Set the time in seconds between sensor captures
                self.sem_bp.set_attribute('sensor_tick', '0.02')

    def _create_scenario(self, config, env_id):
        self.logger.log(f">> Loading scenario data id: {config.data_id}")

        # create scenarios according to different types
        scenario = RouteScenario(
            world=self.world,
            config=config,
            ego_id=env_id,
            max_running_step=self.max_episode_step,
            env_params=self.env_params,
            mode=self.mode,
            logger=self.logger
        )

        # init scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.ego_length = self.ego_vehicle.bounding_box.extent.x * 2
        self.scenario_manager.load_scenario(scenario)  # The scenario manager only controls the RouteScenario
        self.route = self.scenario_manager.route_scenario.route  # the global route
        self.gps_route = self.scenario_manager.route_scenario.gps_route  # the global gps route
        self.global_route_waypoints = self.scenario_manager.route_scenario.global_route_waypoints

    @staticmethod
    def _weather_to_dict(weather):
        if weather is None:
            return None
        weather_fields = [
            'cloudiness',
            'precipitation',
            'precipitation_deposits',
            'wind_intensity',
            'sun_azimuth_angle',
            'sun_altitude_angle',
            'wetness',
            'fog_distance',
            'fog_density',
        ]
        return {field: float(getattr(weather, field, 0.0)) for field in weather_fields}

    @staticmethod
    def _infer_weather_label(weather_dict):
        if not weather_dict:
            return 'unknown'

        precipitation = weather_dict.get('precipitation', 0.0)
        wetness = weather_dict.get('wetness', 0.0)
        cloudiness = weather_dict.get('cloudiness', 0.0)

        if precipitation >= 20.0:
            return 'rainy'
        if wetness >= 20.0:
            return 'wet'
        if cloudiness >= 50.0:
            return 'cloudy'
        return 'clear'

    @staticmethod
    def _infer_time_of_day_label(weather_dict):
        if not weather_dict:
            return 'unknown'

        sun_altitude = weather_dict.get('sun_altitude_angle', 70.0)
        if sun_altitude <= 0.0:
            return 'night'
        if sun_altitude <= 25.0:
            return 'sunset'
        return 'noon'

    @staticmethod
    def _clone_weather(weather):
        cloned = carla.WeatherParameters()
        if weather is None:
            return cloned

        for field in [
            'cloudiness',
            'precipitation',
            'precipitation_deposits',
            'wind_intensity',
            'sun_azimuth_angle',
            'sun_altitude_angle',
            'wetness',
            'fog_distance',
            'fog_density',
        ]:
            if hasattr(weather, field):
                setattr(cloned, field, float(getattr(weather, field)))
        return cloned

    def _override_weather_from_labels(self, weather, parameters):
        if parameters is None:
            return weather

        weather_label = parameters.get('weather_label')
        time_of_day_label = parameters.get('time_of_day_label')
        if weather_label is None and time_of_day_label is None:
            return weather

        overridden = self._clone_weather(weather)
        weather_presets = {
            'clear': {
                'cloudiness': 10.0,
                'precipitation': 0.0,
                'precipitation_deposits': 0.0,
                'wetness': 0.0,
                'fog_density': 0.0,
                'fog_distance': 120.0,
                'wind_intensity': 5.0,
            },
            'rainy': {
                'cloudiness': 85.0,
                'precipitation': 75.0,
                'precipitation_deposits': 60.0,
                'wetness': 70.0,
                'fog_density': 20.0,
                'fog_distance': 45.0,
                'wind_intensity': 35.0,
            },
            'cloudy': {
                'cloudiness': 85.0,
                'precipitation': 0.0,
                'precipitation_deposits': 0.0,
                'wetness': 10.0,
                'fog_density': 5.0,
                'fog_distance': 90.0,
                'wind_intensity': 10.0,
            },
            'wet': {
                'cloudiness': 45.0,
                'precipitation': 0.0,
                'precipitation_deposits': 0.0,
                'wetness': 80.0,
                'fog_density': 0.0,
                'fog_distance': 100.0,
                'wind_intensity': 8.0,
            },
        }
        time_presets = {
            'noon': {'sun_altitude_angle': 70.0},
            'sunset': {'sun_altitude_angle': 15.0},
            'night': {'sun_altitude_angle': -20.0},
        }

        if weather_label is not None:
            for field, value in weather_presets.get(str(weather_label).lower(), {}).items():
                setattr(overridden, field, value)
        if time_of_day_label is not None:
            for field, value in time_presets.get(str(time_of_day_label).lower(), {}).items():
                setattr(overridden, field, value)
        return overridden

    def _get_scene_naming_fields(self):
        parameters = self.config.parameters or {}
        scenario_type_id = int(parameters.get('scenario_type_id', self.config.scenario_id))
        scenario_subtype_id = int(parameters.get('scenario_subtype_id', 1))
        scenario_number = int(parameters.get('scenario_number', self.config.data_id + 1))
        map_name = self.world.get_map().name.split('/')[-1]
        scene_name = (
            f"{map_name}_type{scenario_type_id:03d}_"
            f"subtype{scenario_subtype_id:04d}_scenario{scenario_number:05d}"
        )
        return {
            'scene_name': scene_name,
            'scenario_type_id': scenario_type_id,
            'scenario_subtype_id': scenario_subtype_id,
            'scenario_number': scenario_number,
        }

    def _run_scenario(self):
        self.scenario_manager.run_scenario()  # init the background vehicle

    def register_CBV_sensor(self, CBV):
        blueprint = CarlaDataProvider._blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=CBV)

        # use weak reference to avoid memory leak
        self_weakref = weakref.ref(self)

        def count_collisions(event):
            self_strongref = self_weakref()
            if self_strongref is not None:
                self_strongref.CBVs_collision[event.actor.id] = {
                    'other_actor_id': event.other_actor.id,
                    'normal_impulse': [event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z]
                }

        collision_sensor.listen(lambda event: count_collisions(event))

        self.CBVs_collision_sensor[CBV.id] = collision_sensor
        self.CBVs_collision[CBV.id] = None

    def CBVs_selection(self):
        # when training the ego agent, don't need to calculate the CBV
        if self.scenario_agent_learnable and len(self.CBVs) < 2 and self.time_step % 2 == 0:
            # select the candidates of CBVs
            CBV_candidates = get_CBV_candidates(
                self.ego_vehicle, self.goal_waypoint,
                self.search_radius, self.ego_length
            )
            if CBV_candidates:
                # selecting the CBV
                # 1.Rule-based
                if self.CBVs_select_method == 'rule-based':
                    CBV = find_closest_vehicle(self.ego_vehicle, self.search_radius, CBV_candidates)
                # 2.attention-based
                elif self.CBVs_select_method == 'attention-based':
                    most_relevant_vehicle = self.agent_state_encoder.get_encoded_state(
                        self.ego_vehicle, CBV_candidates, self.waypoints, self.red_light_state
                    )
                    CBV = most_relevant_vehicle
                else:
                    raise ValueError(f'Unknown CBV selecting method {self.CBVs_select_method}')

                if CBV:
                    # if CBV not in the CBVs list, put the new one in
                    if CBV.id not in self.CBVs:
                        self.CBVs[CBV.id] = CBV
                        CBV.set_autopilot(enabled=False)  # prepared to be controlled
                        self.register_CBV_sensor(CBV)
                        CarlaDataProvider.add_CBV(self.ego_vehicle, CBV)

                    # update the nearby vehicles around the CBV
                        self.CBVs_nearby_vehicles[CBV.id] = get_nearby_vehicles(CBV, self.search_radius)
        self.scenario_manager.update_CBV_nearby_vehicles(self.CBVs_nearby_vehicles)

    def _get_camera_target_actor(self, role_name):
        if role_name == 'ego':
            return self.ego_vehicle

        if self.ego_vehicle is None:
            return None

        special_actors = CarlaDataProvider.get_special_actors_by_ego(self.ego_vehicle)
        return special_actors.get(role_name)

    def _remove_actor_camera_sensors(self):
        for role_sensors in self.actor_camera_sensors.values():
            for sensor in role_sensors.values():
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
        self.actor_camera_sensors = {}
        self.camera_actor_ids = {}

    def _remove_actor_lidar_sensors(self):
        for sensor in self.actor_lidar_sensors.values():
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.actor_lidar_sensors = {}
        self.lidar_actor_ids = {}

    def _sync_actor_camera_sensors(self):
        if not self.save_camera_frames or self.eval_mode != 'render':
            return
        self_weakref = weakref.ref(self)

        def get_actor_camera_img(env_self, role_name, view_name, data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            env_self.actor_camera_imgs[role_name][view_name] = array

        for role_name in self.camera_actor_roles:
            actor = self._get_camera_target_actor(role_name)
            current_actor_id = actor.id if actor is not None else None
            previous_actor_id = self.camera_actor_ids.get(role_name)

            if current_actor_id == previous_actor_id and role_name in self.actor_camera_sensors:
                continue

            if role_name in self.actor_camera_sensors:
                for sensor in self.actor_camera_sensors[role_name].values():
                    if sensor is not None and sensor.is_alive:
                        sensor.stop()
                        sensor.destroy()
                self.actor_camera_sensors.pop(role_name, None)

            if actor is None:
                self.camera_actor_ids.pop(role_name, None)
                continue

            self.actor_camera_sensors[role_name] = {}
            for view_name, transform in self.camera_view_transforms.items():
                sensor = self.world.spawn_actor(self.front_camera_bp, transform, attach_to=actor)
                sensor.listen(
                    lambda data, role=role_name, view=view_name, self_ref=self_weakref:
                    get_actor_camera_img(self_ref(), role, view, data) if self_ref() else None
                )
                self.actor_camera_sensors[role_name][view_name] = sensor
            self.camera_actor_ids[role_name] = current_actor_id

    def _sync_actor_lidar_sensors(self):
        if not self.save_camera_frames or not self.save_actor_lidar_frames or self.eval_mode != 'render':
            return
        self_weakref = weakref.ref(self)

        def get_actor_lidar_points(env_self, role_name, data):
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            env_self.actor_lidar_points[role_name] = np.reshape(points, (-1, 4))

        for role_name in self.camera_actor_roles:
            actor = self._get_camera_target_actor(role_name)
            current_actor_id = actor.id if actor is not None else None
            previous_actor_id = self.lidar_actor_ids.get(role_name)

            if current_actor_id == previous_actor_id and role_name in self.actor_lidar_sensors:
                continue

            existing_sensor = self.actor_lidar_sensors.pop(role_name, None)
            if existing_sensor is not None and existing_sensor.is_alive:
                existing_sensor.stop()
                existing_sensor.destroy()

            if actor is None:
                self.lidar_actor_ids.pop(role_name, None)
                continue

            sensor = self.world.spawn_actor(self.actor_lidar_bp, self.actor_lidar_trans, attach_to=actor)
            sensor.listen(
                lambda data, role=role_name, self_ref=self_weakref:
                get_actor_lidar_points(self_ref(), role, data) if self_ref() else None
            )
            self.actor_lidar_sensors[role_name] = sensor
            self.lidar_actor_ids[role_name] = current_actor_id

    def _write_scene_metadata(self, base_dir):
        meta_path = os.path.join(base_dir, 'meta.json')
        os.makedirs(base_dir, exist_ok=True)
        if os.path.exists(meta_path):
            return

        actors_meta = {}
        for role_name in self.metadata_actor_roles:
            actor = self._get_camera_target_actor(role_name)
            if actor is not None:
                actors_meta[role_name] = {
                    'id': actor.id,
                    'type_id': actor.type_id
                }

        weather_dict = self._weather_to_dict(self.world.get_weather())
        scene_naming = self._get_scene_naming_fields()
        metadata = {
            'scenario_id': self.config.scenario_id,
            'data_id': self.config.data_id,
            'map': self.world.get_map().name.split('/')[-1],
            'camera_fps': self.camera_fps,
            'camera_export': {
                'width': self.camera_export_width,
                'height': self.camera_export_height,
                'format': self.camera_export_format,
                'quality': self.camera_export_quality if self.camera_export_format in ('jpg', 'jpeg') else None,
                'optimize': bool(self.camera_export_optimize) if self.camera_export_format in ('jpg', 'jpeg') else None,
            },
            'views': list(self.camera_view_transforms.keys()),
            'actors': actors_meta,
            'parameters': self.config.parameters,
            'target_outcome': self.config.parameters.get('target_outcome') if self.config.parameters is not None else None,
            'weather': weather_dict,
            'weather_label': self._infer_weather_label(weather_dict),
            'time_of_day_label': self._infer_time_of_day_label(weather_dict),
            'scenario_type_id': scene_naming['scenario_type_id'],
            'scenario_subtype_id': scene_naming['scenario_subtype_id'],
            'scenario_number': scene_naming['scenario_number'],
            'scene_name': scene_naming['scene_name'],
            'lidar': {
                'enabled': bool(self.save_actor_lidar_frames),
                'roles': list(self.camera_actor_roles) if self.save_actor_lidar_frames else [],
                'channels': 32,
                'range_m': 70,
                'vertical_fov_deg': 30,
                'rotation_frequency_hz': 10,
            },
            'accident_type': 'pending',
        }
        with open(meta_path, 'w', encoding='utf-8') as meta_file:
            json.dump(metadata, meta_file, indent=2)

    def _update_scene_metadata_result(self, base_dir):
        meta_path = os.path.join(base_dir, 'meta.json')
        if not os.path.exists(meta_path):
            return

        with open(meta_path, 'r', encoding='utf-8') as meta_file:
            metadata = json.load(meta_file)

        metadata['accident_type'] = 'A' if self.scene_had_ego_collision else 'normal'

        with open(meta_path, 'w', encoding='utf-8') as meta_file:
            json.dump(metadata, meta_file, indent=2)

    def _save_front_camera_frame(self):
        if not self.save_camera_frames or self.eval_mode != 'render':
            return
        if getattr(self, 'time_step', 0) % self.capture_stride != 0:
            return

        scenario_name = f"Scenario{self.config.scenario_id}"
        map_name = self.world.get_map().name.split('/')[-1]
        base_dir = os.path.join(
            self.output_dir,
            f"{scenario_name}_{map_name}",
            "camera_frames",
            f"data_{self.config.data_id:04d}"
        )
        self._write_scene_metadata(base_dir)

        frame_ext = 'jpg' if self.camera_export_format in ('jpg', 'jpeg') else self.camera_export_format
        frame_name = f"frame_{self.front_camera_frame_idx:04d}.{frame_ext}"
        for role_name, role_images in self.actor_camera_imgs.items():
            if role_name not in self.actor_camera_sensors:
                continue
            for view_name, image_array in role_images.items():
                view_dir = os.path.join(base_dir, role_name, view_name)
                os.makedirs(view_dir, exist_ok=True)
                save_path = os.path.join(view_dir, frame_name)
                self._queue_export_job({
                    'type': 'image',
                    'path': save_path,
                    'array': image_array.copy(),
                    'frame_ext': frame_ext,
                    'quality': self.camera_export_quality,
                    'optimize': self.camera_export_optimize,
                })
        self.front_camera_frame_idx += 1

    def _save_actor_lidar_frame(self):
        if not self.save_camera_frames or not self.save_actor_lidar_frames or self.eval_mode != 'render':
            return
        if getattr(self, 'time_step', 0) % self.capture_stride != 0:
            return

        scenario_name = f"Scenario{self.config.scenario_id}"
        map_name = self.world.get_map().name.split('/')[-1]
        base_dir = os.path.join(
            self.output_dir,
            f"{scenario_name}_{map_name}",
            "camera_frames",
            f"data_{self.config.data_id:04d}"
        )
        self._write_scene_metadata(base_dir)

        frame_name = f"frame_{max(self.front_camera_frame_idx - 1, 0):04d}.npy"
        for role_name, point_cloud in self.actor_lidar_points.items():
            if role_name not in self.actor_lidar_sensors:
                continue
            lidar_dir = os.path.join(base_dir, role_name, 'lidar')
            os.makedirs(lidar_dir, exist_ok=True)
            self._queue_export_job({
                'type': 'npy',
                'path': os.path.join(lidar_dir, frame_name),
                'array': point_cloud.copy(),
            })

    def reset(self, config, env_id):
        self.config = config
        self.env_id = env_id

        if getattr(config, 'weather', None) is not None:
            config.weather = self._override_weather_from_labels(config.weather, getattr(config, 'parameters', None))
            self.world.set_weather(config.weather)

        self._create_sensors()
        # create RouteScenario, scenario manager, ego_vehicle etc.
        self._create_scenario(config, env_id)

        # generate the initial background vehicles
        self._run_scenario()
        self._attach_sensor()

        # first update the info in the CarlaDataProvider
        CarlaDataProvider.on_carla_tick()

        # route planner for ego vehicle
        self.routeplanner = RoutePlanner(self.ego_vehicle, self.max_waypt, self.global_route_waypoints)
        self.waypoints, self.goal_waypoint, self.red_light_state = self.routeplanner.run_step()

        # Update time_steps
        self.time_step = 0
        self.reset_step += 1

        # find ego nearby vehicles
        self.ego_nearby_vehicles = get_nearby_vehicles(self.ego_vehicle, self.search_radius)

        # set controlled bv
        self.CBVs_selection()
        self._sync_actor_camera_sensors()
        self._sync_actor_lidar_sensors()

        # Get actors' polygon list (for visualization)
        if self.birdeye_render:
            self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
            # self.walker_polygons = [self._get_actor_polygons('walker.*')]

        # applying setting can tick the world and get data from sensors
        # removing this block will cause error: AttributeError: 'NoneType' object has no attribute 'raw_data'
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()

        self._save_front_camera_frame()
        self._save_actor_lidar_frame()

        return self._get_obs(), self._get_info(next_info=False, reset=True)

    def _attach_sensor(self):
        if self.eval_mode == 'render':
            self_weakref = weakref.ref(self)  # weak reference of self
            # Add lidar sensor
            if not self.disable_lidar:
                self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
                self.lidar_sensor.listen(lambda data, self_ref=self_weakref: get_lidar_data(self_ref(), data) if self_ref() else None)

            def get_lidar_data(ego_self, data):
                ego_self.lidar_data = data

            # Add camera sensor
            self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
            self.camera_sensor.listen(lambda data, self_ref=self_weakref: get_camera_img(self_ref(), data) if self_ref() else None)

            def get_camera_img(ego_self, data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                ego_self.BGR_img = copy.deepcopy(array)
                array = array[:, :, ::-1]
                ego_self.camera_img = array

            # Add sem_camera sensor
            if self.enable_sem:
                self.sem_sensor = self.world.spawn_actor(self.sem_bp, self.sem_trans, attach_to=self.ego_vehicle)
                self.sem_sensor.listen(lambda data, self_ref=self_weakref: get_sem_img(self_ref(), data) if self_ref() else None)

            if self.save_camera_frames:
                self._sync_actor_camera_sensors()
                if self.save_actor_lidar_frames:
                    self._sync_actor_lidar_sensors()

            def get_sem_img(ego_self, data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, 2]  # from PlanT
                ego_self.sem_img = array

    def visualize_actors(self):
        # visualize the past trajectory of all the actor on the map
        if self.eval_mode == 'render' and self.draw_debug_overlays:
            draw_trajectory(self.world, self.ego_vehicle, self.time_step)

            # if the ego agent is learnable and need to viz the route, then draw the target waypoints
            if self.viz_route:
                waypoint_route = np.array([[node[0], node[1]] for node in self.waypoints])
                draw_route(self.world, self.ego_vehicle, waypoint_route)

    def step_before_tick(self, ego_action, scenario_action):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp

                # update the CBV action
                self.scenario_manager.get_update(timestamp, scenario_action)

                # if CBV has changed, update the ego CBV distance
                update_goal_CBV_dis(self.ego_vehicle, self.goal_waypoint)

                # Calculate acceleration and steering
                if not self.auto_ego:
                    if not self.ego_agent_learnable:
                        # the rule-based action
                        throttle = ego_action[0]
                        steer = ego_action[1]
                        brake = ego_action[2]
                        throttle, steer, brake = self._apply_collision_target_ego_override(throttle, steer, brake)
                        act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
                    else:
                        # the learnable agent action
                        ego_action = process_ego_action(ego_action, acc_range=self.acc_range, steering_range=self.steering_range)
                        ego_action[0], ego_action[1], ego_action[2] = self._apply_collision_target_ego_override(
                            ego_action[0], ego_action[1], ego_action[2]
                        )
                        # apply ego control
                        act = carla.VehicleControl(throttle=float(ego_action[0]), steer=float(ego_action[1]), brake=float(ego_action[2]))
                    self.ego_vehicle.apply_control(act)  # apply action of the ego vehicle on the next tick

            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    def step_after_tick(self):
        if self.birdeye_render:
            # Append actors' polygon lists
            vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
            self.vehicle_polygons.append(vehicle_poly_dict)
            while len(self.vehicle_polygons) > self.max_past_step:
                self.vehicle_polygons.pop(0)
            # walker_poly_dict = self._get_actor_polygons('walker.*')
            # self.walker_polygons.append(walker_poly_dict)
            # while len(self.walker_polygons) > self.max_past_step:
            #     self.walker_polygons.pop(0)

        # After tick, update all the actors' velocity map, location map and transform map
        CarlaDataProvider.on_carla_tick()

        # route planner
        # self.waypoints: the waypoints from the waypoint buffer, needed to be followed
        self.waypoints, self.goal_waypoint, self.red_light_state = self.routeplanner.run_step()

        # find ego nearby vehicles
        self.ego_nearby_vehicles = get_nearby_vehicles(self.ego_vehicle, self.search_radius)

        extra_status = {}
        if self.eval_mode == 'analysis':
            # update the BVs record when evaluating
            extra_status.update(
                get_records(self.ego_vehicle, self.CBVs_collision, self.ego_nearby_vehicles,
                            self.goal_waypoint, self.goal_point_radius, self.search_radius)
            )

        # update the running status and check whether terminate or not
        self.scenario_manager.update_running_status(extra_status)

        # update the self.ego_collide status
        self.get_ego_collision_status()

        origin_info = self._get_info(next_info=True)  # info of old CBV

        # if CBV collided, then remove it
        self._remove_and_clean_CBV(origin_info)

        # select the new CBV
        self.CBVs_selection() if self.scenario_manager.running else None
        self._sync_actor_camera_sensors()
        self._sync_actor_lidar_sensors()

        updated_CBVs_info = self._get_info(next_info=False)  # info of new CBV

        self.visualize_actors()  # visualize the controlled bv and the waypoints in clients side after tick
        self._save_front_camera_frame()
        self._save_actor_lidar_frame()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return self._get_obs(), self._get_reward(), self._terminal(), [origin_info, updated_CBVs_info]

    def _get_info(self, next_info, reset=False):
        info = {}
        # info for scenario agents to take action (scenario obs)
        info.update(self.scenario_manager.route_scenario.update_info(goal_waypoint=self.goal_waypoint))

        # the feasibility needs the ego info (without route info)
        if self.mode == 'collect_feasibility_data':
            info.update(self.scenario_manager.route_scenario.update_ego_info(self.ego_nearby_vehicles))
            info['ego_min_dis'] = get_ego_min_dis(self.ego_vehicle, self.ego_nearby_vehicles, self.search_radius)
            info['ego_collide'] = float(self.ego_collide)

        # the feasibility-guided method need CBV-related ego_obs
        if self.use_feasibility:
            info['ego_CBV_obs'] = {}
            info['ego_CBV_dis'] = {}
            for CBV_id, CBV in self.CBVs.items():
                info['ego_CBV_obs'][CBV_id] = self.scenario_manager.route_scenario.update_ego_info(ego_nearby_vehicles=[CBV])['ego_obs']
                info['ego_CBV_dis'][CBV_id] = get_min_distance_across_bboxes(self.ego_vehicle, CBV)

        # when resetting
        if reset:
            info.update({
                'route_waypoints': self.global_route_waypoints,  # the global route waypoints
                'gps_route': self.gps_route,  # the global gps route
                'route': self.route,  # the global route
            })

        # when after the tick before selecting a new CBV
        elif next_info:
            # the total reward for the CBV training
            info['CBVs_reward'] = self._get_scenario_reward()

            # if CBV collide with other vehicles, then terminate
            info['CBVs_terminated'] = self._get_CBVs_terminated()

            # if Ego stuck, timeout or max step, then truncated
            info['CBVs_truncated'] = self._get_CBVs_truncated()

        return info

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in CarlaDataProvider._world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self):
        if self.birdeye_render:
            # set ego information for birdeye_render
            self.birdeye_render.set_hero(self.ego_vehicle, self.ego_vehicle.id)
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            # self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints

            # render birdeye image with the birdeye_render
            birdeye_render_types = ['roadmap', 'actors', 'waypoints']
            birdeye_surface = self.birdeye_render.render(birdeye_render_types)
            birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
            center = (int(birdeye_surface.shape[0] / 2), int(birdeye_surface.shape[1] / 2))
            width = height = int(self.display_size / 2)
            birdeye = birdeye_surface[center[0] - width:center[0] + width, center[1] - height:center[1] + height]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            if not self.disable_lidar:
                # get Lidar image
                point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
                point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
                x = point_cloud[:, 0:1]
                y = point_cloud[:, 1:2]
                z = point_cloud[:, 2:3]
                intensity = point_cloud[:, 3:4]
                point_cloud = np.concatenate([y, -x, z], axis=1)
                # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin, and z is set to be two bins.
                y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
                x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
                z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
                # Get lidar image according to the bins
                lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
                lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
                lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
                wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
                wayptimg = np.expand_dims(wayptimg, axis=2)
                wayptimg = np.fliplr(np.rot90(wayptimg, 3))
                # Get the final lidar image
                lidar = np.concatenate((lidar, wayptimg), axis=2)
                lidar = np.flip(lidar, axis=1)
                lidar = np.rot90(lidar, 1) * 255

                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id * self.display_size))

                # display lidar image
                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, self.env_id * self.display_size))

                # display camera image
                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size * 2, self.env_id * self.display_size))
            else:
                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id * self.display_size))

                # display camera image
                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size, self.env_id * self.display_size))

                # display masked viz 3rd person
                if self.enable_sem:
                    masked_img = get_masked_viz_3rd_person(self.BGR_img, self.sem_img)
                    masked_image = resize(masked_img, (self.obs_size, self.obs_size)) * 255
                    masked_image_surface = rgb_to_display_surface(masked_image, self.display_size)
                    self.display.blit(masked_image_surface, (self.display_size * 2, self.env_id * self.display_size))

        if self.agent_obs_type == 'ego_state':
            # Ego state
            ego_trans = CarlaDataProvider.get_transform(self.ego_vehicle)
            ego_loc = ego_trans.location
            ego_pos = np.array([ego_loc.x, ego_loc.y])
            ego_speed = calculate_abs_velocity(CarlaDataProvider.get_velocity(self.ego_vehicle))  # m/s
            ego_compass = np.deg2rad(ego_trans.rotation.yaw)  # the yaw angle in radius
            ego_state = {
                'gps': ego_pos,
                'speed': ego_speed,
                'compass': ego_compass
            }
            obs = {
                'ego_state': ego_state,
            }
        elif self.agent_obs_type == 'ego_obs':
            # the ego obs for RL training needs the route_info
            obs = self.scenario_manager.route_scenario.update_ego_info(self.ego_nearby_vehicles, waypoints=self.waypoints)
        elif self.agent_obs_type == 'no_obs':
            obs = None
        else:
            raise NotImplementedError(f"Unsupported obs tyep: {self.agent_obs_type}.")
        return obs

    def _get_reward(self):
        """ Calculate the step reward. """
        r_collision = -1 if self.ego_collide else 0

        # reward for steering:
        r_steer = -self.ego_vehicle.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        # reward for speed tracking
        v = CarlaDataProvider.get_velocity(self.ego_vehicle)

        # cost for too fast
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)
        r_fast = -1 if lspeed_lon > self.desired_speed else 0

        # cost for lateral acceleration
        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon ** 2

        # combine all rewards
        # r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
        # reward from "Interpretable End-to-End Urban Autonomous Driving With Latent Deep Reinforcement Learning"
        r = 10 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1
        return r

    def _get_scenario_reward(self):
        """
            sparse reward:
            -1:CBV collide with other bvs or not on the road
            0:CBV normal driving
            1:CBV collide with ego vehicle
            goal_CBV_dis_reward ~ [-1, 1]: the ratio of (init_goal_CBV_dis-current_goal_CBV_dis)/init_goal_CBV_dis
        """
        CBVs_reward = {}
        for CBV_id, CBV in self.CBVs.items():
            # prevent the CBV getting too close to the other bvs
            # CBV_min_dis, CBV_min_dis_reward = get_CBV_bv_reward(self.CBVs[CBV_id], self.search_radius, self.CBVs_nearby_vehicles[CBV_id])

            # encourage CBV to get closer to the goal point
            delta_dis, CBV_goal_dis = get_CBV_ego_reward(self.ego_vehicle, CBV, self.goal_waypoint)  # [-1, 1]

            # CBV collision punish (collide with another vehicle, don't avoid ego vehicle collision)
            collision_punish = -1 if self.CBVs_collision[CBV_id] is not None and self.CBVs_collision[CBV_id]['other_actor_id'] != self.ego_vehicle.id else 0

            # terminal reward (reach the goal)
            if CBV_goal_dis < self.goal_point_radius:
                terminal_reward = 1
                CarlaDataProvider.CBV_reach_goal(self.ego_vehicle, CBV)
            else:
                terminal_reward = 0

            # final scenario agent rewards
            CBVs_reward[CBV_id] = delta_dis + 15 * collision_punish + 15 * terminal_reward

        return CBVs_reward

    def _get_CBVs_terminated(self):
        CBVs_terminated = {}
        for CBV_id in self.CBVs:
            # if CBV collide with the other vehicles, then CBV terminated
            if self.CBVs_collision[CBV_id] is not None:
                CBVs_terminated[CBV_id] = True
            # if CBV reach the goal, then CBV terminated
            elif CBV_id in CarlaDataProvider.get_CBVs_reach_goal_by_ego(self.ego_vehicle).keys():
                CBVs_terminated[CBV_id] = True
            else:
                CBVs_terminated[CBV_id] = False
        return CBVs_terminated

    def _get_CBVs_truncated(self):
        CBVs_truncated = {}
        for CBV_id, CBV in self.CBVs.items():
            if not self.scenario_manager.running:
                # if the Ego stops, then the CBV is truncated
                CBVs_truncated[CBV_id] = True
            elif not check_interaction(self.ego_vehicle, CBV, self.ego_length, delta_forward_angle=100, ego_fov=200):
                # loose condition to check truncated
                CBVs_truncated[CBV_id] = True
            elif check_CBV_BV_stuck(CBV, self.ego_vehicle, max_distance=8, angle=10):
                # the CBV already reached the goal stuck the current CBV
                CBVs_truncated[CBV_id] = True
            else:
                CBVs_truncated[CBV_id] = False

        return CBVs_truncated

    def get_ego_collision_status(self):
        if self.scenario_manager.ego_collision:
            # from the view of ego vehicle (maybe not correct since ignore too close collision)
            self.ego_collide = True
        else:
            # if the ego collision detector didn't detect the collision, need to use CBV collision detector
            self.ego_collide = False
            for collision_event in self.CBVs_collision.values():
                if collision_event is not None and collision_event['other_actor_id'] == self.ego_vehicle.id:
                    self.ego_collide = True
                    break

        if self.ego_collide:
            self.scene_had_ego_collision = True

        self.logger.log(f'>> Ego collide', color='yellow') if self.ego_collide else None

    def _terminal(self):
        return not self.scenario_manager.running

    def _write_scene_result(self):
        if not self.save_camera_frames or self.output_dir is None or self.config is None:
            return

        scenario_name = f"Scenario{self.config.scenario_id}"
        map_name = self.world.get_map().name.split('/')[-1]
        base_dir = os.path.join(
            self.output_dir,
            f"{scenario_name}_{map_name}",
            "camera_frames",
            f"data_{self.config.data_id:04d}"
        )
        os.makedirs(base_dir, exist_ok=True)
        self._update_scene_metadata_result(base_dir)
        result_path = os.path.join(base_dir, 'scene_result.json')
        final_record = self.scenario_manager.running_record[-1] if self.scenario_manager.running_record else {}
        result = {
            'ego_collision': self.scene_had_ego_collision,
            'time_steps': self.time_step,
            'route_completion': final_record.get('route_complete'),
            'current_game_time': final_record.get('current_game_time')
        }
        with open(result_path, 'w', encoding='utf-8') as result_file:
            json.dump(result, result_file, indent=2)

    def _remove_sensor(self):
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
            self.lidar_sensor = None
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
        self._remove_actor_camera_sensors()
        self._remove_actor_lidar_sensors()
        if self.sem_sensor is not None:
            self.sem_sensor.stop()
            self.sem_sensor.destroy()
            self.sem_sensor = None
        if self.CBVs_collision_sensor:
            # remove the collision sensor that has not been destroyed
            for sensor in self.CBVs_collision_sensor.values():
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            self.CBVs_collision_sensor = {}

    def _remove_CBV_sensor(self, CBV_id):
        sensor = self.CBVs_collision_sensor.pop(CBV_id, None)
        if sensor is not None and sensor.is_alive:
            sensor.stop()
            sensor.destroy()
            self.CBVs_collision.pop(CBV_id)

    def _remove_ego(self):
        if self.ego_vehicle is not None and CarlaDataProvider.actor_id_exists(self.ego_vehicle.id):
            CarlaDataProvider.remove_actor_by_id(self.ego_vehicle.id)
        self.ego_vehicle = None

    def _remove_and_clean_CBV(self, info):
        # remove the truncated CBV from the CBV list and set them free to normal bvs
        CBVs_truncated = info['CBVs_truncated']
        for CBV_id, truncated in CBVs_truncated.items():
            if truncated:
                CBV = self.CBVs.pop(CBV_id, None)
                if CBV is not None:
                    # remove the CBV collision sensor
                    self._remove_CBV_sensor(CBV_id)
                    # remove the truncated CBV from existing CBV lists
                    CBV.set_autopilot(enabled=True)  # set the original CBV to normal bvs
                    self.CBVs_nearby_vehicles.pop(CBV_id)
                    CarlaDataProvider.CBV_back_to_BV(self.ego_vehicle, CBV)

        # clean the terminated CBV
        CBVs_terminated = info['CBVs_terminated']
        for CBV_id, terminated in CBVs_terminated.items():
            if terminated:
                CBV = self.CBVs.pop(CBV_id, None)
                if CBV is not None:
                    # remove sensor
                    self._remove_CBV_sensor(CBV_id)
                    if CBV_id in CarlaDataProvider.get_CBVs_reach_goal_by_ego(self.ego_vehicle).keys():
                        # set the goal reaching CBV free
                        CBV.set_autopilot(enabled=True)
                    else:
                        CarlaDataProvider.CBV_terminate(self.ego_vehicle, CBV)
                        # clean the CBV from the environment
                        if CarlaDataProvider.actor_id_exists(CBV_id):
                            CarlaDataProvider.remove_actor_by_id(CBV_id)
                    self.CBVs_nearby_vehicles.pop(CBV_id)

    def _reset_variables(self):
        self.CBVs = {}
        self.CBVs_nearby_vehicles = {}
        self.gps_route = None
        self.route = None
        self.global_route_waypoints = None
        self.waypoints = None
        self.goal_waypoint = None
        self.ego_collide = False
        self.scene_had_ego_collision = False
        self.CBVs_collision = {}
        self.front_camera_frame_idx = 0
        self.actor_camera_sensors = {}
        self.camera_actor_ids = {}
        self.actor_lidar_sensors = {}
        self.lidar_actor_ids = {}
        self.actor_camera_imgs = {
            role_name: {
                view_name: np.zeros((self.camera_export_height, self.camera_export_width, 3), dtype=np.uint8)
                for view_name in getattr(self, 'camera_view_transforms', {'front': None})
            }
            for role_name in self.camera_actor_roles
        }
        self.actor_lidar_points = {
            role_name: np.zeros((0, 4), dtype=np.float32)
            for role_name in self.camera_actor_roles
        }

    def clean_up(self):
        self._flush_export_jobs()
        self._write_scene_result()

        # remove the render sensor only when evaluating
        self._remove_sensor()

        # destroy criterion sensors on the ego vehicle
        self.scenario_manager.clean_up()

        # remove the ego vehicle after removing all the sensors
        self._remove_ego()

        # remove temp variables
        self._reset_variables()
