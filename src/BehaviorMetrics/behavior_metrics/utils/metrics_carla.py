#!/usr/bin/env python

"""This module contains the metrics manager.
This module is in charge of generating metrics for a brain execution.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time
import os
import rosbag
import re

from bagpy import bagreader
from utils.logger import logger
from scripts import plot_tensorboard_perc_histogram

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import carla

PLOTS_FONTSIZE=30

def circuit_distance_completed(checkpoints, lap_point):
    previous_point = []
    diameter = 0
    for i, point in enumerate(checkpoints):
        current_point = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        if i != 0:
            dist = (previous_point - current_point) ** 2
            dist = np.sum(dist, axis=0)
            dist = np.sqrt(dist)
            diameter += dist
        if point is lap_point:
            break
        previous_point = current_point
    return diameter

from scipy.spatial import cKDTree

def get_metrics(experiment_metrics, experiment_metrics_bag_filename, map_waypoints, experiment_metrics_filename, config, waypoints_info, clockwise):

    # waypoint_xy = np.array([
    #     [wp.transform.location.x, wp.transform.location.y]
    #     for wp in map_waypoints
    # ])
    # kdtree = cKDTree(waypoint_xy)

    time_counter = 5
    while not os.path.exists(experiment_metrics_bag_filename):
        time.sleep(1)
        time_counter -= 1
        if time_counter <= 0:
            ValueError(f"{experiment_metrics_bag_filename} isn't a file!")
            return {}

    try:
        bag_reader = bagreader(experiment_metrics_bag_filename)
    except rosbag.bag.ROSBagException:
        return {}

    csv_files = []
    for topic in bag_reader.topics:
        data = bag_reader.message_by_topic(topic)
        csv_files.append(data)

    data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-odometry.csv'
    dataframe_pose = pd.read_csv(data_file)
    checkpoints = []
    for index, row in dataframe_pose.iterrows():
        checkpoints.append(row)

    if config.task == 'follow_lane_traffic':
        data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-npc_vehicle_1-odometry.csv'
        dataframe_pose = pd.read_csv(data_file)
        checkpoints_2 = []
        for index, row in dataframe_pose.iterrows():
            checkpoints_2.append(row)

    data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/clock.csv'
    dataframe_clock = pd.read_csv(data_file)
    clock_points = []
    for index, row in dataframe_clock.iterrows():
        clock_points.append(row)
    start_clock = clock_points[0]
    seconds_start = start_clock['clock.secs']
    seconds_end = clock_points[len(clock_points) - 1]['clock.secs']

    collision_points = []
    if '/carla/ego_vehicle/collision' in bag_reader.topics:
        data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-collision.csv'
        dataframe_collision = pd.read_csv(data_file)
        for index, row in dataframe_collision.iterrows():
            collision_points.append(row)

    lane_invasion_points = []
    if '/carla/ego_vehicle/lane_invasion' in bag_reader.topics:
        data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-lane_invasion.csv'
        dataframe_lane_invasion = pd.read_csv(data_file, on_bad_lines='skip')
        for index, row in dataframe_lane_invasion.iterrows():
            lane_invasion_points.append(row)

    data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-speedometer.csv'
    dataframe_speedometer = pd.read_csv(data_file)
    speedometer_points = []
    for index, row in dataframe_speedometer.iterrows():
        speedometer_points.append(row)

    data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-vehicle_status.csv'
    dataframe_vehicle_status = pd.read_csv(data_file)
    vehicle_status_points = []
    brake_points = []
    checkpoints_steer = []
    for index, row in dataframe_vehicle_status.iterrows():
        vehicle_status_points.append(row)
        if row["control.brake"] > 0:
            action_applied = -row["control.brake"]
        else:
            action_applied = row["control.throttle"]
        brake_points.append(action_applied)
        checkpoints_steer.append(row["control.steer"])

    if len(checkpoints) > 1:
        starting_point = checkpoints[0]
        starting_point = (starting_point['pose.pose.position.x'], starting_point['pose.pose.position.y'])
        experiment_metrics['starting_point'] = starting_point
        experiment_metrics = get_distance_completed(experiment_metrics, checkpoints)
        experiment_metrics = get_average_speed(experiment_metrics, speedometer_points, checkpoints, waypoints_info, clockwise)
        experiment_metrics = get_suddenness_control_commands(experiment_metrics, vehicle_status_points)
        experiment_metrics, collisions_checkpoints = get_collisions(experiment_metrics, collision_points, dataframe_pose)
        experiment_metrics, lane_invasion_checkpoints = get_lane_invasions(experiment_metrics, lane_invasion_points, dataframe_pose)
        experiment_metrics['experiment_total_simulated_time'] = seconds_end - seconds_start
        if config.task == 'follow_lane_traffic':
            experiment_metrics = get_distance_other_vehicle(experiment_metrics, checkpoints, checkpoints_2)

        if 'bird_eye_view_images' in experiment_metrics:
            experiment_metrics['bird_eye_view_images_per_second'] = experiment_metrics['bird_eye_view_images'] / experiment_metrics['experiment_total_simulated_time']
            experiment_metrics['bird_eye_view_unique_images_per_second'] = experiment_metrics['bird_eye_view_unique_images'] / experiment_metrics['experiment_total_simulated_time']

        logger.info("getting position deviation and completed distance")

        experiment_metrics = get_position_deviation_and_effective_completed_distance(experiment_metrics, checkpoints, map_waypoints, experiment_metrics_filename, speedometer_points, collisions_checkpoints, lane_invasion_checkpoints, brake_points, checkpoints_steer, waypoints_info, clockwise)
        experiment_metrics['completed_laps'] = get_completed_laps(checkpoints, starting_point)
        # shutil.rmtree(experiment_metrics_bag_filename.split('.bag')[0])
        return experiment_metrics
    else:
        return {}

def get_completed_laps(checkpoints, starting_point):
    points_to_start_count = 50
    completed_laps = 0
    for x, checkpoint in enumerate(checkpoints):
        if points_to_start_count > 0:
            points_to_start_count -= 1
        else:
            point_1 = np.array([checkpoint['pose.pose.position.x'], checkpoint['pose.pose.position.y']])
            point_2 = np.array([starting_point[0], starting_point[1]])
            dist = (point_2 - point_1) ** 2
            dist = np.sum(dist, axis=0)
            dist = np.sqrt(dist)
            if dist < 0.5:
                completed_laps += 1
                points_to_start_count = 50
    
    return completed_laps

def get_distance_completed(experiment_metrics, checkpoints):
    end_point = checkpoints[len(checkpoints) - 1]
    experiment_metrics['completed_distance'] = circuit_distance_completed(checkpoints, end_point)
    return experiment_metrics


def get_average_speed(experiment_metrics, speedometer_points, checkpoints, waypoints_info, clockwise):
    """
    Assign speeds to the nearest waypoint and determine direction (right/left)
    """
    previous_speed = 0
    speedometer_points_sum = 0
    suddenness_distance_speeds = []
    speed_points = []

    speeds_by_waypoint = []
    waypoint_order = []

    for i, point in enumerate(speedometer_points):
        speed_point = point.data * 3.6  # km/h
        speedometer_points_sum += speed_point
        suddenness_distance_speed = abs(speed_point - previous_speed)
        suddenness_distance_speeds.append(suddenness_distance_speed)
        previous_speed = speed_point
        speed_points.append(speed_point)

        agent_location = (
            checkpoints[i]['pose.pose.position.x'],
            - checkpoints[i]['pose.pose.position.y'] # Note that ros is publishing y in opposite sign than Carla transfomrs
        )

        # Find closest waypoint (brute force)
        distances = [
            math.sqrt((wp['transform'].location.x - agent_location[0]) ** 2 +
                      (wp['transform'].location.y - agent_location[1]) ** 2)
            for wp in waypoints_info
        ]
        closest_idx = int(np.argmin(distances))
        speeds_by_waypoint.append(speed_point)
        waypoint_order.append(closest_idx)

    experiment_metrics['average_speed'] = speedometer_points_sum / len(speedometer_points)
    experiment_metrics['suddenness_distance_speed'] = sum(suddenness_distance_speeds) / len(suddenness_distance_speeds)
    experiment_metrics['max_speed'] = max(speed_points)
    experiment_metrics['min_speed'] = min(speed_points)
    experiment_metrics['speeds'] = speed_points
    experiment_metrics['speeds_by_waypoint'] = speeds_by_waypoint
    experiment_metrics['waypoint_order'] = waypoint_order
    experiment_metrics['waypoints_info_colors'] = [wp['color'] for wp in waypoints_info]
    experiment_metrics['clockwise'] = bool(clockwise)

    return experiment_metrics

def get_suddenness_control_commands(experiment_metrics, vehicle_status_points):
    previous_commanded_throttle = 0
    previous_commanded_steer = 0
    previous_commanded_brake = 0
    suddenness_distance_control_commands = []
    suddenness_distance_throttle = []
    suddenness_distance_steer = []
    suddenness_distance_brake_command = []

    for point in vehicle_status_points:
        throttle = point['control.throttle']
        steer = point['control.steer']
        brake_command = point['control.brake']

        a = np.array((throttle, steer, brake_command))
        b = np.array((previous_commanded_throttle, previous_commanded_steer, previous_commanded_brake))
        distance = np.linalg.norm(a - b)
        suddenness_distance_control_commands.append(distance)

        a = np.array((throttle))
        b = np.array((previous_commanded_throttle))
        distance_throttle = np.linalg.norm(a - b)
        suddenness_distance_throttle.append(distance_throttle)

        a = np.array((steer))
        b = np.array((previous_commanded_steer))
        distance_steer = np.linalg.norm(a - b)
        suddenness_distance_steer.append(distance_steer)

        a = np.array((brake_command))
        b = np.array((previous_commanded_brake))
        distance_brake_command = np.linalg.norm(a - b)
        suddenness_distance_brake_command.append(distance_brake_command)

        previous_commanded_throttle = throttle
        previous_commanded_steer = steer
        previous_commanded_brake = brake_command

    experiment_metrics['suddenness_distance_control_commands'] = sum(suddenness_distance_control_commands) / len(suddenness_distance_control_commands)
    experiment_metrics['suddenness_distance_throttle'] = sum(suddenness_distance_throttle) / len(suddenness_distance_throttle)
    experiment_metrics['suddenness_distance_steer'] = sum(suddenness_distance_steer) / len(suddenness_distance_steer)
    experiment_metrics['suddenness_distance_brake_command'] = sum(suddenness_distance_brake_command) / len(suddenness_distance_brake_command)
    return experiment_metrics


def get_collisions(experiment_metrics, collision_points, df_checkpoints):
    collisions_checkpoints = []
    collisions_checkpoints_different = []
    collisions_actors_different = []
    previous_collisions_checkpoints_x, previous_collisions_checkpoints_y = 0, 0
    for point in collision_points:
        idx = (df_checkpoints['Time'] - point['Time']).abs().idxmin()
        collision_point = df_checkpoints.iloc[idx]
        collisions_checkpoints.append(collision_point)
        point_1 = np.array([collision_point['pose.pose.position.x'], collision_point['pose.pose.position.y']])
        point_2 = np.array([previous_collisions_checkpoints_x, previous_collisions_checkpoints_y])
        dist = (point_2 - point_1) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        if dist > 1:
            collisions_checkpoints_different.append(collision_point)
            collisions_actors_different.append(point['other_actor_id'])
        previous_collisions_checkpoints_x, previous_collisions_checkpoints_y = collision_point['pose.pose.position.x'], collision_point['pose.pose.position.y']

    experiment_metrics['collisions'] = len(collisions_checkpoints_different)
    experiment_metrics['collision_actor_ids'] = collisions_actors_different
    return experiment_metrics, collisions_checkpoints

def get_lane_invasions(experiment_metrics, lane_invasion_points, df_checkpoints):
    lane_invasion_checkpoints = []
    lane_invasion_checkpoints_different = []
    previous_lane_invasion_checkpoints_x, previous_lane_invasion_checkpoints_y = 0, 0
    previous_time = 0
    for point in lane_invasion_points:
        idx = (df_checkpoints['Time'] - point['Time']).abs().idxmin()
        lane_invasion_point = df_checkpoints.iloc[idx]

        lane_invasion_checkpoints.append(lane_invasion_point)
        point_1 = np.array([lane_invasion_point['pose.pose.position.x'], lane_invasion_point['pose.pose.position.y']])
        point_2 = np.array([previous_lane_invasion_checkpoints_x, previous_lane_invasion_checkpoints_y])
        dist = (point_2 - point_1) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        if dist > 1 and point['Time'] - previous_time > 0.5:
            lane_invasion_checkpoints_different.append(lane_invasion_point)
        previous_time = point['Time']
        previous_lane_invasion_checkpoints_x, previous_lane_invasion_checkpoints_y = lane_invasion_point['pose.pose.position.x'], lane_invasion_point['pose.pose.position.y']

    experiment_metrics['lane_invasions'] = len(lane_invasion_checkpoints_different)
    return experiment_metrics, lane_invasion_checkpoints


# def get_brake_points(experiment_metrics, brake_points, df_checkpoints):
#     brake_checkpoints = []
#     brake_checkpoints_different = []
#     previous_brake_checkpoints_x, previous_brake_checkpoints_y = 0, 0
#     previous_time = 0
#     for point in brake_points:
#         idx = (df_checkpoints['Time'] - point['Time']).abs().idxmin()
#         brake_point = df_checkpoints.iloc[idx]
#
#         brake_checkpoints.append(point)
#         point_1 = np.array([brake_point['pose.pose.position.x'], brake_point['pose.pose.position.y']])
#         point_2 = np.array([previous_brake_checkpoints_x, previous_brake_checkpoints_y])
#         dist = (point_2 - point_1) ** 2
#         dist = np.sum(dist, axis=0)
#         dist = np.sqrt(dist)
#         if dist > 1 and point['Time'] - previous_time > 0.5:
#             brake_checkpoints_different.append(brake_point)
#         previous_time = point['Time']
#         previous_brake_checkpoints_x, previous_brake_checkpoints_y = brake_point['pose.pose.position.x'], brake_point['pose.pose.position.y']
#
#     experiment_metrics['brakes'] = len(brake_checkpoints)
#     return experiment_metrics, brake_checkpoints


def find_closest_waypoints(checkpoints_array, map_waypoints_array, chunk_size=10000):
    N = len(checkpoints_array)
    M = len(map_waypoints_array)
    best_indices = np.zeros(N, dtype=np.int32)
    min_dists = np.full(N, np.inf)

    for start_idx in range(0, M, chunk_size):
        end_idx = min(start_idx + chunk_size, M)
        map_chunk = map_waypoints_array[start_idx:end_idx]  # (chunk_size, 3)

        # Compute distance matrix: (N, chunk_size)
        distances = np.linalg.norm(checkpoints_array[:, None, :] - map_chunk[None, :, :], axis=2)

        # Find where current distances are better
        closer = distances < min_dists[:, None]
        new_min_dists = distances.min(axis=1)
        new_indices = distances.argmin(axis=1) + start_idx

        update_mask = closer.any(axis=1)
        min_dists[update_mask] = new_min_dists[update_mask]
        best_indices[update_mask] = new_indices[update_mask]

    return best_indices


def get_position_deviation_and_effective_completed_distance(experiment_metrics, checkpoints, map_waypoints, experiment_metrics_filename, speedometer, collision_points, lane_invasion_checkpoints, brake_checkpoints, checkpoints_steer, waypoints_info, right):
    map_waypoints_tuples = []
    map_waypoints_tuples_x = []
    map_waypoints_tuples_y = []
    for waypoint in map_waypoints:
        if (experiment_metrics['carla_map'] == 'Carla/Maps/Town04'
                or experiment_metrics['carla_map'] == 'Carla/Maps/Town04_Opt'
        ):
            map_waypoints_tuples_x.append(-waypoint.transform.location.x)
            map_waypoints_tuples_y.append(waypoint.transform.location.y)
            map_waypoints_tuples.append((-waypoint.transform.location.x, waypoint.transform.location.y))
        elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town06'
                or experiment_metrics['carla_map'] == 'Carla/Maps/Town06_Opt'
                or experiment_metrics['carla_map'] == 'Carla/Maps/Town10HD'):
            map_waypoints_tuples_x.append(waypoint.transform.location.x)
            map_waypoints_tuples_y.append(-waypoint.transform.location.y)
            map_waypoints_tuples.append((waypoint.transform.location.x, -waypoint.transform.location.y))
        else:
            map_waypoints_tuples_x.append(waypoint.transform.location.x)
            map_waypoints_tuples_y.append(waypoint.transform.location.y)
            map_waypoints_tuples.append((waypoint.transform.location.x, waypoint.transform.location.y))

    logger.info("got waypoints")
    checkpoints_tuples = []
    checkpoints_tuples_x = []
    checkpoints_tuples_y = []
    checkpoints_speeds = []
    for i, point in enumerate(checkpoints):
        current_checkpoint = np.array([point['pose.pose.position.x'], point['pose.pose.position.y'], speedometer[i]['data']*3.6])
        if (experiment_metrics['carla_map'] == 'Carla/Maps/Town01'
                or experiment_metrics['carla_map'] == 'Carla/Maps/Town01_Opt'):
            checkpoint_x = (max(map_waypoints_tuples_x) + min(map_waypoints_tuples_x))-current_checkpoint[0]
            checkpoint_y = -point['pose.pose.position.y']
        elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town03'
              or experiment_metrics['carla_map'] == 'Carla/Maps/Town07'
              or experiment_metrics['carla_map'] == 'Carla/Maps/Town03_Opt'
              or experiment_metrics['carla_map'] == 'Carla/Maps/Town07_Opt'
              or experiment_metrics['carla_map'] == 'Carla/Maps/Town05'
              or experiment_metrics['carla_map'] == 'Carla/Maps/Town05_Opt'
              or experiment_metrics['carla_map'] == 'Carla/Maps/Town02'
              or experiment_metrics['carla_map'] == 'Carla/Maps/Town02_Opt'
            ):
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]
        elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town04' or experiment_metrics['carla_map'] == 'Carla/Maps/Town04_Opt'):
            checkpoint_x = -current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]
        else:
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = current_checkpoint[1]
        checkpoints_tuples_x.append(checkpoint_x)
        checkpoints_tuples_y.append(checkpoint_y)
        checkpoints_speeds.append(current_checkpoint[2])
        checkpoints_tuples.append((checkpoint_x, checkpoint_y, current_checkpoint[2]))

    logger.info("got checkpoints")

    checkpoints_array = np.array(checkpoints_tuples)  # Shape: (num_checkpoints, 2 or 3)
    checkpoints_array = checkpoints_array[:, :2]
    map_waypoints_array = np.array(map_waypoints_tuples)  # Shape: (num_waypoints, 2 or 3)

    num_checkpoints = checkpoints_array.shape[0]
    num_waypoints = map_waypoints_array.shape[0]

    best_checkpoint_points = np.zeros_like(checkpoints_array)  # Closest waypoints
    min_dists = np.full(num_checkpoints, np.inf)  # Start with large distances

    # List to store covered checkpoints
    covered_checkpoints = []

    # Process waypoints in chunks
    for start_idx in range(0, num_waypoints, 10000):
        end_idx = min(start_idx + 10000, num_waypoints)
        map_chunk = map_waypoints_array[start_idx:end_idx]  # Extract chunk of waypoints

        # Compute distances for the current chunk
        distances_chunk = np.linalg.norm(
            checkpoints_array[:, np.newaxis, :] - map_chunk[np.newaxis, :, :], axis=2
        )

        # Update the minimum distances and corresponding best checkpoint points
        is_closer = distances_chunk < min_dists[:, np.newaxis]
        min_dists = np.where(is_closer.any(axis=1), distances_chunk.min(axis=1), min_dists)
        best_indices_chunk = is_closer.argmax(axis=1)

        for i, closer in enumerate(is_closer.any(axis=1)):
            if closer:
                best_checkpoint_points[i] = map_chunk[best_indices_chunk[i]]

    # Separate x, y coordinates for final output
    best_checkpoint_points_x = best_checkpoint_points[:, 0].tolist()
    best_checkpoint_points_y = best_checkpoint_points[:, 1].tolist()

    # Filter covered checkpoints
    for i, dist in enumerate(min_dists):
        best_x, best_y = best_checkpoint_points[i, :2]
        if dist < 1:  # Check distance condition
            if not covered_checkpoints or (
                    covered_checkpoints[-1][0] != best_x or covered_checkpoints[-1][1] != best_y):
                covered_checkpoints.append((best_x, best_y))

    # min_dists = []
    # best_checkpoint_points_x = []
    # best_checkpoint_points_y = []
    # covered_checkpoints = []
    # for error_counter, checkpoint in enumerate(checkpoints_tuples):
    #     min_dist = 100
    #     for x, perfect_checkpoint in enumerate(map_waypoints_tuples):
    #         point_1 = np.array([checkpoint[0], checkpoint[1]])
    #         point_2 = np.array([perfect_checkpoint[0], perfect_checkpoint[1]])
    #         dist = (point_2 - point_1) ** 2
    #         dist = np.sum(dist, axis=0)
    #         dist = np.sqrt(dist)
    #         if dist < min_dist:
    #             min_dist = dist
    #             best_checkpoint = x
    #             best_checkpoint_point_x = point_2[0]
    #             best_checkpoint_point_y = point_2[1]
    #     best_checkpoint_points_x.append(best_checkpoint_point_x)
    #     best_checkpoint_points_y.append(best_checkpoint_point_y)
    #     if min_dist < 100:
    #         min_dists.append(min_dist)
    #         if len(covered_checkpoints) == 0 or (
    #                 len(covered_checkpoints) > 0 and covered_checkpoints[len(covered_checkpoints) - 1][
    #             0] != best_checkpoint_point_x and covered_checkpoints[len(covered_checkpoints) - 1][
    #                     1] != best_checkpoint_point_y):
    #             if min_dist < 1:
    #                 covered_checkpoints.append((best_checkpoint_point_x, best_checkpoint_point_y))

    experiment_metrics['effective_completed_distance'] = len(covered_checkpoints)*0.5
    experiment_metrics['position_deviation_mean'] = sum(min_dists) / len(min_dists)
    experiment_metrics['position_deviation_total_err'] = sum(min_dists)
    experiment_metrics['position_deviation_mean_per_km'] = experiment_metrics['position_deviation_mean'] / (experiment_metrics['effective_completed_distance']/1000)
    starting_point_map = (checkpoints_tuples_x[0], checkpoints_tuples_y[0])
    experiment_metrics['starting_point_map'] = starting_point_map
    if experiment_metrics['collisions'] > 0:
        experiment_metrics['collisions_per_km'] = experiment_metrics['collisions'] / (experiment_metrics['effective_completed_distance']/1000)
    else:
        experiment_metrics['collisions_per_km'] = 0
    if experiment_metrics['lane_invasions'] > 0:
        experiment_metrics['lane_invasions_per_km'] = experiment_metrics['lane_invasions'] / (experiment_metrics['effective_completed_distance']/1000)
    else:
        experiment_metrics['lane_invasions_per_km'] = 0
    experiment_metrics['suddenness_distance_control_command_per_km'] = experiment_metrics['suddenness_distance_control_commands'] / (experiment_metrics['effective_completed_distance']/1000)
    experiment_metrics['suddenness_distance_throttle_per_km'] = experiment_metrics['suddenness_distance_throttle'] / (experiment_metrics['effective_completed_distance']/1000)
    experiment_metrics['suddenness_distance_steer_per_km'] = experiment_metrics['suddenness_distance_steer'] / (experiment_metrics['effective_completed_distance']/1000)
    experiment_metrics['suddenness_distance_brake_command_per_km'] = experiment_metrics['suddenness_distance_brake_command'] / (experiment_metrics['effective_completed_distance']/1000)
    experiment_metrics['suddenness_distance_speed_per_km'] = experiment_metrics['suddenness_distance_speed'] / (experiment_metrics['effective_completed_distance']/1000)
    experiment_metrics['position_deviations'] = min_dists.tolist()

    logger.info("creating maps")
    create_experiment_maps(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, best_checkpoint_points_x, best_checkpoint_points_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, collision_points, lane_invasion_checkpoints, brake_checkpoints, min_dists, checkpoints_steer)
    logger.info("creating histograms")
    create_speed_histograms_plot(experiment_metrics, experiment_metrics_filename, "deviation", min_dists)
    create_speed_histograms_plot(experiment_metrics, experiment_metrics_filename, "speed", checkpoints_speeds)
    logger.info("creating speed line")
    create_speed_line_plot(experiment_metrics, experiment_metrics_filename, checkpoints_speeds, checkpoints, waypoints_info, right)
    return experiment_metrics


def create_speed_histograms_plot(experiment_metrics, experiment_metrics_filename, metric_name, checkpoints):
    city = experiment_metrics['carla_map'].split('/')[-1]
    metricname = f"{metric_name}_histogram"
    fig, ax = plt.subplots(figsize=(10, 5))
    title = f"{metric_name} Histogram ({experiment_metrics['experiment_model']} - {city})"
    ax.set_title(title)

    plot_tensorboard_perc_histogram.plot_histogram_from_metric_lists(
        ax,
        metricname,
        title,
        comp_1={'tag': experiment_metrics['experiment_model'], 'metrics': checkpoints}
        # comp_2={'tag': '', 'metrics': {'speeds': []}},  # Empty placeholders
        # comp_3={'tag': '', 'metrics': {'speeds': []}},
    )

    fig.tight_layout()
    filename = f"{experiment_metrics_filename}_{metric_name}_histogram.png"
    fig.savefig(filename)
    plt.close()

def create_speed_line_plot(experiment_metrics,
                           experiment_metrics_filename,
                           checkpoints_speeds,
                           checkpoints,
                           waypoints_info,
                           right):
    """
        Assign speeds to the nearest waypoint and determine direction (right/left)
        """
    city = experiment_metrics['carla_map'].split('/')[-1]
    previous_speed = 0
    speedometer_points_sum = 0
    suddenness_distance_speeds = []
    speed_points = []

    speeds_by_waypoint = []
    waypoint_order = []

    for i, point in enumerate(checkpoints_speeds):
        speed_point = point  # km/h
        speedometer_points_sum += speed_point
        suddenness_distance_speed = abs(speed_point - previous_speed)
        suddenness_distance_speeds.append(suddenness_distance_speed)
        previous_speed = speed_point
        speed_points.append(speed_point)

        agent_location = (
            checkpoints[i]['pose.pose.position.x'],
            - checkpoints[i]['pose.pose.position.y']
        # Note that ros is publishing y in opposite sign than Carla transfomrs
        )

        # Find closest waypoint (brute force)
        distances = [
            math.sqrt((wp['transform'].location.x - agent_location[0]) ** 2 +
                      (wp['transform'].location.y - agent_location[1]) ** 2)
            for wp in waypoints_info
        ]
        closest_idx = int(np.argmin(distances))
        speeds_by_waypoint.append(speed_point)
        waypoint_order.append(closest_idx)

    waypoints_info_colors = [wp['color'] for wp in waypoints_info]
    waypoint_colors = [dict_to_color(color) for color in waypoints_info_colors]

    plot_sparse_line(
        y_values=speeds_by_waypoint,
        waypoint_order=waypoint_order,
        waypoint_colors=waypoint_colors,
        direction_right=right,
        title=f"speeds progress ({experiment_metrics['experiment_model']} - {city})",
        x_label='position index',
        y_label='speeds',
        save_path=f"{experiment_metrics_filename}_lineplot.png"
    )

    return experiment_metrics

def create_experiment_maps(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, best_checkpoint_points_x, best_checkpoint_points_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, collision_points, lane_invasion_checkpoints, brake_checkpoints, min_dists, checkpoints_steer):
    difference_x = 0
    difference_y = 0
    starting_point_landmark = 0
    while difference_x < 1 and difference_y < 1 and starting_point_landmark < len(checkpoints_tuples_x)-1:
        difference_x = abs(checkpoints_tuples_x[starting_point_landmark] - checkpoints_tuples_x[0])
        difference_y = abs(checkpoints_tuples_y[starting_point_landmark] - checkpoints_tuples_y[0])
        if difference_x < 1 and difference_y < 1:
            starting_point_landmark += 1

    difference_x = 0
    difference_y = 0
    finish_point_landmark = len(checkpoints_tuples_x)-1
    while difference_x < 1 and difference_y < 1 and finish_point_landmark > 0:
        difference_x = abs(checkpoints_tuples_x[finish_point_landmark] - checkpoints_tuples_x[len(checkpoints_tuples_x)-1])
        difference_y = abs(checkpoints_tuples_y[finish_point_landmark] - checkpoints_tuples_y[len(checkpoints_tuples_x)-1])
        if difference_x < 1 and difference_y < 1:
            finish_point_landmark -= 1

    # create_brake_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y,
    #                  checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark,
    #                  finish_point_landmark, collision_points, brake_checkpoints)
    # create_rewards_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y,
    #                  checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark,
    #                  finish_point_landmark, collision_points, brake_checkpoints, min_dists, checkpoints_steer)
    # create_steer_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y,
    #                  checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark,
    #                  finish_point_landmark, collision_points, brake_checkpoints, min_dists, checkpoints_steer)
    # logger.info("creating dist map")
    # create_dist_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y,
    #                  checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark,
    #                  finish_point_landmark, collision_points, brake_checkpoints, min_dists, checkpoints_steer)
    # create_speed_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y,
    #                 checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark,
    #                 finish_point_landmark, collision_points, brake_checkpoints, min_dists, checkpoints_steer)
    # create_debugging_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark)
    if experiment_metrics['collisions'] > 0:
        logger.info("creating collisions map")
        create_collisions_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark, collision_points)
    if experiment_metrics['lane_invasions'] > 0:
        logger.info("creating lane invasions map")
        create_lane_invasions_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark, lane_invasion_checkpoints)


def convert_ros_checkpoint_to_carla_checkpoint(experiment_metrics, current_checkpoint):
    if (experiment_metrics['carla_map'] == 'Carla/Maps/Town01' or experiment_metrics[
        'carla_map'] == 'Carla/Maps/Town02' or \
            experiment_metrics['carla_map'] == 'Carla/Maps/Town01_Opt' or experiment_metrics[
                'carla_map'] == 'Carla/Maps/Town02_Opt'):
        # (max(map_waypoints_tuples_x) + min(map_waypoints_tuples_x)) - current_checkpoint[0]
        checkpoint_x = current_checkpoint[0]
        checkpoint_y = -current_checkpoint[1]
    elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town03' or experiment_metrics[
        'carla_map'] == 'Carla/Maps/Town07' or \
          experiment_metrics['carla_map'] == 'Carla/Maps/Town03_Opt' or experiment_metrics[
              'carla_map'] == 'Carla/Maps/Town07_Opt'):
        checkpoint_x = current_checkpoint[0]
        checkpoint_y = -current_checkpoint[1]
    elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town04' or experiment_metrics[
        'carla_map'] == 'Carla/Maps/Town04_Opt'):
        checkpoint_x = -current_checkpoint[0]
        checkpoint_y = -current_checkpoint[1]
    else:
        checkpoint_x = current_checkpoint[0]
        checkpoint_y = current_checkpoint[1]
    return checkpoint_x, checkpoint_y


def create_debugging_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark):
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()
    colors = ["#00FF00", "#FF0000", "#000000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='b', marker="s", label='Map waypoints')
    # ax.scatter(best_checkpoint_points_x, best_checkpoint_points_y, s=10, c='g', marker="o", label='Map waypoints for position deviation')
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c=checkpoints_speeds, cmap='hot_r', marker="o",
                      label='Experiment waypoints', vmin=0,
                      vmax=(experiment_metrics['max_speed'] if experiment_metrics['max_speed'] > 30 else 30))
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0],
               label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark], checkpoints_tuples_y[starting_point_landmark], s=100,
               marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x) - 1], checkpoints_tuples_y[len(checkpoints_tuples_x) - 1],
               s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark], checkpoints_tuples_y[finish_point_landmark], s=100,
               marker="o", color=colors[2])
    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})

    full_text = ''
    for key, value in experiment_metrics.items():
        # print(key, value)
        full_text += ' * ' + str(key) + ' : ' + str(value) + '\n'
    plt.figtext(0.1, 0.01, full_text, wrap=True, horizontalalignment='left', fontsize=PLOTS_FONTSIZE)

    plt.grid(True)
    plt.subplots_adjust(bottom=0.4)
    plt.title(experiment_metrics['experiment_model'], fontsize=PLOTS_FONTSIZE)
    fig.savefig(experiment_metrics_filename + 'debug.png', dpi=fig.dpi)


def create_collisions_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark, collision_points):
    collision_checkpoints_tuples_x = []
    collision_checkpoints_tuples_y = []
    for i, point in enumerate(collision_points):
        current_checkpoint = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        checkpoint_x, checkpoint_y = convert_ros_checkpoint_to_carla_checkpoint(experiment_metrics, current_checkpoint)
        collision_checkpoints_tuples_x.append(checkpoint_x)
        collision_checkpoints_tuples_y.append(checkpoint_y)


    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#00FF00", "#FF0000", "#000000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='b', marker="s", label='Map waypoints')
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c=checkpoints_speeds, cmap='hot_r', marker="o", label='Experiment waypoints', vmin=0, vmax=(experiment_metrics['max_speed'] if experiment_metrics['max_speed']>30 else 30))
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark], checkpoints_tuples_y[starting_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x)-1], checkpoints_tuples_y[len(checkpoints_tuples_x)-1], s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark], checkpoints_tuples_y[finish_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(collision_checkpoints_tuples_x, collision_checkpoints_tuples_y, s=200, marker="o", color='y', label='Collisions')

    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})
    plt.grid(True)
    plt.title(experiment_metrics['experiment_model'] + ' Collisions', fontsize=PLOTS_FONTSIZE)
    fig.savefig(experiment_metrics_filename + '_collisions.png', dpi=fig.dpi)

def create_speed_map(experiment_metrics, experiment_metrics_filename,
                     map_waypoints_tuples_x, map_waypoints_tuples_y,
                     checkpoints_tuples_x, checkpoints_tuples_y,
                     checkpoints_speeds,
                     starting_point_landmark, finish_point_landmark):

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()
    colors = ["#00FF00", "#FF0000", "#000000"]

    # Plot base map waypoints
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y,
               s=10, c='#ADD8E6', marker="s", label='Map waypoints')

    # Plot experiment waypoints colored by speed
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y,
                      s=10, c=checkpoints_speeds, cmap='cool', marker="o",
                      label='Experiment waypoints', vmin=0)

    # Mark key points
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0],
               s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark],
               checkpoints_tuples_y[starting_point_landmark],
               s=100, marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[-1], checkpoints_tuples_y[-1],
               s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark],
               checkpoints_tuples_y[finish_point_landmark],
               s=100, marker="o", color=colors[2])

    # Colorbar and styling
    fig.colorbar(plot, shrink=0.5, label="Speed (m/s)")
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})
    plt.grid(True)
    plt.title(experiment_metrics['experiment_model'] + ' Speeds', fontsize=PLOTS_FONTSIZE)

    # Save the figure
    fig.savefig(experiment_metrics_filename + '_speeds.png', dpi=fig.dpi)

def create_dist_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark, lane_invasion_checkpoints, brake_checkpoints, min_dists, checkpoints_steer):
    brake_checkpoints_tuples_x = []
    brake_checkpoints_tuples_y = []

    x = len(checkpoints_tuples_y) - len(brake_checkpoints)

    if x > 0:
        checkpoints_tuples_x = checkpoints_tuples_x[:-x]
        checkpoints_tuples_y = checkpoints_tuples_y[:-x]
    elif x < 0:
        min_dists = min_dists[:-x]

    min_len = min(len(checkpoints_tuples_x), len(checkpoints_tuples_y), len(min_dists))
    checkpoints_tuples_x = checkpoints_tuples_x[:min_len]
    checkpoints_tuples_y = checkpoints_tuples_y[:min_len]
    min_dists = min_dists[:min_len]

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#00FF00", "#FF0000", "#000000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='#ADD8E6', marker="s", label='Map waypoints')
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c=min_dists, cmap='hot_r', marker="o", label='Experiment waypoints', vmin=-0.2, vmax=0.2)
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark], checkpoints_tuples_y[starting_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x)-1], checkpoints_tuples_y[len(checkpoints_tuples_x)-1], s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark], checkpoints_tuples_y[finish_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(brake_checkpoints_tuples_x, brake_checkpoints_tuples_y, s=200, marker="o", color='y', label='Steer')

    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})
    plt.grid(True)
    plt.title(experiment_metrics['experiment_model'] + ' Dists', fontsize=PLOTS_FONTSIZE)
    fig.savefig(experiment_metrics_filename + '_dists.png', dpi=fig.dpi)


def create_steer_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark, lane_invasion_checkpoints, brake_checkpoints, min_dists, checkpoints_steer):
    brake_checkpoints_tuples_x = []
    brake_checkpoints_tuples_y = []

    x = len(checkpoints_tuples_y) - len(brake_checkpoints)

    if x > 0:
        checkpoints_tuples_x = checkpoints_tuples_x[:-x]
        checkpoints_tuples_y = checkpoints_tuples_y[:-x]
    elif x < 0:
        checkpoints_steer = checkpoints_steer[:-x]

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#00FF00", "#FF0000", "#000000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='#ADD8E6', marker="s", label='Map waypoints')
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c=checkpoints_steer, cmap='hot_r', marker="o", label='Experiment waypoints', vmin=-0.2, vmax=0.2)
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark], checkpoints_tuples_y[starting_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x)-1], checkpoints_tuples_y[len(checkpoints_tuples_x)-1], s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark], checkpoints_tuples_y[finish_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(brake_checkpoints_tuples_x, brake_checkpoints_tuples_y, s=200, marker="o", color='y', label='Steer')

    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})
    plt.grid(True)
    plt.title(experiment_metrics['experiment_model'] + ' Steer', fontsize=PLOTS_FONTSIZE)
    fig.savefig(experiment_metrics_filename + '_steer.png', dpi=fig.dpi)

def create_rewards_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark, lane_invasion_checkpoints, brake_checkpoints, min_dists, checkpoints_steer):
    brake_checkpoints_tuples_x = []
    brake_checkpoints_tuples_y = []

    x = len(checkpoints_tuples_y) - len(brake_checkpoints)

    if x > 0:
        checkpoints_tuples_x = checkpoints_tuples_x[:-x]
        checkpoints_tuples_y = checkpoints_tuples_y[:-x]
    elif x < 0:
        checkpoints_speeds = checkpoints_speeds[:-x]
        min_dists = min_dists[:-x]
        checkpoints_steer = checkpoints_steer[:-x]

    d_rewards = np.power(1 - min_dists, 1)  # Element-wise power operation

    checkpoints_speeds = np.array(checkpoints_speeds)  # Ensure this is a NumPy array
    d_rewards = np.array(d_rewards)  # Ensure d_rewards is a NumPy array
    checkpoints_steer = np.array(checkpoints_steer)

    # Calculate rewards
    rewards = np.log(checkpoints_speeds) * np.power(d_rewards, (checkpoints_speeds / 5) + 1)
    # rewards -= 5 * checkpoints_steer

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#00FF00", "#FF0000", "#000000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='#ADD8E6', marker="s", label='Map waypoints')
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c=rewards, cmap='hot_r', marker="o", label='Experiment waypoints', vmin=0, vmax=3)
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark], checkpoints_tuples_y[starting_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x)-1], checkpoints_tuples_y[len(checkpoints_tuples_x)-1], s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark], checkpoints_tuples_y[finish_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(brake_checkpoints_tuples_x, brake_checkpoints_tuples_y, s=200, marker="o", color='y', label='Rewards')

    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})
    plt.grid(True)
    plt.title(experiment_metrics['experiment_model'] + ' Rewards', fontsize=PLOTS_FONTSIZE)
    fig.savefig(experiment_metrics_filename + '_rewards.png', dpi=fig.dpi)

def create_brake_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark, lane_invasion_checkpoints, brake_checkpoints):
    brake_checkpoints_tuples_x = []
    brake_checkpoints_tuples_y = []

    x = len(checkpoints_tuples_y) - len(brake_checkpoints)

    if x > 0:
        checkpoints_tuples_x = checkpoints_tuples_x[:-x]
        checkpoints_tuples_y = checkpoints_tuples_y[:-x]
    elif x < 0:
        brake_checkpoints = brake_checkpoints[:-x]


    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#00FF00", "#FF0000", "#000000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='#ADD8E6', marker="s", label='Map waypoints')
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c=brake_checkpoints, cmap='hot_r', marker="o", label='Experiment waypoints', vmin=-1, vmax=1)
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark], checkpoints_tuples_y[starting_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x)-1], checkpoints_tuples_y[len(checkpoints_tuples_x)-1], s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark], checkpoints_tuples_y[finish_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(brake_checkpoints_tuples_x, brake_checkpoints_tuples_y, s=200, marker="o", color='y', label='action')

    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})
    plt.grid(True)
    plt.title(experiment_metrics['experiment_model'] + ' Action', fontsize=PLOTS_FONTSIZE)
    fig.savefig(experiment_metrics_filename + '_speed_action.png', dpi=fig.dpi)

def create_lane_invasions_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds, starting_point_landmark, finish_point_landmark, lane_invasion_checkpoints):
    lane_invasion_checkpoints_tuples_x = []
    lane_invasion_checkpoints_tuples_y = []
    for i, point in enumerate(lane_invasion_checkpoints):
        current_checkpoint = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        if (experiment_metrics['carla_map'] == 'Carla/Maps/Town01' or experiment_metrics['carla_map'] == 'Carla/Maps/Town02' or \
            experiment_metrics['carla_map'] == 'Carla/Maps/Town01_Opt' or experiment_metrics['carla_map'] == 'Carla/Maps/Town02_Opt'):
            checkpoint_x = (max(map_waypoints_tuples_x) + min(map_waypoints_tuples_x))-current_checkpoint[0]
            checkpoint_y = -point['pose.pose.position.y']
        elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town03' or experiment_metrics['carla_map'] == 'Carla/Maps/Town07' or \
            experiment_metrics['carla_map'] == 'Carla/Maps/Town03_Opt' or experiment_metrics['carla_map'] == 'Carla/Maps/Town07_Opt'):
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]
        elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town04' or experiment_metrics['carla_map'] == 'Carla/Maps/Town04_Opt'):
            checkpoint_x = -current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]
        else:
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = current_checkpoint[1]
        lane_invasion_checkpoints_tuples_x.append(checkpoint_x)
        lane_invasion_checkpoints_tuples_y.append(checkpoint_y)


    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#00FF00", "#FF0000", "#000000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='b', marker="s", label='Map waypoints')
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c=checkpoints_speeds, cmap='hot_r', marker="o", label='Experiment waypoints', vmin=0, vmax=(experiment_metrics['max_speed'] if experiment_metrics['max_speed']>30 else 30))
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark], checkpoints_tuples_y[starting_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x)-1], checkpoints_tuples_y[len(checkpoints_tuples_x)-1], s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark], checkpoints_tuples_y[finish_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(lane_invasion_checkpoints_tuples_x, lane_invasion_checkpoints_tuples_y, s=200, marker="o", color='y', label='Lane invasions')

    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})
    plt.grid(True)
    plt.title(experiment_metrics['experiment_model'] + ' Lane invasions', fontsize=PLOTS_FONTSIZE)
    fig.savefig(experiment_metrics_filename + '_lane_invasion.png', dpi=fig.dpi)


def get_tensorboard_comparisons(experiments_starting_time, base_dir='./logs/Tensorboard'):
    comparisons = []

    # DDPG, SAC, PPO subdirs
    for algo in ['ddpg', 'sac', 'ppo']:
        algo_path = os.path.join(base_dir, algo)
        if not os.path.exists(algo_path):
            continue

        valid_log_dirs = []

        # Walk only one level deep
        for item in os.listdir(algo_path):
            full_path = os.path.join(algo_path, item)

            # Check if it's a valid timestamped directory
            if os.path.isdir(full_path) and re.match(r'^\d{8}-\d{6}$', item):
                # Check modification time
                if os.stat(full_path).st_mtime > experiments_starting_time:
                    # Calculate total directory size
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(full_path):
                        for f in filenames:
                            try:
                                fp = os.path.join(dirpath, f)
                                total_size += os.path.getsize(fp)
                            except FileNotFoundError:
                                continue

                    # Filter: skip folders smaller than 500 bytes
                    if total_size >= 500:
                        valid_log_dirs.append(os.path.abspath(full_path))
                    else:
                        print(f"Skipping small folder: {full_path} ({total_size} bytes)")

        if valid_log_dirs:
            comparisons.append({
                'log_dir': sorted(valid_log_dirs),
                'tag': algo.upper()
            })

    return comparisons

def get_aggregated_experiments_list(experiments_starting_time):
    current_experiment_folders = []
    root = './'
    folders = list(os.walk(root))[1:]
    for folder in folders:
        if len(folder[0].split('/')) == 2 and folder[2] and experiments_starting_time < os.stat(folder[0]).st_mtime and re.search("\./[0-9]+-[0-9]+", folder[0]):
            current_experiment_folders.append(folder)
    current_experiment_folders.sort()

    dataframes = []
    for folder in current_experiment_folders:
        try:
            r = re.compile(".*\.json")
            json_list = list(filter(r.match, folder[2])) # Read Note below
            df = pd.read_json(folder[0] + '/' + json_list[0], orient='index').T
            dataframes.append(df)
        except:
            print('Broken experiment: ' + folder[0])
            shutil.rmtree(folder[0])

    result = pd.concat(dataframes)
    result.index = result['experiment_model'].values.tolist()
    result.loc[result['collisions'] > 0, 'position_deviation_mean'] = float("nan")
    result.loc[result['collisions'] > 0, 'effective_completed_distance'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_control_commands'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_throttle'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_steer'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_brake_command'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_control_command_per_km'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_throttle_per_km'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_steer_per_km'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_brake_command_per_km'] = float("nan")
    result.loc[result['collisions'] > 0, 'completed_distance'] = float("nan")
    result.loc[result['collisions'] > 0, 'average_speed'] = float("nan")
    result.loc[result['collisions'] > 0, 'position_deviation_mean'] = float("nan")
    result.loc[result['collisions'] > 0, 'position_deviation_mean_per_km'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_speed'] = float("nan")
    result.loc[result['collisions'] > 0, 'suddenness_distance_speed_per_km'] = float("nan")

    return result

def get_maps_colors():
    maps_colors = {
        'Carla/Maps/Town01': 'red',
        'Carla/Maps/Town02': 'green',
        'Carla/Maps/Town02_Opt': 'green',
        'Carla/Maps/Town03': 'blue',
        'Carla/Maps/Town04': 'grey',
        'Carla/Maps/Town05': 'black',
        'Carla/Maps/Town06': 'pink',
        'Carla/Maps/Town07': 'orange',
        'Carla/Maps/Town10HD': 'yellow',
    }
    return maps_colors

# def get_model_colors():
#     model_colors = {
#         'ppo': 'black',
#         'sac': 'darkred',
#         'ddpg': 'darkblue',
#         'ddpg_classic': 'darkblue',
#         'ddpg_normalize': 'white',
#         'ddpg_not_normalize': 'purple',
#     }
#     return model_colors

def get_color_handles():
    red_patch = mpatches.Patch(color='red', label='Map01')
    green_patch = mpatches.Patch(color='green', label='Map02')
    blue_patch = mpatches.Patch(color='blue',  label='Map03')
    grey_patch = mpatches.Patch(color='grey',  label='Map04')
    black_patch = mpatches.Patch(color='black',  label='Map05')
    pink_patch = mpatches.Patch(color='pink',  label='Map06')
    orange_patch = mpatches.Patch(color='orange',  label='Map07')
    yellow_patch = mpatches.Patch(color='yellow',  label='Map10HD')
    color_handles = [red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch, yellow_patch]

    return color_handles

def get_all_experiments_aggregated_metrics(result, experiments_starting_time_str, experiments_metrics_and_titles):
    maps_colors = get_maps_colors()
    color_handles = get_color_handles()
    models = result['experiment_model'].unique()

    for experiment_metric_and_title in experiments_metrics_and_titles:
        metric_name = experiment_metric_and_title['metric']
        if 'histogram_' in metric_name or 'line_' in metric_name:
            continue

        fig, axes = plt.subplots(1, len(models), figsize=(40, 10), sharey=True)
        if len(models) == 1:
            axes = [axes]  # ensure iterable

        for ax, model in zip(axes, models):
            model_data = result[result['experiment_model'] == model]
            bar_values = model_data[metric_name]
            maps = model_data['carla_map']

            for i, (height, map_name) in enumerate(zip(bar_values, maps)):
                ax.bar(
                    i,
                    height,
                    color=maps_colors[map_name],
                    linewidth=0
                )

            ax.set_title(model, fontsize=PLOTS_FONTSIZE)
            ax.set_xticks([])
            labels = ['c' if clockwise else 'ac' for clockwise in model_data['clockwise']]  # list of 'r' or 'l' for each bar
            ax.set_xticks(range(len(labels)))  # numeric positions
            ax.set_xticklabels(labels, rotation=0, ha='clockwise')
            ax.tick_params(axis='y', labelsize=40)
            ax.tick_params(axis='x', labelsize=40)

        # Add figure title
        fig.suptitle(experiment_metric_and_title['title'], fontsize=PLOTS_FONTSIZE, y=1.02)

        # Adjust layout to make space for the legend below the subplots
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Add global legend below the subplots
        fig.legend(handles=color_handles, loc='lower center', ncol=len(color_handles), fontsize=20, bbox_to_anchor=(0.5, 0.01))

        plt.savefig(f"{experiments_starting_time_str}/{metric_name}.png", bbox_inches='tight')
        plt.close()

def plot_sparse_line(y_values, waypoint_order, direction_right, waypoint_colors, title, x_label, y_label, save_path, max_gap=50):
    """
    y_values: list of speeds in order (duplicates allowed)
    waypoint_order: list of waypoint indices in the same order as y_values
    max_gap: maximum allowed jump between consecutive waypoints before breaking the line
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    # --- Break line into segments when gap is too large ---
    segments_x = []
    segments_y = []
    current_x = [waypoint_order[0]]
    current_y = [y_values[0]]

    for prev_wp, prev_y, wp, y in zip(waypoint_order[:-1], y_values[:-1], waypoint_order[1:], y_values[1:]):
        if abs(wp - prev_wp) <= max_gap:  # "consecutive enough"
            current_x.append(wp)
            current_y.append(y)
        else:
            # Save current segment and start new one
            segments_x.append(current_x)
            segments_y.append(current_y)
            current_x = [wp]
            current_y = [y]
    # Add last segment
    segments_x.append(current_x)
    segments_y.append(current_y)

    # Plot each segment as a separate line
    for seg_x, seg_y in zip(segments_x, segments_y):
        ax.plot(seg_x, seg_y, linewidth=1.5, color="blue")

    # Start and End markers
    if waypoint_order:
        if direction_right:
            ax.scatter(waypoint_order[:10], y_values[:10],
                       color="green", s=10, label="Start", zorder=3)
            ax.scatter(waypoint_order[-10:], y_values[-10:],
                       color="red", s=10, label="End", zorder=3)
        else:
            ax.scatter(waypoint_order[-10:], y_values[-10:],
                       color="green", s=10, label="Start", zorder=3)
            ax.scatter(waypoint_order[:10], y_values[:10],
                       color="red", s=10, label="End", zorder=3)

    # --- Sector strip above plot ---
    ax_activity = ax.inset_axes([0.0, 1.01, 1.0, 0.05])  # strip above plot

    total = max(waypoint_order) + 1 if waypoint_order else 0
    rgb_array = np.zeros((1, total, 3), dtype=np.float32)

    for i in range(total):
        if i < len(waypoint_colors):
            c = waypoint_colors[i]
        else:
            c = carla.Color(128, 128, 128)  # fallback gray
        rgb_array[0, i] = [c.r / 255.0, c.g / 255.0, c.b / 255.0]

    ax_activity.imshow(rgb_array, aspect="auto", extent=[0, total, 0, 1])
    ax_activity.set_xticks([])
    ax_activity.set_yticks([])
    ax_activity.set_xlim(ax.get_xlim())

    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()

def dict_to_color(data: dict) -> carla.Color:
    return carla.Color(r=data['r'], g=data['g'], b=data['b'])

def get_per_model_aggregated_metrics(result, experiments_starting_time_str, experiments_metrics_and_titles):
    maps_colors = get_maps_colors()
    color_handles = get_color_handles()
    unique_experiment_models = result['experiment_model'].unique()

    for unique_experiment_model in unique_experiment_models:
        unique_model_experiments = result.loc[result['experiment_model'].eq(unique_experiment_model)]
        colors = [maps_colors[i] for i in unique_model_experiments['carla_map']]

        for experiment_metric_and_title in experiments_metrics_and_titles:
            metric = experiment_metric_and_title['metric']

            if metric.startswith('line_'):
                # Plot per model-town line plot
                for idx, row in unique_model_experiments.iterrows():
                    city = row['carla_map'].split('/')[-1]
                    metricname = metric.replace('line_', '')
                    y_values = row[metricname]

                    if not isinstance(y_values, (list, np.ndarray)):
                        continue
                    waypoint_colors = [dict_to_color(color) for color in row['waypoints_info_colors']]
                    plot_sparse_line(
                        y_values=row[metricname],
                        waypoint_order=row['waypoint_order'],
                        waypoint_colors=waypoint_colors,
                        direction_right= row['clockwise'],
                        title=f"{experiment_metric_and_title['title']} Progress ({unique_experiment_model} - {city})",
                        x_label=experiment_metric_and_title.get('x_axis_label', 'Index'),
                        y_label=experiment_metric_and_title.get('y_axis_label', metricname),
                        save_path=f"{experiments_starting_time_str}/{unique_experiment_model}_{city}_{metricname}_lineplot.png"
                    )

            elif 'histogram_' in metric:
                # Plot per model-town histogram
                for idx, row in unique_model_experiments.iterrows():
                    city = row['carla_map'].split('/')[-1]
                    metricname = metric.replace('histogram_', '')
                    fig, ax = plt.subplots(figsize=(10, 5))
                    title = f"{experiment_metric_and_title['title']} Histogram ({unique_experiment_model} - {city})"
                    ax.set_title(title)

                    plot_tensorboard_perc_histogram.plot_histogram_from_metric_lists(
                        ax,
                        metricname,
                        experiment_metric_and_title['title'],
                        comp_1={'tag': unique_experiment_model, 'metrics': row[metricname]},
                        # comp_2={'tag': '', 'metrics': {'speeds': []}},  # Empty placeholders
                        # comp_3={'tag': '', 'metrics': {'speeds': []}},
                        x_axis=experiment_metric_and_title.get('x_axis', None),
                        x_bins=experiment_metric_and_title.get('x_bins', 10),
                        multiplier=experiment_metric_and_title.get('multiplier', 1)
                    )

                    fig.tight_layout()
                    filename = f"{experiments_starting_time_str}/{unique_experiment_model}_{city}_{metric}.png"
                    fig.savefig(filename)
                    plt.close()
            else:
                # Normal barplot
                fig = plt.figure(figsize=(40, 10))
                ax = unique_model_experiments[metric].plot.bar(color=colors)

                # Title and labels with bigger font
                plt.title(
                    experiment_metric_and_title['title'] + ' with ' + unique_experiment_model,
                    fontsize=PLOTS_FONTSIZE
                )
                ax.set_xlabel(ax.get_xlabel(), fontsize=PLOTS_FONTSIZE)
                ax.set_ylabel(ax.get_ylabel(), fontsize=PLOTS_FONTSIZE)

                # Ticks
                plt.xticks(rotation=90, fontsize=PLOTS_FONTSIZE)
                plt.yticks(fontsize=PLOTS_FONTSIZE)

                # Legend
                plt.legend(
                    handles=color_handles,
                    fontsize=PLOTS_FONTSIZE,
                    bbox_to_anchor=(1.05, 1),  # shift legend outside the right
                    loc='upper left',
                    borderaxespad=0.
                )
                fig.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right
                filename = f"{experiments_starting_time_str}/{unique_experiment_model}_{metric}.png"
                plt.savefig(filename)
                plt.close()

def get_all_experiments_aggregated_metrics_boxplot(result, experiments_starting_time_str, experiments_metrics_and_titles):
    maps_colors = get_maps_colors()
    color_handles = get_color_handles()
    for experiment_metric_and_title in experiments_metrics_and_titles:
        if 'histogram_' in experiment_metric_and_title['metric']:
            continue
        if 'line_' in experiment_metric_and_title['metric']:
            continue

        fig, ax = plt.subplots(figsize=(40, 10))
        dataframes = []
        max_value = 0

        # Build dataframes for each model/map combination
        for model_name in result['experiment_model'].unique():
            for carla_map in result['carla_map'].unique():
                com_dict = {
                    'model_name': model_name,
                    model_name + '-' + carla_map: result.loc[
                        (result['experiment_model'] == model_name) &
                        (result['carla_map'] == carla_map)
                        ][experiment_metric_and_title['metric']].tolist(),
                    'carla_map': carla_map
                }
                df = pd.DataFrame(data=com_dict)
                dataframes.append(df)
                max_value = max(max_value, df[model_name + '-' + carla_map].max())

        # Prepare full list and colors
        full_list = []
        colors = []
        for carla_map in result['carla_map'].unique():
            for experiment_model in result['experiment_model'].unique():
                full_list.append(experiment_model + '-' + carla_map)
                colors.append(maps_colors[carla_map])

        # Concatenate dataframes
        result_by_experiment_model = pd.concat(dataframes)

        # Boxplot
        ax, props = result_by_experiment_model.boxplot(
            column=full_list,
            showfliers=True, sym='k.',
            return_type='both',
            patch_artist=True,
            ax=ax
        )

        # Set box colors
        for patch, color in zip(props['boxes'], colors):
            patch.set_facecolor(color)

        # Set title and axis labels
        ax.set_title(experiment_metric_and_title['title'] + ' boxplot', fontsize=PLOTS_FONTSIZE)
        ax.set_xlabel('Model', fontsize=PLOTS_FONTSIZE)
        ax.set_ylabel(experiment_metric_and_title['metric'], fontsize=PLOTS_FONTSIZE)

        # Set tick labels and sizes
        ax.set_xticklabels([model.split('-')[0] for model in full_list], rotation=0)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)

        # Move legend below the plot
        ax.legend(
            handles=color_handles,
            fontsize=PLOTS_FONTSIZE,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(color_handles)
        )

        # Adjust layout to leave space for legend
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Adjust y-axis limit
        if max_value > 0:
            ax.set_ylim(0, max_value + max_value * 0.1)

        # Save figure
        plt.savefig(
            f"{experiments_starting_time_str}/{experiment_metric_and_title['metric']}_boxplot.png",
            bbox_inches='tight'
        )
        plt.close()


def get_distance_other_vehicle(experiment_metrics, checkpoints, checkpoints_2):
    dangerous_distance = 0  
    close_distance = 0
    medium_distance = 0
    great_distance = 0
    total_distance = 0
    
    dangerous_distance_pct_km = 0  
    close_distance_pct_km = 0
    medium_distance_pct_km = 0
    great_distance_pct_km = 0
    total_distance_pct_km = 0

    for i, (point, point_2) in enumerate(zip(checkpoints, checkpoints_2)):
        current_checkpoint = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        current_checkpoint_2 = np.array([point_2['pose.pose.position.x'], point_2['pose.pose.position.y']])
        
        if i != 0:
            distance_front = np.linalg.norm(current_checkpoint - current_checkpoint_2)
            distance = np.linalg.norm(previous_point - current_checkpoint)

            # Analyzing
            if 20 < distance_front < 50:
                great_distance += distance
                total_distance += distance
            elif 15 < distance_front <= 20:
                medium_distance += distance
                total_distance += distance
            elif 6 < distance_front <= 15:
                close_distance += distance
                total_distance += distance
            elif distance_front <= 6:
                dangerous_distance += distance
                total_distance += distance
                
        previous_point = current_checkpoint
            
    experiment_metrics['dangerous_distance_km'] = dangerous_distance
    experiment_metrics['close_distance_km'] = close_distance
    experiment_metrics['medium_distance_km'] = medium_distance
    experiment_metrics['great_distance_km'] = great_distance
    experiment_metrics['total_distance_to_front_car'] = total_distance
    
    experiment_metrics['dangerous_distance_pct_km'] = (total_distance and dangerous_distance / total_distance or 0) * 100
    experiment_metrics['close_distance_pct_km'] = (total_distance and close_distance / total_distance or 0) * 100
    experiment_metrics['medium_distance_pct_km'] = (total_distance and medium_distance / total_distance or 0) * 100
    experiment_metrics['great_distance_pct_km'] = (total_distance and great_distance / total_distance or 0) * 100
    
    return experiment_metrics