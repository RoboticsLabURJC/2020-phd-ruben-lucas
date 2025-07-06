import os
import pickle
import weakref
from collections import Counter, defaultdict
import math
import time
import carla
import random
import cv2
import torch
from numpy import random
import psutil
import numpy as np
import json
import traceback
from collections import deque

from pyglet.libs.x11.xlib import None_
from sympy.solvers.ode import infinitesimals
import threading
import rl_studio.config_loader as config_loader
from rl_studio.envs.carla.utils.modified_tensorboard import ModifiedTensorBoard

from rl_studio.envs.carla.followlane.followlane_env import FollowLaneEnv
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils
from PIL import Image
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from gymnasium import spaces
import gc

from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.manual_control import CameraManager
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
    CustomTimer,
)
import pygame

from rl_studio.envs.carla.utils.ground_truth.camera_geometry import (
    get_intrinsic_matrix,
    project_polyline,
    check_inside_image,
    create_lane_lines,
    get_matrix_global,
    CameraGeometry,
)

from PIL import Image, ImageTk, ImageDraw

NO_DETECTED = 0

def select_device(logger=None, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            if logger:
                logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        if logger:
            logger.info(f'Using torch {torch.__version__} CPU')

    if logger:
        logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def draw_dash(index, dist, ll_segment):
    height, width = ll_segment.shape

    # List of relative positions around 'dist' to set
    offsets = [-5, -4, -3, -2, -1]

    for offset in offsets:
        x = index
        y = dist + offset
        if 0 <= x < height and 0 <= y < width:
            ll_segment[x, y] = 255


def calculate_midpoints(input_array):
    midpoints = []
    for i in range(0, len(input_array) - 1, 2):
        midpoint = (input_array[i] + input_array[i + 1]) // 2
        midpoints.append(midpoint)
    return midpoints


def add_midpoints(ll_segment_channel, index, dist):
    draw_dash(index, dist, ll_segment_channel)
    draw_dash(index + 2, dist, ll_segment_channel)
    draw_dash(index + 1, dist, ll_segment_channel)
    draw_dash(index - 1, dist, ll_segment_channel)
    draw_dash(index - 2, dist, ll_segment_channel)


def connect_dashed_lines(ll_seg_mask):
    # TODO
    return ll_seg_mask


def discard_not_confident_centers(center_lane_indexes):
    # Count the occurrences of each list size leaving out of the equation the non-detected
    size_counter = Counter(len(inner_list) for inner_list in center_lane_indexes if NO_DETECTED not in inner_list)
    # Check if size_counter is empty, which mean no centers found
    if not size_counter:
        return center_lane_indexes
    # Find the most frequent size
    # most_frequent_size = max(size_counter, key=size_counter.get)

    # Iterate over inner lists and set elements to 1 if the size doesn't match majority
    result = []
    for inner_list in center_lane_indexes:
        # if len(inner_list) != most_frequent_size:
        if len(inner_list) < 1:  # If we don't see the 2 lanes, we discard the row
            inner_list = [NO_DETECTED] * len(inner_list)  # Set all elements to 1
        result.append(inner_list)

    return result


def choose_lane(distance_to_center_normalized, center_points):
    close_lane_indexes = [min(enumerate(inner_array), key=lambda x: abs(x[1]))[0] for inner_array in
                          distance_to_center_normalized]
    distances = [array[index] for array, index in zip(distance_to_center_normalized, close_lane_indexes)]
    centers = [array[index] for array, index in zip(center_points, close_lane_indexes)]
    return distances, centers


def wasDetected(center_lanes):
    for i in range(len(center_lanes)):
        if abs(center_lanes[i]) > 0.1:
            return False
    return True


def getTransformFromPoints(points):
    # print(f"OJO!! {points[0]}")
    # print(f"{points[1]}")
    # print(f"{points[2]}")
    # print(f"{points[3]}")
    # print(f"{points[4]}")
    # print(f"{points[5]}")
    return carla.Transform(
        carla.Location(
            x=points[0],
            y=points[1],
            z=points[2],
        ),
        carla.Rotation(
            pitch=points[3],
            yaw=points[4],
            roll=points[5],
        ),
    )

def interpolate_lane_points(lane_points: np.ndarray, num_points: int = 20, start_y: int = 640) -> np.ndarray:
    """
    Interpolates `num_points` equidistant points along a polyline that starts at y = `start_y`
    (image bottom) and ends at lane_points[-1], extrapolating the initial segment if needed.

    Args:
        lane_points (np.ndarray): Array of shape (N, 2) representing the lane boundary points.
        num_points (int): Number of interpolated points to return.
        start_y (int): The y-coordinate from which to start the interpolation (e.g., 640).

    Returns:
        np.ndarray: Interpolated points along the extended lane, shape (num_points, 2)
    """
    if lane_points.shape[0] < 2:
        return np.zeros((num_points, 2), dtype=np.float32)

    p0 = lane_points[0]
    p1 = lane_points[1]

    dy = p1[1] - p0[1]
    dx = p1[0] - p0[0]

    if dy == 0:
        slope = 0
    else:
        slope = dx / dy

    # Extrapolate backward to y = start_y
    delta_y = start_y - p0[1]
    extrapolated_x = p0[0] + slope * delta_y
    extrapolated_point = np.array([extrapolated_x, start_y])

    # Only prepend if extrapolated point is below p0 (i.e., y > p0[1])
    if start_y > p0[1]:
        lane_points = np.vstack([extrapolated_point, lane_points])

    # Compute arc length along the extended polyline
    deltas = np.diff(lane_points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)

    # Interpolate along arc length
    target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)
    interp_x = np.interp(target_lengths, cumulative_lengths, lane_points[:, 0])
    interp_y = np.interp(target_lengths, cumulative_lengths, lane_points[:, 1])

    interpolated = np.stack((interp_x, interp_y), axis=1)
    return interpolated.astype(np.int32)

def calculate_v_goal(mean_curvature, final_curvature, center_distance, curvature_weight=150):
    mean_curv = max(0, mean_curvature - 1) * 25
    dist_error = abs(center_distance) * 10
    v_goal = max(6, 25 - (mean_curv + dist_error))

    # Snap to closest value in the allowed list
    # allowed_values = [6, 11, 16, 21, 26]
    # v_goal = min(allowed_values, key=lambda x: abs(x - v_goal))

    # print(f"----------------------------------------")
    # print(f"monitoring last modification bro! dist_minus -> {dist_error}")
    # print(f"monitoring last modification bro! curv_minus -> {mean_curv}")
    # print(f"monitoring last modification bro! v_goal -> {v_goal}")
    # print(f"----------------------------------------")
    return v_goal

def curvature_from_three_points(p1, p2, p3):
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    if area == 0:
        return 0
    return (4 * area) / (a * b * c)

def calculate_max_curveture_from_centers(center_points):
    curvatures = []
    for i in range(1, len(center_points) - 1):
        k = curvature_from_three_points(
            np.array(center_points[i - 1]),
            np.array(center_points[i]),
            np.array(center_points[i + 1])
        )
        curvatures.append(k)
    return max(curvatures)

def average_curvature_from_centers(center_points):
    total_angle = 0.0

    for i in range(1, len(center_points) - 1):
        p1 = np.array(center_points[i - 1])
        p2 = np.array(center_points[i])
        p3 = np.array(center_points[i + 1])

        v1 = p2 - p1
        v2 = p3 - p2

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            continue  # skip degenerate segment

        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.arccos(cos_theta)  # radians

        total_angle += angle

    return total_angle  # in radians

def calculate_max_curveture_from_centers_v1(center_points, window_size=5):
    assert window_size >= 3 and window_size % 2 == 1, "window_size must be an odd number >= 3"

    curvatures = []
    half_window = window_size // 2

    points = np.array(center_points)

    for i in range(half_window, len(points) - half_window):
        # Extract a window of points
        window = points[i - half_window:i + half_window + 1]

        # First derivative (velocity): central difference
        dx = np.gradient(window[:, 0])
        dy = np.gradient(window[:, 1])

        # Second derivative (acceleration): central difference
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Compute curvature κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx[half_window] * ddy[half_window] - dy[half_window] * ddx[half_window])
        denominator = (dx[half_window] ** 2 + dy[half_window] ** 2) ** 1.5
        curvature = numerator / denominator if denominator > 1e-6 else 0.0

        curvatures.append(curvature)

    return max(curvatures) if curvatures else 0.0

def calculate_curvature_from(x, y):
    try:
        # Create a mask for elements in `centers` that are NOT 1
        mask = y != 0

        # Apply the mask to both arrays
        x = x[mask]
        y = y[mask]

        if len(x) < 3:
            print("Not enough points to fit a 2nd degree polynomial.")
            return -1.0

        coefficients = np.polyfit(x, y, 2)  # Returns [a, b, c]
        a, b, c = coefficients

        x_mid = x[2]  # Use the middle point
        y_prime = 2 * a * x_mid + b
        y_double_prime = 2 * a
        curvature = abs(y_double_prime) / ((1 + y_prime ** 2) ** (3 / 2))
    except Exception as e:
        print("exception calculating the curvature" + repr(e))
        curvature = -1
    return curvature

def trim_polyline_to_image(points_2d, image_shape):
    h, w = image_shape
    trimmed = []

    for p in points_2d:
        if 0 <= p[0] < w and 0 <= p[1] < h:
            trimmed.append(p)

    return np.array(trimmed)


def get_closest_to_center_border(x_left_points, x_right_points):
    center = 320  # Image center

    # Convert to numpy arrays for efficient computation
    x_left_points = np.array(x_left_points)
    x_right_points = np.array(x_right_points)

    # Filter out points that are -1 and consider only points from index 300 onwards
    valid_left_points = x_left_points[300:][x_left_points[300:] != -1]
    valid_right_points = x_right_points[300:][x_right_points[300:] != -1]

    # Compute mean distance to center for each list
    left_dist = np.mean(np.abs(valid_left_points - center)) if valid_left_points.size > 0 else float('inf')
    right_dist = np.mean(np.abs(valid_right_points - center)) if valid_right_points.size > 0 else float('inf')

    # Return the list closer to center
    return x_left_points if left_dist < right_dist else x_right_points

def get_more_separated_line(line1_points, line2_points, middle_line_points):
    """
    Given two lane lines and a reference middle line (all as y-indexed x-values),
    return the one (line1 or line2) whose points are more separated on average from the middle line.

    Points with value -1 in either line or middle_line are excluded.
    """
    def average_separation(line_points, middle_line_points):
        separations = []
        for y in range(300, len(line_points)):
            x1 = line_points[y]
            xm = middle_line_points[y]
            if x1 != -1 and xm != -1:
                separations.append(abs(x1 - xm))
        return sum(separations) / len(separations) if separations else 0

    sep1 = average_separation(line1_points, middle_line_points)
    sep2 = average_separation(line2_points, middle_line_points)

    return line1_points if sep1 > sep2 else line2_points



def get_evenly_separated_points(closest_border, num_points):
    """
    Selects `num_points` evenly spaced (x, y) pairs along closest_border.
    If exact y-values don't exist or are -1, interpolates between valid points.
    Guarantees exactly `num_points` output, using (-1, -1) as fallback.
    """
    closest_border = list(closest_border)
    height = len(closest_border)

    # Collect all valid (y, x) pairs
    valid_points = [(y, closest_border[y]) for y in range(height) if closest_border[y] != -1]

    if len(valid_points) < 2:
        # Not enough valid points to interpolate
        return [(-1, -1)] * num_points

    # Sort by y to ensure order
    valid_points.sort()

    # Interpolate over y-axis
    y_vals, x_vals = zip(*valid_points)

    # Generate num_points evenly spaced y-values
    min_y = y_vals[0]
    max_y = y_vals[-1]
    if num_points == 1:
        sampled_y = [int(round((min_y + max_y) / 2))]
    else:
        step = (max_y - min_y) / (num_points - 1)
        sampled_y = [min(height - 1, max(0, int(round(min_y + i * step)))) for i in range(num_points)]

    # Interpolate x at sampled_y
    interpolated_points = []
    for y in sampled_y:
        # Find two closest known points around y
        lower = None
        upper = None
        for i in range(len(y_vals)):
            if y_vals[i] <= y:
                lower = (y_vals[i], x_vals[i])
            if y_vals[i] >= y:
                upper = (y_vals[i], x_vals[i])
                break

        if lower and upper:
            y0, x0 = lower
            y1, x1 = upper
            if y0 == y1:
                x = x0
            else:
                # Linear interpolation
                x = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
            interpolated_points.append((int(round(x)), int(y)))
        else:
            interpolated_points.append((-1, -1))

    return interpolated_points


def flatten_points_and_merge(border_points, line_points):
    # Flatten each list: [x1, y1, x2, y2, ..., xN, yN]
    flat_border = [coord for point in border_points for coord in point]
    flat_line = [coord for point in line_points for coord in point]

    # Concatenate them
    state = flat_border + flat_line
    return state

def flatten_points(points):
    # Flatten each list: [x1, y1, x2, y2, ..., xN, yN]
    flat_line = [coord for point in points for coord in point]

    return flat_line

def get_center_points(line_points, border_points):
    """
    For each (x_border, y) in border_points, finds the corresponding x_line from line_points
    (or interpolated via polynomial regression), and returns the midpoint between (x_line, y)
    and (x_border, y).

    Returns:
    - List of (int(center_x), int(y)) center points.
    """
    if not line_points or not border_points:
        return []

    # Extract valid (y, x) points from line_points
    yx_valid = [(y, x) for y, x in enumerate(line_points) if x != -1]

    if len(yx_valid) < 2:
        # Not enough points to fit curve → fallback to using border x-values
        return [(int(x), int(y)) for x, y in border_points]

    # Fit polynomial x = f(y)
    y_vals, x_vals = zip(*yx_valid)
    y_vals = np.array(y_vals)
    x_vals = np.array(x_vals)

    try:
        degree = 3 if len(y_vals) > 2 else 1
        poly = np.poly1d(np.polyfit(y_vals, x_vals, deg=degree))
    except Exception as e:
        raise RuntimeError(f"Polynomial fitting failed: {e}")

    # For each border_point, estimate x_line and compute center
    center_points = []
    for x_border, y in border_points:
        if 0 <= y < len(line_points):
            if line_points[y] != -1:
                x_line = line_points[y]
            else:
                x_line = poly(y)
            x_center = (x_border + x_line) / 2
            center_points.append((int(round(x_center)), int(y)))

    return center_points


def normalize_centers(centers):
    x_centers = centers[:, 0]
    x_centers_normalized = (x_centers / 512).tolist()
    states = x_centers_normalized
    y_centers = centers[:, 1]
    states = states + (y_centers / 640).tolist()  # Returns a list
    return states, x_centers_normalized


class FollowLaneStaticWeatherNoTraffic(FollowLaneEnv):
    def __init__(self, **config):

        self.NON_DETECTED = -1
        self.dashboard = config.get("dashboard")
        self.estimated_steps = config.get("estimated_steps")

        self.start = time.time()

        self.last_lane_id = None
        self.prev_throtle = 0
        self.episode_d_reward = 0
        self.curves_states = 0
        self.abs_w_no_curves_avg = 0
        self.throttle_action_avg_no_curves = 0
        self.throttle_action_avg_curves = 0
        self.throttle_action_difference = 0
        self.throttle_action_std_dev_curves = 0
        self.throttle_action_variance_curves = 0
        self.throttle_action_std_dev_no_curves = 0
        self.throttle_action_variance_no_curves = 0
        self.episode_d_deviation = 0
        self.episode_v_eff_reward = 0
        self.location_actions = {}
        self.location_rewards = {}
        self.location_next_states = {}
        self.location_stats = []
        self.show_images = False
        self.show_all_points = False
        self.config = config
        self.tensorboard_location_writers = {}
        self.previous_action = [0, 0]

        self.update_from_hot_config(config)

        self.actions = config.get("actions")
        self.stage = config.get("stage")

        self.lane_points = None

        self.use_curves_state = config.get("use_curves_state")

        self.failures = 0
        self.tensorboard = config.get("tensorboard")
        self.tensorboard_logs_dir = config.get("tensorboard_logs_dir")

        self.actor_list = []
        self.episodes_speed = []

        ###### init class variables
        FollowLaneCarlaConfig.__init__(self, **config)
        self.projected_x = config.get("projected_x_row")
        self.sync_mode = config["sync"]
        self.front_car = config.get("front_car")
        self.front_car_spawn_points = config.get("front_car_spawn_points")
        self.reset_threshold = config["reset_threshold"] if self.sync_mode else 1
        self.spawn_points = config.get("spawn_points")
        self.detection_mode = config.get("detection_mode")
        self.device = select_device()
        self.fixed_delta_seconds = config.get("fixed_delta_seconds")
        self.appended_states = config.get("appended_states", 0)

        if self.detection_mode == 'yolop':
            from rl_studio.envs.carla.utils.yolop.YOLOP import get_net
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

            self.steps_stopped = 0
            # INIT YOLOP
            self.yolop_model = get_net()
            checkpoint = torch.load(
                "/home/alumnos/rlucasz/Escritorio/RL-Studio/rl_studio/envs/carla/utils/yolop/weights/End-to-end.pth",
                map_location=self.device)
            self.yolop_model.load_state_dict(checkpoint['state_dict'])
            self.yolop_model = self.yolop_model.to(self.device)
        elif self.detection_mode == "lane_detector_v2":
            self.lane_model = torch.load(
                '/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/utils/lane_det/fastai_torch_lane_detector_model.pth',
                map_location=self.device)
            self.lane_model.eval()
        elif self.detection_mode == "lane_detector":
            self.lane_model = torch.load(
                'envs/carla/utils/lane_det/best_model_torch.pth').to(self.device)
            self.lane_model.eval()
        elif self.detection_mode == "lane_detector_v2_poly":
            self.lane_model = torch.load(
                'envs/carla/utils/lane_det/best_model_torch.pth').to(self.device)
            self.lane_model.eval()
        else:
            self.camera_transform = carla.Transform(
                carla.Location(x=-2, y=0, z=2),
                carla.Rotation(pitch=-2, yaw=0, roll=0.0)
            )

            # Translation matrix, convert vehicle reference system to camera reference system

            self.trafo_matrix_vehicle_to_cam = np.array(
                self.camera_transform.get_inverse_matrix()
            )
            self.k = None

        # self.display_manager = None
        # self.vehicle = None
        # self.actor_list = []
        self.timer = CustomTimer()
        self.step_count = 1
        self.step_count_curves = 1
        self.step_count_no_curves = 1
        self.episode = 0
        self.deviated = 0
        self.all_steps = 0
        self.cumulated_reward = 0
        self.ports = config["carla_client"]
        self.carla_ip = config["carla_server"]

        ## -- display manager
        self.display_manager = DisplayManager(
            grid_size=[2, 3],
            window_size=[1500, 800],
            headless=False
        )

        try:
            self.initialize_carla()
        except Exception as e:
            print("Waiting for CARLA to become available to reconnect...")
            self.reconnect_to_carla()

        self.car = None

        self.perfect_distance_pixels = None
        self.perfect_distance_normalized = None

        self.no_detected = [[0]] * len(self.x_row)

        self.num_points = 20
        num_states = self.num_points * 2
        # num_states = 0
        # num_states += len(self.x_row) if self.x_row is not None else 0
        # num_states += len(self.projected_x) if self.projected_x is not None else 0
        num_states += self.appended_states
        if self.actions.get("b") is not None:
            self.action_space = spaces.Box(low=np.array([self.actions["v"][0],
                                                         self.actions["w"][0],
                                                         self.actions["b"][0]]),
                                           high=np.array([self.actions["v"][1],
                                                          self.actions["w"][1],
                                                          self.actions["b"][1]]),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=np.array([self.actions["v"][0],
                                                         self.actions["w"][0]]),
                                           high=np.array([self.actions["v"][1],
                                                          self.actions["w"][1]]),
                                           dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(num_states,), dtype=np.float32)

    def setup_car_fix_pose(self, init):
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = carla.Transform(
            carla.Location(
                x=init.transform.location.x,
                y=init.transform.location.y,
                z=init.transform.location.z,
            ),
            carla.Rotation(
                pitch=init.transform.rotation.pitch,
                yaw=init.transform.rotation.yaw,
                roll=init.transform.rotation.roll,
            ),
        )

        vehicle = self.world.spawn_actor(car_bp, location)
        while vehicle is None:
            vehicle = self.world.spawn_actor(car_bp, location)

        self.acor_list.append(vehicle)

        return vehicle

    def close(self):
        self.destroy_all_actors()
        self.display_manager.destroy()

    def doReset(self):
        ep_time = time.time() - self.start
        print("Step time:", ep_time / self.step_count)
        if self.tensorboard is not None:
            self.calculate_and_report_episode_stats()

        self.close()

        if self.episode % 25 == 0:  # Adjust step frequency as needed
            gc.collect()
            # self.reload_worlds()
            self.load_any_world()
            # print(psutil.Process(os.getpid()).memory_info().rss / 1e6, "MB")
            # print(self.world.get_actors().filter("*"))
            # print("Thread count:", threading.active_count())

        if self.episode % 10 == 0:
            hot_config = config_loader.load_hot_config()
            self.update_from_hot_config(hot_config)

        if self.stage == "w":
            self.fixed_random_throttle = random.uniform(0.4, 0.8)

        self.episode += 1

        self.v_goal_buffer = deque(maxlen=10)

        self.location_stats_episode = {
            "actions": self.location_actions,
            "rewards": self.location_rewards,
            "next_states": self.location_next_states
        }
        self.location_stats.append(self.location_stats_episode)
        self.location_actions = {}
        self.location_rewards = {}
        self.location_next_states = {}
        self.last_centers = None
        self.lane_points = None

        # if self.episode % 20 == 0:
        #     self.tensorboard.save_location_stats(self.location_stats)

        self.steps_stopped = 0
        self.collision_hist = []
        self.invasion_hist = []
        self.actor_list = []
        self.previous_time = 0

        init_pose = self.set_init_pose()
        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        time.sleep(0.5)

        self.car.v_component = 0
        self.car.d_reward = 0
        self.car.v_eff_reward = 0
        self.car.v_punish = 0
        self.car.zig_zag_punish = 0
        self.car.v_goal = 0

        self.vary_car_orientation()  # Call the orientation variation method
        # self.set_init_speed()

        self.start_location_tag = get_car_location_tag(init_pose)

        self.car.apply_control(carla.VehicleControl(steer=0.0))

        # AutoCarlaUtils.show_image("image", self.front_camera_1_5.front_camera, 1)
        # AutoCarlaUtils.show_image("bird_view", self.birds_eye_camera.front_camera, 1)

        # raw_image = self.get_resized_image(self.front_camera_1_5.front_camera)
        # segmentated_image = self.get_resized_image(self.front_camera_1_5_segmentated.front_camera)
        raw_image = self.front_camera_1_5.front_camera
        segmentated_image = self.front_camera_1_5_segmentated.front_camera
        segmentated_raw_data = self.front_camera_1_5_segmentated.raw_data


        (ll_segment,
         misalignment,
         center_distance,
         center_points) = self.detect_center_line_perfect(raw_image, num_points=self.num_points)

        final_curvature = calculate_max_curveture_from_centers(center_points)
        mean_curvature = average_curvature_from_centers(center_points)

        states, x_centers_normalized = normalize_centers(center_points)
        v_goal = calculate_v_goal(mean_curvature,
                                  final_curvature,
                                  center_distance)

        states.append(0)
        states.append(0)
        # states.append(final_curvature)
        states.append(0)
        states.append(0)
        states.append(v_goal/25)
        # states.append(0)
        # states.append(0)
        # if self.use_curves_state:
        #     states.append(curve)
        state_size = len(states)

        self.cumulated_reward = 0
        self.step_count = 1
        self.step_count_curves = 1
        self.step_count_no_curves = 1
        self.episodes_speed = []
        self.episode_d_deviation = 0
        self.prev_throtle = 0
        self.curves_states = 0
        self.abs_w_no_curves_avg = 0
        self.throttle_action_avg_no_curves = 0
        self.throttle_action_avg_curves = 0
        self.throttle_action_difference = 0
        self.throttle_action_std_dev_curves = 0
        self.throttle_action_variance_curves = 0
        self.throttle_action_std_dev_no_curves = 0
        self.throttle_action_variance_no_curves = 0

        self.start = time.time()
        return np.array(states), state_size

    def reset(self, seed=None, options=None):
        try:
            return self.doReset()
        except Exception as e:
            print(f"Waiting for CARLA to become available to reset... after {e}")
            traceback.print_exc()
            self.reconnect_to_carla()
            return self.reset()

    def calculate_and_report_episode_stats(self):
        if len(self.episodes_speed) == 0:
            return
        episode_time = self.step_count * self.fixed_delta_seconds
        self.avg_speed = np.mean(self.episodes_speed)
        self.max_speed = np.max(self.episodes_speed)
        # cum_d_reward = np.sum(self.episodes_d_reward)
        # max_reward = np.max(self.episodes_reward)
        # steering_std_dev = np.std(self.episodes_steer)
        self.advanced_meters = self.avg_speed * episode_time
        if self.curves_states > 5:
            self.throttle_action_difference = self.throttle_action_avg_no_curves - self.throttle_action_avg_curves
        else:
            self.throttle_action_difference = None
            self.throttle_action_avg_curves = None
            self.throttle_action_std_dev_curves = None
            self.throttle_action_variance_curves = None
        # completed = 1 if self.step_count >= self.env_params.estimated_steps else 0
        self.update_all_tensorboard_stats()
        self.update_reward_monitor(self.cumulated_reward)

    ####################################################
    ####################################################

    def find_lane_center(self, mask, i, center_image):
        # Find the indices of 1s in the array
        mask_array = np.array(mask)
        indices = np.where(mask_array > 0.8)[0]

        # If there are no 1s or only one set of 1s, return None
        if len(indices) < 2:
            # TODO (Ruben) For the moment we will be dealing with no detection as a fixed number
            return self.miss_detection(i, center_image)

        # Find the indices where consecutive 1s change to 0
        diff_indices = np.where(np.diff(indices) > 1)[0]
        # If there is only one set of 1s, return None
        if len(diff_indices) == 0:
            return self.miss_detection(i, center_image)

        interested_line_borders = np.array([], dtype=np.int8)
        # print(indices)
        for index in diff_indices:
            interested_line_borders = np.append(interested_line_borders, indices[index])
            interested_line_borders = np.append(interested_line_borders, int(indices[index + 1]))

        midpoints = calculate_midpoints(interested_line_borders)
        self.no_detected[i] = midpoints

        return midpoints

    def calculate_center(self, mask):
        width = mask.shape[1]
        center_image = width / 2
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        # ## As we drive in the right lane, we get from right to left
        # lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from the right lane to center
        center_lane_indexes = [
            self.find_lane_center(lines[x], x, center_image) for x, _ in enumerate(lines)
        ]

        # this part consists of checking the number of lines detected in all rows
        # then discarding the rows (set to 1) in which more or less centers are detected
        center_lane_indexes = discard_not_confident_centers(center_lane_indexes)

        center_lane_distances = [
            [center_image - x for x in inner_array] for inner_array in center_lane_indexes
        ]

        # Calculate the average position of the lane lines
        ## normalized distance
        distance_to_center_normalized = [
            np.array(x) / (width - center_image) for x in center_lane_distances
        ]

        return center_lane_indexes, distance_to_center_normalized

    def find_first_lanes_index(self, image):
        for i in range(image.shape[0]):
            line = image[i, :]
            # Find the indices of 1s in the array
            image_array = np.array(line)
            indices = np.where(image_array > 0.8)[0]
            diff_indices = np.where(np.diff(indices) > 1)[0]
            if len(diff_indices) == 0:
                continue

            interested_line_borders = np.array([], dtype=np.int8)

            for index in diff_indices:
                interested_line_borders = np.append(interested_line_borders, indices[index])
                interested_line_borders = np.append(interested_line_borders, int(indices[index + 1]))
            return interested_line_borders
        return None

    def calculate_states(self, mask):
        width = mask.shape[1]
        center_image = width / 2
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        ## As we drive in right lane, we get from right to left
        lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from right lane to center
        inv_index_right = [
            np.argmax(lines_inversed[x]) for x, _ in enumerate(lines_inversed)
        ]
        index_right = [
            width - inv_index_right[x] for x, _ in enumerate(inv_index_right)
        ]
        distance_to_center = [
            width - inv_index_right[x] - center_image
            for x, _ in enumerate(inv_index_right)
        ]
        ## normalized distances
        distance_to_center_normalized = [
            abs(float((center_image - index_right[i]) / center_image))
            for i, _ in enumerate(index_right)
        ]
        # pixels_in_state = mask.shape[1] / self.num_regions
        # states = [int(value / pixels_in_state) for _, value in enumerate(index_right)]
        states = distance_to_center_normalized

        return states, distance_to_center, distance_to_center_normalized

    def preprocess_image(self, red_mask):
        ## first, we cut the upper image
        img_sliced = self.slice_image(red_mask)
        ## -- convert to GRAY
        gray_mask = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        ## --  aplicamos mascara para convertir a BLANCOS Y NEGROS
        _, white_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

        return white_mask

    def draw_waypoints(self, spawn_points, init, target, lane_id, life_time):
        filtered_waypoints = []
        # i = init
        # for waypoint in spawn_points[init + 1: target + 2]:
        i = 0
        for waypoint in spawn_points:
            filtered_waypoints.append(waypoint)
            string = f"[{waypoint.road_id},{waypoint.lane_id},{i}]"
            # if waypoint.lane_id == lane_id:
            if i != target:
                self.world.debug.draw_string(
                    waypoint.transform.location,
                    f"X - {string}",
                    draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=0),
                    life_time=life_time,
                    persistent_lines=True,
                )
            else:
                self.world.debug.draw_string(
                    waypoint.transform.location,
                    f"X - {string}",
                    draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=life_time,
                    persistent_lines=True,
                )
            i += 1

        return filtered_waypoints

    def get_target_waypoint(self, target_waypoint, life_time):
        """
        draw target point
        """
        self.world.debug.draw_string(
            target_waypoint.transform.location,
            "O",
            draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0),
            life_time=life_time,
            persistent_lines=True,
        )

    def setup_car_random_pose(self, spawn_points):
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        if self.spawn_points is not None:
            spawn_point_index = random.randint(0, len(spawn_points))
            spawn_point = spawn_points[spawn_point_index]
            if random.random() <= 1:  # TODO make it configurable
                location = getTransformFromPoints(spawn_point)
            else:
                location = random.choice(self.world.get_map().get_spawn_points())

            vehicle = self.world.spawn_actor(car_bp, location)
            while vehicle is None:
                vehicle = self.world.spawn_actor(car_bp, location)
        else:
            location = random.choice(self.world.get_map().get_spawn_points())
            vehicle = self.world.try_spawn_actor(car_bp, location)
            while vehicle is None:
                vehicle = self.world.try_spawn_actor(car_bp, location)
        self.actor_list.append(vehicle)
        vehicle.reward = 0
        vehicle.error = 0
        return vehicle, location

    # Method to add the obstacle detector sensor to the vehicle
    def add_obstacle_detector(self, vehicle, world):
        """
        Adds an obstacle detector to the vehicle in CARLA world and starts listening to obstacles.
        """
        # Get the blueprint library for the world
        blueprint_library = world.get_blueprint_library()

        # Find the obstacle detector sensor blueprint
        obstacle_sensor_bp = blueprint_library.find('sensor.other.obstacle')

        # Set the position of the sensor (front of the vehicle)
        sensor_transform = carla.Transform(carla.Location(x=10, z=1))  # Adjust the position as needed

        # Visualize the position of the obstacle sensor by spawning a sphere
        # Get the blueprint library
        # blueprint_library = world.get_blueprint_library()

        # List all available blueprints
        # for blueprint in blueprint_library:
        #    print(f"Blueprint: {blueprint.id}")

        # Spawn the obstacle sensor and attach it to the vehicle
        obstacle_sensor = world.spawn_actor(obstacle_sensor_bp, sensor_transform, attach_to=vehicle)

        # Listen for obstacle data
        obstacle_sensor.listen(lambda data: self.on_obstacle_detected(data))

        self.actor_list.append(obstacle_sensor)

    # Callback function to handle the obstacle detection data
    def on_obstacle_detected(self, obstacle_data):
        """
        Callback function to process detected obstacles.
        """
        # Access the obstacle data directly (not as a list)
        print(obstacle_data.other_actor)
        print(f"Obstacle detected at: ")

    def add_lidar_to_vehicle(self, world, vehicle, lidar_range=50.0, channels=32, rotation_frequency=10.0,
                             points_per_second=56000):
        # Get the Lidar blueprint
        blueprint_library = world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

        # Configure the Lidar settings
        lidar_bp.set_attribute('range', '100')  # Maximum range in meters
        # lidar_bp.set_attribute('channels', str(channels))  # Number of vertical channels
        # lidar_bp.set_attribute('rotation_frequency', str(rotation_frequency))  # Rotations per second
        # lidar_bp.set_attribute('points_per_second', str(points_per_second))  # Points per second
        lidar_bp.set_attribute('upper_fov', '5')  # Top angle of vertical FOV (fixed direction)
        lidar_bp.set_attribute('lower_fov', '1')  # Bottom angle of vertical FOV (fixed direction)
        lidar_bp.set_attribute('horizontal_fov', '0')  # Single horizontal line
        lidar_bp.set_attribute('channels', '30')  # Single vertical line
        lidar_bp.set_attribute('channels', '20')  # Single vertical line
        lidar_bp.set_attribute('rotation_frequency', '20')  # Keep as is for 10 Hz
        lidar_bp.set_attribute('points_per_second', '5000')  # Fewer points for a narrow horizontal sweep

        # Set the Lidar sensor's position relative to the vehicle
        lidar_transform = carla.Transform(
            carla.Location(x=1.5, z=0),  # x is forward, z is height
            carla.Rotation(pitch=0, yaw=5, roll=0)
        )

        # Spawn the Lidar sensor and attach it to the vehicle
        self.lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        self.actor_list.append(self.lidar_sensor)

        self.lidar_sensor.listen(lambda data: self.process_lidar_data(data))

    def lidar_point_to_world(self, lidar_detection):
        # Get the sensor's transformation
        sensor_transform = self.lidar_sensor.get_transform()
        sensor_location = sensor_transform.location
        sensor_rotation = sensor_transform.rotation

        # Extract relative point from lidar detection
        relative_point = lidar_detection.point  # Example: (x, y, z) in sensor's frame

        # Convert sensor rotation to radians
        yaw = math.radians(sensor_rotation.yaw)
        pitch = math.radians(sensor_rotation.pitch)
        roll = math.radians(sensor_rotation.roll)

        # Rotation matrix for the sensor
        rotation_matrix = np.array([
            [
                math.cos(yaw) * math.cos(pitch),
                math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.sin(yaw) * math.cos(roll),
                math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll)
            ],
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll),
                math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll)
            ],
            [
                -math.sin(pitch),
                math.cos(pitch) * math.sin(roll),
                math.cos(pitch) * math.cos(roll)
            ]
        ])

        # Transform the relative point to the world frame
        relative_vector = np.array([relative_point.x, relative_point.y, relative_point.z])
        world_vector = np.dot(rotation_matrix, relative_vector)

        # Add the sensor's global location
        world_x = sensor_location.x + world_vector[0]
        world_y = sensor_location.y + world_vector[1]
        world_z = sensor_location.z + world_vector[2]

        return carla.Location(x=world_x, y=world_y, z=world_z)

    def process_lidar_data(self, data):
        car_location = self.car.get_location()
        min_distance = float('inf')

        # Define the range of lateral (Y) distance for "front" filtering (optional)
        lateral_limit = 2.0  # 2 meters to the left and right of the vehicle's center

        # Define the angle range for the "cone" of front-facing points
        front_angle_limit = 30.0  # 30 degrees (left and right) from the center of the vehicle

        for detection in data:
            # Transform LiDAR point to world coordinates
            world_point = self.lidar_point_to_world(detection)

            # Only consider points ahead of the vehicle and within the cone
            # if world_point.x > car_location.x and abs(angle) <= front_angle_limit and abs(world_point.y) < lateral_limit:
            if True:
                # Compute distance to the car
                distance = car_location.distance(world_point)
                if distance < min_distance:
                    min_distance = distance

                # Visualize the LiDAR point in front of the vehicle
                # self.world.debug.draw_point(world_point, size=0.1, color=carla.Color(255, 0, 0), life_time=0.1)
        self.lidar_front_distance = min_distance if not math.isinf(min_distance) else 100
        # print(f"Closest object in front of the vehicle is {min_distance} meters away.")

    def setup_lane_sensor(self, vehicle):
        lane_invasion = self.world.get_blueprint_library().find("sensor.other.lane_invasion")
        transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lane_invasion = self.world.spawn_actor(
            lane_invasion, transform, attach_to=vehicle
        )
        self.actor_list.append(self.lane_invasion)
        self.lane_invasion.listen(lambda event: self.invasion_data(event))

    def invasion_data(self, event):
        self.invasion_hist.append(event)

    def setup_col_sensor(self, vehicle):
        colsensor = self.world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=vehicle
        )
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

    def collision_data(self, event):
        self.collision_hist.append(event)

    def destroy_all_actors(self):
        if len(self.actor_list) <= 0:
            return
        for actor in self.actor_list[::-1]:
            # for actor in self.actor_list:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            actor.destroy()
        # print(f"\nin self.destroy_all_actors(), actor : {actor}\n")

        self.actor_list.clear()
        # .client.apply_batch(
        #    [carla.command.DestroyActor(x) for x in self.actor_list[::-1]]
        # )

    def show_debug_points(self, centers):
        if not self.debug_waypoints:
            return
        average_abs = sum(abs(x) for x in centers) / len(centers)
        # if average_abs > 0.8:
        #     color = carla.Color(r=255, g=0, b=0)
        # else:
        #     green_value = max(int((1 - average_abs * 2 ) * 255), 0 )
        #     color = carla.Color(r=0, g=green_value, b=0)
        green_value = max(int((1 - average_abs * 2) * 255), 0)
        color = carla.Color(r=0, g=green_value, b=0)

        self.world.debug.draw_string(
            self.car.get_transform().location,
            "X",
            draw_shadow=False,
            color=color,
            life_time=10000000,
            persistent_lines=True,
        )
        if self.step_count % 100 == 1:
            print(self.car.get_transform())

    def project_line(self, y_points, x_points, x_normalized, new_y_points):
        # Filter indices where x_normalized != 1
        filtered_indices = [i for i, x in enumerate(x_normalized) if x != 1]
        filtered_x_points = [x_points[i] for i in filtered_indices]
        filtered_y_points = [y_points[i] for i in filtered_indices]
        filtered_x_normalized = [x_normalized[i] for i in filtered_indices]
        if len(filtered_indices) == 0:
            all_x_points = np.concatenate((x_points, x_points)).astype(int).tolist()
            all_x_normalized = np.concatenate((x_normalized, x_normalized)).tolist()

            return all_x_points, all_x_normalized

        # Create interpolation functions using the filtered points
        interpolation_function = interp1d(
            filtered_y_points, filtered_x_points, kind="linear", fill_value="extrapolate"
        )
        interpolation_function_normalized = interp1d(
            filtered_y_points, filtered_x_normalized, kind="linear", fill_value="extrapolate"
        )

        # Calculate x-values for the new y-values
        projected_x_points = interpolation_function(new_y_points)
        projected_x_normalized_points = interpolation_function_normalized(new_y_points)
        projected_x_normalized_points = np.where(np.isnan(projected_x_normalized_points), 1,
                                                 projected_x_normalized_points)
        projected_x_points = np.where(np.isnan(projected_x_points), 1, projected_x_points)

        # Append the projections to the original arrays
        all_x_points = np.concatenate((projected_x_points, x_points)).astype(int).tolist()
        all_x_normalized = np.concatenate((projected_x_normalized_points, x_normalized)).tolist()

        return all_x_points, all_x_normalized

    def apply_step(self, action):
        self.all_steps += 1
        # action[1] = action[1] * 0.5
        # action[0] = 0.6

        self.car.v_component = 0
        self.car.d_reward = 0
        self.car.v_eff_reward = 0
        self.car.v_punish = 0
        self.car.zig_zag_punish = 0
        self.car.v_goal = 0

        params = self.control(action)

        now = time.time()
        elapsed_time = now - self.previous_time

        if self.sync_mode:
            self.world.tick()
        else:
            # if elapsed_time < self.forced_freq:
            #     wait_time = self.forced_freq - elapsed_time
            #     time.sleep(wait_time)
            self.world.wait_for_tick()

        now = time.time()
        params["fps"] = 1 / (now - self.previous_time)
        self.previous_time = now

        # raw_image = self.get_resized_image(self.front_camera_1_5.front_camera)
        raw_image = self.front_camera_1_5.front_camera

        (ll_segment,
         misalignment,
         center_distance,
         centers) = self.detect_center_line_perfect(raw_image, num_points=self.num_points)

        final_curvature = calculate_max_curveture_from_centers(centers)
        mean_curvature = average_curvature_from_centers(centers)

        states, x_centers_normalized = normalize_centers(centers)
        centers_image = np.zeros(raw_image.shape, dtype=np.uint8)
        for index in range(len(centers)):
            if centers[index][0] == self.NON_DETECTED:
                continue
            cv2.circle(centers_image, (centers[index][0], centers[index][1]), radius=3,
                      color=(255, 255, 0), thickness=-1)
            # cv2.circle(ll_segment, (line_points[index][0], line_points[index][1]), radius=3, color=(0, 0, 255),
            #            thickness=-1)
        gray_overlay = cv2.cvtColor(centers_image, cv2.COLOR_BGR2GRAY)
        mask = gray_overlay > 10
        mask_3ch = np.stack([mask] * 3, axis=-1)

        stacked_image = np.where(mask_3ch, centers_image, raw_image)
        _, center_distance, _ = self.get_lane_position(self.car, self.map)
        # v_goal = calculate_v_goal(final_curvature, center_distance, curvature_weight=150)
        v_goal = calculate_v_goal(mean_curvature,
                                  final_curvature,
                                  center_distance)
        self.v_goal_buffer.append(v_goal)
        v_goal = sum(self.v_goal_buffer) / len(self.v_goal_buffer)

        params["final_curvature"] = final_curvature
        reward, done, has_crashed = self.rewards_easy(center_distance, action, params, v_goal=v_goal, x_centers_normalized=x_centers_normalized)
        self.previous_action = action

        self.car.v_goal = v_goal
        self.car.reward = reward
        self.cumulated_reward = self.cumulated_reward + reward
        self.episodes_speed.append(params["velocity"])

        # states = right_lane_normalized_distances

        states.append(params["velocity"] / 25)
        states.append(params["steering_angle"])
        #states.append(final_curvature)
        # states.append(misalignment)
        states.append(action[0])
        states.append(action[1])
        states.append(v_goal/25)
       # states.append(last_points[0])
       # states.append(last_points[1])
        # states.append(self.lidar_front_distance/100)
        # states.append(curvature * 10)
        # states.append(params["angular_velocity"]/100)
        # if self.use_curves_state:
        #     states.append(curve)

        if self.visualize:
            # self.show_ll_seg_image(centers, ll_segment) # for carla_perfect_lines
            # self.show_image("road_and_lines", road_and_lines)
            # self.show_debug_points(centers)
            self.show_image('overlayed_image', stacked_image) # for carla_segmentated
            self.display_manager.render(vehicle=self.car)

        # bad_left_perception = self.has_bad_perception(lane_left_points[:, 0], max_bad_real_states=3)
        # bad_right_perception = self.has_bad_perception(lane_right_points[:, 0], max_bad_real_states=3)
        # params["bad_perception"] = bad_left_perception and bad_right_perception
        # # # TODO It is a known glitch. Remove when different environment than town04 long straight
        # # if params["bad_perception"] and not out:
        # if params["bad_perception"] and not has_crashed:
        #     print('bad perception')
        #  #   if self.failures < 3:
        #  #       self.failures += 1
        #     done = False
        #     # return self.apply_step(action)
        #     # return self.apply_step([0.1, 0.05 * random.choice([1, -1]), 0])
        #     #else:
        #     #    print("bad perception!")
        #     #    self.failures = 0
        #     #    done = True
        # #else:
        # #    self.failures = 0
        #
        if not all(x == 0 for x in states):
            # self.last_centers = x_midpoints.copy()
            self.last_centers = x_centers_normalized.copy()
        # print(np.array(states))
        return np.array(states), reward, done, done, params

    def calculate_recommended_speed(self, right_center_lane, f=0.15, e=0.06):
        # Step 1: Curvature
        x = np.array(self.x_row)
        y = np.array(right_center_lane)

        coefficients = np.polyfit(x, y, 2)
        a, b, c = coefficients

        x_mid = x[2]
        y_prime = 2 * a * x_mid + b
        y_double_prime = 2 * a
        curvature = abs(y_double_prime) / ((1 + y_prime ** 2) ** (3 / 2))

        # Step 2: Convert to Radius
        radius = 1 / curvature if curvature != 0 else float('inf')  # Handle straight lines

        # Step 3: Compute recommended velocity
        g = 9.81  # m/s²
        v_squared = radius * g * (f + e)
        v = np.sqrt(v_squared)

        return v  # Velocity in m/s

    ################################################################################
    def step(self, action):
        # time.sleep(1)
        self.step_count += 1

        # test code to verify the initial position
        # for _ in range(100):
        #     time.sleep(0.2)
        #     states, reward, done, done, params = self.apply_step([0.5, 0.0])

        # if self.step_count <= 30 and self.step_count % 10 == 0:
        #     self.set_init_speed()

        # TODO OJO, puede dañar el entrenamiento, pero con eso nos aseguramos
        # que trabaja con esta velocidad una vez aterriza en el suelo y se
        # endereza
        try:
            if 5 < self.step_count <= 10:
                self.set_init_speed()

            # for _ in range(1):  # Apply the action for 3 consecutive steps
            states, reward, done, done, params = self.apply_step(action)
        except Exception as e:
            print(f"Waiting for CARLA to become available to step... after {e}")
            traceback.print_exc()
            self.reconnect_to_carla()
            states, reward, done, done, params = self.step(action)

        return states, reward, done, done, params

    def control(self, action):
        if float(action[0]) < 0:
            brake = -float(action[0])
            throttle = 0
        else:
            brake = 0
            throttle = float(action[0])

        if self.stage in ("r", "v"):
            self.car.apply_control(carla.VehicleControl(throttle=throttle, brake=brake,
                                                        steer=float(action[1])))
        else:
            action[0] = self.fixed_random_throttle
            self.car.apply_control(carla.VehicleControl(
                throttle=float(action[0]),
                steer=float(action[1])))

            if self.step_count == 10 or self.step_count == 20:
                transform = self.car.get_transform()
                forward_vector = transform.get_forward_vector()

                # Scale the forward vector by the desired speed
                target_velocity = carla.Vector3D(
                    x=forward_vector.x * self.speed,
                    y=forward_vector.y * self.speed,
                    z=forward_vector.z * self.speed  # Typically 0 unless you want vertical motion
                )
                self.car.set_target_velocity(target_velocity)

        params = {}

        v = self.car.get_velocity()
        params["velocity"] = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        # params["angular_velocity"] = self.car.get_angular_velocity().z
        params["angular_velocity"] = self.get_lateral_velocity()

        w_angle = self.car.get_control().steer
        params["steering_angle"] = w_angle

        return params

    def get_lateral_velocity(self):
        # Get velocity and rotation
        velocity = self.car.get_velocity()
        transform = self.car.get_transform()
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Create unit vector for vehicle's **right** direction
        right_vector = np.array([
            -np.sin(yaw_rad),  # x component
            np.cos(yaw_rad),  # y component
            0
        ])

        # Vehicle velocity vector
        vel_vector = np.array([velocity.x, velocity.y, velocity.z])

        # Project velocity onto right_vector
        lateral_vel = np.dot(vel_vector, right_vector)
        return lateral_vel

    def scale_velocity(self, velocity, v_max=30):
        # Logarithmic scaling formula
        return math.log(1 + velocity) / math.log(1 + v_max)

    def rewards_easy(self, center_distance, action, params, v_goal=None, x_centers_normalized=None):
        # distance_error = error[3:]  # We are just rewarding the 3 lowest points!
        # distance_error = [abs(x) for x in centers_distances]

        ## EARLY RETURNS
        params["distance_error"] = center_distance
        params["d_reward"] = 0
        params["v_reward"] = 0
        params["v_eff_reward"] = 0
        params["reward"] = 0

        # car_deviated_punish = -100 if self.stage == "w" else -5 * max(0, action[0])
        car_deviated_punish = -20
        lane_changed_punish = -1

        # TODO (Ruben) OJO! Que tienen que ser todos  < 0.3!! Revisar si esto no es demasiado restrictivo
        #  En curvas
        # done, states_above_threshold = self.has_bad_perception(distance_error, self.reset_threshold,
        #                                                        len(distance_error) // 3)
        #if done:
        #    print(f"car deviated after step {self.step_count}")
        #    self.deviated += 1
        #    return car_deviated_punish, done, False
        # return -5 * action[0], done, crash

        if self.is_out(center_distance):
            return car_deviated_punish, True, self.has_crashed()

        # if x_centers is not None and self.centers_switched(x_centers):
        #     return lane_changed_punish, True, False

        # DISTANCE REWARD CALCULATION
        # d_rewards = []
        # for _, dist_error in enumerate(distance_error):
        #     if dist_error is None or 0 > dist_error or dist_error > 1:
        #         continue
        #     d_rewards.append(math.pow((1 - dist_error), 3))
        #
        # # TODO ignore non detected centers
        # d_reward = 0 if len(d_rewards) == 0 else sum(d_rewards) / len(d_rewards)
        d_reward = (1 - abs(center_distance)) ** 3

        v = params["velocity"]

        # REWARD CALCULATION
        beta = self.beta
        if v < self.punish_ineffective_vel:
            self.steps_stopped += 1
            if self.steps_stopped > 100:
                print("too much time stopped")
                return car_deviated_punish, True, False
            # return car_deviated_punish + (action[0] * d_reward), False, False
            reward = action[0] * d_reward/5
            reward -= self.calculate_punish(params, action, v_goal, v, center_distance, x_centers_normalized)
            return reward, False, False
        else:
            self.steps_stopped = 0

        # print(f"monitoring last modification bro! current_v -> {params['velocity']}")
        v_eff_reward = self.calculate_v_reward(v_goal, v, d_reward)

        if self.stage in ("v", "r"):
            if v > 32:
                return -1 * max(action[0], 0), True, False
            if v > 28:
                v_eff_reward = 0
                # v_eff_reward = -max(action[0], 0)

        # TOTAL REWARD CALCULATION
        beta = 1 if self.stage == "w" else beta
        d_reward_component = beta * d_reward
        # d_reward_component = beta * pos_reward
        v_reward_component = (1 - beta) * v_eff_reward
        # progress_reward_component = advanced * 0.01
        # aligned_component = abs(0.5 - np.mean(x_centers_normalized)) * 5

        function_reward = d_reward_component + v_reward_component

        # PUNISH CALCULATION
        function_reward -= self.calculate_punish(params, action, v_goal, v, center_distance, x_centers_normalized)

        self.episode_v_eff_reward = self.episode_v_eff_reward + (
                v_eff_reward - self.episode_v_eff_reward) / self.step_count
        params["v_eff_reward"] = v_eff_reward
        # function_reward = d_reward * v_reward
        params["reward"] = function_reward

        self.car.error = center_distance

        self.episode_d_reward = self.episode_d_reward + (d_reward - self.episode_d_reward) / self.step_count
        self.episode_d_deviation = self.episode_d_deviation + (center_distance - self.episode_d_deviation) / self.step_count
        if center_distance < 0.05:
            # Step 1: Increase step count first
            self.step_count_no_curves += 1
            # --- Throttle action stats using Welford's algorithm ---
            delta = action[0] - self.throttle_action_avg_no_curves
            self.throttle_action_avg_no_curves += delta / self.step_count_no_curves
            delta2 = action[0] - self.throttle_action_avg_no_curves
            self.throttle_action_variance_no_curves += delta * delta2
            if self.step_count_no_curves > 1:
                self.throttle_action_std_dev_no_curves = (
                                                                 self.throttle_action_variance_no_curves / (
                                                                     self.step_count_no_curves - 1)
                                                         ) ** 0.5
            else:
                self.throttle_action_std_dev_no_curves = 0.0  # Not enough data yet
            # --- Absolute angular velocity (action[1]) average ---
            self.abs_w_no_curves_avg += (
                                                abs(action[1]) - self.abs_w_no_curves_avg
                                        ) / self.step_count_no_curves

        else:
            self.curves_states += 1
            # Step 1: Increase step count first
            self.step_count_curves += 1
            delta = action[0] - self.throttle_action_avg_curves
            self.throttle_action_avg_curves += delta / self.step_count_curves
            delta2 = action[0] - self.throttle_action_avg_curves
            self.throttle_action_variance_curves += delta * delta2
            if self.step_count_curves > 1:
                self.throttle_action_std_dev_curves = (self.throttle_action_variance_curves / (
                            self.step_count_curves - 1)) ** 0.5
            else:
                self.throttle_action_std_dev_curves = 0.0  # Not enough data yet

        if self.step_count > self.estimated_steps:
            print("episode finished")
            return function_reward, True, False

        return function_reward, False, False

    def slice_image(self, red_mask):
        height = red_mask.shape[0]
        image_middle_line = (height) // 2
        img_sliced = red_mask[image_middle_line:]
        return img_sliced.copy()

    def get_resized_image(self, sensor_data, new_width=640):
        # Assuming sensor_data is the image obtained from the sensor
        # Convert sensor_data to a numpy array or PIL Image if needed
        # For example, if sensor_data is a numpy array:
        # sensor_data = Image.fromarray(sensor_data)
        sensor_data = np.array(sensor_data, copy=True)

        # Get the current width and height
        try:
            height = sensor_data.shape[0]
            width = sensor_data.shape[1]
        except Exception:
            height = 540
            width = 640

        # Calculate the new height to maintain the aspect ratio
        new_height = int((new_width / width) * height)

        resized_img = Image.fromarray(sensor_data).resize((new_width, new_height))

        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        resized_img_np = np.array(resized_img)

        return resized_img_np

    def extract_green_lines(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for the green color in HSV space
        lower_green = np.array([35, 100, 200])
        upper_green = np.array([85, 255, 255])

        # Create a mask for the green color
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        return green_mask

    def draw_line_through_points(self, points, image):
        # Convert the points list to a format compatible with OpenCV (a numpy array)
        points = np.array(points, dtype=np.int32)  # Make sure points are integers

        # Draw a polyline through the points
        cv2.polylines(image, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        return image


    def find_waypoint_by_lane_id(self, lane_id, location, map_api):
        """
        Helper to find a waypoint near the current location with a specific lane_id.
        This can be improved based on how to search waypoints around the car.
        """
        # For simplicity, get waypoint at location but filter lane_id
        wp = map_api.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp.lane_id == lane_id:
            return wp
        # Could add more sophisticated search around location if needed
        return None

    def calculate_lateral_distance(self, location, transform):
        """
        Calculate lateral distance of location from lane center transform.
        """
        dx = location.x - transform.location.x
        dy = location.y - transform.location.y
        # Assuming lane direction is along transform.rotation.yaw
        yaw_rad = np.deg2rad(transform.rotation.yaw)
        # Project vector onto lateral axis (perpendicular to lane direction)
        lateral_dist = abs(-np.sin(yaw_rad) * dx + np.cos(yaw_rad) * dy)
        return lateral_dist

    def adjust_lane_polyline_for_offset(self, polyline, lateral_offset):
        adjusted_polyline = []
        for i in range(len(polyline) - 1):
            p1 = polyline[i]
            p2 = polyline[i + 1]

            direction = p2[:2] - p1[:2]
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue

            direction /= norm
            right_vec = np.array([-direction[1], direction[0]])

            p1_adjusted = p1[:2] + lateral_offset * right_vec
            adjusted_polyline.append(np.array([p1_adjusted[0], p1_adjusted[1], p1[2]]))

        # Include the last point using the last segment's direction
        if len(adjusted_polyline) > 0:
            p_last = polyline[-1]
            p_last_adjusted = adjusted_polyline[-1]  # Reuse last direction
            adjusted_polyline.append(np.array([p_last_adjusted[0], p_last_adjusted[1], p_last[2]]))

        return np.array(adjusted_polyline)

    def detect_center_line_perfect(self, ll_segment, num_points=20):
        ll_segment = cv2.cvtColor(ll_segment, cv2.COLOR_BGR2GRAY)

        height = ll_segment.shape[0]
        width = ll_segment.shape[1]

        trafo_matrix_global_to_camera = get_matrix_global(self.car, self.trafo_matrix_vehicle_to_cam)

        if self.k is None:
            self.k = get_intrinsic_matrix(120, width, height)

        _, center_distance, alignment = self.get_lane_position(self.car, self.map)
        opposite = alignment < 0.5
        misalignment = (1 - abs(alignment)) * 10

        center_list, current_wp = (
            self.get_stable_lane_lines(opposite=opposite))

        if center_list is None or len(center_list) < 2:
            interpolated_center = np.full((num_points, 2), self.NON_DETECTED)
        else:
            projected_center = project_polyline(
                center_list, trafo_matrix_global_to_camera, self.k, image_shape=ll_segment.shape
            ).astype(np.int32)

            # Discard non visible points from projected centers
            h, w = ll_segment.shape[:2]
            mask = np.array([
                0 <= pt[0] < w and 0 <= pt[1] < h
                for pt in projected_center
            ])

            if np.sum(mask) < 2:
                interpolated_center = np.full((num_points, 2), self.NON_DETECTED)
            else:
                # Apply the same mask to both 2D and 3D data
                visible_center = projected_center[mask]
                # In lane_points just remove the points that are not visible because they are behind the car
                for i, keep in enumerate(mask):
                    if keep:
                        first_true_index = i
                        break
                else:
                    # If mask is all False, remove all points
                    first_true_index = len(mask)
                self.lane_points["center"] = self.lane_points["center"][first_true_index:]

                interpolated_center = interpolate_lane_points(visible_center, num_points)

        # If interpolation failed, return dummy points
        return ll_segment, misalignment, center_distance, interpolated_center

    def get_stable_lane_lines(self, segment_length: int = 80,
                              opposite: bool = False,
                              exclude_junctions: bool = False,
                              only_turns: bool = False):
        """
        Returns a stable rolling lane poly-line.
        On the first call it builds the N-point list from the car's current waypoint.
        On subsequent calls it only advances the list when the car has really moved
        past the first stored point, preventing abrupt lane switches.

        Returns
        -------
        center_list : np.ndarray  shape=(N,3)
        left_boundary : np.ndarray
        right_boundary : np.ndarray
        current_wp : carla.Waypoint       (the new reference waypoint)
        """

        # ------------------------------------------------------------------
        # 1)  Build the poly-line once
        # ------------------------------------------------------------------
        if self.lane_points is None or self.step_count < 30:
            wp = self.map.get_waypoint(
                self.car.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            center, left_b, right_b, last_wp = create_lane_lines(
                wp, opposite=opposite,
                exclude_junctions=exclude_junctions,
                only_turns=only_turns)

            # Keep both numpy arrays *and* the last carla.Waypoint for fast extension
            self.lane_points = {
                "center": center.tolist(),  # lists are easier to pop/append
                "last_wp": last_wp  # last waypoint object
            }

        while len(self.lane_points["center"]) < 90:
            last_wp = self.lane_points["last_wp"]

            # Try to move forward or backward depending on direction
            if opposite:
                next_wps = last_wp.previous(1.0)
            else:
                next_wps = last_wp.next(1.0)

            if not next_wps:
                break  # No more waypoints available, cannot extend further

            # Get new waypoint
            new_wp = next_wps[0]
            self.lane_points["last_wp"] = new_wp  # update stored last_wp

            # Compute new center/left/right lane points
            center_np = carla_vec_to_np_array(new_wp.transform.location)

            self.lane_points["center"].append(center_np.tolist())

        # ------------------------------------------------------------------
        # 3)  Return as numpy arrays
        # ------------------------------------------------------------------
        center_arr = np.asarray(self.lane_points["center"])
        return center_arr, self.lane_points["last_wp"]


    def get_lane_position(self, vehicle: carla.Vehicle, map: carla.Map):
        """
        Determines the vehicle's position relative to the lane.

        Returns:
            A dictionary containing:
            - lane_side: "left", "right", or "center"
            - lane_offset: Distance from the lane center (positive: left, negative: right)
            - lane_alignment: Alignment of vehicle and lane directions (1: aligned, -1: opposite)
        """

        waypoint = map.get_waypoint(
            vehicle.get_transform().location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # Vehicle's forward vector
        vehicle_forward = vehicle.get_transform().get_forward_vector()
        vehicle_forward_np = np.array([vehicle_forward.x, vehicle_forward.y])

        # Lane's forward vector
        waypoint_forward = waypoint.transform.get_forward_vector()
        waypoint_forward_np = np.array([waypoint_forward.x, waypoint_forward.y])

        # Vector from waypoint to vehicle
        vehicle_location = vehicle.get_transform().location
        waypoint_location = waypoint.transform.location
        waypoint_to_vehicle = carla.Location(
            vehicle_location.x - waypoint_location.x,
            vehicle_location.y - waypoint_location.y,
            vehicle_location.z - waypoint_location.z
        )
        waypoint_to_vehicle_np = np.array([waypoint_to_vehicle.x, waypoint_to_vehicle.y])

        # 1. Lane Side (Left/Right)
        cross_product = np.cross(vehicle_forward_np, waypoint_forward_np)
        lane_side = "center"
        if cross_product > 0.1:
            lane_side = "left"
        elif cross_product < -0.1:
            lane_side = "right"

        # 2. Lane Offset (Distance to Center)
        # Project waypoint_to_vehicle onto a vector perpendicular to lane_forward
        lane_right_np = np.array([-waypoint_forward_np[1], waypoint_forward_np[0]])  # 90-degree rotation
        lane_offset = np.dot(waypoint_to_vehicle_np, lane_right_np)
        lane_offset /= np.linalg.norm(lane_right_np)

        # 3. Lane Alignment (Forward/Backward)/home/ruben/Desktop/2020-phd-ruben-lucas/src/RL-Studio/rl_studio/checkpoints/follow_lane_carla_sac_auto_carla_baselines/20250506-082233/best_model.zip
        lane_alignment = np.dot(vehicle_forward_np, waypoint_forward_np)

        return lane_side, lane_offset, lane_alignment



    def show_ll_seg_image(self, centers, ll_segment, suffix="", name='ll_seg'):
        ll_segment_stacked = self.get_stacked_image(centers, ll_segment)
        # We now show the segmentation and center lane postprocessing
        self.show_image(name + suffix, ll_segment_stacked)

    def has_crashed(self):
        if len(self.collision_hist) > 0:  # te has chocado, baby
            print("crash!")
            return True
        return False

    def has_invaded(self):
        if len(self.invasion_hist) > 0:  # te has chocado, baby
            print("lane invaded!")
            return True
        return False

    def is_out(self, center_distance):
        # if self.has_invaded() or self.has_crashed() or abs(center_distance) > 1:  # te has chocado, baby
        if self.has_crashed() or abs(center_distance) > 1:  # te has chocado, baby
            self.deviated += 1
            return True

        return False

    def set_init_pose(self):
        ## ---  Car
        waypoints_town = self.world.get_map().generate_waypoints(5.0)
        # set self driving car
        if self.alternate_pose:
            self.car, init_pose = self.setup_car_random_pose(self.spawn_points)
        elif self.waypoints_init is not None:
            init_waypoint = waypoints_town[self.waypoints_init]
            if self.show_all_points:
                self.draw_waypoints(
                    waypoints_town,
                    self.waypoints_init,
                    self.waypoints_target,
                    self.waypoints_lane_id,
                    2000,
                )
            self.car = self.setup_car_fix_pose(init_waypoint)
            init_pose = init_waypoint.transform
        else:  # TODO: hacer en el caso que se quiera poner el target con .next()
            init_waypoint = waypoints_town[self.waypoints_init]
            waypoints_lane = init_waypoint.next_until_lane_end(1000)
            waypoints_next = init_waypoint.next(1000)
            print(f"{init_waypoint.transform.location.x = }")
            print(f"{init_waypoint.transform.location.y = }")
            print(f"{init_waypoint.lane_id = }")
            print(f"{init_waypoint.road_id = }")
            print(f"{len(waypoints_lane) = }")
            print(f"{len(waypoints_next) = }")
            w_road = []
            w_lane = []
            for x in waypoints_next:
                w_road.append(x.road_id)
                w_lane.append(x.lane_id)

            counter_lanes = Counter(w_lane)
            counter_road = Counter(w_road)
            print(f"{counter_lanes = }")
            print(f"{counter_road = }")

            self.car = self.setup_car_fix_pose(init_waypoint)

        # set other car
        if self.front_car not in (None, "none"):
            self.setup_car_random_pose(self.front_car_spawn_points)

        ## --- Sensor collision
        self.setup_col_sensor(self.car)
        self.setup_lane_sensor(self.car)
        # self.add_obstacle_detector(self.car, self.world)
        self.add_lidar_to_vehicle(self.world, self.car)

        # Create a camera sensor blueprint
        # camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        # camera_bp.set_attribute("image_size_x", "800")
        # camera_bp.set_attribute("image_size_y", "600")
        # camera_bp.set_attribute("fov", "90")


        self.birds_eye_camera = SensorManager(
            self.world,
            self.display_manager,
            "BIRD_VIEW",
            carla.Transform(carla.Location(x=0, y=0, z=100), carla.Rotation(pitch=-90)),
            self.car,
            {},
            display_pos=[1, 0],
        )

        self.front_camera_1_5 = SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(
                carla.Location(x=2, y=0, z=2),
                carla.Rotation(pitch=-2, yaw=0, roll=0.0)
            ),
            self.car,
            {},
            display_pos=[0, 0],
        )

        self.front_camera_1_5_segmentated = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCamera",
            carla.Transform(
                carla.Location(x=2, y=0, z=2),
                carla.Rotation(pitch=-2, yaw=0, roll=0.0)
            ),
            self.car,
            {},
            display_pos=[0, 1],
        )

        self.ego_camera = SensorManager(
            self.world,
            self.display_manager,
            "BIRD_VIEW",
            carla.Transform(carla.Location(x=-2, y=0, z=4), carla.Rotation(pitch=-10)),
            self.car,
            {},
            display_pos=[1, 1],
        )

        # self.front_camera_1_5_red_mask = SensorManager(
        #     self.world,
        #     self.display_manager,
        #     "RedMask",
        #     carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+0)),
        #     self.car,
        #     {},
        #     display_pos=[0, 2],
        # )

        return init_pose

    def merge_and_extend_lines(self, lines, ll_segment):
        # Merge parallel lines
        merged_lines = []
        for line in lines if lines is not None else []:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Compute the angle of the line

            # Check if there is a similar line in the merged lines
            found = False
            for merged_line in merged_lines:
                angle_diff = abs(merged_line['angle'] - angle)
                if angle_diff < 20 and abs(angle) > 25:  # Adjust this threshold based on your requirement
                    # Merge the lines by averaging their coordinates
                    merged_line['x1'] = (merged_line['x1'] + x1) // 2
                    merged_line['y1'] = (merged_line['y1'] + y1) // 2
                    merged_line['x2'] = (merged_line['x2'] + x2) // 2
                    merged_line['y2'] = (merged_line['y2'] + y2) // 2
                    found = True
                    break

            if not found and abs(angle) > 25:
                merged_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'angle': angle})

        # Draw the merged lines on the original image
        merged_image = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8
        # if len(merged_lines) < 2 or len(merged_lines) > 2:
        #    print("ii")
        for line in merged_lines:
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            cv2.line(merged_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the original image with merged lines
        cv2.imshow('Merged Lines', merged_image) if self.sync_mode and self.show_images else None

        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Step 5: Perform linear regression on detected lines
        # Iterate over detected lines
        for line in merged_lines if lines is not None else []:
            # Extract endpoints of the line
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']

            # Fit a line to the detected points
            vx, vy, x0, y0 = cv2.fitLine(np.array([[x1, y1], [x2, y2]], dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculate the slope and intercept of the line
            slope = vy / vx

            # Extend the line if needed (e.g., to cover the entire image width)
            extended_y1 = ll_segment.shape[0] - 1  # Bottom of the image
            extended_x1 = x0 + (extended_y1 - y0) / slope
            extended_y2 = 0  # Upper part of the image
            extended_x2 = x0 + (extended_y2 - y0) / slope

            if extended_x1 > 2147483647 or extended_x2 > 2147483647 or extended_y1 > 2147483647 or extended_y2 > 2147483647:
                cv2.line(line_mask, (int(x0), 0), (int(x0), ll_segment.shape[0] - 1), (255, 0, 0), 2)
                continue
            # Draw the extended line on the image
            cv2.line(line_mask, (int(extended_x1), extended_y1), (int(extended_x2), extended_y2), (255, 0, 0), 2)
        return line_mask

    # def has_bad_perception(self, distances_error, threshold=0.3, max_bad_real_states=1):
    #     done = False
    #     states_above_threshold = sum(1 for state_value in distances_error if state_value > threshold)
    #
    #     if states_above_threshold is None:
    #         states_above_threshold = 0
    #
    #     if states_above_threshold > max_bad_real_states:  # salimos porque no detecta linea a la derecha
    #         done = True
    #     return done, states_above_threshold

    def has_bad_perception(self, detected_points, max_bad_real_states=1):
        states_above_threshold = sum(1 for state_value in detected_points if state_value == 0)

        if states_above_threshold is None:
            states_above_threshold = 0

        if states_above_threshold > max_bad_real_states:  # salimos porque no detecta linea a la derecha
            return True
        return False

    def vary_car_orientation(self):
        """
        Slightly varies the orientation (yaw) of the vehicle.

        Args:
            vehicle: The carla.Vehicle object.
        """

        # Get the current transform (position and rotation) of the vehicle
        transform = self.car.get_transform()
        rotation = transform.rotation

        # Generate a small random yaw offset (e.g., between -5 and 5 degrees)
        yaw_offset = random.uniform(-8.0, 8.0)  # Adjust range as needed

        # and randomly just get the car around 180%
        if random.random() < 0.5:
            yaw_offset += 180.0

        # Create a new rotation with the modified yaw
        new_rotation = carla.Rotation(pitch=rotation.pitch, yaw=rotation.yaw + yaw_offset, roll=rotation.roll)

        # Create a new transform with the modified rotation
        new_transform = carla.Transform(transform.location, new_rotation)

        # Apply the new transform to the vehicle
        self.car.set_transform(new_transform)

    def is_facing_lane_direction(self, threshold_degrees=90):
        car_forward = self.car.get_transform().get_forward_vector()
        waypoint = self.map.get_waypoint(self.car.get_location())
        lane_forward = waypoint.transform.get_forward_vector()

        # Normalize vectors
        def normalize(v):
            mag = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            return carla.Vector3D(v.x / mag, v.y / mag, v.z / mag)

        car_f = normalize(car_forward)
        lane_f = normalize(lane_forward)

        dot = car_f.x * lane_f.x + car_f.y * lane_f.y + car_f.z * lane_f.z

        angle = math.degrees(math.acos(dot))

        return angle < threshold_degrees

    def set_init_speed(self):
        transform = self.car.get_transform()
        forward_vector = transform.get_forward_vector()
        rnd = random.random()
        # self.speed = 30
        # self.speed = 32 if rnd < 0.5 else random.randint(10, 36)
        self.speed = 0 if rnd < 0.5 else 20 \
            if rnd < 0.7 else random.randint(0, 26)
        # if not self.is_facing_lane_direction():
            # self.speed *= -1

        target_velocity = carla.Vector3D(
            x=forward_vector.x * self.speed,
            y=forward_vector.y * self.speed,
            z=0
        )
        self.car.set_target_velocity(target_velocity)

    def init_world(self):
        if isinstance(self.towns, list):
            for i in range(len(self.towns)):
                world = self.load_world(self.clients[i], self.towns[i])
                self.worlds.append(world)
            self.world = random.choice(self.worlds)
        else:
            self.world = self.load_world(self.client, self.towns)
            self.worlds.append(self.world)

    def load_world(self, client, town):
        # print(f"\n maps in carla 0.9.13: {client.get_available_maps()}\n")
        # traffic_manager = client.get_trafficmanager(self.config["manager_port"])
        world = client.load_world(town)
        print(f"loading world {town}")
        time.sleep(2.0)  # Needed to the simulator to be ready. TODO May be decrease to 1?
        self.load_world_settings(world)
        print(f"Current World Settings: {world.get_settings()}")
        return world

    def load_any_world(self):
        town = random.choice(self.towns)
        self.world = self.client.load_world(town)
        print(f"loading world {town}")
        time.sleep(2.0)  # Needed to the simulator to be ready. TODO May be decrease to 1?
        self.load_world_settings(self.world)
        # print(f"Current World Settings: {self.world.get_settings()}")
        self.map = self.world.get_map()

    def centers_switched(self, distances):
        """
        It returns true if more than half of coordinates (x or y) differs more than 10 pixels
        """
        if self.last_centers is None or self.step_count < 20:
            return False
        if all(x == 0 for x in distances):
            return False # Tolerate missing frames

        differents = 0
        for i in range(len(distances)):
            if abs(distances[i] - self.last_centers[i]) > 0.2:
                differents += 1
                if differents > len(distances) // 4:
                    print("centers switched!")
                    # print(f"from {self.last_centers} to {distances}!")
                    return True
        return False

    def reload_worlds(self):
        for client in self.clients:
            client.reload_world()
            world = client.get_world()
            self.load_world_settings(world)

    def load_world_settings(self, world):
        settings = world.get_settings()
        self.forced_freq = self.config.get("async_forced_delta_seconds")
        if self.sync_mode:
            settings.max_substep_delta_time = 0.02
            settings.fixed_delta_seconds = self.config.get("fixed_delta_seconds")
            settings.synchronous_mode = True
            # traffic_manager.set_synchronous_mode(True)
        # else:
        # traffic_manager.set_synchronous_mode(False)
        world.apply_settings(settings)

    def update_from_hot_config(self, config):
        self.debug_waypoints = config.get("debug_waypoints")
        self.visualize = config.get("visualize")

    def miss_detection(self, i, center_image):
        return [int((center_image * 2) - 1)] if self.no_detected[i][0] > center_image else [0]

    TRAINING_ID = os.getpid()  # or generate a UUID if needed
    REWARD_FILE_PATH = f"/tmp/rlstudio_reward_monitor_{TRAINING_ID}.json"

    def update_reward_monitor(self, avg_reward):
        try:
            with open(self.REWARD_FILE_PATH, "w") as f:
                json.dump({"avg_reward": avg_reward, "timestamp": time.time()}, f)
        except Exception as e:
            print(f"Could not write reward monitor file: {e}")

    def update_tensorboard_stats(self, tensorboard):
        tensorboard.update_stats(
            steps_episode=self.step_count,
            cum_rewards=self.cumulated_reward,
            crashed=self.deviated,
            d_reward=self.episode_d_reward,
            v_reward=self.episode_v_eff_reward,
            avg_speed=self.avg_speed,
            max_speed=self.max_speed,
            abs_w_no_curves_avg=self.abs_w_no_curves_avg,
            # throttle_difference=self.throttle_action_difference,
            # throttle_curves=self.throttle_action_avg_curves,
            # throttle_no_curves=self.throttle_action_avg_no_curves,
            # throttle_curves_std=self.throttle_action_std_dev_curves,
            # throttle_no_curves_std=self.throttle_action_std_dev_no_curves,
            # cum_d_reward=cum_d_reward,
            # max_reward=max_reward,
            # steering_std_dev=steering_std_dev,
            advanced_meters=self.advanced_meters,
            # actor_loss=self.actor_loss,
            # critic_loss=self.critic_loss,
            # cpu=self.cpu_usages,
            # gpu=self.gpu_usages,
            # collisions = self.crash,
            # completed = completed
        )

    def show_image(self, name, image):
        cv2.imshow(name, image)
        cv2.waitKey(1)  # 1 millisecond

    def initialize_carla(self):
        while self.clients is None:
            try:
                if isinstance(self.ports, list):
                    self.clients = [
                        init_client(self.carla_ip, port)
                        for port in self.ports
                    ]
                else:
                    self.client = init_client(self.carla_ip, self.ports)
                    self.clients = [self.client]
                self.towns = self.config["town"]

                self.worlds = []
                self.world = None
                #self.init_world()
                self.load_any_world()
            except Exception as e:
                print(f"Waiting for CARLA server... after {e}")
                traceback.print_exc()
                # time.sleep(20)

    def get_stacked_image(self, centers, ll_segment):
        if self.detection_mode == "carla_perfect":
            ll_segment_int8 = ll_segment
        else:
            ll_segment_int8 = (ll_segment * 255).astype(np.uint8)

        height, width = ll_segment.shape

        ll_segment_all = [np.copy(ll_segment_int8), np.copy(ll_segment_int8), np.copy(ll_segment_int8)]

        # Draw the midpoint used as right center lane
        # for index, dist in zip(self.x_row, centers):
        #     add_midpoints(ll_segment_all[0], index, dist)
        # for x, y in centers:
        #     if 0 <= y < ll_segment_all[0].shape[0] and 0 <= x < ll_segment_all[0].shape[1]:
        #         add_midpoints(ll_segment_all[2], x, y)

        # Draw a line for the selected perception points
        # for index in self.x_row:
        #     for i in range(630):
        #         ll_segment_all[0][index][i] = 255

        # Add interpolated lane points to the first channel
        for pt in centers:
            x, y = int(pt[0]), int(pt[1])
            add_midpoints(ll_segment_all[2], y, x)
        # for pt in interpolated_right:
        #     x, y = int(pt[0]), int(pt[1])
        #         add_midpoints(ll_segment_all[2], y, x)

        return np.stack(ll_segment_all, axis=-1)

    def reconnect_to_carla(self):
        # time.sleep(20)
        self.clients = None
        try:
            self.initialize_carla()
        except Exception as e:
            print(f"Waiting for CARLA to become available to reconnect... after {e}")
            traceback.print_exc()
            self.reconnect_to_carla()

    def update_all_tensorboard_stats(self):
        self.update_tensorboard_stats(self.tensorboard)
        if self.spawn_points is None:
            return
        if self.start_location_tag not in self.tensorboard_location_writers:
            self.tensorboard_location_writers[self.start_location_tag] = ModifiedTensorBoard(
                log_dir=f"{self.tensorboard_logs_dir}/{self.start_location_tag}"
            )
        self.update_tensorboard_stats(self.tensorboard_location_writers[self.start_location_tag])
        # self.tensorboard.update_fps(self.step_fps)

    def calculate_punish(self, params, action, v_goal, v, center_distance, x_centers_normalized):
        punish = 0

        # is_centered = abs(x_centers_normalized[0] - x_centers_normalized[-1]) <= 0.3
        # if params["final_curvature"] < 0.02 or is_centered:
        punish += self.punish_zig_zag_value * abs(action[1])
        self.car.zig_zag_punish = punish
        # punish += self.car.zig_zag_punish * abs(action[1] - self.previous_action[1]) * 10

        if abs(center_distance) > 0.2 and v > 15:
            punish += (v - 15) * 0.5
            self.car.v_punish = (v - 15) * 0.5

        if v_goal > v + 3 and action[0] < 0.4:
            punish += 1
            self.car.v_punish += 1

        if v_goal < v + 3 and action[0] > 0.7:
            punish += 1
            self.car.v_punish += 1

        if self.stage in ("v", "r"):
            punish += 1 if action[0] > 0.95 else 0

        return punish

    def calculate_v_reward(self, v_goal, v, d_reward):
        # Sigmoid v_difference_error .......................
        # Center at 10, stretch across 0–20
        # center = 10
        # scale = 1 / 2.0  # adjust this for more/less steepness
        # v_difference_error = 1 / (1 + np.exp(-scale * (diff - center))) # Sigmoid ....
        # v_component = 1 - v_difference_error
        # diff = max(0.5, abs(v_goal - float(v)))
        # v_component = (1 / diff) * 10
        diff = abs(v_goal - float(v))
        v_difference_error = diff / 2 # proportion error related to v_goal
        v_component = max(0, 10 - v_difference_error)
        v_eff_reward = v_component * d_reward

        self.car.v_component = v_component
        self.car.d_reward = d_reward
        self.car.v_eff_reward = v_eff_reward

        # print(f"----------------------------------------")
        # print(f" d_reward = {d_reward}")
        # print(f"(v_component = {v_component} | v_reward = {v_eff_reward}")
        # print(f"----------------------------------------")
        return v_eff_reward


def init_client(ip, port):
    client = carla.Client(
        ip,
        port,
    )
    client.set_timeout(20.0)
    return client

def get_car_location_tag(transform):
    location = transform.location
    rotation = transform.rotation
    x, y, z = location.x, location.y, location.z
    pitch, roll, yaw = rotation.pitch, rotation.roll, rotation.yaw
    return f"x{int(x)}_y{int(y)}_z{int(z)}_p{int(pitch)}_r{int(roll)}_y{int(yaw)}";

def carla_vec_to_np_array(vec):
    """
    Converts a CARLA Location or Vector3D into a NumPy array [x, y, z].
    """
    return np.array([vec.x, vec.y, vec.z])