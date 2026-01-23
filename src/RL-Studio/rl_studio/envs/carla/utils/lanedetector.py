import cv2
import torch
import numpy as np
import carla
from rl_studio.envs.carla.utils.logger import logger
import os

from collections import Counter


def choose_lane(distance_to_center_normalized, center_points):
    close_lane_indexes = [min(enumerate(inner_array), key=lambda x: abs(x[1]))[0] for inner_array in
                          distance_to_center_normalized]
    distances = [array[index] for array, index in zip(distance_to_center_normalized, close_lane_indexes)]
    centers = [array[index] for array, index in zip(center_points, close_lane_indexes)]
    return distances, centers

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

def calculate_midpoints(input_array):
    midpoints = []
    for i in range(0, len(input_array) - 1, 2):
        midpoint = (input_array[i] + input_array[i + 1]) // 2
        midpoints.append(midpoint)
    return midpoints


def getTransformFromPoints(points):
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

def normalize_centers(centers):
    x_centers = centers[:, 0]
    x_centers_normalized = (x_centers / 640).tolist()
    states = x_centers_normalized
    y_centers = centers[:, 1]
    y_centers_normalized = (y_centers / 512).tolist()
    states = states + y_centers_normalized # Returns a list
    return states, x_centers_normalized, y_centers_normalized


NO_DETECTED = 0

def select_device(device='', batch_size=None):
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

    return torch.device('cuda:0' if cuda else 'cpu')


class LaneDetector:
    def __init__(self, **config):
        self.config = config
        self.detection_mode = config.get("detection_mode")
        self.device = select_device()
        self.transform = None
        self.yolop_model = None
        self.masked_ll_segment = None

        if self.detection_mode == 'yolop_v2':
            import torchvision.transforms as transforms
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.yolop_model = torch.jit.load(
                "envs/carla/utils/yolop/weights/yolopv2.pt",
                map_location=self.device
            ).float()
            self.yolop_model.to(self.device)
            self.yolop_model.eval()
        else:
            raise ValueError(f"Detection mode '{self.detection_mode}' is not supported by this LaneDetector.")

    def detect_lanes(self, image):
        if self.detection_mode == 'yolop_v2':
            return self._detect_lanes_yolop_v2(image)
        return None, None, None

    def _detect_lanes_yolop_v2(self, image):
        # 1. Model Inference (Unchanged)
        img_tensor = self.transform(image).to(self.device).unsqueeze(0)
        with torch.no_grad():
            results = self.yolop_model(img_tensor)
            drivable_area_mask = results[1]

        ll_seg_mask = torch.argmax(drivable_area_mask, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        height, width = ll_seg_mask.shape
        center_x_image = width // 2

        # --- NEW: SANITIZATION PRE-PROCESS ---
        # A. Morphological Opening (Removes small noise/dust)
        kernel = np.ones((5, 5), np.uint8)
        # ll_seg_mask = cv2.morphologyEx(ll_seg_mask, cv2.MORPH_OPEN, kernel)
        # B. Morphological Closing (Fills small holes inside the lane)
        ll_seg_mask = cv2.morphologyEx(ll_seg_mask, cv2.MORPH_CLOSE, kernel)

        # 2. ISOLATE EGO-LANE (Consensus Filter)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ll_seg_mask, connectivity=8)
        best_label = -1
        min_dist = float('inf')

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 500: continue  # Ignore tiny artifacts

            cx, cy = centroids[i]
            dist = abs(cx - center_x_image) + abs(cy - height)
            if dist < min_dist:
                min_dist = dist
                best_label = i

        if best_label == -1:
            # FALLBACK: If we lost the lane, use previous fit if available
            return self.handle_detection_failure(ll_seg_mask)

        ego_mask = (labels == best_label).astype(np.uint8)

        # 3. SCAN INNER BOUNDARIES WITH WIDTH CONSTRAINT
        filtered_points = []
        # Only look at the bottom half of the image (more reliable)
        scan_limit = int(height * 0.5)

        for y in range(height - 1, scan_limit, -5):
            row = ego_mask[y, :]
            detected_pixels = np.where(row > 0)[0]

            if len(detected_pixels) > 20:  # Ensure the road is wide enough to be real
                left_edge = detected_pixels[0]
                right_edge = detected_pixels[-1]

                # SANITY CHECK: If the 'lane' is too wide (e.g. 80% of image),
                # we are likely at an intersection; use a weighted center instead
                lane_width = right_edge - left_edge
                if lane_width > width * 0.8:
                    # Weighted center: Bias toward the side the car is already on
                    center_x = (left_edge + right_edge) // 2
                else:
                    center_x = (left_edge + right_edge) // 2

                filtered_points.append([center_x, y])

        if len(filtered_points) < 5:
            return self.handle_detection_failure(ll_seg_mask)

        filtered_points = np.array(filtered_points)

        # 4. ROBUST POLYNOMIAL SMOOTHING
        try:
            # Use Weighted Fit: Give more weight to points closer to the car
            y_coords = filtered_points[:, 1]
            x_coords = filtered_points[:, 0]
            weights = np.linspace(1, 0.1, len(y_coords))  # Near points (bottom) = weight 1, far = 0.1

            fit = np.polyfit(y_coords, x_coords, 2, w=weights)
            curve_fn = np.poly1d(fit)

            # Temporal Smoothing: 80% new, 20% old (prevents jitter)
            if hasattr(self, 'prev_fit'):
                fit = 0.8 * fit + 0.2 * self.prev_fit
                curve_fn = np.poly1d(fit)
            self.prev_fit = fit

            plot_y = np.linspace(y_coords.min(), height - 1, 30)
            plot_x = curve_fn(plot_y)
            poly_points = np.column_stack((plot_x, plot_y)).astype(np.int32)
        except:
            poly_points = filtered_points

        # 5. VISUALIZATION
        self.masked_ll_segment = ll_seg_mask * 255
        debug_image = cv2.cvtColor(self.masked_ll_segment, cv2.COLOR_GRAY2BGR)
        for pt in filtered_points:
            cv2.circle(debug_image, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
        if len(poly_points) > 0:
            cv2.polylines(debug_image, [poly_points], False, (0, 255, 255), 2)

        cv2.imshow("Sanitized Lane Detection", debug_image)
        cv2.waitKey(1)

        return ll_seg_mask, poly_points, filtered_points

    def handle_detection_failure(self, current_mask=None):
        """Fallback logic when YOLOP sees nothing."""
        if hasattr(self, 'prev_fit'):
            # Extrapolate based on last known curve
            curve_fn = np.poly1d(self.prev_fit)
            if current_mask is not None:
                h, w = current_mask.shape
            else:
                h, w = 720, 1280
            plot_y = np.linspace(h // 2, h - 1, 30)
            plot_x = curve_fn(plot_y)
            poly_points = np.column_stack((plot_x, plot_y)).astype(np.int32)
            if current_mask is not None:
                return current_mask, poly_points, poly_points
            else:
                return None, poly_points, poly_points
        if current_mask is not None:
            return current_mask, [], []
        else:
            return None, [], []

    def _find_lane_center(self, mask, i, center_image):
        # Find the indices of 1s in the array
        mask_array = np.array(mask)
        indices = np.where(mask_array > 0.1)[0]

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

    def _calculate_center(self, mask):
        width = mask.shape[1]
        center_image = width / 2
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x__row)]
        # ## As we drive in the right lane, we get from right to left
        # lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from the right lane to center
        center_lane_indexes = [
            self._find_lane_center(lines[x], x, center_image) for x, _ in enumerate(lines)
        ]

        # this part consists of checking the number of lines detected in all rows
        # then discarding the rows (set to 1) in which more or less centers are detected
        center_lane_indexes = discard_not_confident_centers(center_lane_indexes)

        center_lane_distances = [
            [center_image - x for x in inner_array] for inner_array in center_lane_indexes
        ]

        # Calculate the average position of the lane lines
        ## normalized distance
        distances_to_center_normalized = [
            np.array(x) / (width - center_image) for x in center_lane_distances
        ]

        return center_lane_indexes, distances_to_center_normalized
    
    
    def get_resized_image(self, sensor_data, target_w=640, target_h=384):
        # Check if it's a CARLA raw sensor object or already a numpy array
        if hasattr(sensor_data, 'raw_data'):
            # Case: Raw CARLA sensor data
            array = np.frombuffer(sensor_data.raw__data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
            rgb_image = array[:, :, :3]
        else:
            # Case: It's already a numpy array (e.g. from a previous cv2 step)
            rgb_image = sensor_data
            # If it has 4 channels (BGRA), drop the 4th
            if rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[:, :, :3]

        # Optimized Resize using OpenCV
        resized_img = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return resized_img

