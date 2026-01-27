import argparse
import math
import os, sys
import shutil
import time
from pathlib import Path
from sklearn.linear_model import LinearRegression
from statsmodels.distributions.empirical_distribution import ECDF
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from collections import Counter

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from rl_studio.envs.carla.utils.DemoDataset import LoadImages, LoadStreams
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

# detection_mode = 'yolop'
# detection_mode = 'lane_detector'
# detection_mode = 'programmatic'

show_images = False
apply_mask = True

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

device = select_device()
images_high = 380
upper_limit = 200
x_row = [ upper_limit + 10, 270, 300, 320, images_high - 10 ]
NO_DETECTED = 0
THRESHOLDS_PERC = [0.1, 0.3, 0.5, 0.7, 0.9]
PERFECT_THRESHOLD = 0.9

from rl_studio.envs.carla.utils.yolop.YOLOP import get_net
import torchvision.transforms as transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
# INIT YOLOP
yolop_model = get_net()
checkpoint = torch.load("envs/carla/utils/yolop/weights/End-to-end.pth",
                        map_location=device)
yolop_model.load_state_dict(checkpoint['state_dict'])
yolop_model = yolop_model.to(device)

# Ensure you have the YOLOPv2 source files in your utils folder
# 1. Standard Transforms (Keep these, they are the same for v2)
import torchvision.transforms as transforms

yolop_v2_lines_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. INIT YOLOPv2
# Load the model directly using torch.jit or the model class
yolop_v2_lines_model = torch.jit.load(
    "envs/carla/utils/yolop/weights/yolopv2.pt",
    map_location=device
).float()  # Ensure it is in float for quality
yolop_v2_lines_model.to(device)
yolop_v2_lines_model.eval()  # CRITICAL for inference quality

lane_model_v3 = torch.load('envs/carla/utils/lane_det/best_model_torch.pth')
lane_model = None

def post_process(ll_segment):
    ''''
    Lane line post-processing
    '''
    # ll_segment = morphological_process(ll_segment, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # ll_segment = morphological_process(ll_segment, kernel_size=20, func_type=cv2.MORPH_CLOSE)
    # return ll_segment
    # ll_segment = morphological_process(ll_segment, kernel_size=4, func_type=cv2.MORPH_OPEN)
    # ll_segment = morphological_process(ll_segment, kernel_size=8, func_type=cv2.MORPH_CLOSE)

    # Step 1: Create a binary mask image representing the trapeze
    mask = np.zeros_like(ll_segment)
    # pts = np.array([[300, 250], [-500, 600], [800, 600], [450, 260]], np.int32)
    pts = np.array([[280, 200], [-50, 600], [630, 600], [440, 200]], np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))  # Fill trapeze region with white (255)
    cv2.imshow("applied_mask", mask) if show_images else None

    # Step 2: Apply the mask to the original image
    ll_segment_masked = cv2.bitwise_and(ll_segment, mask)
    ll_segment_excluding_mask = cv2.bitwise_not(mask)
    # Apply the exclusion mask to ll_segment
    ll_segment_excluded = cv2.bitwise_and(ll_segment, ll_segment_excluding_mask)
    cv2.imshow("discarded", ll_segment_excluded) if show_images else None

    return ll_segment_masked


def post_process_hough_lane_det_v1(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    #ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    #ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
    #edges = cv2.Canny(ll_segment, 50, 100)

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 90,  # Angle resolution in radians
        threshold=7,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=60  # Max allowed gap between line for joining them
    )
    # Sort lines by their length
    # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

    # Create a blank image to draw lines
    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Iterate over points
    for points in lines if lines is not None else []:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Postprocess the detected lines
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
    # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
    # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
    cv2.imshow("hough", line_mask) if show_images else None

    return lines

# TODO It is not feasible for online. Since it is calling predict thrice. Optimize
def post_process_hough_lane_det(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    # ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    # ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
    # edges = cv2.Canny(ll_segment, 50, 100)
    # Extract coordinates of non-zero points
    nonzero_points = np.argwhere(ll_segment == 255)
    if len(nonzero_points) == 0:
        return None

    # Extract x and y coordinates
    x = nonzero_points[:, 1].reshape(-1, 1)  # Reshape for scikit-learn input
    y = nonzero_points[:, 0]

    # Fit linear regression model
    model = LinearRegression()
    model.fit(x, y)

    # Predict y values based on x
    y_pred = model.predict(x)

    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Draw the linear regression line
    for i in range(len(x)):
        cv2.circle(line_mask, (x[i][0], int(y_pred[i])), 2, (255, 0, 0), -1)

    cv2.imshow("result", line_mask) if show_images else None

    # Find the minimum and maximum x coordinates
    min_x = np.min(x)
    max_x = np.max(x)

    # Find the corresponding predicted y-values for the minimum and maximum x coordinates
    y1 = int(model.predict([[min_x]]))
    y2 = int(model.predict([[max_x]]))

    # Define the line segment
    line_segment = (min_x, y1, max_x, y2)

    return line_segment


def post_process_hough_yolop_v1(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi/60,  # Angle resolution in radians
        threshold=8,  # Min number of votes for valid line
        minLineLength=8,  # Min allowed length of line
        maxLineGap=20  # Max allowed gap between line for joining them
    )

    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Draw the detected lines on the blank image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw lines in white (255, 255, 255)

    # Apply dilation to the line image

    # edges = cv2.Canny(line_mask, 50, 100)

    # cv2.imshow("intermediate_hough", ll_segment) if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 90,  # Angle resolution in radians
        threshold=35,  # Min number of votes for valid line
        minLineLength=15,  # Min allowed length of line
        maxLineGap=20  # Max allowed gap between line for joining them
    )
    # Sort lines by their length
    # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

    # Create a blank image to draw lines
    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Iterate over points
    for points in lines if lines is not None else []:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Postprocess the detected lines
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
    # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
    # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
    cv2.imshow("hough", line_mask)  if show_images else None

    return lines

def post_process_hough_yolop(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    cv2.imshow("preprocess", ll_segment) if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 90,  # Angle resolution in radians
        threshold=40,  # Min number of votes for valid line
        minLineLength=10,  # Min allowed length of line
        maxLineGap=30  # Max allowed gap between line for joining them
    )
    # Sort lines by their length
    # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

    # Create a blank image to draw lines
    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Iterate over points
    for points in lines if lines is not None else []:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Postprocess the detected lines
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
    # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
    # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
    cv2.imshow("hough", line_mask)  if show_images else None

    return lines


def post_process_hough_programmatic(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi/60,  # Angle resolution in radians
        threshold=20,  # Min number of votes for valid line
        minLineLength=10,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
    )

    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Draw the detected lines on the blank image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw lines in white (255, 255, 255)

    # Apply dilation to the line image

    edges = cv2.Canny(line_mask, 50, 100)

    cv2.imshow("intermediate_hough", edges)  if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 60,  # Angle resolution in radians
        threshold=20,  # Min number of votes for valid line
        minLineLength=13,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
    )
    # Sort lines by their length
    # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

    # Create a blank image to draw lines
    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Iterate over points
    for points in lines if lines is not None else []:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Postprocess the detected lines
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
    # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
    # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
    cv2.imshow("hough", line_mask) if show_images else None

    return lines

def detect_yolop(raw_image):
    # Get names and colors
    names = yolop_model.module.names if hasattr(yolop_model, 'module') else yolop_model.names

    # Run inference
    img = transform(raw_image).to(device)
    img = img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    det_out, da_seg_out, ll_seg_out = yolop_model(img)

    ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=int(1), mode='bicubic')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    return ll_seg_mask


# global variable for temporal smoothing
prev_fit = None


def run_yolop_v2_inference(raw_image):
    # Resize as your model expects
    resized_image = cv2.resize(raw_image, (640, 384))

    img = yolop_v2_lines_transform(resized_image).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    outputs = yolop_v2_lines_model(img)

    da_seg_out = outputs[1]
    ll_seg_out = outputs[2]

    return da_seg_out, ll_seg_out, resized_image


def process_yolop_v2_drivable_output(da_seg_out, resized_image):
    # Interpolate back to original resolution
    da_seg_mask = torch.nn.functional.interpolate(
        da_seg_out,
        size=(resized_image.shape[0], resized_image.shape[1]),
        mode='bilinear',
        align_corners=False
    )

    # Apply softmax and select the drivable class (usually class 1)
    da_seg_mask = torch.softmax(da_seg_mask, dim=1)
    da_seg_mask = da_seg_mask[0][1, :, :] # Select class 1 for drivable area

    # Squeeze and convert to numpy for OpenCV operations
    probs_np = da_seg_mask.squeeze().cpu().numpy()

    # Threshold to create a binary mask
    binary_mask = (probs_np > 0.5).astype(np.uint8) * 255

    # Morphological Closing to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask

def handle_detection_failure(ll_seg_mask):
    """
    Handles cases where lane detection fails.
    For the benchmark, we return empty lists, indicating no detection.
    """
    # Create an empty debug viz of the same size as the mask
    debug_image = np.zeros((ll_seg_mask.shape[0], ll_seg_mask.shape[1], 3), dtype=np.uint8)
    return np.zeros_like(ll_seg_mask), [], [], debug_image




def detect_yolop_v2_lines(raw_image, force_classical_fallback=False, ll_seg_out_from_inference=None):
    if ll_seg_out_from_inference is None:
        # Resize as your model expects
        raw_image = cv2.resize(raw_image, (640, 384))

        img = yolop_v2_lines_transform(raw_image).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        outputs = yolop_v2_lines_model(img)

        ll_seg_out = outputs[2]
    else:
        ll_seg_out = ll_seg_out_from_inference
        # When ll_seg_out_from_inference is provided, we assume raw_image is already resized
        # and is the resized_image from run_yolop_v2_inference

    ll_seg_mask = torch.nn.functional.interpolate(
        ll_seg_out,
        size=(raw_image.shape[0], raw_image.shape[1]),
        mode='bilinear',
        align_corners=False
    )

    probs = torch.sigmoid(ll_seg_mask)
    probs_np = probs.squeeze().cpu().numpy()

    # Otsu threshold
    _, yolo_mask = cv2.threshold(
        (probs_np * 255).astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    yolo_mask = cv2.morphologyEx(yolo_mask, cv2.MORPH_CLOSE, kernel)

    # ==========================
    # CLASSICAL CV FALLBACK
    # ==========================

    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    # Light blur only (avoid destroying lines)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=50,
        maxLineGap=40
    )

    cv_mask = np.zeros_like(yolo_mask)

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(cv_mask, (x1,y1), (x2,y2), 255, 2)

    if not force_classical_fallback:
        return yolo_mask

    cv_mask = classical_lane_fallback(raw_image)

    # Apply road ROI
    cv_mask = apply_road_roi(cv_mask, keep_ratio=0.35)

    # Now filter lane-like regions
    # cv_mask_filtered = filter_lane_like_regions(cv_mask)

    hybrid_mask = cv2.bitwise_or(yolo_mask, cv_mask)

    return hybrid_mask

def apply_road_roi(mask, keep_ratio=0.25):
    """
    Keeps only the bottom portion of the image.
    keep_ratio = 0.45 means bottom 45% is kept.
    """
    h = mask.shape[0]
    roi_mask = np.zeros_like(mask)
    roi_mask[int(h*(1-keep_ratio)):, :] = 255
    return cv2.bitwise_and(mask, roi_mask)

def filter_lane_like_regions(white_mask):

    filtered = np.zeros_like(white_mask)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )

    h, w = white_mask.shape

    for i in range(1, num_labels):  # skip background
        x,y,w_box,h_box,area = stats[i]

        # --- FILTER RULES ---

        # 1. Remove tiny areas
        if area < 80:
            continue

        # 2. Aspect ratio: long and thin
        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        if aspect < 2.5:
            continue

        # 3. Orientation: mostly vertical
        component = (labels == i).astype(np.uint8) * 255
        ys, xs = np.where(component > 0)
        if len(xs) < 30:
            continue

        vx, vy, _, _ = cv2.fitLine(
            np.column_stack([xs, ys]), cv2.DIST_L2, 0, 0.01, 0.01
        )

        angle = abs(np.arctan2(vy, vx)) * 180 / np.pi
        if angle < 25 or angle > 155:
            continue  # reject near-horizontal

        # 4. Position: bottom half only
        cy = centroids[i][1]
        if cy < h * 0.45:
            continue

        # Passed all filters → keep
        filtered[labels == i] = 255

    return filtered


def classical_lane_fallback(raw_image):

    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

    # Detect white-ish pixels (lane paint)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean noise
    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    # Keep only bottom half (where lanes are)
    h = white_mask.shape[0]
    white_mask[:h//2, :] = 0

    # Optional thinning
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    return white_mask

from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

def _extend_line(skeleton, height):
    if skeleton is None:
        return None, None

    points = np.argwhere(skeleton > 0)
    if len(points) < 900: # Need at least a few points to do anything
        return skeleton, None

    # Create a copy to draw extensions on and a debug visualizer
    extended_mask = skeleton.copy()
    debug_viz = cv2.cvtColor(skeleton.copy(), cv2.COLOR_GRAY2BGR)
    width = skeleton.shape[1]

    # Sort points from top to bottom
    points = points[points[:, 0].argsort()]

    # --- SMOOTHING STEP ---
    try:
        # Window size must be odd and smaller than the number of points
        window_length = min(len(points) // 3, 51) # Window up to 51, or 1/3 of points
        if window_length < 5: window_length = 5
        if window_length % 2 == 0: window_length += 1

        if len(points) > window_length:
            # Smooth the x-coordinates of the line
            smoothed_x = savgol_filter(points[:, 1], window_length=window_length, polyorder=2)
            # Update points with the smoothed x-values
            points = np.array(list(zip(points[:, 0], smoothed_x))).astype(np.int32)
    except Exception:
        pass # If smoothing fails for any reason, proceed with the original noisy line
    # --- END SMOOTHING ---

    y_coords = points[:, 0]
    min_y, max_y = y_coords.min(), y_coords.max()

    # --- Downward Extension ---
    BORDER_TOLERANCE = 5 # pixels
    if max_y < height - 5: # 5-pixel tolerance for bottom
        num_points_to_use = 900
        lower_points = points[-num_points_to_use:]
        if len(lower_points) < 2:
            lower_points = points

        # Draw the points being used for downward extension in RED
        for p in lower_points:
            cv2.circle(debug_viz, (p[1], p[0]), 3, (0, 0, 255), -1)

        if len(lower_points) >= 2:
            y_fit = lower_points[:, 0]
            x_fit = lower_points[:, 1]
            try:
                # Use a more robust linear fit if we don't have enough points for a quadratic one
                degree = 2 if len(np.unique(y_fit)) > 2 else 1
                fit = np.polyfit(y_fit, x_fit, degree)
                line_fn = np.poly1d(fit)

                original_bottom_x = int(np.mean(points[y_coords == max_y, 1]))
                original_bottom_point = (original_bottom_x, max_y)

                # ONLY extend downwards if the line is NOT hitting the left/right border
                if not (original_bottom_x < BORDER_TOLERANCE or original_bottom_x > width - 1 - BORDER_TOLERANCE):
                    target_y = height - 1
                    target_x = int(line_fn(target_y))
                    target_x = np.clip(target_x, 0, width - 1)
                    target_point = (target_x, target_y)

                    cv2.line(extended_mask, original_bottom_point, target_point, 255, 2)
            except (np.linalg.LinAlgError, TypeError):
                pass # If fit fails, we just don't extend downwards

    # --- Upward Extension ---
    if min_y > height // 3: # Only extend up if the line starts below the 25% upper part of the image
        num_points_to_use = 900
        upper_points = points[:num_points_to_use]
        if len(upper_points) < 2:
            upper_points = points

        # Draw the points being used for upward extension in BLUE
        for p in upper_points:
            cv2.circle(debug_viz, (p[1], p[0]), 3, (255, 0, 0), -1)

        if len(upper_points) >= 2:
            y_fit = upper_points[:, 0]
            x_fit = upper_points[:, 1]
            try:
                degree = 2 if len(np.unique(y_fit)) > 2 else 1
                fit = np.polyfit(y_fit, x_fit, degree)
                line_fn = np.poly1d(fit)

                original_top_x = int(np.mean(points[y_coords == min_y, 1]))
                original_top_point = (original_top_x, min_y)

                target_y = height // 3
                target_x = int(line_fn(target_y))
                target_x = np.clip(target_x, 0, width - 1)
                target_point = (target_x, target_y)

                cv2.line(extended_mask, original_top_point, target_point, 255, 2)
            except (np.linalg.LinAlgError, TypeError):
                pass # If fit fails, we just don't extend upwards

    return extended_mask, debug_viz


def _line_crosses_white(p1, p2, binary_mask):
    """Returns True if line between p1 and p2 crosses white pixels"""
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))

    # Create a mask with just the line
    line_mask = np.zeros_like(binary_mask)
    cv2.line(line_mask, p1, p2, 255, 1)

    # Check if any white pixel from original mask overlaps the line
    overlap = cv2.bitwise_and(binary_mask, line_mask)
    return np.any(overlap > 0)

def extend_polyline_from_fit(fit_fn, y_min, y_max, width, num_points=100):
    ys = np.linspace(y_min, y_max, num_points).astype(int)
    xs = np.clip(fit_fn(ys), 0, width - 1).astype(int)
    return np.array(list(zip(xs, ys))).reshape((-1, 1, 2))

def _find_and_draw_lane_boundaries(binary_mask, save_path_intermediate_dir=None, img_path=None, reference_point=None):
    height, width = binary_mask.shape
    debug_viz = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Step 1: Pre-process and Get All Extended Lines ---
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    
    all_extended_lines_mask = np.zeros_like(binary_mask)
    lines_debug_viz = np.zeros((height, width, 3), dtype=np.uint8)

    # OJO COMMENTED LINE EXTENSION
    # for i in range(1, num_labels):
    #     # Filter based on length (width or height) instead of area to better detect thin lines
    #     if stats[i, cv2.CC_STAT_WIDTH] < 20 or stats[i, cv2.CC_STAT_HEIGHT] > 100:
    #         segment_mask = (labels == i).astype(np.uint8) * 255
    #         extended_line, segment_debug_viz = _extend_line(segment_mask, height)
    #         if extended_line is not None:
    #             all_extended_lines_mask = cv2.bitwise_or(all_extended_lines_mask, extended_line)
    #         if segment_debug_viz is not None:
    #             lines_debug_viz = cv2.bitwise_or(lines_debug_viz, segment_debug_viz)

    # --- Step 1b: Save the debug visualization of points used for fitting ---
    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_1b_points_used_for_fit.png"), lines_debug_viz)


    # --- Step 2: Robustly Track All Potential Center Paths ---
    active_paths = []
    terminated_paths = []
    MAX_PATH_JUMP_DISTANCE = 10

    for y in range(height // 2, height, 4):
        # row = all_extended_lines_mask[y, :] # OJO COMMENTED DUE TO NOT USING EXTENDED LINES
        row = closed_mask[y, :]

        # --- Correct Midpoint Calculation: Find gaps between lines, not points within them ---
        # Pad the row to find changes at the edges
        padded_row = np.pad(row, (1, 1), 'constant')
        diffs = np.diff(padded_row.astype(np.int32))

        # Find start and end points of white segments
        starts = np.where(diffs > 0)[0]
        ends = np.where(diffs < 0)[0]

        current_midpoints = []
        # Calculate the midpoint of the black space BETWEEN the white segments
        if len(starts) >= 2:
            for i in range(len(starts) - 1):
                # Midpoint is between the end of the first segment and the start of the second
                midpoint_x = (ends[i] + starts[i+1]) // 2
                current_midpoints.append([midpoint_x, y])
        # --- End Correct Midpoint Calculation ---

        if not active_paths and current_midpoints:
            for midpt in current_midpoints:
                active_paths.append([midpt])
            continue

        new_active_paths = []
        unmatched_midpoints = list(current_midpoints)

        for path in active_paths:
            last_point = path[-1]
            closest_midpoint = None
            min_dist = float('inf')

            # Find the closest available midpoint for the current path
            temp_closest = None
            for midpt in unmatched_midpoints:
                dist = np.linalg.norm(np.array(last_point) - np.array(midpt))

                if dist >= MAX_PATH_JUMP_DISTANCE:
                    continue

                # NEW: forbid crossing white lines
                if _line_crosses_white(last_point, midpt, closed_mask):
                    continue

                if dist < min_dist:
                    min_dist = dist
                    temp_closest = midpt

            if temp_closest is not None:
                path.append(temp_closest)
                new_active_paths.append(path)
                unmatched_midpoints.remove(temp_closest)
            else:
                # This path is terminated
                terminated_paths.append(path)

        # Any remaining midpoints start new paths
        for midpt in unmatched_midpoints:
            new_active_paths.append([midpt])

        active_paths = new_active_paths

    terminated_paths.extend(active_paths)

    # --- Step 3: Save debug image of all found paths ---
    all_paths_viz = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
    for i, path in enumerate(terminated_paths):
        if len(path) > 1:
            path_pts = np.array(path).reshape((-1, 1, 2))
            cv2.polylines(all_paths_viz, [path_pts], isClosed=False, color=colors[i % len(colors)], thickness=2)
    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_2_all_candidate_paths.png"), all_paths_viz)

    # --- Step 4: Select the best path ---
    best_path = None
    best_score = float('inf')
    long_paths = [path for path in terminated_paths if len(path) > 10]

    if not long_paths:
        return np.zeros_like(binary_mask), [], [], debug_viz, None, None, np.zeros_like(binary_mask), np.zeros_like(binary_mask), np.zeros_like(binary_mask), np.zeros_like(binary_mask)

    # First, iterate through all paths to find the best one based on the score
    for path in long_paths:
        path_points = np.array(path)
        y_coords = path_points[:, 1]
        x_coords = path_points[:, 0]

        try:
            # Polyfit level 2
            fit_2 = np.polyfit(y_coords, x_coords, 2)
            curve_fn_2 = np.poly1d(fit_2)

            # Use level 2 polyfit to determine the best path by checking its intersection at the bottom of the image
            bottom_y = height - 1
            bottom_x = curve_fn_2(bottom_y)

            if reference_point is None:
                reference_x = width / 2
            else:
                reference_x = reference_point[0]

            distance_to_reference = abs(bottom_x - reference_x)

            if distance_to_reference < best_score:
                best_score = distance_to_reference
                best_path = path_points

        except (np.linalg.LinAlgError, TypeError):
            continue

    # --- Step 4a: Save intermediate visualizations IF requested ---
    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        polyfit_viz_2 = np.zeros((height, width, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        # Now, iterate again to draw all the candidates
        for i, path in enumerate(long_paths):
            path_points = np.array(path)
            y_coords = path_points[:, 1]
            x_coords = path_points[:, 0]
            try:
                fit_2 = np.polyfit(y_coords, x_coords, 2)
                curve_fn_2 = np.poly1d(fit_2)

                plot_y = np.linspace(y_coords.min(), y_coords.max(), 100).astype(int)
                color = colors[i % len(colors)]

                # Draw level 2
                plot_x_2 = np.clip(curve_fn_2(plot_y), 0, width - 1).astype(int)
                pts_2 = np.array(list(zip(plot_x_2, plot_y))).reshape((-1, 1, 2))
                cv2.polylines(polyfit_viz_2, [pts_2], isClosed=False, color=color, thickness=2)
            except (np.linalg.LinAlgError, TypeError):
                continue # Skip drawing if fit fails
                
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_2b_polyfit_deg2.png"), polyfit_viz_2)

    # --- Step 5: Create Final Mask from the Best Path (FORCED EXTENSION) ---
    final_mask = np.zeros_like(binary_mask)
    if best_path is not None and len(best_path) > 1:
        try:
            fit = np.polyfit(best_path[:, 1], best_path[:, 0], 2)
            curve_fn = np.poly1d(fit)

            # FORCE extension from bottom to middle of image
            plot_y_min = height // 2  # <-- always extend up to middle
            plot_y_max = height - 1  # bottom

            plot_y = np.linspace(plot_y_min, plot_y_max, 120).astype(int)
            plot_x = np.clip(curve_fn(plot_y), 0, width - 1).astype(int)
            path_pts = np.array(list(zip(plot_x, plot_y))).reshape((-1, 1, 2))

            cv2.polylines(final_mask, [path_pts], isClosed=False, color=255, thickness=2)

        except (np.linalg.LinAlgError, TypeError):
            cv2.polylines(final_mask, [best_path.reshape((-1, 1, 2))], isClosed=False, color=255, thickness=2)

    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_1_all_extended_lines.png"), all_extended_lines_mask)
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_3_final_path.png"), final_mask)

    return final_mask, [], [], debug_viz, None, None, final_mask, lines_debug_viz, all_paths_viz, all_extended_lines_mask


def detect_lanes_yolop_v2_drivable_agent(image, save_path_intermediate_dir=None, img_path=None):
    with torch.no_grad():
        da_seg_out, _, resized_image = run_yolop_v2_inference(image)
        drivable_mask = process_yolop_v2_drivable_output(da_seg_out, resized_image)

    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_raw_drivable.png"), drivable_mask)

    height, width = drivable_mask.shape
    
    # --- EGO LANE ISOLATION with STATIC TRAPEZOIDAL ROI ---
    # 1. Define the vertices of the trapezoid that represents the ego lane.
    # These points are fine-tuned for a 640x384 image with a standard forward-facing camera.
    roi_vertices = np.array([
        [0, height],          # Bottom-left
        [width, height],      # Bottom-right
        [width // 2 + 80, height // 2], # Top-right
        [width // 2 - 80, height // 2]  # Top-left
    ], dtype=np.int32)

    # 2. Create a black mask and draw the filled trapezoid onto it.
    roi_mask = np.zeros_like(drivable_mask)
    cv2.fillPoly(roi_mask, [roi_vertices], 255)

    # 3. Apply the ROI mask to the drivable area mask.
    # This effectively "cuts out" and keeps only the drivable area within our ego lane.
    ego_lane_mask = cv2.bitwise_and(drivable_mask, drivable_mask, mask=roi_mask)

    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        # Save the isolated ego lane for debugging
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_ego_lane_mask.png"), ego_lane_mask)
        # Also save a visualization of the trapezoid on the original image
        roi_viz = image.copy()
        cv2.polylines(roi_viz, [roi_vertices], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_roi_visualization.png"), roi_viz)
    # --- END EGO LANE ISOLATION ---

    center_points = []
    output_mask = np.zeros_like(drivable_mask)

    # Find center of the ISOLATED ego lane for each row
    for y in range(height // 2, height, 4): # Scan lower half
        row = ego_lane_mask[y, :]
        white_pixels = np.where(row > 0)[0]
        if len(white_pixels) > 1:
            leftmost = white_pixels[0]
            rightmost = white_pixels[-1]
            mid_x = (leftmost + rightmost) // 2
            center_points.append([mid_x, y])

    # Polyfit to get a smooth curve
    poly_points = []
    if len(center_points) > 10:
        try:
            c_pts = np.array(center_points)
            fit = np.polyfit(c_pts[:, 1], c_pts[:, 0], 2)
            y_min, y_max = c_pts[:, 1].min(), height - 1
            plot_y = np.linspace(y_min, y_max, 25).astype(int)
            fit_x = np.poly1d(fit)(plot_y).astype(int)

            pts = np.array([np.transpose(np.vstack([fit_x, plot_y]))], np.int32)
            cv2.polylines(output_mask, pts, False, 255, 2)
            poly_points = pts.squeeze() if pts.shape[0] > 0 else []
        except Exception as e:
            print(f"Drivable area polyfit Error: {e}")
            pass
    
    # --- The rest of the function remains the same ---
    center_image_x = width // 2
    num_agent_points = 10
    center_lanes = []
    if len(poly_points) > 0:
        y_min = np.min(poly_points[:, 1])
        y_max = np.max(poly_points[:, 1])
        target_y_coords = np.linspace(y_min, y_max, num_agent_points).astype(int)

        fit = np.polyfit(poly_points[:, 1], poly_points[:, 0], 2)
        curve_fn = np.poly1d(fit)

        for y_coord in target_y_coords:
            x_val = np.clip(curve_fn(y_coord), 0, width - 1)
            center_lanes.append([int(x_val), int(y_coord)])

        center_lanes = np.array(center_lanes)
        valid_points_x = center_lanes[:, 0]
        raw_errors = (valid_points_x - center_image_x)
        weights = np.linspace(0.2, 1.0, num_agent_points)
        weighted_error = np.sum(raw_errors * weights) / np.sum(weights)
        MAX_DEV = width // 4
        distance_to_center = np.clip(weighted_error / MAX_DEV, -1.0, 1.0)
    else:
        center_lanes = np.array([[0, 0] for _ in range(num_agent_points)])
        distance_to_center = 1.0

    # Visualization
    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).name
        debug_viz = cv2.cvtColor(drivable_mask, cv2.COLOR_GRAY2BGR)
        debug_viz[ego_lane_mask > 0] = [0, 50, 0]
        for pt in center_lanes:
            cv2.circle(debug_viz, (pt[0], pt[1]), 4, (0, 255, 0), -1)
        if len(poly_points) > 0:
            cv2.polylines(debug_viz, [poly_points.reshape((-1, 1, 2))], False, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"drivable_centers_{base_name}"), debug_viz)

    return (output_mask / 255).astype(np.uint8), distance_to_center, center_lanes, center_lanes

def detect_lanes_yolop_v2_hybrid_agent(image, save_path_intermediate_dir=None, img_path=None, force_drivable_fallback=False, force_classical_fallback=False, reference_point=None):
    # Step 1 & 2: Calculate both drivable area and lines first
    with torch.no_grad():
        da_seg_out, ll_seg_out, resized_image = run_yolop_v2_inference(image)
        ll_segment = detect_yolop_v2_lines(resized_image, force_classical_fallback, ll_seg_out_from_inference=ll_seg_out)
        # drivable_mask = process_yolop_v2_drivable_output(da_seg_out, resized_image)

    height, width = ll_segment.shape
    base_name = Path(img_path).stem if img_path else "debug"

    # --- EGO LANE ISOLATION with STATIC TRAPEZOIDAL ROI ---
    # 1. Define the vertices of the trapezoid that represents the ego lane.
    # These points are fine-tuned for a 640x384 image with a standard forward-facing camera.
    roi_vertices = np.array([
        [0, height - 20],          # Bottom-left
        [width, height - 20],      # Bottom-right
        [width, height // 1.8], # Top-right
        [0, height // 1.8]  # Top-left
    ], dtype=np.int32)

    # 2. Create a black mask and draw the filled trapezoid onto it.
    roi_mask = np.zeros_like(ll_segment)
    cv2.fillPoly(roi_mask, [roi_vertices], 255)
    ll_segment = cv2.bitwise_and(ll_segment, ll_segment, mask=roi_mask)

    if save_path_intermediate_dir:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_0_raw_line_mask.png"), ll_segment)

    # Use the simplified line-finding logic which now only extends lines
    _, _, _, _, _, _, all_extended_lines_mask, points_for_extension, paths_mask, extended_lines_mask = _find_and_draw_lane_boundaries(ll_segment, save_path_intermediate_dir, reference_point=reference_point)

    # Step 3: Check if any lines were extended or if fallback is forced.
    # if np.any(all_extended_lines_mask) and not force_drivable_fallback: # OJO!! FALBACK when town03 added
    if True:
        # --- PRIMARY LOGIC: Lines were detected. Use the extended lines mask directly. ---
        if save_path_intermediate_dir:
            print(f"[{base_name}] SUCCESS: Lines detected. Returning all extended lines.")
        final_mask = all_extended_lines_mask
    else:
        # --- FALLBACK LOGIC: No lines detected, use drivable area ---
        if save_path_intermediate_dir:
            print(f"[{base_name}] FALLBACK: No lines detected. Using drivable-area logic.")
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_1_drivable_mask.png"), drivable_mask)

        roi_vertices = np.array([
            [0, height], [width, height],
            [width // 2 + 80, height // 2], [width // 2 - 80, height // 2]
        ], dtype=np.int32)
        roi_mask = np.zeros_like(drivable_mask)
        cv2.fillPoly(roi_mask, [roi_vertices], 255)
        ego_lane_mask = cv2.bitwise_and(drivable_mask, drivable_mask, mask=roi_mask)
        
        if save_path_intermediate_dir:
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_2_roi_applied.png"), ego_lane_mask)

        center_points = []
        for y in range(height // 2, height, 4):
            row = ego_lane_mask[y, :]
            white_pixels = np.where(row > 0)[0]
            if len(white_pixels) > 1:
                center_points.append([(white_pixels[0] + white_pixels[-1]) // 2, y])

        final_mask = np.zeros_like(drivable_mask)
        if len(center_points) > 10:
            c_pts = np.array(center_points)
            try:
                fit = np.polyfit(c_pts[:, 1], c_pts[:, 0], 2)
                curve_fn = np.poly1d(fit)
                y_min, y_max = c_pts[:, 1].min(), height - 1
                plot_y = np.linspace(y_min, y_max, 50).astype(int)
                fit_x = np.clip(curve_fn(plot_y).astype(int), 0, width - 1)

                pts = np.array([np.transpose(np.vstack([fit_x, plot_y]))], np.int32)
                cv2.polylines(final_mask, pts, False, 255, 2)
            except Exception as e:
                 print(f"[{base_name}] ERROR in drivable fallback polyfit: {e}")

    # --- UNIFIED STEP 4: Calculate 10 points and distance from the FINAL MASK ---
    center_lanes = []
    distance_to_center = 1.0 # Default to max error

    path_points = np.argwhere(final_mask > 0)
    if len(path_points) > 10:
        # Swap columns so we have (x, y)
        path_points = path_points[:, [1, 0]]
        
        try:
            fit = np.polyfit(path_points[:, 1], path_points[:, 0], 2)
            curve_fn = np.poly1d(fit)
            
            y_coords = path_points[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Generate 10 evenly spaced points
            plot_y = np.linspace(y_min, y_max, 10).astype(int)
            fit_x = np.clip(curve_fn(plot_y).astype(int), 0, width - 1)

            center_lanes = np.transpose(np.vstack([fit_x, plot_y])).astype(np.int32)
            
            # Calculate weighted error for distance_to_center
            raw_errors = (center_lanes[:, 0] - (width // 2))
            weights = np.linspace(0.2, 1.0, len(raw_errors))
            weighted_error = np.sum(raw_errors * weights) / np.sum(weights)
            distance_to_center = np.clip(weighted_error / (width // 4), -1.0, 1.0)
        except Exception as e:
            print(f"[{base_name}] ERROR in final unified polyfit: {e}")

    if len(center_lanes) == 0:
        center_lanes = np.array([[0, 0] for _ in range(10)])

    # Visualization for both paths
    final_output_viz = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Draw the final calculated polyline or extended lines
    final_output_viz[final_mask > 0] = [0, 255, 0] # Green for final path
    # Draw the 10 evenly spaced center points
    for p in center_lanes:
         cv2.circle(final_output_viz, (p[0], p[1]), 5, (0, 0, 255), -1) # Red dots for the 10 points
    if save_path_intermediate_dir:
       cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_3_final_output.png"), final_output_viz)

    return (final_mask / 255).astype(np.uint8), distance_to_center, center_lanes, center_lanes, ll_segment, extended_lines_mask, paths_mask, points_for_extension


def detect_lanes_yolop_v2_lines_agent(image, save_path_intermediate_dir=None, img_path=None, force_classical_fallback=False):
    with torch.no_grad():
        ll_segment = detect_yolop_v2_lines(image, force_classical_fallback)
    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_raw.png"), ll_segment)

    # Use the simplified line-finding logic which now only extends lines
    output_mask, _, _, debug_viz, _, _, all_extended_lines_mask, _, _ = _find_and_draw_lane_boundaries(ll_segment, save_path_intermediate_dir, img_path)

    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_Edge-Optimized_Consensus_Debug.png"), debug_viz)

    # Return the combined extended line mask directly and default values
    final_mask = all_extended_lines_mask if all_extended_lines_mask is not None else np.zeros_like(ll_segment)
    distance_to_center = 0.0 # Placeholder
    center_lanes = np.array([[image.shape[1] // 2, y] for y in range(image.shape[0] // 2, image.shape[0], image.shape[0] // 20)])

    return (final_mask / 255).astype(np.uint8), distance_to_center, center_lanes, center_lanes


def normalize_centers(centers):
    x_centers = centers[:, 0]
    x_centers_normalized = (x_centers / 640).tolist()
    states = x_centers_normalized
    y_centers = centers[:, 1]
    y_centers_normalized = (y_centers / 384).tolist() # Adjusted for 384 height
    states = states + y_centers_normalized # Returns a list
    return states, x_centers_normalized, y_centers_normalized
        
def merge_and_extend_lines(lines, ll_segment):
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
    for line in merged_lines:
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
        cv2.line(merged_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the merged lines
    cv2.imshow('Merged Lines', merged_image) if show_images else None

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

def detect_lane_detector(raw_image):
    image_tensor = raw_image.transpose(2, 0, 1).astype('float32') / 255
    x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
    model_output = torch.softmax(lane_model.forward(x_tensor), dim=1).cpu().numpy()
    return model_output

def detect_lane_detector_v3(raw_image):
    image_tensor = raw_image.transpose(2, 0, 1).astype('float32') / 255
    x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
    model_output = torch.softmax(lane_model_v3.forward(x_tensor), dim=1).cpu().numpy()
    return model_output

def lane_detection_overlay(image, left_mask, right_mask):
    res = np.copy(image)
    # We show only points with probability higher than 0.5
    res[left_mask > 0.5, :] = [255,0,0]
    res[right_mask > 0.5,:] = [0, 0, 255]
    return res

def extract_green_lines(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the green color in HSV space
    lower_green = np.array([35, 100, 200])
    upper_green = np.array([85, 255, 255])

    # Create a mask for the green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    return green_mask

def detect_lines(raw_image, detection_mode, processing_mode, img_path, save_path_intermediate_dir=None, force_drivable_fallback=False, force_classical_fallback=False):
    if detection_mode == 'carla_perfect':

        green_mask = extract_green_lines(raw_image)

        lines = post_process_hough_programmatic(green_mask)

        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ll_segment = cv2.Canny(blur, 50, 100)
        ll_segment = post_process(ll_segment)

    elif detection_mode == 'programmatic':
        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # mask_white = cv2.inRange(gray, 200, 255)
        # mask_image = cv2.bitWiseAnd(gray, mask_white)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ll_segment = cv2.Canny(blur, 50, 100)
        cv2.imshow("raw", ll_segment) if show_images else None
        processed = post_process(ll_segment)
        if processing_mode == 'none':
            lines = processed
        else:
            lines = post_process_hough_programmatic(processed if apply_mask else ll_segment)
    elif detection_mode == 'yolop_v2_lines':
        detected_lines, distance_to_center, distance_to_center_normalized, centers = detect_lanes_yolop_v2_lines_agent(raw_image, save_path_intermediate_dir, img_path, force_classical_fallback)
        x_centers = [point[0] for point in centers]
        return detected_lines, distance_to_center, distance_to_center_normalized, np.array(x_centers)
    elif detection_mode == 'yolop_v2_drivable':
        detected_lines, distance_to_center, distance_to_center_normalized, centers = detect_lanes_yolop_v2_drivable_agent(raw_image, save_path_intermediate_dir, img_path)
        x_centers = [point[0] for point in centers]
        return detected_lines, distance_to_center, distance_to_center_normalized, np.array(x_centers)
    elif detection_mode == 'yolop_v2_hybrid':
        detected_lines, distance_to_center, distance_to_center_normalized, centers, raw_image, extended_image, paths_image, points_for_extension = detect_lanes_yolop_v2_hybrid_agent(raw_image, save_path_intermediate_dir, img_path, force_drivable_fallback, force_classical_fallback)
        x_centers = [point[0] for point in centers]
        return detected_lines, distance_to_center, distance_to_center_normalized, np.array(x_centers), raw_image, extended_image, paths_image, points_for_extension
    elif detection_mode == 'yolop':
        with torch.no_grad():
            ll_segment = (detect_yolop(raw_image) * 255).astype(np.uint8)
        cv2.imshow("raw", ll_segment) if show_images else None
        if processing_mode == 'none':
            lines = ll_segment
        else:
            processed = post_process(ll_segment)
            lines = post_process_hough_yolop(processed if apply_mask else ll_segment)
    elif detection_mode == "lane_det_v3":
        with torch.no_grad():
            ll_segment, left_mask, right_mask = detect_lane_detector_v3(raw_image)[0]
        ll_segment = np.zeros_like(raw_image)
        ll_segment = lane_detection_overlay(ll_segment, left_mask, right_mask)
        cv2.imshow("raw", ll_segment) if show_images else None
        # Extract blue and red channels
        if processing_mode == 'none':
            lines = ll_segment
        else:
            # Display the grayscale image
            ll_segment = post_process(ll_segment)
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

            lines = []
            left_line = post_process_hough_lane_det(blue_channel)
            if left_line is not None:
                lines.append([left_line])
            right_line = post_process_hough_lane_det(red_channel)
            if right_line is not None:
                lines.append([right_line])
            ll_segment = 0.5 * blue_channel + 0.5 * red_channel
            ll_segment = cv2.convertScaleAbs(ll_segment)
    else:
        with torch.no_grad():
            ll_segment, left_mask, right_mask = detect_lane_detector(raw_image)[0]
        ll_segment = np.zeros_like(raw_image)
        ll_segment = lane_detection_overlay(ll_segment, left_mask, right_mask)
        cv2.imshow("raw", ll_segment) if show_images else None
        # Extract blue and red channels
        if processing_mode == 'none':
            lines = ll_segment
        else:
            # Display the grayscale image
            ll_segment = post_process(ll_segment)
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

            lines = []
            left_line = post_process_hough_lane_det(blue_channel)
            if left_line is not None:
                lines.append([left_line])
            right_line = post_process_hough_lane_det(red_channel)
            if right_line is not None:
                lines.append([right_line])
            ll_segment = 0.5 * blue_channel + 0.5 * red_channel
            ll_segment = cv2.convertScaleAbs(ll_segment)

    if processing_mode == 'none':
        detected_lines = ll_segment
        # detected_lines = (detected_lines // 255).astype(np.uint8)  # Keep the lower one-third of the image
    else:
        detected_lines = merge_and_extend_lines(lines, ll_segment)
        # line_mask = morphological_process(line_mask, kernel_size=15, func_type=cv2.MORPH_CLOSE)
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # line_mask = cv2.dilate(line_mask, (15, 15), iterations=15)
        # line_mask = cv2.erode(line_mask, (5, 5), iterations=20)

        # TODO (Ruben) It is quite hardcoded and unrobust. Fix this to enable all lines and more than
        # 1 lane detection and cameras in other positions
        boundary_y = detected_lines.shape[1] * 1 // 3
        # Copy the lower part of the source image into the target image
        detected_lines[:boundary_y, :] = 0
        detected_lines = (detected_lines // 255).astype(np.uint8)  # Keep the lower one-third of the image

    (
        center_lanes_old,
        distance_to_center_normalized,
    ) = calculate_center_v1(detected_lines)
    right_lane_normalized_distances, right_center_lane = choose_lane_v1(distance_to_center_normalized,
                                                                            center_lanes_old)
    centers = np.array(right_lane_normalized_distances)
    distance_to_center = np.mean(centers)

    return detected_lines, distance_to_center, distance_to_center_normalized, centers


def choose_lane(distance_to_center_normalized, center_points):
    last_row = len(x_row) - 1
    closest_lane_index = min(enumerate(distance_to_center_normalized[last_row]), key=lambda x: abs(x[1]))[0]
    distances = [array[closest_lane_index] if len(array) > closest_lane_index else min(array) for array in distance_to_center_normalized]
    centers = [array[closest_lane_index] if len(array) > closest_lane_index else min(array) for array in center_points]
    return distances, centers

def choose_lane_v1(distance_to_center_normalized, center_points):
    close_lane_indexes = [min(enumerate(inner_array), key=lambda x: abs(x[1]))[0] for inner_array in
                          distance_to_center_normalized]
    distances = [array[index] for array, index in zip(distance_to_center_normalized, close_lane_indexes)]
    centers = [array[index] for array, index in zip(center_points, close_lane_indexes)]
    return distances, centers


def find_lane_center(mask):
    # Find the indices of 1s in the array
    mask_array = np.array(mask)
    indices = np.where(mask_array > 0.8)[0]

    # If there are no 1s or only one set of 1s, return None
    if len(indices) < 2:
        # TODO (Ruben) For the moment we will be dealing with no detection as a fixed number
        return [NO_DETECTED]

    # Find the indices where consecutive 1s change to 0
    diff_indices = np.where(np.diff(indices) > 1)[0]
    # If there is only one set of 1s, return None
    if len(diff_indices) == 0:
        return [NO_DETECTED]

    interested_line_borders = np.array([], dtype=np.int8)
    for index in diff_indices:
        interested_line_borders = np.append(interested_line_borders, indices[index])
        interested_line_borders = np.append(interested_line_borders, int(indices[index+1]))

    midpoints = calculate_midpoints(interested_line_borders)
    return midpoints


def calculate_midpoints(input_array):
    midpoints = []
    for i in range(0, len(input_array) - 1, 2):
        midpoint = (input_array[i] + input_array[i + 1]) // 2
        midpoints.append(midpoint)
    return midpoints


def detect_missing_points(lines):
    num_points = len(lines)
    max_line_points = max(len(line) for line in lines)
    missing_line_count = sum(1 for line in lines if len(line) < max_line_points)

    return missing_line_count > 0 and missing_line_count <= num_points // 2


def interpolate_missing_points(input_lists, x_row):
    # Find the index of the list with the maximum length
    max_length_index = max(range(len(input_lists)), key=lambda i: len(input_lists[i]))

    # Determine the length of the complete lists
    complete_length = len(input_lists[max_length_index])

    # Initialize a list to store the inferred list
    inferred_list = []

    # Iterate over each index in x_row
    for i, x_value in enumerate(x_row):
        # If the current index is in the list with incomplete points
        if len(input_lists[i]) < complete_length:
            interpolated_list = []
            for points_i in range(complete_length):
                # TODO calculates interpolated point of missing line and then build the interpolated_list
                # Since it is not trivial, we just discard this point from the moment
                interpolated_y = NO_DETECTED
                interpolated_list.append(interpolated_y)
            inferred_list.append(interpolated_list)
        else:
            # If the current list is complete, simply append the corresponding y value
            inferred_list.append(input_lists[i])

    return inferred_list


def calculate_lines_percent(mask):
    width = mask.shape[1]
    center_image = width / 2

    line_size = images_high - upper_limit
    line_thresholds = np.array(mask[upper_limit: images_high]) > 0.8

    # Initialize counters
    positive_count = 0
    negative_count = 0

    # Iterate through each set of indices
    for thresholds in line_thresholds:
        indices = np.where(thresholds)[0]

        # Check if any index is positive
        if any(index - center_image > 0 for index in indices):
            positive_count += 1
        # Check if any index is negative
        if any(index - center_image < 0 for index in indices):
            negative_count += 1

    left_lane_perc = negative_count / line_size
    right_lane_perc = positive_count / line_size

    return left_lane_perc, right_lane_perc

def calculate_center_v1(mask):
    width = mask.shape[1]
    center_image = width / 2
    lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]
    center_lane_indexes = [
        find_lane_center(lines[x]) for x, _ in enumerate(lines)
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


def calculate_center(mask):
    width = mask.shape[1]
    center_image = width / 2
    lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]
    center_lane_indexes = [
        find_lane_center(lines[x]) for x, _ in enumerate(lines)
    ]

    if detect_missing_points(center_lane_indexes):
        center_lane_indexes = interpolate_missing_points(center_lane_indexes, x_row)
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
        if len(inner_list) < 1: # If we don't see the 2 lanes, we discard the row
            inner_list = [NO_DETECTED] * len(inner_list)  # Set all elements to 1
        result.append(inner_list)

    return result

def get_ll_seg_image(dists, ll_segment, suffix="",  name='ll_seg'):
    ll_segment_int8 = (ll_segment * 255).astype(np.uint8)
    ll_segment_all = [np.copy(ll_segment_int8),np.copy(ll_segment_int8),np.copy(ll_segment_int8)]

    # draw the midpoint used as right center lane
    for index, dist in zip(x_row, dists):
        # Set the value at the specified index and distance to 1
        add_midpoints(ll_segment_all[0], index, dist)

    # draw a line for the selected perception points
    for index in x_row:
        for i in range(630):
            ll_segment_all[0][index][i] = 255
    ll_segment_stacked = np.stack(ll_segment_all, axis=-1)
    # We now show the segmentation and center lane postprocessing
    cv2.imshow(name + suffix, ll_segment_stacked) if show_images else None
    return ll_segment_stacked

def add_midpoints(ll_segment, index, dist):
    # Set the value at the specified index and distance to 1
    draw_dash(index, dist, ll_segment)
    draw_dash(index + 2, dist, ll_segment)
    draw_dash(index + 1, dist, ll_segment)
    draw_dash(index - 1, dist, ll_segment)
    draw_dash(index - 2, dist, ll_segment)

def draw_dash(index, dist, ll_segment):
    ll_segment[index, dist - 1] = 255  # <-- here is the real calculated center
    ll_segment[index, dist - 3] = 255
    ll_segment[index, dist - 2] = 255
    ll_segment[index, dist - 4] = 255
    ll_segment[index, dist - 5] = 255
    ll_segment[index, dist - 6] = 255


def calculate_and_plot_lines_counts_above(save_results_benchmark_path, percs, yolop_left_perc, yolop_right_perc,
                                                yolop_v2_lines_left_perc, yolop_v2_lines_right_perc,
                                                lane_detector_left_perc, lane_detector_right_perc,
                                                lane_detector_v3_left_perc, lane_detector_v3_right_perc,
                                                programmatic_left_perc, programmatic_right_perc,
                                                perfect_left_perc, perfect_right_perc, processing_mode):

    for perc in percs:
        yolop_counts = calculate_lines_counts_above(perc, yolop_left_perc, yolop_right_perc)
        yolop_v2_lines_counts = calculate_lines_counts_above(perc, yolop_v2_lines_left_perc, yolop_v2_lines_right_perc)
        lane_det_counts = calculate_lines_counts_above(perc, lane_detector_left_perc,
                                                       lane_detector_right_perc)
        lane_det_v3_counts = calculate_lines_counts_above(perc, lane_detector_v3_left_perc,
                                                          lane_detector_v3_right_perc)
        prog_counts = calculate_lines_counts_above(perc, programmatic_left_perc, programmatic_right_perc)

        perfect_counts = calculate_lines_counts_above(perc, perfect_left_perc, perfect_right_perc)

        subplots = 6 if processing_mode != "none" else 5

        fig2, axs2 = plt.subplots(1, subplots, figsize=(18, 6))

        axs2[0].bar(['Just left', 'Just right', 'both', 'none'], yolop_counts, color=['blue', 'green'])
        axs2[0].set_title('YOLOP')
        axs2[0].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[0].set_ylim(0, 100)

        axs2[1].bar(['Just left', 'Just right', 'both', 'none'], yolop_v2_lines_counts, color=['blue', 'green'])
        axs2[1].set_title('yolop_v2_lines')
        axs2[1].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[1].set_ylim(0, 100)

        axs2[2].bar(['Just left', 'Just right', 'both', 'none'], lane_det_counts, color=['blue', 'green'])
        axs2[2].set_title('Lane Detector')
        axs2[2].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[2].set_ylim(0, 100)

        axs2[3].bar(['Just left', 'Just right', 'both', 'none'], lane_det_v3_counts, color=['blue', 'green'])
        axs2[3].set_title('Lane Detector V3')
        axs2[3].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[3].set_ylim(0, 100)

        axs2[4].bar(['Just left', 'Just right', 'both', 'none'], perfect_counts, color=['blue', 'green'])
        axs2[4].set_title('Perfect')
        axs2[4].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[4].set_ylim(0, 100)

        if processing_mode != "none":
            axs2[5].bar(['Just left', 'Just right', 'both', 'none'], prog_counts, color=['blue', 'green'])
            axs2[5].set_title('Programmatic')
            axs2[5].set_ylabel(f'percentage of images above {perc * 100}% detected')
            axs2[5].set_ylim(0, 100)

        plt.savefig(save_results_benchmark_path / f'plot_{perc * 100}_above.png')
        plt.close()
    pass


def calculate_lines_counts_above(threshold, left_perc, right_perc):
    return [
        (np.sum((np.array(left_perc) >= threshold) & (np.array(right_perc) < threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) < threshold) & (np.array(right_perc) >= threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) >= threshold) & (np.array(right_perc) >= threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) < threshold) & (np.array(right_perc) < threshold)) / len(left_perc)) * 100,
    ]

def perform_all_benchmarking(dataset, processing_mode, detection_modes=None, save_intermediate=False, force_drivable_fallback=False, force_classical_fallback=False):
    if detection_modes is None:
        detection_modes = []
    # Create the save directory if it doesn't exist
    save_results_benchmark_dir = str(opt.save_dir + "/benchmark/" + processing_mode + "/")
    save_results_benchmark_path = Path(save_results_benchmark_dir)
    save_results_benchmark_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(save_results_benchmark_dir):
        for file_name in os.listdir(save_results_benchmark_dir):
            file_path = os.path.join(save_results_benchmark_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {save_results_benchmark_dir}: {e}")

    all_modes = {
        "yolop": "YOLOP",
        "yolop_v2_lines": "yolop_v2_lines",
        "yolop_v2_drivable": "yolop_v2_drivable",
        "yolop_v2_hybrid": "yolop_v2_hybrid",
        "lane_det_v3": "Lane Detector_v3",
        "lane_detector": "Lane Detector_v2",
        "programmatic": "Programmatic",
        "carla_perfect": "Perfect"
    }
    
    modes_to_run = detection_modes if detection_modes else all_modes.keys()

    results = {}

    for mode in modes_to_run:
        if mode in all_modes:
            if processing_mode == 'none' and 'yolop_v2' in mode:
                print(f"Skipping {mode} for processing_mode {processing_mode}.")
                continue
            print(f"Running benchmark on {mode} with processing_mode: {processing_mode}")
            times, errors, left_perc, right_perc, percentage = benchmark_one(dataset, mode, processing_mode, save_intermediate, force_drivable_fallback, force_classical_fallback)
            results[mode] = {
                "times": times, "errors": errors, "left_perc": left_perc,
                "right_perc": right_perc, "percentage": percentage, "label": all_modes[mode]
            }

    if not results:
        print("No benchmarks were run. Exiting.")
        return
        
    # Dynamically build lists for plotting
    labels = [res["label"] for res in results.values()]
    percentages = [res["percentage"] for res in results.values()]
    times = [np.mean(res["times"]) for res in results.values()]
    errors = [np.mean(res["errors"]) if res["errors"] else 0 for res in results.values()]
    
    # Generate colors for the plots
    base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

    # Plot 1: Percentage of Detected Images
    plt.figure(figsize=(10, 10))
    plt.bar(labels, percentages, color=colors)
    plt.title('Percentage of Detected Images')
    plt.ylabel('Percentage of images perfectly detected')
    plt.ylim(0, 100)
    plt.savefig(save_results_benchmark_path / 'plot0.png')
    plt.close()

    # Plot 2: Average Times
    plt.figure(figsize=(10, 10))
    plt.bar(labels, times, color=colors)
    plt.title('Average Times')
    plt.ylabel('Time (s)')
    plt.savefig(save_results_benchmark_path / 'plottimes.png')
    plt.close()
    
    # Plot 3: Average errors
    plt.figure(figsize=(10, 6))
    pixel_errors = np.array(errors) * 320  # Assuming 320 is the scaling factor
    plt.bar(labels, pixel_errors, color=colors)
    plt.xlabel('Perception Mode')
    plt.ylabel('Average Error (pixels)')
    plt.title('Average error when both lines detected')
    plt.savefig(save_results_benchmark_path / 'ploterrolane.png')
    plt.close()
    
    print(f"All plots saved to {save_results_benchmark_path}")


def detect(opt):
    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)

    detection_modes = opt.detection_modes if opt.detection_modes else []
    perform_all_benchmarking(dataset, "none", detection_modes, opt.save_intermediate_images, opt.force_drivable_fallback, opt.force_classical_fallback)
    perform_all_benchmarking(dataset, "postprocess", detection_modes, opt.save_intermediate_images, opt.force_drivable_fallback, opt.force_classical_fallback)

def benchmark_one(dataset, detection_mode, processing_mode, save_intermediate=False, force_drivable_fallback=False, force_classical_fallback=False):
    global prev_fit
    prev_fit = None
    print("running benchmarking on " + detection_mode)
    # Run inference
    t0 = time.time()

    save_path_bad_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/bad")
    save_path_good_dir = str(opt.save_dir + detection_mode + "/" + processing_mode +  "/good")
    save_path_bad_raw_dir = str(opt.save_dir + detection_mode + "/" + processing_mode +  "/bad_raw")
    save_path_out_raw_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/good_raw")
    save_results_metrics_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/metrics")
    save_path_intermediate_dir = None
    if save_intermediate:
        save_path_intermediate_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/intermediate")

    # TODO avoid this duplicated code
    if not os.path.exists(save_path_bad_dir):
        os.makedirs(save_path_bad_dir)

    if not os.path.exists(save_path_good_dir):
        os.makedirs(save_path_good_dir)

    if not os.path.exists(save_path_bad_raw_dir):
        os.makedirs(save_path_bad_raw_dir)

    if not os.path.exists(save_path_out_raw_dir):
        os.makedirs(save_path_out_raw_dir)

    if not os.path.exists(save_results_metrics_dir):
        os.makedirs(save_results_metrics_dir)
        
    if save_intermediate and not os.path.exists(save_path_intermediate_dir):
        os.makedirs(save_path_intermediate_dir)

    directories = [save_path_bad_dir, save_path_good_dir, save_path_bad_raw_dir, save_path_out_raw_dir,
                   save_results_metrics_dir]
    if save_intermediate:
        directories.append(save_path_intermediate_dir)
    for directory_path in directories:
        if os.path.exists(directory_path):
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {directory_path}: {e}")

    detected = 0
    all_images = 0
    all_avg_errors_when_detected = []
    all_left_perc = []
    all_right_perc = []
    times = []
    for i, (path, img, img_det, vid_cap, shapes) in enumerate(dataset):
        all_images += 1

        save_path_bad = str(save_path_bad_dir + '/' + Path(path).name)
        save_path_good = str(save_path_good_dir + '/' + Path(path).name)
        save_path_bad_raw = str(save_path_bad_raw_dir + '/' + Path(path).name)
        save_path_out_raw = str(save_path_out_raw_dir + '/' + Path(path).name)

        height = img.shape[0]
        width = img.shape[1]

        # resized_img = Image.fromarray(img).resize((640, 384))
        resized_img_np = cv2.resize(img, (640, 384), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        # resized_img_np = np.array(resized_img)
        start = time.time()

        ll_seg_out_list, distance_to_center, center_lanes_normalized, x_centers, _, _, _, _= detect_lines(resized_img_np, detection_mode, processing_mode, path, save_path_intermediate_dir, force_drivable_fallback, force_classical_fallback)

        ll_seg_out = ll_seg_out_list

        # Original 'good'/'bad' logic for other models
        if processing_mode == "none":
            left_percent, right_percent = calculate_lines_percent(ll_seg_out)
            if left_percent > PERFECT_THRESHOLD and right_percent > PERFECT_THRESHOLD:
                detected += 1
                cv2.imwrite(save_path_good, ll_seg_out)
                cv2.imwrite(save_path_out_raw, img)
            else:
                cv2.imwrite(save_path_bad, ll_seg_out)
                cv2.imwrite(save_path_bad_raw, img)
        else:
            overlay = resized_img_np.copy()
            # Create a color mask for the overlay
            color_mask = np.zeros_like(resized_img_np)
            # ll_seg_out is a single-channel mask of detected lines
            color_mask[ll_seg_out > 0] = [0, 0, 255]  # Red for detected lines

            # Blend the original image with the color mask
            overlay = cv2.addWeighted(overlay, 1, color_mask, 0.5, 0)

            if wasDetected(x_centers.tolist()):
                detected += 1
                cv2.imwrite(save_path_good, overlay)
                cv2.imwrite(save_path_out_raw, img)
            else:
                cv2.imwrite(save_path_bad, overlay)
                cv2.imwrite(save_path_bad_raw, img)

        processing_time = time.time() - start
        times.append(processing_time)
        
        # This part below is now redundant for yolop_v2_lines but needed for other models' metrics
        ll_seg_out_raw = ll_seg_out
        (
            center_lanes,
            distance_to_center_normalized,
        ) = calculate_center_v1(ll_seg_out)
        right_lane_normalized_distances, right_center_lane = choose_lane_v1(distance_to_center_normalized, center_lanes)

        ll_segment_stacked = get_ll_seg_image(right_center_lane, ll_seg_out)
        
        left_percent, right_percent = calculate_lines_percent(ll_seg_out)
        all_left_perc.append(left_percent)
        all_right_perc.append(right_percent)

        if dataset.mode == 'images':
            total_error = 0
            detected_points = 0
            for x in right_lane_normalized_distances:
                if abs(x) != 1:
                    detected_points += 1
                    total_error += abs(x)

            if detected_points > 0:
                average_abs = total_error / detected_points
                all_avg_errors_when_detected.append(average_abs)

        else:
            cv2.imshow('image', ll_seg_out)
            cv2.waitKey(1)  # 1 millisecond

        cv2.waitKey(0) if show_images else None

    cv2.waitKey(10000) if show_images else None
    print('Done. (%.3fs)' % (time.time() - t0))
    print(f"total good -> {detected} images of {all_images} were detected = {(detected/all_images) * 100}%.")

    file_name = processing_mode + "_errors.pkl"
    save_results_metrics_errors_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_errors_file, 'wb') as file:
        pickle.dump(all_avg_errors_when_detected, file)

    file_name = processing_mode + "_left.pkl"
    save_results_metrics_left_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_left_file, 'wb') as file:
        pickle.dump(all_left_perc, file)

    file_name = processing_mode + "_right.pkl"
    save_results_metrics_right_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_right_file, 'wb') as file:
        pickle.dump(all_right_perc, file)

    return times, all_avg_errors_when_detected, all_left_perc, all_right_perc, (detected/all_images) * 100


def wasDetected(labels: list):
    if not labels:
        return False  # Empty list is a failure

    # Check for yolop_v2_lines failure case (all zeros)
    is_all_zeros = all(x == 0 for x in labels)
    if is_all_zeros:
        return False

    # Check for other modes' failure case (all abs(1))
    is_all_ones = all(abs(x) == 1 for x in labels)
    if is_all_ones:
        return False

    # If neither failure case is met, it's a success
    return True

def anyDetected(labels: list):
    for i in range(len(labels)):
        if abs(labels[i]) < 0.8:
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/home/ruben/Desktop/broken_p', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='/home/ruben/Desktop/broken_p_output/', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--detection-modes', nargs='+', type=str, help='List of detection modes to benchmark')
    parser.add_argument('--save-intermediate-images', action='store_true', help='save all intermediate images for debugging')
    parser.add_argument('--force-drivable-fallback', action='store_true', help='force the hybrid agent to use drivable area fallback logic')
    parser.add_argument('--force-classical-fallback', action='store_true', help='force to only use classical lane fallback.')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)