#========VISION

import cv2
import numpy as np

import config

_kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def to_hsv(image):
    """
    Blur + convert BGR image to HSV, with brightness normalization so
    detection stays stable as room lighting changes through the day.

    CLAHE (adaptive histogram equalization) is applied to just the V
    channel: it flattens out bright/dark differences across the frame
    and over time WITHOUT touching hue, so your color ranges keep
    working even when the field gets brighter/dimmer overall.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if config.USE_BRIGHTNESS_NORMALIZATION:
        h, s, v = cv2.split(hsv)
        v = _clahe.apply(v)
        hsv = cv2.merge([h, s, v])

    return hsv


def clean_mask(mask):
    """
    The same open+close cleanup main.py applies to every color mask.
    Public so hsv_tuner.py can call this too - otherwise the tuner's
    preview doesn't match what main.py actually detects.
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel, iterations=2)
    return mask


def sample_color_at(hsv_image, center_x, center_y, radius):
    """
    Look at a circular (square-cropped) region of the HSV image and
    return the color name whose mask covers the largest fraction of
    that region, if it clears GEM_COLOR_MIN_RATIO. Else None.
    """

    h, w = hsv_image.shape[:2]

    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(w, center_x + radius)
    y2 = min(h, center_y + radius)

    if x2 <= x1 or y2 <= y1:
        return None

    region = hsv_image[y1:y2, x1:x2]
    region_area = region.shape[0] * region.shape[1]

    if region_area == 0:
        return None

    best_color = None
    best_ratio = 0.0

    for name, (ranges, _box_color) in config.COLORS.items():

        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(region, np.array(lower), np.array(upper))

        ratio = cv2.countNonZero(mask) / float(region_area)

        if ratio > best_ratio:
            best_ratio = ratio
            best_color = name

    if best_ratio >= config.GEM_COLOR_MIN_RATIO:
        return best_color

    return None


def detect_target_circles_and_gems(image, hsv_image, robot_roi):
    """
    Run color detection twice per color:
      - once on a robot-blanked HSV frame, to find big circular
        blobs (fixed target circles)
      - once on the full HSV frame, to find small blobs (loose
        gems), which are NOT blanked near the robot so a gem next
        to the robot doesn't disappear

    Returns (detected_objects, field_gems) - both lists of dicts.
    """

    height, width = image.shape[:2]

    hsv_for_targets = hsv_image.copy()
    if robot_roi is not None:
        x_min, y_min, x_max, y_max = robot_roi
        hsv_for_targets[y_min:y_max, x_min:x_max] = 0

    min_area = height * width * config.TARGET_AREA_FRACTION
    gem_min_area = height * width * config.GEM_AREA_FRACTION
    padding = config.TARGET_BOX_PADDING

    detected_objects = []
    field_gems = []

    for name, (ranges, box_color) in config.COLORS.items():

        # ---- target circles ----
        target_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            target_mask |= cv2.inRange(hsv_for_targets, np.array(lower), np.array(upper))
        target_mask = clean_mask(target_mask)

        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / float(h)
            if aspect_ratio < 0.7 or aspect_ratio > 1.4:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6:
                continue

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width - 1, x + w + padding)
            y2 = min(height - 1, y + h + padding)

            detected_objects.append({
                "color": name,
                "box_color": box_color,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center_x": x + w // 2, "center_y": y + h // 2,
            })

        # ---- loose field gems ----
        gem_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            gem_mask |= cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        gem_mask = clean_mask(gem_mask)

        contours, _ = cv2.findContours(gem_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            area = cv2.contourArea(contour)
            if area < gem_min_area or area >= min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            field_gems.append({
                "color": name,
                "box_color": box_color,
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "center_x": x + w // 2, "center_y": y + h // 2,
            })

    return detected_objects, field_gems


def get_side(target_index):
    """Field rule: index 1-3 -> TOP SIDE, index 4-6 -> BOTTOM SIDE."""
    return "TOP" if target_index <= 3 else "BOTTOM"


def assign_target_indices(detected_objects, height):
    """
    Split detected target-circle objects into TOP/BOTTOM by y-position,
    sort each side left->right, keep at most 3 per side, and assign
    target_index 1-3 (top) / 4-6 (bottom).
    """

    division_y = int(height * config.SIDE_DIVISION)

    top_objects = [o for o in detected_objects if o["center_y"] < division_y]
    bottom_objects = [o for o in detected_objects if o["center_y"] >= division_y]

    top_objects.sort(key=lambda obj: obj["center_x"])
    bottom_objects.sort(key=lambda obj: obj["center_x"])

    top_objects = top_objects[:3]
    bottom_objects = bottom_objects[:3]

    target_circles = []

    for i, obj in enumerate(top_objects):
        obj["target_index"] = i + 1
        target_circles.append(obj)

    for i, obj in enumerate(bottom_objects):
        obj["target_index"] = i + 4
        target_circles.append(obj)

    return target_circles, division_y