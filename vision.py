import math

import cv2
import numpy as np

import config
import field_area
import color_toggles

# A loose "gem" that sits on / touching a base is really part of that base
# (glare, a reflection, a piece of the base's rim), not something to go and
# pick up. Two checks catch it:
#  - inside a detected base circle, grown by this factor to cover the
#    base's white rim (see split_gems_by_targets)
BASE_EXCLUDE_MARGIN = 1.25
#  - inside ANY big colored blob (a base that wasn't recognized as a clean
#    circle, e.g. half hidden by the robot). Only blobs up to this many times
#    TARGET_AREA_FRACTION count, so a huge floor-colored area can't hide
#    every real gem. Set to 0 to turn this check off.
BIG_BLOB_MAX_AREA_FACTOR = 30
BIG_BLOB_MARGIN = 1.1

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


def get_field_roi_pixels(height, width):
    """
    (Old rectangle boundary - still used as the fallback when no
    click-point polygon has been saved.)

    Convert config.FIELD_ROI_*_FRAC (0.0-1.0 fractions) into an actual
    pixel rectangle (x1, y1, x2, y2) for this frame's size. Clamped and
    sorted so a slider glitch (e.g. X2 < X1) can't produce a broken
    rectangle.
    """

    x1 = int(config.FIELD_ROI_X1_FRAC * width)
    y1 = int(config.FIELD_ROI_Y1_FRAC * height)
    x2 = int(config.FIELD_ROI_X2_FRAC * width)
    y2 = int(config.FIELD_ROI_Y2_FRAC * height)

    x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
    y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))

    return (x1, y1, x2, y2)


def get_field_polygon_pixels(height, width):
    """
    The click-point field area (saved by field_roi_tuner.py) as an
    int32 pixel array shaped (N, 2), or None if no polygon is saved.
    """
    points = field_area.get_points()
    if points is None:
        return None
    return field_area.to_pixels(points, height, width)


def apply_field_roi(hsv_image, roi):
    """
    Zero out everything OUTSIDE the field boundary rectangle so it can
    never be detected as a target circle or gem - this is how a wall
    (or anything else outside the play area) sharing a gem's color
    gets excluded, even though its HSV values would otherwise match.
    """

    x1, y1, x2, y2 = roi
    masked = np.zeros_like(hsv_image)
    masked[y1:y2, x1:x2] = hsv_image[y1:y2, x1:x2]
    return masked


def apply_field_polygon(hsv_image, polygon):
    """
    Same idea as apply_field_roi, but for the click-point shape: zero
    out everything outside the polygon, so a tilted / non-square field
    (and the wall around it) is handled exactly.
    """

    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(hsv_image, hsv_image, mask=mask)


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

    enabled_colors = color_toggles.filter_enabled_colors(config.COLORS)

    for name, (ranges, _box_color) in enabled_colors.items():

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

    Both passes are first restricted to the field area, so anything
    outside it - like a wall that happens to share a gem's color - is
    excluded entirely. The field area is the click-point polygon from
    field_roi_tuner.py if one is saved, otherwise the old rectangle
    (config.FIELD_ROI_*_FRAC).

    Returns (detected_objects, field_gems) - both lists of dicts.
    """

    height, width = image.shape[:2]

    polygon = get_field_polygon_pixels(height, width)
    if polygon is not None:
        hsv_image = apply_field_polygon(hsv_image, polygon)
    else:
        field_roi = get_field_roi_pixels(height, width)
        hsv_image = apply_field_roi(hsv_image, field_roi)

    hsv_for_targets = hsv_image.copy()
    if robot_roi is not None:
        x_min, y_min, x_max, y_max = robot_roi
        hsv_for_targets[y_min:y_max, x_min:x_max] = 0

    min_area = height * width * config.TARGET_AREA_FRACTION
    gem_min_area = height * width * config.GEM_AREA_FRACTION
    padding = config.TARGET_BOX_PADDING

    detected_objects = []
    field_gems = []
    big_blobs = []   # (x, y, radius) of colored blobs too big to be gems

    enabled_colors = color_toggles.filter_enabled_colors(config.COLORS)

    for name, (ranges, box_color) in enabled_colors.items():

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

            if area >= min_area:
                # Too big to be a gem - but remember it, so small blobs
                # sitting inside it can be recognized as part of a base.
                if BIG_BLOB_MAX_AREA_FACTOR > 0 and area <= BIG_BLOB_MAX_AREA_FACTOR * min_area:
                    (bx, by), br = cv2.minEnclosingCircle(contour)
                    big_blobs.append((bx, by, br))
                continue

            if area < gem_min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            field_gems.append({
                "color": name,
                "box_color": box_color,
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "center_x": x + w // 2, "center_y": y + h // 2,
            })

    # Flag gems that sit inside a big colored blob (see BIG_BLOB_* above).
    for gem in field_gems:
        gem["inside_big_blob"] = any(
            math.hypot(gem["center_x"] - bx, gem["center_y"] - by) <= br * BIG_BLOB_MARGIN
            for bx, by, br in big_blobs
        )

    return detected_objects, field_gems


def split_gems_by_targets(field_gems, target_circles):
    """
    Split the loose gems into (kept, ignored). A gem is IGNORED - never
    chased, never grabbed - when it is really part of a base:

      - it sits inside a base circle (grown by BASE_EXCLUDE_MARGIN so the
        base's white rim counts, and by the gem's own size so a gem
        overlapping the edge counts), or
      - it was flagged inside_big_blob by detect_target_circles_and_gems.

    Uses the same circle the drawing shows: centered on the base's box,
    radius (width + height) / 4. Call this AFTER target_memory +
    assign_target_indices, so locked bases (even ones hidden under the
    robot) are all included.
    """

    kept = []
    ignored = []

    for gem in field_gems:
        on_a_base = gem.get("inside_big_blob", False)

        if not on_a_base:
            gem_half = max(gem["x2"] - gem["x1"], gem["y2"] - gem["y1"]) / 2.0

            for target in target_circles:
                cx = (target["x1"] + target["x2"]) / 2.0
                cy = (target["y1"] + target["y2"]) / 2.0
                radius = ((target["x2"] - target["x1"]) + (target["y2"] - target["y1"])) / 4.0

                distance = math.hypot(gem["center_x"] - cx, gem["center_y"] - cy)
                if distance <= radius * BASE_EXCLUDE_MARGIN + gem_half:
                    on_a_base = True
                    break

        (ignored if on_a_base else kept).append(gem)

    return kept, ignored


def remove_gems_inside_targets(field_gems, target_circles):
    """Same as split_gems_by_targets, but returns only the gems to keep."""
    return split_gems_by_targets(field_gems, target_circles)[0]


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