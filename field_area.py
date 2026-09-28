"""
field_area.py
=============
Shared helper for the field-area POLYGON (any number of corner points,
not just a rectangle).

The points are stored in robot_overrides.json under "FIELD_ROI_POINTS"
as a list of [x, y] pairs, each a FRACTION of the frame (0.0 - 1.0), so
they still work if the camera resolution changes. Example:

    "FIELD_ROI_POINTS": [[0.12, 0.10], [0.88, 0.14], [0.92, 0.90], [0.08, 0.86]]

Used by field_roi_tuner.py (draw/save), vision.py (mask out everything
outside the area) and drawing.py (show the outline).

The points are read ONCE when the program starts - restart main.py
after changing them.
"""

import json
import math
import os

import numpy as np

OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_overrides.json")
KEY = "FIELD_ROI_POINTS"


def _valid(points):
    if not isinstance(points, list) or len(points) < 3:
        return False
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            return False
        if not all(isinstance(v, (int, float)) for v in p):
            return False
    return True


def load_points():
    """Saved polygon as a list of [fx, fy], or None if none is saved."""
    try:
        with open(OVERRIDE_PATH, "r") as f:
            points = json.load(f).get(KEY)
    except (OSError, json.JSONDecodeError):
        return None
    return [list(map(float, p)) for p in points] if _valid(points) else None


def save_points(points):
    """Save the polygon (or remove it when points is None/too short),
    keeping every other setting in robot_overrides.json."""
    try:
        with open(OVERRIDE_PATH, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}

    if points is not None and len(points) >= 3:
        data[KEY] = [[round(x, 4), round(y, 4)] for x, y in points]
    else:
        data.pop(KEY, None)

    with open(OVERRIDE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def order_points(points):
    """Sort points by angle around their center, so the polygon never
    crosses itself no matter what order they were clicked in. (Assumes
    a roughly convex area, which a field normally is.)"""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def to_pixels(points, height, width):
    """Fractions -> int32 pixel array shaped (N, 2) for cv2.fillPoly/polylines."""
    return np.array(
        [[int(x * width), int(y * height)] for x, y in points], dtype=np.int32
    )


# ---- loaded once at import (i.e. once per program start) ----
_points = load_points()


def get_points():
    return _points