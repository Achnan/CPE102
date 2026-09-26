# ============================================================
# COLOR SETTINGS
# Each entry: name -> (list of (lower_hsv, upper_hsv) ranges, box_color_bgr)
# ============================================================

COLORS = {
    "red": (
        [
            ((0, 80, 35), (7, 255, 255)),
            ((165, 70, 35), (179, 255, 255))
        ],
        (0, 0, 255)
    ),

    "green": (
        [
            # Your target: HSV(140°, 36%, 68%)
            # OpenCV approximate center: (70, 92, 173)
            ((55, 40, 110), (85, 160, 230))
        ],
        (0, 180, 0)
    ),

    "blue": (
        [
            # Strong/dark normal blue:
            # H = 90-135, S must be high
            ((90, 150, 30), (135, 255, 210))
        ],
        (255, 0, 0)
    ),

    "orange": (
        [
            ((8, 90, 50), (20, 255, 255))
        ],
        (0, 100, 255)
    ),

    "purple": (
        [
            ((145, 80, 40), (164, 255, 255))
        ],
        (180, 0, 180)
    ),

    "light_blue": (
        [
            # Your target: HSV(191°, 45%, 75%)
            # OpenCV approximate center: (95, 115, 191)
            ((83, 81, 157), (107, 179, 255))
        ],
        (191, 131, 105)  # Light-blue label/box color in BGR
    ),
}


# ============================================================
# ARUCO / ROBOT SETTINGS
# ============================================================

ROBOT_MARKER_ID = 0
ARUCO_DICT_NAME = "DICT_4X4_50"

# How far around the marker (in pixels) to blank out when looking
# for TARGET CIRCLES, so the robot body itself is never mistaken
# for one. Field gems are NOT blanked with this.
ROBOT_MASK_PADDING = 60


# ============================================================
# PICKUP / PLACEMENT CIRCLE (in front of the robot)
# ============================================================

GRIPPER_FORWARD_OFFSET = 80
PICKUP_RADIUS = 25
PICKUP_DISTANCE = 25          # gem/target center must be within this to count
GEM_SAMPLE_RADIUS = PICKUP_RADIUS
GEM_COLOR_MIN_RATIO = 0.15    # min fraction of sample area to call a color "held"


# ============================================================
# DETECTION / MORPHOLOGY SETTINGS
# ============================================================

TARGET_AREA_FRACTION = 0.002      # min area (of frame) to count as a target circle
GEM_AREA_FRACTION = 0.00015       # min area to count as a loose gem
TARGET_BOX_PADDING = 20
MORPH_KERNEL_SIZE = 5

# Flatten brightness differences across the frame / across the day
# before HSV thresholding (CLAHE on the V channel only). Turn this
# off if you'd rather tune raw, un-normalized HSV values.
USE_BRIGHTNESS_NORMALIZATION = True


# ============================================================
# FIELD LAYOUT SETTINGS
# ============================================================

SIDE_DIVISION = 0.5   # fraction of frame height splitting TOP / BOTTOM

TURN_ANGLE_THRESHOLD = 15   # degrees - "close enough" to forward
STOP_DISTANCE = 50          # pixels - drawn "aim" circle radius around a target


# ============================================================
# AUTO-LOAD TUNED HSV RANGES (from hsv_tuner.py)
# If hsv_overrides.json exists next to this file, it overrides the
# built-in COLORS ranges above (box colors stay from COLORS). This
# means you can re-tune colors any day just by running hsv_tuner.py
# and pressing 's' - no code editing needed, ever.
# ============================================================

import json
import os

_OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsv_overrides.json")


def _load_hsv_overrides():

    if not os.path.exists(_OVERRIDE_PATH):
        return

    try:
        with open(_OVERRIDE_PATH, "r") as f:
            overrides = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[config] Warning: could not read {_OVERRIDE_PATH} ({e}); using built-in HSV ranges.")
        return

    for name, ranges in overrides.items():

        if name not in COLORS:
            print(f"[config] Warning: hsv_overrides.json has unknown color '{name}', skipping.")
            continue

        _old_ranges, box_color = COLORS[name]
        new_ranges = [(tuple(lower), tuple(upper)) for lower, upper in ranges]
        COLORS[name] = (new_ranges, box_color)

    print(f"[config] Loaded tuned HSV ranges from {_OVERRIDE_PATH}")


_load_hsv_overrides()