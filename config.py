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
            ((88, 60, 160), (105, 149, 255))
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
# ESP32 CONNECTION SETTINGS
# ============================================================

# EDIT THIS — the ESP32's IP address on your Wi-Fi network. Get it from
# the Arduino Serial Monitor after the ESP32 connects
# (look for "[WIFI] Connected! IP address: ...").
ESP32_IP = "10.57.103.178"

ESP32_PORT = 80
ESP32_COMMAND_PATH = "/command"

# How long to wait for the ESP32 to respond before giving up on one
# request. Keep this short - if the ESP32 doesn't answer quickly, it's
# better to skip and try again next resend than block the loop.
ESP32_REQUEST_TIMEOUT_SECONDS = 1.0

# esp32_link won't re-send the SAME action more often than this many
# seconds (unless force=True) - avoids flooding the ESP32 with
# identical FORWARD/STOP packets every single video frame. Must stay
# comfortably under the ESP32's own 1.5s auto-stop safety timeout, or
# the robot will stop itself between resends.
MIN_COMMAND_INTERVAL_SECONDS = 0.2


# ============================================================
# FIELD BOUNDARY (excludes stuff like a wall sharing a gem's color)
# Stored as FRACTIONS of frame width/height (0.0-1.0), not raw pixels,
# so it stays valid even if you switch to a different camera
# resolution. Defaults to the whole frame (no exclusion) until you
# run field_roi_tuner.py to set it.
# ============================================================

FIELD_ROI_X1_FRAC = 0.0
FIELD_ROI_Y1_FRAC = 0.0
FIELD_ROI_X2_FRAC = 1.0
FIELD_ROI_Y2_FRAC = 1.0


# ============================================================
# PICKUP / PLACEMENT CIRCLE (in front of the robot)
# ============================================================

GRIPPER_FORWARD_OFFSET = 80
PICKUP_RADIUS = 25
PICKUP_DISTANCE = 25          # gem/target center must be within this to count

# While approaching a gem (not holding anything yet), the gripper opens
# proactively once the gem is within THIS distance - wider than
# PICKUP_DISTANCE - so the jaws are already open by the time the robot
# gets close, instead of arriving with jaws still closed from the last
# cycle and pushing the gem out of the way. Must stay bigger than
# PICKUP_DISTANCE (open early, then close once truly aligned).
PICKUP_PREOPEN_DISTANCE = 70

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

# Once the robot commits to chasing a specific gem (or a specific
# target circle), it keeps chasing THAT one - matched by color + being
# within this many pixels of where it was last seen - instead of
# re-picking "whichever is nearest" fresh every frame. Without this,
# the nearest gem can flip between two candidates as the robot moves,
# making it constantly re-aim instead of committing to one and
# finishing the pickup. Only when the locked gem/target can no longer
# be found nearby (picked up, or a false-positive that vanished) does
# it pick a new one. Raise this if real detection jitter is causing
# the lock to break too easily; lower it if it's locking onto the
# wrong object when two of the same color are close together.
TARGET_LOCK_MAX_DRIFT_PX = 40


# ============================================================
# AUTO-LOAD TUNED ROBOT SETTINGS (from robot_config_tuner.py)
# If robot_overrides.json exists next to this file, it overrides the
# pickup/navigation constants above (GRIPPER_FORWARD_OFFSET,
# PICKUP_RADIUS, PICKUP_DISTANCE, TURN_ANGLE_THRESHOLD, STOP_DISTANCE,
# ROBOT_MASK_PADDING, GEM_COLOR_MIN_RATIO). Re-tune any time by running
# robot_config_tuner.py and pressing 's' - no code editing needed.
# ============================================================

import json
import os

_ROBOT_OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_overrides.json")

# Only these names can be overridden this way - guards against a typo'd
# or malicious JSON key silently creating/overwriting an unrelated
# module attribute.
_ROBOT_OVERRIDABLE_NAMES = {
    "GRIPPER_FORWARD_OFFSET", "PICKUP_RADIUS", "PICKUP_DISTANCE",
    "TURN_ANGLE_THRESHOLD", "STOP_DISTANCE", "ROBOT_MASK_PADDING",
    "GEM_COLOR_MIN_RATIO", "TARGET_LOCK_MAX_DRIFT_PX", "PICKUP_PREOPEN_DISTANCE",
    "FIELD_ROI_X1_FRAC", "FIELD_ROI_Y1_FRAC", "FIELD_ROI_X2_FRAC", "FIELD_ROI_Y2_FRAC",
}


def _load_robot_overrides():

    if not os.path.exists(_ROBOT_OVERRIDE_PATH):
        return

    try:
        with open(_ROBOT_OVERRIDE_PATH, "r") as f:
            overrides = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[config] Warning: could not read {_ROBOT_OVERRIDE_PATH} ({e}); using built-in values.")
        return

    # These must stay integers - cv2 drawing calls (circle radius, box
    # padding, etc.) require an int and crash on a float. Python's `/`
    # division always returns a float even when dividing by 1 (e.g.
    # 30 / 1 == 30.0), so a saved override can end up as a float even
    # though the built-in default is an int - cast these back
    # explicitly regardless of what the JSON actually contained.
    _INT_NAMES = {
        "GRIPPER_FORWARD_OFFSET", "PICKUP_RADIUS", "PICKUP_DISTANCE",
        "TURN_ANGLE_THRESHOLD", "STOP_DISTANCE", "ROBOT_MASK_PADDING",
        "TARGET_LOCK_MAX_DRIFT_PX", "PICKUP_PREOPEN_DISTANCE",
    }

    for name, value in overrides.items():

        if name not in _ROBOT_OVERRIDABLE_NAMES:
            print(f"[config] Warning: robot_overrides.json has unknown setting '{name}', skipping.")
            continue

        if name in _INT_NAMES:
            value = int(round(value))

        globals()[name] = value

    # GEM_SAMPLE_RADIUS is derived from PICKUP_RADIUS - keep it in sync
    # if PICKUP_RADIUS was just overridden.
    globals()["GEM_SAMPLE_RADIUS"] = globals()["PICKUP_RADIUS"]

    print(f"[config] Loaded tuned robot settings from {_ROBOT_OVERRIDE_PATH}")


_load_robot_overrides()


# ============================================================
# AUTO-LOAD TUNED HSV RANGES (from hsv_tuner.py)
# If hsv_overrides.json exists next to this file, it overrides the
# built-in COLORS ranges above (box colors stay from COLORS). This
# means you can re-tune colors any day just by running hsv_tuner.py
# and pressing 's' - no code editing needed, ever.
# ============================================================

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