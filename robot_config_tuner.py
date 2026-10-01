"""
Robot Config Tuner
===================
A settings panel for everything that used to require editing code:
    - Drive speed / turn speed / turn duty % (auto-pushed to the ESP32
      live as you drag - no re-upload, and no key needed to apply it).
      Turn Duty % is the fix for "turn speed is still too fast no matter
      what I set" - see the note above ESP32_SLIDERS below for why.
    - Gripper open/closed angle for BOTH servos - servo 1 (pin 32) and
      servo 2 (pin 33) each have their own open/closed sliders, so you
      can test and tune each side of the claw independently
      (auto-pushed the same way)
    - Wi-Fi-loss command timeout (auto-pushed the same way)
    - Pickup radius / pickup distance / gripper offset / turn angle
      threshold / stop-circle size / robot-mask padding / held-color
      ratio (Python-side - saved to robot_overrides.json, which
      config.py auto-loads next time main.py runs)

ESP32-side sliders (Drive Speed, Turn Speed, both servos' angles, Cmd
Timeout) push to the robot automatically about 0.3s after you stop
moving the slider - drag it, let go, and the robot updates on its own.
Press 'e' any time to force an immediate push without waiting.

Gripper angles only move the claw the next time it grabs/releases, so
after changing one, press 'g' (grab) or 'r' (release) here to see it.

Drive the robot directly from here to feel the effect of a new speed
or gripper setting, without running the full vision pipeline. Each
key press sends ONE command - the ESP32's own command-timeout failsafe
will auto-stop it if you don't press anything else, so tap again to
keep moving or press space to stop now.

Controls:
    i / k / j / l   - forward / backward / left / right (one nudge each)
    space           - stop immediately
    g               - grab (close gripper - both servos)
    r               - release (open gripper - both servos)
    1 / 2           - nudge servo 1 (left)  angle down / up by 5 degrees
                      and send it IMMEDIATELY - moves ONLY that servo,
                      no mirroring (for testing/finding its real
                      open/closed angles)
    3 / 4           - nudge servo 2 (right) angle down / up by 5 degrees,
                      same as above but for the other servo
    e               - force-push the ESP32 sliders right now (bypasses
                      the auto-push debounce - useful if you want it
                      to apply instantly instead of waiting ~0.3s)
    p               - save the Python-side sliders to robot_overrides.json
                      (takes effect next time you run main.py)
    b               - both of the above at once
    q / ESC         - quit (sends a final STOP first)
"""

import json
import os
import time

import cv2
import numpy as np
import requests

import config
import esp32_link

ROBOT_OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_overrides.json")

WINDOW = "Robot Config Tuner"

# How often the ESP32-side sliders (speed/gripper/timeout) are allowed
# to auto-push a new value while being dragged. Without this, moving a
# slider fires a new HTTP request on every single pixel of movement -
# this waits until the slider has been still for this long before
# actually sending, so a drag sends one request when you stop, not
# fifty while you're moving it.
AUTO_PUSH_DEBOUNCE_SECONDS = 0.3

# Degrees per press of the manual per-servo test keys (1/2/3/4).
SERVO_TEST_STEP_DEG = 5

# ---- ESP32-side sliders: (label, min, max, default) ----
# Servo defaults match the two-servo firmware.
# TURN SPEED FIX: on many small DC gear motors, a single PWM value can't
# give a real slow turn - below some PWM the motor doesn't move at all, and
# just above that it jumps almost straight to fast, with no usable middle
# ground. "Turn Duty" fixes this: the firmware pulses the motor ON (at "Turn
# Speed") and OFF rapidly while turning, and Turn Duty is the percentage of
# time spent ON - THIS is what actually controls the average turn speed now.
# If turning still feels too fast at low Turn Duty, also try RAISING Turn
# Speed itself (so each pulse has enough torque to actually move) while
# LOWERING Turn Duty (so it moves less of the time) - e.g. Turn Speed 220,
# Turn Duty 25, rather than a weak constant PWM that barely moves at all.
ESP32_SLIDERS = [
    ("Drive Speed", 0, 255, 160),
    ("Turn Speed", 0, 255, 140),
    ("Turn Duty", 1, 100, 60),           # % of each pulse cycle spent ON while turning
    ("Grip1 Open", 0, 180, 140),        # servo 1 (pin 32)
    ("Grip1 Close", 0, 180, 95),        # servo 1
    ("Grip2 Open", 0, 180, 0),          # servo 2 (pin 33)
    ("Grip2 Close", 0, 180, 45),        # servo 2
    ("Cmd TO x100", 5, 50, 15),   # 5-50 -> 500-5000ms, step 100ms
]

# Native OpenCV trackbar labels get truncated (especially on macOS) past
# roughly 12-14 characters, no matter how wide the window is - that's why
# the labels above are kept short. PRETTY_LABEL supplies the full
# descriptive text used ONLY in our own drawn panel below, which has no
# such limit.
PRETTY_LABEL = {
    "Drive Speed": "Drive Speed",
    "Turn Speed": "Turn Speed",
    "Turn Duty": "Turn Duty % (avg turn speed)",
    "Grip1 Open": "Gripper1 Open Angle",
    "Grip1 Close": "Gripper1 Closed Angle",
    "Grip2 Open": "Gripper2 Open Angle",
    "Grip2 Close": "Gripper2 Closed Angle",
    "Cmd TO x100": "Cmd Timeout ms (x100)",
    "Grip Offset": "Gripper Fwd Offset",
    "Pickup Rad": "Pickup Radius",
    "Pickup Dist": "Pickup Distance",
    "Turn Thresh": "Turn Angle Threshold",
    "Stop Rad": "Stop Circle Radius",
    "Mask Pad": "Robot Mask Padding",
    "HeldMin x100": "Held Color Min % (x100)",
    "Speed:Gem": "Drive Speed - hunting for a gem",
    "Speed:Base": "Drive Speed - returning to base",
}

# ---- Python-side sliders: (label, min, max, default, config_name, scale) ----
# scale converts the integer trackbar position back to the real value
# (e.g. GEM_COLOR_MIN_RATIO is stored 0.0-1.0 but a trackbar needs ints,
# so we use a 0-100 slider and divide by 100).
PYTHON_SLIDERS = [
    ("Grip Offset", 0, 200, config.GRIPPER_FORWARD_OFFSET, "GRIPPER_FORWARD_OFFSET", 1),
    ("Pickup Rad", 5, 100, config.PICKUP_RADIUS, "PICKUP_RADIUS", 1),
    ("Pickup Dist", 5, 100, config.PICKUP_DISTANCE, "PICKUP_DISTANCE", 1),
    ("Turn Thresh", 1, 45, config.TURN_ANGLE_THRESHOLD, "TURN_ANGLE_THRESHOLD", 1),
    ("Stop Rad", 10, 150, config.STOP_DISTANCE, "STOP_DISTANCE", 1),
    ("Mask Pad", 0, 150, config.ROBOT_MASK_PADDING, "ROBOT_MASK_PADDING", 1),
    ("HeldMin x100", 5, 60, int(config.GEM_COLOR_MIN_RATIO * 100), "GEM_COLOR_MIN_RATIO", 100),
    # how far past a base's drawn circle a gem still counts as ON the base (x100: 125 = 1.25)
    ("BaseRim x100", 100, 250, int(round(getattr(config, "BASE_EXCLUDE_MARGIN", 1.25) * 100)), "BASE_EXCLUDE_MARGIN", 100),
    # Manual two-speed drive - main.py pushes whichever of these matches the
    # robot's current phase (not holding a gem vs holding one) to the
    # ESP32's DRIVE_SPEED live, whenever that phase changes. Both are
    # defined in config.py (and loaded from robot_overrides.json there).
    ("Speed:Gem", 0, 255, config.DRIVE_SPEED_GEM, "DRIVE_SPEED_GEM", 1),
    ("Speed:Base", 0, 255, config.DRIVE_SPEED_BASE, "DRIVE_SPEED_BASE", 1),
]

# The plain "Drive Speed" slider above (in ESP32_SLIDERS) is still only for
# this tuner's own manual test-driving (i/j/k/l) - it has no lasting effect
# once main.py is running, since main.py pushes Speed:Gem / Speed:Base
# live based on which phase the robot is in.


def _nothing(_value):
    pass


def fetch_esp32_config():
    """GET the ESP32's current live settings, so sliders start where the
    robot actually is right now instead of always resetting to defaults."""

    url = f"http://{config.ESP32_IP}:{config.ESP32_PORT}/config"
    try:
        response = requests.get(url, timeout=config.ESP32_REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[robot_config_tuner] Could not fetch current ESP32 config: {e}")
    return None


def push_esp32_config(values):
    """POST new speed/gripper/timeout settings to the ESP32 - takes effect immediately."""

    url = f"http://{config.ESP32_IP}:{config.ESP32_PORT}/config"
    payload = {
        "drive_speed": values["Drive Speed"],
        "turn_speed": values["Turn Speed"],
        "turn_duty_percent": values["Turn Duty"],
        "gripper_open_angle": values["Grip1 Open"],
        "gripper_closed_angle": values["Grip1 Close"],
        "gripper2_open_angle": values["Grip2 Open"],
        "gripper2_closed_angle": values["Grip2 Close"],
        "command_timeout_ms": values["Cmd TO x100"] * 100,
    }
    try:
        response = requests.post(url, json=payload, timeout=config.ESP32_REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            print(f"[robot_config_tuner] Pushed to ESP32: {response.json()}")
            return True
        print(f"[robot_config_tuner] ESP32 rejected config push: {response.status_code} {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[robot_config_tuner] Failed to push config to ESP32: {e}")
    return False


def push_servo_test_angle(servo1_angle=None, servo2_angle=None):
    """
    Move ONE servo directly to an exact angle, bypassing GRAB/RELEASE and
    the mirroring logic entirely - for testing each side independently.
    Pass only the one(s) you want to move; None leaves the other alone.
    Not saved to flash on the ESP32 - it's a live test move only.
    """

    url = f"http://{config.ESP32_IP}:{config.ESP32_PORT}/config"
    payload = {}
    if servo1_angle is not None:
        payload["servo1_angle"] = int(servo1_angle)
    if servo2_angle is not None:
        payload["servo2_angle"] = int(servo2_angle)
    if not payload:
        return False

    try:
        response = requests.post(url, json=payload, timeout=config.ESP32_REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            print(f"[robot_config_tuner] Manual servo move: {payload} -> {response.json()}")
            return True
        print(f"[robot_config_tuner] ESP32 rejected servo test move: {response.status_code} {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[robot_config_tuner] Failed to send servo test move: {e}")
    return False


def save_python_overrides(values):
    """
    Save the Python-side (vision/navigation) sliders to
    robot_overrides.json. Merges with whatever's already in the file
    rather than overwriting it, since field_roi_tuner.py writes to
    this same file for the field-boundary settings - overwriting
    outright would wipe those out.
    """

    existing = {}
    if os.path.exists(ROBOT_OVERRIDE_PATH):
        try:
            with open(ROBOT_OVERRIDE_PATH, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    for label, _lo, _hi, _default, name, scale in PYTHON_SLIDERS:
        # scale == 1 means this is really an integer setting (pixels,
        # degrees) - keep it an int. Only scale != 1 (e.g. the 0-100
        # percent slider for GEM_COLOR_MIN_RATIO) actually needs float
        # division.
        existing[name] = values[label] if scale == 1 else values[label] / scale

    with open(ROBOT_OVERRIDE_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[robot_config_tuner] Saved Python-side settings to {ROBOT_OVERRIDE_PATH}")



# ============================================================================
# UI drawing helpers (all in one place so main()'s loop stays readable)
# ============================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG_COLOR = (28, 28, 30)
PANEL_COLOR = (40, 40, 44)
DIVIDER_COLOR = (60, 60, 64)
TEXT_COLOR = (225, 225, 225)
MUTED_TEXT = (150, 150, 150)
ACCENT = (60, 200, 255)      # cyan - ESP32/robot-side settings
ACCENT_2 = (255, 150, 60)    # orange - Python-side settings
OK_COLOR = (90, 220, 90)
WARN_COLOR = (60, 200, 255)
FAIL_COLOR = (80, 80, 235)
CANVAS_WIDTH = 620
MARGIN = 16


def draw_panel_bg(canvas, y0, y1, color=PANEL_COLOR):
    cv2.rectangle(canvas, (MARGIN - 8, y0), (canvas.shape[1] - MARGIN + 8, y1), color, -1)


def draw_section_title(canvas, y, text, color=ACCENT):
    cv2.putText(canvas, text, (MARGIN, y), FONT, 0.6, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (MARGIN, y + 8), (canvas.shape[1] - MARGIN, y + 8), DIVIDER_COLOR, 1)
    return y + 26


def draw_value_bar(canvas, y, label, value, lo, hi, bar_color, value_text=None):
    """One row: label on the left, a filled progress bar + numeric value on
    the right, showing where `value` sits between lo and hi."""

    label_w = 180
    bar_x = MARGIN + label_w
    bar_w = canvas.shape[1] - MARGIN - bar_x - 60
    bar_h = 12
    bar_y = y - bar_h + 3

    cv2.putText(canvas, label, (MARGIN, y), FONT, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 55, 58), -1)
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / float(hi - lo)))
    fill_w = int(bar_w * frac)
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 94), 1)

    text = value_text if value_text is not None else str(value)
    cv2.putText(canvas, text, (bar_x + bar_w + 10, y), FONT, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)

    return y + 24


def draw_pill(canvas, x, y, text, color):
    """Small rounded status badge (approximated with a filled rectangle)."""

    (tw, th), _ = cv2.getTextSize(text, FONT, 0.48, 1)
    pad_x, pad_y = 10, 6
    x1, y1 = x, y - th - pad_y
    x2, y2 = x + tw + pad_x * 2, y + pad_y
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
    cv2.putText(canvas, text, (x1 + pad_x, y - 2), FONT, 0.48, (15, 15, 15), 1, cv2.LINE_AA)
    return x2


def draw_key_chip(canvas, x, y, key_text, desc_text):
    """One 'keycap + description' chip, used in the control legend. Returns
    the x position AFTER the description text, so the caller can safely
    start the next chip there without overlapping this one."""

    (tw, th), _ = cv2.getTextSize(key_text, FONT, 0.5, 1)
    (dw, _), _ = cv2.getTextSize(desc_text, FONT, 0.46, 1)
    chip_w = tw + 16
    chip_h = 22
    cv2.rectangle(canvas, (x, y - chip_h + 4), (x + chip_w, y + 4), (65, 65, 70), -1)
    cv2.rectangle(canvas, (x, y - chip_h + 4), (x + chip_w, y + 4), (110, 110, 115), 1)
    cv2.putText(canvas, key_text, (x + 8, y - 4), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, desc_text, (x + chip_w + 8, y - 4), FONT, 0.46, MUTED_TEXT, 1, cv2.LINE_AA)
    return x + chip_w + 8 + dw


def main():

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    # Resizable so you can drag it bigger or maximize it (the OS maximize
    # button / double-click title bar). A plain OpenCV window can't force
    # true borderless fullscreen while keeping the trackbars usable and
    # working the same across Windows/macOS/Linux, so "maximize the window"
    # is the reliable way to get a bigger view here.
    cv2.resizeWindow(WINDOW, CANVAS_WIDTH, 900)

    all_sliders = [(label, lo, hi) for label, lo, hi, _default in ESP32_SLIDERS]
    all_sliders += [(label, lo, hi) for label, lo, hi, _default, _name, _scale in PYTHON_SLIDERS]

    for label, lo, hi in all_sliders:
        cv2.createTrackbar(label, WINDOW, lo, hi, _nothing)

    # ---- seed ESP32 sliders from the robot's actual current values ----
    live_config = fetch_esp32_config()
    if live_config:
        cv2.setTrackbarPos("Drive Speed", WINDOW, live_config.get("drive_speed", 160))
        cv2.setTrackbarPos("Turn Speed", WINDOW, live_config.get("turn_speed", 140))
        cv2.setTrackbarPos("Turn Duty", WINDOW, live_config.get("turn_duty_percent", 60))
        cv2.setTrackbarPos("Grip1 Open", WINDOW, live_config.get("gripper_open_angle", 140))
        cv2.setTrackbarPos("Grip1 Close", WINDOW, live_config.get("gripper_closed_angle", 95))
        cv2.setTrackbarPos("Grip2 Open", WINDOW, live_config.get("gripper2_open_angle", 0))
        cv2.setTrackbarPos("Grip2 Close", WINDOW, live_config.get("gripper2_closed_angle", 45))
        cv2.setTrackbarPos(
            "Cmd TO x100", WINDOW,
            int(live_config.get("command_timeout_ms", 1500) / 100)
        )
        print("[robot_config_tuner] Loaded current settings from the ESP32.")
    else:
        for label, _lo, _hi, default in ESP32_SLIDERS:
            cv2.setTrackbarPos(label, WINDOW, default)
        print("[robot_config_tuner] Could not reach ESP32 - sliders start at defaults. "
              "Live test driving (i/j/k/l) won't work until it's reachable.")

    # ---- manual per-servo test angles (independent of the open/closed
    # sliders above) - start from whatever the ESP32 reports it's actually
    # at right now, so pressing 1/2/3/4 nudges from the real position.
    if live_config:
        servo1_test_angle = int(live_config.get("servo1_current_angle", 90))
        servo2_test_angle = int(live_config.get("servo2_current_angle", 90))
    else:
        servo1_test_angle = 90
        servo2_test_angle = 90

    # ---- seed Python sliders from config.py's current values (already
    # reflects robot_overrides.json if one exists, since config.py loads
    # it on import) ----
    for label, _lo, _hi, _default, name, scale in PYTHON_SLIDERS:
        cv2.setTrackbarPos(label, WINDOW, int(getattr(config, name, _default) * scale))

    print(__doc__)

    # Rough estimate; actual layout below can vary a little based on how
    # many sliders/keys there are, so this pads generously and the drawing
    # code just uses however much of the canvas it needs.
    canvas_height = 170 + 26 * (len(ESP32_SLIDERS) + 1) + 26 * len(PYTHON_SLIDERS) + 230
    last_action = "stopped"
    last_action_ok = None  # None = nothing sent yet, True/False = last send result

    # ---- auto-push state ----
    # last_seen_esp32_values: the ESP32-relevant slider values as of
    # last loop iteration - used to detect "did a slider just move".
    # last_pushed_esp32_values: what we last successfully sent - used
    # to know whether we're currently in sync with the robot.
    # pending_since: when the sliders most recently changed (resets
    # the debounce timer); None means nothing is waiting to be sent.
    # Seeded from the ACTUAL current trackbar positions (which were
    # just set from the robot's live config, or defaults if it was
    # unreachable) - not the raw ESP32_SLIDERS defaults - so startup
    # correctly shows "synced" instead of firing a redundant push.
    initial_esp32_values = {
        label: cv2.getTrackbarPos(label, WINDOW) for label, _lo, _hi, _default in ESP32_SLIDERS
    }
    last_seen_esp32_values = dict(initial_esp32_values)
    last_pushed_esp32_values = dict(initial_esp32_values) if live_config else None
    pending_since = None
    push_status = "synced" if live_config else "not yet pushed - could not reach robot at startup"

    # Bar color per ESP32 slider label, so speed/gripper/timeout are each
    # visually distinct at a glance.
    esp32_bar_color = {
        "Drive Speed": ACCENT, "Turn Speed": ACCENT, "Turn Duty": (255, 220, 80),
        "Grip1 Open": (200, 120, 255), "Grip1 Close": (200, 120, 255),
        "Grip2 Open": (255, 140, 220), "Grip2 Close": (255, 140, 220),
        "Cmd TO x100": (140, 200, 140),
    }

    while True:

        values = {label: cv2.getTrackbarPos(label, WINDOW) for label, _lo, _hi in all_sliders}

        esp32_values_now = {label: values[label] for label, _lo, _hi, _default in ESP32_SLIDERS}

        if esp32_values_now != last_seen_esp32_values:
            # A slider just moved - (re)start the debounce timer rather
            # than sending immediately, so a drag sends once when you
            # stop, not on every intermediate position.
            pending_since = time.time()
            last_seen_esp32_values = esp32_values_now

        if pending_since is not None and (time.time() - pending_since) >= AUTO_PUSH_DEBOUNCE_SECONDS:
            if push_esp32_config(values):
                last_pushed_esp32_values = dict(esp32_values_now)
                push_status = "synced"
            else:
                push_status = "push failed - robot unreachable?"
            pending_since = None

        # ==================================================================
        # DRAW
        # ==================================================================

        display = np.zeros((canvas_height, CANVAS_WIDTH, 3), dtype=np.uint8)
        display[:] = BG_COLOR

        # ---- title bar ----
        cv2.putText(display, "ROBOT CONFIG TUNER", (MARGIN, 34), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(display, (MARGIN, 46), (CANVAS_WIDTH - MARGIN, 46), DIVIDER_COLOR, 1)

        y = 76

        # ---- ESP32-side section ----
        section_top = y - 22
        y = draw_section_title(display, y, "ESP32  -  speed / gripper / timeout  (live-pushed)", ACCENT)
        for label, lo, hi, _default in ESP32_SLIDERS:
            y = draw_value_bar(display, y, PRETTY_LABEL.get(label, label), values[label], lo, hi,
                                esp32_bar_color.get(label, ACCENT))

        # sync status pill, top-right of this section
        if pending_since is not None:
            status_text = f"sending in {max(0.0, AUTO_PUSH_DEBOUNCE_SECONDS - (time.time() - pending_since)):.1f}s"
            status_color = WARN_COLOR
        elif esp32_values_now == last_pushed_esp32_values:
            status_text = "synced"
            status_color = OK_COLOR
        else:
            status_text = "push failed"
            status_color = FAIL_COLOR
        draw_pill(display, CANVAS_WIDTH - MARGIN - 130, section_top + 20, status_text, status_color)

        y += 6
        y = draw_section_title(display, y, "MANUAL SERVO TEST  -  independent, not mirrored", (255, 200, 0))
        y = draw_value_bar(display, y, "Servo 1 (pin 33)", servo1_test_angle, 0, 180, (255, 170, 60))
        y = draw_value_bar(display, y, "Servo 2 (pin 32)", servo2_test_angle, 0, 180, (255, 170, 60))

        y += 6
        y = draw_section_title(display, y, "PYTHON-SIDE  -  vision / navigation  (save with 'p')", ACCENT_2)
        for label, lo, hi, _default, name, scale in PYTHON_SLIDERS:
            real_value = values[label] / scale
            value_text = f"{real_value:.2f}" if scale != 1 else str(real_value)
            y = draw_value_bar(display, y, PRETTY_LABEL.get(label, label), values[label], lo, hi, ACCENT_2, value_text=value_text)

        # ---- drive status ----
        y += 10
        if last_action_ok is None:
            drive_text, drive_color = f"Last drive command: {last_action}", MUTED_TEXT
        elif last_action_ok:
            drive_text, drive_color = f"Last drive command: {last_action}  ->  reached robot OK", OK_COLOR
        else:
            drive_text, drive_color = f"Last drive command: {last_action}  ->  FAILED to reach robot!", FAIL_COLOR
        cv2.putText(display, drive_text, (MARGIN, y), FONT, 0.55, drive_color, 2, cv2.LINE_AA)
        y += 28

        # ---- control legend, as keycap chips wrapped across rows ----
        y = draw_section_title(display, y, "CONTROLS", MUTED_TEXT)
        chips = [
            ("I/K/J/L", "drive"), ("SPACE", "stop"),
            ("G", "grab"), ("R", "release"),
            ("1/2", "servo1 -/+5"), ("3/4", "servo2 -/+5"),
            ("E", "push now"), ("P", "save settings"),
            ("B", "push + save"), ("Q/ESC", "quit"),
        ]
        x = MARGIN
        row_h = 30
        for key_text, desc_text in chips:
            (tw, _), _ = cv2.getTextSize(key_text, FONT, 0.5, 1)
            (dw, _), _ = cv2.getTextSize(desc_text, FONT, 0.46, 1)
            chip_total_w = tw + 16 + 8 + dw + 18
            if x + chip_total_w > CANVAS_WIDTH - MARGIN:
                x = MARGIN
                y += row_h
            x = draw_key_chip(display, x, y, key_text, desc_text) + 10
        y += row_h

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            esp32_link.send_command("STOP", force=True)
            break

        elif key == ord('i'):
            last_action = "FORWARD"
            last_action_ok = esp32_link.send_command("FORWARD", force=True)

        elif key == ord('k'):
            last_action = "BACKWARD"
            last_action_ok = esp32_link.send_command("BACKWARD", force=True)

        elif key == ord('j'):
            last_action = "LEFT"
            last_action_ok = esp32_link.send_command("LEFT", force=True)

        elif key == ord('l'):
            last_action = "RIGHT"
            last_action_ok = esp32_link.send_command("RIGHT", force=True)

        elif key == ord(' '):
            last_action = "STOP"
            last_action_ok = esp32_link.send_command("STOP", force=True)

        elif key == ord('g'):
            last_action = "GRAB"
            last_action_ok = esp32_link.send_command("GRAB", force=True)

        elif key == ord('r'):
            last_action = "RELEASE"
            last_action_ok = esp32_link.send_command("RELEASE", force=True)

        elif key == ord('1'):
            servo1_test_angle = max(0, servo1_test_angle - SERVO_TEST_STEP_DEG)
            push_servo_test_angle(servo1_angle=servo1_test_angle)

        elif key == ord('2'):
            servo1_test_angle = min(180, servo1_test_angle + SERVO_TEST_STEP_DEG)
            push_servo_test_angle(servo1_angle=servo1_test_angle)

        elif key == ord('3'):
            servo2_test_angle = max(0, servo2_test_angle - SERVO_TEST_STEP_DEG)
            push_servo_test_angle(servo2_angle=servo2_test_angle)

        elif key == ord('4'):
            servo2_test_angle = min(180, servo2_test_angle + SERVO_TEST_STEP_DEG)
            push_servo_test_angle(servo2_angle=servo2_test_angle)

        elif key == ord('e'):
            if push_esp32_config(values):
                last_pushed_esp32_values = dict(esp32_values_now)
                push_status = "synced"
            else:
                push_status = "push failed - robot unreachable?"
            pending_since = None

        elif key == ord('p'):
            save_python_overrides(values)

        elif key == ord('b'):
            if push_esp32_config(values):
                last_pushed_esp32_values = dict(esp32_values_now)
                push_status = "synced"
            else:
                push_status = "push failed - robot unreachable?"
            pending_since = None
            save_python_overrides(values)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()