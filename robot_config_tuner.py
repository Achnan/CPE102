"""
Robot Config Tuner
===================
A settings panel for everything that used to require editing code:
    - Drive speed / turn speed (auto-pushed to the ESP32 live as you
      drag - no re-upload, and no key needed to apply it)
    - Gripper open/closed angle (auto-pushed the same way)
    - Wi-Fi-loss command timeout (auto-pushed the same way)
    - Pickup radius / pickup distance / gripper offset / turn angle
      threshold / stop-circle size / robot-mask padding / held-color
      ratio (Python-side - saved to robot_overrides.json, which
      config.py auto-loads next time main.py runs)

ESP32-side sliders (Drive Speed, Turn Speed, Gripper angles, Cmd
Timeout) push to the robot automatically about 0.3s after you stop
moving the slider - drag it, let go, and the robot updates on its own.
Press 'e' any time to force an immediate push without waiting.

Drive the robot directly from here to feel the effect of a new speed
or gripper setting, without running the full vision pipeline. Each
key press sends ONE command - the ESP32's own command-timeout failsafe
will auto-stop it if you don't press anything else, so tap again to
keep moving or press space to stop now.

Controls:
    i / k / j / l   - forward / backward / left / right (one nudge each)
    space           - stop immediately
    g               - grab (close gripper)
    r               - release (open gripper)
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

# ---- ESP32-side sliders: (label, min, max, default) ----
ESP32_SLIDERS = [
    ("Drive Speed", 0, 255, 160),
    ("Turn Speed", 0, 255, 140),
    ("Gripper Open Angle", 0, 180, 60),
    ("Gripper Closed Angle", 0, 180, 150),
    ("Cmd Timeout ms (x100)", 5, 50, 15),   # 5-50 -> 500-5000ms, step 100ms
]

# ---- Python-side sliders: (label, min, max, default, config_name, scale) ----
# scale converts the integer trackbar position back to the real value
# (e.g. GEM_COLOR_MIN_RATIO is stored 0.0-1.0 but a trackbar needs ints,
# so we use a 0-100 slider and divide by 100).
PYTHON_SLIDERS = [
    ("Gripper Fwd Offset", 0, 200, config.GRIPPER_FORWARD_OFFSET, "GRIPPER_FORWARD_OFFSET", 1),
    ("Pickup Radius", 5, 100, config.PICKUP_RADIUS, "PICKUP_RADIUS", 1),
    ("Pickup Distance", 5, 100, config.PICKUP_DISTANCE, "PICKUP_DISTANCE", 1),
    ("Turn Angle Threshold", 1, 45, config.TURN_ANGLE_THRESHOLD, "TURN_ANGLE_THRESHOLD", 1),
    ("Stop Circle Radius", 10, 150, config.STOP_DISTANCE, "STOP_DISTANCE", 1),
    ("Robot Mask Padding", 0, 150, config.ROBOT_MASK_PADDING, "ROBOT_MASK_PADDING", 1),
    ("Held Color Min % (x100)", 5, 60, int(config.GEM_COLOR_MIN_RATIO * 100), "GEM_COLOR_MIN_RATIO", 100),
]


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
        "gripper_open_angle": values["Gripper Open Angle"],
        "gripper_closed_angle": values["Gripper Closed Angle"],
        "command_timeout_ms": values["Cmd Timeout ms (x100)"] * 100,
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


def main():

    cv2.namedWindow(WINDOW)

    all_sliders = [(label, lo, hi) for label, lo, hi, _default in ESP32_SLIDERS]
    all_sliders += [(label, lo, hi) for label, lo, hi, _default, _name, _scale in PYTHON_SLIDERS]

    for label, lo, hi in all_sliders:
        cv2.createTrackbar(label, WINDOW, lo, hi, _nothing)

    # ---- seed ESP32 sliders from the robot's actual current values ----
    live_config = fetch_esp32_config()
    if live_config:
        cv2.setTrackbarPos("Drive Speed", WINDOW, live_config.get("drive_speed", 170))
        cv2.setTrackbarPos("Turn Speed", WINDOW, live_config.get("turn_speed", 140))
        cv2.setTrackbarPos("Gripper Open Angle", WINDOW, live_config.get("gripper_open_angle", 60))
        cv2.setTrackbarPos("Gripper Closed Angle", WINDOW, live_config.get("gripper_closed_angle", 150))
        cv2.setTrackbarPos(
            "Cmd Timeout ms (x100)", WINDOW,
            int(live_config.get("command_timeout_ms", 1500) / 100)
        )
        print("[robot_config_tuner] Loaded current settings from the ESP32.")
    else:
        for label, _lo, _hi, default in ESP32_SLIDERS:
            cv2.setTrackbarPos(label, WINDOW, default)
        print("[robot_config_tuner] Could not reach ESP32 - sliders start at defaults. "
              "Live test driving (i/j/k/l) won't work until it's reachable.")

    # ---- seed Python sliders from config.py's current values (already
    # reflects robot_overrides.json if one exists, since config.py loads
    # it on import) ----
    for label, _lo, _hi, _default, name, scale in PYTHON_SLIDERS:
        cv2.setTrackbarPos(label, WINDOW, int(getattr(config, name) * scale))

    print(__doc__)

    canvas_height = 60 + 25 * len(all_sliders) + 210
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

        display = np.zeros((canvas_height, 520, 3), dtype=np.uint8)
        display[:] = (40, 40, 40)

        y = 30
        cv2.putText(display, "ESP32 SETTINGS sync (speed/gripper/timeout only):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 30
        for label, _lo, _hi, _default in ESP32_SLIDERS:
            cv2.putText(display, f"{label}: {values[label]}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25

        if pending_since is not None:
            status_text = f"Sending in {max(0.0, AUTO_PUSH_DEBOUNCE_SECONDS - (time.time() - pending_since)):.1f}s..."
            status_color = (0, 255, 255)
        elif esp32_values_now == last_pushed_esp32_values:
            status_text = "Synced with robot"
            status_color = (0, 255, 0)
        else:
            status_text = push_status
            status_color = (0, 0, 255)
        cv2.putText(display, status_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y += 25

        y += 15
        cv2.putText(display, "Python-side (vision/nav) settings:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 30
        for label, _lo, _hi, _default, name, scale in PYTHON_SLIDERS:
            real_value = values[label] / scale
            cv2.putText(display, f"{label}: {real_value}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25

        y += 15
        if last_action_ok is None:
            drive_text = f"Last drive command: {last_action} (none sent yet)"
            drive_color = (200, 200, 200)
        elif last_action_ok:
            drive_text = f"Last drive command: {last_action} -> reached robot OK"
            drive_color = (0, 255, 0)
        else:
            drive_text = f"Last drive command: {last_action} -> FAILED to reach robot!"
            drive_color = (0, 0, 255)
        cv2.putText(display, drive_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, drive_color, 2)
        y += 30
        for line in [
            "i/k/j/l = forward/backward/left/right   space = stop",
            "g = grab   r = release",
            "e = push speeds/gripper to robot now",
            "p = save pickup/nav settings to file    b = both",
            "q / ESC = quit",
        ]:
            cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 22

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