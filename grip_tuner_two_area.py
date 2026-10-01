"""
Grip Tuner - TWO areas (for main_two_area.py)
==============================================
(grip_tuner.py is the one-area tuner for main.py.)
Size and place BOTH gripper areas while WATCHING them on the live camera
image next to the robot.

    HOLD area (magenta) - where a gem must be when the claw CLOSES. The colour
                          inside it tells the robot whether it is holding
                          something, so make it about the size of one gem.
        Hold Offset     - how far in front of the robot's centre it sits (px)
        Hold Radius     - its size (px)
        Arrive Dist     - how close a gem must get to the hold spot to count as
                          "arrived, grab now" (thin green ring). Keep it about
                          the hold radius plus a little.

    TIP area (cyan)     - the claw's front end. The robot stops and opens the
                          claw before a gem reaches it, and only creeps once a
                          gem is inside it.
        Tip Offset      - how far in front of the robot's centre it sits (px)
        Tip Radius      - its size (px). Bigger = the claw opens earlier.

    Grip Angle          - rotates the direction both areas are measured in,
                          away from the robot's raw ArUco heading. Use it if the
                          top-view camera is mounted at a slight slant, so
                          "straight ahead" in the image isn't where the claw
                          really points.

How to set it (this matters more than the exact numbers):
    1. Press 'o' to OPEN the claw and look at it on screen.
    2. Put the TIP circle's centre on the very front of the open claw.
    3. Put a gem where it should sit when the claw closes, and move the HOLD
       circle onto it. Shrink Hold Radius until the circle is about one gem.
    4. Turn the robot to a few different headings; both circles should stay on
       the right parts of the claw at every heading (if not, adjust Grip Angle).
    5. Press 's' to save.

Controls:
    o        - OPEN the claw   (sends RELEASE to the robot)
    c        - CLOSE the claw  (sends GRAB to the robot)
    s        - save everything to robot_overrides.json
               (main.py picks it up next time it starts)
    q / ESC  - quit

The robot must be in view so its ArUco marker is detected. Don't run
this at the same time as main.py - only one program can use the camera.
"""

import json
import math
import os

import cv2

import config
import drawing
import esp32_link
import robot_tracker

ROBOT_OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_overrides.json")

WINDOW = "Grip Tuner"

OFFSET_MAX = 300
RADIUS_MIN, RADIUS_MAX = 5, 100
ANGLE_MAX_DEG = 45   # slider covers -45 to +45 degrees of slant

# (trackbar name, config name, min, max)  - names are short on purpose: native
# OpenCV trackbar labels get cut off past ~12 characters on macOS.
SLIDERS = [
    ("Hold Offset", "GRIPPER_FORWARD_OFFSET", 0, OFFSET_MAX),
    ("Hold Radius", "PICKUP_RADIUS", RADIUS_MIN, RADIUS_MAX),
    ("Arrive Dist", "PICKUP_DISTANCE", RADIUS_MIN, RADIUS_MAX),
    ("Tip Offset", "GRIPPER_TIP_OFFSET", 0, OFFSET_MAX),
    ("Tip Radius", "GRIPPER_TIP_RADIUS", RADIUS_MIN, RADIUS_MAX),
]
ANGLE_SLIDER = "Grip Angle"

# Used when config.py is an older one without the tip settings (so the tuner
# still opens; saving them only takes effect with the newer config.py).
TIP_DEFAULTS = {"GRIPPER_TIP_OFFSET": 105, "GRIPPER_TIP_RADIUS": 30}

HOLD_COLOR = (255, 0, 255)
TIP_COLOR = (255, 220, 0)
ARRIVE_COLOR = (0, 220, 0)


def _nothing(_value):
    pass


def save_values(values, angle_deg):
    """Merge into robot_overrides.json so other saved settings are kept."""

    existing = {}
    if os.path.exists(ROBOT_OVERRIDE_PATH):
        try:
            with open(ROBOT_OVERRIDE_PATH, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    for name, value in values.items():
        existing[name] = int(value)
    existing["GRIPPER_ANGLE_OFFSET_DEG"] = int(angle_deg)

    with open(ROBOT_OVERRIDE_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[grip_tuner] Saved {values} angle={angle_deg} to {ROBOT_OVERRIDE_PATH}")


def _put(img, text, y, color=(0, 255, 255), scale=0.55):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return y + 26


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    detector = robot_tracker.make_detector()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for trackbar, name, lo, hi in SLIDERS:
        start = min(max(int(getattr(config, name, TIP_DEFAULTS.get(name, 0))), lo), hi)
        cv2.createTrackbar(trackbar, WINDOW, start, hi, _nothing)
    start_angle = min(max(int(getattr(config, "GRIPPER_ANGLE_OFFSET_DEG", 0)), -ANGLE_MAX_DEG), ANGLE_MAX_DEG)
    cv2.createTrackbar(ANGLE_SLIDER, WINDOW, start_angle + ANGLE_MAX_DEG, ANGLE_MAX_DEG * 2, _nothing)

    print(__doc__)

    claw_text = "claw: unknown (press o / c)"
    claw_color = (200, 200, 200)
    status, status_frames_left = "", 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera")
            break

        values = {}
        for trackbar, name, lo, hi in SLIDERS:
            values[name] = max(lo, cv2.getTrackbarPos(trackbar, WINDOW))
        angle_deg = cv2.getTrackbarPos(ANGLE_SLIDER, WINDOW) - ANGLE_MAX_DEG

        # Push the slider values into config so the SAME functions main.py
        # uses (robot_tracker + drawing) pick them up live.
        for name, value in values.items():
            setattr(config, name, value)
        config.GRIPPER_ANGLE_OFFSET_DEG = angle_deg

        hold_off = values["GRIPPER_FORWARD_OFFSET"]
        hold_r = values["PICKUP_RADIUS"]
        arrive = values["PICKUP_DISTANCE"]
        tip_off = values["GRIPPER_TIP_OFFSET"]
        tip_r = values["GRIPPER_TIP_RADIUS"]

        result = frame.copy()

        robot_info = robot_tracker.detect_robot(frame, detector)
        drawing.draw_robot_heading(result, robot_info)

        hold_center = robot_tracker.gripper_center_of(robot_info)
        tip_center = robot_tracker.tip_center_of(robot_info)

        if hold_center is not None and tip_center is not None:
            # thin ring: how close counts as "arrived"
            cv2.circle(result, (int(hold_center[0]), int(hold_center[1])), arrive, ARRIVE_COLOR, 1)
            drawing.draw_tip_circle(result, tip_center)
            drawing.draw_pickup_circle(result, hold_center)
            rc = robot_info["center"]
            cv2.line(result, (int(rc[0]), int(rc[1])), (int(tip_center[0]), int(tip_center[1])),
                     (200, 200, 200), 1)
            cv2.putText(result, "TIP", (int(tip_center[0]) + tip_r + 4, int(tip_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TIP_COLOR, 2, cv2.LINE_AA)
            cv2.putText(result, "HOLD", (int(hold_center[0]) + hold_r + 4, int(hold_center[1]) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, HOLD_COLOR, 2, cv2.LINE_AA)

        y = 25
        y = _put(result, f"HOLD  offset {hold_off}px  radius {hold_r}px   arrive within {arrive}px", y, HOLD_COLOR)
        y = _put(result, f"TIP   offset {tip_off}px  radius {tip_r}px   (bigger radius = claw opens earlier)", y, TIP_COLOR)
        y = _put(result, f"tip is {tip_off - hold_off:+d}px beyond the hold spot   slant {angle_deg:+d} deg", y, (255, 255, 255))
        if tip_off < hold_off:
            y = _put(result, "WARNING: the tip should be FURTHER out than the hold spot", y, (0, 0, 255))
        if arrive > hold_r + 15:
            y = _put(result, "note: Arrive Dist is much bigger than the hold circle - a gem can count as arrived yet sit outside it",
                     y, (0, 165, 255), 0.45)
        if robot_info["center"] is None:
            y = _put(result, "Robot not detected - put it in view", y, (0, 0, 255))
        y = _put(result, claw_text, y, claw_color)

        if status_frames_left > 0:
            y = _put(result, status, y, (0, 255, 0))
            status_frames_left -= 1

        cv2.putText(result, "o = open claw   c = close claw   s = save   q = quit",
                    (10, result.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(WINDOW, result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('o'):
            ok = esp32_link.send_command("RELEASE", force=True)
            claw_text, claw_color = (("claw: OPEN command reached the robot", (0, 255, 0)) if ok
                                     else ("claw: OPEN command FAILED to reach the robot", (0, 0, 255)))
        elif key == ord('c'):
            ok = esp32_link.send_command("GRAB", force=True)
            claw_text, claw_color = (("claw: CLOSE command reached the robot", (0, 255, 0)) if ok
                                     else ("claw: CLOSE command FAILED to reach the robot", (0, 0, 255)))
        elif key == ord('s'):
            save_values(values, angle_deg)
            status = "Saved! Restart main.py to apply."
            status_frames_left = 90

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()