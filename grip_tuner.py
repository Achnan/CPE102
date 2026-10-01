"""
Grip Tuner
==========
Adjust where the grip spot sits and how big the pickup circle is, while
WATCHING it on the live camera image next to the robot.

    Gripper Fwd Offset  - how far in front of the robot's center the grip
                          spot (magenta dot) sits, in pixels
    Pickup Radius       - size of the magenta pickup circle, in pixels
    Grip Angle Offset   - rotates the direction the offset is measured in,
                          away from the robot's raw ArUco heading. Use this
                          if your top-view camera is mounted at a slight
                          slant, so "straight ahead" in the image doesn't
                          line up with where the gripper actually is -
                          the magenta dot will trace a slanted line instead
                          of a straight one as you turn the robot.

Line up the magenta dot with where the gripper actually closes (put a
gem in the claw and check, ideally with the robot turned to a couple of
different headings so the slant is corrected everywhere, not just at one
angle), then press 's' to save.

Controls:
    o        - OPEN the claw   (sends RELEASE to the robot)
    c        - CLOSE the claw  (sends GRAB to the robot)
    s        - save the values to robot_overrides.json
               (main.py picks them up next time it starts)
    q / ESC  - quit

Use 'o' and 'c' to see where the magenta circle sits on the claw when it is
open and when it is closed. The screen tells you whether each command really
reached the robot.

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

OFFSET_MAX = 200
RADIUS_MIN, RADIUS_MAX = 5, 100
ANGLE_MAX_DEG = 45   # slider covers -45 to +45 degrees of slant


def _nothing(_value):
    pass


def save_values(offset, radius, angle_deg):
    """Merge into robot_overrides.json so other saved settings are kept."""

    existing = {}
    if os.path.exists(ROBOT_OVERRIDE_PATH):
        try:
            with open(ROBOT_OVERRIDE_PATH, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing["GRIPPER_FORWARD_OFFSET"] = int(offset)
    existing["PICKUP_RADIUS"] = int(radius)
    existing["GRIPPER_ANGLE_OFFSET_DEG"] = int(angle_deg)

    with open(ROBOT_OVERRIDE_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[grip_tuner] Saved offset={offset}, radius={radius}, angle={angle_deg} to {ROBOT_OVERRIDE_PATH}")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    detector = robot_tracker.make_detector()

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar(
        "Gripper Fwd Offset", WINDOW,
        min(int(config.GRIPPER_FORWARD_OFFSET), OFFSET_MAX), OFFSET_MAX, _nothing
    )
    cv2.createTrackbar(
        "Pickup Radius", WINDOW,
        min(max(int(config.PICKUP_RADIUS), RADIUS_MIN), RADIUS_MAX), RADIUS_MAX, _nothing
    )
    current_angle = int(getattr(config, "GRIPPER_ANGLE_OFFSET_DEG", 0))
    current_angle = min(max(current_angle, -ANGLE_MAX_DEG), ANGLE_MAX_DEG)
    cv2.createTrackbar(
        "Grip Angle Offset", WINDOW,
        current_angle + ANGLE_MAX_DEG, ANGLE_MAX_DEG * 2, _nothing
    )

    print(__doc__)

    status = ""
    status_frames_left = 0
    claw_text, claw_color = "claw: unknown (press o / c)", (200, 200, 200)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera")
            break

        offset = cv2.getTrackbarPos("Gripper Fwd Offset", WINDOW)
        radius = max(RADIUS_MIN, cv2.getTrackbarPos("Pickup Radius", WINDOW))
        angle_deg = cv2.getTrackbarPos("Grip Angle Offset", WINDOW) - ANGLE_MAX_DEG

        # Push the slider values into config so the SAME functions main.py
        # uses (robot_tracker + drawing) pick them up live.
        config.GRIPPER_FORWARD_OFFSET = offset
        config.PICKUP_RADIUS = radius
        config.GRIPPER_ANGLE_OFFSET_DEG = angle_deg

        result = frame.copy()

        robot_info = robot_tracker.detect_robot(frame, detector)
        drawing.draw_robot_heading(result, robot_info)

        gripper_center = robot_tracker.gripper_center_of(robot_info)
        drawing.draw_pickup_circle(result, gripper_center)

        cv2.putText(result, f"Gripper Fwd Offset: {offset}   Pickup Radius: {radius}   Grip Angle Offset: {angle_deg} deg",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if robot_info["center"] is None or gripper_center is None:
            cv2.putText(result, "Robot not detected - put it in view",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            rc = robot_info["center"]
            dist = math.hypot(gripper_center[0] - rc[0], gripper_center[1] - rc[1])
            cv2.line(result, (int(rc[0]), int(rc[1])),
                     (int(gripper_center[0]), int(gripper_center[1])), (255, 0, 255), 1)
            cv2.putText(result, f"grip spot is {dist:.0f}px from robot center",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.putText(result, claw_text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, claw_color, 2)

        if status_frames_left > 0:
            cv2.putText(result, status, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            status_frames_left -= 1

        cv2.putText(result, "o = open claw   c = close claw   s = save   q = quit", (10, result.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

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
            save_values(offset, radius, angle_deg)
            status = "Saved! Restart main.py to apply."
            status_frames_left = 90

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()