import threading
import queue
import time

import cv2

import config
import robot_tracker
import vision
import navigation
import drawing
import esp32_link
from target_memory import TargetMemory   # NEW: makes the base circles sticky


# How often prints. Printing every frame adds real per-frame cost for
# no benefit - the console can't be read that fast anyway.
LOG_EVERY_N_FRAMES = 5

# How often the background thread re-sends the current movement command
# while nothing has changed. Must stay comfortably under the ESP32's
# 1.5s auto-stop timeout.
RESEND_INTERVAL_SEC = 0.2


class CommandSender:
    """
    Owns all communication with esp32_link so the camera loop never has
    to wait on a network round-trip.

    - set_command(cmd)  : non-blocking, just updates "what should be
                            driving right now" - call this every frame.
    - send_priority(cmd): non-blocking, queues a one-shot command
                            (GRAB / RELEASE / STOP-on-exit) to go out
                            immediately, ahead of the next resend.

    A single background thread does the actual esp32_link.send_command()
    calls: priority commands first, otherwise it resends the latest
    movement command every RESEND_INTERVAL_SEC. The main loop is never
    blocked by this, however slow or flaky the Wi-Fi link is.
    """

    def __init__(self, resend_interval=RESEND_INTERVAL_SEC):
        self._lock = threading.Lock()
        self._current_command = "STOP"
        self._priority_queue = queue.Queue()
        self._resend_interval = resend_interval
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_command(self, command):
        with self._lock:
            self._current_command = command

    def send_priority(self, command):
        self._priority_queue.put(command)

    def _run(self):
        last_sent = 0.0
        last_command = None
        while self._running:
            try:
                command = self._priority_queue.get(timeout=0.02)
                esp32_link.send_command(command, force=True)
                last_sent = time.monotonic()
                continue
            except queue.Empty:
                pass

            now = time.monotonic()
            with self._lock:
                command = self._current_command

            if command != last_command:
                # The command just CHANGED (e.g. LEFT -> STOP): send it right
                # away instead of waiting for the next resend tick. Waiting
                # up to RESEND_INTERVAL_SEC made the robot keep turning
                # after the camera said stop, which caused overshoot.
                esp32_link.send_command(command, force=True)
                last_command = command
                last_sent = now
            elif now - last_sent >= self._resend_interval:
                esp32_link.send_command(command)
                last_sent = now

            time.sleep(0.01)

    def stop(self, final_command="STOP"):
        self._running = False
        self._thread.join(timeout=1.0)
        # send the final stop directly, blocking is fine here - we're exiting anyway
        esp32_link.send_command(final_command, force=True)


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    aruco_detector = robot_tracker.make_detector()
    sender = CommandSender()

    # NEW: remembers each color base by position and locks it in place
    # once seen long enough, so the robot driving on top of a base (and
    # blocking the camera's view of it) doesn't make it disappear.
    # Press 't' in the video window to forget them and re-detect.
    target_memory = TargetMemory()

    # Tracks whether the gripper was holding something last frame, so
    # GRAB/RELEASE only get sent once per pickup/placement — not on
    # every single frame the robot happens to sit inside the zone.
    was_holding = False
    grabbed_this_cycle = False
    placed_this_cycle = False

    frame_count = 0

    while True:

        ret, image = cap.read()

        if not ret:
            print("Cannot read camera")
            break

        frame_count += 1
        should_log = (frame_count % LOG_EVERY_N_FRAMES == 0)

        height, width = image.shape[:2]
        result = image.copy()

        # ---- robot ----
        robot_info = robot_tracker.detect_robot(image, aruco_detector)
        drawing.draw_robot_heading(result, robot_info)

        gripper_center = robot_tracker.gripper_center_of(robot_info)
        drawing.draw_pickup_circle(result, gripper_center)

        # ---- color detection ----
        hsv_image = vision.to_hsv(image)

        detected_objects, field_gems = vision.detect_target_circles_and_gems(
            image, hsv_image, robot_info["roi"]
        )

        # NEW: swap this frame's raw detections for the remembered set
        # (locked bases stay put even when hidden under the robot).
        # This MUST come before assign_target_indices, which numbers the
        # bases left->right - otherwise a hidden base would shift the
        # numbering of the others.
        detected_objects = target_memory.update(detected_objects)

        target_circles, _division_y = vision.assign_target_indices(detected_objects, height)

        # NEW: a gem sitting inside a base is not a target to pick up
        # (e.g. one the robot already placed there).
        field_gems, ignored_gems = vision.split_gems_by_targets(field_gems, target_circles)

        # ---- held gem color, sampled from the pickup circle ----
        held_gem_color = None
        if gripper_center is not None:
            held_gem_color = vision.sample_color_at(
                hsv_image,
                int(gripper_center[0]), int(gripper_center[1]),
                config.GEM_SAMPLE_RADIUS
            )

        if should_log:
            print(f"robot holding -> {held_gem_color}")

            if robot_info["center"] is not None:
                print(
                    f"robot position -> "
                    f"({int(robot_info['center'][0])}, {int(robot_info['center'][1])}), "
                    f"heading -> {robot_info['heading_deg']:.1f} deg"
                )
            else:
                print("robot position -> not detected")

        # ---- navigation ----
        nav = navigation.compute_navigation(
            robot_info["center"], robot_info["heading_deg"], gripper_center,
            held_gem_color, field_gems, target_circles
        )

        drawing.draw_navigation(
            result, robot_info["center"], gripper_center, held_gem_color, nav
        )

        if should_log:
            print(
                f"nav command -> {nav['nav_command']} "
                f"(aiming at: {nav['nav_target_label']}, angle_diff: {nav['angle_diff']}, "
                f"ready_to_grab: {nav['ready_to_grab']}, ready_to_place: {nav['ready_to_place']})"
            )

        # ========================================================
        # SEND COMMAND TO ESP32 OVER WI-FI
        #
        # These calls are all non-blocking now: they just hand the
        # command to CommandSender, which talks to the ESP32 from a
        # background thread. The camera loop never waits on the
        # network, so a slow or flaky Wi-Fi link no longer slows
        # down frame capture. GRAB/RELEASE still go out ahead of the
        # regular resend via send_priority(); the ESP32's 1.5s
        # auto-stop is satisfied by the background thread's own
        # resend timer, independent of frame rate.
        # ========================================================

        if nav["ready_to_grab"] and not grabbed_this_cycle:
            sender.send_priority("GRAB")
            grabbed_this_cycle = True

        elif nav["ready_to_place"] and not placed_this_cycle:
            sender.send_priority("RELEASE")
            placed_this_cycle = True

        else:
            sender.set_command(nav["nav_command"])

        # Reset the one-shot guards when the held/not-held state
        # actually flips, so the next pickup/placement can trigger
        # GRAB/RELEASE again.
        is_holding = held_gem_color is not None

        if is_holding and not was_holding:
            placed_this_cycle = False   # just picked up - ready to place next

        if not is_holding and was_holding:
            grabbed_this_cycle = False  # just placed/dropped - ready to grab next

        was_holding = is_holding

        # ---- draw field layout ----
        # (the white center division line is no longer drawn)
        drawing.draw_field_boundary(result, height, width)
        drawing.draw_target_circles(result, target_circles)
        drawing.draw_field_gems(result, field_gems)
        drawing.draw_ignored_gems(result, ignored_gems)

        if should_log:
            for obj in target_circles:
                side = vision.get_side(obj["target_index"])
                lock_note = "locked" if obj.get("locked") else "locking..."
                print(f"target {obj['target_index']} ({side}) -> {obj['color']} [{lock_note}]")
            print()

            for gem in field_gems:
                print(f"gem -> {gem['color']} at ({gem['center_x']}, {gem['center_y']})")
            if ignored_gems:
                print(f"ignored {len(ignored_gems)} gem(s) sitting on a base")
            print()

        # ---- display ----
        cv2.imshow("Overview Camera - Target Circles", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("t"):
            target_memory.reset()
            print("[main] Target memory cleared - re-detecting bases.")

    # Make sure the robot doesn't keep driving after the script exits.
    sender.stop(final_command="STOP")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()