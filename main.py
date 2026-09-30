import threading
import queue
import time

import cv2
import requests

import config
import robot_tracker
import vision
import navigation
import motion
import drawing
import esp32_link
from target_memory import TargetMemory   # makes the base circles sticky


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


# ---- manual two-speed drive -----------------------------------------------
# A different DRIVE_SPEED for "hunting for a gem" vs. "hauling one back to a
# base". The ESP32 stores ONE drive speed at a time, so this pushes the right
# one whenever the phase changes. Tune with the "Speed:Gem" / "Speed:Base"
# sliders in robot_config_tuner.py. motion.py measures how fast the robot
# really drives and sizes its moves to match, so these can be set to what
# the motors need to move reliably while carrying a gem.
DRIVE_SPEED_GEM = config.DRIVE_SPEED_GEM
DRIVE_SPEED_BASE = config.DRIVE_SPEED_BASE


def push_drive_speed(speed):
    """
    Tell the ESP32 to use this DRIVE_SPEED from now on. Runs the HTTP
    request in a short-lived background thread so a slow/flaky Wi-Fi link
    never stalls the camera loop - only called when the phase changes.
    """

    def _send():
        url = f"http://{config.ESP32_IP}:{config.ESP32_PORT}/config"
        try:
            requests.post(url, json={"drive_speed": int(speed)}, timeout=config.ESP32_REQUEST_TIMEOUT_SECONDS)
            print(f"[main] Drive speed -> {speed}")
        except requests.exceptions.RequestException as e:
            print(f"[main] Failed to push drive speed {speed}: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ---- grab / place ---------------------------------------------------------
# All movement timing (turn pulses, driving, stopping to look again, stuck
# handling) now lives in motion.py. main.py only decides WHEN to grab or
# release, and asks motion.py for the occasional back-up.
#
# After every GRAB, wait GRAB_VERIFY_DELAY_SEC for the claw to close and the
# camera to settle, then check whether something is really held. If not,
# back up GRAB_RETRY_BACKUP_SEC, re-open the claw on the next approach, and
# re-plan from scratch.
GRAB_VERIFY_DELAY_SEC = 0.6
GRAB_RETRY_BACKUP_SEC = 0.5

# After releasing a gem on its base: wait for the claw to open, then back
# away before turning, so the open claw doesn't sweep the gem off the base.
RELEASE_OPEN_WAIT_SEC = 0.4
POST_RELEASE_BACKUP_SEC = 0.4


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    aruco_detector = robot_tracker.make_detector()
    sender = CommandSender()
    mover = motion.MotionController()

    # Remembers each color base by position and locks it in place once
    # seen long enough, so the robot driving on top of a base (and
    # blocking the camera's view of it) doesn't make it disappear.
    # Press 't' in the video window to forget them and re-detect.
    target_memory = TargetMemory()

    current_phase_speed = DRIVE_SPEED_GEM
    push_drive_speed(current_phase_speed)

    # When to check whether the last GRAB actually picked something up.
    grab_verify_at = None

    # ---- held-color LOCK ------------------------------------------------
    # Once a GRAB is verified, that gem's colour is locked in and used until
    # it is released - it does NOT depend on the camera still seeing the gem
    # (the claw usually covers it). The camera reading is only used at the
    # moment of verifying a GRAB.
    locked_held_color = None
    was_holding = False
    grabbed_this_cycle = False
    placed_this_cycle = False
    opened_this_cycle = False

    # ---- held-color debounce -------------------------------------------
    # A raw colour reading at the grip spot must repeat HOLD_CONFIRM_FRAMES
    # frames in a row before it's trusted, and be absent HOLD_RELEASE_FRAMES
    # in a row before it's dropped.
    HOLD_CONFIRM_FRAMES = 5
    HOLD_RELEASE_FRAMES = 5
    confirmed_held_color = None
    _candidate_color = None
    _candidate_streak = 0
    _miss_streak = 0

    frame_count = 0

    while True:

        ret, image = cap.read()

        if not ret:
            print("Cannot read camera")
            break

        frame_count += 1
        should_log = (frame_count % LOG_EVERY_N_FRAMES == 0)
        now = time.monotonic()

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

        # Remembered bases (locked ones stay put even when hidden under the
        # robot). MUST come before assign_target_indices.
        detected_objects = target_memory.update(detected_objects)

        target_circles, _division_y = vision.assign_target_indices(detected_objects, height)

        # A gem sitting inside a base is not a target to pick up.
        field_gems, ignored_gems = vision.split_gems_by_targets(field_gems, target_circles)

        # ---- held gem color, sampled from the pickup circle ----
        raw_held_color = None
        if gripper_center is not None:
            raw_held_color = vision.sample_color_at(
                hsv_image,
                int(gripper_center[0]), int(gripper_center[1]),
                config.GEM_SAMPLE_RADIUS
            )

        if raw_held_color is not None:
            _miss_streak = 0
            if raw_held_color == _candidate_color:
                _candidate_streak += 1
            else:
                _candidate_color = raw_held_color
                _candidate_streak = 1
            if _candidate_streak >= HOLD_CONFIRM_FRAMES:
                confirmed_held_color = raw_held_color
        else:
            _candidate_color = None
            _candidate_streak = 0
            _miss_streak += 1
            if _miss_streak >= HOLD_RELEASE_FRAMES:
                confirmed_held_color = None

        held_gem_color = confirmed_held_color

        # ---- WHAT to go for (navigation) ----
        nav = navigation.compute_navigation(
            robot_info["center"], robot_info["heading_deg"], gripper_center,
            locked_held_color, field_gems, target_circles
        )

        # ---- HOW to move there (motion) ----
        pose = None
        if robot_info["center"] is not None and robot_info["heading_deg"] is not None:
            pose = (
                float(robot_info["center"][0]),
                float(robot_info["center"][1]),
                float(robot_info["heading_deg"]) + getattr(config, "GRIPPER_ANGLE_OFFSET_DEG", 0),
            )
        move = mover.update(now, pose, nav)

        if move["give_up"]:
            print(f"[main] {move['note']}")
            navigation.abandon_current_target()

        # Grab/place when motion says the robot has arrived (it may also
        # arrive when stuck but already close - see motion.py).
        nav["ready_to_grab"] = move["arrived"] and locked_held_color is None
        nav["ready_to_place"] = move["arrived"] and locked_held_color is not None
        nav["nav_command"] = move["command"]

        drawing.draw_navigation(
            result, robot_info["center"], gripper_center, locked_held_color, nav
        )

        # ---- grab / place / drive ----
        if nav["ready_to_grab"] and not grabbed_this_cycle:
            sender.set_command("STOP")
            sender.send_priority("GRAB")
            grabbed_this_cycle = True
            grab_verify_at = now + GRAB_VERIFY_DELAY_SEC

        elif nav["ready_to_place"] and not placed_this_cycle:
            sender.set_command("STOP")
            sender.send_priority("RELEASE")
            placed_this_cycle = True
            print(f"[main] RELEASE sent - releasing locked color {locked_held_color}")
            locked_held_color = None   # unlock: no longer holding anything
            mover.request_backup(now, POST_RELEASE_BACKUP_SEC, delay=RELEASE_OPEN_WAIT_SEC)

        else:
            # Open the claw in advance, once per approach.
            if nav["ready_to_open"] and not opened_this_cycle:
                sender.send_priority("RELEASE")
                opened_this_cycle = True
            sender.set_command(move["command"])

        # ---- did the GRAB we sent a moment ago actually pick something up? ----
        if grab_verify_at is not None and now >= grab_verify_at:
            grab_verify_at = None
            if held_gem_color is not None:
                locked_held_color = held_gem_color
                print(f"[main] GRAB verified and locked -> holding {locked_held_color}")
            else:
                print("[main] GRAB did not pick anything up - backing up and retrying")
                grabbed_this_cycle = False
                opened_this_cycle = False   # re-open the claw on the next approach
                navigation.force_replan()
                mover.reset_target()
                mover.request_backup(now, GRAB_RETRY_BACKUP_SEC)

        is_holding = locked_held_color is not None

        desired_phase_speed = DRIVE_SPEED_BASE if is_holding else DRIVE_SPEED_GEM
        if desired_phase_speed != current_phase_speed:
            push_drive_speed(desired_phase_speed)
            current_phase_speed = desired_phase_speed

        # Reset the one-shot guards when the held/not-held state flips.
        if is_holding and not was_holding:
            placed_this_cycle = False   # just picked up - ready to place next
        if not is_holding and was_holding:
            grabbed_this_cycle = False  # just placed/dropped - ready to grab next
            opened_this_cycle = False   # ready to pre-open for the next gem too
        was_holding = is_holding

        # ---- logging ----
        if should_log:
            print(f"robot holding -> {locked_held_color} (camera currently reads: {held_gem_color})")
            if robot_info["center"] is not None:
                print(
                    f"robot position -> "
                    f"({int(robot_info['center'][0])}, {int(robot_info['center'][1])}), "
                    f"heading -> {robot_info['heading_deg']:.1f} deg"
                )
            else:
                print("robot position -> not detected")
            angle = nav["angle_diff"]
            dist = nav["gripper_distance"]
            print(
                f"target -> {nav['nav_target_label']}"
                f" (angle {angle:+.0f} deg, {dist:.0f}px)" if angle is not None else
                f"target -> {nav['nav_target_label']}"
            )
            print(
                f"motion -> {move['state']}: {move['note']} | sent {move['command']} | "
                f"learned turn {mover.turn_rate:.0f} deg/s, drive {mover.drive_rate:.0f} px/s "
                f"at speed {current_phase_speed}"
            )

        # ---- draw field layout ----
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