import threading
import queue
import time

import cv2
import requests

import config
import robot_tracker
import vision
import navigation
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


# ---- manual two-speed drive (auto speed removed) -------------------------
# Back to plain manual speeds you set yourself - a different DRIVE_SPEED for
# "hunting for a gem" vs. "hauling one back to a base". The ESP32 only
# stores ONE drive speed at a time, so main.py pushes whichever of these
# matches the current phase, but ONLY when the phase actually changes (not
# continuously) - these are fixed numbers you tune, not an auto-computed
# ramp. Adjust them with the "Speed:Gem" / "Speed:Base" sliders in
# robot_config_tuner.py (saved to robot_overrides.json), or edit the
# defaults below directly.
DRIVE_SPEED_GEM = getattr(config, "DRIVE_SPEED_GEM", 200)     # searching for / approaching a gem
DRIVE_SPEED_BASE = getattr(config, "DRIVE_SPEED_BASE", 130)   # hauling a held gem back to its base


def push_drive_speed(speed):
    """
    Tell the ESP32 to use this DRIVE_SPEED from now on. Runs the HTTP
    request in a short-lived background thread so a slow/flaky Wi-Fi link
    never stalls the camera loop - only called when the phase changes, not
    every frame.
    """

    def _send():
        url = f"http://{config.ESP32_IP}:{config.ESP32_PORT}/config"
        try:
            requests.post(url, json={"drive_speed": int(speed)}, timeout=config.ESP32_REQUEST_TIMEOUT_SECONDS)
            print(f"[main] Drive speed -> {speed}")
        except requests.exceptions.RequestException as e:
            print(f"[main] Failed to push drive speed {speed}: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ---- stop before every movement action ------------------------------------
# Before the robot does ANY movement action (FORWARD/BACKWARD/LEFT/RIGHT),
# it stops for this long first - every time the command actually changes,
# with no exceptions (including coming from a real STOP). This trades a
# little speed for a lot of stability: every action starts from a dead
# stop instead of blending into whatever the robot was already doing.
ACTION_CHANGE_STOP_SEC = 0.2

# ---- grab-verify + retry (fix for a gem shoved behind the claw) ----------
# A GRAB can miss - the gem was pushed out of position (see
# navigation.py's FINAL_APPROACH_FACTOR comment for the usual cause) and
# the claw closes on nothing. Left alone, the robot would just keep re-
# aiming at wherever that shoved gem now appears to be, pushing it further
# every frame - the same loop the final-approach lock is meant to prevent,
# but this catches it even if a bad grab still slips through.
#
# Fix: after every GRAB, wait GRAB_VERIFY_DELAY_SEC for the claw to finish
# closing and the camera to settle, then check whether held_gem_color
# actually became non-None. If it's still empty, the grab failed - back
# the robot up for GRAB_RETRY_BACKUP_SEC (so whatever's stuck near the
# claw is freed) and drop the current target lock so the next attempt
# re-evaluates from scratch instead of chasing the same shoved gem.
GRAB_VERIFY_DELAY_SEC = 0.6
GRAB_RETRY_BACKUP_SEC = 0.5


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    aruco_detector = robot_tracker.make_detector()
    sender = CommandSender()

    # Remembers each color base by position and locks it in place once
    # seen long enough, so the robot driving on top of a base (and
    # blocking the camera's view of it) doesn't make it disappear.
    # Press 't' in the video window to forget them and re-detect.
    target_memory = TargetMemory()

    # Manual two-speed drive: start in "hunting for a gem" mode, since
    # nothing is held at startup.
    current_phase_speed = DRIVE_SPEED_GEM
    push_drive_speed(current_phase_speed)

    # Grab-verify + retry state (see GRAB_VERIFY_DELAY_SEC above).
    # grab_verify_at: when to check whether the last GRAB actually worked,
    # or None if no GRAB is currently pending verification.
    # backup_until: while time.monotonic() is before this, the robot is
    # forced to BACKWARD (overrides everything else) to shake loose
    # whatever's stuck near the claw after a failed grab.
    grab_verify_at = None
    backup_until = 0.0

    # Stop-before-every-action state (see ACTION_CHANGE_STOP_SEC above).
    # last_sent_command: the last command actually sent (any of
    # FORWARD/BACKWARD/LEFT/RIGHT/STOP), or None before anything is sent.
    last_sent_command = None
    action_pause_until = 0.0

    # Tracks whether the gripper was holding something last frame, so
    # GRAB/RELEASE only get sent once per pickup/placement — not on
    # every single frame the robot happens to sit inside the zone.
    was_holding = False

    # ---- held-color LOCK ------------------------------------------------
    # Once a GRAB is verified to have actually picked something up, that
    # gem's color is locked in and used for navigation/placement for as
    # long as the robot is holding it - it does NOT depend on the camera
    # still being able to see the gem's color while it's held (the claw
    # usually covers most of it, so the raw camera reading is unreliable
    # exactly while it matters most). The camera's held-color reading
    # (held_gem_color, debounced above) is now used ONLY at the moment of
    # verifying a GRAB - never while already holding something.
    # None = not holding anything; a color name = holding that color,
    # locked, regardless of what the camera currently sees.
    locked_held_color = None
    grabbed_this_cycle = False
    placed_this_cycle = False

    # Tracks whether the claw has already been pre-opened for the gem
    # currently being approached, so RELEASE is only sent once per
    # approach instead of every frame while inside the pre-open zone.
    opened_this_cycle = False

    # ---- held-color debounce -------------------------------------------
    # A single frame of sample_color_at() at the grip spot is NOT trusted
    # on its own - it can flicker (the robot's own frame/shadow, motion
    # blur, partial occlusion), and since ready_to_open / ready_to_grab /
    # ready_to_place / GRAB / RELEASE all depend on "are we holding
    # something right now", a flickering raw reading makes the claw seem
    # unsure whether it's holding or about to open. The raw reading must
    # repeat for HOLD_CONFIRM_FRAMES frames in a row before it's trusted as
    # "now holding X", and must be absent for HOLD_RELEASE_FRAMES frames in
    # a row before it's trusted as "no longer holding" - only
    # confirmed_held_color (not the raw reading) is ever passed to
    # navigation or used to decide GRAB/RELEASE.
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

        # Swap this frame's raw detections for the remembered set (locked
        # bases stay put even when hidden under the robot). This MUST come
        # before assign_target_indices, which numbers the bases left->right
        # - otherwise a hidden base would shift the numbering of the others.
        detected_objects = target_memory.update(detected_objects)

        target_circles, _division_y = vision.assign_target_indices(detected_objects, height)

        # A gem sitting inside a base is not a target to pick up (e.g. one
        # the robot already placed there).
        field_gems, ignored_gems = vision.split_gems_by_targets(field_gems, target_circles)

        # ---- held gem color, sampled from the pickup circle ----
        raw_held_color = None
        if gripper_center is not None:
            raw_held_color = vision.sample_color_at(
                hsv_image,
                int(gripper_center[0]), int(gripper_center[1]),
                config.GEM_SAMPLE_RADIUS
            )

        # Debounce the raw reading before trusting it (see the comment
        # where these variables are initialized, above the main loop).
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

        # ---- navigation ----
        # Uses locked_held_color (see above), NOT the raw held_gem_color -
        # once holding something, what's actually held doesn't change just
        # because the claw is currently blocking the camera's view of it.
        nav = navigation.compute_navigation(
            robot_info["center"], robot_info["heading_deg"], gripper_center,
            locked_held_color, field_gems, target_circles
        )

        drawing.draw_navigation(
            result, robot_info["center"], gripper_center, locked_held_color, nav
        )

        if should_log:
            print(
                f"nav command -> {nav['nav_command']} "
                f"(aiming at: {nav['nav_target_label']}, angle_diff: {nav['angle_diff']}, "
                f"ready_to_open: {nav['ready_to_open']}, "
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

        if time.monotonic() < backup_until:
            # A grab just failed - back away first, ignoring everything
            # else navigation might want to do this frame (see
            # GRAB_RETRY_BACKUP_SEC above).
            sender.set_command("BACKWARD")

        elif nav["ready_to_grab"] and not grabbed_this_cycle:
            sender.send_priority("GRAB")
            grabbed_this_cycle = True
            grab_verify_at = time.monotonic() + GRAB_VERIFY_DELAY_SEC

        elif nav["ready_to_place"] and not placed_this_cycle:
            sender.send_priority("RELEASE")
            placed_this_cycle = True
            print(f"[main] RELEASE sent - releasing locked color {locked_held_color}")
            locked_held_color = None   # unlock: no longer holding anything

        else:
            # Pre-open the claw once, on the way in, before the robot is
            # actually close enough to grab. This does NOT block driving:
            # RELEASE is sent as a one-shot priority command (a brief pause
            # on the ESP32 side while the servo moves), then the regular
            # movement command below keeps being sent every frame as usual.
            if nav["ready_to_open"] and not opened_this_cycle:
                sender.send_priority("RELEASE")
                opened_this_cycle = True

            # Decide the ACTUAL command to send this frame. Every time the
            # command actually CHANGES - to anything, including going from
            # a real STOP into a new movement - the robot is forced to
            # STOP first for ACTION_CHANGE_STOP_SEC, then the new command
            # takes over. This is deliberately blanket (every action gets
            # a settle pause), not just reversals.
            desired_command = nav["nav_command"]
            now = time.monotonic()

            if desired_command == "STOP":
                # Stopping itself never needs a pre-pause.
                last_sent_command = "STOP"
                effective_command = "STOP"

            elif now < action_pause_until:
                effective_command = "STOP"

            elif last_sent_command is not None and desired_command != last_sent_command:
                # The command just changed - pause first, then apply it.
                # Commit to the new command NOW (even though the output
                # this frame is still STOP) so that once the pause ends,
                # the final "else" branch below sends it normally instead
                # of re-detecting the same "change" forever.
                action_pause_until = now + ACTION_CHANGE_STOP_SEC
                last_sent_command = desired_command
                effective_command = "STOP"

            else:
                effective_command = desired_command
                last_sent_command = desired_command

            sender.set_command(effective_command)

        # Grab-verify: did the GRAB we sent a moment ago actually pick
        # something up? This is the ONLY moment the raw camera reading
        # (held_gem_color) is consulted - once it confirms a color, that
        # color gets LOCKED in and the camera is no longer needed for it.
        if grab_verify_at is not None and time.monotonic() >= grab_verify_at:
            grab_verify_at = None
            if held_gem_color is not None:
                locked_held_color = held_gem_color
                print(f"[main] GRAB verified and locked -> holding {locked_held_color}")
            else:
                print("[main] GRAB did not pick anything up - backing up and retrying")
                grabbed_this_cycle = False
                backup_until = time.monotonic() + GRAB_RETRY_BACKUP_SEC
                navigation.force_replan()

        is_holding = locked_held_color is not None

        # Manual two-speed drive: push the matching fixed speed whenever the
        # phase (holding a gem or not) actually changes - not continuously.
        desired_phase_speed = DRIVE_SPEED_BASE if is_holding else DRIVE_SPEED_GEM
        if desired_phase_speed != current_phase_speed:
            push_drive_speed(desired_phase_speed)
            current_phase_speed = desired_phase_speed

        # Reset the one-shot guards when the held/not-held state actually
        # flips, so the next pickup/placement can trigger GRAB/RELEASE again.
        if is_holding and not was_holding:
            placed_this_cycle = False   # just picked up - ready to place next

        if not is_holding and was_holding:
            grabbed_this_cycle = False  # just placed/dropped - ready to grab next
            opened_this_cycle = False   # ready to pre-open for the next gem too

        was_holding = is_holding

        # ---- draw field layout ----
        # (the white center division line has been removed - draw_division_line
        # no longer exists in drawing.py, so it is NOT called here)
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