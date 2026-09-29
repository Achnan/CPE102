import time

import numpy as np

import config

# ---- anti-oscillation steering (turn overshoot fix) ----------------------
# The robot's turn speed is fixed, and there is camera + Wi-Fi delay between
# "decide to stop turning" and "the wheels actually stop", so a plain
# "turn until the angle is small" overshoots, flips to the other direction,
# overshoots again... The three settings below damp that out.

# True  = measure the turn angle from the grip spot (what you asked for).
# False = measure it from the robot center (the point it actually spins
# around). Both agree once the robot is lined up, but the grip-spot angle
# changes faster while turning, so if it still wobbles, try False.
STEER_FROM_GRIPPER = True

# Turning is done in short pulses (turn a little, STOP, let the camera catch
# up, re-measure). The pulse gets longer the bigger the angle error is:
TURN_PULSE_PERIOD_SEC = 0.30   # one "turn + pause" cycle
MIN_TURN_DUTY = 0.30           # smallest fraction of the cycle spent turning
FULL_TURN_ANGLE = 50.0         # at this error (deg) or more: turn continuously

# Once it is driving FORWARD, only start turning again if the error grows
# past TURN_ANGLE_THRESHOLD * this factor (stops FORWARD <-> turn flicker).
FORWARD_RELEASE_FACTOR = 1.5

# ---- pre-open the gripper on approach --------------------------------------
# Only matters while NOT holding a gem (target = a gem to pick up). Once the
# gripper gets within PRE_OPEN_DISTANCE_FACTOR x PICKUP_DISTANCE of the gem,
# nav["ready_to_open"] turns True so main.py can send RELEASE (open the claw)
# BEFORE the robot is actually close enough to grab - so the claw is already
# open and ready by the time it reaches the gem, instead of arriving with a
# closed claw and pushing the gem away.
#
# Must be > 1.0 (a pre-open zone smaller than the pickup zone would never
# trigger before ready_to_grab does). If you'd rather set an exact pixel
# distance instead of a multiple of PICKUP_DISTANCE, add
# GRIPPER_PRE_OPEN_DISTANCE to config.py - it's read here if present and
# overrides the factor.
PRE_OPEN_DISTANCE_FACTOR = 1.4   # lowered from 2.5 - that made ready_to_open
                                 # true for most of the approach, which felt
                                 # like the claw was "always open". Now it only
                                 # opens once genuinely close.

# ---- stuck-in-a-turning-loop fix --------------------------------------------
# Problem this fixes: when a gem is right at the edge of the gripper/claw,
# spinning to line up with it can physically nudge or drag the gem along
# with the robot's own frame, so the angle to it never actually closes -
# the robot ends up spinning forever, chasing something that moves with it.
#
# Fix: track how long the robot has been continuously TURNING toward the
# same locked target without making real progress (gripper_distance getting
# meaningfully smaller). If it's been turning for STUCK_TURN_TIMEOUT_SEC
# without at least STUCK_PROGRESS_PX of improvement:
#   - If it's already fairly close (within STUCK_GRAB_DISTANCE_FACTOR x
#     PICKUP_DISTANCE), stop trying to perfectly align and just force the
#     grab/place right now - the claw is usually wide enough to catch
#     something that close even if the angle isn't perfect, and forcing it
#     beats spinning forever.
#   - Otherwise, give up on this specific gem/target for a while (it goes on
#     a short blacklist so it isn't immediately re-chosen as "nearest" the
#     very next frame) and let the robot pick a different one.
STUCK_TURN_TIMEOUT_SEC = 3.0        # how long to tolerate turning with no progress
STUCK_PROGRESS_PX = 15              # must close by at least this much to not be "stuck"
STUCK_GRAB_DISTANCE_FACTOR = 1.6    # within this x PICKUP_DISTANCE -> force the grab/place
STUCK_BLACKLIST_SEC = 4.0           # how long an abandoned target is skipped for

# ---- final-approach lock (fix for the gem getting shoved behind the claw) -
# Problem this fixes: small heading corrections right up until the last
# moment can nudge a gem sideways or behind the claw instead of into it,
# especially once it's already close enough to be touching the gripper -
# and once that happens the robot just keeps re-aiming at wherever the
# (now-shoved) gem ended up, pushing it further every frame.
#
# Fix: once the robot is ALREADY lined up (angle within TURN_ANGLE_THRESHOLD)
# and gets within FINAL_APPROACH_FACTOR x PICKUP_DISTANCE of the target, the
# heading is locked - no more re-aiming for the rest of this approach. It
# just drives straight in, ignoring any apparent angle drift from here on
# (which close-up is more likely to be the gem being pushed than the robot
# actually being misaligned). The lock releases only when the target is
# reached or a different target is picked (see the lock_key check below).
FINAL_APPROACH_FACTOR = 2.0

_final_lock = {"lock_key": None, "locked": False}


def _reset_final_lock():
    _final_lock["lock_key"] = None
    _final_lock["locked"] = False


def force_replan():
    """
    Drop the current target lock and all approach state, so the next frame
    picks a fresh nearest target instead of continuing to chase whatever
    was locked. Call this from main.py after a failed grab (the gem may
    have been shoved out of the gripper's expected position, so continuing
    to chase its last known spot is unlikely to help).
    """
    _clear_lock()
    _reset_stuck_tracking()

_steer = {"forwarding": False}

# turning_since: monotonic time the current stuck-tracked turn started, or
# None if not currently accumulating (reset whenever real progress is made
# or the locked target changes).
# best_distance: closest gripper_distance seen since tracking started for
# this lock - used to detect "not actually getting closer".
_stuck = {"lock_key": None, "turning_since": None, "best_distance": None}

# Single-slot blacklist: the one most recently abandoned stuck target, so it
# isn't immediately re-picked as "nearest" right after giving up on it.
_blacklist = {"color": None, "x": None, "y": None, "until": 0.0}


def _reset_stuck_tracking():
    _stuck["lock_key"] = None
    _stuck["turning_since"] = None
    _stuck["best_distance"] = None
    _reset_final_lock()


def _is_blacklisted(item):
    if time.monotonic() >= _blacklist["until"]:
        return False
    if item["color"] != _blacklist["color"]:
        return False
    dx = item["center_x"] - _blacklist["x"]
    dy = item["center_y"] - _blacklist["y"]
    return (dx * dx + dy * dy) ** 0.5 <= config.TARGET_LOCK_MAX_DRIFT_PX


def _angle_diff(angle_to_target, heading_deg):
    diff = angle_to_target - heading_deg

    while diff > 180:
        diff -= 360

    while diff < -180:
        diff += 360

    return diff


# ---- target lock state ----------------------------------------------------
# Persists across calls so the robot commits to one physical gem/target
# instead of re-evaluating "which one is nearest" every single frame.
_locked = {
    "kind": None,
    "color": None,
    "x": None,
    "y": None
}


def _clear_lock():
    _locked["kind"] = None
    _locked["color"] = None
    _locked["x"] = None
    _locked["y"] = None


def _set_lock(kind, item):
    _locked["kind"] = kind
    _locked["color"] = item["color"]
    _locked["x"] = item["center_x"]
    _locked["y"] = item["center_y"]


def _find_locked_match(items, color, x, y):
    """
    Find whichever item in `items` is the same color and within
    TARGET_LOCK_MAX_DRIFT_PX of the locked position.
    """

    best = None
    best_dist = None

    for item in items:

        if item["color"] != color:
            continue

        dx = item["center_x"] - x
        dy = item["center_y"] - y

        distance = (dx * dx + dy * dy) ** 0.5

        if distance <= config.TARGET_LOCK_MAX_DRIFT_PX:

            if best_dist is None or distance < best_dist:
                best_dist = distance
                best = item

    return best


def _nearest(items, origin):
    """
    Find the item closest to the origin.

    For gem selection, origin is the GRIP SPOT.
    """

    nearest_item = None
    nearest_distance = None

    for item in items:

        dx = item["center_x"] - origin[0]
        dy = item["center_y"] - origin[1]

        distance = np.sqrt(dx * dx + dy * dy)

        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_item = item

    return nearest_item


def compute_navigation(
    robot_center,
    robot_heading_deg,
    gripper_center,
    held_gem_color,
    field_gems,
    target_circles
):
    """
    Decide what the robot should do this frame.

    NOT HOLDING A GEM:
        - Choose a GEM from field_gems.
        - Move toward that gem.
        - Once close enough (PRE_OPEN_DISTANCE_FACTOR x PICKUP_DISTANCE),
          set ready_to_open = True so main.py opens the claw in advance.
        - STOP when the gem enters the pickup circle.
        - Set ready_to_grab = True.

    HOLDING A GEM:
        - Use held_gem_color.
        - Find the TARGET CIRCLE with the same color.
        - Move toward that target circle.
        - STOP when the target enters the pickup circle.
        - Set ready_to_place = True.

    STUCK-IN-A-TURN protection: if turning toward the locked target for too
    long without making real progress (see STUCK_* constants above), either
    forces the grab/place (if already close) or abandons that target for a
    few seconds and lets a different one be chosen - see _stuck / _blacklist.

    The robot therefore follows:

        GEM
         |
        (claw opens early on approach)
         |
        GRAB
         |
        TARGET CIRCLE
         |
        RELEASE
    """

    nav = {
        "nav_command": "STOP",
        "nav_target_point": None,
        "nav_target_label": None,
        "angle_diff": None,
        "gripper_distance": None,
        "ready_to_grab": False,
        "ready_to_place": False,
        "ready_to_open": False,
    }

    # ------------------------------------------------------------------
    # Robot or gripper not detected
    # ------------------------------------------------------------------

    if robot_center is None or gripper_center is None:
        _steer["forwarding"] = False
        _reset_stuck_tracking()
        return nav

    # ==================================================================
    # STEP 1: NOT HOLDING A GEM
    #
    # Choose from field_gems.
    #
    # field_gems are the detected loose gems (the square detection
    # boxes shown around the gems in your camera view).
    # ==================================================================

    if held_gem_color is None:

        # We are not holding anything.
        # Therefore we should NOT look for a target circle.
        #
        # If a target lock remains from a previous placement,
        # remove it.
        if _locked["kind"] == "target":
            _clear_lock()

        chosen = None

        # --------------------------------------------------------------
        # Continue following the same gem if possible
        # --------------------------------------------------------------

        if _locked["kind"] == "gem":

            chosen = _find_locked_match(
                field_gems,
                _locked["color"],
                _locked["x"],
                _locked["y"]
            )

        # --------------------------------------------------------------
        # If we found the locked gem, update its position
        # --------------------------------------------------------------

        if chosen is not None:

            _set_lock("gem", chosen)

        else:

            # The previous gem disappeared or there is no lock yet.
            #
            # Choose a new nearest GEM (skipping anything on the
            # stuck-target blacklist - see STUCK_* above).
            #
            # IMPORTANT:
            # This searches field_gems, NOT target_circles.
            _clear_lock()

            candidates = [g for g in field_gems if not _is_blacklisted(g)]

            chosen = _nearest(
                candidates,
                gripper_center
            )

            if chosen is not None:
                _set_lock("gem", chosen)

        # --------------------------------------------------------------
        # Aim at the selected GEM
        # --------------------------------------------------------------

        if chosen is not None:

            nav["nav_target_point"] = (
                chosen["center_x"],
                chosen["center_y"]
            )

            nav["nav_target_label"] = (
                f"gem:{chosen['color']}"
            )

    # ==================================================================
    # STEP 2: HOLDING A GEM
    #
    # Now find the destination circle with the same color.
    # ==================================================================

    else:

        # We are holding a gem.
        #
        # Therefore we should no longer follow the old gem.
        if _locked["kind"] == "gem":
            _clear_lock()

        chosen = None

        # --------------------------------------------------------------
        # Continue following the same target circle if possible
        # --------------------------------------------------------------

        if (
            _locked["kind"] == "target"
            and _locked["color"] == held_gem_color
        ):

            chosen = _find_locked_match(
                target_circles,
                held_gem_color,
                _locked["x"],
                _locked["y"]
            )

        # --------------------------------------------------------------
        # Update existing target lock
        # --------------------------------------------------------------

        if chosen is not None:

            _set_lock("target", chosen)

        else:

            # No valid target lock.
            #
            # Find circles that have the same color as the held gem
            # (skipping anything on the stuck-target blacklist).
            _clear_lock()

            matching = [
                t for t in target_circles
                if t["color"] == held_gem_color and not _is_blacklisted(t)
            ]

            if matching:

                chosen = min(
                    matching,
                    key=lambda t:
                        (t["center_x"] - gripper_center[0]) ** 2
                        + (t["center_y"] - gripper_center[1]) ** 2
                )

                _set_lock("target", chosen)

        # --------------------------------------------------------------
        # Aim at the selected TARGET CIRCLE
        # --------------------------------------------------------------

        if chosen is not None:

            nav["nav_target_point"] = (
                chosen["center_x"],
                chosen["center_y"]
            )

            nav["nav_target_label"] = (
                f"target {chosen['target_index']}:{chosen['color']}"
            )

    # ==================================================================
    # No target found
    # ==================================================================

    if nav["nav_target_point"] is None:

        _steer["forwarding"] = False
        _reset_stuck_tracking()
        return nav

    tx, ty = nav["nav_target_point"]

    # ==================================================================
    # Calculate angle to target
    # ==================================================================

    aim_origin = (
        gripper_center
        if STEER_FROM_GRIPPER
        else robot_center
    )

    angle_to_target = np.degrees(
        np.arctan2(
            ty - aim_origin[1],
            tx - aim_origin[0]
        )
    )

    angle_diff = _angle_diff(
        angle_to_target,
        robot_heading_deg
    )

    nav["angle_diff"] = angle_diff

    # ==================================================================
    # Calculate distance from GRIPPER to target
    # ==================================================================

    gdx = tx - gripper_center[0]
    gdy = ty - gripper_center[1]

    gripper_distance = np.sqrt(
        gdx * gdx + gdy * gdy
    )

    nav["gripper_distance"] = gripper_distance

    # ==================================================================
    # Stuck-tracking bookkeeping - do this before the pickup-zone check so
    # reaching the zone (a real success) always clears the tracker below.
    # ==================================================================

    lock_key = (_locked["kind"], _locked["color"])
    if _stuck["lock_key"] != lock_key:
        # A different physical target than last frame - start fresh.
        _stuck["lock_key"] = lock_key
        _stuck["turning_since"] = None
        _stuck["best_distance"] = gripper_distance
    elif _stuck["best_distance"] is None or gripper_distance < _stuck["best_distance"] - STUCK_PROGRESS_PX:
        # Real progress was made since tracking started - forget any
        # accumulated stuck time and reset the "best" baseline.
        _stuck["best_distance"] = gripper_distance
        _stuck["turning_since"] = None

    if _final_lock["lock_key"] != lock_key:
        _final_lock["lock_key"] = lock_key
        _final_lock["locked"] = False

    if (
        not _final_lock["locked"]
        and gripper_distance <= config.PICKUP_DISTANCE * FINAL_APPROACH_FACTOR
        and abs(angle_diff) <= config.TURN_ANGLE_THRESHOLD
    ):
        # Already lined up and close enough - lock the heading now, before
        # any contact with the gem has a chance to nudge it and throw off
        # a fresh angle reading.
        _final_lock["locked"] = True

    # ==================================================================
    # Check whether target is inside pickup/gripper circle
    # ==================================================================

    in_pickup_zone = (
        gripper_distance <= config.PICKUP_DISTANCE
    )

    # If we are NOT holding a gem:
    # target = gem
    # therefore entering the circle means READY TO GRAB.
    if held_gem_color is None:

        nav["ready_to_grab"] = in_pickup_zone

        # ---- pre-open check (approach zone, bigger than the pickup zone) ----
        pre_open_distance = getattr(
            config, "GRIPPER_PRE_OPEN_DISTANCE",
            config.PICKUP_DISTANCE * PRE_OPEN_DISTANCE_FACTOR
        )
        nav["ready_to_open"] = gripper_distance <= pre_open_distance

    # If we ARE holding a gem:
    # target = destination circle
    # therefore entering the circle means READY TO PLACE.
    else:

        nav["ready_to_place"] = in_pickup_zone

    # ==================================================================
    # Target reached
    # ==================================================================

    if in_pickup_zone:

        _steer["forwarding"] = False
        _reset_stuck_tracking()

        nav["nav_command"] = "STOP"

        return nav

    # ==================================================================
    # Final-approach lock - once engaged (see above), skip re-aiming
    # entirely and just keep driving straight in.
    # ==================================================================

    if _final_lock["locked"]:
        _steer["forwarding"] = True
        _stuck["turning_since"] = None
        nav["nav_command"] = "FORWARD"
        return nav

    # ==================================================================
    # Hysteresis
    # ==================================================================

    threshold = config.TURN_ANGLE_THRESHOLD

    if _steer["forwarding"]:
        threshold *= FORWARD_RELEASE_FACTOR

    # ==================================================================
    # Move forward if angle is small enough
    # ==================================================================

    if abs(angle_diff) <= threshold:

        _steer["forwarding"] = True
        _stuck["turning_since"] = None   # making forward progress, not stuck-turning

        nav["nav_command"] = "FORWARD"

        return nav

    # ==================================================================
    # Otherwise turn - but first check we're not stuck turning in place
    # ==================================================================

    _steer["forwarding"] = False

    now = time.monotonic()
    if _stuck["turning_since"] is None:
        _stuck["turning_since"] = now

    turning_elapsed = now - _stuck["turning_since"]

    if turning_elapsed >= STUCK_TURN_TIMEOUT_SEC:

        if gripper_distance <= config.PICKUP_DISTANCE * STUCK_GRAB_DISTANCE_FACTOR:
            # Close enough already - stop fighting the angle (likely being
            # dragged along by the claw) and just take it now.
            if held_gem_color is None:
                nav["ready_to_grab"] = True
            else:
                nav["ready_to_place"] = True

            nav["nav_command"] = "STOP"
            _reset_stuck_tracking()
            return nav

        # Too far away to force it - give up on this one for a while so it
        # isn't immediately re-chosen as "nearest" next frame.
        _blacklist["color"] = _locked["color"]
        _blacklist["x"] = _locked["x"]
        _blacklist["y"] = _locked["y"]
        _blacklist["until"] = now + STUCK_BLACKLIST_SEC

        _clear_lock()
        _reset_stuck_tracking()

        nav["nav_command"] = "STOP"
        nav["nav_target_point"] = None
        nav["nav_target_label"] = None
        return nav

    direction = (
        "RIGHT"
        if angle_diff > 0
        else "LEFT"
    )

    # ==================================================================
    # Pulsed turning
    # ==================================================================

    duty = min(
        1.0,
        max(
            MIN_TURN_DUTY,
            abs(angle_diff) / FULL_TURN_ANGLE
        )
    )

    if duty >= 0.99:

        nav["nav_command"] = direction

    else:

        phase = (
            time.monotonic()
            % TURN_PULSE_PERIOD_SEC
        )

        nav["nav_command"] = (
            direction
            if phase < duty * TURN_PULSE_PERIOD_SEC
            else "STOP"
        )

    return nav