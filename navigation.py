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
PRE_OPEN_DISTANCE_FACTOR = 2.5

_steer = {"forwarding": False}


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
            # Choose a new nearest GEM.
            #
            # IMPORTANT:
            # This searches field_gems, NOT target_circles.
            _clear_lock()

            chosen = _nearest(
                field_gems,
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
            # Find circles that have the same color as the held gem.
            _clear_lock()

            matching = [
                t for t in target_circles
                if t["color"] == held_gem_color
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

        nav["nav_command"] = "STOP"

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

        nav["nav_command"] = "FORWARD"

        return nav

    # ==================================================================
    # Otherwise turn
    # ==================================================================

    _steer["forwarding"] = False

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