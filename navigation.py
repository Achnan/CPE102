import numpy as np

import config


def _angle_diff(angle_to_target, heading_deg):
    diff = angle_to_target - heading_deg
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


# ---- target lock state ----
# Persists across calls (this module is used by exactly one robot/loop
# at a time) so the robot commits to one physical gem/target instead of
# re-evaluating "which one is nearest" fresh every single frame - see
# the big comment in compute_navigation() for why that matters.
_locked = {"kind": None, "color": None, "x": None, "y": None}


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
    TARGET_LOCK_MAX_DRIFT_PX of the locked position - i.e. "probably
    still the same physical object", not just "happens to be closest
    to the robot right now".
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


def _nearest(items, robot_center):
    nearest_item = None
    nearest_distance = None

    for item in items:
        dx = item["center_x"] - robot_center[0]
        dy = item["center_y"] - robot_center[1]
        distance = np.sqrt(dx * dx + dy * dy)

        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_item = item

    return nearest_item


def compute_navigation(robot_center, robot_heading_deg, gripper_center,
                        held_gem_color, field_gems, target_circles):
    """
    Decide what the robot should do this frame.

    - If not holding a gem: aim at a gem. STOP + ready_to_grab once
      that gem is inside the pickup circle.
    - If holding a gem: aim at the target circle matching
      held_gem_color. STOP + ready_to_place once inside the pickup circle.

    TARGET LOCKING: rather than picking "whichever is nearest" fresh
    every frame (which can flip back and forth between two similarly-
    distant candidates as the robot moves, making it constantly re-aim
    instead of finishing the approach), this commits to one specific
    gem/target and keeps chasing THAT one - matched by color and being
    within TARGET_LOCK_MAX_DRIFT_PX of where it was last seen - until
    it's reached (picked up/placed) or it can no longer be found nearby
    (e.g. a false-positive that vanished). Only then is a new nearest
    one chosen.

    Returns a dict with everything the drawing/UI layer needs:
        nav_command, nav_target_point, nav_target_label,
        angle_diff, gripper_distance, ready_to_grab, ready_to_place
    """

    nav = {
        "nav_command": "STOP",
        "nav_target_point": None,
        "nav_target_label": None,
        "angle_diff": None,
        "gripper_distance": None,
        "ready_to_grab": False,
        "ready_to_place": False,
    }

    if robot_center is None or gripper_center is None:
        return nav

    if held_gem_color is None:
        # Not holding anything - we want a GEM lock. Drop any leftover
        # target-circle lock from a previous placement cycle.
        if _locked["kind"] == "target":
            _clear_lock()

        chosen = None
        if _locked["kind"] == "gem":
            chosen = _find_locked_match(field_gems, _locked["color"], _locked["x"], _locked["y"])

        if chosen is not None:
            _set_lock("gem", chosen)   # refresh the lock to its current position
        else:
            # No valid lock yet, or the locked gem is gone - pick a
            # fresh nearest one and start a new lock.
            _clear_lock()
            chosen = _nearest(field_gems, robot_center)
            if chosen is not None:
                _set_lock("gem", chosen)

        if chosen is not None:
            nav["nav_target_point"] = (chosen["center_x"], chosen["center_y"])
            nav["nav_target_label"] = f"gem:{chosen['color']}"

    else:
        # Holding a gem - we want a TARGET lock. Drop any leftover gem lock.
        if _locked["kind"] == "gem":
            _clear_lock()

        chosen = None
        if _locked["kind"] == "target" and _locked["color"] == held_gem_color:
            chosen = _find_locked_match(target_circles, held_gem_color, _locked["x"], _locked["y"])

        if chosen is not None:
            _set_lock("target", chosen)
        else:
            _clear_lock()
            matching = [t for t in target_circles if t["color"] == held_gem_color]
            if matching:
                chosen = min(
                    matching,
                    key=lambda t: (t["center_x"] - robot_center[0]) ** 2
                    + (t["center_y"] - robot_center[1]) ** 2
                )
                _set_lock("target", chosen)

        if chosen is not None:
            nav["nav_target_point"] = (chosen["center_x"], chosen["center_y"])
            nav["nav_target_label"] = f"target {chosen['target_index']}:{chosen['color']}"

    if nav["nav_target_point"] is None:
        return nav

    tx, ty = nav["nav_target_point"]

    angle_to_target = np.degrees(
        np.arctan2(ty - robot_center[1], tx - robot_center[0])
    )
    angle_diff = _angle_diff(angle_to_target, robot_heading_deg)
    nav["angle_diff"] = angle_diff

    gdx = tx - gripper_center[0]
    gdy = ty - gripper_center[1]
    gripper_distance = np.sqrt(gdx * gdx + gdy * gdy)
    nav["gripper_distance"] = gripper_distance

    in_pickup_zone = gripper_distance <= config.PICKUP_DISTANCE

    if held_gem_color is None:
        nav["ready_to_grab"] = in_pickup_zone
    else:
        nav["ready_to_place"] = in_pickup_zone

    if in_pickup_zone:
        nav["nav_command"] = "STOP"
    elif angle_diff > config.TURN_ANGLE_THRESHOLD:
        nav["nav_command"] = "RIGHT"
    elif angle_diff < -config.TURN_ANGLE_THRESHOLD:
        nav["nav_command"] = "LEFT"
    else:
        nav["nav_command"] = "FORWARD"

    return nav

    tx, ty = nav["nav_target_point"]

    angle_to_target = np.degrees(
        np.arctan2(ty - robot_center[1], tx - robot_center[0])
    )
    angle_diff = _angle_diff(angle_to_target, robot_heading_deg)
    nav["angle_diff"] = angle_diff

    gdx = tx - gripper_center[0]
    gdy = ty - gripper_center[1]
    gripper_distance = np.sqrt(gdx * gdx + gdy * gdy)
    nav["gripper_distance"] = gripper_distance

    in_pickup_zone = gripper_distance <= config.PICKUP_DISTANCE

    if held_gem_color is None:
        nav["ready_to_grab"] = in_pickup_zone
    else:
        nav["ready_to_place"] = in_pickup_zone

    if in_pickup_zone:
        nav["nav_command"] = "STOP"
    elif angle_diff > config.TURN_ANGLE_THRESHOLD:
        nav["nav_command"] = "RIGHT"
    elif angle_diff < -config.TURN_ANGLE_THRESHOLD:
        nav["nav_command"] = "LEFT"
    else:
        nav["nav_command"] = "FORWARD"

    return nav