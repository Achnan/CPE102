"""
navigation.py - decides WHAT the robot is going for, never HOW it moves.

    - which gem to chase (nearest to the grip spot, then locked so the
      robot commits to it) or, while holding a gem, which base to deliver to
    - geometry to that target: angle_diff (how far to turn) and
      gripper_distance (how far the grip spot still is from it)
    - ready_to_open: close enough that the claw should open in advance

HOW to get there (turn pulses, driving, settling, stuck detection) lives in
motion.py. Keeping the two apart is the point of this redesign: earlier
versions had navigation pulsing turns, main.py adding settle pauses on top,
and a stuck timer cutting turns off - three timers fighting over the same
wheels, which is what produced the turn/stop/turn loop.
"""

import time

import numpy as np

import config

# Measure the turn angle from the robot CENTER (the point it actually spins
# around) instead of from the grip spot. Once lined up, both give the same
# answer - the grip spot sits on the robot's heading line, so it still ends
# up on the target. But from the grip spot the angle becomes unstable when a
# target is very close to or behind the claw (it can flip to 180 degrees),
# which is what made the robot spin around a gem stuck beside the claw.
# Set True to go back to measuring from the grip spot.
STEER_FROM_GRIPPER = False

# The claw opens in advance once the grip spot is within this many times
# PICKUP_DISTANCE of the gem (or GRIPPER_PRE_OPEN_DISTANCE px, if that is
# defined in config.py).
PRE_OPEN_DISTANCE_FACTOR = 1.4

# How long a target motion.py gave up on is skipped before it may be
# chosen again.
ABANDON_BLACKLIST_SEC = 4.0


# ---- target lock --------------------------------------------------------
# The robot commits to one physical gem/base and keeps chasing THAT one
# (same colour, within TARGET_LOCK_MAX_DRIFT_PX of where it was last seen)
# instead of re-picking "whichever is nearest" every frame.
_locked = {"kind": None, "color": None, "x": None, "y": None}

# Increases every time a brand-new target is chosen. motion.py uses it to
# know when to throw away per-target progress (stuck counters etc.).
_lock_generation = 0

# The one most recently abandoned target, skipped until "until".
_blacklist = {"color": None, "x": None, "y": None, "until": 0.0}


def _clear_lock():
    _locked.update(kind=None, color=None, x=None, y=None)


def _new_lock(kind, item):
    global _lock_generation
    _lock_generation += 1
    _locked.update(kind=kind, color=item["color"], x=item["center_x"], y=item["center_y"])


def _refresh_lock(item):
    _locked.update(x=item["center_x"], y=item["center_y"])


def force_replan():
    """Forget the current target so the next frame picks a fresh one."""
    _clear_lock()


def abandon_current_target():
    """Skip the current target for ABANDON_BLACKLIST_SEC, then replan."""
    if _locked["kind"] is not None:
        _blacklist.update(
            color=_locked["color"], x=_locked["x"], y=_locked["y"],
            until=time.monotonic() + ABANDON_BLACKLIST_SEC,
        )
    _clear_lock()


def _is_blacklisted(item):
    if time.monotonic() >= _blacklist["until"] or item["color"] != _blacklist["color"]:
        return False
    dx = item["center_x"] - _blacklist["x"]
    dy = item["center_y"] - _blacklist["y"]
    return (dx * dx + dy * dy) ** 0.5 <= config.TARGET_LOCK_MAX_DRIFT_PX


def _find_locked_match(items, color, x, y):
    """Same colour and within TARGET_LOCK_MAX_DRIFT_PX of the locked spot."""
    best, best_dist = None, None
    for item in items:
        if item["color"] != color:
            continue
        d = ((item["center_x"] - x) ** 2 + (item["center_y"] - y) ** 2) ** 0.5
        if d <= config.TARGET_LOCK_MAX_DRIFT_PX and (best_dist is None or d < best_dist):
            best, best_dist = item, d
    return best


def _nearest(items, origin):
    best, best_dist = None, None
    for item in items:
        d = np.hypot(item["center_x"] - origin[0], item["center_y"] - origin[1])
        if best_dist is None or d < best_dist:
            best, best_dist = item, d
    return best


def _angle_diff(angle_to_target, heading_deg):
    diff = angle_to_target - heading_deg
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def _choose(kind, candidates, gripper_center):
    """Keep the locked target of this kind if still visible, else lock the nearest."""
    chosen = None
    if _locked["kind"] == kind:
        chosen = _find_locked_match(candidates, _locked["color"], _locked["x"], _locked["y"])
    if chosen is not None:
        _refresh_lock(chosen)
        return chosen

    _clear_lock()
    chosen = _nearest([c for c in candidates if not _is_blacklisted(c)], gripper_center)
    if chosen is not None:
        _new_lock(kind, chosen)
    return chosen


def compute_navigation(robot_center, robot_heading_deg, gripper_center,
                       held_gem_color, field_gems, target_circles):
    """
    Returns a dict:
        nav_target_point, nav_target_label, target_id,
        angle_diff        - degrees to turn; > 0 means turn RIGHT
        gripper_distance  - px from the grip spot to the target
        ready_to_grab / ready_to_place - grip spot inside PICKUP_DISTANCE
        ready_to_open     - inside the pre-open zone (only while not holding)
        nav_command       - placeholder "STOP"; main.py fills in the command
                            motion.py actually decided, for display/logging
    """

    nav = {
        "nav_command": "STOP",
        "nav_target_point": None,
        "nav_target_label": None,
        "target_id": None,
        "angle_diff": None,
        "gripper_distance": None,
        "center_distance": None,
        "ready_to_grab": False,
        "ready_to_place": False,
        "ready_to_open": False,
    }

    if robot_center is None or gripper_center is None or robot_heading_deg is None:
        return nav

    if held_gem_color is None:
        chosen = _choose("gem", field_gems, gripper_center)
        if chosen is not None:
            nav["nav_target_label"] = f"gem:{chosen['color']}"
    else:
        matching = [t for t in target_circles if t["color"] == held_gem_color]
        chosen = _choose("target", matching, gripper_center)
        if chosen is not None:
            nav["nav_target_label"] = f"target {chosen['target_index']}:{chosen['color']}"

    if chosen is None:
        return nav

    tx, ty = chosen["center_x"], chosen["center_y"]
    nav["nav_target_point"] = (tx, ty)
    nav["target_id"] = (_locked["kind"], _lock_generation)

    # The robot's real forward direction is the ArUco heading corrected by
    # the grip slant from grip_tuner.py (the same correction gripper_center
    # already uses), so steering and the grip spot always agree.
    heading = robot_heading_deg + getattr(config, "GRIPPER_ANGLE_OFFSET_DEG", 0)
    origin = gripper_center if STEER_FROM_GRIPPER else robot_center
    bearing = np.degrees(np.arctan2(ty - origin[1], tx - origin[0]))
    nav["angle_diff"] = float(_angle_diff(bearing, heading))

    distance = float(np.hypot(tx - gripper_center[0], ty - gripper_center[1]))
    nav["gripper_distance"] = distance
    # used by motion.py to work out how far SIDEWAYS the claw would miss
    nav["center_distance"] = float(np.hypot(tx - robot_center[0], ty - robot_center[1]))

    arrived = distance <= config.PICKUP_DISTANCE
    if held_gem_color is None:
        nav["ready_to_grab"] = arrived
        pre_open = getattr(config, "GRIPPER_PRE_OPEN_DISTANCE",
                           config.PICKUP_DISTANCE * PRE_OPEN_DISTANCE_FACTOR)
        nav["ready_to_open"] = distance <= pre_open
    else:
        nav["ready_to_place"] = arrived

    return nav