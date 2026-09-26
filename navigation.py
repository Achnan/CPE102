#=======NAVI

import numpy as np

import config


def _angle_diff(angle_to_target, heading_deg):
    diff = angle_to_target - heading_deg
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def compute_navigation(robot_center, robot_heading_deg, gripper_center,
                        held_gem_color, field_gems, target_circles):
    """
    Decide what the robot should do this frame.

    - If not holding a gem: aim at the nearest loose gem. STOP + ready_to_grab
      once that gem is inside the pickup circle.
    - If holding a gem: aim at the closest target circle matching
      held_gem_color. STOP + ready_to_place once inside the pickup circle.

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
        # --- find nearest loose gem ---
        nearest_gem = None
        nearest_distance = None

        for gem in field_gems:
            dx = gem["center_x"] - robot_center[0]
            dy = gem["center_y"] - robot_center[1]
            distance = np.sqrt(dx * dx + dy * dy)

            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_gem = gem

        if nearest_gem is not None:
            nav["nav_target_point"] = (nearest_gem["center_x"], nearest_gem["center_y"])
            nav["nav_target_label"] = f"gem:{nearest_gem['color']}"

    else:
        # --- find closest target circle of the held color ---
        matching = [t for t in target_circles if t["color"] == held_gem_color]

        if matching:
            target = min(
                matching,
                key=lambda t: (t["center_x"] - robot_center[0]) ** 2
                + (t["center_y"] - robot_center[1]) ** 2
            )
            nav["nav_target_point"] = (target["center_x"], target["center_y"])
            nav["nav_target_label"] = f"target {target['target_index']}:{target['color']}"

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