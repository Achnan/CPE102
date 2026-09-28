import cv2

import config
import vision


def draw_robot_heading(result, robot_info):
    if robot_info["center"] is None or robot_info["forward_unit"] is None:
        return

    center = robot_info["center"]
    forward_vector = robot_info["forward_unit"] * config.GRIPPER_FORWARD_OFFSET / 2
    arrow_end = (center + forward_vector * 3).astype(int)

    cv2.arrowedLine(
        result, tuple(center.astype(int)), tuple(arrow_end),
        (0, 255, 255), 3, tipLength=0.3
    )

    if robot_info["marker_ids"] is not None:
        cv2.aruco.drawDetectedMarkers(
            result, robot_info["marker_corners"], robot_info["marker_ids"]
        )


def draw_pickup_circle(result, gripper_center):
    if gripper_center is None:
        return
    gx, gy = int(gripper_center[0]), int(gripper_center[1])
    cv2.circle(result, (gx, gy), config.PICKUP_RADIUS, (255, 0, 255), 2)
    cv2.circle(result, (gx, gy), 4, (255, 0, 255), -1)


def draw_field_boundary(result, height, width):
    """Draw the field area so it's visible where detection is
    restricted to - anything outside it (like a wall) is excluded from
    color detection.

    Uses the click-point polygon from field_roi_tuner.py if one is
    saved; otherwise falls back to the old rectangle
    (config.FIELD_ROI_*_FRAC)."""

    polygon = vision.get_field_polygon_pixels(height, width)

    if polygon is not None:
        cv2.polylines(result, [polygon], True, (0, 255, 255), 2)
        cv2.putText(
            result, "field boundary", (int(polygon[:, 0].min()) + 5, int(polygon[:, 1].min()) + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )
        return

    x1, y1, x2, y2 = vision.get_field_roi_pixels(height, width)

    # Only draw it if it's actually restricting something - a
    # full-frame default (0,0,1,1) would just draw a border around
    # the whole image, which isn't useful information.
    is_full_frame = (x1 == 0 and y1 == 0 and x2 == width and y2 == height)
    if is_full_frame:
        return

    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        result, "field boundary", (x1 + 5, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )


# NOTE: draw_division_line() has been removed (the white line across the
# middle of the frame). Delete any call to it in main.py.


def draw_target_circles(result, target_circles):
    """Draws each color base as a CIRCLE (instead of a box).

    The circle is centered on the base's bounding box, with a radius
    that fits it. Works best with target_memory.TargetMemory.update()
    applied first, so a base stays drawn (locked in place) even while
    the robot is on top of it and blocking the camera's view.
    """

    for obj in target_circles:
        box_color = obj["box_color"]

        cx = int((obj["x1"] + obj["x2"]) / 2)
        cy = int((obj["y1"] + obj["y2"]) / 2)
        width = obj["x2"] - obj["x1"]
        height = obj["y2"] - obj["y1"]
        radius = max(1, int((width + height) / 4))

        cv2.circle(result, (cx, cy), radius, box_color, 3)

        cv2.putText(
            result, f"{obj['target_index']}: {obj['color']}",
            (cx - radius, max(30, cy - radius - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA
        )


def draw_field_gems(result, field_gems):
    for gem in field_gems:
        box_color = gem["box_color"]
        cv2.rectangle(result, (gem["x1"], gem["y1"]), (gem["x2"], gem["y2"]), box_color, 2)
        cv2.circle(result, (gem["center_x"], gem["center_y"]), 4, box_color, -1)


def draw_ignored_gems(result, ignored_gems):
    """Gems that were detected but IGNORED because they sit on a base.
    Drawn gray so you can see what was rejected (and why the robot isn't
    chasing it)."""

    for gem in ignored_gems:
        cv2.rectangle(result, (gem["x1"], gem["y1"]), (gem["x2"], gem["y2"]), (150, 150, 150), 1)
        cv2.putText(
            result, "ignored", (gem["x1"], max(12, gem["y1"] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA
        )


def draw_navigation(result, robot_center, gripper_center, held_gem_color, nav):
    """Draws the aim line, target circle, PICK!/PLACE! label and status text."""

    if nav["nav_target_point"] is None or gripper_center is None:
        return

    tx, ty = nav["nav_target_point"]
    ready = nav["ready_to_grab"] or nav["ready_to_place"]

    label = (
        "PICK!" if nav["ready_to_grab"] else
        "PLACE!" if nav["ready_to_place"] else
        "HOLDING" if held_gem_color else
        "PICKUP"
    )
    label_color = (0, 255, 0) if ready else (255, 0, 255)

    cv2.putText(
        result, label,
        (int(gripper_center[0]) - 35, int(gripper_center[1]) - config.PICKUP_RADIUS - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2, cv2.LINE_AA
    )

    # aim line starts at the GRIP SPOT (the point navigation now steers from)
    cv2.line(
        result, (int(gripper_center[0]), int(gripper_center[1])), (int(tx), int(ty)),
        (0, 255, 255), 2
    )

    cv2.circle(
        result, (int(tx), int(ty)), config.STOP_DISTANCE,
        (0, 255, 0) if ready else (0, 255, 255), 2
    )

    cv2.putText(
        result,
        f"{nav['nav_command']} -> {nav['nav_target_label']} "
        f"(gripper_d={nav['gripper_distance']:.0f}, a={nav['angle_diff']:.0f})",
        (int(robot_center[0]) + 10, int(robot_center[1]) + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
    )