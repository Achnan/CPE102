import cv2
import numpy as np

import config


def make_detector():
    """Create the ArUco detector once at startup."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config.ARUCO_DICT_NAME)
    )
    aruco_params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def detect_robot(image, detector):
    """
    Find the robot's ArUco marker in `image`.

    Returns a dict with:
        center          : np.array([x, y]) or None
        heading_deg     : float or None
        forward_unit    : np.array([x, y]) or None (unit vector, "facing" direction)
        roi             : (x_min, y_min, x_max, y_max) or None
        marker_corners  : raw corners list (for drawing)
        marker_ids      : raw ids array (for drawing)
    """

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    info = {
        "center": None,
        "heading_deg": None,
        "forward_unit": None,
        "roi": None,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
    }

    if marker_ids is None:
        return info

    for corner, marker_id in zip(marker_corners, marker_ids.flatten()):

        if marker_id != config.ROBOT_MARKER_ID:
            continue

        pts = corner.reshape(4, 2)

        x_min = int(pts[:, 0].min()) - config.ROBOT_MASK_PADDING
        y_min = int(pts[:, 1].min()) - config.ROBOT_MASK_PADDING
        x_max = int(pts[:, 0].max()) + config.ROBOT_MASK_PADDING
        y_max = int(pts[:, 1].max()) + config.ROBOT_MASK_PADDING

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width - 1, x_max)
        y_max = min(height - 1, y_max)

        info["roi"] = (x_min, y_min, x_max, y_max)

        # Corners are 0=TL, 1=TR, 2=BR, 3=BL.
        # Midpoint of top edge minus center = "forward" direction.
        top_left = pts[0]
        top_right = pts[1]

        center = pts.mean(axis=0)
        forward_vector = (top_left + top_right) / 2.0 - center

        info["center"] = center
        info["heading_deg"] = np.degrees(
            np.arctan2(forward_vector[1], forward_vector[0])
        )

        forward_norm = np.linalg.norm(forward_vector)
        if forward_norm > 0:
            info["forward_unit"] = forward_vector / forward_norm

        break

    return info


def rotate_vector(vec, angle_deg):
    """
    Rotate a 2D vector by angle_deg (standard math rotation: positive angle
    turns from +x toward +y). Used to slant the grip spot when the top-view
    camera isn't mounted perfectly perpendicular to the field, so "forward"
    in the image doesn't quite line up with where the gripper physically is.
    """

    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x, y = vec
    return np.array([x * cos_a - y * sin_a, x * sin_a + y * cos_a])


def gripper_center_of(robot_info):
    """
    Compute the pickup/placement circle center in front of the robot.

    GRIPPER_ANGLE_OFFSET_DEG (default 0, tuned live in grip_tuner.py) rotates
    the direction the offset is applied in, away from the robot's raw ArUco
    heading - this compensates for a top-view camera that's mounted at a
    slight slant, where "straight ahead" in the image isn't quite where the
    gripper actually is relative to the marker.
    """

    if robot_info["center"] is None or robot_info["forward_unit"] is None:
        return None

    # grip_tuner.py sets config.GRIPPER_ANGLE_OFFSET_DEG live, and config.py
    # loads the saved value from robot_overrides.json on a normal run.
    angle_offset = getattr(config, "GRIPPER_ANGLE_OFFSET_DEG", 0.0)
    direction = robot_info["forward_unit"]
    if angle_offset:
        direction = rotate_vector(direction, angle_offset)

    return robot_info["center"] + direction * config.GRIPPER_FORWARD_OFFSET


def tip_center_of(robot_info):
    """
    Centre of the claw's TIP area: the front end of the claw, a little further
    along the same (slanted) direction as the grip spot. motion.py uses the
    distance from here to a gem to decide when the claw must be open.
    """

    if robot_info["center"] is None or robot_info["forward_unit"] is None:
        return None

    angle_offset = getattr(config, "GRIPPER_ANGLE_OFFSET_DEG", 0.0)
    direction = robot_info["forward_unit"]
    if angle_offset:
        direction = rotate_vector(direction, angle_offset)

    tip_offset = getattr(config, "GRIPPER_TIP_OFFSET", config.GRIPPER_FORWARD_OFFSET + 25)
    return robot_info["center"] + direction * tip_offset