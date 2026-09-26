#======ROBOTTRACK

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


def gripper_center_of(robot_info):
    """Compute the pickup/placement circle center in front of the robot."""

    if robot_info["center"] is None or robot_info["forward_unit"] is None:
        return None

    return robot_info["center"] + robot_info["forward_unit"] * config.GRIPPER_FORWARD_OFFSET