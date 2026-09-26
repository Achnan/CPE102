#=======MAIN

import cv2

import config
import robot_tracker
import vision
import navigation
import drawing


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    aruco_detector = robot_tracker.make_detector()

    while True:

        ret, image = cap.read()

        if not ret:
            print("Cannot read camera")
            break

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

        target_circles, division_y = vision.assign_target_indices(detected_objects, height)

        # ---- held gem color, sampled from the pickup circle ----
        held_gem_color = None
        if gripper_center is not None:
            held_gem_color = vision.sample_color_at(
                hsv_image,
                int(gripper_center[0]), int(gripper_center[1]),
                config.GEM_SAMPLE_RADIUS
            )

        print(f"robot holding -> {held_gem_color}")

        if robot_info["center"] is not None:
            print(
                f"robot position -> "
                f"({int(robot_info['center'][0])}, {int(robot_info['center'][1])}), "
                f"heading -> {robot_info['heading_deg']:.1f} deg"
            )
        else:
            print("robot position -> not detected")

        # ---- navigation ----
        nav = navigation.compute_navigation(
            robot_info["center"], robot_info["heading_deg"], gripper_center,
            held_gem_color, field_gems, target_circles
        )

        drawing.draw_navigation(
            result, robot_info["center"], gripper_center, held_gem_color, nav
        )

        print(
            f"nav command -> {nav['nav_command']} "
            f"(aiming at: {nav['nav_target_label']}, angle_diff: {nav['angle_diff']}, "
            f"ready_to_grab: {nav['ready_to_grab']}, ready_to_place: {nav['ready_to_place']})"
        )

        # ---- draw field layout ----
        drawing.draw_division_line(result, width, division_y)
        drawing.draw_target_circles(result, target_circles)
        drawing.draw_field_gems(result, field_gems)

        for obj in target_circles:
            side = vision.get_side(obj["target_index"])
            print(f"target {obj['target_index']} ({side}) -> {obj['color']}")
        print()

        for gem in field_gems:
            print(f"gem -> {gem['color']} at ({gem['center_x']}, {gem['center_y']})")
        print()

        # ---- display ----
        cv2.imshow("Overview Camera - Target Circles", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()