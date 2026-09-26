import cv2

import config
import robot_tracker
import vision
import navigation
import drawing
import esp32_link


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    aruco_detector = robot_tracker.make_detector()

    # Tracks whether the gripper was holding something last frame, so
    # GRAB/RELEASE only get sent once per pickup/placement — not on
    # every single frame the robot happens to sit inside the zone.
    was_holding = False
    grabbed_this_cycle = False
    placed_this_cycle = False

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

        # ========================================================
        # SEND COMMAND TO ESP32 OVER WI-FI
        #
        # GRAB/RELEASE fire once per event (edge-triggered off
        # ready_to_grab/ready_to_place going True), sent with
        # force=True so they aren't delayed by the send throttle.
        # Every other frame just resends the current movement
        # command (FORWARD/LEFT/RIGHT/STOP) — this is required,
        # not optional: the ESP32 auto-stops itself if it doesn't
        # hear ANY command for 1.5 seconds.
        # ========================================================

        if nav["ready_to_grab"] and not grabbed_this_cycle:
            esp32_link.send_command("GRAB", force=True)
            grabbed_this_cycle = True

        elif nav["ready_to_place"] and not placed_this_cycle:
            esp32_link.send_command("RELEASE", force=True)
            placed_this_cycle = True

        else:
            esp32_link.send_command(nav["nav_command"])

        # Reset the one-shot guards when the held/not-held state
        # actually flips, so the next pickup/placement can trigger
        # GRAB/RELEASE again.
        is_holding = held_gem_color is not None

        if is_holding and not was_holding:
            placed_this_cycle = False   # just picked up - ready to place next

        if not is_holding and was_holding:
            grabbed_this_cycle = False  # just placed/dropped - ready to grab next

        was_holding = is_holding

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

    # Make sure the robot doesn't keep driving after the script exits.
    esp32_link.send_command("STOP", force=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()