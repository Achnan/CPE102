"""
Live HSV Tuner
==============
Run this any time lighting has changed and your color detection looks
off. Adjust sliders until the MASK window shows clean white blobs where
your colored objects are (and black everywhere else), then press 's'
to save. main.py will automatically pick up the saved values next time
you run it - no code editing required.

This tuner runs the SAME pipeline main.py uses (CLAHE brightness
normalization + the same open/close mask cleanup), so what you see
here should match what main.py actually detects.

One difference still remains: main.py blanks out a box around the
robot's ArUco marker before scanning for TARGET CIRCLES specifically
(so the robot body is never mistaken for one). This tuner has no
robot marker to detect, so it can't preview that blanking - if a
target circle sits very close to where the robot happens to be
parked, trust main.py's own display over the tuner for that spot.

Controls:
    Click a color button at the top   - pick which color you're tuning
    n / p                             - same thing, next / previous color
    H/S/V min/max sliders             - the HSV range for the selected color + range
    [ / ]                             - switch between multiple ranges for this color
                                         (e.g. "red" wraps around 0/179, so it needs 2)
    +                                 - add a new range to the current color (clone current)
    -                                 - remove the current range (only if more than 1 left)
    s                                 - save ALL colors' current ranges to hsv_overrides.json
    q / ESC                           - quit without saving further changes
"""

import copy
import json
import os

import cv2
import numpy as np

import config
import vision

OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsv_overrides.json")

WINDOW_MAIN = "HSV Tuner - camera"
WINDOW_MASK = "HSV Tuner - mask / result"

BUTTON_ROW_HEIGHT = 44   # pixels reserved at the top of the frame for color buttons


def _nothing(_value):
    pass


def draw_color_buttons(display, color_names, working, selected_index):
    """Draw one clickable button per color across the top of `display`."""

    width = display.shape[1]
    btn_w = max(1, width // len(color_names))

    for i, name in enumerate(color_names):
        x1 = i * btn_w
        x2 = width if i == len(color_names) - 1 else x1 + btn_w
        box_color = working[name]["box_color"]

        cv2.rectangle(display, (x1, 0), (x2, BUTTON_ROW_HEIGHT), box_color, -1)

        # pick readable text color based on button brightness
        brightness = 0.114 * box_color[0] + 0.587 * box_color[1] + 0.299 * box_color[2]
        text_color = (255, 255, 255) if brightness < 140 else (0, 0, 0)

        cv2.putText(
            display, name, (x1 + 6, BUTTON_ROW_HEIGHT - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
        )

        if i == selected_index:
            cv2.rectangle(display, (x1, 0), (x2 - 1, BUTTON_ROW_HEIGHT - 1), (255, 255, 255), 3)

    return btn_w


def button_index_for_click(x, y, width, num_colors):
    """Return which color button was clicked, or None if the click was elsewhere."""

    if y < 0 or y > BUTTON_ROW_HEIGHT:
        return None

    btn_w = max(1, width // num_colors)
    idx = x // btn_w
    return min(idx, num_colors - 1)


def build_working_copy():
    """Deep-copy just the ranges (as mutable lists) out of config.COLORS."""
    working = {}
    for name, (ranges, box_color) in config.COLORS.items():
        working[name] = {
            "ranges": [[list(lower), list(upper)] for lower, upper in ranges],
            "box_color": box_color,
        }
    return working


def save_overrides(working):
    to_save = {name: data["ranges"] for name, data in working.items()}
    with open(OVERRIDE_PATH, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"[hsv_tuner] Saved HSV ranges to {OVERRIDE_PATH}")


def main():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    working = build_working_copy()
    color_names = list(working.keys())

    state = {"color_index": 0, "range_index": 0, "frame_width": 640}

    cv2.namedWindow(WINDOW_MAIN)
    cv2.namedWindow(WINDOW_MASK)

    def on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        idx = button_index_for_click(x, y, state["frame_width"], len(color_names))
        if idx is not None and idx != state["color_index"]:
            state["color_index"] = idx
            state["range_index"] = 0
            push_sliders_from_state()

    cv2.setMouseCallback(WINDOW_MAIN, on_mouse)

    cv2.createTrackbar("H min", WINDOW_MAIN, 0, 179, _nothing)
    cv2.createTrackbar("S min", WINDOW_MAIN, 0, 255, _nothing)
    cv2.createTrackbar("V min", WINDOW_MAIN, 0, 255, _nothing)
    cv2.createTrackbar("H max", WINDOW_MAIN, 179, 179, _nothing)
    cv2.createTrackbar("S max", WINDOW_MAIN, 255, 255, _nothing)
    cv2.createTrackbar("V max", WINDOW_MAIN, 255, 255, _nothing)

    def push_sliders_from_state():
        name = color_names[state["color_index"]]
        ranges = working[name]["ranges"]
        state["range_index"] = min(state["range_index"], len(ranges) - 1)
        lower, upper = ranges[state["range_index"]]

        cv2.setTrackbarPos("H min", WINDOW_MAIN, lower[0])
        cv2.setTrackbarPos("S min", WINDOW_MAIN, lower[1])
        cv2.setTrackbarPos("V min", WINDOW_MAIN, lower[2])
        cv2.setTrackbarPos("H max", WINDOW_MAIN, upper[0])
        cv2.setTrackbarPos("S max", WINDOW_MAIN, upper[1])
        cv2.setTrackbarPos("V max", WINDOW_MAIN, upper[2])

    push_sliders_from_state()

    print(__doc__)

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera")
            break

        state["frame_width"] = frame.shape[1]

        name = color_names[state["color_index"]]
        ranges = working[name]["ranges"]
        box_color = working[name]["box_color"]

        # ---- read current slider values back into the working copy ----
        h_min = cv2.getTrackbarPos("H min", WINDOW_MAIN)
        s_min = cv2.getTrackbarPos("S min", WINDOW_MAIN)
        v_min = cv2.getTrackbarPos("V min", WINDOW_MAIN)
        h_max = cv2.getTrackbarPos("H max", WINDOW_MAIN)
        s_max = cv2.getTrackbarPos("S max", WINDOW_MAIN)
        v_max = cv2.getTrackbarPos("V max", WINDOW_MAIN)

        ranges[state["range_index"]] = [[h_min, s_min, v_min], [h_max, s_max, v_max]]

        # ---- build mask using the SAME pipeline main.py uses ----
        # (vision.to_hsv applies the same CLAHE brightness normalization,
        # and clean_mask applies the same open/close cleanup - so what
        # you see here is what main.py will actually detect.)
        hsv = vision.to_hsv(frame)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = vision.clean_mask(mask)

        result = cv2.bitwise_and(frame, frame, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # ---- draw clickable color buttons + info text on the main window ----
        display = frame.copy()
        draw_color_buttons(display, color_names, working, state["color_index"])

        text_top = BUTTON_ROW_HEIGHT + 25
        info_lines = [
            f"Color: {name}  (range {state['range_index'] + 1}/{len(ranges)})",
            f"Lower: {[h_min, s_min, v_min]}   Upper: {[h_max, s_max, v_max]}",
            "click a button above or n/p to switch color   [ / ] range   + add   - remove   s save   q quit",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(
                display, line, (10, text_top + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA
            )

        cv2.imshow(WINDOW_MAIN, display)
        cv2.imshow(WINDOW_MASK, np.hstack([mask_bgr, result]))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('n'):
            state["color_index"] = (state["color_index"] + 1) % len(color_names)
            state["range_index"] = 0
            push_sliders_from_state()

        elif key == ord('p'):
            state["color_index"] = (state["color_index"] - 1) % len(color_names)
            state["range_index"] = 0
            push_sliders_from_state()

        elif key == ord('['):
            state["range_index"] = (state["range_index"] - 1) % len(ranges)
            push_sliders_from_state()

        elif key == ord(']'):
            state["range_index"] = (state["range_index"] + 1) % len(ranges)
            push_sliders_from_state()

        elif key == ord('+'):
            ranges.append(copy.deepcopy(ranges[state["range_index"]]))
            state["range_index"] = len(ranges) - 1
            push_sliders_from_state()
            print(f"[hsv_tuner] Added range #{len(ranges)} for '{name}'")

        elif key == ord('-'):
            if len(ranges) > 1:
                ranges.pop(state["range_index"])
                state["range_index"] = max(0, state["range_index"] - 1)
                push_sliders_from_state()
                print(f"[hsv_tuner] Removed a range for '{name}', {len(ranges)} left")
            else:
                print(f"[hsv_tuner] '{name}' must keep at least 1 range")

        elif key == ord('s'):
            save_overrides(working)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()