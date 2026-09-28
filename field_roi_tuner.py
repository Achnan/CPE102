"""
Field Area Tuner  (click-point version)
=======================================
Mark the play area with corner POINTS instead of a rectangle, so it can
match a tilted / non-square field (and cut out a wall of the same color).
Anything OUTSIDE the shape is ignored by detection in main.py.

Controls:
    left click on empty space   - add a point
    left click + drag a point   - move it
    right click a point         - delete it   (or press z to undo the last one added)
    c                           - clear all points (back to no restriction)
    q / ESC                     - quit

You need at least 3 points. Click the corners of the field in ANY order -
they are sorted automatically so the outline never crosses itself. It is
AUTO-SAVED to robot_overrides.json every time you add / move / delete a
point. Restart main.py afterwards to apply it.

The area outside the shape is shown darkened: what you see bright is
exactly what main.py will look at.

Don't run this at the same time as main.py - only one program can use
the camera at once.
"""

import cv2
import numpy as np

import config
import field_area

WINDOW = "Field Area Tuner"
GRAB_RADIUS = 14   # pixels: how close a click must be to grab an existing point

# Points in CLICK order (so 'z' can undo the last one), as fractions 0-1.
points = []
drag_index = None
frame_size = [0, 0]
status = ""


def _load_start_points():
    saved = field_area.load_points()
    if saved:
        return saved

    # No polygon saved yet: start from the old rectangle (if one was set)
    # so its four corners can just be nudged into shape.
    x1 = getattr(config, "FIELD_ROI_X1_FRAC", 0.0)
    y1 = getattr(config, "FIELD_ROI_Y1_FRAC", 0.0)
    x2 = getattr(config, "FIELD_ROI_X2_FRAC", 1.0)
    y2 = getattr(config, "FIELD_ROI_Y2_FRAC", 1.0)
    if (x1, y1, x2, y2) != (0.0, 0.0, 1.0, 1.0):
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return []


def _clamp01(v):
    return max(0.0, min(1.0, v))


def _to_frac(x, y):
    w, h = frame_size
    return [_clamp01(x / w), _clamp01(y / h)]


def _nearest_point(x, y):
    w, h = frame_size
    best, best_d = None, GRAB_RADIUS
    for i, (fx, fy) in enumerate(points):
        d = ((fx * w - x) ** 2 + (fy * h - y) ** 2) ** 0.5
        if d <= best_d:
            best, best_d = i, d
    return best


def save():
    global status
    if len(points) < 3:
        status = f"{len(points)} point(s) - need at least 3 (not saved yet)"
        return
    field_area.save_points(field_area.order_points(points))
    status = "Saved! Restart main.py to apply."
    print(f"[field_roi_tuner] Saved {len(points)} points.")


def on_mouse(event, x, y, _flags, _param):
    global drag_index, status

    if frame_size[0] == 0:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = _nearest_point(x, y)
        if idx is None:
            points.append(_to_frac(x, y))
            idx = len(points) - 1
        drag_index = idx
        status = ""

    elif event == cv2.EVENT_MOUSEMOVE and drag_index is not None:
        points[drag_index] = _to_frac(x, y)

    elif event == cv2.EVENT_LBUTTONUP and drag_index is not None:
        drag_index = None
        save()

    elif event == cv2.EVENT_RBUTTONDOWN:
        idx = _nearest_point(x, y)
        if idx is not None:
            points.pop(idx)
            save()


def main():
    global status

    points[:] = _load_start_points()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)

    while True:
        ret, image = cap.read()
        if not ret:
            print("Cannot read camera")
            break

        h, w = image.shape[:2]
        frame_size[0], frame_size[1] = w, h

        display = image.copy()

        if len(points) >= 3:
            polygon = field_area.to_pixels(field_area.order_points(points), h, w)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            inside = mask.astype(bool)
            display = cv2.convertScaleAbs(image, alpha=0.35)
            display[inside] = image[inside]
            cv2.polylines(display, [polygon], True, (0, 255, 255), 2)
            info = f"Field area: {len(points)} points"
        elif len(points) > 0:
            info = f"{len(points)} point(s) - add at least {3 - len(points)} more"
        else:
            info = "No field area set - click the corners of your field"

        for i, (fx, fy) in enumerate(points):
            p = (int(fx * w), int(fy * h))
            cv2.circle(display, p, 6, (0, 0, 255), -1)
            cv2.circle(display, p, 8, (255, 255, 255), 1)
            cv2.putText(display, str(i + 1), (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, "click = add   drag = move   right-click / z = delete   c = clear   q = quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if status:
            cv2.putText(display, status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('z') and points:
            points.pop()
            save()
        elif key == ord('c'):
            points.clear()
            field_area.save_points(None)
            status = "Cleared - no restriction. Restart main.py to apply."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()