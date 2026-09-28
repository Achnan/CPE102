"""
target_memory.py
================
Makes the color-base circles "sticky".

Problem: when the robot drives on top of a base, it blocks the camera's
view (and main/vision also blank the robot's area on purpose), so the
detector stops finding that base and it vanishes.

Fix: every base is remembered BY POSITION. Once a base has been seen for
LOCK_AFTER_FRAMES frames, its position is LOCKED (median of the samples,
so a noisy frame can't skew it) and never changes again, no matter what
the camera sees afterwards.

IMPORTANT - where this sits in the pipeline:

    detected_objects, field_gems = vision.detect_target_circles_and_gems(...)
    detected_objects = target_memory.update(detected_objects)      # <-- HERE
    target_circles, division_y = vision.assign_target_indices(detected_objects, height)

It goes BEFORE assign_target_indices, not after. That function numbers
bases by sorting left->right, so if a base disappears the numbering of
the others would shift. Feeding it the remembered (full) list keeps the
numbering stable.

Rules:
  - A new detection near an existing remembered base is treated as that
    same base (and ignored if the base is already locked).
  - A brand-new base only appears in the output after it has been seen
    CONFIRM_AFTER_FRAMES times, so a one-frame false detection never
    shows up.
  - A base that was never confirmed and stops being seen is forgotten.
  - At most MAX_TARGETS bases are remembered (3 top + 3 bottom).
"""

import math

import numpy as np

CONFIRM_AFTER_FRAMES = 5        # sightings before a new base is trusted/shown
LOCK_AFTER_FRAMES = 15          # sightings before a base is frozen in place
DROP_UNCONFIRMED_AFTER = 10     # frames unseen before an unconfirmed base is forgotten
MAX_TARGETS = 6                 # 3 top + 3 bottom


class TargetMemory:

    def __init__(
        self,
        confirm_after=CONFIRM_AFTER_FRAMES,
        lock_after=LOCK_AFTER_FRAMES,
        drop_unconfirmed_after=DROP_UNCONFIRMED_AFTER,
        max_targets=MAX_TARGETS,
    ):
        self.confirm_after = confirm_after
        self.lock_after = lock_after
        self.drop_unconfirmed_after = drop_unconfirmed_after
        self.max_targets = max_targets
        self._slots = []

    def reset(self):
        """Forget every base so they get re-detected and re-locked."""
        self._slots.clear()

    @staticmethod
    def _radius(obj):
        return max(obj["x2"] - obj["x1"], obj["y2"] - obj["y1"]) / 2.0

    def _find_slot(self, obj):
        """Nearest remembered base whose circle this detection overlaps."""
        best_slot = None
        best_dist = None

        for slot in self._slots:
            s = slot["obj"]
            dist = math.hypot(obj["center_x"] - s["center_x"], obj["center_y"] - s["center_y"])
            limit = max(self._radius(s), self._radius(obj))
            if dist <= limit and (best_dist is None or dist < best_dist):
                best_slot, best_dist = slot, dist

        return best_slot

    def update(self, detected_objects):
        """
        Feed this frame's detections in; get back the full list of
        remembered bases (including ones currently hidden under the
        robot). Same dict format as the detections, plus "locked".
        """

        for slot in self._slots:
            slot["updated"] = False

        for obj in detected_objects:

            slot = self._find_slot(obj)

            if slot is None:
                if len(self._slots) >= self.max_targets:
                    continue
                slot = {"obj": dict(obj), "samples": [], "seen": 0,
                        "missed": 0, "locked": False, "updated": False}
                self._slots.append(slot)

            slot["updated"] = True

            if slot["locked"]:
                continue   # frozen - ignore whatever the camera sees now

            slot["obj"] = dict(obj)
            slot["samples"].append(
                (obj["x1"], obj["y1"], obj["x2"], obj["y2"], obj["center_x"], obj["center_y"])
            )
            slot["seen"] += 1
            slot["missed"] = 0

            if slot["seen"] >= self.lock_after:
                x1, y1, x2, y2, cx, cy = np.median(np.array(slot["samples"]), axis=0)
                slot["obj"].update(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    center_x=int(cx), center_y=int(cy),
                )
                slot["locked"] = True
                slot["samples"] = []

        # anything not seen this frame: count the miss, forget flukes
        for slot in self._slots:
            if not slot["updated"]:
                slot["missed"] += 1

        self._slots = [
            s for s in self._slots
            if s["locked"] or s["seen"] >= self.confirm_after
            or s["missed"] <= self.drop_unconfirmed_after
        ]

        output = []
        for slot in self._slots:
            if slot["locked"] or slot["seen"] >= self.confirm_after:
                obj = dict(slot["obj"])
                obj["locked"] = slot["locked"]
                output.append(obj)
        return output