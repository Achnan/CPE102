"""
gem_counter.py
==============
Counts the gems the camera currently sees - per colour and in total - and
draws them as a small panel in a corner of main.py's window.

Keys (in the video window, handled by main.py)
    c   show / hide the panel   (hidden, a tiny "c = show stone counts" reminder stays)
    v   switch between the FULL panel and a one-line COMPACT bar (when you need
        to see more of the field); pressing it while hidden shows the panel

What the panel shows
    * one row per colour: a swatch, the name, a bar and the count. A colour with
      no gems is dimmed so the ones that exist stand out; one switched off with
      'x' in hsv_tuner.py shows "off" instead of a misleading 0.
    * a green "+1" / orange "-1" next to a number for a moment after it changes
      (and a tinted row), so you can SEE a gem being picked up or appearing.
    * TOTAL of all colours.
    * ON BASES x of y, with a progress bar: gems already sitting on a base,
      out of every gem seen (loose + on a base). A gem the robot is carrying is
      in neither number while it is in the claw, so the bar dips briefly.

What is counted
    TOTAL and the colour rows count LOOSE gems on the field (the ones the robot
    could still pick up). Gems on a base are counted only in the ON BASES line.

Two things to know about the numbers
    * They are detections, not a head-count. Gems that touch can merge into a
      single blob, which counts as ONE, so a pile reads lower than it really is.
    * A raw count flickers by +-1 from frame to frame (a gem half hidden by the
      claw, a blob that splits and rejoins), which makes a number unreadable.
      So the panel shows the MEDIAN of the last SMOOTH_FRAMES frames. At 30 fps
      the default of 15 is about half a second of delay. Set it to 1 to see the
      raw count.

Settings are the constants just below.
"""

import time
from collections import deque

import cv2
import numpy as np

import config

try:
    import color_toggles
except ImportError:          # the toggle feature is optional
    color_toggles = None

SMOOTH_FRAMES = 15           # median over this many frames; 1 = raw count
CORNER = "top-right"         # top-right, top-left, bottom-right or bottom-left.
                             # It only covers the picture you look at (detection is done before it is drawn),
                             # so move it if it hides something you want to watch.
SHOW_AT_START = True         # False = start hidden (press c to show)
COMPACT_AT_START = False     # True = start with the one-line bar
CHANGE_FLASH_SEC = 1.6       # how long a "+1" / "-1" stays after a count changes
TOGGLE_REFRESH_SEC = 1.0     # how often to re-read which colours are switched off

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, DIM, GREY = (235, 235, 235), (115, 115, 115), (160, 160, 160)
TITLE, GOOD, DOWN = (0, 220, 255), (0, 255, 0), (0, 150, 255)   # BGR


class GemCounter:

    def __init__(self, smooth_frames=SMOOTH_FRAMES):
        self.history = {name: deque(maxlen=max(1, smooth_frames)) for name in config.COLORS}
        self.on_base = deque(maxlen=max(1, smooth_frames))
        self.visible = SHOW_AT_START
        self.compact = COMPACT_AT_START
        self._toggles = {}
        self._toggles_read_at = -1e9
        self._shown = {}      # key -> the value last displayed
        self._flash = {}      # key -> (value before the change, time of the latest change)

    # ------------------------------------------------------------------
    # on / off, size
    def toggle(self):
        """Show / hide the panel. Returns True if it is now visible."""
        self.visible = not self.visible
        return self.visible

    def toggle_compact(self):
        """Full <-> one-line bar (and show it if it was hidden). Returns True if now compact."""
        self.compact = not self.compact
        self.visible = True
        return self.compact

    # ------------------------------------------------------------------
    def _disabled(self):
        if color_toggles is None:
            return set()
        now = time.monotonic()
        if now - self._toggles_read_at >= TOGGLE_REFRESH_SEC:
            self._toggles = color_toggles.load_toggles()
            self._toggles_read_at = now
        return {name for name, on in self._toggles.items() if not on}

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    def update(self, field_gems, ignored_gems=()):
        """Feed this frame's detections (call it every frame, shown or not, so the
        numbers are ready the moment the panel is switched on). Returns
        {"per_color": {name: count or None if the colour is off}, "total": n, "on_base": n}."""

        raw = {name: 0 for name in config.COLORS}
        for gem in field_gems:
            if gem["color"] in raw:
                raw[gem["color"]] += 1
        for name, count in raw.items():
            self.history[name].append(count)
        self.on_base.append(len(ignored_gems))

        disabled = self._disabled()
        per_color = {}
        for name in config.COLORS:
            per_color[name] = None if name in disabled else self._median(self.history[name])

        counts = {
            "per_color": per_color,
            "total": sum(c for c in per_color.values() if c is not None),
            "on_base": self._median(self.on_base),
        }
        self._track_changes(counts)
        return counts

    def _track_changes(self, counts):
        """Remember what changed and when, for the +1 / -1 markers."""
        now = time.monotonic()
        current = dict(counts["per_color"])
        current["__total"] = counts["total"]
        for key, value in current.items():
            if value is None:
                self._shown.pop(key, None)
                self._flash.pop(key, None)
                continue
            old = self._shown.get(key)
            if old is not None and value != old:
                flash = self._flash.get(key)
                recent = flash is not None and now - flash[1] <= CHANGE_FLASH_SEC
                self._flash[key] = (flash[0] if recent else old, now)   # keep the starting value while changes keep coming
            self._shown[key] = value

    def _delta(self, key, value, now):
        """How much `key` changed within the last CHANGE_FLASH_SEC (0 = nothing to show)."""
        flash = self._flash.get(key)
        if flash is None or value is None or now - flash[1] > CHANGE_FLASH_SEC:
            return 0
        return value - flash[0]

    # ------------------------------------------------------------------
    # drawing
    def draw(self, image, counts):
        """Draw the panel (or just the reminder if hidden) in the CORNER of `image`, in place."""
        now = time.monotonic()
        h, w = image.shape[:2]
        ui = max(1.0, w / 1280.0)                 # grow everything on big frames
        if not self.visible:
            self._draw_hint(image, w, h, ui)
        elif self.compact:
            self._draw_compact(image, counts, w, h, ui, now)
        else:
            self._draw_full(image, counts, w, h, ui, now)

    @staticmethod
    def _origin(w, h, panel_w, panel_h, ui):
        margin = int(12 * ui)
        x0 = margin if "left" in CORNER else w - panel_w - margin
        y0 = margin if "top" in CORNER else h - panel_h - margin
        return x0, y0

    @staticmethod
    def _backdrop(image, x0, y0, x1, y1, alpha=0.78):
        """Darken the picture behind the text so it stays readable."""
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.shape[1], x1), min(image.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return False
        roi = image[y0:y1, x0:x1]
        cv2.addWeighted(np.full_like(roi, 22), alpha, roi, 1 - alpha, 0, roi)
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (110, 110, 110), 1)
        return True

    @staticmethod
    def _text_w(text, scale, thick):
        return cv2.getTextSize(text, FONT, scale, thick)[0][0]

    def _draw_hint(self, image, w, h, ui):
        text, scale, thick = "c = show stone counts", 0.5 * ui, max(1, int(round(ui)))
        tw = self._text_w(text, scale, thick)
        pad = int(6 * ui)
        x0, y0 = self._origin(w, h, tw + 2 * pad, int(22 * ui), ui)
        if self._backdrop(image, x0, y0, x0 + tw + 2 * pad, y0 + int(22 * ui), alpha=0.6):
            cv2.putText(image, text, (x0 + pad, y0 + int(16 * ui)), FONT, scale, GREY, thick, cv2.LINE_AA)

    def _tint_row(self, image, x0, y0, x1, y1, color):
        """A faint colour wash behind a row whose number just changed."""
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.shape[1], x1), min(image.shape[0], y1)
        if x1 > x0 and y1 > y0:
            roi = image[y0:y1, x0:x1]
            cv2.addWeighted(np.full_like(roi, color), 0.30, roi, 0.70, 0, roi)

    # -- full panel ------------------------------------------------------
    def _draw_full(self, image, counts, w, h, ui, now):
        scale, thick = 0.6 * ui, max(1, int(round(1.5 * ui)))
        small = scale * 0.78
        row_h, pad, sw, gap = int(28 * ui), int(12 * ui), int(15 * ui), int(8 * ui)
        names = list(config.COLORS)
        panel_w = int(285 * ui)
        bar_row = int(20 * ui)
        panel_h = pad * 2 + row_h * (len(names) + 3) + bar_row
        x0, y0 = self._origin(w, h, panel_w, panel_h, ui)
        x1, y1 = x0 + panel_w, y0 + panel_h
        if not self._backdrop(image, x0, y0, x1, y1):
            return

        def put(text, x, y, color=WHITE, s=scale):
            cv2.putText(image, text, (x, y), FONT, s, color, thick, cv2.LINE_AA)

        def put_right(text, y, color=WHITE, s=scale, right=None):
            put(text, (x1 - pad if right is None else right) - self._text_w(text, s, thick), y, color, s)

        name_w = max(self._text_w(n, scale, thick) for n in names)
        num_w = self._text_w("00", scale, thick)
        delta_w = self._text_w("+9", scale, thick)
        bar_x0 = x0 + pad + sw + gap + name_w + gap
        bar_x1 = x1 - pad - num_w - gap - delta_w - gap
        bar_h = max(4, int(8 * ui))
        biggest = max([c for c in counts["per_color"].values() if c is not None] + [1])

        # title + the key hints
        y = y0 + pad + int(row_h * 0.7)
        put("STONES SEEN", x0 + pad, y, TITLE)
        for hint in ("c hide  v size", "c hide"):
            room = panel_w - 2 * pad - self._text_w("STONES SEEN", scale, thick)
            if self._text_w(hint, small * 0.85, thick) + int(14 * ui) <= room:
                put_right(hint, y, DIM, small * 0.85)
                break
        y += row_h

        # one row per colour
        for name in names:
            count = counts["per_color"][name]
            delta = self._delta(name, count, now)
            if delta:
                self._tint_row(image, x0 + 1, y - int(row_h * 0.72), x1 - 1, y + int(row_h * 0.22),
                               (0, 110, 0) if delta > 0 else (0, 60, 130))
            box_color = config.COLORS[name][1]
            top = y - sw + int(2 * ui)
            has = bool(count)
            if has:
                cv2.rectangle(image, (x0 + pad, top), (x0 + pad + sw, top + sw), box_color, -1)
                cv2.rectangle(image, (x0 + pad, top), (x0 + pad + sw, top + sw), (255, 255, 255), 1)
            else:   # nothing of this colour: an empty swatch
                cv2.rectangle(image, (x0 + pad, top), (x0 + pad + sw, top + sw), box_color, 1)
            put(name, x0 + pad + sw + gap, y, WHITE if has else DIM)

            if count is None:
                put_right("off", y, DIM)
            else:
                if bar_x1 > bar_x0:
                    by = y - bar_h
                    cv2.rectangle(image, (bar_x0, by), (bar_x1, by + bar_h), (60, 60, 60), -1)
                    fill = int((bar_x1 - bar_x0) * count / biggest)
                    if fill > 0:
                        cv2.rectangle(image, (bar_x0, by), (bar_x0 + fill, by + bar_h), box_color, -1)
                put_right(str(count), y, WHITE if has else DIM)
                if delta:
                    put_right(f"{delta:+d}", y, GOOD if delta > 0 else DOWN,
                              right=x1 - pad - num_w - gap)
            y += row_h

        # total
        cv2.line(image, (x0 + pad, y - int(row_h * 0.75)), (x1 - pad, y - int(row_h * 0.75)), (150, 150, 150), 1)
        total, delta = counts["total"], self._delta("__total", counts["total"], now)
        put("TOTAL", x0 + pad, y, GOOD)
        put_right(str(total), y, GOOD)
        if delta:
            put_right(f"{delta:+d}", y, GOOD if delta > 0 else DOWN, right=x1 - pad - num_w - gap)
        y += row_h

        # progress: how many of everything seen is already on a base
        on_base = counts["on_base"]
        seen = on_base + total
        put(f"ON BASES  {on_base} of {seen}", x0 + pad, y, WHITE if on_base else DIM, small)
        by = y + int(7 * ui)
        cv2.rectangle(image, (x0 + pad, by), (x1 - pad, by + bar_h), (60, 60, 60), -1)
        if seen:
            cv2.rectangle(image, (x0 + pad, by),
                          (x0 + pad + int((panel_w - 2 * pad) * on_base / seen), by + bar_h), GOOD, -1)

    # -- one-line bar ----------------------------------------------------
    def _draw_compact(self, image, counts, w, h, ui, now):
        scale, thick = 0.6 * ui, max(1, int(round(1.5 * ui)))
        small = scale * 0.7
        pad, sw, gap = int(10 * ui), int(13 * ui), int(5 * ui)
        panel_h = int(32 * ui)

        names = list(config.COLORS)
        parts = []   # (kind, payload, width)
        total_text = f"TOTAL {counts['total']}"
        parts.append(("total", total_text, self._text_w(total_text, scale, thick)))
        for name in names:
            count = counts["per_color"][name]
            text = "-" if count is None else str(count)
            parts.append(("color", (name, count, text), sw + gap + self._text_w(text, scale, thick)))
        base_text = f"bases {counts['on_base']}   v = full"
        parts.append(("bases", base_text, self._text_w(base_text, small, thick)))

        spacing = int(14 * ui)
        panel_w = pad * 2 + sum(p[2] for p in parts) + spacing * (len(parts) - 1)
        x0, y0 = self._origin(w, h, panel_w, panel_h, ui)
        if not self._backdrop(image, x0, y0, x0 + panel_w, y0 + panel_h):
            return

        y = y0 + int(panel_h * 0.68)
        x = x0 + pad
        for kind, payload, width in parts:
            if kind == "total":
                delta = self._delta("__total", counts["total"], now)
                color = GOOD if delta >= 0 else DOWN
                cv2.putText(image, payload, (x, y), FONT, scale, color if delta else GOOD, thick, cv2.LINE_AA)
            elif kind == "color":
                name, count, text = payload
                box_color = config.COLORS[name][1]
                top = y - sw + int(2 * ui)
                if count:
                    cv2.rectangle(image, (x, top), (x + sw, top + sw), box_color, -1)
                    cv2.rectangle(image, (x, top), (x + sw, top + sw), (255, 255, 255), 1)
                else:
                    cv2.rectangle(image, (x, top), (x + sw, top + sw), box_color, 1)
                delta = self._delta(name, count, now)
                color = (GOOD if delta > 0 else DOWN) if delta else (WHITE if count else DIM)
                cv2.putText(image, text, (x + sw + gap, y), FONT, scale, color, thick, cv2.LINE_AA)
            else:
                cv2.putText(image, payload, (x, y), FONT, small, GREY, thick, cv2.LINE_AA)
            x += width + spacing

    # ------------------------------------------------------------------
    @staticmethod
    def as_text(counts):
        """One line for the console log."""
        parts = [f"{name} {c}" for name, c in counts["per_color"].items() if c is not None]
        return f"stones seen -> {', '.join(parts)} | total {counts['total']} | on bases {counts['on_base']}"