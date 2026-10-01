"""
Live HSV Tuner
==============
Run this any time lighting has changed and your color detection looks
off. Adjust sliders until the MASK panel shows clean white blobs where
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

Everything lives in ONE window now: color buttons, sliders, the live
camera feed (with detected blobs outlined), the raw mask, and the
masked result are all shown together so you never have to hunt for a
second window.

EYEDROPPER (new): instead of guessing slider values, pick the color
straight from the camera image:
    1. Click a color button (e.g. red) to choose which color you're setting.
    2. Press 'e' to turn the eyedropper ON.
    3. Click on the object in the CAMERA panel. The HSV range for that
       color is set automatically (and the sliders jump to it).
    4. Click more spots on the same kind of object (bright side, shadow
       side...) and the range WIDENS to cover all of them.
    5. Fine-tune with the sliders as usual, or change the two "Pick tol"
       sliders (how much room to add around what you clicked) - the next
       click re-applies with the new tolerance.
Reds that wrap around the hue seam (0/179) are handled for you: the
eyedropper creates two ranges automatically when needed.
Note: an eyedropper click REPLACES all ranges of the selected color
(it recalculates from your clicked samples). Slider tweaks made after
a click are kept until your next click.

EXCLUDING (new): a range is a box - one min and one max for each of H, S and V -
so the only way to drop a wrongly detected thing is to raise a min or lower a
max. The tuner looks at the pixels of what you clicked, tries every channel and
side, and puts the new limit in the MIDDLE of the gap between that thing and your
good clicks, on the channel where the gap is widest (a cut that barely squeezes
between them would lose the gem when the light changes). The footer says what it
did. If that thing looks the same as your good clicks it says so and changes
nothing - lower "Pick tol SV" or click a cleaner good sample instead. Exclusions
are kept when you add more good clicks, and cleared when you switch colour.

Controls:
    Click a color button at the top   - pick which color you're tuning
    n / p                             - same thing, next / previous color
    H/S/V min/max sliders             - the HSV range for the selected color + range
    [ / ]                             - switch between multiple ranges for this color
                                         (e.g. "red" wraps around 0/179, so it needs 2)
    +                                 - add a new range to the current color (clone current)
    -                                 - remove the current range (only if more than 1 left)
    e                                 - eyedropper on / off (click the CAMERA panel to sample)
    x                                 - enable/disable the selected color for detection
                                         (a disabled color is completely skipped by
                                         main.py - as if it weren't tuned at all - until
                                         you turn it back on here; its saved HSV ranges
                                         are untouched, so you can still tune it while
                                         it's off and re-enable it once it's fixed)
    m                                 - EXCLUDE mode (eyedropper must be on, and you need at least
                                         one good click first). Click something that is being
                                         detected but should NOT be (a robot part, a neighbouring
                                         colour): the range is narrowed so that thing drops out
                                         while every good click stays. Press m again to go back
                                         to adding. See "EXCLUDING" below.
    u                                 - eyedropper: undo the last click (in exclude mode: the last exclusion)
    r                                 - eyedropper: forget all clicks AND exclusions (next click starts fresh)
    s                                 - save ALL colors' current ranges to hsv_overrides.json
    q / ESC                           - quit without saving further changes
"""

import copy
import json
import os
import time

import cv2
import numpy as np

import config
import vision
import color_toggles

OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsv_overrides.json")

WINDOW_NAME = "HSV Tuner"

# ---- layout constants -------------------------------------------------
BUTTON_ROW_HEIGHT = 46      # color picker buttons across the top
INFO_BAR_HEIGHT = 46        # selected color / range / HSV values readout
FOOTER_HEIGHT = 30          # condensed keybinding hints + status message
PANEL_HEADER_HEIGHT = 22    # "CAMERA" / "MASK" / "RESULT" labels
PANEL_GAP = 6               # gap between the three panels
PANEL_WIDTH = 400           # each of the 3 panels is resized to this width
MARGIN = 8                  # outer margin around the whole canvas

BG_COLOR = (32, 32, 32)
PANEL_BORDER = (90, 90, 90)
TEXT_COLOR = (230, 230, 230)
MUTED_TEXT = (150, 150, 150)
ACCENT = (60, 200, 255)
SAVE_FLASH_COLOR = (90, 220, 90)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---- eyedropper settings ---------------------------------------------
SAMPLE_RADIUS = 4           # each click averages a (2*4+1)=9x9 pixel patch
DEFAULT_TOL_H = 10          # hue room added either side of what you clicked
DEFAULT_TOL_SV = 50         # saturation/value room added either side


def _nothing(_value):
    pass


# ------------------------------------------------------------------ #
# Eyedropper helpers
# ------------------------------------------------------------------ #

def sample_hsv(hsv, cx, cy, radius=SAMPLE_RADIUS):
    """
    Median (H, S, V) of a small patch around (cx, cy), taken from the
    SAME hsv image the detection uses (blur + CLAHE), so the range it
    produces matches what main.py will actually see.

    Hue is a circle (0 and 179 are neighbours), so a plain median of a
    red patch could land on a nonsense value like 90. Hues are measured
    relative to the clicked pixel first, then the median is taken.
    """

    h_img, w_img = hsv.shape[:2]
    cx = min(max(cx, 0), w_img - 1)
    cy = min(max(cy, 0), h_img - 1)

    x1, x2 = max(0, cx - radius), min(w_img, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(h_img, cy + radius + 1)

    patch = hsv[y1:y2, x1:x2].reshape(-1, 3).astype(int)
    if patch.size == 0:
        return None

    ref = int(hsv[cy, cx][0])
    hue_diff = ((patch[:, 0] - ref + 90) % 180) - 90
    h = int(round(ref + np.median(hue_diff))) % 180
    s = int(np.median(patch[:, 1]))
    v = int(np.median(patch[:, 2]))
    return h, s, v


def hue_arc(hues):
    """
    Smallest arc of the 0-179 hue circle that contains every hue in
    `hues`, as (start, end). If start > end the arc wraps across 0/179.
    """

    hs = sorted(set(hues))
    n = len(hs)
    if n == 1:
        return hs[0], hs[0]

    best_gap, best_i = -1, 0
    for i in range(n):
        gap = (hs[(i + 1) % n] - hs[i]) % 180
        if gap > best_gap:
            best_gap, best_i = gap, i

    return hs[(best_i + 1) % n], hs[best_i]


def hue_ranges(start, end, tol):
    """
    Turn an arc + tolerance into 1 or 2 plain (lo, hi) hue ranges within
    0-179. Two ranges come back when the arc crosses the 0/179 seam
    (typical for red), because cv2.inRange can't wrap around by itself.
    """

    span = ((end - start) % 180) + 2 * tol
    if span >= 179:
        return [(0, 179)]

    lo = (start - tol) % 180
    hi = lo + span

    if hi <= 179:
        return [(lo, hi)]
    return [(lo, 179), (0, hi - 180)]


def ranges_from_samples(samples, tol_h, tol_sv):
    """List of [[h,s,v]_lower, [h,s,v]_upper] built from all clicked samples."""

    start, end = hue_arc([s[0] for s in samples])

    s_lo = max(0, min(s[1] for s in samples) - tol_sv)
    s_hi = min(255, max(s[1] for s in samples) + tol_sv)
    v_lo = max(0, min(s[2] for s in samples) - tol_sv)
    v_hi = min(255, max(s[2] for s in samples) + tol_sv)

    return [
        [[int(h_lo), int(s_lo), int(v_lo)], [int(h_hi), int(s_hi), int(v_hi)]]
        for h_lo, h_hi in hue_ranges(start, end, tol_h)
    ]


# ------------------------------------------------------------------ #
# Exclusion helpers
# ------------------------------------------------------------------ #

# ---- exclusion: narrow a colour range so a wrongly-detected blob drops out ----------------------
EXCLUDE_FULL_FRACTION = 0.95   # a cut that removes at least this much of the blob counts as "clean"
EXCLUDE_MIN_USEFUL = 0.6       # a partial cut must remove at least this much; below it the blob looks too much like your good clicks
BLOB_MAX_PIXELS = 6000         # a huge blob is sub-sampled to this many pixels
_CHANNEL = "HSV"               # channel index -> name
_SCALE = (179.0, 255.0, 255.0) # how big each channel is, to compare gaps between channels fairly
_MIN_GAP = (2, 8, 8)           # a cut needs at least this much room between the blob and your clicks


def _inside(points, lower, upper):
    """Boolean mask of the rows of `points` (N x 3) lying inside the box [lower, upper]."""
    points = np.asarray(points)
    return np.all((points >= np.asarray(lower)) & (points <= np.asarray(upper)), axis=1)


def cut_blob_out(ranges, blob, kept):
    """
    Narrow the HSV ranges so the pixels of `blob` (N x 3, the wrongly detected
    thing) stop matching, while EVERY point in `kept` (your good clicks) still
    matches.

    A range is a box: one min and one max per channel, so the only possible edit
    is to raise a min or lower a max, slicing a slab off one side of the box.
    For each range holding part of the blob, every channel and side is tried:

      * If the blob and your clicks are separated on that channel (there is a
        gap between them of at least _MIN_GAP), the new limit goes in the MIDDLE
        of the gap, so both sides keep as much room as possible against lighting
        changes. Of all such cuts the one with the WIDEST gap (relative to the
        channel's size) wins - a cut that barely squeezes between them would
        lose the gem as soon as the light shifts. Ties prefer H, S, V in that order.
      * If nothing separates cleanly, the best partial cut is used: the limit goes
        HALFWAY between the old limit and your nearest good click (so half the
        original margin survives instead of hugging your click), and it is only
        used if it still removes at least EXCLUDE_MIN_USEFUL of the blob. The
        result is flagged PARTIAL.

    Returns (new_ranges, message) on success, or (None, message). `ranges` is
    not modified.
    """

    blob = np.asarray(blob, dtype=int)
    kept = np.asarray(kept, dtype=int).reshape(-1, 3)
    new_ranges = [[list(lo), list(hi)] for lo, hi in ranges]
    notes, all_clean, touched, to_drop = [], True, 0, []

    for index, (lower, upper) in enumerate(new_ranges):
        in_blob = blob[_inside(blob, lower, upper)]
        if len(in_blob) == 0:
            continue
        touched += 1
        good = kept[_inside(kept, lower, upper)]

        if len(good) == 0:
            # none of your good clicks live in this piece of the range, so the
            # whole piece is unnecessary (e.g. the far side of red's 0/179 wrap)
            to_drop.append(index)
            notes.append(f"removed range {index + 1} (none of your good clicks are in it)")
            continue

        clean, partial = [], []
        for ch in range(3):
            values = in_blob[:, ch]

            # --- raise the minimum: everything BELOW the new min is cut ---
            blob_edge = int(np.percentile(values, 95))        # top of the blob (ignoring its top 5%)
            kept_edge = int(good[:, ch].min())                # lowest value you want to keep
            if blob_edge + 1 + _MIN_GAP[ch] <= kept_edge:
                t = (blob_edge + 1 + kept_edge) // 2
                if t > lower[ch]:
                    clean.append(((kept_edge - blob_edge) / _SCALE[ch], ch, "min", t,
                                  float(np.mean(values < t)), kept_edge - t))
            elif kept_edge > lower[ch]:
                t = lower[ch] + (kept_edge - lower[ch]) // 2
                if t > lower[ch]:
                    partial.append((float(np.mean(values < t)), ch, "min", t, kept_edge - t))

            # --- lower the maximum: everything ABOVE the new max is cut ---
            blob_edge = int(np.percentile(values, 5))         # bottom of the blob (ignoring its bottom 5%)
            kept_edge = int(good[:, ch].max())                # highest value you want to keep
            if kept_edge + _MIN_GAP[ch] + 1 <= blob_edge:
                t = (kept_edge + blob_edge) // 2
                if t < upper[ch]:
                    clean.append(((blob_edge - kept_edge) / _SCALE[ch], ch, "max", t,
                                  float(np.mean(values > t)), t - kept_edge))
            elif kept_edge < upper[ch]:
                t = kept_edge + (upper[ch] - kept_edge) // 2
                if t < upper[ch]:
                    partial.append((float(np.mean(values > t)), ch, "max", t, t - kept_edge))

        if clean:
            widest = max(c[0] for c in clean)
            gap, ch, side, t, fraction, room = min(
                (c for c in clean if c[0] >= widest - 0.02), key=lambda c: c[1])   # near-ties: H, S, V
            verb = f"raised {_CHANNEL[ch]} min to {t}" if side == "min" else f"lowered {_CHANNEL[ch]} max to {t}"
            where = "above" if side == "min" else "below"     # your clicks sit on the KEPT side of the cut
            note = f"{verb} (removes {100 * fraction:.0f}% of that blob; your clicks keep a margin of {room} {where} it)"
        elif partial and max(p[0] for p in partial) >= EXCLUDE_MIN_USEFUL:
            fraction, ch, side, t, room = max(partial, key=lambda p: (round(p[0], 3), -p[1]))
            verb = f"raised {_CHANNEL[ch]} min to {t}" if side == "min" else f"lowered {_CHANNEL[ch]} max to {t}"
            note = f"{verb} (removes only {100 * fraction:.0f}% - the rest looks like your clicks)"
            all_clean = False
        else:
            best = max((p[0] for p in partial), default=0.0)
            notes.append(f"range {index + 1}: only {100 * best:.0f}% of that blob can be cut without losing a good click")
            all_clean = False
            continue

        if side == "min":
            lower[ch] = t
        else:
            upper[ch] = t
        notes.append(note)

    if touched == 0:
        return None, "that blob is not inside the current range - nothing to cut"

    survivors = [r for i, r in enumerate(new_ranges) if i not in to_drop]
    if not survivors:
        return None, "none of your good clicks are inside the range - click the colour you want first"

    unchanged = not to_drop and all(r == [list(lo), list(hi)] for r, (lo, hi) in zip(new_ranges, ranges))
    if unchanged:
        return None, ("can't exclude it: it looks the same as your good clicks ("
                      + "; ".join(notes) + "). Lower 'Pick tol SV', or click a cleaner good sample")

    message = "Excluded: " + "; ".join(notes) + f" | all {len(kept)} good click{'s' if len(kept) != 1 else ''} kept"
    if not all_clean:
        message += " | PARTIAL - click more of it, or lower 'Pick tol'"
    return survivors, message


def blob_near_click(mask, hsv, fx, fy, radius=14):
    """
    HSV pixels (N x 3) of the detected blob under (fx, fy): the connected piece
    of `mask` containing that point, limited to `radius` px around it so a blob
    that also touches something you DO want doesn't get mixed in. None if the
    click is not on anything currently detected.
    """

    if mask[fy, fx] == 0:
        return None

    _count, labels = cv2.connectedComponents(mask)
    same_blob = labels == labels[fy, fx]

    h, w = mask.shape[:2]
    y0, y1, x0, x1 = max(0, fy - radius), min(h, fy + radius + 1), max(0, fx - radius), min(w, fx + radius + 1)
    window = np.zeros_like(same_blob)
    window[y0:y1, x0:x1] = True

    ys, xs = np.nonzero(same_blob & window)
    if len(ys) > BLOB_MAX_PIXELS:
        keep = np.linspace(0, len(ys) - 1, BLOB_MAX_PIXELS).astype(int)
        ys, xs = ys[keep], xs[keep]
    return hsv[ys, xs].astype(int)


# ------------------------------------------------------------------ #
# Drawing helpers
# ------------------------------------------------------------------ #

def draw_color_buttons(canvas, color_names, working, selected_index, hover_index, toggles):
    """Draw one clickable button per color across the top of `canvas`.
    A disabled color's button is dimmed to near-gray with an "OFF" tag,
    so it's obvious at a glance which colors main.py will actually use."""

    width = canvas.shape[1]
    btn_w = max(1, width // len(color_names))

    for i, name in enumerate(color_names):
        x1 = i * btn_w
        x2 = width if i == len(color_names) - 1 else x1 + btn_w
        box_color = working[name]["box_color"]
        enabled = toggles.get(name, True)

        is_selected = i == selected_index
        is_hover = i == hover_index and not is_selected

        if not enabled:
            # desaturate toward dark gray so a disabled color is visually
            # unmistakable even before reading any text
            fill = tuple(int(c * 0.25 + 40 * 0.75) for c in box_color)
        else:
            fill = box_color
            if is_hover:
                # lighten slightly on hover so it feels responsive
                fill = tuple(min(255, int(c * 1.25) + 15) for c in box_color)

        cv2.rectangle(canvas, (x1, 0), (x2 - 1, BUTTON_ROW_HEIGHT), fill, -1)

        brightness = 0.114 * fill[0] + 0.587 * fill[1] + 0.299 * fill[2]
        text_color = (255, 255, 255) if brightness < 140 else (20, 20, 20)

        label = name if not is_selected else f"* {name}"
        if not enabled:
            label += "  OFF"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        tx = x1 + max(6, (btn_w - tw) // 2)
        ty = BUTTON_ROW_HEIGHT // 2 + th // 2

        cv2.putText(canvas, label, (tx, ty), FONT, 0.55, text_color, 2, cv2.LINE_AA)

        # thin separators between buttons
        if i > 0:
            cv2.line(canvas, (x1, 0), (x1, BUTTON_ROW_HEIGHT), (0, 0, 0), 1)

        if is_selected:
            cv2.rectangle(canvas, (x1 + 2, 2), (x2 - 3, BUTTON_ROW_HEIGHT - 3), (255, 255, 255), 2)

    cv2.line(canvas, (0, BUTTON_ROW_HEIGHT), (width, BUTTON_ROW_HEIGHT), (0, 0, 0), 2)
    return btn_w


def draw_info_bar(canvas, y0, name, range_index, num_ranges, lower, upper, box_color, enabled, hsv_pixel=None):
    """Readout of the currently selected color/range + swatches for lower/upper."""

    width = canvas.shape[1]
    y1 = y0 + INFO_BAR_HEIGHT
    cv2.rectangle(canvas, (0, y0), (width, y1), (48, 48, 48), -1)
    cv2.line(canvas, (0, y1), (width, y1), (0, 0, 0), 2)

    pad = 10
    cy = y0 + INFO_BAR_HEIGHT // 2

    # color name swatch + label
    cv2.rectangle(canvas, (pad, y0 + 10), (pad + 26, y1 - 10), box_color, -1)
    cv2.rectangle(canvas, (pad, y0 + 10), (pad + 26, y1 - 10), (255, 255, 255), 1)

    status = "ENABLED" if enabled else "DISABLED - press x to re-enable"
    status_color = (120, 255, 120) if enabled else (100, 100, 255)
    title = f"{name}  -  range {range_index + 1}/{num_ranges}   [{status}]"
    cv2.putText(canvas, title, (pad + 36, cy + 6), FONT, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    (title_w, _), _ = cv2.getTextSize(title, FONT, 0.6, 2)

    # HSV lower/upper swatches, right-aligned
    def hsv_to_bgr(h, s, v):
        patch = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    swatch_w, swatch_h = 40, 22
    x = width - pad - swatch_w
    y = y0 + (INFO_BAR_HEIGHT - swatch_h) // 2

    cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h), hsv_to_bgr(*upper), -1)
    cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h), (255, 255, 255), 1)
    cv2.putText(canvas, "max", (x, y - 6), FONT, 0.4, MUTED_TEXT, 1, cv2.LINE_AA)

    x -= swatch_w + 14
    cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h), hsv_to_bgr(*lower), -1)
    cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h), (255, 255, 255), 1)
    cv2.putText(canvas, "min", (x, y - 6), FONT, 0.4, MUTED_TEXT, 1, cv2.LINE_AA)

    values_text = f"H {lower[0]:3d}-{upper[0]:3d}   S {lower[1]:3d}-{upper[1]:3d}   V {lower[2]:3d}-{upper[2]:3d}"
    x -= 14
    (tw, th), _ = cv2.getTextSize(values_text, FONT, 0.5, 1)
    cv2.putText(canvas, values_text, (x - tw, cy + 5), FONT, 0.5, MUTED_TEXT, 1, cv2.LINE_AA)


def make_panel(image, label, target_w, is_mask=False, label_color=ACCENT):
    """Resize `image` to target_w (keeping aspect ratio) and add a header + border."""

    h, w = image.shape[:2]
    scale = target_w / float(w)
    target_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    panel = np.zeros((PANEL_HEADER_HEIGHT + target_h, target_w, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)
    panel[PANEL_HEADER_HEIGHT:, :] = resized

    cv2.putText(panel, label, (6, PANEL_HEADER_HEIGHT - 6), FONT, 0.5, label_color, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (target_w - 1, panel.shape[0] - 1), PANEL_BORDER, 1)
    return panel


def draw_footer(canvas, y0, flash_message=None, flash_color=SAVE_FLASH_COLOR):
    width = canvas.shape[1]
    y1 = y0 + FOOTER_HEIGHT
    cv2.rectangle(canvas, (0, y0), (width, y1), (20, 20, 20), -1)

    if flash_message:
        text, color = flash_message, flash_color
    else:
        text = "click/n/p color | [ ] range | +/- range | e eyedropper | m exclude | x on/off | s save | q quit"
        color = MUTED_TEXT

    cv2.putText(canvas, text, (10, y0 + FOOTER_HEIGHT - 9), FONT, 0.48, color, 1, cv2.LINE_AA)


def button_index_for_click(x, y, width, num_colors):
    """Return which color button was clicked/hovered, or None if outside the row."""

    if y < 0 or y > BUTTON_ROW_HEIGHT:
        return None

    btn_w = max(1, width // num_colors)
    idx = x // btn_w
    return min(idx, num_colors - 1)


def find_detections(mask):
    """Contours in the cleaned mask, for drawing outlines on the camera panel."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 40]


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


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    working = build_working_copy()
    color_names = list(working.keys())
    toggles = color_toggles.load_toggles()   # {name: False} for anything turned off

    canvas_width = MARGIN * 2 + PANEL_WIDTH * 3 + PANEL_GAP * 2

    state = {
        "color_index": 0,
        "range_index": 0,
        "hover_index": None,
        "flash_message": None,
        "flash_until": 0.0,
        "frame_count": 0,
        "cached_contours": [],
        # eyedropper
        "eyedropper": False,
        "samples": [],           # (h, s, v) of every click for the selected color
        "sample_points": [],     # (x, y) in camera-frame pixels, for the on-screen markers
        "last_hsv": None,        # the latest frame's HSV image (what clicks sample from)
        "last_mask": None,       # the latest cleaned mask (what "exclude" clicks look at)
        "mode": "add",           # eyedropper click mode: "add" a good colour or "exclude" a wrong one
        "exclusions": [],        # [{"blob": N x 3 HSV pixels, "point": (x, y)}] for the selected colour
        "camera_panel": None,    # (canvas_x, canvas_y, scale) of the camera panel, for click mapping
    }

    # recomputing contours every single frame is the priciest cosmetic step;
    # only refresh them every DETECTION_EVERY frames to keep the UI snappy
    DETECTION_EVERY = 2

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    def flash(text, seconds=1.5):
        state["flash_message"] = text
        state["flash_until"] = time.time() + seconds

    def reset_samples():
        state["samples"] = []
        state["sample_points"] = []
        state["exclusions"] = []
        state["mode"] = "add"

    def apply_samples():
        """Recompute the selected color's ranges from ALL clicked samples."""
        name = color_names[state["color_index"]]

        if not state["samples"]:
            return

        tol_h = cv2.getTrackbarPos("Pick tol H", WINDOW_NAME)
        tol_sv = cv2.getTrackbarPos("Pick tol SV", WINDOW_NAME)

        new_ranges = ranges_from_samples(state["samples"], tol_h, tol_sv)

        # clicks recompute the range from scratch, so put every exclusion back
        lost = 0
        for exclusion in state["exclusions"]:
            cut, _message = cut_blob_out(new_ranges, exclusion["blob"], state["samples"])
            if cut is None:
                lost += 1
            else:
                new_ranges = cut

        working[name]["ranges"] = new_ranges
        state["range_index"] = 0
        push_sliders_from_state()

        n = len(state["samples"])
        extra = " (wraps 0/179 - 2 ranges)" if len(working[name]["ranges"]) > 1 else ""
        excl = f", {len(state['exclusions']) - lost} exclusion(s) re-applied" if state["exclusions"] else ""
        warn = f" - {lost} exclusion(s) no longer possible" if lost else ""
        flash(f"Eyedropper: '{name}' set from {n} click{'s' if n != 1 else ''}{extra}{excl}{warn}")

    def eyedropper_click(x, y):
        hsv = state["last_hsv"]
        layout = state["camera_panel"]
        if hsv is None or layout is None:
            return

        x0, y0, scale = layout
        fx = int((x - x0) / scale)
        fy = int((y - y0) / scale)

        h_img, w_img = hsv.shape[:2]
        if not (0 <= fx < w_img and 0 <= fy < h_img):
            return   # clicked outside the camera picture

        sample = sample_hsv(hsv, fx, fy)
        if sample is None:
            return

        state["samples"].append(sample)
        state["sample_points"].append((fx, fy))
        apply_samples()

    def exclude_click(x, y):
        """Click on something detected that should NOT be: narrow the range so it drops out."""
        hsv = state["last_hsv"]
        mask = state["last_mask"]
        layout = state["camera_panel"]
        if hsv is None or mask is None or layout is None:
            return

        if not state["samples"]:
            flash("Exclude needs a good click first: press m, click the colour you WANT, then m again")
            return

        x0, y0, scale = layout
        fx = int((x - x0) / scale)
        fy = int((y - y0) / scale)
        h_img, w_img = hsv.shape[:2]
        if not (0 <= fx < w_img and 0 <= fy < h_img):
            return

        blob = blob_near_click(mask, hsv, fx, fy)
        if blob is None:
            flash("That spot is not being detected right now - nothing to exclude", 2.5)
            return

        name = color_names[state["color_index"]]
        new_ranges, message = cut_blob_out(working[name]["ranges"], blob, state["samples"])
        if new_ranges is None:
            flash(message[0].upper() + message[1:], 4.0)
            print(f"[hsv_tuner] {name}: {message}")
            return

        working[name]["ranges"] = new_ranges
        state["range_index"] = min(state["range_index"], len(new_ranges) - 1)
        push_sliders_from_state()
        state["exclusions"].append({"blob": blob, "point": (fx, fy)})
        flash(message, 4.0)
        print(f"[hsv_tuner] {name}: {message}")

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            state["hover_index"] = button_index_for_click(x, y, canvas_width, len(color_names))
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        idx = button_index_for_click(x, y, canvas_width, len(color_names))
        if idx is not None:
            if idx != state["color_index"]:
                state["color_index"] = idx
                state["range_index"] = 0
                reset_samples()
                push_sliders_from_state()
            return

        if state["eyedropper"]:
            if state["mode"] == "exclude":
                exclude_click(x, y)
            else:
                eyedropper_click(x, y)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # sliders grouped as min/max pairs per channel, easier to reason about
    cv2.createTrackbar("H min", WINDOW_NAME, 0, 179, _nothing)
    cv2.createTrackbar("H max", WINDOW_NAME, 179, 179, _nothing)
    cv2.createTrackbar("S min", WINDOW_NAME, 0, 255, _nothing)
    cv2.createTrackbar("S max", WINDOW_NAME, 255, 255, _nothing)
    cv2.createTrackbar("V min", WINDOW_NAME, 0, 255, _nothing)
    cv2.createTrackbar("V max", WINDOW_NAME, 255, 255, _nothing)

    # eyedropper tolerance: how much room to add around the clicked color
    cv2.createTrackbar("Pick tol H", WINDOW_NAME, DEFAULT_TOL_H, 40, _nothing)
    cv2.createTrackbar("Pick tol SV", WINDOW_NAME, DEFAULT_TOL_SV, 120, _nothing)

    def push_sliders_from_state():
        name = color_names[state["color_index"]]
        ranges = working[name]["ranges"]
        state["range_index"] = min(state["range_index"], len(ranges) - 1)
        lower, upper = ranges[state["range_index"]]

        cv2.setTrackbarPos("H min", WINDOW_NAME, lower[0])
        cv2.setTrackbarPos("H max", WINDOW_NAME, upper[0])
        cv2.setTrackbarPos("S min", WINDOW_NAME, lower[1])
        cv2.setTrackbarPos("S max", WINDOW_NAME, upper[1])
        cv2.setTrackbarPos("V min", WINDOW_NAME, lower[2])
        cv2.setTrackbarPos("V max", WINDOW_NAME, upper[2])

    push_sliders_from_state()

    print(__doc__)

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera")
            break

        name = color_names[state["color_index"]]
        ranges = working[name]["ranges"]
        box_color = working[name]["box_color"]

        # ---- read current slider values back into the working copy ----
        h_min = cv2.getTrackbarPos("H min", WINDOW_NAME)
        h_max = cv2.getTrackbarPos("H max", WINDOW_NAME)
        s_min = cv2.getTrackbarPos("S min", WINDOW_NAME)
        s_max = cv2.getTrackbarPos("S max", WINDOW_NAME)
        v_min = cv2.getTrackbarPos("V min", WINDOW_NAME)
        v_max = cv2.getTrackbarPos("V max", WINDOW_NAME)

        lower_hsv = [h_min, s_min, v_min]
        upper_hsv = [h_max, s_max, v_max]
        ranges[state["range_index"]] = [lower_hsv, upper_hsv]

        # ---- build mask using the SAME pipeline main.py uses ----
        hsv = vision.to_hsv(frame)
        state["last_hsv"] = hsv   # eyedropper clicks sample from this

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = vision.clean_mask(mask)

        state["last_mask"] = mask   # exclude clicks look at what is detected right now

        result = cv2.bitwise_and(frame, frame, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # outline detected blobs directly on the camera feed for a quick sanity check.
        # contour search is the priciest cosmetic step, so only refresh it every
        # few frames - the sliders/mask themselves still update every frame.
        state["frame_count"] += 1
        if state["frame_count"] % DETECTION_EVERY == 0 or not state["cached_contours"]:
            state["cached_contours"] = find_detections(mask)

        camera_view = frame.copy()
        for contour in state["cached_contours"]:
            cv2.drawContours(camera_view, [contour], -1, box_color, 2)

        # markers where the eyedropper was clicked
        for i, (px, py) in enumerate(state["sample_points"], start=1):
            cv2.circle(camera_view, (px, py), SAMPLE_RADIUS + 4, (255, 255, 255), 2)
            cv2.circle(camera_view, (px, py), SAMPLE_RADIUS + 4, (0, 0, 0), 1)
            cv2.putText(camera_view, str(i), (px + 10, py - 8), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # red crosses where something was excluded
        for (px, py) in [e["point"] for e in state["exclusions"]]:
            cv2.line(camera_view, (px - 9, py - 9), (px + 9, py + 9), (0, 0, 255), 3)
            cv2.line(camera_view, (px - 9, py + 9), (px + 9, py - 9), (0, 0, 255), 3)

        # ---- assemble the single combined canvas ----
        top_h = BUTTON_ROW_HEIGHT + INFO_BAR_HEIGHT

        if state["eyedropper"] and state["mode"] == "exclude":
            camera_label, camera_label_color = "CAMERA - EXCLUDE: click the wrong part", (0, 0, 255)
        elif state["eyedropper"]:
            camera_label, camera_label_color = "CAMERA - EYEDROPPER ON: click the object", (0, 255, 0)
        else:
            camera_label, camera_label_color = "CAMERA + DETECTIONS", ACCENT

        panels = [
            make_panel(camera_view, camera_label, PANEL_WIDTH, label_color=camera_label_color),
            make_panel(mask_bgr, "MASK", PANEL_WIDTH),
            make_panel(result, "RESULT", PANEL_WIDTH),
        ]
        panel_h = max(p.shape[0] for p in panels)
        panels = [
            p if p.shape[0] == panel_h
            else cv2.copyMakeBorder(p, 0, panel_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            for p in panels
        ]

        # remember where the camera picture sits on the canvas, so a mouse
        # click can be mapped back to a pixel of the real camera frame
        state["camera_panel"] = (
            MARGIN,
            top_h + MARGIN + PANEL_HEADER_HEIGHT,
            PANEL_WIDTH / float(frame.shape[1]),
        )

        canvas_h = MARGIN * 2 + top_h + panel_h + FOOTER_HEIGHT
        canvas = np.zeros((canvas_h, canvas_width, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR

        draw_color_buttons(canvas, color_names, working, state["color_index"], state["hover_index"], toggles)
        draw_info_bar(canvas, BUTTON_ROW_HEIGHT, name, state["range_index"], len(ranges),
                      lower_hsv, upper_hsv, box_color, toggles.get(name, True))

        x = MARGIN
        y = top_h + MARGIN
        for p in panels:
            canvas[y:y + p.shape[0], x:x + p.shape[1]] = p
            x += p.shape[1] + PANEL_GAP

        footer_message, footer_color = None, SAVE_FLASH_COLOR
        if state["flash_message"] and time.time() < state["flash_until"]:
            footer_message = state["flash_message"]
        elif state["eyedropper"] and state["mode"] == "exclude":
            footer_message = (
                f"EXCLUDE: click something wrongly detected ({len(state['exclusions'])} done) "
                f"| u undo | r restart | m back to adding"
            )
            footer_color = (90, 90, 255)
        elif state["eyedropper"]:
            n = len(state["samples"])
            footer_message = (
                f"EYEDROPPER: click '{name}' on the CAMERA panel ({n} click{'s' if n != 1 else ''}) "
                f"- more clicks widen | m exclude | u undo | r restart | e off"
            )
            footer_color = ACCENT
        draw_footer(canvas, canvas_h - FOOTER_HEIGHT, flash_message=footer_message, flash_color=footer_color)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('n'):
            state["color_index"] = (state["color_index"] + 1) % len(color_names)
            state["range_index"] = 0
            reset_samples()
            push_sliders_from_state()

        elif key == ord('p'):
            state["color_index"] = (state["color_index"] - 1) % len(color_names)
            state["range_index"] = 0
            reset_samples()
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
            flash(f"Added range #{len(ranges)} for '{name}'")
            print(f"[hsv_tuner] {state['flash_message']}")

        elif key == ord('-'):
            if len(ranges) > 1:
                ranges.pop(state["range_index"])
                state["range_index"] = max(0, state["range_index"] - 1)
                push_sliders_from_state()
                flash(f"Removed a range for '{name}', {len(ranges)} left")
                print(f"[hsv_tuner] {state['flash_message']}")
            else:
                flash(f"'{name}' must keep at least 1 range")
                print(f"[hsv_tuner] {state['flash_message']}")

        elif key == ord('e'):
            state["eyedropper"] = not state["eyedropper"]
            if state["eyedropper"]:
                reset_samples()   # a fresh session starts from the next click
            flash("Eyedropper ON - click the object in the CAMERA panel"
                  if state["eyedropper"] else "Eyedropper OFF")

        elif key == ord('m'):
            if state["mode"] == "add":
                if not state["eyedropper"]:
                    state["eyedropper"] = True
                    reset_samples()
                state["mode"] = "exclude"
                flash("EXCLUDE mode: click something detected that should NOT be (press m to go back)"
                      if state["samples"] else
                      "EXCLUDE mode - but first click the colour you WANT (press m, click it, m again)", 3.0)
            else:
                state["mode"] = "add"
                flash("Back to ADD mode - click the colour you want")

        elif key == ord('u') and state["mode"] == "exclude":
            if state["exclusions"]:
                state["exclusions"].pop()
                apply_samples()
                flash(f"Last exclusion undone ({len(state['exclusions'])} left)")
            else:
                flash("No exclusion to undo")

        elif key == ord('u'):
            if state["samples"]:
                state["samples"].pop()
                state["sample_points"].pop()
                if state["samples"]:
                    apply_samples()
                else:
                    flash("Eyedropper: all clicks undone (range unchanged)")
            else:
                flash("Eyedropper: nothing to undo")

        elif key == ord('r'):
            reset_samples()
            flash("Eyedropper: clicks and exclusions cleared - next click starts fresh")

        elif key == ord('s'):
            save_overrides(working)
            flash(f"Saved to {os.path.basename(OVERRIDE_PATH)}")

        elif key == ord('x'):
            new_state = color_toggles.toggle(name)
            toggles = color_toggles.load_toggles()
            flash(f"'{name}' is now {'ENABLED' if new_state else 'DISABLED'} for detection")
            print(f"[hsv_tuner] {name} -> {'enabled' if new_state else 'disabled'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()