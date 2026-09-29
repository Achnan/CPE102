"""
color_toggles.py
================
Lets you turn individual colors ON/OFF for detection, without deleting
their HSV ranges or touching hsv_overrides.json. A color you disable
here is completely skipped by vision.py (both target-circle and loose-
gem detection, and the held-color check) - as if it weren't in
config.COLORS at all - until you turn it back on.

This is separate from hsv_overrides.json on purpose: that file holds
WHAT a color looks like (its HSV ranges); this file holds WHETHER a
color is currently allowed to be detected at all. Keeping them apart
means toggling a color off and on never touches its tuned ranges.

Stored in color_toggles.json, next to this file, as:
    {"red": true, "blue": false, ...}
A color not listed here is treated as enabled (the default), so this
file only needs to record colors you've actually turned OFF.

Used by:
    - hsv_tuner.py   - the 'x' key toggles the selected color and saves
    - vision.py      - filters config.COLORS down to only enabled ones
                       before every detection pass
"""

import json
import os

TOGGLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_toggles.json")


def load_toggles():
    """{color_name: True/False} for every color that has been explicitly
    set. A color missing from this dict should be treated as enabled."""

    if not os.path.exists(TOGGLES_PATH):
        return {}
    try:
        with open(TOGGLES_PATH, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {name: bool(value) for name, value in data.items()}


def save_toggles(toggles):
    with open(TOGGLES_PATH, "w") as f:
        json.dump(toggles, f, indent=2)


def is_enabled(color_name, toggles=None):
    if toggles is None:
        toggles = load_toggles()
    return toggles.get(color_name, True)   # default: enabled


def set_enabled(color_name, enabled):
    """Turn one color on/off and save immediately. Returns the new toggles dict."""
    toggles = load_toggles()
    if enabled:
        # Leave it out of the file entirely when re-enabled, so the file
        # only ever lists colors that are actually turned off.
        toggles.pop(color_name, None)
    else:
        toggles[color_name] = False
    save_toggles(toggles)
    return toggles


def toggle(color_name):
    """Flip one color's enabled state and save. Returns the new state (True/False)."""
    toggles = load_toggles()
    new_state = not is_enabled(color_name, toggles)
    set_enabled(color_name, new_state)
    return new_state


def filter_enabled_colors(colors_dict, toggles=None):
    """
    Given a dict shaped like config.COLORS ({name: (ranges, box_color)}),
    return a new dict with disabled colors removed. Call this fresh each
    time you need it (it's cheap - a small dict comprehension) rather
    than caching, so a toggle made in the tuner while main.py is running
    takes effect the moment main.py reloads it (see vision.py).
    """

    if toggles is None:
        toggles = load_toggles()
    return {
        name: value for name, value in colors_dict.items()
        if toggles.get(name, True)
    }