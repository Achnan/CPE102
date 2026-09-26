"""
esp32_link.py

Sends movement/gripper commands to the ESP32 over Wi-Fi, matching the
ESP32 firmware exactly:

    POST http://<ESP32_IP>/command
    Body (JSON): {"type": "COMMAND", "action": "FORWARD"}

The ESP32 only executes — it has no idea what "gem" or "target circle"
means. Every action this module can send is one the firmware already
understands: FORWARD, BACKWARD, LEFT, RIGHT, STOP, GRAB, RELEASE.

Important: the ESP32 stops itself automatically if it doesn't receive
ANY command for 1.5 seconds (its own COMMAND_TIMEOUT_MS safety net).
That means movement commands must be resent regularly while the robot
should keep moving — this module does NOT do that on its own; main.py
must call send_command() every frame (or on some steady timer) for as
long as it wants the robot moving. This module only throttles how
OFTEN repeats actually go out over the network (see
MIN_COMMAND_INTERVAL_SECONDS in config.py) — it does not decide when
movement should start or stop.
"""

import time

import requests

import config


_VALID_ACTIONS = {
    "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "GRAB", "RELEASE"
}

_last_action_sent = None
_last_send_time = 0.0


def _build_url():
    return f"http://{config.ESP32_IP}:{config.ESP32_PORT}{config.ESP32_COMMAND_PATH}"


def send_command(action, force=False):
    """
    Send one action to the ESP32.

    action : one of FORWARD/BACKWARD/LEFT/RIGHT/STOP/GRAB/RELEASE
    force  : bypass the MIN_COMMAND_INTERVAL_SECONDS throttle — use
             this for GRAB/RELEASE, which should fire the moment
             they're decided, not wait for the next throttle window.

    Returns True if the ESP32 responded 200 OK, False otherwise
    (including on connection errors / timeouts — these are caught
    here so a dropped Wi-Fi packet never crashes the vision loop).
    """

    global _last_action_sent, _last_send_time

    action = action.upper()

    if action not in _VALID_ACTIONS:
        print(f"[esp32_link] Refusing to send unknown action: {action}")
        return False

    now = time.time()

    if not force and action == _last_action_sent:
        if now - _last_send_time < config.MIN_COMMAND_INTERVAL_SECONDS:
            return True  # too soon to repeat the same command - not an error

    try:
        response = requests.post(
            _build_url(),
            json={"type": "COMMAND", "action": action},
            timeout=config.ESP32_REQUEST_TIMEOUT_SECONDS,
        )

        _last_action_sent = action
        _last_send_time = now

        if response.status_code != 200:
            print(
                f"[esp32_link] ESP32 returned status "
                f"{response.status_code} for action {action}: "
                f"{response.text}"
            )
            return False

        return True

    except requests.exceptions.RequestException as e:
        # Covers connection refused, timeout, host unreachable, etc.
        # Printed but not raised — a single dropped command shouldn't
        # stop the vision loop from continuing to try next frame.
        print(f"[esp32_link] Failed to send {action}: {e}")
        return False