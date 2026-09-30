"""
motion.py - decides HOW the robot moves toward the target navigation.py chose.

It is the ONLY place that decides movement timing. The idea is "move, stop,
look again":

    far away, lined up    -> FORWARD continuously, stop if it drifts off line
    not lined up          -> one turn PULSE sized to the angle error, then stop
    close to the target   -> short forward HOPs sized to the distance, then stop
    after every pulse/hop -> STOP and SETTLE (wait for the camera and Wi-Fi
                             to catch up) before measuring and deciding again

Why this instead of turning continuously until the angle is small: the
camera and Wi-Fi add delay, so a continuous turn always overshoots, turns
back, overshoots again. Sizing each pulse from a measured angle and then
looking again converges in a few steps with no overshoot loop. It also makes
"stop before every action" part of the design instead of a patch on top.

Self-calibration: nobody knows exactly how many degrees the robot turns per
second at your Turn Speed / Turn Duty, or how many pixels per second it
drives at your drive speed - and both change with battery level. After every
pulse/hop, the controller measures what really happened and updates its
estimate, so later pulses get the size right. It starts by assuming the
robot is FAST, so the first pulses are short and safe, and adapts from there.

Stuck handling (replaces the old 3-second timer that also fired during
normal slow turns): the robot counts as stuck only when several pulses in a
row make NO progress - the angle doesn't improve (e.g. a gem is being
dragged along by the claw) or hops don't get it closer (e.g. a gem is being
pushed ahead). Then it either grabs/places anyway if it's close enough, or
gives up on that target so navigation picks another.

The Arduino firmware is unchanged - this only uses the existing
FORWARD / BACKWARD / LEFT / RIGHT / STOP commands.
"""

import math

import config

# ---- timing -------------------------------------------------------------
SETTLE_SEC = 0.30          # stop this long after each pulse/hop before looking again

# ---- turning --------------------------------------------------------------
KEEP_DRIVING_FACTOR = 1.5  # while driving, only stop to re-aim past TURN_ANGLE_THRESHOLD x this
TURN_GAIN = 0.8            # aim to remove 80% of the angle error per pulse (never overshoot)
TURN_MIN_SEC = 0.08
TURN_MAX_SEC = 0.45
TURN_RATE_START = 240.0    # deg/s assumed at start - deliberately FAST so first pulses are short
TURN_RATE_LIMITS = (15.0, 900.0)

# ---- driving --------------------------------------------------------------
SLOW_ZONE_FACTOR = 3.0     # inside PICKUP_DISTANCE x this, switch from driving to hops
HOP_GAIN = 0.7             # each hop covers 70% of the remaining distance
HOP_MIN_SEC = 0.08
HOP_MAX_SEC = 0.35
DRIVE_RATE_START = 400.0   # px/s assumed at start - deliberately FAST so first hops are short
DRIVE_RATE_LIMITS = (20.0, 2000.0)

# ---- stuck / arrival ------------------------------------------------------
STUCK_TURN_TRIES = 6       # this many turn pulses in a row without the angle improving
STUCK_HOP_TRIES = 6        # this many hops in a row without getting closer
ANGLE_PROGRESS_DEG = 3.0
DIST_PROGRESS_PX = 3.0
OVERSHOOT_PX = 10.0        # final approach: grip spot passed the target by this much
FORCE_ARRIVE_FACTOR = 1.6  # stuck/passed but within PICKUP_DISTANCE x this -> grab/place anyway
MAX_MISSES = 2             # passed beside the target this many times -> give up on it

# Close to the target, "lined up" is judged by how far SIDEWAYS the claw's
# straight path would miss the target, not by angle: 15 degrees is fine from
# far away (it re-aims later) but at close range it passes beside the gem.
# The claw path may miss by at most PICKUP_DISTANCE x this.
LATERAL_TOLERANCE_FACTOR = 0.5

LEARN_RATE = 0.5           # how fast the rate estimates follow new measurements


def _wrap(deg):
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class MotionController:

    def __init__(self):
        self.turn_rate = TURN_RATE_START
        self.drive_rate = DRIVE_RATE_START

        self.state = "IDLE"       # IDLE, TURN, DRIVE, HOP, BACKUP, WAIT, SETTLE, ARRIVED
        self.command = "STOP"
        self.action_end = 0.0
        self.settle_end = 0.0
        self.action = None        # what was just done, for calibration after settling
        self.pending_backup = None
        self.note = ""

        self.target_id = None
        self.action_target = None   # target a continuous DRIVE was started for
        self.min_distance_seen = 0.0
        self._reset_target_progress()

    # ------------------------------------------------------------------
    def _reset_target_progress(self, keep_misses=False):
        self.best_angle = None
        self.turn_fails = 0
        self.final_approach = False
        self.min_distance = None
        self.hop_fails = 0
        if not keep_misses:
            self.misses = 0

    def reset_target(self):
        """Call after a failed grab etc. - forget progress toward the target."""
        self._reset_target_progress()
        if self.state == "ARRIVED":
            self.state = "IDLE"

    def request_backup(self, now, seconds, delay=0.0):
        """Reverse for `seconds`, optionally after standing still for `delay`
        (e.g. to let the claw finish opening before backing away)."""
        self._reset_target_progress()
        if delay > 0:
            self.state = "WAIT"
            self.action_end = now + delay
            self.pending_backup = seconds
        else:
            self._start("BACKUP", "BACKWARD", now, seconds, None)

    # ------------------------------------------------------------------
    def _start(self, state, command, now, duration, pose):
        self.state = state
        self.command = command
        self.action_end = now + duration
        self.action = {"kind": state, "start": now, "duration": duration, "pose": pose}

    def _begin_settle(self, now):
        if self.action is not None and self.state == "DRIVE":
            self.action["duration"] = now - self.action["start"]
        self.state = "SETTLE"
        self.command = "STOP"
        self.settle_end = now + SETTLE_SEC

    def _learn(self, pose):
        """After settling, compare where we are with where the action started."""
        a, self.action = self.action, None
        if a is None or pose is None or a["pose"] is None or a["duration"] < 0.05:
            return
        if a["kind"] == "TURN":
            turned = abs(_wrap(pose[2] - a["pose"][2]))
            observed = max(turned, 0.5) / a["duration"]
            self.turn_rate = _clamp((1 - LEARN_RATE) * self.turn_rate + LEARN_RATE * observed,
                                    *TURN_RATE_LIMITS)
        elif a["kind"] in ("DRIVE", "HOP"):
            moved = math.hypot(pose[0] - a["pose"][0], pose[1] - a["pose"][1])
            observed = max(moved, 0.5) / a["duration"]
            self.drive_rate = _clamp((1 - LEARN_RATE) * self.drive_rate + LEARN_RATE * observed,
                                     *DRIVE_RATE_LIMITS)

    def _stuck(self, distance, why):
        self.state = "ARRIVED" if distance <= config.PICKUP_DISTANCE * FORCE_ARRIVE_FACTOR else "IDLE"
        self.command = "STOP"
        if self.state == "ARRIVED":
            self.note = f"stuck ({why}) but close - taking it here"
            return self._out(arrived=True)
        self.note = f"stuck ({why}) - giving up on this target"
        self._reset_target_progress()
        return self._out(give_up=True)

    def _out(self, arrived=False, give_up=False):
        return {"command": self.command, "state": self.state,
                "arrived": arrived, "give_up": give_up, "note": self.note}

    # ------------------------------------------------------------------
    def update(self, now, pose, nav):
        """
        pose: (x, y, heading_deg) of the robot center, or None if not seen.
              heading must include the grip slant (see navigation.py).
        nav:  the dict from navigation.compute_navigation().
        Returns {"command", "state", "arrived", "give_up", "note"}.
        """

        has_target = pose is not None and nav["nav_target_point"] is not None
        if has_target and nav["target_id"] != self.target_id:
            self.target_id = nav["target_id"]
            self._reset_target_progress()
            if self.state == "ARRIVED":
                self.state = "IDLE"

        # ---- finish whatever is in progress ------------------------------
        if self.state == "WAIT":
            if now < self.action_end:
                return self._out()
            seconds, self.pending_backup = self.pending_backup, None
            self._start("BACKUP", "BACKWARD", now, seconds, None)

        if self.state in ("TURN", "HOP", "BACKUP"):
            if now < self.action_end:
                return self._out()
            self._begin_settle(now)

        if self.state == "DRIVE":
            keep = config.TURN_ANGLE_THRESHOLD * KEEP_DRIVING_FACTOR
            slow_zone = config.PICKUP_DISTANCE * SLOW_ZONE_FACTOR
            if (has_target and nav["target_id"] == self.action_target
                    and abs(nav["angle_diff"]) <= keep and nav["gripper_distance"] > slow_zone):
                return self._out()
            self._begin_settle(now)

        if self.state == "SETTLE":
            if now < self.settle_end:
                return self._out()
            self._learn(pose)
            self.state = "IDLE"

        if self.state == "ARRIVED":
            return self._out(arrived=True)

        # ---- decide the next action ---------------------------------------
        self.command = "STOP"
        if not has_target:
            self.state = "IDLE"
            self.note = "no target" if pose is not None else "robot not detected"
            return self._out()

        angle = nav["angle_diff"]
        distance = nav["gripper_distance"]

        if distance <= config.PICKUP_DISTANCE:
            self.state = "ARRIVED"
            self.note = "arrived"
            return self._out(arrived=True)

        if self.final_approach and self.min_distance is not None \
                and distance > self.min_distance + OVERSHOOT_PX:
            if self.min_distance <= config.PICKUP_DISTANCE * FORCE_ARRIVE_FACTOR:
                self.state = "ARRIVED"
                self.note = "passed just by it - taking it here"
                return self._out(arrived=True)
            # Went past it without the claw getting close: back up and retry.
            self.misses += 1
            if self.misses > MAX_MISSES:
                return self._stuck(distance, "keeps passing beside it")
            back = _clamp((distance + config.PICKUP_DISTANCE) / self.drive_rate, 0.15, 0.6)
            self._reset_target_progress(keep_misses=True)
            self._start("BACKUP", "BACKWARD", now, back, None)
            self.note = f"missed it (closest {self.min_distance_seen:.0f}px) - backing up to retry"
            return self._out()

        far = distance > config.PICKUP_DISTANCE * SLOW_ZONE_FACTOR
        if far:
            needs_turn = abs(angle) > config.TURN_ANGLE_THRESHOLD
        else:
            center_distance = nav.get("center_distance") or distance
            sideways_miss = center_distance * math.sin(math.radians(min(abs(angle), 90.0)))
            needs_turn = sideways_miss > config.PICKUP_DISTANCE * LATERAL_TOLERANCE_FACTOR
        # Only turn if even the shortest possible pulse would leave a smaller
        # error than now - otherwise it would just swing to the other side.
        can_improve = self.turn_rate * TURN_MIN_SEC < 1.8 * abs(angle)

        # ---- turn pulse (not during the final approach: once lined up and
        # close, a gem nudged sideways by the claw must not start a chase) --
        if not self.final_approach and needs_turn and can_improve:
            if self.best_angle is None or abs(angle) < self.best_angle - ANGLE_PROGRESS_DEG:
                self.best_angle = abs(angle)
                self.turn_fails = 0
            else:
                self.turn_fails += 1
                if self.turn_fails >= STUCK_TURN_TRIES:
                    return self._stuck(distance, "turning does not reduce the angle")

            duration = _clamp(TURN_GAIN * abs(angle) / self.turn_rate, TURN_MIN_SEC, TURN_MAX_SEC)
            direction = "RIGHT" if angle > 0 else "LEFT"
            self._start("TURN", direction, now, duration, pose)
            self.note = f"turn {direction} {duration:.2f}s for {angle:+.0f} deg"
            return self._out()

        self.best_angle = None
        self.turn_fails = 0

        # ---- lined up and far: drive continuously -------------------------
        if not self.final_approach and far:
            self._start("DRIVE", "FORWARD", now, 0.0, pose)
            self.action_target = nav["target_id"]
            self.note = f"drive, {distance:.0f}px to go"
            return self._out()

        # ---- lined up and close: short hops, heading locked ----------------
        if self.final_approach:
            if distance < self.min_distance - DIST_PROGRESS_PX:
                self.hop_fails = 0
            else:
                self.hop_fails += 1
                if self.hop_fails >= STUCK_HOP_TRIES:
                    return self._stuck(distance, "hops do not get closer")
        self.final_approach = True
        self.min_distance = distance if self.min_distance is None else min(self.min_distance, distance)
        self.min_distance_seen = self.min_distance

        remaining = distance - config.PICKUP_DISTANCE * 0.5
        duration = _clamp(HOP_GAIN * remaining / self.drive_rate, HOP_MIN_SEC, HOP_MAX_SEC)
        self._start("HOP", "FORWARD", now, duration, pose)
        self.note = f"hop {duration:.2f}s, {distance:.0f}px to go"
        return self._out()