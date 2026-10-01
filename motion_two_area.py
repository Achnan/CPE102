"""
motion_two_area.py (TWO-area version of motion.py) - decides HOW the robot moves toward the target navigation.py chose.

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
looking again converges in a few steps with no overshoot loop.

TWO GRIPPER AREAS (set in grip_tuner.py)
    TIP area  - the claw's front end. The robot drives until a gem is a
                little outside the tip circle (far enough to stop in time),
                then STOPS, opens the claw, and WAITS for the servo to finish
                opening. Only then does it move in, and once the gem is
                inside the tip circle it only creeps. So the claw is open
                before anything touches it - the tips used to reach the gem
                while the claw was still closed and shove it.
    HOLD area - where the gem must be when the claw closes. Arriving means
                the gem is within PICKUP_DISTANCE of the hold spot; main.py
                then grabs and checks the gem really is inside the hold circle.

Pushing: if a creep makes the gem slide forward together with the robot, the
tip is shoving it. The robot backs up (with the claw still open) and tries
again; after MAX_PUSHES it gives up on that gem.

Self-calibration: after every pulse/hop the controller measures what really
happened (degrees turned, pixels driven) and updates its estimate, so later
moves get the size right. It starts by assuming the robot is FAST, so the
first moves are short and safe.

Stuck handling is based on progress, not time: several pulses in a row that
don't improve the angle, or hops that don't get closer, mean stuck. Then it
takes the gem if it is already close, or gives that target up.

The Arduino firmware is unchanged - this only uses the existing
FORWARD / BACKWARD / LEFT / RIGHT / STOP commands (and asks main.py to send
RELEASE to open the claw).
"""

import math

import border_guard
import config

# ---- timing -------------------------------------------------------------
SETTLE_SEC = 0.30          # stop this long after each pulse/hop before looking again

# ---- the claw ---------------------------------------------------------------
OPEN_WAIT_SEC = 0.5        # how long the servo needs to open the claw - raise if it is slow
STOP_LATENCY_SEC = 0.25    # camera + Wi-Fi delay: stop driving this much travel BEFORE the tip circle
STOP_MARGIN_LIMITS = (10.0, 70.0)   # px, clamp for that stopping distance
CREEP_PX = 15.0            # while a gem is inside the tip circle, each hop aims for at most this far

# ---- turning --------------------------------------------------------------
KEEP_DRIVING_FACTOR = 1.5  # while driving, only stop to re-aim past TURN_ANGLE_THRESHOLD x this
TURN_GAIN = 0.8            # aim to remove 80% of the angle error per pulse (never overshoot)
TURN_MIN_SEC = 0.08
TURN_MAX_SEC = 0.45
TURN_RATE_START = 240.0    # deg/s assumed at start - deliberately FAST so first pulses are short
TURN_RATE_LIMITS = (15.0, 900.0)

# ---- driving --------------------------------------------------------------
SLOW_ZONE_FACTOR = 3.0     # bases: inside PICKUP_DISTANCE x this, switch from driving to hops
HOP_GAIN = 0.7             # each hop covers 70% of the remaining distance
HOP_MIN_SEC = 0.08
HOP_MAX_SEC = 0.35
DRIVE_RATE_START = 400.0   # px/s assumed at start - deliberately FAST so first hops are short
DRIVE_RATE_LIMITS = (20.0, 2000.0)

# ---- stuck / arrival / pushing --------------------------------------------
STUCK_TURN_TRIES = 6       # this many turn pulses in a row without the angle improving
STUCK_HOP_TRIES = 6        # this many hops in a row without getting closer
ANGLE_PROGRESS_DEG = 3.0
DIST_PROGRESS_PX = 3.0
OVERSHOOT_PX = 10.0        # final approach: hold spot passed the target by this much
FORCE_ARRIVE_FACTOR = 1.6  # stuck/passed but within PICKUP_DISTANCE x this -> grab/place anyway
MAX_MISSES = 2             # passed beside the target this many times -> give up on it
PUSH_MOVE_PX = 8.0         # the gem slid at least this far forward during one hop = being pushed
MAX_PUSHES = 2             # pushed it this many times -> give up on it

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


class MotionController(border_guard.BorderMixin):

    def __init__(self):
        self.turn_rate = TURN_RATE_START
        self.drive_rate = DRIVE_RATE_START

        self.state = "IDLE"   # IDLE, TURN, DRIVE, HOP, BACKUP, WAIT, OPENING, SETTLE, ARRIVED
        self.command = "STOP"
        self.action_end = 0.0
        self.settle_end = 0.0
        self.action = None        # what was just done, for calibration after settling
        self.pending_backup = None
        self.note = ""

        self.target_id = None
        self.action_target = None   # target a continuous DRIVE was started for
        self.min_distance_seen = 0.0
        self._border_setup()        # lost-marker recovery + last-move memory (border_guard.py)
        self.claw_ready = True      # refreshed every update()
        self._reset_target_progress()

    # ------------------------------------------------------------------
    def _reset_target_progress(self, keep_misses=False, keep_pushes=False):
        self.best_angle = None
        self.turn_fails = 0
        self.final_approach = False
        self.min_distance = None
        self.hop_fails = 0
        if not keep_misses:
            self.misses = 0
        if not keep_pushes:
            self.pushes = 0

    def reset_target(self):
        """Call after a failed grab etc. - forget progress toward the target."""
        self._reset_target_progress()
        if self.state in ("ARRIVED", "OPENING"):
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
        # long enough for the robot to finish moving AND for that to show up in the (delayed) camera picture:
        self.settle_end = now + max(SETTLE_SEC, self._drive_coast)

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

    def _out(self, arrived=False, give_up=False, open_now=False):
        return {"command": self.command, "state": self.state,
                "arrived": arrived, "give_up": give_up, "open_now": open_now,
                "note": self.note}

    def _open_claw(self, why, claw_open_since):
        """Stand still until the claw is open. Asks main.py to send RELEASE
        whenever it does not know the claw to be open."""
        self.state = "OPENING"
        self.command = "STOP"
        self.note = why
        return self._out(open_now=(claw_open_since is None))

    def _stuck(self, nav, why, claw_open_since):
        distance = nav["gripper_distance"]
        close = distance <= config.PICKUP_DISTANCE * FORCE_ARRIVE_FACTOR
        if close and self._needs_claw(nav) and not self.claw_ready:
            return self._open_claw(f"stuck ({why}) but close - opening the claw first", claw_open_since)
        self.command = "STOP"
        if close:
            self.state = "ARRIVED"
            self.note = f"stuck ({why}) but close - taking it here"
            return self._out(arrived=True)
        self.state = "IDLE"
        self.note = f"stuck ({why}) - giving up on this target"
        self._reset_target_progress()
        return self._out(give_up=True)

    @staticmethod
    def _needs_claw(nav):
        """Gem targets need the claw open before contact; bases do not."""
        tid = nav.get("target_id")
        return bool(tid) and tid[0] == "gem"

    def _far(self, nav):
        """Still far enough to keep driving continuously?"""
        if self._needs_claw(nav) and nav.get("tip_distance") is not None:
            # Stop early enough that the robot (still moving while the stop
            # command travels through camera + Wi-Fi) halts OUTSIDE the tip
            # circle, with time to open the claw before anything touches it.
            margin = _clamp(self.drive_rate * STOP_LATENCY_SEC, *STOP_MARGIN_LIMITS)
            return nav["tip_distance"] > getattr(config, "GRIPPER_TIP_RADIUS", 30) + margin
        return nav["gripper_distance"] > config.PICKUP_DISTANCE * SLOW_ZONE_FACTOR

    # ------------------------------------------------------------------
    def _border_denied(self, nav, why, claw_open_since):
        """
        The next forward move would carry the robot out of the safe zone (see
        border_guard.py). Either the target is already close enough to take it
        from here, or it cannot be reached without leaving the camera's view -
        then give it up rather than risk losing the robot.
        """
        distance = nav["gripper_distance"]
        is_gem = bool(nav.get("target_id")) and nav["target_id"][0] == "gem"
        radius = nav.get("target_radius") or 0.0
        reach = config.PICKUP_DISTANCE * FORCE_ARRIVE_FACTOR
        if not is_gem:
            reach = max(reach, 0.8 * radius)    # a base is big: the grip spot only has to be ON it
        self.command = "STOP"
        if distance <= reach:
            if is_gem and not self.claw_ready:
                return self._open_claw("at the border - opening the claw first", claw_open_since)
            self.state = "ARRIVED"
            self.note = f"at the border - {'taking it' if is_gem else 'placing it'} from here ({why})"
            return self._out(arrived=True)
        self.state = "IDLE"
        self.note = f"cannot reach it without leaving the camera's view ({why}) - giving up on this target"
        self._reset_target_progress()
        return self._out(give_up=True)

    def update(self, now, pose, nav, claw_open_since=None, zone=None):
        """
        pose: (x, y, heading_deg) of the robot center, or None if the marker is not seen.
        nav:  the dict from navigation.compute_navigation().
        claw_open_since: when main.py sent the open command (None = closed / unknown).
        zone: a border_guard.SafeZone (or None = no border): forward/backward moves that
              would end outside it are refused or cut short, and a target that cannot be
              reached without leaving it is given up.
        With pose None the robot does not just stand there: it runs the lost-marker
        recovery (see border_guard.py).
        Returns {"command", "state", "arrived", "give_up", "open_now", "note"}.
        """
        if pose is None:
            out = self._lost(now)
        else:
            self._found(now, pose, zone)
            out = self._update_core(now, pose, nav, claw_open_since, zone)
        if pose is not None:               # the recovery's own commands are never remembered
            self._record_command(out["command"], now)
        return out

    def _update_core(self, now, pose, nav, claw_open_since=None, zone=None):
        """
        pose: (x, y, heading_deg) of the robot center, or None if not seen.
              heading must include the grip slant (see navigation.py).
        nav:  the dict from navigation.compute_navigation().
        claw_open_since: time (same clock as `now`) when main.py sent the open
              command, or None if the claw is closed / not known to be open.
        Returns {"command", "state", "arrived", "give_up", "open_now", "note"}.
        """

        self.claw_ready = (claw_open_since is not None
                           and now >= claw_open_since + OPEN_WAIT_SEC)

        has_target = pose is not None and nav["nav_target_point"] is not None
        if has_target and nav["target_id"] != self.target_id:
            self.target_id = nav["target_id"]
            self._reset_target_progress()
            if self.state in ("ARRIVED", "OPENING"):
                self.state = "IDLE"

        # ---- finish whatever is in progress ------------------------------
        if self.state == "WAIT":
            if now < self.action_end:
                return self._out()
            seconds, self.pending_backup = self.pending_backup, None
            self._start("BACKUP", "BACKWARD", now, seconds, None)

        if self.state in ("TURN", "HOP", "BACKUP"):
            if now < self.action_end:
                if zone is not None and self.state != "TURN" and self._border_stops(zone, pose, now):
                    self._begin_settle(now)
                else:
                    return self._out()
            else:
                self._begin_settle(now)

        if self.state == "DRIVE" and zone is not None and self._border_stops(zone, pose, now):
            self._begin_settle(now)

        if self.state == "DRIVE":
            keep = config.TURN_ANGLE_THRESHOLD * KEEP_DRIVING_FACTOR
            if (has_target and nav["target_id"] == self.action_target
                    and abs(nav["angle_diff"]) <= keep and self._far(nav)):
                return self._out()
            self._begin_settle(now)

        if self.state == "SETTLE":
            if now < self.settle_end:
                return self._out()
            finished = self.action
            self._learn(pose)
            self.state = "IDLE"
            self._border_report(zone, pose)

            # Did that hop push the gem instead of closing in on it?
            if (has_target and finished is not None and finished["kind"] == "HOP"
                    and finished.get("gem_point") is not None
                    and finished.get("target") == nav["target_id"]):
                gx0, gy0 = finished["gem_point"]
                dx = nav["nav_target_point"][0] - gx0
                dy = nav["nav_target_point"][1] - gy0
                moved = math.hypot(dx, dy)
                heading = math.radians(pose[2])
                ahead = dx * math.cos(heading) + dy * math.sin(heading)
                if moved >= PUSH_MOVE_PX and ahead >= 0.6 * moved:
                    self.pushes += 1
                    if self.pushes > MAX_PUSHES:
                        return self._stuck(nav, "keeps pushing the gem instead of entering it",
                                           claw_open_since)
                    back = _clamp((2 * ahead + 15) / self.drive_rate, 0.15, 0.5)
                    self._reset_target_progress(keep_misses=True, keep_pushes=True)
                    self._start("BACKUP", "BACKWARD", now, back, None)
                    self.note = f"pushing the gem ({ahead:.0f}px ahead) - backing up to try again"
                    return self._out()

        if self.state == "OPENING":
            if has_target and self.claw_ready:
                self.state = "IDLE"
            elif has_target:
                return self._out(open_now=(claw_open_since is None))
            else:
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
        needs_claw = self._needs_claw(nav)

        if distance <= config.PICKUP_DISTANCE:
            if needs_claw and not self.claw_ready:
                return self._open_claw("at the gem - opening the claw first", claw_open_since)
            self.state = "ARRIVED"
            self.note = "arrived"
            return self._out(arrived=True)

        if zone is not None and self._band_escape(zone, pose, now):
            return self._out()

        if self.final_approach and self.min_distance is not None \
                and distance > self.min_distance + OVERSHOOT_PX:
            if self.min_distance <= config.PICKUP_DISTANCE * FORCE_ARRIVE_FACTOR:
                if needs_claw and not self.claw_ready:
                    return self._open_claw("passed just by it - opening the claw first",
                                           claw_open_since)
                self.state = "ARRIVED"
                self.note = "passed just by it - taking it here"
                return self._out(arrived=True)
            # Went past it without the claw getting close: back up and retry.
            self.misses += 1
            if self.misses > MAX_MISSES:
                return self._stuck(nav, "keeps passing beside it", claw_open_since)
            back = _clamp((distance + config.PICKUP_DISTANCE) / self.drive_rate, 0.15, 0.6)
            self._reset_target_progress(keep_misses=True)
            self._start("BACKUP", "BACKWARD", now, back, None)
            self.note = f"missed it (closest {self.min_distance_seen:.0f}px) - backing up to retry"
            return self._out()

        far = self._far(nav)

        # Open the claw BEFORE any more aiming. Turning pivots about the robot's
        # centre, and the claw tip is ~100px out from it, so even a modest
        # correction swings the tip sideways by tens of pixels - straight into
        # the gem if the claw is not open yet (seen in simulation: the tip went
        # from 68px to 8px away without the robot driving at all).
        if needs_claw and not far and not self.claw_ready:
            return self._open_claw("stopped short of the gem - opening the claw", claw_open_since)

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
                    return self._stuck(nav, "turning does not reduce the angle", claw_open_since)

            duration = _clamp(TURN_GAIN * abs(angle) / self.turn_rate, TURN_MIN_SEC, TURN_MAX_SEC)
            direction = "RIGHT" if angle > 0 else "LEFT"
            self._start("TURN", direction, now, duration, pose)
            self.note = f"turn {direction} {duration:.2f}s for {angle:+.0f} deg"
            return self._out()

        self.best_angle = None
        self.turn_fails = 0

        # ---- lined up and far: drive continuously -------------------------
        if not self.final_approach and far:
            if zone is not None:
                ok, why = zone.allows((pose[0], pose[1]), pose[2], 1,
                                      self._speed() * self._coast())
                if not ok:
                    fitted = self._fit_hop(zone, pose)
                    if fitted is None:
                        return self._border_denied(nav, why, claw_open_since)
                    self._start("HOP", "FORWARD", now, fitted, pose)
                    self.action["fitted"] = True
                    self.note = f"border: little room left - short timed hop {fitted:.2f}s ({why})"
                    return self._out()
            self._start("DRIVE", "FORWARD", now, 0.0, pose)
            self.action_target = nav["target_id"]
            tip = nav.get("tip_distance")
            self.note = (f"drive, tip {tip:.0f}px from the gem" if needs_claw and tip is not None
                         else f"drive, {distance:.0f}px to go")
            return self._out()

        # ---- close: the claw must be OPEN before moving in ----------------
        if needs_claw and not self.claw_ready:
            return self._open_claw("stopped short of the gem - opening the claw", claw_open_since)

        # ---- lined up and close: short hops, heading locked ----------------
        if self.final_approach:
            if distance < self.min_distance - DIST_PROGRESS_PX:
                self.hop_fails = 0
            else:
                self.hop_fails += 1
                if self.hop_fails >= STUCK_HOP_TRIES:
                    return self._stuck(nav, "hops do not get closer", claw_open_since)
        self.final_approach = True
        self.min_distance = distance if self.min_distance is None else min(self.min_distance, distance)
        self.min_distance_seen = self.min_distance

        remaining = distance - config.PICKUP_DISTANCE * 0.5
        hop_px = HOP_GAIN * remaining
        creeping = needs_claw and nav.get("in_tip_zone")
        if creeping:
            hop_px = min(hop_px, CREEP_PX)   # a gem is inside the tip circle: no big lunges
        duration = _clamp(hop_px / self.drive_rate, HOP_MIN_SEC, HOP_MAX_SEC)
        fitted = False
        if zone is not None:
            ok, why = zone.allows((pose[0], pose[1]), pose[2], 1,
                                  self._speed() * (duration + self._command_lag()))
            if not ok:
                shorter = self._fit_hop(zone, pose)
                if shorter is None:
                    return self._border_denied(nav, why, claw_open_since)
                duration, fitted = min(duration, shorter), True
        self._start("HOP", "FORWARD", now, duration, pose)
        if fitted:
            self.action["fitted"] = True
        self.action["gem_point"] = nav["nav_target_point"]
        self.action["target"] = nav["target_id"]
        self.note = f"{'creep' if creeping else 'hop'} {duration:.2f}s, {distance:.0f}px to go"
        return self._out()