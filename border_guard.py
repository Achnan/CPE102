"""
border_guard.py
===============
Keeps the robot inside the camera's picture - and gets it back if it leaves
anyway. The real field is bigger than what the camera sees, and once the ArUco
marker leaves the picture the robot is blind: the program can no longer tell
where it is, and you may only pick the robot up and replace it once.

Three layers (motion.py / motion_two_area.py and navigation*.py call this):

  1. KEEP-IN ZONE  A virtual border = the camera frame shrunk by a margin (and,
     if you saved a field outline with field_roi_tuner.py, cut down to that
     outline too). Before any forward/backward move, the robot's position AFTER
     the move - including the distance it keeps coasting while the stop command
     travels through the camera and Wi-Fi - must still be inside it. If not,
     the move is refused or cut short.

  2. REACHABLE TARGETS  The claw reaches about GRIPPER_FORWARD_OFFSET px beyond
     the robot's centre, so a gem slightly past the border can still be picked
     up with the robot staying inside. Gems farther out than that are ignored
     instead of chased. A base near the edge is approached as far as the border
     allows, and the gem is released there if the grip spot is already on the
     base.

  2b. NO TURNING NEAR THE EDGE  Turning pivots the marker about its centre, and a
     square marker needs its centre at least half a diagonal (0.71 side) from an
     edge to stay fully in view at EVERY heading. A robot whose centre has
     drifted closer than the margin (it coasted a little farther than predicted)
     does not turn there: before its next decision it makes ONE timed hop
     straight out of the band - forward or BACKWARD, whichever really gains
     clearance from ALL the edges along the way (in a corner the nearest edge
     changes, so the direction is judged on the whole path, not the nearest edge
     at the start) - and only then goes back to normal control. (In simulation,
     all of the marker losses that remained after the move checks were turns
     made inside this band.)

  3. LOST-MARKER RECOVERY  If the marker disappears right after the robot was
     moving, it most likely moved out of the picture. The robot RETRACES ITS
     STEPS: it plays back, in reverse and with each direction flipped (turn
     left <-> turn right, forward <-> back), every movement command sent since
     the last time it was seen safely inside the border - plus the distance it
     keeps coasting after a stop command - and stops the instant the marker is
     seen again. (Reversing only the LAST command is not enough: the controller
     often turns to re-aim just after the border stopped a drive, and then the
     last command is a turn although it was the drive that took the robot out.)
     If the retrace is not enough it backs up in short pulses. It never moves
     blind unless it was moving moments before: a marker lost while the robot
     sat still (a hand in the way, a reflection) just makes it wait.

How big is the margin?
    MARGIN_MARKERS marker lengths, plus the marker's own half-diagonal (so the
    WHOLE marker stays in view - the detector needs all four corners) - measured
    from the marker itself every frame. So it follows the camera: a different
    camera, height or zoom changes the marker's size in the picture and the
    margin follows. On top of that every move is checked with the distance the
    robot will still coast (speed x CAMERA_LATENCY_SEC + COMMAND_LATENCY_SEC).

    Make MARGIN_MARKERS bigger to be safer (you lose a strip along the edges),
    smaller to use more of the picture.

Set ENABLED = False to switch all of this off.
"""

import math
import time

import cv2
import numpy as np

# ---- settings -----------------------------------------------------------
ENABLED = True
MARGIN_MARKERS = 0.4         # extra clearance beyond the marker's half-diagonal, in marker lengths
CAMERA_LATENCY_SEC = 0.15    # the pose we see is this old
COMMAND_LATENCY_SEC = 0.15   # a stop command takes this long to take effect (Wi-Fi + one frame)
                             # (these two are only the STARTING guess for how far the robot coasts after
                             #  a stop: after every border stop it measures how far it really coasted
                             #  and uses that if it is more - see BorderMixin._border_report)
DEFAULT_MARKER_SIDE = 60.0   # px; used until the marker has been seen once
USE_FIELD_POLYGON = True     # also stay inside the outline from field_roi_tuner.py, if one is saved
REACH_SLACK_PX = 20          # gems must be reachable with this much to spare (hop size is coarse)

# lost-marker recovery
LOST_CONFIRM_SEC = 0.3       # marker must be missing this long before anything happens (flicker guard)
RECENT_MOVE_SEC = 2.5        # only back up blindly if the robot was driving this recently
RECOVER_PULSE_SEC = 0.35     # each blind back-up pulse
RECOVER_WAIT_SEC = 0.5       # stand still after each pulse and look for the marker
RECOVER_MAX_PULSES = 8       # then give up, stand still, and shout
SETTLE_AFTER_FOUND_SEC = 0.3
FIT_SAFETY = 1.15            # a border-fitted hop is planned to use 1/1.15 of the room: the speed is only an estimate
FIT_MIN_SEC = 0.08           # a fitted hop shorter than this is not worth making: the target is out of reach
FIT_MAX_SEC = 1.0            # ...and never longer than this
ESCAPE_HYSTERESIS_PX = 8     # aim to end this far past the inner edge of the border band
ESCAPE_MAX_PX = 300.0        # an escape hop never plans to travel farther than this
ESCAPE_MAX_SEC = 0.8         # ...or longer than this
ESCAPE_MIN_GAIN_PX = 6.0     # driving straight is only worth it if it gains at least this much clearance
RETRACE_MAX_SEC = 3.0        # never replay more than this much movement
RETRACE_MIN_SEG = 0.06       # a command segment shorter than this is still replayed for this long
REPEAT_WINDOW_SEC = 4.0      # lost again this soon after a recovery = the recovery overshot: undo half of its last step
WIGGLE_SEC = (0.12, 0.24, 0.24, 0.36, 0.36, 0.48)   # last resort: turn left, right, left... by growing amounts
FRAME_SEC = 0.034            # one camera frame: how long a command counts for each time it is sent
COAST_MAX_SEC = 1.2          # the learned coasting time never goes above this
START_SPEED_GUESS = 400.0    # px/s the guard assumes for the robot until it has measured a real drive (deliberately fast)
COAST_SAFETY = 1.25          # every prediction uses the coasting time x this: delays jitter from frame to frame


# ---- the marker as a ruler ---------------------------------------------
def marker_side_of(robot_info, marker_id):
    """Side length (px) of the robot's marker in this frame, or None if it is not visible."""
    corners, ids = robot_info.get("marker_corners"), robot_info.get("marker_ids")
    if ids is None or corners is None or len(corners) == 0:
        return None
    for corner, found in zip(corners, np.asarray(ids).flatten()):
        if found != marker_id:
            continue
        pts = np.asarray(corner).reshape(4, 2)
        return float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
    return None


# ---- the zone -------------------------------------------------------------
_outline_cache = {}


class SafeZone:

    def __init__(self, width, height, marker_side, polygon=None):
        self.width, self.height = int(width), int(height)
        self.side = float(marker_side)
        self.half_diag = 0.7071 * self.side           # the whole marker must stay inside
        self.margin = self.half_diag + MARGIN_MARKERS * self.side
        self.polygon = polygon                        # (N, 1, 2) float32 or None

    def signed_distance(self, p):
        """Distance from p to the nearest border: positive inside, negative outside."""
        x, y = float(p[0]), float(p[1])
        d = min(x, self.width - x, y, self.height - y)
        if self.polygon is not None:
            d = min(d, cv2.pointPolygonTest(self.polygon, (x, y), True))
        return d

    def allows(self, center, heading_deg, direction, travel_px):
        """
        May the robot move `travel_px` along (direction=+1) or against (-1) its
        heading from `center`? Returns (ok, reason). A move is fine if it ends
        inside the margin, or if it at least heads INWARD (so a robot already
        in the margin band can always drive out of it).
        """
        a = math.radians(heading_deg)
        end = (center[0] + direction * travel_px * math.cos(a), center[1] + direction * travel_px * math.sin(a))
        d_now, d_end = self.signed_distance(center), self.signed_distance(end)
        if d_end >= self.margin or d_end > d_now + 1.0:
            return True, ""
        return False, f"it would end {d_end:.0f}px from the border (needs {self.margin:.0f})"

    def room(self, center, heading_deg, direction, limit=1500.0):
        """
        How far (px) can the robot drive along (direction=+1) or against (-1) its heading before its
        centre would be closer to the border than the margin? 0 if it is already in the margin band and
        heading further out; `limit` if it never gets there.
        """
        a = math.radians(heading_deg)
        dx, dy = direction * math.cos(a), direction * math.sin(a)
        d0 = self.signed_distance(center)
        previous, t, step = d0, 0.0, 4.0
        while t < limit:
            t += step
            d = self.signed_distance((center[0] + dx * t, center[1] + dy * t))
            if d < self.margin and d < previous - 1e-9:
                return max(0.0, t - step) if d0 >= self.margin else 0.0
            previous = d
        return limit

    def gem_reachable(self, point, hold_offset):
        """Can the claw reach `point` while the robot's centre stays inside the margin?"""
        return self.signed_distance(point) >= self.margin - (hold_offset - REACH_SLACK_PX)

    # ---- drawing support ----
    def outline(self):
        """Contours (full-frame pixels) of the margin line, for drawing."""
        if self.polygon is None:
            m = int(round(self.margin))                      # just the frame, inset: exact
            return [np.array([[m, m], [self.width - m, m], [self.width - m, self.height - m], [m, self.height - m]],
                             np.int32).reshape(-1, 1, 2)]

        # with a field outline the margin line follows its shape: use a distance
        # transform on a half-size picture, cached because it is too slow to redo each frame
        key = (self.width, self.height, int(self.margin // 4), self.polygon.tobytes())
        if key in _outline_cache:
            return _outline_cache[key]
        k = 2
        w, h = max(8, self.width // k), max(8, self.height // k)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [(self.polygon.reshape(-1, 2) / k).astype(np.int32)], 255)
        mask[0, :] = mask[-1, :] = 0
        mask[:, 0] = mask[:, -1] = 0
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3) * k
        safe = (dist >= self.margin).astype(np.uint8) * 255
        found = cv2.findContours(safe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c * k for c in (found[0] if len(found) == 2 else found[1])]
        if len(_outline_cache) > 16:
            _outline_cache.clear()
        _outline_cache[key] = contours
        return contours


_polygon_cache = {"key": None, "polygon": None}


def _field_polygon(width, height):
    if not USE_FIELD_POLYGON:
        return None
    key = (width, height)
    if _polygon_cache["key"] == key:
        return _polygon_cache["polygon"]
    polygon = None
    try:
        import field_area
        points = field_area.get_points()
        if points:
            polygon = field_area.to_pixels(points, height, width).astype(np.float32).reshape(-1, 1, 2)
    except Exception:
        polygon = None
    _polygon_cache.update(key=key, polygon=polygon)
    return polygon


def make_zone(width, height, marker_side=None):
    """The zone for this frame size (None if the guard is switched off)."""
    if not ENABLED:
        return None
    return SafeZone(width, height, marker_side or DEFAULT_MARKER_SIDE, _field_polygon(width, height))


def draw_zone(image, zone):
    """Draw the border the robot is held inside (orange)."""
    if zone is None:
        return
    contours = zone.outline()
    if contours:
        cv2.polylines(image, contours, True, (0, 165, 255), 2, cv2.LINE_AA)
        x, y = contours[0].reshape(-1, 2)[0]
        cv2.putText(image, "robot border", (int(x) + 6, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 165, 255), 1, cv2.LINE_AA)


def draw_banner(image, text, color=(0, 0, 255)):
    """A bar across the top of the picture for things you must not miss."""
    h, w = image.shape[:2]
    ui = max(1.0, w / 1280.0)
    bar = int(44 * ui)
    roi = image[0:bar, 0:w]
    cv2.addWeighted(np.full_like(roi, 20), 0.75, roi, 0.25, 0, roi)
    scale = 0.8 * ui
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(image, text, (max(8, (w - tw) // 2), int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


# ---- recovery when the marker is lost ----------------------------------------
_REAL = ("FORWARD", "BACKWARD", "LEFT", "RIGHT")
_INVERT = {"FORWARD": "BACKWARD", "BACKWARD": "FORWARD", "LEFT": "RIGHT", "RIGHT": "LEFT"}


class Recovery:
    """Decides what to do each frame while the marker cannot be seen."""

    def __init__(self):
        self.reset(quiet=True)

    def reset(self, quiet=False):
        if not quiet and self.lost_since is not None and self.phase not in ("idle",):
            print(f"[border_guard] marker found again after {time.monotonic() - self.lost_since:.1f}s "
                  f"({self.steps_done} retrace step{'s' if self.steps_done != 1 else ''}, "
                  f"{self.pulses} back-up pulse{'s' if self.pulses != 1 else ''})")
        self.lost_since = None
        self.phase = "idle"       # idle -> retrace -> (wait <-> pulse) -> holding / gave_up
        self.queue = []           # retrace steps still to play: [(command, seconds)]
        self.label = "retracing"
        self.total_steps = 0
        self.steps_done = 0
        self.last_played = None   # (command, seconds) of the last real movement step this recovery played
        self.current = None       # (command, ends_at) of the retrace step being played
        self.direction = None     # fallback back-up direction
        self.pulses = 0
        self.until = 0.0
        self.note = ""

    @property
    def active(self):
        return self.lost_since is not None

    def step(self, now, make_plan):
        """
        make_plan(now): -> (steps, fallback_direction). steps = [(command, seconds), ...] to play, in order.
        Returns the command to send this frame.
        """
        if self.lost_since is None:
            self.lost_since = now
        if now - self.lost_since < LOST_CONFIRM_SEC:
            self.note = "marker not seen - waiting a moment"
            return "STOP"

        if self.phase == "idle":
            steps, self.direction = make_plan(now)
            if steps:
                self.phase = "retrace"
                self.label = "retracing"
                self.queue = list(steps)
                self.total_steps = len(steps)
                text = ", ".join(f"{c} {d:.2f}s" for c, d in steps)
                print(f"[border_guard] MARKER LOST right after moving - it probably left the picture. "
                      f"Retracing my steps: {text}")
            elif self.direction is not None:
                self.phase = "wait"
                self.until = now
                print(f"[border_guard] MARKER LOST - going {self.direction} in short pulses until it is seen again")
            else:
                self.phase = "holding"
                print("[border_guard] MARKER LOST but the robot had not been moving - "
                      "holding still (it may be hidden, not gone)")

        if self.phase == "holding":
            self.note = "marker lost - holding still"
            return "STOP"

        if self.phase == "gave_up":
            self.note = "MARKER LOST - recovery failed, robot stopped"
            return "STOP"

        if self.phase == "retrace":
            if self.current is not None and now < self.current[1]:
                self.note = f"marker lost - retracing {self.steps_done}/{self.total_steps}: {self.current[0]}"
                return self.current[0]
            if self.current is not None:
                self.current = None
            if self.queue:
                command, seconds = self.queue.pop(0)
                self.current = (command, now + seconds)
                self.steps_done += 1
                if command != "STOP":
                    self.last_played = (command, seconds)
                self.note = f"marker lost - {self.label} {self.steps_done}/{self.total_steps}: {command}"
                return command
            # the whole retrace is played and the marker is still not back
            if self.direction is not None:
                print("[border_guard] retrace finished and the marker is still not seen - "
                      f"going {self.direction} in short pulses")
                self.phase, self.until = "wait", now + RECOVER_WAIT_SEC
            elif self.label != "searching":
                # nothing known to back up along: turn left and right by growing amounts. If the robot's centre
                # is still in the picture, some headings put all four marker corners back inside it.
                print("[border_guard] retrace finished and the marker is still not seen - searching by turning "
                      "left and right")
                self.label = "searching"
                for k, seconds in enumerate(WIGGLE_SEC):
                    self.queue += [("LEFT" if k % 2 == 0 else "RIGHT", seconds), ("STOP", RECOVER_WAIT_SEC * 0.7)]
                self.total_steps = self.steps_done + len(self.queue)
                self.current = None
                return "STOP"
            else:
                self.phase = "gave_up"
                print("[border_guard] still not seen after searching - stopping. "
                      "Pick the robot up and put it back (you get one chance)")
                self.note = "MARKER LOST - recovery failed, robot stopped"
                return "STOP"

        if self.phase == "wait":
            if now < self.until:
                self.note = f"marker lost - looking ({self.pulses}/{RECOVER_MAX_PULSES})"
                return "STOP"
            if self.pulses >= RECOVER_MAX_PULSES:
                self.phase = "gave_up"
                print("[border_guard] MARKER STILL LOST after "
                      f"{RECOVER_MAX_PULSES} pulses - stopping. Pick the robot up and put it back (you get one chance)")
                self.note = "MARKER LOST - recovery failed, robot stopped"
                return "STOP"
            self.phase = "pulse"
            self.until = now + RECOVER_PULSE_SEC
            self.pulses += 1

        if self.phase == "pulse":
            if now < self.until:
                self.note = f"marker lost - going {self.direction} ({self.pulses}/{RECOVER_MAX_PULSES})"
                return self.direction
            self.phase = "wait"
            self.until = now + RECOVER_WAIT_SEC
            self.note = f"marker lost - looking ({self.pulses}/{RECOVER_MAX_PULSES})"
            return "STOP"

        return "STOP"


class BorderMixin:
    """
    Added to MotionController (motion.py and motion_two_area.py). Expects the
    host class to have: state, command, action, pending_backup, drive_rate,
    action_end, settle_end, note, _out() and _reset_target_progress().
    """

    def _border_setup(self):
        self._recovery = Recovery()
        self._cmd_log = []             # [command, started, ended] of the movement commands sent while the marker was seen
        self._last_seen = 0.0          # last time the marker was seen
        self._safe_seen = 0.0          # last time it was seen with the robot well inside the border: where a retrace aims to return to
        self._last_fix = None          # (command, seconds, time) of the last step of a recovery that found the marker again
        self._stop_pending = False     # the guard cut a move short and has not yet checked where the robot ended up
        self._coast_probe = None       # (x, y) where a drive was cut short, to measure how far it really coasted
        self._drive_coast = CAMERA_LATENCY_SEC + COMMAND_LATENCY_SEC   # seconds the robot coasts after a drive is stopped
        self._escalated = False        # already raised the coasting time for the current loss
        self._v_obs = None             # px/s measured during long, steady drives (None until one has been seen)
        self._drive_since = None       # when the current DRIVE began
        self._pose_hist = []           # (t, x, y) during the current DRIVE

    # ---- what was the robot doing? --------------------------------------
    def _record_command(self, command, now):
        """Remember the movement commands sent while the marker was visible (never the recovery's own)."""
        if command not in _REAL:
            return
        self._last_fix = None          # the robot is under normal control again
        if self._cmd_log and self._cmd_log[-1][0] == command and now - self._cmd_log[-1][2] <= 0.15:
            self._cmd_log[-1][2] = now + FRAME_SEC
        else:
            self._cmd_log.append([command, now, now + FRAME_SEC])
        while self._cmd_log and self._cmd_log[0][2] < now - 12.0:
            self._cmd_log.pop(0)

    def _retrace_plan(self, now):
        """
        Steps that bring the robot back to where it was last seen safely inside the border: every
        command sent since then (the camera and Wi-Fi delays mean the pose we saw was a little older than
        the commands sent just before it, so the window starts one coasting time earlier), newest first,
        each direction flipped. The robot moves for exactly as long as each command was sent (the delay only
        shifts it in time), so the replay needs no extra "coasting" time; commands still in flight when the
        marker vanished are finished long before the recovery starts. Segments are
        clipped to that window and the total is capped at RETRACE_MAX_SEC, so a long stay near the border
        never makes it replay old, unrelated moves.

        Lost again soon after a recovery found the marker? Then the recovery overshot (the camera was
        late to show it): undo half of its last step instead.

        Also returns the direction for the fallback back-up pulses: the opposite of the NET driving over
        the last RECENT_MOVE_SEC (forward time minus backward time). The newest command alone is not a
        safe guide: after a long drive toward the edge the robot may have reversed for an instant, and
        undoing only that last reversal would send it further out.
        """
        if self._last_fix is not None and now - self._last_fix[2] <= REPEAT_WINDOW_SEC and not self._cmd_log:
            command, seconds, _when = self._last_fix
            half = max(RETRACE_MIN_SEG, seconds * 0.5)
            self._last_fix = (_INVERT[command], half, now)       # a second overshoot halves it again
            return [(_INVERT[command], half)], None

        since = self._safe_seen - self._drive_coast
        steps, total = [], 0.0
        for command, started, ended in reversed(self._cmd_log):
            if ended <= since:
                break
            seconds = max(RETRACE_MIN_SEG, ended - max(started, since))
            if total + seconds > RETRACE_MAX_SEC:
                break
            steps.append((_INVERT[command], seconds))
            total += seconds

        # Forward and backward steps with no turn between them are on the same line: only their net
        # matters. (Without this, undoing "drive out, then back up a little" would first drive OUT again.)
        merged, line = [], 0.0
        for command, seconds in steps:
            if command in ("FORWARD", "BACKWARD"):
                line += seconds if command == "FORWARD" else -seconds
                continue
            if abs(line) >= RETRACE_MIN_SEG:
                merged.append(("FORWARD" if line > 0 else "BACKWARD", abs(line)))
            line = 0.0
            merged.append((command, seconds))
        if abs(line) >= RETRACE_MIN_SEG:
            merged.append(("FORWARD" if line > 0 else "BACKWARD", abs(line)))
        steps = merged

        net, horizon = 0.0, self._last_seen - RECENT_MOVE_SEC
        for command, started, ended in self._cmd_log:
            if command in ("FORWARD", "BACKWARD") and ended > horizon:
                net += (1.0 if command == "FORWARD" else -1.0) * (ended - max(started, horizon))
        fallback = None if abs(net) < 0.1 else ("BACKWARD" if net > 0 else "FORWARD")
        return steps, fallback

    # ---- the marker is gone / back --------------------------------------
    def _speed(self):
        """
        How fast the robot really moves (px/s), for every distance the border code works out. The motion
        controller's own estimate (drive_rate) can be far too LOW when camera + Wi-Fi are slow: it
        measures how far the robot moved after a short pause, before the robot has even started moving
        on the delayed command (in simulation it fell from 400 to ~30 px/s) - and a speed that is too low
        means distances are under-predicted and timed hops come out far too long. So: the speed
        measured during long steady drives if there is one, else the larger of the learned rate and a
        fast starting guess; never below the learned rate.
        """
        if self._v_obs is not None:
            return max(self._v_obs, self.drive_rate)
        return max(self.drive_rate, START_SPEED_GUESS)

    def _watch_speed(self, now, pose):
        """Measure the speed from successive poses during a long steady drive (after the delays have passed)."""
        if self.state != "DRIVE":
            self._drive_since = None
            self._pose_hist.clear()
            return
        if self._drive_since is None:
            self._drive_since = now
        self._pose_hist.append((now, pose[0], pose[1]))
        while self._pose_hist and self._pose_hist[0][0] < now - 0.35:
            self._pose_hist.pop(0)
        if now - self._drive_since < self._drive_coast + 0.5 or len(self._pose_hist) < 2:
            return                                         # still inside the camera + Wi-Fi delay: not moving at full speed yet
        t0, x0, y0 = self._pose_hist[0]
        if now - t0 >= 0.25:
            v = math.hypot(pose[0] - x0, pose[1] - y0) / (now - t0)
            if 20.0 < v < 3000.0:
                self._v_obs = v if self._v_obs is None else max(self._v_obs, v)

    def _coast(self):
        """Seconds the robot is assumed to keep moving after a stop is decided: learned, plus a cushion."""
        return self._drive_coast * COAST_SAFETY

    def _command_lag(self):
        """Wi-Fi + command delay, scaled up by however much slower than assumed the robot has turned out to be."""
        return COMMAND_LATENCY_SEC * self._coast() / (CAMERA_LATENCY_SEC + COMMAND_LATENCY_SEC)

    def _lost(self, now):
        command = self._recovery.step(now, self._retrace_plan)
        # The marker is really gone (not just a flicker: the recovery has moved past its wait) right after the
        # robot was moving: it coasted farther than the border check assumed. Be more careful from now on.
        if self._recovery.phase != "idle" and not self._escalated:
            self._escalated = True
            if any(seg[0] in ("FORWARD", "BACKWARD") and seg[2] > now - 3.0 for seg in self._cmd_log):
                new = min(COAST_MAX_SEC, self._drive_coast * 1.4)
                if new > self._drive_coast + 1e-6:
                    print(f"[border_guard] the marker was lost right after driving, so the robot coasts farther than the "
                          f"{self._drive_coast:.2f}s assumed - the border check now assumes {new:.2f}s")
                    self._drive_coast = new
        self.state = "LOST"
        self.command = command
        self.action = None
        self.pending_backup = None
        self.note = self._recovery.note
        return self._out()

    def _found(self, now, pose=None, zone=None):
        if self._recovery.active:
            played = self._recovery.last_played
            self._recovery.reset()
            self._cmd_log.clear()          # those moves are undone now; do not replay them again
            self._escalated = False
            self._last_fix = (played[0], played[1], now) if played else None
        if self.state == "LOST":
            self.state = "SETTLE"          # we just stopped / backed up: let it settle, then look afresh
            self.command = "STOP"
            self.action = None
            self.settle_end = now + SETTLE_AFTER_FOUND_SEC
            self._reset_target_progress()
        if pose is not None:
            self._watch_speed(now, pose)
            self._last_seen = now
            if zone is None or zone.signed_distance((pose[0], pose[1])) >= zone.margin:
                self._safe_seen = now

    # ---- a move that does not fit: make it shorter instead of giving up -----
    def _fit_hop(self, zone, pose, direction=1):
        """
        A full-speed drive (or the planned hop) would coast past the border. A hop is TIMED - it ends
        by the clock, not by what the camera shows a moment late - so one sized to the room that is
        left cannot overshoot by more than the command delay. Returns its duration in seconds, or
        None when there is no room worth a hop (then the target really cannot be reached).
        """
        room = zone.room((pose[0], pose[1]), pose[2], direction)
        usable = (room - self._speed() * self._command_lag()) / FIT_SAFETY
        seconds = usable / self._speed()
        if seconds < FIT_MIN_SEC:
            return None
        return min(seconds, FIT_MAX_SEC)

    # ---- leaving the border band without turning -------------------------
    def _band_escape(self, zone, pose, now):
        """
        Called when the motion logic is about to decide its next move. If the robot's centre is closer
        to the border than the margin, start ONE timed hop straight out of the band instead of letting
        the normal logic turn there. Returns True if a hop was started.

        The direction (forward or backward along the heading) is judged on the real distance to the
        nearest border in the whole picture, sampled along the path: it has to reach the clearance we want
        without ever being worse than where the robot stands now. That matters in corners, where the
        nearest edge changes along the way. If neither direction gains anything (the heading runs along
        the edge), nothing is done and the normal logic carries on.
        """
        x, y, heading = pose
        d0 = zone.signed_distance((x, y))
        if d0 >= zone.margin:
            return False
        want = zone.margin + ESCAPE_HYSTERESIS_PX
        a = math.radians(heading)
        best = None                                        # (reached, travel_px, direction)
        for direction in (1, -1):
            dx, dy = direction * math.cos(a), direction * math.sin(a)
            gain, gain_at, hit = 0.0, 0.0, None
            t = 0.0
            while t < ESCAPE_MAX_PX:
                t += 4.0
                d = zone.signed_distance((x + dx * t, y + dy * t))
                if d < d0 - 1.0:
                    break                                  # this way gets worse before it gets better
                if d - d0 > gain:
                    gain, gain_at = d - d0, t
                if d >= want:
                    hit = t
                    break
            if hit is not None:
                candidate = (1, hit, direction)
            elif gain >= ESCAPE_MIN_GAIN_PX:
                candidate = (0, gain_at, direction)
            else:
                continue
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
        if best is None:
            return False
        _reached, travel, direction = best
        seconds = min(ESCAPE_MAX_SEC, travel / self._speed())
        if seconds < 0.08:
            seconds = 0.08
        name = "FORWARD" if direction > 0 else "BACKWARD"
        self._start("HOP", name, now, seconds, pose)
        self.action["fitted"] = True
        self.note = (f"in the border band ({d0:.0f}px of {zone.margin:.0f}px): one {seconds:.2f}s hop "
                     f"{name.lower()} out of it, no turning near the edge")
        return True

    # ---- the border checks ---------------------------------------------
    def _border_stops(self, zone, pose, now):
        """Should the forward/backward move in progress be cut short?"""
        if self.action is not None and self.action.get("fitted"):
            return False                  # already sized to the room left: it ends by the clock
        direction = -1 if self.state == "BACKUP" else 1
        remaining = 0.0 if self.state == "DRIVE" else max(0.0, self.action_end - now)
        seconds = self._coast() + remaining
        ok, why = zone.allows((pose[0], pose[1]), pose[2], direction, self._speed() * seconds)
        if not ok:
            self.note = f"border: {why} - stopping"
            self._stop_pending = True
            if self.state == "DRIVE":
                self._coast_probe = (pose[0], pose[1])
            return True
        return False

    def _border_report(self, zone, pose):
        """
        Self-check, and self-correction: after the guard stopped a move, see where the robot REALLY
        ended up. If a drive coasted further than assumed (the real camera or Wi-Fi is slower than
        the starting guess), the coasting time used by the border check is raised to match, so the
        next stop is early enough.
        """
        if not self._stop_pending or zone is None:
            return
        self._stop_pending = False

        if self._coast_probe is not None:
            coasted = math.hypot(pose[0] - self._coast_probe[0], pose[1] - self._coast_probe[1])
            observed = coasted / self._speed()
            self._coast_probe = None
            nominal = CAMERA_LATENCY_SEC + COMMAND_LATENCY_SEC
            if observed > self._drive_coast:
                new = min(COAST_MAX_SEC, observed)               # never trust a faster stop than the slowest seen
                if new > self._drive_coast * 1.05:
                    print(f"[border_guard] the robot coasted {coasted:.0f}px after the stop = {observed:.2f}s of travel, "
                          f"more than the {self._drive_coast:.2f}s assumed - the border check now assumes {new:.2f}s "
                          f"(+{(COAST_SAFETY - 1) * 100:.0f}% cushion)")
                self._drive_coast = new
            elif observed < self._drive_coast * 0.6:
                self._drive_coast = max(nominal, 0.9 * self._drive_coast + 0.1 * observed)   # relax very slowly

        d = zone.signed_distance((pose[0], pose[1]))
        if d >= zone.half_diag:
            print(f"[border_guard] stopped near the border: the marker settled {d:.0f}px from the edge "
                  f"(it needs {zone.half_diag:.0f}px to stay fully in view, margin is {zone.margin:.0f}px) - OK")
        else:
            print(f"[border_guard] WARNING: after the stop the marker is only {d:.0f}px from the edge - less than the "
                  f"{zone.half_diag:.0f}px it needs to stay fully in view (the coasting time is being adjusted; "
                  f"raise MARGIN_MARKERS if this repeats)")