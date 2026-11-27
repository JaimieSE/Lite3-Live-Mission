#!/usr/bin/env python3
"""Group 08 mission reporter for the Dog NUC.

Consumes the Pi's perception JSON (std_msgs/String, ROS domain 8):
  /group8pi/perception/detections    YOLO objects (incl. hand_tools), QR text, identified hazard signs
  /group8pi/perception/scenes        Claude room label (scene.type) and scene.evidence  [BEST_EFFORT]
  /group8pi/perception/scene_events  scene_pending / scene_result / scene_failed       [RELIABLE]
assigns sightings to M01-M05, resolves the arena cell from TF (map -> base_link) at the original
camera capture time, and sends one booklet v6.4 seven-key POST per mission.

Pose rule: the pose is looked up only at capture_stamp_ns. If TF cannot answer yet, the frame
stays pending and is retried until it leaves the TF history window; it is never replaced by a
later pose. Arena frame (booklet p.15): origin at the start pose, +X initial forward, +Y left,
cells 1.02 m apart, pos = cell centre.

Position contract (agreed with the Pi session, 2026-11-26, operator: "Room cell only"): a mission position comes
only from a room photo, as room_scan_result.cell = [i, j], the room cell roam put in its capture_request. Roam's
course grid has the same origin and axes as the arena (start cell = (0, 0)), so its cell (i, j) is arena cell
(i, j) and pos = [i * 1.02, j * 1.02]. Sightings while walking carry no measured position: they are never placed
at the robot's cell or guessed; they only supply QR text for M04 (pos null).
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import queue
import re
import statistics
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DETECTIONS_TOPIC = "/group8pi/perception/detections"
SCENES_TOPIC = "/group8pi/perception/scenes"
EVENTS_TOPIC = "/group8pi/perception/scene_events"
ROOM_SCAN_TOPIC = "/group8pi/perception/room_scans"
ENDPOINTS = {
    "A": "http://192.168.10.141:8001/task_report_A",
    "B": "http://192.168.10.141:8001/task_report_B",
}
CELL_M = 1.02
TASK_TYPES = ("object", "hazard", "survivor", "qr")
SCENE_TYPES = ("clothes_shop", "fitness_centre", "kitchen", "market", "workshop",
               "clinic", "cleaning_room", "furniture_shop")
OBJECT_LABELS = ("medical_equipment", "hand_tools", "food", "water", "utensils")
HAZARD_LABELS = ("high_voltage", "radioactive", "chemical", "biological", "flammable")
WIRE_KEYS = ("group_id", "task_id", "task_type", "scene_type", "label", "pos", "message")
GROUP_ID = re.compile(r"^group08_[A-Za-z0-9]{4}$")
DEFAULT_GROUP_FILE = Path(__file__).resolve().parent / "group_id.txt"
# Booklet v6.4 p.14: fixed for practice, prelim and finals.
BOOKLET_MISSIONS = [
    {"task_id": "M01", "task_type": "object", "label": "hand_tools", "description":
     "Rescue teams need tools to clear debris and access damaged areas. "
     "Locate hand tools and report their location."},
    {"task_id": "M02", "task_type": "hazard", "description":
     "A potential hazard has been reported in the affected area. Locate the hazard "
     "warning sign, identify the type of hazard, and report its location."},
    {"task_id": "M03", "task_type": "survivor", "description":
     "Locate a conscious survivor who has left a message for rescuers (represented by "
     "a QR code). Report the type of area where the survivor is found and their location."},
    {"task_id": "M04", "task_type": "qr", "description":
     "Find and decode the QR code beside the conscious survivor (M03) and report the "
     "survivor's message exactly as decoded."},
    {"task_id": "M05", "task_type": "survivor", "description":
     "Locate another survivor who is weak or unconscious and unable to communicate "
     "(without a QR code). Report the type of area where the survivor is found and "
     "their location."},
]
_TYPE_ALIASES = {"hazard sign": "hazard", "hazard_sign": "hazard", "qr code": "qr", "qr_code": "qr"}


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# ---------------------------------------------------------------- missions
def _vocab_in(text, vocab):
    words = " " + re.sub(r"[^a-z]+", " ", text.lower()) + " "
    return [v for v in vocab if f" {v.replace('_', ' ')} " in words]


def normalise_missions(raw):
    """Accept a list, {"tasks"|"missions": [...]}, or {"M01": "description", ...}."""
    if isinstance(raw, dict):
        raw = raw.get("tasks") or raw.get("missions") or [
            {"task_id": k, "description": v} for k, v in raw.items()]
    defaults = {m["task_id"]: m for m in BOOKLET_MISSIONS}
    missions = {}
    for item in raw:
        tid = str(item.get("task_id") or item.get("id"))
        kind = str(item.get("task_type") or defaults.get(tid, {}).get("task_type", "")).lower()
        kind = _TYPE_ALIASES.get(kind, kind)
        if kind not in TASK_TYPES:
            raise ValueError(f"{tid}: unknown task_type {kind!r}")
        text = str(item.get("description") or "")
        labels = [item["label"]] if item.get("label") else _vocab_in(
            text, OBJECT_LABELS if kind == "object" else HAZARD_LABELS if kind == "hazard" else ())
        rooms = [item["scene_type"]] if item.get("scene_type") else _vocab_in(text, SCENE_TYPES)
        missions[tid] = dict(task_id=tid, task_type=kind, description=text,
                             label=labels[0] if len(labels) == 1 else None,
                             room=rooms[0] if len(rooms) == 1 else None)
    survivors = [m for m in missions.values() if m["task_type"] == "survivor"]
    if len(survivors) != 2 or sum(m["task_type"] == "qr" for m in missions.values()) != 1:
        raise ValueError("Expected two survivor missions and one QR mission")
    for m in survivors:   # the QR-linked survivor is the conscious one (booklet FAQ 7)
        text = m["description"].lower()
        m["needs_qr"] = "qr" in text and "without a qr" not in text
    if sum(m["needs_qr"] for m in survivors) != 1:
        raise ValueError("Could not tell which survivor mission is QR-linked")
    if any(m["task_type"] == "object" and not m["label"] for m in missions.values()):
        raise ValueError("Object mission has no exact target label")
    return missions


def retrieve_missions(url=None, path=None, timeout=5.0):
    """All five descriptions in ONE HTTP GET, or a local file, or the booklet list."""
    if url:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
            raw = json.load(r)
        log("Retrieved missions in one request from", url)
    elif path:
        raw = json.loads(Path(path).read_text())
    else:
        raw = BOOKLET_MISSIONS
    return normalise_missions(raw)


def load_group_id(value=None, path=DEFAULT_GROUP_FILE):
    """--group-id wins; otherwise the one-line group_id.txt, so the code needs no edit on the day."""
    if value:
        return value.strip()
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "group08_XXXX"


# ------------------------------------------------------------------ parsing
def _key(rec):
    return (int(rec["capture_stamp_ns"]), str(rec["frame_id"]))


def parse_detection(data):
    """Pi detections record (schema_version 1) -> normalised observation."""
    rec = json.loads(data)
    objects = [dict(name=o["name"], cls=o.get("class"), conf=float(o.get("confidence", 0)),
                    box=[o["box"][k] for k in ("x1", "y1", "x2", "y2")], base_xy=o.get("base_xy"))
               for o in rec.get("objects") or []]
    qr = [c["text"] for c in rec.get("qr_codes") or [] if c.get("text")]
    hazards = []
    for s in rec.get("warning_sign_candidates") or []:   # Pi classifier puts the class in `label`
        label = s.get("label") or s.get("hazard_label")
        if label in HAZARD_LABELS:
            hazards.append(dict(label=label, conf=float(s.get("confidence", 1.0)), base_xy=s.get("base_xy")))
    hazards += [dict(label=o["name"], conf=o["conf"], base_xy=o.get("base_xy"))
                for o in objects if o["name"] in HAZARD_LABELS]
    return dict(key=_key(rec), frame=rec.get("frame"), objects=objects, qr=qr, hazards=hazards,
                sign_candidates=len(rec.get("warning_sign_candidates") or []),
                stair_curtain=rec.get("stair_curtain"))


def parse_scene(data):
    """A /scenes record or a scene_result event -> {key, type, evidence, model, request_id}."""
    rec = json.loads(data)
    scene = rec.get("scene") or {}
    return dict(key=_key(rec), type=scene.get("type"), evidence=list(scene.get("evidence") or []),
                model=rec.get("model"), request_id=rec.get("request_id"))


def parse_room_scan(data):
    """A room_scan_result from the Pi, or None (other events, bad JSON)."""
    rec = json.loads(data)
    if not isinstance(rec, dict) or rec.get("event") != "room_scan_result" or not rec.get("scan_id"):
        return None
    return rec


def scan_observations(scan):
    """Room scan -> one (observation, scene | None) per captured view, in the detections format.

    Claude's findings and the locally decoded QR text are attached to the view they were seen in, under that
    view's own capture stamp and camera frame, so the existing capture-time pose lookup places each finding.
    Unclear findings and unknown hazard symbols are left out (they can still be settled by a later look)."""
    room = scan.get("room") or {}
    cell = room_cell(scan.get("cell"))
    out = []
    for view in scan.get("views") or []:
        vid = view.get("view_id")
        stamp, frame_id = view.get("capture_stamp_ns"), view.get("frame_id")
        if type(stamp) is not int or not frame_id:
            continue
        objects, hazards = [], []
        for f in scan.get("findings") or []:
            if f.get("view_id") != vid or not f.get("clear"):
                continue
            if f.get("kind") == "person":
                objects.append(dict(name="person", cls=None, conf=0.9, box=None, confirmed=True))
            elif f.get("kind") == "hand_tools":
                objects.append(dict(name="hand_tools", cls=None, conf=0.9, box=None))
            elif f.get("kind") == "hazard_sign" and f.get("label") in HAZARD_LABELS:
                hazards.append(dict(label=f["label"], conf=0.9))
        qr = [q["text"] for q in scan.get("qr") or [] if q.get("view_id") == vid and q.get("text")]
        obs = dict(key=(stamp, str(frame_id)), frame=view.get("frame"), objects=objects, qr=qr, hazards=hazards,
                   sign_candidates=0, stair_curtain=None, scan_id=scan["scan_id"], view_id=vid, cell=cell)
        cues = [str(e.get("cue")) for e in room.get("evidence") or [] if e.get("cue")]
        scene = None
        if room.get("type") in SCENE_TYPES and cues:
            scene = dict(key=obs["key"], type=room["type"], evidence=cues, model=scan.get("model"),
                         request_id=scan["scan_id"])
        out.append((obs, scene))
    return out


# ------------------------------------------------------------------ geometry
def to_arena(map_pose, origin):
    """map-frame (x, y, yaw) -> arena (x, y, yaw) relative to the start pose `origin`."""
    x, y, yaw = map_pose
    ox, oy, oyaw = origin
    dx, dy = x - ox, y - oy
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    return c * dx - s * dy, s * dx + c * dy, math.atan2(math.sin(yaw - oyaw), math.cos(yaw - oyaw))


def cell_of(xy):
    return (round(xy[0] / CELL_M), round(xy[1] / CELL_M))


def cell_centre(cell):
    return [round(cell[0] * CELL_M, 2), round(cell[1] * CELL_M, 2)]


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def target_base_xy(obs):
    """The target's [x, y] in base_link for this frame (Pi depth, base_xy): the person first, then a hazard sign,
    then a mission object. None when none of them carries one."""
    for item in [o for o in obs.get("objects") or [] if o.get("name") == "person"] + list(obs.get("hazards") or []) + \
            [o for o in obs.get("objects") or [] if o.get("name") in OBJECT_LABELS]:
        xy = item.get("base_xy")
        if isinstance(xy, (list, tuple)) and len(xy) == 2 and all(isinstance(v, (int, float)) for v in xy):
            return float(xy[0]), float(xy[1])
    return None


QR_LINK_WINDOW_S = 30.0       # a walking QR and a room scan in the same visit: this close in capture time
QR_LINK_POSE_M = 1.2          # ... and the dog this close to where it stood for the scan
QR_LINK_SETTLE_S = QR_LINK_WINDOW_S + 60.0   # decide only once every scan of the window can have answered (Claude
                                             # answers in 10-60 s): never link before a second room could show up


def room_cell(value):
    """room_scan_result.cell [i, j] (roam's course grid = arena cell indices) -> (i, j), or None if unusable."""
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) and abs(v) <= 25 for v in value):
        return None
    return int(value[0]), int(value[1])


class ClockMonitor:
    """Pi capture stamp vs NUC receive time. Negative lag means the Pi clock runs ahead."""

    def __init__(self, window=200, max_ahead_ms=50.0, max_lag_ms=1500.0):
        self.lags = deque(maxlen=window)
        self.max_ahead_ms, self.max_lag_ms = max_ahead_ms, max_lag_ms
        self.warned = None

    def add(self, capture_ns, received_ns):
        self.lags.append((received_ns - capture_ns) / 1e6)
        if len(self.lags) >= 10:
            verdict = self.verdict()
            if verdict != "ok" and verdict != self.warned:
                log("CLOCK WARNING:", verdict, self.summary())
            self.warned = verdict if verdict != "ok" else None

    def verdict(self):
        if not self.lags:
            return "no data"
        if min(self.lags) < -self.max_ahead_ms:
            return "Pi clock ahead of NUC: capture stamps are in the NUC's future"
        if statistics.median(self.lags) > self.max_lag_ms:
            return "Pi clock behind NUC or link very slow"
        return "ok"

    def summary(self):
        if not self.lags:
            return {}
        return dict(samples=len(self.lags), median_ms=round(statistics.median(self.lags), 1),
                    min_ms=round(min(self.lags), 1), max_ms=round(max(self.lags), 1))


# ------------------------------------------------------------------ state
class Encounter:
    """Sightings made from one arena cell without a long gap."""

    def __init__(self, number, cell, now):
        self.id, self.cell, self.first, self.last = number, cell, now, now
        self.scene = None
        self.person_hits = set()      # capture stamps with a confident person
        self.person_confirmed = False  # a room scan (Claude) saw a clear survivor here
        self.qr = {}                  # text -> first frame key
        self.objects = {}             # label -> best confidence
        self.hazards = {}             # label -> best confidence


class MissionTracker:
    def __init__(self, missions, poster, group_id, pose_at, *, encounter_gap_s=15.0,
                 person_min_hits=3, person_min_conf=0.5, object_min_conf=0.4,
                 tf_history_s=120.0, scene_orphan_s=30.0, look_ahead_m=0.0, qr_survivor_max_cells=1,
                 room_cell_only=True, pos_source=None, m05_after_s=None):
        self.missions, self.poster, self.group_id = missions, poster, group_id
        self.pose_at = pose_at        # capture_stamp_ns -> arena (x, y, yaw) or None if not yet known
        self.gap, self.person_min_hits = encounter_gap_s, person_min_hits
        self.person_min_conf, self.object_min_conf = person_min_conf, object_min_conf
        self.tf_history_s, self.scene_orphan_s = tf_history_s, scene_orphan_s
        self.look_ahead_m, self.qr_survivor_max_cells = look_ahead_m, qr_survivor_max_cells
        # Operator 2026-11-27: a survivor with no QR beside it for this long goes out as M05 (no QR) instead of
        # waiting for M03 forever (26 Nov runs: two cleaning_room survivors were never sent). None: never.
        self.m05_after_s = m05_after_s
        # True (competition): positions only from room photos. False: the old robot-cell placement (tests only).
        self.room_cell_only = room_cell_only
        # Position source: "room_cell" (default, competition): only room photos, at the room's cell.
        # "base_xy" (option): room photos at the room's cell; other detections at the capture-time pose plus the
        # target's base_xy from the Pi's depth (PI_BASE_XY=1 on the Pi). Never the robot's own cell.
        self.pos_source = pos_source or ("room_cell" if room_cell_only else "robot_cell")
        if self.pos_source == "base_xy":
            self.room_cell_only = False
        self.frames = {}              # key -> dict(obs, received, pose, encounter, scene)
        self.unlocated = deque()      # keys waiting for their capture-time pose
        self.orphan_scenes = {}       # key -> (received, scene) that beat their detections
        self.encounters = []
        self.qr_seen = {}             # text -> scene type for M04 (no pose needed)
        self.state = {}               # task_id -> "sending" | "accepted"
        self.reports = {}             # task_id -> last payload handed to the poster
        self.rejected = {}            # task_id -> payloads the server rejected
        self.notes = {}
        self.now = 0.0
        self.scan_views = set()       # (scan_id, view_id) already taken in: a re-sent result adds only new views
        # Operator 2026-11-27 ("why is M03 not detected"): a QR decoded from a live walking frame has no cell. It is
        # linked to a room cell when decoded in the same visit as that room's scan: within QR_LINK_WINDOW_S of one of
        # the scan's view stamps, only one room cell scanned in that window, and the dog's map pose at the QR frame
        # within QR_LINK_POSE_M of where it stood for the scan. Otherwise it stays unlinked (reason logged).
        self.loose_qr = []            # dict(text, stamp_ns, done)
        self.scan_visits = []         # dict(scan_id, cell, stamps_ns, pose [x, y, yaw] map or None)
        self.linked_qr_cells = set()
        self.scanned_cells = set()    # room cells a Claude room scan has looked at
        self.scan_qr_cells = set()    # ... whose scan reported a qr_code (read or not): M03 candidates, never M05
        self.map_pose_at = None       # capture_stamp_ns -> map (x, y, yaw) or None (set by run_ros)
        self.accept_dry_run_scans = False
        self.stats = dict(frames=0, located=0, pose_expired=0, scenes=0, scenes_joined=0,
                          scenes_orphaned=0, scenes_unusable=0)

    # ---- input
    def on_detection(self, obs, now):
        self.now = max(self.now, now)
        relevant = (obs["qr"] or obs["hazards"] or
                    any(o["name"] == "person" or o["name"] in OBJECT_LABELS for o in obs["objects"]))
        if not relevant or obs["key"] in self.frames:
            return
        if obs.get("cell") is None and (self.room_cell_only or
                                        (self.pos_source == "base_xy" and target_base_xy(obs) is None)):
            for text in obs["qr"]:            # QR text needs no position (M04 pos is null)
                if not obs.get("scan_id") and not any(q["text"] == text and abs(q["stamp_ns"] - obs["key"][0]) < 5e9
                                                      for q in self.loose_qr):
                    self.loose_qr.append(dict(text=text, stamp_ns=obs["key"][0], done=False, first_now=now))
                self.qr_seen.setdefault(text, None)
            self.stats["unplaced"] = self.stats.get("unplaced", 0) + 1
            if obs.get("scan_id"):
                log(f"room scan {obs['scan_id']} has no usable cell; its findings get no position")
            if obs["qr"]:
                self.evaluate()
            return
        self.stats["frames"] += 1
        frame = self.frames[obs["key"]] = dict(obs=obs, received=now, pose=None, encounter=None,
                                               scene=None)
        for text in obs["qr"]:
            self.qr_seen.setdefault(text, None)
        orphan = self.orphan_scenes.pop(obs["key"], None)
        if orphan:
            self._attach(frame, orphan[1])
        self.unlocated.append(obs["key"])
        self.tick(now)

    def on_room_scan(self, scan, now):
        """A Pi room scan: each captured view becomes an observation at its own capture stamp (pose looked up at
        that instant), with the room Claude named. Re-sent results (after an extra look) add only new views."""
        if scan.get("dry_run") and not self.accept_dry_run_scans:
            log(f"room scan {scan.get('scan_id')} is a dry run; ignored for reporting")
            return
        stamps = [v.get("capture_stamp_ns") for v in scan.get("views") or [] if type(v.get("capture_stamp_ns")) is int]
        scan_cell = room_cell(scan.get("cell"))
        if scan_cell is not None and not (scan.get("dry_run") and not self.accept_dry_run_scans):
            self.scanned_cells.add(scan_cell)
            if scan.get("qr") or any(f.get("kind") == "qr_code" for f in scan.get("findings") or []):
                self.scan_qr_cells.add(scan_cell)
        if room_cell(scan.get("cell")) is not None and stamps:
            self.scan_visits.append(dict(scan_id=scan["scan_id"], cell=room_cell(scan.get("cell")), stamps_ns=stamps,
                                         pose=scan.get("pose")))
        for obs, scene in scan_observations(scan):
            tag = (scan["scan_id"], obs["view_id"])
            if tag in self.scan_views:
                continue
            self.scan_views.add(tag)
            had = obs["key"] in self.frames
            self.on_detection(obs, now)
            if scene is not None and obs["key"] in self.frames and not had:
                self.on_scene(scene, now)
            elif obs["qr"] and obs["key"] in self.frames:
                self.qr_seen.update({t: self.qr_seen.get(t) for t in obs["qr"]})
        log(f"room scan {scan['scan_id']}: room {(scan.get('room') or {}).get('type')}, "
            f"{len(scan.get('findings') or [])} finding(s), QR {[q.get('text') for q in scan.get('qr') or []]}")
        self.evaluate()

    def on_scene(self, scene, now):
        self.now = max(self.now, now)
        self.stats["scenes"] += 1
        if scene["type"] not in SCENE_TYPES or not scene["evidence"]:
            self.stats["scenes_unusable"] += 1
            log(f"scene {scene['type']!r} for {scene['key']} has no booklet scene_type with evidence")
            return
        frame = self.frames.get(scene["key"])
        if frame is None:
            self.orphan_scenes[scene["key"]] = (now, scene)
            self.stats["scenes_orphaned"] += 1
            return
        self._attach(frame, scene)
        log(f"scene {scene['type']} joined frame {frame['obs']['frame']} "
            f"{now - frame['received']:.1f}s after its detections"
            + (f" (encounter {frame['encounter'].id}, cell {frame['encounter'].cell})"
               if frame["encounter"] else " (pose still pending)"))
        self.evaluate()

    def tick(self, now):
        """Retry capture-time poses, expire old buffers, and re-evaluate."""
        self.now = max(self.now, now)
        for _ in range(len(self.unlocated)):
            key = self.unlocated.popleft()
            frame = self.frames.get(key)
            if frame is None:
                continue
            if frame["obs"].get("cell") is not None and self.pos_source in ("room_cell", "base_xy"):
                self._locate(frame, "room cell", now)
                continue
            pose = self.pose_at(key[0])
            if pose is not None:
                self._locate(frame, pose, now)
            elif now - frame["received"] > self.tf_history_s:
                self.stats["pose_expired"] += 1
                log(f"frame {frame['obs']['frame']}: no TF at capture time within "
                    f"{self.tf_history_s:.0f}s; its sightings stay unreported")
            else:
                self.unlocated.append(key)
        for key in [k for k, (t, _) in self.orphan_scenes.items() if now - t > self.scene_orphan_s]:
            del self.orphan_scenes[key]
        for key in [k for k, f in self.frames.items()
                    if now - f["received"] > self.tf_history_s and f["pose"] is not None]:
            del self.frames[key]
        self.evaluate()

    def _attach(self, frame, scene):
        frame["scene"] = scene
        self.stats["scenes_joined"] += 1
        if frame["encounter"] is not None:
            frame["encounter"].scene = scene
        for text in frame["obs"]["qr"]:
            self.qr_seen[text] = scene["type"]

    def _locate(self, frame, pose, now):
        obs = frame["obs"]
        if pose == "room cell":
            cell = obs["cell"]
        elif self.pos_source == "base_xy":
            x, y, yaw = pose
            bx, by = target_base_xy(obs)
            cell = cell_of((x + bx * math.cos(yaw) - by * math.sin(yaw), y + bx * math.sin(yaw) + by * math.cos(yaw)))
        else:
            x, y, yaw = pose
            cell = cell_of((x + self.look_ahead_m * math.cos(yaw), y + self.look_ahead_m * math.sin(yaw)))
        enc = self._encounter(cell, now)
        frame["pose"], frame["encounter"] = pose, enc
        self.stats["located"] += 1
        for o in obs["objects"]:
            if o["name"] == "person" and o["conf"] >= self.person_min_conf:
                enc.person_hits.add(obs["key"][0])
                enc.person_confirmed = enc.person_confirmed or bool(o.get("confirmed"))
            elif o["name"] in OBJECT_LABELS and o["conf"] >= self.object_min_conf:
                enc.objects[o["name"]] = max(o["conf"], enc.objects.get(o["name"], 0))
        for h in obs["hazards"]:
            enc.hazards[h["label"]] = max(h["conf"], enc.hazards.get(h["label"], 0))
        for text in obs["qr"]:
            enc.qr.setdefault(text, obs["key"])
        if frame["scene"] is not None:
            enc.scene = frame["scene"]

    def _encounter(self, cell, now):
        for e in reversed(self.encounters):
            if e.cell == cell and now - e.last <= self.gap:
                e.last = now
                return e
        e = Encounter(len(self.encounters) + 1, cell, now)
        self.encounters.append(e)
        return e

    # ---- assignment
    def _link_loose_qr(self):
        for q in self.loose_qr:
            if q["done"] or self.now - q["first_now"] < QR_LINK_SETTLE_S:
                continue
            near = [v for v in self.scan_visits
                    if min(abs(s - q["stamp_ns"]) for s in v["stamps_ns"]) <= QR_LINK_WINDOW_S * 1e9]
            if not near:
                q["done"] = True
                self._candidate("M03", "held", reason=f"QR {q['text']!r}: no room scan within "
                                                      f"{QR_LINK_WINDOW_S:g}s of it; not linked to a cell")
                continue
            cells = {v["cell"] for v in near}
            if len(cells) > 1:
                q["done"] = True
                self._candidate("M03", "held", reason=f"QR {q['text']!r}: rooms {sorted(cells)} scanned in the same "
                                                      f"window; ambiguous, not linked")
                continue
            visit = near[0]
            here = self.map_pose_at(q["stamp_ns"]) if self.map_pose_at else None
            there = visit.get("pose")
            if here is None or not there:                     # TF lag (1-6 s) is long past by now
                q["done"] = True
                self._candidate("M03", "held", reason=f"QR {q['text']!r}: no map pose at its frame or the scan's; "
                                                          f"not linked")
                continue
            gap = math.hypot(here[0] - there[0], here[1] - there[1])
            q["done"] = True
            if gap > QR_LINK_POSE_M:
                self._candidate("M03", "held", reason=f"QR {q['text']!r}: dog {gap:.2f} m from where it scanned room "
                                                      f"{visit['cell']}; not the same visit, not linked")
                continue
            self.linked_qr_cells.add(visit["cell"])
            log(f"QR {q['text']!r} linked to room {visit['cell']} (scan {visit['scan_id']}, same visit, "
                f"{gap:.2f} m from the scan pose)")
            self._candidate("M03", "linked", reason=f"QR {q['text']!r} -> room {visit['cell']} "
                                                    f"(scan {visit['scan_id']}, {gap:.2f} m)")

    def evaluate(self):
        self._link_loose_qr()
        by_type = {}
        for m in self.missions.values():
            by_type.setdefault(m["task_type"], []).append(m)
        scened = [e for e in self.encounters if e.scene]
        for m in by_type.get("object", []):
            self._first(m, [(e, m["label"]) for e in scened if m["label"] in e.objects])
        for m in by_type.get("hazard", []):
            self._first(m, [(e, lab) for e in scened for lab in e.hazards if m["label"] in (None, lab)])
        self._survivors(by_type.get("survivor", []))
        for m in by_type.get("qr", []):
            if self._open(m["task_id"]) and self.qr_seen:
                if len(self.qr_seen) > 1:      # operator 2026-11-27: post only confirmed answers, never a pick
                    self._note(m["task_id"], f"{len(self.qr_seen)} distinct QR payloads {sorted(self.qr_seen)}; "
                                             f"held (no confirmed one)")
                    continue
                text, room = next(iter(self.qr_seen.items()))
                self._report(m, room or "", "", None, text)

    def _open(self, tid):
        return tid not in self.state

    def _first(self, mission, candidates):
        if not self._open(mission["task_id"]):
            return
        candidates = [(e, lab) for e, lab in candidates
                      if not mission["room"] or e.scene["type"] == mission["room"]]
        if candidates:
            enc, label = candidates[0]
            self._report(mission, enc.scene["type"], label, cell_centre(enc.cell))

    def survivor_cells(self):
        """Confirmed survivors, one per cell (booklet FAQ 8: at most one per cell)."""
        cells = {}
        for e in self.encounters:
            if e.scene and (len(e.person_hits) >= self.person_min_hits or e.person_confirmed):
                cells.setdefault(e.cell, e)
        return cells

    def qr_cells(self):
        return {e.cell for e in self.encounters if e.qr} | self.linked_qr_cells

    def _survivors(self, missions):
        conscious = next(m for m in missions if m["needs_qr"])
        other = next(m for m in missions if not m["needs_qr"])
        survivors, qr_cells = self.survivor_cells(), self.qr_cells()
        qr_distance = {c: min((chebyshev(c, q) for q in qr_cells), default=math.inf) for c in survivors}
        near_qr = sorted((d, c) for c, d in qr_distance.items() if d <= self.qr_survivor_max_cells)
        qr_visit_open = any(e.qr and self.now - e.last <= self.gap for e in self.encounters)
        # Operator 2026-11-27 ("It should just report"): a survivor whose own room scan showed a qr_code is M03 at
        # once, decoded text or not (M03's payload carries no message; the QR text goes in M04). Two such rooms: held.
        qr_rooms = sorted(c for c in survivors if c in self.scan_qr_cells)
        if self._open(conscious["task_id"]) and len(qr_rooms) == 1:
            cell = qr_rooms[0]
            log(f"{conscious['task_id']}: survivor {cell} with a QR in her room scan")
            self._report(conscious, survivors[cell].scene["type"], "", cell_centre(cell))
        elif self._open(conscious["task_id"]) and len(qr_rooms) > 1:
            self._note(conscious["task_id"], f"{len(qr_rooms)} survivors with a QR in their room {qr_rooms}; "
                                             f"held (no confirmed one)")
        if self._open(conscious["task_id"]) and not qr_rooms:
            # The QR may be decoded from the neighbouring cell (booklet FAQ 4), so take the survivor
            # closest to it; two equally close survivors stay pending.
            # A survivor in the QR's own cell is final; a neighbour only once the QR visit is over,
            # since the survivor beside the QR may not be confirmed yet when the code is decoded.
            if near_qr and (near_qr[0][0] == 0 or not qr_visit_open) and (
                    len(near_qr) == 1 or near_qr[0][0] < near_qr[1][0]):
                cell = near_qr[0][1]
                self._report(conscious, survivors[cell].scene["type"], "", cell_centre(cell))
            else:
                self._note(conscious["task_id"], f"{len(survivors)} confirmed survivor cell(s), {len(near_qr)} next "
                                                 f"to a decoded/linked QR")
        if not self._open(other["task_id"]):
            return
        # M05 (operator 2026-11-27: "M05 waits until M03 is confirmed" - "This should not be the case"): the other
        # survivor is one with NO QR anywhere near her: her room scan(s) reported no qr_code, no decoded or linked
        # QR within qr_survivor_max_cells, and not M03's cell. A survivor whose room scan showed a QR stays an M03
        # candidate and is never M05. Unscanned (walking) survivors only count once M03 is out.
        m03 = self.reports.get(conscious["task_id"]) or {}
        m03_cell = tuple(m03["_cell"]) if m03.get("_cell") else None
        free = sorted(c for c in survivors if c != m03_cell and qr_distance[c] > self.qr_survivor_max_cells
                      and c not in self.scan_qr_cells and (c in self.scanned_cells or m03_cell is not None))
        if len(free) == 1:
            cell = free[0]
            log(f"{other['task_id']}: survivor {cell} has no QR near her (room scan saw none); the other survivor")
            self._report(other, survivors[cell].scene["type"], "", cell_centre(cell))
        elif len(free) > 1:
            self._note(other["task_id"], f"{len(free)} survivors without a QR {free}; held (no confirmed one)")
        elif not self._m05_fallback(other, survivors, qr_distance, qr_visit_open):
            beside = sorted(c for c in survivors if c != m03_cell)
            self._note(other["task_id"], f"{len(survivors)} confirmed survivor cell(s); none without a QR near her"
                                         + (f" ({beside} had a QR in view or beside her: M03 candidates)" if beside
                                            else ""))

    def _m05_fallback(self, other, survivors, qr_distance, qr_visit_open):
        """M03 is still open: send a survivor with no decoded QR beside it as M05 once it has been confirmed for
        m05_after_s and no QR visit is going on. The one farthest from any QR, then the earliest seen."""
        if self.m05_after_s is None or not self._open(other["task_id"]) or qr_visit_open:
            return False
        ready = sorted((-qr_distance[c], e.first, c) for c, e in survivors.items()
                       if qr_distance[c] > self.qr_survivor_max_cells and self.now - e.first >= self.m05_after_s)
        if not ready:
            return False
        cell = ready[0][2]
        log(f"{other['task_id']}: survivor {cell} has no QR beside it after {self.m05_after_s:g}s; "
            f"reporting it as the survivor without a QR")
        self._report(other, survivors[cell].scene["type"], "", cell_centre(cell))
        return True

    # ---- reporting
    def _candidate(self, tid, status, **fields):
        """Operator 2026-11-27: compute and keep everything - each held or posted answer, with why."""
        path = getattr(self, "candidates_path", None)
        if path:
            with Path(path).open("a") as f:
                f.write(json.dumps(dict(time=time.time(), task_id=tid, status=status, **fields), default=str) + "\n")

    def _note(self, tid, text):
        if self.notes.get(tid) != text:
            self.notes[tid] = text
            log(f"{tid} pending: {text}")
            self._candidate(tid, "held", reason=text)

    def _report(self, mission, scene_type, label, pos, message=""):
        payload = {"group_id": self.group_id, "task_id": mission["task_id"],
                   "task_type": mission["task_type"], "scene_type": scene_type,
                   "label": label, "pos": pos, "message": message}
        validate_payload(payload)
        if payload in self.rejected.get(mission["task_id"], []):
            self._candidate(mission["task_id"], "held", reason="server already refused this exact report",
                            payload=payload)
            return                    # the server already refused this exact report
        self._candidate(mission["task_id"], "dry_run" if getattr(self.poster, "dry_run", False) else "posted", payload=payload)
        self.state[mission["task_id"]] = "sending"
        self.reports[mission["task_id"]] = dict(
            payload, _cell=[round(p / CELL_M) for p in pos] if pos else None)
        self.poster.send(payload, self.on_ack)

    def on_ack(self, payload, accepted, response):
        tid = payload["task_id"]
        if accepted:
            self.state[tid] = "dry_run" if response == "dry_run" else "accepted"
            log(f"{tid} {self.state[tid]}")
        else:
            self.state.pop(tid, None)  # reopen: a different later observation may be sent
            self.rejected.setdefault(tid, []).append(payload)
            log(f"{tid} rejected: {response}")


def validate_payload(p):
    assert tuple(p) == WIRE_KEYS, f"keys {tuple(p)}"
    assert p["group_id"].startswith("group08_"), p["group_id"]
    assert p["task_type"] in TASK_TYPES
    assert p["scene_type"] in SCENE_TYPES + ("",)
    assert p["label"] in OBJECT_LABELS + HAZARD_LABELS + ("",)
    assert p["pos"] is None or (len(p["pos"]) == 2 and all(isinstance(v, float) for v in p["pos"]))
    assert isinstance(p["message"], str)
    if p["task_type"] == "qr":
        assert p["message"] and p["pos"] is None and p["label"] == ""
    else:
        assert p["message"] == "" and p["pos"] is not None and p["scene_type"]
    if p["task_type"] == "survivor":
        assert p["label"] == ""


# ------------------------------------------------------------------ HTTP
def accepted_response(payload, status, text):
    try:
        body = json.loads(text)
    except ValueError:
        return False
    return (status == 200 and isinstance(body, dict) and body.get("status") == "accepted"
            and body.get("task_id") == payload["task_id"])


class Poster:
    """POSTs on a background thread, one at a time. Network errors retry; answers are final."""

    def __init__(self, url, dry_run, log_path, timeout=5.0, retry_s=2.0):
        self.url, self.dry_run, self.timeout, self.retry_s = url, dry_run, timeout, retry_s
        self.log_path = Path(log_path) if log_path else None
        self.queue = queue.Queue()
        self.acks = queue.Queue()      # (callback, payload, accepted, response) for the main thread
        self.thread = None
        self.stopping = False

    def send(self, payload, callback):
        print("PAYLOAD", json.dumps(payload), flush=True)
        if self.dry_run:
            self._record(payload, None, "dry_run")
            callback(payload, True, "dry_run")
            return
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        self.queue.put((payload, callback))

    def drain(self):
        """Deliver acknowledgements on the caller's thread (call from the ROS timer)."""
        while True:
            try:
                callback, payload, accepted, response = self.acks.get_nowait()
            except queue.Empty:
                return
            callback(payload, accepted, response)

    def wait(self, timeout=10.0):
        end = time.monotonic() + timeout
        while (self.queue.unfinished_tasks and time.monotonic() < end):
            time.sleep(0.05)
        self.drain()

    def _run(self):
        while not self.stopping:
            payload, callback = self.queue.get()
            while True:
                try:
                    req = Request(self.url, json.dumps(payload, allow_nan=False).encode(), method="POST",
                                  headers={"Content-Type": "application/json"})
                    with urlopen(req, timeout=self.timeout) as r:
                        status, text = r.status, r.read(8192).decode(errors="replace")
                except HTTPError as exc:
                    status, text = exc.code, exc.read(8192).decode(errors="replace")
                except (URLError, OSError) as exc:
                    log(f"{payload['task_id']} POST failed ({exc}); retrying in {self.retry_s:g}s")
                    time.sleep(self.retry_s)
                    continue
                break
            log(f"{payload['task_id']} HTTP {status}: {text}")
            self._record(payload, status, text)
            self.acks.put((callback, payload, accepted_response(payload, status, text), text))
            self.queue.task_done()

    def _record(self, payload, status, response):
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"time": time.time(), "url": self.url, "payload": payload,
                                    "http_status": status, "response": response}) + "\n")


# ------------------------------------------------------------------ ROS
def run_ros(args, missions, poster, group_id):
    import rclpy
    from rclpy.duration import Duration
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init()
    node = rclpy.create_node("group08_mission_reporter")
    tf = Buffer(cache_time=Duration(seconds=args.tf_history))
    TransformListener(tf, node)
    origin = [args.origin]
    if args.origin:
        log(f"Arena origin from --origin: x={args.origin[0]:.3f} y={args.origin[1]:.3f} "
            f"yaw={math.degrees(args.origin[2]):.1f}deg")
    clock = ClockMonitor()

    def planar(t):
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def pose_at(stamp_ns):
        if origin[0] is None:
            return None
        try:   # exactly the capture time; no fallback to a later pose
            t = tf.lookup_transform(args.map_frame, args.base_frame, Time(nanoseconds=stamp_ns))
        except TransformException:
            return None
        return to_arena(planar(t), origin[0])

    tracker = MissionTracker(missions, poster, group_id, pose_at,
                             person_min_hits=args.person_min_hits, tf_history_s=args.tf_history,
                             pos_source=args.pos_source,
                             m05_after_s=args.m05_after if args.m05_after >= 0 else None)
    seed_accepted(tracker, args.accepted_from)

    def map_pose_at(stamp_ns):                  # map frame, for linking a walking QR to the room scanned there
        try:
            return planar(tf.lookup_transform(args.map_frame, args.base_frame, Time(nanoseconds=stamp_ns)))
        except TransformException:
            return None
    tracker.map_pose_at = map_pose_at
    tracker.candidates_path = Path(args.report_log).with_name("candidates.jsonl") if args.report_log else None
    log(f"Position source: {args.pos_source}; survivor without a QR goes out as M05 after "
        f"{args.m05_after:g}s" if args.m05_after >= 0 else f"Position source: {args.pos_source}")
    # Dry-run room scans (bench tests) only count when nothing is POSTed.
    tracker.accept_dry_run_scans = poster.dry_run

    def capture_origin():
        if origin[0] is not None:
            return
        try:
            origin[0] = planar(tf.lookup_transform(args.map_frame, args.base_frame, Time()))
        except TransformException:
            return
        log(f"Arena origin = current {args.map_frame}->{args.base_frame}; start this with the dog "
            f"in the start cell facing arena +X, or pass --origin: x={origin[0][0]:.3f} "
            f"y={origin[0][1]:.3f} yaw={math.degrees(origin[0][2]):.1f}deg")

    def on_det(msg):
        try:
            obs = parse_detection(msg.data)
        except (ValueError, KeyError, TypeError) as exc:
            log("bad detection message:", exc)
            return
        clock.add(obs["key"][0], time.time_ns())
        if obs["qr"] or obs["hazards"] or any(o["name"] != "person" or o["conf"] >= .5 for o in obs["objects"]):
            log(f"frame {obs['frame']} objects={[o['name'] for o in obs['objects']]} qr={obs['qr']} "
                f"hazards={[h['label'] for h in obs['hazards']]}")
        tracker.on_detection(obs, time.monotonic())

    def on_scene(msg):
        try:
            tracker.on_scene(parse_scene(msg.data), time.monotonic())
        except (ValueError, KeyError, TypeError) as exc:
            log("bad scene message:", exc)

    def on_event(msg):
        try:
            event = json.loads(msg.data)
            if event.get("event") == "scene_result":
                on_scene(msg)       # same join; /scenes carries a copy for older consumers
            elif event.get("event") == "scene_failed":
                log(f"scene request {event.get('request_id')} failed: {event.get('error')}")
        except (ValueError, AttributeError) as exc:
            log("bad scene event:", exc)

    def on_room_scan(msg):
        try:
            scan = parse_room_scan(msg.data)
        except (ValueError, TypeError) as exc:
            log("bad room scan message:", exc)
            return
        if scan is not None:
            tracker.on_room_scan(scan, time.monotonic())

    best_effort = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(String, ROOM_SCAN_TOPIC, on_room_scan,
                             QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
    node.create_subscription(String, DETECTIONS_TOPIC, on_det, best_effort)
    if args.scene_source == "scenes":
        node.create_subscription(String, SCENES_TOPIC, on_scene, best_effort)
    else:
        node.create_subscription(String, EVENTS_TOPIC, on_event,
                                 QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
    node.create_timer(0.5, capture_origin)
    node.create_timer(0.5, lambda: (poster.drain(), tracker.tick(time.monotonic())))
    node.create_timer(30.0, lambda: log("clock", clock.verdict(), clock.summary(), "stats", tracker.stats))
    log("pos = room cell only: room_scan_result.cell [i, j] -> [i*1.02, j*1.02]; walking sightings give QR text only")
    log(f"Listening on {DETECTIONS_TOPIC} and "
        f"{SCENES_TOPIC if args.scene_source == 'scenes' else EVENTS_TOPIC}; "
        f"{'DRY RUN' if poster.dry_run else 'POST ' + poster.url} as {group_id}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        poster.wait()
        summary(tracker)
        log("clock", clock.verdict(), clock.summary())
        node.destroy_node()
        rclpy.try_shutdown()


def parse_origin(text):
    x, y, yaw = (float(v) for v in text.split(","))
    return x, y, math.radians(yaw)


def seed_accepted(tracker, paths):
    """Mark tasks the server already ACCEPTED in earlier report logs of this run as done, so a restarted reporter
    (e.g. manual scene fallback after roam exits, or dry -> live) never re-sends them. Dry runs are not accepted."""
    for path in paths or []:
        try:
            lines = Path(path).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                rec = json.loads(line)
                if accepted_response(rec["payload"], rec.get("http_status"), rec.get("response") or ""):
                    tracker.state[rec["payload"]["task_id"]] = "accepted"
            except (ValueError, KeyError, TypeError):
                continue
    if tracker.state:
        log(f"already accepted earlier in this run (not re-sent): {sorted(tracker.state)}")


def run_replay(path, missions, poster, group_id, accepted_from=None):
    """JSONL lines: {"topic", "data", "t" (s), "pose": [x, y, yaw] arena | null} - pose at capture."""
    poses = {}
    tracker = MissionTracker(missions, poster, group_id, lambda stamp: poses.get(stamp))
    seed_accepted(tracker, accepted_from)
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if rec["topic"] == DETECTIONS_TOPIC:
            obs = parse_detection(rec["data"])
            if rec.get("pose"):
                poses[obs["key"][0]] = tuple(rec["pose"])
            tracker.on_detection(obs, rec["t"])
        elif rec["topic"] == ROOM_SCAN_TOPIC:
            scan = parse_room_scan(rec["data"])
            if scan is not None:
                for view in scan.get("views") or []:
                    if rec.get("poses") and view.get("view_id") in rec["poses"]:
                        poses[view["capture_stamp_ns"]] = tuple(rec["poses"][view["view_id"]])
                tracker.accept_dry_run_scans = poster.dry_run
                tracker.on_room_scan(scan, rec["t"])
        elif rec["topic"] == SCENES_TOPIC or json.loads(rec["data"]).get("event") == "scene_result":
            tracker.on_scene(parse_scene(rec["data"]), rec["t"])
    poster.wait()
    summary(tracker)
    return tracker


def summary(tracker):
    for tid in tracker.missions:
        log(f"{tid}: {tracker.state.get(tid, 'not reported')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-id", help=f"Registered Group 08 ID (default: {DEFAULT_GROUP_FILE.name})")
    p.add_argument("--endpoint", default="A", help="A, B, or a full URL")
    p.add_argument("--live", action="store_true", help="Send POSTs (default is dry run)")
    p.add_argument("--missions-url", help="Organizer retrieval endpoint (one GET for all five)")
    p.add_argument("--missions-file", help="Local mission JSON (default: booklet v6.4 list)")
    p.add_argument("--replay", help="Replay a JSONL recording instead of subscribing")
    p.add_argument("--report-log", default="mission_reports.jsonl")
    p.add_argument("--accepted-from", action="append", metavar="LOG",
                   help="earlier mission_reports.jsonl of this run: tasks accepted there are not re-sent (repeatable)")
    p.add_argument("--scene-source", choices=("events", "scenes"), default="events",
                   help="Read Claude results from scene_events (RELIABLE) or the older /scenes topic")
    p.add_argument("--map-frame", default="map")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--tf-history", type=float, default=120.0,
                   help="Seconds of TF and detections kept for late scenes and late TF")
    p.add_argument("--origin", type=parse_origin, metavar="X,Y,YAW_DEG",
                   help="Start pose in the map frame (from the operator launcher); default: pose at startup")
    p.add_argument("--person-min-hits", type=int, default=3)
    p.add_argument("--m05-after", type=float, default=-1.0, metavar="S",
                   help="Send a survivor with no QR beside it as M05 after S seconds while M03 is open "
                        "(negative: wait for M03 as before)")
    p.add_argument("--pos-source", choices=("room_cell", "base_xy"), default="room_cell",
                   help="room_cell (default): positions only from room photos, at the room's cell. base_xy: also "
                        "place other detections from the Pi's depth (needs PI_BASE_XY=1 on the Pi)")
    args = p.parse_args(argv)
    group_id = load_group_id(args.group_id)
    if args.live and not GROUP_ID.match(group_id):
        p.error(f"--live needs the registered group08_<4 alphanumerics>, got {group_id!r}")
    url = ENDPOINTS.get(args.endpoint.upper(), args.endpoint)
    missions = retrieve_missions(args.missions_url, args.missions_file)
    for m in missions.values():
        log(f"{m['task_id']} {m['task_type']} label={m['label']} room={m['room']}"
            + (f" needs_qr={m['needs_qr']}" if m["task_type"] == "survivor" else ""))
    poster = Poster(url, not args.live, args.report_log)
    if args.replay:
        run_replay(args.replay, missions, poster, group_id, args.accepted_from)
    else:
        run_ros(args, missions, poster, group_id)


if __name__ == "__main__":
    sys.exit(main())
