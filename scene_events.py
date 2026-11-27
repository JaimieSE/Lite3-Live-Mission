"""Confirm mission candidates across frames, deduplicate encounters, and schedule Claude scene requests.

ROS-free so it can be unit-tested. rgbd_perception.py feeds it one record per processed frame and
publishes the events it returns on /group8pi/perception/scene_events (std_msgs/String JSON):

  scene_pending  a confirmed new encounter needs a Claude request; the NUC holds navigation so the
                 dog stops (operator 2026-11-26: Claude only sees frames taken while the dog stands still)
  scene_captured the still frame was taken and sent to Claude; the NUC may release the hold
  scene_result   Claude returned a room label for that request's (still) frame
  scene_failed   the request failed, so the NUC should release any hold for request_id

Every event carries request_id, encounter_id (plus encounter_ids when one frame confirmed several
encounters), capture_stamp_ns, frame_id and trigger. Encounters are keyed by Pi capture time only;
the Pi has no pose, so the NUC deduplicates repeat encounters of one target by arena cell.
"""
from collections import deque
import itertools
import os
import json
from pathlib import Path
import time

EVENT_SCHEMA_VERSION = 1
MISSION_OBJECTS = frozenset({"hand_tools", "medical_equipment", "food", "water", "utensils"})
HAZARD_LABELS = frozenset({"high_voltage", "radioactive", "chemical", "biological", "flammable"})


def candidate_keys(objects, signs, qr_codes, person_confidence=0.5, object_confidence=0.4,
                   object_labels=MISSION_OBJECTS):
    """(kind, label) keys worth a scene request in one frame. Unidentified triangles are ignored."""
    keys = set()
    for item in objects:
        name, confidence = item.get("name"), item.get("confidence", 0)
        if name == "person" and confidence >= person_confidence:
            keys.add(("person", "person"))
        elif name in object_labels and confidence >= object_confidence:
            keys.add(("object", name))
    for sign in signs:
        if sign.get("label") in HAZARD_LABELS:
            keys.add(("warning_sign", sign["label"]))
    for code in qr_codes:
        if code.get("text"):
            keys.add(("qr", code["text"]))
    return keys


class EncounterGate:
    """A key becomes a confirmed encounter after `confirm` hits within the last `window` frames.

    One encounter lasts while its key keeps appearing within `gap_s` (Pi capture time). A key
    that returns after a longer gap starts a new encounter, which must be confirmed again.
    """

    def __init__(self, confirm=2, window=3, gap_s=8.0):
        if not 1 <= confirm <= window:
            raise ValueError("confirm must be between 1 and window")
        self.confirm, self.window, self.gap_ns = confirm, window, int(gap_s * 1e9)
        self.recent = deque(maxlen=window)     # sets of keys from the last frames
        self.active = {}                       # key -> dict(id, last_ns, confirmed)
        self.numbers = itertools.count(1)

    def update(self, keys, stamp_ns):
        """Returns encounters confirmed by this frame: [(encounter_id, (kind, label))]."""
        for key in [k for k, e in self.active.items() if stamp_ns - e["last_ns"] > self.gap_ns]:
            del self.active[key]
            for seen in self.recent:          # an expired encounter's frames cannot confirm the next
                seen.discard(key)
        self.recent.append(set(keys))
        confirmed = []
        for key in sorted(keys):
            encounter = self.active.get(key)
            if encounter is None:
                encounter = self.active[key] = dict(id=None, last_ns=stamp_ns, confirmed=False)
            encounter["last_ns"] = stamp_ns
            hits = sum(key in frame for frame in self.recent)
            if not encounter["confirmed"] and hits >= self.confirm:
                encounter["confirmed"] = True
                encounter["id"] = f"{key[0]}-{next(self.numbers)}"
                confirmed.append((encounter["id"], key))
        return confirmed


class SceneScheduler:
    """At most one Claude request in flight and at least `min_interval_s` between request starts.

    Operator 2026-11-26: Claude only sees frames taken while the dog stands still. A request starts
    with scene_pending (the NUC holds navigation), then waits for the first still frame (`still=True`
    in offer) and sends that one with scene_captured. No still frame within `still_wait_s` ends the
    request with scene_failed "no_still_frame".

    Encounters confirmed while busy wait in a short queue (newest frame per batch). When the queue
    is full the oldest batch is dropped with a scene_failed event, so the NUC never waits on it;
    no scene_pending is published for a batch until its request actually starts.
    """

    def __init__(self, submit, min_interval_s=4.0, queue=2, hold_timeout_s=5.0, model=None,
                 clock=time.monotonic, wall_ns=time.time_ns, on_start=None, still_wait_s=15.0):
        self.submit, self.min_interval_s, self.hold_timeout_s = submit, min_interval_s, hold_timeout_s
        self.still_wait_s = still_wait_s
        self.on_start = on_start     # called with (request, image) just before the Claude request
        self.model, self.clock, self.wall_ns = model, clock, wall_ns
        self.queue = deque()
        self.queue_size = queue
        self.waiting = None          # request dict waiting for a still frame (scene_pending sent)
        self.pending = None          # (future, request dict)
        self.last_start = -float("inf")
        self.numbers = itertools.count(1)
        self.stats = dict(requested=0, results=0, failed=0, dropped=0, no_still=0)

    def offer(self, image, frame_info, encounters, still=True):
        """Queue one frame's confirmed encounters; send `image` if a request waits and the dog is
        still in this frame. Returns events to publish now."""
        events = []
        if encounters:
            if len(self.queue) >= self.queue_size:
                dropped = self.queue.popleft()
                self.stats["dropped"] += 1
                events.append(self._event("scene_failed", dropped, error="dropped_rate_limited"))
            self.queue.append(dict(frame=frame_info, encounters=encounters, request_id=None))
        events += self.poll()
        if self.waiting is not None and still and image is not None:
            events.append(self._capture(image, frame_info))
        return events

    def poll(self):
        events = []
        if self.pending is not None and self.pending[0].done():
            future, request = self.pending
            self.pending = None
            try:
                scene = future.result()
            except Exception as exc:  # network, API and validation errors all release the hold
                self.stats["failed"] += 1
                events.append(self._event("scene_failed", request,
                                          error=f"{type(exc).__name__}: {exc}"[:200]))
            else:
                self.stats["results"] += 1
                events.append(self._event("scene_result", request, scene=scene))
        if self.waiting is not None and self.clock() - self.waiting["since"] > self.still_wait_s:
            request, self.waiting = self.waiting, None
            self.stats["no_still"] += 1
            events.append(self._event("scene_failed", request, error="no_still_frame"))
        if (self.pending is None and self.waiting is None and self.queue
                and self.clock() - self.last_start >= self.min_interval_s):
            request = self.queue.pop()           # newest frame first; older queued batches go stale
            stale = list(self.queue)
            self.queue.clear()
            request["request_id"] = f"scene-{next(self.numbers)}"
            for old in stale:                    # fold older encounters into this request
                request["encounters"] = old["encounters"] + request["encounters"]
            self.last_start = request["since"] = self.clock()
            self.waiting = request
            events.append(self._event("scene_pending", request, hold_timeout_s=self.hold_timeout_s))
        return events

    def cancel(self, error="shutdown"):
        """scene_failed for a request still waiting for a still frame (releases the NUC hold)."""
        if self.waiting is None:
            return []
        request, self.waiting = self.waiting, None
        return [self._event("scene_failed", request, error=error)]

    def _capture(self, image, frame_info):
        request, self.waiting = self.waiting, None
        request["frame"] = frame_info            # results carry the still frame's stamp
        self.stats["requested"] += 1
        if self.on_start is not None:
            try:
                self.on_start(request, image)
            except Exception as exc:  # logging must never block the request or the hold
                print(f"scene log failed for {request['request_id']}: {exc}", flush=True)
        self.pending = (self.submit(image), request)
        return self._event("scene_captured", request,
                           waited_s=round(self.clock() - request["since"], 2))

    def _event(self, kind, request, **extra):
        encounters = request["encounters"]
        trigger_kind, trigger_label = encounters[-1][1]
        event = dict(event=kind, schema_version=EVENT_SCHEMA_VERSION,
                     request_id=request["request_id"] or f"dropped-{encounters[-1][0]}",
                     encounter_id=encounters[-1][0], encounter_ids=[e[0] for e in encounters],
                     capture_stamp_ns=request["frame"]["capture_stamp_ns"],
                     frame_id=request["frame"]["frame_id"], frame=request["frame"].get("frame"),
                     trigger=dict(kind=trigger_kind, label=trigger_label),
                     triggers=[dict(encounter_id=e[0], kind=e[1][0], label=e[1][1]) for e in encounters],
                     model=self.model, published_ns=self.wall_ns())
        event.update(extra)
        return event


class SceneLog:
    """Keep what Claude was shown and what it answered, one folder per perception-service start.

    <root>/<YYYYmmdd-HHMMSS>/<request_id>.jpg  the image exactly as sent (scaled to at most 768 px)
    <root>/<YYYYmmdd-HHMMSS>/requests.jsonl    one line per finished request: trigger, encounters, frame,
                                               capture time, room label, evidence or error, latency
    Request ids restart at scene-1 whenever the service restarts, hence the per-start folder.
    """

    def __init__(self, root, max_side=768, wall_ns=time.time_ns):
        self.dir = Path(root) / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_side, self.wall_ns = max_side, wall_ns
        self.started_ns = {}

    def started(self, request, image):
        import cv2
        height, width = image.shape[:2]
        scale = min(1, self.max_side / max(height, width))
        if scale < 1:
            image = cv2.resize(image, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(self.dir / f"{request['request_id']}.jpg"), image)
        self.started_ns[request["request_id"]] = self.wall_ns()

    def finished(self, event):
        if event["event"] not in ("scene_result", "scene_failed"):
            return
        rid = event["request_id"]
        started = self.started_ns.pop(rid, None)
        scene = event.get("scene") or {}
        image = self.dir / f"{rid}.jpg"
        record = dict(request_id=rid, outcome=event["event"].removeprefix("scene_"),
                      requested_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started / 1e9))
                      if started else None,
                      latency_s=round((self.wall_ns() - started) / 1e9, 2) if started else None,
                      trigger=event.get("trigger"), triggers=event.get("triggers"),
                      frame=event.get("frame"), capture_stamp_ns=event.get("capture_stamp_ns"),
                      frame_id=event.get("frame_id"), model=event.get("model"),
                      room=scene.get("type"), evidence=scene.get("evidence"), error=event.get("error"),
                      image=image.name if image.exists() else None)
        with (self.dir / "requests.jsonl").open("a") as out:
            out.write(json.dumps(record) + "\n")


# Auto-mode gate (operator 2026-11-26): Claude scene requests only while autonomous navigation is
# driving, not in manual mode, unless PI_SCENE_TRIGGER=always. publish_perception.py (NUC domain)
# watches /nav_cmd_vel and records the non-zero commands of the last MOVES_KEEP_S in AUTO_FILE;
# rgbd_perception.py (Pi loopback domain) checks it with auto_allows_scene before each request and
# with still_at before sending a frame (the dog stands still: no navigation command for still_s).
AUTO_FILE = Path(os.getenv("PI_AUTO_FILE") or "/dev/shm/group8pi_auto.json")
NAV_COMMAND_TOPIC = "/nav_cmd_vel"
MOVES_KEEP_S = 20.0


def nav_command_active(linear_x, linear_y, angular_z, linear_eps=0.02, angular_eps=0.05):
    """True when a navigation Twist actually asks the robot to move."""
    return abs(linear_x) > linear_eps or abs(linear_y) > linear_eps or abs(angular_z) > angular_eps


def write_auto_state(path, nav_wall, moves=None):
    """Atomically record the wall time (s) of the last non-zero autonomous navigation command and the
    recent ones (`moves`, wall seconds, oldest first)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    moves = [round(m, 3) for m in (moves if moves is not None else [nav_wall])]
    tmp.write_text(json.dumps({"nav_wall": nav_wall, "moves": moves, "topic": NAV_COMMAND_TOPIC}))
    os.replace(tmp, path)


def auto_allows_scene(path, now, hold_s=10.0):
    """(allowed, reason): allowed when navigation commanded motion within the last hold_s seconds."""
    try:
        nav_wall = json.loads(Path(path).read_text()).get("nav_wall")
    except (OSError, ValueError, AttributeError):
        return False, "manual mode (no autonomous navigation command seen)"
    if nav_wall is None or now - nav_wall > hold_s:
        return False, f"manual mode (no autonomous navigation command for {hold_s:g} s)"
    return True, "auto mode"


def still_at(path, stamp, still_s=1.0):
    """True when no navigation command asked the dog to move during the still_s seconds before
    `stamp` (wall seconds, the frame's capture time). Motion after the capture does not matter."""
    try:
        state = json.loads(Path(path).read_text())
        moves = state.get("moves") or [state["nav_wall"]]
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return True                      # never driven: nothing commanded the dog to move
    return not any(stamp - still_s <= m <= stamp for m in moves)
