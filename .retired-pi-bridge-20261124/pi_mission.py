"""Mission decisions from the Raspberry Pi's compact perception results.

The Pi (group8pi) runs the camera, YOLO, QR, warning-sign and stair/curtain detectors and an optional
local VLM. It republishes JSON (std_msgs/String, BEST_EFFORT, VOLATILE) into ROS domain 8:
  /group8pi/perception/detections  one message per processed frame
  /group8pi/perception/scenes      asynchronous scene results for some of those frames
pi_bridge.py is the only NUC process that subscribes to them. This module is ROS-free and unit-tested:
it buffers detections, matches late scene results by (capture_stamp_ns, frame_id), groups sightings
into encounters, and decides M01-M05 from the retrieved mission descriptions. It never commands
motion and never submits a score; insufficient evidence stays pending.
"""

from collections import OrderedDict
import json
import math

DETECTIONS_TOPIC = '/group8pi/perception/detections'
SCENES_TOPIC = '/group8pi/perception/scenes'
UNKNOWN = 'unknown'


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _first(mapping, *keys, default=None):
    for key in keys:
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return mapping[key]
    return default


def _list(mapping, *keys):
    value = _first(mapping, *keys, default=[])
    return value if isinstance(value, list) else []


def _box(value):
    """[x1, y1, x2, y2] from a list or from ultralytics' {'x1', 'y1', 'x2', 'y2'}; None otherwise."""
    if isinstance(value, dict):
        value = [value.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(_number(v) for v in value):
        return [float(v) for v in value]
    return None


def frame_key(message):
    """(capture_stamp_ns, frame_id) from a top-level or nested `frame` object, else None."""
    sources = [message]
    if isinstance(message.get('frame'), dict):
        sources.insert(0, message['frame'])
    for source in sources:
        stamp, frame_id = source.get('capture_stamp_ns'), source.get('frame_id')
        if type(stamp) is int and stamp > 0 and _text(frame_id):
            return stamp, frame_id
        if isinstance(stamp, str) and stamp.isdigit() and _text(frame_id):
            return int(stamp), frame_id
    return None


def parse_detections(text):
    """Normalized frame dict or None.

    Pi format (rgbd_perception.py + publish_perception.py, schema_version 1):
      {"frame": int, "capture_stamp_ns": int, "frame_id": "<camera optical frame>",
       "objects": [{"name": "person", "class": 0, "confidence": 0.45, "box": {"x1", "y1", "x2", "y2"}}],
       "qr_codes": [{"text": str, "corners": [[x, y] x4]}],
       "warning_sign_candidates": [{"box": [x1, y1, x2, y2], "triangle": [...], "yellow_fraction", "area"}],
       "stair_curtain": {"obstacle": "stairs"|"curtain"|"unknown", "reason": "confirmed"|<why not>,
                         "stamp_ns", "distance_m", ...},
       "schema_version": 1, "source": "group8pi"}
    frame_id is the camera's frame (the same for every frame); only (capture_stamp_ns, frame_id) is unique.
    """
    try:
        message = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(message, dict):
        return None
    key = frame_key(message)
    if key is None:
        return None
    frame = message.get('frame')
    sequence = frame.get('seq', frame.get('index')) if isinstance(frame, dict) else frame
    objects, people = [], []
    for item in _list(message, 'objects', 'detections', 'yolo'):
        if not isinstance(item, dict):
            continue
        label = _first(item, 'name', 'label')        # ultralytics: 'class' is the numeric class id
        confidence = _first(item, 'confidence', 'score', 'conf')
        if not _text(label) or not _number(confidence):
            continue
        entry = dict(label=label.strip().casefold(), confidence=float(confidence),
                     bbox=_box(_first(item, 'box', 'bbox', 'xyxy')), track_id=_first(item, 'track_id', 'track'))
        (people if entry['label'] == 'person' else objects).append(entry)
    for item in _list(message, 'persons', 'people'):
        confidence = _first(item, 'confidence', 'score', 'conf', default=0.0)
        if _number(confidence):
            people.append(dict(label='person', confidence=float(confidence),
                               bbox=_box(_first(item, 'box', 'bbox', 'xyxy')),
                               track_id=_first(item, 'track_id', 'track')))
    qr = []
    for item in _list(message, 'qr_codes', 'qr', 'qrs'):
        payload = item if isinstance(item, str) else _first(item, 'text', 'payload', 'data')
        decoded = isinstance(item, str) or item.get('decoded', True) is not False
        if decoded and _text(payload) and payload not in qr:
            qr.append(payload)
    signs = []
    for item in _list(message, 'warning_signs', 'warning_sign_candidates', 'signs', 'yellow_sign_candidates'):
        if not isinstance(item, dict):
            continue
        hazard = _first(item, 'hazard', 'hazard_label', 'identified_as', 'symbol', 'label')   # 'label': Pi hazard classifier (2026-11-24)
        signs.append(dict(
            id=_first(item, 'id', 'candidate_id', 'track_id'), bbox=_box(item.get('box')),
            shape=_first(item, 'shape', 'kind', default='yellow_triangle'),
            confidence=float(item['confidence']) if _number(item.get('confidence')) else None,
            hazard=hazard.strip().casefold() if _text(hazard) and hazard.strip().casefold() != UNKNOWN else None))
    stair_curtain = []
    raw = message.get('stair_curtain')
    for item in ([raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        kind = _first(item, 'obstacle', 'kind', 'type', default=UNKNOWN)
        state = _first(item, 'reason', 'status', 'state', default=UNKNOWN)
        # Only a confirmed stairs/curtain is usable; stale, timestamp_mismatch, waiting... are unknown.
        usable = kind in ('stairs', 'curtain') and state == 'confirmed'
        stair_curtain.append(dict(kind=kind if usable else UNKNOWN, state=state if usable else UNKNOWN,
                                  raw_state=state, distance_m=item.get('distance_m') if usable else None))
    return dict(key=key, frame=sequence, objects=objects, people=people, qr=qr, signs=signs,
                stair_curtain=stair_curtain)


def parse_scene(text):
    """{'key', 'type', 'evidence', 'model', 'confidence'} or None. type is 'unknown' if not usable."""
    try:
        message = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(message, dict):
        return None
    key = frame_key(message)
    if key is None:
        return None
    scene = message.get('scene') if isinstance(message.get('scene'), dict) else message
    kind = scene.get('type')
    evidence = scene.get('evidence')
    if isinstance(evidence, list):
        evidence = [cue for cue in evidence if _text(cue)]
    elif _text(evidence):
        evidence = [evidence]
    else:
        evidence = []
    kind = kind.strip().casefold() if _text(kind) else UNKNOWN
    confidence = _first(scene, 'confidence', default=message.get('confidence'))
    if not evidence:
        kind = UNKNOWN                       # a scene label without visible evidence is not evidence
    return dict(key=key, type=kind, evidence=evidence, model=message.get('model'),
                confidence=float(confidence) if _number(confidence) else None)


def _overlap(first, second):
    try:
        left, top = max(first[0], second[0]), max(first[1], second[1])
        right, bottom = min(first[2], second[2]), min(first[3], second[3])
        inter = max(0.0, right - left) * max(0.0, bottom - top)
        union = ((first[2] - first[0]) * (first[3] - first[1])
                 + (second[2] - second[0]) * (second[3] - second[1]) - inter)
        return inter / union if union > 0 else 0.0
    except (TypeError, IndexError):
        return 0.0


class Encounter:
    """One continuous sighting of one thing (a person, an object label, a sign) across nearby frames."""

    def __init__(self, number, kind, label, key, item):
        self.id = f'{kind}-{number}'
        self.kind, self.label = kind, label
        self.keys = [key]
        self.first_ns = self.last_ns = key[0]
        self.bbox, self.track_id = item.get('bbox'), item.get('track_id')
        self.best_confidence = item.get('confidence') or 0.0
        self.hazards = {item['hazard']} if item.get('hazard') else set()

    def add(self, key, item):
        self.keys.append(key)
        self.last_ns = max(self.last_ns, key[0])
        self.bbox = item.get('bbox') or self.bbox
        self.best_confidence = max(self.best_confidence, item.get('confidence') or 0.0)
        if item.get('hazard'):
            self.hazards.add(item['hazard'])


class PiMission:
    """Buffers Pi frames, attaches scenes, and decides missions. All times are seconds unless _ns."""

    def __init__(self, tasks, *, scene_timeout_s=20.0, retention_s=300.0, encounter_gap_s=5.0,
                 person_confidence=0.4, survivor_frames=2, object_confidence=0.4, scene_confidence=0.0,
                 max_frames=5000):
        self.tasks = {task['id']: task for task in tasks}
        self.scene_timeout_s, self.retention_s = scene_timeout_s, retention_s
        self.gap_ns = int(encounter_gap_s * 1e9)
        self.person_confidence, self.survivor_frames = person_confidence, survivor_frames
        self.object_confidence, self.scene_confidence = object_confidence, scene_confidence
        self.max_frames = max_frames
        self.frames = OrderedDict()          # key -> frame dict (+ received, scene, scene_state)
        self.orphan_scenes = OrderedDict()   # key -> (received, scene) for scenes that beat their frame
        self.encounters = []
        self.qr_payloads = OrderedDict()     # payload -> first frame key
        self.stats = dict(detections=0, scenes=0, scenes_matched=0, scenes_late=0, scenes_orphaned=0,
                          scenes_expired=0, scenes_missing=0, duplicates=0, rejected=0)
        self.latest_stair_curtain = []
        self.archived_scenes = {}            # scenes of frames that aged out of the buffer
        self.count = 0

    # ---- ingestion -------------------------------------------------------------------------------
    def add_detections(self, text, received):
        frame = parse_detections(text)
        if frame is None:
            self.stats['rejected'] += 1
            return None
        key = frame['key']
        if key in self.frames:
            self.stats['duplicates'] += 1
            return key
        self.stats['detections'] += 1
        frame.update(received=received, scene=None, scene_state='pending')
        self.frames[key] = frame
        orphan = self.orphan_scenes.pop(key, None)
        if orphan is not None:
            self._attach(frame, orphan[1], late=False)
        self.latest_stair_curtain = frame['stair_curtain']
        for payload in frame['qr']:
            self.qr_payloads.setdefault(payload, key)
        for person in frame['people']:
            if person['confidence'] >= self.person_confidence:
                self._encounter('survivor', 'person', key, person)
        for item in frame['objects']:
            if item['confidence'] >= self.object_confidence:
                self._encounter('object', item['label'], key, item)
        for sign in frame['signs']:
            self._encounter('sign', 'warning_sign', key, sign)
        self.expire(received)
        return key

    def add_scene(self, text, received):
        scene = parse_scene(text)
        if scene is None:
            self.stats['rejected'] += 1
            return None
        self.stats['scenes'] += 1
        frame = self.frames.get(scene['key'])
        if frame is None:
            # The scene can beat its detections over the link; hold it briefly instead of dropping it.
            self.orphan_scenes[scene['key']] = (received, scene)
            self.stats['scenes_orphaned'] += 1
        else:
            self._attach(frame, scene, late=frame['scene_state'] == 'missing')
        self.expire(received)
        return scene['key']

    def _attach(self, frame, scene, late):
        frame['scene'], frame['scene_state'] = scene, 'matched'
        self.stats['scenes_matched'] += 1
        if late:
            self.stats['scenes_late'] += 1

    def _encounter(self, kind, label, key, item):
        best = None
        for encounter in self.encounters:
            if encounter.kind != kind or encounter.label != label:
                continue
            if key[0] - encounter.last_ns > self.gap_ns or key[0] < encounter.first_ns - self.gap_ns:
                continue
            if item.get('track_id') is not None and encounter.track_id is not None:
                score = 2.0 if item['track_id'] == encounter.track_id else -1.0
            elif item.get('bbox') is not None and encounter.bbox is not None:
                score = _overlap(item['bbox'], encounter.bbox)
                score = score if score >= 0.1 else -1.0
            else:
                score = 0.5
            if score >= 0 and (best is None or score > best[0]):
                best = (score, encounter)
        if best is not None and key not in best[1].keys:
            best[1].add(key, item)
            return best[1]
        self.count += 1
        encounter = Encounter(self.count, kind, label, key, item)
        self.encounters.append(encounter)
        return encounter

    def expire(self, now):
        for key, frame in self.frames.items():
            if frame['scene_state'] == 'pending' and now - frame['received'] > self.scene_timeout_s:
                frame['scene_state'] = 'missing'
                self.stats['scenes_missing'] += 1
        for key in [k for k, (when, _) in self.orphan_scenes.items() if now - when > self.scene_timeout_s]:
            del self.orphan_scenes[key]
            self.stats['scenes_expired'] += 1
        # Frames are dropped from the buffer, but encounters keep their keys and the scene they earned.
        while self.frames and (len(self.frames) > self.max_frames
                               or now - next(iter(self.frames.values()))['received'] > self.retention_s):
            key, frame = self.frames.popitem(last=False)
            if frame['scene'] is not None:
                self.archived_scenes[key] = frame['scene']

    # ---- evidence --------------------------------------------------------------------------------
    def scene_of(self, key):
        frame = self.frames.get(key)
        scene = frame['scene'] if frame else self.archived_scenes.get(key)
        if scene is None:
            return None
        if scene['type'] == UNKNOWN:
            return scene
        if scene['confidence'] is not None and scene['confidence'] < self.scene_confidence:
            return dict(scene, type=UNKNOWN)
        return scene

    def encounter_scene(self, encounter):
        """(scene type, evidence keys) from frames of this encounter only; ('unknown', ...) if mixed."""
        votes = {}
        pending = 0
        for key in encounter.keys:
            scene = self.scene_of(key)
            if scene is None:
                frame = self.frames.get(key)
                pending += frame is not None and frame['scene_state'] == 'pending'
                continue
            if scene['type'] != UNKNOWN:
                votes.setdefault(scene['type'], []).append(key)
        if len(votes) == 1:
            kind, keys = next(iter(votes.items()))
            return kind, keys
        if len(votes) > 1:
            return 'conflicting', sorted(k for keys in votes.values() for k in keys)
        return ('pending' if pending else UNKNOWN), []

    def survivors(self):
        """Confirmed survivor encounters, deduplicated into identities. Unconfirmed ones are reported."""
        identities, unconfirmed = OrderedDict(), []
        for encounter in self.encounters:
            if encounter.kind != 'survivor':
                continue
            scene, scene_keys = self.encounter_scene(encounter)
            confirmed = len(encounter.keys) >= self.survivor_frames
            entry = dict(encounter=encounter.id, scene=scene, frames=len(encounter.keys),
                         first=list(encounter.keys[0]), scene_evidence=[list(k) for k in scene_keys],
                         confidence=round(encounter.best_confidence, 3))
            if not confirmed or scene in (UNKNOWN, 'pending', 'conflicting'):
                entry['reason'] = ('needs survivor confirmation' if not confirmed
                                   else f'scene {scene}; identity cannot be deduplicated')
                unconfirmed.append(entry)
                continue
            # One survivor per scene: repeat sightings in the same scene are the same survivor.
            identity = identities.setdefault(f'survivor:{scene}', dict(entry, identity=f'survivor:{scene}',
                                                                       sightings=[]))
            identity['sightings'].append(encounter.id)
        return list(identities.values()), unconfirmed

    # ---- decisions -------------------------------------------------------------------------------
    def _labels(self, task):
        labels = set(task.get('labels', [])) | set(task.get('detector_labels', []))
        if _text(task.get('target')):
            labels.add(task['target'])
        return {label.casefold() for label in labels}

    def _object_decision(self, task, kind):
        labels = self._labels(task)
        best, candidates = None, []
        for encounter in self.encounters:
            if kind == 'hazard' and encounter.kind == 'sign':
                hazards = encounter.hazards & labels
                if not hazards:
                    candidates.append(dict(encounter=encounter.id, reason='warning triangle needs hazard '
                                           'identification' if not encounter.hazards else
                                           f'identified as {sorted(encounter.hazards)}, not this hazard'))
                    continue
                label = sorted(hazards)[0]
            elif encounter.kind == 'object' and encounter.label in labels:
                label = encounter.label
            else:
                continue
            scene, scene_keys = self.encounter_scene(encounter)
            entry = dict(encounter=encounter.id, label=label, scene=scene,
                         evidence=[list(k) for k in (scene_keys or encounter.keys[:1])],
                         confidence=round(encounter.best_confidence, 3))
            room = task.get('room')
            if scene in (UNKNOWN, 'pending', 'conflicting') and (task.get('scene_required') or room):
                candidates.append(dict(entry, reason=f'scene {scene}'))
                continue
            if room and scene != room:
                candidates.append(dict(entry, reason=f'scene {scene} is not {room}'))
                continue
            if best is None or entry['confidence'] > best['confidence']:
                best = entry
        return best, candidates

    def decide(self):
        decisions = {}
        identities, unconfirmed = self.survivors()
        for task_id, task in self.tasks.items():
            kind = task.get('task_type') or ('qr' if task.get('kind') == 'qr' else 'object')
            decision = dict(task_id=task_id, task_type=kind, status='pending')
            if kind == 'qr':
                pattern = task.get('qr_prefix')
                payloads = [p for p in self.qr_payloads if not pattern or p.startswith(pattern)]
                if len(payloads) == 1:
                    decision.update(status='ready', message=payloads[0],
                                    evidence=[list(self.qr_payloads[payloads[0]])])
                else:
                    decision['reason'] = ('no decoded QR yet' if not payloads
                                          else f'{len(payloads)} distinct QR payloads; needs attribution')
                    decision['candidates'] = payloads
            elif kind in ('object', 'hazard'):
                best, candidates = self._object_decision(task, kind)
                if best:
                    decision.update(status='ready', label=best['label'], scene=best['scene'],
                                    evidence=best['evidence'], confidence=best['confidence'],
                                    encounter=best['encounter'])
                else:
                    decision['reason'] = 'no matching sighting' if not candidates else 'insufficient evidence'
                decision['candidates'] = candidates
            decisions[task_id] = decision
        self._survivor_decisions(decisions, identities, unconfirmed)
        return decisions

    def _survivor_decisions(self, decisions, identities, unconfirmed):
        survivor_tasks = [t for t in self.tasks.values() if t.get('task_type') == 'survivor'
                          or (t.get('target') == 'survivor' and not t.get('task_type'))]
        described = [t for t in survivor_tasks if t.get('room')]
        other = [t for t in survivor_tasks if not t.get('room')]
        claimed = set()
        for task in described:                                    # M03
            decision = decisions[task['id']]
            match = next((i for i in identities if i['scene'] == task['room']), None)
            if match is None:
                decision['reason'] = f'no confirmed survivor in {task["room"]}'
            elif task.get('relation') and not _text(task.get('relation_evidence')):
                # The Pi reports scene type only; an entrance-relative relation needs its own evidence.
                decision.update(reason=f'survivor confirmed in {task["room"]}; relation '
                                       f'"{task["relation"]}" not observable from Pi results',
                                identity=match['identity'], evidence=match['scene_evidence'])
                claimed.add(match['identity'])
            else:
                decision.update(status='ready', scene=match['scene'], identity=match['identity'],
                                encounter=match['encounter'], evidence=match['scene_evidence'])
                claimed.add(match['identity'])
            decision['candidates'] = identities + unconfirmed
        described_rooms = {t['room'] for t in described}
        for task in other:                                        # M05
            decision = decisions[task['id']]
            pool = [i for i in identities if i['identity'] not in claimed]
            if described:
                # Distinct from the described survivor by definition when seen in another scene.
                pool = [i for i in pool if i['scene'] not in described_rooms]
            elif not claimed:
                pool = []
            if len(pool) >= 1:
                pick = pool[0]
                decision.update(status='ready', scene=pick['scene'], identity=pick['identity'],
                                encounter=pick['encounter'], evidence=pick['scene_evidence'])
                if len(pool) > 1:
                    decision['note'] = f'{len(pool)} distinct candidates; kept the first confirmed'
            else:
                decision['reason'] = 'no distinct confirmed survivor outside the described scene yet'
            decision['candidates'] = identities + unconfirmed

    def status(self):
        return dict(stats=dict(self.stats, buffered_frames=len(self.frames),
                               orphan_scenes=len(self.orphan_scenes), encounters=len(self.encounters)),
                    stair_curtain=self.latest_stair_curtain, decisions=self.decide())
