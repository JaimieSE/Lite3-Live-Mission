"""Record live camera samples and pass their detections to the mission store."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2

from rescue_mission.store import Store


def box_overlap(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


class LiveStore:
    def __init__(self, database, interval, objectives=None, person_reid=None, on_observation=None, on_event=None):
        self.path = Path(database).expanduser().resolve()
        self.store = Store(self.path)
        self.interval = interval
        self.last_sample = None
        self.person_tracks = {}
        self.last_qr_handoff = {}
        self.last_qr_probe = -float("inf")
        self.visible_labels = set()
        self.person_reid = person_reid
        self.on_observation = on_observation
        self.on_event = on_event
        try:
            with self.store.db:
                self.store.db.execute("""
                    CREATE TABLE IF NOT EXISTS camera_frames (
                        id TEXT PRIMARY KEY,
                        observed_at TEXT NOT NULL,
                        image_jpeg BLOB NOT NULL,
                        detections_json TEXT NOT NULL
                    )
                """)
            if objectives is not None:
                seed = json.loads(objectives.expanduser().read_text(encoding="utf-8"))
                if not isinstance(seed, list) or not all(isinstance(obj, dict) for obj in seed):
                    raise ValueError("Objectives must be a JSON array of objects")
                self.store.seed(seed)
            objectives = self.store.rows("objective")
            targets = {row["target"].strip().casefold() for row in objectives}
            aliases = {label.strip().casefold() for row in objectives for label in row.get("detector_labels", [])}
            if any(row.get("task_type") == "hazard" for row in objectives):
                aliases.add("hazard_sign")
            self.objective_targets = targets
            self.allowed_labels = targets | aliases | ({"person"} if "survivor" in targets else set())
            self.qr_records = {row["text"]: row for row in self.store.rows("qr")}
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.store.close()

    def new_person_detections(self, objects, frame_time):
        self.person_tracks = {
            track_id: (box, last_seen)
            for track_id, (box, last_seen) in self.person_tracks.items()
            if frame_time - last_seen <= 1.0
        }
        people = {
            index: obj["xyxy"] for index, obj in enumerate(objects)
            if obj["label"] == "person"
        }
        matches = []
        for index, box in people.items():
            for track_id, (previous, _) in self.person_tracks.items():
                overlap = box_overlap(box, previous)
                if overlap >= 0.25:
                    matches.append((overlap, index, track_id))

        matched_detections = set()
        matched_tracks = set()
        for _, index, track_id in sorted(matches, reverse=True):
            if index in matched_detections or track_id in matched_tracks:
                continue
            self.person_tracks[track_id] = (people[index], frame_time)
            matched_detections.add(index)
            matched_tracks.add(track_id)

        new_detections = people.keys() - matched_detections
        for index in new_detections:
            self.person_tracks[uuid4().hex] = (people[index], frame_time)
        return matched_detections, bool(new_detections)

    def qr_needs_location(self, qr):
        if not qr["decoded"] or not qr["text"].strip():
            return False
        previous = self.qr_records.get(qr["text"], {})
        return not Store.valid_location(previous.get("location"))

    def record(self, frame, detections, *, handoff=None):
        frame_time = detections["frame_received"]
        qr_updates = {}
        for qr in detections["qr_codes"]:
            if not self.qr_needs_location(qr):
                continue
            text = qr["text"]
            has_position = Store.valid_location(qr.get("location"))
            retry = handoff is not None and handoff.locate_qr and (
                frame_time - self.last_qr_handoff.get(text, -float("inf")) >= self.interval
            )
            if has_position or ((text not in self.qr_records or retry) and text not in qr_updates):
                qr_updates[text] = qr
        sample_due = self.last_sample is None or frame_time - self.last_sample >= self.interval
        objects = detections["objects"]
        if self.objective_targets:
            objects = [obj for obj in objects if obj["label"].casefold() in self.allowed_labels]
        if self.person_reid is None:
            repeated_people, new_person = self.new_person_detections(objects, frame_time)
        else:
            repeated_people = self.person_reid.repeated_detections(frame, objects)
            new_person = any(obj["label"] == "person" and index not in repeated_people
                             for index, obj in enumerate(objects))
        newly_visible = False
        if handoff is not None:
            labels = {obj["label"] for obj in objects if obj["label"] != "person"}
            newly_visible = bool(labels - self.visible_labels)
            self.visible_labels = labels
        unreadable = []
        if handoff is not None and frame_time - self.last_qr_probe >= self.interval:
            unreadable = [qr for qr in detections["qr_codes"] if not qr["decoded"]]
        if not (sample_due or qr_updates or new_person or newly_visible or unreadable):
            return

        observed_at = detections["received_at_utc"]
        metadata = {key: value for key, value in detections.items() if key != "frame_received"}
        metadata["objects"] = [
            obj for index, obj in enumerate(objects)
            if index not in repeated_people
        ]
        evidence_ref = self.save_frame(frame, observed_at, metadata)
        self.last_sample = frame_time

        recovered = []
        try:
            if handoff is not None:
                self.last_qr_handoff.update(dict.fromkeys(qr_updates, frame_time))
                if unreadable:
                    self.last_qr_probe = frame_time
                recovered = handoff.locate(self, frame, detections, metadata["objects"],
                                          qr_updates.values(), unreadable)
        finally:
            observations = self.record_findings(metadata["objects"], [*qr_updates.values(), *recovered],
                                                evidence_ref, observed_at)
        return observations

    def save_frame(self, frame, observed_at, metadata):
        frame_id = uuid4().hex
        success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Could not encode camera evidence")
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO camera_frames VALUES (?, ?, ?, ?)",
                (frame_id, observed_at, jpeg.tobytes(), json.dumps(metadata, allow_nan=False)),
            )
        return f"sqlite:{self.path}#camera_frames/{frame_id}"

    def record_findings(self, objects, qr_codes, evidence_ref, observed_at):
        for qr in qr_codes:
            if qr["text"] not in self.qr_records or Store.valid_location(qr.get("location")):
                self.record_qr(qr, evidence_ref, observed_at)

        frame_id = evidence_ref.rsplit("/", 1)[-1]
        observations = []
        for index, detection in enumerate(objects):
            detection_id = f"{frame_id}-object-{index}"
            observation = {
                "id": detection_id, "entity_id": detection_id,
                "label": detection["label"], "confidence": detection["confidence"],
                "evidence_ref": evidence_ref, "verified": False, "xyxy": detection["xyxy"],
            }
            if "location" in detection:
                observation["location"] = detection["location"]
                observation["location_source"] = detection["location_source"]
                for key in ("location_evidence_ref", "location_observed_at", "location_match"):
                    if key in detection:
                        observation[key] = detection[key]
            elif detection.get("location_issue"):
                observation["location_issue"] = detection["location_issue"]
            self.apply("observation", observed_at, observation)
            observations.append(observation)
            if self.on_observation is not None:
                self.on_observation(observation)
        return observations

    def record_qr(self, qr, evidence_ref, observed_at):
        text = qr["text"]
        previous = self.qr_records.get(text)
        if previous is None:
            qr_id = "qr-" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            data = {
                "id": qr_id, "text": text, "evidence_ref": qr.get("decode_evidence_ref", evidence_ref),
                "decoder": qr["source"], "corners": qr["corners"],
            }
            if "decode_evidence_ref" in qr:
                data["wide_evidence_ref"] = evidence_ref
            first_seen = observed_at
            event_id = qr_id
        else:
            data = {key: value for key, value in previous.items() if key not in ("source", "observed_at")}
            first_seen = previous["observed_at"]
            event_id = data["id"] + "-location"

        if Store.valid_location(qr.get("location")):
            data.update(
                location=qr["location"],
                location_source=qr["location_source"],
                location_observed_at=qr.get("location_observed_at", observed_at),
                location_evidence_ref=qr.get("location_evidence_ref", evidence_ref),
            )
            if "location_match" in qr:
                data["location_match"] = qr["location_match"]
            data.pop("location_issue", None)
        elif qr.get("location_issue"):
            data["location_issue"] = qr["location_issue"]
        self.apply("qr", first_seen, data, event_id=event_id)
        self.qr_records[text] = self.store.get("qr", data["id"])

    def apply(self, kind, observed_at, data, *, event_id=None):
        event = {
            "event_id": event_id or data["id"], "type": kind, "source": "live_scene_understanding",
            "observed_at": observed_at, "data": data,
        }
        if self.store.apply(event) and self.on_event is not None:
            self.on_event(event)
