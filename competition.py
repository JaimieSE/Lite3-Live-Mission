"""Attempt configuration, arena coordinates, and improvement-only task reporting."""

import argparse
from contextlib import contextmanager
import fcntl
import json
from math import cos, floor, isfinite, sin
import os
import re
from pathlib import Path
import sqlite3
from time import sleep
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from rescue_mission.store import Store



V63 = "sutd-v6.3"
V63_TASK_TYPES = {'M01': 'object', 'M02': 'hazard', 'M03': 'survivor', 'M04': 'qr', 'M05': 'survivor'}
V63_SCENES = (
    "clothes_shop", "fitness_centre", "kitchen", "market", "workshop",
    "clinic", "cleaning_room", "furniture_shop",
)
V63_LABELS = {
    "object": ("medical_equipment", "hand_tools", "food", "water", "utensils"),
    "hazard": ("high_voltage", "radioactive", "chemical", "biological", "flammable"),
}


def validate_v63(profile):
    if re.fullmatch(r"group[0-9]{2}_[A-Za-z0-9]{4}", str(profile["group_id"])) is None:
        raise ValueError("v6.3 requires group##_XXXX")
    if not profile["scene_categories"] or any(scene not in V63_SCENES for scene in profile["scene_categories"]):
        raise ValueError("Use v6.3 scene categories")
    types = {task["id"]: task.get("task_type") for task in profile["tasks"]}
    if types != V63_TASK_TYPES:
        raise ValueError("Use the five fixed mission IDs and types, M01–M05")
    for task in profile["tasks"]:
        if task["id"] == "M05" and (task.get("room") or task.get("relation")):
            raise ValueError("M05 has no specified room or spatial relation")
        kind = task.get("task_type")
        if kind not in (*V63_LABELS, "survivor", "qr") or (kind == "qr") != (task["kind"] == "qr"):
            raise ValueError(f"Set the v6.3 task_type for {task['id']}")
        if (kind == "survivor") != (task["target"] == "survivor"):
            raise ValueError(f"Check the survivor target for {task['id']}")
        if kind in V63_LABELS and any(label not in V63_LABELS[kind] for label in task["labels"]):
            raise ValueError(f"Use the v6.3 {kind} labels for {task['id']}")
    if profile["mode"] == "competition" and profile.get("organizer_contract_confirmed") is not True:
        raise ValueError("Confirm the organizers' final task list before competition use")
    return profile

def number(value):
    return type(value) in (int, float) and isfinite(value)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def read_json(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def validate_profile(profile):
    if not isinstance(profile, dict) or profile.get("mode") not in ("practice", "competition"):
        raise ValueError("Set profile mode to practice or competition")
    group = profile.get("group_id")
    if not text(group) and type(group) is not int:
        raise ValueError("Set the registered group_id")
    tasks = profile.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, dict) for task in tasks):
        raise ValueError("Load a nonempty task list")
    ids = set()
    for task in tasks:
        if not text(task.get("id")) or task["id"] in ids or not text(task.get("target")):
            raise ValueError("Each task needs a unique ID and a target")
        ids.add(task["id"])
        if task.get("mode", "single") != "single" or task.get("kind") not in ("object", "qr"):
            raise ValueError("Scored tasks must be individual object or QR findings")
        if task["kind"] == "object":
            labels = task.get("labels")
            if not isinstance(labels, list) or not labels or not all(map(text, labels)):
                raise ValueError(f"Set the allowed report labels for {task['id']}")
        aliases = task.get("detector_labels", [])
        if not isinstance(aliases, list) or not all(map(text, aliases)):
            raise ValueError(f"Invalid detector_labels for {task['id']}")
        Store.validate_objective(task)
    sizes = profile.get("cell_size_m")
    if not isinstance(sizes, list) or len(sizes) != 2 or not all(number(s) and s > 0 for s in sizes):
        raise ValueError("Set cell_size_m to the measured X and Y cell dimensions")
    scenes = profile.get("scene_categories")
    if not isinstance(scenes, list) or not all(map(text, scenes)):
        raise ValueError("Set the organizer's scene categories")
    if any(task.get("room") and task["room"] not in scenes for task in tasks):
        raise ValueError("Task room must match an allowed scene category")
    if profile.get("contract") == V63:
        return validate_v63(profile)
    if profile.get("contract") is not None:
        raise ValueError("Unknown scoring contract")
    wire = profile.get("wire")
    keys = {"group", "task", "position", "label", "scene", "message"}
    if not isinstance(wire, dict) or not isinstance(wire.get("fields"), dict) or set(wire["fields"]) != keys:
        raise ValueError("Configure wire.fields from the scoring server contract")
    names = list(wire["fields"].values())
    if not all(map(text, names)) or len(set(names)) != len(names):
        raise ValueError("Wire field names must be distinct nonempty strings")
    if wire.get("position_format") not in ("xy_array", "xy_object"):
        raise ValueError("Set wire.position_format to xy_array or xy_object")
    acceptance = wire.get("acceptance")
    if acceptance is not None and (not isinstance(acceptance, dict)
                                   or not text(acceptance.get("field")) or "value" not in acceptance):
        raise ValueError("Wire acceptance needs a response field and value")
    if profile["mode"] == "competition" and profile.get("organizer_contract_confirmed") is not True:
        raise ValueError("Competition requires the final task list and confirmed scoring contract")
    return profile


def arena_position(location, origin, sizes, error):
    if not Store.valid_location(location) or origin is None or not number(error) or not 0 <= error <= 0.5:
        return None
    dx, dy = location["x"] - origin[0], location["y"] - origin[1]
    c, s = cos(origin[2]), sin(origin[2])
    coordinates = (c * dx + s * dy, -s * dx + c * dy)
    centers = [floor(value / size + 0.5) * size for value, size in zip(coordinates, sizes)]
    # Withhold the cell when the position uncertainty could place it across an edge.
    if any(abs(value - center) + error >= size / 2 for value, center, size in zip(coordinates, centers, sizes)):
        return None
    return [round(center, 6) for center in centers]


def prepare_navigation(state_path):
    """Claim a ready navigation run once, without clearing or restarting a live stack."""
    state = dict(line.split("=", 1) for line in state_path.read_text().splitlines() if "=" in line)
    folder = Path(state["RUN_DIR"]).resolve()
    if not (folder / "ready.json").is_file() or read_json(folder / "ready.json").get("verdict") != "READY":
        raise ValueError("Start a fresh navigation run and wait for NAVIGATION READY")
    os.killpg(int(state["PGID"]), 0)
    if (folder / "live_mission_competition.attempt").exists() or any((folder / "roam").glob("*")):
        raise ValueError("This navigation run has already roamed. Start a fresh map and route first")
    return folder


@contextmanager
def transaction(db):
    db.execute("BEGIN IMMEDIATE")
    with db:
        yield


class Competition:
    def __init__(self, store):
        self.store = store
        attempt = store.get("competition_config", "attempt")
        if attempt is None:
            raise ValueError("Database is not a competition-profile run")
        self.profile = attempt["profile"]
        self.origin = attempt["arena_origin_map"]
        self.tasks = {task["id"]: task for task in self.profile["tasks"]}

    @staticmethod
    def initialize(store, profile, origin, nav_run=None):
        if origin is not None and (len(origin) != 3 or not all(map(number, origin))):
            raise ValueError("Arena origin must contain finite x, y, and yaw")
        if profile["mode"] == "competition" and (origin is None or nav_run is None):
            raise ValueError("Competition needs fresh navigation and a measured start pose")
        with transaction(store.db):
            if store.get("competition_config", "attempt") or any(
                store.rows(kind) for kind in ("objective", "observation", "qr", "request")
            ):
                raise ValueError("Competition profile requires a fresh mission database")
            for task in profile["tasks"]:
                store.put("objective", dict(task, status="pending", findings=[]))
            store.put("competition_config", {
                "id": "attempt", "profile": profile, "arena_origin_map": origin,
                "navigation_run": str(nav_run) if nav_run else None,
            })

    @staticmethod
    def matches(task, observation):
        label = observation.get("label")
        return (label == task["target"] or label in task.get("detector_labels", [])
                or (task["target"] == "survivor" and label == "person"))

    def inspect(self):
        """Advisory candidates derived from this attempt's mapped robot observations."""
        finished = {row["id"] for row in self.store.rows("competition_best") if row["position"] is not None}
        candidates = []
        for row in self.store.rows("observation"):
            if not Store.valid_location(row.get("location")):
                continue
            ids = [task["id"] for task in self.tasks.values() if task["kind"] == "object"
                   and task["id"] not in finished and self.matches(task, row)]
            if ids:
                candidates.append(dict(observation_id=row["id"], task_ids=ids,
                                       location=row["location"], evidence_ref=row["evidence_ref"]))
        return candidates

    def stage(self, validation):
        task = self.tasks[validation["task_id"]]
        evidence_kind = "qr" if task["kind"] == "qr" else "observation"
        observation = self.store.get(evidence_kind, validation["observation_id"])
        if observation is None or not text(observation.get("evidence_ref")):
            raise ValueError("Use saved evidence from this attempt")
        report = dict(id=task["id"], observation_id=observation["id"], evidence_ref=observation["evidence_ref"],
                      position=None, measured_position=observation.get("location"))
        if evidence_kind == "qr":
            if not text(observation.get("text")):
                raise ValueError("The QR clue must be decoded")
            confidence = validation.get("identification_confidence", 1.0)
            if not number(confidence) or not 0.8 <= confidence <= 1:
                raise ValueError("Use a validated QR identification")
            report.update(message=observation["text"], confidence=confidence, entity_id=observation["id"])
        else:
            if not self.matches(task, observation):
                raise ValueError("The observation does not match the task")
            label, scene = validation["label"], validation.get("scene")
            confidence = validation["identification_confidence"]
            if label not in task["labels"] or not number(confidence) or not 0.8 <= confidence <= 1:
                raise ValueError("Use a validated identification and an allowed task label")
            if scene is not None and scene not in self.profile["scene_categories"]:
                raise ValueError("Scene category is outside the organizer's list")
            if (task.get("scene_required") or self.profile.get("contract") == V63) and scene is None:
                raise ValueError("The task requires an identified scene category")
            if task.get("room") and scene != task["room"]:
                raise ValueError("The task requires the specified room")
            if task.get("relation") and (validation.get("relation") != task["relation"]
                                         or not text(validation.get("relation_reference"))):
                raise ValueError("The task requires an entrance-relative observation")
            entity_id = validation.get("entity_id")
            if entity_id is None and task["target"] != "survivor":
                entity_id = observation["entity_id"]
            if not text(entity_id):
                raise ValueError("Provide a stable identity for this finding")
            error = validation.get("location_error_m")
            position = arena_position(observation.get("location"), self.origin, self.profile["cell_size_m"], error)
            report.update(label=label, scene=scene, confidence=confidence, entity_id=entity_id,
                          position=position, location_error_m=error if position is not None else None)

        with transaction(self.store.db):
            previous = self.store.get("competition_best", report["id"])
            if previous is not None and not improves(report, previous):
                return False
            if any(row["id"] != report["id"]
                   and self.tasks[row["id"]]["kind"] == task["kind"]
                   and (row["entity_id"] == report["entity_id"] or row["observation_id"] == report["observation_id"])
                   for row in self.store.rows("competition_best")):
                raise ValueError("This finding is already assigned to another task")
            self.store.put("competition_best", report)
        return True

    def ingest(self, validation):
        if not isinstance(validation, dict):
            raise ValueError("Expected a validation object")
        if validation.get("type") != "coverage":
            return self.stage(validation)
        cells = validation.get("searched_cells")
        if not isinstance(cells, list) or not all(
            isinstance(cell, list) and len(cell) == 2 and all(type(index) is int for index in cell)
            for cell in cells
        ):
            raise ValueError("Coverage needs a list of integer grid cell indices")
        with transaction(self.store.db):
            previous = self.store.get("competition_coverage", "coverage") or {}
            known = set(map(tuple, previous.get("searched_cells", [])))
            updated = known | set(map(tuple, cells))
            if updated == known:
                return False
            self.store.put("competition_coverage", dict(id="coverage", searched_cells=sorted(updated)))
        return True

    def collect_qr(self):
        if self.profile["mode"] != "practice" and self.profile.get("contract") != V63:
            return
        qr_tasks = [task for task in self.tasks.values() if task["kind"] == "qr"]
        if len(qr_tasks) != 1:
            return
        clues = self.store.rows("qr")
        # Multiple distinct clues require task-specific attribution.
        if len(clues) == 1:
            self.stage({"task_id": qr_tasks[0]["id"], "observation_id": clues[0]["id"]})

    def payload(self, report):
        if self.profile.get("contract") == V63:
            kind = self.tasks[report["id"]]["task_type"]
            return {
                "group_id": self.profile["group_id"],
                "task_id": report["id"],
                "task_type": kind,
                "scene_type": report.get("scene") or "",
                "label": (report.get("label") or "") if kind in V63_LABELS else "",
                "pos": None if kind == "qr" else report["position"],
                "message": report["message"] if kind == "qr" else "",
            }
        wire = self.profile["wire"]
        position = report["position"]
        if position is not None and wire["position_format"] == "xy_object":
            position = dict(x=position[0], y=position[1])
        fields = dict(group=self.profile["group_id"], task=report["id"], position=position)
        fields.update({key: report[key] for key in ("label", "scene", "message") if report.get(key) is not None})
        return {wire["fields"][key]: value for key, value in fields.items()}

    def status(self):
        reports = [
            {"task_id": row["id"], "evidence_ref": row["evidence_ref"],
             "measured_position": row["measured_position"], "payload": self.payload(row),
             "submitted": self.submitted(row)}
            for row in self.store.rows("competition_best")
        ]
        report_status = {row["task_id"]: "submitted" if row["submitted"] else "ready"
                         for row in reports}
        coverage = self.store.get("competition_coverage", "coverage") or {}
        return {
            "mode": self.profile["mode"], "arena_origin_map": self.origin,
            "tasks": [{"task_id": task_id, "target": task["target"],
                       "status": report_status.get(task_id, "pending")}
                      for task_id, task in self.tasks.items()],
            "reports": reports,
            "inspection_candidates": self.inspect(),
            "searched_cell_count": len(coverage.get("searched_cells", [])),
        }

    def submitted(self, row):
        previous = self.store.get("competition_sent", row["id"])
        return previous is not None and previous["payload"] == self.payload(row)


def improves(new, old):
    if new["confidence"] < old["confidence"]:
        return False
    if old["position"] is not None:
        if new["position"] is None or new["location_error_m"] > old["location_error_m"]:
            return False
        if new["position"] != old["position"] and new["location_error_m"] >= old["location_error_m"]:
            return False
    if old.get("scene") is not None and new.get("scene") is None:
        return False
    for key in ("label", "scene", "message"):
        if old.get(key) is not None and new.get(key) != old[key] and new["confidence"] <= old["confidence"]:
            return False
    return (new["confidence"] > old["confidence"]
            or new["position"] is not None and (
                old["position"] is None or new["location_error_m"] < old["location_error_m"])
            or old.get("scene") is None and new.get("scene") is not None)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def submit(competition, database, server):
    url = urlsplit(server)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.query or url.fragment:
        raise ValueError("Supply the scorer base URL without credentials, query, or fragment")
    if competition.profile["mode"] == "practice" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Practice profiles can submit only to a local mock server")
    opener = build_opener(NoRedirect)
    # One sender per attempt prevents an older request racing a newer submission.
    with database.with_suffix(database.suffix + ".send.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for task_id in competition.tasks:
            row = competition.store.get("competition_best", task_id)
            if row is None or competition.submitted(row):
                continue
            payload = competition.payload(row)
            request = Request(server.rstrip("/") + "/task_report",
                              data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                              headers={"Content-Type": "application/json"}, method="POST")
            with opener.open(request, timeout=5) as response:
                if competition.profile.get("contract") == V63:
                    acceptance = {"field": "status", "value": "accepted"}
                else:
                    acceptance = competition.profile["wire"].get("acceptance")
                if acceptance:
                    body = json.loads(response.read(65536))
                    if (not isinstance(body, dict)
                            or body.get(acceptance["field"]) != acceptance["value"]
                            or (competition.profile.get("contract") == V63 and body.get("task_id") != task_id)):
                        raise ValueError(f"Scorer did not acknowledge task {task_id}")
            with competition.store.db:
                competition.store.put("competition_sent", dict(id=task_id, payload=payload))
            print(f"Accepted {task_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("stage").add_argument("validation")
    sender = commands.add_parser("submit")
    sender.add_argument("--server", required=True)
    sender.add_argument("--watch", action="store_true", help="Send new improvements as they appear")
    args = parser.parse_args()
    database = args.db.expanduser().resolve()
    if not database.is_file():
        parser.error(f"Database does not exist: {database}")
    store = Store(database)
    try:
        competition = Competition(store)
        if args.command == "stage":
            if competition.profile["mode"] != "practice":
                raise ValueError("Run-time validations in competition mode come from the robot's ROS adapter")
            print(json.dumps({"improved": competition.ingest(read_json(args.validation))}))
        elif args.command == "submit":
            submit(competition, database, args.server)
            while args.watch:
                sleep(1)
                submit(competition, database, args.server)
        else:
            print(json.dumps(competition.status(), indent=2))
    except KeyboardInterrupt:
        pass
    except (KeyError, OSError, URLError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        store.close()


if __name__ == "__main__":
    main()
