"""Tests for pi_mission_reporter and nav_hold. Run: python3 -m pytest -q -p no:cacheprovider"""
import copy
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import sys
import threading

import pytest

import nav_hold
import pi_mission_reporter as r

LIVE = Path(__file__).parent / "fixtures" / "pi_detections_live_20261124.jsonl"
TEMPLATE = json.loads(json.loads(LIVE.read_text().splitlines()[-1])["data"])  # real Pi message
GID = "group08_RJ08"
S = 10**9
PERSON = ("person", 0, 0.8)
TOOLS = ("hand_tools", 0, 0.9)


def det(stamp, objects=(), qr=(), signs=()):
    m = copy.deepcopy(TEMPLATE)
    m.update(capture_stamp_ns=stamp, frame=stamp // S, qr_codes=[{"text": t, "corners": []} for t in qr],
             warning_sign_candidates=list(signs),
             objects=[{"name": n, "class": c, "confidence": conf,
                       "box": {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 220.0}} for n, c, conf in objects])
    return json.dumps(m)


def scene(stamp, kind, evidence=("prep counter", "stove"), frame_id=None, event=False):
    rec = {"capture_stamp_ns": stamp, "frame_id": frame_id or TEMPLATE["frame_id"], "frame": stamp // S,
           "scene": {"type": kind, "evidence": list(evidence)}, "model": "claude-opus-5-5"}
    if event:
        rec.update(event="scene_result", request_id="scene-1", encounter_id="person-1")
    return json.dumps(rec)


class FakePoster:
    def __init__(self, accept=True):
        self.sent, self.accept = [], accept

    def send(self, payload, callback):
        self.sent.append(payload)
        callback(payload, self.accept, "ok" if self.accept else '{"status":"rejected"}')


def tracker(poses=None, accept=True, **kw):
    kw.setdefault("room_cell_only", False)    # most tests below predate the room-cell contract
    poses = {} if poses is None else poses
    poster = FakePoster(accept)
    t = r.MissionTracker(r.retrieve_missions(), poster, GID, lambda stamp: poses.get(stamp), **kw)
    return t, poster.sent, poses


def see(t, poses, stamp, pose, now=None, **kw):
    poses[stamp] = pose
    t.on_detection(r.parse_detection(det(stamp, **kw)), stamp / S if now is None else now)


# ---------------------------------------------------------------- reporter
def test_booklet_missions_and_group_file(tmp_path):
    m = r.retrieve_missions()
    assert m["M01"]["label"] == "hand_tools" and m["M02"]["label"] is None
    assert m["M03"]["needs_qr"] and not m["M05"]["needs_qr"]
    (tmp_path / "g").write_text("group08_RJ08\n")
    assert r.load_group_id(None, tmp_path / "g") == "group08_RJ08"
    assert r.load_group_id("group08_ZZ99", tmp_path / "g") == "group08_ZZ99"


def test_recorded_live_messages_parse_and_report_nothing():
    t, sent, poses = tracker()
    for i, line in enumerate(LIVE.read_text().splitlines()):
        obs = r.parse_detection(json.loads(line)["data"])
        assert obs["key"][1] == "camera_color_optical_frame"
        poses[obs["key"][0]] = (0.0, 0.0, 0.0)
        t.on_detection(obs, float(i))
    assert sent == []


def test_delayed_scene_uses_capture_time_pose_while_dog_moves():
    """Hand tools seen in cell (1,2); the dog walks on; Claude answers 9 s later."""
    t, sent, poses = tracker()
    see(t, poses, 1 * S, (1.0, 2.1, 0.0), objects=[TOOLS])
    for k in range(2, 10):                                   # robot keeps moving through (3,0)
        see(t, poses, k * S, (3.0 + 0.1 * k, 0.1, 0.0), objects=[PERSON])
    assert sent == []
    t.on_scene(r.parse_scene(scene(1 * S, "workshop", ("tool bench", "pegboard"), event=True)), 10.0)
    assert sent == [{"group_id": GID, "task_id": "M01", "task_type": "object", "scene_type": "workshop",
                     "label": "hand_tools", "pos": [1.02, 2.04], "message": ""}]


def test_pose_missing_at_capture_keeps_report_pending_then_uses_capture_pose():
    t, sent, poses = tracker()
    t.on_detection(r.parse_detection(det(5 * S, [TOOLS])), 5.0)        # TF not there yet
    t.on_scene(r.parse_scene(scene(5 * S, "workshop")), 6.0)
    t.tick(7.0)
    assert sent == [] and t.stats["located"] == 0
    poses[5 * S] = (2.0, 1.0, 0.0)                                      # TF for that instant arrives
    poses[9 * S] = (0.0, 0.0, 0.0)                                      # a later pose must not be used
    t.tick(8.0)
    assert sent[0]["pos"] == [2.04, 1.02]


def test_pose_never_available_never_reports():
    t, sent, _ = tracker(tf_history_s=10)
    t.on_detection(r.parse_detection(det(5 * S, [TOOLS])), 5.0)
    t.on_scene(r.parse_scene(scene(5 * S, "workshop")), 6.0)
    t.tick(20.0)
    assert sent == [] and t.stats["pose_expired"] == 1


def test_scene_join_needs_stamp_and_frame_id_and_booklet_room():
    t, sent, poses = tracker()
    see(t, poses, 5 * S, (0, 0, 0), objects=[TOOLS])
    t.on_scene(r.parse_scene(scene(5 * S, "workshop", frame_id="other_frame")), 6.0)
    t.on_scene(r.parse_scene(scene(5 * S, "corridor")), 6.0)
    t.on_scene(r.parse_scene(scene(5 * S, "workshop", evidence=())), 6.0)
    assert sent == []


def test_scene_before_detections_is_held_and_joined():
    t, sent, poses = tracker()
    t.on_scene(r.parse_scene(scene(5 * S, "workshop")), 5.0)
    see(t, poses, 5 * S, (1.0, 0.0, 0.0), now=6.0, objects=[TOOLS])
    assert [p["task_id"] for p in sent] == ["M01"]


def test_hazard_needs_classifier_label():
    t, sent, poses = tracker()
    tri = {"box": [1, 2, 30, 40], "triangle": [[1, 40], [15, 2], [30, 40]], "yellow_fraction": .9, "area": 500.0}
    see(t, poses, 7 * S, (0, 0, 0), signs=[tri])
    t.on_scene(r.parse_scene(scene(7 * S, "kitchen")), 8.0)
    assert sent == []
    see(t, poses, 9 * S, (0, 1.0, 0), signs=[dict(tri, label="high_voltage", confidence=.93)])
    t.on_scene(r.parse_scene(scene(9 * S, "kitchen")), 10.0)
    assert sent == [{"group_id": GID, "task_id": "M02", "task_type": "hazard", "scene_type": "kitchen",
                     "label": "high_voltage", "pos": [0.0, 1.02], "message": ""}]


def visit_survivor(t, poses, start, pose, room, qr=None):
    for k in range(4):
        see(t, poses, (start + k) * S, pose, objects=[PERSON], qr=[qr] if qr and k == 1 else ())
    t.on_scene(r.parse_scene(scene((start + 1) * S, room, ("shelves",))), start + 6.0)


def test_two_distinct_survivors_m05_first_waits_for_qr_survivor():
    t, sent, poses = tracker(m05_after_s=None)    # the M03-first ordering, fallback off
    visit_survivor(t, poses, 100, (2.0, 1.0, 0.0), "market")                 # unconscious, seen first
    assert sent == [], "M05 must wait until the QR-linked survivor is known"
    visit_survivor(t, poses, 200, (0.05, 2.0, 0.0), "kitchen", qr="Please help me.")
    by = {p["task_id"]: p for p in sent}
    assert [p["task_id"] for p in sent] == ["M04", "M03", "M05"]
    assert by["M04"] == {"group_id": GID, "task_id": "M04", "task_type": "qr", "scene_type": "",
                         "label": "", "pos": None, "message": "Please help me."}
    assert by["M03"]["scene_type"] == "kitchen" and by["M03"]["pos"] == [0.0, 2.04]
    assert by["M05"]["scene_type"] == "market" and by["M05"]["pos"] == [2.04, 1.02]
    assert by["M03"]["label"] == by["M05"]["label"] == by["M05"]["message"] == ""


def test_qr_survivor_is_m03_and_far_survivor_preferred_for_m05():
    t, sent, poses = tracker(m05_after_s=None)    # the M03-first ordering, fallback off
    visit_survivor(t, poses, 100, (1.0, 2.0, 0.0), "clinic")                  # beside the QR cell
    visit_survivor(t, poses, 200, (3.0, 0.0, 0.0), "market")                  # far from any QR
    assert sent == []
    visit_survivor(t, poses, 300, (0.0, 2.0, 0.0), "kitchen", qr="Please help me.")
    by = {p["task_id"]: p for p in sent}
    assert by["M03"]["pos"] == [0.0, 2.04] and by["M05"]["pos"] == [3.06, 0.0]


def test_two_survivors_equally_close_to_qr_keep_m03_pending():
    t, sent, poses = tracker(m05_after_s=None)    # the M03-first ordering, fallback off
    visit_survivor(t, poses, 100, (0.0, 1.0, 0.0), "kitchen")
    visit_survivor(t, poses, 200, (0.0, 3.0, 0.0), "clinic")
    see(t, poses, 300 * S, (0.0, 2.0, 0.0), qr=["Please help me."])           # QR read between them
    assert [p["task_id"] for p in sent] == ["M04"]


def test_same_survivor_revisited_is_one_survivor():
    t, sent, poses = tracker()
    visit_survivor(t, poses, 100, (0.0, 2.0, 0.0), "kitchen", qr="Please help me.")
    visit_survivor(t, poses, 200, (0.1, 1.9, 0.0), "kitchen", qr="Please help me.")
    assert sorted(p["task_id"] for p in sent) == ["M03", "M04"]


def test_single_person_frame_is_not_a_confirmed_survivor():
    t, sent, poses = tracker()
    see(t, poses, 1 * S, (0, 0, 0), objects=[PERSON], qr=["hi"])
    t.on_scene(r.parse_scene(scene(1 * S, "clinic")), 2.0)
    assert [p["task_id"] for p in sent] == ["M04"]


def test_repeated_sightings_do_not_duplicate_reports():
    t, sent, poses = tracker()
    for rep in range(3):
        base = 1000 * (rep + 1)
        for k in range(4):
            see(t, poses, (base + k) * S, (0, 2.0, 0), objects=[PERSON, TOOLS], qr=["Please help me."])
        t.on_scene(r.parse_scene(scene(base * S, "kitchen")), base + 5)
        visit_survivor(t, poses, base + 50, (2.0, 1.0, 0), "market")
    ids = [p["task_id"] for p in sent]
    assert sorted(ids) == ["M01", "M03", "M04", "M05"] and len(ids) == len(set(ids))


def test_rejected_report_reopens_but_never_resends_identical_payload():
    t, sent, poses = tracker(accept=False)
    see(t, poses, 1 * S, (0, 0, 0), objects=[TOOLS])
    t.on_scene(r.parse_scene(scene(1 * S, "workshop")), 2.0)
    see(t, poses, 3 * S, (0, 0, 0), objects=[TOOLS])
    t.tick(4.0)
    assert len(sent) == 1 and "M01" not in t.state


def test_clock_monitor_flags_pi_ahead():
    c = r.ClockMonitor()
    for i in range(20):
        c.add(i * S, i * S + 80_000_000)
    assert c.verdict() == "ok" and c.summary()["median_ms"] == 80
    c.add(100 * S, 100 * S - 200_000_000)
    assert "ahead" in c.verdict()


# ---------------------------------------------------------------- HTTP against a mock server
class MockScorer(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockScorer.received.append((self.path, self.headers["Content-Type"], body))
        ok = list(body) == list(r.WIRE_KEYS) and body["scene_type"] != "clothes_shop"
        reply = ({"status": "accepted", "group_id": "group08", "task_id": body["task_id"],
                  "message": "Submission recorded"} if ok else
                 {"status": "rejected", "group_id": "group08", "task_id": body["task_id"],
                  "reason": "Invalid scene_type"})
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def scorer():
    MockScorer.received = []
    server = HTTPServer(("127.0.0.1", 0), MockScorer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/task_report_A"
    server.shutdown()


def test_mock_server_receives_exact_payloads_once(scorer, tmp_path):
    poster = r.Poster(scorer, dry_run=False, log_path=tmp_path / "log.jsonl")
    poses = {}
    t = r.MissionTracker(r.retrieve_missions(), poster, GID, lambda s: poses.get(s), room_cell_only=False)
    for rep in range(3):                                   # three visits to each scene
        visit_survivor(t, poses, 100 + 1000 * rep, (0, 2.0, 0), "kitchen", qr="Please help me.")
        poster.wait()
        visit_survivor(t, poses, 200 + 1000 * rep, (2.0, 1.0, 0), "market")
        poster.wait()
    paths = {p for p, _, _ in MockScorer.received}
    bodies = [b for _, _, b in MockScorer.received]
    assert paths == {"/task_report_A"}
    assert {ct for _, ct, _ in MockScorer.received} == {"application/json"}
    assert sorted(b["task_id"] for b in bodies) == ["M03", "M04", "M05"]
    assert all(list(b) == list(r.WIRE_KEYS) for b in bodies)
    assert t.state == {"M04": "accepted", "M03": "accepted", "M05": "accepted"}  # real HTTP acks
    log = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert all(e["http_status"] == 200 and '"accepted"' in e["response"] for e in log)


def test_mock_server_rejection_is_recorded_not_counted(scorer, tmp_path):
    poster = r.Poster(scorer, dry_run=False, log_path=None)
    poses = {}
    t = r.MissionTracker(r.retrieve_missions(), poster, GID, lambda s: poses.get(s), room_cell_only=False)
    see(t, poses, 1 * S, (0, 0, 0), objects=[TOOLS])
    t.on_scene(r.parse_scene(scene(1 * S, "clothes_shop", ("racks",))), 2.0)
    poster.wait()
    assert len(MockScorer.received) == 1 and "M01" not in t.state and t.rejected["M01"]


# ---------------------------------------------------------------- navigation hold
def ev(kind, rid="scene-1", enc=("person-1",), stamp=100 * S, **extra):
    return dict(event=kind, request_id=rid, encounter_id=enc[-1], encounter_ids=list(enc),
                capture_stamp_ns=stamp, frame_id="camera_color_optical_frame",
                trigger={"kind": "person", "label": "person"}, **extra)


def test_hold_releases_on_matching_result_only():
    h = nav_hold.HoldController(5.0)
    out = h.on_event(ev("scene_pending"), 0.0, 100 * S)
    assert out["hold"] and out["request_id"] == "scene-1" and out["expires_wall_ns"] == 105 * S
    assert h.on_event(ev("scene_result", rid="scene-9"), 1.0, 101 * S) is None
    out = h.on_event(ev("scene_result"), 2.0, 102 * S)
    assert out["hold"] is False and out["reason"] == "scene_result"


def test_hold_releases_once_the_still_frame_is_taken():
    h = nav_hold.HoldController(5.0)
    h.on_event(ev("scene_pending"), 0.0, 100 * S)
    assert h.on_event(ev("scene_captured", rid="scene-9"), 1.0, 101 * S) is None
    out = h.on_event(ev("scene_captured", stamp=102 * S), 2.0, 102 * S)
    assert out["hold"] is False and out["reason"] == "scene_captured"
    assert h.on_event(ev("scene_result", stamp=102 * S), 6.0, 106 * S) is None        # result comes later


def test_hold_resumes_after_timeout_and_late_result_does_not_rehold():
    h = nav_hold.HoldController(5.0)
    h.on_event(ev("scene_pending"), 0.0, 100 * S)
    assert h.tick(4.9, 104 * S) is None
    out = h.tick(5.0, 105 * S)
    assert out["hold"] is False and "timeout" in out["reason"]
    assert h.on_event(ev("scene_result"), 9.0, 109 * S) is None and h.active is None


def test_hold_once_per_encounter_failure_and_stale_events():
    h = nav_hold.HoldController(5.0)
    h.on_event(ev("scene_pending"), 0.0, 100 * S)
    assert h.on_event(ev("scene_failed", error="TimeoutError"), 1.0, 101 * S)["hold"] is False
    assert h.on_event(ev("scene_pending", rid="scene-2"), 2.0, 102 * S) is None     # same encounter again
    assert h.on_event(ev("scene_pending", rid="scene-3", enc=("qr-2",), stamp=90 * S), 3.0, 103 * S) is None
    assert h.on_event(ev("scene_pending", rid="scene-4", enc=("object-3",), stamp=103 * S), 3.5, 103 * S)["hold"]
    assert h.on_event(ev("scene_pending", rid="scene-5", enc=("qr-6",), stamp=104 * S), 4.0, 104 * S) is None
    assert nav_hold.HoldController(5.0, enabled=False).on_event(ev("scene_pending"), 0, 100 * S) is None


# ---------------------------------------------------------------- Pi events -> NUC, end to end
def test_pi_scheduler_events_drive_hold_and_late_report():
    sys.path.insert(0, str(Path.home() / "usb-camera-perception"))
    import scene_events as se
    futures, sent_images = [Future()], []
    clock = [0.0]
    sch = se.SceneScheduler(lambda image: sent_images.append(image) or futures[0], min_interval_s=4,
                            clock=lambda: clock[0], wall_ns=lambda: int(clock[0] * S) + 100 * S,
                            model="claude-opus-5-5")
    gate = se.EncounterGate()
    hold = nav_hold.HoldController(5.0)
    t, sent, poses = tracker()
    events = []
    frames = [(1.0, 2.0, 0.0, False), (1.0, 2.0, 0.0, False), (2.0, 2.0, 0.0, False),   # walking
              (2.0, 2.0, 0.0, True)]                                                      # stopped
    for k, (x, y, yaw, still) in enumerate(frames):
        stamp = 100 * S + k * S
        clock[0] = float(k)
        raw = det(stamp, [TOOLS])
        rec = json.loads(raw)
        see(t, poses, stamp, (x, y, yaw), now=float(k), objects=[TOOLS])
        keys = se.candidate_keys(rec["objects"], [], [])
        for e in sch.offer(f"img{k}", {"frame": k, "capture_stamp_ns": stamp, "frame_id": rec["frame_id"]},
                           gate.update(keys, stamp), still):
            events.append(e)
            out = hold.on_event(json.loads(json.dumps(e)), clock[0], int(clock[0] * S) + 100 * S)
            if out:
                events.append(out)
        if k == 2:
            assert events[-1]["hold"] is True and sent_images == []           # held, nothing sent yet
    pending = [e for e in events if e.get("event") == "scene_pending"]
    assert len(pending) == 1 and pending[0]["capture_stamp_ns"] == 101 * S     # confirmed on 2nd frame
    assert sent_images == ["img3"]                                             # only the still frame
    assert events[-2]["event"] == "scene_captured" and events[-1]["hold"] is False
    clock[0] = 7.5
    futures[0].set_result({"type": "workshop", "evidence": ["pegboard"]})
    result = sch.poll()[0]
    assert result["event"] == "scene_result" and result["capture_stamp_ns"] == 103 * S
    t.on_scene(r.parse_scene(json.dumps(result)), 8.0)                         # late result still reports
    assert sent[0]["task_id"] == "M01" and sent[0]["pos"] == [2.04, 2.04]     # pose of the still frame


def test_neighbour_survivor_becomes_m03_after_qr_visit_ends():
    t, sent, poses = tracker()
    see(t, poses, 100 * S, (0.0, 2.0, 0.0), qr=["Please help me."])           # QR read from cell (0,2)
    visit_survivor(t, poses, 101, (1.0, 2.0, 0.0), "kitchen")                  # survivor in next cell
    assert [p["task_id"] for p in sent] == ["M04"]                             # QR visit still open
    t.tick(200.0)
    assert sent[-1]["task_id"] == "M03" and sent[-1]["pos"] == [1.02, 2.04]


# ---------------------------------------------------------------- room scans (Pi camera, Claude, local QR)
def room_scan(scan_id, room, views, findings=(), qr=(), dry_run=False, evidence=None):
    """views: [(view_id, stamp)]; findings: [(kind, label, view_id, clear)]; qr: [(view_id, text)]"""
    return {"event": "room_scan_result", "scan_id": scan_id, "dry_run": dry_run, "model": "claude-opus-5-5",
            "views": [{"view_id": v, "capture_stamp_ns": st, "frame_id": "camera_color_optical_frame", "frame": i,
                       "image": f"/x/{scan_id}/{v}.jpg", "qr": [t for vv, t in qr if vv == v]}
                      for i, (v, st) in enumerate(views)],
            "room": {"type": room, "evidence": evidence if evidence is not None else
                     [{"view_id": views[0][0], "cue": f"{room} props"}]},
            "findings": [{"kind": k, "label": l, "view_id": v, "evidence": "seen", "clear": c}
                         for k, l, v, c in findings],
            "qr": [{"view_id": v, "text": t, "source": "local"} for v, t in qr],
            "additional_view": {"needed": False}}


def scan_tracker(poses):
    t, sent, _ = tracker(poses)
    t.accept_dry_run_scans = True
    return t, sent


def test_room_scan_views_join_capture_time_poses_and_report_missions():
    poses = {10 * S: (1.0, 2.0, 1.57), 13 * S: (1.05, 1.95, -1.57),          # turned left, then right
             12 * S: (9.0, 9.0, 0.0)}                                        # a later pose that must not be used
    t, sent = scan_tracker(poses)
    t.on_room_scan(r.parse_room_scan(json.dumps(room_scan(
        "scan-1", "kitchen", [("left", 10 * S), ("right", 13 * S)],
        findings=[("hazard_sign", "high_voltage", "left", True), ("hand_tools", "hand_tools", "left", True),
                  ("person", "", "right", True)],
        qr=[("right", "My friend is in the market.")]))), 20.0)
    t.tick(200.0)                                                            # QR visit over: neighbours allowed
    by = {p["task_id"]: p for p in sent}
    assert by["M04"]["message"] == "My friend is in the market." and by["M04"]["pos"] is None
    assert by["M02"] == {"group_id": GID, "task_id": "M02", "task_type": "hazard", "scene_type": "kitchen",
                         "label": "high_voltage", "pos": [1.02, 2.04], "message": ""}
    assert by["M01"]["label"] == "hand_tools" and by["M01"]["pos"] == [1.02, 2.04]
    assert by["M03"]["scene_type"] == "kitchen" and by["M03"]["pos"] == [1.02, 2.04] and by["M03"]["label"] == ""
    assert "M05" not in by


def test_room_scan_m05_waits_for_qr_survivor_and_is_distinct():
    poses = {10 * S: (3.0, 0.0, 0.0), 11 * S: (3.0, 0.0, 0.0), 20 * S: (0.0, 2.0, 0.0), 21 * S: (0.0, 2.0, 0.0)}
    t, sent = scan_tracker(poses)
    t.on_room_scan(room_scan("scan-1", "market", [("left", 10 * S), ("right", 11 * S)],
                             findings=[("person", "", "left", True)]), 15.0)
    assert sent == []                                                        # QR association unclear: pending
    t.on_room_scan(room_scan("scan-2", "kitchen", [("left", 20 * S), ("right", 21 * S)],
                             findings=[("person", "", "right", True)], qr=[("right", "Please help me.")]), 25.0)
    by = {p["task_id"]: p for p in sent}
    assert by["M03"]["pos"] == [0.0, 2.04] and by["M03"]["scene_type"] == "kitchen"
    assert by["M05"]["pos"] == [3.06, 0.0] and by["M05"]["scene_type"] == "market"


def test_room_scan_revisit_and_resent_result_do_not_duplicate():
    poses = {10 * S: (1.0, 0.0, 0.0), 11 * S: (1.0, 0.0, 0.0), 50 * S: (1.1, 0.1, 0.0), 51 * S: (1.1, 0.1, 0.0)}
    t, sent = scan_tracker(poses)
    first = room_scan("scan-1", "workshop", [("left", 10 * S), ("right", 11 * S)],
                      findings=[("hand_tools", "hand_tools", "left", True)])
    t.on_room_scan(first, 15.0)
    t.on_room_scan(first, 16.0)                                              # same result re-sent (extra look)
    t.on_room_scan(room_scan("scan-9", "workshop", [("left", 50 * S), ("right", 51 * S)],
                             findings=[("hand_tools", "hand_tools", "right", True)]), 55.0)   # revisit
    assert [p["task_id"] for p in sent] == ["M01"]


def test_room_scan_unclear_unknown_and_dry_run_are_not_reported():
    poses = {10 * S: (0.0, 0.0, 0.0)}
    t, sent, _ = tracker(poses)                                              # live reporter: dry runs ignored
    t.on_room_scan(room_scan("dry", "kitchen", [("left", 10 * S)], findings=[("hand_tools", "", "left", True)],
                             dry_run=True), 1.0)
    assert sent == []
    t.accept_dry_run_scans = True
    t.on_room_scan(room_scan("s2", "kitchen", [("left", 10 * S)],
                             findings=[("hazard_sign", "unknown", "left", True), ("hand_tools", "", "left", False)]), 2.0)
    assert sent == []
    t.on_room_scan(room_scan("s3", "unknown", [("left", 12 * S)], findings=[("hand_tools", "", "left", True)],
                             evidence=[]), 3.0)
    poses[12 * S] = (0.0, 0.0, 0.0)
    t.tick(4.0)
    assert sent == []                                                        # no room: object report waits


def test_single_forward_view_from_in_front_of_the_room():
    """The dog stands in the cell in front of the room and takes ONE forward photo: Claude's findings and the
    OpenCV QR text (decoded where Claude saw a QR code) compile into the mission answers."""
    poses = {30 * S: (1.0, 0.0, 1.57)}
    t, sent = scan_tracker(poses)
    scan = room_scan("view-1_1-9", "kitchen", [("forward", 30 * S)],
                     findings=[("hazard_sign", "high_voltage", "forward", True), ("person", "", "forward", True),
                               ("qr_code", "", "forward", True)],
                     qr=[("forward", "My friend is in the market.")])
    for q in scan["qr"]:
        q["source"] = "opencv"
    scan["qr_unreadable"] = []
    t.on_room_scan(r.parse_room_scan(json.dumps(scan)), 35.0)
    by = {p["task_id"]: p for p in sent}
    assert by["M04"]["message"] == "My friend is in the market."
    assert by["M02"]["label"] == "high_voltage" and by["M02"]["scene_type"] == "kitchen"
    assert by["M03"]["scene_type"] == "kitchen" and by["M03"]["label"] == ""


def test_claude_failed_but_opencv_qr_still_reports_m04():
    poses = {30 * S: (1.0, 0.0, 1.57)}
    t, sent = scan_tracker(poses)
    scan = room_scan("view-1_1-10", "unknown", [("forward", 30 * S)], qr=[("forward", "Please help me.")],
                     evidence=[])
    scan.update(claude=False, error="APITimeoutError: timed out")
    t.on_room_scan(r.parse_room_scan(json.dumps(scan)), 35.0)
    assert [p["task_id"] for p in sent] == ["M04"] and sent[0]["message"] == "Please help me."


def test_start_pose_is_cell_zero_zero():
    origin = (4.2, -1.3, 0.9)
    x, y, _ = r.to_arena(origin, origin)
    assert r.cell_of((x, y)) == (0, 0)



# ---------------------------------------------------------------- pos = room cell only (2026-11-26)
def test_room_scan_cell_is_the_position_not_the_robot_pose():
    t, sent, poses = tracker(room_cell_only=True)
    t.accept_dry_run_scans = True
    scan = room_scan("view-3_-2-100", "kitchen", [("forward", 50 * S)],
                     findings=[("hazard_sign", "high_voltage", "forward", True)])
    scan["cell"] = [3, -2]
    t.on_room_scan(scan, 60.0)                       # no pose known at all: not needed
    m02 = [p for p in sent if p["task_id"] == "M02"]
    assert m02 and m02[0]["pos"] == [3.06, -2.04] and m02[0]["scene_type"] == "kitchen"


def test_walking_sightings_get_no_position_but_qr_still_reports():
    t, sent, poses = tracker(room_cell_only=True)
    for k in range(5):
        see(t, poses, (20 + k) * S, (1.0, 0.0, 0.0), objects=[PERSON, TOOLS], qr=("help in kitchen",),
            signs=[{"box": [1, 2, 30, 40], "label": "high_voltage", "confidence": .93}])
        t.on_scene(r.parse_scene(scene((20 + k) * S, "kitchen")), 21.0 + k)
    assert [p["task_id"] for p in sent] == ["M04"] and sent[0]["pos"] is None


def test_room_scan_without_cell_reports_no_position():
    t, sent, poses = tracker(room_cell_only=True)
    t.accept_dry_run_scans = True
    poses[50 * S] = (0.0, 0.0, 0.0)
    t.on_room_scan(room_scan("view-x", "kitchen", [("forward", 50 * S)],
                             findings=[("hazard_sign", "high_voltage", "forward", True)]), 60.0)
    t.tick(61.0)
    assert [p for p in sent if p["pos"] is not None] == []


def test_room_cell_validation():
    assert r.room_cell([2, -1]) == (2, -1)
    assert r.room_cell([2.5, 1]) is None and r.room_cell([True, 1]) is None
    assert r.room_cell(None) is None and r.room_cell([40, 0]) is None


# ---------------------------------------------------------------- position source (room cell default, base_xy option)
def det_xy(stamp, objects=(), signs=()):
    """A detections record whose objects carry base_xy: objects = [(name, conf, base_xy)]."""
    m = json.loads(det(stamp, [(n, 0, c) for n, c, _ in objects], signs=signs))
    for o, (_, _, xy) in zip(m["objects"], objects):
        o["base_xy"] = xy
    return json.dumps(m)


def test_default_is_room_cell_and_yolo_sightings_get_no_position():
    poses = {5 * S: (0.0, 0.0, 0.0)}
    t2 = r.MissionTracker(r.retrieve_missions(), FakePoster(), GID, lambda s: poses.get(s))
    assert t2.pos_source == "room_cell" and t2.room_cell_only


def test_base_xy_places_the_target_not_the_robot():
    poses = {5 * S: (1.0, 0.0, math.pi / 2)}                 # robot in cell (1,0), facing +Y
    poster = FakePoster()
    t = r.MissionTracker(r.retrieve_missions(), poster, GID, lambda s: poses.get(s), pos_source="base_xy")
    t.on_detection(r.parse_detection(det_xy(5 * S, [("hand_tools", 0.9, [2.0, 0.0])])), 5.0)   # 2 m ahead
    t.on_scene(r.parse_scene(scene(5 * S, "workshop")), 6.0)
    assert poster.sent and poster.sent[0]["pos"] == [1.02, 2.04]                   # target cell (1,2), not (1,0)


def test_base_xy_null_gives_no_position_and_never_the_robot_cell():
    poses = {5 * S: (1.0, 0.0, 0.0)}
    poster = FakePoster()
    t = r.MissionTracker(r.retrieve_missions(), poster, GID, lambda s: poses.get(s), pos_source="base_xy")
    t.on_detection(r.parse_detection(det_xy(5 * S, [("hand_tools", 0.9, None)])), 5.0)
    t.on_scene(r.parse_scene(scene(5 * S, "workshop")), 6.0)
    t.tick(7.0)
    assert poster.sent == []


def test_base_xy_mode_still_uses_the_room_cell_for_room_photos():
    poster = FakePoster()
    t = r.MissionTracker(r.retrieve_missions(), poster, GID, lambda s: None, pos_source="base_xy")
    t.accept_dry_run_scans = True
    scan = room_scan("view-2_1-5", "kitchen", [("forward", 30 * S)],
                     findings=[("hazard_sign", "high_voltage", "forward", True)])
    scan["cell"] = [2, 1]
    t.on_room_scan(r.parse_room_scan(json.dumps(scan)), 31.0)
    t.tick(32.0)
    assert poster.sent[0]["task_id"] == "M02" and poster.sent[0]["pos"] == [2.04, 1.02]


# ------------------------------------------------ M05 fallback (operator 2026-11-27)
def test_survivor_without_a_qr_goes_out_as_m05_after_the_wait():
    """26 Nov: two cleaning_room survivors were held forever waiting for M03; with --m05-after they go out as M05.
    Off by default since 27 Nov (operator: post only confirmed answers)."""
    t, sent, poses = tracker(m05_after_s=30.0)
    visit_survivor(t, poses, 100, (2.0, 1.0, 0.0), "cleaning_room")
    t.tick(100 + 20.0)
    assert sent == [], "not before m05_after_s"
    t.tick(100 + 31.0)
    assert [(p["task_id"], p["scene_type"], p["pos"]) for p in sent] == [("M05", "cleaning_room", [2.04, 1.02])]
    # the QR survivor found later is still M03 (and M04)
    visit_survivor(t, poses, 300, (0.05, 4.0, 0.0), "kitchen", qr="Please help me.")
    by = {p["task_id"]: p for p in sent}
    assert by["M03"]["scene_type"] == "kitchen" and by["M03"]["pos"] == [0.0, 4.08] and "M04" in by
    assert [p["task_id"] for p in sent].count("M05") == 1


def test_survivor_beside_a_qr_is_never_the_m05_fallback():
    t, sent, poses = tracker(m05_after_s=0.0)
    see(t, poses, 50 * S, (0.0, 2.0, 0.0), qr=["Please help me."])          # QR read, no survivor yet
    visit_survivor(t, poses, 100, (1.0, 2.0, 0.0), "clinic")                 # one cell from the QR
    t.tick(400.0)
    assert "M05" not in [p["task_id"] for p in sent]


def test_m05_fallback_can_be_switched_off():
    t, sent, poses = tracker(m05_after_s=None)
    visit_survivor(t, poses, 100, (2.0, 1.0, 0.0), "cleaning_room")
    t.tick(1000.0)
    assert [p["task_id"] for p in sent] == []



def test_confirmed_only_no_timed_m05_and_no_qr_pick_by_default():
    """Operator 2026-11-27: post only confirmed answers. A survivor with no QR beside her is not sent as M05 on a
    timer, and with two distinct QR texts M04 is held instead of posting the first."""
    t, sent, poses = tracker()
    visit_survivor(t, poses, 100, (2.0, 1.0, 0.0), "cleaning_room")
    t.tick(100 + 120.0)
    assert [p["task_id"] for p in sent] == []
    t2, sent2, _ = tracker()
    t2.qr_seen.update({"First clue.": None, "Second clue.": None})
    t2.evaluate()
    assert [p for p in sent2 if p["task_id"] == "M04"] == []


def _room_cell_tracker(map_poses):
    t, sent, _ = tracker(room_cell_only=True)
    t.accept_dry_run_scans = True
    t.map_pose_at = lambda stamp: map_poses.get(stamp)
    return t, sent


def _survivor_room(scan_id, cell, stamp, pose, room="cleaning_room", qr_in_scan=False):
    # qr_in_scan=False: the link tests check the walking-QR route only (a qr_code in the scan reports M03 directly)
    scan = room_scan(scan_id, room, [("centre", stamp)], findings=[("person", "", "centre", True)]
                     + ([("qr_code", "", "centre", True)] if qr_in_scan else []))
    scan.update(cell=list(cell), pose=list(pose))
    return r.parse_room_scan(json.dumps(scan))


def test_walking_qr_links_to_the_room_scanned_in_the_same_visit_and_m03_follows():
    """27 Nov 06:44 Arena B: the QR was decoded from a walking frame (no cell); the cleaning-room scan saw the
    survivor and the QR but could not read it. Same visit (12 s apart, same spot) -> QR linked -> M03."""
    t, sent = _room_cell_tracker({100 * S: (2.10, 0.05, 0.0)})
    t.on_detection(r.parse_detection(det(100 * S, qr=["My friend is in the market."])), 100.0)
    t.on_room_scan(_survivor_room("fallback-2_0-x", (2, 0), 112 * S, (2.0, 0.0, 0.0)), 125.0)
    assert "M03" not in [p["task_id"] for p in sent]                         # not before the window has settled
    t.tick(100.0 + r.QR_LINK_SETTLE_S + 1)
    by = {p["task_id"]: p for p in sent}
    assert by["M04"]["message"] == "My friend is in the market." and by["M04"]["pos"] is None
    assert by["M03"]["scene_type"] == "cleaning_room" and by["M03"]["pos"] == [2.04, 0.0]


def test_walking_qr_is_not_linked_when_two_rooms_were_scanned_or_the_dog_was_elsewhere():
    t, sent = _room_cell_tracker({100 * S: (2.1, 0.0, 0.0)})
    t.on_detection(r.parse_detection(det(100 * S, qr=["Clue."])), 100.0)
    t.on_room_scan(_survivor_room("a", (2, 0), 105 * S, (2.0, 0.0, 0.0)), 110.0)
    t.on_room_scan(_survivor_room("b", (0, 3), 110 * S, (0.0, 3.0, 0.0)), 115.0)
    t.tick(100.0 + r.QR_LINK_SETTLE_S + 1)
    assert "M03" not in [p["task_id"] for p in sent]                        # ambiguous window
    t2, sent2 = _room_cell_tracker({100 * S: (6.0, 4.0, 0.0)})             # QR decoded 4+ m from the scan spot
    t2.on_detection(r.parse_detection(det(100 * S, qr=["Clue."])), 100.0)
    t2.on_room_scan(_survivor_room("c", (2, 0), 110 * S, (2.0, 0.0, 0.0)), 115.0)
    t2.tick(100.0 + r.QR_LINK_SETTLE_S + 1)
    assert "M03" not in [p["task_id"] for p in sent2]


def _scan_room(t, scan_id, cell, stamp, room, qr_seen_in_scan):
    findings = [("person", "", "centre", True)] + ([("qr_code", "", "centre", True)] if qr_seen_in_scan else [])
    scan = room_scan(scan_id, room, [("centre", stamp)], findings=findings)
    scan.update(cell=list(cell), pose=[cell[0] * 1.0, cell[1] * 1.0, 0.0])
    t.on_room_scan(r.parse_room_scan(json.dumps(scan)), stamp / S)


def test_m05_goes_out_without_waiting_for_m03_and_never_with_a_qr_beside_her():
    """Operator 2026-11-27: M05 must not wait for M03. Cleaning room (2,0): survivor + an unreadable QR -> M03
    candidate, never M05. Market (4,2): survivor, no QR in its scan -> M05 at once."""
    t, sent = _room_cell_tracker({})
    _scan_room(t, "cleaning", (2, 0), 100 * S, "cleaning_room", qr_seen_in_scan=True)
    t.tick(200.0)
    assert [p["task_id"] for p in sent] == ["M03"]                         # the QR survivor is M03, never M05
    _scan_room(t, "market", (4, 2), 300 * S, "market", qr_seen_in_scan=False)
    by = {p["task_id"]: p for p in sent}
    assert list(by) == ["M03", "M05"] and by["M05"]["scene_type"] == "market" and by["M05"]["pos"] == [4.08, 2.04]


def test_two_survivors_without_a_qr_are_held_for_m05():
    t, sent = _room_cell_tracker({})
    _scan_room(t, "a", (4, 2), 100 * S, "market", qr_seen_in_scan=False)
    _scan_room(t, "b", (0, 5), 200 * S, "kitchen", qr_seen_in_scan=False)
    t.tick(300.0)
    # the first went out at once; a second QR-less survivor is then not posted again
    assert [p["task_id"] for p in sent].count("M05") == 1


def test_after_m03_the_remaining_survivor_beside_a_qr_is_still_never_m05():
    t, sent = _room_cell_tracker({100 * S: (2.0, 0.0, 0.0)})
    t.on_detection(r.parse_detection(det(100 * S, qr=["Help."])), 100.0)
    _scan_room(t, "cleaning", (2, 0), 110 * S, "cleaning_room", qr_seen_in_scan=True)
    t.tick(100.0 + r.QR_LINK_SETTLE_S + 1)                                # QR linked to (2,0) -> M03
    _scan_room(t, "kitchen", (0, 4), 300 * S, "kitchen", qr_seen_in_scan=True)      # later: another QR-room survivor
    t.tick(400.0)
    ids = [p["task_id"] for p in sent]
    assert "M03" in ids and "M05" not in ids



def test_m03_reports_at_once_from_a_qr_in_her_room_scan_and_m05_from_a_room_without_one():
    """Operator 2026-11-27 ("It should just report"): no waiting chains. Cleaning room (2,0) scan: survivor + QR
    (not decoded) -> M03 now; market (4,2) scan: survivor, no QR -> M05 now."""
    t, sent = _room_cell_tracker({})
    _scan_room(t, "cleaning", (2, 0), 100 * S, "cleaning_room", qr_seen_in_scan=True)
    by = {p["task_id"]: p for p in sent}
    assert by["M03"]["scene_type"] == "cleaning_room" and by["M03"]["pos"] == [2.04, 0.0] and by["M03"]["message"] == ""
    _scan_room(t, "market", (4, 2), 200 * S, "market", qr_seen_in_scan=False)
    by = {p["task_id"]: p for p in sent}
    assert by["M05"]["scene_type"] == "market" and by["M05"]["pos"] == [4.08, 2.04]

