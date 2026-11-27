# Pi scene events, navigation hold and mission reporting

All topics are on ROS domain 8 and carry `std_msgs/msg/String` JSON.

| Topic | Publisher | QoS | Purpose |
|---|---|---|---|
| `/group8pi/perception/detections` | Pi | BEST_EFFORT | One record per processed frame: `objects` (COCO plus `hand_tools`), `qr_codes`, `warning_sign_candidates` (identified hazard in `label`), `stair_curtain`, obstacle candidates |
| `/group8pi/perception/scenes` | Pi | BEST_EFFORT | Claude result in the original format, for older consumers |
| `/group8pi/perception/scene_events` | Pi | RELIABLE, depth 20 | `scene_pending`, `scene_captured`, `scene_result`, `scene_failed` |
| `/group8/navigation/hold_request` | NUC `nav_hold.py` | RELIABLE, TRANSIENT_LOCAL, depth 1 | Hold request for roam |

## Scene event contract (schema_version 1)

```json
{"event": "scene_pending", "schema_version": 1, "request_id": "scene-7",
 "encounter_id": "person-12", "encounter_ids": ["person-12"],
 "capture_stamp_ns": 1795533166429167021, "frame_id": "camera_color_optical_frame", "frame": 412,
 "trigger": {"kind": "person", "label": "person"},
 "triggers": [{"encounter_id": "person-12", "kind": "person", "label": "person"}],
 "model": "claude-opus-5-5", "published_ns": 1795533166529167021, "hold_timeout_s": 5.0}
```

`scene_pending` carries the frame that confirmed the encounter (usually taken while walking). No image
is sent yet. `scene_captured` (adds `waited_s`) carries the still frame that was actually sent to Claude.
A `scene_result` adds `"scene": {"type": "<booklet scene_type or unknown>", "evidence": [...]}`.
A `scene_failed` adds `"error"`. Captures, results and failures reuse the request's `request_id` and carry
the still frame's `capture_stamp_ns` and `frame_id` (the trigger frame's if no still frame was taken). The `trigger.kind` is `person`, `object`, `warning_sign` or `qr`.

These rules apply on the Pi (`~/usb-camera-perception/scene_events.py`):

- **Confirmation (operator 2026-11-26: one sighting).** A single frame is enough: the first sighting sends
  `scene_pending`, which holds the dog, and starts the Claude request (`PI_CONFIRM_FRAMES=1`,
  `PI_CONFIRM_WINDOW=1`; before this it was 2 of 3 frames). Unidentified yellow triangles never trigger.
- **One encounter per sighting.** A label that is not seen for 8 s (`PI_ENCOUNTER_GAP_S`) starts a new
  encounter. The Pi has no pose, so the NUC deduplicates repeat encounters of the same target by arena cell.
- **Still frames only (operator 2026-11-26).** Claude only sees a frame taken after at least 1 s
  (`PI_STILL_S`) with no non-zero `/nav_cmd_vel` command. A request starts with `scene_pending`, which
  makes the NUC hold navigation, and then waits for the first still frame. If the dog is already
  standing, that is the same frame. With no still frame within 15 s (`PI_STILL_WAIT_S`), the request
  ends with `scene_failed` and `"error": "no_still_frame"`. Floor tilt is not checked.
- **Rate limit.** At most one Claude request runs at a time, with at least 10 s
  (`--scene-interval`) between starts. Encounters confirmed while a request is busy are merged into the
  next request. A batch dropped because the queue is full emits `scene_failed` with
  `"error": "dropped_rate_limited"`, and no `scene_pending` is ever sent for it.

## Navigation hold (`nav_hold.py`)

- The node holds once per encounter, on the first `scene_pending` that contains a new encounter
  ID, and only one hold is active at a time.
- It releases on the matching `request_id`'s `scene_captured` (the still frame is taken, so the dog may
  move on), `scene_result` or `scene_failed`, or after `--timeout` seconds (default 5). It also releases on
  shutdown.
- It skips a `scene_pending` whose capture is older than the timeout.
- It publishes `{"hold": bool, "request_id", "encounter_ids", "reason", "expires_wall_ns", "stamp_ns"}`.
  It never sends velocity commands and never clears a stop it did not request.

Roam needs these changes, which are not done yet:

1. Subscribe to `/group8/navigation/hold_request` with RELIABLE and TRANSIENT_LOCAL QoS.
2. When `hold` is true and `time.time_ns() < expires_wall_ns`, finish the current motion and stop at
   the next safe stopping point. Do not stop mid-step on stairs or terrain, and do not stop while
   crossing a curtain.
3. Combine this hold with roam's own manual stop and e-stop using OR. Resume only when all of them
   are clear. A hold message whose `expires_wall_ns` has passed counts as clear, so a crashed hold
   node cannot freeze the robot.

## Mission reporter (`pi_mission_reporter.py`)

Run it as a dry run first, then live:

```bash
# run the reporter in the ROS domain 8 environment (source ~/lite3_ws/operator/env.sh)
python3 pi_mission_reporter.py --origin X,Y,YAW_DEG                  # dry run: prints payloads
python3 pi_mission_reporter.py --origin X,Y,YAW_DEG --live --endpoint A
```

- **Group ID.** It is read from `group_id.txt` (one line, currently `group08_RJ08`) or `--group-id`.
  `--live` refuses an ID that does not match `group08_` plus 4 characters.
- **Missions.** `--missions-url` makes one GET that retrieves all five. Without it, the reporter uses
  the booklet v6.4 list.
- **Scene joins.** A scene joins a detection only on the same `(capture_stamp_ns, frame_id)`. A scene that
  arrives before its detections is held for 30 s.
- **Pose.** The pose is `map → base_link` at `capture_stamp_ns`, from a 120 s TF buffer
  (`--tf-history`). If TF cannot answer for that instant, the frame stays pending and is retried. It is
  never replaced by a later pose.
- **Arena coordinates.** Positions are relative to `--origin`, which should be the start pose captured
  by the operator launcher. Without it, the pose at startup is used, so start the reporter in the start
  cell facing arena +X. Positions are snapped to 1.02 m cell centres.
- **Survivors.**
  - A survivor counts as confirmed after 3 person frames in one cell plus a scene from that visit.
  - M03 is the survivor closest to a decoded QR. A survivor in the QR's own cell is taken at once; a
    neighbouring survivor only after the QR visit ends. Two equally close survivors leave M03 pending.
  - M05 waits for M03. It then takes a different cell, preferring one away from every QR.
- **Duplicates.**
  - A task is done only when the server answers `"status": "accepted"` with the same `task_id`.
  - A rejection reopens the task, but the same payload is never sent again.
  - "Accepted" means the format was valid, not that the answer is correct; the practice server
    also returns `score`.
- **Clock check.** The reporter compares Pi capture stamps with NUC receive time. It warns if the Pi
  is more than 50 ms ahead, or if the median lag exceeds 1.5 s.

Tests: `python3 -m pytest -q -p no:cacheprovider test_pi_mission_reporter.py` (24 tests, including a
mock scoring server and a Pi-scheduler-to-hold end-to-end run).

## Room scans (2026-11-26)

The camera (Orbbec) is on the Pi, so the Pi takes every photo. The NUC's `scene_capture.py --relay` cannot get a
frame on the NUC and is no longer needed for room scans.

At a room cell, roam (`roam.py`, `_capture_tick` → posture look with `scan`) runs one posture-look round,
left then right. At each side, once the view has settled, it publishes:

`/group8/navigation/capture_request` (String JSON, RELIABLE): `{"id", "scan_id", "view_id": "left"|"right"|"extra-<side>", "final", "qr_only", "reason": "room", "cell", "pose", "session"}`.

The Pi's `publish_perception.py` drops each request into a RAM inbox (`/dev/shm/group8pi_capture`). From there,
`rgbd_perception.py` (`room_scan.RoomScanner`) handles it:

- **Capture.** Each view is taken from the first STILL frame captured ≥ 0.5 s after the request arrived. It is
  stored at full resolution and its QR codes are decoded locally, with the text kept exactly as read.
- **Ack per view.** `/group8/navigation/capture_ack` `{"id", "status": "captured"|"timeout", "scan_id",
  "view_id", "capture_stamp_ns", "frame_id", "qr", "stored"}`. A view times out after 6 s. Roam retries once
  (2 attempts, 8 s each), then moves on.
- **One Claude request.** After the final view, every view of the scan goes to Claude together (`claude-opus-5-5`,
  structured JSON, toy props read as real ones). A `qr_only` close-up is decoded locally, with no Claude call.
  A scan whose final view never comes is sent after 20 s with whatever views it has.
- **Results.** `/group8pi/perception/room_scans` (RELIABLE): `room_scan_view`, `room_scan_pending`,
  `room_scan_result` `{"scan_id", "views": [{view_id, capture_stamp_ns, frame_id, pose, qr}], "room": {type,
  evidence: [{view_id, cue}]}, "findings": [{kind: person|hand_tools|hazard_sign|qr_code, label, view_id,
  evidence, clear}], "qr": [{view_id, text, source: "local"}], "additional_view": {needed, view_id, reason},
  "dry_run"}`, and `room_scan_failed`.
- **Extra view.** Centred, roam waits up to 10 s for the result. When `additional_view.needed`, it looks once
  more toward that side (`extra-<side>`, `qr_only` when only a QR is unclear).

`pi_mission_reporter.py` turns each view into an observation at its own capture stamp and frame ID, with the
room Claude named. The pose is `map→base_link` at that stamp, and positions use the arena cell convention.
Missions are assigned as follows:

- **M01:** `hand_tools`.
- **M02:** the exact hazard label from the symbol.
- **M03:** a clear survivor nearest the decoded QR.
- **M04:** the local QR text, unchanged.
- **M05:** the other survivor, held pending until M03 is known.

Unclear findings and an `unknown` room are not reported. Re-sent results and revisits are deduplicated by
scan/view and by cell. Scans marked `dry_run` are ignored unless the reporter is itself in dry run.
