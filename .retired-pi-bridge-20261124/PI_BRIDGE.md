# Pi perception → NUC mission decisions

The Raspberry Pi (group8pi) runs the camera and all vision inference. The NUC (group8) only consumes
its compact JSON results and decides M01–M05. **Dry run only:** nothing here moves the robot or
submits a score.

## Who consumes what (ROS domain 8)

| Topic | Publisher | NUC consumer |
|---|---|---|
| `/group8pi/perception/detections` | Pi `group8pi_perception` (BEST_EFFORT, VOLATILE) | `pi_bridge.py` only |
| `/group8pi/perception/scenes` | Pi `group8pi_perception` (BEST_EFFORT, VOLATILE) | `pi_bridge.py` only |
| `/live_mission/scoring_validation` | (unchanged) | `live_perception.py --competition` |
| `/mission/events` | `live_perception.py --publish-mission-events` | `rescue_mission mission_node` |

`pi_bridge.py` is a separate process. It does not write to a recorder database, so it can't
conflict with `live_perception.py` or `competition.sh`.

## Run

```bash
cd ~/live_mission
./pi_bridge.sh --dry-run --missions competition-v6.3-practice.json
```

Each run writes `data/pi/<UTC>/messages.jsonl`, which holds every received message with its receive time,
and `status.json`, which holds the decisions, buffer statistics and the latest stair/curtain state. To re-run the logic offline
on a recording:

```bash
python3 pi_bridge.py --dry-run --missions competition-v6.3-practice.json --replay data/pi/<UTC>/messages.jsonl
```

Tests: `python3 -m pytest -q -p no:cacheprovider tests/test_pi_mission.py`

## Decision rules (`pi_mission.py`)

- **Frames and scenes.** Frames are keyed by `(capture_stamp_ns, frame_id)`. A scene result joins only
  the frame with both the same stamp and the same frame_id. A scene that arrives before its detections
  is held for `--scene-timeout` (20 s). A frame without a scene after that timeout is *scene-unknown*.
  A later scene still attaches while the frame is buffered (5 min).
- **Unknown scenes.** A scene of type `unknown` counts as no scene. So does a scene without evidence.
- **Encounters.** Sightings of the same thing less than 5 s apart on the Pi clock form one encounter (`--encounter-gap`; the Pi produces about 0.6 frames/s). A
  scene counts for a sighting only if it came from a frame of that same encounter.
- **M01.** Needs an object whose label is in the task's labels. If the task requires a scene, that
  scene must be known.
- **M02.** A yellow triangle needs a hazard identification that matches the task's labels. YOLO
  "triangle" alone stays pending.
- **M03.** Needs a confirmed survivor: a person seen in at least 2 frames of one encounter, in the
  described scene. If the task also names an entrance-relative relation, M03 stays pending, because
  the Pi results do not show that relation.
- **M04.** Reports the decoded QR payload. Two or more different payloads stay pending until it is
  clear which one the task means.
- **M05.** Survivors are deduplicated to one identity per scene. M05 takes a confirmed identity outside
  M03's described scene, so it can resolve before M03. A survivor with an unknown scene stays pending.
- **Stairs and curtains.** `unknown`, `stale` and `timestamp_mismatch` observations are reported as
  unknown.

## Status (2026-11-24)

- **Pi → NUC link works.** 150 of 150 test messages arrived on `/group8pi/link_test`, sent with the
  Pi's own `nuc-dds.xml`. Access from the NUC is `ssh group8pi`, using key `~/.ssh/group8pi_ed25519`.
- **Message format checked** against the Pi source (`rgbd_perception.py`, `publish_perception.py`) and
  against 111 real detector frames (`tests/fixtures/`). The `frame_id` is the camera's frame and is the
  same on every frame, so frames are keyed by stamp and frame_id together.
- **Live receipt works.** On 2026-11-24 the NUC received 56 detection messages in 2 minutes. None were lost
  (Pi frames 496–551 all arrived) and all 56 parsed. The recording is in `data/pi/20261124-first-live/`.
  The Pi produces only about 0.6 frames/s.
- **No scene results yet.** The Pi service now sets `PI_SCENE_MODEL=claude-opus-5-5` and reads the
  API key from `~/.config/group8pi/claude.env`, but the key is missing, so scene recognition is off.
- **Stairs and curtains are mostly unusable.** 41 of 56 live frames report `timestamp_mismatch`.
  That needs fixing on the Pi side, and the NUC treats those frames as unknown.
- **No hazard identification.** Pi warning signs are yellow-triangle candidates only, so M02 cannot
  complete from Pi data yet.
- **No mission retrieval or submission.** The organizer API contract has not been found, and
  `--missions-url` is refused.

## When the camera is reconnected

1. On the Pi, `systemctl --user status group8pi-perception.service` should stay active and stop restarting.
2. On the NUC, run `./pi_bridge.sh --dry-run --missions competition-v6.3-practice.json --seconds 120`.
   You should see `DISCOVERY` lines for both topics and a nonzero `received` count in `status.json`.
3. With `PI_SCENE_MODEL` set on the Pi, check `scenes_matched` and `scenes_late` in `status.json`. Then
   replay `messages.jsonl` and confirm the scenes join the right frames.
