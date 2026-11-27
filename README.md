# Live camera and objective store

[Command reference](COMMANDS.md)

Run in the group8 Remote Desktop session. This folder reuses the existing
scene-understanding Python environment, model and camera helpers, alongside the
objective package's SQLite store. Run one live recorder per database.

```bash
cd ~/live_mission
./run_live.sh
```

Press **Q** or Ctrl+C to stop. For a short camera check, use
`./run_live.sh --headless --seconds 30`. This uses the Orbbec RGB-D camera and
starts no navigation nodes. The local detector needs no API key.
The default uses only the Orbbec RGB-D camera for preview, detection, saved
evidence, and candidate map positions. Pressing S saves the annotated Orbbec
inference frame. `--rgbd` explicitly selects the same mode.

Each launch creates a timestamped library under `data/runs/` with its own
`mission.sqlite3` database. It prints the path at startup and saves results
as they arrive. Pressing **Q**, pressing **Ctrl+C**, or reaching the requested
duration leaves the library on disk. Manually saved images and optional sign
results go in the same library.

The database contains JPEG frames and detection metadata from the live camera,
sampled every five seconds by default.
Adjust the cadence with `--sample-interval SECONDS`. Newly decoded QR text is
recorded immediately with its evidence frame, once per distinct text per
database. A new YOLO `person` sighting is saved immediately with its evidence
frame. Overlapping detections are tracked across frames and saved once per
sighting, including printed depictions. A track expires one second after its
last match, so a person who returns may be saved again. The preview continues
to show all detections. QR text is stored verbatim. Object detections are
unverified mission observations with image coordinates. Yellow-sign candidates appear in the frame
metadata. This is a sampled record rather than a continuous video archive.
Database size grows with the run length and capture rate.

In another terminal, open the latest run’s summary:

```bash
cd ~/live_mission
./status.sh
```

With no path, status shows the most recently started library, even while the
recorder is running. Use `--all` for full JSON, including unverified detections
and evidence references, or `--confirmed` for JSON with confirmed observations.
Pass a database path to view an earlier library or the receiving mission node.

```bash
./status.sh --all
./status.sh /path/to/earlier/mission.sqlite3
```

To load objectives for a run:

```bash
./run_live.sh --objectives /path/to/your/objectives.json
./status.sh
```

An optional `--db PATH` places the fresh database at a custom path. If you
specifically want to continue an existing database, use `--resume PATH`.
That database will include the earlier records, including decoded QR clues.

When a database has objectives, the recorder saves only object detections
whose YOLO labels match those targets. A `survivor` objective enables YOLO's
`person` label, including printed depictions. Decoded QR records still save
independently. This filter applies to new writes, and the camera preview still
shows the detector output. A database with no objectives saves all YOLO classes.

Each objective needs an `id` and `target`. The existing store matches
objective targets to verified observation labels. Orbbec depth can provide
candidate map coordinates when calibration and TF are available. Completing
objectives still requires separately validated evidence. QR text never
creates objectives.

The mission records link to evidence using the frame ID in `evidence_ref`.
To export the newest JPEG from a run, pass the database path printed by status:

```bash
python3 - /path/to/run/mission.sqlite3 <<'PY'
from contextlib import closing
from pathlib import Path
import sqlite3
import sys

database = Path(sys.argv[1]).resolve().as_uri() + "?mode=ro"
with closing(sqlite3.connect(database, uri=True)) as db:
    row = db.execute(
        "SELECT image_jpeg FROM camera_frames ORDER BY observed_at DESC LIMIT 1"
    ).fetchone()
if row is None:
    raise SystemExit("No camera frames recorded")
Path("latest.jpg").write_bytes(row[0])
PY
```

## Remembering people after they leave view

Run the optional BoxMOT BoT-SORT and OSNet mode with a fresh test database:

```bash
./run_live.sh --reid
./status.sh --all
```

BoT-SORT handles short gaps. After a track expires, the recorder compares the
returning detection with anonymous appearance embeddings from this run. A
strong match with a clear lead over the next candidate suppresses a duplicate
observation. Two simultaneous detections cannot claim the same identity.
The match thresholds are provisional. Clothing, viewpoint and image quality
can change the result, especially for printed depictions.

The appearance memory is held in RAM and resets when the recorder stops.
The tracker uses the front camera alone. The default recorder command
continues to use bounding-box overlap for short-term deduplication.

## Search wide, locate with Orbbec

```bash
./run_live.sh --wide-search
./status.sh
```

Wide-search processes the existing wide feed at up to 4 frames per second
(`--scout-fps` overrides the processing limit). Live Mission opens its Orbbec
RGB, depth, and calibration subscriptions when a saved target needs inspection.
Each handoff gives the other camera up to about two seconds to return fresh
evidence. Nearby handoffs reuse the subscriptions. After three idle seconds,
Live Mission releases them and discards cached frames. It leaves the shared
Orbbec driver and other subscribers alone.

A QR decoded with the wide camera needs no handoff for scoring. If the wide
camera sees an unreadable QR, the Orbbec can decode it after a unique visual
match, even without a map transform. Use `--locate-wide-qr` to also request
depth for QR codes already decoded by the wide camera. Objects and tracked
warning-sign candidates need a unique visual match before RGB-D positioning.
If depth, transforms or matching are unavailable, the wide evidence remains
a candidate. The usual objective filter applies. An optional
`--scene-understanding` classifies saved wide views for v6.3 task attribution.

Keep navigation mapping running during wide search to provide the `map`
transform. The run summary shows positions and explains why a sighting could
not be located. In `./status.sh --all`,
`evidence_ref` is the wide image for a new sighting and
`location_evidence_ref` is the Orbbec image used for the estimate.
`location_match` records how the two views were associated. Positions remain
unverified. Use `--headless` to keep both feeds active without the preview.

## RGB-D candidate locations

Orbbec RGB-D supplies the preview, mission detections, and candidate locations
when started with the default command:

```bash
./run_live.sh
./status.sh --all
```

The recorder subscribes to the color and depth images and calibration on ROS
domain 8 by default. It uses nearby sensor timestamps, camera calibration
and TF to project native depth into the color image. A consistent depth
cluster near the center of a detected box can provide a candidate position
in `map`. Missing calibration, transforms or usable depth leaves the image
observation without a position.

Decoded QR clues use depth inside their four detected corners to estimate a
map position. `./status.sh` shows the coordinates with the clue. If the first
scan has no usable depth or transform, the clue still saves. A later scan can
add its position to that same record. `./status.sh --all` shows the original QR
evidence and the frame used for the position estimate.

For RViz, add a Marker display on `/live_mission/candidate_markers` with fixed
frame `map`. Each new saved observation with a position publishes an amber
sphere. The database stores the estimated position under `location`. Markers
are published while the recorder is running. Existing database records are
available through status.

RGB-D positions remain unverified. They do not complete objectives. This
mode does not add semantic detectors or change shared ROS configuration.

To send newly saved RGB-D QR and observation events to the existing ROS mission
node, add `--publish-mission-events`. Start the receiving node first and use a
**different database file for each process**. The recorder commits evidence
locally before publishing. Events are sent live, so the receiving node does
not receive earlier database records when it starts. Keep the recorder database
available because the event evidence references point to its saved images.
The event publisher does not interpret QR text or verify candidate locations.

## Competition scoring

Use `competition-v6.3-practice.json` for the booklet's wire format and 1.02 m
cell spacing. The example task list and `group08_TEST` are for practice.
Practice submissions are restricted to a loopback mock server. For the
competition, copy the profile, enter the registered group ID and final task
list, and set `mode` to `competition` and `organizer_contract_confirmed`
to `true` after checking them. Each task needs its booklet `task_type` and
internal evidence matcher. Use the organizer-provided scoring base URL
with `competition.sh submit --server`.

Scoring positions are two-dimensional arena cell centres. QR tasks send
the decoded text with a null position. Survivors send the scene and
position, with an empty wire label. The existing evidence validator
supplies task attribution in competition mode.


`competition-practice.json` illustrates M01–M05 with mock labels, cell sizes,
and wire fields. Use the organizers' final task list and API contract for a
competition profile. Set `mode` to `competition` and
`organizer_contract_confirmed` to true after entering the confirmed
`group_id`, X/Y `cell_size_m`, `scene_categories`, task labels, and `wire`
field names and coordinate format. Configure `wire.acceptance` if the server
reports acceptance in JSON. Each survivor has a separate single task ID.
The old objective list is still available for ordinary practice.

For a competition attempt, place the robot at the centre of the starting cell,
facing arena +X. Start a fresh navigation mapping session through the operator
workflow and wait for readiness before starting the recorder. The competition
launcher claims that navigation session once, rejects sessions with prior
roaming logs, starts a fresh database, and captures the start pose from
`map → base_link`. It uses that heading for arena +X and its left for arena
+Y. It does not reset a shared navigation stack or erase previous run logs.

```bash
./run_live.sh --rgbd --competition competition-v6.3-practice.json
# Use the database path printed at startup in another terminal.
./competition.sh --db /path/to/mission.sqlite3 status
```

The local status shows staged reports, original 3D measurements, submission
state, and robot-observed inspection candidates. For object reports, send
validated findings as JSON in `std_msgs/msg/String` on
`/live_mission/scoring_validation`. The validator supplies `task_id`,
`observation_id`, exact `label`, and `identification_confidence`. Survivors
also require a stable `entity_id` across observations. Supply `scene` and
entrance-relative `relation` and `relation_reference` where required by the
task. A task can set `scene_required` to require a category when its expected
room is unspecified. An optional validated `location_error_m` enables cell-centre conversion
when the uncertainty fits within one cell and is at most 0.5 m. Otherwise a
valid identification can stage a report with null position.

For QR tasks send the task ID and saved QR record ID as `observation_id`.
The report takes its message from the stored clue and sends null position.
Practice can stage one unambiguous decoded clue automatically. Competition
uses task attribution from the robot's validator. Practice also accepts a
validation JSON file via `./competition.sh --db PATH stage FILE`.
Report validation can include `{"type":"coverage","searched_cells":[[0,0]]}`
from the robot's coverage logic. Coverage is tracked independently of scoring.

After validating the organizer profile on a mock or practice setup, an
explicit sender command can post accepted improvements as evidence arrives:

```bash
./competition.sh --db /path/to/mission.sqlite3 submit --server http://localhost:8080 --watch
```

Practice profiles are restricted to loopback. The sender calls
`POST /task_report`, remembers acknowledged payloads, and skips repeat or
weaker results. The recorder itself does not send a score. The current
scene-understanding pipeline still needs a trusted validator for task-specific
semantics and localization accuracy, and roaming still needs an inspection
consumer before it can act on the advisory candidates.
