# Live mission command reference

Run these commands in the **group8 Remote Desktop session**. Keep the
recorder, results view, and navigation in separate terminals. Each recorder
launch creates a fresh library under `data/runs/` and saves it as it records.

## Start perception

Orbbec-only preview, detection, and depth with short-term person deduplication:

```bash
cd ~/live_mission
./run_live.sh
```

Orbbec RGB-D with BoT-SORT and appearance matching after re-entry:

```bash
cd ~/live_mission
./run_live.sh --reid
```

Wide camera only, including for detection:

```bash
cd ~/live_mission
./run_live.sh --wide
```

`--rgbd` explicitly selects the default Orbbec-only mode. Add `--headless --seconds 30` to record for 30 seconds without a preview.
In the preview, press **S** to save the current evidence frame and **Q** to
stop. **Ctrl+C** also stops the recorder.

Search for targets with the wide camera and estimate map positions with
Orbbec depth:

```bash
cd ~/live_mission
./run_live.sh --wide-search
```

Wide-search wakes the Orbbec subscriptions only for a handoff. QR codes already
decoded on the wide camera skip depth unless you add `--locate-wide-qr`.

## View results

In another terminal, view the most recent run:

```bash
cd ~/live_mission
./status.sh
./status.sh --all
./status.sh --all | less
```

The summary shows objective progress, observation counts, recent detections,
map positions, and QR clues. The full JSON includes unverified detections and
evidence references. In `less`, type `/person` and press **Enter** to search,
**n** to go to the next match, and **q** to quit. Rerun the command to refresh.
If the pager is hard to use, run the command without `| less`.

To open an earlier run, use `./status.sh /path/to/mission.sqlite3` with the
database path printed during that run. To continue writing to that database
explicitly, pass `--resume /path/to/mission.sqlite3` to the recorder.

For RGB-D map pins, set RViz **Fixed Frame** to `map` and add a **Marker**
display for `/live_mission/candidate_markers`. The pins are unverified
candidates.

## Load an objective list

Pass your actual objectives JSON to a fresh run. The command below uses the
bundled practice list as an example:

```bash
cd ~/live_mission
./run_live.sh --rgbd \
  --objectives ~/mission_rgb_01a1edca/src/rescue_mission/config/objectives.json
./status.sh
```

Loaded objectives filter saved object detections. A `survivor` target allows
YOLO's `person` class. QR clues are saved independently. If you already have a
separate objective node running, add `--publish-mission-events` to the RGB-D
recorder command to forward newly saved events. Give each process its own
database and point `./status.sh --summary` at the objective node's `db_path`
to see its results.

## Preview competition reports

With dry Nav running, start RGB-D mission recording and room classification
together:

```bash
./run_live.sh --rgbd --competition competition-v6.3-practice.json --scene-understanding
```

Room classification sends occasional saved RGB frames to Claude and prompts
for an API key if needed. The recorder keeps the linked classification and
stages matching task reports locally. Use `--max-requests` and `--cooldown`
to control calls. Entrance-relative findings still need supported relation
evidence. Scoring cell positions still need a verified location uncertainty.


```bash
cd ~/live_mission
./run_live.sh --rgbd --competition competition-v6.3-practice.json
./competition.sh --db /path/to/mission.sqlite3 status
```

Use the database path printed by the recorder. See the [competition profile
instructions](README.md#competition-scoring) before using an organizer server.

## Start and check navigation

In a separate terminal, run the read-only checks first:

```bash
cd ~/lite3_ws
operator/preflight.sh
```

### Robot stationary (dry navigation)

For recording while the robot stays still, after preflight reports `READY`:

```bash
operator/start_navigation.sh
operator/status.sh
```

This starts mapping, Nav2 and terrain monitoring in **dry mode**: no robot
commands. If navigation is already running, check `operator/status.sh` before
starting another instance.

### Robot roaming (autonomous run)

`operator/roam.sh` starts navigation itself and then drives the robot:

```bash
cd ~/lite3_ws
export LITE3_LIO_SOURCE=local
operator/stop_navigation.sh
operator/roam.sh start --physical
```

Type `ARM` when asked. Navigation starts (30–60 s) and prints
`NAVIGATION READY`, then roam asks you to type `ROAM`. The robot moves only
after `ROAM`, at regular (normal) speed.

**Start the recorder between `NAVIGATION READY` and typing `ROAM`.** A
competition attempt claims the navigation session at the start pose and
refuses a session that has already roamed. It also needs that navigation run's
readiness verdict to be `READY`; a `DEGRADED` start (for example LiDAR-link
warnings) is refused. Add `--publish-mission-events` so roam logs each finding
in its run report:

```bash
cd ~/live_mission
./run_live.sh --rgbd --publish-mission-events
```

Roam only logs mission events; they do not change how it moves. While it runs:

```bash
operator/roam.sh status
operator/roam.sh pause
operator/roam.sh resume
operator/roam.sh stop
```

The handheld remote STOP works at all times.

### Stop navigation

When finished, dry or roaming:

```bash
operator/stop_navigation.sh
```

See the [live mission README](README.md) and the
[Lite3 operator guide](../lite3_ws/operator/README.md) for details.
