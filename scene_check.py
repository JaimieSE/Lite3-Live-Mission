#!/usr/bin/env python3
"""Manual scene assessment, the competition way, with no navigation or roam running.

  1. (--report live|dry) start pi_mission_reporter.py for this check; its log goes to a new
     ~/lite3_ws/log/operator/competition-<time>-scene-check/ folder so the dashboard's Mission reports panel shows it.
  2. Send the Pi one room scan: capture_request per view (same scan_id, final on the last view) for the room cell YOU
     give (--cell i,j: course/arena grid, start cell (0,0), +X start heading, +Y left). The NUC never estimates pos.
  3. Wait for each view's capture_ack and the Pi's room_scan_result (room type, findings, QR), print them, then give
     the reporter time to POST, and stop it.

The Pi takes each view from its first still frame after the request. Between views, point the camera at the next
part of the room yourself and press Enter (--no-wait: send the views back to back).

  ./scene_check.sh --cell 1,0 --report live --arena B          # competition: POSTs to task_report_B
  ./scene_check.sh --cell 1,0 --views centre,left,right --report dry
  ./scene_check.sh --cell 1,0 --auto --report live              # scan by itself when a QR / person shows up

--auto (operator 2026-11-27): watch /group8pi/perception/detections and start a room scan when the Pi decodes a QR
text not scanned yet, or when it reports a person box (if the Pi's person detector is on; 60 s cooldown). Claude's
room scan then confirms the person and the room. Runs until Ctrl+C; the reporter stays up the whole time.

Manual scene fallback (operator 2026-11-27), when autonomy is stuck and the operator drives by hand:
  ~/live_mission/scene_check.sh --fallback on --arena B [--live]
  ~/live_mission/scene_check.sh --fallback off
on:  pauses roam (operator/roam.sh pause: goal cancelled, map, Nav2, localization and the A1 origin stay up; never
     resumed here), reuses the running mission reporter or starts one (dry unless --live; --pos-source room_cell;
     --accepted-from the run's own log, so nothing already accepted is re-sent; listed in competition.pids so
     competition_start.sh stop ends it), and starts the watcher. The watcher reads map->base_link (TF) and roam's
     course grid + rooms (roam status.json). When the dog has stood still for STILL_S inside a known room, or in the
     cell in front of a room facing into it (within FACE_DEG of a grid axis), it sends ONE centre view for that room
     cell: once per visit (re-armed when the dog leaves its cell), one scan pending at a time, up to
     MAX_ATTEMPTS attempts for a blurry/failed capture, results accepted up to RESULT_WAIT_S after the request.
     Stale pose, lost localization, no grid or an ambiguous doorway: no scan, reason logged. It publishes only
     capture requests: never a velocity, mode or gait command.
off: stops the watcher (scan ownership back to roam). Roam stays paused until the operator runs operator/roam.sh
     resume. The reporter keeps running (it only reports Pi results; competition_start.sh stop ends it).
Without roam (teleoperating from the start, operator 2026-11-27): start dry navigation with the dog in A1 facing
arena +X (operator/start_navigation.sh), then --fallback on. The watcher builds the course grid itself from /map with
roam's own CourseGrid, wall_gaps and reachability (origin = map frame origin = where navigation started, or
--origin X,Y,YAW_DEG), and logs the room list whenever it changes. Rooms you drive into still count (roam would drop a
visited cell from its rooms). A new run folder competition-<time>-manual is used when no roam is alive.
Log: <run folder>/scene_fallback.log and scene_fallback.jsonl (each scan: scan_id, target cell, map pose, room-cell
centre in map frame - a cell centre, not a measured target - and the Pi's capture stamps / frame IDs).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

CAPTURE_REQUEST_TOPIC = '/group8/navigation/capture_request'
CAPTURE_ACK_TOPIC = '/group8/navigation/capture_ack'
ROOM_SCAN_TOPIC = '/group8pi/perception/room_scans'
DETECTIONS_TOPIC = '/group8pi/perception/detections'
PERSON_COOLDOWN_S = 60.0
PERSON_MIN_CONF = 0.5


def triggers(detection_json):
    """(qr_texts, person_seen) from one Pi detections message."""
    try:
        msg = json.loads(detection_json)
    except (TypeError, ValueError):
        return [], False
    if not isinstance(msg, dict):
        return [], False
    qr = [q.get('text') for q in msg.get('qr_codes') or [] if isinstance(q, dict) and q.get('text')]
    person = any(isinstance(o, dict) and o.get('name') == 'person' and (o.get('confidence') or 0) >= PERSON_MIN_CONF
                 for o in msg.get('objects') or [])
    return qr, person
ROOT = Path(__file__).resolve().parent
LOG_ROOT = Path(__import__('os').environ.get('SCENE_CHECK_LOG_ROOT') or Path.home() / 'lite3_ws' / 'log' / 'operator')
LITE3_WS = Path.home() / 'lite3_ws'
ROAM_DIR = LOG_ROOT / 'roam'
FALLBACK_STATE = LOG_ROOT / 'scene_fallback.json'
STILL_S, STILL_M, STILL_DEG = 1.0, 0.05, 3.0          # "the dog stops": pose this steady for this long
FACE_DEG = 25.0                                        # facing a neighbour cell: within this of a grid axis
POSE_STALE_S = 3.0                                     # map->base_link stamp older than this: stale (the LIO
                                                       # chain lags 1-2 s, measured 2026-11-27: median 1.75 s)
RESULT_WAIT_S = 60.0                                   # a Claude answer took 10.3 s in a dry run: wait well past it
MAX_ATTEMPTS = 2                                       # bounded retry for blurry / failed captures


def scan_target(grid, rooms, pose):
    """(target room cell, None) or (None, why not). grid: lite3_movement_tests.course_grid.CourseGrid; rooms: set of
    (i, j); pose: (x, y, yaw) in map. Inside a room: that room. Else the room in the facing direction (grid axis,
    +i = grid heading, +j = its left). Facing between two axes next to a room: ambiguous doorway, no scan."""
    import math
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]          # 0, 90, 180, 270 deg from the grid heading
    here = grid.cell(pose[0], pose[1])
    if here in rooms:
        return here, None
    rel = math.degrees(math.remainder(pose[2] - grid.heading, math.tau)) % 360.0
    k = int(round(rel / 90.0)) % 4
    off_axis = abs(math.remainder(rel - 90.0 * k, 360.0))
    if off_axis > FACE_DEG:
        lo = int(math.floor(rel / 90.0)) % 4
        near = [(here[0] + dirs[i][0], here[1] + dirs[i][1]) for i in (lo, (lo + 1) % 4)]
        if any(c in rooms for c in near):
            return None, f'facing between grid axes ({off_axis:.0f} deg off): doorway could be {near}'
        return None, 'not in or facing a known room'
    ahead = (here[0] + dirs[k][0], here[1] + dirs[k][1])
    if ahead in rooms:
        return ahead, None
    return None, 'not in or facing a known room'


def parse_cell(text):
    i, j = (int(v) for v in text.split(','))
    return [i, j]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cell', type=parse_cell, metavar='I,J', help='room cell (pos = I*1.02, J*1.02)')
    p.add_argument('--fallback', choices=('on', 'off', 'run'), help='manual scene fallback (see above)')
    p.add_argument('--live', action='store_true', help='fallback: reporter POSTs to the arena server (default dry)')
    p.add_argument('--origin', default='0,0,0', metavar='X,Y,YAW_DEG',
                   help='fallback without roam: A1 pose in the map frame (default: map origin = navigation start)')
    p.add_argument('--views', default='centre', help='comma-separated view ids, final = the last (default centre)')
    p.add_argument('--report', choices=('off', 'dry', 'live'), default='off',
                   help='run the mission reporter for this check: live POSTs to the scoring server')
    p.add_argument('--arena', choices=('A', 'B'),
                   help="scoring arena; default: the running competition's (its reporter), else B")
    p.add_argument('--restart-reporter', action='store_true',
                   help='fallback on: replace the running reporter (new code) instead of reusing it; same run folder')
    p.add_argument('--arena-override', action='store_true',
                   help='allow an --arena different from the running competition (otherwise refused)')
    p.add_argument('--no-wait', action='store_true', help='do not wait for Enter between views')
    p.add_argument('--auto', action='store_true', help='scan by itself when a new QR text or a person is seen')
    p.add_argument('--detections-topic', default=DETECTIONS_TOPIC, help='(testing) watch this topic instead')
    p.add_argument('--view-timeout', type=float, default=10.0)
    p.add_argument('--result-timeout', type=float, default=60.0)
    p.add_argument('--report-wait', type=float, default=10.0, help='seconds the reporter gets after the result')
    args = p.parse_args(argv)
    if args.fallback != 'off':
        running, source = competition_arena()
        if args.arena is None:
            args.arena = running or 'B'
            print(f"Arena {args.arena} ({source if running else 'no running competition found; default B'})")
        elif running and args.arena != running and not args.arena_override:
            p.error(f"--arena {args.arena} disagrees with the running competition's arena {running} ({source}). "
                    f"Drop --arena to use {running}, or add --arena-override if you really mean {args.arena}.")
        elif running and args.arena != running:
            print(f"WARNING: arena {args.arena} overrides the running competition's {running} ({source})")
    if args.fallback == 'on':
        return fallback_on(args)
    if args.fallback == 'off':
        return fallback_off()
    if args.fallback == 'run':
        return fallback_run(args)
    if args.cell is None:
        p.error('--cell I,J is required (or use --fallback on|off)')
    views = [v.strip() for v in args.views.split(',') if v.strip()]

    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    reporter = folder = None
    if args.report != 'off':
        folder = LOG_ROOT / f"competition-{datetime.now():%Y%m%d-%H%M%S}-scene-check"
        folder.mkdir(parents=True)
        cmd = [sys.executable, str(ROOT / 'pi_mission_reporter.py'), '--endpoint', args.arena,
               '--report-log', str(folder / 'mission_reports.jsonl')]
        if args.report == 'live':
            cmd.append('--live')
        reporter = subprocess.Popen(cmd, stdout=(folder / 'reporter.log').open('w'), stderr=subprocess.STDOUT,
                                    cwd=str(ROOT))
        print(f'Reporter ({args.report.upper()}, arena {args.arena}) -> {folder}/reporter.log', flush=True)
        time.sleep(4.0)                                   # subscribed before the scan result can arrive
        if reporter.poll() is not None:
            sys.exit(f'Reporter exited early; see {folder}/reporter.log')

    rclpy.init()
    node = rclpy.create_node('group08_scene_check')
    reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    pub = node.create_publisher(String, CAPTURE_REQUEST_TOPIC, reliable)
    acks, results = {}, []
    current = {'scan_id': None}
    seen = {'qr': [], 'person': False}

    def on_ack(msg):
        try:
            ack = json.loads(msg.data)
        except ValueError:
            return
        if ack.get('scan_id') == current['scan_id']:
            acks[ack.get('id')] = ack

    def on_scan(msg):
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        if event.get('scan_id') == current['scan_id']:
            results.append(event)

    def on_detection(msg):
        qr, person = triggers(msg.data)
        seen['qr'].extend(t for t in qr if t not in seen['qr'])
        seen['person'] = seen['person'] or person

    node.create_subscription(String, CAPTURE_ACK_TOPIC, on_ack, reliable)
    node.create_subscription(String, ROOM_SCAN_TOPIC, on_scan, reliable)
    if args.auto:
        node.create_subscription(String, args.detections_topic, on_detection,
                                 QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT))

    def spin_until(done, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not done():
            rclpy.spin_once(node, timeout_sec=0.1)
        return done()

    spin_until(lambda: pub.get_subscription_count() > 0, 8.0)   # let discovery reach the Pi
    if pub.get_subscription_count() == 0:
        print('WARNING: no subscriber on the capture request topic yet (is the Pi service up?)', flush=True)
    def run_scan(why):
        scan_id = current['scan_id'] = f"scan-check-{uuid.uuid4().hex[:8]}"
        results.clear()
        print(f'Room scan {scan_id} ({why}): cell {args.cell} '
              f'(pos {[round(args.cell[0] * 1.02, 2), round(args.cell[1] * 1.02, 2)]}), views {views}', flush=True)
        for n, view in enumerate(views):
            if n and not args.no_wait and not args.auto:
                input(f'Point the camera for view "{view}", hold still, then press Enter... ')
            rid = f'{scan_id}-{view}'
            for attempt in (1, 2):
                rid_try = rid if attempt == 1 else f'{rid}-retry'
                req = {'id': rid_try, 'reason': 'room', 'cell': args.cell, 'pose': None, 'session': 'scene-check',
                       'scan_id': scan_id, 'view_id': view, 'final': n == len(views) - 1}
                pub.publish(String(data=json.dumps(req)))
                got = spin_until(lambda: rid_try in acks, args.view_timeout)
                ack = acks.get(rid_try) or {}
                print(f'  view {view}: {ack.get("status", "no ack")} {ack.get("summary") or ""}'.rstrip(), flush=True)
                if got and ack.get('status') == 'captured':
                    break
        spin_until(lambda: any(e.get('event') in ('room_scan_result', 'room_scan_failed') for e in results),
                   args.result_timeout)
        final = next((e for e in reversed(results) if e.get('event') in ('room_scan_result', 'room_scan_failed')), None)
        if final is None:
            print(f'No room_scan_result within {args.result_timeout:.0f}s', flush=True)
            return 1
        else:
            room = final.get('room') or {}
            print(f"Result: {final['event']} | room {room.get('type')} | cell {final.get('cell')} | "
                  f"claude {final.get('claude')} | error {final.get('error')}", flush=True)
            for cue in room.get('evidence') or []:
                print(f"  room cue ({cue.get('view_id')}): {cue.get('cue')}", flush=True)
            for f in final.get('findings') or []:
                print(f"  finding: {f.get('kind')} {f.get('label')!r} in {f.get('view_id')} clear={f.get('clear')} "
                      f"- {f.get('evidence')}", flush=True)
            for q in final.get('qr') or []:
                print(f"  QR ({q.get('view_id')}): {q.get('text')!r}", flush=True)
        return 0

    code = 0
    try:
        if not args.auto:
            code = run_scan('manual')
        else:
            scanned_qr, last_person = set(), -float('inf')
            print('Auto: watching the Pi for a new QR text or a person (Ctrl+C to stop)', flush=True)
            while True:
                rclpy.spin_once(node, timeout_sec=0.2)
                new_qr = [t for t in seen['qr'] if t not in scanned_qr]
                person = seen['person'] and time.monotonic() - last_person >= PERSON_COOLDOWN_S
                seen['person'] = False
                if new_qr or person:
                    scanned_qr.update(new_qr)
                    if person:
                        last_person = time.monotonic()
                    run_scan(f"QR {new_qr[0]!r}" if new_qr else 'person seen')
                    seen['qr'] = [t for t in seen['qr'] if t in scanned_qr]
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        if reporter is not None:
            time.sleep(args.report_wait)
            reporter.send_signal(2)
            try:
                reporter.wait(timeout=15)
            except subprocess.TimeoutExpired:
                reporter.kill()
            print(f'Reporter stopped. Posts: {folder}/mission_reports.jsonl', flush=True)
            log = folder / 'mission_reports.jsonl'
            for line in (log.read_text().splitlines() if log.exists() else []):
                rec = json.loads(line)
                print(f"  {rec['payload'].get('task_id')} -> HTTP {rec.get('http_status')} {rec.get('response')}",
                      flush=True)
            if not log.exists():
                print('  (no reports were sent)', flush=True)
    return code



# ------------------------------------------------------------------ manual scene fallback (operator 2026-11-27)
def _alive(pid):
    import os
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _roam_alive():
    try:
        return _alive(json.loads((ROAM_DIR / 'status.json').read_text()).get('pid'))
    except (OSError, ValueError):
        return False


def _run_folder():
    """With roam alive: the current competition run folder (newest competition-*, not a bench scene-check).
    Without roam (teleoperating from the start): a new competition-<time>-manual folder."""
    runs = [d for d in sorted(LOG_ROOT.glob('competition-*')) if d.is_dir() and not d.name.endswith('-scene-check')]
    if runs and _roam_alive():
        return runs[-1]
    folder = LOG_ROOT / f"competition-{datetime.now():%Y%m%d-%H%M%S}-manual"
    folder.mkdir(parents=True)
    return folder


def _reporters():
    """[(pid, argv)] of running pi_mission_reporter.py processes."""
    out = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        argv = [a.decode(errors='replace') for a in argv if a]
        if any(a.endswith('pi_mission_reporter.py') for a in argv):
            out.append((int(proc.name), argv))
    return out


def _arg(argv, name):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def competition_arena():
    """(arena 'A'/'B' or None, where it came from): the running mission reporters' --endpoint when they agree, else
    the newest physical competition run's launcher reporter.log ("POST .../task_report_X")."""
    import re
    ends = {(_arg(argv, '--endpoint') or 'A').upper() for _, argv in _reporters()}
    ends = {e[-1] if e.startswith('HTTP') else e for e in ends}
    if len(ends) == 1:
        return ends.pop(), 'running mission reporter --endpoint'
    for run in sorted(LOG_ROOT.glob('competition-*-physical'), reverse=True)[:1]:
        try:
            m = re.search(r'task_report_([AB])', (run / 'reporter.log').read_text(errors='replace')[:20000])
        except OSError:
            m = None
        if m:
            return m.group(1), f'{run.name}/reporter.log'
    return None, None


def fallback_on(args):
    import os
    import signal
    state = json.loads(FALLBACK_STATE.read_text()) if FALLBACK_STATE.exists() else {}
    if _alive(state.get('watcher_pid')):
        print(f"Fallback already on (watcher pid {state['watcher_pid']}); run --fallback off first to change it")
        return 1
    # a reporter already in the wanted mode keeps its run folder (watcher log and reports stay together)
    keep = None if args.restart_reporter else next((argv for _, argv in _reporters() if ('--live' in argv) == bool(args.live)
                 and (_arg(argv, '--endpoint') or 'A').upper() == args.arena and _arg(argv, '--report-log')), None)
    # a reporter being replaced (mode switch / restart) hands over its run folder: its accepted answers carry over
    replaced = [Path(_arg(argv, '--report-log')).parent for _, argv in _reporters() if _arg(argv, '--report-log')]
    if keep and Path(_arg(keep, '--report-log')).parent.is_dir():
        run = Path(_arg(keep, '--report-log')).parent
    elif replaced and replaced[0].is_dir():
        run = replaced[0]
    else:
        run = _run_folder()
    log_path = run / 'mission_reports.jsonl'
    # 1. hand driving to the operator: pause roam (never resumed here). Nav2, SLAM/map and the A1 origin stay up.
    roam = {}
    try:
        roam = json.loads((ROAM_DIR / 'status.json').read_text())
    except (OSError, ValueError):
        pass
    if roam.get('state') == 'ROAMING' and _alive(roam.get('pid')):
        out = subprocess.run([str(LITE3_WS / 'operator' / 'roam.sh'), 'pause'], capture_output=True, text=True,
                             timeout=20)
        print(f"Roam paused: {(out.stdout or out.stderr).strip()[:200]}")
        for _ in range(20):
            time.sleep(0.5)
            try:
                if json.loads((ROAM_DIR / 'status.json').read_text()).get('state') != 'ROAMING':
                    break
            except (OSError, ValueError):
                pass
        else:
            print('WARNING: roam still reports ROAMING; use the remote to take over and check operator/roam.sh status')
    else:
        print(f"Roam is {roam.get('state', 'not running')}{'' if _alive(roam.get('pid')) else ' (no live process)'}: "
              f"nothing to pause")
    print('Take the handheld remote (switch the robot to manual / remote driving there). This tool sends no motion.')
    # 2. mission reporter: reuse one in the right mode, else (re)start it with the run's accepted state
    want_live = bool(args.live)
    reporter_pid = None
    for pid, argv in _reporters():
        live = '--live' in argv
        endpoint = (_arg(argv, '--endpoint') or 'A').upper()
        if live == want_live and endpoint == args.arena and not args.restart_reporter:
            reporter_pid = pid
            print(f"Reusing mission reporter pid {pid} ({'LIVE' if live else 'DRY'}, arena {endpoint}, "
                  f"log {_arg(argv, '--report-log')})")
            break
        os.kill(pid, signal.SIGINT)
        print(f"Stopped reporter pid {pid} ({'LIVE' if live else 'DRY'}, arena {endpoint}) to switch mode")
        for _ in range(30):
            if not _alive(pid):
                break
            time.sleep(0.5)
    if reporter_pid is None:
        cmd = [sys.executable, str(ROOT / 'pi_mission_reporter.py'), '--endpoint', args.arena,
               '--report-log', str(log_path), '--pos-source', 'room_cell']
        for earlier in sorted(run.glob('mission_reports*.jsonl')):
            cmd += ['--accepted-from', str(earlier)]
        if want_live:
            cmd.append('--live')
        proc = subprocess.Popen(cmd, stdout=(run / 'reporter-fallback.log').open('a'), stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=str(ROOT), start_new_session=True)
        reporter_pid = proc.pid
        with (LOG_ROOT / 'competition.pids').open('a') as f:    # competition_start.sh stop ends it too
            f.write(f'reporter {reporter_pid}\n')
        print(f"Started mission reporter pid {reporter_pid} ({'LIVE' if want_live else 'DRY'}, arena {args.arena}) "
              f"-> {run}/reporter-fallback.log, reports {log_path}")
    # 3. the watcher
    watcher = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--fallback', 'run',
                                '--arena', args.arena, '--result-timeout', str(args.result_timeout),
                                '--origin', args.origin],
                               stdout=(run / 'scene_fallback.log').open('a'), stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, cwd=str(ROOT), start_new_session=True)
    FALLBACK_STATE.write_text(json.dumps({'watcher_pid': watcher.pid, 'reporter_pid': reporter_pid,
                                          'live': want_live, 'arena': args.arena, 'run': str(run),
                                          'since': datetime.now().isoformat()}))
    print(f"Manual scene fallback ON (watcher pid {watcher.pid}): scans when the dog stops in / facing a known room. "
          f"Log: {run}/scene_fallback.log")
    return 0


def fallback_off():
    import signal
    import os
    state = json.loads(FALLBACK_STATE.read_text()) if FALLBACK_STATE.exists() else {}
    pid = state.get('watcher_pid')
    if _alive(pid):
        os.kill(int(pid), signal.SIGINT)
        for _ in range(20):
            if not _alive(pid):
                break
            time.sleep(0.5)
        print(f"Fallback watcher pid {pid} stopped" + (' (still alive: check it)' if _alive(pid) else ''))
    else:
        print('Fallback was not running')
    FALLBACK_STATE.unlink(missing_ok=True)
    print('Scan ownership is back with roam. Roam stays paused: resume it yourself with operator/roam.sh resume. '
          'The mission reporter keeps running (competition_start.sh stop ends it).')
    return 0


def fallback_run(args):
    """The watcher process (started by --fallback on)."""
    import math
    import rclpy
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import qos_profile_system_default
    from lite3_movement_tests.course_grid import CourseGrid, MIN_FREE_M2, ROOM_WALLS
    from lite3_movement_tests import wall_gaps as WG
    from lite3_movement_tests.roam_ros import grid_from_occupancy
    ox, oy, oyaw = (float(v) for v in args.origin.split(','))

    run = Path(json.loads(FALLBACK_STATE.read_text())['run']) if FALLBACK_STATE.exists() else _run_folder()
    record = (run / 'scene_fallback.jsonl').open('a')

    def log(text, **fields):
        print(f"{datetime.now():%H:%M:%S} {text}", flush=True)
        if fields:
            record.write(json.dumps(dict(fields, time=time.time(), text=text)) + '\n')
            record.flush()

    rclpy.init()
    node = rclpy.create_node('group08_scene_fallback')
    tf = Buffer(cache_time=Duration(seconds=10.0))
    TransformListener(tf, node)
    reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    pub = node.create_publisher(String, CAPTURE_REQUEST_TOPIC, reliable)
    acks, results = {}, {}
    node.create_subscription(String, CAPTURE_ACK_TOPIC, lambda m: _keep(acks, m.data, 'id'), reliable)
    node.create_subscription(String, ROOM_SCAN_TOPIC, lambda m: _keep(results, m.data, 'scan_id'), reliable)
    latest_map = {}
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(OccupancyGrid, '/map', lambda m: latest_map.update(msg=m), latched)
    own = {'grid': None, 'rooms': set(), 'stamp': None, 't': 0.0, 'crumbs': []}

    def own_grid(pose, now):
        """No roam: the course grid from /map, as roam's grid worker computes it (rooms ignore 'visited')."""
        if pose is not None and (not own['crumbs'] or math.hypot(pose[0] - own['crumbs'][-1][0],
                                                                  pose[1] - own['crumbs'][-1][1]) >= 0.3):
            own['crumbs'].append((pose[0], pose[1]))
        msg = latest_map.get('msg')
        if msg is None:
            return None, set()
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec, msg.info.width, msg.info.height)
        if stamp != own['stamp'] and now - own['t'] >= 3.0 and pose is not None:
            grid = grid_from_occupancy(msg)
            gaps = WG.gap_mask(grid, own['crumbs'])
            reach = WG.reachable_mask(grid, pose[:2], gaps[0], own['crumbs'])
            g = CourseGrid(ox, oy, math.radians(oyaw))
            g.update(grid, None, own['crumbs'], now, walls=gaps[0], reach=reach)
            rooms = {c for c, info in g.cells.items() if info.get('walls', 0) >= ROOM_WALLS
                     and info['free_m2'] >= MIN_FREE_M2 and info.get('reachable', True)}
            if rooms != own['rooms']:
                log(f"rooms (from the map, A1 = [0,0]): {sorted(rooms)}", event='rooms', rooms=sorted(rooms))
            own.update(grid=g, rooms=rooms, stamp=stamp, t=now)
        return own['grid'], own['rooms']
    poses = []                    # (t, x, y, yaw) for the still check
    armed_from = None             # the cell the dog stood in when it last scanned: re-armed after leaving it
    pending = None                # dict(scan_id, rid, target, sent, attempt, pose)
    grid_now = None
    last_note = {}

    def note(key, text):
        import re
        kind = re.sub(r'[0-9.]+', '#', text)                # 'stale (1.5s old)' and '(1.6s old)' are one reason
        if last_note.get(key) != kind or time.monotonic() - last_note.get(key + '_t', 0) > 10:
            last_note[key], last_note[key + '_t'] = kind, time.monotonic()
            log(f'no scan: {text}', event='skip', reason=text)

    def send(target, pose, attempt, grid):
        end = time.monotonic() + 5.0          # a request published before the Pi's subscription matches is lost
        while pub.get_subscription_count() == 0 and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            log('WARNING: no Pi subscriber on the capture request topic; sending anyway')
        scan_id = f"fallback-{target[0]}_{target[1]}-{uuid.uuid4().hex[:8]}"
        rid = f'{scan_id}-centre'
        req = {'id': rid, 'reason': 'room', 'cell': list(target), 'pose': [round(v, 3) for v in pose],
               'session': 'scene-fallback', 'scan_id': scan_id, 'view_id': 'centre', 'final': True}
        pub.publish(String(data=json.dumps(req)))
        cx, cy = grid.centre(*target)
        log(f"scan {scan_id}: room cell {list(target)} (report pos {[round(target[0] * 1.02, 2), round(target[1] * 1.02, 2)]})"
            f" from map pose ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg), attempt {attempt}",
            event='scan_request', scan_id=scan_id, request=req,
            internal_location={'room_cell_centre': {'frame_id': 'map', 'x': round(cx, 3), 'y': round(cy, 3),
                                                    'kind': 'room cell centre (course grid), not a measured target'},
                               'measured_target': None})
        return dict(scan_id=scan_id, rid=rid, target=target, sent=time.monotonic(), attempt=attempt, pose=pose)

    log(f"watching map->base_link; arena {args.arena}; one centre view per room visit")
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            # ---- a pending scan: wait for its answer (late answers welcome), bounded retries
            if pending is not None:
                ack = acks.get(pending['rid'])
                res = results.get(pending['scan_id'])
                if res and res.get('event') in ('room_scan_result', 'room_scan_failed'):
                    ok = res['event'] == 'room_scan_result'
                    room = (res.get('room') or {}).get('type')
                    log(f"result {pending['scan_id']} after {now - pending['sent']:.1f}s: {res['event']} room {room}, "
                        f"findings {[(f.get('kind'), f.get('label')) for f in res.get('findings') or []]}, "
                        f"QR {[q.get('text') for q in res.get('qr') or []]}",
                        event='scan_result', scan_id=pending['scan_id'], cell=res.get('cell'), ok=ok, room=room,
                        findings=res.get('findings'), qr=res.get('qr'), error=res.get('error'),
                        views=[{k: v.get(k) for k in ('view_id', 'capture_stamp_ns', 'frame_id')}
                               for v in res.get('views') or []])
                    retry = not ok
                elif ack and ack.get('status') in ('blurry', 'timeout'):
                    log(f"view of {pending['scan_id']}: {ack['status']}", event='scan_view', scan_id=pending['scan_id'],
                        status=ack['status'])
                    retry = True
                elif now - pending['sent'] > args.result_timeout:
                    log(f"no result for {pending['scan_id']} within {args.result_timeout:.0f}s", event='scan_timeout',
                        scan_id=pending['scan_id'])
                    retry = True
                else:
                    continue
                if retry and pending['attempt'] < MAX_ATTEMPTS and grid_now is not None:
                    pending = send(pending['target'], pending['pose'], pending['attempt'] + 1, grid_now)
                else:
                    pending = None
                continue
            # ---- where is the dog, is it still, which room is it in / facing
            grid_now = None
            try:
                t = tf.lookup_transform('map', 'base_link', Time())
            except TransformException as exc:
                note('tf', f'localization lost (no map->base_link: {str(exc)[:80]})')
                continue
            age = (node.get_clock().now() - Time.from_msg(t.header.stamp)).nanoseconds / 1e9
            if age > POSE_STALE_S:
                note('tf', f'map pose stale ({age:.1f}s old)')
                continue
            q = t.transform.rotation
            pose = (t.transform.translation.x, t.transform.translation.y,
                    math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
            stamp = Time.from_msg(t.header.stamp).nanoseconds / 1e9
            cg = None
            try:
                status = json.loads((ROAM_DIR / 'status.json').read_text())
                cg = status.get('course_grid') if _alive(status.get('pid')) or status.get('pid') is None else None
            except (OSError, ValueError):
                pass
            if cg:                                              # roam alive: its grid and rooms
                grid_now = CourseGrid(cg['origin'][0], cg['origin'][1], cg['heading_rad'], cg.get('cell_m', 1.0))
                rooms = {tuple(c) for c in cg.get('rooms') or []}
            else:                                               # teleoperated from the start: our own grid
                grid_now, rooms = own_grid(pose, now)
                if grid_now is None:
                    note('grid', 'no /map yet (is navigation running?)')
                    continue
            if not poses or stamp > poses[-1][0]:               # still-ness by the poses' own stamps: a stalled
                poses = [p for p in poses if stamp - p[0] <= STILL_S + 1.0] + [(stamp, *pose)]   # TF never looks still
            here = grid_now.cell(pose[0], pose[1])
            if armed_from is not None and here != armed_from:
                armed_from = None                                   # left the cell: the next visit may scan
            if armed_from is not None:
                continue
            window = [p for p in poses if stamp - p[0] <= STILL_S]
            if len(window) < 3 or poses[0][0] > stamp - STILL_S or any(
                    math.hypot(p[1] - pose[0], p[2] - pose[1]) > STILL_M or
                    abs(math.degrees(math.remainder(p[3] - pose[2], math.tau))) > STILL_DEG for p in window):
                continue                                            # moving (or not long enough still)
            target, why = scan_target(grid_now, rooms, pose)
            if target is None:
                note('target', why)
                continue
            armed_from = here
            pending = send(target, pose, 1, grid_now)
    except KeyboardInterrupt:
        pass
    finally:
        log('fallback watcher stopped')
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def _keep(store, text, key):
    try:
        msg = json.loads(text)
    except (TypeError, ValueError):
        return
    if isinstance(msg, dict) and msg.get(key):
        if key == 'scan_id' and msg.get('event') not in ('room_scan_result', 'room_scan_failed'):
            return
        store[msg[key]] = msg


if __name__ == '__main__':
    sys.exit(main())
