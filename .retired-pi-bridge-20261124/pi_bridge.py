#!/usr/bin/env python3
"""NUC consumer for the Raspberry Pi's perception topics; dry-run mission decisions only.

This is the one NUC process that subscribes to /group8pi/perception/detections and
/group8pi/perception/scenes (ROS domain 8). It feeds them to pi_mission.PiMission and writes the
M01-M05 decisions to a status file. It publishes nothing, commands no motion and submits nothing:
retrieval and submission wait for the organizer's API contract.

  ./pi_bridge.sh --dry-run --missions competition-v6.3-practice.json
  ./pi_bridge.sh --dry-run --missions ... --replay data/pi/<run>/messages.jsonl   # offline, no ROS
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

from pi_mission import DETECTIONS_TOPIC, SCENES_TOPIC, PiMission

ROOT = Path(__file__).resolve().parent


def load_tasks(path):
    data = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
    tasks = data.get('tasks') if isinstance(data, dict) else data
    if not isinstance(tasks, list) or not all(isinstance(t, dict) and t.get('id') for t in tasks):
        raise ValueError('Missions file needs a list of tasks with ids (a profile or a bare list)')
    return tasks


def write_status(path, mission, extra):
    status = dict(mission.status(), **extra, written_at=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2, default=list), encoding='utf-8')
    temporary.replace(path)
    return status


def summary(decisions):
    return {task_id: (d['status'], d.get('message') or d.get('label') or d.get('identity') or d.get('reason'))
            for task_id, d in decisions.items()}


def replay(mission, path, status_path, speed):
    """Feed a recorded messages.jsonl back through the decision logic with its original receive times."""
    first = None
    with Path(path).open(encoding='utf-8') as lines:
        for line in lines:
            row = json.loads(line)
            received = row['rx_mono'] if 'rx_mono' in row else row['rx_ns'] / 1e9
            first = received if first is None else first
            if speed:
                time.sleep(max(0.0, (received - first) / speed - (time.monotonic() - replay.started)))
            (mission.add_detections if row['topic'] == DETECTIONS_TOPIC else mission.add_scene)(
                row['data'], received)
            last = received
    mission.expire(last + mission.scene_timeout_s + 1 if first is not None else 0)
    return write_status(status_path, mission, dict(mode='replay', source=str(path)))


def run_live(mission, args, status_path, record_path):
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node('group8_pi_mission')
    qos = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    record = record_path.open('a', encoding='utf-8')
    counts = {DETECTIONS_TOPIC: 0, SCENES_TOPIC: 0}

    def handler(topic, add):
        def receive(message):
            received = time.monotonic()
            counts[topic] += 1
            record.write(json.dumps(dict(topic=topic, rx_mono=received, rx_ns=time.time_ns(),
                                         data=message.data)) + '\n')
            record.flush()
            add(message.data, received)
        return receive

    node.create_subscription(String, DETECTIONS_TOPIC, handler(DETECTIONS_TOPIC, mission.add_detections), qos)
    node.create_subscription(String, SCENES_TOPIC, handler(SCENES_TOPIC, mission.add_scene), qos)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    started = time.monotonic()
    next_write, last_summary, reported_discovery = started, None, False
    print(f'Listening on {DETECTIONS_TOPIC} and {SCENES_TOPIC} (BEST_EFFORT, VOLATILE). '
          f'Dry run: no motion, no submission. Status: {status_path}', flush=True)
    try:
        while not stop and rclpy.ok() and (not args.seconds or time.monotonic() - started < args.seconds):
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if not reported_discovery and now - started > args.discovery_wait:
                reported_discovery = True
                for topic in counts:
                    publishers = node.get_publishers_info_by_topic(topic)
                    if not publishers:
                        print(f'DISCOVERY: no publisher on {topic}', flush=True)
                    for endpoint in publishers:
                        print(f'DISCOVERY: {topic} <- {endpoint.node_namespace.rstrip("/")}/{endpoint.node_name} '
                              f'{endpoint.qos_profile.reliability.name}/{endpoint.qos_profile.durability.name}',
                              flush=True)
            if now >= next_write:
                next_write = now + 1.0
                mission.expire(now)
                status = write_status(status_path, mission, dict(mode='live-dry-run', received=dict(counts),
                                                                 uptime_s=round(now - started, 1)))
                current = summary(status['decisions'])
                if current != last_summary:
                    print('DECISIONS:', json.dumps(current), flush=True)
                    last_summary = current
    except KeyboardInterrupt:
        pass
    finally:
        mission.expire(time.monotonic())
        write_status(status_path, mission, dict(mode='live-dry-run', received=dict(counts), stopped=True))
        record.close()
        node.destroy_node()
        rclpy.shutdown()
        print(f'Received {counts}. Messages: {record_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', required=True,
                        help='Required: decisions only. Motion and official submission are not implemented.')
    parser.add_argument('--missions', type=Path, required=True,
                        help='Mission descriptions M01-M05: a competition profile or a list of tasks')
    parser.add_argument('--missions-url', help='Organizer mission retrieval (disabled until the contract is known)')
    parser.add_argument('--replay', type=Path, help='Replay a recorded messages.jsonl instead of subscribing')
    parser.add_argument('--speed', type=float, default=0.0, help='Replay speed factor; 0 = as fast as possible')
    parser.add_argument('--out', type=Path, help='Output folder (default data/pi/<UTC time>)')
    parser.add_argument('--seconds', type=float, default=0.0, help='Stop after this long (0 = until Ctrl+C)')
    parser.add_argument('--scene-timeout', type=float, default=20.0,
                        help='Seconds to wait for a scene result before treating the frame as scene-unknown')
    parser.add_argument('--discovery-wait', type=float, default=8.0)
    parser.add_argument('--encounter-gap', type=float, default=5.0,
                        help='Max seconds between sightings of one encounter (the Pi runs ~0.6 frames/s)')
    args = parser.parse_args()
    if args.missions_url:
        parser.error('Mission retrieval is disabled: the organizer API contract has not been confirmed')
    try:
        tasks = load_tasks(args.missions)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    folder = args.out or ROOT / 'data' / 'pi' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder.mkdir(parents=True, exist_ok=True)
    mission = PiMission(tasks, scene_timeout_s=args.scene_timeout, encounter_gap_s=args.encounter_gap)
    status_path = folder / 'status.json'
    if args.replay:
        replay.started = time.monotonic()
        status = replay(mission, args.replay, status_path, args.speed)
        print(json.dumps(dict(stats=status['stats'], decisions=summary(status['decisions'])), indent=2))
        return
    if os.environ.get('ROS_DOMAIN_ID') != '8':
        parser.error('Run with ROS_DOMAIN_ID=8 (use pi_bridge.sh)')
    run_live(mission, args, status_path, folder / 'messages.jsonl')


if __name__ == '__main__':
    sys.exit(main())
