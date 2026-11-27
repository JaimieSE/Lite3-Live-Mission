"""Dry-run tests for the Pi perception -> NUC mission decisions. No ROS, motion or network."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi_mission import DETECTIONS_TOPIC, SCENES_TOPIC, PiMission, parse_detections, parse_scene  # noqa: E402

PROFILE = json.loads((ROOT / 'competition-v6.3-practice.json').read_text())
BASE_NS = 1_795_453_413_224_217_000
CAMERA_FRAME = 'camera_color_optical_frame'          # the Orbbec frame_id: the same on every frame
CLASS_IDS = {'person': 0, 'medical_equipment': 80}


def tasks(m03_relation=False):
    result = copy.deepcopy(PROFILE['tasks'])
    if not m03_relation:
        result[2].pop('relation')
    return result


def stamp(index):
    return BASE_NS + index * 200_000_000          # 5 Hz frames on the Pi clock


def yolo(name, confidence, box):
    """One entry of ultralytics Results.summary(decimals=3), as the Pi publishes it."""
    return dict(name=name, **{'class': CLASS_IDS.get(name, 99)}, confidence=confidence,
                box=dict(zip(('x1', 'y1', 'x2', 'y2'), map(float, box))))


def sign(box, **extra):
    return dict(box=box, triangle=[[box[0], box[3]], [box[2], box[3]], [box[0], box[1]]],
                yellow_fraction=0.9, area=2659.0, **extra)


def detections(index, people=(), objects=(), qr=(), signs=(), stair=None):
    message = dict(frame=index, capture_stamp_ns=stamp(index), frame_id=CAMERA_FRAME,
                   objects=[yolo(name, c, b) for name, c, b in objects] + [yolo('person', c, b) for c, b in people],
                   qr_codes=[dict(text=t, corners=[[0, 0], [9, 0], [9, 9], [0, 9]]) for t in qr],
                   warning_sign_candidates=list(signs),
                   stair_curtain=stair or dict(obstacle='unknown', reason='no_confirmed_candidate',
                                               stamp_ns=stamp(index)),
                   schema_version=1, source='group8pi')
    return json.dumps(message)


def scene(index, kind, evidence=('stove', 'sink'), frame_id=CAMERA_FRAME):
    return json.dumps(dict(frame=index, capture_stamp_ns=stamp(index), frame_id=frame_id,
                           scene=dict(type=kind, evidence=list(evidence)), model='gemma3:4b',
                           schema_version=1, source='group8pi'))


LEFT, RIGHT = [10, 10, 100, 200], [400, 10, 500, 200]


def test_parses_the_pi_message_format():
    frame = parse_detections(detections(3, people=[(0.41, [291, 1, 344, 85])], objects=[('chair', 0.44, LEFT)],
                                        qr=['HELLO', 'HELLO'], signs=[sign([540, 205, 640, 258])]))
    assert frame['key'] == (stamp(3), CAMERA_FRAME) and frame['frame'] == 3
    assert frame['people'] == [dict(label='person', confidence=0.41, bbox=[291.0, 1.0, 344.0, 85.0], track_id=None)]
    assert frame['objects'][0]['label'] == 'chair'                  # name, not the numeric class id
    assert frame['qr'] == ['HELLO'] and frame['signs'][0]['hazard'] is None
    assert frame['signs'][0]['bbox'] == [540.0, 205.0, 640.0, 258.0]
    assert parse_detections('{"objects": []}') is None            # no frame identifiers -> rejected
    assert parse_scene(json.dumps(dict(capture_stamp_ns=1, frame_id='a', scene=dict(type='kitchen'))))['type'] \
        == 'unknown'                                                # a label without evidence is not evidence


def test_unusable_stair_curtain_states_are_unknown():
    mission = PiMission(tasks())
    for i, state in enumerate(('stale_camera', 'timestamp_mismatch', 'waiting_for_rgbd', 'confirmed')):
        mission.add_detections(detections(i, stair=dict(obstacle='stairs', reason=state, distance_m=1.2)), i)
        latest = mission.latest_stair_curtain[0]
        assert latest['kind'] == ('stairs' if state == 'confirmed' else 'unknown')
        assert latest['distance_m'] == (1.2 if state == 'confirmed' else None)


def test_delayed_and_out_of_order_scenes_match_by_stamp_and_frame_id():
    mission = PiMission(tasks(), scene_timeout_s=20)
    for i in range(3):
        mission.add_detections(detections(i, objects=[('medical_equipment', 0.8, LEFT)]), float(i))
    assert mission.decide()['M01']['status'] == 'pending'
    mission.add_scene(scene(2, 'clinic', frame_id='other_camera'), 5.0)  # same stamp, other frame_id
    assert mission.decide()['M01']['status'] == 'pending'
    mission.add_scene(scene(1, 'clinic'), 9.0)                          # 8 s late, out of order
    decision = mission.decide()['M01']
    assert decision['status'] == 'ready' and decision['scene'] == 'clinic'
    assert decision['evidence'] == [[stamp(1), CAMERA_FRAME]]


def test_scene_that_arrives_before_its_detections_is_held():
    mission = PiMission(tasks())
    mission.add_scene(scene(4, 'kitchen'), 0.0)
    mission.add_detections(detections(4, objects=[('medical_equipment', 0.9, LEFT)]), 0.3)
    assert mission.decide()['M01']['scene'] == 'kitchen'
    assert mission.stats['scenes_orphaned'] == 1 and mission.stats['scenes_matched'] == 1


def test_missing_scene_keeps_mission_pending_and_late_result_still_counts():
    mission = PiMission(tasks(), scene_timeout_s=5)
    mission.add_detections(detections(0, objects=[('medical_equipment', 0.9, LEFT)]), 0.0)
    mission.expire(30.0)
    decision = mission.decide()['M01']
    assert decision['status'] == 'pending' and decision['candidates'][0]['reason'] == 'scene unknown'
    assert mission.stats['scenes_missing'] == 1
    mission.add_scene(scene(0, 'clinic'), 31.0)                         # after the timeout, still buffered
    assert mission.decide()['M01']['status'] == 'ready' and mission.stats['scenes_late'] == 1


def test_scene_for_an_expired_frame_is_dropped_not_misattributed():
    mission = PiMission(tasks(), scene_timeout_s=5, retention_s=10)
    mission.add_detections(detections(0, objects=[('medical_equipment', 0.9, LEFT)]), 0.0)
    mission.add_detections(detections(100), 50.0)                        # frame 0 ages out
    mission.add_scene(scene(0, 'clinic'), 51.0)                         # its scene has no frame to join
    mission.expire(60.0)
    assert mission.decide()['M01']['status'] == 'pending' and mission.stats['scenes_expired'] == 1


def test_unknown_scene_type_is_not_evidence():
    mission = PiMission(tasks())
    mission.add_detections(detections(0, objects=[('medical_equipment', 0.9, LEFT)]), 0.0)
    mission.add_scene(scene(0, 'unknown'), 1.0)
    assert mission.decide()['M01']['status'] == 'pending'


def test_warning_triangle_needs_hazard_identification():
    mission = PiMission(tasks())
    mission.add_detections(detections(0, signs=[sign([540, 205, 640, 258])]), 0.0)
    mission.add_scene(scene(0, 'workshop', ('workbench',)), 1.0)
    decision = mission.decide()['M02']
    assert decision['status'] == 'pending'
    assert 'needs hazard identification' in decision['candidates'][0]['reason']
    # The Pi does not identify hazards today; these use the field a future identifier would add.
    mission.add_detections(detections(1, signs=[sign([541, 205, 641, 258], hazard='radioactive')]), 1.2)
    assert mission.decide()['M02']['status'] == 'pending'                # wrong hazard for this mission
    mission.add_detections(detections(2, signs=[sign([542, 205, 642, 258], hazard='high_voltage')]), 1.4)
    decision = mission.decide()['M02']
    assert decision['status'] == 'ready' and decision['label'] == 'high_voltage'
    assert decision['scene'] == 'workshop'                              # scene from the same encounter


def test_qr_payload_reported_once_and_ambiguity_stays_pending():
    mission = PiMission(tasks())
    mission.add_detections(detections(0, qr=['ROOM-B LEFT']), 0.0)
    mission.add_detections(detections(1, qr=['ROOM-B LEFT']), 0.2)
    assert mission.decide()['M04'] == dict(mission.decide()['M04'], status='ready', message='ROOM-B LEFT')
    mission.add_detections(detections(2, qr=['OTHER']), 0.4)
    assert mission.decide()['M04']['status'] == 'pending'


def test_single_frame_person_is_not_a_confirmed_survivor():
    mission = PiMission(tasks())
    mission.add_detections(detections(0, people=[(0.9, LEFT)]), 0.0)
    mission.add_scene(scene(0, 'kitchen'), 1.0)
    decisions = mission.decide()
    assert decisions['M03']['status'] == decisions['M05']['status'] == 'pending'
    assert decisions['M03']['candidates'][0]['reason'] == 'needs survivor confirmation'


def see_survivor(mission, first, room, at, box=LEFT, frames=3, scene_delay=4.0):
    for i in range(first, first + frames):
        mission.add_detections(detections(i, people=[(0.85, box)]), at + (i - first) * 0.2)
    mission.add_scene(scene(first + 1, room, ('shelves',)), at + scene_delay)


def test_m05_survivor_seen_before_m03_is_kept_and_assigned():
    mission = PiMission(tasks())
    see_survivor(mission, 0, 'clinic', 0.0)                              # the other survivor, first
    decisions = mission.decide()
    assert decisions['M03']['status'] == 'pending'
    assert decisions['M05']['status'] == 'ready' and decisions['M05']['identity'] == 'survivor:clinic'
    see_survivor(mission, 50, 'clinic', 30.0, box=RIGHT)                 # repeat sighting, new encounter
    see_survivor(mission, 100, 'kitchen', 60.0)                          # the described survivor
    decisions = mission.decide()
    assert decisions['M03']['status'] == 'ready' and decisions['M03']['scene'] == 'kitchen'
    assert decisions['M03']['evidence'] == [[stamp(101), CAMERA_FRAME]]
    assert decisions['M05']['identity'] == 'survivor:clinic' and 'note' not in decisions['M05']
    identities, _ = mission.survivors()
    assert [i['identity'] for i in identities] == ['survivor:clinic', 'survivor:kitchen']
    assert len(identities[0]['sightings']) == 2                          # deduplicated repeat


def test_m03_uses_scene_from_its_own_encounter_only():
    mission = PiMission(tasks())
    mission.add_detections(detections(0), 0.0)
    mission.add_scene(scene(0, 'kitchen'), 0.5)                         # a kitchen, but nobody seen there
    see_survivor(mission, 30, 'clinic', 10.0)                           # survivor is in another room
    assert mission.decide()['M03']['status'] == 'pending'


def test_repeat_sightings_in_the_described_room_never_become_m05():
    mission = PiMission(tasks())
    see_survivor(mission, 0, 'kitchen', 0.0)
    see_survivor(mission, 40, 'kitchen', 20.0, box=RIGHT)
    decisions = mission.decide()
    assert decisions['M03']['status'] == 'ready' and decisions['M05']['status'] == 'pending'


def test_survivor_with_delayed_then_missing_scene_stays_pending():
    mission = PiMission(tasks(), scene_timeout_s=5)
    for i in range(3):
        mission.add_detections(detections(i, people=[(0.9, LEFT)]), i * 0.2)
    mission.expire(20.0)
    decisions = mission.decide()
    assert decisions['M05']['status'] == 'pending'
    assert 'scene unknown' in decisions['M05']['candidates'][0]['reason']


def test_m03_relation_is_not_inferred_from_pi_results():
    mission = PiMission(tasks(m03_relation=True))
    see_survivor(mission, 0, 'kitchen', 0.0)
    decision = mission.decide()['M03']
    assert decision['status'] == 'pending' and 'relation "right" not observable' in decision['reason']


def test_bridge_replay_is_dry_run_only(tmp_path):
    rows = []
    for i in range(3):
        rows.append(dict(topic=DETECTIONS_TOPIC, rx_mono=i * 0.2, data=detections(i, people=[(0.9, LEFT)])))
    rows.append(dict(topic=SCENES_TOPIC, rx_mono=6.0, data=scene(1, 'clinic', ('shelves',))))
    rows.append(dict(topic=DETECTIONS_TOPIC, rx_mono=7.0, data=detections(9, qr=['CLUE'])))
    record = tmp_path / 'messages.jsonl'
    record.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    bridge = [sys.executable, str(ROOT / 'pi_bridge.py')]
    result = subprocess.run(bridge + ['--dry-run', '--missions', str(ROOT / 'competition-v6.3-practice.json'),
                                      '--replay', str(record), '--out', str(tmp_path / 'out')],
                            capture_output=True, text=True, check=True)
    decisions = json.loads(result.stdout)['decisions']
    assert decisions['M04'] == ['ready', 'CLUE'] and decisions['M05'] == ['ready', 'survivor:clinic']
    assert json.loads((tmp_path / 'out' / 'status.json').read_text())['mode'] == 'replay'
    refused = subprocess.run(bridge + ['--missions', 'x.json'], capture_output=True, text=True)
    assert refused.returncode == 2 and '--dry-run' in refused.stderr
    refused = subprocess.run(bridge + ['--dry-run', '--missions', str(ROOT / 'competition-v6.3-practice.json'),
                                       '--missions-url', 'http://example.invalid/missions'],
                             capture_output=True, text=True)
    assert refused.returncode == 2 and 'contract has not been confirmed' in refused.stderr


def test_real_pi_frames_parse_and_match_a_delayed_scene():
    rows = [json.loads(line) for line in (ROOT / 'tests' / 'fixtures' / 'pi_real_frames.jsonl').open()]
    mission = PiMission(tasks())
    people = [r for r in rows if any(o['name'] == 'person' for o in json.loads(r['data'])['objects'])]
    target = json.loads(people[1]['data'])
    for row in rows:
        mission.add_detections(row['data'], row['rx_mono'])
        if row is people[3]:                                         # VLM result lands two frames later
            mission.add_scene(json.dumps(dict(frame=target['frame'], capture_stamp_ns=target['capture_stamp_ns'],
                                              frame_id=target['frame_id'], model='gemma3:4b',
                                              scene=dict(type='clinic', evidence=['examination bed']))),
                              row['rx_mono'])
    assert mission.stats['rejected'] == 0 and mission.stats['detections'] == 111
    assert len([e for e in mission.encounters if e.kind == 'survivor']) == 1
    decisions = mission.decide()
    assert decisions['M05']['status'] == 'ready' and decisions['M05']['identity'] == 'survivor:clinic'
    assert decisions['M05']['evidence'] == [[target['capture_stamp_ns'], target['frame_id']]]
    assert decisions['M02']['status'] == 'pending'                   # triangles carry no hazard identification
    assert mission.latest_stair_curtain[0]['kind'] == 'unknown'
