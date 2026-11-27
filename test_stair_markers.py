"""Unit tests for stair_markers.py (no ROS)."""
import json
import math

from stair_markers import CX, CY, StairMap, build_markers, candidate_bearing_distance, stair_sightings, to_map


def box(x1, y1, x2, y2):
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)


def detections(stamp, stairs=(), sc=None):
    msg = dict(frame=1, capture_stamp_ns=stamp, frame_id='camera_color_optical_frame', objects=[],
               stair_curtain=sc or dict(obstacle='unknown', reason='no_confirmed_candidate', stamp_ns=stamp))
    if stairs is not None:
        msg['stair_candidates'] = [dict(name='stairs', **{'class': 0}, confidence=c, box=b) for c, b in stairs]
    return json.dumps(msg)


def test_centred_box_is_straight_ahead_and_lower_edge_is_nearer():
    bearing, far, ranged = candidate_bearing_distance(box(CX - 50, 150, CX + 50, 300))
    assert abs(bearing) < 1e-6 and ranged
    _, near, _ = candidate_bearing_distance(box(CX - 50, 300, CX + 50, 470))
    assert near < far
    # 15 deg down, 0.15 m high: the image centre row hits the floor at 0.15 / tan(15 deg) = 0.56 m
    _, centre, _ = candidate_bearing_distance(box(CX - 50, 100, CX + 50, CY))
    assert abs(centre - 0.15 / math.tan(math.radians(15))) < 0.01


def test_box_on_the_left_has_a_positive_bearing():
    bearing, _, _ = candidate_bearing_distance(box(0, 200, 100, 300))
    assert math.degrees(bearing) > 20


def test_box_bottom_above_horizon_has_unknown_range():
    _, distance, ranged = candidate_bearing_distance(box(200, 0, 400, 40))
    assert not ranged and distance == 1.5


def test_sightings_from_yolo_and_confirmed_rgbd_only():
    assert stair_sightings(detections(5, stairs=None)) == []
    yolo = stair_sightings(detections(5, stairs=[(0.82, box(300, 200, 340, 300))]))
    assert len(yolo) == 1 and yolo[0]['source'] == 'yolo' and yolo[0]['stamp_ns'] == 5
    for reason in ('timestamp_mismatch', 'stale_camera', 'awaiting_consensus'):
        assert stair_sightings(detections(5, sc=dict(obstacle='stairs', reason=reason, distance_m=1.0))) == []
    rgbd = stair_sightings(detections(5, sc=dict(obstacle='stairs', reason='confirmed', distance_m=1.2,
                                                 heading_correction_deg=-10.0, stamp_ns=4)))
    assert rgbd[0]['source'] == 'rgbd' and rgbd[0]['stamp_ns'] == 4 and rgbd[0]['distance_m'] == 1.2
    assert abs(math.degrees(rgbd[0]['bearing']) + 10) < 1e-9


def test_to_map_uses_robot_heading():
    x, y = to_map((1.0, 2.0, math.pi / 2), dict(bearing=0.0, distance_m=1.0))
    assert abs(x - 1.0) < 1e-9 and abs(y - 3.0) < 1e-9


def test_repeat_sightings_merge_and_rgbd_confirms():
    places = StairMap(merge_m=0.75)
    yolo = dict(source='yolo', confidence=0.8, ranged=True)
    places.add((2.0, 0.0), yolo, True, 0)
    places.add((2.3, 0.1), dict(yolo, confidence=0.9), True, 1)
    places.add((5.0, 0.0), yolo, False, 2)
    assert len(places.places) == 2 and places.places[0]['sightings'] == 2
    assert StairMap.label(places.places[0]) == 'STAIRS YOLO 0.90 x2'
    assert StairMap.label(places.places[1]) == 'STAIRS YOLO 0.80 (pose approx)'
    places.add((2.4, 0.0), dict(source='rgbd', confidence=1.0, ranged=True), True, 3)
    assert places.places[0]['confirmed'] and places.places[0]['xy'] == (2.4, 0.0)
    assert StairMap.label(places.places[0]).startswith('STAIRS confirmed')


def test_markers_block_and_label_per_place():
    places = StairMap()
    places.add((1.0, 2.0), dict(source='yolo', confidence=0.75, ranged=True), True, 0)
    markers = build_markers(places.places)
    assert [m['type'] for m in markers] == ['CUBE', 'TEXT_VIEW_FACING']
    assert markers[1]['text'] == 'STAIRS YOLO 0.75' and markers[1]['position'][:2] == (1.0, 2.0)
