#!/usr/bin/env python3
"""Label stairs seen by the Raspberry Pi in RViz.

Input  /group8pi/perception/detections (std_msgs/String JSON, BEST_EFFORT, ROS domain 8). Two Pi sources:
  stair_candidates[]  obstacle YOLO (arena-obstacles model, every 3rd frame, conf >= 0.70): image box only.
                      Distance comes from where the box's bottom edge meets the floor, using the Pi's
                      camera mount (0.15 m high, 15 deg down) and the Orbbec colour intrinsics below.
  stair_curtain       RGB-D advisory, only when obstacle == "stairs" and reason == "confirmed": measured
                      distance_m (base_link forward) and heading_correction_deg.
The Claude scene model only classifies rooms; it does not report stairs.
Output /group8pi/stair_markers (visualization_msgs/MarkerArray, RELIABLE + TRANSIENT_LOCAL, frame map):
  an orange block and a "STAIRS" label per place, sightings within --merge-m merged. RGB-D confirmed places are
  solid red-orange and say "confirmed"; YOLO-only places are translucent and say "YOLO <confidence>".
The pose is map -> base_link at the frame's capture stamp; if TF has no pose that old, the newest pose is used
and the label says "(pose approx)". Display only: never publishes motion or holds.
"""
import argparse
import json
import math
import sys
import time

DETECTIONS_TOPIC = '/group8pi/perception/detections'
MARKERS_TOPIC = '/group8pi/stair_markers'
# Orbbec Gemini 335 colour, 640 x 480 (Pi /camera/color/camera_info, 2026-11-26)
FX, FY, CX, CY = 461.02, 460.94, 322.12, 237.93
CAMERA_HEIGHT_M, CAMERA_PITCH_DEG = 0.15, 15.0      # the Pi service's --height-m / --pitch-deg
MIN_M, MAX_M, FALLBACK_M = 0.3, 4.0, 1.5


def _num(v):
    return type(v) in (int, float) and math.isfinite(v)


def candidate_bearing_distance(box, *, fx=FX, fy=FY, cx=CX, cy=CY, height=CAMERA_HEIGHT_M, pitch_deg=CAMERA_PITCH_DEG):
    """(bearing_rad left-positive, distance_m along the floor, estimated: bool) from a stairs box {x1,y1,x2,y2}."""
    u = (box['x1'] + box['x2']) / 2.0
    bearing = -math.atan2(u - cx, fx)
    depression = math.radians(pitch_deg) + math.atan2(box['y2'] - cy, fy)   # ray through the box's bottom edge
    if depression <= math.radians(2.0):
        return bearing, FALLBACK_M, False          # bottom edge at/above the horizon: no floor intersection
    distance = height / math.tan(depression) / math.cos(bearing)
    if not MIN_M <= distance <= MAX_M:
        return bearing, min(max(distance, MIN_M), MAX_M), False
    return bearing, distance, True


def stair_sightings(text):
    """[{stamp_ns, bearing, distance_m, source, confidence, ranged}] from one Pi detections message."""
    try:
        msg = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(msg, dict):
        return []
    out = []
    stamp = msg.get('capture_stamp_ns')
    for item in msg.get('stair_candidates') or []:
        box = item.get('box') if isinstance(item, dict) else None
        if not (isinstance(box, dict) and all(_num(box.get(k)) for k in ('x1', 'y1', 'x2', 'y2'))
                and type(stamp) is int and _num(item.get('confidence'))):
            continue
        bearing, distance, ranged = candidate_bearing_distance(box)
        out.append(dict(stamp_ns=stamp, bearing=bearing, distance_m=distance, source='yolo',
                        confidence=float(item['confidence']), ranged=ranged))
    sc = msg.get('stair_curtain')
    if isinstance(sc, dict) and sc.get('obstacle') == 'stairs' and sc.get('reason') == 'confirmed' \
            and _num(sc.get('distance_m')) and 0 < sc['distance_m'] < 5:
        sc_stamp = sc.get('stamp_ns') if type(sc.get('stamp_ns')) is int and sc['stamp_ns'] > 0 else stamp
        if type(sc_stamp) is int:
            heading = sc.get('heading_correction_deg')
            out.append(dict(stamp_ns=sc_stamp, bearing=math.radians(heading) if _num(heading) else 0.0,
                            distance_m=float(sc['distance_m']), source='rgbd', confidence=1.0, ranged=True))
    return out


def to_map(pose, sighting):
    """Map (x, y) of a sighting from the robot pose (x, y, yaw) at its capture time."""
    x, y, yaw = pose
    a = yaw + sighting['bearing']
    return x + sighting['distance_m'] * math.cos(a), y + sighting['distance_m'] * math.sin(a)


class StairMap:
    """Stair places in the map; repeat sightings within merge_m are one place."""

    def __init__(self, merge_m=0.75):
        self.merge_m = merge_m
        self.places = []

    def add(self, xy, sighting, pose_exact, now):
        best = min(self.places, key=lambda p: math.dist(p['xy'], xy), default=None)
        if best is not None and math.dist(best['xy'], xy) <= self.merge_m:
            place = best
            weight = 3.0 if sighting['source'] == 'rgbd' else 1.0
            if sighting['source'] == 'rgbd' and not place['confirmed']:
                place['xy'], place['weight'] = xy, weight           # measured beats estimated
            else:
                total = place['weight'] + weight
                place['xy'] = tuple((p * place['weight'] + q * weight) / total for p, q in zip(place['xy'], xy))
                place['weight'] = total
        else:
            place = dict(id=len(self.places), xy=xy, weight=0.0, sightings=0, confirmed=False,
                         best_confidence=0.0, ranged=False, pose_exact=False)
            self.places.append(place)
            place['weight'] = 3.0 if sighting['source'] == 'rgbd' else 1.0
        place['sightings'] += 1
        place['confirmed'] |= sighting['source'] == 'rgbd'
        place['ranged'] |= sighting['ranged']
        place['pose_exact'] |= pose_exact
        place['best_confidence'] = max(place['best_confidence'], sighting['confidence'])
        place['last_seen'] = now
        return place

    @staticmethod
    def label(place):
        text = 'STAIRS ' + ('confirmed' if place['confirmed'] else f"YOLO {place['best_confidence']:.2f}")
        if place['sightings'] > 1:
            text += f" x{place['sightings']}"
        if not place['confirmed'] and not place['ranged']:
            text += ' (range unknown)'
        if not place['pose_exact']:
            text += ' (pose approx)'
        return text


def build_markers(places, frame_id='map'):
    """Plain dicts (one per marker) so the layout is testable without ROS."""
    markers = []
    for place in places:
        x, y = place['xy']
        alpha = 0.9 if place['confirmed'] else 0.45
        markers.append(dict(ns='stairs_block', id=place['id'], type='CUBE', frame_id=frame_id,
                            position=(x, y, 0.15), scale=(0.6, 0.6, 0.3), color=(1.0, 0.35, 0.0, alpha)))
        markers.append(dict(ns='stairs_label', id=place['id'], type='TEXT_VIEW_FACING', frame_id=frame_id,
                            position=(x, y, 0.75), scale=(0.0, 0.0, 0.28), color=(1.0, 0.55, 0.0, 1.0),
                            text=StairMap.label(place)))
    return markers


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--topic', default=DETECTIONS_TOPIC)
    p.add_argument('--merge-m', type=float, default=0.75)
    p.add_argument('--map-frame', default='map')
    p.add_argument('--base-frame', default='base_link')
    args = p.parse_args(argv)

    import rclpy
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray

    rclpy.init()
    node = rclpy.create_node('group08_stair_markers')
    tf = Buffer(cache_time=Duration(seconds=60.0))
    TransformListener(tf, node)
    stairs = StairMap(args.merge_m)
    pub = node.create_publisher(MarkerArray, MARKERS_TOPIC, QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def planar(t):
        q = t.transform.rotation
        return (t.transform.translation.x, t.transform.translation.y,
                math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def pose_at(stamp_ns):
        try:
            return planar(tf.lookup_transform(args.map_frame, args.base_frame, Time(nanoseconds=stamp_ns))), True
        except TransformException:
            try:
                return planar(tf.lookup_transform(args.map_frame, args.base_frame, Time())), False
            except TransformException:
                return None, False

    def publish():
        out = MarkerArray()
        stamp = node.get_clock().now().to_msg()
        for m in build_markers(stairs.places, args.map_frame):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = m['frame_id'], stamp
            marker.ns, marker.id, marker.action = m['ns'], m['id'], Marker.ADD
            marker.type = getattr(Marker, m['type'])
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = m['position']
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = m['scale']
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = m['color']
            marker.text = m.get('text', '')
            out.markers.append(marker)
        pub.publish(out)

    def on_detections(msg):
        for sighting in stair_sightings(msg.data):
            pose, exact = pose_at(sighting['stamp_ns'])
            if pose is None:
                print(time.strftime('%H:%M:%S'), 'stairs seen but no map pose yet; not placed', flush=True)
                continue
            place = stairs.add(to_map(pose, sighting), sighting, exact, time.monotonic())
            print(time.strftime('%H:%M:%S'), f"{sighting['source']} stairs {sighting['distance_m']:.2f} m "
                  f"{math.degrees(sighting['bearing']):+.0f} deg -> place {place['id']} "
                  f"({place['xy'][0]:.2f}, {place['xy'][1]:.2f}): {StairMap.label(place)}", flush=True)
            publish()

    node.create_subscription(String, args.topic, on_detections,
                             QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
    node.create_timer(2.0, publish)
    print(f'Labelling stairs from {args.topic} on {MARKERS_TOPIC} (RViz MarkerArray, Transient Local)', flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
