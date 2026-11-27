#!/usr/bin/env python3
"""Camera capture round trip for roam (operator 2026-11-25, item 7).

  dog stops in a pocket -> roam publishes /group8/navigation/capture_request {id, cell, pose, reason}
  -> this service takes the next camera frame -> asks Claude what the scene shows -> stores the picture and the
  answer for upload later -> publishes /group8/navigation/capture_ack {id, status, summary, stored} -> dog moves off.

Camera: the robot's front camera as it reaches the NUC (/camera/color/image_raw/compressed, ~1.8 Hz); --topic for
another one. That stream exists only on the NUC, so with the Claude key on the Pi run two halves:
  NUC:  scene_capture.py --relay      takes the frame for each request, publishes it as a capture job
  Pi:   scene_capture.py --from-jobs  asks Claude, stores, acks (key from ~/.config/group8pi/claude.env) Never moves the robot. Claude: claude-opus-5-5 (low effort, structured JSON), ANTHROPIC_API_KEY (or an
`ant auth login` profile). Without credentials, or when Claude is slow, the picture is stored as 'pending' and the
dog is acked anyway; `--analyse-pending` analyses the stored pending pictures later.

Store: <out>/captures/<id>.jpg and <out>/captures.jsonl (one record per capture: the upload queue, status
'analysed' / 'pending' / 'failed'). Uploading to the competition server is a separate step (the mission reporter).

  python3 scene_capture.py                     # run (default store ~/live_mission/captures)
  python3 scene_capture.py --analyse-pending   # analyse stored pending pictures, no ROS
"""
import argparse
import base64
import json
import os
import threading
import time
from pathlib import Path

REQUEST_TOPIC = '/group8/navigation/capture_request'
JOB_TOPIC = '/group8/navigation/capture_job'     # NUC relay -> Pi: the request with the picture attached
ACK_TOPIC = '/group8/navigation/capture_ack'
IMAGE_TOPIC = '/camera/color/image_raw/compressed'
MODEL = 'claude-opus-5-5'
FRESH_FRAME_S = 2.0          # wait at most this long for a frame taken after the dog stopped
CLAUDE_TIMEOUT_S = 8.0       # ack the dog after this even without Claude's answer (stored as pending)

SCHEMA = {
    'type': 'object',
    'properties': {
        'scene_type': {'type': 'string', 'description': 'kind of room or area, e.g. kitchen, market, corridor'},
        'summary': {'type': 'string', 'description': 'one short sentence: what is in view'},
        'people': {'type': 'integer', 'description': 'people or survivors visible'},
        'hazards': {'type': 'array', 'items': {'type': 'string'}},
        'objects': {'type': 'array', 'items': {'type': 'string'}},
        'qr_visible': {'type': 'boolean'},
    },
    'required': ['scene_type', 'summary', 'people', 'hazards', 'objects', 'qr_visible'],
    'additionalProperties': False,
}
PROMPT = ('This is a frame from the front camera of a quadruped robot exploring a search-and-rescue practice course '
          'made of 1 x 1 m cells with low walls and blocks. The robot has just walked into a pocket or small room. '
          'Describe what is in view for the mission log: the kind of area, people or survivors, hazards (signs, '
          'spills, wires, fire), notable objects, and whether a QR code is visible. Report only what you can see.')


def analyse(jpeg_bytes, model=MODEL, timeout_s=CLAUDE_TIMEOUT_S):
    """Claude's structured description of one JPEG. Returns (status, dict or error text)."""
    import anthropic
    client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)
    data = base64.standard_b64encode(jpeg_bytes).decode('ascii')
    try:
        response = client.beta.messages.create(
            model=model, max_tokens=4000,
            betas=['server-side-fallback-2026-07-01'], fallbacks='default',   # refusal -> fallback model
            output_config={'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}},
            messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': data}},
                {'type': 'text', 'text': PROMPT}]}])
    except TypeError as exc:                                 # no API key / profile on this computer
        if 'authentication' in str(exc).lower():
            return 'pending', 'no Claude credentials'
        raise
    except anthropic.AuthenticationError:
        return 'pending', 'no valid Claude credentials'
    except anthropic.RateLimitError:
        return 'pending', 'rate limited'
    except anthropic.APITimeoutError:
        return 'pending', 'timed out'
    except anthropic.APIConnectionError as exc:
        return 'pending', f'no connection: {exc}'[:120]
    except anthropic.APIStatusError as exc:
        return 'failed', f'API error {exc.status_code}: {exc.message}'[:160]
    if response.stop_reason == 'refusal':
        return 'failed', 'refused'
    text = next((b.text for b in response.content if b.type == 'text'), '')
    try:
        return 'analysed', json.loads(text)
    except ValueError:
        return 'failed', f'unparsable answer: {text[:80]}'


class Store:
    def __init__(self, out):
        self.dir = Path(out).expanduser()
        (self.dir / 'captures').mkdir(parents=True, exist_ok=True)
        self.log = self.dir / 'captures.jsonl'
        self.lock = threading.Lock()

    def save(self, rid, jpeg, request, status, result, stamp):
        img = self.dir / 'captures' / f'{rid}.jpg'
        img.write_bytes(jpeg)
        rec = {'id': rid, 'time': time.time(), 'frame_stamp': stamp, 'request': request, 'image': str(img),
               'status': status, 'result': result if status == 'analysed' else None,
               'error': result if status != 'analysed' else None, 'uploaded': False}
        with self.lock, self.log.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        return rec

    def records(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def rewrite(self, records):
        with self.lock:
            tmp = self.log.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(r) + '\n' for r in records))
            os.replace(tmp, self.log)


def analyse_pending(store, model):
    recs = store.records()
    done = 0
    for r in recs:
        if r['status'] != 'pending':
            continue
        status, result = analyse(Path(r['image']).read_bytes(), model, timeout_s=60.0)
        if status == 'analysed':
            r.update(status='analysed', result=result, error=None)
            done += 1
        else:
            r['error'] = result
        print(f"{r['id']}: {r['status']} {r['error'] or ''}", flush=True)
    store.rewrite(recs)
    print(f'analysed {done} pending capture(s)', flush=True)


def run(args):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    store = Store(args.out)
    rclpy.init()
    node = rclpy.create_node('group8_scene_capture')
    reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    ack_pub = node.create_publisher(String, ACK_TOPIC, reliable)
    frame = {'t': 0.0, 'jpeg': None, 'stamp': None}

    def on_image(msg):
        frame.update(t=time.monotonic(), jpeg=bytes(msg.data),
                     stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    if not args.from_jobs:
        node.create_subscription(CompressedImage, args.topic, on_image, qos_profile_sensor_data)
    job_pub = node.create_publisher(String, JOB_TOPIC, reliable) if args.relay else None

    def relay(request):
        if request.get('reason') == 'stairs_vlm_confirmation':
            return  # the Pi captures its own camera frame for stairs
        rid, asked = request.get('id'), time.monotonic()
        while time.monotonic() - asked < FRESH_FRAME_S and frame['t'] < asked:
            time.sleep(0.05)
        if frame['jpeg'] is None or (request.get('reason') == 'stairs_vlm_confirmation' and frame['t'] < asked):
            ack_pub.publish(String(data=json.dumps({'id': rid, 'status': 'no_image', 'summary': 'no camera frame',
                                                    'stored': False})))
            return
        job = dict(request, jpeg_b64=base64.standard_b64encode(frame['jpeg']).decode('ascii'),
                   frame_stamp=frame['stamp'])
        job_pub.publish(String(data=json.dumps(job)))
        print(f"{time.strftime('%H:%M:%S')} {rid} relayed {len(frame['jpeg'])} B to the Pi", flush=True)

    def handle(request):
        rid, asked = request.get('id'), time.monotonic()
        if request.get('jpeg_b64'):                         # a job from the NUC relay: the picture is attached
            frame.update(t=asked, jpeg=base64.standard_b64decode(request.pop('jpeg_b64')),
                         stamp=request.pop('frame_stamp', None))
        while time.monotonic() - asked < FRESH_FRAME_S and frame['t'] < asked:
            time.sleep(0.05)                                # a frame taken after the dog stopped
        if frame['jpeg'] is None:
            ack = {'id': rid, 'status': 'no_image', 'summary': 'no camera frame', 'stored': False}
        else:
            jpeg, stamp = frame['jpeg'], frame['stamp']
            try:
                status, result = analyse(jpeg, args.model, args.claude_timeout) if not args.no_claude \
                    else ('pending', 'Claude disabled')
            except Exception as exc:  # noqa: BLE001 - never leave the dog waiting: store and ack anyway
                status, result = 'pending', f'{type(exc).__name__}: {exc}'[:160]
            rec = store.save(rid, jpeg, request, status, result, stamp)
            summary = result.get('summary') if status == 'analysed' else f'stored ({result})'
            if status == 'analysed' and result.get('scene_type'):
                summary = f"{result['scene_type']}: {summary}"
            ack = {'id': rid, 'status': status, 'summary': (summary or '')[:200], 'stored': True,
                   'image': rec['image']}
        ack_pub.publish(String(data=json.dumps(ack)))      # the dog moves off on this
        print(f"{time.strftime('%H:%M:%S')} {rid} cell={request.get('cell')} -> {ack['status']}: "
              f"{ack['summary']} ({time.monotonic() - asked:.1f} s)", flush=True)

    def on_request(msg):
        try:
            request = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(request, dict) and request.get('id'):
            threading.Thread(target=relay if args.relay else handle, args=(request,), daemon=True).start()
    topic = JOB_TOPIC if args.from_jobs else REQUEST_TOPIC
    node.create_subscription(String, topic, on_request, reliable)
    role = 'relay (frames -> Pi)' if args.relay else f'{args.model}; store {store.dir}'
    print(f'scene_capture: {args.topic if not args.from_jobs else "jobs"} -> {role}; listening on {topic}',
          flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default='~/live_mission/captures_store', help='store directory')
    ap.add_argument('--topic', default=IMAGE_TOPIC, help='sensor_msgs/CompressedImage topic')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--claude-timeout', type=float, default=CLAUDE_TIMEOUT_S)
    ap.add_argument('--no-claude', action='store_true', help='store pictures only (analyse later)')
    ap.add_argument('--relay', action='store_true', help='NUC half: attach the camera frame, publish a capture job')
    ap.add_argument('--from-jobs', action='store_true', help='Pi half: take capture jobs (picture attached)')
    ap.add_argument('--analyse-pending', action='store_true', help='analyse stored pending pictures and exit')
    args = ap.parse_args()
    if args.analyse_pending:
        analyse_pending(Store(args.out), args.model)
        return 0
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
