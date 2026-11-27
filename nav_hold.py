#!/usr/bin/env python3
"""Hold navigation once per confirmed Pi encounter until the Pi has a still frame for Claude.

Input  /group8pi/perception/scene_events  (std_msgs/String JSON, RELIABLE) from the Pi:
         scene_pending | scene_captured | scene_result | scene_failed with request_id, encounter_id(s),
         capture_stamp_ns, frame_id
Output /group8/navigation/hold_request    (std_msgs/String JSON, RELIABLE + TRANSIENT_LOCAL):
         {"hold": bool, "request_id", "encounter_ids", "reason", "expires_wall_ns", "stamp_ns"}

Roam owns motion: it should stop at its next safe stopping point while hold is true and the
message has not expired, and OR this with its own manual and emergency stops. This node never
publishes velocity, never clears a stop it did not request, and releases its hold on the matching
scene_captured (operator 2026-11-26: the Pi sends Claude only a frame taken while the dog stands
still, so the dog may move on once that frame is taken), result or failure, or after --timeout
seconds (default 5). The Claude result arrives later and is used for reporting by
pi_mission_reporter with the still frame's capture stamp.
"""
import argparse
import json
import sys
import time

EVENTS_TOPIC = "/group8pi/perception/scene_events"
HOLD_TOPIC = "/group8/navigation/hold_request"


class HoldController:
    """Pure state machine; `now` is monotonic seconds, `wall_ns` is Unix ns for the expiry field."""

    def __init__(self, timeout_s=5.0, max_event_age_s=None, enabled=True):
        self.timeout_s = timeout_s
        self.max_event_age_s = timeout_s if max_event_age_s is None else max_event_age_s
        self.enabled = enabled
        self.held_encounters = set()   # each encounter holds navigation at most once
        self.active = None             # dict(request_id, encounter_ids, deadline, deadline_wall_ns)
        self.log = []                  # (now, text) for tests and status

    def on_event(self, event, now, wall_ns):
        """Returns a hold_request message dict when the output changes, else None."""
        kind, rid = event.get("event"), event.get("request_id")
        if kind == "scene_pending":
            return self._pending(event, now, wall_ns)
        if kind in ("scene_captured", "scene_result", "scene_failed") and self.active and rid == self.active["request_id"]:
            return self._release(f"{kind}" + (f": {event.get('error')}" if event.get("error") else ""),
                                 now, wall_ns)
        return None

    def tick(self, now, wall_ns):
        if self.active and now >= self.active["deadline"]:
            return self._release(f"timeout after {self.timeout_s:g}s", now, wall_ns)
        return None

    def _pending(self, event, now, wall_ns):
        encounters = set(event.get("encounter_ids") or [event.get("encounter_id")]) - {None}
        age_s = (wall_ns - int(event.get("capture_stamp_ns", wall_ns))) / 1e9
        if not self.enabled:
            reason = "holds disabled"
        elif not encounters or encounters <= self.held_encounters:
            reason = "encounter already held once"
        elif self.active is not None:
            reason = f"already holding for {self.active['request_id']}"
        elif age_s > self.max_event_age_s:
            reason = f"event is {age_s:.1f}s old"
        else:
            self.held_encounters |= encounters
            self.active = dict(request_id=event["request_id"], encounter_ids=sorted(encounters),
                               deadline=now + self.timeout_s,
                               deadline_wall_ns=wall_ns + int(self.timeout_s * 1e9))
            self.log.append((now, f"hold {event['request_id']} {sorted(encounters)}"))
            return self.message(True, f"scene_pending {event.get('trigger', {}).get('kind', '')}", wall_ns)
        self.held_encounters |= encounters
        self.log.append((now, f"no hold for {event.get('request_id')}: {reason}"))
        return None

    def _release(self, reason, now, wall_ns):
        message = self.message(False, reason, wall_ns)
        self.log.append((now, f"release {self.active['request_id']}: {reason}"))
        self.active = None
        return message

    def message(self, hold, reason, wall_ns):
        active = self.active or {}
        return dict(hold=hold, request_id=active.get("request_id"),
                    encounter_ids=active.get("encounter_ids", []), reason=reason,
                    expires_wall_ns=active.get("deadline_wall_ns") if hold else None, stamp_ns=wall_ns)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timeout", type=float, default=5.0, help="Seconds before navigation resumes anyway")
    p.add_argument("--disable", action="store_true", help="Log events but never request a hold")
    args = p.parse_args(argv)
    if args.timeout <= 0:
        p.error("--timeout must be positive")

    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("group08_nav_hold")
    controller = HoldController(args.timeout, enabled=not args.disable)
    out = node.create_publisher(String, HOLD_TOPIC, QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def publish(message):
        if message is not None:
            out.publish(String(data=json.dumps(message)))
            print(time.strftime("%H:%M:%S"), "HOLD" if message["hold"] else "RESUME",
                  message["request_id"], message["reason"], flush=True)

    def on_event(msg):
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        before = len(controller.log)
        publish(controller.on_event(event, time.monotonic(), time.time_ns()))
        for _, text in controller.log[before:]:
            if text.startswith("no hold"):
                print(time.strftime("%H:%M:%S"), text, flush=True)

    node.create_subscription(String, EVENTS_TOPIC, on_event,
                             QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
    node.create_timer(0.1, lambda: publish(controller.tick(time.monotonic(), time.time_ns())))
    publish(controller.message(False, "startup", time.time_ns()))
    print(f"Holding navigation on {EVENTS_TOPIC} scene_pending, timeout {args.timeout:g}s; "
          f"publishing {HOLD_TOPIC}", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if controller.active:     # never leave a hold behind
            publish(controller._release("shutdown", time.monotonic(), time.time_ns()))
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
