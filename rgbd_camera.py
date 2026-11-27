"""Read the ROS RGB-D camera and estimate unverified map positions."""

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from math import atan2, isfinite
from queue import Empty, Full, Queue
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

MAX_PAIR_GAP_NS = 60_000_000
SCOUT_IDLE_SECONDS = 3.0


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def image_array(msg, dtype, channels=1):
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    return np.ndarray(
        (msg.height, msg.width, channels), dtype=dtype, buffer=msg.data,
        strides=(msg.step, dtype.itemsize * channels, dtype.itemsize),
    )


def camera_parameters(info, image):
    if (info is None or not image.header.frame_id
            or info.header.frame_id != image.header.frame_id
            or (info.width, info.height) != (image.width, image.height)
            or info.distortion_model not in ("", "plumb_bob", "rational_polynomial")):
        return None
    matrix = np.array(info.k).reshape(3, 3)
    distortion = np.array(info.d) if info.d else None
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        return None
    if distortion is not None and (len(distortion) not in (4, 5, 8, 12, 14)
                                   or not np.isfinite(distortion).all()):
        return None
    return matrix, distortion


def transform_points(points, transform):
    q = transform.rotation
    vector = np.array([q.x, q.y, q.z])
    cross = 2 * np.cross(vector, points)
    t = transform.translation
    return points + q.w * cross + np.cross(vector, cross) + [t.x, t.y, t.z]


class RGBDCamera:
    def __init__(self, publish_mission_events=False, receive_validations=False, on_demand=False):
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.depth_ready = threading.Condition(self.lock)
        self.error = None
        self.sample = None
        self.depth_frames = deque(maxlen=12)
        self.color_info = None
        self.depth_info = None
        self.rays = None
        self.marker_id = 0
        self.context = Context()
        rclpy.init(context=self.context, signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node(f"live_mission_rgbd_{os.getpid()}", context=self.context)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self.node, spin_thread=False)
        self.on_demand = on_demand
        self.image_subscriptions = []
        self.stream_requests = Queue()
        self.stream_idle_at = None
        if not on_demand:
            self.open_image_streams()
        self.markers = self.node.create_publisher(
            Marker, "/live_mission/candidate_markers",
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.event_publisher = None
        if publish_mission_events:
            self.event_publisher = self.node.create_publisher(String, "/mission/events", 10)
        self.validations = Queue(maxsize=64)
        if receive_validations:
            self.node.create_subscription(String, "/live_mission/scoring_validation",
                                          self.on_validation, 10)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.spin, daemon=True)
        self.thread.start()

    def open_image_streams(self):
        if self.image_subscriptions:
            return
        try:
            for message, topic, callback in (
                (Image, "/camera/color/image_raw", self.on_color),
                (Image, "/camera/depth/image_raw", self.on_depth),
                (CameraInfo, "/camera/color/camera_info", self.on_color_info),
                (CameraInfo, "/camera/depth/camera_info", self.on_depth_info),
            ):
                subscription = self.node.create_subscription(message, topic, callback, qos_profile_sensor_data)
                self.image_subscriptions.append(subscription)
        except Exception:
            self.close_image_streams()
            raise

    def close_image_streams(self):
        for subscription in self.image_subscriptions:
            self.node.destroy_subscription(subscription)
        self.image_subscriptions.clear()
        with self.depth_ready:
            self.sample = self.color_info = self.depth_info = None
            self.depth_frames.clear()
            self.depth_ready.notify_all()

    def service_stream_requests(self, now):
        # The executor thread owns image subscription creation and destruction.
        try:
            while True:
                reply = self.stream_requests.get_nowait()
                if reply is None:
                    self.stream_idle_at = now + SCOUT_IDLE_SECONDS
                    continue
                self.stream_idle_at = None
                try:
                    was_active = bool(self.image_subscriptions)
                    self.open_image_streams()
                    if not was_active:
                        print("Orbbec scout subscriptions active.", flush=True)
                    self.error = None
                    reply.put_nowait(True)
                except Exception as exc:
                    self.error = f"Orbbec subscription failed: {type(exc).__name__}"
                    reply.put_nowait(False)
        except Empty:
            pass
        if self.stream_idle_at is not None and now >= self.stream_idle_at:
            if self.image_subscriptions:
                self.close_image_streams()
                print("Orbbec scout subscriptions released.", flush=True)
            self.stream_idle_at = None

    @contextmanager
    def inspection(self):
        if not self.on_demand:
            yield True
            return
        reply = Queue(maxsize=1)
        self.stream_requests.put(reply)
        self.executor.wake()
        try:
            try:
                active = reply.get(timeout=0.75)
            except Empty:
                active = False
            yield active
        finally:
            self.stream_requests.put(None)
            self.executor.wake()

    def on_validation(self, message):
        try:
            self.validations.put_nowait(json.loads(message.data))
        except (ValueError, Full):
            self.node.get_logger().warning("Discarded invalid or excess scoring validation")

    def next_validation(self):
        try:
            return self.validations.get_nowait()
        except Empty:
            return None

    def start_pose(self):
        try:
            stamped = self.tf.lookup_transform("map", "base_link", Time())
        except TransformException:
            return None
        sample_time = stamp_ns(stamped.header)
        if sample_time <= 0 or abs(self.node.get_clock().now().nanoseconds - sample_time) > 1_000_000_000:
            return None
        position, q = stamped.transform.translation, stamped.transform.rotation
        yaw = atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        pose = [position.x, position.y, yaw]
        return pose if all(map(isfinite, pose)) else None

    def spin(self):
        try:
            while not self.stop.is_set():
                if self.on_demand:
                    self.service_stream_requests(time.monotonic())
                self.executor.spin_once(timeout_sec=0.1)
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.try_shutdown()

    def on_color(self, msg):
        if msg.encoding not in ("bgr8", "rgb8"):
            raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")
        frame = image_array(msg, np.uint8, 3)
        frame = frame.copy() if msg.encoding == "bgr8" else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        stamp = stamp_ns(msg.header)
        if stamp <= 0:
            return
        sec, ns = divmod(stamp, 1_000_000_000)
        utc = datetime.fromtimestamp(sec, timezone.utc).replace(microsecond=ns // 1000)
        with self.lock:
            self.sample = time.monotonic(), utc.isoformat(), frame, msg

    def on_depth(self, msg):
        if stamp_ns(msg.header) > 0:
            with self.depth_ready:
                self.depth_frames.append(msg)
                self.depth_ready.notify()

    def on_color_info(self, msg):
        with self.lock:
            self.color_info = msg

    def on_depth_info(self, msg):
        with self.lock:
            self.depth_info = msg

    def latest(self):
        with self.lock:
            return self.sample

    def pair_depth(self, color):
        color_stamp = stamp_ns(color.header)
        with self.depth_ready:
            # Keep the matching frame before inference can outlast the depth queue.
            self.depth_ready.wait_for(
                lambda: self.depth_frames and stamp_ns(self.depth_frames[-1].header) >= color_stamp,
                timeout=0.1,
            )
            depth = min(
                self.depth_frames, key=lambda msg: abs(stamp_ns(msg.header) - color_stamp),
                default=None,
            )
            color_info, depth_info = self.color_info, self.depth_info
        if depth is None or abs(stamp_ns(depth.header) - color_stamp) > MAX_PAIR_GAP_NS:
            return None
        return depth, color_info, depth_info

    def has_map_transform(self, color):
        return self.tf.can_transform("map", color.header.frame_id, Time())

    def locate(self, objects, color, depth_pair, qr_codes=()):
        if depth_pair is None or not (objects or qr_codes):
            return
        depth, color_info, depth_info = depth_pair
        color_params = camera_parameters(color_info, color)
        depth_params = camera_parameters(depth_info, depth)
        if color_params is None or depth_params is None:
            return "camera calibration unavailable"
        try:
            to_map = self.tf.lookup_transform(
                "map", color.header.frame_id, Time.from_msg(color.header.stamp),
            ).transform
            # Use each sensor timestamp so motion between the two images is accounted for.
            to_color = self.tf.lookup_transform_full(
                color.header.frame_id, Time.from_msg(color.header.stamp),
                depth.header.frame_id, Time.from_msg(depth.header.stamp), "map",
            ).transform
        except TransformException:
            return "map or camera transform unavailable at the image timestamps"

        if depth.encoding == "16UC1":
            depths = image_array(depth, np.uint16)[::2, ::2, 0].astype(np.float32) * 0.001
        elif depth.encoding == "32FC1":
            depths = image_array(depth, np.float32)[::2, ::2, 0].astype(np.float32)
        else:
            return

        matrix, distortion = depth_params
        key = depths.shape, tuple(depth_info.k), tuple(depth_info.d)
        if self.rays is None or self.rays[0] != key:
            rows, cols = np.indices(depths.shape, dtype=np.float32)
            pixels = np.stack((2 * cols, 2 * rows), axis=-1).reshape(-1, 1, 2)
            xy = cv2.undistortPoints(pixels, matrix, distortion).reshape(-1, 2)
            self.rays = key, np.column_stack((xy, np.ones(len(xy))))
        depths = depths.ravel()
        valid = np.isfinite(depths) & (depths >= 0.15) & (depths <= 8.0)
        if not valid.any():
            return
        points = transform_points(self.rays[1][valid] * depths[valid, None], to_color)
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1)]
        if not len(points):
            return
        matrix, distortion = color_params
        projected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)
        pixels = projected.reshape(-1, 2)

        for obj in objects:
            location = self.target_location(obj["xyxy"], points, pixels, to_map)
            if location is not None:
                obj["location"] = location
                obj["location_source"] = "rgbd_depth"

        for qr in qr_codes:
            location = self.qr_location(qr["corners"], points, pixels, to_map)
            if location is not None:
                qr["location"] = location
                qr["location_source"] = "rgbd_depth"

    @classmethod
    def qr_location(cls, corners, points, pixels, to_map):
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            return None
        hull = cv2.convexHull(corners).reshape(-1, 2)
        if len(hull) != 4 or cv2.contourArea(hull) <= 0:
            return None

        # Stay inside the QR edges to avoid including depth from its surroundings.
        center = hull.mean(axis=0)
        inner = center + 0.65 * (hull - center)
        inside = np.ones(len(pixels), dtype=bool)
        for start, end in zip(inner, np.roll(inner, -1, axis=0)):
            dx, dy = end - start
            inside &= dx * (pixels[:, 1] - start[1]) >= dy * (pixels[:, 0] - start[0])
        return cls.cluster_location(points[inside], to_map, minimum=8)

    @classmethod
    def target_location(cls, box, points, pixels, to_map):
        low, high = np.asarray(box).reshape(2, 2)
        center = (low + high) / 2
        half = (high - low) * 0.2
        inside = (np.abs(pixels - center) <= half).all(axis=1)
        return cls.cluster_location(points[inside], to_map, minimum=12)

    @staticmethod
    def cluster_location(samples, to_map, minimum):
        if len(samples) < minimum:
            return None
        depth = np.median(samples[:, 2])
        cluster = samples[np.abs(samples[:, 2] - depth) <= max(0.06, depth * 0.04)]
        if len(cluster) < max(minimum, len(samples) * 0.65):
            return None
        position = transform_points(np.median(cluster, axis=0), to_map)
        if np.isfinite(position).all():
            return dict(frame_id="map", x=float(position[0]), y=float(position[1]), z=float(position[2]))
        return None

    def publish_event(self, event):
        message = String()
        message.data = json.dumps(event, allow_nan=False)
        self.event_publisher.publish(message)

    def publish_candidate(self, observation):
        location = observation.get("location")
        if location is None:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = self.node.get_name()
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = location["x"]
        marker.pose.position.y = location["y"]
        marker.pose.position.z = location["z"]
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.18
        marker.color.r, marker.color.g, marker.color.a = 1.0, 0.65, 0.9
        self.markers.publish(marker)
