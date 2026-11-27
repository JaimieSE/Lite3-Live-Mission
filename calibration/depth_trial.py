"""Offline depth-filter trial using an unvalidated front-camera alignment."""
import argparse
import json
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points


def project(scan):
    cloud = deserialize_message(scan.read_bytes(), PointCloud2)
    if cloud.header.frame_id != "livox_frame":
        raise ValueError(f"Unexpected point cloud frame: {cloud.header.frame_id}")
    points = read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
    lidar_xyz = np.column_stack([points[axis] for axis in ("x", "y", "z")])
    optical_axes = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    rotation = optical_axes @ Rotation.from_euler("y", 15, degrees=True).as_matrix()
    xyz = lidar_xyz @ rotation.T + [0, 0.05, -0.07]
    xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.08)]
    pixels = xyz[:, :2] / xyz[:, 2, None] * 556 + [639.5, 359.5]
    return xyz, pixels


def classify(xyz, pixels, box):
    low, high = np.array(box).reshape(2, 2)
    if np.any(high <= low):
        raise ValueError("Box needs increasing x and y coordinates")
    mask = ((pixels >= low) & (pixels < high)).all(axis=1)
    points = xyz[mask]
    cells = ((pixels[mask] - low) / (high - low) * 3).astype(int)
    occupied = len(np.unique(cells, axis=0))
    result = {"decision": "skip_uncertain", "points": len(points), "covered_cells": occupied}
    if len(points) < 24 or occupied < 6:
        return result

    # Retain the strongest plane, then require its support across the image box.
    rng = np.random.default_rng(6)
    best = None
    for _ in range(160):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-6:
            continue
        distance = (points - a) @ (normal / length)
        inliers = np.abs(distance) <= 0.025
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None:
        return result

    fraction = float(best.mean())
    plane_cells = len(np.unique(cells[best], axis=0))
    result.update(plane_fraction=round(fraction, 3), plane_cells=plane_cells,
                  plane_median_depth_m=round(float(np.median(points[best, 2])), 3))
    if fraction >= 0.9 and plane_cells >= 6:
        result["decision"] = "keep_flat_candidate"
    elif fraction < 0.7:
        result["decision"] = "skip_nonplanar"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Saved 1280×720 camera frame")
    parser.add_argument("box", type=int, nargs=4, help="Object box, x1 y1 x2 y2")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Saved image does not exist")
    stamp = float(args.image.stem)
    scans = list(args.image.parent.glob("*.cdr"))
    if not scans:
        parser.error("No LiDAR scans in the image folder")
    scan = min(scans, key=lambda path: abs(float(path.stem) - stamp))
    gap = abs(float(scan.stem) - stamp)
    result = {"image": args.image.name, "scan": scan.name,
              "timestamp_gap_ms": round(gap * 1000), "box": args.box}
    if gap > 0.10:
        result.update(decision="skip_uncertain", reason="timestamp gap")
    else:
        result.update(classify(*project(scan), args.box))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
