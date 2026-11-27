"""Match wide-camera sightings against a short burst of Orbbec frames."""

import time

import cv2
import numpy as np

HANDOFF_SECONDS = 2.0


class WideHandoff:
    def __init__(self, camera, detect_objects, detect_qr, stop, *, locate_qr=False, detect_signs=None):
        self.camera = camera
        self.detect_objects = detect_objects
        self.detect_qr = detect_qr
        self.stop = stop
        self.locate_qr = locate_qr
        self.detect_signs = detect_signs
        self.sift = cv2.SIFT_create(nfeatures=240)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def features(self, frame, detection):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection["xyxy"]
        x1, x2 = np.clip([x1, x2], 0, width).astype(int)
        y1, y2 = np.clip([y1, y2], 0, height).astype(int)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[12:-12, 12:-12] = 255
        if detection["label"] == "qr_candidate":
            # Shared finder patterns are weak identity evidence for an unreadable code.
            mask[:78, :78] = mask[:78, -78:] = mask[-78:, :78] = 0
        keypoints, descriptors = self.sift.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), mask)
        if descriptors is None or len(descriptors) < 10:
            return None
        return np.asarray([point.pt for point in keypoints], dtype=np.float32) / 224, descriptors

    def visual_match(self, first, second):
        if first is None or second is None:
            return False
        source, source_desc = first
        target, target_desc = second
        forward = self.matcher.knnMatch(source_desc, target_desc, k=2)
        backward = self.matcher.knnMatch(target_desc, source_desc, k=2)
        reverse = {
            pair[0].queryIdx: pair[0].trainIdx for pair in backward
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        }
        pairs = [
            (pair[0].queryIdx, pair[0].trainIdx) for pair in forward
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
            and reverse.get(pair[0].trainIdx) == pair[0].queryIdx
        ]
        if len(pairs) < 10:
            return False
        a, b = np.asarray(pairs).T
        _, inliers = cv2.findHomography(source[a], target[b], cv2.RANSAC, 0.035)
        if inliers is None:
            return False
        keep = inliers.ravel().astype(bool)
        if keep.sum() < max(10, 0.65 * len(pairs)):
            return False
        return all(cv2.contourArea(cv2.convexHull(points[keep])) >= 0.045
                   for points in (source[a], target[b]))

    def object_matches(self, wide_objects, rgb_objects, frame):
        labels = {detection["label"] for detection, _ in wide_objects}
        rgb_objects = [detection for detection in rgb_objects if detection["label"] in labels]
        rgb_features = [self.features(frame, detection) for detection in rgb_objects]
        matches = [
            [j for j, rgb in enumerate(rgb_objects)
             if wide["label"] == rgb["label"] and self.visual_match(features, rgb_features[j])]
            for wide, features in wide_objects
        ]
        # Require a unique match in both directions, including alternatives without depth.
        counts = np.bincount([j for row in matches for j in row], minlength=len(rgb_objects))
        return [(wide, rgb_objects[row[0]], "visual_features")
                for (wide, _), row in zip(wide_objects, matches)
                if len(row) == 1 and counts[row[0]] == 1]

    def qr_candidate(self, qr):
        low = np.min(qr["corners"], axis=0)
        high = np.max(qr["corners"], axis=0)
        return dict(qr, label="qr_candidate", xyxy=[*map(int, low), *map(int, high)])

    def locate(self, recorder, frame, detections, objects, qr_codes, unreadable_qr=()):
        wide_objects = []
        for detection in objects:
            features = self.features(frame, detection)
            if features is None:
                detection["location_issue"] = "too few visual features for a cross-camera match"
            else:
                wide_objects.append((detection, features))
        requested_qr = list(qr_codes) if self.locate_qr else []
        qr_codes = {
            qr["text"]: qr for qr in requested_qr
            if sum(other["decoded"] and other["text"] == qr["text"] for other in detections["qr_codes"]) == 1
        }
        for qr in requested_qr:
            if qr["text"] not in qr_codes:
                qr["location_issue"] = "multiple wide-camera QR detections have the same text"
        unresolved = []
        for qr in unreadable_qr:
            candidate = self.qr_candidate(qr)
            features = self.features(frame, candidate)
            if features is not None:
                unresolved.append((candidate, features))
        if not (wide_objects or qr_codes or unresolved):
            return []
        known_text = {qr["text"] for qr in detections["qr_codes"] if qr["decoded"]}
        with self.camera.inspection() as active:
            return self.inspect_targets(recorder, detections["frame_received"], wide_objects,
                                        qr_codes, unresolved, known_text, active)

    def inspect_targets(self, recorder, wide_received, wide_objects, qr_codes, unresolved,
                        known_text, active):
        print("Wide target found. Checking Orbbec.", flush=True)
        deadline = time.monotonic() + HANDOFF_SECONDS
        recovered = []
        last_frame = None
        completed = set()
        located = 0
        issue = "no fresh Orbbec color frame" if active else "Orbbec subscription unavailable"
        target_issues = {}
        while (active and (len(completed) < len(wide_objects) or qr_codes or unresolved)
               and not self.stop.is_set() and time.monotonic() < deadline):
            if self.camera.error:
                issue = "Orbbec camera unavailable"
                break
            sample = self.camera.latest()
            if (sample is None or sample[0] == last_frame or sample[0] < wide_received - 0.15
                    or time.monotonic() - sample[0] > 0.5):
                self.stop.wait(0.02)
                continue
            last_frame, observed_at, rgb_frame, color = sample
            rgb_qr = self.detect_qr(rgb_frame) if qr_codes or unresolved else []
            if unresolved:
                candidates = [self.qr_candidate(qr) for qr in rgb_qr
                              if qr["decoded"] and qr["text"].strip() and qr["text"] not in known_text]
                pairs = self.object_matches(unresolved, candidates, rgb_frame)
                for wide, decoded, _ in pairs:
                    if decoded["text"] in known_text:
                        continue
                    qr = {key: decoded[key] for key in ("decoded", "text", "source", "corners")}
                    qr["decode_evidence_ref"] = recorder.save_frame(rgb_frame, observed_at, {
                        "camera_role": "wide_search_qr_decode",
                        "timestamp_source": "ROS color image header",
                        "qr_codes": [dict(qr)],
                    })
                    recovered.append(qr)
                    known_text.add(qr["text"])
                matched_ids = {id(wide) for wide, _, _ in pairs}
                unresolved = [item for item in unresolved if id(item[0]) not in matched_ids]
            if len(completed) == len(wide_objects) and not qr_codes:
                continue
            if not self.camera.has_map_transform(color):
                issue = "map transform unavailable"
                self.stop.wait(0.05)
                continue
            depth_pair = self.camera.pair_depth(color)
            if depth_pair is None:
                issue = "no time-matched Orbbec depth frame"
                continue

            qr_matches = []
            for text, wide in qr_codes.items():
                candidates = [qr for qr in rgb_qr if qr["decoded"] and qr["text"] == text]
                if len(candidates) == 1:
                    qr_matches.append((wide, candidates[0], "qr_text"))
            pairs = []
            if len(completed) < len(wide_objects):
                labels = {wide["label"] for wide, _ in wide_objects if id(wide) not in completed}
                rgb_objects = self.detect_objects(rgb_frame) if labels - {"hazard_sign"} else []
                if self.detect_signs is not None and "hazard_sign" in labels:
                    rgb_objects.extend(dict(label="hazard_sign", xyxy=sign["box"])
                                       for sign in self.detect_signs(rgb_frame))
                pairs = [(wide, rgb, method) for wide, rgb, method
                         in self.object_matches(wide_objects, rgb_objects, rgb_frame)
                         if id(wide) not in completed]

            issue = "no unique Orbbec match"
            if not (pairs or qr_matches):
                continue
            projection_issue = self.camera.locate(
                [rgb for _, rgb, _ in pairs], color, depth_pair,
                qr_codes=[rgb for _, rgb, _ in qr_matches],
            )
            for wide, rgb, _ in pairs + qr_matches:
                if "location" not in rgb:
                    target_issues[id(wide)] = projection_issue or "matched target had no reliable depth"
            matched = [(wide, rgb, method) for wide, rgb, method in pairs + qr_matches if "location" in rgb]
            if not matched:
                continue

            reference = recorder.save_frame(rgb_frame, observed_at, {
                "timestamp_source": "ROS color image header",
                "camera_role": "wide_search_location",
                "image_width": rgb_frame.shape[1],
                "image_height": rgb_frame.shape[0],
                "objects": [rgb for _, rgb, method in matched if method == "visual_features"],
                "qr_codes": [rgb for _, rgb, method in matched if method == "qr_text"],
            })
            for wide, rgb, method in matched:
                wide.update(location=rgb["location"], location_source=rgb["location_source"],
                            location_evidence_ref=reference, location_observed_at=observed_at,
                            location_match=method)
                if method == "qr_text":
                    del qr_codes[wide["text"]]
                else:
                    completed.add(id(wide))
            located += len(matched)
        missed = [wide for wide, _ in wide_objects if id(wide) not in completed] + list(qr_codes.values())
        for target in missed:
            target["location_issue"] = target_issues.get(id(target), issue)
        reasons = ", ".join(sorted({target["location_issue"] for target in missed}))
        details = f" Position unavailable: {reasons}." if reasons else ""
        print(f"Back to wide search. Positions saved: {located}. QR decodes: {len(recovered)}.{details}", flush=True)
        return recovered
