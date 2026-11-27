"""BoT-SORT tracking and anonymous appearance memory for one recorder run."""

from uuid import uuid4

import numpy as np
import torch
from boxmot import BotSort


class PersonReID:
    def __init__(self, weights, confidence):
        if not weights.is_file():
            raise FileNotFoundError(f"Missing local ReID weights: {weights}")
        self.tracker = BotSort(
            reid_weights=weights,
            device=torch.device("cpu"),
            half=False,
            track_high_thresh=max(0.01, confidence - 0.001),
            new_track_thresh=confidence,
            cmc_method="sof",
        )
        self.track_entities = {}
        self.gallery = {}

    def repeated_detections(self, frame, objects):
        indices = [i for i, obj in enumerate(objects) if obj["label"] == "person"]
        detections = np.asarray([
            [*objects[i]["xyxy"], objects[i]["confidence"], 0] for i in indices
        ], dtype=np.float32).reshape(-1, 6)
        if not indices:
            self.tracker.update(detections, frame)
            return set()

        embeddings = self.tracker.model.get_features(detections[:, :4], frame)
        tracks = self.tracker.update(detections, frame, embeddings)
        track_for_detection = {int(row[7]): int(row[4]) for row in tracks}
        known_entities = {
            index: self.track_entities[track]
            for index, track in track_for_detection.items()
            if track in self.track_entities
        }
        claimed = set()
        repeated = set()
        for index in sorted(range(len(indices)), key=lambda i: i not in known_entities):
            track = track_for_detection.get(index)
            feature = embeddings[index]
            if not np.isfinite(feature).all():
                feature = None
            entity = known_entities.get(index)
            if entity is None and feature is not None:
                entity = self.match(feature)
            if entity is None or entity in claimed:
                entity = uuid4().hex
                if feature is not None:
                    self.gallery[entity] = feature.copy()
            else:
                repeated.add(indices[index])
            claimed.add(entity)
            if track is not None:
                self.track_entities[track] = entity
        return repeated

    def match(self, feature):
        ranked = sorted(
            ((float(feature @ stored), entity) for entity, stored in self.gallery.items()),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.88:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08:
            return None
        return ranked[0][1]
