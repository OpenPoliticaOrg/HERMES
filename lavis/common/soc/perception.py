"""Perception pipeline interfaces for detector/tracker/ReID plug-in stacks."""

from abc import ABC, abstractmethod


class Detector(ABC):
    @abstractmethod
    def detect(self, frame):
        """Return detection list: [{bbox, class_id, score}, ...]."""


class Tracker(ABC):
    @abstractmethod
    def update(self, detections):
        """Return track list: [{track_id, bbox, score}, ...]."""


class ReIDModel(ABC):
    @abstractmethod
    def embed(self, crop):
        """Return vector embedding for an entity crop."""


class NullDetector(Detector):
    def detect(self, frame):
        return []


class NullTracker(Tracker):
    def update(self, detections):
        return []


class NullReID(ReIDModel):
    def embed(self, crop):
        return None
