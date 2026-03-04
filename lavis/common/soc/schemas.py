"""Canonical SOC schemas used by runtime services and integrations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


LIFECYCLE_STATES = {"entered", "continued", "reentered", "exited"}
OBSERVATION_SOURCES = {"schedule", "auto_motion", "detector_tracker"}
THREAT_POLICY_ACTIONS = {
    "review_required",
    "escalate_level_1",
    "escalate_level_2",
}
THREAT_SEVERITIES = {"info", "low", "medium", "high", "critical"}


def utc_now_iso():
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    return ts.isoformat().replace("+00:00", "Z")


def _clamp01(value):
    try:
        x = float(value)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass
class EntityTrackEvent:
    event_id: str
    timestamp_utc: str
    site_id: str
    camera_id: str
    entity_id_local: str
    entity_id_global: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None
    track_confidence: float = 0.0
    reid_embedding_ref: Optional[str] = None
    lifecycle_state: str = "continued"
    context_label: Optional[str] = None
    context_confidence: float = 0.0
    observation_source: str = "detector_tracker"
    entity_event_sequences: Optional[List[Dict[str, Any]]] = None
    entity_lifecycle: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.lifecycle_state not in LIFECYCLE_STATES:
            self.lifecycle_state = "continued"
        if self.observation_source not in OBSERVATION_SOURCES:
            self.observation_source = "detector_tracker"
        self.track_confidence = _clamp01(self.track_confidence)
        self.context_confidence = _clamp01(self.context_confidence)
        if self.entity_id_global is not None:
            self.entity_id_global = str(self.entity_id_global)
        if self.bbox is not None:
            self.bbox = self._normalize_bbox(self.bbox)

    @staticmethod
    def _normalize_bbox(bbox):
        if not isinstance(bbox, dict):
            return None
        out = {}
        for key in ["x1", "y1", "x2", "y2", "x", "y", "w", "h"]:
            if key not in bbox:
                continue
            try:
                out[key] = float(bbox[key])
            except Exception:
                continue
        return out if out else None

    def to_dict(self):
        return {
            "event_id": str(self.event_id),
            "timestamp_utc": str(self.timestamp_utc),
            "site_id": str(self.site_id),
            "camera_id": str(self.camera_id),
            "entity_id_local": str(self.entity_id_local),
            "entity_id_global": self.entity_id_global,
            "bbox": self.bbox,
            "track_confidence": float(self.track_confidence),
            "reid_embedding_ref": self.reid_embedding_ref,
            "lifecycle_state": self.lifecycle_state,
            "context_label": self.context_label,
            "context_confidence": float(self.context_confidence),
            "observation_source": self.observation_source,
            "entity_event_sequences": self.entity_event_sequences,
            "entity_lifecycle": self.entity_lifecycle,
            "metadata": dict(self.metadata),
        }


@dataclass
class ThreatEvent:
    threat_type: str
    severity: str
    confidence_calibrated: float
    entity_refs: List[str] = field(default_factory=list)
    camera_refs: List[str] = field(default_factory=list)
    clip_ref: Optional[str] = None
    markov_state: Optional[str] = None
    anomaly_score: float = 0.0
    fusion_score: float = 0.0
    policy_action: str = "review_required"
    explanations: List[str] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=utc_now_iso)
    site_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence_calibrated = _clamp01(self.confidence_calibrated)
        self.anomaly_score = _clamp01(self.anomaly_score)
        self.fusion_score = _clamp01(self.fusion_score)
        if self.policy_action not in THREAT_POLICY_ACTIONS:
            self.policy_action = "review_required"
        if self.severity not in THREAT_SEVERITIES:
            self.severity = "medium"
        self.entity_refs = [str(x) for x in self.entity_refs if x is not None]
        self.camera_refs = [str(x) for x in self.camera_refs if x is not None]

    def to_dict(self):
        return {
            "threat_type": str(self.threat_type),
            "severity": self.severity,
            "confidence_calibrated": float(self.confidence_calibrated),
            "entity_refs": list(self.entity_refs),
            "camera_refs": list(self.camera_refs),
            "clip_ref": self.clip_ref,
            "markov_state": self.markov_state,
            "anomaly_score": float(self.anomaly_score),
            "fusion_score": float(self.fusion_score),
            "policy_action": self.policy_action,
            "explanations": list(self.explanations),
            "timestamp_utc": str(self.timestamp_utc),
            "site_id": self.site_id,
            "metadata": dict(self.metadata),
        }
