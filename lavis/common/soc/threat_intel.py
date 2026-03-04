"""Threat taxonomy, hybrid anomaly scoring, and incident fusion services."""

import json
import math
from collections import defaultdict, deque
from datetime import datetime

from .schemas import ThreatEvent


SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


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


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _severity_max(a, b):
    if SEVERITY_ORDER.get(a, -1) >= SEVERITY_ORDER.get(b, -1):
        return a
    return b


def _to_epoch_seconds(ts_utc):
    if ts_utc is None:
        return None
    if isinstance(ts_utc, (int, float)):
        return float(ts_utc)
    text = str(ts_utc)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _hash_embedding_ref(value):
    if value is None:
        return 0.0
    text = str(value)
    acc = 0
    for ch in text:
        acc = (acc * 131 + ord(ch)) % 1000003
    return float(acc) / 1000003.0


class ThreatTaxonomyV2:
    def __init__(
        self,
        classes=None,
        event_mappings=None,
        keyword_mappings=None,
        severity_policy_map=None,
        minimum_candidate_confidence=0.25,
    ):
        self.classes = classes or {
            "weapon": {"severity": "critical", "policy_action": "escalate_level_2"},
            "assault_fight": {"severity": "high", "policy_action": "escalate_level_2"},
            "fire_smoke": {"severity": "critical", "policy_action": "escalate_level_2"},
            "intrusion": {"severity": "high", "policy_action": "escalate_level_1"},
            "anomalous_behavior": {
                "severity": "medium",
                "policy_action": "review_required",
            },
        }
        self.event_mappings = event_mappings or {}
        self.keyword_mappings = keyword_mappings or {
            "weapon": "weapon",
            "gun": "weapon",
            "knife": "weapon",
            "fight": "assault_fight",
            "assault": "assault_fight",
            "fire": "fire_smoke",
            "smoke": "fire_smoke",
            "intrusion": "intrusion",
            "trespass": "intrusion",
            "loiter": "anomalous_behavior",
        }
        self.severity_policy_map = severity_policy_map or {
            "critical": "escalate_level_2",
            "high": "escalate_level_1",
            "medium": "review_required",
            "low": "review_required",
            "info": "review_required",
        }
        self.minimum_candidate_confidence = _clamp01(minimum_candidate_confidence)

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as fp:
            payload = json.load(fp)

        event_mappings = {}
        raw_event_map = payload.get("event_to_threat", payload.get("event_mappings", {}))
        if isinstance(raw_event_map, dict):
            for event_id, meta in raw_event_map.items():
                if isinstance(meta, str):
                    event_mappings[str(event_id)] = {"threat_type": str(meta)}
                elif isinstance(meta, dict):
                    event_mappings[str(event_id)] = dict(meta)
        elif isinstance(raw_event_map, list):
            for item in raw_event_map:
                if not isinstance(item, dict):
                    continue
                event_id = item.get("event_id")
                threat_type = item.get("threat_type")
                if not event_id or not threat_type:
                    continue
                event_mappings[str(event_id)] = dict(item)

        return cls(
            classes=payload.get("classes", None),
            event_mappings=event_mappings,
            keyword_mappings=payload.get("keyword_mappings", None),
            severity_policy_map=payload.get("severity_policy_map", None),
            minimum_candidate_confidence=payload.get("minimum_candidate_confidence", 0.25),
        )

    def _class_meta(self, threat_type):
        meta = self.classes.get(threat_type, {})
        severity = str(meta.get("severity", "medium"))
        policy_action = str(
            meta.get("policy_action", self.severity_policy_map.get(severity, "review_required"))
        )
        return severity, policy_action

    def classify(self, event_id, event_label=None, base_confidence=0.0):
        event_id = str(event_id or "")
        event_label = str(event_label or "")
        confidence = _clamp01(base_confidence)

        if event_id in self.event_mappings:
            item = self.event_mappings[event_id]
            threat_type = str(item.get("threat_type", "anomalous_behavior"))
            severity = str(item.get("severity", self._class_meta(threat_type)[0]))
            policy_action = str(item.get("policy_action", self._class_meta(threat_type)[1]))
            mapped_conf = _clamp01(item.get("base_confidence", confidence))
            return {
                "threat_type": threat_type,
                "severity": severity,
                "policy_action": policy_action,
                "base_confidence": max(confidence, mapped_conf),
                "source": "event_mapping",
            }

        text = f"{event_id} {event_label}".lower()
        for keyword, threat_type in self.keyword_mappings.items():
            if str(keyword).lower() in text:
                severity, policy_action = self._class_meta(threat_type)
                return {
                    "threat_type": str(threat_type),
                    "severity": severity,
                    "policy_action": policy_action,
                    "base_confidence": confidence,
                    "source": "keyword_mapping",
                }

        return {
            "threat_type": "anomalous_behavior",
            "severity": "low",
            "policy_action": self.severity_policy_map.get("low", "review_required"),
            "base_confidence": confidence,
            "source": "fallback",
        }

    def should_emit_candidate(self, confidence):
        return float(confidence) >= self.minimum_candidate_confidence


class EmbeddingNoveltyModel:
    """Lightweight open-set novelty estimator per ecological context."""

    def __init__(self, max_points_per_context=500):
        self.max_points_per_context = max(32, int(max_points_per_context))
        self.context_points = defaultdict(lambda: deque(maxlen=self.max_points_per_context))

    def _to_scalar_feature(self, embedding_ref):
        return _hash_embedding_ref(embedding_ref)

    def score(self, context_label, embedding_ref):
        key = str(context_label or "default")
        points = self.context_points[key]
        feature = self._to_scalar_feature(embedding_ref)
        if len(points) == 0:
            points.append(feature)
            return 0.5
        mean = sum(points) / float(len(points))
        variance = sum((p - mean) ** 2 for p in points) / float(max(1, len(points) - 1))
        std = math.sqrt(max(variance, 1e-6))
        z = abs(feature - mean) / max(std, 1e-3)
        points.append(feature)
        # Convert z-score to bounded [0,1] novelty probability.
        return _clamp01(1.0 - math.exp(-0.5 * z))


class TemporalRuleEngine:
    def __init__(self, loitering_window_threshold=8, reentry_weight=0.3):
        self.loitering_window_threshold = max(2, int(loitering_window_threshold))
        self.reentry_weight = _clamp01(reentry_weight)

    def score(self, entity_summary):
        if not isinstance(entity_summary, dict):
            return 0.0, []

        explanations = []
        score = 0.0

        sequence_length = int(entity_summary.get("sequence_length", 0))
        if sequence_length >= self.loitering_window_threshold:
            score += 0.6
            explanations.append(
                f"loitering_windows={sequence_length}>=threshold={self.loitering_window_threshold}"
            )

        lifecycle = str(entity_summary.get("lifecycle_state", "continued"))
        if lifecycle == "reentered":
            score += self.reentry_weight
            explanations.append("reentry_pattern_detected")

        state = entity_summary.get("markov_state") or {}
        if _safe_float(state.get("prob", 0.0), 0.0) < 0.35:
            score += 0.2
            explanations.append("high_markov_uncertainty")

        return _clamp01(score), explanations


class HybridAnomalyScorer:
    def __init__(
        self,
        markov_weight=0.45,
        novelty_weight=0.35,
        temporal_weight=0.20,
        novelty_model=None,
        temporal_rules=None,
    ):
        self.markov_weight = _clamp01(markov_weight)
        self.novelty_weight = _clamp01(novelty_weight)
        self.temporal_weight = _clamp01(temporal_weight)
        total = self.markov_weight + self.novelty_weight + self.temporal_weight
        if total <= 1e-9:
            total = 1.0
        self.markov_weight /= total
        self.novelty_weight /= total
        self.temporal_weight /= total

        self.novelty_model = novelty_model or EmbeddingNoveltyModel()
        self.temporal_rules = temporal_rules or TemporalRuleEngine()

    def score(self, context_label, entity_summary, markov_state, reid_embedding_ref):
        markov_prob = _safe_float((markov_state or {}).get("prob", 0.0), 0.0)
        markov_anomaly = _clamp01(1.0 - markov_prob)

        novelty = self.novelty_model.score(context_label, reid_embedding_ref)
        temporal, temporal_explanations = self.temporal_rules.score(entity_summary)

        score = (
            self.markov_weight * markov_anomaly
            + self.novelty_weight * novelty
            + self.temporal_weight * temporal
        )
        explanations = [
            f"markov_surprise={markov_anomaly:.3f}",
            f"novelty={novelty:.3f}",
            f"temporal={temporal:.3f}",
        ] + temporal_explanations
        return _clamp01(score), explanations


class IncidentFusionService:
    """Merge entity-level threat candidates into incident episodes."""

    def __init__(self, merge_window_seconds=8.0):
        self.merge_window_seconds = max(0.5, float(merge_window_seconds))
        self._episodes = {}
        self._counter = 0

    def _new_incident_id(self):
        self._counter += 1
        return f"incident_{self._counter:08d}"

    def _match_key(self, threat_event):
        event_ts = _to_epoch_seconds(threat_event.timestamp_utc)
        if event_ts is None:
            event_ts = 0.0
        bucket = int(event_ts // self.merge_window_seconds)
        site_id = str(threat_event.site_id or "site_default")
        threat_type = str(threat_event.threat_type)
        entity_anchor = ""
        if threat_event.entity_refs:
            entity_anchor = str(sorted(threat_event.entity_refs)[0])
        return (site_id, threat_type, bucket, entity_anchor)

    def fuse(self, threat_event):
        if not isinstance(threat_event, ThreatEvent):
            return None

        key = self._match_key(threat_event)
        episode = self._episodes.get(key)
        if episode is None:
            incident_id = self._new_incident_id()
            episode = {
                "incident_id": incident_id,
                "site_id": threat_event.site_id,
                "threat_type": threat_event.threat_type,
                "severity": threat_event.severity,
                "camera_refs": set(threat_event.camera_refs),
                "entity_refs": set(threat_event.entity_refs),
                "confidence_max": float(threat_event.confidence_calibrated),
                "anomaly_max": float(threat_event.anomaly_score),
                "events": [threat_event],
            }
            self._episodes[key] = episode
        else:
            episode["camera_refs"].update(threat_event.camera_refs)
            episode["entity_refs"].update(threat_event.entity_refs)
            episode["severity"] = _severity_max(episode["severity"], threat_event.severity)
            episode["confidence_max"] = max(
                float(episode["confidence_max"]), float(threat_event.confidence_calibrated)
            )
            episode["anomaly_max"] = max(
                float(episode["anomaly_max"]), float(threat_event.anomaly_score)
            )
            episode["events"].append(threat_event)

        # Fusion score increases with corroboration across cameras/entities.
        corroboration = 0.0
        corroboration += min(len(episode["camera_refs"]), 4) * 0.12
        corroboration += min(len(episode["entity_refs"]), 4) * 0.10
        corroboration += min(len(episode["events"]), 5) * 0.08
        fusion_score = _clamp01(
            0.55 * float(episode["confidence_max"])
            + 0.25 * float(episode["anomaly_max"])
            + 0.20 * corroboration
        )
        return {
            "incident_id": episode["incident_id"],
            "threat_type": episode["threat_type"],
            "severity": episode["severity"],
            "camera_refs": sorted(list(episode["camera_refs"])),
            "entity_refs": sorted(list(episode["entity_refs"])),
            "fusion_score": fusion_score,
            "event_count": len(episode["events"]),
        }
