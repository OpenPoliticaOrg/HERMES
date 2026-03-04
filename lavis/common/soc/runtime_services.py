"""Runtime service handlers aligned to planned SOC gRPC service contracts."""

from typing import Any, Dict

from .routing import NATS_SUBJECTS
from .schemas import EntityTrackEvent, ThreatEvent, utc_now_iso
from .service_contracts import (
    AlertDispatchService as AlertDispatchServiceContract,
    EntityFusionService as EntityFusionServiceContract,
    FeedbackIngestService as FeedbackIngestServiceContract,
    InferenceProfileServiceContract,
    IngestGatewayService as IngestGatewayServiceContract,
    ThreatScoringService as ThreatScoringServiceContract,
)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _to_entity_track_event(payload, default_site_id, default_camera_id):
    payload = _as_dict(payload)
    return EntityTrackEvent(
        event_id=str(payload.get("event_id", "unknown_event")),
        timestamp_utc=str(payload.get("timestamp_utc", utc_now_iso())),
        site_id=str(payload.get("site_id", default_site_id)),
        camera_id=str(payload.get("camera_id", default_camera_id)),
        entity_id_local=str(payload.get("entity_id_local", "__scene__")),
        entity_id_global=payload.get("entity_id_global"),
        bbox=payload.get("bbox"),
        track_confidence=_safe_float(payload.get("track_confidence", 0.0), 0.0),
        reid_embedding_ref=payload.get("reid_embedding_ref"),
        lifecycle_state=str(payload.get("lifecycle_state", "continued")),
        context_label=payload.get("context_label"),
        context_confidence=_safe_float(payload.get("context_confidence", 0.0), 0.0),
        observation_source=str(payload.get("observation_source", "detector_tracker")),
        entity_event_sequences=payload.get("entity_event_sequences"),
        entity_lifecycle=payload.get("entity_lifecycle"),
        metadata=_as_dict(payload.get("metadata")),
    )


def _to_threat_event(payload, default_site_id, default_camera_id):
    payload = _as_dict(payload)
    return ThreatEvent(
        threat_type=str(payload.get("threat_type", "anomalous_behavior")),
        severity=str(payload.get("severity", "medium")),
        confidence_calibrated=_safe_float(payload.get("confidence_calibrated", 0.0), 0.0),
        entity_refs=list(payload.get("entity_refs", [])),
        camera_refs=list(payload.get("camera_refs", [default_camera_id])),
        clip_ref=payload.get("clip_ref"),
        markov_state=payload.get("markov_state"),
        anomaly_score=_safe_float(payload.get("anomaly_score", 0.0), 0.0),
        fusion_score=_safe_float(payload.get("fusion_score", 0.0), 0.0),
        policy_action=str(payload.get("policy_action", "review_required")),
        explanations=list(payload.get("explanations", [])),
        timestamp_utc=str(payload.get("timestamp_utc", utc_now_iso())),
        site_id=str(payload.get("site_id", default_site_id)),
        metadata=_as_dict(payload.get("metadata")),
    )


class IngestGatewayRuntimeService(IngestGatewayServiceContract):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def ingest_observation(self, request):
        request = _as_dict(request)
        if isinstance(request.get("result"), dict):
            payload = request["result"]
        else:
            track = _to_entity_track_event(
                request.get("track_event"),
                default_site_id=self.orchestrator.site_id,
                default_camera_id=self.orchestrator.camera_id,
            )
            bbox = track.bbox if isinstance(track.bbox, dict) else {}
            bbox_xyxy = [
                _safe_float(bbox.get("x1", 0.0), 0.0),
                _safe_float(bbox.get("y1", 0.0), 0.0),
                _safe_float(bbox.get("x2", 0.0), 0.0),
                _safe_float(bbox.get("y2", 0.0), 0.0),
            ]
            payload = {
                "timestamp_utc": track.timestamp_utc,
                self.orchestrator.context_field: track.context_label,
                "entity_observation_source": track.observation_source,
                "markov_state": {
                    "event_id": track.event_id,
                    "prob": track.track_confidence,
                },
                "entity_event_sequences": [
                    {
                        "entity_id": track.entity_id_local,
                        "lifecycle_state": track.lifecycle_state,
                        "markov_state": {
                            "event_id": track.event_id,
                            "prob": track.track_confidence,
                        },
                        "event_predictions": [],
                        "metadata": {
                            "bbox_xyxy_norm": bbox_xyxy,
                            "reid_embedding_ref": track.reid_embedding_ref,
                        },
                    }
                ],
                "entity_lifecycle": track.entity_lifecycle or {},
            }

        soc_payload = self.orchestrator.process_result(payload)
        return {
            "status": "ok",
            "timestamp_utc": payload.get("timestamp_utc", utc_now_iso()),
            "soc_payload": soc_payload,
        }


class InferenceProfileRuntimeService(InferenceProfileServiceContract):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def resolve_profile(self, request):
        request = _as_dict(request)
        role_hint = str(request.get("role_hint", "")).lower().strip()
        force_profile = request.get("force_profile")
        hw = dict(self.orchestrator.hardware_snapshot or {})
        if role_hint in {"edge", "core"}:
            hw["site_role"] = role_hint
        profile = self.orchestrator.profile_service.select_profile(
            hardware_snapshot=hw,
            force_profile=force_profile,
        )
        return profile.to_dict()


class EntityFusionRuntimeService(EntityFusionServiceContract):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def upsert_entity_track(self, request):
        request = _as_dict(request)
        track = _to_entity_track_event(
            request.get("track_event"),
            default_site_id=self.orchestrator.site_id,
            default_camera_id=self.orchestrator.camera_id,
        )
        entity_id_global = self.orchestrator._entity_global_id(
            track.entity_id_local,
            timestamp_utc=track.timestamp_utc,
            reid_embedding_ref=track.reid_embedding_ref,
        )
        payload = track.to_dict()
        payload["entity_id_global"] = entity_id_global
        return {
            "status": "ok",
            "entity_id_local": track.entity_id_local,
            "entity_id_global": entity_id_global,
            "track_event": payload,
        }


class ThreatScoringRuntimeService(ThreatScoringServiceContract):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    @staticmethod
    def _clamp01(value):
        x = _safe_float(value, 0.0)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def score_threat(self, request):
        request = _as_dict(request)
        track = _to_entity_track_event(
            request.get("track_event"),
            default_site_id=self.orchestrator.site_id,
            default_camera_id=self.orchestrator.camera_id,
        )
        threat_cls = self.orchestrator.taxonomy.classify(
            event_id=track.event_id,
            event_label=None,
            base_confidence=track.track_confidence,
        )
        anomaly_score, explanations = self.orchestrator.anomaly_scorer.score(
            context_label=track.context_label,
            entity_summary={},
            markov_state={"event_id": track.event_id, "prob": track.track_confidence},
            reid_embedding_ref=track.reid_embedding_ref,
        )
        confidence = self._clamp01(
            0.65 * _safe_float(threat_cls.get("base_confidence", 0.0), 0.0)
            + 0.35 * _safe_float(anomaly_score, 0.0)
        )
        confidence = self.orchestrator.confidence_calibrator.calibrate(confidence)
        if not self.orchestrator.taxonomy.should_emit_candidate(confidence):
            return {"status": "suppressed", "threat_event": None}

        threat_event = ThreatEvent(
            threat_type=threat_cls.get("threat_type", "anomalous_behavior"),
            severity=threat_cls.get("severity", "medium"),
            confidence_calibrated=confidence,
            entity_refs=[track.entity_id_global or track.entity_id_local],
            camera_refs=[track.camera_id],
            clip_ref=None,
            markov_state=track.event_id,
            anomaly_score=anomaly_score,
            fusion_score=0.0,
            policy_action=threat_cls.get("policy_action", "review_required"),
            explanations=[f"taxonomy_source={threat_cls.get('source', 'unknown')}"]
            + list(explanations),
            timestamp_utc=track.timestamp_utc,
            site_id=track.site_id,
            metadata={
                "entity_id_local": track.entity_id_local,
                "track_confidence": track.track_confidence,
            },
        )
        fused = self.orchestrator.fusion_service.fuse(threat_event)
        if isinstance(fused, dict):
            threat_event.fusion_score = self._clamp01(fused.get("fusion_score", 0.0))
            threat_event.metadata["incident_id"] = fused.get("incident_id")
            threat_event.metadata["incident_event_count"] = fused.get("event_count")

        return {"status": "ok", "threat_event": threat_event.to_dict()}


class AlertDispatchRuntimeService(AlertDispatchServiceContract):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def dispatch_alert(self, request):
        request = _as_dict(request)
        threat_event = _to_threat_event(
            request.get("threat_event"),
            default_site_id=self.orchestrator.site_id,
            default_camera_id=self.orchestrator.camera_id,
        )
        confirmed = bool(request.get("confirmed", False))
        subject = (
            NATS_SUBJECTS["threat_alert_confirmed"]
            if confirmed
            else NATS_SUBJECTS["threat_alert_candidate"]
        )
        payload = threat_event.to_dict()

        self.orchestrator.routing_service.publish(subject, payload)
        dispatched = self.orchestrator.routing_service.dispatch_step(
            max_dispatch=max(1, int(request.get("max_dispatch", 1)))
        )
        bus_results = []
        for item in dispatched:
            if not isinstance(item, dict):
                continue
            try:
                bus_item = self.orchestrator.message_bus.publish(
                    item.get("subject"), item.get("payload", {})
                )
                bus_results.append(bus_item)
            except Exception:
                continue

        return {
            "status": "ok",
            "subject": subject,
            "dispatched_count": len(dispatched),
            "bus_publish_count": len(bus_results),
        }


class FeedbackIngestRuntimeService(FeedbackIngestServiceContract):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def ingest_feedback(self, request):
        request = _as_dict(request)
        case_id = request.get("case_id")
        if not case_id:
            return {"status": "error", "error": "missing_case_id"}
        actor = str(request.get("analyst_id", "analyst"))
        label = str(request.get("label", "unlabeled"))
        notes = str(request.get("notes", ""))
        feedback = self.orchestrator.workflow_service.ingest_feedback(
            case_id=case_id,
            actor=actor,
            label=label,
            notes=notes,
        )
        if feedback is None:
            return {"status": "error", "error": "feedback_ingest_failed"}
        return {"status": "ok", "feedback": feedback}


class SOCRuntimeServiceSuite:
    """Convenience wrapper bundling concrete runtime services."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.ingest_gateway = IngestGatewayRuntimeService(orchestrator)
        self.inference_profile = InferenceProfileRuntimeService(orchestrator)
        self.entity_fusion = EntityFusionRuntimeService(orchestrator)
        self.threat_scoring = ThreatScoringRuntimeService(orchestrator)
        self.alert_dispatch = AlertDispatchRuntimeService(orchestrator)
        self.feedback_ingest = FeedbackIngestRuntimeService(orchestrator)

    @classmethod
    def from_json_config(cls, path):
        from .runtime import SOCOrchestrator

        return cls(orchestrator=SOCOrchestrator.from_json_config(path))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ingest_gateway": self.ingest_gateway,
            "inference_profile": self.inference_profile,
            "entity_fusion": self.entity_fusion,
            "threat_scoring": self.threat_scoring,
            "alert_dispatch": self.alert_dispatch,
            "feedback_ingest": self.feedback_ingest,
        }
