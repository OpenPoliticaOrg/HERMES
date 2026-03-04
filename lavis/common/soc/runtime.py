"""End-to-end SOC runtime orchestration for streaming inference outputs."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from .calibration import ConfidenceCalibrator
from .federation import EntityFederationService
from .ingestion_health import IngestionHealthMonitor
from .message_bus import InMemoryMessageBus, NATSJetStreamMessageBus
from .mlops import (
    CanaryRolloutManager,
    DriftMonitor,
    RolloutGuardrailPolicy,
    SLOMonitor,
    SignedModelRegistry,
)
from .profiles import InferenceProfileService, detect_runtime_hardware
from .routing import NATS_SUBJECTS, RuntimeRoutingPolicyService
from .schemas import EntityTrackEvent, ThreatEvent, utc_now_iso
from .security import ImmutableAuditLog, RBACPolicyEngine, TransportSecurityConfig
from .stores import (
    ClickHouseEventStore,
    FilesystemClipStore,
    InMemoryEventStore,
    InMemoryHotStateStore,
    RedisHotStateStore,
)
from .threat_intel import (
    HybridAnomalyScorer,
    IncidentFusionService,
    ThreatTaxonomyV2,
)
from .workflow import AlertWorkflowService


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


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


class SOCOrchestrator:
    """Build canonical events, score threats, and route SOC messages."""

    def __init__(
        self,
        site_id="site_demo",
        camera_id="camera_demo",
        context_field="ecological_context",
        taxonomy=None,
        anomaly_scorer=None,
        fusion_service=None,
        routing_service=None,
        workflow_service=None,
        ingestion_health_monitor=None,
        profile_service=None,
        confidence_calibrator=None,
        entity_federation_service=None,
        message_bus=None,
        hot_state_store=None,
        event_store=None,
        clip_store=None,
        rbac_engine=None,
        audit_log=None,
        transport_security_config=None,
        model_registry=None,
        canary_rollout_manager=None,
        rollout_guardrail_policy=None,
        drift_monitor=None,
        slo_monitor=None,
        service_account_id="svc_soc_runtime",
        rollout_id=None,
        observation_source_default="detector_tracker",
    ):
        self.site_id = str(site_id)
        self.camera_id = str(camera_id)
        self.context_field = str(context_field)
        self.observation_source_default = str(observation_source_default)

        self.taxonomy = taxonomy or ThreatTaxonomyV2()
        self.anomaly_scorer = anomaly_scorer or HybridAnomalyScorer()
        self.fusion_service = fusion_service or IncidentFusionService()
        self.routing_service = routing_service or RuntimeRoutingPolicyService()
        self.workflow_service = workflow_service or AlertWorkflowService()
        self.ingestion_health_monitor = ingestion_health_monitor or IngestionHealthMonitor()
        self.profile_service = profile_service or InferenceProfileService()
        self.confidence_calibrator = confidence_calibrator or ConfidenceCalibrator()
        self.entity_federation_service = (
            entity_federation_service or EntityFederationService()
        )
        self.message_bus = message_bus or InMemoryMessageBus()
        self.hot_state_store = hot_state_store or InMemoryHotStateStore()
        self.event_store = event_store or InMemoryEventStore()
        self.clip_store = clip_store
        self.rbac_engine = rbac_engine or RBACPolicyEngine()
        self.audit_log = audit_log or ImmutableAuditLog()
        self.transport_security_config = (
            transport_security_config or TransportSecurityConfig()
        )
        self.model_registry = model_registry or SignedModelRegistry()
        self.canary_rollout_manager = canary_rollout_manager or CanaryRolloutManager()
        self.rollout_guardrail_policy = (
            rollout_guardrail_policy or RolloutGuardrailPolicy()
        )
        self.drift_monitor = drift_monitor or DriftMonitor()
        self.slo_monitor = slo_monitor or SLOMonitor()
        self.service_account_id = str(service_account_id)
        self.rollout_id = str(rollout_id) if rollout_id else None
        self.transport_security_status = self.transport_security_config.validate()

        self.hardware_snapshot = detect_runtime_hardware()
        self.active_profile = self.profile_service.select_profile(self.hardware_snapshot)
        if self.rollout_id:
            canary_profile_id = self.canary_rollout_manager.resolve_profile(
                rollout_id=self.rollout_id,
                site_id=self.site_id,
                camera_id=self.camera_id,
            )
            if canary_profile_id in getattr(self.profile_service, "profiles", {}):
                self.active_profile = self.profile_service.profiles[canary_profile_id]

        self._global_entity_cache = {}

    @classmethod
    def from_json_config(cls, path):
        config_path = Path(path).resolve()
        with open(config_path, "r") as fp:
            payload = json.load(fp)

        taxonomy = None
        taxonomy_path = payload.get("threat_taxonomy_path")
        if taxonomy_path:
            resolved = Path(taxonomy_path)
            if not resolved.is_absolute():
                rel = resolved
                candidates = [
                    config_path.parent / rel,
                    Path.cwd() / rel,
                ]
                for parent in config_path.parents:
                    candidates.append(parent / rel)
                resolved = candidates[0]
                for candidate in candidates:
                    if candidate.exists():
                        resolved = candidate
                        break
            taxonomy = ThreatTaxonomyV2.from_file(str(resolved))
        elif isinstance(payload.get("threat_taxonomy"), dict):
            taxonomy = ThreatTaxonomyV2(**payload.get("threat_taxonomy"))

        workflow = AlertWorkflowService(
            sla_seconds_by_severity=payload.get("sla_seconds_by_severity"),
            runbooks=payload.get("runbooks"),
        )

        routing_cfg = payload.get("routing", {}) if isinstance(payload.get("routing"), dict) else {}
        routing = RuntimeRoutingPolicyService(
            max_backlog=routing_cfg.get("max_backlog", 5000),
            congestion_soft_limit=routing_cfg.get("congestion_soft_limit", 3000),
            max_retries=routing_cfg.get("max_retries", 3),
            priority_drop_threshold=routing_cfg.get("priority_drop_threshold", 60),
        )

        ingest_cfg = payload.get("ingestion_health", {}) if isinstance(payload.get("ingestion_health"), dict) else {}
        ingestion_monitor = IngestionHealthMonitor(
            expected_fps=ingest_cfg.get("expected_fps", 10.0),
            fps_window=ingest_cfg.get("fps_window", 60),
            drop_timeout_seconds=ingest_cfg.get("drop_timeout_seconds", 5.0),
            max_timestamp_skew_seconds=ingest_cfg.get("max_timestamp_skew_seconds", 2.0),
            max_fps_drift_ratio=ingest_cfg.get("max_fps_drift_ratio", 0.35),
        )

        calibration_cfg = (
            payload.get("confidence_calibration", {})
            if isinstance(payload.get("confidence_calibration"), dict)
            else {}
        )
        confidence_calibrator = ConfidenceCalibrator.from_dict(calibration_cfg)

        federation_cfg = (
            payload.get("entity_federation", {})
            if isinstance(payload.get("entity_federation"), dict)
            else {}
        )
        federation = EntityFederationService(
            max_time_delta_seconds=federation_cfg.get("max_time_delta_seconds", 20.0),
            min_embedding_similarity=federation_cfg.get("min_embedding_similarity", 0.85),
        )

        integrations_cfg = (
            payload.get("integrations", {})
            if isinstance(payload.get("integrations"), dict)
            else {}
        )
        message_bus = cls._build_message_bus(integrations_cfg.get("message_bus", {}))
        hot_state_store = cls._build_hot_store(integrations_cfg.get("hot_store", {}))
        event_store = cls._build_event_store(integrations_cfg.get("event_store", {}))
        clip_store = cls._build_clip_store(
            integrations_cfg.get("clip_store", {}),
            config_path=config_path,
        )

        security_cfg = (
            payload.get("security", {})
            if isinstance(payload.get("security"), dict)
            else {}
        )
        rbac_engine = RBACPolicyEngine.from_dict(security_cfg.get("rbac", {}))
        transport_cfg = (
            security_cfg.get("transport", {})
            if isinstance(security_cfg.get("transport"), dict)
            else {}
        )
        transport_security_config = TransportSecurityConfig(
            mtls_internal=transport_cfg.get("mtls_internal", False),
            tls_external=transport_cfg.get("tls_external", False),
            cert_path=transport_cfg.get("cert_path", ""),
            key_path=transport_cfg.get("key_path", ""),
            ca_path=transport_cfg.get("ca_path", ""),
        )
        audit_log = ImmutableAuditLog()
        service_account_id = security_cfg.get("service_account_id", "svc_soc_runtime")

        mlops_cfg = (
            payload.get("mlops", {})
            if isinstance(payload.get("mlops"), dict)
            else {}
        )
        model_registry = SignedModelRegistry.from_dict(
            mlops_cfg.get("model_registry", {})
        )
        for item in mlops_cfg.get("bootstrap_models", []):
            if not isinstance(item, dict):
                continue
            artifact_path = item.get("artifact_path")
            model_id = item.get("model_id")
            version = item.get("version", "v0")
            if not artifact_path or not model_id:
                continue
            resolved_artifact = Path(artifact_path)
            if not resolved_artifact.is_absolute():
                rel = resolved_artifact
                candidates = [config_path.parent / rel, Path.cwd() / rel]
                for parent in config_path.parents:
                    candidates.append(parent / rel)
                resolved_artifact = None
                for candidate in candidates:
                    if candidate.exists():
                        resolved_artifact = candidate
                        break
            if resolved_artifact is None or not Path(resolved_artifact).exists():
                logging.warning(
                    f"Bootstrap model artifact missing for model_id={model_id}: {artifact_path}"
                )
                continue
            try:
                model_registry.register_model(
                    model_id=model_id,
                    version=version,
                    artifact_path=str(resolved_artifact),
                    metadata=item.get("metadata", {}),
                )
            except Exception as exc:
                logging.warning(f"Bootstrap model registration failed: {exc}")
        canary_cfg = (
            mlops_cfg.get("canary_rollout", {})
            if isinstance(mlops_cfg.get("canary_rollout"), dict)
            else {}
        )
        canary_rollout_manager = CanaryRolloutManager.from_dict(canary_cfg)
        for rollout in canary_cfg.get("rollouts", []):
            if not isinstance(rollout, dict):
                continue
            rollout_id = rollout.get("rollout_id")
            baseline_profile = rollout.get("baseline_profile")
            canary_profile = rollout.get("canary_profile")
            if not rollout_id or not baseline_profile or not canary_profile:
                continue
            canary_rollout_manager.set_rollout(
                rollout_id=rollout_id,
                baseline_profile=baseline_profile,
                canary_profile=canary_profile,
                canary_ratio=rollout.get("canary_ratio", None),
            )
        active_rollout_id = canary_cfg.get("active_rollout", None)

        drift_cfg = (
            mlops_cfg.get("drift_monitor", {})
            if isinstance(mlops_cfg.get("drift_monitor"), dict)
            else {}
        )
        drift_monitor = DriftMonitor(
            class_window=drift_cfg.get("class_window", 200),
            embedding_window=drift_cfg.get("embedding_window", 200),
            alert_window=drift_cfg.get("alert_window", 300),
            class_drift_threshold=drift_cfg.get("class_drift_threshold", 0.12),
            embedding_drift_threshold=drift_cfg.get("embedding_drift_threshold", 0.30),
            alert_z_threshold=drift_cfg.get("alert_z_threshold", 3.0),
        )

        slo_cfg = (
            mlops_cfg.get("slo_monitor", {})
            if isinstance(mlops_cfg.get("slo_monitor"), dict)
            else {}
        )
        slo_monitor = SLOMonitor(
            latency_window=slo_cfg.get("latency_window", 500),
            target_p95_latency_gpu_s=slo_cfg.get("target_p95_latency_gpu_s", 2.0),
            target_p95_latency_cpu_s=slo_cfg.get("target_p95_latency_cpu_s", 4.0),
            error_budget=slo_cfg.get("error_budget", 0.01),
        )
        guardrail_cfg = (
            mlops_cfg.get("guardrails", {})
            if isinstance(mlops_cfg.get("guardrails"), dict)
            else {}
        )
        rollout_guardrail_policy = RolloutGuardrailPolicy.from_dict(guardrail_cfg)

        return cls(
            site_id=payload.get("site_id", "site_demo"),
            camera_id=payload.get("camera_id", "camera_demo"),
            context_field=payload.get("context_field", "ecological_context"),
            taxonomy=taxonomy,
            workflow_service=workflow,
            routing_service=routing,
            ingestion_health_monitor=ingestion_monitor,
            confidence_calibrator=confidence_calibrator,
            entity_federation_service=federation,
            message_bus=message_bus,
            hot_state_store=hot_state_store,
            event_store=event_store,
            clip_store=clip_store,
            rbac_engine=rbac_engine,
            audit_log=audit_log,
            transport_security_config=transport_security_config,
            model_registry=model_registry,
            canary_rollout_manager=canary_rollout_manager,
            rollout_guardrail_policy=rollout_guardrail_policy,
            drift_monitor=drift_monitor,
            slo_monitor=slo_monitor,
            service_account_id=service_account_id,
            rollout_id=active_rollout_id,
            observation_source_default=payload.get("observation_source_default", "detector_tracker"),
        )

    @staticmethod
    def _build_message_bus(cfg):
        cfg = cfg if isinstance(cfg, dict) else {}
        bus_type = str(cfg.get("type", "in_memory")).lower()
        if bus_type == "nats":
            try:
                return NATSJetStreamMessageBus(
                    servers=cfg.get("servers", ["nats://127.0.0.1:4222"]),
                    stream_name=cfg.get("stream_name", "HERMES"),
                    timeout_seconds=cfg.get("timeout_seconds", 2.0),
                )
            except Exception as exc:
                logging.warning(
                    f"Failed to initialize NATS message bus ({exc}). Falling back to in-memory bus."
                )
        return InMemoryMessageBus()

    @staticmethod
    def _build_hot_store(cfg):
        cfg = cfg if isinstance(cfg, dict) else {}
        store_type = str(cfg.get("type", "in_memory")).lower()
        if store_type == "redis":
            try:
                return RedisHotStateStore(
                    url=cfg.get("url", "redis://127.0.0.1:6379/0"),
                    key_prefix=cfg.get("key_prefix", "hermes:soc"),
                )
            except Exception as exc:
                logging.warning(
                    f"Failed to initialize Redis hot store ({exc}). Using in-memory hot store."
                )
        return InMemoryHotStateStore()

    @staticmethod
    def _build_event_store(cfg):
        cfg = cfg if isinstance(cfg, dict) else {}
        store_type = str(cfg.get("type", "in_memory")).lower()
        if store_type == "clickhouse":
            try:
                return ClickHouseEventStore(
                    host=cfg.get("host", "127.0.0.1"),
                    port=cfg.get("port", 8123),
                    username=cfg.get("username", "default"),
                    password=cfg.get("password", ""),
                    database=cfg.get("database", "default"),
                )
            except Exception as exc:
                logging.warning(
                    f"Failed to initialize ClickHouse event store ({exc}). "
                    "Using in-memory event store."
                )
        return InMemoryEventStore(
            max_events_per_table=cfg.get("max_events_per_table", 20000)
        )

    @staticmethod
    def _build_clip_store(cfg, config_path):
        cfg = cfg if isinstance(cfg, dict) else {}
        store_type = str(cfg.get("type", "none")).lower()
        if store_type not in {"filesystem", "fs"}:
            return None
        root_dir = cfg.get("root_dir", "logs/soc_clips")
        root = Path(root_dir)
        if not root.is_absolute():
            rel = root
            candidates = [Path.cwd() / rel, config_path.parent / rel]
            for parent in config_path.parents:
                candidates.append(parent / rel)
            root = candidates[0]
            for candidate in candidates:
                if candidate.parent.exists():
                    root = candidate
                    break
        try:
            return FilesystemClipStore(root_dir=str(root))
        except Exception as exc:
            logging.warning(f"Failed to initialize clip store ({exc}).")
            return None

    def _audit(self, action, details=None, timestamp_utc=None):
        if self.audit_log is None:
            return None
        if not self._is_authorized("audit:write"):
            return None
        try:
            return self.audit_log.append(
                actor=self.service_account_id,
                action=str(action),
                details=details if isinstance(details, dict) else {},
                timestamp_utc=timestamp_utc or utc_now_iso(),
            )
        except Exception as exc:
            logging.warning(f"Audit append failed: {exc}")
            return None

    def _is_authorized(self, action):
        if self.rbac_engine is None:
            return True
        try:
            return bool(
                self.rbac_engine.authorize(
                    account_id=self.service_account_id,
                    action=str(action),
                    site_id=self.site_id,
                )
            )
        except Exception as exc:
            logging.warning(f"RBAC authorize failure for action={action}: {exc}")
            return False

    def _route_publish(self, subject, payload, action):
        if not self._is_authorized(action):
            self._audit(
                action="rbac.denied",
                details={"action": action, "subject": subject},
            )
            return False
        self.routing_service.publish(subject, payload)
        return True

    def _entity_global_id(
        self, entity_id_local, timestamp_utc=None, reid_embedding_ref=None
    ):
        key = (self.site_id, str(entity_id_local))
        if self.entity_federation_service is not None:
            try:
                return self.entity_federation_service.resolve(
                    site_id=self.site_id,
                    camera_id=self.camera_id,
                    entity_id_local=str(entity_id_local),
                    timestamp_utc=timestamp_utc,
                    embedding_ref=reid_embedding_ref,
                )
            except Exception as exc:
                logging.warning(
                    f"Entity federation resolution failed for {entity_id_local}: {exc}. "
                    "Falling back to local global ID cache."
                )
        if key not in self._global_entity_cache:
            self._global_entity_cache[key] = f"{self.site_id}:global:{entity_id_local}"
        return self._global_entity_cache[key]

    def _observation_source(self, result):
        source = result.get("entity_observation_source") or self.observation_source_default
        source = str(source)
        if "schedule" in source:
            return "schedule"
        if "auto_motion" in source:
            return "auto_motion"
        return "detector_tracker"

    @staticmethod
    def _extract_bbox(entity_summary):
        metadata = entity_summary.get("metadata")
        if not isinstance(metadata, dict):
            return None

        if isinstance(metadata.get("bbox_xyxy_norm"), list) and len(metadata["bbox_xyxy_norm"]) == 4:
            x1, y1, x2, y2 = metadata["bbox_xyxy_norm"]
            return {
                "x1": _safe_float(x1),
                "y1": _safe_float(y1),
                "x2": _safe_float(x2),
                "y2": _safe_float(y2),
            }

        if isinstance(metadata.get("bbox_xywh_norm"), list) and len(metadata["bbox_xywh_norm"]) == 4:
            x, y, w, h = metadata["bbox_xywh_norm"]
            return {
                "x": _safe_float(x),
                "y": _safe_float(y),
                "w": _safe_float(w),
                "h": _safe_float(h),
            }
        return None

    def _build_entity_events(self, result, timestamp_utc):
        out = []
        context_label = result.get(self.context_field)
        observation_source = self._observation_source(result)
        entity_sequences = result.get("entity_event_sequences") or []
        entity_lifecycle = result.get("entity_lifecycle") or {}

        for item in entity_sequences:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("entity_id", "__scene__"))
            markov_state = item.get("markov_state") or {}
            reid_embedding_ref = (item.get("metadata") or {}).get("reid_embedding_ref")
            event_id = markov_state.get("event_id") or (
                item.get("observation_state") or {}
            ).get("event_id")
            if not event_id:
                event_id = "unknown_event"

            track_event = EntityTrackEvent(
                event_id=str(event_id),
                timestamp_utc=timestamp_utc,
                site_id=self.site_id,
                camera_id=self.camera_id,
                entity_id_local=entity_id,
                entity_id_global=self._entity_global_id(
                    entity_id,
                    timestamp_utc=timestamp_utc,
                    reid_embedding_ref=reid_embedding_ref,
                ),
                bbox=self._extract_bbox(item),
                track_confidence=_safe_float(markov_state.get("prob", 0.0), 0.0),
                reid_embedding_ref=reid_embedding_ref,
                lifecycle_state=str(item.get("lifecycle_state", "continued")),
                context_label=str(context_label) if context_label is not None else None,
                context_confidence=1.0 if context_label is not None else 0.0,
                observation_source=observation_source,
                entity_event_sequences=item.get("event_sequence"),
                entity_lifecycle=entity_lifecycle,
                metadata={
                    "window_step": item.get("window_step"),
                    "sequence_length": item.get("sequence_length"),
                    "entity_status": item.get("entity_status"),
                },
            )
            out.append(track_event)

        return out

    def _build_threat_events(self, result, entity_track_events):
        out = []
        entity_by_id = {}
        for item in result.get("entity_event_sequences") or []:
            if isinstance(item, dict):
                entity_by_id[str(item.get("entity_id", "__scene__"))] = item

        for event in entity_track_events:
            entity_summary = entity_by_id.get(event.entity_id_local, {})
            markov_state = entity_summary.get("markov_state") or {}
            event_predictions = entity_summary.get("event_predictions") or []
            event_label = None
            if event_predictions and isinstance(event_predictions, list):
                event_label = (event_predictions[0] or {}).get("label")

            threat_cls = self.taxonomy.classify(
                event_id=event.event_id,
                event_label=event_label,
                base_confidence=event.track_confidence,
            )

            anomaly_score, explanations = self.anomaly_scorer.score(
                context_label=event.context_label,
                entity_summary=entity_summary,
                markov_state=markov_state,
                reid_embedding_ref=event.reid_embedding_ref,
            )

            confidence = _clamp01(
                0.65 * _safe_float(threat_cls.get("base_confidence", 0.0), 0.0)
                + 0.35 * float(anomaly_score)
            )
            confidence = self.confidence_calibrator.calibrate(confidence)
            if not self.taxonomy.should_emit_candidate(confidence):
                continue

            clip_ref = None
            if self.clip_store is not None and self._is_authorized("store:clip"):
                try:
                    clip_id = (
                        f"{self.site_id}_{self.camera_id}_"
                        f"{event.entity_id_local}_{event.timestamp_utc}"
                    ).replace(":", "_")
                    clip_ref = self.clip_store.store_clip(
                        clip_id=clip_id,
                        clip_payload={
                            "event_id": event.event_id,
                            "entity_id_local": event.entity_id_local,
                            "timestamp_utc": event.timestamp_utc,
                        },
                        metadata={"site_id": self.site_id, "camera_id": self.camera_id},
                    )
                except Exception as exc:
                    logging.warning(f"Clip store write failed: {exc}")
            elif self.clip_store is not None:
                self._audit(
                    action="rbac.denied",
                    details={"action": "store:clip", "entity_id": event.entity_id_local},
                )

            threat = ThreatEvent(
                threat_type=threat_cls.get("threat_type", "anomalous_behavior"),
                severity=threat_cls.get("severity", "medium"),
                confidence_calibrated=confidence,
                entity_refs=[event.entity_id_global or event.entity_id_local],
                camera_refs=[self.camera_id],
                clip_ref=clip_ref,
                markov_state=markov_state.get("event_id"),
                anomaly_score=anomaly_score,
                fusion_score=0.0,
                policy_action=threat_cls.get("policy_action", "review_required"),
                explanations=[
                    f"taxonomy_source={threat_cls.get('source', 'unknown')}",
                ]
                + explanations,
                timestamp_utc=event.timestamp_utc,
                site_id=self.site_id,
                metadata={
                    "entity_id_local": event.entity_id_local,
                    "track_confidence": event.track_confidence,
                },
            )

            fused = self.fusion_service.fuse(threat)
            if isinstance(fused, dict):
                threat.fusion_score = _clamp01(fused.get("fusion_score", 0.0))
                threat.metadata["incident_id"] = fused.get("incident_id")
                threat.metadata["incident_event_count"] = fused.get("event_count")
                # Escalate if corroborated strongly across events/entities/cameras.
                if threat.fusion_score >= 0.8 and threat.policy_action == "review_required":
                    threat.policy_action = "escalate_level_1"

            out.append(threat)

        return out

    def _store_event(self, table, event, action="store:event"):
        if not self._is_authorized(action):
            self._audit(
                action="rbac.denied",
                details={"action": action, "table": table},
            )
            return False
        if self.event_store is None:
            return False
        try:
            return bool(self.event_store.append(table, event))
        except Exception as exc:
            logging.warning(f"Event store append failed for {table}: {exc}")
            return False

    def _set_hot_state(self, key, value, ttl_seconds=None, action="store:hot_state"):
        if not self._is_authorized(action):
            self._audit(
                action="rbac.denied",
                details={"action": action, "key": key},
            )
            return False
        if self.hot_state_store is None:
            return False
        try:
            return bool(self.hot_state_store.set(key, value, ttl_seconds=ttl_seconds))
        except Exception as exc:
            logging.warning(f"Hot store update failed for key={key}: {exc}")
            return False

    def _publish_to_bus(self, subject, payload, action="emit:bus_publish"):
        if not self._is_authorized(action):
            self._audit(
                action="rbac.denied",
                details={"action": action, "subject": subject},
            )
            return None
        if self.message_bus is None:
            return None
        try:
            return self.message_bus.publish(subject=subject, payload=payload)
        except Exception as exc:
            logging.warning(f"Message bus publish failed for {subject}: {exc}")
            return None

    def _evaluate_rollout_guardrails(self, drift_metrics, slo_metrics, timestamp_utc):
        if self.rollout_guardrail_policy is None:
            return {}

        decision = self.rollout_guardrail_policy.evaluate(
            drift_metrics=drift_metrics,
            slo_metrics=slo_metrics,
        )
        guardrails = dict(decision)
        guardrails["rollout_id"] = self.rollout_id
        guardrails["active_profile_before"] = (
            self.active_profile.profile_id if self.active_profile is not None else None
        )
        guardrails["rolled_back"] = False
        guardrails["rollout_active"] = False

        if not self.rollout_id or self.canary_rollout_manager is None:
            guardrails["active_profile_after"] = guardrails["active_profile_before"]
            return guardrails

        rollout_cfg = self.canary_rollout_manager.rollouts.get(str(self.rollout_id))
        if not isinstance(rollout_cfg, dict):
            guardrails["active_profile_after"] = guardrails["active_profile_before"]
            return guardrails

        guardrails["rollout_active"] = bool(rollout_cfg.get("active", False))
        if decision.get("triggered") and bool(rollout_cfg.get("active", False)):
            rollback_cfg = self.canary_rollout_manager.rollback(self.rollout_id)
            baseline_profile_id = None
            if isinstance(rollback_cfg, dict):
                baseline_profile_id = rollback_cfg.get("baseline_profile")

            if (
                baseline_profile_id
                and self.profile_service is not None
                and baseline_profile_id in getattr(self.profile_service, "profiles", {})
            ):
                self.active_profile = self.profile_service.profiles[baseline_profile_id]

            reason = "|".join(decision.get("alarm_reasons", []))
            history_item = self.rollout_guardrail_policy.record_rollback(
                rollout_id=self.rollout_id,
                reason=reason,
                timestamp_utc=timestamp_utc,
            )
            rollback_event = {
                "timestamp_utc": timestamp_utc,
                "site_id": self.site_id,
                "camera_id": self.camera_id,
                "rollout_id": self.rollout_id,
                "baseline_profile": baseline_profile_id,
                "active_profile_after": (
                    self.active_profile.profile_id
                    if self.active_profile is not None
                    else baseline_profile_id
                ),
                "alarm_reasons": list(decision.get("alarm_reasons", [])),
                "history": history_item,
            }
            self._store_event("mlops_rollout_events", rollback_event)
            self._set_hot_state(
                key=f"mlops:rollout:{self.rollout_id}",
                value=rollback_event,
                ttl_seconds=86400,
            )
            self._audit(
                action="mlops.rollback",
                details=rollback_event,
                timestamp_utc=timestamp_utc,
            )
            guardrails["rolled_back"] = True
            guardrails["rollback_event"] = rollback_event
            guardrails["rollout_active"] = False

        guardrails["active_profile_after"] = (
            self.active_profile.profile_id if self.active_profile is not None else None
        )
        return guardrails

    def process_result(self, result):
        if not isinstance(result, dict):
            return {}

        started_at = time.perf_counter()
        timestamp_utc = result.get("timestamp_utc") or utc_now_iso()
        self._audit(
            action="process.start",
            details={
                "site_id": self.site_id,
                "camera_id": self.camera_id,
                "window_index": result.get("window_index"),
                "frame_index": result.get("frame_index"),
            },
            timestamp_utc=timestamp_utc,
        )

        health = self.ingestion_health_monitor.update(
            site_id=self.site_id,
            camera_id=self.camera_id,
            capture_timestamp_utc=timestamp_utc,
            frame_index=result.get("frame_index"),
        )
        self._set_hot_state(
            key=f"ingestion_health:{self.site_id}:{self.camera_id}",
            value=health,
            ttl_seconds=300,
        )
        self._store_event("ingestion_health_events", health)

        self._route_publish(
            NATS_SUBJECTS["video_obs_raw"],
            {
                "timestamp_utc": timestamp_utc,
                "site_id": self.site_id,
                "camera_id": self.camera_id,
                "window_index": result.get("window_index"),
                "frame_index": result.get("frame_index"),
                "context_label": result.get(self.context_field),
                "event_predictions": result.get("event_predictions", []),
            },
            action="emit:video_obs_raw",
        )

        entity_events = self._build_entity_events(result, timestamp_utc)
        for entity_event in entity_events:
            entity_payload = entity_event.to_dict()
            self._route_publish(
                NATS_SUBJECTS["video_entity_tracks"],
                entity_payload,
                action="emit:entity_track",
            )
            self._store_event("entity_track_events", entity_payload)
            self._set_hot_state(
                key=f"entity:{entity_event.entity_id_global}",
                value=entity_payload,
                ttl_seconds=600,
            )

        top_state = result.get("markov_state") or {}
        if top_state:
            posterior_payload = {
                "timestamp_utc": timestamp_utc,
                "site_id": self.site_id,
                "camera_id": self.camera_id,
                "context_label": result.get(self.context_field),
                "markov_state": top_state,
                "markov_posterior": result.get("markov_posterior"),
            }
            self._route_publish(
                NATS_SUBJECTS["video_event_posterior"],
                posterior_payload,
                action="emit:event_posterior",
            )
            self._store_event("video_event_posterior", posterior_payload)

        threat_events = self._build_threat_events(result, entity_events)
        case_updates = []
        for threat in threat_events:
            threat_payload = threat.to_dict()
            self._route_publish(
                NATS_SUBJECTS["threat_alert_candidate"],
                threat_payload,
                action="emit:threat_candidate",
            )
            self._store_event("threat_events", threat_payload)
            case = self.workflow_service.open_candidate_case(threat)
            if case is not None:
                case_updates.append(case)
                self._route_publish(
                    NATS_SUBJECTS["soc_case_updates"],
                    case,
                    action="emit:case_update",
                )
                self._store_event("soc_case_updates", case)
                self._set_hot_state(
                    key=f"case:{case.get('case_id')}",
                    value=case,
                    ttl_seconds=86400,
                )

        drift_metrics = {}
        if self.drift_monitor is not None and self._is_authorized("monitor:drift"):
            drift_metrics = self.drift_monitor.update(
                class_event_ids=[x.event_id for x in entity_events],
                embedding_refs=[
                    x.reid_embedding_ref
                    for x in entity_events
                    if x.reid_embedding_ref is not None
                ],
                alert_count=len(threat_events),
            )
            self._store_event("drift_metrics", drift_metrics)

        dispatched = self.routing_service.dispatch_step(max_dispatch=64)
        bus_publish_results = []
        for message in dispatched:
            if not isinstance(message, dict):
                continue
            bus_result = self._publish_to_bus(
                subject=message.get("subject"),
                payload=message.get("payload", {}),
            )
            if bus_result is not None:
                bus_publish_results.append(bus_result)
            self._store_event("dispatch_log", message)

        processing_seconds = max(0.0, time.perf_counter() - started_at)
        slo_metrics = {}
        if self.slo_monitor is not None and self._is_authorized("monitor:slo"):
            slo_metrics = self.slo_monitor.record(
                processing_seconds=processing_seconds,
                success=True,
                profile_id=self.active_profile.profile_id,
            )
            self._store_event("slo_metrics", slo_metrics)

        rollout_guardrails = self._evaluate_rollout_guardrails(
            drift_metrics=drift_metrics,
            slo_metrics=slo_metrics,
            timestamp_utc=timestamp_utc,
        )

        self._audit(
            action="process.end",
            details={
                "processing_seconds": processing_seconds,
                "entity_event_count": len(entity_events),
                "threat_event_count": len(threat_events),
                "case_update_count": len(case_updates),
            },
            timestamp_utc=timestamp_utc,
        )

        return {
            "profile": self.active_profile.to_dict(),
            "hardware": dict(self.hardware_snapshot),
            "processing_seconds": processing_seconds,
            "ingestion_health": health,
            "entity_track_events": [x.to_dict() for x in entity_events],
            "threat_events": [x.to_dict() for x in threat_events],
            "case_updates": case_updates,
            "routing_metrics": self.routing_service.metrics(),
            "routing_dispatched": dispatched,
            "message_bus_publish_results": bus_publish_results,
            "dead_letter": self.routing_service.dead_letter_snapshot(limit=10),
            "workflow_snapshot": self.workflow_service.snapshot(),
            "sla_breaches": self.workflow_service.sla_breaches(),
            "drift_metrics": drift_metrics,
            "slo_metrics": slo_metrics,
            "rollout_guardrails": rollout_guardrails,
            "hot_store_stats": (
                self.hot_state_store.stats() if self.hot_state_store is not None else {}
            ),
            "event_store_stats": (
                self.event_store.stats() if self.event_store is not None else {}
            ),
            "message_bus_stats": (
                self.message_bus.stats()
                if self.message_bus is not None and hasattr(self.message_bus, "stats")
                else {}
            ),
            "entity_federation_snapshot": (
                self.entity_federation_service.snapshot()
                if self.entity_federation_service is not None
                else {}
            ),
            "confidence_calibration": self.confidence_calibrator.to_dict(),
            "security": {
                "service_account_id": self.service_account_id,
                "rbac": self.rbac_engine.snapshot()
                if self.rbac_engine is not None
                else {},
                "transport": dict(self.transport_security_status),
                "audit_integrity": self.audit_log.verify()
                if self.audit_log is not None
                else {},
            },
            "mlops": {
                "model_registry": self.model_registry.snapshot()
                if self.model_registry is not None
                else {},
                "canary_rollouts": self.canary_rollout_manager.snapshot()
                if self.canary_rollout_manager is not None
                else {},
                "guardrails": (
                    self.rollout_guardrail_policy.snapshot()
                    if self.rollout_guardrail_policy is not None
                    else {}
                ),
            },
        }
