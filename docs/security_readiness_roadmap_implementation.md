# Real-World Security Readiness Roadmap: Phase-1 Implementation

This document maps the roadmap to concrete code added in this repository.

## Implemented Foundation (Now)

## Core architecture
- **Edge-agent compatible ingestion abstraction**
  - `lavis/common/soc/interop.py`
  - ONVIF discovery/profile sync service + VMS-style interfaces:
    - `CameraSourceProvider`
    - `ClipExporter`
    - `PTZControlProxy` (read-only)
- **Kubernetes-core compatible service layer contracts**
  - `docs/proto/hermes_soc_services.proto`
  - service interfaces: `IngestGatewayService`, `InferenceProfileService`, `EntityFusionService`, `ThreatScoringService`, `AlertDispatchService`, `FeedbackIngestService`
  - concrete local handlers: `lavis/common/soc/runtime_services.py` (`SOCRuntimeServiceSuite`)
- **Message fabric subject conventions (NATS JetStream aligned)**
  - `lavis/common/soc/routing.py`
  - `lavis/common/soc/message_bus.py`
  - subjects:
    - `video.obs.raw`
    - `video.entity.tracks`
    - `video.event.posterior`
    - `threat.alert.candidate`
    - `threat.alert.confirmed`
    - `soc.case.updates`

## Canonical schemas
- `lavis/common/soc/schemas.py`
- `EntityTrackEvent` fields:
  - `event_id`, `timestamp_utc`, `site_id`, `camera_id`
  - `entity_id_local`, `entity_id_global`
  - `bbox`, `track_confidence`, `reid_embedding_ref`
  - `lifecycle_state` (`entered|continued|reentered|exited`)
  - `context_label`, `context_confidence`
  - `observation_source` (`schedule|auto_motion|detector_tracker`)
- `ThreatEvent` fields:
  - `threat_type`, `severity`, `confidence_calibrated`
  - `entity_refs[]`, `camera_refs[]`, `clip_ref`
  - `markov_state`, `anomaly_score`, `fusion_score`
  - `policy_action` (`review_required|escalate_level_1|escalate_level_2`)
  - `explanations[]`

## Perception/runtime profile routing
- `lavis/common/soc/profiles.py`
- dynamic routing implemented:
  - `edge_gpu_profile`
  - `edge_cpu_profile`
  - `core_gpu_profile`
  - fallback: `auto_motion_fallback`

## Threat + anomaly intelligence
- `lavis/common/soc/threat_intel.py`
- `lavis/common/soc/calibration.py`
- implemented:
  - Threat taxonomy v2 (`ThreatTaxonomyV2`) with event mapping + keyword fallback
  - Hybrid anomaly scorer (`HybridAnomalyScorer`) combining:
    - Markov surprise
    - context-open-set novelty (`EmbeddingNoveltyModel`)
    - temporal rules (`TemporalRuleEngine`)
  - Confidence calibration layer (`ConfidenceCalibrator`) with:
    - identity
    - temperature scaling
    - isotonic-table mapping
  - Incident fusion (`IncidentFusionService`) with corroboration-aware `fusion_score`

## Metadata federation and storage integration
- `lavis/common/soc/federation.py`
- `lavis/common/soc/stores.py`
- implemented:
  - cross-camera/global entity continuity with `EntityFederationService`
  - hot-state adapters:
    - `InMemoryHotStateStore`
    - `RedisHotStateStore` (optional external)
  - event store adapters:
    - `InMemoryEventStore`
    - `ClickHouseEventStore` (optional external)
  - filesystem clip persistence via `FilesystemClipStore`

## Security, governance, and MLOps reliability
- `lavis/common/soc/security.py`
- `lavis/common/soc/mlops.py`
- implemented:
  - RBAC policy engine with site-scoped service-account authorization
  - immutable hash-chained audit log integrity checks
  - transport security config validation hooks (mTLS/TLS paths)
  - signed model registry (`SignedModelRegistry`) for artifact integrity metadata
  - canary rollout manager (`CanaryRolloutManager`) for staged profile assignment
  - sustained-alarm rollout guardrails (`RolloutGuardrailPolicy`) with rollback history
  - drift monitor (`DriftMonitor`) for:
    - class prior drift
    - embedding drift
    - alert volume anomalies
  - SLO monitor (`SLOMonitor`) for p95 latency and availability budget checks

## Coordination/message passing runtime policy
- `lavis/common/soc/routing.py`
- implemented:
  - subject-priority queue routing
  - congestion-aware dropping of low-priority channels
  - retry and dead-letter queue handling
  - per-subject backlog and dispatch counters

## Human-in-loop SOC workflow
- `lavis/common/soc/workflow.py`
- implemented:
  - states: `candidate -> analyst_review -> confirmed|dismissed`
  - acknowledgment tracking
  - severity-SLA breach checks
  - runbook binding by severity
  - feedback ingest
  - immutable audit log entries for state and analyst actions

## Orchestrated runtime integration
- `lavis/common/soc/runtime.py`
- integrated into live stream in:
  - `stream_online.py`
- new stream flags:
  - `--soc-enable`
  - `--soc-config`
  - `--soc-site-id`
  - `--soc-camera-id`
  - `--soc-threat-taxonomy-path`
  - `--soc-print-routing`
- emitted output fields per window:
  - `soc_profile`
  - `soc_hardware`
  - `soc_ingestion_health`
  - `soc_entity_track_events`
  - `soc_threat_events`
  - `soc_case_updates`
  - `soc_routing_metrics`
  - `soc_routing_dispatched`
  - `soc_message_bus_publish_results`
  - `soc_message_bus_stats`
  - `soc_dead_letter`
  - `soc_sla_breaches`
  - `soc_processing_seconds`
  - `soc_drift_metrics`
  - `soc_slo_metrics`
  - `soc_rollout_guardrails`
  - `soc_hot_store_stats`
  - `soc_event_store_stats`
  - `soc_entity_federation`
  - `soc_confidence_calibration`
  - `soc_security`
  - `soc_mlops`

## Setup/check/test assets
- Example SOC data/config:
  - `data/soc/example_threat_taxonomy_v2.json`
  - `data/soc/example_soc_runtime_config.json`
  - `data/soc/example_onvif_inventory.json`
- Smoke test:
  - `tools/soc_readiness_smoke.py`
- Setup doctor:
  - `tools/soc_readiness_doctor.py`
- Multi-feed coordination simulator:
  - `tools/soc_coordination_sim.py`
- Security + MLOps smoke test:
  - `tools/soc_mlops_smoke.py`
- Runtime service-handler smoke test:
  - `tools/soc_services_smoke.py`
- External integration probe:
  - `tools/soc_integration_probe.py`
- gRPC runtime server + smoke:
  - `tools/soc_grpc_server.py`
  - `tools/soc_grpc_smoke.py`
- Runtime service + probe docs:
  - `docs/soc_runtime_services_and_integration_probe.md`
- Run scripts:
  - `run_scripts/soc/test.sh`
  - `run_scripts/soc/doctor.sh`
  - `run_scripts/soc/live_viz.sh`
  - `run_scripts/soc/coordination_test.sh`
  - `run_scripts/soc/mlops_test.sh`
  - `run_scripts/soc/services_test.sh`
  - `run_scripts/soc/integration_probe.sh`
  - `run_scripts/soc/grpc_server.sh`
  - `run_scripts/soc/grpc_smoke.sh`

## What is intentionally phase-1 (not fully production-complete yet)
- ONVIF/VMS adapters are interface-level + static inventory sync stubs, not full vendor SDK coverage.
- Runtime routing is in-process policy logic; external NATS/JetStream plumbing remains deployment wiring.
- Redis/ClickHouse/object store integration is contract-level and runtime-ready but not hard-wired in these scripts.
- gRPC `.proto` and concrete service handlers are implemented, but no long-running gRPC server process is bundled in this patch.
- Full detector/tracker/ReID model stack selection is profile-routed; concrete model binaries and canary rollout tooling are not bundled in this patch.
- Autonomous high-impact actuation remains intentionally absent; workflow is analyst-first.

## User Flow (Analyst + Operator)
1. Operator starts live stream with SOC enabled.
2. Each window updates classification + context-conditioned Markov posterior.
3. Entity lifecycle is tracked (`entered/reentered/exited`) and canonical `EntityTrackEvent`s are emitted.
4. Threat candidates are generated with calibrated confidence and anomaly/fusion scores.
5. Candidate alerts become SOC cases (`candidate`) and route through analyst review.
6. Analyst acknowledges, confirms, or dismisses; all actions are audited.
7. Feedback is ingested for future active-learning/model tuning loops.

## Quick Start
```bash
# Dependency + environment check + smoke
bash run_scripts/soc/doctor.sh

# SOC smoke test JSON report
bash run_scripts/soc/test.sh

# Live visualization run
bash run_scripts/soc/live_viz.sh 0 warehouse_entry

# Multi-node coordination + message-passing KPI run
bash run_scripts/soc/coordination_test.sh

# Security + MLOps primitives run
bash run_scripts/soc/mlops_test.sh

# Runtime service handlers (gRPC-aligned local services) smoke test
bash run_scripts/soc/services_test.sh

# Probe configured external integrations (NATS/Redis/ClickHouse/transport files)
bash run_scripts/soc/integration_probe.sh --json

# Run gRPC runtime server
bash run_scripts/soc/grpc_server.sh --host 127.0.0.1 --port 50051

# Validate gRPC endpoints end-to-end
bash run_scripts/soc/grpc_smoke.sh --print-json
```

## Acceptance-test coverage included in smoke
- Canonical schema emission for entity tracks
- Threat candidate generation and case creation
- Analyst workflow transitions + feedback ingest
- Congestion handling preserving critical channel dispatch
- ONVIF discovery/profile sync sanity
- Message-bus publish path, hot/event store updates, and clip persistence
- Entity federation global-ID continuity checks
- RBAC/audit integrity, drift/SLO metrics, and model-registry bootstrap checks
- Rollout guardrail decisions and rollback metadata emission
- Runtime service-handler flow (ingest/profile/fusion/threat/dispatch/feedback)
- External integration probe for configured production adapters
- gRPC service endpoint flow over network channel
