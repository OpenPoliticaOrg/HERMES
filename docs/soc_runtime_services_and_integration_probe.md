# SOC Runtime Services and Integration Probe

## Purpose
This document describes:
- executable service handlers aligned to the planned SOC gRPC interfaces
- integration probe workflow for external backends (NATS/Redis/ClickHouse/transport files)

## Runtime service handlers
Implementation:
- `lavis/common/soc/runtime_services.py`

Service suite:
- `SOCRuntimeServiceSuite`
  - `ingest_gateway` (`IngestGatewayRuntimeService`)
  - `inference_profile` (`InferenceProfileRuntimeService`)
  - `entity_fusion` (`EntityFusionRuntimeService`)
  - `threat_scoring` (`ThreatScoringRuntimeService`)
  - `alert_dispatch` (`AlertDispatchRuntimeService`)
  - `feedback_ingest` (`FeedbackIngestRuntimeService`)

Notes:
- Interfaces align with `docs/proto/hermes_soc_services.proto`.
- The suite is local/in-process by default (no long-running gRPC server required).
- These handlers operate directly on `SOCOrchestrator` state.

## gRPC runtime server
Implementation:
- `tools/soc_grpc_server.py`

Run:
```bash
bash run_scripts/soc/grpc_server.sh --host 127.0.0.1 --port 50051
```

Design:
- proto Python messages are generated at runtime using `protoc` to `logs/soc_proto_gen`
- generic gRPC method handlers are registered per service
- no `grpcio-tools` dependency is required for server operation

Smoke:
```bash
bash run_scripts/soc/grpc_smoke.sh --print-json
```

## Guardrail behavior surfaced by services/runtime
- MLOps rollout guardrails evaluate sustained drift/SLO alarm windows.
- On trigger, active canary rollout is rolled back to baseline profile.
- Rollback metadata is emitted in `rollout_guardrails` and persisted/audited.

## Smoke test
Run:
```bash
bash run_scripts/soc/services_test.sh
```

Coverage:
- profile resolution
- ingest-to-threat flow
- entity fusion/global ID continuity
- threat scoring
- alert dispatch
- analyst feedback ingest

## Integration probe
Implementation:
- `tools/soc_integration_probe.py`
- wrapper: `run_scripts/soc/integration_probe.sh`

Run:
```bash
bash run_scripts/soc/integration_probe.sh --json
```

Probe behavior:
- If backend type is configured as in-memory, the check is skipped.
- If backend type is configured as external, the check is required:
  - NATS: connect + flush
  - Redis: ping
  - ClickHouse: `SELECT 1`
  - filesystem clip store: write test
  - transport security paths: certificate/key/CA existence

Pass criteria:
- all required configured checks pass

## Recommended operator flow
1. `bash run_scripts/soc/doctor.sh`
2. `bash run_scripts/soc/integration_probe.sh --json`
3. `bash run_scripts/soc/services_test.sh`
4. `bash run_scripts/soc/live_viz.sh 0 warehouse_entry`
