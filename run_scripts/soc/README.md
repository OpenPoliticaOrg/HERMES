# SOC Security Readiness Quick Run

## 1) Doctor (dependency + smoke checks)
```bash
bash run_scripts/soc/doctor.sh
```

## 2) Smoke test
```bash
bash run_scripts/soc/test.sh
```

## 3) Live run with visualization
```bash
bash run_scripts/soc/live_viz.sh 0 warehouse_entry
```

## 4) Multi-feed coordination + message passing test
```bash
bash run_scripts/soc/coordination_test.sh
```

## 5) Security + MLOps primitive test
```bash
bash run_scripts/soc/mlops_test.sh
```

## 6) Runtime service handler smoke test
```bash
bash run_scripts/soc/services_test.sh
```

## 7) External integration probe
```bash
bash run_scripts/soc/integration_probe.sh --json
```

## 8) gRPC runtime server
```bash
bash run_scripts/soc/grpc_server.sh --host 127.0.0.1 --port 50051
```

## 9) gRPC smoke test
```bash
bash run_scripts/soc/grpc_smoke.sh --print-json
```

## 10) Minimal web dashboard client
```bash
# one command: spawn gRPC server + open dashboard stream
bash run_scripts/soc/dashboard.sh --spawn-grpc-server --demo-stream
```

Dashboard analyst controls:
- set analyst id in the input box
- per-case actions: `Ack`, `Confirm`, `Dismiss` (executed via gRPC case-management endpoints)

## 11) Dashboard smoke test
```bash
bash run_scripts/soc/dashboard_smoke.sh --print-json
```

Arguments for `coordination_test.sh`:
1. `steps` (default `24`)
2. `cameras` (default `cam_a01,cam_a02,cam_a03`)
3. `output_json` (default `logs/soc_coordination/summary.json`)
4. `soc_config` (default `data/soc/example_soc_runtime_config.json`)

Arguments for `live_viz.sh`:
1. `video_source` (default `0`)
2. `ecological_context` (default `warehouse_entry`)
3. `question` (default `what is the activity in the video?`)
4. `sequence_id` (default `cam_soc0`)
5. `output_jsonl` (default `logs/soc_live_viz.jsonl`)
6. `checkpoint_path` (optional)
7. `soc_config` (default `data/soc/example_soc_runtime_config.json`)

Live SOC output fields added to each JSONL row:
- `soc_profile`
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
