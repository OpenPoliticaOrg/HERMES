# SOC Coordination and Message-Passing Testing

This guide validates coordination across multiple surveillance feed nodes.

## What this test measures
- Cross-camera entity continuity (`cross_camera_handoff_consistency`)
- Critical-channel preservation under congestion
- p95 SOC processing latency per window
- Per-subject message volume on the shared bus
- Shared hot/event store updates and entity federation footprint

## Run
```bash
bash run_scripts/soc/coordination_test.sh
```

Optional arguments:
```bash
bash run_scripts/soc/coordination_test.sh \
  36 \
  cam_a01,cam_a02,cam_a03,cam_a04 \
  logs/soc_coordination/custom_summary.json \
  data/soc/example_soc_runtime_config.json
```

## Output
Default report path:
- `logs/soc_coordination/summary.json`

Important fields:
- `coordination_kpis.cross_camera_handoff_consistency`
- `coordination_kpis.critical_channel_preserved_under_congestion`
- `coordination_kpis.p95_processing_latency_ms`
- `coordination_kpis.message_subject_counts`
- `store_stats.entity_federation`

## Interpreting results
- `cross_camera_handoff_consistency` near `1.0` means global IDs remain stable during handoffs.
- `critical_channel_preserved_under_congestion=true` means threat-critical routing was prioritized correctly.
- If `p95_processing_latency_ms` grows unexpectedly, inspect model/profile selection and queue backlog.

## Related files
- `tools/soc_coordination_sim.py`
- `run_scripts/soc/coordination_test.sh`
- `lavis/common/soc/federation.py`
- `lavis/common/soc/routing.py`
- `lavis/common/soc/runtime.py`
