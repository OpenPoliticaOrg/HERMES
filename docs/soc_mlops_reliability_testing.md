# SOC MLOps and Reliability Testing

This document covers testing for:
- signed model registry
- canary rollout assignment
- drift/SLO monitors
- RBAC and immutable audit integrity

## 1) Run primitive smoke tests
```bash
bash run_scripts/soc/mlops_test.sh
```

Checks performed:
- model artifact registration + signature verification
- deterministic canary assignment with non-degenerate split
- drift monitor output validity
- SLO monitor p95/availability output validity
- RBAC allow/deny behavior
- immutable audit chain integrity

## 2) Run full SOC doctor (includes MLOps smoke)
```bash
bash run_scripts/soc/doctor.sh
```

## 3) Verify runtime payload fields
When SOC runtime is enabled (`stream_online.py --soc-enable`), verify fields:
- `soc_drift_metrics`
- `soc_slo_metrics`
- `soc_security`
- `soc_mlops`
- `soc_processing_seconds`

## 4) Canary rollout behavior validation
Canary resolution is configured under `mlops.canary_rollout` in:
- `data/soc/example_soc_runtime_config.json`

Expected behavior:
- `active_rollout` chooses baseline vs canary profile by deterministic hash of site/camera.
- changing `canary_ratio` adjusts canary coverage.

## 5) Model registry bootstrap validation
Bootstrap model entries are configured under `mlops.bootstrap_models`.
For each entry:
- artifact hash is computed
- signature is generated
- entry appears in `soc_mlops.model_registry` output

## 6) Rollout guardrails (implemented)
Guardrail output is emitted in:
- `soc_rollout_guardrails`

Rollback trigger logic (default config):
- requires sustained alarms across `consecutive_alarm_windows`
- evaluates drift alarms immediately
- evaluates SLO alarms only after `min_samples_for_slo` gate

When triggered:
- active canary rollout is rolled back to baseline profile
- rollback event is persisted and audited (`mlops.rollback`)

## Related files
- `lavis/common/soc/mlops.py`
- `lavis/common/soc/security.py`
- `tools/soc_mlops_smoke.py`
- `run_scripts/soc/mlops_test.sh`
