# SOC Runtime Algorithm Specs

## 1) Context-conditioned Markov baseline
Given Markov posterior max probability `p_markov_max`, anomaly component:

`markov_surprise = 1 - p_markov_max`

## 2) Open-set novelty (per ecological context)
For each context `c`, maintain historical scalarized embedding features.

`z = |x - mean_c| / std_c`

Novelty probability:

`novelty = 1 - exp(-0.5 * z)`

## 3) Temporal rule score
Rule features include:
- loitering windows exceeding threshold
- re-entry pattern
- high Markov uncertainty

Temporal score is clamped to `[0,1]` from weighted rule contributions.

## 4) Hybrid anomaly score

`anomaly = w_m * markov_surprise + w_n * novelty + w_t * temporal`

Where default normalized weights are:
- `w_m = 0.45`
- `w_n = 0.35`
- `w_t = 0.20`

## 5) Calibrated threat confidence
From taxonomy base confidence `p_tax` and anomaly score `a`:

`confidence_calibrated = clamp(0.65 * p_tax + 0.35 * a)`

Candidate alerts are emitted if:

`confidence_calibrated >= minimum_candidate_confidence`

Optional calibration layer:
- `identity`: no change
- `temperature`: `sigmoid(logit(p) / T)`
- `isotonic`: piecewise monotonic mapping from reliability table

## 6) Incident fusion score
Incidents are merged by `(site_id, threat_type, time_bucket, entity_anchor)`.

Fusion score:

`fusion = clamp(0.55 * conf_max + 0.25 * anomaly_max + 0.20 * corroboration)`

Corroboration grows with:
- number of unique cameras
- number of unique entities
- number of supporting events

## 7) Routing/backpressure policy
Messages are prioritized by subject class. Under congestion:
- low-priority queues are dropped first
- critical channels are preserved
- delivery failures enter retry queue
- retry-exceeded messages enter dead-letter queue

## 8) Workflow state machine
States:
- `candidate`
- `analyst_review`
- `confirmed`
- `dismissed`

Transitions:
- `candidate -> analyst_review` (ack)
- `candidate/analyst_review -> confirmed|dismissed`

SLA breach detection:
- alert if non-terminal case elapsed time exceeds severity SLA target.

## 9) Cross-camera entity federation
For each local entity observation, assign global ID by:
1. preserving existing local-to-global link when available
2. otherwise matching recent global entities by embedding-reference similarity and time proximity
3. creating a new global entity when no match meets threshold

Matching constraints:
- `|t_current - t_last_seen| <= max_time_delta_seconds`
- `sim(embedding_ref_current, embedding_ref_last) >= min_embedding_similarity`

## 10) Coordination KPI definitions
- `cross_camera_handoff_consistency`:
  - fraction of cross-camera handoff stages where the same physical entity kept the same global ID
- `critical_channel_preserved_under_congestion`:
  - whether at least one critical alert message was dispatched despite queue congestion
- `p95_processing_latency_ms`:
  - p95 processing time for per-window orchestrator updates

## 11) Drift monitoring
Drift monitor outputs:
- `class_prior_drift`:
  - Jensen-Shannon-style divergence between baseline and recent class-event distributions
- `embedding_drift`:
  - absolute delta between baseline and current embedding-hash means
- `alert_volume_zscore`:
  - z-score of current alert count against rolling alert window statistics

Alarm triggers:
- `class_prior_drift >= class_drift_threshold`
- `embedding_drift >= embedding_drift_threshold`
- `|alert_volume_zscore| >= alert_z_threshold`

## 12) SLO monitoring
Per-window SOC processing is tracked with:
- `latency_p95_s`
- `availability`

Targets:
- GPU profile target p95 <= `2.0s`
- CPU profile target p95 <= `4.0s`
- availability target >= `1 - error_budget`

## 13) Security and governance checks
- RBAC authorization evaluated per action and site scope.
- Unauthorized actions generate `rbac.denied` audit entries.
- Audit log integrity is verified with hash chaining:
  - each entry stores `prev_hash`
  - `entry_hash = sha256(serialized_entry_without_hash)`

## 14) Canary rollout guardrails and rollback
Guardrail policy evaluates sustained alarms over windows:

`trigger = (alarm_reasons != empty) AND (consecutive_alarm_hits >= required_hits)`

Alarm reasons can include:
- drift alarms: `class_prior_drift_alarm`, `embedding_drift_alarm`, `alert_volume_alarm`
- SLO alarms (after sample gate): `latency_alarm`, `availability_alarm`

If `trigger=true` and rollout is active:
- rollout is marked inactive (`rollback`)
- active profile is forced to rollout baseline profile
- rollback event is appended to event store + hot state
- immutable audit entry `mlops.rollback` is recorded
