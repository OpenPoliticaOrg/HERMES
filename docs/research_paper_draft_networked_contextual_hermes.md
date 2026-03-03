# Networked Context-Conditional Event Inference and Coordination for Multi-Feed Surveillance

**Draft manuscript for repository extension work (internal preprint)**

## Abstract

We present a practical extension of HERMES for real-time, multi-feed surveillance analysis by coupling context-aware event inference with network-level message-passing optimization. The system combines (i) dynamic event candidate routing, (ii) pluggable observation classifiers, (iii) context-conditional inhomogeneous Markov filtering with sliding-window updates, (iv) entity-centric sequence and lifecycle tracking, and (v) policy-driven network transport simulation for inter-node coordination. On synthetic surveillance mesh scenarios with staged stress contexts (crowding, uplink loss, node degradation), the proposed framework supports online adaptation and policy comparison between receding-horizon min-cost linear programming and decentralized backpressure routing. Results from Monte Carlo experiments indicate improved timely handoff delivery and reduced stale-event rates under weighted policy selection. We release implementation details, configs, scripts, and reproducible evaluation pipelines in-repo.

## 1. Introduction

Long-form video understanding systems are increasingly deployed in distributed camera environments where model accuracy and communication reliability jointly determine operational quality. In surveillance settings, event-level decisions depend on:

1. local perception confidence,
2. temporal continuity under changing context,
3. cross-node transport constraints (latency, congestion, loss), and
4. coordination quality metrics such as handoff success and duplicate alert suppression.

Most existing pipelines optimize these components separately. We instead build an integrated stack in which contextual event posteriors and network transport evaluation are developed and tested together. Our objective is to produce a system that is directly runnable and tunable for engineering deployment studies.

### Contributions

1. **Dynamic contextual event inference stack** in HERMES:
   - taxonomy-driven candidate routing,
   - configurable observation classifier fusion,
   - context-conditional Markov filtering with sliding windows,
   - entity-centric event sequence updates and lifecycle state tracking.
2. **Online diagnostics**:
   - live posterior/matrix visualization,
   - interactive ecological context switching during streaming,
   - entity trajectory timelines (entered/reentered/active/exited/inactive).
3. **Network coordination simulator**:
   - min-cost LP and backpressure policies under context-conditioned link/traffic changes.
4. **Surveillance-specific KPI layer**:
   - timely handoff success, stale-message rate, duplicate alert rate, recovery behavior.
5. **Weighted policy ranking**:
   - direction-aware normalized scoring with customizable operational priorities.

## 2. Related Work

### 2.1 Long-form video understanding

HERMES targets temporal coherence and episodic-semantic modeling for long-form understanding. Our extension does not alter core model pretraining objectives; it augments runtime inference and system-level evaluation.

### 2.2 Sequential event filtering

Markov filtering is widely used for temporal smoothing but is often homogeneous and context-agnostic. We add context-conditioned transitions, optional higher-order prior blending, and sliding-window online re-filtering.

### 2.3 Network control for distributed analytics

Queue-aware control methods such as backpressure provide stability under uncertain load, while optimization-based routing (e.g., min-cost flow/LP) offers stronger immediate cost control. We compare both in a unified packet simulator with surveillance semantics.

## 3. System Overview

Let video observations arrive in windows \(t=1,\dots,T\). For each window:

1. The classifier ranks event candidates from taxonomy-selected label subsets.
2. Observation-level scores are fused by user-defined classifier sets.
3. A context-conditional Markov module updates posterior event beliefs.
4. Entity timelines are updated per observation window.
5. Event messages are routed across node network policies for downstream fusion.

This yields a perception-to-transport loop where uncertainty and transport quality can be jointly analyzed.

## 4. Methods

### 4.1 Dynamic event inference

Given question/context input \(x_t\), the model produces ranked candidates \(\{(e_i, s_i)\}\). Observation classifier fusion maps these into scores \(\tilde{p}_t(e)\), combining model confidence and custom rules/prototypes.

### 4.2 Context-conditional Markov filtering

Posterior update:

\[
\hat{p}_t = p_{t-1} T_{c_t,t}, \quad
p_t(e) \propto \hat{p}_t(e)\,\tilde{p}_t(e)
\]

where \(T_{c_t,t}\) may vary by context \(c_t\), schedule, or provider.

Sliding-window mode recomputes posteriors over last \(W\) observations; higher-order mode blends recent priors across \(k\) steps.

### 4.3 Symbolic transfer entropy diagnostics

Optional symbolic matrix transfer entropy tracks directed dependence from source symbols (e.g., context labels) to target event-state symbols with configurable source/target orders.

### 4.4 Entity sequence and lifecycle tracking

Per-window entity observations update per-entity event sequences and Markov state
history. The runtime also emits lifecycle sets:

- `entered_entities`
- `reentered_entities`
- `exited_entities`
- active/inactive entity sets

These lifecycle outputs are visualized in a live trajectory strip across windows.

### 4.5 Network message passing model

We model directed graph \(G=(V,E)\), edge capacities \(C_e(t)\), delays \(d_e\), and loss \(p_e(t)\). Traffic commodities \(k\) include source, destination, rate, type, TTL, and replication count.

#### Policy A: Receding-horizon min-cost LP

\[
\min_{x_{e,k}(t)\ge 0}\sum_{e,k} c_{e,k}(t)\,x_{e,k}(t)
\]

subject to edge-capacity and queue-availability constraints.

#### Policy B: Backpressure

\[
P_{e,k}(t)=\left(Q_{u,k}(t)-Q_{v,k}(t)\right)
+ \beta \Delta\text{dist}_{u\to v,k}
- \gamma d_e
\]

Positive-pressure commodities are allocated edge capacity greedily.

### 4.6 Coordination KPIs

For surveillance relevance, we track:

- unique delivery ratio,
- stale unique delivery ratio (TTL-based),
- duplicate delivery ratio,
- handoff success and timely handoff success,
- alert duplicate rate,
- backlog recovery after context switches.

### 4.7 Weighted policy ranking

For each metric \(m\), per-policy mean is normalized to \([0,1]\) with direction-aware scaling (maximize/minimize). Final score:

\[
S(\pi)=\frac{\sum_m w_m\,\text{norm}_m(\pi)}{\sum_m w_m}
\]

allowing operator-driven priorities via weight vector \(w\).

## 5. Implementation

### 5.1 Key modules

- Event stack:
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/lavis/common/event_taxonomy.py`
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/lavis/common/event_observation.py`
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/lavis/common/event_markov.py`
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/stream_online.py`
- Network stack:
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/tools/network_message_passing_sim.py`
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/tools/network_message_passing_kpi_report.py`

### 5.2 Configs and scripts

- Surveillance config:
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/network_message_passing/surveillance_mesh.json`
- Reproducible runs:
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/surveillance_test.sh`
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/surveillance_monte_carlo.sh`
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/kpi_report.sh`

## 6. Experimental Protocol

### 6.1 Scenario

Surveillance mesh with 4 camera nodes, 2 edge nodes, and 1 fusion node. Context phases:

1. `normal`
2. `crowded` (higher event traffic)
3. `uplink_loss` (higher uplink loss)
4. `cam2_failure` (degraded camera uplinks)

### 6.2 Evaluation

- Single-run policy comparison
- Monte Carlo runs across random seeds
- KPI aggregation and weighted ranking

### 6.3 Reproducibility commands

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/surveillance_monte_carlo.sh 20 320

bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/kpi_report.sh \
  "logs/network_message_passing/monte_carlo/run_*.json" \
  "logs/network_message_passing/monte_carlo_summary.json" \
  "handoff_timely_success_rate=4,p95_latency=3,stale_unique_ratio=3,delivery_ratio=2"
```

## 7. Results Summary (preliminary, synthetic)

In representative Monte Carlo tests, `min_cost_lp` shows:

- higher delivery ratios,
- lower p95 latency,
- higher timely handoff success,
- lower stale unique delivery rates

relative to backpressure under the provided synthetic workload and objective settings.

Backpressure remains useful as a decentralized baseline and may become preferable under different cost priorities or limited centralized solver availability.

## 8. Discussion

### 8.1 Why integration matters

Perception quality alone is insufficient if coordination traffic cannot meet timeliness constraints. Conversely, high-throughput routing without contextual event smoothing can amplify noisy alerts. The combined stack enables co-analysis.

### 8.2 Operational tuning

Weighted ranking allows policy selection based on deployment priorities:

- strict safety operations: increase `handoff_timely_success_rate`, `stale_unique_ratio` penalties
- bandwidth-constrained deployments: increase `drop_ratio`, `backlog_peak`, `recovery_steps_avg` penalties

### 8.3 Limitations

1. Simulator is synthetic and packet-level.
2. LP is one-step receding-horizon, not global finite-horizon optimization.
3. No explicit compute-node scheduling or model inference latency coupling yet.
4. Entity lifecycle currently depends on provided per-window
   `entity_observations`; automatic detector/tracker/re-identification from raw
   video is not integrated in this module.

## 9. Conclusion

We introduced a practical extension to HERMES that combines context-aware event inference and network coordination benchmarking for multi-feed surveillance. The resulting system is runnable, testable, and tunable end-to-end with explicit coordination KPIs and weighted policy ranking. This provides a concrete foundation for transitioning from model-centric evaluation to system-centric deployment studies.

## 10. Checklist for release-ready manuscript

1. Add real-world trace replay (camera metadata + network telemetry).
2. Add ablations over Markov order/window/context mismatch.
3. Add policy sensitivity analysis over ranking weights.
4. Add confidence intervals and significance tests across larger seed sets.
5. Add qualitative case studies (successful vs failed handoff episodes).
