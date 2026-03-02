# Optimal Message Passing Between Network Nodes

This document describes the simulator and algorithm specs implemented in:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/tools/network_message_passing_sim.py`

## 1) Problem setup

We model a directed network \(G=(V,E)\) with per-edge:

- capacity \(C_e(t)\) packets/step
- delay \(d_e\) steps
- loss probability \(p_e(t)\)

Traffic is a set of commodities \(k \in \mathcal{K}\), each with source \(s_k\), destination \(t_k\), and arrival rate \(\lambda_k(t)\).

At each step \(t\), queues \(Q_{n,k}(t)\) store packets of commodity \(k\) at node \(n\).

Context \(c(t)\) can modify:

- traffic scale \(\lambda_k(t)\)
- edge capacities \(C_e(t)\)
- edge losses \(p_e(t)\)

## 2) Objective

Typical objective combines:

\[
\text{maximize throughput} \quad \text{while minimizing latency, drop, backlog}
\]

Equivalent weighted cost objective:

\[
J = \alpha_1 \cdot \text{delay} + \alpha_2 \cdot \text{drop} + \alpha_3 \cdot \text{queue backlog}
\]

The simulator compares two online policies against these metrics.

## 3) Policy A: Receding-horizon min-cost LP (`min_cost_lp`)

### Decision variables

\[
x_{e,k}(t) \ge 0
\]

packets of commodity \(k\) to send on edge \(e=(u \rightarrow v)\) at time \(t\).

### Constraints

Edge capacity:

\[
\sum_k x_{e,k}(t) \le C_e(t)
\]

Queue availability:

\[
\sum_{e \in \text{out}(n)} x_{e,k}(t) \le Q_{n,k}(t), \quad n \neq t_k
\]

### Cost

For each candidate move \((e,k)\):

\[
c_{e,k}(t) =
w_d \cdot d_e
 + w_s \cdot \text{dist}(v,t_k)
 + w_q \cdot \sum_j Q_{v,j}(t)
 - R_{\text{send}}
\]

where:

- \(\text{dist}(v,t_k)\): shortest-delay remaining distance
- \(R_{\text{send}}\): transmit reward (encourages shipping packets now)

The LP is solved each step via `scipy.optimize.linprog` (HiGHS), then projected to feasible integers for packet movement.

### Interpretation

This is a centralized MPC-style one-step optimizer. It is near-optimal for immediate routing decisions but not a full finite-horizon global optimum.

## 4) Policy B: Backpressure (`backpressure`)

For each edge \(e=(u \rightarrow v)\) and commodity \(k\), pressure score:

\[
P_{e,k}(t) =
 \big(Q_{u,k}(t) - Q_{v,k}(t)\big)
 + \beta \cdot \big(\text{dist}(u,t_k)-\text{dist}(v,t_k)\big)
 - \gamma \cdot d_e
\]

If \(P_{e,k}(t)>0\), commodity \(k\) is eligible on edge \(e\). Capacity is allocated to highest-pressure commodities first.

### Interpretation

Backpressure is fully online and adaptive; it is robust under changing loads but can trade off some path efficiency for stability.

## 5) Context-conditioned network behavior

`context_schedule` activates context names by start step. Each context can set:

- `traffic_scale`: per-commodity traffic multipliers
- `edge_capacity_scale`: per-edge capacity multipliers
- `edge_loss_add` / `edge_loss_scale`: loss perturbations

This enables scenario testing such as `normal -> rush -> degraded`.

## 6) Metrics

Collected per run:

- total injected, delivered, dropped
- delivery ratio, drop ratio
- average and p95 latency
- per-commodity versions of the above
- time series: injected/delivered/dropped/backlog/in-transit

Outputs:

- JSON metrics summary
- dashboard plot (throughput, backlog, cumulative delivered, delivery ratio)

## 7) Config schema

Example config:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/network_message_passing/example_network.json`

Main fields:

- `nodes`: list of node IDs
- `edges`: list of directed links `{id, src, dst, capacity, delay, loss}`
- `traffic`: commodities `{id, src, dst, rate, kind, ttl_steps, copies}`
- `default_context`, `context_schedule`, `contexts`
- `simulation`: `steps`, `seed`, `traffic_mode`
- `policies`: default hyperparameters for both policies

Commodity fields used for coordination KPIs:

- `kind`: e.g. `alert`, `track`, `handoff`, `heartbeat`
- `ttl_steps`: stale threshold
- `copies`: number of replicated copies per unique message (tests duplicate pressure)

## 8) How to run

Compare both policies:

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/compare.sh
```

Single policy:

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/run.sh min_cost_lp
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/run.sh backpressure
```

Tune policy hyperparameters directly:

```bash
python /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/tools/network_message_passing_sim.py \
  --config /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/network_message_passing/example_network.json \
  --policy both \
  --lp-alpha-delay 1.2 \
  --lp-alpha-distance 1.0 \
  --lp-alpha-downstream-queue 0.1 \
  --bp-delay-weight 0.3 \
  --output-json /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/logs/network_message_passing/tuned.json \
  --plot-path /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/logs/network_message_passing/tuned.png
```

## 9) Surveillance coordination KPIs

The simulator now reports:

- `totals.unique_delivery_ratio`
- `totals.stale_unique_ratio`
- `totals.duplicate_delivery_ratio`
- `coordination_kpis.handoff_success_rate`
- `coordination_kpis.handoff_timely_success_rate`
- `coordination_kpis.alert_duplicate_rate`

Surveillance config included:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/network_message_passing/surveillance_mesh.json`

Run once:

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/surveillance_test.sh
```

Monte Carlo + KPI report:

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/surveillance_monte_carlo.sh 20 320
```

Automatic policy ranking is included in the KPI report:

- Normalizes each metric across policies to \([0,1]\) (direction-aware).
- Combines normalized metrics with weighted average.
- Outputs ranked policies with per-metric contribution breakdown.

Default ranking emphasizes:

- `handoff_timely_success_rate`
- `p95_latency`
- `stale_unique_ratio`
- `delivery_ratio`

You can override ranking weights:

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/kpi_report.sh \
  /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/logs/network_message_passing/monte_carlo/run_*.json \
  /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/logs/network_message_passing/monte_carlo_summary.json \
  "handoff_timely_success_rate=4,p95_latency=3,stale_unique_ratio=3,delivery_ratio=2"
```
