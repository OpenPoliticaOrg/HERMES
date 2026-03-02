# HERMES Repository and Extensions Documentation

This document covers:

1. Core HERMES repository usage
2. Dynamic event tagging + context-conditional Markov extensions
3. Network message passing and surveillance coordination evaluation extensions
4. Reproducible run flows for setup, testing, and benchmarking

## 1) Repository overview

Primary upstream focus:

- Long-form video understanding model training/inference across MovieCORE, LVU, Breakfast, COIN, and MovieChat.

Extension focus added in this workspace:

- Dynamic event taxonomy routing for classification candidates
- Pluggable observation classifiers (prototype/binary/custom)
- Context-conditional inhomogeneous Markov filtering with sliding windows
- Live streaming with interactive context switching and Markov diagnostics
- Network message-passing simulator with policy comparison
- Surveillance-specific coordination KPI evaluation and Monte Carlo reporting

## 2) Important directories

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/lavis/`
  - model/tasks/common modules
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/`
  - dataset and example configs
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/tools/`
  - simulation, doctor, smoke, and KPI scripts
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/`
  - runnable shell wrappers
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/docs/`
  - design notes and algorithm docs

## 3) Dynamic event + Markov extension architecture

Main files:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/lavis/common/event_taxonomy.py`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/lavis/common/event_observation.py`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/lavis/common/event_markov.py`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/lavis/tasks/classification.py`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/stream_online.py`

### 3.1 Data flow (rank mode)

1. Candidate event classes are selected from taxonomy.
2. Model ranking provides per-candidate confidence.
3. Observation classifier set fuses model scores + prototype/binary/custom features.
4. Context-conditional Markov update smooths posterior online.
5. Streaming path emits:
   - top predictions
   - observation scores
   - Markov posterior/state
   - optional Markov debug internals

### 3.2 Key capabilities

- Context-conditioned transition matrices (`transition_mode=context`)
- Inhomogeneous transitions (schedule/provider support)
- Sliding-window re-filtering (`window_size`)
- Higher-memory prior blending (`markov_order`)
- Optional symbolic matrix transfer entropy diagnostics (`transfer_entropy_mode=symbolic_matrix`)
- Interactive context control in live dashboard

### 3.3 Core configs

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/taxonomy/example_event_taxonomy.json`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/taxonomy/example_observation_classifiers.json`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/taxonomy/example_markov_chain.json`

## 4) Dynamic event + Markov runbook

### 4.1 Health checks

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/context_markov/doctor.sh
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/context_markov/test.sh
```

### 4.2 Live visualization

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/context_markov/live_viz.sh 0 salon
```

Interactive keys (matplotlib window focused):

- `[` / left arrow: previous context
- `]` / right arrow: next context
- `1`..`9`: direct context selection
- `a`: auto/manual toggle
- `q`/`Esc`: quit

### 4.3 Hyperparameter sweeps

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/context_markov/sweep.sh
```

Sweeps:

- `markov_order`
- `window_size`
- symbolic TE target/source orders

## 5) Network message passing extension architecture

Main files:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/tools/network_message_passing_sim.py`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/network_message_passing/example_network.json`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/data/network_message_passing/surveillance_mesh.json`

### 5.1 Policies

- `min_cost_lp`: receding-horizon LP under queue/capacity constraints
- `backpressure`: queue-differential online routing with delay/progress terms

### 5.2 Context-conditioned stress testing

Context phases can modify:

- traffic rates
- edge capacity multipliers
- edge loss perturbations

### 5.3 Surveillance coordination KPIs

Per run output includes:

- `totals.delivery_ratio`, `totals.drop_ratio`
- `totals.unique_delivery_ratio`
- `totals.stale_unique_ratio`
- `totals.duplicate_delivery_ratio`
- `coordination_kpis.handoff_success_rate`
- `coordination_kpis.handoff_timely_success_rate`
- `coordination_kpis.alert_duplicate_rate`

## 6) Surveillance evaluation runbook

### 6.1 Single scenario run

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/surveillance_test.sh
```

### 6.2 Monte Carlo policy comparison

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/surveillance_monte_carlo.sh 20 320
```

### 6.3 KPI aggregation and weighted ranking

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/kpi_report.sh
```

Custom ranking priorities:

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/network_message_passing/kpi_report.sh \
  "logs/network_message_passing/monte_carlo/run_*.json" \
  "logs/network_message_passing/monte_carlo_summary.json" \
  "handoff_timely_success_rate=4,p95_latency=3,stale_unique_ratio=3,delivery_ratio=2"
```

## 7) Output artifacts

Default outputs:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/logs/context_markov/*.jsonl`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/logs/network_message_passing/*.json`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/logs/network_message_passing/*.png`

## 8) Reproducibility notes

- Set fixed seeds for Monte Carlo comparability.
- Keep same config and horizon across policy comparisons.
- For fair policy comparison in simulator:
  - arrival randomness and link-loss randomness are seeded deterministically and separated internally.

## 9) Limitations

- `min_cost_lp` is one-step receding-horizon optimization, not full finite-horizon optimal control.
- Simulator is packet-level and synthetic; it does not include full vision-model compute scheduling or GPU contention.
- KPI semantics depend on config assumptions (TTL/copies/context stress patterns).

## 10) Suggested next engineering milestones

1. Integrate simulator outputs with real camera metadata replay traces.
2. Add policy tuning loop (Bayesian optimization over weights).
3. Add failure-injection templates (edge cut, burst loss, clock skew, delayed acknowledgments).
4. Add end-to-end benchmark reports (CSV + plots + markdown summary generator).

## 11) Paper and presentation artifact build

### 11.1 Paper (LaTeX -> PDF)

Source:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/paper/networked_contextual_hermes.tex`

Build:

```bash
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/papers/build_networked_contextual_hermes_paper.sh
```

Output:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/paper/networked_contextual_hermes.pdf`

### 11.2 Presentation (Markdown -> HTML + PDF)

Source:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/docs/presentation_networked_contextual_hermes.md`
- Visual assets generated for the deck:
  - `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/docs/assets/networked_contextual_hermes/`

Build:

```bash
# Optional: refresh presentation visuals from simulator outputs
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/presentations/build_networked_contextual_hermes_visuals.sh

# Build HTML + PDF deck
bash /Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/run_scripts/presentations/build_networked_contextual_hermes_slides.sh
```

Outputs:

- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/docs/presentation_networked_contextual_hermes.html`
- `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/docs/presentation_networked_contextual_hermes.pdf`
