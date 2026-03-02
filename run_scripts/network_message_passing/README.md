# Network Message Passing Simulator

## Quick start (compare both policies)
```bash
bash run_scripts/network_message_passing/compare.sh
```

Outputs:
- `logs/network_message_passing/compare.json`
- `logs/network_message_passing/compare.png`

## Run one policy
```bash
bash run_scripts/network_message_passing/run.sh min_cost_lp
bash run_scripts/network_message_passing/run.sh backpressure
```

Arguments for `run.sh`:
1. `policy`: `min_cost_lp` | `backpressure` | `both`
2. `config_path`: network JSON config
3. `steps`: simulation steps
4. `output_json`: metrics/results output file
5. `plot_path`: dashboard PNG output file

## Direct CLI usage
```bash
python tools/network_message_passing_sim.py \
  --config data/network_message_passing/example_network.json \
  --policy both \
  --steps 180 \
  --output-json logs/network_message_passing/results.json \
  --plot-path logs/network_message_passing/dashboard.png
```

Useful flags:
- `--traffic-mode poisson|deterministic`
- `--lp-send-reward`, `--lp-alpha-delay`, `--lp-alpha-distance`, `--lp-alpha-downstream-queue`
- `--bp-delay-weight`
- `--show-plot`

## Surveillance coordination test
Single run:
```bash
bash run_scripts/network_message_passing/surveillance_test.sh
```

This uses:
- `data/network_message_passing/surveillance_mesh.json`
- context phases: normal, crowded, uplink_loss, cam2_failure
- KPI fields: delivery, latency, stale ratio, duplicate ratio, handoff success

Monte Carlo summary:
```bash
bash run_scripts/network_message_passing/surveillance_monte_carlo.sh 20 320
```

Outputs:
- `logs/network_message_passing/monte_carlo/run_*.json`
- `logs/network_message_passing/monte_carlo_summary.json`

Recompute KPI summary from existing Monte Carlo runs:
```bash
bash run_scripts/network_message_passing/kpi_report.sh
```

Custom ranking priorities (metric weights):
```bash
bash run_scripts/network_message_passing/kpi_report.sh \
  logs/network_message_passing/monte_carlo/run_*.json \
  logs/network_message_passing/monte_carlo_summary.json \
  "handoff_timely_success_rate=4,p95_latency=3,stale_unique_ratio=3,delivery_ratio=2"
```
