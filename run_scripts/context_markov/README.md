# Context Markov Quick Run

## 1) Fast smoke test (no model checkpoint required)
```bash
bash run_scripts/context_markov/test.sh
```

This validates:
- taxonomy routing
- prototype/binary observation scoring
- context-conditional Markov updates
- sliding-window + context switch behavior

## 1.5) Full setup doctor (deps + files + smoke)
```bash
bash run_scripts/context_markov/doctor.sh
```

Optional:
```bash
# with checkpoint and webcam probe
bash run_scripts/context_markov/doctor.sh /path/model.pth 0
```

## 2) Live stream run (webcam or file)
```bash
bash run_scripts/context_markov/live.sh 0 salon
```

Arguments:
1. `video_source` (default `0`): camera index or video file path
2. `ecological_context` (default `salon`)
3. `question` (default `what is the activity in the video?`)
4. `sequence_id` (default `cam0`)
5. `output_jsonl` (default `logs/context_markov_live.jsonl`)
6. `checkpoint_path` (optional)

Example with file input and checkpoint:
```bash
bash run_scripts/context_markov/live.sh /path/video.mp4 garage \
  "what is the activity in the video?" \
  cam_file \
  logs/context_markov_file.jsonl \
  /path/model.pth
```

## 2.5) Live stream with matrix/debug visualization
```bash
bash run_scripts/context_markov/live_viz.sh 0 salon
```

This enables:
- per-step Markov debug payload (`markov_debug`)
- live matplotlib dashboard with:
  - top classifications
  - posterior state probabilities
  - transition matrix used at each step
  - context + step diagnostics
- interactive context switching from keyboard (without restart)

Optional `live_viz.sh` arg 7:
1. `context_options` (comma-separated list), example: `garage,salon,street`

Optional `live_viz.sh` args 8-11:
1. `markov_order` (>=1)
2. `window_size` (>=0)
3. `te_target_order` (>=1, enables symbolic TE)
4. `te_source_order` (>=1, enables symbolic TE)

Hotkeys (focus the matplotlib window first):
- `[` or left arrow: previous context
- `]` or right arrow: next context
- `1`..`9`: jump to context index in diagnostics panel
- `a`: toggle auto/manual context mode
- `q` or `Esc`: stop stream

## 3) Hyperparameter sweep (order/window/TE)
```bash
bash run_scripts/context_markov/sweep.sh
```

Arguments:
1. `markov_cfg` (default example config)
2. `orders` (default `1,2,3`)
3. `window_sizes` (default `0,6,12`)
4. `te_target_orders` (default `1,2`)
5. `te_source_orders` (default `1,2`)
6. `steps_per_phase` (default `25`)
7. `output_json` (default `logs/context_markov_sweep.json`)

Example:
```bash
bash run_scripts/context_markov/sweep.sh \
  data/taxonomy/example_markov_chain.json \
  1,2,3,4 \
  0,4,8,12 \
  1,2,3 \
  1,2 \
  30 \
  logs/my_sweep.json
```

The sweep prints a ranking table and writes full metrics JSON.

## 4) Context schedule (optional)
Use `stream_online.py` directly with a context schedule JSON:
```json
{"0": "garage", "30": "street", "80": "salon"}
```
