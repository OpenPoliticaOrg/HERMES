# Dynamic Event Tagging and Mapping Design

## Quick Run
- Setup doctor:
```bash
bash run_scripts/context_markov/doctor.sh
```
- Fast smoke test:
```bash
bash run_scripts/context_markov/test.sh
```
- Live stream:
```bash
bash run_scripts/context_markov/live.sh 0 salon
```

## Goal
Enable HERMES classification to:
1. Rank events from a dynamic candidate set (classifier-driven), not only free-form generation.
2. Map predicted labels to stable `event_id`s using a taxonomy.
3. Emit structured outputs for downstream event graph/timeline systems.

## Current Gaps
- Classification currently uses `model.generate(...)` and exact string match.
- Candidate ranking (`predict_class`) exists in the model but is not used by the classification task.
- No stable event ID mapping layer exists between model text and downstream systems.

## Proposed Architecture

### 1) Taxonomy Layer
A JSON file defines canonical events and aliases:

```json
{
  "schema_version": 1,
  "events": [
    {
      "event_id": "coin:put_on_hair_extensions",
      "label": "put on hair extensions",
      "aliases": ["apply hair extensions"],
      "dataset": "coin_cls",
      "parents": ["domain:nursing_and_care"],
      "metadata": {"domain": "nursing and care"}
    }
  ]
}
```

Semantics:
- `event_id`: stable identifier for storage/joins.
- `label`: canonical classifier string.
- `aliases`: optional surface forms mapped to same `event_id`.
- `dataset`: optional dataset scope (`coin_cls`, `lvu_cls`, etc.).
- `parents`: optional hierarchy links.

### 2) Classifier Layer
The same JSON optionally defines classifier configs:

```json
{
  "classifiers": [
    {
      "classifier_id": "lvu_director",
      "dataset": "lvu_cls",
      "active": true,
      "prompt": "Question: {} Answer:",
      "candidate_event_ids": ["lvu:director:steven", "lvu:director:ron"],
      "rules": {"question_contains_any": ["director"]}
    }
  ]
}
```

Semantics:
- `candidate_event_ids`: explicit candidate space to rank.
- `rules.question_contains_any`: optional per-sample routing.
- `prompt`: optional classifier-specific prompt template.

### 3) Runtime Selection
Per sample:
1. Select active classifiers for dataset.
2. If classifier has `rules`, match question text.
3. First matched classifier wins; else fallback to default label space.
4. Call `predict_class` with selected candidate labels.
5. Convert ranked candidate indices to `{label, event_id}`.

### 4) Output Contract
Each prediction record contains both backward-compatible and structured fields:

```json
{
  "image_id": "xZecGPPhbHE",
  "caption": ["put on hair extensions", "comb hair"],
  "classifier_id": "coin_default",
  "question": "what is the activity in the video?",
  "event_predictions": [
    {
      "rank": 1,
      "candidate_index": 17,
      "label": "put on hair extensions",
      "event_id": "coin:put_on_hair_extensions",
      "confidence": 0.82
    }
  ]
}
```

Notes:
- `caption` is preserved to avoid breaking existing top-1/top-5 metric code.
- `confidence` is computed from softmax over negative candidate losses.

## Config Knobs
Add run config options:
- `classification_mode`: `generate` | `rank` (default `generate` for compatibility)
- `event_taxonomy_path`: path to taxonomy JSON
- `classifier_prompt`: fallback prompt template (default `{}`)
- `rank_n_segments`: candidate chunking for memory/speed (default `1`)
- `rank_topk`: number of labels/tags to emit (default `5`)
- `observation_classifier_path`: path to prototype/binary classifier set JSON
- `markov_chain_path`: path to Markov chain JSON
- `markov_sequence_mode`: `prefix_before_underscore` | `image_id`
- `markov_topk`: number of posterior states to export per observation
- `markov_context_field`: sample field carrying ecological context (default `ecological_context`)
- `markov_debug`: include `T_t`, prior/predicted/likelihood/posterior internals in output

Example CLI override:

```bash
torchrun --nproc_per_node=1 train.py \
  --cfg-path lavis/projects/hermes/cls_coin.yaml \
  --options \
  run.evaluate True \
  run.classification_mode rank \
  run.event_taxonomy_path data/taxonomy/example_event_taxonomy.json \
  run.classifier_prompt "{}" \
  run.observation_classifier_path data/taxonomy/example_observation_classifiers.json \
  run.markov_chain_path data/taxonomy/example_markov_chain.json \
  run.markov_sequence_mode prefix_before_underscore \
  run.markov_topk 5 \
  run.rank_n_segments 2 \
  run.rank_topk 5
```

## Prototype/Binary Classifier Set
You can attach custom classifiers per event:

```json
{
  "combination": "weighted_mean",
  "classifiers": [
    {
      "classifier_id": "proto_puton",
      "event_id": "coin:put_on_hair_extensions",
      "type": "prototype_label",
      "weight": 0.7,
      "include_model_score": true,
      "model_weight": 0.3,
      "params": {
        "prototypes": ["put on hair extensions", "apply hair extensions"],
        "use_confidence_weight": true
      }
    },
    {
      "classifier_id": "binary_puton_conf",
      "event_id": "coin:put_on_hair_extensions",
      "type": "confidence_threshold",
      "weight": 1.0,
      "params": {
        "threshold": 0.45,
        "low_score": 0.1,
        "high_score": 0.9
      }
    },
    {
      "classifier_id": "custom_python_classifier",
      "event_id": "coin:change_car_tire",
      "type": "python",
      "class_path": "my_project.classifiers:MyBinaryClassifier",
      "params": {
        "alpha": 0.2
      }
    }
  ]
}
```

Built-in types:
- `prototype_label`: similarity to prototype phrases.
- `confidence_threshold`: binary decision from model confidence.
- `keyword_binary`: keyword presence in question/predicted labels.
- `python`: load your own class (`module.path:ClassName`) implementing `score(context) -> [0,1]`.

## Markov Chain Config
Use observation scores to update sequence belief:

```json
{
  "states": [
    "coin:put_on_hair_extensions",
    "coin:change_car_tire"
  ],
  "transition_mode": "context",
  "context_key": "ecological_context",
  "context_fallback": "default",
  "context_initial": {
    "garage": [0.2, 0.8],
    "salon": [0.8, 0.2],
    "default": [0.5, 0.5]
  },
  "context_transitions": {
    "garage": [[0.6, 0.4], [0.2, 0.8]],
    "salon": [[0.9, 0.1], [0.5, 0.5]],
    "default": [[0.85, 0.15], [0.25, 0.75]]
  },
  "initial": [0.5, 0.5],
  "window_size": 12,
  "transition": [
    [0.9, 0.1],
    [0.2, 0.8]
  ],
  "smoothing": 1e-9,
  "learn_transitions": false
}
```

Update rule per observation:
1. Predict: `p_t^- = p_{t-1} * T`
2. Correct: `p_t ∝ p(obs_t | state_t) * p_t^-`
3. Normalize to get posterior state probabilities.

Notes:
- `transition_mode = "homogeneous"` uses one fixed matrix (`transition`).
- `transition_mode = "context"` picks transition and prior from ecological context labels.
- `transition_mode = "schedule"` selects `T_t` by step index from `transition_schedule`.
- `transition_mode = "provider"` calls a Python provider for custom `T_t = f(context, t, obs)`.
- `window_size > 0` enables sliding-window online filtering; posterior is recomputed over the last `window_size` observations.

## Online Video Feed Adapter
Use `/Users/ajithsenthil/Desktop/CompPsychoVid/HERMES/stream_online.py` for live/file streaming:

```bash
python stream_online.py \
  --cfg-path lavis/projects/hermes/cls_coin.yaml \
  --video-source 0 \
  --question "what is the activity in the video?" \
  --sequence-id cam0 \
  --ecological-context garage \
  --context-field ecological_context \
  --ecological-context-by-window /path/to/context_schedule.json \
  --chunk-seconds 3 \
  --stride-seconds 1 \
  --options \
    run.classification_mode rank \
    run.event_taxonomy_path data/taxonomy/example_event_taxonomy.json \
    run.observation_classifier_path data/taxonomy/example_observation_classifiers.json \
    run.markov_chain_path data/taxonomy/example_markov_chain.json \
    run.markov_context_field ecological_context
```

`--ecological-context-by-window` expects JSON like:
```json
{"0": "garage", "30": "street", "80": "salon"}
```
Meaning: use `garage` from window 0, switch to `street` at window 30, then `salon` at window 80.

Each window outputs JSON with:
- ranked event labels (`event_predictions`)
- observation model scores (`observation_scores`)
- online Markov posterior (`markov_posterior`, `markov_state`)
- optional Markov internals (`markov_debug`) when `--debug-markov` or `run.markov_debug=True`

For one-command live visualization:
```bash
bash run_scripts/context_markov/live_viz.sh 0 salon
```

## Fallback Behavior
If taxonomy/classifier config is missing:
- Build candidate label space from dataset annotations (`annotation[*].label`).
- Map label to event ID as `null`.

## Why This Design
- Minimal code risk: reuses existing model ranking path.
- Dynamic by construction: classifier rules can switch candidate sets per sample.
- Stable IDs: decouples model text from analytics/storage schemas.
- Backward compatible outputs and metrics.
