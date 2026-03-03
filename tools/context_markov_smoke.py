#!/usr/bin/env python3
"""
Fast smoke test for context-conditional event observation + Markov updates.

Does not require model checkpoints; validates config wiring and online update logic.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Context Markov smoke test")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--taxonomy",
        default="data/taxonomy/example_event_taxonomy.json",
        help="Taxonomy JSON path (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--observation",
        default="data/taxonomy/example_observation_classifiers.json",
        help="Observation classifier JSON path (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--markov",
        default="data/taxonomy/example_markov_chain.json",
        help="Markov JSON path (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print JSON summary instead of plain text logs.",
    )
    return parser.parse_args()


def _resolve(root, path_like):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return root / p


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def main():
    args = parse_args()
    root = Path(args.repo_root).resolve()

    taxonomy_path = _resolve(root, args.taxonomy)
    observation_path = _resolve(root, args.observation)
    markov_path = _resolve(root, args.markov)

    event_taxonomy_mod = _load_module(
        root / "lavis/common/event_taxonomy.py", "event_taxonomy_local"
    )
    event_obs_mod = _load_module(
        root / "lavis/common/event_observation.py", "event_observation_local"
    )
    event_markov_mod = _load_module(
        root / "lavis/common/event_markov.py", "event_markov_local"
    )
    entity_seq_mod = _load_module(
        root / "lavis/common/entity_event_sequence.py", "entity_event_sequence_local"
    )

    taxonomy = event_taxonomy_mod.EventTaxonomy(str(taxonomy_path))
    obs_set = event_obs_mod.ObservationClassifierSet.from_file(str(observation_path))
    markov = event_markov_mod.EventMarkovChain.from_file(str(markov_path))

    # 1) Taxonomy route check.
    selector = taxonomy.select_candidates(
        dataset_name="lvu_cls",
        question_text="what is the director of the movie?",
        fallback_labels=["steven", "ron"],
        fallback_prompt="{}",
    )
    if selector["classifier_id"] != "lvu_director":
        fail("Taxonomy classifier routing failed for lvu_director.")

    # 2) Observation classifier check.
    obs_context = {
        "question": "what is the director of the movie?",
        "event_predictions": [
            {"label": "steven", "event_id": "lvu:director:steven", "confidence": 0.7},
            {"label": "ron", "event_id": "lvu:director:ron", "confidence": 0.2},
        ],
    }
    obs_scores = obs_set.score_events(
        base_context=obs_context,
        candidate_event_ids=["lvu:director:steven", "lvu:director:ron"],
        model_scores={"lvu:director:steven": 0.7, "lvu:director:ron": 0.2},
    )
    if obs_scores["lvu:director:steven"] <= obs_scores["lvu:director:ron"]:
        fail("Observation classifier scoring is not preferring steven as expected.")

    # 3) Context-conditional Markov check.
    salon_post = markov.update(
        sequence_id="demo_salon",
        observation_scores={
            "coin:put_on_hair_extensions": 0.55,
            "coin:change_car_tire": 0.45,
        },
        context={"ecological_context": "salon"},
    )
    garage_post = markov.update(
        sequence_id="demo_garage",
        observation_scores={
            "coin:put_on_hair_extensions": 0.55,
            "coin:change_car_tire": 0.45,
        },
        context={"ecological_context": "garage"},
    )

    if (
        salon_post["coin:put_on_hair_extensions"]
        <= garage_post["coin:put_on_hair_extensions"]
    ):
        fail("Context-conditioned Markov transition/prior did not react to ecology.")

    # 4) Sliding-window + context switch sanity.
    seq = "demo_switch"
    for _ in range(3):
        markov.update(
            sequence_id=seq,
            observation_scores={
                "coin:put_on_hair_extensions": 0.7,
                "coin:change_car_tire": 0.3,
            },
            context={"ecological_context": "salon"},
        )
    switched = None
    for _ in range(3):
        switched = markov.update(
            sequence_id=seq,
            observation_scores={
                "coin:put_on_hair_extensions": 0.3,
                "coin:change_car_tire": 0.7,
            },
            context={"ecological_context": "garage"},
        )
    if switched["coin:change_car_tire"] <= 0.5:
        fail("Online context switch did not move posterior toward garage-expected state.")

    # 5) Markov order + symbolic transfer entropy debug check.
    with open(markov_path, "r") as fp:
        markov_payload = json.load(fp)
    markov_te = event_markov_mod.EventMarkovChain(
        states=markov_payload.get("states", []),
        transition=markov_payload.get("transition", []),
        initial=markov_payload.get("initial"),
        smoothing=markov_payload.get("smoothing", 1e-9),
        learn_transitions=markov_payload.get("learn_transitions", False),
        transition_mode=markov_payload.get("transition_mode", "homogeneous"),
        transition_schedule=markov_payload.get("transition_schedule", []),
        transition_provider=None,
        window_size=0,
        context_key=markov_payload.get("context_key", "ecological_context"),
        context_transitions=markov_payload.get("context_transitions", {}),
        context_initial=markov_payload.get("context_initial", {}),
        context_fallback=markov_payload.get("context_fallback", "default"),
        markov_order=3,
        transfer_entropy_mode="symbolic_matrix",
        transfer_entropy_source="context",
        transfer_entropy_target_order=2,
        transfer_entropy_source_order=1,
    )
    te_value = None
    for idx, ctx in enumerate(["salon", "salon", "garage", "garage", "garage"]):
        _, dbg = markov_te.update(
            sequence_id="demo_te",
            observation_scores={
                "coin:put_on_hair_extensions": 0.8 if ctx == "salon" else 0.2,
                "coin:change_car_tire": 0.2 if ctx == "salon" else 0.8,
            },
            context={"ecological_context": ctx},
            return_debug=True,
        )
        if dbg.get("markov_order") != 3:
            fail("Markov order debug value is incorrect.")
        te_dbg = dbg.get("transfer_entropy") or {}
        if idx >= 3 and te_dbg.get("value") is not None:
            te_value = float(te_dbg.get("value"))
    if te_value is None:
        fail("Symbolic transfer entropy did not produce a value.")

    # 6) Entity sequence tracker check.
    markov_entity = event_markov_mod.EventMarkovChain.from_file(str(markov_path))
    entity_tracker = entity_seq_mod.EntityEventSequenceTracker(
        markov_chain=markov_entity,
        context_field="ecological_context",
        history_limit=8,
        default_entity_id="__scene__",
        default_markov_topk=3,
        default_observation_topk=3,
        default_missing_tolerance=0,
    )
    entity_tracker.begin_window(
        base_sequence_id="cam0",
        context={"ecological_context": "salon"},
        image_id="cam0_0",
    )
    e1 = entity_tracker.update_entity(
        base_sequence_id="cam0",
        entity_id="person_1",
        observation_scores={
            "coin:put_on_hair_extensions": 0.82,
            "coin:change_car_tire": 0.18,
        },
        context={"ecological_context": "salon"},
        image_id="cam0_0",
    )
    e2 = entity_tracker.update_entity(
        base_sequence_id="cam0",
        entity_id="person_2",
        observation_scores={
            "coin:put_on_hair_extensions": 0.22,
            "coin:change_car_tire": 0.78,
        },
        context={"ecological_context": "garage"},
        image_id="cam0_0",
    )
    life0 = entity_tracker.finalize_window("cam0")

    entity_tracker.begin_window(
        base_sequence_id="cam0",
        context={"ecological_context": "garage"},
        image_id="cam0_1",
    )
    e3 = entity_tracker.update_entity(
        base_sequence_id="cam0",
        entity_id="person_1",
        observation_scores={
            "coin:put_on_hair_extensions": 0.35,
            "coin:change_car_tire": 0.65,
        },
        context={"ecological_context": "garage"},
        image_id="cam0_1",
    )
    life1 = entity_tracker.finalize_window("cam0")

    entity_tracker.begin_window(
        base_sequence_id="cam0",
        context={"ecological_context": "garage"},
        image_id="cam0_2",
    )
    e4 = entity_tracker.update_entity(
        base_sequence_id="cam0",
        entity_id="person_2",
        observation_scores={
            "coin:put_on_hair_extensions": 0.21,
            "coin:change_car_tire": 0.79,
        },
        context={"ecological_context": "garage"},
        image_id="cam0_2",
    )
    life2 = entity_tracker.finalize_window("cam0")

    if e1 is None or e2 is None or e3 is None:
        fail("Entity sequence tracker did not return updates.")
    if e4 is None:
        fail("Entity sequence tracker did not return re-entry update.")
    if int(e3.get("sequence_length", 0)) != 2:
        fail("Entity sequence length did not increment for repeated entity observations.")
    if int(e4.get("sequence_length", 0)) != 2:
        fail("Entity sequence length is incorrect after entity re-entry.")
    if len(e3.get("event_sequence", [])) != 2:
        fail("Entity event sequence history is missing expected steps.")
    if e3.get("observation_state", {}).get("event_id") != "coin:change_car_tire":
        fail("Entity observation top event is incorrect.")
    if e2.get("lifecycle_state") != "entered":
        fail("Initial lifecycle state for person_2 should be entered.")
    if "person_2" not in life1.get("exited_entities", []):
        fail("person_2 was not marked as exited on missed window.")
    if e4.get("lifecycle_state") != "reentered":
        fail("person_2 should be marked as reentered after returning.")
    if "person_1" not in life2.get("exited_entities", []):
        fail("person_1 was not marked as exited when unobserved in window 3.")
    if "person_2" not in life2.get("reentered_entities", []):
        fail("Window lifecycle summary is missing reentered person_2.")

    result = {
        "status": "ok",
        "paths": {
            "taxonomy": str(taxonomy_path),
            "observation": str(observation_path),
            "markov": str(markov_path),
        },
        "checks": {
            "taxonomy_classifier_id": selector["classifier_id"],
            "observation_scores": obs_scores,
            "salon_post": salon_post,
            "garage_post": garage_post,
            "switch_post": switched,
            "te_bits": te_value,
            "entity_sequence_person_1_len": int(e3.get("sequence_length", 0)),
            "entity_sequence_person_2_len": int(e4.get("sequence_length", 0)),
            "entity_lifecycle_window_1_exited": life1.get("exited_entities", []),
            "entity_lifecycle_window_2_reentered": life2.get("reentered_entities", []),
        },
    }

    if args.print_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("[OK] Context Markov smoke test passed.")
        print(
            "[OK] salon vs garage:",
            salon_post["coin:put_on_hair_extensions"],
            garage_post["coin:put_on_hair_extensions"],
        )
        print("[OK] switch posterior:", switched)


if __name__ == "__main__":
    main()
