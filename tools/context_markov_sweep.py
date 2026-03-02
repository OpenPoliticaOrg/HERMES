#!/usr/bin/env python3
"""
Hyperparameter sweep for context-conditional Markov filtering.

Sweeps:
- markov_order
- window_size
- symbolic transfer-entropy orders
"""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


def _load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_int_list(raw, default):
    if raw is None or str(raw).strip() == "":
        return list(default)
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if len(out) == 0:
        return list(default)
    return out


def _parse_str_list(raw, default):
    if raw is None or str(raw).strip() == "":
        return list(default)
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    if len(out) == 0:
        return list(default)
    return out


def _resolve(root, value):
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def parse_args():
    parser = argparse.ArgumentParser(description="Context Markov hyperparameter sweep")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--markov",
        default="data/taxonomy/example_markov_chain.json",
        help="Markov chain JSON path (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--orders",
        default="1,2,3",
        help="Comma-separated Markov orders to test.",
    )
    parser.add_argument(
        "--window-sizes",
        default="0,6,12",
        help="Comma-separated window sizes to test.",
    )
    parser.add_argument(
        "--te-mode",
        choices=["none", "symbolic_matrix"],
        default="symbolic_matrix",
        help="Transfer entropy mode for sweep runs.",
    )
    parser.add_argument(
        "--te-target-orders",
        default="1,2",
        help="Comma-separated TE target orders to test.",
    )
    parser.add_argument(
        "--te-source-orders",
        default="1,2",
        help="Comma-separated TE source orders to test.",
    )
    parser.add_argument(
        "--contexts",
        default=None,
        help=(
            "Comma-separated context phase sequence. "
            "If omitted, inferred from context_initial keys."
        ),
    )
    parser.add_argument(
        "--steps-per-phase",
        type=int,
        default=25,
        help="Number of simulated updates per context phase.",
    )
    parser.add_argument(
        "--switch-threshold",
        type=float,
        default=0.6,
        help="Target posterior threshold for switch latency.",
    )
    parser.add_argument(
        "--high-prob",
        type=float,
        default=0.8,
        help="Observation probability assigned to expected state in simulation.",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=25,
        help="Max table rows to print.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save full sweep results JSON.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON payload to stdout.",
    )
    return parser.parse_args()


def _build_chain(event_markov_mod, payload, order, window_size, te_mode, te_t, te_s):
    provider = event_markov_mod.EventMarkovChain._load_transition_provider(
        payload.get("transition_provider"),
        payload.get("transition_provider_params", {}),
    )
    return event_markov_mod.EventMarkovChain(
        states=payload.get("states", []),
        transition=payload.get("transition", []),
        initial=payload.get("initial"),
        smoothing=payload.get("smoothing", 1e-9),
        learn_transitions=payload.get("learn_transitions", False),
        transition_mode=payload.get("transition_mode", "homogeneous"),
        transition_schedule=payload.get("transition_schedule", []),
        transition_provider=provider,
        window_size=window_size,
        context_key=payload.get("context_key", "ecological_context"),
        context_transitions=payload.get("context_transitions", {}),
        context_initial=payload.get("context_initial", {}),
        context_fallback=payload.get("context_fallback", "default"),
        markov_order=order,
        transfer_entropy_mode=te_mode,
        transfer_entropy_source=payload.get("transfer_entropy_source", "context"),
        transfer_entropy_target_order=te_t,
        transfer_entropy_source_order=te_s,
    )


def _infer_contexts(payload):
    keys = list((payload.get("context_initial") or {}).keys())
    fallback = payload.get("context_fallback")
    keys = [k for k in keys if k != fallback]
    if len(keys) >= 2:
        return keys[:2]
    if len(keys) == 1:
        return [keys[0], fallback or "default"]
    return ["salon", "garage"]


def _expected_state_by_context(payload, states, contexts):
    expected = {}
    context_initial = payload.get("context_initial", {}) or {}
    for idx, ctx in enumerate(contexts):
        vec = context_initial.get(ctx)
        if vec is not None and len(vec) == len(states):
            expected[ctx] = states[int(np.argmax(np.asarray(vec, dtype=float)))]
        else:
            expected[ctx] = states[min(idx, len(states) - 1)]
    return expected


def _observation_scores(states, expected_state, high_prob):
    n = len(states)
    if n == 1:
        return {states[0]: 1.0}
    p_high = min(max(float(high_prob), 0.01), 0.99)
    p_low = (1.0 - p_high) / float(max(1, n - 1))
    out = {s: p_low for s in states}
    out[expected_state] = p_high
    return out


def _last_not_none(values):
    for v in reversed(values):
        if v is not None:
            return v
    return None


def _format_num(v, nd=4):
    if v is None:
        return "n/a"
    return f"{float(v):.{nd}f}"


def run():
    args = parse_args()
    root = Path(args.repo_root).resolve()
    markov_path = _resolve(root, args.markov)
    if not markov_path.exists():
        raise FileNotFoundError(f"Markov config not found: {markov_path}")

    with open(markov_path, "r") as fp:
        payload = json.load(fp)

    states = payload.get("states", [])
    if len(states) < 2:
        raise ValueError("Sweep requires at least 2 states in Markov config.")

    contexts = _parse_str_list(args.contexts, _infer_contexts(payload))
    if len(contexts) < 2:
        raise ValueError("Provide at least 2 contexts for switching behavior.")

    orders = [max(1, x) for x in _parse_int_list(args.orders, [1])]
    windows = [max(0, x) for x in _parse_int_list(args.window_sizes, [0])]
    te_target_orders = [max(1, x) for x in _parse_int_list(args.te_target_orders, [1])]
    te_source_orders = [max(1, x) for x in _parse_int_list(args.te_source_orders, [1])]
    steps_per_phase = max(2, int(args.steps_per_phase))

    event_markov_mod = _load_module(
        root / "lavis/common/event_markov.py", "event_markov_local_sweep"
    )

    expected_state = _expected_state_by_context(payload, states, contexts)

    combos = list(
        itertools.product(
            orders,
            windows,
            te_target_orders,
            te_source_orders,
        )
    )
    results = []

    for order, window_size, te_t, te_s in combos:
        chain = _build_chain(
            event_markov_mod=event_markov_mod,
            payload=payload,
            order=order,
            window_size=window_size,
            te_mode=args.te_mode,
            te_t=te_t,
            te_s=te_s,
        )

        phase_probs = {ctx: [] for ctx in contexts}
        switch_latency = {ctx: None for ctx in contexts[1:]}
        te_values = []
        final_posterior = None

        for phase_idx, ctx in enumerate(contexts):
            target_state = expected_state[ctx]
            obs_scores = _observation_scores(states, target_state, args.high_prob)
            for local_step in range(steps_per_phase):
                posterior, debug = chain.update(
                    sequence_id="sweep_seq",
                    observation_scores=obs_scores,
                    context={"ecological_context": ctx},
                    return_debug=True,
                )
                final_posterior = posterior

                p_target = float(posterior.get(target_state, 0.0))
                phase_probs[ctx].append(p_target)
                if phase_idx > 0 and switch_latency[ctx] is None and p_target >= args.switch_threshold:
                    switch_latency[ctx] = local_step + 1

                te_debug = debug.get("transfer_entropy") or {}
                te_values.append(te_debug.get("value"))

        phase_mean = {
            ctx: float(np.mean(vals)) if len(vals) > 0 else None
            for ctx, vals in phase_probs.items()
        }
        phase_final = {
            ctx: float(vals[-1]) if len(vals) > 0 else None
            for ctx, vals in phase_probs.items()
        }

        row = {
            "markov_order": int(order),
            "window_size": int(window_size),
            "te_mode": args.te_mode,
            "te_target_order": int(te_t),
            "te_source_order": int(te_s),
            "contexts": list(contexts),
            "expected_state_by_context": expected_state,
            "steps_per_phase": steps_per_phase,
            "switch_threshold": float(args.switch_threshold),
            "phase_mean_target_prob": phase_mean,
            "phase_final_target_prob": phase_final,
            "switch_latency_steps": switch_latency,
            "transfer_entropy_final_bits": _last_not_none(te_values),
            "transfer_entropy_max_bits": (
                float(max([x for x in te_values if x is not None]))
                if any(x is not None for x in te_values)
                else None
            ),
            "final_posterior": final_posterior,
        }
        results.append(row)

    final_ctx = contexts[-1]
    switch_ctx = contexts[1]
    results.sort(
        key=lambda r: (
            -(r["phase_mean_target_prob"].get(final_ctx) or 0.0),
            (r["switch_latency_steps"].get(switch_ctx) or 10**9),
            -(r["transfer_entropy_final_bits"] or -10**9),
        )
    )

    if not args.print_json:
        print("Top sweep results (higher final-context fit is better):")
        print(
            "order window te_t te_s "
            f"mean_{final_ctx} switch_{switch_ctx} te_final_bits te_max_bits"
        )
        for row in results[: max(1, args.max_report)]:
            print(
                f"{row['markov_order']:>5} "
                f"{row['window_size']:>6} "
                f"{row['te_target_order']:>4} "
                f"{row['te_source_order']:>4} "
                f"{_format_num(row['phase_mean_target_prob'].get(final_ctx), 4):>10} "
                f"{str(row['switch_latency_steps'].get(switch_ctx)):>12} "
                f"{_format_num(row['transfer_entropy_final_bits'], 4):>12} "
                f"{_format_num(row['transfer_entropy_max_bits'], 4):>10}"
            )

    payload_out = {
        "config": {
            "markov_path": str(markov_path),
            "orders": orders,
            "window_sizes": windows,
            "te_mode": args.te_mode,
            "te_target_orders": te_target_orders,
            "te_source_orders": te_source_orders,
            "contexts": contexts,
            "steps_per_phase": steps_per_phase,
            "switch_threshold": args.switch_threshold,
            "high_prob": args.high_prob,
        },
        "results": results,
    }

    if args.output_json:
        out_path = _resolve(root, args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(payload_out, fp, indent=2, sort_keys=True)
        if not args.print_json:
            print(f"\nSaved full sweep results to: {out_path}")

    if args.print_json:
        print(json.dumps(payload_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()

