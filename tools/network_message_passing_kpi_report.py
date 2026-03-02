#!/usr/bin/env python3
"""
Aggregate KPI report across multiple simulator result JSON files.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


METRIC_DIRECTIONS = {
    "delivery_ratio": "max",
    "unique_delivery_ratio": "max",
    "drop_ratio": "min",
    "avg_latency": "min",
    "p95_latency": "min",
    "stale_unique_ratio": "min",
    "duplicate_delivery_ratio": "min",
    "handoff_success_rate": "max",
    "handoff_timely_success_rate": "max",
    "alert_duplicate_rate": "min",
    "backlog_peak": "min",
    "recovery_steps_avg": "min",
}

DEFAULT_RANKING_WEIGHTS = {
    "delivery_ratio": 2.0,
    "unique_delivery_ratio": 1.0,
    "drop_ratio": 1.0,
    "avg_latency": 1.0,
    "p95_latency": 2.0,
    "stale_unique_ratio": 2.0,
    "duplicate_delivery_ratio": 1.0,
    "handoff_success_rate": 2.0,
    "handoff_timely_success_rate": 3.0,
    "alert_duplicate_rate": 1.0,
    "backlog_peak": 1.0,
    "recovery_steps_avg": 1.0,
}


def _resolve(root, value):
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def parse_args():
    parser = argparse.ArgumentParser(description="Network message passing KPI report")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--input-glob",
        default="logs/network_message_passing/monte_carlo/run_*.json",
        help="Glob pattern for result files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output summary JSON path.",
    )
    parser.add_argument(
        "--recovery-window",
        type=int,
        default=10,
        help="Pre-switch backlog averaging window.",
    )
    parser.add_argument(
        "--recovery-tolerance",
        type=float,
        default=0.1,
        help="Allowed backlog overshoot over baseline during recovery search.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full summary JSON.",
    )
    parser.add_argument(
        "--ranking-weights",
        default=None,
        help=(
            "Optional comma-separated metric=weight list. "
            "Example: handoff_timely_success_rate=4,p95_latency=3,stale_unique_ratio=3"
        ),
    )
    return parser.parse_args()


def _safe_mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _safe_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.std(vals))


def _parse_weight_overrides(raw_value):
    if raw_value is None:
        return {}
    text = str(raw_value).strip()
    if not text:
        return {}
    out = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid ranking weight token '{token}'. Expected metric=weight."
            )
        key, value = token.split("=", 1)
        key = key.strip()
        if key not in METRIC_DIRECTIONS:
            raise ValueError(
                f"Unknown ranking metric '{key}'. "
                f"Valid metrics: {sorted(METRIC_DIRECTIONS.keys())}"
            )
        out[key] = float(value.strip())
    return out


def _build_ranking_weights(overrides):
    weights = dict(DEFAULT_RANKING_WEIGHTS)
    for key, value in overrides.items():
        weights[key] = float(value)
    return weights


def _policy_metric_means(summary):
    by_policy = {}
    for policy, pdata in summary.get("policies", {}).items():
        by_policy[policy] = {}
        for metric in METRIC_DIRECTIONS.keys():
            by_policy[policy][metric] = (
                pdata.get(metric, {}).get("mean")
                if isinstance(pdata.get(metric), dict)
                else None
            )
    return by_policy


def _normalize_metric_scores(metric, policy_values):
    direction = METRIC_DIRECTIONS.get(metric, "max")
    values = [v for v in policy_values.values() if v is not None]
    if len(values) == 0:
        return {p: 0.5 for p in policy_values.keys()}

    lo = float(min(values))
    hi = float(max(values))
    if hi - lo <= 1e-12:
        return {p: 1.0 for p in policy_values.keys()}

    out = {}
    for policy, value in policy_values.items():
        if value is None:
            out[policy] = 0.5
            continue
        v = float(value)
        if direction == "max":
            out[policy] = (v - lo) / (hi - lo)
        else:
            out[policy] = (hi - v) / (hi - lo)
    return out


def _build_policy_ranking(summary, ranking_weights):
    by_policy_metrics = _policy_metric_means(summary)
    policies = sorted(by_policy_metrics.keys())
    if len(policies) == 0:
        return {"weights": ranking_weights, "ranking": []}

    policy_score = {p: 0.0 for p in policies}
    policy_weight = {p: 0.0 for p in policies}
    contributions = {p: {} for p in policies}

    for metric, weight in ranking_weights.items():
        w = float(weight)
        if w <= 0:
            continue
        metric_values = {p: by_policy_metrics[p].get(metric) for p in policies}
        norm_scores = _normalize_metric_scores(metric, metric_values)
        for p in policies:
            contrib = w * float(norm_scores[p])
            policy_score[p] += contrib
            policy_weight[p] += w
            contributions[p][metric] = {
                "weight": w,
                "raw_value": metric_values[p],
                "normalized_score": float(norm_scores[p]),
                "weighted_contribution": contrib,
            }

    ranking_rows = []
    for p in policies:
        total_w = policy_weight[p]
        if total_w <= 0:
            final_score = None
        else:
            final_score = float(policy_score[p] / total_w)
        ranking_rows.append(
            {
                "policy": p,
                "score": final_score,
                "score_numerator": float(policy_score[p]),
                "score_denominator": float(total_w),
                "metric_contributions": contributions[p],
            }
        )

    ranking_rows.sort(
        key=lambda x: (x["score"] is not None, x["score"] if x["score"] is not None else -1.0),
        reverse=True,
    )

    return {"weights": ranking_weights, "ranking": ranking_rows}


def _recovery_steps_for_run(timeseries, recovery_window=10, recovery_tolerance=0.1):
    contexts = timeseries.get("context", [])
    backlog = timeseries.get("backlog", [])
    if len(contexts) != len(backlog) or len(contexts) == 0:
        return []

    rec = []
    for i in range(1, len(contexts)):
        prev_c = contexts[i - 1]
        curr_c = contexts[i]
        if curr_c == prev_c:
            continue
        if curr_c == "drain" or prev_c == "drain":
            continue
        start = max(0, i - recovery_window)
        baseline_slice = backlog[start:i]
        if len(baseline_slice) == 0:
            continue
        baseline = float(np.mean(baseline_slice))
        threshold = baseline * (1.0 + float(recovery_tolerance))
        rec_steps = None
        for t in range(i, len(backlog)):
            if backlog[t] <= threshold:
                rec_steps = int(t - i)
                break
        rec.append(
            {
                "from_context": prev_c,
                "to_context": curr_c,
                "switch_step": int(i),
                "baseline_backlog": baseline,
                "recovery_steps": rec_steps,
            }
        )
    return rec


def _collect_run_metrics(run_payload, recovery_window=10, recovery_tolerance=0.1):
    rows = []
    for policy, result in run_payload.get("results", {}).items():
        totals = result.get("totals", {})
        coord = result.get("coordination_kpis", {})
        ts = result.get("timeseries", {})
        rec = _recovery_steps_for_run(
            ts,
            recovery_window=recovery_window,
            recovery_tolerance=recovery_tolerance,
        )
        rec_values = [x.get("recovery_steps") for x in rec if x.get("recovery_steps") is not None]
        row = {
            "policy": policy,
            "delivery_ratio": totals.get("delivery_ratio"),
            "unique_delivery_ratio": totals.get("unique_delivery_ratio"),
            "drop_ratio": totals.get("drop_ratio"),
            "avg_latency": totals.get("avg_latency"),
            "p95_latency": totals.get("p95_latency"),
            "stale_unique_ratio": totals.get("stale_unique_ratio"),
            "duplicate_delivery_ratio": totals.get("duplicate_delivery_ratio"),
            "handoff_success_rate": coord.get("handoff_success_rate"),
            "handoff_timely_success_rate": coord.get("handoff_timely_success_rate"),
            "alert_duplicate_rate": coord.get("alert_duplicate_rate"),
            "backlog_peak": (
                int(max(ts.get("backlog", [0]))) if len(ts.get("backlog", [])) > 0 else None
            ),
            "recovery_steps_avg": (
                float(np.mean(rec_values)) if len(rec_values) > 0 else None
            ),
            "recovery_events": rec,
        }
        rows.append(row)
    return rows


def run():
    args = parse_args()
    root = Path(args.repo_root).resolve()
    pattern = str(_resolve(root, args.input_glob))
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {pattern}")

    by_policy = {}
    per_run = []
    for f in files:
        with open(f, "r") as fp:
            payload = json.load(fp)
        rows = _collect_run_metrics(
            payload,
            recovery_window=args.recovery_window,
            recovery_tolerance=args.recovery_tolerance,
        )
        per_run.append({"file": f, "rows": rows})
        for r in rows:
            pol = r["policy"]
            by_policy.setdefault(pol, []).append(r)

    metrics = [
        "delivery_ratio",
        "unique_delivery_ratio",
        "drop_ratio",
        "avg_latency",
        "p95_latency",
        "stale_unique_ratio",
        "duplicate_delivery_ratio",
        "handoff_success_rate",
        "handoff_timely_success_rate",
        "alert_duplicate_rate",
        "backlog_peak",
        "recovery_steps_avg",
    ]

    summary = {"num_files": len(files), "files": files, "policies": {}}
    for policy, rows in by_policy.items():
        summary["policies"][policy] = {}
        for m in metrics:
            vals = [r.get(m) for r in rows]
            summary["policies"][policy][m] = {
                "mean": _safe_mean(vals),
                "std": _safe_std(vals),
            }

    ranking_overrides = _parse_weight_overrides(args.ranking_weights)
    ranking_weights = _build_ranking_weights(ranking_overrides)
    ranking = _build_policy_ranking(summary, ranking_weights)

    print(f"Loaded {len(files)} runs.")
    for policy in sorted(summary["policies"].keys()):
        s = summary["policies"][policy]
        print(f"\nPolicy: {policy}")
        print(
            "  delivery_ratio(mean±std): "
            f"{s['delivery_ratio']['mean']} ± {s['delivery_ratio']['std']}"
        )
        print(
            "  p95_latency(mean±std): "
            f"{s['p95_latency']['mean']} ± {s['p95_latency']['std']}"
        )
        print(
            "  handoff_success(mean±std): "
            f"{s['handoff_success_rate']['mean']} ± {s['handoff_success_rate']['std']}"
        )
        print(
            "  handoff_timely(mean±std): "
            f"{s['handoff_timely_success_rate']['mean']} ± {s['handoff_timely_success_rate']['std']}"
        )
        print(
            "  alert_duplicate(mean±std): "
            f"{s['alert_duplicate_rate']['mean']} ± {s['alert_duplicate_rate']['std']}"
        )
        print(
            "  stale_unique(mean±std): "
            f"{s['stale_unique_ratio']['mean']} ± {s['stale_unique_ratio']['std']}"
        )
        print(
            "  recovery_steps(mean±std): "
            f"{s['recovery_steps_avg']['mean']} ± {s['recovery_steps_avg']['std']}"
        )

    print("\nPolicy ranking (0..1, higher is better):")
    for i, row in enumerate(ranking["ranking"], start=1):
        print(f"  {i}. {row['policy']}: {row['score']}")

    out_payload = {
        "summary": summary,
        "ranking": ranking,
        "per_run": per_run,
    }

    if args.output_json:
        out = _resolve(root, args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fp:
            json.dump(out_payload, fp, indent=2, sort_keys=True)
        print(f"\nSaved summary JSON to: {out}")

    if args.print_json:
        print(json.dumps(out_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
