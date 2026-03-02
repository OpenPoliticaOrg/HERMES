#!/usr/bin/env python3
"""
Build visual assets used by the networked contextual HERMES presentation.
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


METRIC_CONFIG = [
    ("delivery_ratio", "Delivery ratio", "max"),
    ("unique_delivery_ratio", "Unique delivery ratio", "max"),
    ("handoff_timely_success_rate", "Timely handoff", "max"),
    ("p95_latency", "p95 latency", "min"),
    ("stale_unique_ratio", "Stale unique ratio", "min"),
    ("drop_ratio", "Drop ratio", "min"),
]


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def _load_json(path: Path):
    with path.open("r") as fp:
        return json.load(fp)


def _short_state_name(name: str) -> str:
    out = str(name).split(":")[-1]
    return out.replace("_", " ")


def _policy_label(policy: str) -> str:
    if policy == "min_cost_lp":
        return "Min-cost LP"
    if policy == "backpressure":
        return "Backpressure"
    return policy


def _metric_fmt(metric: str, value):
    if value is None:
        return "-"
    if "ratio" in metric or "rate" in metric:
        return f"{float(value):.3f}"
    if "latency" in metric:
        return f"{float(value):.2f}"
    return f"{float(value):.2f}"


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_architecture(out_path: Path):
    fig, ax = plt.subplots(figsize=(15.0, 4.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    nodes = [
        ("Video Window", "#dbeafe"),
        ("Event Candidates", "#bfdbfe"),
        ("Classifier Fusion", "#93c5fd"),
        ("Context Markov", "#60a5fa"),
        ("Event Messages", "#38bdf8"),
        ("Routing Policy", "#22d3ee"),
        ("Fusion + KPIs", "#67e8f9"),
    ]
    xs = np.linspace(0.08, 0.92, len(nodes))
    ys = [0.66, 0.33, 0.66, 0.33, 0.66, 0.33, 0.66]
    w = 0.12
    h = 0.18

    for i, ((label, color), x, y) in enumerate(zip(nodes, xs, ys)):
        box = FancyBboxPatch(
            (x - w / 2.0, y - h / 2.0),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#0f172a",
            facecolor=color,
            alpha=0.95,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, color="#0f172a")
        if i < len(nodes) - 1:
            x2 = xs[i + 1]
            y2 = ys[i + 1]
            arrow = FancyArrowPatch(
                (x + w / 2.0, y),
                (x2 - w / 2.0, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color="#1e3a8a",
                connectionstyle="arc3,rad=0.05",
            )
            ax.add_patch(arrow)

    ax.text(
        0.5,
        0.94,
        "Perception -> Temporal Filtering -> Network Coordination",
        ha="center",
        va="center",
        fontsize=14,
        color="#0b1f3a",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "Context-conditioned transitions and routing policy are evaluated online for each observation step.",
        ha="center",
        va="center",
        fontsize=10,
        color="#1f2937",
    )
    _save(fig, out_path)


def plot_network_topology(network_cfg, out_path: Path):
    nodes = network_cfg.get("nodes", [])
    edges = network_cfg.get("edges", [])
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.axis("off")

    pos = {
        "Cam1": (-2.2, 1.2),
        "Cam2": (-0.8, 1.2),
        "Cam3": (0.8, 1.2),
        "Cam4": (2.2, 1.2),
        "EdgeNorth": (-1.1, 0.0),
        "EdgeSouth": (1.1, 0.0),
        "Fusion": (0.0, -1.2),
    }
    if any(n not in pos for n in nodes):
        angle = np.linspace(0, 2 * np.pi, num=len(nodes), endpoint=False)
        pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, angle)}

    edge_pairs = {(e["src"], e["dst"]) for e in edges}
    max_capacity = max([int(e.get("capacity", 1)) for e in edges] + [1])

    for e in edges:
        src = e["src"]
        dst = e["dst"]
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        has_reverse = (dst, src) in edge_pairs
        rad = 0.12 if (has_reverse and src < dst) else (-0.12 if has_reverse else 0.0)
        width = 0.8 + 2.2 * (float(e.get("capacity", 1)) / max_capacity)
        color = "#475569"
        if "Fusion" in (src, dst):
            color = "#1d4ed8"
        elif src.startswith("Cam") and dst.startswith("Edge"):
            color = "#0f766e"
        elif src.startswith("Cam") and dst.startswith("Cam"):
            color = "#7c2d12"
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            alpha=0.8,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

    for n in nodes:
        x, y = pos[n]
        if n.startswith("Cam"):
            fc = "#bfdbfe"
            ec = "#1d4ed8"
        elif n.startswith("Edge"):
            fc = "#bbf7d0"
            ec = "#15803d"
        else:
            fc = "#fde68a"
            ec = "#b45309"
        ax.scatter([x], [y], s=1500, color=fc, edgecolor=ec, linewidths=2.2, zorder=3)
        ax.text(x, y, n, ha="center", va="center", fontsize=10, color="#0f172a", zorder=4)

    ax.text(
        0.0,
        2.02,
        "Surveillance Mesh Topology",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#0b1f3a",
    )
    ax.text(
        0.0,
        -2.1,
        "Edge colors: camera->edge uplinks (teal), edge/fusion backbone (blue), camera handoff links (brown).",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
    )
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.35, 2.2)
    _save(fig, out_path)


def _context_spans(context_series):
    if not context_series:
        return []
    spans = []
    start = 0
    curr = context_series[0]
    for i, c in enumerate(context_series[1:], start=1):
        if c != curr:
            spans.append((start, i - 1, curr))
            start = i
            curr = c
    spans.append((start, len(context_series) - 1, curr))
    return spans


def plot_backlog_timeline(single_run, out_path: Path):
    results = single_run.get("results", {})
    fig, ax = plt.subplots(figsize=(12.8, 5.4))

    policy_colors = {
        "min_cost_lp": "#1d4ed8",
        "backpressure": "#b91c1c",
    }
    context_colors = {
        "normal": "#93c5fd",
        "crowded": "#fdba74",
        "uplink_loss": "#fecaca",
        "cam2_failure": "#fca5a5",
    }

    context_series = None
    for policy, payload in results.items():
        ts = payload.get("timeseries", {})
        x = np.arange(len(ts.get("backlog", [])))
        y = np.asarray(ts.get("backlog", []), dtype=float)
        if context_series is None:
            context_series = list(ts.get("context", []))
        ax.plot(
            x,
            y,
            label=_policy_label(policy),
            linewidth=2.2,
            color=policy_colors.get(policy, "#334155"),
            alpha=0.95,
        )

    for start, end, ctx in _context_spans(context_series):
        c = context_colors.get(ctx, "#cbd5e1")
        ax.axvspan(start, end, color=c, alpha=0.18)
        xm = 0.5 * (start + end)
        ax.text(
            xm,
            ax.get_ylim()[1] * 0.97,
            ctx,
            ha="center",
            va="top",
            fontsize=9,
            color="#334155",
        )
        ax.axvline(start, color="#64748b", alpha=0.22, linewidth=1.1)

    ax.set_title("Queue Backlog Under Context Stress", fontsize=14, color="#0b1f3a", fontweight="bold")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Total backlog packets")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left")
    _save(fig, out_path)


def plot_markov_heatmaps(markov_cfg, out_path: Path):
    states = [_short_state_name(s) for s in markov_cfg.get("states", [])]
    state_labels = [textwrap.fill(s, width=11) for s in states]
    ctx_mats = markov_cfg.get("context_transitions", {})
    contexts = list(ctx_mats.keys())
    if len(contexts) == 0:
        return
    contexts = contexts[: min(4, len(contexts))]

    fig, axes = plt.subplots(1, len(contexts), figsize=(4.9 * len(contexts), 4.3))
    if len(contexts) == 1:
        axes = [axes]

    im = None
    for idx, (ax, ctx) in enumerate(zip(axes, contexts)):
        m = np.asarray(ctx_mats[ctx], dtype=float)
        im = ax.imshow(m, vmin=0.0, vmax=1.0, cmap="YlGnBu")
        ax.set_title(ctx, fontsize=12, color="#0f172a")
        ax.set_xticks(np.arange(len(states)))
        ax.set_yticks(np.arange(len(states)))
        ax.set_xticklabels(state_labels, rotation=22, ha="right", fontsize=8)
        if idx == 0:
            ax.set_yticklabels(state_labels, fontsize=8)
            ax.set_ylabel("Current state")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                color = "white" if m[i, j] >= 0.55 else "#111827"
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
        ax.set_xlabel("Next state")
    fig.subplots_adjust(wspace=0.3, bottom=0.2, top=0.83)

    fig.suptitle("Context-Specific Transition Matrices", fontsize=14, color="#0b1f3a", fontweight="bold")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Transition probability")
    _save(fig, out_path)


def plot_policy_kpis(summary_payload, out_path: Path):
    policies = summary_payload.get("summary", {}).get("policies", {})
    ranking_rows = summary_payload.get("ranking", {}).get("ranking", [])
    if len(policies) == 0:
        return

    ranked = [r["policy"] for r in ranking_rows if "policy" in r]
    policy_names = ranked + [p for p in sorted(policies.keys()) if p not in ranked]

    raw = np.full((len(METRIC_CONFIG), len(policy_names)), np.nan, dtype=float)
    norm = np.full_like(raw, np.nan)

    for i, (metric, _label, direction) in enumerate(METRIC_CONFIG):
        vals = []
        for j, p in enumerate(policy_names):
            v = policies[p].get(metric, {}).get("mean")
            if v is None:
                continue
            raw[i, j] = float(v)
            vals.append(float(v))
        if len(vals) == 0:
            continue
        lo = min(vals)
        hi = max(vals)
        if hi - lo <= 1e-12:
            norm[i, :] = 1.0
            continue
        for j in range(len(policy_names)):
            v = raw[i, j]
            if np.isnan(v):
                norm[i, j] = 0.5
            elif direction == "max":
                norm[i, j] = (v - lo) / (hi - lo)
            else:
                norm[i, j] = (hi - v) / (hi - lo)

    ranking_scores = {}
    for row in ranking_rows:
        if row.get("score") is not None:
            ranking_scores[row["policy"]] = float(row["score"])
    ranking_values = [ranking_scores.get(p, 0.0) for p in policy_names]

    fig = plt.figure(figsize=(11.8, 7.2))
    gs = GridSpec(2, 1, height_ratios=[3.6, 1.3], hspace=0.36, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(norm, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax0.set_title("Policy KPI Comparison (normalized by direction)", fontsize=14, color="#0b1f3a", fontweight="bold")
    ax0.set_xticks(np.arange(len(policy_names)))
    ax0.set_xticklabels([_policy_label(p) for p in policy_names], fontsize=10)
    ax0.set_yticks(np.arange(len(METRIC_CONFIG)))
    ax0.set_yticklabels([label for _metric, label, _d in METRIC_CONFIG], fontsize=10)

    for i, (metric, _label, _direction) in enumerate(METRIC_CONFIG):
        for j, p in enumerate(policy_names):
            value = raw[i, j]
            text = _metric_fmt(metric, None if np.isnan(value) else value)
            color = "black" if np.isnan(norm[i, j]) or norm[i, j] < 0.58 else "white"
            ax0.text(j, i, text, ha="center", va="center", fontsize=9.5, color=color)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.024, pad=0.015)
    cbar.set_label("Normalized score")

    ax1 = fig.add_subplot(gs[1, 0])
    bar_colors = ["#1d4ed8" if p == "min_cost_lp" else "#b91c1c" for p in policy_names]
    ax1.bar(np.arange(len(policy_names)), ranking_values, color=bar_colors, alpha=0.92)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(np.arange(len(policy_names)))
    ax1.set_xticklabels([_policy_label(p) for p in policy_names], fontsize=10)
    ax1.set_ylabel("Composite score")
    ax1.set_title("Weighted ranking score", fontsize=12, color="#0f172a")
    ax1.grid(axis="y", alpha=0.2, linestyle="--")
    for j, val in enumerate(ranking_values):
        ax1.text(j, val + 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    _save(fig, out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build presentation visual assets")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--single-run-json",
        default="logs/network_message_passing/surveillance_single.json",
        help="Path to single-run simulator output JSON.",
    )
    parser.add_argument(
        "--summary-json",
        default="logs/network_message_passing/monte_carlo_summary.json",
        help="Path to Monte Carlo summary JSON.",
    )
    parser.add_argument(
        "--network-config",
        default="data/network_message_passing/surveillance_mesh.json",
        help="Path to surveillance network config JSON.",
    )
    parser.add_argument(
        "--markov-config",
        default="data/taxonomy/example_markov_chain.json",
        help="Path to Markov config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/assets/networked_contextual_hermes",
        help="Directory for generated presentation images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.repo_root).resolve()

    single_run_path = _resolve(root, args.single_run_json)
    summary_path = _resolve(root, args.summary_json)
    network_cfg_path = _resolve(root, args.network_config)
    markov_cfg_path = _resolve(root, args.markov_config)
    out_dir = _resolve(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )

    plot_architecture(out_dir / "architecture_pipeline.png")

    if network_cfg_path.exists():
        network_cfg = _load_json(network_cfg_path)
        plot_network_topology(network_cfg, out_dir / "surveillance_topology.png")
    else:
        print(f"[warn] Missing network config: {network_cfg_path}")

    if markov_cfg_path.exists():
        markov_cfg = _load_json(markov_cfg_path)
        plot_markov_heatmaps(markov_cfg, out_dir / "markov_context_heatmaps.png")
    else:
        print(f"[warn] Missing Markov config: {markov_cfg_path}")

    if single_run_path.exists():
        single = _load_json(single_run_path)
        plot_backlog_timeline(single, out_dir / "backlog_context_timeline.png")
    else:
        print(f"[warn] Missing single-run JSON: {single_run_path}")

    if summary_path.exists():
        summary = _load_json(summary_path)
        plot_policy_kpis(summary, out_dir / "policy_kpi_comparison.png")
    else:
        print(f"[warn] Missing summary JSON: {summary_path}")

    print(f"Generated presentation visuals in: {out_dir}")


if __name__ == "__main__":
    main()
