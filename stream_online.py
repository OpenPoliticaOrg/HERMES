"""
Online streaming inference for event observations + Markov updates.

Example:
python stream_online.py \
  --cfg-path lavis/projects/hermes/cls_coin.yaml \
  --video-source 0 \
  --question "what is the activity in the video?" \
  --sequence-id cam0 \
  --options \
    run.classification_mode rank \
    run.event_taxonomy_path data/taxonomy/example_event_taxonomy.json \
    run.observation_classifier_path data/taxonomy/example_observation_classifiers.json \
    run.markov_chain_path data/taxonomy/example_markov_chain.json
"""

import argparse
import json
import logging
from collections import deque

import numpy as np
import torch

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.entity_observation_adapter import MotionEntityObservationAdapter
from lavis.datasets.data_utils import prepare_sample

# imports modules for registration
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.models import *  # noqa: F401,F403
from lavis.processors import *  # noqa: F401,F403
from lavis.runners import *  # noqa: F401,F403
from lavis.tasks import *  # noqa: F401,F403


def _unique_labels(values):
    seen = set()
    out = []
    for value in values:
        if value is None:
            continue
        label = str(value).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


class InteractiveContextController:
    def __init__(
        self,
        context_options=None,
        default_context=None,
        schedule_enabled=False,
    ):
        self.context_options = _unique_labels(context_options or [])
        if default_context:
            self.context_options = _unique_labels(
                [default_context] + self.context_options
            )

        self.default_context = default_context
        self.schedule_enabled = bool(schedule_enabled)
        self.mode = "auto" if self.schedule_enabled else "manual"
        self.current_context = str(default_context) if default_context else None
        if self.current_context is None and self.context_options:
            self.current_context = self.context_options[0]

        self.should_quit = False
        self.last_key = None

    def _add_context_if_new(self, context):
        if context is None:
            return
        label = str(context).strip()
        if not label:
            return
        if label not in self.context_options:
            self.context_options.append(label)

    def set_manual_context(self, context):
        self._add_context_if_new(context)
        if context is not None:
            self.current_context = str(context)
        self.mode = "manual"

    def set_context_by_index(self, index):
        if index < 0 or index >= len(self.context_options):
            return False
        self.current_context = self.context_options[index]
        self.mode = "manual"
        return True

    def step_context(self, delta):
        if len(self.context_options) == 0:
            return None
        if self.current_context not in self.context_options:
            self.current_context = self.context_options[0]
            self.mode = "manual"
            return self.current_context

        idx = self.context_options.index(self.current_context)
        next_idx = (idx + int(delta)) % len(self.context_options)
        self.current_context = self.context_options[next_idx]
        self.mode = "manual"
        return self.current_context

    def toggle_auto_manual(self):
        if self.mode == "auto":
            self.mode = "manual"
        else:
            self.mode = "auto"
        return self.mode

    def resolve_for_window(self, scheduled_context):
        self._add_context_if_new(scheduled_context)

        if self.mode == "auto":
            if scheduled_context is not None:
                self.current_context = str(scheduled_context)
                return self.current_context, "auto_schedule"
            if self.default_context is not None:
                self.current_context = str(self.default_context)
                return self.current_context, "auto_default"
            if self.current_context is not None:
                return self.current_context, "auto_hold"
            if self.context_options:
                self.current_context = self.context_options[0]
                return self.current_context, "auto_options"
            return None, "auto_none"

        if self.current_context is None:
            if scheduled_context is not None:
                self.current_context = str(scheduled_context)
            elif self.default_context is not None:
                self.current_context = str(self.default_context)
            elif self.context_options:
                self.current_context = self.context_options[0]
        return self.current_context, "manual"

    def current_index(self):
        if self.current_context is None:
            return None
        if self.current_context not in self.context_options:
            return None
        return self.context_options.index(self.current_context)

    def options_as_text(self, max_items=8):
        if not self.context_options:
            return "-"
        rendered = []
        for idx, label in enumerate(self.context_options[:max_items]):
            marker = "*" if label == self.current_context else " "
            rendered.append(f"{idx + 1}:{label}{marker}")
        if len(self.context_options) > max_items:
            rendered.append("...")
        return ", ".join(rendered)

    def on_keypress(self, key):
        if key is None:
            return
        key = str(key).lower()
        self.last_key = key

        if key in ("q", "escape"):
            self.should_quit = True
            return

        if key in ("a",):
            self.toggle_auto_manual()
            return

        if key in ("left", "["):
            self.step_context(-1)
            return

        if key in ("right", "]"):
            self.step_context(1)
            return

        if key in ("0",):
            if len(self.context_options) > 0:
                self.set_context_by_index(0)
            return

        if key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            self.set_context_by_index(int(key) - 1)


class LiveMarkovVisualizer:
    def __init__(
        self,
        max_classes=5,
        max_states=8,
        context_field="ecological_context",
        context_controller=None,
        max_entity_rows=10,
        max_entity_windows=32,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm

        self.max_classes = max_classes
        self.max_states = max_states
        self.context_field = context_field
        self.context_controller = context_controller
        self.max_entity_rows = max(1, int(max_entity_rows))
        self.max_entity_windows = max(4, int(max_entity_windows))
        self.plt = plt
        self.plt.ion()
        self.ListedColormap = ListedColormap
        self.BoundaryNorm = BoundaryNorm

        self.entity_window_states = {}
        self.entity_ids_seen = set()
        self.entity_code_colors = {
            0: "#FFFFFF",  # not tracked
            1: "#D9D9D9",  # inactive
            2: "#BFD8FF",  # active hold (not observed this window)
            3: "#3B82F6",  # active observed
            4: "#22C55E",  # entered
            5: "#06B6D4",  # re-entered
            6: "#EF4444",  # exited
        }
        self.entity_code_labels = {
            1: "inactive",
            2: "active_hold",
            3: "active_observed",
            4: "entered",
            5: "reentered",
            6: "exited",
        }

        self.fig = self.plt.figure(figsize=(14, 9))
        grid = self.fig.add_gridspec(
            3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.38, wspace=0.28
        )
        self.ax_class = self.fig.add_subplot(grid[0, 0])
        self.ax_state = self.fig.add_subplot(grid[0, 1])
        self.ax_matrix = self.fig.add_subplot(grid[1, 0])
        self.ax_info = self.fig.add_subplot(grid[1, 1])
        self.ax_entity = self.fig.add_subplot(grid[2, :])
        self.matrix_cbar = None
        self.entity_cbar = None
        self.fig.canvas.manager.set_window_title("HERMES Live Markov Monitor")
        if self.context_controller is not None:
            self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event):
        if self.context_controller is None:
            return
        self.context_controller.on_keypress(event.key)

    @staticmethod
    def _truncate(text, n=28):
        text = str(text)
        if len(text) <= n:
            return text
        return text[: n - 3] + "..."

    def _render_entity_ids(self, values, max_items=3, width=12):
        values = values or []
        if len(values) == 0:
            return "-"
        rendered = [self._truncate(v, n=width) for v in values[:max_items]]
        if len(values) > max_items:
            rendered.append("...")
        return ", ".join(rendered)

    def _extract_transition(self, result):
        debug = result.get("markov_debug") or {}
        if "last_transition" in debug and debug["last_transition"] is not None:
            return np.array(debug["last_transition"], dtype=float)
        window = debug.get("window") or {}
        trace = window.get("window_trace") or []
        if len(trace) > 0 and trace[-1].get("transition") is not None:
            return np.array(trace[-1]["transition"], dtype=float)
        return None

    @staticmethod
    def _sorted_entities_for_display(entity_set):
        return sorted(entity_set, key=lambda x: str(x))

    def _update_entity_timeline_state(self, result):
        window_index = result.get("window_index")
        if window_index is None:
            return
        try:
            window_index = int(window_index)
        except Exception:
            return

        lifecycle = result.get("entity_lifecycle") or {}
        if not isinstance(lifecycle, dict):
            lifecycle = {}

        entered = set(lifecycle.get("entered_entities") or [])
        reentered = set(lifecycle.get("reentered_entities") or [])
        exited = set(lifecycle.get("exited_entities") or [])
        active = set(lifecycle.get("active_entities") or [])
        inactive = set(lifecycle.get("inactive_entities") or [])
        observed = set(lifecycle.get("observed_entities") or [])

        entities = set()
        entities.update(entered)
        entities.update(reentered)
        entities.update(exited)
        entities.update(active)
        entities.update(inactive)
        entities.update(observed)

        for item in result.get("entity_event_sequences") or []:
            if not isinstance(item, dict):
                continue
            entity_id = item.get("entity_id")
            if entity_id is not None:
                entities.add(str(entity_id))

        if len(entities) == 0:
            self.entity_window_states[window_index] = {}
            return

        state_map = {}
        for entity_id in entities:
            entity_id = str(entity_id)
            if entity_id in exited:
                code = 6
            elif entity_id in reentered:
                code = 5
            elif entity_id in entered:
                code = 4
            elif entity_id in active and entity_id in observed:
                code = 3
            elif entity_id in active:
                code = 2
            elif entity_id in inactive:
                code = 1
            else:
                code = 0
            state_map[entity_id] = code
            self.entity_ids_seen.add(entity_id)

        self.entity_window_states[window_index] = state_map

        if len(self.entity_window_states) > self.max_entity_windows * 3:
            keep_from = max(window_index - self.max_entity_windows * 2, 0)
            self.entity_window_states = {
                k: v for k, v in self.entity_window_states.items() if int(k) >= keep_from
            }

    def _draw_entity_timeline(self, result):
        self._update_entity_timeline_state(result)
        self.ax_entity.clear()
        if self.entity_cbar is not None:
            try:
                self.entity_cbar.remove()
            except Exception:
                pass
            self.entity_cbar = None

        window_index = result.get("window_index")
        if window_index is None:
            self.ax_entity.axis("off")
            self.ax_entity.set_title("Entity Trajectories")
            return
        try:
            window_index = int(window_index)
        except Exception:
            self.ax_entity.axis("off")
            self.ax_entity.set_title("Entity Trajectories")
            return

        all_windows = sorted(self.entity_window_states.keys())
        if len(all_windows) == 0:
            self.ax_entity.axis("off")
            self.ax_entity.set_title("Entity Trajectories")
            return

        min_window = max(window_index - self.max_entity_windows + 1, 0)
        windows = [w for w in all_windows if int(w) >= min_window and int(w) <= window_index]
        if len(windows) == 0:
            windows = [window_index]

        entities = self._sorted_entities_for_display(self.entity_ids_seen)
        if len(entities) == 0:
            self.ax_entity.axis("off")
            self.ax_entity.set_title("Entity Trajectories")
            return

        active_entities = result.get("entity_lifecycle", {}).get("active_entities") or []
        active_set = set([str(x) for x in active_entities])
        entities = sorted(
            entities,
            key=lambda eid: (0 if eid in active_set else 1, str(eid)),
        )[: self.max_entity_rows]

        mat = np.zeros((len(entities), len(windows)), dtype=float)
        for wi, w in enumerate(windows):
            state_map = self.entity_window_states.get(int(w), {})
            for ei, entity_id in enumerate(entities):
                mat[ei, wi] = float(state_map.get(entity_id, 0))

        colors = [self.entity_code_colors[idx] for idx in sorted(self.entity_code_colors.keys())]
        cmap = self.ListedColormap(colors)
        boundaries = np.arange(-0.5, len(colors) + 0.5, 1.0)
        norm = self.BoundaryNorm(boundaries, cmap.N)

        im = self.ax_entity.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        self.ax_entity.set_title("Entity Trajectories (Window x Entity)")
        self.ax_entity.set_xlabel("Window Index")
        self.ax_entity.set_ylabel("Entity")
        self.ax_entity.set_yticks(np.arange(len(entities)))
        self.ax_entity.set_yticklabels([self._truncate(x, n=18) for x in entities], fontsize=8)

        x_tick_step = max(1, int(np.ceil(len(windows) / 8.0)))
        x_tick_idx = list(range(0, len(windows), x_tick_step))
        if len(windows) > 0 and (len(windows) - 1) not in x_tick_idx:
            x_tick_idx.append(len(windows) - 1)
        self.ax_entity.set_xticks(x_tick_idx)
        self.ax_entity.set_xticklabels([str(windows[idx]) for idx in x_tick_idx], fontsize=8)

        legend_codes = [4, 5, 3, 2, 6, 1]
        legend_labels = [self.entity_code_labels[c] for c in legend_codes]
        ticks = [float(c) for c in legend_codes]
        self.entity_cbar = self.fig.colorbar(im, ax=self.ax_entity, fraction=0.024, pad=0.01, ticks=ticks)
        self.entity_cbar.ax.set_yticklabels(legend_labels, fontsize=7)

    def update(self, result):
        event_predictions = (result.get("event_predictions") or [])[: self.max_classes]
        class_labels = [self._truncate(x.get("label", "")) for x in event_predictions]
        class_scores = [float(x.get("confidence", 0.0)) for x in event_predictions]

        markov_posterior = (result.get("markov_posterior") or [])[: self.max_states]
        state_labels = [self._truncate(x.get("event_id", "")) for x in markov_posterior]
        state_scores = [float(x.get("prob", 0.0)) for x in markov_posterior]

        self.ax_class.clear()
        if len(class_labels) > 0:
            self.ax_class.barh(class_labels[::-1], class_scores[::-1], color="#4C78A8")
            self.ax_class.set_xlim(0.0, 1.0)
        self.ax_class.set_title("Top Classifications")
        self.ax_class.set_xlabel("Confidence")

        self.ax_state.clear()
        if len(state_labels) > 0:
            self.ax_state.barh(state_labels[::-1], state_scores[::-1], color="#F58518")
            self.ax_state.set_xlim(0.0, 1.0)
        self.ax_state.set_title("Markov Posterior")
        self.ax_state.set_xlabel("Probability")

        transition = self._extract_transition(result)
        self.ax_matrix.clear()
        if self.matrix_cbar is not None:
            try:
                self.matrix_cbar.remove()
            except Exception:
                pass
            self.matrix_cbar = None
        if transition is not None and transition.ndim == 2:
            im = self.ax_matrix.imshow(transition, cmap="Blues", vmin=0.0, vmax=1.0)
            self.ax_matrix.set_title("Transition Matrix (Current Step)")
            self.ax_matrix.set_xlabel("To State")
            self.ax_matrix.set_ylabel("From State")
            for i in range(transition.shape[0]):
                for j in range(transition.shape[1]):
                    self.ax_matrix.text(
                        j,
                        i,
                        f"{transition[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=7,
                    )
            # Avoid stacking multiple colorbars by recreating figure-level colorbar per update.
            # Matplotlib handles replacing safely in interactive mode.
            self.matrix_cbar = self.fig.colorbar(
                im, ax=self.ax_matrix, fraction=0.046, pad=0.04
            )
        else:
            self.ax_matrix.axis("off")
            self.ax_matrix.text(
                0.5,
                0.5,
                "Enable --debug-markov to view matrix trace",
                ha="center",
                va="center",
            )

        self.ax_info.clear()
        self.ax_info.axis("off")
        debug = result.get("markov_debug") or {}
        te_debug = debug.get("transfer_entropy") or {}
        te_val = te_debug.get("value")
        if te_val is None:
            te_text = "-"
        else:
            te_text = f"{float(te_val):.4f} bits"
        lines = [
            f"Sequence: {result.get('sequence_id')}",
            f"Window: {result.get('window_index')}",
            f"Frame Index: {result.get('frame_index')}",
            f"Context: {result.get(self.context_field)}",
            f"Context Mode: {result.get('context_mode', '-')}",
            f"Context Source: {result.get('context_source', '-')}",
            f"Entity Obs Mode: {result.get('entity_observation_mode', '-')}",
            f"Entity Obs Source: {result.get('entity_observation_source', '-')}",
            f"Top Class: {class_labels[0] if class_labels else '-'}",
            f"Top State: {result.get('markov_state', {}).get('event_id', '-')}",
            f"Transition Mode: {debug.get('transition_mode', '-')}",
            f"Markov Order: {debug.get('markov_order', '-')}",
            f"TE: {te_text}",
            f"Step: {debug.get('step_idx', '-')}",
        ]
        entity_sequences = result.get("entity_event_sequences") or []
        lines.append(f"Entities Updated: {len(entity_sequences)}")
        for entity_item in entity_sequences[:3]:
            entity_state = entity_item.get("markov_state") or {}
            entity_name = self._truncate(entity_item.get("entity_id", "-"), n=16)
            state_label = self._truncate(entity_state.get("event_id", "-"), n=24)
            state_prob = entity_state.get("prob", None)
            if state_prob is None:
                lines.append(f"  {entity_name}: {state_label}")
            else:
                lines.append(f"  {entity_name}: {state_label} ({float(state_prob):.2f})")
        if len(entity_sequences) > 3:
            lines.append("  ...")
        entity_lifecycle = result.get("entity_lifecycle") or {}
        if isinstance(entity_lifecycle, dict) and len(entity_lifecycle) > 0:
            entered = entity_lifecycle.get("entered_entities") or []
            reentered = entity_lifecycle.get("reentered_entities") or []
            exited = entity_lifecycle.get("exited_entities") or []
            active_count = entity_lifecycle.get("active_count", 0)
            tracked_count = entity_lifecycle.get("total_tracked_entities", 0)
            lines.append(
                f"Lifecycle: +{len(entered)} re+{len(reentered)} -{len(exited)} "
                f"(active {active_count}/{tracked_count})"
            )
            lines.append(f"Entered: {self._render_entity_ids(entered)}")
            lines.append(f"Exited: {self._render_entity_ids(exited)}")
        if self.context_controller is not None:
            lines.extend(
                [
                    f"Contexts: {self.context_controller.options_as_text()}",
                    "Keys: [ ]/arrow switch, 1-9 set, a auto/manual, q quit",
                ]
            )
        self.ax_info.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace")
        self.ax_info.set_title("Step Diagnostics")

        self._draw_entity_timeline(result)
        self.fig.tight_layout()
        self.plt.pause(0.001)

    def close(self):
        try:
            self.plt.close(self.fig)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Online streaming event inference")
    parser.add_argument("--cfg-path", required=True, help="Path to project config YAML.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Config overrides in key value format, same as train.py.",
    )
    parser.add_argument(
        "--video-source",
        required=True,
        help="Camera index (e.g. 0) or video file path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional finetuned checkpoint path to load after model construction.",
    )
    parser.add_argument(
        "--question",
        default="what is the activity in the video?",
        help="Observation question passed as text_input.",
    )
    parser.add_argument(
        "--sequence-id",
        default="stream0",
        help="Sequence identifier used for online Markov state.",
    )
    parser.add_argument(
        "--context-field",
        default="ecological_context",
        help="Sample field name for ecological context metadata.",
    )
    parser.add_argument(
        "--ecological-context",
        default=None,
        help="Default ecological context label (e.g., kitchen, street, forest).",
    )
    parser.add_argument(
        "--ecological-context-by-window",
        default=None,
        help=(
            "Optional JSON file mapping start window index to context label. "
            "Example: {\"0\":\"kitchen\", \"30\":\"street\"}"
        ),
    )
    parser.add_argument(
        "--entity-observations-by-window",
        default=None,
        help=(
            "Optional JSON file mapping window index to entity observations. "
            "Each entry may be a list of entity dicts or {\"entities\": [...]}."
        ),
    )
    parser.add_argument(
        "--entity-observation-mode",
        choices=["none", "schedule", "auto_motion"],
        default="none",
        help=(
            "Entity observation source mode. "
            "`schedule` uses --entity-observations-by-window; "
            "`auto_motion` extracts moving blobs from video windows."
        ),
    )
    parser.add_argument(
        "--entity-motion-min-area",
        type=int,
        default=500,
        help="Minimum contour area (px) for auto-motion entities.",
    )
    parser.add_argument(
        "--entity-motion-iou-threshold",
        type=float,
        default=0.25,
        help="IoU threshold for auto-motion track association.",
    )
    parser.add_argument(
        "--entity-motion-max-tracks",
        type=int,
        default=12,
        help="Maximum active auto-motion tracks.",
    )
    parser.add_argument(
        "--entity-motion-max-missed",
        type=int,
        default=3,
        help="Maximum missed windows before auto-motion track removal.",
    )
    parser.add_argument(
        "--context-options",
        default=None,
        help=(
            "Optional comma-separated context labels for interactive switching. "
            "Example: kitchen,street,office"
        ),
    )
    parser.add_argument(
        "--interactive-context",
        action="store_true",
        help=(
            "Enable keyboard-based context switching in the matplotlib window "
            "(requires --visualize matplotlib)."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="Temporal chunk duration (seconds) per inference window.",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=1.0,
        help="Stride duration (seconds) between successive online inferences.",
    )
    parser.add_argument(
        "--fps-override",
        type=float,
        default=0.0,
        help="Optional FPS override if capture metadata is unreliable.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=-1,
        help="Stop after this many windows; -1 means run until stream ends.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional output JSONL file to append streaming results.",
    )
    parser.add_argument(
        "--debug-markov",
        action="store_true",
        help="Include per-step Markov internals (T_t, priors, likelihood, posterior).",
    )
    parser.add_argument(
        "--markov-order-override",
        type=int,
        default=0,
        help="Optional override for Markov memory order (>=1). 0 keeps config value.",
    )
    parser.add_argument(
        "--markov-window-size-override",
        type=int,
        default=-1,
        help="Optional override for Markov sliding window size (>=0). -1 keeps config value.",
    )
    parser.add_argument(
        "--te-mode-override",
        choices=["none", "symbolic_matrix"],
        default=None,
        help="Optional override for transfer entropy mode.",
    )
    parser.add_argument(
        "--te-target-order-override",
        type=int,
        default=0,
        help="Optional override for TE target order (>=1). 0 keeps config value.",
    )
    parser.add_argument(
        "--te-source-order-override",
        type=int,
        default=0,
        help="Optional override for TE source order (>=1). 0 keeps config value.",
    )
    parser.add_argument(
        "--visualize",
        choices=["none", "matplotlib"],
        default="none",
        help="Live visualization mode.",
    )
    parser.add_argument(
        "--entity-missing-tolerance",
        type=int,
        default=0,
        help=(
            "Entity exits after this many consecutive unobserved windows "
            "(0 exits immediately on first missed window)."
        ),
    )
    return parser.parse_args()


def _parse_video_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def _frame_to_tensor(frame_bgr, cv2_module):
    frame_rgb = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2RGB)
    # (H, W, C) uint8 -> (C, H, W) float32
    return torch.from_numpy(frame_rgb).permute(2, 0, 1).float()


def _sample_evenly(frames, num_frames):
    if len(frames) == 0:
        raise ValueError("Cannot sample from empty frame list.")
    if len(frames) == 1:
        return [frames[0]] * num_frames
    idxs = np.rint(np.linspace(0, len(frames) - 1, num_frames)).astype(int).tolist()
    return [frames[idx] for idx in idxs]


def _choose_eval_dataset(dataset_splits):
    if "test" in dataset_splits:
        return dataset_splits["test"]
    if "val" in dataset_splits:
        return dataset_splits["val"]
    if "train" in dataset_splits:
        return dataset_splits["train"]
    raise RuntimeError("No usable dataset split found for processor extraction.")


def _load_context_schedule(path):
    if not path:
        return []
    with open(path, "r") as fp:
        payload = json.load(fp)
    schedule = []
    for k, v in payload.items():
        schedule.append((int(k), str(v)))
    schedule.sort(key=lambda x: x[0])
    return schedule


def _resolve_context_label(default_context, schedule, window_idx):
    chosen = default_context
    for start_window, label in schedule:
        if window_idx >= start_window:
            chosen = label
    return chosen


def _load_markov_context_options(markov_chain_path):
    if not markov_chain_path:
        return []
    try:
        with open(markov_chain_path, "r") as fp:
            payload = json.load(fp)
    except Exception as exc:
        logging.warning(
            f"Failed to parse Markov config at {markov_chain_path}: {exc}. "
            "Skipping Markov context option discovery."
        )
        return []

    values = []
    context_transitions = payload.get("context_transitions", {})
    context_initial = payload.get("context_initial", {})
    values.extend(context_transitions.keys())
    values.extend(context_initial.keys())

    context_fallback = payload.get("context_fallback")
    if context_fallback:
        values.append(context_fallback)
    return _unique_labels(values)


def _load_entity_observation_schedule(path):
    if not path:
        return {}
    with open(path, "r") as fp:
        payload = json.load(fp)

    out = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            out[int(key)] = value
        return out

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "window_index" not in item:
                continue
            out[int(item["window_index"])] = item.get("entities", item)
        return out

    return out


def _resolve_entity_observations(schedule, window_idx, default_empty=False):
    if not schedule:
        return None
    value = schedule.get(int(window_idx))
    if value is None:
        return [] if default_empty else None
    if isinstance(value, dict):
        entities = value.get("entities")
        if isinstance(entities, list):
            return entities
        return [value]
    if isinstance(value, list):
        return value
    return None


def _parse_context_options(raw_value):
    if not raw_value:
        return []
    values = [token.strip() for token in str(raw_value).split(",")]
    return _unique_labels(values)


def _build_context_options(args, cfg, context_schedule):
    options = []
    options.extend(_parse_context_options(args.context_options))
    if args.ecological_context is not None:
        options.append(args.ecological_context)
    options.extend([label for _, label in context_schedule])

    markov_chain_path = cfg.run_cfg.get("markov_chain_path", None)
    options.extend(_load_markov_context_options(markov_chain_path))
    return _unique_labels(options)


def _safe_int(value):
    try:
        out = int(value)
    except Exception:
        return None
    if out <= 0:
        return None
    return out


def _resolve_num_model_frames(cfg, dataset_name, default_value=8):
    dataset_cfg = cfg.datasets_cfg[dataset_name]
    candidates = [
        cfg.model_cfg.get("num_frames", None),
        cfg.model_cfg.get("num_frms", None),
        dataset_cfg.get("num_frames", None),
        dataset_cfg.get("num_frms", None),
    ]
    for value in candidates:
        parsed = _safe_int(value)
        if parsed is not None:
            dataset_cfg["num_frames"] = int(parsed)
            return int(parsed)

    fallback = int(default_value)
    logging.warning(
        "Could not infer num_frames from model/dataset config. "
        f"Falling back to {fallback} frames."
    )
    dataset_cfg["num_frames"] = fallback
    return fallback


def main():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for stream_online.py. Install with `pip install opencv-python`."
        ) from exc

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    cfg = Config(args)
    dataset_name = list(cfg.datasets_cfg.keys())[0]
    num_model_frames = _resolve_num_model_frames(cfg, dataset_name)
    if not str(cfg.model_cfg.arch).endswith("_hermes"):
        cfg.model_cfg.arch += "_hermes"

    # Force rank mode for online event filtering.
    cfg.run_cfg.classification_mode = "rank"
    cfg.run_cfg.markov_debug = bool(args.debug_markov or args.visualize != "none")

    task = tasks.setup_task(cfg)
    if hasattr(task, "entity_sequence_missing_tolerance"):
        task.entity_sequence_missing_tolerance = max(0, int(args.entity_missing_tolerance))
    if (
        hasattr(task, "entity_sequence_tracker")
        and task.entity_sequence_tracker is not None
        and hasattr(task.entity_sequence_tracker, "default_missing_tolerance")
    ):
        task.entity_sequence_tracker.default_missing_tolerance = max(
            0, int(args.entity_missing_tolerance)
        )
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)

    chain = getattr(task, "event_markov_chain", None)
    if chain is not None:
        if args.markov_order_override >= 1:
            chain.markov_order = int(args.markov_order_override)
            chain.sequence_history = {}
        if args.markov_window_size_override >= 0:
            chain.window_size = int(args.markov_window_size_override)
            chain.sequence_windows = {}
        if args.te_mode_override is not None:
            chain.transfer_entropy_mode = args.te_mode_override
            chain.sequence_symbols = {}
        if args.te_target_order_override >= 1:
            chain.transfer_entropy_target_order = int(args.te_target_order_override)
            chain.sequence_symbols = {}
        if args.te_source_order_override >= 1:
            chain.transfer_entropy_source_order = int(args.te_source_order_override)
            chain.sequence_symbols = {}
        logging.info(
            "Markov runtime settings: "
            f"order={chain.markov_order}, "
            f"window_size={chain.window_size}, "
            f"te_mode={chain.transfer_entropy_mode}, "
            f"te_target_order={chain.transfer_entropy_target_order}, "
            f"te_source_order={chain.transfer_entropy_source_order}"
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device)
    model.eval()

    eval_dataset = _choose_eval_dataset(datasets[dataset_name])
    vis_processor = eval_dataset.vis_processor

    capture = cv2.VideoCapture(_parse_video_source(args.video_source))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.video_source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if args.fps_override > 0:
        fps = float(args.fps_override)
    if fps <= 0:
        fps = 30.0

    chunk_frames = max(1, int(round(args.chunk_seconds * fps)))
    stride_frames = max(1, int(round(args.stride_seconds * fps)))
    frame_buffer = deque(maxlen=chunk_frames)
    frame_buffer_bgr = deque(maxlen=chunk_frames)
    frame_idx = 0
    next_infer_frame = chunk_frames
    window_idx = 0
    context_schedule = _load_context_schedule(args.ecological_context_by_window)
    entity_observation_schedule = _load_entity_observation_schedule(
        args.entity_observations_by_window
    )
    entity_observation_mode = str(args.entity_observation_mode)
    if (
        entity_observation_mode == "none"
        and args.entity_observations_by_window
        and len(entity_observation_schedule) > 0
    ):
        entity_observation_mode = "schedule"
    if entity_observation_mode == "schedule" and len(entity_observation_schedule) == 0:
        logging.warning(
            "Entity observation mode `schedule` requested but no schedule payload was loaded. "
            "Falling back to mode `none`."
        )
        entity_observation_mode = "none"
    motion_adapter = None
    if entity_observation_mode == "auto_motion":
        motion_adapter = MotionEntityObservationAdapter(
            min_area=args.entity_motion_min_area,
            iou_threshold=args.entity_motion_iou_threshold,
            max_tracks=args.entity_motion_max_tracks,
            max_missed=args.entity_motion_max_missed,
        )

    context_options = _build_context_options(args, cfg, context_schedule)
    use_interactive_context = bool(args.interactive_context and args.visualize == "matplotlib")
    if args.interactive_context and args.visualize != "matplotlib":
        logging.warning(
            "--interactive-context requires --visualize matplotlib. "
            "Running without interactive context controls."
        )
    context_controller = None
    if use_interactive_context:
        context_controller = InteractiveContextController(
            context_options=context_options,
            default_context=args.ecological_context,
            schedule_enabled=(len(context_schedule) > 0),
        )
        logging.info(
            "Interactive context controls enabled. "
            "Focus the matplotlib window and use keys: [ ] / arrow, 1-9, a, q."
        )

    fout = open(args.output_jsonl, "a") if args.output_jsonl else None
    visualizer = None
    if args.visualize == "matplotlib":
        try:
            visualizer = LiveMarkovVisualizer(
                context_field=args.context_field,
                context_controller=context_controller,
            )
        except Exception as exc:
            logging.warning(
                f"Failed to initialize matplotlib visualizer ({exc}). Continuing without GUI."
            )
            visualizer = None
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_idx += 1
            frame_buffer.append(_frame_to_tensor(frame_bgr, cv2))
            frame_buffer_bgr.append(frame_bgr.copy())

            if len(frame_buffer) < chunk_frames:
                continue
            if frame_idx < next_infer_frame:
                continue

            next_infer_frame += stride_frames

            sampled = _sample_evenly(list(frame_buffer), num_model_frames)
            video = torch.stack(sampled, dim=1)  # (C, T, H, W)
            video = vis_processor(video)
            scheduled_context = _resolve_context_label(
                args.ecological_context, context_schedule, window_idx
            )
            ecological_context = scheduled_context
            if len(context_schedule) > 0:
                context_source = "schedule"
            elif args.ecological_context is not None:
                context_source = "fixed_default"
            else:
                context_source = "none"

            if context_controller is not None:
                ecological_context, context_source = context_controller.resolve_for_window(
                    scheduled_context
                )

            sample = {
                "image": video.unsqueeze(0),
                "text_input": [args.question],
                "image_id": [f"{args.sequence_id}_{window_idx}"],
            }
            if ecological_context is not None:
                sample[args.context_field] = [ecological_context]
            entity_observations = None
            if entity_observation_mode == "schedule":
                entity_observations = _resolve_entity_observations(
                    entity_observation_schedule,
                    window_idx,
                    default_empty=True,
                )
            elif entity_observation_mode == "auto_motion":
                entity_observations = motion_adapter.observe_window(
                    window_frames_bgr=list(frame_buffer_bgr),
                    window_idx=window_idx,
                    cv2_module=cv2,
                )
            if entity_observations is not None:
                # Batch shape [B], each item is a per-window entity list.
                sample["entity_observations"] = [entity_observations]
            sample = prepare_sample(sample, cuda_enabled=(device.type == "cuda"))

            with torch.no_grad():
                result = task.valid_step(model=model, samples=sample)[0]

            # Enrich with stream metadata.
            result["sequence_id"] = args.sequence_id
            result["window_index"] = window_idx
            result["fps"] = fps
            result["chunk_seconds"] = args.chunk_seconds
            result["stride_seconds"] = args.stride_seconds
            result["frame_index"] = frame_idx
            result[args.context_field] = ecological_context
            if context_controller is not None:
                context_mode = context_controller.mode
            elif len(context_schedule) > 0:
                context_mode = "schedule"
            elif args.ecological_context is not None:
                context_mode = "fixed"
            else:
                context_mode = "none"
            result["context_mode"] = context_mode
            result["context_source"] = context_source
            result["context_options"] = (
                list(context_controller.context_options)
                if context_controller is not None
                else list(context_options)
            )
            result["entity_observation_mode"] = entity_observation_mode
            if entity_observations is not None:
                if entity_observation_mode == "schedule":
                    if len(entity_observations) > 0:
                        result["entity_observation_source"] = "window_schedule"
                    else:
                        result["entity_observation_source"] = "window_schedule_empty"
                elif entity_observation_mode == "auto_motion":
                    if len(entity_observations) > 0:
                        result["entity_observation_source"] = "auto_motion"
                    else:
                        result["entity_observation_source"] = "auto_motion_empty"

            line = json.dumps(result)
            print(line, flush=True)
            if fout is not None:
                fout.write(line + "\n")
                fout.flush()
            if visualizer is not None:
                visualizer.update(result)
            if context_controller is not None and context_controller.should_quit:
                logging.info("Quit requested via keyboard. Stopping stream.")
                break

            window_idx += 1
            if args.max_windows >= 0 and window_idx >= args.max_windows:
                break
    finally:
        capture.release()
        if fout is not None:
            fout.close()
        if visualizer is not None:
            visualizer.close()


if __name__ == "__main__":
    main()
