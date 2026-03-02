"""
Event Markov-chain filtering for sequential event observations.
"""

import importlib
import json
import logging
import os
from collections import defaultdict

import numpy as np


def _normalize(vec, eps=1e-12):
    vec = np.asarray(vec, dtype=np.float64)
    total = vec.sum()
    if total <= eps:
        return None
    return vec / total


def _normalize_rows(mat, eps=1e-12):
    mat = np.asarray(mat, dtype=np.float64)
    out = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        row = mat[i]
        row_sum = row.sum()
        if row_sum <= eps:
            out[i] = np.ones(mat.shape[1], dtype=np.float64) / float(mat.shape[1])
        else:
            out[i] = row / row_sum
    return out


def _symbolic_transfer_entropy(
    source_symbols,
    target_symbols,
    target_order=1,
    source_order=1,
    smoothing=1e-9,
    log_base=2.0,
):
    """
    Matrix/count based symbolic transfer entropy:
    TE(source -> target) with configurable history orders.
    """
    n = min(len(source_symbols), len(target_symbols))
    target_order = max(1, int(target_order))
    source_order = max(1, int(source_order))
    if n <= max(target_order, source_order):
        return None

    source = [str(x) for x in source_symbols[:n]]
    target = [str(x) for x in target_symbols[:n]]
    target_alphabet = sorted(set(target))
    if len(target_alphabet) == 0:
        return None
    n_target = float(len(target_alphabet))

    start_t = max(target_order, source_order) - 1

    c_joint = defaultdict(float)  # (y_next, y_hist, x_hist)
    c_yx = defaultdict(float)  # (y_hist, x_hist)
    c_yy = defaultdict(float)  # (y_next, y_hist)
    c_y = defaultdict(float)  # (y_hist)
    total = 0.0

    for t in range(start_t, n - 1):
        y_next = target[t + 1]
        y_hist = tuple(target[t - target_order + 1 : t + 1])
        x_hist = tuple(source[t - source_order + 1 : t + 1])

        c_joint[(y_next, y_hist, x_hist)] += 1.0
        c_yx[(y_hist, x_hist)] += 1.0
        c_yy[(y_next, y_hist)] += 1.0
        c_y[y_hist] += 1.0
        total += 1.0

    if total <= 0:
        return None

    log_denom = np.log(float(log_base))
    te = 0.0
    alpha = float(max(smoothing, 0.0))

    for (y_next, y_hist, x_hist), count in c_joint.items():
        p_joint = count / total
        p_y_given_yx = (count + alpha) / (c_yx[(y_hist, x_hist)] + alpha * n_target)
        p_y_given_y = (c_yy[(y_next, y_hist)] + alpha) / (
            c_y[y_hist] + alpha * n_target
        )
        if p_y_given_yx <= 0 or p_y_given_y <= 0:
            continue
        te += p_joint * (np.log(p_y_given_yx) - np.log(p_y_given_y)) / log_denom

    return float(te)


class EventMarkovChain:
    def __init__(
        self,
        states,
        transition,
        initial=None,
        smoothing=1e-9,
        learn_transitions=False,
        transition_mode="homogeneous",
        transition_schedule=None,
        transition_provider=None,
        window_size=0,
        context_key="ecological_context",
        context_transitions=None,
        context_initial=None,
        context_fallback="default",
        markov_order=1,
        transfer_entropy_mode="none",
        transfer_entropy_source="context",
        transfer_entropy_target_order=1,
        transfer_entropy_source_order=1,
    ):
        self.states = list(states)
        self.num_states = len(self.states)
        if self.num_states == 0:
            raise ValueError("Markov chain requires at least one state.")

        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        self.base_transition = self._validate_transition(transition)
        self.transition_mode = str(transition_mode)
        self.transition_provider = transition_provider
        self.transition_schedule = self._parse_transition_schedule(
            transition_schedule or []
        )
        self.context_key = str(context_key)
        self.context_fallback = context_fallback
        self.context_transitions = self._parse_context_transitions(
            context_transitions or {}
        )
        self.context_initial = self._parse_context_initial(context_initial or {})
        self.window_size = int(window_size)
        if self.window_size < 0:
            self.window_size = 0

        if initial is None:
            initial = np.ones(self.num_states, dtype=np.float64) / float(self.num_states)
        else:
            initial = _normalize(initial)
            if initial is None:
                initial = np.ones(self.num_states, dtype=np.float64) / float(
                    self.num_states
                )
        self.initial = initial

        self.markov_order = max(1, int(markov_order))
        self.smoothing = float(smoothing)
        self.learn_transitions = bool(learn_transitions)
        self.transition_counts = np.ones_like(self.base_transition) * self.smoothing
        self.sequence_posteriors = {}
        self.sequence_steps = {}
        self.sequence_windows = {}
        self.sequence_history = {}
        self.sequence_symbols = {}

        self.transfer_entropy_mode = str(transfer_entropy_mode or "none")
        self.transfer_entropy_source = str(transfer_entropy_source or "context")
        self.transfer_entropy_target_order = max(1, int(transfer_entropy_target_order))
        self.transfer_entropy_source_order = max(1, int(transfer_entropy_source_order))
        supported_te_modes = {"none", "symbolic_matrix"}
        if self.transfer_entropy_mode not in supported_te_modes:
            logging.warning(
                f"Unsupported transfer_entropy_mode={self.transfer_entropy_mode}. "
                "Disabling transfer entropy."
            )
            self.transfer_entropy_mode = "none"

        if self.learn_transitions and (
            self.transition_mode != "homogeneous" or self.window_size > 0
        ):
            logging.warning(
                "learn_transitions is only applied for homogeneous, non-windowed mode. "
                "Disabling adaptive transition learning."
            )
            self.learn_transitions = False

    def _validate_transition(self, transition):
        transition = np.asarray(transition, dtype=np.float64)
        if transition.shape != (self.num_states, self.num_states):
            raise ValueError(
                "Transition matrix shape mismatch: "
                f"got {transition.shape}, expected {(self.num_states, self.num_states)}."
            )
        return _normalize_rows(transition)

    def _parse_transition_schedule(self, schedule):
        parsed = []
        for entry in schedule:
            start_step = int(entry.get("start_step", 0))
            end_step = entry.get("end_step")
            if end_step is not None:
                end_step = int(end_step)
            matrix = self._validate_transition(entry.get("transition", []))
            parsed.append(
                {
                    "start_step": start_step,
                    "end_step": end_step,
                    "transition": matrix,
                }
            )
        parsed.sort(key=lambda x: x["start_step"])
        return parsed

    def _parse_context_transitions(self, context_transitions):
        parsed = {}
        for key, matrix in context_transitions.items():
            parsed[str(key)] = self._validate_transition(matrix)
        return parsed

    def _parse_context_initial(self, context_initial):
        parsed = {}
        for key, vec in context_initial.items():
            vec = _normalize(vec)
            if vec is None or len(vec) != self.num_states:
                continue
            parsed[str(key)] = vec
        return parsed

    @staticmethod
    def _load_transition_provider(path, params):
        if not path:
            return None
        if ":" not in path:
            raise ValueError(
                f"Invalid transition provider '{path}'. Use module.path:ClassName."
            )
        module_name, class_name = path.split(":", 1)
        module = importlib.import_module(module_name)
        cls_obj = getattr(module, class_name)
        provider = cls_obj(**params)
        if not hasattr(provider, "get_transition"):
            raise ValueError("Transition provider must implement get_transition().")
        return provider

    @classmethod
    def from_file(cls, config_path):
        if not config_path:
            raise ValueError("config_path is required.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Markov config not found: {config_path}")

        with open(config_path, "r") as fp:
            payload = json.load(fp)

        states = payload.get("states", [])
        transition = payload.get("transition", [])
        initial = payload.get("initial")
        smoothing = payload.get("smoothing", 1e-9)
        learn_transitions = payload.get("learn_transitions", False)
        transition_mode = payload.get("transition_mode", "homogeneous")
        transition_schedule = payload.get("transition_schedule", [])
        window_size = payload.get("window_size", 0)
        context_key = payload.get("context_key", "ecological_context")
        context_transitions = payload.get("context_transitions", {})
        context_initial = payload.get("context_initial", {})
        context_fallback = payload.get("context_fallback", "default")
        markov_order = payload.get("markov_order", 1)
        transfer_entropy_mode = payload.get("transfer_entropy_mode", "none")
        transfer_entropy_source = payload.get("transfer_entropy_source", "context")
        transfer_entropy_target_order = payload.get("transfer_entropy_target_order", 1)
        transfer_entropy_source_order = payload.get("transfer_entropy_source_order", 1)
        transition_provider = cls._load_transition_provider(
            payload.get("transition_provider"),
            payload.get("transition_provider_params", {}),
        )
        return cls(
            states=states,
            transition=transition,
            initial=initial,
            smoothing=smoothing,
            learn_transitions=learn_transitions,
            transition_mode=transition_mode,
            transition_schedule=transition_schedule,
            transition_provider=transition_provider,
            window_size=window_size,
            context_key=context_key,
            context_transitions=context_transitions,
            context_initial=context_initial,
            context_fallback=context_fallback,
            markov_order=markov_order,
            transfer_entropy_mode=transfer_entropy_mode,
            transfer_entropy_source=transfer_entropy_source,
            transfer_entropy_target_order=transfer_entropy_target_order,
            transfer_entropy_source_order=transfer_entropy_source_order,
        )

    def _observation_vector(self, observation_scores):
        obs = np.ones(self.num_states, dtype=np.float64) * self.smoothing
        for event_id, score in observation_scores.items():
            idx = self.state_to_idx.get(event_id)
            if idx is None:
                continue
            obs[idx] = max(float(score), self.smoothing)
        return obs

    def _vec_to_state_map(self, vec):
        return {state: float(vec[idx]) for state, idx in self.state_to_idx.items()}

    def _transition_from_schedule(self, step_idx):
        chosen = None
        for entry in self.transition_schedule:
            if step_idx < entry["start_step"]:
                continue
            end_step = entry["end_step"]
            if end_step is not None and step_idx > end_step:
                continue
            chosen = entry["transition"]
        if chosen is None:
            return self.base_transition
        return chosen

    def _resolve_context_label(self, context):
        if context is None:
            return None
        if isinstance(context, str):
            return context
        if isinstance(context, dict):
            if self.context_key in context and context[self.context_key] is not None:
                return str(context[self.context_key])
            if "ecological_context" in context and context["ecological_context"] is not None:
                return str(context["ecological_context"])
            if "context" in context and context["context"] is not None:
                return str(context["context"])
        return None

    def _transition_from_context(self, context_label):
        if context_label in self.context_transitions:
            return self.context_transitions[context_label]
        if self.context_fallback in self.context_transitions:
            return self.context_transitions[self.context_fallback]
        return self.base_transition

    def _initial_from_context(self, context_label):
        if context_label in self.context_initial:
            return self.context_initial[context_label]
        if self.context_fallback in self.context_initial:
            return self.context_initial[self.context_fallback]
        return self.initial

    def _get_transition(self, sequence_id, step_idx, observation_scores, context=None):
        context_label = self._resolve_context_label(context)
        if self.transition_mode == "context":
            return self._transition_from_context(context_label)
        if self.transition_mode == "schedule":
            return self._transition_from_schedule(step_idx)
        if self.transition_mode == "provider" and self.transition_provider is not None:
            try:
                matrix = self.transition_provider.get_transition(
                    sequence_id=sequence_id,
                    step_idx=step_idx,
                    observation_scores=observation_scores,
                    context=context,
                    context_label=context_label,
                    default_transition=self.base_transition,
                    states=self.states,
                )
            except TypeError:
                # Backward-compatible call signature for existing providers.
                matrix = self.transition_provider.get_transition(
                    sequence_id=sequence_id,
                    step_idx=step_idx,
                    observation_scores=observation_scores,
                    default_transition=self.base_transition,
                    states=self.states,
                )
            return self._validate_transition(matrix)
        return self.base_transition

    def _apply_transition(self, prev, transition_matrix, obs, return_debug=False):
        pred = prev @ transition_matrix
        post_unnorm = pred * obs
        post = _normalize(post_unnorm)
        if post is None:
            post = np.ones(self.num_states, dtype=np.float64) / float(self.num_states)
        if return_debug:
            return post, {
                "prior": self._vec_to_state_map(prev),
                "predicted": self._vec_to_state_map(pred),
                "likelihood": self._vec_to_state_map(obs),
                "posterior": self._vec_to_state_map(post),
                "transition": transition_matrix.tolist(),
            }
        return post

    def _prior_from_history(self, history, default_prior):
        if len(history) == 0:
            return default_prior
        if self.markov_order <= 1:
            return history[-1]

        take = min(self.markov_order, len(history))
        selected = history[-take:]
        prior = np.mean(np.stack(selected, axis=0), axis=0)
        prior_norm = _normalize(prior)
        if prior_norm is None:
            return history[-1]
        return prior_norm

    def _symbol_for_target(self, posterior_vec):
        idx = int(np.argmax(posterior_vec))
        return self.states[idx]

    def _symbol_for_source(self, context_label, observation_scores):
        if self.transfer_entropy_source == "context":
            if context_label is None:
                return "__none__"
            return str(context_label)

        if self.transfer_entropy_source == "observation_argmax":
            if not observation_scores:
                return "__none__"
            return max(observation_scores.items(), key=lambda x: float(x[1]))[0]

        return "__none__"

    def _append_symbols_and_compute_te(
        self, sequence_id, context_label, observation_scores, posterior_vec
    ):
        seq = self.sequence_symbols.setdefault(
            sequence_id, {"source": [], "target": []}
        )
        seq["source"].append(
            self._symbol_for_source(context_label, observation_scores)
        )
        seq["target"].append(self._symbol_for_target(posterior_vec))

        if self.transfer_entropy_mode == "none":
            return None

        if self.transfer_entropy_mode == "symbolic_matrix":
            value = _symbolic_transfer_entropy(
                source_symbols=seq["source"],
                target_symbols=seq["target"],
                target_order=self.transfer_entropy_target_order,
                source_order=self.transfer_entropy_source_order,
                smoothing=self.smoothing,
                log_base=2.0,
            )
            return {
                "mode": "symbolic_matrix",
                "source": self.transfer_entropy_source,
                "target_order": int(self.transfer_entropy_target_order),
                "source_order": int(self.transfer_entropy_source_order),
                "value": value,
            }

        return None

    def _update_windowed(
        self,
        sequence_id,
        step_idx,
        observation_scores,
        context=None,
        return_debug=False,
    ):
        window = self.sequence_windows.setdefault(sequence_id, [])
        window.append(
            {
                "step_idx": step_idx,
                "observation_scores": observation_scores,
                "context": context,
            }
        )
        if self.window_size > 0 and len(window) > self.window_size:
            del window[0 : len(window) - self.window_size]

        first_context_label = self._resolve_context_label(window[0].get("context"))
        initial_prior = self._initial_from_context(first_context_label)
        history = []
        window_trace = []
        for item in window:
            context_label = self._resolve_context_label(item.get("context"))
            t = self._get_transition(
                sequence_id=sequence_id,
                step_idx=item["step_idx"],
                observation_scores=item["observation_scores"],
                context=item.get("context"),
            )
            obs = self._observation_vector(item["observation_scores"])
            prior = self._prior_from_history(history, initial_prior)
            if return_debug:
                post, step_debug = self._apply_transition(
                    prior, t, obs, return_debug=True
                )
                step_debug["step_idx"] = int(item["step_idx"])
                step_debug["context_label"] = context_label
                step_debug["markov_order"] = int(self.markov_order)
                window_trace.append(step_debug)
            else:
                post = self._apply_transition(prior, t, obs)
            history.append(post)

        if return_debug:
            return post, {
                "window_size_effective": len(window),
                "window_start_step": int(window[0]["step_idx"]),
                "window_end_step": int(window[-1]["step_idx"]),
                "initial_context_label": first_context_label,
                "window_trace": window_trace,
            }

        return post

    def reset_sequence(self, sequence_id):
        self.sequence_posteriors.pop(sequence_id, None)
        self.sequence_steps.pop(sequence_id, None)
        self.sequence_windows.pop(sequence_id, None)
        self.sequence_history.pop(sequence_id, None)
        self.sequence_symbols.pop(sequence_id, None)

    def update(self, sequence_id, observation_scores, context=None, return_debug=False):
        step_idx = int(self.sequence_steps.get(sequence_id, 0))
        context_label = self._resolve_context_label(context)

        debug = {
            "sequence_id": sequence_id,
            "step_idx": step_idx,
            "transition_mode": self.transition_mode,
            "context_label": context_label,
            "window_size_config": int(self.window_size),
            "markov_order": int(self.markov_order),
            "transfer_entropy_mode": self.transfer_entropy_mode,
        }

        if self.window_size > 0:
            if return_debug:
                post, window_debug = self._update_windowed(
                    sequence_id,
                    step_idx,
                    observation_scores,
                    context=context,
                    return_debug=True,
                )
                debug["window"] = window_debug
                if window_debug.get("window_trace"):
                    debug["last_transition"] = window_debug["window_trace"][-1].get(
                        "transition"
                    )
            else:
                post = self._update_windowed(
                    sequence_id, step_idx, observation_scores, context=context
                )
        else:
            obs = self._observation_vector(observation_scores)
            history = self.sequence_history.get(sequence_id, [])
            initial_prior = self._initial_from_context(context_label)
            prev = self._prior_from_history(history, initial_prior)
            transition = self._get_transition(
                sequence_id=sequence_id,
                step_idx=step_idx,
                observation_scores=observation_scores,
                context=context,
            )
            if return_debug:
                post, step_debug = self._apply_transition(
                    prev, transition, obs, return_debug=True
                )
                debug.update(step_debug)
            else:
                post = self._apply_transition(prev, transition, obs)

            if self.learn_transitions:
                self.transition_counts += np.outer(prev, post)
                self.base_transition = _normalize_rows(self.transition_counts)

        self.sequence_posteriors[sequence_id] = post
        history = self.sequence_history.setdefault(sequence_id, [])
        history.append(post)
        if len(history) > max(self.markov_order, 1):
            del history[0 : len(history) - max(self.markov_order, 1)]
        self.sequence_steps[sequence_id] = step_idx + 1

        transfer_entropy = self._append_symbols_and_compute_te(
            sequence_id=sequence_id,
            context_label=context_label,
            observation_scores=observation_scores,
            posterior_vec=post,
        )

        posterior_map = self._vec_to_state_map(post)
        if return_debug:
            debug["transfer_entropy"] = transfer_entropy
            debug["posterior"] = posterior_map
            return posterior_map, debug

        return posterior_map
