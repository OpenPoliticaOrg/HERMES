"""
Entity-centric online event sequence and lifecycle tracking.
"""

from copy import deepcopy


def _clamp01(value):
    v = float(value)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def normalize_observation_scores(raw_scores):
    """
    Normalize observation scores into {event_id: score} map.

    Supports:
    - dict[event_id] -> numeric
    - list[{event_id, score|prob|confidence}]
    """
    out = {}
    if raw_scores is None:
        return out

    if isinstance(raw_scores, dict):
        for event_id, score in raw_scores.items():
            if event_id is None:
                continue
            event_id = str(event_id).strip()
            if not event_id:
                continue
            try:
                out[event_id] = _clamp01(score)
            except Exception:
                continue
        return out

    if isinstance(raw_scores, (list, tuple)):
        for item in raw_scores:
            if not isinstance(item, dict):
                continue
            event_id = item.get("event_id")
            if event_id is None:
                continue
            event_id = str(event_id).strip()
            if not event_id:
                continue
            raw_score = item.get("score", item.get("prob", item.get("confidence", 0.0)))
            try:
                out[event_id] = _clamp01(raw_score)
            except Exception:
                continue
        return out

    return out


def scores_from_event_predictions(event_predictions):
    """
    Build {event_id: confidence} map from event prediction list.
    Keeps max confidence if event_id appears multiple times.
    """
    out = {}
    if not isinstance(event_predictions, (list, tuple)):
        return out
    for pred in event_predictions:
        if not isinstance(pred, dict):
            continue
        event_id = pred.get("event_id")
        if event_id is None:
            continue
        event_id = str(event_id).strip()
        if not event_id:
            continue
        raw_conf = pred.get("confidence", pred.get("score", pred.get("prob", 0.0)))
        try:
            conf = _clamp01(raw_conf)
        except Exception:
            continue
        prev = out.get(event_id)
        if prev is None or conf > prev:
            out[event_id] = conf
    return out


class EntityEventSequenceTracker:
    def __init__(
        self,
        markov_chain=None,
        context_field="ecological_context",
        history_limit=64,
        default_entity_id="__scene__",
        default_markov_topk=5,
        default_observation_topk=5,
        default_missing_tolerance=0,
    ):
        self.markov_chain = markov_chain
        self.context_field = str(context_field)
        self.history_limit = max(1, int(history_limit))
        self.default_entity_id = str(default_entity_id)
        self.default_markov_topk = max(1, int(default_markov_topk))
        self.default_observation_topk = max(1, int(default_observation_topk))
        self.default_missing_tolerance = max(0, int(default_missing_tolerance))

        self.entity_timelines = {}
        self.sequence_windows = {}
        self.sequence_entity_state = {}
        self.sequence_window_state = {}

    def reset(self):
        self.entity_timelines = {}
        self.sequence_windows = {}
        self.sequence_entity_state = {}
        self.sequence_window_state = {}

    @staticmethod
    def _normalize_base_sequence_id(base_sequence_id):
        value = str(base_sequence_id).strip() if base_sequence_id is not None else "seq"
        return value if value else "seq"

    def _normalize_entity_id(self, entity_id):
        value = str(entity_id).strip() if entity_id is not None else self.default_entity_id
        return value if value else self.default_entity_id

    def get_timeline(self, entity_sequence_id):
        timeline = self.entity_timelines.get(entity_sequence_id, [])
        return deepcopy(timeline)

    def get_entity_states(self, base_sequence_id):
        base = self._normalize_base_sequence_id(base_sequence_id)
        return deepcopy(self.sequence_entity_state.get(base, {}))

    def compose_entity_sequence_id(self, base_sequence_id, entity_id):
        base = self._normalize_base_sequence_id(base_sequence_id)
        ent = self._normalize_entity_id(entity_id)
        return f"{base}::entity::{ent}"

    @staticmethod
    def _top_state_from_scores(score_map):
        if not score_map:
            return None
        event_id, score = max(score_map.items(), key=lambda x: float(x[1]))
        return {"event_id": event_id, "prob": float(score)}

    @staticmethod
    def _sort_scores(score_map):
        return sorted(score_map.items(), key=lambda x: float(x[1]), reverse=True)

    @staticmethod
    def _unique_sorted(values):
        return sorted(set(values))

    def _normalize_posterior_without_markov(self, observation_scores):
        values = [max(float(v), 0.0) for v in observation_scores.values()]
        total = sum(values)
        if total <= 0.0:
            n = len(observation_scores)
            if n == 0:
                return {}
            uni = 1.0 / float(n)
            return {k: uni for k in observation_scores.keys()}
        out = {}
        for event_id, value in observation_scores.items():
            out[event_id] = float(max(float(value), 0.0) / total)
        return out

    def _context_label(self, context):
        if isinstance(context, dict):
            return context.get(self.context_field, context.get("context", None))
        return context

    def begin_window(
        self,
        base_sequence_id,
        context=None,
        image_id=None,
        question=None,
        metadata=None,
    ):
        base = self._normalize_base_sequence_id(base_sequence_id)
        if base in self.sequence_window_state:
            self.finalize_window(base)

        step = int(self.sequence_windows.get(base, 0)) + 1
        self.sequence_windows[base] = step
        self.sequence_window_state[base] = {
            "step": step,
            "context": self._context_label(context),
            "image_id": image_id,
            "question": question,
            "metadata": metadata if isinstance(metadata, dict) else {},
            "observed_entities": set(),
            "updated_entities": [],
            "entered_entities": [],
            "reentered_entities": [],
            "continued_entities": [],
            "exited_entities": [],
        }
        return step

    def _ensure_window(
        self,
        base_sequence_id,
        context=None,
        image_id=None,
        question=None,
        metadata=None,
    ):
        base = self._normalize_base_sequence_id(base_sequence_id)
        window_state = self.sequence_window_state.get(base)
        if window_state is None:
            self.begin_window(
                base_sequence_id=base,
                context=context,
                image_id=image_id,
                question=question,
                metadata=metadata,
            )
            window_state = self.sequence_window_state.get(base)
        return base, window_state

    def _track_entity_lifecycle(self, base_sequence_id, entity_id):
        base = self._normalize_base_sequence_id(base_sequence_id)
        entity_id = self._normalize_entity_id(entity_id)
        window_state = self.sequence_window_state.get(base)
        if window_state is None:
            self.begin_window(base)
            window_state = self.sequence_window_state[base]

        entity_states = self.sequence_entity_state.setdefault(base, {})
        entity_state = entity_states.get(entity_id)
        lifecycle_state = "continued"

        if entity_state is None:
            lifecycle_state = "entered"
            entity_state = {
                "entity_id": entity_id,
                "active": True,
                "first_seen_step": int(window_state["step"]),
                "last_seen_step": int(window_state["step"]),
                "last_exit_step": None,
                "seen_windows": 0,
                "missed_windows": 0,
                "enter_count": 1,
                "exit_count": 0,
            }
            entity_states[entity_id] = entity_state
            window_state["entered_entities"].append(entity_id)
        elif not bool(entity_state.get("active", False)):
            lifecycle_state = "reentered"
            entity_state["active"] = True
            entity_state["enter_count"] = int(entity_state.get("enter_count", 0)) + 1
            window_state["reentered_entities"].append(entity_id)
        else:
            lifecycle_state = "continued"
            window_state["continued_entities"].append(entity_id)

        entity_state["last_seen_step"] = int(window_state["step"])
        entity_state["missed_windows"] = 0
        entity_state["seen_windows"] = int(entity_state.get("seen_windows", 0)) + 1
        if entity_id not in window_state["updated_entities"]:
            window_state["updated_entities"].append(entity_id)
        window_state["observed_entities"].add(entity_id)

        return lifecycle_state, deepcopy(entity_state), int(window_state["step"])

    def finalize_window(self, base_sequence_id, missing_tolerance=None):
        base = self._normalize_base_sequence_id(base_sequence_id)
        window_state = self.sequence_window_state.pop(base, None)
        if window_state is None:
            return {
                "window_step": int(self.sequence_windows.get(base, 0)),
                "observed_entities": [],
                "updated_entities": [],
                "entered_entities": [],
                "reentered_entities": [],
                "continued_entities": [],
                "exited_entities": [],
                "active_entities": [],
                "inactive_entities": [],
                "observed_count": 0,
                "active_count": 0,
                "total_tracked_entities": 0,
                "entity_states": {},
            }

        tol = self.default_missing_tolerance
        if missing_tolerance is not None:
            tol = max(0, int(missing_tolerance))

        observed = set(window_state["observed_entities"])
        entity_states = self.sequence_entity_state.setdefault(base, {})
        exited_entities = []

        for entity_id, entity_state in entity_states.items():
            if entity_id in observed:
                continue
            if not bool(entity_state.get("active", False)):
                continue
            entity_state["missed_windows"] = int(entity_state.get("missed_windows", 0)) + 1
            if int(entity_state["missed_windows"]) > tol:
                entity_state["active"] = False
                entity_state["last_exit_step"] = int(window_state["step"])
                entity_state["exit_count"] = int(entity_state.get("exit_count", 0)) + 1
                exited_entities.append(entity_id)

        window_state["exited_entities"].extend(exited_entities)

        active_entities = self._unique_sorted(
            [
                entity_id
                for entity_id, entity_state in entity_states.items()
                if bool(entity_state.get("active", False))
            ]
        )
        inactive_entities = self._unique_sorted(
            [
                entity_id
                for entity_id, entity_state in entity_states.items()
                if not bool(entity_state.get("active", False))
            ]
        )

        summary = {
            "window_step": int(window_state["step"]),
            "image_id": window_state.get("image_id"),
            "question": window_state.get("question"),
            self.context_field: window_state.get("context"),
            "metadata": deepcopy(window_state.get("metadata", {})),
            "observed_entities": self._unique_sorted(observed),
            "updated_entities": self._unique_sorted(window_state.get("updated_entities", [])),
            "entered_entities": self._unique_sorted(window_state.get("entered_entities", [])),
            "reentered_entities": self._unique_sorted(window_state.get("reentered_entities", [])),
            "continued_entities": self._unique_sorted(window_state.get("continued_entities", [])),
            "exited_entities": self._unique_sorted(window_state.get("exited_entities", [])),
            "active_entities": active_entities,
            "inactive_entities": inactive_entities,
            "observed_count": len(observed),
            "active_count": len(active_entities),
            "total_tracked_entities": len(entity_states),
            "entity_states": deepcopy(entity_states),
        }
        return summary

    def update_entity(
        self,
        base_sequence_id,
        entity_id,
        observation_scores,
        context=None,
        image_id=None,
        question=None,
        metadata=None,
        markov_debug=False,
        markov_topk=None,
        observation_topk=None,
    ):
        observation_scores = normalize_observation_scores(observation_scores)
        if len(observation_scores) == 0:
            return None

        markov_topk = (
            self.default_markov_topk if markov_topk is None else max(1, int(markov_topk))
        )
        observation_topk = (
            self.default_observation_topk
            if observation_topk is None
            else max(1, int(observation_topk))
        )

        base_sequence_id, _ = self._ensure_window(
            base_sequence_id=base_sequence_id,
            context=context,
            image_id=image_id,
            question=question,
            metadata=metadata,
        )
        entity_id = self._normalize_entity_id(entity_id)
        lifecycle_state, entity_state, window_step = self._track_entity_lifecycle(
            base_sequence_id=base_sequence_id,
            entity_id=entity_id,
        )
        entity_sequence_id = self.compose_entity_sequence_id(base_sequence_id, entity_id)

        posterior_map = None
        markov_state = None
        markov_debug_payload = None
        if self.markov_chain is not None:
            if markov_debug:
                posterior_map, markov_debug_payload = self.markov_chain.update(
                    sequence_id=entity_sequence_id,
                    observation_scores=observation_scores,
                    context=context,
                    return_debug=True,
                )
            else:
                posterior_map = self.markov_chain.update(
                    sequence_id=entity_sequence_id,
                    observation_scores=observation_scores,
                    context=context,
                )
            markov_state = self._top_state_from_scores(posterior_map)
        else:
            posterior_map = self._normalize_posterior_without_markov(observation_scores)
            markov_state = self._top_state_from_scores(posterior_map)

        observation_state = self._top_state_from_scores(observation_scores)
        context_label = self._context_label(context)

        step_value = None
        if isinstance(markov_debug_payload, dict):
            step_idx = markov_debug_payload.get("step_idx")
            if step_idx is not None:
                try:
                    step_value = int(step_idx) + 1
                except Exception:
                    step_value = None

        timeline = self.entity_timelines.setdefault(entity_sequence_id, [])
        if step_value is None:
            step_value = len(timeline) + 1

        record = {
            "step": int(step_value),
            "window_step": int(window_step),
            "entity_id": entity_id,
            "lifecycle_state": lifecycle_state,
            "image_id": image_id,
            "question": question,
            self.context_field: context_label,
            "observation_event_id": (
                observation_state["event_id"] if observation_state is not None else None
            ),
            "observation_score": (
                observation_state["prob"] if observation_state is not None else None
            ),
            "markov_event_id": markov_state["event_id"] if markov_state is not None else None,
            "markov_prob": markov_state["prob"] if markov_state is not None else None,
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
        timeline.append(record)
        if len(timeline) > self.history_limit:
            del timeline[0 : len(timeline) - self.history_limit]

        posterior_sorted = self._sort_scores(posterior_map)
        posterior_top = posterior_sorted[:markov_topk]
        observation_sorted = self._sort_scores(observation_scores)
        observation_top = observation_sorted[:observation_topk]

        summary = {
            "entity_id": entity_id,
            "entity_sequence_id": entity_sequence_id,
            "window_step": int(window_step),
            "lifecycle_state": lifecycle_state,
            "entity_status": entity_state,
            "sequence_length": len(timeline),
            "event_sequence": deepcopy(timeline),
            "observation_state": {
                "event_id": observation_state["event_id"],
                "score": float(observation_state["prob"]),
            }
            if observation_state is not None
            else None,
            "observation_scores": [
                {"event_id": event_id, "score": float(score)}
                for event_id, score in observation_top
            ],
            "markov_state": markov_state,
            "markov_posterior": [
                {"event_id": event_id, "prob": float(prob)}
                for event_id, prob in posterior_top
            ],
        }
        if markov_debug:
            summary["markov_debug"] = markov_debug_payload

        return summary
