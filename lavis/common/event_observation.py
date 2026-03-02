"""
Observation-level event classifiers for dynamic event tagging.
"""

import importlib
import json
import os
import re


def _normalize_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"([.!\"()*#:;~])", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _token_set(text):
    return set(_normalize_text(text).split())


def _clamp_prob(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


class BaseObservationClassifier:
    def score(self, context):
        raise NotImplementedError


class PrototypeLabelClassifier(BaseObservationClassifier):
    """
    Scores event observation via similarity between predicted labels and prototypes.
    """

    def __init__(self, prototypes=None, use_confidence_weight=True):
        self.prototypes = prototypes or []
        self.use_confidence_weight = use_confidence_weight
        self.prototype_tokens = [_token_set(proto) for proto in self.prototypes]

    @staticmethod
    def _jaccard(a_tokens, b_tokens):
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        if union == 0:
            return 0.0
        return inter / union

    def score(self, context):
        best = 0.0
        for pred in context.get("event_predictions", []):
            pred_label = pred.get("label", "")
            pred_conf = float(pred.get("confidence", 0.0))
            pred_tokens = _token_set(pred_label)
            for proto_tokens in self.prototype_tokens:
                sim = self._jaccard(pred_tokens, proto_tokens)
                if self.use_confidence_weight:
                    sim *= pred_conf
                if sim > best:
                    best = sim
        return _clamp_prob(best)


class ConfidenceThresholdClassifier(BaseObservationClassifier):
    """
    Binary confidence classifier from model confidence for target event.
    """

    def __init__(self, threshold=0.5, low_score=0.1, high_score=0.9):
        self.threshold = float(threshold)
        self.low_score = float(low_score)
        self.high_score = float(high_score)

    def score(self, context):
        model_conf = float(context.get("model_event_confidence", 0.0))
        if model_conf >= self.threshold:
            return _clamp_prob(self.high_score)
        return _clamp_prob(self.low_score)


class KeywordBinaryClassifier(BaseObservationClassifier):
    """
    Binary classifier based on keywords in question or labels.
    """

    def __init__(self, keywords=None, target_fields=None, hit_score=0.8, miss_score=0.2):
        self.keywords = [_normalize_text(k) for k in (keywords or [])]
        self.target_fields = target_fields or ["question", "labels"]
        self.hit_score = float(hit_score)
        self.miss_score = float(miss_score)

    def _collect_text(self, context):
        chunks = []
        if "question" in self.target_fields:
            chunks.append(context.get("question", ""))
        if "labels" in self.target_fields:
            chunks.extend(
                [pred.get("label", "") for pred in context.get("event_predictions", [])]
            )
        return " ".join(chunks)

    def score(self, context):
        haystack = _normalize_text(self._collect_text(context))
        if any(keyword in haystack for keyword in self.keywords):
            return _clamp_prob(self.hit_score)
        return _clamp_prob(self.miss_score)


class ObservationClassifierSpec:
    def __init__(
        self,
        classifier_id,
        event_id,
        classifier,
        weight=1.0,
        include_model_score=True,
        model_weight=1.0,
    ):
        self.classifier_id = classifier_id
        self.event_id = event_id
        self.classifier = classifier
        self.weight = float(weight)
        self.include_model_score = bool(include_model_score)
        self.model_weight = float(model_weight)


class ObservationClassifierSet:
    def __init__(self, specs=None, combination="weighted_mean"):
        self.specs = specs or []
        self.combination = combination
        self.by_event_id = {}
        for spec in self.specs:
            self.by_event_id.setdefault(spec.event_id, []).append(spec)

    @classmethod
    def from_file(cls, config_path):
        if not config_path:
            raise ValueError("config_path is required.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Observation classifier config not found: {config_path}"
            )
        with open(config_path, "r") as fp:
            payload = json.load(fp)

        specs = []
        for item in payload.get("classifiers", []):
            classifier_id = item.get("classifier_id")
            event_id = item.get("event_id")
            classifier_type = item.get("type")
            if not classifier_id or not event_id or not classifier_type:
                continue
            if item.get("active", True) is False:
                continue

            params = item.get("params", {})
            classifier = cls._build_classifier(classifier_type, item, params)

            specs.append(
                ObservationClassifierSpec(
                    classifier_id=classifier_id,
                    event_id=event_id,
                    classifier=classifier,
                    weight=item.get("weight", 1.0),
                    include_model_score=item.get("include_model_score", True),
                    model_weight=item.get("model_weight", 1.0),
                )
            )

        combination = payload.get("combination", "weighted_mean")
        return cls(specs=specs, combination=combination)

    @staticmethod
    def _build_python_classifier(class_path, params):
        if ":" not in class_path:
            raise ValueError(
                f"Invalid class_path '{class_path}'. Use 'module.path:ClassName'."
            )
        module_name, class_name = class_path.split(":", 1)
        module = importlib.import_module(module_name)
        cls_obj = getattr(module, class_name)
        classifier = cls_obj(**params)
        if not hasattr(classifier, "score"):
            raise ValueError(f"Custom classifier {class_path} must implement score().")
        return classifier

    @classmethod
    def _build_classifier(cls, classifier_type, item, params):
        if classifier_type == "prototype_label":
            return PrototypeLabelClassifier(
                prototypes=params.get("prototypes", []),
                use_confidence_weight=params.get("use_confidence_weight", True),
            )
        if classifier_type == "confidence_threshold":
            return ConfidenceThresholdClassifier(
                threshold=params.get("threshold", 0.5),
                low_score=params.get("low_score", 0.1),
                high_score=params.get("high_score", 0.9),
            )
        if classifier_type == "keyword_binary":
            return KeywordBinaryClassifier(
                keywords=params.get("keywords", []),
                target_fields=params.get("target_fields", ["question", "labels"]),
                hit_score=params.get("hit_score", 0.8),
                miss_score=params.get("miss_score", 0.2),
            )
        if classifier_type == "python":
            class_path = item.get("class_path")
            if not class_path:
                raise ValueError("python classifier requires class_path.")
            return cls._build_python_classifier(class_path, params)

        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    @staticmethod
    def _combine_weighted_mean(scored_items):
        if not scored_items:
            return None
        num = 0.0
        den = 0.0
        for score, weight in scored_items:
            num += float(score) * float(weight)
            den += float(weight)
        if den <= 0.0:
            return None
        return _clamp_prob(num / den)

    @staticmethod
    def _combine_noisy_or(scored_items):
        if not scored_items:
            return None
        prod = 1.0
        for score, weight in scored_items:
            # Weight as exponent in noisy-or likelihood space.
            prod *= (1.0 - _clamp_prob(score)) ** max(float(weight), 0.0)
        return _clamp_prob(1.0 - prod)

    def _combine(self, scored_items):
        if self.combination == "noisy_or":
            return self._combine_noisy_or(scored_items)
        return self._combine_weighted_mean(scored_items)

    def score_events(self, base_context, candidate_event_ids, model_scores):
        """
        Returns:
            dict[event_id] -> observation probability in [0, 1].
        """
        out = {}
        for event_id in candidate_event_ids:
            if not event_id:
                continue
            specs = self.by_event_id.get(event_id, [])
            model_score = _clamp_prob(float(model_scores.get(event_id, 0.0)))

            scored_items = []
            for spec in specs:
                context = dict(base_context)
                context["event_id"] = event_id
                context["model_event_confidence"] = model_score
                value = _clamp_prob(spec.classifier.score(context))
                scored_items.append((value, spec.weight))
                if spec.include_model_score:
                    scored_items.append((model_score, spec.model_weight))

            combined = self._combine(scored_items)
            if combined is None:
                combined = model_score
            out[event_id] = _clamp_prob(combined)

        return out
