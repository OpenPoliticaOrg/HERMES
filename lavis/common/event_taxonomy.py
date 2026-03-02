"""
Utilities for dynamic event tagging and label-to-event mapping.
"""

import json
import os
import re
from collections import defaultdict


def normalize_event_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"([.!\"()*#:;~])", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class EventTaxonomy:
    def __init__(self, taxonomy_path):
        if not taxonomy_path:
            raise ValueError("taxonomy_path is required.")
        if not os.path.exists(taxonomy_path):
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

        with open(taxonomy_path, "r") as fp:
            payload = json.load(fp)

        self.path = taxonomy_path
        self.schema_version = payload.get("schema_version", 1)
        self.default_prompt = payload.get("defaults", {}).get("classifier_prompt", "{}")

        self.events_by_id = {}
        self.dataset_event_ids = defaultdict(list)
        self.dataset_label_to_event_id = {}
        self.global_label_to_event_id = {}

        for event in payload.get("events", []):
            event_id = event.get("event_id")
            label = event.get("label")
            if not event_id or not label:
                continue

            event_dataset = event.get("dataset")
            self.events_by_id[event_id] = event

            if event_dataset:
                self.dataset_event_ids[event_dataset].append(event_id)

            label_variants = [label] + event.get("aliases", [])
            for variant in label_variants:
                norm = normalize_event_text(variant)
                if not norm:
                    continue
                if event_dataset:
                    self.dataset_label_to_event_id[(event_dataset, norm)] = event_id
                # Global fallback map. If duplicates exist, first entry wins.
                if norm not in self.global_label_to_event_id:
                    self.global_label_to_event_id[norm] = event_id

        self.classifiers_by_dataset = defaultdict(list)
        for classifier in payload.get("classifiers", []):
            dataset = classifier.get("dataset")
            if not dataset:
                continue
            self.classifiers_by_dataset[dataset].append(classifier)

    def resolve_event_id(self, label, dataset_name=None):
        norm = normalize_event_text(label)
        if not norm:
            return None
        if dataset_name and (dataset_name, norm) in self.dataset_label_to_event_id:
            return self.dataset_label_to_event_id[(dataset_name, norm)]
        return self.global_label_to_event_id.get(norm)

    def _classifier_matches(self, classifier, question_text):
        if classifier.get("active", True) is False:
            return False

        rules = classifier.get("rules", {})
        contains_any = rules.get("question_contains_any")
        if contains_any:
            q = normalize_event_text(question_text)
            if not any(normalize_event_text(token) in q for token in contains_any):
                return False

        return True

    def _labels_for_event_ids(self, event_ids):
        labels = []
        mapped_event_ids = []
        for event_id in event_ids:
            event = self.events_by_id.get(event_id)
            if not event:
                continue
            label = event.get("label")
            if not label:
                continue
            labels.append(label)
            mapped_event_ids.append(event_id)
        return labels, mapped_event_ids

    def select_candidates(
        self,
        dataset_name,
        question_text,
        fallback_labels,
        fallback_prompt="{}",
    ):
        classifiers = self.classifiers_by_dataset.get(dataset_name, [])
        for classifier in classifiers:
            if not self._classifier_matches(classifier, question_text):
                continue

            event_ids = classifier.get("candidate_event_ids", [])
            labels, mapped_event_ids = self._labels_for_event_ids(event_ids)
            if labels:
                return {
                    "classifier_id": classifier.get("classifier_id", "taxonomy_classifier"),
                    "prompt": classifier.get("prompt", self.default_prompt) or fallback_prompt,
                    "candidate_labels": labels,
                    "candidate_event_ids": mapped_event_ids,
                }

        dataset_event_ids = self.dataset_event_ids.get(dataset_name, [])
        labels, mapped_event_ids = self._labels_for_event_ids(dataset_event_ids)
        if labels:
            return {
                "classifier_id": "taxonomy_dataset_fallback",
                "prompt": self.default_prompt or fallback_prompt,
                "candidate_labels": labels,
                "candidate_event_ids": mapped_event_ids,
            }

        resolved_event_ids = [
            self.resolve_event_id(label, dataset_name=dataset_name)
            for label in fallback_labels
        ]
        return {
            "classifier_id": "label_space_fallback",
            "prompt": self.default_prompt or fallback_prompt,
            "candidate_labels": fallback_labels,
            "candidate_event_ids": resolved_event_ids,
        }
