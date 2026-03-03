"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import numpy as np
import torch

import torch.distributed as dist
from lavis.common.entity_event_sequence import (
    EntityEventSequenceTracker,
    normalize_observation_scores,
    scores_from_event_predictions,
)
from lavis.common.dist_utils import main_process, is_dist_avail_and_initialized
from lavis.common.event_markov import EventMarkovChain
from lavis.common.event_observation import ObservationClassifierSet
from lavis.common.event_taxonomy import EventTaxonomy
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.logger import MetricLogger
from lavis.datasets.data_utils import prepare_sample


@registry.register_task("classification")
class ClassificationTask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        report_metric=True,
        verb_only=False,
        noun_only=False,
        dataset_name=None,
        log_dir=None,
        classification_mode="generate",
        event_taxonomy_path=None,
        classifier_prompt="{}",
        rank_n_segments=1,
        rank_topk=5,
        observation_classifier_path=None,
        markov_chain_path=None,
        markov_sequence_mode="prefix_before_underscore",
        markov_topk=5,
        markov_context_field="ecological_context",
        markov_debug=False,
        entity_default_id="__scene__",
        entity_sequence_history=64,
        entity_sequence_observation_topk=5,
        entity_sequence_missing_tolerance=0,
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric
        
        self.verb_only = verb_only
        self.noun_only = noun_only
        self.dataset_name = dataset_name
        self.log_dir = log_dir
        self.classification_mode = classification_mode
        self.classifier_prompt = classifier_prompt
        self.rank_n_segments = rank_n_segments
        self.rank_topk = rank_topk
        self.markov_sequence_mode = markov_sequence_mode
        self.markov_topk = markov_topk
        self.markov_context_field = markov_context_field
        self.markov_debug = bool(markov_debug)
        self.entity_default_id = str(entity_default_id)
        self.entity_sequence_history = max(1, int(entity_sequence_history))
        self.entity_sequence_observation_topk = max(
            1, int(entity_sequence_observation_topk)
        )
        self.entity_sequence_missing_tolerance = max(
            0, int(entity_sequence_missing_tolerance)
        )
        self.default_candidate_labels = []

        self.event_taxonomy = None
        self.observation_classifier_set = None
        self.event_markov_chain = None
        self.entity_sequence_tracker = None

        if event_taxonomy_path:
            try:
                self.event_taxonomy = EventTaxonomy(event_taxonomy_path)
                logging.info(f"Loaded event taxonomy from {event_taxonomy_path}")
            except Exception as exc:
                logging.warning(
                    f"Failed to load taxonomy at {event_taxonomy_path}: {exc}. "
                    "Falling back to dataset label space."
                )

        if observation_classifier_path:
            try:
                self.observation_classifier_set = ObservationClassifierSet.from_file(
                    observation_classifier_path
                )
                logging.info(
                    f"Loaded observation classifiers from {observation_classifier_path}"
                )
            except Exception as exc:
                logging.warning(
                    f"Failed to load observation classifiers at "
                    f"{observation_classifier_path}: {exc}. "
                    "Falling back to model confidence only."
                )

        if markov_chain_path:
            try:
                self.event_markov_chain = EventMarkovChain.from_file(markov_chain_path)
                logging.info(f"Loaded Markov chain from {markov_chain_path}")
            except Exception as exc:
                logging.warning(
                    f"Failed to load Markov chain at {markov_chain_path}: {exc}. "
                    "Markov updates disabled."
                )

        self.entity_sequence_tracker = EntityEventSequenceTracker(
            markov_chain=self.event_markov_chain,
            context_field=self.markov_context_field,
            history_limit=self.entity_sequence_history,
            default_entity_id=self.entity_default_id,
            default_markov_topk=self.markov_topk,
            default_observation_topk=self.entity_sequence_observation_topk,
            default_missing_tolerance=self.entity_sequence_missing_tolerance,
        )

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        log_dir = run_cfg.get("log_dir", None)
        classification_mode = run_cfg.get("classification_mode", "generate")
        event_taxonomy_path = run_cfg.get("event_taxonomy_path", None)
        classifier_prompt = run_cfg.get("classifier_prompt", "{}")
        rank_n_segments = run_cfg.get("rank_n_segments", 1)
        rank_topk = run_cfg.get("rank_topk", 5)
        observation_classifier_path = run_cfg.get("observation_classifier_path", None)
        markov_chain_path = run_cfg.get("markov_chain_path", None)
        markov_sequence_mode = run_cfg.get(
            "markov_sequence_mode", "prefix_before_underscore"
        )
        markov_topk = run_cfg.get("markov_topk", 5)
        markov_context_field = run_cfg.get("markov_context_field", "ecological_context")
        markov_debug = run_cfg.get("markov_debug", False)
        entity_default_id = run_cfg.get("entity_default_id", "__scene__")
        entity_sequence_history = run_cfg.get("entity_sequence_history", 64)
        entity_sequence_observation_topk = run_cfg.get(
            "entity_sequence_observation_topk", 5
        )
        entity_sequence_missing_tolerance = run_cfg.get(
            "entity_sequence_missing_tolerance", 0
        )

        report_metric = run_cfg.get("report_metric", True)
        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            dataset_name=list(cfg.datasets_cfg.keys())[0],
            log_dir=log_dir,
            classification_mode=classification_mode,
            event_taxonomy_path=event_taxonomy_path,
            classifier_prompt=classifier_prompt,
            rank_n_segments=rank_n_segments,
            rank_topk=rank_topk,
            observation_classifier_path=observation_classifier_path,
            markov_chain_path=markov_chain_path,
            markov_sequence_mode=markov_sequence_mode,
            markov_topk=markov_topk,
            markov_context_field=markov_context_field,
            markov_debug=markov_debug,
            entity_default_id=entity_default_id,
            entity_sequence_history=entity_sequence_history,
            entity_sequence_observation_topk=entity_sequence_observation_topk,
            entity_sequence_missing_tolerance=entity_sequence_missing_tolerance,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)
        dataset_splits = datasets.get(self.dataset_name, {})
        self.default_candidate_labels = self._extract_label_space(dataset_splits)
        return datasets

    def _extract_label_space(self, dataset_splits):
        label_set = set()
        for _, dataset in dataset_splits.items():
            ann = getattr(dataset, "annotation", None)
            if ann is None:
                continue

            if isinstance(ann, dict):
                iterable = ann.values()
            else:
                iterable = ann

            for item in iterable:
                if not isinstance(item, dict):
                    continue
                label = item.get("label")
                if label:
                    label_set.add(label)

        return sorted(label_set)

    def _select_classifier_candidates(self, question_text):
        if self.event_taxonomy is not None:
            return self.event_taxonomy.select_candidates(
                dataset_name=self.dataset_name,
                question_text=question_text,
                fallback_labels=self.default_candidate_labels,
                fallback_prompt=self.classifier_prompt,
            )

        return {
            "classifier_id": "dataset_label_space",
            "prompt": self.classifier_prompt,
            "candidate_labels": list(self.default_candidate_labels),
            "candidate_event_ids": [None] * len(self.default_candidate_labels),
        }

    @staticmethod
    def _row_tensor(data, row_idx):
        if isinstance(data, list):
            row = data[row_idx]
            if row.dim() == 2:
                row = row[0]
            return row
        return data[row_idx]

    def _infer_sequence_id(self, image_id):
        image_id = str(image_id)
        if self.markov_sequence_mode == "image_id":
            return image_id
        if self.markov_sequence_mode == "prefix_before_underscore":
            if "_" in image_id:
                return image_id.split("_")[0]
            return image_id
        return image_id

    @staticmethod
    def _sort_dict_items_by_value(dct):
        return sorted(dct.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _normalize_batch_field(value, batch_size):
        if value is None:
            return [None] * batch_size
        if isinstance(value, str):
            return [value] * batch_size
        if torch.is_tensor(value):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            out = list(value)
            if len(out) < batch_size:
                out.extend([None] * (batch_size - len(out)))
            return out[:batch_size]
        return [value] * batch_size

    @staticmethod
    def _normalize_entity_observations(value):
        if value is None:
            return []
        if isinstance(value, dict):
            if isinstance(value.get("entities"), list):
                return [x for x in value.get("entities") if isinstance(x, dict)]
            return [value]
        if isinstance(value, (list, tuple)):
            return [x for x in value if isinstance(x, dict)]
        return []

    def _entity_observation_scores(
        self,
        entity_item,
        fallback_context,
        fallback_question,
        fallback_classifier_id,
        fallback_event_predictions,
        fallback_observation_scores,
        fallback_candidate_event_ids,
    ):
        raw_obs = entity_item.get("observation_scores")
        observation_scores = normalize_observation_scores(raw_obs)

        entity_event_predictions = entity_item.get("event_predictions")
        if not isinstance(entity_event_predictions, list):
            entity_event_predictions = fallback_event_predictions

        if len(observation_scores) > 0:
            return observation_scores, entity_event_predictions

        model_score_map = scores_from_event_predictions(entity_event_predictions)
        candidate_event_ids = entity_item.get("candidate_event_ids")
        if not isinstance(candidate_event_ids, list) or len(candidate_event_ids) == 0:
            candidate_event_ids = [
                pred.get("event_id")
                for pred in entity_event_predictions
                if isinstance(pred, dict) and pred.get("event_id")
            ]
        candidate_event_ids = [str(x) for x in candidate_event_ids if x]
        if len(candidate_event_ids) == 0:
            candidate_event_ids = list(fallback_candidate_event_ids)

        if (
            self.observation_classifier_set is not None
            and len(candidate_event_ids) > 0
            and len(entity_event_predictions) > 0
        ):
            obs_context = {
                "image_id": entity_item.get("image_id", None),
                "question": entity_item.get("question", fallback_question),
                "classifier_id": entity_item.get(
                    "classifier_id", fallback_classifier_id
                ),
                self.markov_context_field: entity_item.get(
                    self.markov_context_field,
                    entity_item.get("context", fallback_context),
                ),
                "event_predictions": entity_event_predictions,
            }
            try:
                observation_scores = self.observation_classifier_set.score_events(
                    base_context=obs_context,
                    candidate_event_ids=candidate_event_ids,
                    model_scores=model_score_map,
                )
            except Exception as exc:
                logging.warning(
                    f"Entity observation scoring failed for entity_id="
                    f"{entity_item.get('entity_id', self.entity_default_id)}: {exc}. "
                    "Falling back to model score map."
                )
                observation_scores = {}

        if len(observation_scores) == 0:
            if len(candidate_event_ids) > 0:
                observation_scores = {
                    event_id: float(model_score_map.get(event_id, 0.0))
                    for event_id in candidate_event_ids
                }
            elif len(model_score_map) > 0:
                observation_scores = {k: float(v) for k, v in model_score_map.items()}
            else:
                observation_scores = normalize_observation_scores(fallback_observation_scores)

        return observation_scores, entity_event_predictions

    def valid_step(self, model, samples):
        if self.classification_mode == "rank":
            return self._valid_step_rank(model=model, samples=samples)
        return self._valid_step_generate(model=model, samples=samples)

    def _valid_step_generate(self, model, samples):
        results = []
        generated = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            num_captions=self.num_beams,
        )

        img_ids = samples["image_id"]
        for i, img_id in enumerate(img_ids):
            caption_list = generated[i * self.num_beams : (i + 1) * self.num_beams]
            results.append({"caption": caption_list, "image_id": img_id})
        return results

    def _valid_step_rank(self, model, samples):
        image_ids = samples.get("image_id", samples.get("question_id", None))
        if image_ids is None:
            image_ids = [str(i) for i in range(samples["image"].size(0))]

        text_inputs = samples.get("text_input", [""] * len(image_ids))
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        ecological_contexts = self._normalize_batch_field(
            samples.get(self.markov_context_field, None),
            len(image_ids),
        )
        entity_observations_batch = self._normalize_batch_field(
            samples.get("entity_observations", None),
            len(image_ids),
        )
        has_explicit_entity_observations = "entity_observations" in samples

        classifier_ids = []
        prompts = []
        candidate_labels = []
        candidate_event_ids = []

        for question in text_inputs:
            selection = self._select_classifier_candidates(question)
            labels = selection["candidate_labels"]
            event_ids = selection["candidate_event_ids"]

            if len(labels) == 0:
                labels = ["unknown"]
                event_ids = [None]

            classifier_ids.append(selection["classifier_id"])
            prompts.append(selection["prompt"])
            candidate_labels.append(labels)
            candidate_event_ids.append(event_ids)

        same_candidates = all(labels == candidate_labels[0] for labels in candidate_labels)
        model_candidates = candidate_labels[0] if same_candidates else candidate_labels

        rank_samples = {
            "image": samples["image"],
            "text_input": text_inputs,
            "prompt": prompts,
        }
        for key in ["context", "history", "caption"]:
            if key in samples:
                rank_samples[key] = samples[key]

        rank_output = model.predict_class(
            rank_samples,
            candidates=model_candidates,
            n_segments=self.rank_n_segments,
            return_losses=True,
        )
        ranks = rank_output["ranks"]
        losses = rank_output["losses"]

        results = []
        for i, image_id in enumerate(image_ids):
            rank_i = self._row_tensor(ranks, i)
            loss_i = self._row_tensor(losses, i).float()
            probs_i = torch.softmax(-loss_i, dim=-1)

            labels_i = candidate_labels[i]
            event_ids_i = candidate_event_ids[i]
            topk = min(self.rank_topk, len(labels_i))
            best_indices = rank_i[:topk].tolist()

            caption_list = []
            event_predictions = []
            for rank_position, idx in enumerate(best_indices, start=1):
                idx = int(idx)
                label = labels_i[idx]
                caption_list.append(label)
                event_predictions.append(
                    {
                        "rank": rank_position,
                        "candidate_index": idx,
                        "label": label,
                        "event_id": event_ids_i[idx],
                        "confidence": float(probs_i[idx].item()),
                    }
                )

            base_sequence_id = self._infer_sequence_id(image_id)

            model_score_map = {
                pred["event_id"]: pred["confidence"]
                for pred in event_predictions
                if pred.get("event_id")
            }
            candidate_event_ids_i = [event_id for event_id in event_ids_i if event_id]

            observation_context = {
                "image_id": image_id,
                "question": text_inputs[i],
                "classifier_id": classifier_ids[i],
                self.markov_context_field: ecological_contexts[i],
                "event_predictions": event_predictions,
            }

            if self.observation_classifier_set is not None:
                observation_scores = self.observation_classifier_set.score_events(
                    base_context=observation_context,
                    candidate_event_ids=candidate_event_ids_i,
                    model_scores=model_score_map,
                )
            else:
                observation_scores = {
                    event_id: float(model_score_map.get(event_id, 0.0))
                    for event_id in candidate_event_ids_i
                }

            markov_sequence_id = None
            markov_posterior = None
            markov_state = None
            markov_debug = None
            if self.event_markov_chain is not None and len(observation_scores) > 0:
                markov_sequence_id = base_sequence_id
                markov_context = {
                    self.markov_context_field: ecological_contexts[i],
                    "image_id": image_id,
                    "question": text_inputs[i],
                }
                if self.markov_debug:
                    posterior_map, markov_debug = self.event_markov_chain.update(
                        sequence_id=markov_sequence_id,
                        observation_scores=observation_scores,
                        context=markov_context,
                        return_debug=True,
                    )
                else:
                    posterior_map = self.event_markov_chain.update(
                        sequence_id=markov_sequence_id,
                        observation_scores=observation_scores,
                        context=markov_context,
                    )
                posterior_sorted = self._sort_dict_items_by_value(posterior_map)
                posterior_top = posterior_sorted[: self.markov_topk]
                markov_posterior = [
                    {"event_id": event_id, "prob": float(prob)}
                    for event_id, prob in posterior_top
                ]
                if posterior_sorted:
                    markov_state = {
                        "event_id": posterior_sorted[0][0],
                        "prob": float(posterior_sorted[0][1]),
                    }

            observation_score_items = self._sort_dict_items_by_value(observation_scores)
            observation_score_items = observation_score_items[: self.rank_topk]
            observation_score_list = [
                {"event_id": event_id, "score": float(score)}
                for event_id, score in observation_score_items
            ]

            tracker_window_context = {
                self.markov_context_field: ecological_contexts[i],
                "image_id": image_id,
                "question": text_inputs[i],
            }
            self.entity_sequence_tracker.begin_window(
                base_sequence_id=base_sequence_id,
                context=tracker_window_context,
                image_id=image_id,
                question=text_inputs[i],
                metadata={"classifier_id": classifier_ids[i]},
            )

            entity_items = self._normalize_entity_observations(entity_observations_batch[i])
            if len(entity_items) == 0 and not has_explicit_entity_observations:
                entity_items = [
                    {
                        "entity_id": self.entity_default_id,
                        "observation_scores": observation_scores,
                        "event_predictions": event_predictions,
                        self.markov_context_field: ecological_contexts[i],
                        "question": text_inputs[i],
                    }
                ]

            entity_event_sequences = []
            for entity_item in entity_items:
                entity_obs_scores, entity_event_predictions = self._entity_observation_scores(
                    entity_item=entity_item,
                    fallback_context=ecological_contexts[i],
                    fallback_question=text_inputs[i],
                    fallback_classifier_id=classifier_ids[i],
                    fallback_event_predictions=event_predictions,
                    fallback_observation_scores=observation_scores,
                    fallback_candidate_event_ids=candidate_event_ids_i,
                )
                if len(entity_obs_scores) == 0:
                    continue

                entity_context = {
                    self.markov_context_field: entity_item.get(
                        self.markov_context_field,
                        entity_item.get("context", ecological_contexts[i]),
                    ),
                    "image_id": image_id,
                    "question": entity_item.get("question", text_inputs[i]),
                    "entity_id": entity_item.get("entity_id", self.entity_default_id),
                }
                entity_summary = self.entity_sequence_tracker.update_entity(
                    base_sequence_id=base_sequence_id,
                    entity_id=entity_item.get("entity_id", self.entity_default_id),
                    observation_scores=entity_obs_scores,
                    context=entity_context,
                    image_id=image_id,
                    question=entity_item.get("question", text_inputs[i]),
                    metadata=entity_item.get("metadata", {}),
                    markov_debug=self.markov_debug,
                    markov_topk=self.markov_topk,
                    observation_topk=self.entity_sequence_observation_topk,
                )
                if entity_summary is None:
                    continue
                if entity_event_predictions is not None:
                    entity_summary["event_predictions"] = entity_event_predictions
                entity_event_sequences.append(entity_summary)

            entity_lifecycle = self.entity_sequence_tracker.finalize_window(
                base_sequence_id=base_sequence_id,
                missing_tolerance=self.entity_sequence_missing_tolerance,
            )

            results.append(
                {
                    "image_id": image_id,
                    "caption": caption_list,
                    "classifier_id": classifier_ids[i],
                    "question": text_inputs[i],
                    self.markov_context_field: ecological_contexts[i],
                    "event_predictions": event_predictions,
                    "observation_scores": observation_score_list,
                    "markov_sequence_id": markov_sequence_id,
                    "markov_posterior": markov_posterior,
                    "markov_state": markov_state,
                    "markov_debug": markov_debug,
                    "entity_event_sequences": entity_event_sequences,
                    "entity_lifecycle": entity_lifecycle,
                }
            )

        return results

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10

        results = []
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()
        return results

    def after_evaluation(self, val_result, split_name, epoch, dataset, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics_cls(
                eval_result_file=eval_result_file, split_name=split_name, dataset=dataset
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics_cls(self, eval_result_file, split_name, dataset):
        gt_dict = dataset.annotation

        with open(eval_result_file, "r") as fp:
            prediction_list = json.load(fp)

        match_video_list = []
        for prediction in prediction_list:
            image_id = prediction["image_id"]
            caption_list = prediction["caption"]
            label = gt_dict[image_id]["label"]

            match_video = [1 if caption == label else 0 for caption in caption_list]
            match_video_list.append(match_video)
        match = np.array(match_video_list)

        top_1 = match[:, :1].max(1).mean() * 100
        top_5 = match[:, :5].max(1).mean() * 100

        result = {"top1": top_1, "top5": top_5}

        print(f"top1: {top_1:.2f} top5: {top_5:.2f}\n")
        result["agg_metrics"] = result["top1"]
        return result
