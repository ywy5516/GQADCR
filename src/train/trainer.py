#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import requests
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.trainer_utils import EvalPrediction, seed_worker, speed_metrics

BASENAME = Path(__file__).stem

from src.eval import CustomeEvaluator
from src.model import CustomBertModel, CustomDPRModel, CustomRobertaModel
from src.train.loss import calculate_align_loss, calculate_rank_loss
from src.train.script_args import CustomScriptArguments


class CustomTrainer(Trainer):
    def __init__(
        self,
        model: Union[CustomBertModel, CustomRobertaModel, CustomDPRModel],
        frozen_model: Union[CustomBertModel, CustomRobertaModel, CustomDPRModel],
        args: TrainingArguments,
        script_args: CustomScriptArguments,
        data_collator=None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class,
            model_init,
            compute_loss_func,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.frozen_model = frozen_model
        self.script_args = script_args
        self.evaluator = CustomeEvaluator(self.script_args.qrel_trec_file)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.script_args.train_dataset_ratio < 1.0:
            train_size = int(len(train_dataset) * self.script_args.train_dataset_ratio)
            dataloader_params["sampler"] = RandomSampler(train_dataset, replacement=False, num_samples=train_size)
        else:
            dataloader_params["shuffle"] = True

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def encode(self, model, texts: List[str], max_length: int, titles: Optional[List[str]] = None) -> Tensor:
        if titles is not None:
            assert len(texts) == len(titles), f"If titles is set, the length of texts and titles must be equal"
            texts = [f"{title} {text}" for title, text in zip(titles, texts)]
        inputs = self.processing_class(
            texts,
            max_length=max_length,
            padding="longest",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        outputs = cast(BaseModelOutputWithPooling, model(**inputs))
        return outputs.pooler_output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # bt_sample_ids = cast(List[str], inputs["sample_id"])
        bt_queries = cast(List[str], inputs["query"])
        bt_rewrite_queries = cast(List[str], inputs["rewrite_query"])
        bt_pos_passages = cast(List[str], inputs["positives"])
        bt_neg_passages = cast(List[List[str]], inputs["negatives"])

        bt_query_embs = self.encode(model, bt_queries, self.script_args.max_concat_length)
        with torch.no_grad():
            bt_rewrite_embs = self.encode(self.frozen_model, bt_rewrite_queries, self.script_args.max_query_length)
            bt_positive_embs = self.encode(self.frozen_model, bt_pos_passages, self.script_args.max_passage_length)
            bt_negative_embs = [
                self.encode(self.frozen_model, neg_lst, self.script_args.max_passage_length)
                for neg_lst in bt_neg_passages
            ]  # List[torch.Tensor]
            bt_negative_embs = torch.stack(bt_negative_embs, dim=0)  # (B, N_neg, D)

        if self.script_args.loss_ablation == "none":
            rank_loss = calculate_rank_loss(
                bt_query_embs,
                bt_positive_embs,
                bt_negative_embs,
                model.device,
                self.script_args.loss_normalize,
                self.script_args.rank_temperature,
            )
            align_loss = calculate_align_loss(
                bt_query_embs,
                bt_rewrite_embs,
                model.device,
                self.script_args.loss_normalize,
                self.script_args.align_temperature,
            )
            return self.script_args.alpha * align_loss + self.script_args.beta * rank_loss
        elif self.script_args.loss_ablation == "no_query":
            return calculate_rank_loss(
                bt_query_embs,
                bt_positive_embs,
                bt_negative_embs,
                model.device,
                self.script_args.loss_normalize,
                self.script_args.rank_temperature,
            )
        else:
            return calculate_align_loss(
                bt_query_embs,
                bt_rewrite_embs,
                model.device,
                self.script_args.loss_normalize,
                self.script_args.align_temperature,
            )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss, None, None

    def _convert_retrieval_result_to_eval_format(self, retrieval_result: Dict[str, list]) -> dict:
        data = dict()
        for sample_id, node_lst in retrieval_result.items():
            if sample_id not in data:
                data[sample_id] = dict()
            for i, node in enumerate(node_lst):
                passage_id = node["id"]
                relevance = 200 - i - 1
                data[sample_id][passage_id] = relevance
        return data

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict:
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = dict()
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()
        model = self.model.eval()
        metrics = dict()
        num_samples = len(eval_dataset)
        if self.script_args.use_retriever and self.script_args.retriever_type == "dense":
            results = dict()  # Dict[str, List[dict]]
            batch_url = f"http://0.0.0.0:{self.script_args.port}/batch_retrieve_from_embs"
            dataloader = DataLoader(eval_dataset, batch_size=self.script_args.retriever_batch_size)
            for batch in tqdm(
                dataloader, desc=f"[{BASENAME}] retrieval in progress, the length of dataloader is {len(dataloader)}"
            ):
                bt_queries = cast(List[str], batch["query"])
                with torch.no_grad():
                    bt_embs = self.encode(model, bt_queries, self.script_args.max_concat_length).cpu()
                if self.script_args.retriever_normalize:
                    bt_embs = torch.nn.functional.normalize(bt_embs, p=2, dim=-1)
                payload = {
                    "query_embs": bt_embs.tolist(),
                    "query_ids": cast(List[str], batch["sample_id"]),
                    "top_k": 100,
                }
                response = requests.post(batch_url, json=payload)
                response.raise_for_status()
                results.update(cast(dict, response.json()))
            runs_data = self._convert_retrieval_result_to_eval_format(results)  # Dict[str, Dict[str, int]]
            retrieval_metric = self.evaluator.evaluate_from_data(runs_data)
            for k, v in retrieval_metric.items():
                metrics[f"{metric_key_prefix}_{k}"] = v
        else:
            all_losses = list()
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc=f"[{BASENAME}] evaluate compute loss"):
                    loss = self.compute_loss(model, batch)
                    all_losses.append(loss.cpu().float())
            metrics[f"{metric_key_prefix}_loss"] = np.mean(all_losses).item()
        torch.cuda.empty_cache()

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix, start_time, num_samples=num_samples, num_steps=int(num_samples / total_batch_size)
            )
        )
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics
