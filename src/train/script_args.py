#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.config import MAX_CONCAT_LENGTH, MAX_PASSAGE_LENGTH, MAX_QUERY_LENGTH


@dataclass
class CustomScriptArguments:
    train_model_path: str
    train_model_cls: Literal["dpr", "roberta", "bert"]
    train_model_type: Literal["query", "passage"]
    frozen_model_path: str
    frozen_model_cls: Literal["dpr", "roberta", "bert"]
    frozen_model_type: Literal["query", "passage"]
    pooling_mode: Literal["cls", "mean"]

    input_key: Literal["convq", "convqa"]
    rewrite_key: str
    train_input_dir: str
    test_input_dir: str
    qrel_trec_file: str  # for test dataset

    device_map: str = field(default="cuda:0")
    train_dataset_ratio: float = field(default=1.0)
    only_last_turn: bool = field(default=False)

    port: Optional[int] = field(default=None)  # for dense retriever
    use_retriever: bool = field(default=False)
    retriever_type: Optional[str] = field(default=None)
    retriever_normalize: bool = field(default=True)
    retriever_batch_size: Optional[int] = field(default=None)

    alpha: float = field(default=1.0)
    beta: float = field(default=1.0)
    loss_normalize: bool = field(default=True)
    align_temperature: float = field(default=0.07)
    rank_temperature: float = field(default=0.07)

    max_query_length: int = field(default=MAX_QUERY_LENGTH)
    max_passage_length: int = field(default=MAX_PASSAGE_LENGTH)
    max_concat_length: int = field(default=MAX_CONCAT_LENGTH)

    cache_dir: str = field(default="cache/train")
    loss_ablation: Literal["none", "no_query", "no_passage"] = field(default="none")
