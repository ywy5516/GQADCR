#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from typing import Tuple, cast

import requests
from loguru import logger
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.trainer_utils import set_seed

sys.path.append(os.path.abspath("."))

from src.data import DevDataset, TrainDataset
from src.model import load_model
from src.train.script_args import CustomScriptArguments
from src.train.trainer import CustomTrainer
from src.utils import check_if_dir_file_exist

BASENAME = Path(__file__).stem


def get_retriever_type(port: int):
    try:
        retriever_type = requests.get(f"http://0.0.0.0:{port}/get_retriever_type").json()["type"]
        if retriever_type not in ("sparse", "dense"):
            raise ValueError(f"[{BASENAME}] invalid retriever type: {retriever_type}")
        return retriever_type
    except Exception as e:
        logger.error(f"[{BASENAME}] Failed to get the retriever's type: {e}")


if __name__ == "__main__":
    # python src/train/main.py
    parser = HfArgumentParser((CustomScriptArguments, TrainingArguments))
    script_args, training_args = cast(
        Tuple[CustomScriptArguments, TrainingArguments], parser.parse_args_into_dataclasses()
    )
    set_seed(training_args.seed)
    check_if_dir_file_exist(
        script_args.train_model_path,
        script_args.frozen_model_path,
        script_args.train_input_dir,
        script_args.test_input_dir,
        script_args.qrel_trec_file,
    )
    if script_args.use_retriever is True:
        if script_args.port is None:
            raise ValueError(f"[{BASENAME}] `port` must be specified when using evaluation metrics")
        if get_retriever_type(script_args.port) != script_args.retriever_type:
            raise ValueError(f"[{BASENAME}] the `retriever_type` does not match the actual implementation.")
        if not script_args.retriever_batch_size:
            raise ValueError(f"[{BASENAME}] when using the retriever, you must specify the `retriever_batch_size`.")
        min_batch_size = 10 if script_args.retriever_type == "sparse" else 1500
        if not script_args.retriever_batch_size:
            script_args.retriever_batch_size = min_batch_size
        else:
            script_args.retriever_batch_size = min(min_batch_size, script_args.retriever_batch_size)

    train_model = load_model(
        script_args.train_model_cls,
        script_args.train_model_type,
        script_args.train_model_path,
        script_args.device_map,
        script_args.pooling_mode,
    )
    frozen_model = load_model(
        script_args.frozen_model_cls,
        script_args.frozen_model_type,
        script_args.frozen_model_path,
        script_args.device_map,
        script_args.pooling_mode,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.train_model_path, use_fast=False, clean_up_tokenization_spaces=True
    )
    frozen_model.requires_grad_(False)
    frozen_model.eval()

    Path(script_args.cache_dir).mkdir(exist_ok=True, parents=True)
    train_dataset = TrainDataset(
        script_args.train_input_dir,
        script_args.input_key,
        script_args.rewrite_key,
        only_last_turn=script_args.only_last_turn,
    )
    test_dataset = DevDataset(
        script_args.test_input_dir,
        script_args.input_key,
        only_last_turn=script_args.only_last_turn,
    )
    data_collator = train_dataset.custom_collate_fn

    trainer = CustomTrainer(
        model=train_model,
        frozen_model=frozen_model,
        args=training_args,
        script_args=script_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
