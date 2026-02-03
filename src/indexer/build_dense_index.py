#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer_utils import set_seed

sys.path.append(os.path.abspath("."))

from src.config import MAX_PASSAGE_LENGTH
from src.data import CustomCorpusDataset
from src.model import BaseEncoder
from src.utils import check_if_dir_file_exist

BASENAME = Path(__file__).stem
NUM_PER_BLOCK_DOCS = 5000000


def save_to_npy(passage_ids: np.ndarray, passage_embs: np.ndarray, block_id: int, save_dir: str):
    id_save_path = os.path.join(save_dir, f"id_block_{block_id}.npy")
    emb_save_path = os.path.join(save_dir, f"emb_block_{block_id}.npy")
    np.save(id_save_path, passage_ids)
    np.save(emb_save_path, passage_embs)


def build_dense_index(args):
    check_if_dir_file_exist(args.corpus_dir, args.pretrained_model_path)
    encoder = BaseEncoder(args.model_cls, args.model_type, args.pretrained_model_path, args.device, args.pooling_mode)
    encoder.model.requires_grad_(False)
    encoder.model.eval()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    dataset = CustomCorpusDataset(args.corpus_dir, args.start, args.end)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, drop_last=False)
    logger.info(f"[{BASENAME}] total length of dataset is {len(dataset)}")

    start_time = datetime.now()
    passage_ids, passage_embs = list(), list()
    count_embs = 0
    cur_block_id = args.start_block_id
    for batch in tqdm(dataloader, desc=f"[{BASENAME}] building index from {Path(args.corpus_dir).name}"):
        try:
            bt_ids = batch["id"].tolist()
        except:
            bt_ids = batch["id"]
        with torch.no_grad():
            bt_outputs = encoder.encode(batch["contents"], max_length=args.max_length)
        bt_embs = bt_outputs.pooler_output.cpu()
        if args.is_normalize:
            bt_embs = torch.nn.functional.normalize(bt_embs, p=2, dim=-1)  # 归一化
        assert len(bt_ids) == bt_embs.shape[0], f"[{BASENAME}] batch size 不一致"
        passage_ids.extend(bt_ids[:])
        passage_embs.append(bt_embs.numpy())

        if len(passage_ids) >= NUM_PER_BLOCK_DOCS:
            save_to_npy(np.array(passage_ids), np.concatenate(passage_embs, axis=0), cur_block_id, args.save_dir)
            count_embs += len(passage_ids)
            del passage_ids, passage_embs
            gc.collect()
            elapse_time = datetime.now() - start_time
            logger.debug(f"[{BASENAME}] passages saved in block {cur_block_id}, elapsed time: {elapse_time}")

            start_time = datetime.now()
            passage_ids, passage_embs = list(), list()
            cur_block_id += 1
    if len(passage_ids) > 0:
        logger.warning(f"[{BASENAME}] the final remaining number is {len(passage_ids)}")
        save_to_npy(np.array(passage_ids), np.concatenate(passage_embs, axis=0), cur_block_id, args.save_dir)
        count_embs += len(passage_ids)
        del passage_ids, passage_embs
        gc.collect()
    logger.success(f"[{BASENAME}] {count_embs} passages are saved in {cur_block_id - args.start_block_id + 1} blocks")


if __name__ == "__main__":
    # python src/indexer/build_dense_index.py
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus_dir", type=str, default="corpus/parsed_qrecc")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--start_block_id", type=int, default=1)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--model_cls", type=str, choices=["dpr", "roberta", "bert"], required=True)
    parser.add_argument("--model_type", type=str, choices=["query", "passage"], required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--pooling_mode", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument("--is_normalize", action="store_true")
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=MAX_PASSAGE_LENGTH)

    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO"], default="DEBUG")
    parser.add_argument("--log_path", type=str, default="log/build_dense_index.log")

    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add(args.log_path, rotation="5 MB", retention="1 week", level="DEBUG")

    set_seed(args.seed)
    logger.debug(vars(args))
    build_dense_index(args)
