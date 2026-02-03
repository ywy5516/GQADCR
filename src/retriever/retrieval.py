#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List, cast

import requests
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.std import tqdm
from transformers.trainer_utils import set_seed

sys.path.append(Path(".").absolute().as_posix())

from src.config import MAX_CONCAT_LENGTH
from src.data import RetrievalDataset
from src.utils import check_if_dir_file_exist

BASENAME = Path(__file__).stem


def retrieve_results(dataset, batch_size: int, port: int, max_length: int, is_normalize: bool) -> dict:
    results = dict()  # {sample_id: [{id1: score1}, {id2: score2}, ...]}
    batch_url = f"http://0.0.0.0:{port}/batch_retrieve"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    for batch in tqdm(dataloader, desc=f"[{BASENAME}] retrieval in progress"):
        payload = {
            "queries": cast(List[str], batch["query"]),
            "query_ids": cast(List[str], batch["sample_id"]),
            "max_length": max_length,
            "is_normalize": is_normalize,
            "top_k": 100,
        }
        response = requests.post(batch_url, json=payload)
        response.raise_for_status()
        results.update(cast(dict, response.json()))
    return results  # Dict[str, List[dict]]


if __name__ == "__main__":
    # python src/retriever/retrieval.py
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--input_key", type=str, required=True)
    parser.add_argument("--only_last_turn", action="store_true")
    parser.add_argument("--max_length", type=int, default=MAX_CONCAT_LENGTH)
    parser.add_argument("--is_normalize", action="store_true")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()
    set_seed(args.seed)
    check_if_dir_file_exist(args.input_dir)

    retriever_type = requests.get(f"http://0.0.0.0:{args.port}/get_retriever_type").json()["type"]
    if retriever_type not in ("sparse", "dense"):
        raise ValueError(f"[{BASENAME}] invalid retriever type: {retriever_type}")

    start_time = datetime.now()
    dataset = RetrievalDataset(args.input_dir, args.input_key, only_last_turn=args.only_last_turn)
    logger.debug(f"[{BASENAME}] the length of dataset is {len(dataset)}")

    min_batch_size = 10 if retriever_type == "sparse" else 1500
    batch_size = min_batch_size if not args.batch_size else min(min_batch_size, args.batch_size)
    all_results = retrieve_results(dataset, batch_size, args.port, args.max_length, args.is_normalize)
    elapse_time = datetime.now() - start_time
    logger.info(f"[{BASENAME}] total time: {str(elapse_time)}")

    with open(args.save_path, "w", encoding="utf-8") as f:
        for sample_id, node_lst in all_results.items():
            for i, node in enumerate(node_lst):
                passage_id, score = node["id"], node["score"]
                # sample_id 0 passage_id rank rank_score real_score
                f.write(f"{sample_id} 0 {passage_id} {i + 1} {200 - i - 1} {score}")
                f.write("\n")
    logger.success(f"[{BASENAME}] retrieval result is saved to {args.save_path}")
