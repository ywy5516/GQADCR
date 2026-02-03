#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, cast

import requests
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer_utils import set_seed

sys.path.append(os.path.abspath("."))

from src.config import MAX_QUERY_LENGTH
from src.data import RetrievalDataset
from src.utils import (
    check_if_dir_file_exist,
    join_and_filter_dirs,
    join_and_filter_files,
    load_all_corpus_map,
    load_json,
    write_json,
)

BASENAME = Path(__file__).stem


def retrieve_results(args, retriever_type: str) -> dict:
    dataset = RetrievalDataset(args.input_dir, args.input_key, only_last_turn=False)
    logger.debug(f"[{BASENAME}] the length of dataset is {len(dataset)}")
    if retriever_type == "sparse":
        batch_size = 5
    else:
        batch_size = min(8500, len(dataset)) if not args.batch_size else args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    results = dict()  # {sample_id: [{id1: score1}, {id2: score2}]}
    batch_url = f"http://0.0.0.0:{args.port}/batch_retrieve"

    for batch in tqdm(dataloader, desc=f"[{BASENAME}] {retriever_type} retrieval in progress"):
        payload = {
            "queries": cast(List[str], batch["query"]),
            "query_ids": cast(List[str], batch["sample_id"]),
            "max_length": args.max_length,
            "is_normalize": args.is_normalize,
            "top_k": args.top_k,
        }
        response = requests.post(batch_url, json=payload)
        response.raise_for_status()
        results.update(cast(dict, response.json()))
    return results


def main(args):
    set_seed(args.seed)
    check_if_dir_file_exist(args.corpus_dir, args.input_dir)
    retriever_type = requests.get(f"http://0.0.0.0:{args.port}/get_retriever_type").json()["type"]
    if retriever_type not in ("sparse", "dense"):
        raise ValueError(f"[{BASENAME}] invalid retriever type {retriever_type}")

    results = retrieve_results(args, retriever_type)  # {sample_id: [{id1: score1}, {id2: score2}]}
    pids_to_texts = load_all_corpus_map(args.corpus_dir)  # Load all passage_id: texts mappings

    conversation_dirs = join_and_filter_dirs(args.input_dir, sorted(os.listdir(args.input_dir)))
    for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] loading all conversations"):
        turn_file_list = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
        for turn_file in turn_file_list:
            data = cast(dict, load_json(turn_file))
            if data["sample_id"] not in results or len(data["positive_pids"]) == 0:
                data[f"{args.output_key}_pids"] = list()
                data[f"{args.output_key}_passages"] = list()
                write_json(data, turn_file)
                continue

            nodes = cast(List[dict], results[data["sample_id"]])
            retrieval_pids = [int(node["id"]) for node in nodes][:10]
            exist_passage_ids = set(data["positive_pids"])
            if data.get("ctx_positive_pids"):
                for ctx_pids in cast(List[List[int]], data["ctx_positive_pids"]):
                    exist_passage_ids.update(ctx_pids)
            hard_negative_pids = [pid for pid in retrieval_pids if pid not in exist_passage_ids]
            hard_negative_passages = [pids_to_texts.get(pid) for pid in hard_negative_pids]
            data[f"{args.output_key}_pids"] = hard_negative_pids
            data[f"{args.output_key}_passages"] = hard_negative_passages
            write_json(data, turn_file)
    logger.success(f"[{BASENAME}] {Path(args.input_dir).name} sampling completed.")


if __name__ == "__main__":
    # python src/retriever/negative_sample.py
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--input_key", type=str, required=True, choices=["cur_utt", "convq", "convqa"])
    parser.add_argument("--output_key", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=MAX_QUERY_LENGTH, help="64 384 512")
    parser.add_argument("--is_normalize", action="store_true")
    parser.add_argument("--top_k", type=int, default=100)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--port", type=int, required=True)

    main(parser.parse_args())
