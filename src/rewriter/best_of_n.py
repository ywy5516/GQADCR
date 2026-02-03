#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, cast

import numpy as np
import requests
from loguru import logger
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from src.utils import (
    check_if_dir_file_exist,
    join_and_filter_dirs,
    join_and_filter_files,
    load_json,
    md5_str,
    pickle_load,
    pickle_store,
)

BASENAME = Path(__file__).stem


def load_qrels(qrel_trec_file: str) -> Dict[str, Dict[str, int]]:
    with open(qrel_trec_file, "r", encoding="utf-8") as f:
        qrel_data = f.readlines()
    qrels = dict()
    for line in qrel_data:
        line = line.strip().split()
        sample_id, passage_id, relevance = line[0], line[2], int(line[3])
        if sample_id not in qrels:
            qrels[sample_id] = dict()
        qrels[sample_id][passage_id] = 1 if relevance >= 1 else 0
    return qrels


def mrr(run_data: Dict[str, Dict[str, float]], qrel_data: Dict[str, Dict[str, int]]):
    mrr_lst = list()
    for sample_id, run_item in run_data.items():
        run_lst = sorted(run_item.items(), key=lambda x: x[1], reverse=True)
        qrel_lst = list()
        for passage_id, _ in run_lst:
            qrel_lst.append(qrel_data[sample_id].get(passage_id, 0))

        tmp_mrr = 0.0
        for i, rel in enumerate(qrel_lst):
            if rel >= 1:
                tmp_mrr = 1.0 / (i + 1)
                break
        mrr_lst.append(tmp_mrr)
    return round(np.average(mrr_lst) * 100, 3)


def main(args):
    check_if_dir_file_exist(args.input_dir, args.qrel_trec_file)
    retriever_type = requests.get(f"http://0.0.0.0:{args.port}/get_retriever_type").json()["type"]
    if retriever_type not in ("sparse", "dense"):
        raise ValueError(f"[{BASENAME}] invalid retriever type: {retriever_type}")
    elif retriever_type == "dense":
        raise NotImplementedError(f"[{BASENAME}] has not implemented dense retrieval")
    retrieve_url = f"http://0.0.0.0:{args.port}/retrieve"

    qrels = load_qrels(args.qrel_trec_file)
    start_time = datetime.now()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    cache_filename = md5_str(args.input_dir, args.multi_rewrite_key, args.final_key)
    cache_pkl_path = os.path.join(args.cache_dir, cache_filename + ".pkl")
    logger.debug(f"[{BASENAME}] the cache path is {cache_pkl_path}")
    success_map = pickle_load(cache_pkl_path) if Path(cache_pkl_path).exists() else dict()

    conversation_data = list()
    conversation_dirs = join_and_filter_dirs(args.input_dir, sorted(os.listdir(args.input_dir)))
    for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] loading all conversations"):
        turn_file_lst = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
        for turn_file in turn_file_lst:
            data = cast(dict, load_json(turn_file))
            if args.multi_rewrite_key not in data or not isinstance(data[args.multi_rewrite_key], list):
                raise ValueError(f"[{BASENAME}] the key {args.multi_rewrite_key} are not in turn_file {turn_file}")
            conversation_data.append(data)

    for data in tqdm(conversation_data, desc=f"[{BASENAME}] filtering all conversations"):
        sample_id = cast(str, data["sample_id"])
        if sample_id in success_map:
            continue
        if len(data["positive_pids"]) == 0:
            success_map[sample_id] = data.get("oracle_utt", data["cur_utt"])
        else:
            max_mrr_score = 0.0
            best_rewrite_utt = None
            multi_rewrite_lst = cast(list, copy(data[args.multi_rewrite_key]))
            multi_rewrite_lst.append(data.get("oracle_utt", data["cur_utt"]))
            for rewrite_utt in set(multi_rewrite_lst):
                run_data = {sample_id: dict()}
                response = requests.post(retrieve_url, json={"query": rewrite_utt, "query_id": sample_id})
                response.raise_for_status()
                node_lst = response.json()[sample_id]
                for node in node_lst:
                    passage_id, score = node["id"], node["score"]
                    run_data[sample_id][passage_id] = score
                mrr_score = mrr(run_data, {sample_id: deepcopy(qrels[sample_id])})
                if mrr_score > max_mrr_score:
                    max_mrr_score = mrr_score
                    best_rewrite_utt = rewrite_utt
            if not best_rewrite_utt:
                logger.warning(f"[{BASENAME}] {sample_id} no suitable query rewrite found.")
                alternative = Counter(multi_rewrite_lst).most_common(1)[0][0]
                success_map[sample_id] = data.get("oracle_utt", alternative)
            else:
                success_map[sample_id] = best_rewrite_utt
        pickle_store(success_map, cache_pkl_path)

    elapsed_time = datetime.now() - start_time
    logger.success(f"[{BASENAME}] best of n finished, total elapsed time is {elapsed_time}")


if __name__ == "__main__":
    # python src/rewriter/best_of_n.py
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="cache/best_of_n")
    parser.add_argument("--multi_rewrite_key", type=str, required=True)
    parser.add_argument("--final_key", type=str, required=True)
    parser.add_argument("--qrel_trec_file", type=str, required=True)
    parser.add_argument("--port", type=int, default=8006)

    main(parser.parse_args())
