#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from argparse import ArgumentParser
from copy import copy
from pathlib import Path
from typing import cast

from loguru import logger
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from src.config import LLM_API_BASE, LLM_API_KEY
from src.rewriter.client import CustomOpenAI, build_zero_shot_prompt
from src.utils import (
    check_if_dir_file_exist,
    join_and_filter_dirs,
    join_and_filter_files,
    load_json,
    md5_str,
    pickle_load,
    pickle_store,
    write_json,
)

BASENAME = Path(__file__).stem


def main(args):
    check_if_dir_file_exist(args.input_dir)
    cache_dir = str(os.path.join(args.cache_dir, args.dataset_name + "/" + args.dataset_type))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_pkl_path = os.path.join(cache_dir, md5_str(args.input_dir, args.rewrite_key, args.ablation) + ".pkl")
    logger.debug(f"[{BASENAME}] the cache path is {cache_pkl_path}")
    success_set = pickle_load(cache_pkl_path) if Path(cache_pkl_path).exists() else set()

    agent = CustomOpenAI(args.api_key, args.api_base)
    conversation_dirs = join_and_filter_dirs(args.input_dir, sorted(os.listdir(args.input_dir)))
    for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] Loading conversations"):
        turn_file_lst = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
        for turn_file in turn_file_lst:
            if turn_file in success_set:
                continue
            data = cast(dict, load_json(turn_file))
            sample_id = cast(str, data["sample_id"])
            ctx_utt_lst = copy(data["ctx_utt_list"])
            ctx_response_lst = (
                copy(data["ctx_rationale_list"]) if args.dataset_name == "topiocqa" else copy(data["ctx_response_list"])
            )
            assert len(ctx_utt_lst) == len(ctx_response_lst), f"{turn_file} has length problem"
            new_response = data["rationale"] if data.get("rationale") else data["cur_response"]

            if sample_id.endswith("_1"):
                rewrite_utt_lst = [data.get("oracle_utt", data["cur_utt"])]
            elif len(data["positive_pids"]) < 1 and data.get("oracle_utt"):
                # Handle cases in the QReCC dataset where there are no target text passages and human rewrites exist
                rewrite_utt_lst = [data["oracle_utt"]]
            else:
                if args.ablation == "pr":
                    prompt = build_zero_shot_prompt(ctx_utt_lst, ctx_response_lst, data["cur_utt"])
                else:
                    prompt = build_zero_shot_prompt(ctx_utt_lst, ctx_response_lst, data["cur_utt"], new_response)

                try:
                    rewrite_utt_lst = agent.rewrite_utterance_multi(args.model_name, prompt, args.multi_rewrite_times)
                except Exception as e:
                    logger.error(f"[{BASENAME}] {sample_id} rewrite failed, use previous one")
                    raise e

            data[args.rewrite_key] = rewrite_utt_lst
            write_json(data, turn_file)
            success_set.add(turn_file)

        pickle_store(success_set, cache_pkl_path)

    logger.success(f"[{BASENAME}] rewrite finished successfully")


if __name__ == "__main__":
    # python src/rewriter/multi_rewrite.py
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, choices=["topiocqa", "qrecc"], required=True)
    parser.add_argument("--dataset_type", type=str, choices=["train", "dev"], required=True)
    parser.add_argument("--cache_dir", type=str, default="cache/multi_rewrite")
    parser.add_argument("--rewrite_key", type=str, required=True)
    parser.add_argument("--multi_rewrite_times", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=LLM_API_KEY)
    parser.add_argument("--api_base", type=str, default=LLM_API_BASE)
    parser.add_argument("--ablation", type=str, choices=["none", "pr"], default="none")

    main(parser.parse_args())
