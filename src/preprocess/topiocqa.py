#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import sys
from argparse import ArgumentParser
from copy import copy
from pathlib import Path
from typing import cast

import ujson as json
from loguru import logger
from tqdm import tqdm

sys.path.append(".")

from src.utils import (
    check_if_dir_file_exist,
    join_and_filter_dirs,
    join_and_filter_files,
    load_all_corpus_map,
    load_json,
    stream_load_jsonl,
    write_json,
)

MAX_PASSAGES_PER_FILE = 500000
BASENAME = Path(__file__).stem


def parse_collection(tsv_file: str, output_dir: str):
    """Parse the original corpus file"""
    check_if_dir_file_exist(tsv_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    idx, file_idx = 0, 1
    output_file_path = os.path.join(output_dir, "{:03d}.jsonl".format(file_idx))
    output_obj = open(output_file_path, "w", encoding="utf-8", newline="\n")
    with open(tsv_file, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin, delimiter="\t")
        for row in tqdm(reader, desc=f"[{BASENAME}] reading from tsv file"):
            if row[0] == "id":  # ['id', 'text', 'title'] id begin from 1
                continue

            text, title = row[1], " ".join(row[2].split(" [SEP] ")).strip()
            # "contents": " ".join([title, text])
            output_obj.write(json.dumps({"id": int(row[0]), "contents": text}, ensure_ascii=False) + "\n")
            idx += 1

            if idx % MAX_PASSAGES_PER_FILE == 0:
                output_obj.close()
                file_idx += 1
                output_file_path = os.path.join(output_dir, "{:03d}.jsonl".format(file_idx))
                output_obj = open(output_file_path, "w", encoding="utf-8", newline="\n")
    output_obj.close()
    logger.success(f"[{BASENAME}] convert topiocqa collection to jsonl files")


def parse_topiocqa_data(input_file: str, output_dir: str):
    check_if_dir_file_exist(input_file)
    for line in tqdm(stream_load_jsonl(input_file), desc=f"[{BASENAME}] parsing input file {Path(input_file).name}"):
        conversation_id = line["Conversation_no"]
        turn_id = line["Turn_no"]
        tmp_dir = os.path.join(output_dir, str(conversation_id).zfill(4))
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        tmp_json_file = os.path.join(tmp_dir, "{:02d}.json".format(turn_id))

        sample_id = "{}_{}".format(conversation_id, turn_id)
        cur_utt = line["Question"].strip()
        positive_pids = list()
        gold_passage = cast(dict, line["Gold_passage"])
        if gold_passage:
            pid = int(gold_passage["id"].lstrip("wiki:"))
            positive_pids.append(pid)
        parsed_record = {
            "sample_id": sample_id,
            "cur_utt": cur_utt,
            "cur_response": line["Answer"].strip(),
            "topic": line["Topic"],
            # "sub_topic": line["Topic_section"],
            "rationale": line["Rationale"].strip(),  # str
            # "additional_answers": line["Additional_answers"],  # list[dict]
            "positive_pids": positive_pids,
        }
        write_json(parsed_record, tmp_json_file)
    logger.success(f"[{BASENAME}] TopiOCQA file {input_file} has been parsed")


def parse_topiocqa_data_to_qrels(input_file: str, qrels_file: str):
    """Extract positive samples from original dataset file to compose `qrels.tsv`"""
    check_if_dir_file_exist(input_file)
    with open(qrels_file, "w", encoding="utf-8") as f:
        for line in tqdm(stream_load_jsonl(input_file)):
            sample_id = f"{line['Conversation_no']}_{line['Turn_no']}"
            gold_passage = cast(dict, line["Gold_passage"])
            if gold_passage:
                pid = int(gold_passage["id"].lstrip("wiki:"))
                f.write("{}\t{}\t{}\t{}".format(sample_id, 0, pid, 1))
                f.write("\n")
    logger.success(f"[{BASENAME}] Convert topiocqa file {input_file} to qrels file")


def conversation_pids2texts(input_dir: str, pids_to_texts: dict):
    """Convert `positive_pids` in session file to text"""
    check_if_dir_file_exist(input_dir)
    conversation_dirs = join_and_filter_dirs(input_dir, sorted(os.listdir(input_dir)))
    for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] converting id to text in {input_dir}"):
        turn_file_list = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
        for turn_file in turn_file_list:
            data = cast(dict, load_json(turn_file))
            sample_id = data["sample_id"]
            if len(data["positive_pids"]) == 0:
                logger.warning(f"[{BASENAME}] topiocqa sample_id {sample_id} has no positive passages")
                data["positive_passages"] = list()
            else:
                positive_passages = list()
                for pid in data["positive_pids"]:
                    text = pids_to_texts.get(pid, None)
                    if text is None:
                        logger.error(f"[{BASENAME}] sample_id {sample_id}, pid {pid} not found in collection")
                    positive_passages.append(text)
                data["positive_passages"] = positive_passages

                if not data["rationale"]:
                    data["rationale"] = positive_passages[0]
            write_json(data, turn_file)
    logger.success(f"[{BASENAME}] convert all pids to texts in {input_dir}")


def combine_conversation_history(input_dir: str):
    """Concatenate conversation history in a single session"""
    check_if_dir_file_exist(input_dir)
    conversation_dirs = join_and_filter_dirs(input_dir, sorted(os.listdir(input_dir)))
    for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] combining history from {input_dir}"):
        turn_file_list = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
        ctx_utt_list = list()
        ctx_response_list = list()
        ctx_rationale_list = list()
        for turn_file in turn_file_list:
            data = cast(dict, load_json(turn_file))
            if not data["cur_response"]:
                data["cur_response"] = "UNANSWERABLE"
            data["ctx_utt_list"] = copy(ctx_utt_list)
            data["ctx_response_list"] = copy(ctx_response_list)
            data["ctx_rationale_list"] = copy(ctx_rationale_list)
            assert len(data["ctx_utt_list"]) == len(data["ctx_response_list"])

            ctx_utt_list.append(data["cur_utt"])
            ctx_response_list.append(data["cur_response"])
            ctx_rationale_list.append(data["rationale"])
            write_json(data, turn_file)
    logger.success(f"[{BASENAME}] Finished combining the history of {input_dir}")


def main(args):
    parse_collection(args.collection_input_file, args.collection_output_dir)

    parse_topiocqa_data(args.train_input_file, args.train_output_dir)
    parse_topiocqa_data(args.dev_input_file, args.dev_output_dir)

    parse_topiocqa_data_to_qrels(args.train_input_file, args.train_qrels_file)
    parse_topiocqa_data_to_qrels(args.dev_input_file, args.dev_qrels_file)

    pids_to_texts = load_all_corpus_map(args.collection_output_dir)
    conversation_pids2texts(args.train_output_dir, pids_to_texts)
    conversation_pids2texts(args.dev_output_dir, pids_to_texts)
    del pids_to_texts

    combine_conversation_history(args.train_output_dir)
    combine_conversation_history(args.dev_output_dir)


if __name__ == "__main__":
    # python src/preprocess/topiocqa.py
    # https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv
    # https://huggingface.co/datasets/McGill-NLP/TopiOCQA/tree/main/data
    parser = ArgumentParser()
    parser.add_argument("--collection_input_file", type=str, default="corpus/topiocqa/full_wiki_segments.tsv")
    parser.add_argument("--collection_output_dir", type=str, default="corpus/parsed_topiocqa")

    parser.add_argument("--train_input_file", type=str, default="corpus/topiocqa/topiocqa_train.jsonl")
    parser.add_argument("--train_output_dir", type=str, default="data/topiocqa/parsed_train")
    parser.add_argument("--train_qrels_file", type=str, default="data/topiocqa/train_qrels.tsv")

    parser.add_argument("--dev_input_file", type=str, default="corpus/topiocqa/topiocqa_valid.jsonl")
    parser.add_argument("--dev_output_dir", type=str, default="data/topiocqa/parsed_dev")
    parser.add_argument("--dev_qrels_file", type=str, default="data/topiocqa/dev_qrels.tsv")

    main(parser.parse_args())
