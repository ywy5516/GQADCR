#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import shutil
import sys
from argparse import ArgumentParser
from copy import copy
from pathlib import Path
from typing import List, cast

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
    write_json,
)

MAX_PASSAGES_PER_FILE = 500000
BASENAME = Path(__file__).stem


def parse_collection(input_dir: str, output_dir: str):
    """Parse the original corpus file"""
    passage_dirs = [
        os.path.join(input_dir, "commoncrawl"),
        os.path.join(input_dir, "wayback"),
        os.path.join(input_dir, "wayback-backfill"),
    ]
    check_if_dir_file_exist(*passage_dirs)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    idx = file_idx = 1
    pid_to_raw_pid = dict()
    output_file_path = os.path.join(output_dir, "{:03d}.jsonl".format(file_idx))
    output_obj = open(output_file_path, "w", encoding="utf-8", newline="\n")
    for dirpath in passage_dirs:
        filenames = sorted(os.listdir(dirpath))
        for filename in tqdm(filenames, desc=f"[{BASENAME}] processing {dirpath}"):
            with open(os.path.join(dirpath, filename), "r", encoding="utf-8") as fin:
                for line in fin.readlines():
                    line = cast(dict, json.loads(line))
                    pid_to_raw_pid[idx] = line["id"]
                    new_line = {"id": idx, "contents": line["contents"]}
                    output_obj.write(json.dumps(new_line, ensure_ascii=False) + "\n")

                    if idx % MAX_PASSAGES_PER_FILE == 0:
                        output_obj.close()
                        file_idx += 1
                        output_file_path = os.path.join(output_dir, "{:03d}.jsonl".format(file_idx))
                        output_obj = open(output_file_path, "w", encoding="utf-8", newline="\n")
                    idx += 1
    output_obj.close()
    pkl_path = os.path.join(output_dir, "pid_to_raw_pid.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(pid_to_raw_pid, f)
    logger.success(f"[{BASENAME}] collection file has been parsed")
    return pkl_path


def parse_qrecc_data(pkl_path: str, input_file: str, output_dir: str):
    check_if_dir_file_exist(pkl_path, input_file)

    with open(pkl_path, "rb") as f:
        pid_to_raw_pid = cast(dict, pickle.load(f))
    raw_pid_to_pid = dict()
    for pid, raw_pid in pid_to_raw_pid.items():
        raw_pid_to_pid[raw_pid] = pid
    del pid_to_raw_pid

    raw_data = cast(List[dict], load_json(input_file))
    for line in tqdm(raw_data, desc=f"[{BASENAME}] parsing file {Path(input_file).name}"):
        conversation_id = line["Conversation_no"]
        turn_id = line["Turn_no"]
        tmp_dir = os.path.join(output_dir, str(conversation_id).zfill(5))
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        tmp_json_file = os.path.join(tmp_dir, "{:02d}.json".format(turn_id))

        sample_id = "{}_{}".format(conversation_id, turn_id)
        cur_utt = line["Question"].strip() if int(turn_id) > 1 else line["Truth_rewrite"].strip()
        positive_pids = list()
        for raw_pid in line["Truth_passages"]:
            new_pid = raw_pid_to_pid[raw_pid]
            positive_pids.append(new_pid)

        parsed_record = {
            "sample_id": sample_id,
            "conversation_source": line["Conversation_source"],
            "cur_utt": cur_utt,
            "oracle_utt": line["Truth_rewrite"].strip(),
            "cur_response": line["Truth_answer"].strip(),
            "positive_pids": positive_pids,
        }
        write_json(parsed_record, tmp_json_file)
    logger.success(f"[{BASENAME}] QReCC file {input_file} has been parsed")


def parse_qrecc_data_to_qrels(pkl_path: str, input_file: str, qrels_file: str):
    """Extract positive samples from original dataset file to compose `qrels.tsv`"""
    check_if_dir_file_exist(pkl_path, input_file)

    with open(pkl_path, "rb") as f:
        pid_to_raw_pid = cast(dict, pickle.load(f))
    raw_pid_to_pid = dict()
    for pid, raw_pid in pid_to_raw_pid.items():
        raw_pid_to_pid[raw_pid] = pid
    del pid_to_raw_pid

    data = cast(List[dict], load_json(input_file))
    with open(qrels_file, "w", encoding="utf-8") as f:
        for line in tqdm(data, desc=f"[{BASENAME}] Parsing qrecc dev file {Path(input_file).name}"):
            sample_id = f"{line['Conversation_no']}_{line['Turn_no']}"
            for raw_pid in line["Truth_passages"]:
                f.write("{}\t{}\t{}\t{}".format(sample_id, 0, raw_pid_to_pid[raw_pid], 1))
                f.write("\n")
    logger.success(f"[{BASENAME}] Convert qrecc file {input_file} to qrels file")


def clean_invalid_conversations(input_dir: str):
    """Remove sessions that have no positive samples in any round or only one round"""
    check_if_dir_file_exist(input_dir)
    invalid_conversation_dirs = set()
    conversation_dirs = join_and_filter_dirs(input_dir, sorted(os.listdir(input_dir)))
    for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] checking all conversations"):
        turn_file_list = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
        if len(turn_file_list) <= 1:
            invalid_conversation_dirs.add(conversation_dir)
            continue
        flag = False
        for turn_file in turn_file_list:
            data = cast(dict, load_json(turn_file))
            if data["positive_pids"]:
                flag = True
                break
        if flag is False:
            invalid_conversation_dirs.add(conversation_dir)
    for invalid_dir in invalid_conversation_dirs:
        shutil.rmtree(invalid_dir)
        logger.debug(f"[{BASENAME}] deleted invalid conversation {invalid_dir}")
    logger.success(f"[{BASENAME}] cleaned {len(invalid_conversation_dirs)} invalid conversations in {input_dir}")


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
                data["positive_passages"] = list()
            else:
                positive_passages = list()
                for pid in data["positive_pids"]:
                    text = pids_to_texts.get(pid, None)
                    if text is None:
                        logger.error(f"[{BASENAME}] sample_id {sample_id}, pid {pid} not found in collection")
                    positive_passages.append(text)
                data["positive_passages"] = positive_passages
            write_json(data, turn_file)
    logger.success(f"[{BASENAME}] convert all pids to texts in {input_dir}")


def combine_conversation_history(input_dir: str):
    """Concatenate conversation history in a single session"""
    check_if_dir_file_exist(input_dir)
    conversation_dirs = join_and_filter_dirs(input_dir, sorted(os.listdir(input_dir)))
    for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] combining history from {input_dir}"):
        turn_file_list = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
        ctx_utt_list = list()
        ctx_oracle_list = list()
        ctx_response_list = list()

        for turn_file in turn_file_list:
            data = cast(dict, load_json(turn_file))
            if not data["cur_response"]:
                data["cur_response"] = "UNANSWERABLE"
            data["ctx_utt_list"] = copy(ctx_utt_list)
            data["ctx_oracle_list"] = copy(ctx_oracle_list)
            data["ctx_response_list"] = copy(ctx_response_list)
            assert len(data["ctx_utt_list"]) == len(data["ctx_response_list"])

            ctx_utt_list.append(data["cur_utt"])
            ctx_oracle_list.append(data["oracle_utt"])
            ctx_response_list.append(data["cur_response"])
            write_json(data, turn_file)
    logger.success(f"[{BASENAME}] Finished combining the history of {input_dir}")


def main(args):
    collection_pkl_path = parse_collection(args.collection_input_dir, args.collection_output_dir)

    parse_qrecc_data(collection_pkl_path, args.train_input_file, args.train_output_dir)
    parse_qrecc_data(collection_pkl_path, args.dev_input_file, args.dev_output_dir)

    parse_qrecc_data_to_qrels(collection_pkl_path, args.train_input_file, args.train_qrels_file)
    parse_qrecc_data_to_qrels(collection_pkl_path, args.dev_input_file, args.dev_qrels_file)

    clean_invalid_conversations(args.train_output_dir)
    clean_invalid_conversations(args.dev_output_dir)

    pids_to_texts = load_all_corpus_map(args.collection_output_dir)
    conversation_pids2texts(args.train_output_dir, pids_to_texts)
    conversation_pids2texts(args.dev_output_dir, pids_to_texts)
    del pids_to_texts

    combine_conversation_history(args.train_output_dir)
    combine_conversation_history(args.dev_output_dir)


if __name__ == "__main__":
    # python src/preprocess/qrecc.py
    # https://github.com/scai-conf/SCAI-QReCC-21
    parser = ArgumentParser()
    parser.add_argument("--collection_input_dir", type=str, default="corpus/qrecc/")
    parser.add_argument("--collection_output_dir", type=str, default="corpus/parsed_qrecc")

    parser.add_argument("--train_input_file", type=str, default="corpus/qrecc/scai-qrecc21-training-turns.json")
    parser.add_argument("--train_output_dir", type=str, default="data/qrecc/parsed_train")
    parser.add_argument("--train_qrels_file", type=str, default="data/qrecc/train_qrels.tsv")

    parser.add_argument("--dev_input_file", type=str, default="corpus/qrecc/scai-qrecc21-test-turns.json")
    parser.add_argument("--dev_output_dir", type=str, default="data/qrecc/parsed_dev")
    parser.add_argument("--dev_qrels_file", type=str, default="data/qrecc/dev_qrels.tsv")

    main(parser.parse_args())
