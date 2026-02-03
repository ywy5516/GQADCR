#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path
from typing import List, Union, cast

from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import (
    join_and_filter_dirs,
    join_and_filter_files,
    load_json,
    stream_load_jsonl,
)

BASENAME = Path(__file__).stem


class CustomCorpusDataset(Dataset):
    def __init__(self, corpus_dir: str, start: int, end: int):
        """Load preprocessed corpus files"""
        super().__init__()
        self.jsonl_filepaths = self._init_jsonl_filepaths(corpus_dir, start, end)
        self.data = self._read_jsonl_data()

    @staticmethod
    def _init_jsonl_filepaths(corpus_dir: str, start: int, end: int) -> List[str]:
        jsonl_filepaths = join_and_filter_files(corpus_dir, sorted(os.listdir(corpus_dir)), ".jsonl")
        n = len(jsonl_filepaths)
        if start < 0 or start >= n:
            raise IndexError(f"[{BASENAME}] start is not in range [0, {n})")
        if start < end:
            jsonl_filepaths = jsonl_filepaths[start:end]
        else:
            logger.warning(f"[{BASENAME}] start >= end ({start} >= {end}), skip end")
            jsonl_filepaths = jsonl_filepaths[start:]
        logger.debug(jsonl_filepaths)
        return jsonl_filepaths

    def _read_jsonl_data(self) -> List[dict]:
        data = list()
        for jsonl_filepath in tqdm(self.jsonl_filepaths, desc=f"[{BASENAME}] loading jsonl to build dense index"):
            for line in stream_load_jsonl(jsonl_filepath):
                data.append(line)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class DataBase:
    @staticmethod
    def _convert_list_to_set(input_lst: Union[List[str], List[List[str]]]) -> set:
        result = set()
        for item in input_lst:
            if isinstance(item, list):
                result.update(item)
            else:
                result.add(item)
        return result

    @staticmethod
    def _combine_conv_q(cur_utt: str, ctx_utt_lst: List[str]):
        utt_lst = ctx_utt_lst + [cur_utt]
        return " [SEP] ".join(utt_lst[::-1])

    @staticmethod
    def _combine_conv_ctx(cur_utt: str, ctx_utt_lst: List[str], ctx_response_lst: List[str]):
        conv_ctx = list()
        for utterance, response in zip(ctx_utt_lst, ctx_response_lst):
            conv_ctx.append(utterance.rstrip("?").rstrip(".").rstrip())
            conv_ctx.append(response)
        conv_ctx.append(cur_utt)  # q1, a1, q2, a2, ..., q_k
        return " [SEP] ".join(conv_ctx[::-1])  # q_k [SEP] a_{k-1} [SEP] q_{k-1} ... [SEP] a1 [SEP] q1

    @staticmethod
    def custom_collate_fn(batch):
        raise NotImplementedError


class RetrievalDataset(Dataset, DataBase):
    def __init__(self, input_dir: str, input_key: str, **kwargs):
        super().__init__()
        self.data = self._load_conversation_data(input_dir, input_key, **kwargs)

    def _load_conversation_data(self, input_dir: str, input_key: str, **kwargs) -> List[dict]:
        examples = list()
        conversation_dirs = join_and_filter_dirs(input_dir, sorted(os.listdir(input_dir)))
        for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] loading all conversations"):
            turn_file_list = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
            if kwargs.get("only_last_turn", False):
                turn_file_list = [turn_file_list[-1]]
            for turn_file in turn_file_list:
                data = cast(dict, load_json(turn_file))
                if len(data["positive_pids"]) == 0:
                    continue
                assert len(data["ctx_utt_list"]) == len(data["ctx_response_list"])

                if input_key == "convq":
                    query_str = self._combine_conv_q(data["cur_utt"], data["ctx_utt_list"])
                elif input_key == "convqa":
                    query_str = self._combine_conv_ctx(data["cur_utt"], data["ctx_utt_list"], data["ctx_response_list"])
                elif input_key == "convqp":
                    raise ValueError(f"[{BASENAME}] does not support convqp")
                elif input_key not in data:
                    raise KeyError(f"[{BASENAME}] the key {input_key} does not exist in {turn_file}")
                else:
                    query_str = data[input_key]
                examples.append({"sample_id": data["sample_id"], "query": query_str})
        return examples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class DevDataset(RetrievalDataset):
    def __init__(self, input_dir: str, input_key: str, **kwargs):
        super().__init__(input_dir, input_key, **kwargs)


class TrainDataset(Dataset, DataBase):
    def __init__(self, input_dir: str, input_key: str, rewrite_key: str, **kwargs):
        super().__init__()
        self.required_keys = (
            "cur_utt",
            "cur_response",
            "ctx_utt_list",
            "ctx_response_list",
            "positive_passages",
        )
        self.data = self._load_conversation_data(input_dir, input_key, rewrite_key, **kwargs)

    def _check_keys_exist(self, sample_id: str, conversation_data: dict):
        for key in self.required_keys:
            if key not in conversation_data:
                raise KeyError(f"[{BASENAME}] the key {key} does not exist in the data of {sample_id}")

    def _load_conversation_data(self, input_dir: str, input_key: str, rewrite_key: str, **kwargs):
        examples = list()
        conversation_dirs = join_and_filter_dirs(input_dir, sorted(os.listdir(input_dir)))
        for conversation_dir in tqdm(conversation_dirs, desc=f"[{BASENAME}] loading train conversations"):
            turn_file_list = join_and_filter_files(conversation_dir, sorted(os.listdir(conversation_dir)), ".json")
            if len(turn_file_list) == 1:
                continue
            if kwargs.get("only_last_turn", False):
                turn_file_list = [turn_file_list[-1]]

            for turn_file in turn_file_list:
                data = cast(dict, load_json(turn_file))
                sample_id = data["sample_id"]
                if len(data["positive_passages"]) == 0:
                    continue
                self._check_keys_exist(sample_id, data)
                if rewrite_key not in data:
                    raise KeyError(f"[{BASENAME}] the key {rewrite_key} does not exist in {turn_file}")
                if not isinstance(data[rewrite_key], str):
                    raise ValueError(f"[{BASENAME}] the key {rewrite_key} is not a string in {turn_file}")

                if input_key == "convq":
                    query_str = self._combine_conv_q(data["cur_utt"], data["ctx_utt_list"])
                    hard_negative_passages = set(data["hard_convq_passages"][:5])
                elif input_key == "convqa":
                    query_str = self._combine_conv_ctx(data["cur_utt"], data["ctx_utt_list"], data["ctx_response_list"])
                    hard_negative_passages = set(data["hard_convqa_passages"][:5])
                else:
                    raise KeyError(f"[{BASENAME}] invalid input_key {input_key} in {sample_id}")
                rewrite_query = data[rewrite_key]
                positive_passages = set(data["positive_passages"])

                positives = list(positive_passages)
                examples.append(
                    {
                        "sample_id": sample_id,
                        "query": query_str,
                        "rewrite_query": rewrite_query,
                        "positives": positives,  # List[str]
                        "negatives": list(hard_negative_passages),  # List[str]
                    }
                )
        return examples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    @staticmethod
    def custom_collate_fn(batch: List[dict]):
        """fix RuntimeError: each element in list of batch should be of equal size"""
        result = dict()
        min_neg_len = min([len(item["negatives"]) for item in batch])
        min_neg_len = min(min_neg_len, 10)
        for k in batch[0].keys():
            if k == "positives":
                result[k] = [random.choice(item[k]) for item in batch]
            elif k == "negatives":
                result[k] = [
                    item[k] if len(item[k]) == min_neg_len else random.sample(item[k], min_neg_len) for item in batch
                ]
            else:
                result[k] = [item[k] for item in batch]
        return result
