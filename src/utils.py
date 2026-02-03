#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import ujson as json
from loguru import logger
from torch import Tensor
from tqdm import tqdm

BASENAME = Path(__file__).stem


def sentence_embedding(
    last_hidden_state: Tensor, attention_mask: Optional[Tensor] = None, pooling_mode: str = "cls"
) -> Tensor:
    # last_hidden_state (batch_size, sequence_length, hidden_size)
    # attention_mask[..., None] (batch_size, sequence_length, 1)
    if pooling_mode == "cls":
        pooled_output = last_hidden_state[:, 0, :]
    else:
        if attention_mask is None:
            raise ValueError("When `pooling_mode` is `mean`, attention mask must be provided")
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        pooled_output = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return pooled_output


def md5_str(*args) -> str:
    input_str = "_".join(str(arg) for arg in args)
    md5_obj = md5()
    md5_obj.update(input_str.encode("utf-8"))
    return md5_obj.hexdigest()


def write_json(json_obj: Any, filepath: str) -> None:
    """Save object to json file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load object from json file"""
    if not os.path.exists(filepath):
        logger.error(f"[{BASENAME}] {filepath} does not exist.")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[{BASENAME}] {filepath} load json error: {e}")
        raise e


def stream_load_jsonl(file_path: str) -> Generator:
    """Read large files using generator"""
    with open(file_path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def pickle_load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_store(data, path: str, high_protocol: bool = False):
    with open(path, "wb") as f:
        if high_protocol:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(data, f)


def check_if_dir_file_exist(*args):
    for path in args:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not exists")


def join_and_filter_dirs(base_path: str, dirnames: List[str]) -> List[str]:
    dirpaths = [os.path.join(base_path, dirname) for dirname in dirnames]
    return [dirpath for dirpath in dirpaths if os.path.isdir(dirpath)]


def join_and_filter_files(base_path: str, filenames: List[str], suffix: Optional[str]) -> List[str]:
    filepaths = [os.path.join(base_path, filename) for filename in filenames]
    filepaths = [filepath for filepath in filepaths if os.path.isfile(filepath)]
    if suffix:
        filepaths = [filepath for filepath in filepaths if filepath.endswith(suffix)]
    return filepaths


def load_all_corpus_map(corpus_dir: str) -> Dict[int, str]:
    """Load all preprocessed corpus files into dictionary mapping"""
    check_if_dir_file_exist(corpus_dir)
    pid2text = dict()
    jsonl_filepaths = join_and_filter_files(corpus_dir, sorted(os.listdir(corpus_dir)), ".jsonl")
    for jsonl_filepath in tqdm(jsonl_filepaths, desc=f"[{BASENAME}] loading all corpus"):
        for line in stream_load_jsonl(jsonl_filepath):
            pid2text[int(line["id"])] = line["contents"]
    return pid2text
