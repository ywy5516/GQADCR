#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from src.retriever.sparse import CustomBM25Retriever, CustomJSONReader
from src.utils import check_if_dir_file_exist

BASENAME = Path(__file__).stem


def build_bm25s_index(corpus_dir: str, save_dir: str, **kwargs):
    logger.debug(f"[{BASENAME}] build bm25s index from {corpus_dir}")
    check_if_dir_file_exist(corpus_dir)
    nodes = list()
    filename_list = sorted(os.listdir(corpus_dir))
    reader = CustomJSONReader(is_jsonl=True, clean_json=False)
    start_time = datetime.now()
    for filename in tqdm(filename_list, desc="loading corpus file"):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(corpus_dir, filename)
        nodes.extend(reader.load_jsonl_nodes(filepath, extra_info={"corpus": corpus_dir, "filepath": filepath}))
    retriever = CustomBM25Retriever(nodes=nodes, **kwargs)
    retriever.persist(save_dir, save_corpus=True)

    del nodes
    del retriever
    gc.collect()
    elapsed_time = datetime.now() - start_time
    logger.info(f"[{BASENAME}] build bm25s index caused {elapsed_time}")


if __name__ == "__main__":
    # python src/indexer/build_sparse_index.py
    parser = ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="corpus/parsed_qrecc")
    parser.add_argument("--save_dir", type=str, default="index/bm25s_qrecc/")
    parser.add_argument("--bm25_method", type=str, default="lucene")
    parser.add_argument("--bm25_k1", type=float, default=0.82, help="topiocqa k1=0.9; qrecc k1=0.82")
    parser.add_argument("--bm25_b", type=float, default=0.68, help="topiocqa b=0.4; qrecc b=0.68")
    parser.add_argument("--bm25_verbose", action="store_true")
    parser.add_argument("--bm25_skip_stemming", action="store_true")

    args = parser.parse_args()
    build_bm25s_index(
        args.corpus_dir,
        args.save_dir,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        similarity_top_k=100,
        verbose=args.bm25_verbose,
        skip_stemming=args.bm25_skip_stemming,
    )
