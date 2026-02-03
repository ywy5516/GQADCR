#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers.trainer_utils import set_seed

sys.path.append(str(Path(".").absolute()))

from src.config import MAX_QUERY_LENGTH
from src.retriever.dense import CustomFaissRetriever
from src.retriever.sparse import CustomBM25Retriever
from src.utils import check_if_dir_file_exist

BASENAME = Path(__file__).stem

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    query_id: str

    # for dense
    max_length: int = MAX_QUERY_LENGTH
    is_normalize: bool = False
    top_k: int = 100


class BatchQueryRequest(BaseModel):
    queries: List[str]
    query_ids: List[str]

    # for dense
    max_length: int = MAX_QUERY_LENGTH
    is_normalize: bool = False
    top_k: int = 100


class BatchEmbRequest(BaseModel):
    # only for dense
    query_embs: List[List[float]]
    query_ids: List[str]
    top_k: int = 100


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest) -> JSONResponse:
    if args.retriever_type == "sparse":
        nodes = retriever.retrieve(request.query)
    else:
        nodes = retriever.batch_retrieve([request.query], request.max_length, request.is_normalize, request.top_k)
        nodes = nodes[0]
    # type(node.node.node_id) == str
    results = {request.query_id: [{"id": node.node.node_id, "score": node.score} for node in nodes]}
    return JSONResponse(results)  # json.loads(response.body)  response.json()


@app.post("/batch_retrieve")
def batch_retrieve_endpoint(request: BatchQueryRequest) -> JSONResponse:
    if args.retriever_type == "sparse":
        nodes = retriever.batch_retrieve(request.queries)
    else:
        nodes = retriever.batch_retrieve(request.queries, request.max_length, request.is_normalize, request.top_k)
    results = dict()
    for query_id, node_list in zip(request.query_ids, nodes):
        result = [{"id": node.node.node_id, "score": node.score} for node in node_list]
        results[query_id] = result
    return JSONResponse(results)


@app.post("/batch_retrieve_from_embs")
def batch_retrieve_from_embs_endpoint(request: BatchEmbRequest) -> JSONResponse:
    if args.retriever_type == "sparse":
        raise NotImplementedError
    nodes = retriever.batch_retrieve_from_embs(request.query_embs, request.top_k)
    results = dict()
    for query_id, node_list in zip(request.query_ids, nodes):
        result = [{"id": node.node.node_id, "score": node.score} for node in node_list]
        results[query_id] = result
    return JSONResponse(results)


@app.get("/get_retriever_type")
def get_retriever_type() -> JSONResponse:
    return JSONResponse({"type": args.retriever_type})


if __name__ == "__main__":
    # python src/retriever/server.py
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retriever_type", type=str, choices=["sparse", "dense"], required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--model_cls", type=str, choices=["dpr", "roberta", "bert"])
    parser.add_argument("--model_type", type=str, choices=["query", "passage"])
    parser.add_argument("--pooling_mode", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--encoder_device", type=str, default="cuda:0")
    parser.add_argument("--faiss_device", type=str, default="cuda:0")

    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--log_level", type=str, choices=["debug", "info"], default="info")

    args = parser.parse_args()
    set_seed(args.seed)
    check_if_dir_file_exist(args.index_dir)
    if args.retriever_type == "dense":
        if not (
            args.pretrained_model_path
            and args.model_cls
            and args.model_type
            and args.encoder_device
            and args.faiss_device
        ):
            raise ValueError(f"[{BASENAME}] some arguments are not defined")
        check_if_dir_file_exist(args.pretrained_model_path)

    if args.retriever_type == "sparse":
        retriever = CustomBM25Retriever.from_persist_dir(args.index_dir, True, True)
    else:
        retriever = CustomFaissRetriever(
            args.pretrained_model_path, args.model_cls, args.model_type, args.encoder_device, args.pooling_mode
        )
        retriever.from_persist_dir(args.index_dir, args.faiss_device, args.embedding_size)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=args.log_level)
