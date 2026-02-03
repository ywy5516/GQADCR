#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Tuple, cast

import faiss
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel

from src.model import BaseEncoder
from src.utils import join_and_filter_files

BASENAME = Path(__file__).stem
NUM_PER_BLOCK_DOCS = 5000000


class CustomBaseNode(BaseModel):
    node_id: str  # 与 sparse 保持一致


class CustomNodeWithScore(BaseModel):
    node: CustomBaseNode
    score: float


class CustomFaissRetriever:
    def __init__(
        self,
        pretrained_model_path: str,
        model_cls: Literal["dpr", "roberta", "bert"],
        model_type: Literal["query", "passage"],
        device: str,
        pooling_mode: Literal["cls", "mean"],
    ):
        self.encoder = BaseEncoder(model_cls, model_type, pretrained_model_path, device, pooling_mode)
        self.encoder.model.requires_grad_(False)
        self.encoder.model.eval()

    @staticmethod
    def _create_faiss_gpu_index(device: str, d: int):
        """create faiss-gpu index"""
        index = faiss.IndexFlatIP(d)  # 内积
        ids_part = device.split(":")[-1]
        gpu_device_list = [int(x.strip()) for x in ids_part.split(",") if x.strip().isdigit()]
        n_gpu_for_faiss = len(gpu_device_list)
        if n_gpu_for_faiss == 1:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, gpu_device_list[0], index)
        else:
            gpu_resources = list()
            temp_mem = -1
            for _ in range(n_gpu_for_faiss):
                res = faiss.StandardGpuResources()
                if temp_mem >= 0:
                    res.setTempMemory(temp_mem)
                gpu_resources.append(res)

            vdev = faiss.Int32Vector()  # gpu_devices_vector
            vres = faiss.GpuResourcesVector()  # gpu_vector_resources
            for i in range(n_gpu_for_faiss):
                vdev.push_back(gpu_device_list[i])
                vres.push_back(gpu_resources[i])
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.usePrecomputed = False
            return faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    def from_persist_dir(self, index_dir: str, device: str, embedding_size: int = 768):
        self.faiss = self._create_faiss_gpu_index(device, embedding_size)
        logger.debug(f"[{BASENAME}] the type of faiss index is {type(self.faiss)}")

        logger.debug(f"[{BASENAME}] load faiss index from {index_dir}")
        npy_filepaths = join_and_filter_files(index_dir, sorted(os.listdir(index_dir)), ".npy")
        emb_files, id_files = dict(), dict()
        for filepath in npy_filepaths:
            filename = Path(filepath).stem
            if filename.startswith("emb_block"):
                emb_files[int(filename.split("_")[2])] = filepath
            elif filename.startswith("id_block"):
                id_files[int(filename.split("_")[2])] = filepath
        sorted_keys = sorted(set(emb_files.keys()) & set(id_files.keys()))
        self.paired_list = [(emb_files[idx], id_files[idx]) for idx in sorted_keys]

    @staticmethod
    def _merge_candidates(
        merged_matrix: List[List[tuple]], candidate_matrix: List[List[tuple]], top_k: int
    ) -> List[List[tuple]]:
        results = list()
        for merged_lst, candidate_lst in zip(merged_matrix, candidate_matrix):
            # (passage_id, score)
            p1 = p2 = cnt = 0
            tmp_lst = list()
            while p1 < len(merged_lst) and p2 < len(candidate_lst) and cnt < top_k:
                if merged_lst[p1][-1] > candidate_lst[p2][-1]:
                    tmp_lst.append(merged_lst[p1])
                    p1 += 1
                else:
                    tmp_lst.append(candidate_lst[p2])
                    p2 += 1
                cnt += 1

            while p1 < len(merged_lst) and cnt < top_k:
                tmp_lst.append(merged_lst[p1])
                p1 += 1
                cnt += 1
            while p2 < len(candidate_lst) and cnt < top_k:
                tmp_lst.append(candidate_lst[p2])
                p2 += 1
                cnt += 1
            results.append(tmp_lst)
        return results

    def batch_retrieve(
        self, queries: List[str], max_length: int, is_normalize: bool = False, top_k: int = 100
    ) -> List[List[CustomNodeWithScore]]:
        with torch.no_grad():
            outputs = self.encoder.encode(queries, max_length=max_length)
        query_embs = outputs.pooler_output.cpu()
        if is_normalize:
            query_embs = torch.nn.functional.normalize(query_embs, p=2, dim=-1)
        query_embs = query_embs.numpy()

        torch.cuda.empty_cache()
        start_time = datetime.now()
        merged_candidate_matrix = None
        for emb_file, id_file in self.paired_list:
            try:
                self.faiss.reset()
            except Exception as e:
                logger.error(f"[{BASENAME}] reset faiss index error: {e}")
            gc.collect()
            torch.cuda.empty_cache()

            logger.debug(f"[{BASENAME}] loading npy files {id_file} and {emb_file}")
            passage_ids = cast(np.ndarray, np.load(id_file, mmap_mode="r"))
            self.faiss.add(np.load(emb_file, mmap_mode="r"))

            distances, indices = cast(Tuple[np.ndarray, np.ndarray], self.faiss.search(query_embs, top_k))
            candidate_matrix = [
                sorted(
                    [(pid, score) for pid, score in zip(passage_ids[indices[i]], distances[i])],
                    key=lambda x: x[1],
                    reverse=True,
                )
                for i in range(distances.shape[0])
            ]
            if merged_candidate_matrix is None:
                merged_candidate_matrix = candidate_matrix
            else:
                merged_candidate_matrix = self._merge_candidates(merged_candidate_matrix, candidate_matrix, top_k)
            self.faiss.reset()

        elapse_time = datetime.now() - start_time
        logger.info(f"[{BASENAME}] search time cost: {str(elapse_time)}")

        nodes = list()
        for merged_candidate_lst in merged_candidate_matrix:
            nodes.append(
                [
                    CustomNodeWithScore(node=CustomBaseNode(node_id=str(pid)), score=score)
                    for pid, score in merged_candidate_lst
                ]
            )
        return nodes

    def batch_retrieve_from_embs(self, query_embs: List[List[float]], top_k: int = 100):
        query_embs = np.array(query_embs)
        start_time = datetime.now()
        merged_candidate_matrix = None
        for emb_file, id_file in self.paired_list:
            try:
                self.faiss.reset()
            except Exception as e:
                logger.error(f"[{BASENAME}] reset faiss index error: {e}")
            gc.collect()
            torch.cuda.empty_cache()

            logger.debug(f"[{BASENAME}] loading npy files {id_file} and {emb_file}")
            passage_ids = cast(np.ndarray, np.load(id_file, mmap_mode="r"))
            self.faiss.add(np.load(emb_file, mmap_mode="r"))

            distances, indices = cast(Tuple[np.ndarray, np.ndarray], self.faiss.search(query_embs, top_k))
            candidate_matrix = [
                sorted(
                    [(pid, score) for pid, score in zip(passage_ids[indices[i]], distances[i])],
                    key=lambda x: x[1],
                    reverse=True,
                )
                for i in range(distances.shape[0])
            ]
            if merged_candidate_matrix is None:
                merged_candidate_matrix = candidate_matrix
            else:
                merged_candidate_matrix = self._merge_candidates(merged_candidate_matrix, candidate_matrix, top_k)
            self.faiss.reset()

        elapse_time = datetime.now() - start_time
        logger.info(f"[{BASENAME}] search time cost: {str(elapse_time)}")

        nodes = list()
        for merged_candidate_lst in merged_candidate_matrix:
            nodes.append(
                [
                    CustomNodeWithScore(node=CustomBaseNode(node_id=str(pid)), score=score)
                    for pid, score in merged_candidate_lst
                ]
            )
        return nodes
