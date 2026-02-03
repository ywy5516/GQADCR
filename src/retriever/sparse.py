#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import bm25s
import Stemmer
import ujson as json
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from tqdm import tqdm

from src.utils import stream_load_jsonl

BASENAME = Path(__file__).stem

DEFAULT_PERSIST_ARGS = {
    "bm25_k1": "bm25_k1",
    "bm25_b": "bm25_b",
    "bm25_method": "bm25_method",
    "similarity_top_k": "similarity_top_k",
    "_verbose": "verbose",
    "skip_stemming": "skip_stemming",
}
DEFAULT_PERSIST_FILENAME = "bm25s.json"


class CustomBM25Retriever(BaseRetriever):
    def __init__(
        self,
        bm25_k1: float = 0.9,
        bm25_b: float = 0.4,
        bm25_method: str = "lucene",
        nodes: Optional[List[BaseNode]] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
        language: str = "en",
        existing_bm25: Optional[bm25s.BM25] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        skip_stemming: bool = False,
        token_pattern: str = r"(?u)\b\w\w+\b",
    ) -> None:
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.bm25_method = bm25_method
        self.stemmer = stemmer or Stemmer.Stemmer("english")
        self.similarity_top_k = similarity_top_k
        self.token_pattern = token_pattern
        self.skip_stemming = skip_stemming

        if existing_bm25 is not None:
            self.bm25 = existing_bm25
            self.corpus = existing_bm25.corpus
        else:
            if nodes is None:
                raise ValueError("Please pass nodes or an existing BM25 object.")

            corpus_tokens = bm25s.tokenize(
                [
                    node.get_content(metadata_mode=MetadataMode.EMBED)
                    for node in tqdm(nodes, total=len(nodes), desc="corpus tokenizing")
                ],
                token_pattern=token_pattern,
                stopwords=language,
                stemmer=None if skip_stemming else self.stemmer,
                show_progress=verbose,
            )
            self.bm25 = bm25s.BM25(k1=bm25_k1, b=bm25_b, method=bm25_method)
            self.bm25.index(corpus_tokens, show_progress=verbose)
            self.corpus = [
                node_to_metadata_dict(node) for node in tqdm(nodes, total=len(nodes), desc="node to metadata dict")
            ]
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    def _get_persist_args(self) -> Dict[str, Any]:
        """Get Persist Args Dict to Save."""
        return {DEFAULT_PERSIST_ARGS[key]: getattr(self, key) for key in DEFAULT_PERSIST_ARGS if hasattr(self, key)}

    def persist(self, save_dir: str, save_corpus: bool = True, **kwargs) -> None:
        """Persist the retriever to a directory."""
        self.bm25.save(save_dir=save_dir, corpus=self.corpus if save_corpus else None, **kwargs)
        with open(os.path.join(save_dir, DEFAULT_PERSIST_FILENAME), "w", encoding="utf-8") as f:
            json.dump(self._get_persist_args(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_persist_dir(cls, save_dir: str, load_corpus: bool = True, mmap: bool = True, **kwargs):
        bm25 = bm25s.BM25.load(save_dir, load_corpus=load_corpus, mmap=mmap, **kwargs)
        with open(os.path.join(save_dir, DEFAULT_PERSIST_FILENAME), "r", encoding="utf-8") as f:
            persist_args = json.load(f)
        return cls(existing_bm25=bm25, **persist_args)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = bm25s.tokenize(
            query,
            stemmer=None if self.skip_stemming else self.stemmer,
            token_pattern=self.token_pattern,
            show_progress=self._verbose,
        )
        indexes, scores = self.bm25.retrieve(tokenized_query, k=self.similarity_top_k, show_progress=self._verbose)

        indexes = indexes[0]  # np.ndarray
        scores = scores[0]  # np.ndarray

        nodes: List[NodeWithScore] = list()
        for idx, score in zip(indexes, scores):
            # idx can be an int or a dict of the node
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(score)))
        return nodes

    def batch_retrieve(self, queries: List[str]) -> List[List[NodeWithScore]]:
        return [self._retrieve(QueryBundle(query)) for query in queries]


class CustomJSONReader(BaseReader):
    def __init__(
        self,
        levels_back: Optional[int] = None,
        collapse_length: Optional[int] = None,
        ensure_ascii: bool = False,
        is_jsonl: Optional[bool] = False,
        clean_json: Optional[bool] = True,
    ):
        super().__init__()
        self.levels_back = levels_back
        self.collapse_length = collapse_length
        self.ensure_ascii = ensure_ascii
        self.is_jsonl = is_jsonl
        self.clean_json = clean_json

    @staticmethod
    def load_jsonl_nodes(input_file: str, extra_info: Optional[dict] = None) -> List[TextNode]:
        if extra_info is None:
            extra_info = dict()
        nodes = list()
        for line in stream_load_jsonl(input_file):
            nodes.append(TextNode(id_=str(line["id"]), text=line["contents"], metadata=extra_info))
        return nodes
