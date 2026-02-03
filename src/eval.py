#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from argparse import ArgumentParser
from typing import Dict, Tuple, cast

import numpy as np
from pytrec_eval import RelevanceEvaluator

sys.path.append(os.path.abspath("."))


class CustomeEvaluator:
    def __init__(self, qrel_trec_file: str, relevance_threshold: int = 1):
        self.qrels_data, self.qrels_ndcg_data = self._read_qrel_data(qrel_trec_file, relevance_threshold)

    @staticmethod
    def _read_qrel_data(qrel_trec_file: str, relevance_threshold: int = 1) -> Tuple[dict, dict]:
        with open(qrel_trec_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data, ndcg_data = dict(), dict()
        for line in lines:
            line = line.strip().split()
            sample_id, passage_id, relevance = line[0], line[2], int(line[3])
            if sample_id not in data:
                data[sample_id] = dict()
                ndcg_data[sample_id] = dict()
            data[sample_id][passage_id] = relevance
            ndcg_data[sample_id][passage_id] = 1 if relevance >= relevance_threshold else 0
        return data, ndcg_data

    def _read_run_data(self, run_trec_file: str) -> dict:
        with open(run_trec_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = dict()
        for line in lines:
            line = line.strip().split()
            # sample_id 0 passage_id rank rank_score real_score
            sample_id, passage_id, relevance = line[0], line[2], int(line[4])
            if sample_id not in data:
                data[sample_id] = dict()
            data[sample_id][passage_id] = relevance
        return data

    @staticmethod
    def _pytrec_mrr(qrels_data: dict, runs_data: dict) -> list:
        evaluator = RelevanceEvaluator(qrels_data, {"recip_rank"})
        scores = evaluator.evaluate(runs_data)
        return [v["recip_rank"] for v in scores.values()]

    @staticmethod
    def _pytrec_ndcg_at_3(qrels_ndcg_data: dict, runs_data: dict):
        evaluator = RelevanceEvaluator(qrels_ndcg_data, {"ndcg_cut.3"})
        scores = evaluator.evaluate(runs_data)
        return np.average([v["ndcg_cut_3"] for v in scores.values()])

    @staticmethod
    def _pytrec_recall_at_10(qrels_data: dict, runs_data: dict) -> list:
        evaluator = RelevanceEvaluator(qrels_data, {"recall.10"})
        scores = evaluator.evaluate(runs_data)
        return [v["recall_10"] for v in scores.values()]

    def evaluate_from_data(self, runs_data: dict):
        evaluator1 = RelevanceEvaluator(self.qrels_data, {"recip_rank", "recall.10"})
        scores1 = cast(Dict[str, dict], evaluator1.evaluate(runs_data))
        mrr_list = [v["recip_rank"] for v in scores1.values()]
        recall_10_list = [v["recall_10"] for v in scores1.values()]

        evaluator2 = RelevanceEvaluator(self.qrels_ndcg_data, {"ndcg_cut.3"})
        scores2 = cast(Dict[str, dict], evaluator2.evaluate(runs_data))
        ndcg_3_list = [v["ndcg_cut_3"] for v in scores2.values()]
        return {
            "MRR": round(np.average(mrr_list) * 100, 3),
            "NDCG@3": round(np.average(ndcg_3_list) * 100, 3),
            "Recall@10": round(np.average(recall_10_list) * 100, 3),
        }

    def evaluate_from_file(self, run_trec_file: str) -> dict:
        runs_data = self._read_run_data(run_trec_file)
        return self.evaluate_from_data(runs_data)


def main(args):
    evaluator = CustomeEvaluator(args.qrel_trec_file, args.relevance_threshold)
    print(evaluator.evaluate_from_file(args.run_trec_file))


if __name__ == "__main__":
    # python src/eval.py
    parser = ArgumentParser()
    parser.add_argument("-r", "--run_trec_file", type=str, required=True)
    parser.add_argument("-q", "--qrel_trec_file", type=str, required=True)
    parser.add_argument("-t", "--relevance_threshold", type=int, default=1)

    main(parser.parse_args())
