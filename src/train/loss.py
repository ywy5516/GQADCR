#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

BASENAME = Path(__file__).stem


def calculate_align_loss(
    query_embs: Tensor,
    rewrite_embs: Tensor,
    device: Union[str, torch.device],
    is_normalize: bool = True,
    temperature: Optional[float] = 0.07,
) -> Tensor:
    """Contrastive loss with in-batch negative samples only"""
    assert query_embs.shape == rewrite_embs.shape
    bs = query_embs.shape[0]  # batch size
    if is_normalize:
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        rewrite_embs = F.normalize(rewrite_embs, p=2, dim=-1)

    logits = torch.matmul(query_embs, rewrite_embs.T)  # B * B
    if temperature is not None:
        logits /= temperature
    labels = torch.arange(bs, device=device)  # 1 * B
    return F.cross_entropy(logits, labels, reduction="mean")


def calculate_rank_loss(
    query_embs: Tensor,
    positive_embs: Tensor,
    negative_embs: Tensor,
    device: Union[str, torch.device],
    is_normalize: bool = True,
    temperature: Optional[float] = 0.07,
) -> Tensor:
    """Calculate retrieval ranking loss, including in-batch negative samples"""
    assert query_embs.shape == positive_embs.shape
    bs = query_embs.shape[0]
    n_neg = negative_embs.shape[1]
    assert bs == len(negative_embs)

    if is_normalize:
        query_embs = F.normalize(query_embs, p=2, dim=-1)  # (B, D)
        positive_embs = F.normalize(positive_embs, p=2, dim=-1)  # (B, D)
        negative_embs = F.normalize(negative_embs, p=2, dim=-1)  # (B, N_neg, D)

    label_matrix = torch.arange(bs, device=device)  # (B,)
    label_matrix = (label_matrix.unsqueeze(0) == label_matrix.unsqueeze(1)).float()  # (B, B)

    batch_scores = torch.matmul(query_embs, positive_embs.T)  # (B, B)
    assert batch_scores.shape == label_matrix.shape
    # Extract cosine similarity values for positive pairs (B, 1)
    pos_scores = batch_scores[label_matrix.bool()].view(bs, -1)
    # Extract cosine similarity values for in-batch negative pairs (B, B-1)
    batch_neg_scores = batch_scores[~label_matrix.bool()].view(bs, -1)

    additional_neg_scores = torch.einsum("bd,bnd->bn", query_embs, negative_embs)  # (B, N_neg)
    assert additional_neg_scores.shape == (bs, n_neg)

    logits = torch.cat([pos_scores, batch_neg_scores, additional_neg_scores], dim=1)  # (B, B+N_neg)

    # The similarity values of positive samples are placed in the first column of logits (index 0)
    labels = torch.zeros(bs, dtype=torch.long, device=device)

    if temperature is not None:
        logits /= temperature

    return F.cross_entropy(logits, labels, reduction="mean")
