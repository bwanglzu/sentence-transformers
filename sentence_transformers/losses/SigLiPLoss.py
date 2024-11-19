from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class SigLIPLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature: float = 1.0, similarity_fct=util.cos_sim) -> None:
        """
        SigLIP contrastive loss with learnable temperature parameter.
        Args:
            model: SentenceTransformer model
            temperature: Initial temperature value
            similarity_fct: Similarity function between embeddings
        """
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.similarity_fct = similarity_fct

    def forward(self, sentence_features: Iterable[dict], labels: Any = None) -> torch.Tensor:
        # Get embeddings for each input
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        # Compute cosine similarities
        similarities = self.similarity_fct(embeddings_a, embeddings_b)

        logits = similarities / torch.abs(self.temperature)

        # Compute sigmoid-based loss
        targets = torch.arange(len(logits), device=logits.device)
        loss_a = nn.functional.binary_cross_entropy_with_logits(
            logits, torch.eye(len(logits), device=logits.device)[targets]
        )

        # Symmetric loss
        loss_b = nn.functional.binary_cross_entropy_with_logits(
            logits.t(), torch.eye(len(logits), device=logits.device)[targets]
        )

        return (loss_a + loss_b) / 2

    def get_config_dict(self) -> dict:
        return {"temperature": self.temperature.item(), "similarity_fct": self.similarity_fct.__name__}
