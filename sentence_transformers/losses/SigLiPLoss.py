from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class SigLIPLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        temperature: float = 1.0,
        similarity_fct=util.cos_sim,
        bidirectional: bool = False,
    ) -> None:
        """
        SigLIP-inspired contrastive loss for sentence transformers.
        Args:
            model: SentenceTransformer model
            temperature: Temperature scaling for logits
            similarity_fct: Similarity function between embeddings
            bidirectional: if set to `True`, will average loss from `q` to `d` and `d` to `q`.
        """
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.similarity_fct = similarity_fct
        self.bidirectional = bidirectional

    def forward(self, sentence_features: Iterable[dict], labels: Any = None) -> torch.Tensor:
        # Get embeddings for each input
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        similarities = self.similarity_fct(embeddings_a, embeddings_b)

        logits = similarities / self.temperature

        targets = torch.arange(len(logits), device=logits.device)
        # Bo: this is bidirectinoal
        loss_a = nn.functional.binary_cross_entropy_with_logits(
            logits, torch.eye(len(logits), device=logits.device)[targets]
        )
        if self.bidirectional:
            loss_b = nn.functional.binary_cross_entropy_with_logits(
                logits.t(), torch.eye(len(logits), device=logits.device)[targets]
            )
            return (loss_a + loss_b) / 2
        else:
            return loss_a

    def get_config_dict(self) -> dict:
        return {"temperature": self.temperature, "similarity_fct": self.similarity_fct.__name__}
