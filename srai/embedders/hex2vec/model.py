"""
Hex2Vec model.

This module contains the embedding model from Hex2Vec paper[1].

References:
    [1] https://dl.acm.org/doi/10.1145/3486635.3491076
"""
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, sigmoid
from torchmetrics.functional import f1_score as f1


class Hex2VecModel(pl.LightningModule):  # type: ignore
    """
    Hex2Vec embedding model.

    This class implements the embedding model from Hex2Vec paper. It is based on a skip-gram model
    with negative sampling and triplet-loss. The model takes vectors of numbers as input (raw counts
    of features) per region and outputs dense embeddings.
    """

    def __init__(self, encoder_sizes: List[int], learning_rate: float = 0.001):
        """
        Initialize Hex2VecModel.

        Args:
            encoder_sizes (List[int]): List of sizes for the encoder layers.
                The first element is the input size (number of features),
                the last element is the output (embedding) size.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
        """
        super().__init__()
        self.learning_rate = learning_rate

        def create_layers(sizes: List[Tuple[int, int]]) -> nn.Sequential:
            layers = []
            for i, (input_size, output_size) in enumerate(sizes):
                linear = nn.Linear(input_size, output_size)
                nn.init.xavier_uniform_(linear.weight)
                layers.append(nn.Linear(input_size, output_size))
                if i != len(sizes) - 1:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        sizes = list(zip(encoder_sizes[:-1], encoder_sizes[1:]))
        self.encoder = create_layers(sizes)

    def forward(self, X_current: torch.Tensor) -> torch.Tensor:
        """
        Calculate embedding for a region.

        Args:
            X_current (torch.Tensor): Region features.
        """
        return self.encoder(X_current)

    def predict_proba(self, X_current: torch.Tensor, X_other: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability of X_current being neighbours with X_other.

        X_current and X_other are assumed to have the same batch size.
        The probabilities are calculated in pairs, i.e. the first element of X_current
        is compared with the first element of X_other.

        Args:
            X_current (torch.Tensor): Current regions.
            X_other (torch.Tensor): Other regions.
        """
        score = self.predict_scores(X_current, X_other)
        return sigmoid(score)

    def predict_scores(self, X_current: torch.Tensor, X_other: torch.Tensor) -> torch.Tensor:
        """
        Predict raw unnormalized scores of X_current being neighbours with X_other.

        X_current and X_other are assumed to have the same batch size.
        The scores are calculated in pairs, i.e. the first element of X_current
        is compared with the first element of X_other.
        In order to get probabilities, use the sigmoid function.

        Args:
            X_current (torch.Tensor): Current regions.
            X_other (torch.Tensor): Other regions.
        """
        X_current_em = self(X_current)
        X_other_em = self(X_other)
        score = torch.mul(X_current_em, X_other_em).sum(dim=1)
        return score

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform one training step.

        One batch of data consists of 5 tensors:
                - X_current: Current regions.
                - X_context: Context regions. The regions assumed to be neighbours
                    of the corresponding regions in X_current.
                - X_negative: Negative regions. The regions assumed to NOT be neighbours
                    of the corresponding regions in X_current.
                - y_pos: Labels for positive pairs. All ones.
                - y_neg: Labels for negative pairs. All zeros.
            The regions in X_current, X_context and X_negative are first embedded using the encoder.
            After that, the dot product of the corresponding embeddings is calculated.
            The loss is calculated as a binary cross-entropy between the dot product and the labels.

        Args:
            batch (List[torch.Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        X_current, X_context, X_negative, y_pos, y_neg = batch
        scores_pos = self.predict_scores(X_current, X_context)
        scores_neg = self.predict_scores(X_current, X_negative)

        scores = torch.cat([scores_pos, scores_neg])
        y = torch.cat([y_pos, y_neg])

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(sigmoid(scores), y.int(), task="binary")
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_f1", f_score, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform one validation step.

        Args:
            batch (List[torch.Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        X_current, X_context, X_negative, y_pos, y_neg = batch
        scores_pos = self.predict_scores(X_current, X_context)
        scores_neg = self.predict_scores(X_current, X_negative)

        scores = torch.cat([scores_pos, scores_neg])
        y = torch.cat([y_pos, y_neg])

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(sigmoid(scores), y.int(), task="binary")
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_f1", f_score, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
