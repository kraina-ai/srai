"""
Hex2Vec model.

This module contains the embedding model from Hex2Vec paper[1].

References:
    [1] https://dl.acm.org/doi/10.1145/3486635.3491076
"""

from typing import TYPE_CHECKING

from srai._optional import import_optional_dependencies
from srai.embedders import Model

if TYPE_CHECKING:  # pragma: no cover
    import torch


class Hex2VecModel(Model):
    """
    Hex2Vec embedding model.

    This class implements the embedding model from Hex2Vec paper. It is based on a skip-gram model
    with negative sampling and triplet-loss. The model takes vectors of numbers as input (raw counts
    of features) per region and outputs dense embeddings.
    """

    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.001):
        """
        Initialize Hex2VecModel.

        Args:
            layer_sizes (List[int]): List of sizes for the model layers.
                The first element is the input size (number of features),
                the last element is the output (embedding) size.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.


        Raises:
            ValueError: If layer_sizes contains less than 2 elements.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        from torch import nn

        super().__init__()
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least 2 elements")

        def create_layers(sizes: list[tuple[int, int]]) -> nn.Sequential:
            layers = []
            for i, (input_size, output_size) in enumerate(sizes):
                linear = nn.Linear(input_size, output_size)
                nn.init.xavier_uniform_(linear.weight)
                layers.append(nn.Linear(input_size, output_size))
                if i != len(sizes) - 1:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        sizes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.encoder = create_layers(sizes)

    def forward(self, X_anchor: "torch.Tensor") -> "torch.Tensor":
        """
        Calculate embedding for a region.

        Args:
            X_anchor (torch.Tensor): Region features.
        """
        return self.encoder(X_anchor)

    def predict_proba(self, X_anchor: "torch.Tensor", X_context: "torch.Tensor") -> "torch.Tensor":
        """
        Predict the probability of X_anchor being neighbours with X_context.

        X_anchor and X_context are assumed to have the same batch size.
        The probabilities are calculated in pairs, i.e. the first element of X_anchor
        is compared with the first element of X_context.

        Args:
            X_anchor (torch.Tensor): Anchor regions.
            X_context (torch.Tensor): Context regions.
        """
        from torch.nn.functional import sigmoid

        score = self.predict_scores(X_anchor, X_context)
        return sigmoid(score)

    def predict_scores(self, X_anchor: "torch.Tensor", X_context: "torch.Tensor") -> "torch.Tensor":
        """
        Predict raw unnormalized scores of X_anchor being neighbours with X_context.

        X_anchor and X_context are assumed to have the same batch size.
        The scores are calculated in pairs, i.e. the first element of X_anchor
        is compared with the first element of X_context.
        In order to get probabilities, use the sigmoid function.

        Args:
            X_anchor (torch.Tensor): Anchor regions.
            X_context (torch.Tensor): Context regions.
        """
        import torch

        X_anchor_em = self(X_anchor)
        X_context_em = self(X_context)
        score = torch.mul(X_anchor_em, X_context_em).sum(dim=1)
        return score

    def training_step(self, batch: list["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        """
        Perform one training step.

        One batch of data consists of 3 tensors:
                - X_anchor: Anchor regions.
                - X_positive: Positive regions. The regions assumed to be neighbours
                    of the corresponding regions in X_anchor.
                - X_negative: Negative regions. The regions assumed to NOT be neighbours
                    of the corresponding regions in X_anchor.
            The regions in X_anchor, X_positive and X_negative are first embedded using the encoder.
            After that, the dot product of the corresponding embeddings is calculated.
            The loss is calculated as a binary cross-entropy between the dot product and the labels.

        Args:
            batch (List[torch.Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import f1_score as f1

        X_anchor, X_positive, X_negative = batch
        scores_pos = self.predict_scores(X_anchor, X_positive)
        scores_neg = self.predict_scores(X_anchor, X_negative)

        scores = torch.cat([scores_pos, scores_neg])
        y_pos = torch.ones_like(scores_pos)
        y_neg = torch.zeros_like(scores_neg)
        y = torch.cat([y_pos, y_neg]).to(X_anchor)

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(F.sigmoid(scores), y.int(), task="binary")
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_f1", f_score, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: list["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        """
        Perform one validation step.

        Args:
            batch (List[torch.Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import f1_score as f1

        X_anchor, X_positive, X_negative = batch
        scores_pos = self.predict_scores(X_anchor, X_positive)
        scores_neg = self.predict_scores(X_anchor, X_negative)

        scores = torch.cat([scores_pos, scores_neg])
        y_pos = torch.ones_like(scores_pos)
        y_neg = torch.zeros_like(scores_neg)
        y = torch.cat([y_pos, y_neg]).to(X_anchor)

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(F.sigmoid(scores), y.int(), task="binary")
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_f1", f_score, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Configure optimizer."""
        import torch

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
