"""
Embedding model for gtfs2vec.

This module contains embedding model from gtfs2vec paper [1].

References:
    1. https://doi.org/10.1145/3486640.3491392
"""

from typing import TYPE_CHECKING, Any

from srai._optional import import_optional_dependencies
from srai.embedders import Model

if TYPE_CHECKING:  # pragma: no cover
    import torch


class GTFS2VecModel(Model):
    """Autoencoder based embedding model for gtfs2vec."""

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 48,
        n_embed: int = 64,
    ) -> None:
        """
        Init GTFS2VecModel.

        Args:
            n_features (int): Number of features.
            n_hidden (int, optional): Number of hidden units. Defaults to 48.
            n_embed (int, optional): Embedding size. Defaults to 64.
        """
        super().__init__()
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        from torch import nn

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_embed)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_embed, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_features)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
        """
        embedding: torch.Tensor = self.encoder(x)
        return embedding

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Configure optimizer."""
        import torch

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: "torch.Tensor", batch_idx: Any) -> "torch.Tensor":
        """
        Training step.

        Args:
            batch (torch.Tensor): Batch.
            batch_idx (Any): Batch index.
        """
        from torch.nn import functional as F

        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
