"""
Embedding model for gtfs2vec.

This module contains embedding model from gtfs2vec paper [1].

References:
    1. https://doi.org/10.1145/3486640.3491392
"""
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class GTFS2VecModel(LightningModule):  # type: ignore
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
        self.n_features = n_features

        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_embed)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_embed, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
        """
        embedding: torch.Tensor = self.encoder(x)
        return embedding

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: Any) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (torch.Tensor): Batch.
            batch_idx (Any): Batch index.
        """
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
