"""
Embedding model for gtfs2vec.

This module contains embedding model from gtfs2vec paper [1].

References:
    [1] https://doi.org/10.1145/3486640.3491392

"""
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class GTFS2VecModel(LightningModule):  # type: ignore
    """Autoencoder based embedding model for gtfs2vec."""

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 48,
        emb_size: int = 64,
        sparsity_lambda: Union[float, None] = None,
    ) -> None:
        """
        Init GTFS2VecModel.

        Args:
            n_features (int): Number of features.
            n_hidden (int, optional): Number of hidden units. Defaults to 48.
            emb_size (int, optional): Embedding size. Defaults to 64.
            sparsity_lambda (Union[float, None], optional): Sparsity lambda. Defaults to None.

        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, emb_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_size, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_features)
        )
        self.sparsity_lambda = sparsity_lambda

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
        if self.sparsity_lambda is not None:
            loss = loss + self.sparsity_lambda * torch.mean(torch.abs(z))
        self.log("train_loss", loss)
        return loss
