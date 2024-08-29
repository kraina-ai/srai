"""
Embedding model for Highway2Vec.

This module contains the embedding model from the `highway2vec` paper [1].

References:
    1. https://doi.org/10.1145/3557918.3565865
"""

from typing import TYPE_CHECKING

from srai._optional import import_optional_dependencies
from srai.embedders import Model

if TYPE_CHECKING:  # pragma: no cover
    import torch


class Highway2VecModel(Model):
    """Autoencoder based embedding model for highway2vec."""

    def __init__(self, n_features: int, n_hidden: int = 64, n_embed: int = 30, lr: float = 1e-3):
        """
        Init Highway2VecModel.

        Args:
            n_features (int): Number of features.
            n_hidden (int, optional): Number of hidden units. Defaults to 64.
            n_embed (int, optional): Embedding size. Defaults to 30.
            lr (float, optional): Learning rate. Defaults to 1e-3.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        from torch import nn

        super().__init__()

        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embed),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features),
        )
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.lr = lr

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
        """
        z: torch.Tensor = self.encoder(x)
        return z

    def training_step(self, batch: "torch.Tensor", batch_idx: int) -> "torch.Tensor":
        """
        Training step.

        Args:
            batch (torch.Tensor): Batch.
            batch_idx (int): Batch index.
        """
        return self._common_step(batch, batch_idx, "train")

    def _common_step(self, batch: "torch.Tensor", batch_idx: int, stage: str) -> "torch.Tensor":
        """
        Perform common step.

        Args:
            batch (torch.Tensor): Batch.
            batch_idx (int): Batch index.
            stage (str): Name of the stage - e.g. train, valid, test.
        """
        import torch.nn.functional as F

        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Configure optimizer."""
        import torch

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
