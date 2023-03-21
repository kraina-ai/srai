"""
Embedding model for Highway2Vec.

This module contains the embedding model from the `highway2vec` paper [1].

References:
    [1] https://doi.org/10.1145/3557918.3565865
"""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim


class Highway2VecModel(pl.LightningModule):  # type: ignore
    """Autoencoder based embedding model for highway2vec."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, latent_dim: int = 30, lr: float = 1e-3):
        """
        Init Highway2VecModel.

        Args:
            in_dim (int): Number of features.
            hidden_dim (int, optional): Number of hidden units. Defaults to 64.
            latent_dim (int, optional): Embedding size. Defaults to 30.
            lr (float, optional): Learning rate. Defaults to 1e-3.
        """
        super().__init__()

        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
        """
        z: torch.Tensor = self.encoder(x)
        return z

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (torch.Tensor): Batch.
            batch_idx (int): Batch index.
        """
        return self._common_step(batch, batch_idx, "train")

    def _common_step(self, batch: torch.Tensor, batch_idx: int, stage: str) -> torch.Tensor:
        """
        Perform common step.

        Args:
            batch (torch.Tensor): Batch.
            batch_idx (int): Batch index.
            stage (str): Name of the stage - e.g. train, valid, test.
        """
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
