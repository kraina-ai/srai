"""
Embedding model for highway2vec.

This module contains the embedding model from `highway2vec` paper [1].

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
        """TODO."""
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
        """TODO."""
        z: torch.Tensor = self.encoder(x)
        return z

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """TODO."""
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """TODO."""
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """TODO."""
        return self._common_step(batch, batch_idx, "test")

    # def _prepare_batch(self, batch, batch_idx):
    #     x = batch
    #     # x = x.view(x.size(0), -1)
    #     return x

    def _common_step(self, batch: torch.Tensor, batch_idx: int, stage: str) -> torch.Tensor:
        """TODO."""
        x = self._prepare_batch(batch, batch_idx)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """TODO."""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
