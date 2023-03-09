"""TODO."""
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, sigmoid
from torchmetrics.functional import f1_score as f1


class Hex2VecModel(pl.LightningModule):  # type: ignore
    """TODO."""

    def __init__(self, encoder_sizes: List[int]):
        """TODO."""
        super().__init__()

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """TODO."""
        return self.encoder(X)

    def predict_proba(self, Xt: torch.Tensor, Xc: torch.Tensor) -> torch.Tensor:
        """TODO."""
        score = self(Xt, Xc)
        return sigmoid(score)

    def predict_scores(self, Xt: torch.Tensor, Xc: torch.Tensor) -> torch.Tensor:
        """TODO."""
        Xt_em = self(Xt)
        Xc_em = self(Xc)
        score = torch.mul(Xt_em, Xc_em).sum(dim=1)
        return score

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """TODO."""
        Xt, Xc, Xn, y_pos, y_neg = batch
        scores_pos = self.predict_scores(Xt, Xc)
        scores_neg = self.predict_scores(Xt, Xn)

        scores = torch.cat([scores_pos, scores_neg])
        y = torch.cat([y_pos, y_neg])

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(sigmoid(scores), y.int(), task="binary")
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_f1", f_score, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """TODO."""
        Xt, Xc, Xn, y_pos, y_neg = batch
        scores_pos = self.predict_scores(Xt, Xc)
        scores_neg = self.predict_scores(Xt, Xn)

        scores = torch.cat([scores_pos, scores_neg])
        y = torch.cat([y_pos, y_neg])

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(sigmoid(scores), y.int(), task="binary")
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_f1", f_score, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """TODO."""
        return torch.optim.Adam(self.parameters(), lr=0.001)
