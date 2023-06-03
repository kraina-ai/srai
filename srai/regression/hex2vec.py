"""
"""
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from srai.utils._optional import import_optional_dependencies
from srai.embedders.hex2vec.model import Hex2VecModel

if TYPE_CHECKING:  # pragma: no cover
    import torch


try:  # pragma: no cover
    from pytorch_lightning import LightningModule

except ImportError:
    from srai.utils._pytorch_stubs import LightningModule


class Hex2VecModelForRegionRegression(LightningModule):  # type: ignore
    """
    Hex2Vec regression model.
    """

    def __init__(self, hex2vec_layer_sizes: List[int], learning_rate: float = 0.001):
        """
        Initialize Hex2VecModel.

        Raises:
            ValueError: If layer_sizes contains less than 2 elements.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        from torch import nn

        super().__init__()
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.hex2vec_model = Hex2VecModel(layer_sizes=layer_sizes)
        self.regression_head = nn.Linear(layer_sizes[-1], 1)
        

    def forward(self, X_anchor: "torch.Tensor") -> "torch.Tensor":
        """
        """
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import f1_score as f1
        x = self.hex2vec_model(X_anchor)
        x = F.relu(x)
        x = self.regression_head(x)
        return x



    def training_step(self, batch: List["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        """
        """
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import mean_squared_error, mean_absolute_error

        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        mae = mean_absolute_error(y_hat, y)
        rmse = mean_squared_error(y_hat, y, squared=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: List["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import mean_squared_error, mean_absolute_error

        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        mae = mean_absolute_error(y_hat, y)
        rmse = mean_squared_error(y_hat, y, squared=False)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Configure optimizer."""
        import torch

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_kwargs(self) -> dict:
        """Get model save kwargs."""
        return {"hex2vec_layer_sizes": self.hex2vec_layer_sizes, "learning_rate": self.learning_rate}

    @classmethod
    def load(cls, path: Path, **kwargs: dict) -> "Hex2VecModelForRegionRegression":
        """
        Load model from a file.

        Args:
            path (str): Path to the file.
            **kwargs (dict): Additional kwargs to pass to the model constructor.
        """
        import torch

        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model


