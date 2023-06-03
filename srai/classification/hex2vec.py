""""""
from pathlib import Path
from typing import TYPE_CHECKING, List

from srai.embedders.hex2vec.model import Hex2VecModel
from srai.utils._optional import import_optional_dependencies

if TYPE_CHECKING:  # pragma: no cover
    import torch


try:  # pragma: no cover
    from pytorch_lightning import LightningModule

except ImportError:
    from srai.utils._pytorch_stubs import LightningModule


class Hex2VecModelForRegionClassification(LightningModule):  # type: ignore
    """Hex2Vec classification model."""

    def __init__(
        self, hex2vec_layer_sizes: List[int], n_classes: int, learning_rate: float = 0.001
    ):
        """
        Initialize Hex2VecModel.

        Raises:
            ValueError: If layer_sizes contains less than 2 elements.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be greater than 1")
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        from torch import nn

        super().__init__()
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.hex2vec_model = Hex2VecModel(layer_sizes=hex2vec_layer_sizes)
        self.classification_head = nn.Linear(hex2vec_layer_sizes[-1], n_classes)

    def forward(self, X_anchor: "torch.Tensor") -> "torch.Tensor":
        """
        Calculate embedding for a region.

        Args:
            X_anchor (torch.Tensor): Region features.
        """
        import torch.nn.functional as F

        x = self.hex2vec_model(X_anchor)
        x = F.relu(x)
        x = F.log_softmax(self.classification_head(x), dim=1)
        return x

    def training_step(self, batch: List["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        """"""
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import f1_score as f1

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f_score = f1(preds, y, task="multiclass", num_classes=self.n_classes, average="macro")

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_f1", f_score, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: List["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import f1_score as f1

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f_score = f1(preds, y, task="multiclass", num_classes=self.n_classes, average="macro")

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        self.log("val_f1", f_score, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Configure optimizer."""
        import torch

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_kwargs(self) -> dict:
        """Get model save kwargs."""
        return {
            "hex2vec_layer_sizes": self.hex2vec_layer_sizes,
            "n_classes": self.n_classes,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def load(cls, path: Path, **kwargs: dict) -> "Hex2VecModelForRegionClassification":
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
