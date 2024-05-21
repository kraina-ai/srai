"""
Regression model module.

This module contains implementation of base model of regression.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn

try:  # pragma: no cover
    import torch  # noqa: F811
    from torch import nn  # noqa: F811

except ImportError:
    from srai.embedders._pytorch_stubs import nn, torch


class RegressionBaseModel(nn.Module):  # type: ignore
    """
    Regression base model.

    Definition of Regression Model
    """

    def __init__(
        self,
        embeddings_size: int,
        linear_sizes: Optional[list[int]] = None,
        activation_function: Optional[nn.Module] = None,
    ):
        """
        Initializaiton of regression module.

        Args:
            embeddings_size (int): size of input embedding
            linear_sizes (Optional[list[int]], optional): sizes of linear layers inside module. \
                Defaults to [500, 1000].
            activation_function (Optional[nn.Module], optional): activation function from torch.nn \
                Defaults to ReLU.
        """
        super().__init__()
        if linear_sizes is None:
            linear_sizes = [500, 1000]
        if activation_function is None:
            activation_function = nn.ReLU()
        self.model = torch.nn.Sequential()
        previous_size = embeddings_size
        for cnt, size in enumerate(linear_sizes):
            self.model.add_module(f"linear_{cnt}", nn.Linear(previous_size, size))
            self.model.add_module(f"ReLU_{cnt}", activation_function)
            previous_size = size
            if cnt % 2:
                self.model.add_module(f"dropout_{cnt}", nn.Dropout(p=0.2))
        self.model.add_module("linear_final", nn.Linear(previous_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Vector data

        Returns:
            torch.Tensor: target value
        """
        return self.model(x)
