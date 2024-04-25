"""
Metrics module.

This module contains implementation of non-standard metrics used by evaluator.
"""

import numpy as np
import torch


def mean_absolute_percentage_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
    """Calculates regression metric: Mean Absolute Percentage Error.

    Args:
        y_true (torch.Tensor): Expected values
        y_pred (torch.Tensor): Predicted values

    Returns:
        np.ndarray: mean absolute percentage error value
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> np.ndarray:
    """Calculates regression metric: Symmetric Mean Absolute Percentage Error.

    Args:
        y_true (torch.Tensor): Expected values
        y_pred (torch.Tensor): Predicted values

    Returns:
        np.ndarray: symmetric mean absolute percentage error value
    """
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
