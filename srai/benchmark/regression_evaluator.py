"""This module contains RegressionEvaluator dataset."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from srai.benchmark import Evaluator

from ._custom_metrics import (
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
)


class RegressionEvaluator(Evaluator):
    """Evaluator for regression task."""

    def __init__(self) -> None:
        """Create the evaluator."""
        super().__init__(task="regression")

    def _compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Calucates regression metrics. Metrics included :  (MSE, RMSE, MAE, MAPE, sMAPE).

        Args:
            predictions (np.ndarray): Predictions return by model.
            labels (np.ndarray): Target values.

        Returns:
            dict[str, float]: dictionary with regression metrics values.
        """
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)
        mape = mean_absolute_percentage_error(labels, predictions)
        smape = symmetric_mean_absolute_percentage_error(labels, predictions)
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "sMAPE": smape,
        }
