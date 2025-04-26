"""This module contains HexRegressionEvaluator dataset."""

from typing import Any, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import srai.datasets as sds
from srai.benchmark import BaseEvaluator

from ._custom_metrics import (
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
)


class HexRegressionEvaluator(BaseEvaluator):
    """Evaluator for regression task."""

    def __init__(self) -> None:
        """Create the evaluator."""
        super().__init__(task="regression")

    def evaluate(
        self,
        dataset: sds.PointDataset,
        predictions: np.ndarray,
        log_metrics: bool = True,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Evaluate regression predictions against test set.

        This regression evaluator is designed for H3 grid predictions.

        Args:
            dataset (sds.PointDataset): Dataset to evaluate.
            predictions (np.ndarray): Predictions returned by your model. Should match regions_id.
            log_metrics (bool, optional): If True, logs metrics to the console. Defaults to True.
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to HF
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            region_ids (list[str]): List of region IDs. Required for region-based evaluators.

        Raises:
            ValueError: If region id for H3 index not found in region_ids.

        Returns:
            dict[str, float]: Dictionary with metrics values for the task.
        """
        region_ids = kwargs.get("region_ids")

        if region_ids is None:
            raise ValueError("Region_ids are required for region-based evaluation.")

        target_column = dataset.target if dataset.target is not None else "count"
        _, h3_test = dataset.get_h3_with_labels()

        if h3_test is None:
            raise ValueError("The function 'get_h3_with_labels' returned None for h3_test.")
        else:
            h3_indexes = h3_test["region_id"].to_list()
            labels = h3_test[target_column].to_numpy()

        region_to_prediction = {
            region_id: prediction for region_id, prediction in zip(region_ids, predictions)
        }

        # order predictions according to the order of region_ids
        try:
            ordered_predictions = [region_to_prediction[h3] for h3 in h3_indexes]
        except KeyError as err:
            raise ValueError(
                "Region id for H3 index {err.args[0]} not found in region_ids."
            ) from err

        region_ids[:] = h3_indexes
        predictions = np.array(ordered_predictions)
        metrics = self._compute_metrics(predictions, labels)
        if log_metrics:
            self._log_metrics(metrics)
        return metrics

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
