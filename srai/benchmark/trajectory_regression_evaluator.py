"""This module contains TrajectoryRegressionEvaluator."""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import srai.datasets as sds
from srai.benchmark import BaseEvaluator

from ._custom_metrics import (
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TrajectoryRegressionEvaluator(BaseEvaluator):
    """Evaluator for regression task."""

    def __init__(self) -> None:
        """Create the evaluator."""
        super().__init__(task="trajectory_regression")

    def evaluate(
        self,
        dataset: sds.PointDataset | sds.TrajectoryDataset,
        predictions: np.ndarray,
        log_metrics: bool = True,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Evaluate regression predictions against test set.

        This regression evaluator is designed for predictions for h3 grid trajectories.

        Args:
            dataset (sds.TrajectoryDataset): Dataset to evaluate.
            predictions (np.ndarray): Predictions returned by your model. Should match trip_id.
            log_metrics (bool, optional): If True, logs metrics to the console. Defaults to True.
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to HF
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            trip_ids (list[str]): List of region IDs. Required for region-based evaluators.

        Raises:
            ValueError: If region id for H3 index not found in region_ids.

        Returns:
            dict[str, float]: Dictionary with metrics values for the task.
        """
        if not isinstance(dataset, sds.TrajectoryDataset):
            raise ValueError("This evaluator only supports TrajectoryDataset.")

        if dataset.version != "TTE":
            raise ValueError(
                f"Trajectory Regression Evaluator is made for regression tasks\
                            such as Travevel Time Estimation (TTE). Your dataset version is\
                              preprocessed for task: {dataset.version}"
            )
        trip_ids = kwargs.get("trip_ids")

        if trip_ids is None:
            raise ValueError("Trip_ids are required for trajectory based evaluation.")

        _, _, h3_test = dataset.get_h3_with_labels()
        # target_column = dataset.target if dataset.target is not None else "count"
        # _, h3_test = dataset.get_h3_with_labels()

        if h3_test is None:
            raise ValueError("The function 'get_h3_with_labels' returned None for h3_test.")
        else:
            trip_indexes = [int(idx) for idx in h3_test[dataset.target].to_list()]
            labels = h3_test["duration"].to_numpy()

        trip_to_prediction = {
            trip_id: prediction for trip_id, prediction in zip(trip_ids, predictions)
        }
        trip_to_prediction_keys = trip_to_prediction.keys()
        available_trip_indexes = set(trip_indexes).intersection(trip_to_prediction_keys)
        missing_trip_indexes = set(trip_indexes).difference(available_trip_indexes)

        if missing_trip_indexes:
            logging.info(
                f"{len(missing_trip_indexes)} trip_ids have no matching trip indexes in\
                         the test set and will be skipped in evaluation. Measuring for \
                          {len(available_trip_indexes)} indexes."
            )

        # Reorder labels and predictions accordingly
        if len(missing_trip_indexes) != len(trip_ids):
            filtered_labels = np.array(
                [label for idx, label in zip(trip_indexes, labels) if idx in trip_to_prediction]
            )
            ordered_predictions = np.array(
                [trip_to_prediction[idx] for idx in available_trip_indexes]
            )

            trip_ids[:] = available_trip_indexes
            predictions = ordered_predictions

            metrics = self._compute_metrics(predictions, filtered_labels)
            if log_metrics:
                self._log_metrics(metrics)
            return metrics
        else:
            raise ValueError("No matching trip ids found in test dataset")

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
