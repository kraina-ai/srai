"""Base class for benchmark evaluation."""

import abc
import logging
from typing import Any, Literal, Optional

import numpy as np

import srai.datasets as sds

logging.basicConfig(level=logging.INFO, format="%(message)s")


class BaseEvaluator(abc.ABC):
    """Abstract class for benchmark evaluators."""

    def __init__(
        self,
        task: Literal[
            "trajectory_regression", "regression", "poi_prediction", "mobility_prediction"
        ],
    ) -> None:
        self.task = task

    @abc.abstractmethod
    def evaluate(
        self,
        dataset: sds.PointDataset | sds.TrajectoryDataset,
        predictions: np.ndarray,
        log_metrics: bool = True,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Evaluate predictions againts test set.

        Args:
            dataset (sds.HuggingFaceDataset): Dataset to evaluate on.
            predictions (np.ndarray): Predictions returned by your model.
            log_metrics (bool, optional): If True, logs metrics to the console. Defaults to True.
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to HF
                Defaults to None.
            **kwargs: Additional keyword arguments depending on the task.

        Keyword Args:
            region_ids (list[str], optional): List of region IDs. Required for region-based\
                  evaluators.
            point_of_interests (np.ndarray, optional): Points of interest. Required for point-based\
                evaluators.

        Returns:
            dict[str, float]: Dictionary with metrics values for the task.

        Note:
            Specific subclasses may require different sets of keyword arguments.
        """
        # if self.task == "regression":
        #     train_gdf, test_gdf = dataset.load(version=f"res_{resolution}", hf_token=hf_token)
        #     target_column = dataset.target if dataset.target is not None else "count"
        #     # h3_indexes, labels = self._get_labels(test, resolution, target_column)
        #     _, h3_test = dataset.get_h3_with_labels(
        #         train_gdf=train_gdf, test_gdf=test_gdf, resolution=resolution
        #     )

        #     if h3_test is None:
        #         raise ValueError("The function 'get_h3_with_labels' returned None for h3_test.")
        #     else:
        #         h3_indexes = h3_test["region_id"].to_list()
        #         labels = h3_test[target_column].to_numpy()

        #     region_to_prediction = {
        #         region_id: prediction for region_id, prediction in zip(region_ids, predictions)
        #     }

        #     # order predictions according to the order of region_ids
        #     try:
        #         ordered_predictions = [region_to_prediction[h3] for h3 in h3_indexes]
        #     except KeyError as err:
        #         raise ValueError(
        #             "Region id for H3 index {err.args[0]} not found in region_ids."
        #         ) from err

        #     region_ids[:] = h3_indexes
        #     predictions = np.array(ordered_predictions)
        #     metrics = self._compute_metrics(predictions, labels)
        #     if log_metrics:
        #         self._log_metrics(metrics)
        #     return metrics
        # else:
        #     raise NotImplementedError
        raise NotImplementedError

    # @abc.abstractmethod
    # def evaluate_on_benchmark(self, model: torch.nn.Module):
    #     """Evaluates the model on all available datasets for the chosen task \
    #         in the benchmark."""
    #     raise NotImplementedError

    @abc.abstractmethod
    def _compute_metrics(self, predictions: np.array, labels: np.array) -> dict[str, float]:
        """
        Computes metrics for the given task.

        Args:
            predictions (np.array): Predictions returned by model as tensor of values.
            labels (np.array): Target values.

        Returns:
            dict[str, float]: Dictionary with task-dedicated metrics values.
        """
        raise NotImplementedError

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """
        Logs evaluation metrics to the console.

        Args:
            metrics (dict[str,float]): Evaluation metrics as a dataset.

        Returns:
            None
        """
        logging.info("Resulting metrics: ")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.4f}")
