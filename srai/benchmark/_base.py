"""Base class for benchmark evaluation."""

import abc
import logging
from typing import Literal, Optional

import geopandas as gpd
import numpy as np

import srai.datasets as sds
from srai.regionalizers import H3Regionalizer

logging.basicConfig(level=logging.INFO, format="%(message)s")


class Evaluator(abc.ABC):
    """Abstract class for benchmark evaluators."""

    def __init__(
        self, task: Literal["trajectory_prediction", "regression", "poi_prediction"]
    ) -> None:
        self.task = task

    def evaluate(
        self,
        dataset: sds.HuggingFaceDataset,
        region_ids: list[str],
        predictions: np.ndarray,
        resolution: int,
        log_metrics: bool = True,
        hf_token: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Evaluate predictions againts test set.

        Args:
            dataset (sds.HuggingFaceDataset): Dataset to evaluate.
            region_ids (list[str]): List of region ids. Should match predictions.
            predictions (np.ndarray,): Predictions returned by your model. Should match regions_id.
            resolution (int): Resolution of the H3 grid.
            log_metrics (bool, optional): If True, logs metrics to the console. Defaults to True.
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to HF
                Defaults to None.

        Returns:
            dict[str, float]: Dictionary with metrics values for the task.
        """
        if self.task == "regression":
            _, test = dataset.load(version=f"res_{resolution}", hf_token=hf_token)
            target_column = dataset.target
            h3_indexes, labels = self._get_labels(test, resolution, target_column)
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
        else:
            raise NotImplementedError

    def get_evaluation_info(
        self,
        dataset: sds.HuggingFaceDataset,
        resolution: int,
        hf_token: Optional[str] = None,
    ) -> list[str]:
        """
        Retrieves the H3 indexes in the test set for a given benchmark.

        Args:
            dataset (sds.HuggingFaceDataset): Dataset to evaluate.
            resolution (int): Resolution of the H3 grid.
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to HF
                Defaults to None.

        Returns:
            list[str]: List of H3 indexes in the test set.
        """
        if self.task == "regression":
            _, test = dataset.load(version=f"res_{resolution}", hf_token=hf_token)
            target_column = dataset.target
            h3_indexes, _ = self._get_labels(test, resolution, target_column)
            return h3_indexes
        else:
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

    def _get_labels(
        self, test_gdf: gpd.GeoDataFrame, resolution: int, target_column: Optional[str]
    ) -> tuple[list[str], np.array]:
        """Returns labels from the dataset."""
        if self.task == "regression":
            if target_column is None:
                target_column = "count"
            gdf_ = test_gdf.copy()
            regionalizer = H3Regionalizer(resolution=resolution)
            regions = regionalizer.transform(test_gdf)
            joined_gdf = gpd.sjoin(test_gdf, regions, how="left", predicate="within")  # noqa: E501
            joined_gdf.rename(columns={"index_right": "h3_index"}, inplace=True)

            averages_hex = joined_gdf.groupby("h3_index").size().reset_index(name=target_column)
            gdf_ = regions.merge(
                averages_hex, how="inner", left_on="region_id", right_on="h3_index"
            )
            return gdf_["h3_index"].to_list(), gdf_[target_column].to_numpy()

        else:
            raise NotImplementedError
