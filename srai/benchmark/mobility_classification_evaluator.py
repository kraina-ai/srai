"""This module contains MobilityClassificationEvaluator."""

import logging
from typing import Any, Optional

import numpy as np

import srai.datasets as sds
from srai.benchmark import BaseEvaluator

from ._custom_metrics import dtw_distance, haversine_sequence, sequence_accuracy

logging.basicConfig(level=logging.INFO, format="%(message)s")


class MobilityClassificationEvaluator(BaseEvaluator):
    """Evaluator for models predicting H3 index trajectories directly."""

    def __init__(self) -> None:
        """Create the evaluator."""
        super().__init__(task="mobility_classification")

    def evaluate(
        self,
        dataset: sds.PointDataset | sds.TrajectoryDataset,
        predictions: list[list[str]],
        log_metrics: bool = True,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Evaluate predicted H3 index sequences against ground truth H3 sequences.

        Args:
            dataset (TrajectoryDataset): Dataset to evaluate.
            predictions (List[List[str]]): Predicted sequences of H3 indexes.
            log_metrics (bool): If True, logs metrics.
            hf_token (str, optional): Ignored.
            **kwargs:
                - trip_ids (List[str]): List of trip IDs corresponding to predictions.

        Returns:
            dict[str, float]: Evaluation metrics.
        """
        if not isinstance(dataset, sds.TrajectoryDataset):
            raise ValueError("This evaluator only supports TrajectoryDataset.")

        trip_ids = kwargs.get("trip_ids")
        if trip_ids is None:
            raise ValueError("`trip_ids` are required for trajectory evaluation.")

        _, h3_test = dataset.get_h3_with_labels()
        if h3_test is None:
            raise ValueError("The function 'get_h3_with_labels' returned None for h3_test.")

        trip_id_col = dataset.target if dataset.target is not None else "trip_id"
        h3_col = "h3_sequence_y"  # Adjust if this column name differs

        # Map predictions to their corresponding trip ID
        trip_to_prediction = {
            int(trip_id): prediction for trip_id, prediction in zip(trip_ids, predictions)
        }

        all_trip_ids = set(map(int, h3_test[trip_id_col].unique()))
        available_trip_ids = [trip_id for trip_id in trip_to_prediction if trip_id in all_trip_ids]
        missing_trip_ids = set(trip_to_prediction.keys()) - set(available_trip_ids)

        if missing_trip_ids:
            logging.info(
                f"{len(missing_trip_ids)} trip_ids have no matching data in the test set "
                f"and will be skipped. Evaluating {len(available_trip_ids)} trip(s)."
            )

        if not available_trip_ids:
            raise ValueError("No matching trip ids found in test dataset.")

        # Build filtered true sequences and predictions
        true_sequences = []
        filtered_predictions = []

        for trip_id in available_trip_ids:
            trip_df = h3_test[h3_test[trip_id_col] == trip_id]
            true_h3_seq = trip_df[h3_col].iloc[0]
            pred_h3_seq = trip_to_prediction[trip_id]

            true_sequences.append(true_h3_seq)
            filtered_predictions.append(pred_h3_seq)

        # Compute metrics
        metrics = self._compute_metrics(true_sequences, filtered_predictions)

        if log_metrics:
            self._log_metrics(metrics)

        return metrics

    def _compute_metrics(
        self,
        true_sequences: list[list[str]],
        pred_sequences: list[list[str]],
    ) -> dict[str, float]:
        """
        Compute trajectory evaluation metrics based on H3 index sequences.

        Metrics included:
            - SequenceAccuracy: Proportion of exact matches between predicted and true H3 cells.
            - MeanHaversineDistance: Average Haversine distance (in meters) between predicted and \
                true H3 coordinates.
            - MeanDTW: Average Dynamic Time Warping distance between predicted and true sequences.

        Args:
            true_sequences (list[list[str]]): Ground truth sequences of H3 cell indexes.
            pred_sequences (list[list[str]]): Predicted sequences of H3 cell indexes.

        Returns:
            dict[str, float]: Dictionary containing the averaged trajectory evaluation metrics.
        """
        acc_list = []
        haversine_list = []
        dtw_list = []

        for true_seq, pred_seq in zip(true_sequences, pred_sequences):
            acc = sequence_accuracy(true_seq, pred_seq)
            hav = haversine_sequence(true_seq, pred_seq)
            dtw = dtw_distance(true_seq, pred_seq)

            acc_list.append(acc)
            haversine_list.append(hav)
            dtw_list.append(dtw)

        return {
            "SequenceAccuracy": float(np.mean(acc_list)),
            "MeanHaversineDistance": float(np.mean(haversine_list)),
            "MeanDTW": float(np.mean(dtw_list)),
        }
