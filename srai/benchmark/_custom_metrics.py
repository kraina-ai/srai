"""
Metrics module.

This module contains implementation of non-standard metrics used by evaluators.
"""

import h3
import numpy as np
from geopy.distance import great_circle

from srai._optional import import_optional_dependencies


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10
) -> float:
    """Calculates regression metric: Mean Absolute Percentage Error.

    Args:
        y_true (np.ndarray): Expected values
        y_pred (np.ndarray): Predicted values
        epsilon (float): Small constant to avoid division by zero (default: 1e-10)

    Returns:
        float: Mean absolute percentage error value
    """
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10
) -> float:
    """Calculates regression metric: Symmetric Mean Absolute Percentage Error.

    Args:
        y_true (np.ndarray): Expected values
        y_pred (np.ndarray): Predicted values
        epsilon (float): Small constant to avoid division by zero (default: 1e-10)

    Returns:
        float: Symmetric mean absolute percentage error value
    """
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denominator) * 100)


def haversine_sequence(true_h3_seq: list[str], pred_h3_seq: list[str]) -> float:
    """
    Compute the average Haversine distance between pairs of H3 cells.

    Args:
        true_h3_seq (List[str]): Ground truth sequence of H3 cell indexes.
        pred_h3_seq (List[str]): Predicted sequence of H3 cell indexes.

    Returns:
        float: Mean Haversine distance in meters between corresponding H3 pairs.
               Returns float('inf') if no valid pairs are found.
    """
    dists: list[float] = []
    for true_h3, pred_h3 in zip(true_h3_seq, pred_h3_seq):
        if true_h3 and pred_h3:
            true_latlon = h3.cell_to_latlng(true_h3)
            pred_latlon = h3.cell_to_latlng(pred_h3)
            dist = great_circle(true_latlon, pred_latlon).meters
            dists.append(dist)
    return float(np.mean(dists)) if dists else float("inf")


def dtw_distance(true_h3_seq: list[str], pred_h3_seq: list[str]) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences of H3 cells.

    Args:
        true_h3_seq (List[str]): Ground truth sequence of H3 cell indexes.
        pred_h3_seq (List[str]): Predicted sequence of H3 cell indexes.

    Returns:
        float: DTW distance between the latitude-longitude paths of the two sequences.
    """
    import_optional_dependencies(dependency_group="datasets", modules=["fastdtw"])
    from fastdtw import fastdtw

    true_coords = [h3.cell_to_latlng(h) for h in true_h3_seq]
    pred_coords = [h3.cell_to_latlng(h) for h in pred_h3_seq]
    distance, _ = fastdtw(true_coords, pred_coords, dist=lambda x, y: great_circle(x, y).meters)
    return float(distance)


def sequence_accuracy(true: list[str], pred: list[str]) -> float:
    """
    Compute accuracy of predicted H3 sequence by exact element-wise match.

    Args:
        true (List[str]): Ground truth sequence of H3 indexes.
        pred (List[str]): Predicted sequence of H3 indexes.

    Returns:
        float: Proportion of elements that match exactly.
    """
    return float(np.mean([t == p for t, p in zip(true, pred)]))
