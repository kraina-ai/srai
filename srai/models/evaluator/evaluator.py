"""
Evaluator module.

This module contains implementation of evaluator for models.
"""

import logging
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from metrics import (
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")


class Evaluator:
    """Evaluator class."""

    def __init__(
        self,
        task: str,
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Evaluator.

        Args:
            task (str): Evaluation task type, possible values:
                     "trajectory_prediction", "regression", "poi_prediction".
            device (Optional[str], optional): Type of device used for evaluation.
                    Defaults to "cuda' if available.

        Raises:
            ValueError: If task type is not supported.
        """
        if task not in ["trajectory_prediction", "regression", "poi_prediction"]:
            raise ValueError(f"Task {task} not supported.")
        self.task = task
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        test_data: Dataset | DataLoader,
        data_loader_params: Optional[dict[str, Any]] = None,
        compute_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], dict[str, float]]] = None,
        compute_loss: Optional[bool] = False,
        loss_fn: Optional[Any] = None,
    ) -> tuple[dict[str, float], np.ndarray] | np.ndarray:
        """
        Evaluates model on a chosen dataset with task-dependent metrics.

        Args:
            test_data (Dataset | DataLoader): The test split of dataset chosen for evaluation.\
                Should contain "X" column with vector embedding, and "y" with target label
            model (nn.Module): Model intended for evaluation.
            data_loader_params (Optional[dict], optional): Parameters passed to DataLoader.\
                Batch size defaults to 64, shuffle defaults to False.
            compute_metrics (Optional[Callable]): function that computes metrics from model\
                predictions and target labels. Has to return dictionary where keys are metrics's \
                names and values are they results
            compute_loss (Optional[bool]): Boolean if to compute loss together with metrics,\
                f.e used in training
            loss_fn (Optional[Any]): function (torch or any custom) that computes loss from\
                model prediction and target labels

        Raises:
            ValueError: If test_dataset is not instance of torch.utils.data.Dataset.
        """
        if not isinstance(test_data, DataLoader):
            data_loader = DataLoader(
                test_data,
                **(
                    data_loader_params
                    if data_loader_params
                    else {"batch_size": 64, "shuffle": False}
                ),
            )
        else:
            data_loader = test_data

        model.eval()
        metrics_per_batch = []
        eval_loss = []
        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(data_loader),
                desc="Evaluation",
                total=len(data_loader),
            ):
                inputs = batch["X"].to(self.device)
                labels = batch["y"].to(self.device)

                outputs = model(inputs, labels=labels)
                if compute_loss:
                    if loss_fn is None:
                        loss_fn = nn.L1Loss()
                        logging.info("Loss function wasn't given. Using default nn.L1Loss()")
                    loss = loss_fn(outputs, labels)
                    eval_loss.append(float(loss.item()))

                if compute_metrics is None:
                    metrics = self.compute_metrics(predictions=outputs, labels=labels)
                else:
                    metrics = compute_metrics(outputs, labels)
                metrics_per_batch.append({"Batch": i, **metrics})

        mean_metrics = {
            key: np.mean([batch[key] for batch in metrics_per_batch])
            for key in metrics_per_batch[0].keys()
            if key != "Batch"
        }

        log_metrics(metrics_per_batch, mean_metrics)
        if compute_loss:
            logging.info(f"Eval loss: {np.mean(eval_loss):.4f}")
            return mean_metrics, np.mean(eval_loss)
        else:
            return mean_metrics

    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """
        Computes metruics for the given task for a single batch.

        Args:
            predictions (torch.Tensor): Predictions return by model.
            labels (torch.Tensor): Target values.

        Returns:
            dict: Dictionary with task-dedicated metrics values.

        Raises:
            NotImplementedError: If task is not supported.
        """
        if self.task == "trajectory_prediction":
            return self.compute_trajectory_prediction_metrics(
                predictions=predictions, labels=labels
            )
        elif self.task == "regression":
            return self.compute_regression_metrics(predictions=predictions, labels=labels)
        elif self.task == "poi_prediction":
            return self.compute_poi_prediction_metrics(predictions=predictions, labels=labels)
        else:
            raise NotImplementedError

    def compute_regression_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """
        Calucates regression metrics. Metrics included :  (MSE, RMSE, MAE, MAPE, sMAPE).

        Args:
            predictions (torch.Tensor): Predictions return by model.
            labels (torch.Tensor): Target values.

        Returns:
            dict: dictionary with regression metrics values.
        """
        mse = mean_squared_error(labels.numpy(), predictions.numpy())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels.numpy(), predictions.numpy())
        mape = mean_absolute_percentage_error(labels.numpy(), predictions.numpy())
        smape = symmetric_mean_absolute_percentage_error(labels.numpy(), predictions.numpy())
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "sMAPE": smape,
        }

    def compute_trajectory_prediction_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """
        Calucates trajectory prediction metrics.

        Args:
            predictions (torch.Tensor): Predictions return by model.
            labels (torch.Tensor): Target values.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def compute_poi_prediction_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """
        Calucates POI prediction metrics.

        Args:
            predictions (torch.Tensor): Predictions return by model.
            labels (torch.Tensor): Target values.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError


def log_metrics(metrics_per_batch: list[dict[str, float]], mean_metrics: dict[str, float]) -> None:
    """
    _summary_.

    Args:
        metrics_per_batch (list): Metrics values per batch.
        mean_metrics (dict): Mean metrics values across all batches.
    """
    logging.info("Metrics per batch:")
    for batch_metrics in metrics_per_batch:
        batch_info = ", ".join(
            [f"{key}={value:.4f}" for key, value in batch_metrics.items() if key != "Batch"]
        )
        logging.info(f"Batch {batch_metrics['Batch']}: {batch_info}")
    logging.info("-----")

    logging.info("Mean metrics across all batches:")
    for key, value in mean_metrics.items():
        logging.info(f"{key}: {value:.4f}")
