"""
Trainer module.

This module contains implementation of trainer for base models.
"""

import copy
import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from srai.models.evaluator import Evaluator


class Trainer:
    """_Trainer Class."""

    def __init__(
        self,
        model: nn.Module,
        training_args: dict[str, Any],
        train_dataset: Dataset,
        eval_dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn,
        save_best: Optional[bool] = False,
        save_dir: Optional[str] = None,
        # lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    ):
        """
        _summary_.

        Args:
            model (nn.Module): Model to train
            training_args (dict[str, Any]): Dictionary with training arguments such as:\
                    batch_size : int,\
                    task : str, \
                    device : str (default cuda),
                    epochs: int, \
                    early_stopping : bool, \
                    metric2look4 : str (a metric used in evaluation to get the best model)
            train_dataset (Dataset): Training dataset with "X" as an embedding column,\
                    "y" as a target label
            eval_dataset (Dataset): Evaluation dataset with "X" as an embedding column,\
                    "y" as a target label
            optimizer (torch.optim.Optimizer): torch optimizer
            loss_fn (torch.nn): torch loss function
            save_best (Optional[bool]): save best model after training
            save_dir (Optional[str]): directory to save the best model

        Raises:
            ValueError: task is not supported in srai library
            ValueError: train_dataset should be instance of torch.utils.data.Dataset
            ValueError: eval_dataset should be instance of torch.utils.data.Dataset
        """
        self.model = model
        self.args = training_args
        self.batch_size = self.args.get("batch_size", 64)
        self.task = self.args.get("task", "regression")
        self.metric2look4 = self.args.get("metric2look4", None)

        if self.task not in ["trajectory_prediction", "regression", "poi_prediction"]:
            raise ValueError(f"Task {self.task} not supported in srai library.")

        # self.lr_scheduler = lr_scheduler
        # if self.lr_scheduler is not None:
        self.optimizer = optimizer
        self.early_stopping = self.args.get("early_stopping", True)
        self.device = self.args.get("device", "cuda")
        self.epochs = self.args.get("epochs", 50)
        self.loss_fn = loss_fn
        self.best_weights = None
        self.evaluator = Evaluator(self.task, self.device)
        self.save_best = save_best
        self.save_dir = save_dir

        if not isinstance(train_dataset, Dataset):
            raise ValueError("train_dataset should be instance of torch.utils.data.Dataset")
        if not isinstance(eval_dataset, Dataset):
            raise ValueError("eval_dataset should be instance of torch.utils.data.Dataset")

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self) -> tuple[nn.Module, list[float], list[float]]:
        """
        Train model on given task.

        Returns:
            nn.Module: best model after training
            list[float]: train loss
            list[float]: eval loss
        Raises:
            NotImplementedError: tasks that haven't been implemented yet
        """
        best_metric = np.inf  # init to infinity
        loss_eval = []
        loss_train = []

        for epoch in range(self.epochs):
            batch_loss_list = []
            self.model.train()
            for batch in tqdm(
                self.train_dataloader,
                desc=f"Epoch: {epoch}",
                total=len(self.train_dataloader),
            ):
                inputs = batch["X"].to(self.device)
                labels = batch["y"].to(self.device)

                outputs = self.model(inputs, labels=labels)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss_list.append(loss.item())

            logging.info(
                f"Epoch [{epoch+1}/{self.epochs}], avg_loss: {np.mean(batch_loss_list):.4f}"
            )
            loss_train.append(np.mean(batch_loss_list))
            metrics, eval_loss = self.evaluator.evaluate(
                self.model,
                self.eval_dataloader,
                compute_loss=True,
                loss_fn=self.loss_fn,
            )
            loss_eval.append(eval_loss)
            if self.metric2look4 is None:
                if self.task == "regression":
                    self.metric2look4 = "MSE"

            if metrics[str(self.metric2look4)] < best_metric:
                best_metric = metrics[str(self.metric2look4)]
                self.best_weights = copy.deepcopy(self.model.state_dict())
                logging.info(
                    f"Best model found at epoch {epoch}, \
                              {self.metric2look4}: {best_metric:.4f}"
                )

        self.model.load_state_dict(self.best_weights)
        if self.save_best:
            if self.save_dir is None:
                self.save_dir = ".."
            torch.save(self.model, os.path.join(self.save_dir, f"best_{self.task}_model.pkl"))
        return self.model, loss_train, loss_eval

    def predict(self) -> None:
        """
        Prediction for h3 index /test dataset.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
