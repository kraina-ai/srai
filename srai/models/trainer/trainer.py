"""
Trainer module.

This module contains implementation of trainer for base models.
"""

import copy
import logging
import os
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from srai.models.evaluator import Evaluator


class Trainer:
    """Trainer Class."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn,
        task: str = "regression",
        batch_size: int = 64,
        epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        early_stopping: bool = True,
        metric2look4: Optional[str] = None,
        save_best: Optional[bool] = False,
        save_dir: Optional[str] = None,
        # lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    ):
        """
        Trainer class.

        Args:
            model (nn.Module): Model to train
            train_dataset (Dataset): Training dataset with "X" as an embedding column,\
                    "y" as a target label
            eval_dataset (Dataset): Evaluation dataset with "X" as an embedding column,\
                    "y" as a target label
            task (str): training task type, possible values:
                     "trajectory_prediction", "regression", "poi_prediction".
                       Default to regression\
            batch_size (int): batch size \
            device (str): device used for training,\
            epochs (int): number of epochs for training \
            early_stopping (bool): stop training if loss haven't decreased in \
            metric2look4 (str): a metric used in evaluation to distinguish the best model
            optimizer (torch.optim.Optimizer): torch optimizer
            loss_fn (torch.nn): torch loss function
            save_best (Optional[bool]): save best model after training
            save_dir (Optional[str]): directory to save the best model

        Raises:
            ValueError: task is not supported in srai library
            ValueError: train_dataset should be instance of torch.utils.data.Dataset
            ValueError: eval_dataset should be instance of torch.utils.data.Dataset
        """
        if task not in ["trajectory_prediction", "regression", "poi_prediction"]:
            raise ValueError(f"Task {task} not supported in srai library.")
        self.task = task
        # self.lr_scheduler = lr_scheduler
        # if self.lr_scheduler is not None:
        self.model = model
        self.model.to(device)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.metric2look4 = metric2look4
        self.early_stopping = early_stopping
        self.device = device
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.best_weights = None
        self.evaluator = Evaluator(self.task, self.device)
        self.save_best = save_best
        self.save_dir = save_dir

        # if not isinstance(train_dataset, Dataset):
        #     raise ValueError("train_dataset should be instance of torch.utils.data.Dataset")
        # if not isinstance(eval_dataset, Dataset):
        #     raise ValueError("eval_dataset should be instance of torch.utils.data.Dataset")

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

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
        stop_counter = 0
        best_metric = np.inf
        prev_eval_loss = np.inf  # init to infinity
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
                labels = batch["y"].to(self.device).reshape(-1, 1)

                outputs = self.model(inputs)
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
                return_metrics=True,
                loss_fn=self.loss_fn,
            )
            loss_eval.append(eval_loss)
            if self.early_stopping:
                if eval_loss >= prev_eval_loss:
                    stop_counter += 1
                    if stop_counter == 5:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break
                else:
                    stop_counter = 0
                prev_eval_loss = eval_loss

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
