"""The benchmark module contains classes for evaluating the performance of a model on a dataset."""

from ._base import Evaluator
from .regression_evaluator import RegressionEvaluator

__all__ = ["Evaluator", "RegressionEvaluator"]
