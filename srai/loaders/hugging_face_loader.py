"""
Hugging Face loader.

This module contains Hugging Face loader.
"""

from typing import Optional

import pandas as pd

from srai._optional import import_optional_dependencies
from srai.loaders import Loader


class HuggingFaceLoader(Loader):
    """
    Hugging Face loader.

    Loader to download dataset from HuggingFace and return a Pandas DataFrame.
    """

    def __init__(self, hf_token: Optional[str] = None) -> None:
        """
        Initialize loader.

        Args:
            hf_token (str): Token from HuggingFace Hub account.
        """
        import_optional_dependencies(
            dependency_group="datasets",
            modules=["datasets"],
        )
        self.hf_token = hf_token

    def load(self, dataset_name: str, name: Optional[str] = None) -> pd.DataFrame:
        """
        Method to load data files from Hugging Face.

        Args:
            dataset_name (str, optional): Name of dataset.
            name (Optional[str], optional): Name of version of dataset e.g. "nyc_bike_2013". \
                Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, name=name, token=self.hf_token, trust_remote_code=True)

        return dataset["train"].to_pandas()
