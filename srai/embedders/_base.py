"""Base class for embedders and models."""

import abc
from pathlib import Path
from typing import Any, TypeVar, Union

import geopandas as gpd
import pandas as pd

from srai.constants import GEOMETRY_COLUMN

try:  # pragma: no cover
    from pytorch_lightning import LightningModule

except ImportError:
    from srai.embedders._pytorch_stubs import LightningModule


class Model(LightningModule):  # type: ignore
    """Class for model based on LightningModule."""

    def get_config(self) -> dict[str, Any]:
        """Get model config."""
        model_config = {
            k: v
            for k, v in vars(self).items()
            if k[0] != "_"
            and k
            not in (
                "training",
                "prepare_data_per_node",
                "allow_zero_length_dataloader_with_multiple_devices",
            )
        }

        return model_config

    def save(self, path: Union[Path, str]) -> None:
        """
        Save the model to a directory.

        Args:
            path (Path): Path to the directory.
        """
        import torch

        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs: Any) -> "Model":
        """
        Load model from a file.

        Args:
            path (Union[Path, str]): Path to the file.
            **kwargs (dict): Additional kwargs to pass to the model constructor.
        """
        import torch

        if isinstance(path, str):
            path = Path(path)

        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model


class Embedder(abc.ABC):
    """Abstract class for embedders."""

    @abc.abstractmethod
    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:  # pragma: no cover
        """
        Embed regions using features.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Returns:
            pd.DataFrame: Embedding and geometry index for each region in regions_gdf.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def save(self, path: Union[Path, str]) -> None:
    #     """
    #     Save the embedder to a directory.

    #     Args:
    #         path (Path): Path to the directory.
    #     """
    #     raise NotImplementedError

    # @classmethod
    # @abc.abstractmethod
    # def load(cls, path: Union[Path, str]) -> "Embedder":
    #     """
    #     Load the embedder from a directory.

    #     Args:
    #         path (Path): Path to the directory.

    #     Returns:
    #         Embedder: The loaded embedder.
    #     """
    #     raise NotImplementedError

    def _validate_indexes(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> None:
        if regions_gdf.index.name is None:
            raise ValueError("regions_gdf must have a named index.")

        if features_gdf.index.name is None:
            raise ValueError("features_gdf must have a named index.")

        if not isinstance(joint_gdf.index, pd.MultiIndex):
            raise ValueError(
                f"joint_gdf.index must be of type pandas.MultiIndex, not {type(joint_gdf.index)}"
            )

        if len(joint_gdf.index.names) != 2:
            raise ValueError(
                f"joint_gdf.index must have 2 levels, has {len(joint_gdf.index.names)}"
            )

        if regions_gdf.index.name != joint_gdf.index.names[0]:
            raise ValueError(
                f"Name of regions_gdf.index ({regions_gdf.index.name}) must be equal to the name of"
                f" the 1st level of joint_gdf.index ({joint_gdf.index.names[0]})"
            )

        if features_gdf.index.name != joint_gdf.index.names[1]:
            raise ValueError(
                f"Name of features_gdf.index ({features_gdf.index.name}) must be equal to the name"
                f" of the 2nd level of joint_gdf.index ({joint_gdf.index.names[1]})"
            )

    def _remove_geometry_if_present(self, data: gpd.GeoDataFrame) -> pd.DataFrame:
        if GEOMETRY_COLUMN in data.columns:
            data = data.drop(columns=GEOMETRY_COLUMN)
        return pd.DataFrame(data)


ModelT = TypeVar("ModelT", bound=Model)
