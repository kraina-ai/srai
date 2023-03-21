"""Base class for embedders."""

import abc

import geopandas as gpd
import pandas as pd

from srai.constants import GEOMETRY_COLUMN


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
            ValueError: If features_gdf is empty and self.expected_output_features is not set.
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        raise NotImplementedError

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
