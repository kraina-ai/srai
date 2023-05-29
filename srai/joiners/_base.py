"""Base class for joiners."""

import abc
from typing import TYPE_CHECKING, Union

import geopandas as gpd

if TYPE_CHECKING:
    import duckdb


class Joiner(abc.ABC):
    """Abstract class for joiners."""

    @abc.abstractmethod
    def transform(
        self,
        regions: Union["duckdb.DuckDBPyRelation", gpd.GeoDataFrame],
        features: Union["duckdb.DuckDBPyRelation", gpd.GeoDataFrame],
    ) -> "duckdb.DuckDBPyRelation":  # pragma: no cover
        """
        Join features to regions.

        Args:
            regions (Union[duckdb.DuckDBPyRelation, gpd.GeoDataFrame]): regions with which features
                are joined
            features (Union[duckdb.DuckDBPyRelation, gpd.GeoDataFrame]): features to be joined
        Returns:
            GeoDataFrame with an intersection of regions and features, which contains
            a MultiIndex and optionally a geometry with the intersection
        """
        raise NotImplementedError
