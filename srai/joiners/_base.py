"""Base class for joiners."""

import abc

import geopandas as gpd


class Joiner(abc.ABC):
    """Abstract class for joiners."""

    @abc.abstractmethod
    def transform(
        self,
        regions: gpd.GeoDataFrame,
        features: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:  # pragma: no cover
        """
        Join features to regions.

        Args:
            regions (gpd.GeoDataFrame): regions with which features are joined
            features (gpd.GeoDataFrame): features to be joined
        Returns:
            GeoDataFrame with an intersection of regions and features, which contains
            a MultiIndex and optionally a geometry with the intersection
        """
        raise NotImplementedError
