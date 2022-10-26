"""Base class for regionizers."""

import abc

import geopandas as gpd


class BaseRegionizer(abc.ABC):
    """Base abstract class for regionizers."""

    @abc.abstractmethod
    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        This one should treat the input as a
        single region.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionized.

        Returns:
            GeoDataFrame with the regionized data.
        """
        raise NotImplementedError
