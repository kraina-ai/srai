"""Base classes for loaders."""

import abc

import geopandas as gpd


class BaseLoader(abc.ABC):
    """Base abstract class for loaders."""

    @abc.abstractmethod
    def download(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Download data for a given area.

        Args:
            gdf: GeoDataFrame with the area of interest.

        Returns:
            GeoDataFrame with the downloaded data.
        """
        raise NotImplementedError
