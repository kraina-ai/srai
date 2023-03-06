"""Base class for regionizers."""

import abc

import geopandas as gpd


class Regionizer(abc.ABC):
    """Abstract class for regionizers."""

    @abc.abstractmethod
    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:  # pragma: no cover
        """
        Regionize a given GeoDataFrame.

        This one should treat the input as a single region.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionized.

        Returns:
            GeoDataFrame with the regionized data.
        """
        raise NotImplementedError

    def _explode_multipolygons(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Explode multipolygons into multiple polygons.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be exploded.

        Returns:
            GeoDataFrame with exploded multipolygons.
        """
        return gdf.explode(index_parts=True).reset_index(drop=True)
