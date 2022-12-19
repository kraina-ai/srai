"""Base class for regionizers."""

import abc

import geopandas as gpd


class BaseRegionizer(abc.ABC):
    """Base abstract class for regionizers."""

    @abc.abstractmethod
    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        This one should treat the input as a single region.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionized.

        Returns:
            GeoDataFrame with the regionized data.

        """
        raise NotImplementedError

    def _set_crs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Set the CRS to WGS84.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be converted.

        Returns:
            GeoDataFrame converted to EPSG:4326.

        """
        return gdf.to_crs(epsg=4326)

    def _explode_multipolygons(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Explode multipolygons into multiple polygons.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be exploded.

        Returns:
            GeoDataFrame with exploded multipolygons.

        """
        return gdf.explode(index_parts=True).reset_index(drop=True)
