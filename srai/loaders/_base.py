"""Base class for loaders."""

import abc
from collections.abc import Iterable
from typing import Any, Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from srai.constants import WGS84_CRS


def prepare_area_gdf_for_loader(
    area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """
    Prepare an area for the loader.

    Loader expects a GeoDataFrame input, but users shouldn't be limited by this requirement.
    All Shapely geometries will by transformed into GeoDataFrame with proper CRS.

    Args:
        area (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
            Area to be parsed into GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Sanitized GeoDataFrame.
    """
    if isinstance(area, gpd.GeoDataFrame):
        # Return a GeoDataFrame with changed CRS
        return area.to_crs(WGS84_CRS)
    elif isinstance(area, gpd.GeoSeries):
        # Create a GeoDataFrame with GeoSeries
        return gpd.GeoDataFrame(geometry=area, crs=WGS84_CRS)
    elif isinstance(area, Iterable):
        # Create a GeoSeries with a list of geometries
        return prepare_area_gdf_for_loader(gpd.GeoSeries(area, crs=WGS84_CRS))
    # Wrap a single geometry with a list
    return prepare_area_gdf_for_loader([area])


class Loader(abc.ABC):
    """Abstract class for loaders."""

    @abc.abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> gpd.GeoDataFrame:  # pragma: no cover
        """
        Load data for a given area.

        Args:
            *args: Positional arguments dependating on a specific loader.
            **kwargs: Keyword arguments dependating on a specific loader.

        Returns:
            GeoDataFrame with the downloaded data.
        """
        raise NotImplementedError

    def _prepare_area_gdf(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
    ) -> gpd.GeoDataFrame:
        return prepare_area_gdf_for_loader(area)
