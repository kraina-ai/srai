"""Base class for loaders."""

import abc
from collections.abc import Iterable
from typing import Any, Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from srai._typing import is_expected_type
from srai.constants import WGS84_CRS
from srai.geodatatable import VALID_GEO_INPUT, GeoDataTable, prepare_geo_input

VALID_AREA_INPUT = Union[
    BaseGeometry,
    Iterable[BaseGeometry],
    gpd.GeoSeries,
    gpd.GeoDataFrame,
    VALID_GEO_INPUT,
]


def prepare_area_input_for_loader(area: VALID_AREA_INPUT) -> GeoDataTable:
    """
    Prepare an area for the loader.

    Loader expects a GeoDataTable input, but users shouldn't be limited by this requirement.
    All Shapely geometries will by transformed into GeoDataTable with proper CRS.

    Args:
        area (VALID_AREA_INPUT): Area to be parsed into GeoDataTable.

    Returns:
        GeoDataTable: Sanitized GeoDataTable.
    """
    if is_expected_type(area, VALID_GEO_INPUT):
        # Return a GeoDataTable from the valid input
        return prepare_geo_input(area)
    elif isinstance(area, gpd.GeoDataFrame):
        # Return a GeoDataTable from GeoDataFrame with changed CRS
        return GeoDataTable.from_geodataframe(area.to_crs(WGS84_CRS))
    elif isinstance(area, gpd.GeoSeries):
        # Create a GeoDataTable from GeoDataFrame with GeoSeries
        return GeoDataTable.from_geodataframe(gpd.GeoDataFrame(geometry=area, crs=WGS84_CRS))
    elif isinstance(area, Iterable):
        # Create a GeoSeries with a list of geometries
        return prepare_area_input_for_loader(gpd.GeoSeries(area, crs=WGS84_CRS))

    # Wrap a single geometry with a list
    return prepare_area_input_for_loader([area])


class Loader(abc.ABC):
    """Abstract class for loaders."""

    @abc.abstractmethod
    def load(
        self, *args: Any, **kwargs: Any
    ) -> Union[GeoDataTable, Iterable[GeoDataTable]]:  # pragma: no cover
        """
        Load data for a given area.

        Args:
            *args: Positional arguments dependating on a specific loader.
            **kwargs: Keyword arguments dependating on a specific loader.

        Returns:
            GeoDataTable or multiple GeoDataTables with the downloaded data.
        """
        raise NotImplementedError

    def _prepare_area_input(self, area: VALID_AREA_INPUT) -> GeoDataTable:
        return prepare_area_input_for_loader(area)
