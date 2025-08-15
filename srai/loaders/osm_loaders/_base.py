"""Base class for OSM loaders."""

import abc
from typing import Union

from srai.geodatatable import GeoDataTable
from srai.loaders import Loader
from srai.loaders._base import VALID_AREA_INPUT
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter


class OSMLoader(Loader, abc.ABC):
    """Abstract class for loaders."""

    OSM_FILTER_GROUP_COLUMN_NAME = "osm_group_"

    @abc.abstractmethod
    def load(
        self,
        area: VALID_AREA_INPUT,
        tags: Union[OsmTagsFilter, GroupedOsmTagsFilter],
    ) -> GeoDataTable:  # pragma: no cover
        """
        Load data for a given area.

        Args:
            area (VALID_AREA_INPUT): Geometry with the area of interest.
            tags (Union[OsmTagsFilter, GroupedOsmTagsFilter]): OSM tags filter.

        Returns:
            GeoDataTable: GeoDataTable with the downloaded data.
        """
        raise NotImplementedError
