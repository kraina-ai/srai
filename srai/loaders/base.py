"""Base class for loaders."""

import abc
from pathlib import Path
from typing import Union

import geopandas as gpd


class BaseLoader(abc.ABC):
    """Base abstract class for loaders."""

    @abc.abstractmethod
    def load(self, area: Union[gpd.GeoDataFrame, Path]) -> gpd.GeoDataFrame:
        """
        Load data for a given area.

        Args:
            area (gdf.GeoDataFrame | Path): GeoDataFrame with the area of interest or a path
                to a file with a geometry.

        Returns:
            GeoDataFrame with the downloaded data.
        """
        raise NotImplementedError
