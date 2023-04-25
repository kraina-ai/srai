"""
Adjacency neighbourhood.

This module contains the AdjacencyNeighbourhood class, that allows to get the neighbours of any
region based on its borders.
"""
from typing import Dict, Hashable, Set

import geopandas as gpd

from srai.constants import GEOMETRY_COLUMN
from srai.neighbourhoods import Neighbourhood


class AdjacencyNeighbourhood(Neighbourhood[Hashable]):
    """
    Adjacency Neighbourhood.

    This class allows to get the neighbours of any region based on common border. Additionally, a
    lookup table is implemented to accelerate repeated queries.

    By default, a lookup table will be populated lazily based on queries. A dedicated function
    `generate_neighbourhoods` allows for precalculation of all the neighbourhoods at once.
    """

    def __init__(self, regions_gdf: gpd.GeoDataFrame) -> None:
        """
        Init AdjacencyNeighbourhood.

        Args:
            regions_gdf (gpd.GeoDataFrame): regions for which a neighbourhood will be calculated.

        Raises:
            ValueError: If regions_gdf doesn't have geometry column.
        """
        if GEOMETRY_COLUMN not in regions_gdf.columns:
            raise ValueError("Regions must have a geometry column.")
        self.regions_gdf = regions_gdf
        self.lookup: Dict[Hashable, Set[Hashable]] = {}

    def generate_neighbourhoods(self) -> None:
        """Generate the lookup table for all regions."""
        for region_id in self.regions_gdf.index:
            if region_id not in self.lookup:
                self.lookup[region_id] = self._get_adjacent_neighbours(region_id)

    def get_neighbours(self, index: Hashable) -> Set[Hashable]:
        """
        Get the direct neighbours of any region using its index.

        Args:
            index (Hashable): Unique identifier of the region.

        Returns:
            Set[Hashable]: Indexes of the neighbours.
        """
        if self._index_incorrect(index):
            return set()

        if index not in self.lookup:
            self.lookup[index] = self._get_adjacent_neighbours(index)

        return self.lookup[index]

    def _get_adjacent_neighbours(self, index: Hashable) -> Set[Hashable]:
        """
        Get the direct neighbours of a region using `touches` [1] operator from the Shapely library.

        Args:
            index (Hashable): Unique identifier of the region.

        Returns:
            Set[Hashable]: Indexes of the neighbours.

        References:
            1. https://shapely.readthedocs.io/en/stable/reference/shapely.touches.html
        """
        current_region = self.regions_gdf.loc[index]
        neighbours = self.regions_gdf[
            self.regions_gdf.geometry.touches(current_region[GEOMETRY_COLUMN])
        ]
        return set(neighbours.index)

    def _index_incorrect(self, index: Hashable) -> bool:
        return index not in self.regions_gdf.index
