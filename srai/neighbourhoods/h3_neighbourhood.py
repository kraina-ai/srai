"""
H3 neighbourhood.

This module contains the H3Neighbourhood class, that allows to get the neighbours of an H3 region.
"""

from typing import Generic, Optional, TypeVar

import geopandas as gpd
import h3.api.basic_int as h3int
import h3.api.basic_str as h3str

from srai.neighbourhoods import Neighbourhood

H3IndexGenericType = TypeVar("H3IndexGenericType", int, str)


class H3Neighbourhood(Neighbourhood[H3IndexGenericType], Generic[H3IndexGenericType]):
    """
    H3 Neighbourhood.

    This class allows to get the neighbours of an H3 region.
    """

    def __init__(
        self, regions_gdf: Optional[gpd.GeoDataFrame] = None, include_center: bool = False
    ) -> None:
        """
        Initializes the H3Neighbourhood.

        If a regions GeoDataFrame is provided, only the neighbours
        that are in the regions GeoDataFrame will be returned by the methods of this instance.
        NOTICE: If a region is a part of the k-th ring of a region
            and is included in the GeoDataFrame, it will be returned
            by get_neighbours_at_distance method with distance k
            even when there is no path of length k between the two regions.

        Args:
            regions_gdf (Optional[gpd.GeoDataFrame], optional): The regions that are being analyzed.
                The H3Neighbourhood will only look for neighbours among these regions.
                Defaults to None.
            include_center (bool): Whether to include the region itself in the neighbours.
            This is the default value used for all the methods of the class,
            unless overridden in the function call.
        """
        super().__init__(include_center)
        self._available_indices: Optional[set[H3IndexGenericType]] = None
        if regions_gdf is not None:
            self._available_indices = set(regions_gdf.index)

    def get_neighbours(
        self, index: H3IndexGenericType, include_center: Optional[bool] = None
    ) -> set[H3IndexGenericType]:
        """
        Get the direct neighbours of an H3 region using its index.

        Args:
            index (H3IndexType): H3 index of the region.
            include_center (Optional[bool]): Whether to include the region itself in the neighbours.
            If None, the value set in __init__ is used. Defaults to None.

        Returns:
            set[H3IndexType]: Indexes of the neighbours.
        """
        return self.get_neighbours_up_to_distance(index, 1, include_center)

    def get_neighbours_up_to_distance(
        self,
        index: H3IndexGenericType,
        distance: int,
        include_center: Optional[bool] = None,
        unchecked: bool = False,
    ) -> set[H3IndexGenericType]:
        """
        Get the neighbours of an H3 region up to a certain distance.

        Args:
            index (H3IndexType): H3 index of the region.
            distance (int): Distance to the neighbours.
            include_center (Optional[bool]): Whether to include the region itself in the neighbours.
                If None, the value set in __init__ is used. Defaults to None.
            unchecked (bool): Whether to check if the neighbours are in the available indices.

        Returns:
            set[H3IndexType]: Indexes of the neighbours up to the given distance.
        """
        if self._distance_incorrect(distance):
            return set()

        neighbours: set[H3IndexGenericType]

        if isinstance(index, str):
            neighbours = set(h3str.grid_disk(index, distance))
        else:
            neighbours = set(h3int.grid_disk(index, distance))

        neighbours = self._handle_center(
            index, distance, neighbours, at_distance=False, include_center_override=include_center
        )
        if unchecked:
            return neighbours
        return self._select_available(neighbours)

    def get_neighbours_at_distance(
        self, index: H3IndexGenericType, distance: int, include_center: Optional[bool] = None
    ) -> set[H3IndexGenericType]:
        """
        Get the neighbours of an H3 region at a certain distance.

        Args:
            index (H3IndexType): H3 index of the region.
            distance (int): Distance to the neighbours.
            include_center (Optional[bool]): Whether to include the region itself in the neighbours.
            If None, the value set in __init__ is used. Defaults to None.

        Returns:
            set[H3IndexType]: Indexes of the neighbours at the given distance.
        """
        if self._distance_incorrect(distance):
            return set()

        neighbours: set[H3IndexGenericType]

        if isinstance(index, str):
            neighbours = set(h3str.grid_ring(index, distance))
        else:
            neighbours = set(h3int.grid_ring(index, distance))

        neighbours = self._handle_center(
            index, distance, neighbours, at_distance=True, include_center_override=include_center
        )
        return self._select_available(neighbours)

    def _select_available(self, indices: set[H3IndexGenericType]) -> set[H3IndexGenericType]:
        if self._available_indices is None:
            return indices
        return indices.intersection(self._available_indices)

    def _distance_incorrect(self, distance: int) -> bool:
        return distance < 0
