"""Utility H3 related functions."""

from collections.abc import Iterable
from typing import List, Literal, Tuple, Union, overload

import geopandas as gpd
import h3
import numpy as np
import numpy.typing as npt
from h3ronpy.arrow.vector import cells_to_wkb_polygons, wkb_to_cells
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from srai.constants import WGS84_CRS


def shapely_geometry_to_h3(
    geometry: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries],
    h3_resolution: int,
    buffer: bool,
) -> List[str]:
    """TODO."""
    wkb = []
    if isinstance(geometry, Iterable):
        wkb = [sub_geometry.wkb for sub_geometry in geometry]
    elif isinstance(geometry, gpd.GeoSeries):
        wkb = geometry.to_wkb()
    else:
        wkb = [geometry.wkb]

    h3_indexes = wkb_to_cells(
        wkb, resolution=h3_resolution, all_intersecting=buffer, flatten=True
    ).unique()

    return [h3.int_to_str(h3_index) for h3_index in h3_indexes.tolist()]


def h3_to_geoseries(h3_index: Union[int, str, Iterable[Union[int, str]]]) -> gpd.GeoSeries:
    """TODO."""
    if isinstance(h3_index, (str, int)):
        return h3_to_geoseries([h3_index])
    else:
        h3_int_indexes = (
            h3_cell if isinstance(h3_cell, int) else h3.str_to_int(h3_cell) for h3_cell in h3_index
        )
        return gpd.GeoSeries.from_wkb(cells_to_wkb_polygons(h3_int_indexes), crs=WGS84_CRS)


@overload
def h3_to_shapely_geometry(h3_index: Union[int, str]) -> Polygon:
    ...


@overload
def h3_to_shapely_geometry(h3_index: Iterable[Union[int, str]]) -> List[Polygon]:
    ...


def h3_to_shapely_geometry(
    h3_index: Union[int, str, Iterable[Union[int, str]]]
) -> Union[Polygon, List[Polygon]]:
    """TODO."""
    if isinstance(h3_index, (str, int)):
        coords = h3.cell_to_boundary(h3_index, geo_json=True)
        return Polygon(coords)
    else:
        return h3_to_geoseries(h3_index).values.tolist()


@overload
def get_local_ij_index(origin_index: str, h3_index: str) -> Tuple[int, int]:
    ...


@overload
def get_local_ij_index(
    origin_index: str, h3_index: List[str], return_as_numpy: Literal[False]
) -> List[Tuple[int, int]]:
    ...


@overload
def get_local_ij_index(
    origin_index: str, h3_index: List[str], return_as_numpy: Literal[True]
) -> npt.NDArray[np.int8]:
    ...


def get_local_ij_index(
    origin_index: str, h3_index: Union[str, List[str]], return_as_numpy: bool = False
) -> Union[Tuple[int, int], List[Tuple[int, int]], npt.NDArray[np.int8]]:
    """
    Calculate the local H3 ij index based on provided origin index.

    Wraps H3's cell_to_local_ij function and centers returned coordinates
    around provided origin cell.

    Args:
        origin_index (str): H3 index of the origin region.
        h3_index (Union[str, List[str]]): H3 index of the second region or list of regions.
        return_as_numpy (bool, optional): Flag whether to return calculated indexes as a Numpy array
            or a list of tuples.

    Returns:
        Union[Tuple[int, int], List[Tuple[int, int]], npt.NDArray[np.int8]]: The local ij index of
            the second region (or regions) with respect to the first one.
    """
    origin_coords = h3.cell_to_local_ij(origin_index, origin_index)
    if isinstance(h3_index, str):
        ijs = h3.cell_to_local_ij(origin_index, h3_index)
        return (origin_coords[0] - ijs[0], origin_coords[1] - ijs[1])
    ijs = np.array([h3.cell_to_local_ij(origin_index, h3_cell) for h3_cell in h3_index])
    local_ijs = np.array(origin_coords) - ijs

    if not return_as_numpy:
        local_ijs = [(coords[0], coords[1]) for coords in local_ijs]

    return local_ijs
