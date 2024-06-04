"""Utility H3 related functions."""

from collections.abc import Iterable
from typing import Literal, Union, overload

import geopandas as gpd
import h3
import numpy as np
import numpy.typing as npt
from h3ronpy.arrow import cells_to_string, grid_disk
from h3ronpy.arrow.vector import ContainmentMode, cells_to_wkb_polygons, wkb_to_cells
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS

__all__ = [
    "shapely_geometry_to_h3",
    "h3_to_geoseries",
    "h3_to_shapely_geometry",
    "get_local_ij_index",
    "ring_buffer_h3_indexes",
    "ring_buffer_geometry",
    "ring_buffer_h3_regions_gdf",
]


def shapely_geometry_to_h3(
    geometry: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
    h3_resolution: int,
    buffer: bool = True,
) -> list[str]:
    """
    Convert Shapely geometry to H3 indexes.

    Args:
        geometry (Union[BaseGeometry, Iterable[BaseGeometry], GeoSeries, GeoDataFrame]):
            Shapely geometry to be converted.
        h3_resolution (int): H3 resolution of the cells. See [1] for a full comparison.
        buffer (bool, optional): Whether to fully cover the geometries with
            H3 Cells (visible on the borders). Defaults to True.

    Returns:
        List[str]: List of H3 indexes that cover a given geometry.

    Raises:
        ValueError: If resolution is not between 0 and 15.

    References:
        1. https://h3geo.org/docs/core-library/restable/
    """
    if not (0 <= h3_resolution <= 15):
        raise ValueError(f"Resolution {h3_resolution} is not between 0 and 15.")

    wkb = []
    if isinstance(geometry, gpd.GeoSeries):
        wkb = geometry.to_wkb()
    elif isinstance(geometry, gpd.GeoDataFrame):
        wkb = geometry[GEOMETRY_COLUMN].to_wkb()
    elif isinstance(geometry, Iterable):
        wkb = [sub_geometry.wkb for sub_geometry in geometry]
    else:
        wkb = [geometry.wkb]

    containment_mode = ContainmentMode.Covers if buffer else ContainmentMode.ContainsCentroid
    h3_indexes = wkb_to_cells(
        wkb, resolution=h3_resolution, containment_mode=containment_mode, flatten=True
    ).unique()

    return [h3.int_to_str(h3_index) for h3_index in h3_indexes.tolist()]


# TODO: write tests (#322)
def h3_to_geoseries(h3_index: Union[int, str, Iterable[Union[int, str]]]) -> gpd.GeoSeries:
    """
    Convert H3 index to GeoPandas GeoSeries.

    Args:
        h3_index (Union[int, str, Iterable[Union[int, str]]]): H3 index (or list of indexes)
            to be converted.

    Returns:
        GeoSeries: Geometries as GeoSeries with default CRS applied.
    """
    if isinstance(h3_index, (str, int)):
        return h3_to_geoseries([h3_index])
    else:
        h3_int_indexes = (
            h3_cell if isinstance(h3_cell, int) else h3.str_to_int(h3_cell) for h3_cell in h3_index
        )
        return gpd.GeoSeries.from_wkb(cells_to_wkb_polygons(h3_int_indexes), crs=WGS84_CRS)


@overload
def h3_to_shapely_geometry(h3_index: Union[int, str]) -> Polygon: ...


@overload
def h3_to_shapely_geometry(h3_index: Iterable[Union[int, str]]) -> list[Polygon]: ...


# TODO: write tests (#322)
def h3_to_shapely_geometry(
    h3_index: Union[int, str, Iterable[Union[int, str]]],
) -> Union[Polygon, list[Polygon]]:
    """
    Convert H3 index to Shapely polygon.

    Args:
        h3_index (Union[int, str, Iterable[Union[int, str]]]): H3 index (or list of indexes)
            to be converted.

    Returns:
        Union[Polygon, List[Polygon]]: Converted polygon (or list of polygons).
    """
    if isinstance(h3_index, (str, int)):
        coords = h3.cell_to_boundary(h3_index, geo_json=True)
        return Polygon(coords)
    return h3_to_geoseries(h3_index).values.tolist()


@overload
def get_local_ij_index(origin_index: str, h3_index: str) -> tuple[int, int]: ...


@overload
def get_local_ij_index(
    origin_index: str, h3_index: list[str], return_as_numpy: Literal[False]
) -> list[tuple[int, int]]: ...


@overload
def get_local_ij_index(
    origin_index: str, h3_index: list[str], return_as_numpy: Literal[True]
) -> npt.NDArray[np.int8]: ...


# Last fallback needed as per documentation:
# https://mypy.readthedocs.io/en/stable/literal_types.html#literal-types
@overload
def get_local_ij_index(
    origin_index: str, h3_index: list[str], return_as_numpy: bool
) -> Union[list[tuple[int, int]], npt.NDArray[np.int8]]: ...


def get_local_ij_index(
    origin_index: str, h3_index: Union[str, list[str]], return_as_numpy: bool = False
) -> Union[tuple[int, int], list[tuple[int, int]], npt.NDArray[np.int8]]:
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


def ring_buffer_h3_indexes(h3_indexes: Iterable[Union[int, str]], distance: int) -> list[str]:
    """
    Buffer H3 indexes by a given number of k-rings.

    List of provided H3 indexes will be buffered by a given distance.

    Args:
        h3_indexes (Iterable[Union[int, str]]): List of H3 indexes to be buffered.
        distance (int): The k-ring buffer distance in H3 cells.

    Returns:
        List[str]: Buffered list of H3 cells containing both original and new cells.
    """
    assert all(
        h3.is_valid_cell(h3_cell) for h3_cell in h3_indexes
    ), "Not all values in h3_indexes are valid H3 cells."

    h3_int_indexes = (
        h3_cell if isinstance(h3_cell, int) else h3.str_to_int(h3_cell) for h3_cell in h3_indexes
    )
    buffered_h3s = set(cells_to_string(grid_disk(h3_int_indexes, distance, flatten=True)).tolist())
    return list(buffered_h3s)


def ring_buffer_geometry(
    geometry: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
    h3_resolution: int,
    distance: int,
) -> Union[gpd.GeoSeries, BaseGeometry]:
    """
    Buffer a Shapely geometry with H3 cells, and return the bounding geometry.

    If a GeoDataFrame is passed, the geometry column will be used and the return will be a GeoSeries

    Args:
        geometry (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
            The geometry to buffer.
        h3_resolution (int): The H3 resolution to use.
        distance (int): The k-ring buffer distance in H3 cells.

    Returns:
        Union[gpd.GeoSeries, BaseGeometry]: The buffered geometry.
    """
    if isinstance(geometry, gpd.GeoDataFrame):
        geometry = geometry[GEOMETRY_COLUMN]
        return geometry.apply(lambda x: ring_buffer_geometry(x, h3_resolution, distance))

    if isinstance(geometry, gpd.GeoSeries):
        return geometry.apply(lambda x: ring_buffer_geometry(x, h3_resolution, distance))

    if isinstance(geometry, Iterable):
        return gpd.GeoSeries([ring_buffer_geometry(x, h3_resolution, distance) for x in geometry])

    assert isinstance(geometry, BaseGeometry)
    h3s = shapely_geometry_to_h3(geometry, h3_resolution, buffer=True)
    # buffer all the h3
    buffered_h3s = ring_buffer_h3_indexes(h3s, distance=distance)
    # get the bounding geometry
    return h3_to_geoseries(buffered_h3s).unary_union


def ring_buffer_h3_regions_gdf(regions_gdf: gpd.GeoDataFrame, distance: int) -> gpd.GeoDataFrame:
    """
    Buffer H3 indexes by a given number of k-rings.

    List of provided H3 indexes will be buffered by a given distance.

    Args:
        regions_gdf (gpd.GeoDataFrame): GeoDataFrame with H3 regions from H3Regionalizer.
        distance (int): The k-ring buffer distance in H3 cells.

    Returns:
        gpd.GeoDataFrame: Buffered regions_gdf with new H3 cells added.
    """
    buffered_h3_indexes = ring_buffer_h3_indexes(h3_indexes=regions_gdf.index, distance=distance)
    buffered_gdf_h3 = gpd.GeoDataFrame(
        data={REGIONS_INDEX: buffered_h3_indexes},
        geometry=h3_to_geoseries(buffered_h3_indexes),
        crs=WGS84_CRS,
    ).set_index(REGIONS_INDEX)
    return buffered_gdf_h3
