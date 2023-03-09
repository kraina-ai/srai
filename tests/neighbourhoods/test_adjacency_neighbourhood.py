from typing import Set

import geopandas as gpd
import pytest
from shapely import geometry

from srai.neighbourhoods import AdjacencyNeighbourhood
from srai.utils.constants import WGS84_CRS


@pytest.fixture  # type: ignore
def no_geometry_gdf() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame()


@pytest.fixture  # type: ignore
def empty_gdf() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with no geometry."""
    return gpd.GeoDataFrame(geometry=[])


@pytest.fixture  # type: ignore
def squares_regions_fixture() -> gpd.GeoDataFrame:
    """
    This GeoDataFrame represents 9 squares on a cartesian plane in a 3 by 3 grid pattern. Squares
    are adjacent by edges and by vertexes. Squares are given compass directions. Visually it looks
    like this ("C" means "CENTER"):

    [["NW", "N", "NE"],
     [ "W", "C",  "E"],
     ["SW", "S", "SE"]]

    Returns:
        GeoDataFrame: A GeoDataFrame containing square regions.
    """
    regions = gpd.GeoDataFrame(
        geometry=[
            geometry.box(minx=0, miny=0, maxx=1, maxy=1),
            geometry.box(minx=1, miny=0, maxx=2, maxy=1),
            geometry.box(minx=2, miny=0, maxx=3, maxy=1),
            geometry.box(minx=0, miny=1, maxx=1, maxy=2),
            geometry.box(minx=1, miny=1, maxx=2, maxy=2),
            geometry.box(minx=2, miny=1, maxx=3, maxy=2),
            geometry.box(minx=0, miny=2, maxx=1, maxy=3),
            geometry.box(minx=1, miny=2, maxx=2, maxy=3),
            geometry.box(minx=2, miny=2, maxx=3, maxy=3),
        ],
        index=["SW", "S", "SE", "W", "CENTER", "E", "NW", "N", "NE"],  # compass directions
        crs=WGS84_CRS,
    )
    return regions


@pytest.fixture  # type: ignore
def rounded_regions_fixture() -> gpd.GeoDataFrame:
    """
    This GeoDataFrame represents 9 small squares with buffer on a cartesian plane in a 3 by 3 grid
    pattern. Regions are adjacent by edges, but not by vertexes. Regions are given compass
    directions. Visually it looks like this ("C" means "CENTER"):

    [["NW", "N", "NE"],
     [ "W", "C",  "E"],
     ["SW", "S", "SE"]]

    Returns:
        GeoDataFrame: A GeoDataFrame containing rounded regions.
    """
    regions = gpd.GeoDataFrame(
        geometry=[
            geometry.box(minx=0, miny=0, maxx=0.5, maxy=0.5).buffer(0.25),
            geometry.box(minx=1, miny=0, maxx=1.5, maxy=0.5).buffer(0.25),
            geometry.box(minx=2, miny=0, maxx=2.5, maxy=0.5).buffer(0.25),
            geometry.box(minx=0, miny=1, maxx=0.5, maxy=1.5).buffer(0.25),
            geometry.box(minx=1, miny=1, maxx=1.5, maxy=1.5).buffer(0.25),
            geometry.box(minx=2, miny=1, maxx=2.5, maxy=1.5).buffer(0.25),
            geometry.box(minx=0, miny=2, maxx=0.5, maxy=2.5).buffer(0.25),
            geometry.box(minx=1, miny=2, maxx=1.5, maxy=2.5).buffer(0.25),
            geometry.box(minx=2, miny=2, maxx=2.5, maxy=2.5).buffer(0.25),
        ],
        index=["SW", "S", "SE", "W", "CENTER", "E", "NW", "N", "NE"],  # compass directions
        crs=WGS84_CRS,
    )
    return regions


def test_no_geometry_gdf_attribute_error(no_geometry_gdf: gpd.GeoDataFrame) -> None:
    """Test checks if GeoDataFrames without geometry are disallowed."""
    with pytest.raises(AttributeError):
        AdjacencyNeighbourhood(no_geometry_gdf)


def test_empty_gdf_empty_set(empty_gdf: gpd.GeoDataFrame) -> None:
    """Test checks if empty GeoDataFrames return empty neighbourhoods."""
    neighbourhood = AdjacencyNeighbourhood(empty_gdf)
    assert neighbourhood.get_neighbours(1) == set()


def test_lazy_loading_empty_set(squares_regions_fixture: gpd.GeoDataFrame) -> None:
    """Test checks if lookup table is empty after init."""
    neighbourhood = AdjacencyNeighbourhood(squares_regions_fixture)
    assert neighbourhood.lookup == dict()


def test_generate_all_neighbourhoods_rounded_regions(
    rounded_regions_fixture: gpd.GeoDataFrame,
) -> None:
    """Test checks `generate_neighbourhoods` function with rounded regions."""
    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture)
    neighbourhood.generate_neighbourhoods()
    assert neighbourhood.lookup == {
        "SW": {"W", "S"},
        "S": {"SW", "CENTER", "SE"},
        "SE": {"E", "S"},
        "W": {"SW", "CENTER", "NW"},
        "CENTER": {"N", "W", "E", "S"},
        "E": {"CENTER", "NE", "SE"},
        "NW": {"N", "W"},
        "N": {"CENTER", "NE", "NW"},
        "NE": {"N", "E"},
    }


def test_generate_all_neighbourhoods_squares_regions(
    squares_regions_fixture: gpd.GeoDataFrame,
) -> None:
    """Test checks `generate_neighbourhoods` function with square regions."""
    neighbourhood = AdjacencyNeighbourhood(squares_regions_fixture)
    neighbourhood.generate_neighbourhoods()
    assert neighbourhood.lookup == {
        "SW": {"W", "S", "CENTER"},
        "S": {"SW", "W", "CENTER", "SE", "E"},
        "SE": {"E", "S", "CENTER"},
        "W": {"N", "SW", "S", "CENTER", "NW"},
        "CENTER": {"SW", "N", "W", "S", "SE", "E", "NW", "NE"},
        "E": {"N", "S", "CENTER", "SE", "NE"},
        "NW": {"W", "N", "CENTER"},
        "N": {"W", "CENTER", "E", "NW", "NE"},
        "NE": {"E", "N", "CENTER"},
    }


def test_adjacency_lazy_loading(rounded_regions_fixture: gpd.GeoDataFrame) -> None:
    """Test checks if lookup table is lazily populated."""
    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture)
    neighbours = neighbourhood.get_neighbours("SW")
    assert neighbours == {"W", "S"}
    assert neighbourhood.lookup == {
        "SW": {"W", "S"},
    }


@pytest.mark.parametrize(  # type: ignore
    "index, distance, neighbours_expected",
    [
        ("SW", -2, set()),
        ("SW", -1, set()),
        ("SW", 0, set()),
        ("SW", 1, {"S", "W"}),
        ("SW", 2, {"CENTER", "SE", "NW"}),
        ("SW", 3, {"N", "E"}),
        ("SW", 4, {"NE"}),
        ("SW", 5, set()),
        ("CENTER", 1, {"N", "E", "S", "W"}),
        ("CENTER", 2, {"NW", "NE", "SW", "SE"}),
        ("CENTER", 3, set()),
        ("N", 1, {"NW", "NE", "CENTER"}),
        ("N", 2, {"E", "S", "W"}),
        ("N", 3, {"SE", "SW"}),
        ("N", 4, set()),
    ],
)
def test_adjacency_lazy_loading_at_distance(
    index: str,
    distance: int,
    neighbours_expected: Set[str],
    rounded_regions_fixture: gpd.GeoDataFrame,
) -> None:
    """Test checks `get_neighbours_at_distance` function with rounded regions."""
    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture)
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance)
    assert neighbours == neighbours_expected


@pytest.mark.parametrize(  # type: ignore
    "index, distance, neighbours_expected",
    [
        ("SW", -2, set()),
        ("SW", -1, set()),
        ("SW", 0, set()),
        ("SW", 1, {"S", "W"}),
        ("SW", 2, {"S", "W", "CENTER", "SE", "NW"}),
        ("SW", 3, {"S", "W", "CENTER", "SE", "NW", "N", "E"}),
        ("SW", 4, {"S", "W", "CENTER", "SE", "NW", "N", "E", "NE"}),
        ("SW", 5, {"S", "W", "CENTER", "SE", "NW", "N", "E", "NE"}),
        ("CENTER", 1, {"N", "E", "S", "W"}),
        ("CENTER", 2, {"N", "E", "S", "W", "NW", "NE", "SW", "SE"}),
        ("CENTER", 3, {"N", "E", "S", "W", "NW", "NE", "SW", "SE"}),
        ("N", 1, {"NW", "NE", "CENTER"}),
        ("N", 2, {"NW", "NE", "CENTER", "E", "S", "W"}),
        ("N", 3, {"NW", "NE", "CENTER", "E", "S", "W", "SE", "SW"}),
        ("N", 4, {"NW", "NE", "CENTER", "E", "S", "W", "SE", "SW"}),
    ],
)
def test_adjacency_lazy_loading_up_to_distance(
    index: str,
    distance: int,
    neighbours_expected: Set[str],
    rounded_regions_fixture: gpd.GeoDataFrame,
) -> None:
    """Test checks `get_neighbours_up_to_distance` function with rounded regions."""
    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture)
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance)
    assert neighbours == neighbours_expected
