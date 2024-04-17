"""Tests for AdjacencyNeighbourhood."""

import geopandas as gpd
import pytest
from shapely import geometry

from srai.constants import WGS84_CRS
from srai.neighbourhoods import AdjacencyNeighbourhood


@pytest.fixture  # type: ignore
def no_geometry_gdf() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with no geometry."""
    return gpd.GeoDataFrame()


@pytest.fixture  # type: ignore
def empty_gdf() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame(geometry=[])


@pytest.fixture  # type: ignore
def squares_regions_fixture() -> gpd.GeoDataFrame:
    """
    Get 9 square regions.

    This GeoDataFrame represents 9 squares on a cartesian plane in a 3 by 3 grid pattern. Squares
    are adjacent by edges and by vertices. Squares are given compass directions. Visually it looks
    like this ("C" means "CENTER"):

    [["NW", "N", "NE"],
     [ "W", "C",  "E"],
     ["SW", "S", "SE"]]

    Returns:
        GeoDataFrame: A GeoDataFrame containing square regions.
    """
    regions = gpd.GeoDataFrame(
        geometry=[
            geometry.box(minx=0, maxx=1, miny=0, maxy=1),
            geometry.box(minx=1, maxx=2, miny=0, maxy=1),
            geometry.box(minx=2, maxx=3, miny=0, maxy=1),
            geometry.box(minx=0, maxx=1, miny=1, maxy=2),
            geometry.box(minx=1, maxx=2, miny=1, maxy=2),
            geometry.box(minx=2, maxx=3, miny=1, maxy=2),
            geometry.box(minx=0, maxx=1, miny=2, maxy=3),
            geometry.box(minx=1, maxx=2, miny=2, maxy=3),
            geometry.box(minx=2, maxx=3, miny=2, maxy=3),
        ],
        index=["SW", "S", "SE", "W", "CENTER", "E", "NW", "N", "NE"],  # compass directions
        crs=WGS84_CRS,
    )
    return regions


@pytest.fixture  # type: ignore
def rounded_regions_fixture() -> gpd.GeoDataFrame:
    """
    Get 9 small rounded square regions.

    This GeoDataFrame represents 9 small squares with buffer on a cartesian plane in a 3 by 3 grid
    pattern. Regions are adjacent by edges, but not by vertices. Regions are given compass
    directions. Visually it looks like this ("C" means "CENTER"):

    [["NW", "N", "NE"],
     [ "W", "C",  "E"],
     ["SW", "S", "SE"]]

    Returns:
        GeoDataFrame: A GeoDataFrame containing rounded regions.
    """
    regions = gpd.GeoDataFrame(
        geometry=[
            geometry.box(minx=0, maxx=0.5, miny=0, maxy=0.5).buffer(0.25),
            geometry.box(minx=1, maxx=1.5, miny=0, maxy=0.5).buffer(0.25),
            geometry.box(minx=2, maxx=2.5, miny=0, maxy=0.5).buffer(0.25),
            geometry.box(minx=0, maxx=0.5, miny=1, maxy=1.5).buffer(0.25),
            geometry.box(minx=1, maxx=1.5, miny=1, maxy=1.5).buffer(0.25),
            geometry.box(minx=2, maxx=2.5, miny=1, maxy=1.5).buffer(0.25),
            geometry.box(minx=0, maxx=0.5, miny=2, maxy=2.5).buffer(0.25),
            geometry.box(minx=1, maxx=1.5, miny=2, maxy=2.5).buffer(0.25),
            geometry.box(minx=2, maxx=2.5, miny=2, maxy=2.5).buffer(0.25),
        ],
        index=["SW", "S", "SE", "W", "CENTER", "E", "NW", "N", "NE"],  # compass directions
        crs=WGS84_CRS,
    )
    return regions


def test_no_geometry_gdf_attribute_error(no_geometry_gdf: gpd.GeoDataFrame) -> None:
    """Test checks if GeoDataFrames without geometry are disallowed."""
    with pytest.raises(ValueError):
        AdjacencyNeighbourhood(no_geometry_gdf)


def test_empty_gdf_empty_set(empty_gdf: gpd.GeoDataFrame) -> None:
    """Test checks if empty GeoDataFrames return empty neighbourhoods."""
    neighbourhood = AdjacencyNeighbourhood(empty_gdf)
    assert not neighbourhood.get_neighbours(1)


def test_empty_gdf_empty_set_include_center(empty_gdf: gpd.GeoDataFrame) -> None:
    """Test checks if empty GeoDataFrames return empty neighbourhoods."""
    neighbourhood = AdjacencyNeighbourhood(empty_gdf, include_center=True)
    assert not neighbourhood.get_neighbours(1)


def test_lazy_loading_empty_set(squares_regions_fixture: gpd.GeoDataFrame) -> None:
    """Test checks if lookup table is empty after init."""
    neighbourhood = AdjacencyNeighbourhood(squares_regions_fixture)
    assert not neighbourhood.lookup


def test_lazy_loading_empty_set_include_center(squares_regions_fixture: gpd.GeoDataFrame) -> None:
    """Test checks if lookup table is empty after init."""
    neighbourhood = AdjacencyNeighbourhood(squares_regions_fixture, include_center=True)
    assert not neighbourhood.lookup


def test_adjacency_lazy_loading(rounded_regions_fixture: gpd.GeoDataFrame) -> None:
    """Test checks if lookup table is lazily populated."""
    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture)
    neighbours = neighbourhood.get_neighbours("SW")
    assert neighbours == {"W", "S"}
    assert neighbourhood.lookup == {
        "SW": {"W", "S"},
    }


def test_adjacency_lazy_loading_include_center(rounded_regions_fixture: gpd.GeoDataFrame) -> None:
    """Test checks if lookup table is lazily populated."""
    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture, include_center=True)
    neighbours = neighbourhood.get_neighbours("SW")
    assert neighbours == {"W", "S", "SW"}
    assert neighbourhood.lookup == {
        "SW": {"W", "S", "SW"},
    }


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


@pytest.mark.parametrize(  # type: ignore
    "index,distance,expected,expected_include_center",
    [
        ("SW", -2, set(), set()),
        ("SW", -1, set(), set()),
        ("SW", 0, set(), {"SW"}),
        ("SW", 1, {"S", "W"}, {"S", "W"}),
        ("SW", 2, {"CENTER", "SE", "NW"}, {"CENTER", "SE", "NW"}),
        ("SW", 3, {"N", "E"}, {"N", "E"}),
        ("SW", 4, {"NE"}, {"NE"}),
        ("SW", 5, set(), set()),
        ("CENTER", 0, set(), {"CENTER"}),
        ("CENTER", 1, {"N", "E", "S", "W"}, {"N", "E", "S", "W"}),
        ("CENTER", 2, {"NW", "NE", "SW", "SE"}, {"NW", "NE", "SW", "SE"}),
        ("CENTER", 3, set(), set()),
        ("N", 0, set(), {"N"}),
        ("N", 1, {"NW", "NE", "CENTER"}, {"NW", "NE", "CENTER"}),
        ("N", 2, {"E", "S", "W"}, {"E", "S", "W"}),
        ("N", 3, {"SE", "SW"}, {"SE", "SW"}),
        ("N", 4, set(), set()),
    ],
)
def test_adjacency_lazy_loading_at_distance(
    index: str,
    distance: int,
    expected: set[str],
    expected_include_center: set[str],
    rounded_regions_fixture: gpd.GeoDataFrame,
) -> None:
    """Test checks `get_neighbours_at_distance` function with rounded regions."""
    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture)
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance)
    assert neighbours == expected
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance, include_center=True)
    assert neighbours == expected_include_center

    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture, include_center=True)
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance)
    assert neighbours == expected_include_center
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance, include_center=False)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "index,distance,expected,expected_include_center",
    [
        ("SW", -2, set(), set()),
        ("SW", -1, set(), set()),
        ("SW", 0, set(), {"SW"}),
        ("SW", 1, {"S", "W"}, {"SW", "S", "W"}),
        ("SW", 2, {"S", "W", "CENTER", "SE", "NW"}, {"SW", "S", "W", "CENTER", "SE", "NW"}),
        (
            "SW",
            3,
            {"S", "W", "CENTER", "SE", "NW", "N", "E"},
            {"SW", "S", "W", "CENTER", "SE", "NW", "N", "E"},
        ),
        (
            "SW",
            4,
            {"S", "W", "CENTER", "SE", "NW", "N", "E", "NE"},
            {"SW", "S", "W", "CENTER", "SE", "NW", "N", "E", "NE"},
        ),
        (
            "SW",
            5,
            {"S", "W", "CENTER", "SE", "NW", "N", "E", "NE"},
            {"SW", "S", "W", "CENTER", "SE", "NW", "N", "E", "NE"},
        ),
        ("CENTER", 0, set(), {"CENTER"}),
        ("CENTER", 1, {"N", "E", "S", "W"}, {"CENTER", "N", "E", "S", "W"}),
        (
            "CENTER",
            2,
            {"N", "E", "S", "W", "NW", "NE", "SW", "SE"},
            {"CENTER", "N", "E", "S", "W", "NW", "NE", "SW", "SE"},
        ),
        (
            "CENTER",
            3,
            {"N", "E", "S", "W", "NW", "NE", "SW", "SE"},
            {"CENTER", "N", "E", "S", "W", "NW", "NE", "SW", "SE"},
        ),
        ("N", 0, set(), {"N"}),
        ("N", 1, {"NW", "NE", "CENTER"}, {"N", "NW", "NE", "CENTER"}),
        ("N", 2, {"NW", "NE", "CENTER", "E", "S", "W"}, {"N", "NW", "NE", "CENTER", "E", "S", "W"}),
        (
            "N",
            3,
            {"NW", "NE", "CENTER", "E", "S", "W", "SE", "SW"},
            {"N", "NW", "NE", "CENTER", "E", "S", "W", "SE", "SW"},
        ),
        (
            "N",
            4,
            {"NW", "NE", "CENTER", "E", "S", "W", "SE", "SW"},
            {"N", "NW", "NE", "CENTER", "E", "S", "W", "SE", "SW"},
        ),
    ],
)
def test_adjacency_lazy_loading_up_to_distance(
    index: str,
    distance: int,
    expected: set[str],
    expected_include_center: set[str],
    rounded_regions_fixture: gpd.GeoDataFrame,
) -> None:
    """Test checks `get_neighbours_up_to_distance` function with rounded regions."""
    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture)
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance)
    assert neighbours == expected
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance, include_center=True)
    assert neighbours == expected_include_center

    neighbourhood = AdjacencyNeighbourhood(rounded_regions_fixture, include_center=True)
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance)
    assert neighbours == expected_include_center
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance, include_center=False)
    assert neighbours == expected
