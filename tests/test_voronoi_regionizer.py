"""Voronoi regionizer tests."""
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from srai.regionizers import VoronoiRegionizer
from srai.utils import _merge_disjointed_gdf_geometries

bbox = box(minx=-180, maxx=180, miny=-90, maxy=90)
bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")


def test_zero_seeds_value_error() -> None:
    """Test checks if zero seeds is disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame({"geometry": []}, index=[])
        VoronoiRegionizer(seeds=seeds_gdf)


def test_empty_gdf_value_error() -> None:
    """Test checks if empty GeoDataFrames are disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame()
        VoronoiRegionizer(seeds=seeds_gdf)


def test_duplicate_seeds_value_error() -> None:
    """Test checks if duplicate points are disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(0, 0), Point(-1, -1), Point(2, 2)]},
            index=[1, 2, 3, 4],
            crs="EPSG:4326",
        )
        VoronoiRegionizer(seeds=seeds_gdf)


def test_single_seed_region() -> None:
    """Test checks if single seed is disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)]},
            index=[1],
            crs="EPSG:4326",
        )
        VoronoiRegionizer(seeds=seeds_gdf)


def test_multiple_seeds_regions() -> None:
    """Test checks if regions are generated correctly."""
    seeds_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(0, 0),
                Point(90, 0),
                Point(180, 0),
                Point(-90, 0),
                Point(0, 90),
                Point(0, -90),
            ]
        },
        index=[1, 2, 3, 4, 5, 6],
        crs="EPSG:4326",
    )
    vr = VoronoiRegionizer(seeds=seeds_gdf)
    result_gdf = vr.transform(gdf=bbox_gdf)
    assert len(result_gdf.index) == 6
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(bbox).is_empty


def test_four_close_seed_region() -> None:
    """Test checks if four close seeds are properly evaluated."""
    seeds_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(17.014997869227177, 51.09919872601259),
                Point(16.935542631959215, 51.09380600286582),
                Point(16.900425, 51.1162552343),
                Point(16.932700, 51.166251),
            ]
        },
        index=[1, 2, 3, 4],
        crs="EPSG:4326",
    )
    vr = VoronoiRegionizer(seeds=seeds_gdf)
    result_gdf = vr.transform(gdf=bbox_gdf)
    assert len(result_gdf.index) == 4
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(bbox).is_empty


def test_default_parameter() -> None:
    """Test checks if regions are generated correctly with a default mask."""
    seeds_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(0, 0),
                Point(90, 0),
                Point(180, 0),
                Point(-90, 0),
                Point(0, 90),
                Point(0, -90),
            ]
        },
        index=[1, 2, 3, 4, 5, 6],
        crs="EPSG:4326",
    )
    vr = VoronoiRegionizer(seeds=seeds_gdf)
    result_gdf = vr.transform(gdf=None)
    assert len(result_gdf.index) == 6
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(bbox).is_empty
