"""Voronoi regionizer tests."""
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from srai.regionizers import VoronoiRegionizer
from srai.utils import _merge_disjointed_gdf_geometries

bbox = box(minx=-180, maxx=180, miny=-90, maxy=90)
bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]})


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
            {"geometry": [Point(0, 0), Point(0, 0)]}, index=[1, 2], crs="EPSG:4326"
        )
        VoronoiRegionizer(seeds=seeds_gdf)


def test_single_seed_region() -> None:
    """Test checks if single seed returns a single bbox."""
    seeds_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]},
        index=[1],
        crs="EPSG:4326",
    )
    vr = VoronoiRegionizer(seeds=seeds_gdf)
    result_gdf = vr.transform(gdf=bbox_gdf)
    assert len(result_gdf.index) == 1
    assert result_gdf.loc[1].geometry.equals(bbox)


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
