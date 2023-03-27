"""Voronoi regionizer tests."""
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.regionizers import VoronoiRegionizer
from srai.regionizers._spherical_voronoi import generate_voronoi_regions
from srai.utils import merge_disjointed_gdf_geometries


def test_empty_gdf_attribute_error(gdf_empty: gpd.GeoDataFrame) -> None:
    """Test checks if empty GeoDataFrames are disallowed."""
    with pytest.raises(AttributeError):
        VoronoiRegionizer(seeds=gdf_empty)


def test_no_crs_gdf_value_error(
    gdf_earth_poles: gpd.GeoDataFrame, gdf_no_crs: gpd.GeoDataFrame
) -> None:
    """Test checks if GeoDataFrames without crs are disallowed."""
    with pytest.raises(ValueError):
        vr = VoronoiRegionizer(seeds=gdf_earth_poles)
        vr.transform(gdf=gdf_no_crs)


def test_duplicate_seeds_value_error() -> None:
    """Test checks if duplicate points are disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame(
            {GEOMETRY_COLUMN: [Point(0, 0), Point(0, 0), Point(-1, -1), Point(2, 2)]},
            index=[1, 2, 3, 4],
            crs=WGS84_CRS,
        )
        VoronoiRegionizer(seeds=seeds_gdf)


def test_single_seed_region() -> None:
    """Test checks if single seed is disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame(
            {GEOMETRY_COLUMN: [Point(0, 0)]},
            index=[1],
            crs=WGS84_CRS,
        )
        VoronoiRegionizer(seeds=seeds_gdf)


def test_single_seed_algorithm_error() -> None:
    """Test checks if single seed is disallowed."""
    with pytest.raises(ValueError):
        generate_voronoi_regions(seeds=[Point(0, 0)], max_meters_between_points=10_000)


def test_multiple_seeds_regions(
    gdf_earth_poles: gpd.GeoDataFrame, gdf_earth_bbox: gpd.GeoDataFrame, earth_bbox: Polygon
) -> None:
    """Test checks if regions are generated correctly."""
    vr = VoronoiRegionizer(seeds=gdf_earth_poles)
    result_gdf = vr.transform(gdf=gdf_earth_bbox)
    assert len(result_gdf.index) == 6
    assert merge_disjointed_gdf_geometries(result_gdf).difference(earth_bbox).is_empty


def test_big_number_of_seeds_regions(gdf_earth_bbox: gpd.GeoDataFrame, earth_bbox: Polygon) -> None:
    """Test checks if regions are generated correctly and multiprocessing working."""
    number_of_points = 1000
    minx, miny, maxx, maxy = earth_bbox.bounds
    rng = np.random.default_rng()
    randx = rng.uniform(minx, maxx, number_of_points)
    randy = rng.uniform(miny, maxy, number_of_points)
    coords = np.vstack((randx, randy)).T

    pts = [p for p in list(map(Point, coords)) if p.covered_by(earth_bbox)]

    random_points_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: pts},
        index=list(range(len(pts))),
        crs=WGS84_CRS,
    )

    vr = VoronoiRegionizer(seeds=random_points_gdf)
    result_gdf = vr.transform(gdf=gdf_earth_bbox)
    assert len(result_gdf.index) == number_of_points
    assert merge_disjointed_gdf_geometries(result_gdf).difference(earth_bbox).is_empty


def test_four_close_seed_region(gdf_earth_bbox: gpd.GeoDataFrame, earth_bbox: Polygon) -> None:
    """Test checks if four close seeds are properly evaluated."""
    seeds_gdf = gpd.GeoDataFrame(
        {
            GEOMETRY_COLUMN: [
                Point(17.014997869227177, 51.09919872601259),
                Point(16.935542631959215, 51.09380600286582),
                Point(16.900425, 51.1162552343),
                Point(16.932700, 51.166251),
            ]
        },
        index=[1, 2, 3, 4],
        crs=WGS84_CRS,
    )
    vr = VoronoiRegionizer(seeds=seeds_gdf)
    result_gdf = vr.transform(gdf=gdf_earth_bbox)
    assert len(result_gdf.index) == 4
    assert merge_disjointed_gdf_geometries(result_gdf).difference(earth_bbox).is_empty


def test_default_parameter(gdf_earth_poles: gpd.GeoDataFrame, earth_bbox: Polygon) -> None:
    """Test checks if regions are generated correctly with a default mask."""
    vr = VoronoiRegionizer(seeds=gdf_earth_poles)
    result_gdf = vr.transform(gdf=None)
    assert len(result_gdf.index) == 6
    assert merge_disjointed_gdf_geometries(result_gdf).difference(earth_bbox).is_empty


@pytest.mark.parametrize(  # type: ignore
    "gdf_fixture",
    [
        "gdf_multipolygon",
        "gdf_poland",
        "gdf_earth_bbox",
    ],
)
def test_clipping_parameter(
    gdf_fixture: str,
    request: Any,
) -> None:
    """Test checks if regions are clipped correctly with a provided mask."""
    gdf: gpd.GeoDataFrame = request.getfixturevalue(gdf_fixture)
    gdf_earth_poles: gpd.GeoDataFrame = request.getfixturevalue("gdf_earth_poles")
    vr = VoronoiRegionizer(seeds=gdf_earth_poles)
    result_gdf = vr.transform(gdf=gdf)
    assert merge_disjointed_gdf_geometries(result_gdf).difference(gdf.iloc[0].geometry).is_empty
