"""Voronoi regionizer tests."""
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import Point

from srai.regionizers import VoronoiRegionizer
from srai.utils import _merge_disjointed_gdf_geometries


def test_empty_gdf_value_error(gdf_empty) -> None:  # type: ignore
    """Test checks if empty GeoDataFrames are disallowed."""
    with pytest.raises(ValueError):
        VoronoiRegionizer(seeds=gdf_empty)


def test_no_crs_gdf_value_error(gdf_earth_poles, gdf_no_crs) -> None:  # type: ignore
    """Test checks if GeoDataFrames without crs are disallowed."""
    with pytest.raises(ValueError):
        vr = VoronoiRegionizer(seeds=gdf_earth_poles)
        vr.transform(gdf=gdf_no_crs)


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


def test_multiple_seeds_regions(  # type: ignore
    gdf_earth_poles, gdf_earth_bbox, earth_bbox
) -> None:
    """Test checks if regions are generated correctly."""
    vr = VoronoiRegionizer(seeds=gdf_earth_poles)
    result_gdf = vr.transform(gdf=gdf_earth_bbox)
    assert len(result_gdf.index) == 6
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(earth_bbox).is_empty


def test_four_close_seed_region(gdf_earth_bbox, earth_bbox) -> None:  # type: ignore
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
    result_gdf = vr.transform(gdf=gdf_earth_bbox)
    assert len(result_gdf.index) == 4
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(earth_bbox).is_empty


def test_default_parameter(gdf_earth_poles, earth_bbox) -> None:  # type: ignore
    """Test checks if regions are generated correctly with a default mask."""
    vr = VoronoiRegionizer(seeds=gdf_earth_poles)
    result_gdf = vr.transform(gdf=None)
    assert len(result_gdf.index) == 6
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(earth_bbox).is_empty


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
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(gdf.iloc[0].geometry).is_empty
