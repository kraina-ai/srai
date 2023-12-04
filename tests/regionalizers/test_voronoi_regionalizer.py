"""Voronoi regionalizer tests."""

from multiprocessing import cpu_count
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pytest
from pymap3d import Ellipsoid
from shapely.geometry import Point, Polygon

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.geometry import merge_disjointed_gdf_geometries
from srai.regionalizers import VoronoiRegionalizer
from srai.regionalizers._spherical_voronoi import (
    _map_from_geocentric,
    _parse_multiprocessing_activation_threshold,
    _parse_num_of_multiprocessing_workers,
    generate_voronoi_regions,
)


def get_random_points(number_of_points: int, seed: int) -> list[Point]:
    """Get random points within WGS84 bounds sampled on a sphere."""
    vec = np.random.default_rng(seed=seed).standard_normal((3, number_of_points))
    vec /= np.linalg.norm(vec, axis=0)
    xi, yi, zi = vec

    unit_sphere_ellipsoid = Ellipsoid(
        semimajor_axis=1, semiminor_axis=1, name="Unit Sphere", model="Unit"
    )

    return [
        Point(_map_from_geocentric(_x, _y, _z, unit_sphere_ellipsoid))
        for _x, _y, _z in zip(xi, yi, zi)
    ]


def list_to_geodataframe(points: list[Point]) -> gpd.GeoDataFrame:
    """Wrap points into GeoDataFrame."""
    return gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: points},
        crs=WGS84_CRS,
    )


def check_if_disjoint(gdf: gpd.GeoDataFrame) -> bool:
    """Check if provided geometries are disjoint."""
    duplicated_regions = (
        gdf.sjoin(gdf, predicate="overlaps").index.value_counts().loc[lambda x: x > 1]
    )
    duplicated_seeds_ids = duplicated_regions.index.to_list()
    return len(duplicated_seeds_ids) == 0


def check_if_seeds_match_regions(seeds: gpd.GeoDataFrame, regions: gpd.GeoDataFrame) -> bool:
    """Check if order of seeds and regions is the same."""
    seeds_index_column_name = f"{seeds.index.name or 'index'}_seeds"
    joined_data = regions.reset_index().sjoin(seeds, rsuffix="seeds")

    if len(joined_data) != len(regions):
        return False

    return cast(bool, (joined_data[REGIONS_INDEX] == joined_data[seeds_index_column_name]).all())


def test_empty_gdf_attribute_error(gdf_empty: gpd.GeoDataFrame) -> None:
    """Test checks if empty GeoDataFrames are disallowed."""
    with pytest.raises(AttributeError):
        VoronoiRegionalizer(seeds=gdf_empty)


def test_no_crs_gdf_value_error(
    gdf_earth_poles: gpd.GeoDataFrame, gdf_no_crs: gpd.GeoDataFrame
) -> None:
    """Test checks if GeoDataFrames without crs are disallowed."""
    with pytest.raises(ValueError):
        vr = VoronoiRegionalizer(seeds=gdf_earth_poles)
        vr.transform(gdf=gdf_no_crs)


def test_duplicate_seeds_value_error() -> None:
    """Test checks if duplicate points are disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame(
            {GEOMETRY_COLUMN: [Point(0, 0), Point(0, 0), Point(-1, -1), Point(2, 2)]},
            index=[1, 2, 3, 4],
            crs=WGS84_CRS,
        )
        VoronoiRegionalizer(seeds=seeds_gdf)


def test_seed_outside_earth_bounding_box_value_error() -> None:
    """Test checks if points outside Earth bounding box are disallowed."""
    with pytest.raises(ValueError):
        seeds = [Point(0, 0), Point(-1, -1), Point(2, 2), Point(200, 200)]
        VoronoiRegionalizer(seeds=seeds)


def test_single_seed_region() -> None:
    """Test checks if single seed is disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame(
            {GEOMETRY_COLUMN: [Point(0, 0)]},
            index=[1],
            crs=WGS84_CRS,
        )
        VoronoiRegionalizer(seeds=seeds_gdf)


def test_single_seed_algorithm_error() -> None:
    """Test checks if single seed is disallowed."""
    with pytest.raises(ValueError):
        generate_voronoi_regions(seeds=[Point(0, 0)], max_meters_between_points=10_000)


@pytest.mark.parametrize(  # type: ignore
    "num_of_multiprocessing_workers,expected_num_of_multiprocessing_workers",
    [
        (-1, cpu_count()),
        (0, 1),
        (1, 1),
        (cpu_count() - 1, cpu_count() - 1),
        (cpu_count(), cpu_count()),
    ],
)
def test_num_of_multiprocessing_workers(
    num_of_multiprocessing_workers: int, expected_num_of_multiprocessing_workers: int
) -> None:
    """Test checks if number of workers is parsed correctly."""
    assert (
        _parse_num_of_multiprocessing_workers(num_of_multiprocessing_workers)
        == expected_num_of_multiprocessing_workers
    )


@pytest.mark.parametrize(  # type: ignore
    "multiprocessing_activation_threshold,expected_multiprocessing_activation_threshold",
    [
        (None, 100),
        (0, 100),
        (1, 1),
        (100, 100),
        (1_000, 1_000),
        (10_000, 10_000),
    ],
)
def test_multiprocessing_activation_threshold(
    multiprocessing_activation_threshold: int,
    expected_multiprocessing_activation_threshold: int,
) -> None:
    """Test checks if multiprocessing activation threshold is parsed correctly."""
    assert (
        _parse_multiprocessing_activation_threshold(multiprocessing_activation_threshold)
        == expected_multiprocessing_activation_threshold
    )


@pytest.mark.parametrize(  # type: ignore
    "max_meters_between_points",
    [100_000, 10_000, 1_000, 500],
)
def test_regions_edge_resolution(
    max_meters_between_points: int,
    gdf_earth_poles: gpd.GeoDataFrame,
    earth_bbox: Polygon,
) -> None:
    """Test checks if regions with different resolution are generated correctly."""
    vr = VoronoiRegionalizer(
        seeds=gdf_earth_poles,
        max_meters_between_points=max_meters_between_points,
        multiprocessing_activation_threshold=6,
    )
    result_gdf = vr.transform()
    assert len(result_gdf.index) == 6
    assert check_if_seeds_match_regions(
        seeds=gdf_earth_poles, regions=result_gdf
    ), "Seeds don't match generated regions"
    assert result_gdf.geometry.unary_union.difference(
        earth_bbox
    ).is_empty, "Result doesn't cover bounding box"
    assert check_if_disjoint(result_gdf), "Result isn't disjoint"


@pytest.mark.parametrize("random_points", [10, 100, 1_000, 10_000])  # type: ignore
def test_multiple_seeds_regions(
    random_points: int,
    earth_bbox: Polygon,
) -> None:
    """Test checks if regions are generated correctly."""
    seed = np.random.default_rng().integers(100_000_000)
    seeds = get_random_points(random_points, seed)
    vr = VoronoiRegionalizer(seeds=seeds)
    result_gdf = vr.transform()
    assert len(result_gdf.index) == random_points
    assert check_if_seeds_match_regions(
        seeds=gpd.GeoDataFrame(geometry=seeds, crs=WGS84_CRS), regions=result_gdf
    ), f"Seeds don't match generated regions (seed: {seed})"
    assert result_gdf.geometry.unary_union.difference(
        earth_bbox
    ).is_empty, f"Result doesn't cover bounding box (seed: {seed})"
    assert check_if_disjoint(result_gdf), f"Result isn't disjoint (seed: {seed})"


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
    vr = VoronoiRegionalizer(seeds=seeds_gdf)
    result_gdf = vr.transform(gdf=gdf_earth_bbox)
    assert len(result_gdf.index) == 4
    assert check_if_seeds_match_regions(
        seeds=seeds_gdf, regions=result_gdf
    ), "Seeds don't match generated regions"
    assert result_gdf.geometry.unary_union.difference(
        earth_bbox
    ).is_empty, "Result doesn't cover bounding box"
    assert check_if_disjoint(result_gdf), "Result isn't disjoint"


def test_default_parameter(gdf_earth_poles: gpd.GeoDataFrame, earth_bbox: Polygon) -> None:
    """Test checks if regions are generated correctly with a default mask."""
    vr = VoronoiRegionalizer(seeds=gdf_earth_poles)
    result_gdf = vr.transform(gdf=None)
    assert len(result_gdf.index) == 6
    assert check_if_seeds_match_regions(
        seeds=gdf_earth_poles, regions=result_gdf
    ), "Seeds don't match generated regions"
    assert result_gdf.unary_union.difference(earth_bbox).is_empty
    assert check_if_disjoint(result_gdf), "Result isn't disjoint"


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
    vr = VoronoiRegionalizer(seeds=gdf_earth_poles)
    result_gdf = vr.transform(gdf=gdf)
    assert merge_disjointed_gdf_geometries(result_gdf).difference(gdf.iloc[0].geometry).is_empty
