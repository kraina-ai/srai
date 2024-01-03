"""Voronoi regionalizer tests."""

from contextlib import nullcontext as does_not_raise
from typing import Any, Union

import geopandas as gpd
import pytest
from overpass import API
from pytest_mock import MockerFixture
from shapely.geometry import Point, box

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.geometry import merge_disjointed_gdf_geometries
from srai.regionalizers import AdministrativeBoundaryRegionalizer

bbox = box(minx=-180, maxx=180, miny=-90, maxy=90)
bbox_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [bbox]}, crs=WGS84_CRS)


@pytest.mark.parametrize(  # type: ignore
    "admin_level,expectation",
    [
        (-1, pytest.raises(ValueError)),
        (0, pytest.raises(ValueError)),
        (1, does_not_raise()),
        (2, does_not_raise()),
        (3, does_not_raise()),
        (4, does_not_raise()),
        (5, does_not_raise()),
        (6, does_not_raise()),
        (7, does_not_raise()),
        (8, does_not_raise()),
        (9, does_not_raise()),
        (10, does_not_raise()),
        (11, does_not_raise()),
        (12, pytest.raises(ValueError)),
    ],
)
def test_admin_level(
    admin_level: int,
    expectation: Any,
) -> None:
    """Test checks if illegal admin_level is disallowed."""
    with expectation:
        AdministrativeBoundaryRegionalizer(admin_level=admin_level)


def test_empty_gdf_attribute_error(gdf_empty) -> None:  # type: ignore
    """Test checks if empty GeoDataFrames are disallowed."""
    with pytest.raises(AttributeError):
        abr = AdministrativeBoundaryRegionalizer(admin_level=4)
        abr.transform(gdf_empty)


def test_no_crs_gdf_value_error(gdf_no_crs) -> None:  # type: ignore
    """Test checks if GeoDataFrames without crs are disallowed."""
    with pytest.raises(ValueError):
        abr = AdministrativeBoundaryRegionalizer(admin_level=4)
        abr.transform(gdf=gdf_no_crs)


@pytest.fixture  # type: ignore
def mock_overpass_api(mocker: MockerFixture) -> None:
    """Mock overpass API."""
    mocker.patch.object(API, "get", return_value={"elements": [{"type": "relation", "id": 2137}]})

    geocoded_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: [box(minx=0, miny=0, maxx=1, maxy=1)]}, crs=WGS84_CRS
    )
    mocker.patch("osmnx.geocode_to_gdf", return_value=geocoded_gdf)


@pytest.mark.parametrize(  # type: ignore
    "toposimplify",
    [
        (True),
        (0.0001),
        (0.001),
        (0.01),
        (False),
        (0),
    ],
)
def test_empty_region_full_bounding_box(toposimplify: Union[bool, float], request: Any) -> None:
    """Test checks if empty region fills required bounding box."""
    request.getfixturevalue("mock_overpass_api")
    request_bbox = box(minx=0, miny=0, maxx=2, maxy=2)
    request_bbox_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [request_bbox]}, crs=WGS84_CRS)
    abr = AdministrativeBoundaryRegionalizer(
        admin_level=4, return_empty_region=True, toposimplify=toposimplify
    )
    result_gdf = abr.transform(gdf=request_bbox_gdf)
    assert merge_disjointed_gdf_geometries(result_gdf).difference(request_bbox).is_empty
    assert "EMPTY" in result_gdf.index


@pytest.mark.parametrize(  # type: ignore
    "toposimplify",
    [
        (True),
        (0.0001),
        (0.001),
        (0.01),
        (False),
        (0),
    ],
)
def test_no_empty_region_full_bounding_box(toposimplify: Union[bool, float], request: Any) -> None:
    """Test checks if no empty region is generated when not needed."""
    request.getfixturevalue("mock_overpass_api")
    request_bbox = box(minx=0, miny=0, maxx=1, maxy=1)
    request_bbox_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [request_bbox]}, crs=WGS84_CRS)
    abr = AdministrativeBoundaryRegionalizer(
        admin_level=2, return_empty_region=True, toposimplify=toposimplify
    )
    result_gdf = abr.transform(gdf=request_bbox_gdf)
    assert merge_disjointed_gdf_geometries(result_gdf).difference(request_bbox).is_empty
    assert "EMPTY" not in result_gdf.index


@pytest.mark.parametrize(  # type: ignore
    "toposimplify",
    [
        (True),
        (0.0001),
        (0.001),
        (0.01),
        (False),
        (0),
    ],
)
def test_points_in_result(toposimplify: Union[bool, float], request: Any) -> None:
    """Test checks case when points are in a requested region."""
    request.getfixturevalue("mock_overpass_api")
    request_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [Point(0.5, 0.5)]}, crs=WGS84_CRS)

    abr = AdministrativeBoundaryRegionalizer(
        admin_level=2, return_empty_region=False, clip_regions=False, toposimplify=toposimplify
    )

    result_gdf = abr.transform(gdf=request_gdf)
    assert request_gdf.geometry[0].covered_by(result_gdf.geometry[0])


@pytest.mark.parametrize(  # type: ignore
    "toposimplify",
    [
        (True),
        (0.0001),
        (0.001),
        (0.01),
        (False),
        (0),
    ],
)
def test_toposimplify_on_real_data(toposimplify: Union[float, bool]) -> None:
    """Test if toposimplify usage covers an entire region."""
    madagascar_bbox = box(
        minx=43.2541870461, miny=-25.6014344215, maxx=50.4765368996, maxy=-12.0405567359
    )
    madagascar_bbox_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [madagascar_bbox]}, crs=WGS84_CRS)

    abr = AdministrativeBoundaryRegionalizer(
        admin_level=4, return_empty_region=True, toposimplify=toposimplify
    )
    madagascar_result_gdf = abr.transform(gdf=madagascar_bbox_gdf)
    assert (
        merge_disjointed_gdf_geometries(madagascar_result_gdf).difference(madagascar_bbox).is_empty
    )


@pytest.mark.parametrize(  # type: ignore
    "return_empty_region",
    [
        (True),
        (False),
    ],
)
def test_regions_not_found_on_real_data(return_empty_region: bool) -> None:
    """Test if warns when can't find any regions."""
    null_island_region = box(minx=0, miny=0, maxx=0.1, maxy=0.1)
    null_island_region_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: [null_island_region]}, crs=WGS84_CRS
    )

    abr = AdministrativeBoundaryRegionalizer(
        admin_level=10, return_empty_region=return_empty_region
    )

    with pytest.warns(RuntimeWarning):
        madagascar_result_gdf = abr.transform(gdf=null_island_region_gdf)

    if return_empty_region:
        assert (
            merge_disjointed_gdf_geometries(madagascar_result_gdf)
            .difference(null_island_region)
            .is_empty
        )
    else:
        assert merge_disjointed_gdf_geometries(madagascar_result_gdf).is_empty
