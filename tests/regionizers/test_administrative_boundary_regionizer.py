"""Voronoi regionizer tests."""
from contextlib import nullcontext as does_not_raise
from typing import Any, Union

import geopandas as gpd
import pytest
from overpass import API
from pytest_mock import MockerFixture
from shapely.geometry import box

from srai.regionizers import AdministrativeBoundaryRegionizer
from srai.utils import _merge_disjointed_gdf_geometries
from srai.utils.constants import WGS84_CRS

bbox = box(minx=-180, maxx=180, miny=-90, maxy=90)
bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=WGS84_CRS)


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
        AdministrativeBoundaryRegionizer(admin_level=admin_level)


def test_empty_gdf_attribute_error(gdf_empty) -> None:  # type: ignore
    """Test checks if empty GeoDataFrames are disallowed."""
    with pytest.raises(AttributeError):
        abr = AdministrativeBoundaryRegionizer(admin_level=4)
        abr.transform(gdf_empty)


def test_no_crs_gdf_value_error(gdf_no_crs) -> None:  # type: ignore
    """Test checks if GeoDataFrames without crs are disallowed."""
    with pytest.raises(ValueError):
        abr = AdministrativeBoundaryRegionizer(admin_level=4)
        abr.transform(gdf=gdf_no_crs)


@pytest.fixture  # type: ignore
def mock_overpass_api(mocker: MockerFixture) -> None:
    """Mock overpass API."""
    mocker.patch.object(API, "get", return_value={"elements": [{"type": "relation", "id": 2137}]})

    geocoded_gdf = gpd.GeoDataFrame(
        {"geometry": [box(minx=0, miny=0, maxx=1, maxy=1)]}, crs=WGS84_CRS
    )
    mocker.patch("osmnx.geocode_to_gdf", return_value=geocoded_gdf)

    mocker.patch("osmnx.downloader._retrieve_from_cache", return_value=None)
    mocker.patch("osmnx.downloader._save_to_cache")


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
    region_bbox = box(minx=0, miny=0, maxx=2, maxy=2)
    region_bbox_gdf = gpd.GeoDataFrame({"geometry": [region_bbox]}, crs=WGS84_CRS)
    abr = AdministrativeBoundaryRegionizer(
        admin_level=4, return_empty_region=True, toposimplify=toposimplify
    )
    result_gdf = abr.transform(gdf=region_bbox_gdf)
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(region_bbox).is_empty
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
    region_bbox = box(minx=0, miny=0, maxx=1, maxy=1)
    region_bbox_gdf = gpd.GeoDataFrame({"geometry": [region_bbox]}, crs=WGS84_CRS)
    abr = AdministrativeBoundaryRegionizer(
        admin_level=2, return_empty_region=True, toposimplify=toposimplify
    )
    result_gdf = abr.transform(gdf=region_bbox_gdf)
    assert _merge_disjointed_gdf_geometries(result_gdf).difference(region_bbox).is_empty
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
def test_toposimplify_on_real_data(toposimplify: Union[float, bool]) -> None:
    """Test if toposimplify usage covers an entire region."""
    madagascar_bbox = box(
        minx=43.2541870461, miny=-25.6014344215, maxx=50.4765368996, maxy=-12.0405567359
    )
    madagascar_bbox_gdf = gpd.GeoDataFrame({"geometry": [madagascar_bbox]}, crs=WGS84_CRS)

    abr = AdministrativeBoundaryRegionizer(
        admin_level=4, return_empty_region=True, toposimplify=toposimplify
    )
    madagascar_result_gdf = abr.transform(gdf=madagascar_bbox_gdf)
    assert (
        _merge_disjointed_gdf_geometries(madagascar_result_gdf).difference(madagascar_bbox).is_empty
    )
