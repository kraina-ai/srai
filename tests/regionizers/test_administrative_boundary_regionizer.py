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
    request: Any,
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
def mock_for_madagascar(mocker: MockerFixture) -> None:
    """Mock overpass API."""
    mocker.patch.object(
        API,
        "get",
        return_value={
            "elements": [
                {
                    "type": "relation",
                    "id": 2137,
                },
            ]
        },
    )

    geocoded_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                box(
                    minx=47,
                    miny=-15,
                    maxx=48,
                    maxy=-14,
                )
            ],
        },
        crs=WGS84_CRS,
    )

    mocker.patch(
        "osmnx.geocode_to_gdf",
        return_value=geocoded_gdf,
    )


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
    request.getfixturevalue("mock_for_madagascar")
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
    print(madagascar_result_gdf)
    assert "EMPTY" in madagascar_result_gdf.index


@pytest.fixture  # type: ignore
def mock_for_asia(mocker: MockerFixture) -> None:
    """Mock overpass API."""
    mocker.patch.object(
        API,
        "get",
        return_value={
            "elements": [
                {
                    "type": "relation",
                    "id": 2137,
                },
            ]
        },
    )

    geocoded_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                box(
                    minx=69,
                    miny=23,
                    maxx=89,
                    maxy=35,
                )
            ],
        },
        crs=WGS84_CRS,
    )

    mocker.patch(
        "osmnx.geocode_to_gdf",
        return_value=geocoded_gdf,
    )


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
    request.getfixturevalue("mock_for_asia")
    asia_bbox = box(
        minx=69.73278412113555,
        miny=24.988848422533074,
        maxx=88.50230949587835,
        maxy=34.846427760404225,
    )
    asia_bbox_gdf = gpd.GeoDataFrame({"geometry": [asia_bbox]}, crs=WGS84_CRS)
    abr = AdministrativeBoundaryRegionizer(
        admin_level=2, return_empty_region=True, toposimplify=toposimplify
    )
    asia_result_gdf = abr.transform(gdf=asia_bbox_gdf)
    assert _merge_disjointed_gdf_geometries(asia_result_gdf).difference(asia_bbox).is_empty
    print(asia_result_gdf)
    assert "EMPTY" not in asia_result_gdf.index
