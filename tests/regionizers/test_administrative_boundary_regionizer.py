"""Voronoi regionizer tests."""
from contextlib import nullcontext as does_not_raise
from typing import Any, Union

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

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
def test_single_points(toposimplify: Union[bool, float]) -> None:
    """Test checks if single points work."""
    country_points_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(19.24530, 52.21614),  # Poland
                Point(10.48674, 51.38001),  # Germany
                Point(14.74938, 47.69628),  # Austria
                Point(15.00989, 49.79905),  # Czechia
            ]
        },
        crs=WGS84_CRS,
    )
    abr = AdministrativeBoundaryRegionizer(
        admin_level=2, return_empty_region=False, clip_regions=False, toposimplify=toposimplify
    )
    countries_result_gdf = abr.transform(gdf=country_points_gdf)
    assert list(countries_result_gdf.index) == ["Poland", "Germany", "Austria", "Czechia"]


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
def test_empty_region_full_bounding_box(toposimplify: Union[bool, float]) -> None:
    """Test checks if empty region fills required bounding box."""
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
    assert "EMPTY" in madagascar_result_gdf.index


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
def test_no_empty_region_full_bounding_box(toposimplify: Union[bool, float]) -> None:
    """Test checks if no empty region is generated when not needed."""
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
    assert "EMPTY" not in asia_result_gdf.index
