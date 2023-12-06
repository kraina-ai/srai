"""Tests for OSMOnlineLoader."""

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import osmnx as ox
import pytest
from packaging import version
from pandas.testing import assert_frame_equal

from srai.constants import WGS84_CRS
from srai.loaders.osm_loaders import OSMOnlineLoader
from srai.loaders.osm_loaders.filters import OsmTagsFilter

if TYPE_CHECKING:  # pragma: no cover
    from shapely.geometry import Polygon


@pytest.fixture  # type: ignore
def mock_osmnx(
    mocker, two_polygons_area_gdf, area_with_no_objects_gdf, amenities_gdf, building_gdf
):
    """Patch `osmnx` functions to return data from predefined gdfs."""
    gdfs = {"amenity": amenities_gdf, "building": building_gdf}
    polygon_1, polygon_2 = two_polygons_area_gdf["geometry"]
    empty_polygon = area_with_no_objects_gdf["geometry"][0]

    def mock_geometries_from_polygon(polygon: "Polygon", tags: OsmTagsFilter) -> gpd.GeoDataFrame:
        tag_key, tag_value = list(tags.items())[0]
        gdf = gdfs[tag_key]
        if tag_value is True:
            tag_res = gdf
        else:
            tag_res = gdf.loc[gdf[tag_key] == tag_value]
        if tag_res.empty:
            return tag_res
        if polygon == polygon_1:
            return tag_res.iloc[:1]
        elif polygon == polygon_2:
            return tag_res.iloc[1:]
        elif polygon == empty_polygon:
            return gpd.GeoDataFrame(crs=WGS84_CRS, geometry=[])
        return None

    if version.parse(ox.__version__) >= version.parse("1.5.0"):
        mocker.patch("osmnx.features_from_polygon", new=mock_geometries_from_polygon)
    else:
        mocker.patch("osmnx.geometries_from_polygon", new=mock_geometries_from_polygon)


@pytest.mark.parametrize(  # type: ignore
    "area_gdf_fixture,query,expected_result_gdf_fixture",
    [
        ("single_polygon_area_gdf", {"amenity": "restaurant"}, "expected_result_single_polygon"),
        ("two_polygons_area_gdf", {"amenity": "restaurant"}, "expected_result_gdf_simple"),
        (
            "two_polygons_area_gdf",
            {"amenity": ["restaurant", "bar"], "building": True},
            "expected_result_gdf_complex",
        ),
        (
            "empty_area_gdf",
            {"amenity": ["restaurant", "bar"], "building": True},
            "empty_result_gdf",
        ),
        (
            "area_with_no_objects_gdf",
            {"amenity": ["restaurant", "bar"], "building": True},
            "empty_result_gdf",
        ),
    ],
)
def test_osm_online_loader(
    area_gdf_fixture: str,
    query: OsmTagsFilter,
    expected_result_gdf_fixture: str,
    request: Any,
):
    """Test `OSMOnlineLoader.load()`."""
    _ = request.getfixturevalue("mock_osmnx")
    area_gdf = request.getfixturevalue(area_gdf_fixture)
    expected_result_gdf = request.getfixturevalue(expected_result_gdf_fixture)
    loader = OSMOnlineLoader()
    res = loader.load(area_gdf, query)
    assert "address" not in res.columns
    assert_frame_equal(res, expected_result_gdf, check_like=True)
