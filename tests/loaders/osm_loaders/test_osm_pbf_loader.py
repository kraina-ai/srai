"""Tests for OSMPbfLoader."""
# from typing import Any

# import geopandas as gpd
# import pandas as pd
# import pytest
# from pandas.testing import assert_frame_equal
# from shapely.geometry import Point, Polygon

# from srai.loaders.osm_loaders import OSMOnlineLoader
# from srai.loaders.osm_loaders.filters.osm_tags_type import osm_tags_type


# @pytest.mark.parametrize(  # type: ignore
#     "area_gdf_fixture,query,expected_result_gdf_fixture",
#     [
#         ("single_polygon_area_gdf", {"amenity": "restaurant"}, "expected_result_single_polygon"),
#         ("two_polygons_area_gdf", {"amenity": "restaurant"}, "expected_result_gdf_simple"),
#         (
#             "two_polygons_area_gdf",
#             {"amenity": ["restaurant", "bar"], "building": True},
#             "expected_result_gdf_complex",
#         ),
#         (
#             "empty_area_gdf",
#             {"amenity": ["restaurant", "bar"], "building": True},
#             "empty_result_gdf",
#         ),
#         (
#             "area_with_no_objects_gdf",
#             {"amenity": ["restaurant", "bar"], "building": True},
#             "empty_result_gdf",
#         ),
#     ],
# )
# def test_osm_online_loader(
#     area_gdf_fixture: str,
#     query: osm_tags_type,
#     expected_result_gdf_fixture: str,
#     request: Any,
# ):
#     """Test `OSMOnlineLoader.load()`."""
#     _ = request.getfixturevalue("mock_osmnx")
#     area_gdf = request.getfixturevalue(area_gdf_fixture)
#     expected_result_gdf = request.getfixturevalue(expected_result_gdf_fixture)
#     loader = OSMOnlineLoader()
#     res = loader.load(area_gdf, query)
#     assert "address" not in res.columns
#     assert_frame_equal(res, expected_result_gdf, check_like=True)
