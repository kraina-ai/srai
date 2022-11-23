"""Intersection joiner tests."""
from unittest import TestCase

import geopandas as gpd
import pandas as pd
import pytest

from srai.joiners import IntersectionJoiner

ut = TestCase()


def test_regions_without_geometry_value_error(
    no_geometry_gdf: gpd.GeoDataFrame, features_gdf: gpd.GeoDataFrame
) -> None:
    """Test checks if regions without geometry are disallowed."""
    with pytest.raises(ValueError):
        IntersectionJoiner().join(regions=no_geometry_gdf, features=features_gdf)


def test_features_without_geometry_value_error(
    regions_gdf: gpd.GeoDataFrame, no_geometry_gdf: gpd.GeoDataFrame
) -> None:
    """Test checks if features without geometry are disallowed."""
    with pytest.raises(ValueError):
        IntersectionJoiner().join(regions=regions_gdf, features=no_geometry_gdf)


def test_empty_regions_value_error(
    empty_gdf: gpd.GeoDataFrame, features_gdf: gpd.GeoDataFrame
) -> None:
    """Test checks if empty regions are disallowed."""
    with pytest.raises(ValueError):
        IntersectionJoiner().join(regions=empty_gdf, features=features_gdf)


def test_empty_features_value_error(
    regions_gdf: gpd.GeoDataFrame, empty_gdf: gpd.GeoDataFrame
) -> None:
    """Test checks if empty features are disallowed."""
    with pytest.raises(ValueError):
        IntersectionJoiner().join(regions=regions_gdf, features=empty_gdf)


def test_correct_multiindex_intersection_joiner(
    regions_gdf: gpd.GeoDataFrame, features_gdf: gpd.GeoDataFrame, joint_multiindex: pd.MultiIndex
) -> None:
    """Test checks if intersection joiner returns correct MultiIndex."""
    joint = IntersectionJoiner().join(regions=regions_gdf, features=features_gdf)

    ut.assertEqual(joint.index.names, joint_multiindex.names)
    ut.assertCountEqual(joint.index, joint_multiindex)
