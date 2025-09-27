"""Tests for S2 utils."""

from contextlib import nullcontext as does_not_raise
from typing import Any

import geopandas as gpd
import pytest
from s2sphere import CellId
from shapely.geometry import Polygon

from srai.constants import WGS84_CRS
from srai.embedders.s2vec import s2_utils


@pytest.mark.parametrize(  # type: ignore
    "token,target_level,expectation",
    [
        ("invalid_token", 18, pytest.raises(ValueError)),
        ("470fc275", 1, pytest.raises(ValueError)),
        ("470fc275", 18, does_not_raise()),
    ],
)
def test_get_children_from_token_incorrect_params(
    token: str, target_level: int, expectation: Any
) -> None:
    """Test that incorrect tokens or target_level raise ValueError."""
    with expectation:
        s2_utils.get_children_from_token(token, target_level)


def test_get_children_from_token() -> None:
    """Test that S2 children cells are properly returned (correct amount of cells and level)."""
    token = "470fc275"
    parent_level = CellId.from_token(token).level()
    target_level = 18

    children = s2_utils.get_children_from_token(token, target_level)
    assert len(children) == 4 ** (target_level - parent_level)
    for child_token in children.index:
        child_level = CellId.from_token(child_token).level()
        assert child_level == target_level


def test_sort_patches() -> None:
    """Test that patches are properly sorted from left to right and top to bottom."""
    polys = [
        Polygon([(1, 1), (1, 2), (2, 2), (2, 1)]),  # top-right
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # bottom-left
        Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),  # bottom-right
        Polygon([(0, 1), (0, 2), (1, 2), (1, 1)]),  # top-left
    ]
    gdf = gpd.GeoDataFrame({"geometry": polys}, crs=WGS84_CRS)
    gdf = gdf.sample(frac=1, random_state=42).reset_index(drop=True)

    sorted_gdf = s2_utils.sort_patches(gdf)
    expected_order = [
        Polygon([(0, 1), (0, 2), (1, 2), (1, 1)]),  # top-left
        Polygon([(1, 1), (1, 2), (2, 2), (2, 1)]),  # top-right
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # bottom-left
        Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),  # bottom-right
    ]
    for sorted_poly, expected_poly in zip(sorted_gdf.geometry, expected_order):
        assert sorted_poly.equals(expected_poly)
