"""Tests for OvertureMapsLoader."""

from typing import Optional
from unittest import TestCase

import pytest
from shapely import box
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from srai.loaders.overturemaps_loader import OvertureMapsLoader

ut = TestCase()
TEST_OVERTUREMAPS_RELEASE_VERSION = "2024-08-20.0"


@pytest.mark.parametrize(  # type: ignore
    "test_geometries,theme_type_pairs,hierarchy_depth,places_confidence_score,include_all_columns,expected_result_length,expected_features_columns_length",
    [
        (
            [
                Polygon(
                    [
                        (7.416769421059001, 43.7346112362936),
                        (7.416769421059001, 43.730681304758946),
                        (7.4218262821731, 43.730681304758946),
                        (7.4218262821731, 43.7346112362936),
                    ]
                )
            ],
            None,
            0,
            0.75,
            True,
            895,
            8,
        ),
        (
            [
                Polygon(
                    [
                        (7.416769421059001, 43.7346112362936),
                        (7.416769421059001, 43.730681304758946),
                        (7.4218262821731, 43.730681304758946),
                        (7.4218262821731, 43.7346112362936),
                    ]
                ),
                box(0, 0, 1, 1),
            ],
            [("base", "water"), ("base", "land_use"), ("base", "land")],
            None,
            0,
            False,
            93,
            20,
        ),
        (
            [
                Polygon(
                    [
                        (7.416769421059001, 43.7346112362936),
                        (7.416769421059001, 43.730681304758946),
                        (7.4218262821731, 43.730681304758946),
                        (7.4218262821731, 43.7346112362936),
                    ]
                ),
                Polygon(
                    [
                        (16.8073393, 51.1389477),
                        (17.0278673, 51.0426754),
                        (17.1762192, 51.1063195),
                        (16.9580276, 51.2093551),
                        (16.8073393, 51.1389477),
                    ]
                ),
            ],
            [("places", "place")],
            99,
            0.9,
            False,
            327,
            205,
        ),
    ],
)
def test_overture_maps_loader(
    test_geometries: list[BaseGeometry],
    theme_type_pairs: Optional[list[tuple[str, str]]],
    hierarchy_depth: Optional[int],
    places_confidence_score: float,
    include_all_columns: bool,
    expected_result_length: int,
    expected_features_columns_length: int,
) -> None:
    """Test `OvertureMapsLoader.load()`."""
    loader = OvertureMapsLoader(
        release=TEST_OVERTUREMAPS_RELEASE_VERSION,
        theme_type_pairs=theme_type_pairs,
        include_all_possible_columns=include_all_columns,
        hierarchy_depth=hierarchy_depth,
        places_minimal_confidence=places_confidence_score,
    )
    result = loader.load(area=test_geometries, ignore_cache=True)

    assert (
        len(result) == expected_result_length
    ), f"Mismatched result length ({len(result)}, {expected_result_length})"
    assert (
        len(result.columns) == expected_features_columns_length + 1
    ), f"Mismatched columns length ({len(result.columns)}, {expected_features_columns_length + 1})"
