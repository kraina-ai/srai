"""Tests for merging OSM Loaders filters."""
from unittest import TestCase

import pytest

from srai.loaders.osm_loaders.filters._typing import (
    grouped_osm_tags_type,
    merge_grouped_osm_tags_type,
    osm_tags_type,
)

ut = TestCase()


@pytest.fixture  # type: ignore
def expected_result_min_count_8m() -> osm_tags_type:
    """Get expected results when using `min_count=8_000_000`."""
    return {
        "natural": ["wood"],
        "landuse": ["farmland", "residential"],
    }


def test_merge_grouped_filters() -> None:
    """Test merging grouped tags filter into a base osm filter."""
    base_filter: osm_tags_type = {}
    grouped_filter: grouped_osm_tags_type = {}
    merged_filters = merge_grouped_osm_tags_type(grouped_filter)

    ut.assertDictEqual(base_filter, merged_filters)
