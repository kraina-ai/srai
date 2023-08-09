"""Tests for merging OSM Loaders filters."""
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest import TestCase

import pytest

from srai.loaders.osm_loaders.filters import OsmTagsFilter, merge_grouped_osm_tags_filter

ut = TestCase()


@pytest.mark.parametrize(  # type: ignore
    "grouped_filter,expected_result_filter,expectation",
    [
        ({"tag_a": True}, {"tag_a": True}, pytest.raises(ValueError)),
        ({"tag_a": "A"}, {"tag_a": "A"}, pytest.raises(ValueError)),
        ({"tag_a": ["A"]}, {"tag_a": ["A"]}, pytest.raises(ValueError)),
        ({}, {}, does_not_raise()),
        ({"group_a": {}}, {}, does_not_raise()),
        ({"group_a": {"tag_a": True}}, {"tag_a": True}, does_not_raise()),
        ({"group_a": {"tag_a": "A"}}, {"tag_a": ["A"]}, does_not_raise()),
        ({"group_a": {"tag_a": ["A"]}}, {"tag_a": ["A"]}, does_not_raise()),
        (
            {"group_a": {"tag_a": "A", "tag_b": "B"}},
            {"tag_a": ["A"], "tag_b": ["B"]},
            does_not_raise(),
        ),
        (
            {"group_a": {"tag_a": ["A"], "tag_b": ["B"]}},
            {"tag_a": ["A"], "tag_b": ["B"]},
            does_not_raise(),
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_a": "A", "tag_b": "B"},
            },
            {"tag_a": ["A"], "tag_b": ["B"]},
            does_not_raise(),
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_c": "C", "tag_d": "D"},
            },
            {"tag_a": ["A"], "tag_b": ["B"], "tag_c": ["C"], "tag_d": ["D"]},
            does_not_raise(),
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_a": "C", "tag_b": "D"},
            },
            {"tag_a": ["A", "C"], "tag_b": ["B", "D"]},
            does_not_raise(),
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_a": ["C", "D"], "tag_b": "E"},
            },
            {"tag_a": ["A", "C", "D"], "tag_b": ["B", "E"]},
            does_not_raise(),
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_a": ["C", "D"], "tag_b": True},
            },
            {"tag_a": ["A", "C", "D"], "tag_b": True},
            does_not_raise(),
        ),
        (
            {
                "group_a": {"tag_a": ["A"], "tag_b": ["B"]},
                "group_b": {"tag_a": ["C", "D"], "tag_b": False},
            },
            {"tag_a": ["A", "C", "D"], "tag_b": ["B"]},
            does_not_raise(),
        ),
        (
            {
                "group_a": {"tag_a": ["A", "C"], "tag_b": ["B", "E"]},
                "group_b": {"tag_a": ["C", "D"], "tag_b": ["B"]},
            },
            {"tag_a": ["A", "C", "D"], "tag_b": ["B", "E"]},
            does_not_raise(),
        ),
    ],
)
def test_merge_grouped_filters(
    grouped_filter: Any, expected_result_filter: OsmTagsFilter, expectation: Any
) -> None:
    """Test merging grouped tags filter into a base osm filter."""
    with expectation:
        merged_filters = merge_grouped_osm_tags_filter(grouped_filter)
        ut.assertDictEqual(expected_result_filter, merged_filters)
