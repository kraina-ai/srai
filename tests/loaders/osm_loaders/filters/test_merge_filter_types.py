"""Tests for merging OSM Loaders filters."""

from typing import Any
from unittest import TestCase

import pytest

from srai.loaders.osm_loaders.filters import OsmTagsFilter, merge_osm_tags_filter

ut = TestCase()


@pytest.mark.parametrize(  # type: ignore
    "osm_tags_filter,expected_result_filter",
    [
        ({"tag_a": True}, {"tag_a": True}),
        ({"tag_a": "A"}, {"tag_a": "A"}),
        ({"tag_a": ["A"]}, {"tag_a": ["A"]}),
        ({}, {}),
        ({"group_a": {}}, {}),
        ({"group_a": {"tag_a": True}}, {"tag_a": True}),
        ({"group_a": {"tag_a": "A"}}, {"tag_a": ["A"]}),
        ({"group_a": {"tag_a": ["A"]}}, {"tag_a": ["A"]}),
        ({"group_a": {"tag_a": "A", "tag_b": "B"}}, {"tag_a": ["A"], "tag_b": ["B"]}),
        ({"group_a": {"tag_a": ["A"], "tag_b": ["B"]}}, {"tag_a": ["A"], "tag_b": ["B"]}),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_a": "A", "tag_b": "B"},
            },
            {"tag_a": ["A"], "tag_b": ["B"]},
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_c": "C", "tag_d": "D"},
            },
            {"tag_a": ["A"], "tag_b": ["B"], "tag_c": ["C"], "tag_d": ["D"]},
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_a": "C", "tag_b": "D"},
            },
            {"tag_a": ["A", "C"], "tag_b": ["B", "D"]},
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_a": ["C", "D"], "tag_b": "E"},
            },
            {"tag_a": ["A", "C", "D"], "tag_b": ["B", "E"]},
        ),
        (
            {
                "group_a": {"tag_a": "A", "tag_b": "B"},
                "group_b": {"tag_a": ["C", "D"], "tag_b": True},
            },
            {"tag_a": ["A", "C", "D"], "tag_b": True},
        ),
        (
            {
                "group_a": {"tag_a": ["A"], "tag_b": ["B"]},
                "group_b": {"tag_a": ["C", "D"], "tag_b": False},
            },
            {"tag_a": ["A", "C", "D"], "tag_b": ["B"]},
        ),
        (
            {
                "group_a": {"tag_a": ["A", "C"], "tag_b": ["B", "E"]},
                "group_b": {"tag_a": ["C", "D"], "tag_b": ["B"]},
            },
            {"tag_a": ["A", "C", "D"], "tag_b": ["B", "E"]},
        ),
        ([{"tag_a": True}], {"tag_a": True}),
        ([{"tag_a": "A"}], {"tag_a": ["A"]}),
        ([{"tag_a": ["A"]}], {"tag_a": ["A"]}),
        ([{}], {}),
        ([{"group_a": {}}], {}),
        (
            [{"tag_a": "A", "tag_b": "B"}, {"tag_a": "A", "tag_b": "B"}],
            {"tag_a": ["A"], "tag_b": ["B"]},
        ),
        (
            [
                {
                    "group_a": {"tag_a": "A", "tag_b": "B"},
                    "group_b": {"tag_a": "A", "tag_b": "B"},
                },
                {"tag_a": "A", "tag_b": "B"},
            ],
            {"tag_a": ["A"], "tag_b": ["B"]},
        ),
        (
            [
                {
                    "group_a": {"tag_a": "A", "tag_b": "B"},
                    "group_b": {"tag_a": "A", "tag_b": "B"},
                },
                {
                    "group_a": {"tag_a": "A", "tag_b": "B"},
                    "group_b": {"tag_a": "A", "tag_b": "B"},
                },
            ],
            {"tag_a": ["A"], "tag_b": ["B"]},
        ),
        ([{}, {}], {}),
        ([{"group_a": {}}, {"group_a": {}}], {}),
        ([{"group_a": {}}, {}], {}),
        (
            [{"tag_a": "A", "tag_b": "B"}, {"tag_c": "C", "tag_d": "D"}],
            {"tag_a": ["A"], "tag_b": ["B"], "tag_c": ["C"], "tag_d": ["D"]},
        ),
    ],
)
def test_merge_osm_tags_filters(
    osm_tags_filter: Any, expected_result_filter: OsmTagsFilter
) -> None:
    """Test merging grouped tags filter into a base osm filter."""
    merged_filters = merge_osm_tags_filter(osm_tags_filter)
    ut.assertDictEqual(expected_result_filter, merged_filters)
