"""Tests for OSM Loaders filters."""

import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any

import pytest
import requests_mock as r_mock
from requests import HTTPError

from srai.loaders.osm_loaders.filters import get_popular_tags


@pytest.fixture  # type: ignore
def popular_filter_api_data() -> dict[str, Any]:
    """Load example taginfo API response data from file."""
    with (Path(__file__).parent / "popular_filter_example.json").open("rt", encoding="utf-8") as f:
        res: dict[str, Any] = json.load(f)
        return res


@pytest.fixture  # type: ignore
def expected_result_all() -> dict[str, list[str]]:
    """Get expected results for whole api data."""
    return {
        "natural": ["wood"],
        "landuse": ["farmland", "residential"],
        "source": [
            "cadastre-dgi-fr source : Direction Générale des Impôts - Cadastre. Mise à jour : 2012",
            "bing",
        ],
        "highway": ["tertiary", "crossing"],
        "building:levels": ["2"],
        "source:date": ["2014-03-24"],
        "layer": ["1"],
    }


@pytest.fixture  # type: ignore
def expected_in_wiki_only() -> dict[str, list[str]]:
    """Get expected results when using `in_wiki_only=True`."""
    return {
        "natural": ["wood"],
        "landuse": ["farmland", "residential"],
        "source": ["bing"],
        "highway": ["tertiary", "crossing"],
        "layer": ["1"],
    }


@pytest.fixture  # type: ignore
def expected_result_min_count_8m() -> dict[str, list[str]]:
    """Get expected results when using `min_count=8_000_000`."""
    return {
        "natural": ["wood"],
        "landuse": ["farmland", "residential"],
    }


@pytest.fixture  # type: ignore
def expected_result_min_fraction() -> dict[str, list[str]]:
    """Get expected results when using `min_fraction=0.001`."""
    return {
        "natural": ["wood"],
    }


@pytest.mark.parametrize(  # type: ignore
    "expected_result_fixture,status_code,in_wiki_only,min_count,min_fraction,expectation",
    [
        ("expected_result_all", 200, False, 0, 0.0, does_not_raise()),
        ("expected_in_wiki_only", 200, True, 0, 0.0, does_not_raise()),
        ("expected_result_min_count_8m", 200, False, 8_000_000, 0.0, does_not_raise()),
        ("expected_result_min_fraction", 200, False, 0, 0.001, does_not_raise()),
        ("expected_result_all", 400, False, 0, 0.0, pytest.raises(HTTPError)),
        ("expected_result_all", 500, False, 0, 0.0, pytest.raises(HTTPError)),
    ],
)
def test_get_popular_tags(
    expected_result_fixture: str,
    status_code: int,
    in_wiki_only: bool,
    min_count: int,
    min_fraction: float,
    expectation: Any,
    request: Any,
):
    """Test downloading popular tags from taginfo API."""
    requests_mock = request.getfixturevalue("requests_mock")
    api_response = request.getfixturevalue("popular_filter_api_data")
    expected_result = request.getfixturevalue(expected_result_fixture)

    with expectation:
        requests_mock.get(r_mock.ANY, json=api_response, status_code=status_code)
        result = get_popular_tags(
            in_wiki_only=in_wiki_only, min_count=min_count, min_fraction=min_fraction
        )
        assert result == expected_result
