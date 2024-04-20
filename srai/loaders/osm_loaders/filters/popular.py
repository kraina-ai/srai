"""
OSM popular tags downloader.

This module exposes a function `get_popular_tags` that uses taginfo[1] API
to download the most popular tags from OSM

References:
    1. https://taginfo.openstreetmap.org/
"""

import operator
from typing import Any

import requests
from functional import seq

from srai.loaders.osm_loaders.filters import OsmTagsFilter

_TAGINFO_API_ADDRESS = "https://taginfo.openstreetmap.org"
_TAGINFO_API_TAGS = _TAGINFO_API_ADDRESS + "/api/4/tags/popular"


def get_popular_tags(
    in_wiki_only: bool = False, min_count: int = 0, min_fraction: float = 0.0
) -> OsmTagsFilter:
    """
    Download the OSM's most popular tags from taginfo api.

    This is a wrapper around the `popular` taginfo api endpoint [1].
    It queries the API, and optionally filters the results
    according to argument values.

    Args:
        in_wiki_only (bool, optional): If True, only return results tags
            that have at least one wiki page. Defaults to False.
        min_count(int, optional): Minimum number of objects in OSM with this tag attached,
            to include the tag in the results. Defaults to 0 (no filtering).
        min_fraction(float, optional): What fraction of all objects have to have this tag attached,
            to include it in the results. Defaults to 0.0 (no filtering).

    Returns:
        Dict[str, List[str]]: dictionary containing the downloaded popular tags.

    References:
        1. https://taginfo.openstreetmap.org/taginfo/apidoc#api_4_tags_popular
    """
    taginfo_api_response = requests.get(
        _TAGINFO_API_TAGS,
        headers={"User-Agent": "SRAI Python package (https://github.com/kraina-ai/srai)"},
    )
    taginfo_api_response.raise_for_status()
    taginfo_data = taginfo_api_response.json()["data"]
    return _parse_taginfo_response(taginfo_data, in_wiki_only, min_count, min_fraction)


def _parse_taginfo_response(
    taginfo_data: list[dict[str, Any]], in_wiki_only: bool, min_count: int, min_fraction: float
) -> OsmTagsFilter:
    result_tags = (
        seq(taginfo_data)
        .filter(lambda t: t["count_all"] >= min_count)
        .filter(lambda t: t["count_all_fraction"] >= min_fraction)
    )
    if in_wiki_only:
        result_tags = result_tags.filter(operator.itemgetter("in_wiki"))
    taginfo_grouped: OsmTagsFilter = (
        result_tags.map(operator.itemgetter("key", "value")).group_by_key().to_dict()
    )
    return taginfo_grouped
