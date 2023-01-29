"""#TODO."""
from typing import Any, Dict, List

import requests
from functional import seq

_TAGINFO_API_ADDRESS = "https://taginfo.openstreetmap.org"
_TAGINFO_API_TAGS = _TAGINFO_API_ADDRESS + "/api/4/tags/popular"


def get_popular_tags(
    in_wiki_only: bool = True, min_count: int = 0, min_fraction: float = 0.0
) -> Dict[str, List[str]]:
    """#TODO."""
    taginfo_api_response = requests.get(_TAGINFO_API_TAGS)
    taginfo_api_response.raise_for_status()
    taginfo_data = taginfo_api_response.json()["data"]
    return _parse_taginfo_response(taginfo_data, in_wiki_only, min_count, min_fraction)


def _parse_taginfo_response(
    taginfo_data: List[Dict[str, Any]], in_wiki_only: bool, min_count: int, min_fraction: float
) -> Dict[str, List[str]]:
    result_tags = (
        seq(taginfo_data)
        .filter(lambda t: t["count_all"] >= min_count)
        .filter(lambda t: t["count_all_fraction"] >= min_fraction)
    )
    if in_wiki_only:
        result_tags = result_tags.filter(lambda t: t["in_wiki"])
    taginfo_grouped: Dict[str, List[str]] = (
        result_tags.map(lambda t: (t["key"], t["value"])).group_by_key().to_dict()
    )
    return taginfo_grouped
