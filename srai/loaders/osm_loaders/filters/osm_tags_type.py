"""Module contains a dedicated type alias for OSM tags filter."""
from typing import Dict, List, Union

osm_tags_type = Dict[str, Union[List[str], str, bool]]

grouped_osm_tags_type = Dict[str, Dict[str, Union[List[str], str, bool]]]
