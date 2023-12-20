from collections.abc import Iterable
from typing import Any, NamedTuple, cast

from srai._typing import is_expected_type


class OsmWayPolygonConfig(NamedTuple):
    """OSM Way polygon features config object."""

    all: Iterable[str]
    allowlist: dict[str, Iterable[str]]
    denylist: dict[str, Iterable[str]]


def parse_dict_to_config_object(raw_config: dict[str, Any]) -> OsmWayPolygonConfig:
    all_tags = raw_config.get("all", [])
    allowlist_tags = raw_config.get("allowlist", {})
    denylist_tags = raw_config.get("denylist", {})
    if not is_expected_type(all_tags, Iterable[str]):
        raise ValueError(f"Wrong type of key: all ({type(all_tags)})")

    if not is_expected_type(allowlist_tags, dict[str, Iterable[str]]):
        raise ValueError(f"Wrong type of key: all ({type(allowlist_tags)})")

    if not is_expected_type(denylist_tags, dict[str, Iterable[str]]):
        raise ValueError(f"Wrong type of key: denylist ({type(denylist_tags)})")

    return OsmWayPolygonConfig(
        all=cast(Iterable[str], all_tags),
        allowlist=cast(dict[str, Iterable[str]], allowlist_tags),
        denylist=cast(dict[str, Iterable[str]], denylist_tags),
    )


# Config based on two sources + manual OSM wiki check
# 1. https://github.com/tyrasd/osm-polygon-features/blob/master/polygon-features.json
# 2. https://github.com/ideditor/id-area-keys/blob/main/areaKeys.json
OSM_WAY_POLYGON_CONFIG_RAW = {
    "all": [
        "allotments",
        "area:highway",
        "boundary",
        "bridge:support",
        "building:part",
        "building",
        "cemetery",
        "club",
        "craft",
        "demolished:building",
        "disused:amenity",
        "disused:leisure",
        "disused:shop",
        "healthcare",
        "historic",
        "industrial",
        "internet_access",
        "junction",
        "landuse",
        "leisure",
        "office",
        "place",
        "police",
        "polling_station",
        "public_transport",
        "residential",
        "ruins",
        "seamark:type",
        "shop",
        "sport",
        "telecom",
        "tourism",
    ],
    "allowlist": {
        "advertising": ["sculpture", "sign"],
        "aerialway": ["station"],
        "barrier": ["city_wall", "hedge", "wall", "toll_booth"],
        "highway": ["services", "rest_area", "platform"],
        "railway": ["station", "turntable", "roundhouse", "platform"],
        "waterway": ["riverbank", "dock", "boatyard", "dam", "fuel"],
    },
    "denylist": {
        "aeroway": ["jet_bridge", "parking_position", "taxiway", "no"],
        "amenity": ["bench", "weighbridge"],
        "attraction": ["river_rafting", "train", "water_slide", "boat_ride"],
        "emergency": ["designated", "destination", "no", "official", "private", "yes"],
        "geological": ["volcanic_caldera_rim", "fault"],
        "golf": ["cartpath", "hole", "path"],
        "indoor": ["corridor", "wall"],
        "man_made": [
            "yes",
            "breakwater",
            "carpet_hanger",
            "crane",
            "cutline",
            "dyke",
            "embankment",
            "goods_conveyor",
            "groyne",
            "pier",
            "pipeline",
            "torii",
            "video_wall",
        ],
        "military": ["trench"],
        "natural": [
            "bay",
            "cliff",
            "coastline",
            "ridge",
            "strait",
            "tree_row",
            "valley",
            "no",
            "arete",
        ],
        "piste:type": ["downhill", "hike", "ice_skate", "nordic", "skitour", "sled", "sleigh"],
        "playground": [
            "balancebeam",
            "rope_traverse",
            "stepping_stone",
            "stepping_post",
            "rope_swing",
            "climbing_slope",
        ],
        "power": ["cable", "line", "minor_line", "insulator", "busbar", "bay", "portal"],
    },
}
