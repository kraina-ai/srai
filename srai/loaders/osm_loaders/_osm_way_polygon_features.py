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
        "historic",
        "landuse",
        "leisure",
        "place",
        "tourism",
        "craft",
        "area:highway",
        "office",
        "building",
        "building:part",
        "shop",
        "boundary",
        "ruins",
        "public_transport",
        "residential",
        "demolished:building",
        "cemetery",
        "bridge:support",
        "club",
        "disused:amenity",
        "polling_station",
        "internet_access",
        "telecom",
        "police",
        "junction",
        "disused:shop",
        "seamark:type",
        "industrial",
        "allotments",
        "healthcare",
    ],
    "allowlist": {
        "waterway": ["riverbank", "dock", "boatyard", "dam", "fuel"],
        "barrier": ["city_wall", "hedge", "wall", "toll_booth"],
        "highway": ["services", "rest_area"],
        "railway": ["station", "turntable", "roundhouse", "platform"],
        "aerialway": ["station"],
        "advertising": ["sculpture", "sign"],
    },
    "denylist": {
        "aeroway": ["jet_bridge", "parking_position", "taxiway", "no"],
        "power": ["cable", "line", "minor_line", "insulator", "busbar", "bay", "portal"],
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
        "amenity": ["bench", "weighbridge"],
        "attraction": ["river_rafting", "train", "water_slide", "boat_ride"],
        "playground": [
            "balancebeam",
            "rope_traverse",
            "stepping_stone",
            "stepping_post",
            "rope_swing",
            "climbing_slope",
        ],
        "piste:type": ["downhill", "hike", "ice_skate", "nordic", "skitour", "sled", "sleigh"],
        "emergency": ["designated", "destination", "no", "official", "private", "yes"],
    },
}

OSM_WAY_POLYGON_CONFIG = parse_dict_to_config_object(OSM_WAY_POLYGON_CONFIG_RAW)
