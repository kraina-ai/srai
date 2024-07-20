"""
Base OSM groups filter.

This module contains the grouped OSM tags filter that was used in ARIC@SIGSPATIAL 2021 paper [1].

References:
    1. https://doi.org/10.1145/3486626.3493434
    1. https://arxiv.org/abs/2111.00990
"""

from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter

BASE_OSM_GROUPS_FILTER: GroupedOsmTagsFilter = {
    "water": {"natural": ["water", "bay", "beach", "coastline"], "waterway": ["riverbank"]},
    "aerialway": {
        "aerialway": [
            "cable_car",
            "gondola",
            "mixed_lift",
            "chair_lift",
            "drag_lift",
            "t-bar",
            "j-bar",
            "platter",
            "rope_tow",
            "magic_carpet",
            "zip_line",
            "goods",
            "station",
        ]
    },
    "airports": {"aeroway": ["aerodrome", "heliport", "spaceport"]},
    "sustenance": {
        "amenity": [
            "bar",
            "bbq",
            "biergarten",
            "cafe",
            "fast_food",
            "food_court",
            "ice_cream",
            "pub",
            "restaurant",
        ]
    },
    "education": {
        "amenity": [
            "college",
            "driving_school",
            "kindergarten",
            "language_school",
            "library",
            "toy_library",
            "music_school",
            "school",
            "university",
        ]
    },
    "transportation": {
        "amenity": [
            "bicycle_parking",
            "bicycle_repair_station",
            "bicycle_rental",
            "boat_rental",
            "boat_sharing",
            "car_rental",
            "car_sharing",
            "car_wash",
            "charging_station",
            "bus_stop",
            "ferry_terminal",
            "fuel",
            "motorcycle_parking",
            "parking",
            "taxi",
            "bus_station",
        ],
        "public_transport": ["station", "stop_position"],
        "railway": ["station", "subway_entrance", "tram_stop"],
        "building": ["train_station"],
        "highway": ["bus_stop"],
    },
    "finances": {"amenity": ["atm", "bank", "bureau_de_change"]},
    "healthcare": {
        "amenity": [
            "baby_hatch",
            "clinic",
            "dentist",
            "doctors",
            "hospital",
            "nursing_home",
            "pharmacy",
            "social_facility",
            "veterinary",
        ]
    },
    "culture_art_entertainment": {
        "amenity": [
            "arts_centre",
            "brothel",
            "casino",
            "cinema",
            "community_centre",
            "gambling",
            "nightclub",
            "planetarium",
            "public_bookcase",
            "social_centre",
            "stripclub",
            "studio",
            "theatre",
        ]
    },
    "other": {
        "amenity": [
            "animal_boarding",
            "animal_shelter",
            "childcare",
            "conference_centre",
            "courthouse",
            "crematorium",
            "embassy",
            "fire_station",
            "grave_yard",
            "internet_cafe",
            "marketplace",
            "monastery",
            "place_of_worship",
            "police",
            "post_office",
            "prison",
            "ranger_station",
            "refugee_site",
            "townhall",
        ]
    },
    "buildings": {
        "building": ["commercial", "industrial", "warehouse"],
        "office": True,
        "waterway": ["dock", "boatyard"],
    },
    "emergency": {"emergency": ["ambulance_station", "defibrillator", "landing_site"]},
    "historic": {
        "historic": [
            "aqueduct",
            "battlefield",
            "building",
            "castle",
            "church",
            "citywalls",
            "fort",
            "memorial",
            "monastery",
            "monument",
            "ruins",
            "tower",
        ]
    },
    "leisure": {
        "leisure": [
            "adult_gaming_centre",
            "amusement_arcade",
            "beach_resort",
            "common",
            "dance",
            "dog_park",
            "escape_game",
            "fitness_centre",
            "fitness_station",
            "garden",
            "hackerspace",
            "horse_riding",
            "ice_rink",
            "marina",
            "miniature_golf",
            "nature_reserve",
            "park",
            "pitch",
            "slipway",
            "sports_centre",
            "stadium",
            "summer_camp",
            "swimming_area",
            "swimming_pool",
            "track",
            "water_park",
        ],
        "amenity": ["public_bath", "dive_centre"],
    },
    "shops": {"shop": True},
    "sport": {"sport": True},
    "tourism": {"tourism": True},
    "greenery": {
        "leisure": ["park"],
        "natural": ["grassland", "scrub"],
        "landuse": [
            "grass",
            "allotments",
            "forest",
            "flowerbed",
            "meadow",
            "village_green",
            "grassland",
            "scrub",
            "garden",
            "park",
            "recreation_ground",
        ],
    },
}
