"""
Geofabrik layers filter.

This module contains the grouped OSM tags filter that is defined by Geofabrik [1].
Based on the document version `0.7.12`.

Note: not all definitions from the document are implemented, such as boundaries or places.

References:
    1. https://www.geofabrik.de/data/geofabrik-osm-gis-standard-0.7.pdf
"""
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter

GEOFABRIK_LAYERS: GroupedOsmTagsFilter = {
    "public": {
        "amenity": [
            "police",
            "fire_station",
            "post_box",
            "post_office",
            "telephone",
            "library",
            "townhall",
            "courthouse",
            "prison",
            "embassy",
            "community_centre",
            "nursing_home",
            "arts_centre",
            "grave_yard",
            "marketplace",
            "recycling",
            "public_building",
        ],
        "office": ["diplomatic"],
        "landuse": ["cemetery"],
    },
    "education": {
        "amenity": [
            "university",
            "school",
            "kindergarten",
            "college",
        ]
    },
    "health": {
        "amenity": [
            "pharmacy",
            "hospital",
            "clinic",
            "doctors",
            "dentist",
            "veterinary",
        ]
    },
    "leisure": {
        "amenity": [
            "theatre",
            "nightclub",
            "cinema",
            "swimming_pool",
            "theatre",
            "theatre",
        ],
        "leisure": [
            "park",
            "playground",
            "dog_park",
            "sports_centre",
            "swimming_pool",
            "water_park",
            "golf_course",
            "stadium",
            "ice_rink",
        ],
        "sport": [
            "swimming",
            "tennis",
        ],
    },
    "catering": {
        "amenity": [
            "restaurant",
            "fast_food",
            "cafe",
            "pub",
            "bar",
            "food_court",
            "biergarten",
        ]
    },
    "accommodation": {
        "tourism": [
            "hotel",
            "motel",
            "bed_and_breakfast",
            "guest_house",
            "hostel",
            "chalet",
            "camp_site",
            "alpine_hut",
            "caravan_site",
        ],
        "amenity": ["shelter"],
    },
    "shopping": {
        "shop": [
            "supermarket",
            "bakery",
            "kiosk",
            "mall",
            "department_store",
            "general",
            "convenience",
            "clothes",
            "florist",
            "chemist",
            "books",
            "butcher",
            "shoes",
            "alcohol",
            "beverages",
            "optician",
            "jewelry",
            "gift",
            "sports",
            "stationery",
            "outdoor",
            "mobile_phone",
            "toys",
            "newsagent",
            "greengrocer",
            "beauty",
            "video",
            "car",
            "bicycle",
            "doityourself",
            "hardware",
            "furniture",
            "computer",
            "garden_centre",
            "hairdresser",
            "car_repair",
            "travel_agency",
            "laundry",
            "dry_cleaning",
        ],
        "amenity": ["car_rental", "car_wash", "car_sharing", "bicycle_rental", "vending_machine"],
        "vending": [
            "cigarettes",
            "parking_tickets",
        ],
    },
    "money": {"amenity": ["bank", "atm"]},
    "tourism": {
        "tourism": [
            "information",
            "attraction",
            "museum",
            "artwork",
            "picnic_site",
            "viewpoint",
            "zoo",
            "theme_park",
        ],
        "historic": [
            "monument",
            "memorial",
            "castle",
            "ruins",
            "archaeological_site",
            "wayside_cross",
            "wayside_shrine",
            "battlefield",
            "fort",
        ],
    },
    "miscpoi": {
        "amenity": [
            "toilets",
            "bench",
            "drinking_water",
            "fountain",
            "hunting_stand",
            "waste_basket",
            "emergency_phone",
            "fire_hydrant",
        ],
        "man_made": [
            "surveillance",
            "tower",
            "water_tower",
            "windmill",
            "lighthouse",
            "wastewater_plant",
            "water_well",
            "watermill",
            "water_works",
        ],
        "emergency": ["phone", "fire_hydrant"],
        "highway": ["emergency_access_point"],
    },
    "pofw": {"amenity": ["place_of_worship"]},
    "natural": {
        "natural": [
            "spring",
            "glacier",
            "peak",
            "cliff",
            "volcano",
            "tree",
            "mine",
            "cave_entrance",
            "beach",
        ]
    },
    "traffic": {
        "highway": [
            "traffic_signals",
            "mini_roundabout",
            "stop",
            "crossing",
            "ford",
            "motorway_junction",
            "turning_circle",
            "speed_camera",
            "street_lamp",
        ],
        "railway": ["level_crossing"],
    },
    "fuel_parking": {"amenity": ["fuel", "parking", "bicycle_parking"], "highway": ["services"]},
    "water_traffic": {
        "leisure": [
            "slipway",
            "marina",
        ],
        "man_made": ["pier"],
        "waterway": [
            "dam",
            "waterfall",
            "lock_gate",
            "weir",
        ],
    },
    "transport": {
        "railway": ["station", "halt", "tram_stop"],
        "public_transport": ["stop_position"],
        "highway": ["bus_stop"],
        "amenity": ["bus_station", "taxi", "ferry_terminal"],
        "aerialway": ["station"],
    },
    "air_traffic": {
        "amenity": ["airport"],
        "aeroway": [
            "aerodrome",
            "airfield",
            "aeroway",
            "helipad",
            "apron",
        ],
        "military": ["airfield"],
    },
    "major_roads": {
        "highway": [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
        ]
    },
    "minor_roads": {
        "highway": [
            "unclassified",
            "residential",
            "living_street",
            "pedestrian",
            "busway",
        ]
    },
    "highway_links": {
        "highway": [
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
        ]
    },
    "very_small_roads": {
        "highway": [
            "service",
            "track",
        ]
    },
    "paths_unsuitable_for_cars": {
        "highway": ["bridleway", "path", "cycleway", "footway", "steps"],
        "cycle": ["designated"],
        "horse": ["designated"],
        "foot": ["designated"],
    },
    "unkown_roads": {"highway": ["road"]},
    "railways": {
        "railway": [
            "rail",
            "light_rail",
            "subway",
            "tram",
            "monorail",
            "narrow_gauge",
            "miniature",
            "funicular",
            "rack",
        ],
        "aerialway": [
            "drag_lift",
            "chair_lift",
            "high_speed_chair_lift",
            "cable_car",
            "gondola",
            "goods",
            "platter",
            "t-bar",
            "j-bar",
            "magic_carpet",
            "zip_line",
            "rope_tow",
            "mixed_lift",
        ],
    },
    "waterways": {
        "waterway": [
            "river",
            "stream",
            "canal",
            "drain",
        ]
    },
    "buildings": {"building": True},
    "landuse": {
        "landuse": [
            "forest",
            "residential",
            "industrial",
            "cemetery",
            "allotments",
            "meadow",
            "commercial",
            "recreation_ground",
            "retail",
            "military",
            "quarry",
            "orchard",
            "vineyard",
            "scrub",
            "grass",
            "military",
            "farmland",
            "farmyard",
        ],
        "leisure": ["park", "common", "nature_reserve", "recreation_ground"],
        "natural": ["wood", "heath"],
        "boundary": ["national_park"],
    },
    "water": {
        "natural": ["water", "glacier", "wetland"],
        "landuse": ["reservoir"],
        "waterway": ["riverbank", "dock"],
    },
}
