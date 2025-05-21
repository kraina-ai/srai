"""Constants used across the project."""

import os
from typing import Literal, get_args

WGS84_CRS = "EPSG:4326"

REGIONS_INDEX_TYPE = Literal["region_id"]
FEATURES_INDEX_TYPE = Literal["feature_id"]
REGIONS_INDEX = get_args(REGIONS_INDEX_TYPE)[0]
FEATURES_INDEX = get_args(FEATURES_INDEX_TYPE)[0]

GEOMETRY_COLUMN = "geometry"

FORCE_TERMINAL = os.getenv("FORCE_TERMINAL_MODE", "false").lower() == "true"
