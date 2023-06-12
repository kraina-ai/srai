"""TODO."""
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

OPENSTREETMAP_FR_POLYGONS_INDEX = "https://download.openstreetmap.fr/polygons"
OPENSTREETMAP_FR_EXTRACTS_INDEX = "https://download.openstreetmap.fr/extracts"

OPENSTREETMAP_FR_INDEX_GDF: Optional[gpd.GeoDataFrame] = None


@dataclass
class OpenstreetmapExtract:
    """TODO."""

    id: str
    url: str
    geometry: BaseGeometry
