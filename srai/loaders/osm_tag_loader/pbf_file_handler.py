"""DOCSTRING TODO."""
import os
from typing import Any, Callable, Dict, Optional, Sequence, Union

import geopandas as gpd
import osmium
import osmium.osm
import shapely.wkb as wkblib
from osmium.osm.types import T_obj
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.loaders.osm_tag_loader.filters.osm_tags_type import osm_tags_type
from srai.utils.constants import FEATURES_INDEX, WGS84_CRS


class PbfFileHandler(osmium.SimpleHandler):  # type: ignore
    """DOCSTRING TODO."""

    _PBAR_FORMAT = "[{}] Parsing pbf file #{}"

    def __init__(self, tags: osm_tags_type, region_geometry: Optional[BaseGeometry] = None) -> None:
        """DOCSTRING TODO."""
        super(PbfFileHandler, self).__init__()
        self.filter_tags = tags
        self.filter_tags_keys = set(self.filter_tags.keys())
        self.region_geometry = region_geometry
        self.wkbfab = osmium.geom.WKBFactory()

    def get_features_gdf(
        self, file_paths: Sequence[Union[str, "os.PathLike[str]"]], region_id: str = "OSM"
    ) -> gpd.GeoDataFrame:
        """DOCSTRING TODO."""
        with tqdm(desc="Parsing pbf file") as self.pbar:
            self._clear_cache()
            for path_no, path in enumerate(file_paths):
                self.path_no = path_no + 1
                description = self._PBAR_FORMAT.format(region_id, str(self.path_no))
                self.pbar.set_description(description)
                self.apply_file(path)
            features_gdf = (
                gpd.GeoDataFrame(data=self.features_cache.values())
                .set_crs(WGS84_CRS)
                .set_index(FEATURES_INDEX)
            )
            self._clear_cache()
        return features_gdf

    def node(self, node: osmium.osm.Node) -> None:
        """DOCSTRING TODO."""
        self._parse_osm_object(
            osm_object=node, osm_type="node", parse_to_wkb_function=self.wkbfab.create_point
        )

    def way(self, way: osmium.osm.Way) -> None:
        """DOCSTRING TODO."""
        self._parse_osm_object(
            osm_object=way, osm_type="way", parse_to_wkb_function=self.wkbfab.create_linestring
        )

    def area(self, area: osmium.osm.Area) -> None:
        """DOCSTRING TODO."""
        self._parse_osm_object(
            osm_object=area,
            osm_type="way" if area.from_way() else "relation",
            parse_to_wkb_function=self.wkbfab.create_multipolygon,
            osm_id=area.orig_id(),
        )

    def _clear_cache(self) -> None:
        self.features_cache: Dict[str, Dict[str, Any]] = {}

    def _parse_osm_object(
        self,
        osm_object: osmium.osm.OSMObject[T_obj],
        osm_type: str,
        parse_to_wkb_function: Callable[..., str],
        osm_id: Optional[int] = None,
    ) -> None:
        self.pbar.update(n=1)

        if osm_id is None:
            osm_id = osm_object.id

        full_osm_id = f"{osm_type}/{osm_id}"

        matching_tags = self._get_matching_tags(osm_object)
        if matching_tags:
            wkb = parse_to_wkb_function(osm_object)
            geometry = wkblib.loads(wkb, hex=True)
            if self.region_geometry is None or geometry.intersects(self.region_geometry):
                if full_osm_id not in self.features_cache:
                    self.features_cache[full_osm_id] = {
                        FEATURES_INDEX: full_osm_id,
                        "geometry": geometry,
                    }
                self.features_cache[full_osm_id].update(matching_tags)

    def _get_matching_tags(self, osm_object: osmium.osm.OSMObject[T_obj]) -> Dict[str, str]:
        matching_tags: Dict[str, str] = {}

        for tag_key in self.filter_tags_keys:
            if tag_key in osm_object.tags:
                object_tag_value = osm_object.tags[tag_key]
                filter_tag_value = self.filter_tags[tag_key]
                if (
                    (isinstance(filter_tag_value, bool) and filter_tag_value)
                    or (isinstance(filter_tag_value, str) and object_tag_value == filter_tag_value)
                    or (isinstance(filter_tag_value, list) and object_tag_value in filter_tag_value)
                ):
                    matching_tags[tag_key] = object_tag_value

        return matching_tags
