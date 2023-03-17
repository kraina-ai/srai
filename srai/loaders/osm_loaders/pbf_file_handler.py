"""
PBF File Handler.

This module contains a handler capable of parsing a PBF file into a GeoDataFrame.
"""
import os
from typing import Any, Callable, Dict, Optional, Sequence, Union

import geopandas as gpd
import osmium
import osmium.osm
import shapely.wkb as wkblib
from osmium.osm.types import T_obj
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.loaders.osm_loaders.filters.osm_tags_type import osm_tags_type
from srai.utils.constants import FEATURES_INDEX, WGS84_CRS


class PbfFileHandler(osmium.SimpleHandler):  # type: ignore
    """
    PbfFileHandler.

    PBF(Protocolbuffer Binary Format)[1] file handler is a wrapper around
    a `SimpleHandler`[2] from the `pyosmium`[3] library capable of parsing an `*.osm.pbf` file.

    Handler requires tags filter to only unpack required objects and additionally can use
    a geometry to filter only intersecting objects.

    Handler inherits functions from the `SimpleHandler`, such as `apply_file` and `apply_buffer`
    but it's discouraged to use them on your own, and instead use dedicated `get_features_gdf`
    function.

    References:
        1. https://wiki.openstreetmap.org/wiki/PBF_Format
        2. https://docs.osmcode.org/pyosmium/latest/ref_osmium.html#osmium.SimpleHandler
        3. https://osmcode.org/pyosmium/
    """

    _PBAR_FORMAT = "[{}] Parsing pbf file #{}"

    def __init__(self, tags: osm_tags_type, region_geometry: Optional[BaseGeometry] = None) -> None:
        """
        Initialize PbfFileHandler.

        Args:
            tags (osm_tags_type): A dictionary
                specifying which tags to download.
                The keys should be OSM tags (e.g. `building`, `amenity`).
                The values should either be `True` for retrieving all objects with the tag,
                string for retrieving a single tag-value pair
                or list of strings for retrieving all values specified in the list.
                `tags={'leisure': 'park}` would return parks from the area.
                `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
                would return parks, all amenity types, bakeries and bicycle shops.
            region_geometry (BaseGeometry, optional): Region which can be used to filter only
                intersecting OSM objects. Defaults to None.
        """
        super(PbfFileHandler, self).__init__()
        self.filter_tags = tags
        self.filter_tags_keys = set(self.filter_tags.keys())
        self.region_geometry = region_geometry
        self.wkbfab = osmium.geom.WKBFactory()

    def get_features_gdf(
        self, file_paths: Sequence[Union[str, "os.PathLike[str]"]], region_id: str = "OSM"
    ) -> gpd.GeoDataFrame:
        """
        Get features GeoDataFrame from a list of PBF files.

        Function parses multiple PBF files and returns a single GeoDataFrame with parsed
        OSM objects.

        This function is a dedicated wrapper around the inherited function `apply_file`.

        Args:
            file_paths (Sequence[Union[str, os.PathLike[str]]]): List of paths to `*.osm.pbf`
                files to be parsed.
            region_id (str, optional): Region name to be set in progress bar.
                Defaults to "OSM".

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with OSM features.
        """
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
        """
        Implementation of the required `node` function.

        See [1] for more information.

        Args:
            node (osmium.osm.Node): Node to be parsed.

        References:
            1. https://docs.osmcode.org/pyosmium/latest/ref_osm.html#osmium.osm.Node
        """
        self._parse_osm_object(
            osm_object=node, osm_type="node", parse_to_wkb_function=self.wkbfab.create_point
        )

    def way(self, way: osmium.osm.Way) -> None:
        """
        Implementation of the required `way` function.

        See [1] for more information.

        Args:
            way (osmium.osm.Way): Way to be parsed.

        References:
            1. https://docs.osmcode.org/pyosmium/latest/ref_osm.html#osmium.osm.Way
        """
        self._parse_osm_object(
            osm_object=way, osm_type="way", parse_to_wkb_function=self.wkbfab.create_linestring
        )

    def area(self, area: osmium.osm.Area) -> None:
        """
        Implementation of the required `area` function.

        See [1] for more information.

        Args:
            area (osmium.osm.Area): Area to be parsed.

        References:
            1. https://docs.osmcode.org/pyosmium/latest/ref_osm.html#osmium.osm.Area
        """
        self._parse_osm_object(
            osm_object=area,
            osm_type="way" if area.from_way() else "relation",
            parse_to_wkb_function=self.wkbfab.create_multipolygon,
            osm_id=area.orig_id(),
        )

    def _clear_cache(self) -> None:
        """Clear memory from accumulated features."""
        self.features_cache: Dict[str, Dict[str, Any]] = {}

    def _parse_osm_object(
        self,
        osm_object: osmium.osm.OSMObject[T_obj],
        osm_type: str,
        parse_to_wkb_function: Callable[..., str],
        osm_id: Optional[int] = None,
    ) -> None:
        """Parse OSM object into a feature with geometry and tags if it matches given criteria."""
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
        """Find matching tags between provided filter and currently parsed OSM object."""
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
