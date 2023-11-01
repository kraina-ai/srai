"""
PBF File Handler.

This module contains a handler capable of parsing a PBF file into a GeoDataFrame.
"""
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union, cast

import geopandas as gpd
import osmium
import osmium.osm
import shapely.wkb as wkblib
from osmium.osm.types import T_obj
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.constants import FEATURES_INDEX, WGS84_CRS
from srai.loaders.osm_loaders.filters import OsmTagsFilter

if TYPE_CHECKING:
    import os


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

    def __init__(
        self,
        tags: Optional[OsmTagsFilter] = None,
        region_geometry: Optional[BaseGeometry] = None,
    ) -> None:
        """
        Initialize PbfFileHandler.

        Args:
            tags (osm_tags_type, optional): A dictionary
                specifying which tags to download.
                The keys should be OSM tags (e.g. `building`, `amenity`).
                The values should either be `True` for retrieving all objects with the tag,
                string for retrieving a single tag-value pair
                or list of strings for retrieving all values specified in the list.
                `tags={'leisure': 'park}` would return parks from the area.
                `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
                would return parks, all amenity types, bakeries and bicycle shops.
                If `None`, handler will allow all of the tags to be parsed. Defaults to `None`.
            region_geometry (BaseGeometry, optional): Region which can be used to filter only
                intersecting OSM objects. Defaults to None.
        """
        super().__init__()
        self.filter_tags = tags
        if self.filter_tags:
            self.filter_tags_keys = set(self.filter_tags.keys())
        else:
            self.filter_tags_keys = set()
        self.region_geometry = region_geometry
        self.wkbfab = osmium.geom.WKBFactory()
        self.features_cache: Dict[str, Dict[str, Any]] = {}
        self.features_count: Optional[int] = None

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
        self._clear_cache()
        if self.features_count is None:
            self._count_features(file_paths, region_id)

        with tqdm(desc="Parsing pbf file", total=self.features_count) as self.pbar:
            for path_no, path in enumerate(file_paths):
                self.path_no = path_no + 1
                description = self._PBAR_FORMAT.format(region_id, str(self.path_no))
                self.pbar.set_description(description)
                self.apply_file(path)
            if self.features_cache:
                features_gdf = (
                    gpd.GeoDataFrame(data=self.features_cache.values())
                    .set_crs(WGS84_CRS)
                    .set_index(FEATURES_INDEX)
                )
            else:
                features_gdf = gpd.GeoDataFrame(
                    index=gpd.pd.Index(name=FEATURES_INDEX, data=[]), crs=WGS84_CRS, geometry=[]
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
        if self.counting_features:
            self._count_feature()
        else:
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
        if self.counting_features:
            self._count_feature()
        else:
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
        if self.counting_features:
            self._count_feature()
        else:
            self._parse_osm_object(
                osm_object=area,
                osm_type="way" if area.from_way() else "relation",
                parse_to_wkb_function=self.wkbfab.create_multipolygon,
                osm_id=area.orig_id(),
            )

    def _clear_cache(self) -> None:
        """Clear memory from accumulated features."""
        self.features_cache.clear()
        self.features_count = None

    def _count_features(
        self, file_paths: Sequence[Union[str, "os.PathLike[str]"]], region_id: str = "OSM"
    ) -> None:
        with tqdm(desc=f"[{region_id}] Counting pbf features") as self.pbar:
            self.counting_features = True
            self.features_count = 0
            for path in file_paths:
                self.apply_file(path)
            self.pbar.update(n=self.features_count % 100_000)
            self.counting_features = False

    def _count_feature(self) -> None:
        self.features_count = cast(int, self.features_count) + 1
        if self.features_count % 100_000 == 0:
            self.pbar.update(n=100_000)

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
            geometry = self._get_osm_geometry(osm_object, parse_to_wkb_function)
            self._add_feature_to_cache(
                full_osm_id=full_osm_id, matching_tags=matching_tags, geometry=geometry
            )

    def _get_matching_tags(self, osm_object: osmium.osm.OSMObject[T_obj]) -> Dict[str, str]:
        """
        Find matching tags between provided filter and currently parsed OSM object.

        If tags filter is `None`, it will copy all tags from the OSM object.
        """
        matching_tags: Dict[str, str] = {}

        if self.filter_tags:
            for tag_key in self.filter_tags_keys:
                if tag_key in osm_object.tags:
                    object_tag_value = osm_object.tags[tag_key]
                    filter_tag_value = self.filter_tags[tag_key]
                    if (
                        (isinstance(filter_tag_value, bool) and filter_tag_value)
                        or (
                            isinstance(filter_tag_value, str)
                            and object_tag_value == filter_tag_value
                        )
                        or (
                            isinstance(filter_tag_value, list)
                            and object_tag_value in filter_tag_value
                        )
                    ):
                        matching_tags[tag_key] = object_tag_value
        else:
            for tag in osm_object.tags:
                matching_tags[tag.k] = tag.v

        return matching_tags

    def _get_osm_geometry(
        self, osm_object: osmium.osm.OSMObject[T_obj], parse_to_wkb_function: Callable[..., str]
    ) -> BaseGeometry:
        """Get geometry from currently parsed OSM object."""
        geometry = None
        try:
            wkb = parse_to_wkb_function(osm_object)
            geometry = wkblib.loads(wkb, hex=True)
        except Exception as ex:
            message = str(ex)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        return geometry

    def _add_feature_to_cache(
        self, full_osm_id: str, matching_tags: Dict[str, str], geometry: Optional[BaseGeometry]
    ) -> None:
        """
        Add OSM feature to cache or update existing one based on ID.

        Some of the `way` features are parsed twice, in form of `LineStrings` and `Polygons` /
        `MultiPolygons`. Additional check ensures that a closed geometry will be always preffered
        over a `LineString`.
        """
        if geometry is not None and self._geometry_is_in_region(geometry):
            if full_osm_id not in self.features_cache:
                self.features_cache[full_osm_id] = {
                    FEATURES_INDEX: full_osm_id,
                    "geometry": geometry,
                }

            if isinstance(geometry, (Polygon, MultiPolygon)):
                self.features_cache[full_osm_id]["geometry"] = geometry

            self.features_cache[full_osm_id].update(matching_tags)

    def _geometry_is_in_region(self, geometry: BaseGeometry) -> bool:
        """Check if OSM geometry intersects with provided region."""
        return self.region_geometry is None or geometry.intersects(self.region_geometry)
