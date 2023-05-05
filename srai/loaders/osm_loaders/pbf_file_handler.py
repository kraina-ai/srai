# flake8: noqa
# type: ignore
"""
PBF File Handler.

This module contains a handler capable of parsing a PBF file into a GeoDataFrame.
"""
import multiprocessing
import queue
import warnings
from multiprocessing.pool import AsyncResult
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import geopandas as gpd
import osmium
import osmium.osm
import shapely.wkb as wkblib
from osmium.osm.types import T_obj
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.constants import FEATURES_INDEX
from srai.loaders.osm_loaders.filters._typing import osm_tags_type

if TYPE_CHECKING:
    import os


class PbfFileHandler:
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
        tags: Optional[osm_tags_type] = None,
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
        self.tags = tags
        # self.filter_tags = tags
        # if self.filter_tags:
        #     self.filter_tags_keys = set(self.filter_tags.keys())
        # else:
        #     self.filter_tags_keys = set()
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

        Args:
            file_paths (Sequence[Union[str, os.PathLike[str]]]): List of paths to `*.osm.pbf`
                files to be parsed.
            region_id (str, optional): Region name to be set in progress bar.
                Defaults to "OSM".

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with OSM features.
        """
        if self.features_count is None:
            self.features_count = self.CountingPbfFileHandler().count_features(
                file_paths, region_id
            )

        # generated_regions.extend(
        #     process_map(
        #         create_regions_func,
        #         region_ids,
        #         desc="Generating regions",
        #         max_workers=num_of_multiprocessing_workers,
        #         chunksize=ceil(total_regions / (4 * num_of_multiprocessing_workers)),
        #     )
        # )
        bar_queue = multiprocessing.Manager().Queue()
        bar_process = multiprocessing.Process(
            target=_update_bar, args=(bar_queue, self.features_count), daemon=True
        )
        bar_process.start()

        # pool_size = multiprocessing.cpu_count()
        pool_size = 2
        pool = multiprocessing.Pool(pool_size)
        pool_results: List[AsyncResult[Any]] = []
        for pool_index in range(pool_size):
            # print(pool_index)

            # features_count: int,
            # tags: Optional[osm_tags_type],
            # region_geometry: Optional[BaseGeometry],
            # pool_size: int,
            # pool_index: int,
            # file_paths: Sequence[Union[str, "os.PathLike[str]"]],
            # region_id: str,
            # bar_queue: Any,

            async_result = pool.apply_async(
                _process_pbf,
                [
                    self.features_count,
                    self.tags,
                    self.region_geometry,
                    pool_size,
                    pool_index,
                    file_paths,
                    region_id,
                    bar_queue,
                ],
            )
            pool_results.append(async_result)
        pool.close()
        pool.join()
        bar_process.terminate()
        print(pool_results)
        print([result.get() for result in pool_results])

        # features_gdf = (
        #     gpd.GeoDataFrame(data=[result.get() for result in pool_results])
        #     .set_crs(WGS84_CRS)
        #     .set_index(FEATURES_INDEX)
        # )

        # return features_gdf

    class CountingPbfFileHandler(osmium.SimpleHandler):
        def count_features(
            self, file_paths: Sequence[Union[str, "os.PathLike[str]"]], region_id: str = "OSM"
        ) -> int:
            with tqdm(desc=f"[{region_id}] Counting pbf features") as self.pbar:
                self.counting_features = True
                self.features_count = 0
                for path in file_paths:
                    self.apply_file(path)
                self.pbar.update(n=self.features_count % 100_000)
                self.counting_features = False
            return self.features_count

        def node(self, node: osmium.osm.Node) -> None:
            self._count_feature()

        def way(self, way: osmium.osm.Way) -> None:
            self._count_feature()

        def area(self, area: osmium.osm.Area) -> None:
            self._count_feature()

        def _count_feature(self) -> None:
            self.features_count += 1
            if self.features_count % 100_000 == 0:
                self.pbar.update(n=100_000)


class _MultithreadingPbfFileHandler(osmium.SimpleHandler):
    def __init__(
        self,
        pool_size: int,
        pool_index: int,
        features_count: Optional[int],
        tags: Optional[osm_tags_type],
        region_geometry: Optional[BaseGeometry],
        bar_queue,
    ) -> None:
        super().__init__()
        self.processing_counter = 0
        self.pool_size = pool_size
        self.pool_index = pool_index
        self.filter_tags = tags
        if self.filter_tags:
            self.filter_tags_keys = set(self.filter_tags.keys())
        else:
            self.filter_tags_keys = set()
        self.region_geometry = region_geometry
        self.wkbfab = osmium.geom.WKBFactory()
        self.features_cache: Dict[str, Dict[str, Any]] = {}
        self.features_count = features_count
        self.bar_queue = bar_queue

        # print(self.processing_counter, self.pool_size, self.pool_index)

    # def get_raw_features(
    #     self, file_paths: Sequence[Union[str, "os.PathLike[str]"]], region_id: str = "OSM"
    # ) -> List[Dict[str, Any]]:
    #     """TODO."""
    #     self._clear_cache()
    #     print(file_paths, self.pool_index)
    #     with self.lock:
    #         # self.pbar = tqdm(desc="Parsing pbf file", total=self.features_count, position=self.pool_index)
    #         self.pbar = tqdm(desc="Parsing pbf file", total=self.features_count)

    #     for path_no, path in enumerate(file_paths):
    #         print(path_no, path)
    #         self.path_no = path_no + 1
    #         description = PbfFileHandler._PBAR_FORMAT.format(region_id, str(self.path_no))
    #         self.pbar.set_description(description)
    #         self.apply_file(path)

    #     with self.lock:
    #         self.pbar.close()

    #     print("cache", len(self.features_cache))

    # return self.features_cache.values()

    def _clear_cache(self) -> None:
        """Clear memory from accumulated features."""
        self.features_cache.clear()

    def node(self, node: osmium.osm.Node) -> None:
        """
        Implementation of the required `node` function.

        See [1] for more information.

        Args:
            node (osmium.osm.Node): Node to be parsed.

        References:
            1. https://docs.osmcode.org/pyosmium/latest/ref_osm.html#osmium.osm.Node
        """
        print("node")
        # self._parse_osm_object(
        #     osm_object=node, osm_type="node", parse_to_wkb_function=self.wkbfab.create_point
        # )

    def way(self, way: osmium.osm.Way) -> None:
        """
        Implementation of the required `way` function.

        See [1] for more information.

        Args:
            way (osmium.osm.Way): Way to be parsed.

        References:
            1. https://docs.osmcode.org/pyosmium/latest/ref_osm.html#osmium.osm.Way
        """
        print("way")
        # self._parse_osm_object(
        #     osm_object=way, osm_type="way", parse_to_wkb_function=self.wkbfab.create_linestring
        # )

    def area(self, area: osmium.osm.Area) -> None:
        """
        Implementation of the required `area` function.

        See [1] for more information.

        Args:
            area (osmium.osm.Area): Area to be parsed.

        References:
            1. https://docs.osmcode.org/pyosmium/latest/ref_osm.html#osmium.osm.Area
        """
        print("area")
        # self._parse_osm_object(
        #     osm_object=area,
        #     osm_type="way" if area.from_way() else "relation",
        #     parse_to_wkb_function=self.wkbfab.create_multipolygon,
        #     osm_id=area.orig_id(),
        # )

    def _parse_osm_object(
        self,
        osm_object: osmium.osm.OSMObject[T_obj],
        osm_type: str,
        parse_to_wkb_function: Callable[..., str],
        osm_id: Optional[int] = None,
    ) -> None:
        """Parse OSM object into a feature with geometry and tags if it matches given criteria."""
        self.processing_counter += 1
        print(self.processing_counter, self.pool_size, self.processing_counter % self.pool_size)

        self.bar_queue.put_nowait(1)

        # if self.processing_counter % self.pool_size == self.pool_index:
        #     # self.pbar.n = self.processing_counter
        #     if osm_id is None:
        #         osm_id = osm_object.id

        #     full_osm_id = f"{osm_type}/{osm_id}"

        #     matching_tags = self._get_matching_tags(osm_object)
        #     if matching_tags:
        #         geometry = self._get_osm_geometry(osm_object, parse_to_wkb_function)
        #         self._add_feature_to_cache(
        #             full_osm_id=full_osm_id, matching_tags=matching_tags, geometry=geometry
        #         )

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
        except RuntimeError as ex:
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


def _update_bar(q: queue.Queue, total: int):
    pbar = tqdm(total=total)

    while True:
        if not q.empty():
            x = q.get()
            pbar.update(x)


def _process_pbf(
    features_count: int,
    tags: Optional[osm_tags_type],
    region_geometry: Optional[BaseGeometry],
    pool_size: int,
    pool_index: int,
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    region_id: str,
    bar_queue: queue.Queue,
):
    # features = simple_handler.get_raw_features(file_paths, region_id)

    # print(bar_queue, file_paths, pool_index, pool_size)

    simple_handler = _MultithreadingPbfFileHandler(
        pool_size, pool_index, features_count, tags, region_geometry, bar_queue
    )

    for path_no, path in enumerate(file_paths):
        print(path_no, path)
        # simple_handler.apply_file(path)
        bar_queue.put_nowait(10)

    features = simple_handler.features_cache.values()

    print("cache", len(features))
    return features
