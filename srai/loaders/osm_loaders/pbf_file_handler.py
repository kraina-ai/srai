"""
PBF File Handler.

This module contains a handler capable of parsing a PBF file into a GeoDataFrame.
"""
import json
import multiprocessing
import secrets
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import osmium
import osmium.osm
import psutil
from osmium.osm.types import T_obj
from pygeos import from_wkt
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.constants import FEATURES_INDEX
from srai.db import get_duckdb_connection
from srai.loaders.osm_loaders.filters._typing import osm_tags_type

if TYPE_CHECKING:
    import os
    from multiprocessing.connection import Connection

    import duckdb

IGNORED_TAGS = ["created_by", "converted_by", "source", "time", "ele", "attribution"]
PROGRESS_BAR_UPDATE_RESOLUTION = 5_000


def read_features_from_pbf_files(
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    tags: Optional[osm_tags_type] = None,
    region_id: str = "OSM",
    filter_region_geometry: Optional[BaseGeometry] = None,
) -> "duckdb.DuckDBPyRelation":
    """
    Get features from a list of PBF files.

    Function parses multiple PBF files and returns a single duckdb relation with parsed
    OSM objects.

    This function is a dedicated wrapper around the inherited function `apply_file`.

    Args:
        file_paths (Sequence[Union[str, os.PathLike[str]]]): List of paths to `*.osm.pbf`
            files to be parsed.
        tags (osm_tags_type, optional): A dictionary specifying which tags to download.
            The keys should be OSM tags (e.g. `building`, `amenity`).
            The values should either be `True` for retrieving all objects with the tag,
            string for retrieving a single tag-value pair
            or list of strings for retrieving all values specified in the list.
            `tags={'leisure': 'park}` would return parks from the area.
            `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
            would return parks, all amenity types, bakeries and bicycle shops.
            If `None`, handler will allow all of the tags to be parsed. Defaults to `None`.
        region_id (str, optional): Region name to be set in progress bar.
            Defaults to "OSM".
        filter_region_geometry (BaseGeometry, optional): Geometry used to filter features
            from the file. If not provided, features won't be filtered. Defaults to `None`.

    Returns:
        duckdb.DuckDBPyRelation: Relation with OSM features.
    """
    features, columns = _run_pbf_processing_in_parallel(
        file_paths=file_paths, region_id=region_id, tags=tags
    )

    if tags:
        columns.update(tags.keys())

    if not features:
        features = [{FEATURES_INDEX: None, "wkt": None}]

    temp_file = tempfile.NamedTemporaryFile(mode="w+")
    for feature in tqdm(features, desc=f"[{region_id}] Saving features to temp file"):
        jout = json.dumps(feature) + "\n"
        temp_file.write(jout)
    temp_file.flush()

    features_relation = _parse_raw_df_to_duckdb(temp_file.name, columns, filter_region_geometry)

    return features_relation


class MultiProcessingHandler(osmium.SimpleHandler):  # type: ignore
    """TODO."""

    def __init__(
        self,
        tags: Optional[osm_tags_type],
        pool_size: int,
        pool_index: int,
        features_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
        bar_queue: "multiprocessing.Queue[Optional[int]]",
    ) -> None:
        """TODO."""
        super().__init__()
        self.processing_counter = 0
        self.parsed_counter = 0
        self.pool_size = pool_size
        self.pool_index = pool_index
        self.features_queue = features_queue
        self.bar_queue = bar_queue

        self.filter_tags = tags
        if self.filter_tags:
            self.filter_tags_keys = set(self.filter_tags.keys())
        else:
            self.filter_tags_keys = set()

        self.wktfab = osmium.geom.WKTFactory()

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
            osm_object=node, osm_type="node", parse_to_wkt_function=self.wktfab.create_point
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
            osm_object=way, osm_type="way", parse_to_wkt_function=self.wktfab.create_linestring
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
            parse_to_wkt_function=self.wktfab.create_multipolygon,
            osm_id=area.orig_id(),
        )

    def _parse_osm_object(
        self,
        osm_object: osmium.osm.OSMObject[T_obj],
        osm_type: str,
        parse_to_wkt_function: Callable[..., str],
        osm_id: Optional[int] = None,
    ) -> None:
        """Parse OSM object into a feature with geometry and tags if it matches given criteria."""
        self.processing_counter += 1

        if self.processing_counter % self.pool_size == self.pool_index:
            self.parsed_counter += 1

            if self.parsed_counter % PROGRESS_BAR_UPDATE_RESOLUTION == 0:
                self.bar_queue.put_nowait(PROGRESS_BAR_UPDATE_RESOLUTION)

            if osm_id is None:
                osm_id = osm_object.id

            full_osm_id = f"{osm_type}/{osm_id}"

            matching_tags = self._get_matching_tags(osm_object)
            if matching_tags:
                wkt = self._get_osm_geometry(osm_object, parse_to_wkt_function)
                self._send_feature(full_osm_id=full_osm_id, matching_tags=matching_tags, wkt=wkt)

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

        matching_tags = {
            tag_key: tag_value
            for tag_key, tag_value in matching_tags.items()
            if tag_key not in IGNORED_TAGS
        }

        return matching_tags

    def _get_osm_geometry(
        self, osm_object: osmium.osm.OSMObject[T_obj], parse_to_wkt_function: Callable[..., str]
    ) -> Optional[str]:
        """Get geometry from currently parsed OSM object."""
        wkt = None
        try:
            wkt = parse_to_wkt_function(osm_object)
            from_wkt(wkt)
        except Exception as ex:
            wkt = None
            message = str(ex)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        return wkt

    def _send_feature(
        self, full_osm_id: str, matching_tags: Dict[str, str], wkt: Optional[str]
    ) -> None:
        """
        Add OSM feature to cache or update existing one based on ID.

        Some of the `way` features are parsed twice, in form of `LineStrings` and `Polygons` /
        `MultiPolygons`. Additional check ensures that a closed geometry will be always preffered
        over a `LineString`.
        """
        if wkt is not None:
            self.features_queue.put_nowait(
                {
                    FEATURES_INDEX: full_osm_id,
                    "wkt": wkt,
                    **matching_tags,
                }
            )


def _run_pbf_processing_in_parallel(
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    region_id: str,
    tags: Optional[osm_tags_type],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    number_of_cpus = psutil.cpu_count()
    pool_size = max(1, number_of_cpus - 2)  # removing 2 workers for Manager and progress bar

    bar_queue: "multiprocessing.Queue[Optional[int]]" = multiprocessing.Queue()
    features_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]" = multiprocessing.Queue()
    receive_connection, send_connection = multiprocessing.Pipe()

    bar_process = multiprocessing.Process(
        target=_update_progress_bar, args=(bar_queue, region_id), daemon=True
    )
    features_process = multiprocessing.Process(
        target=_process_features_queue, args=(features_queue, send_connection), daemon=True
    )
    processes = [
        multiprocessing.Process(
            target=_process_pbf,
            args=(file_paths, tags, pool_index, pool_size, features_queue, bar_queue),
        )
        for pool_index in range(pool_size)
    ]

    try:
        bar_process.start()
        features_process.start()

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        for p in processes:
            p.close()

        features_queue.put_nowait(None)
        bar_queue.put_nowait(None)

        received_result: Tuple[List[Dict[str, Any]], Set[str]] = receive_connection.recv()
        features_process.join()
        features_process.close()
    except:
        for p in processes:
            p.terminate()
        features_process.terminate()
        bar_process.terminate()
        raise

    return received_result


def _update_progress_bar(bar_queue: "multiprocessing.Queue[Optional[int]]", region_id: str) -> None:
    """Update shared progress bar for all processes."""
    with tqdm(desc=f"[{region_id}] Parsing pbf file") as pbar:
        while True:
            x = bar_queue.get()
            if x is None:
                break
            pbar.update(x)


def _process_features_queue(
    features_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]", send_connection: "Connection"
) -> None:
    """
    Add OSM features to cache or update existing ones based on the ID.

    Some of the `way` features are parsed twice, in form of `LineStrings` and `Polygons` /
    `MultiPolygons`. Additional check ensures that a closed geometry will be always preffered over a
    `LineString`.

    Sends back a list of parsed features and a set of columns names.
    """
    features_dict = {}
    columns: Set[str] = set()
    while True:
        feature: Optional[Dict[str, Any]] = features_queue.get()
        if feature is None:
            break

        full_osm_id = feature.pop(FEATURES_INDEX)
        wkt = feature.pop("wkt")
        if full_osm_id not in features_dict:
            features_dict[full_osm_id] = {
                FEATURES_INDEX: full_osm_id,
                "wkt": wkt,
            }

        if wkt.startswith(("POLYGON(", "MULTIPOLYGON(")):
            features_dict[full_osm_id]["wkt"] = wkt

        features_dict[full_osm_id].update(feature)
        columns.update(feature.keys())

    send_connection.send((list(features_dict.values()), columns))


def _process_pbf(
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    tags: Optional[osm_tags_type],
    pool_index: int,
    pool_size: int,
    features_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
    bar_queue: "multiprocessing.Queue[Optional[int]]",
) -> None:
    """Process PBF file using MultiProcessingHandler."""
    simple_handler = MultiProcessingHandler(tags, pool_size, pool_index, features_queue, bar_queue)
    for file_path in file_paths:
        simple_handler.apply_file(file_path)

    bar_queue.put_nowait(simple_handler.parsed_counter % PROGRESS_BAR_UPDATE_RESOLUTION)


def _parse_raw_df_to_duckdb(
    json_file_path: str, columns: Set[str], filter_region_geometry: Optional[BaseGeometry] = None
) -> "duckdb.DuckDBPyRelation":
    """Upload accumulated raw data into a relation and parse geometries."""
    relation_id = secrets.token_hex(nbytes=16)
    relation_name = f"features_{relation_id}"
    query = """
    SELECT * FROM (
        SELECT
            * EXCLUDE (wkt),
            ST_GeomFromText(wkt) geometry
        FROM read_json(
            '{json_file_name}',
            lines=true,
            json_format='records',
            columns={{ "feature_id": 'VARCHAR', "wkt": 'VARCHAR', {columns_definition} }}
        )
        WHERE feature_id IS NOT NULL AND wkt IS NOT NULL
    )
    """
    columns_definition = ", ".join(f"\"{column}\": 'VARCHAR'" for column in sorted(columns))
    filled_query = query.format(
        json_file_name=json_file_path, columns_definition=columns_definition
    )
    features_relation = get_duckdb_connection().sql(filled_query).set_alias(relation_name)

    if filter_region_geometry is not None:
        region_geometry_wkt = filter_region_geometry.wkt
        features_relation = features_relation.filter(
            f"ST_Intersects(geometry, ST_GeomFromText('{region_geometry_wkt}'))"
        )

    features_relation.to_table(relation_name)

    return get_duckdb_connection().table(relation_name)
