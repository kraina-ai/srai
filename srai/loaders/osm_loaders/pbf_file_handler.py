"""
PBF File Handler.

This module contains a handler capable of parsing a PBF file into a GeoDataFrame.
"""
import multiprocessing
import secrets
import tempfile
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import orjson
import osmium
import osmium.osm
import psutil
import redislite
from osmium.osm.types import T_obj
from pygeos import from_wkt
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.constants import FEATURES_INDEX
from srai.db import get_duckdb_connection
from srai.loaders.osm_loaders.filters._typing import osm_tags_type

if TYPE_CHECKING:
    import os

    import duckdb

IGNORED_TAGS = ["created_by", "converted_by", "source", "time", "ele", "attribution"]
PARSED_PROGRESS_BAR_UPDATE_RESOLUTION = 10_000
LOADED_PROGRESS_BAR_UPDATE_RESOLUTION = 1_000
PROGRESS_BAR_FORMAT = "[{}] Processing PBF file (loaded: {})"


def read_features_from_pbf_files(
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    tags: Optional[osm_tags_type] = None,
    region_id: str = "OSM",
    filter_region_geometry: Optional[BaseGeometry] = None,
) -> "duckdb.DuckDBPyRelation":
    """
    Get features from a list of PBF files.

    Function parses provided PBF files and returns a single duckdb relation with loaded
    OSM objects.

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
    tmpdir = tempfile.TemporaryDirectory()
    temp_dir_name = tmpdir.name

    redis_db_file_name = (Path(temp_dir_name) / "redis.db").as_posix()
    json_file_name = (Path(temp_dir_name) / "features.json").as_posix()

    rdb = redislite.Redis(redis_db_file_name, charset="utf-8", decode_responses=True)
    rdb.flushdb()

    _run_pbf_processing_in_parallel(
        file_paths=file_paths,
        region_id=region_id,
        tags=tags,
        redis_db_file_name=redis_db_file_name,
    )

    columns: Set[str] = set()

    with open(json_file_name, "w+") as temp_file:
        for feature_id in tqdm(rdb.keys(), desc=f"[{region_id}] Saving features to temp file"):
            values = rdb.hgetall(feature_id)
            jout = orjson.dumps(values).decode("utf-8") + "\n"
            temp_file.write(jout)
            columns.update(values.keys())

    columns = columns.difference([FEATURES_INDEX, "wkt"])

    if tags:
        columns.update(tags.keys())

    features_relation = _parse_raw_df_to_duckdb(json_file_name, columns, filter_region_geometry)

    tmpdir.cleanup()

    return features_relation


class MultiProcessingHandler(osmium.SimpleHandler):  # type: ignore
    """TODO."""

    def __init__(
        self,
        tags: Optional[osm_tags_type],
        pool_size: int,
        pool_index: int,
        features_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
        bar_queue: "multiprocessing.Queue[Optional[Tuple[str, int]]]",
        redis_db_file_name: str,
    ) -> None:
        """TODO."""
        super().__init__()
        self.processing_counter = 0
        self.parsed_counter = 0
        self.loaded_counter = 0
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

        self.rdb = redislite.Redis(redis_db_file_name)

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

            if self.parsed_counter % PARSED_PROGRESS_BAR_UPDATE_RESOLUTION == 0:
                self.bar_queue.put_nowait(("parsed", PARSED_PROGRESS_BAR_UPDATE_RESOLUTION))

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
            tag_key.lower(): tag_value
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
            with self.rdb.lock(name=f"_lock_{full_osm_id}"):
                feature_exists = bool(self.rdb.exists(full_osm_id))

                if not feature_exists:
                    self.rdb.hmset(
                        full_osm_id,
                        {
                            FEATURES_INDEX: full_osm_id,
                            "wkt": wkt,
                        },
                    )
                    self.loaded_counter += 1

                    if self.loaded_counter % LOADED_PROGRESS_BAR_UPDATE_RESOLUTION == 0:
                        self.bar_queue.put_nowait(("loaded", LOADED_PROGRESS_BAR_UPDATE_RESOLUTION))

                if feature_exists and wkt.startswith(("POLYGON(", "MULTIPOLYGON(")):
                    self.rdb.hset(full_osm_id, "wkt", wkt)

                self.rdb.hmset(full_osm_id, matching_tags)


def _run_pbf_processing_in_parallel(
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    region_id: str,
    tags: Optional[osm_tags_type],
    redis_db_file_name: str,
) -> None:
    number_of_cpus = psutil.cpu_count()
    pool_size = max(1, number_of_cpus - 2)  # removing 2 workers for progress bar and redis instance

    bar_queue: "multiprocessing.Queue[Optional[Tuple[str, int]]]" = multiprocessing.Queue()
    features_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]" = multiprocessing.Queue()

    bar_process = multiprocessing.Process(
        target=_update_progress_bar, args=(bar_queue, region_id), daemon=True
    )
    processes = [
        multiprocessing.Process(
            target=_process_pbf,
            args=(
                file_paths,
                tags,
                pool_index,
                pool_size,
                features_queue,
                bar_queue,
                redis_db_file_name,
            ),
        )
        for pool_index in range(pool_size)
    ]

    try:
        bar_process.start()

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        for p in processes:
            p.close()

        bar_queue.put_nowait(None)
        bar_process.join()
        bar_process.close()
    except:
        for p in processes:
            p.terminate()
        bar_process.terminate()
        raise


def _update_progress_bar(
    bar_queue: "multiprocessing.Queue[Optional[Tuple[str, int]]]", region_id: str
) -> None:
    """Update shared progress bar for all processes."""
    loaded_features = 0

    main_bar = tqdm(desc=PROGRESS_BAR_FORMAT.format(region_id, loaded_features))
    while True:
        x = bar_queue.get()
        if x is None:
            break
        bar_type, update_value = x

        if bar_type == "parsed":
            main_bar.update(update_value)
        elif bar_type == "loaded":
            loaded_features += update_value

        main_bar.set_description_str(desc=PROGRESS_BAR_FORMAT.format(region_id, loaded_features))

    main_bar.close()


# def _process_features_queue(
#     features_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
#     bar_queue: "multiprocessing.Queue[Optional[Tuple[str, int]]]",
#     send_connection: "Connection",
#     temp_dir_name: str,
# ) -> None:
#     """
#     Add OSM features to cache or update existing ones based on the ID.

#     Some of the `way` features are parsed twice, in form of `LineStrings` and `Polygons` /
#     `MultiPolygons`. Additional check ensures that a closed geometry will be always preffered over
# a
#     `LineString`.

#     Sends back a list of parsed features and a set of columns names.
#     """
#     # redis_db_file_name = (Path(temp_dir_name) / "redis.db").as_posix()
#     redis_db_file_name = "redis.db"
#     rdb = redislite.Redis(redis_db_file_name, charset="utf-8", decode_responses=True)
#     # db_file_name = (Path(temp_dir_name) / "features.duckdb").as_posix()
#     # json_file_name = (Path(temp_dir_name) / "features.json").as_posix()
#     json_file_name = "features.json"

#     # db_connection = get_new_duckdb_connection(db_file=db_file_name)

#     # create_query = """
#     # CREATE OR REPLACE TEMP TABLE temp_features
#     # (feature_id VARCHAR PRIMARY KEY, wkt VARCHAR, tags JSON)
#     # """

#     # upsert_query = """
#     # INSERT INTO temp_features (feature_id, wkt, tags)
#     # VALUES ($feature_id, $wkt, $tags)
#     # ON CONFLICT DO UPDATE
#     # SET tags = json_merge_patch(tags, excluded.tags)
#     # """

#     # update_wkt_query = """
#     # UPDATE temp_features
#     # SET wkt = $wkt
#     # WHERE feature_id = $feature_id
#     # """

#     # # Forcing nested JSON column to be saved in `.jsonl` format without quotes
#     # select_data_query = """
#     # COPY (
#     #     SELECT
#     #     json_merge_patch(
#     #         to_json({{feature_id: feature_id, wkt: wkt}}),
#     #         tags
#     #     ) as json
#     #     FROM temp_features
#     # ) TO '{json_file}'
#     # WITH (
#     #     FORMAT 'CSV',
#     #     HEADER false,
#     #     DELIMITER ',',
#     #     ESCAPE '',
#     #     QUOTE ''
#     # );
#     # """

#     # db_connection.execute(create_query)

#     # feature_ids: Set[str] = set()
#     columns: Set[str] = set()

#     open(json_file_name, mode="w").close()

#     while True:
#         feature: Optional[Dict[str, Any]] = features_queue.get()
#         if feature is None:
#             break

#         feature.pop(FEATURES_INDEX)
#         feature.pop("wkt")

#         # # feature_exists = bool(rdb.exists(full_osm_id))

#         # feature_exists = full_osm_id in feature_ids

#         # feature_to_append: Dict[str, Any] = {}
#         # if feature_exists:
#         #     line_number_to_replace = feature_ids[full_osm_id]
#         #     with open(json_file_name, encoding="utf-8") as json_file:
#         #         for line_number, input_line in enumerate(json_file):
#         #             if line_number < line_number_to_replace:
#         #                 continue
#         #             feature_to_append = json.loads(input_line)
#         #             break

#         #     if wkt.startswith(("POLYGON(", "MULTIPOLYGON(")):
#         #         feature_to_append["wkt"] = wkt
#         #     feature_to_append.update(feature)
#         # else:
#         #     parsed_counter += 1
#         #     feature_to_append = {FEATURES_INDEX: full_osm_id, "wkt": wkt, **feature}

#         # feature_ids[full_osm_id] = lines_counter
#         # lines_counter += 1

#         # with open(json_file_name, "a", encoding="utf-8") as json_file:
#         #     json_file.write(json.dumps(feature_to_append))
#         #     json_file.write("\n")

#         # if not feature_exists:
#         #     feature_ids[full_osm_id] = parsed_counter
#         #     parsed_counter += 1
#         #     with open(json_file_name, "a") as json_file:
#         #         line = f'{json.dumps({FEATURES_INDEX: full_osm_id, "wkt": wkt, **feature})}\n'
#         #         json_file.write(line)
#         #         # json_file.write("\n")
#         # else:
#         #     try:
#         #         line_number_to_replace = feature_ids[full_osm_id]
#         #         with open(json_file_name) as json_file, tempfile.NamedTemporaryFile(mode="wt",
# dir=dirname(json_file_name), delete=False) as output:
#         #             tname = output.name
#         #             for line_number, input_line in enumerate(json_file):
#         #                 output_line = input_line
#         #                 if line_number != line_number_to_replace:
#         #                     input_json: Dict[str, Any] = json.loads(input_line)
#         #                     if wkt.startswith(("POLYGON(", "MULTIPOLYGON(")):
#         #                         input_json["wkt"] = wkt
#         #                     input_json.update(feature)
#         #                     output_line = json.dumps(input_json)
#         #                 output.write(output_line)
#         #     except Exception:
#         #         remove(tname)
#         #     else:
#         #         rename(tname, json_file_name)
#             # wkt.startswith(("POLYGON(", "MULTIPOLYGON("))
#             # with fileinput.input(files=json_file_name, inplace=True) as f:
#             # with fileinput.input(files=json_file_name, inplace=True) as f:
#             #     line_number_to_replace = feature_ids[full_osm_id]
#             #     for line_number, input_line in enumerate(f):
#             #         if line_number != line_number_to_replace:
#             #             print(input_line, end="")
#             #             continue
#             #         # if len(input_line) == 0:
#             #         #     continue
#             #         # if line_number < line_number_to_replace:
#             #         #     continue

#             #         try:
#             #             # print(input_line, file=sys.stderr)
#             #             input_json: Dict[str, Any] = json.loads(input_line)
#             #         except:
#             #             print(input_line, file=sys.stderr)
#             #             print("XD", file=sys.stderr)
#             #             raise

#             #         if wkt.startswith(("POLYGON(", "MULTIPOLYGON(")):
#             #             input_json["wkt"] = wkt

#             #         input_json.update(feature)
#             #         output_line = json.dumps(input_json)

#             #         # print(line_number_to_replace, input_line, output_line, "\n", file=sys.std
# err)

#             #         print(output_line, end="\n")
#             #         # break
#             #         # print(line_number, line_number_to_replace, full_osm_id, file=sys.stderr)

#         #     rdb.hmset(full_osm_id, {
#         #         FEATURES_INDEX: full_osm_id,
#         #         "wkt": wkt,
#         #     })

#         # if feature_exists and wkt.startswith(("POLYGON(", "MULTIPOLYGON(")):
#         #     rdb.hset(full_osm_id, "wkt", wkt)

#         # rdb.hmset(full_osm_id, feature)

#         # db_connection.execute(
#         #     query=upsert_query,
#         #     parameters=dict(feature_id=full_osm_id, wkt=wkt, tags=json.dumps(feature))
#         # )

#         # if feature_exists and wkt.startswith(("POLYGON(", "MULTIPOLYGON(")):
#         #     db_connection.execute(
#         #         query=update_wkt_query,
#         #         parameters=dict(feature_id=full_osm_id, wkt=wkt)
#         #     )

#         # feature_exists = full_osm_id in features_dict
#         # existing_feature: Dict[str, Any] = features_dict.get(full_osm_id, {})

#         # if not feature_exists:
#         #     existing_feature = {
#         #         FEATURES_INDEX: full_osm_id,
#         #         "wkt": wkt,
#         #     }
#         #     parsed_counter += 1

#         # if feature_exists and wkt.startswith(("POLYGON(", "MULTIPOLYGON(")):
#         #     existing_feature["wkt"] = wkt

#         # existing_feature.update(feature)
#         # features_dict[full_osm_id] = existing_feature

#         columns.update(feature.keys())

#         # if parsed_counter % progress_bar_resolution == 0:
#         #     bar_queue.put_nowait(("parsed", progress_bar_resolution))

#     # matching_line_numbers = set(feature_ids.values())

#     # with fileinput.input(files=json_file_name, inplace=True) as f:
#     #     for line_number, input_line in enumerate(f):
#     #         if line_number in matching_line_numbers:
#     #             print(input_line, end="")

#     send_connection.send(columns)

#     # bar_queue.put_nowait(("parsed", parsed_counter % progress_bar_resolution))

#     # # temp_file = tempfile.NamedTemporaryFile(mode="w+")
#     with open(json_file_name, "w+") as temp_file:
#     #     # for feature in tqdm(rdb.keys(), desc=f"[{region_id}] Saving features to temp file"):
#         for feature_id in tqdm(rdb.keys(), desc="Saving features to temp file"):
#     #     # for feature in tqdm(feature_values, desc=f"[{region_id}] Saving features to temp file"
# ):
#     #         # values = rdb.hgetall(str(feature_id))
#             values = rdb.hgetall(feature_id)
#             jout = orjson.dumps(values) + "\n"
#             temp_file.write(jout)
#     # # temp_file.flush()

#     # db_connection.execute(select_data_query.format(json_file=json_file_name))

#     # db_connection.close()


def _process_pbf(
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    tags: Optional[osm_tags_type],
    pool_index: int,
    pool_size: int,
    features_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
    bar_queue: "multiprocessing.Queue[Optional[Tuple[str, int]]]",
    redis_db_file_name: str,
) -> None:
    """Process PBF file using MultiProcessingHandler."""
    simple_handler = MultiProcessingHandler(
        tags, pool_size, pool_index, features_queue, bar_queue, redis_db_file_name
    )
    for file_path in file_paths:
        simple_handler.apply_file(file_path)

    if simple_handler.parsed_counter > 0:
        bar_queue.put_nowait(
            ("parsed", simple_handler.parsed_counter % PARSED_PROGRESS_BAR_UPDATE_RESOLUTION)
        )
    if simple_handler.loaded_counter > 0:
        bar_queue.put_nowait(
            ("loaded", simple_handler.loaded_counter % LOADED_PROGRESS_BAR_UPDATE_RESOLUTION)
        )


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

    #  read_json(self: duckdb.DuckDBPyConnection, name: str, *, columns: object = None,
    # sample_size: object = None, maximum_depth: object = None) → duckdb.DuckDBPyRelation
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
