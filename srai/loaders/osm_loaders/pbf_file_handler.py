"""
PBF File Handler.

This module contains a handler capable of parsing a PBF file into a GeoDataFrame.
"""
import secrets
import tempfile
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

import psutil
from shapely.geometry.base import BaseGeometry

from srai.constants import FEATURES_INDEX
from srai.db import get_duckdb_connection, get_new_duckdb_connection
from srai.loaders.osm_loaders.filters._typing import osm_tags_type

if TYPE_CHECKING:
    import os

    import duckdb

LOAD_QUERY = """
INSERT INTO osm_features
SELECT
    feature_type || '/' || COALESCE(osm_id, osm_way_id) feature_id,
    {all_tags_query},
    wkb_geometry
FROM (
    SELECT
        * EXCLUDE (all_tags),
        hstore_to_json(all_tags) all_tags,
        CASE
            WHEN '{layer}' = 'points' THEN 'node'
            WHEN '{layer}' = 'lines' THEN 'way'
            WHEN '{layer}' = 'multilinestrings' THEN 'relation'
            WHEN '{layer}' = 'other_relations' THEN 'relation'
            WHEN '{layer}' = 'multipolygons' AND osm_way_id IS NULL THEN 'relation'
            WHEN '{layer}' = 'multipolygons' AND osm_way_id IS NOT NULL THEN 'way'
        END AS feature_type
    FROM ST_READ(
        '{pbf_file}',
        allowed_drivers = ['OSM'],
        open_options = [
            'INTERLEAVED_READING=YES',
            'CONFIG_FILE={gdal_config_file}',
            'MAX_TMPFILE_SIZE={max_memory_size}',
            'USE_CUSTOM_INDEXING=NO'
        ],
        sequential_layer_scan = true,
        layer = '{layer}'
    )
)
WHERE ({filter_clauses}) {geometry_filter}
"""

GDAL_LAYERS = ["points", "lines", "multilinestrings", "multipolygons", "other_relations"]

SAVE_TABLE_TO_JSON_QUERY = """
COPY (
    SELECT
    json_merge_patch(
        to_json({{feature_id: feature_id, wkt: ST_AsText(ST_GeomFromWKB(wkb_geometry))}}),
        all_tags
    ) as json
    FROM osm_features
) TO '{json_file}'
WITH (
    FORMAT 'CSV',
    HEADER false,
    DELIMITER ',',
    ESCAPE '',
    QUOTE ''
);
"""


def read_features_from_pbf_files(
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    tags: Optional[osm_tags_type] = None,
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
    with tempfile.TemporaryDirectory() as temp_dir_name:
        duckdb_file_name = (Path(temp_dir_name) / "features.duckdb").as_posix()
        json_file_name = (Path(temp_dir_name) / "features.jsonl").as_posix()

        connection = get_new_duckdb_connection(db_file=duckdb_file_name)
        _prepare_temp_db(connection)

        queries = _prepare_queries(file_paths, tags, filter_region_geometry)

        for query in queries:
            connection.execute(query)

        columns = set(
            row[0]
            for row in connection.sql(
                "SELECT DISTINCT UNNEST(json_keys(all_tags)) tag_key FROM osm_features"
            ).fetchall()
        )
        columns = columns.difference([FEATURES_INDEX, "wkt"])

        connection.execute(SAVE_TABLE_TO_JSON_QUERY.format(json_file=json_file_name))
        features_relation = _parse_raw_df_to_duckdb(json_file_name, columns)

    return features_relation


def _prepare_temp_db(connection: "duckdb.DuckDBPyConnection") -> None:
    """Create required macro and table for OSM features."""
    macro_query = """
    CREATE OR REPLACE MACRO hstore_to_json(hstore_string) AS
    json('{' || replace(regexp_replace(hstore_string,'\\s',' ', 'g'),'=>',':') || '}');
    """
    connection.execute(macro_query)

    create_table_query = (
        "CREATE OR REPLACE TEMP TABLE osm_features(feature_id VARCHAR, all_tags JSON, wkb_geometry"
        " WKB_BLOB)"
    )
    connection.execute(create_table_query)


def _prepare_queries(
    file_paths: Sequence[Union[str, "os.PathLike[str]"]],
    tags: Optional[osm_tags_type],
    filter_region_geometry: Optional[BaseGeometry],
) -> List[str]:
    """Prepare SQL queries for loading OSM files."""
    queries = []

    json_schema = _generate_tags_json_transform_schema(tags)
    filter_clauses = _generate_tags_json_filter(tags)
    gdal_config_file_path = Path(__file__).parent / "gdal_config" / "osmconf.ini"

    for file_path in file_paths:
        file_path_object = Path(file_path)
        available_memory = psutil.virtual_memory().available / 1024**2
        file_size = file_path_object.stat().st_size / 1024**2
        mb_memory_size = int(min(available_memory * 0.75, file_size * 5))

        geometry_filter = None

        if filter_region_geometry is not None:
            region_geometry_wkt = filter_region_geometry.wkt
            geometry_filter = f"ST_Intersects(geometry, ST_GeomFromText('{region_geometry_wkt}'))"

        for layer_name in GDAL_LAYERS:
            filled_query = LOAD_QUERY.format(
                pbf_file=file_path_object.as_posix(),
                gdal_config_file=gdal_config_file_path.as_posix(),
                layer=layer_name,
                max_memory_size=mb_memory_size,
                all_tags_query=(
                    "all_tags"
                    if json_schema is None
                    else f"json_transform(all_tags, '{json_schema}') all_tags"
                ),
                filter_clauses=filter_clauses,
                geometry_filter="" if geometry_filter is None else f"AND {geometry_filter}",
            )
            queries.append(filled_query)

    return queries


def _generate_tags_json_transform_schema(
    tags_filter: Optional[osm_tags_type] = None,
) -> Optional[str]:
    """Prepare JSON schema based on tags filter."""
    schema = None

    if tags_filter:
        tags_definitions = [
            f'"{filter_tag_key}": "VARCHAR"' for filter_tag_key in tags_filter.keys()
        ]
        schema = f"{{ {', '.join(tags_definitions)} }}"

    return schema


def _generate_tags_json_filter(tags_filter: Optional[osm_tags_type] = None) -> str:
    """Prepare features filter clauses based on tags filter."""
    filter_clauses = ["(1=1)"]

    if tags_filter:
        filter_clauses.clear()

        def escape(value: str) -> str:
            return value.replace("'", "''")

        for filter_tag_key, filter_tag_value in tags_filter.items():
            if isinstance(filter_tag_value, bool) and filter_tag_value:
                filter_clauses.append(
                    f"(json_extract_string(all_tags, '{filter_tag_key}') IS NOT NULL)"
                )
            elif isinstance(filter_tag_value, str):
                escaped_value = escape(filter_tag_value)
                filter_clauses.append(
                    f"(json_extract_string(all_tags, '{filter_tag_key}') = '{escaped_value}')"
                )
            elif isinstance(filter_tag_value, list) and filter_tag_value:
                values_list = [f"'{escape(value)}'" for value in filter_tag_value]
                filter_clauses.append(
                    f"(json_extract_string(all_tags, '{filter_tag_key}') IN"
                    f" ({', '.join(values_list)}))"
                )

    return " OR ".join(filter_clauses)


def _parse_raw_df_to_duckdb(json_file_path: str, columns: Set[str]) -> "duckdb.DuckDBPyRelation":
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
            records=true,
            format='newline_delimited',
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

    features_relation.to_table(relation_name)

    return get_duckdb_connection().table(relation_name)
