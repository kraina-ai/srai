"""
PBF File Handler.

This module contains a handler capable of parsing a PBF file into a GeoDataFrame.
"""

import hashlib
import json
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from math import floor
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, cast

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from srai._optional import import_optional_dependencies
from srai._typing import is_expected_type
from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, WGS84_CRS
from srai.geometry import get_geometry_hash
from srai.loaders.osm_loaders.filters import (
    GroupedOsmTagsFilter,
    OsmTagsFilter,
    merge_osm_tags_filter,
)

if TYPE_CHECKING:  # pragma: no cover
    import os

    import duckdb


class PbfFileHandler:
    """
    PbfFileHandler.

    PBF(Protocolbuffer Binary Format)[1] file handler is a dedicated `*.osm.pbf` files reader
    based on DuckDB[2] and its spatial extension[3].

    Handler can filter out OSM features based on tags filter and geometry filter
    to limit the result.

    References:
        1. https://wiki.openstreetmap.org/wiki/PBF_Format
        2. https://duckdb.org/
        3. https://github.com/duckdb/duckdb_spatial
    """

    AREA_KEYS_URL = "https://raw.githubusercontent.com/ideditor/id-area-keys/main/areaKeys.json"
    AREA_KEYS_CONFIG = {
        "addr:*": True,
        "advertising": ["billboard"],
        "aerialway": [
            "cable_car",
            "chair_lift",
            "drag_lift",
            "gondola",
            "goods",
            "j-bar",
            "magic_carpet",
            "mixed_lift",
            "platter",
            "rope_tow",
            "t-bar",
            "zip_line",
        ],
        "aeroway": ["jet_bridge", "parking_position", "runway", "taxiway"],
        "allotments": True,
        "amenity": ["bench", "weighbridge"],
        "area:highway": True,
        "attraction": ["dark_ride", "river_rafting", "summer_toboggan", "train", "water_slide"],
        "bridge:support": True,
        "building": True,
        "building:part": True,
        "cemetery": True,
        "club": True,
        "craft": True,
        "demolished:building": True,
        "disused:amenity": True,
        "disused:railway": True,
        "disused:shop": True,
        "emergency": ["designated", "destination", "no", "official", "private", "yes"],
        "golf": ["cartpath", "hole", "path"],
        "healthcare": True,
        "historic": True,
        "indoor": ["corridor", "wall"],
        "industrial": True,
        "internet_access": True,
        "junction": True,
        "landuse": True,
        "leisure": ["slipway", "track"],
        "man_made": [
            "yes",
            "breakwater",
            "carpet_hanger",
            "crane",
            "cutline",
            "dyke",
            "embankment",
            "goods_conveyor",
            "groyne",
            "pier",
            "pipeline",
            "torii",
            "video_wall",
        ],
        "military": ["trench"],
        "natural": ["bay", "cliff", "coastline", "ridge", "strait", "tree_row", "valley"],
        "office": True,
        "piste:type": ["downhill", "hike", "ice_skate", "nordic", "skitour", "sled", "sleigh"],
        "place": True,
        "playground": [
            "activitypanel",
            "balancebeam",
            "basketswing",
            "bridge",
            "climbingwall",
            "hopscotch",
            "horizontal_bar",
            "seesaw",
            "slide",
            "structure",
            "swing",
            "tunnel_tube",
            "water",
            "zipwire",
        ],
        "police": True,
        "polling_station": True,
        "power": ["cable", "line", "minor_line"],
        "public_transport": ["platform"],
        "residential": True,
        "seamark:type": True,
        "shop": True,
        "telecom": True,
        "tourism": ["artwork", "attraction"],
        "traffic_calming": [
            "yes",
            "bump",
            "chicane",
            "choker",
            "cushion",
            "dip",
            "hump",
            "island",
            "mini_bumps",
            "rumble_strip",
        ],
        "waterway": [
            "canal",
            "dam",
            "ditch",
            "drain",
            "fish_pass",
            "lock_gate",
            "river",
            "stream",
            "tidal_channel",
            "weir",
        ],
    }

    # From https://github.com/Binnette/osm-polygon-features/blob/patch-1/polygon-features.json
    POLYGON_FEATURES = [
        {"key": "building", "polygon": "all"},
        {
            "key": "highway",
            "polygon": "whitelist",
            "values": ["services", "rest_area", "escape", "elevator"],
        },
        {
            "key": "natural",
            "polygon": "blacklist",
            "values": ["no", "coastline", "cliff", "ridge", "arete", "tree_row"],
        },
        {"key": "landuse", "polygon": "all"},
        {
            "key": "waterway",
            "polygon": "whitelist",
            "values": ["riverbank", "dock", "boatyard", "dam"],
        },
        {"key": "amenity", "polygon": "all"},
        {"key": "leisure", "polygon": "all"},
        {
            "key": "barrier",
            "polygon": "whitelist",
            "values": ["city_wall", "ditch", "hedge", "retaining_wall", "wall", "spikes"],
        },
        {
            "key": "railway",
            "polygon": "whitelist",
            "values": ["station", "turntable", "roundhouse", "platform"],
        },
        {"key": "area", "polygon": "all"},
        {"key": "boundary", "polygon": "all"},
        {
            "key": "man_made",
            "polygon": "blacklist",
            "values": ["no", "cutline", "embankment", "pipeline"],
        },
        {
            "key": "power",
            "polygon": "whitelist",
            "values": ["plant", "substation", "generator", "transformer"],
        },
        {"key": "place", "polygon": "all"},
        {"key": "shop", "polygon": "all"},
        {"key": "aeroway", "polygon": "blacklist", "values": ["no", "taxiway"]},
        {"key": "tourism", "polygon": "all"},
        {"key": "historic", "polygon": "all"},
        {"key": "public_transport", "polygon": "all"},
        {"key": "office", "polygon": "all"},
        {"key": "building:part", "polygon": "all"},
        {"key": "military", "polygon": "all"},
        {"key": "ruins", "polygon": "all"},
        {"key": "area:highway", "polygon": "all"},
        {"key": "craft", "polygon": "all"},
        {"key": "golf", "polygon": "all"},
        {"key": "indoor", "polygon": "all"},
    ]

    class ConvertedOSMParquetFiles(NamedTuple):
        """List of parquet files read from the `*.osm.pbf` file."""

        nodes_valid_with_tags: "duckdb.DuckDBPyRelation"
        nodes_required_ids: "duckdb.DuckDBPyRelation"
        nodes_filtered_ids: "duckdb.DuckDBPyRelation"

        ways_all_with_tags: "duckdb.DuckDBPyRelation"
        ways_with_unnested_nodes_refs: "duckdb.DuckDBPyRelation"
        ways_required_ids: "duckdb.DuckDBPyRelation"
        ways_filtered_ids: "duckdb.DuckDBPyRelation"

        relations_all_with_tags: "duckdb.DuckDBPyRelation"
        relations_with_unnested_way_refs: "duckdb.DuckDBPyRelation"
        relations_filtered_ids: "duckdb.DuckDBPyRelation"

    class ParsedOSMFeatures(NamedTuple):
        """Final list of parsed features from the `*.osm.pbf` file."""

        nodes: "duckdb.DuckDBPyRelation"
        ways: "duckdb.DuckDBPyRelation"
        relations: "duckdb.DuckDBPyRelation"

    def __init__(
        self,
        tags_filter: Optional[Union[OsmTagsFilter, GroupedOsmTagsFilter]] = None,
        geometry_filter: Optional[BaseGeometry] = None,
        working_directory: Union[str, Path] = "files",
    ) -> None:
        """
        Initialize PbfFileHandler.

        Args:
            tags_filter (OsmTagsFilter, optional): A dictionary
                specifying which tags to download.
                The keys should be OSM tags (e.g. `building`, `amenity`).
                The values should either be `True` for retrieving all objects with the tag,
                string for retrieving a single tag-value pair
                or list of strings for retrieving all values specified in the list.
                `tags={'leisure': 'park}` would return parks from the area.
                `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
                would return parks, all amenity types, bakeries and bicycle shops.
                If `None`, handler will allow all of the tags to be parsed. Defaults to `None`.
            geometry_filter (BaseGeometry, optional): Region which can be used to filter only
                intersecting OSM objects. Defaults to None.
            working_directory (Union[str, Path], optional): Directory where to save
                the parsed `*.parquet` files. Defaults to "files".
        """
        import_optional_dependencies(dependency_group="osm", modules=["duckdb"])
        self.tags_filter = tags_filter
        self.merged_tags_filter = merge_osm_tags_filter(tags_filter) if tags_filter else None
        self.geometry_filter = geometry_filter
        self.working_directory = Path(working_directory)
        self.working_directory.mkdir(parents=True, exist_ok=True)
        self.connection: duckdb.DuckDBPyConnection = None
        self.rows_per_bucket = 1_000_000

    def get_features_gdf(
        self,
        file_paths: Sequence[Union[str, "os.PathLike[str]"]],
        explode_tags: bool = True,
        ignore_cache: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Get features GeoDataFrame from a list of PBF files.

        Function parses multiple PBF files and returns a single GeoDataFrame with parsed
        OSM objects.

        This function is a dedicated wrapper around the inherited function `apply_file`.

        Args:
            file_paths (Sequence[Union[str, os.PathLike[str]]]): List of paths to `*.osm.pbf`
                files to be parsed.
            explode_tags (bool, optional): Whether to split tags into columns based on OSM tag keys.
                Defaults to True.
            ignore_cache: (bool, optional): Whether to ignore precalculated geoparquet files or not.
                Defaults to False.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with OSM features.
        """
        import geoarrow.pyarrow as ga
        from geoarrow.pyarrow import io

        parsed_geoparquet_files = []
        for file_path in file_paths:
            parsed_geoparquet_file = self.convert_pbf_to_gpq(
                file_path, explode_tags=explode_tags, ignore_cache=ignore_cache
            )
            parsed_geoparquet_files.append(parsed_geoparquet_file)

        parquet_tables = [
            io.read_geoparquet_table(parsed_parquet_file)
            for parsed_parquet_file in parsed_geoparquet_files
        ]
        joined_parquet_table: pa.Table = pa.concat_tables(parquet_tables)
        gdf_parquet = gpd.GeoDataFrame(
            data=joined_parquet_table.drop(GEOMETRY_COLUMN).to_pandas(maps_as_pydicts="strict"),
            geometry=ga.to_geopandas(joined_parquet_table.column(GEOMETRY_COLUMN)),
        ).set_index(FEATURES_INDEX)

        return gdf_parquet

    def convert_pbf_to_gpq(
        self,
        pbf_path: Union[str, "os.PathLike[str]"],
        result_file_path: Optional[Union[str, "os.PathLike[str]"]] = None,
        explode_tags: bool = True,
        ignore_cache: bool = False,
    ) -> Path:
        """
        Convert PBF file to GeoParquet file.

        Args:
            pbf_path (Union[str, os.PathLike[str]]): _description_
            result_file_path (Union[str, os.PathLike[str]], optional): Where to save
                the geoparquet file. If not provided, will be generated based on hashes
                from provided tags filter and geometry filter. Defaults to None.
            explode_tags (bool, optional): Whether to split tags into columns based on OSM tag keys.
                Defaults to True.
            ignore_cache (bool, optional): Whether to ignore precalculated geoparquet files or not.
                Defaults to False.

        Returns:
            Path: Path to the generated GeoParquet file.
        """
        with tempfile.TemporaryDirectory(dir=self.working_directory.resolve()) as tmp_dir_name:
            try:
                self._set_up_duckdb_connection(tmp_dir_name)
                result_file_path = result_file_path or self._generate_geoparquet_result_file_path(
                    pbf_path
                )
                parsed_geoparquet_file = self._parse_pbf_file(
                    pbf_path=pbf_path,
                    tmp_dir_name=tmp_dir_name,
                    result_file_path=Path(result_file_path),
                    explode_tags=explode_tags,
                    ignore_cache=ignore_cache,
                )
                return parsed_geoparquet_file
            finally:
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None

    def _set_up_duckdb_connection(self, tmp_dir_name: str) -> None:
        import duckdb

        self.connection = duckdb.connect(database=str(Path(tmp_dir_name) / "db.duckdb"))
        for extension_name in ("parquet", "spatial"):
            self.connection.install_extension(extension_name)
            self.connection.load_extension(extension_name)

        self.connection.sql("""
            CREATE OR REPLACE MACRO linestring_to_linestring_wkt(ls) AS
            'LINESTRING (' || array_to_string([pt.x || ' ' || pt.y for pt in ls], ', ') || ')';
        """)
        self.connection.sql("""
            CREATE OR REPLACE MACRO linestring_to_polygon_wkt(ls) AS
            'POLYGON ((' || array_to_string([pt.x || ' ' || pt.y for pt in ls], ', ') || '))';
        """)

    def _parse_pbf_file(
        self,
        pbf_path: Union[str, "os.PathLike[str]"],
        tmp_dir_name: str,
        result_file_path: Path,
        explode_tags: bool = True,
        ignore_cache: bool = False,
    ) -> Path:
        if not result_file_path.exists() or ignore_cache:
            elements = self.connection.sql(f"SELECT * FROM ST_READOSM('{Path(pbf_path)}');")
            converted_osm_parquet_files = self._prefilter_elements_ids(elements, tmp_dir_name)

            self._delete_directories(
                tmp_dir_name,
                [
                    "nodes_filtered_non_distinct_ids",
                    "nodes_prepared_ids",
                    "ways_valid_ids",
                    "ways_filtered_non_distinct_ids",
                    "relations_valid_ids",
                    "relations_ids",
                ],
            )

            filtered_nodes_with_geometry = self._get_filtered_nodes_with_geometry(
                converted_osm_parquet_files, tmp_dir_name
            )
            self._delete_directories(tmp_dir_name, "nodes_filtered_ids")

            required_nodes_with_structs = self._get_required_nodes_with_structs(
                converted_osm_parquet_files, tmp_dir_name
            )
            self._delete_directories(
                tmp_dir_name,
                [
                    "nodes_valid_with_tags",
                    "nodes_required_ids",
                ],
            )

            required_ways_with_linestrings = self._get_required_ways_with_linestrings(
                converted_osm_parquet_files, required_nodes_with_structs, tmp_dir_name
            )
            self._delete_directories(
                tmp_dir_name,
                [
                    "ways_required_ids_grouped",
                    "ways_required_ids",
                    "ways_with_unnested_nodes_refs",
                    "required_nodes_with_points",
                ],
            )

            filtered_ways_with_proper_geometry = self._get_filtered_ways_with_proper_geometry(
                converted_osm_parquet_files, required_ways_with_linestrings, tmp_dir_name
            )
            self._delete_directories(
                tmp_dir_name,
                [
                    "ways_prepared_ids",
                    "ways_all_with_tags",
                ],
            )

            filtered_relations_with_geometry = self._get_filtered_relations_with_geometry(
                converted_osm_parquet_files, required_ways_with_linestrings, tmp_dir_name
            )
            self._delete_directories(
                tmp_dir_name,
                [
                    "relations_all_with_tags",
                    "relations_with_unnested_way_refs",
                    "relations_filtered_ids",
                    "required_ways_with_linestrings",
                ],
            )

            self._concatenate_results_to_geoparquet(
                PbfFileHandler.ParsedOSMFeatures(
                    nodes=filtered_nodes_with_geometry,
                    ways=filtered_ways_with_proper_geometry,
                    relations=filtered_relations_with_geometry,
                ),
                tmp_dir_name=tmp_dir_name,
                save_file_path=result_file_path,
                explode_tags=explode_tags,
            )

        return result_file_path

    def _generate_geoparquet_result_file_path(
        self, pbf_file_path: Union[str, "os.PathLike[str]"]
    ) -> Path:
        pbf_file_name = Path(pbf_file_path).name.removesuffix(".osm.pbf")

        osm_filter_tags_hash_part = "nofilter"
        if self.tags_filter is not None:
            h = hashlib.new("sha256")
            h.update(json.dumps(self.tags_filter).encode())
            osm_filter_tags_hash_part = h.hexdigest()

        clipping_geometry_hash_part = "noclip"
        if self.geometry_filter is not None:
            clipping_geometry_hash_part = get_geometry_hash(self.geometry_filter)

        result_file_name = (
            f"{pbf_file_name}_{osm_filter_tags_hash_part}_{clipping_geometry_hash_part}.geoparquet"
        )
        return Path(self.working_directory) / result_file_name

    def _prefilter_elements_ids(
        self, elements: "duckdb.DuckDBPyRelation", tmp_dir_name: str
    ) -> ConvertedOSMParquetFiles:
        sql_filter = self._generate_osm_tags_sql_filter()
        filtered_tags_clause = self._generate_filtered_tags_clause()

        is_intersecting = self.geometry_filter is not None

        nodes_prepared_ids_path = Path(tmp_dir_name) / "nodes_prepared_ids"
        nodes_prepared_ids_path.mkdir(parents=True, exist_ok=True)

        ways_prepared_ids_path = Path(tmp_dir_name) / "ways_prepared_ids"
        ways_prepared_ids_path.mkdir(parents=True, exist_ok=True)

        # NODES - VALID (NV)
        # - select all with kind = 'node'
        # - select all with lat and lon not empty
        nodes_valid_with_tags = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT
                id,
                {filtered_tags_clause},
                lon,
                lat
            FROM ({elements.sql_query()})
            WHERE kind = 'node'
            AND lat IS NOT NULL AND lon IS NOT NULL
            """,
            file_path=Path(tmp_dir_name) / "nodes_valid_with_tags",
        )
        # NODES - INTERSECTING (NI)
        # - select all from NV which intersect given geometry filter
        # NODES - FILTERED (NF)
        # - select all from NI with tags filter
        if is_intersecting:
            wkt = cast(BaseGeometry, self.geometry_filter).wkt
            intersection_filter = f"ST_Intersects(ST_Point(lon, lat), ST_GeomFromText('{wkt}'))"
            nodes_intersecting_ids = self._sql_to_parquet_file(
                sql_query=f"""
                SELECT DISTINCT id FROM ({nodes_valid_with_tags.sql_query()}) n
                WHERE {intersection_filter} = true
                """,
                file_path=Path(tmp_dir_name) / "nodes_intersecting_ids",
            )
            self._sql_to_parquet_file(
                sql_query=f"""
                SELECT id FROM ({nodes_valid_with_tags.sql_query()}) n
                SEMI JOIN ({nodes_intersecting_ids.sql_query()}) ni ON n.id = ni.id
                WHERE tags IS NOT NULL AND cardinality(tags) > 0 AND ({sql_filter})
                """,
                file_path=Path(tmp_dir_name) / "nodes_filtered_non_distinct_ids",
            )
        else:
            nodes_intersecting_ids = nodes_valid_with_tags
            self._sql_to_parquet_file(
                sql_query=f"""
                SELECT id FROM ({nodes_valid_with_tags.sql_query()}) n
                WHERE tags IS NOT NULL AND cardinality(tags) > 0 AND ({sql_filter})
                """,
                file_path=Path(tmp_dir_name) / "nodes_filtered_non_distinct_ids",
            )
        nodes_filtered_ids = self._calculate_unique_ids_to_parquet(
            Path(tmp_dir_name) / "nodes_filtered_non_distinct_ids",
            Path(tmp_dir_name) / "nodes_filtered_ids",
        )

        # WAYS - VALID (WV)
        # - select all with kind = 'way'
        # - select all with more then one ref
        # - join all NV to refs
        # - select all where all refs has been joined (total_refs == found_refs)
        self.connection.sql(f"""
            SELECT *
            FROM ({elements.sql_query()}) w
            WHERE kind = 'way' AND len(refs) >= 2
        """).to_view("ways", replace=True)
        ways_all_with_tags = self._sql_to_parquet_file(
            sql_query=f"""
            WITH filtered_tags AS (
                SELECT id, {filtered_tags_clause}
                FROM ways w
                WHERE tags IS NOT NULL AND cardinality(tags) > 0
            )
            SELECT id, tags
            FROM filtered_tags
            WHERE tags IS NOT NULL AND cardinality(tags) > 0
            """,
            file_path=Path(tmp_dir_name) / "ways_all_with_tags",
        )
        ways_with_unnested_nodes_refs = self._sql_to_parquet_file(
            sql_query="""
            SELECT w.id, UNNEST(refs) as ref, UNNEST(range(length(refs))) as ref_idx
            FROM ways w
            """,
            file_path=Path(tmp_dir_name) / "ways_with_unnested_nodes_refs",
        )
        ways_valid_ids = self._sql_to_parquet_file(
            sql_query=f"""
            WITH total_ways_with_nodes_refs AS (
                SELECT id, ref
                FROM ({ways_with_unnested_nodes_refs.sql_query()})
            ),
            unmatched_ways_with_nodes_refs AS (
                SELECT id, ref
                FROM ({ways_with_unnested_nodes_refs.sql_query()}) w
                ANTI JOIN ({nodes_valid_with_tags.sql_query()}) nv ON nv.id = w.ref
            )
            SELECT DISTINCT id
            FROM total_ways_with_nodes_refs
            EXCEPT
            SELECT DISTINCT id
            FROM unmatched_ways_with_nodes_refs
            """,
            file_path=Path(tmp_dir_name) / "ways_valid_ids",
        )
        # WAYS - INTERSECTING (WI)
        # - select all from WV with joining any from NV on ref
        if is_intersecting:
            ways_intersecting_ids = self._sql_to_parquet_file(
                sql_query=f"""
                SELECT DISTINCT uwr.id
                FROM ({ways_with_unnested_nodes_refs.sql_query()}) uwr
                SEMI JOIN ({ways_valid_ids.sql_query()}) wv ON uwr.id = wv.id
                SEMI JOIN ({nodes_intersecting_ids.sql_query()}) n ON n.id = uwr.ref
                """,
                file_path=Path(tmp_dir_name) / "ways_intersecting_ids",
            )
        else:
            ways_intersecting_ids = ways_valid_ids
        # WAYS - FILTERED (WF)
        # - select all from WI with tags filter
        self._sql_to_parquet_file(
            sql_query=f"""
            SELECT id FROM ({ways_all_with_tags.sql_query()}) w
            SEMI JOIN ({ways_intersecting_ids.sql_query()}) wi ON w.id = wi.id
            WHERE {sql_filter}
            """,
            file_path=Path(tmp_dir_name) / "ways_filtered_non_distinct_ids",
        )
        ways_filtered_ids = self._calculate_unique_ids_to_parquet(
            Path(tmp_dir_name) / "ways_filtered_non_distinct_ids",
            ways_prepared_ids_path / "filtered",
        )

        # RELATIONS - VALID (RV)
        # - select all with kind = 'relation'
        # - select all with more then one ref
        # - select all with type in ['boundary', 'multipolygon']
        # - join all WV to refs
        # - select all where all refs has been joined (total_refs == found_refs)
        self.connection.sql(f"""
            SELECT *
            FROM ({elements.sql_query()})
            WHERE kind = 'relation' AND len(refs) > 0
            AND list_contains(map_keys(tags), 'type')
            AND list_has_any(map_extract(tags, 'type'), ['boundary', 'multipolygon'])
        """).to_view("relations", replace=True)
        relations_all_with_tags = self._sql_to_parquet_file(
            sql_query=f"""
            WITH filtered_tags AS (
                SELECT id, {filtered_tags_clause}
                FROM relations r
                WHERE tags IS NOT NULL AND cardinality(tags) > 0
            )
            SELECT id, tags
            FROM filtered_tags
            WHERE tags IS NOT NULL AND cardinality(tags) > 0
            """,
            file_path=Path(tmp_dir_name) / "relations_all_with_tags",
        )
        relations_with_unnested_way_refs = self._sql_to_parquet_file(
            sql_query="""
            WITH unnested_relation_refs AS (
                SELECT
                    r.id,
                    UNNEST(refs) as ref,
                    UNNEST(ref_types) as ref_type,
                    UNNEST(ref_roles) as ref_role,
                    UNNEST(range(length(refs))) as ref_idx
                FROM relations r
            )
            SELECT id, ref, ref_role, ref_idx
            FROM unnested_relation_refs
            WHERE ref_type = 'way'
            """,
            file_path=Path(tmp_dir_name) / "relations_with_unnested_way_refs",
        )
        relations_valid_ids = self._sql_to_parquet_file(
            sql_query=f"""
            WITH total_relation_refs AS (
                SELECT id, ref
                FROM ({relations_with_unnested_way_refs.sql_query()}) frr
            ),
            unmatched_relation_refs AS (
                SELECT id, ref
                FROM ({relations_with_unnested_way_refs.sql_query()}) r
                ANTI JOIN ({ways_valid_ids.sql_query()}) wv ON wv.id = r.ref
            )
            SELECT DISTINCT id
            FROM total_relation_refs
            EXCEPT
            SELECT DISTINCT id
            FROM unmatched_relation_refs
            """,
            file_path=Path(tmp_dir_name) / "relations_valid_ids",
        )
        # RELATIONS - INTERSECTING (RI)
        # - select all from RW with joining any from RV on ref
        if is_intersecting:
            relations_intersecting_ids = self._sql_to_parquet_file(
                sql_query=f"""
                SELECT frr.id
                FROM ({relations_with_unnested_way_refs.sql_query()}) frr
                SEMI JOIN ({relations_valid_ids.sql_query()}) rv ON frr.id = rv.id
                SEMI JOIN ({ways_intersecting_ids.sql_query()}) wi ON wi.id = frr.ref
                """,
                file_path=Path(tmp_dir_name) / "relations_intersecting_ids",
            )
        else:
            relations_intersecting_ids = relations_valid_ids
        # RELATIONS - FILTERED (RF)
        # - select all from RI with tags filter
        relations_ids_path = Path(tmp_dir_name) / "relations_ids"
        relations_ids_path.mkdir(parents=True, exist_ok=True)
        self._sql_to_parquet_file(
            sql_query=f"""
            SELECT id FROM ({relations_all_with_tags.sql_query()}) r
            SEMI JOIN ({relations_intersecting_ids.sql_query()}) ri ON r.id = ri.id
            WHERE {sql_filter}
            """,
            file_path=relations_ids_path / "filtered",
        )
        relations_filtered_ids = self._calculate_unique_ids_to_parquet(
            relations_ids_path / "filtered", Path(tmp_dir_name) / "relations_filtered_ids"
        )

        # WAYS - REQUIRED (WR)
        # - required - all IDs from WF
        #   + all needed to construct relations from RF
        self._sql_to_parquet_file(
            sql_query=f"""
            SELECT ref as id
            FROM ({relations_with_unnested_way_refs.sql_query()}) frr
            SEMI JOIN ({relations_filtered_ids.sql_query()}) fri ON fri.id = frr.id
            """,
            file_path=ways_prepared_ids_path / "required_by_relations",
        )
        ways_required_ids = self._calculate_unique_ids_to_parquet(
            ways_prepared_ids_path, Path(tmp_dir_name) / "ways_required_ids"
        )

        # NODES - REQUIRED (WR)
        # - required - all IDs from NF
        #   + all needed to construct ways from WR
        #   + and needed to construct ways from WF
        self._sql_to_parquet_file(
            sql_query=f"""
            SELECT ref as id
            FROM ({ways_with_unnested_nodes_refs.sql_query()}) uwr
            SEMI JOIN ({ways_required_ids.sql_query()}) rwi ON rwi.id = uwr.id
            """,
            file_path=nodes_prepared_ids_path / "required_by_relations",
        )
        self._sql_to_parquet_file(
            sql_query=f"""
            SELECT ref as id
            FROM ({ways_with_unnested_nodes_refs.sql_query()}) uwr
            SEMI JOIN ({ways_filtered_ids.sql_query()}) fwi ON fwi.id = uwr.id
            """,
            file_path=nodes_prepared_ids_path / "required_by_ways",
        )
        nodes_required_ids = self._calculate_unique_ids_to_parquet(
            nodes_prepared_ids_path, Path(tmp_dir_name) / "nodes_required_ids"
        )

        return PbfFileHandler.ConvertedOSMParquetFiles(
            nodes_valid_with_tags=nodes_valid_with_tags,
            nodes_required_ids=nodes_required_ids,
            nodes_filtered_ids=nodes_filtered_ids,
            ways_all_with_tags=ways_all_with_tags,
            ways_with_unnested_nodes_refs=ways_with_unnested_nodes_refs,
            ways_required_ids=ways_required_ids,
            ways_filtered_ids=ways_filtered_ids,
            relations_all_with_tags=relations_all_with_tags,
            relations_with_unnested_way_refs=relations_with_unnested_way_refs,
            relations_filtered_ids=relations_filtered_ids,
        )

    def _delete_directories(self, tmp_dir_name: str, directories: Union[str, list[str]]) -> None:
        if isinstance(directories, str):
            directories = [directories]
        for directory in directories:
            directory_path = Path(tmp_dir_name) / directory
            if not directory_path.exists():
                continue
            shutil.rmtree(directory_path)

    def _generate_osm_tags_sql_filter(self) -> str:
        """Prepare features filter clauses based on tags filter."""
        filter_clauses = ["(1=1)"]

        if self.merged_tags_filter:
            filter_clauses.clear()

            for filter_tag_key, filter_tag_value in self.merged_tags_filter.items():
                if isinstance(filter_tag_value, bool) and filter_tag_value:
                    filter_clauses.append(f"(list_contains(map_keys(tags), '{filter_tag_key}'))")
                elif isinstance(filter_tag_value, str):
                    escaped_value = self._sql_escape(filter_tag_value)
                    filter_clauses.append(
                        f"list_extract(map_extract(tags, '{filter_tag_key}'), 1) ="
                        f" '{escaped_value}'"
                    )
                elif isinstance(filter_tag_value, list) and filter_tag_value:
                    values_list = [f"'{self._sql_escape(value)}'" for value in filter_tag_value]
                    filter_clauses.append(
                        f"list_extract(map_extract(tags, '{filter_tag_key}'), 1) IN"
                        f" ({', '.join(values_list)})"
                    )

        return " OR ".join(filter_clauses)

    def _generate_filtered_tags_clause(self) -> str:
        """Prepare filtered tags clause by removing tags commonly ignored by OGR."""
        tags_to_ignore = [
            "created_by",
            "converted_by",
            "source",
            "time",
            "ele",
            "note",
            "todo",
            "fixme",
            "FIXME",
            "openGeoDB:",
        ]
        escaped_tags_to_ignore = [f"'{tag}'" for tag in tags_to_ignore]

        return f"""
        map_from_entries(
            [
                tag_entry
                for tag_entry in map_entries(tags)
                if not tag_entry.key in ({','.join(escaped_tags_to_ignore)})
                and not starts_with(tag_entry.key, 'openGeoDB:')
            ]
        ) as tags
        """

    def _sql_escape(self, value: str) -> str:
        """Escape value for SQL query."""
        return value.replace("'", "''")

    def _sql_to_parquet_file(self, sql_query: str, file_path: Path) -> "duckdb.DuckDBPyRelation":
        relation = self.connection.sql(sql_query)
        return self._save_parquet_file(relation, file_path)

    def _save_parquet_file(
        self, relation: "duckdb.DuckDBPyRelation", file_path: Path
    ) -> "duckdb.DuckDBPyRelation":
        self.connection.sql(f"""
            COPY (
                SELECT * FROM ({relation.sql_query()})
            ) TO '{file_path}' (FORMAT 'parquet', PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 25000)
        """)
        return self.connection.sql(f"""
            SELECT * FROM read_parquet('{file_path}/**')
        """)

    def _calculate_unique_ids_to_parquet(
        self, file_path: Path, result_path: Optional[Path] = None
    ) -> "duckdb.DuckDBPyRelation":
        if result_path is None:
            result_path = file_path / "distinct"

        self.connection.sql(f"""
            COPY (
                SELECT id FROM read_parquet('{file_path}/**') GROUP BY id
            ) TO '{result_path}' (FORMAT 'parquet', PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 25000)
        """)

        return self.connection.sql(f"""
            SELECT * FROM read_parquet('{result_path}/**')
        """)

    def _get_filtered_nodes_with_geometry(
        self,
        osm_parquet_files: ConvertedOSMParquetFiles,
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        nodes_with_geometry = self.connection.sql(f"""
            SELECT
                n.id,
                n.tags,
                ST_Point(n.lon, n.lat) geometry
            FROM ({osm_parquet_files.nodes_valid_with_tags.sql_query()}) n
            SEMI JOIN ({osm_parquet_files.nodes_filtered_ids.sql_query()}) fn ON n.id = fn.id
        """)
        nodes_parquet = self._save_parquet_file_with_geometry(
            relation=nodes_with_geometry,
            file_path=Path(tmp_dir_name) / "filtered_nodes_with_geometry",
        )
        return nodes_parquet

    def _get_required_nodes_with_structs(
        self,
        osm_parquet_files: ConvertedOSMParquetFiles,
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        nodes_with_structs = self.connection.sql(f"""
            SELECT
                n.id,
                struct_pack(x := lon, y := lat)::POINT_2D point
            FROM ({osm_parquet_files.nodes_valid_with_tags.sql_query()}) n
            SEMI JOIN ({osm_parquet_files.nodes_required_ids.sql_query()}) rn ON n.id = rn.id
        """)
        nodes_parquet = self._save_parquet_file(
            relation=nodes_with_structs,
            file_path=Path(tmp_dir_name) / "required_nodes_with_points",
        )
        return nodes_parquet

    def _get_required_ways_with_linestrings(
        self,
        osm_parquet_files: ConvertedOSMParquetFiles,
        required_nodes_with_structs: "duckdb.DuckDBPyRelation",
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        total_required_ways = osm_parquet_files.ways_required_ids.count("id").fetchone()[0]

        required_ways_with_linestrings_path = Path(tmp_dir_name) / "required_ways_with_linestrings"
        required_ways_with_linestrings_path.mkdir(parents=True, exist_ok=True)

        if total_required_ways == 0:
            empty_file_path = str(required_ways_with_linestrings_path / "empty.parquet")
            self.connection.sql("CREATE OR REPLACE TABLE x(id STRING, linestring LINESTRING_2D);")
            self.connection.table("x").to_parquet(empty_file_path)
            return self.connection.read_parquet(empty_file_path)

        groups = floor(total_required_ways / self.rows_per_bucket)
        grouped_required_ways_ids_path = Path(tmp_dir_name) / "ways_required_ids_grouped"
        self.connection.sql(f"""
            COPY (
                SELECT
                    *,
                    floor(
                        row_number() OVER (ORDER BY id) / {self.rows_per_bucket}
                    )::INTEGER as "group",
                FROM ({osm_parquet_files.ways_required_ids.sql_query()})
            ) TO '{grouped_required_ways_ids_path}'
            (FORMAT 'parquet', PARTITION_BY ("group"), ROW_GROUP_SIZE 25000)
        """)

        for group in range(groups + 1):
            current_required_ways_ids_group_path = grouped_required_ways_ids_path / f"group={group}"
            current_required_ways_ids_group_relation = self.connection.sql(f"""
                SELECT * FROM read_parquet('{current_required_ways_ids_group_path}/**')
            """)

            ways_with_linestrings = self.connection.sql(f"""
                SELECT id, list(point)::LINESTRING_2D linestring
                FROM (
                    SELECT w.id, n.point
                    FROM ({osm_parquet_files.ways_with_unnested_nodes_refs.sql_query()}) w
                    SEMI JOIN ({current_required_ways_ids_group_relation.sql_query()}) rw
                    ON w.id = rw.id
                    JOIN ({required_nodes_with_structs.sql_query()}) n
                    ON n.id = w.ref
                    ORDER BY w.id, w.ref_idx
                )
                GROUP BY id
            """)
            self._save_parquet_file(
                relation=ways_with_linestrings,
                file_path=required_ways_with_linestrings_path / f"group={group}",
            )

        ways_parquet = self.connection.sql(f"""
            SELECT * FROM read_parquet('{required_ways_with_linestrings_path}/**')
        """)
        return ways_parquet

    def _get_filtered_ways_with_proper_geometry(
        self,
        osm_parquet_files: ConvertedOSMParquetFiles,
        required_ways_with_linestrings: "duckdb.DuckDBPyRelation",
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        ways_with_proper_geometry = self.connection.sql(f"""
            WITH required_ways_with_linestrings AS (
                SELECT
                    w.id,
                    w.tags,
                    w_l.linestring,
                    -- Filter below is based on `_is_closed_way_a_polygon` function from OSMnx
                    -- Filter values taken from https://wiki.openstreetmap.org/wiki/Overpass_turbo/Polygon_Features
                    (
                        -- if first and last nodes are the same
                        ST_Equals(linestring[1]::POINT_2D, linestring[-1]::POINT_2D)
                        -- if the element doesn't have any tags leave it as a Linestring
                        AND tags IS NOT NULL
                        -- if the element is specifically tagged 'area':'no' -> LineString
                        AND NOT (
                            list_contains(map_keys(tags), 'area')
                            AND list_extract(map_extract(tags, 'area'), 1) = 'no'
                        )
                        AND (
                            -- if all features with that key should be polygons -> Polygon
                            list_has_any(map_keys(tags), [
                                'amenity', 'area', 'area:highway', 'boundary',
                                'building', 'building:part', 'craft', 'golf',
                                'historic', 'indoor', 'landuse', 'leisure',
                                'military', 'office', 'place', 'public_transport',
                                'ruins', 'shop', 'tourism'
                            ])
                            -- if the value for that key in the element
                            -- is in the passlist -> Polygon
                            OR (
                                list_contains(map_keys(tags), 'barrier')
                                AND list_has_any(
                                    map_extract(tags, 'barrier'),
                                    ['city_wall', 'ditch', 'hedge', 'retaining_wall', 'spikes']
                                )
                            )
                            OR (
                                list_contains(map_keys(tags), 'highway')
                                AND list_has_any(
                                    map_extract(tags, 'highway'),
                                    ['services', 'rest_area', 'escape', 'elevator']
                                )
                            )
                            OR (
                                list_contains(map_keys(tags), 'power')
                                AND list_has_any(
                                    map_extract(tags, 'power'),
                                    ['plant', 'substation', 'generator', 'transformer']
                                )
                            )
                            OR (
                                list_contains(map_keys(tags), 'railway')
                                AND list_has_any(
                                    map_extract(tags, 'railway'),
                                    ['station', 'turntable', 'roundhouse', 'platform']
                                )
                            )
                            OR (
                                list_contains(map_keys(tags), 'waterway')
                                AND list_has_any(
                                    map_extract(tags, 'waterway'),
                                    ['riverbank', 'dock', 'boatyard', 'dam']
                                )
                            )
                            -- if the value for that key in the element
                            -- is not in the blocklist -> Polygon
                            OR (
                                list_contains(map_keys(tags), 'aeroway')
                                AND NOT list_has_any(map_extract(tags, 'aeroway'), ['taxiway'])
                            )
                            OR (
                                list_contains(map_keys(tags), 'man_made')
                                AND NOT list_has_any(
                                    map_extract(tags, 'man_made'),
                                    ['cutline', 'embankment', 'pipeline']
                                )
                            )
                            OR (
                                list_contains(map_keys(tags), 'natural')
                                AND NOT list_has_any(
                                    map_extract(tags, 'natural'),
                                    ['coastline', 'cliff', 'ridge', 'arete', 'tree_row']
                                )
                            )
                        )
                    ) AS is_polygon
                FROM ({required_ways_with_linestrings.sql_query()}) w_l
                SEMI JOIN ({osm_parquet_files.ways_filtered_ids.sql_query()}) fw ON w_l.id = fw.id
                JOIN ({osm_parquet_files.ways_all_with_tags.sql_query()}) w ON w.id = w_l.id
            ),
            proper_geometries AS (
                SELECT
                    id,
                    tags,
                    (CASE
                        WHEN is_polygon
                        THEN linestring_to_polygon_wkt(linestring)
                        ELSE linestring_to_linestring_wkt(linestring)
                    END)::GEOMETRY AS geometry
                FROM
                    required_ways_with_linestrings w
            )
            SELECT id, tags, geometry FROM proper_geometries
        """)
        ways_parquet = self._save_parquet_file_with_geometry(
            relation=ways_with_proper_geometry,
            file_path=Path(tmp_dir_name) / "filtered_ways_with_geometry",
        )
        return ways_parquet

    def _get_filtered_relations_with_geometry(
        self,
        osm_parquet_files: ConvertedOSMParquetFiles,
        required_ways_with_linestrings: "duckdb.DuckDBPyRelation",
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        relations_with_geometry = self.connection.sql(f"""
            WITH unnested_relations AS (
                SELECT
                    r.id,
                    COALESCE(r.ref_role, 'outer') as ref_role,
                    r.ref,
                    linestring_to_linestring_wkt(w.linestring)::GEOMETRY as geometry
                FROM ({osm_parquet_files.relations_with_unnested_way_refs.sql_query()}) r
                SEMI JOIN ({osm_parquet_files.relations_filtered_ids.sql_query()}) fr
                ON r.id = fr.id
                JOIN ({required_ways_with_linestrings.sql_query()}) w
                ON w.id = r.ref
                ORDER BY r.id, r.ref_idx
            ),
            relations_with_geometries AS (
                SELECT id, ref_role, geom geometry
                FROM (
                    SELECT
                        id,
                        ref_role,
                        UNNEST(
                            ST_Dump(ST_LineMerge(ST_Collect(list(geometry)))), recursive := true
                        ),
                    FROM unnested_relations
                    GROUP BY id, ref_role
                )
                WHERE ST_NPoints(geom) >= 4
            ),
            valid_relations AS (
                SELECT id, is_valid
                FROM (
                    SELECT
                        id,
                        bool_and(
                            ST_Equals(ST_StartPoint(geometry), ST_EndPoint(geometry))
                        ) is_valid
                    FROM relations_with_geometries
                    GROUP BY id
                )
                WHERE is_valid = true
            ),
            unioned_geometries AS (
                SELECT id, ref_role, ST_Union_Agg(ST_MakePolygon(geometry)) geometry
                FROM relations_with_geometries
                SEMI JOIN valid_relations ON relations_with_geometries.id = valid_relations.id
                GROUP BY id, ref_role
            ),
            final_geometries AS (
                SELECT
                    outers.id,
                    CASE WHEN inners.id IS NOT NULL THEN
                        ST_Difference(outers.geometry, inners.geometry)
                    ELSE
                        outers.geometry
                    END AS geometry
                FROM (
                    SELECT * FROM
                    unioned_geometries
                    WHERE ref_role = 'outer'
                ) outers
                LEFT JOIN (
                    SELECT * FROM
                    unioned_geometries
                    WHERE ref_role = 'inner'
                ) inners
                ON outers.id = inners.id
            )
            SELECT r_g.id, r.tags, r_g.geometry
            FROM final_geometries r_g
            JOIN ({osm_parquet_files.relations_all_with_tags.sql_query()}) r
            ON r.id = r_g.id
        """)
        relations_parquet = self._save_parquet_file_with_geometry(
            relation=relations_with_geometry,
            file_path=Path(tmp_dir_name) / "filtered_relations_with_geometry",
        )
        return relations_parquet

    def _save_parquet_file_with_geometry(
        self, relation: "duckdb.DuckDBPyRelation", file_path: Path
    ) -> "duckdb.DuckDBPyRelation":
        self.connection.sql(f"""
            COPY (
                SELECT
                    * EXCLUDE (geometry), ST_AsWKB(geometry) geometry_wkb
                FROM ({relation.sql_query()})
            ) TO '{file_path}' (FORMAT 'parquet', PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 25000)
        """)
        return self.connection.sql(f"""
            SELECT * EXCLUDE (geometry_wkb), ST_GeomFromWKB(geometry_wkb) geometry
            FROM read_parquet('{file_path}/**')
        """)

    def _concatenate_results_to_geoparquet(
        self,
        parsed_data: ParsedOSMFeatures,
        tmp_dir_name: str,
        save_file_path: Path,
        explode_tags: bool,
    ) -> None:
        import geoarrow.pyarrow as ga
        from geoarrow.pyarrow import io

        select_clauses = [
            *self._generate_osm_tags_sql_select(parsed_data, explode_tags),
            "geometry",
        ]

        node_select_clauses = ["'node/' || id as feature_id", *select_clauses]
        way_select_clauses = ["'way/' || id as feature_id", *select_clauses]
        relation_select_clauses = ["'relation/' || id as feature_id", *select_clauses]

        unioned_features = self.connection.sql(f"""
            SELECT {', '.join(node_select_clauses)}
            FROM ({parsed_data.nodes.sql_query()}) n
            UNION ALL
            SELECT {', '.join(way_select_clauses)}
            FROM ({parsed_data.ways.sql_query()}) w
            UNION ALL
            SELECT {', '.join(relation_select_clauses)}
            FROM ({parsed_data.relations.sql_query()}) r
        """)

        grouped_features = self._parse_features_relation_to_groups(unioned_features, explode_tags)

        valid_features_full_relation = self.connection.sql(f"""
            SELECT * FROM ({grouped_features.sql_query()})
            WHERE ST_IsValid(geometry)
        """)

        valid_features_parquet_path = Path(tmp_dir_name) / "osm_valid_elements"
        valid_features_parquet_relation = self._save_parquet_file_with_geometry(
            valid_features_full_relation,
            valid_features_parquet_path,
        )

        valid_features_parquet_table = pq.read_table(valid_features_parquet_path)

        is_empty = valid_features_parquet_table.num_rows == 0

        if not is_empty:
            geometry_column = ga.as_wkb(
                ga.with_crs(valid_features_parquet_table.column("geometry_wkb"), WGS84_CRS)
            )
        else:
            geometry_column = ga.as_wkb(gpd.GeoSeries([], crs=WGS84_CRS))

        valid_features_parquet_table = valid_features_parquet_table.append_column(
            GEOMETRY_COLUMN, geometry_column
        )
        valid_features_parquet_table = valid_features_parquet_table.drop("geometry_wkb")

        parquet_tables = [valid_features_parquet_table]

        invalid_features_full_relation = self.connection.sql(f"""
            SELECT * FROM ({grouped_features.sql_query()}) a
            ANTI JOIN ({valid_features_parquet_relation.sql_query()}) b
            ON a.feature_id = b.feature_id
        """)

        total_nodes = parsed_data.nodes.count("id").fetchone()[0]
        total_ways = parsed_data.ways.count("id").fetchone()[0]
        total_relations = parsed_data.relations.count("id").fetchone()[0]
        total_features = total_nodes + total_ways + total_relations

        valid_features = valid_features_parquet_relation.count("feature_id").fetchone()[0]

        invalid_features = total_features - valid_features

        if invalid_features > 0:
            groups = floor(invalid_features / self.rows_per_bucket)
            grouped_invalid_features_result_parquet = (
                Path(tmp_dir_name) / "osm_invalid_elements_grouped"
            )
            self.connection.sql(f"""
                COPY (
                    SELECT
                        * EXCLUDE (geometry), ST_AsWKB(geometry) geometry_wkb,
                        floor(
                            row_number() OVER (ORDER BY feature_id) / {self.rows_per_bucket}
                        )::INTEGER as "group",
                    FROM ({invalid_features_full_relation.sql_query()})
                ) TO '{grouped_invalid_features_result_parquet}'
                (FORMAT 'parquet', PARTITION_BY ("group"), ROW_GROUP_SIZE 25000)
            """)

            for group in range(groups + 1):
                current_invalid_features_group_path = (
                    grouped_invalid_features_result_parquet / f"group={group}"
                )
                current_invalid_features_group_table = pq.read_table(
                    current_invalid_features_group_path
                ).drop("group")
                valid_geometry_column = ga.as_wkb(
                    ga.as_geoarrow(
                        ga.to_geopandas(
                            ga.with_crs(
                                current_invalid_features_group_table.column("geometry_wkb"),
                                WGS84_CRS,
                            )
                        ).make_valid()
                    )
                )

                current_invalid_features_group_table = (
                    current_invalid_features_group_table.append_column(
                        GEOMETRY_COLUMN, valid_geometry_column
                    )
                )
                current_invalid_features_group_table = current_invalid_features_group_table.drop(
                    "geometry_wkb"
                )
                parquet_tables.append(current_invalid_features_group_table)

        joined_parquet_table: pa.Table = pa.concat_tables(parquet_tables)

        is_empty = joined_parquet_table.num_rows == 0

        empty_columns = []
        for column_name in joined_parquet_table.column_names:
            if column_name in (FEATURES_INDEX, GEOMETRY_COLUMN):
                continue
            if (
                is_empty
                or pa.compute.all(
                    pa.compute.is_null(joined_parquet_table.column(column_name))
                ).as_py()
            ):
                empty_columns.append(column_name)

        if empty_columns:
            joined_parquet_table = joined_parquet_table.drop(empty_columns)

        io.write_geoparquet_table(
            joined_parquet_table, save_file_path, primary_geometry_column=GEOMETRY_COLUMN
        )

    def _generate_osm_tags_sql_select(
        self, parsed_data: ParsedOSMFeatures, explode_tags: bool
    ) -> list[str]:
        """Prepare features filter clauses based on tags filter."""
        osm_tag_keys_select_clauses = []

        # TODO: elif keep other tags
        if not self.merged_tags_filter and not explode_tags:
            osm_tag_keys_select_clauses = ["tags"]
        elif not self.merged_tags_filter and explode_tags:
            osm_tag_keys = set()
            for elements in (
                parsed_data.nodes,
                parsed_data.ways,
                parsed_data.relations,
            ):
                found_tag_keys = [row[0] for row in self.connection.sql(f"""
                        SELECT DISTINCT UNNEST(map_keys(tags)) tag_key
                        FROM ({elements.sql_query()})
                """).fetchall()]
                osm_tag_keys.update(found_tag_keys)
            osm_tag_keys_select_clauses = [
                f"list_extract(map_extract(tags, '{osm_tag_key}'), 1) as \"{osm_tag_key}\""
                for osm_tag_key in sorted(list(osm_tag_keys))
            ]
        elif self.merged_tags_filter and not explode_tags:
            filter_tag_clauses = []
            for filter_tag_key, filter_tag_value in self.merged_tags_filter.items():
                if isinstance(filter_tag_value, bool) and filter_tag_value:
                    filter_tag_clauses.append(f"tag_entry.key = '{filter_tag_key}'")
                elif isinstance(filter_tag_value, str):
                    escaped_value = self._sql_escape(filter_tag_value)
                    filter_tag_clauses.append(
                        f"(tag_entry.key = '{filter_tag_key}' AND tag_entry.value ="
                        f" '{escaped_value}')"
                    )
                elif isinstance(filter_tag_value, list) and filter_tag_value:
                    values_list = [f"'{self._sql_escape(value)}'" for value in filter_tag_value]
                    filter_tag_clauses.append(
                        f"(tag_entry.key = '{filter_tag_key}' AND tag_entry.value IN"
                        f" ({', '.join(values_list)}))"
                    )
            osm_tag_keys_select_clauses = [f"""
                map_from_entries(
                    [
                        tag_entry
                        for tag_entry in map_entries(tags)
                        if {" OR ".join(filter_tag_clauses)}
                    ]
                ) as tags
            """]
        elif self.merged_tags_filter and explode_tags:
            for filter_tag_key, filter_tag_value in self.merged_tags_filter.items():
                if isinstance(filter_tag_value, bool) and filter_tag_value:
                    osm_tag_keys_select_clauses.append(
                        f"list_extract(map_extract(tags, '{filter_tag_key}'), 1) as"
                        f' "{filter_tag_key}"'
                    )
                elif isinstance(filter_tag_value, str):
                    escaped_value = self._sql_escape(filter_tag_value)
                    osm_tag_keys_select_clauses.append(f"""
                        CASE WHEN list_extract(
                            map_extract(tags, '{filter_tag_key}'), 1
                        ) = '{escaped_value}'
                        THEN '{escaped_value}'
                        ELSE NULL
                        END as "{filter_tag_key}"
                    """)
                elif isinstance(filter_tag_value, list) and filter_tag_value:
                    values_list = [f"'{self._sql_escape(value)}'" for value in filter_tag_value]
                    osm_tag_keys_select_clauses.append(f"""
                        CASE WHEN list_extract(
                            map_extract(tags, '{filter_tag_key}'), 1
                        ) IN ({', '.join(values_list)})
                        THEN list_extract(map_extract(tags, '{filter_tag_key}'), 1)
                        ELSE NULL
                        END as "{filter_tag_key}"
                    """)

        if len(osm_tag_keys_select_clauses) > 100:
            warnings.warn(
                "Select clause contains more than 100 columns"
                f" (found {len(osm_tag_keys_select_clauses)} columns)."
                " Query might fail with insufficient memory resources."
                " Consider applying more restrictive OsmTagsFilter for parsing.",
                stacklevel=1,
            )

        return osm_tag_keys_select_clauses

    def _parse_features_relation_to_groups(
        self,
        features_relation: "duckdb.DuckDBPyRelation",
        explode_tags: bool,
    ) -> "duckdb.DuckDBPyRelation":
        """
        Optionally group raw OSM features into groups defined in `GroupedOsmTagsFilter`.

        Creates new features based on definition from `GroupedOsmTagsFilter`.
        Returns transformed DuckDB relation with columns based on group names from the filter.
        Values are built by concatenation of matching tag key and value with
        an equal sign (eg. amenity=parking). Since many tags can match a definition
        of a single group, a first match is used as a feature value.

        Args:
            features_relation (duckdb.DuckDBPyRelation): Generated features from the loader.
            explode_tags (bool, optional): Whether to split tags into columns based on OSM tag keys.
                Defaults to True.

        Returns:
            duckdb.DuckDBPyRelation: Parsed features_relation.
        """
        if not self.tags_filter or not is_expected_type(self.tags_filter, GroupedOsmTagsFilter):
            return features_relation

        grouped_features_relation: "duckdb.DuckDBPyRelation"
        grouped_tags_filter = cast(GroupedOsmTagsFilter, self.tags_filter)

        if explode_tags:
            case_clauses = []
            for group_name in sorted(grouped_tags_filter.keys()):
                osm_filter = grouped_tags_filter[group_name]
                case_when_clauses = []
                for osm_tag_key, osm_tag_value in osm_filter.items():
                    if isinstance(osm_tag_value, bool) and osm_tag_value:
                        case_when_clauses.append(
                            f"WHEN \"{osm_tag_key}\" IS NOT NULL THEN '{osm_tag_key}=' ||"
                            f' "{osm_tag_key}"'
                        )
                    elif isinstance(osm_tag_value, str):
                        escaped_value = self._sql_escape(osm_tag_value)
                        case_when_clauses.append(
                            f"WHEN \"{osm_tag_key}\" = '{escaped_value}' THEN '{osm_tag_key}=' ||"
                            f' "{osm_tag_key}"'
                        )
                    elif isinstance(osm_tag_value, list) and osm_tag_value:
                        values_list = [f"'{self._sql_escape(value)}'" for value in osm_tag_value]
                        case_when_clauses.append(
                            f"WHEN \"{osm_tag_key}\" IN ({', '.join(values_list)}) THEN"
                            f" '{osm_tag_key}=' || \"{osm_tag_key}\""
                        )
                case_clause = f'CASE {" ".join(case_when_clauses)} END AS "{group_name}"'
                case_clauses.append(case_clause)

            joined_case_clauses = ", ".join(case_clauses)
            grouped_features_relation = self.connection.sql(f"""
                SELECT feature_id, {joined_case_clauses}, geometry
                FROM ({features_relation.sql_query()})
            """)
        else:
            case_clauses = []
            group_names = sorted(grouped_tags_filter.keys())
            for group_name in group_names:
                osm_filter = grouped_tags_filter[group_name]
                case_when_clauses = []
                for osm_tag_key, osm_tag_value in osm_filter.items():
                    element_clause = f"element_at(tags, '{osm_tag_key}')[1]"
                    if isinstance(osm_tag_value, bool) and osm_tag_value:
                        case_when_clauses.append(
                            f"WHEN {element_clause} IS NOT NULL THEN '{osm_tag_key}=' ||"
                            f" {element_clause}"
                        )
                    elif isinstance(osm_tag_value, str):
                        escaped_value = self._sql_escape(osm_tag_value)
                        case_when_clauses.append(
                            f"WHEN {element_clause} = '{escaped_value}' THEN '{osm_tag_key}=' ||"
                            f" {element_clause}"
                        )
                    elif isinstance(osm_tag_value, list) and osm_tag_value:
                        values_list = [f"'{self._sql_escape(value)}'" for value in osm_tag_value]
                        case_when_clauses.append(
                            f"WHEN {element_clause} IN ({', '.join(values_list)}) THEN"
                            f" '{osm_tag_key}=' || {element_clause}"
                        )
                case_clause = f'CASE {" ".join(case_when_clauses)} END'
                case_clauses.append(case_clause)

            group_names_as_sql_strings = [f"'{group_name}'" for group_name in group_names]
            groups_map = (
                f"map([{', '.join(group_names_as_sql_strings)}], [{', '.join(case_clauses)}])"
            )
            non_null_groups_map = f"""map_from_entries(
                [
                    tag_entry
                    for tag_entry in map_entries({groups_map})
                    if tag_entry.value IS NOT NULL
                ]
            ) as tags"""

            grouped_features_relation = self.connection.sql(f"""
                SELECT feature_id, {non_null_groups_map}, geometry
                FROM ({features_relation.sql_query()})
            """)

        return grouped_features_relation
