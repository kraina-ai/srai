"""
PBF File Handler.

This module contains a handler capable of parsing a PBF file into a GeoDataFrame.
"""

import hashlib
import json
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from srai._optional import import_optional_dependencies
from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, WGS84_CRS
from srai.geometry import get_geometry_hash
from srai.loaders.osm_loaders.filters import OsmTagsFilter

if TYPE_CHECKING:
    import os

    import duckdb


class PbfFileHandler:
    """
    PbfFileHandler.

    PBF(Protocolbuffer Binary Format)[1] file handler is a ...
    """

    class FilteredOSMIds(NamedTuple):
        """"""

        nodes_required: "duckdb.DuckDBPyRelation"
        nodes_filtered: "duckdb.DuckDBPyRelation"
        ways_required: "duckdb.DuckDBPyRelation"
        ways_filtered: "duckdb.DuckDBPyRelation"
        relations_filtered: "duckdb.DuckDBPyRelation"

    class ParsedOSMFeatures(NamedTuple):
        """"""

        nodes: "duckdb.DuckDBPyRelation"
        ways: "duckdb.DuckDBPyRelation"
        relations: "duckdb.DuckDBPyRelation"

    def __init__(
        self,
        tags: Optional[OsmTagsFilter] = None,
        region_geometry: Optional[BaseGeometry] = None,
        working_directory: Union[str, Path] = "files",
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
            working_directory (Union[str, Path], optional): Directory where to save
                the parsed `*.parquet` files. Defaults to "files".
        """
        import_optional_dependencies(dependency_group="osm", modules=["duckdb"])
        self.filter_tags = tags
        self.region_geometry = region_geometry
        self.working_directory = Path(working_directory)
        self.working_directory.mkdir(parents=True, exist_ok=True)

    def get_features_gdf(
        self, file_paths: Sequence[Union[str, "os.PathLike[str]"]]
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
        import geoarrow.pyarrow as ga
        from geoarrow.pyarrow import io

        with tempfile.TemporaryDirectory(dir=self.working_directory) as tmp_dir_name:
            self._set_up_duckdb_connection(tmp_dir_name)
            parsed_geoparquet_files = []
            for path_no, path in enumerate(file_paths):
                self.path_no = path_no + 1
                parsed_geoparquet_file = self._parse_pbf_file(path, tmp_dir_name)
                parsed_geoparquet_files.append(parsed_geoparquet_file)

        parquet_tables = [
            io.read_geoparquet_table(parsed_parquet_file)
            for parsed_parquet_file in parsed_geoparquet_files
        ]
        joined_parquet_table: pa.Table = pa.concat_tables(parquet_tables)
        gdf_parquet = gpd.GeoDataFrame(
            data=joined_parquet_table.drop(GEOMETRY_COLUMN).to_pandas(),
            geometry=ga.to_geopandas(joined_parquet_table.column(GEOMETRY_COLUMN)),
        ).set_index(FEATURES_INDEX)

        return gdf_parquet

    def _set_up_duckdb_connection(self, tmp_dir_name: str) -> None:
        import duckdb

        self.connection = duckdb.connect(database=str(Path(tmp_dir_name) / "db.duckdb"))
        for extension_name in ["parquet", "spatial"]:
            self.connection.install_extension(extension_name)
            self.connection.load_extension(extension_name)

    def _parse_pbf_file(self, pbf_path: Union[str, "os.PathLike[str]"], tmp_dir_name: str) -> Path:
        result_file_name = self._generate_geoparquet_result_file_name(pbf_path)
        result_file_path = Path(self.working_directory) / result_file_name
        if not result_file_path.exists():
            elements = self.connection.sql(f"SELECT * FROM ST_READOSM('{Path(pbf_path)}');")
            filtered_elements_ids = self._prefilter_elements_ids(elements, tmp_dir_name)

            nodes_with_geometry = self._parse_nodes(elements, filtered_elements_ids, tmp_dir_name)
            ways_with_linestrings_geometry = self._parse_ways_to_linestrings(
                elements, nodes_with_geometry, filtered_elements_ids, tmp_dir_name
            )
            ways_with_proper_geometry = self._parse_ways_to_proper_geometry(
                ways_with_linestrings_geometry, tmp_dir_name
            )
            relations_with_geometry = self._parse_relations(
                elements, ways_with_linestrings_geometry, filtered_elements_ids, tmp_dir_name
            )

            self._concatenate_results_to_geoparquet(
                PbfFileHandler.ParsedOSMFeatures(
                    nodes=nodes_with_geometry,
                    ways=ways_with_proper_geometry,
                    relations=relations_with_geometry,
                ),
                tmp_dir_name=tmp_dir_name,
                save_file_path=result_file_path,
            )

        return result_file_path

    def _sql_to_parquet_file(self, sql_query: str, file_path: Path) -> "duckdb.DuckDBPyRelation":
        relation = self.connection.sql(sql_query)
        return self._save_parquet_file(relation, file_path)

    def _save_parquet_file(
        self, relation: "duckdb.DuckDBPyRelation", file_path: Path
    ) -> "duckdb.DuckDBPyRelation":
        self.connection.sql(f"""
            COPY (
                SELECT * FROM ({relation.sql_query()})
            ) TO '{file_path}' (FORMAT 'parquet')
        """)
        return self.connection.sql(f"""
            SELECT * FROM read_parquet('{file_path}')
        """)

    def _save_parquet_file_with_geometry(
        self, elements: "duckdb.DuckDBPyRelation", file_path: Path
    ) -> "duckdb.DuckDBPyRelation":
        self.connection.sql(f"""
            COPY (
                SELECT * EXCLUDE (geometry), ST_AsWKB(geometry) geometry_wkb FROM ({elements.sql_query()})
            ) TO '{file_path}' (FORMAT 'parquet')
        """)
        return self.connection.sql(f"""
            SELECT * EXCLUDE (geometry_wkb), ST_GeomFromWKB(geometry_wkb) geometry
            FROM read_parquet('{file_path}')
        """)

    def _prefilter_elements_ids(
        self, elements: "duckdb.DuckDBPyRelation", tmp_dir_name: str
    ) -> FilteredOSMIds:
        sql_filter = self._generate_osm_tags_sql_filter()
        intersection_filter = (
            f"ST_Intersects(ST_Point(lon, lat), ST_GeomFromText('{self.region_geometry.wkt}'))"
            if self.region_geometry is not None
            else "true"
        )
        # NODES - VALID (NV)
        # - select all with kind = 'node'
        # - select all with lat and lon not empty
        self.connection.sql(f"""
            SELECT * FROM ({elements.sql_query()}) w
            WHERE kind = 'node'
            AND lat IS NOT NULL AND lon IS NOT NULL
        """).to_view("nodes", replace=True)
        # print(
        #     "nodes - view",
        #     self.connection.sql("SELECT DISTINCT id FROM nodes").count("id").fetchone(),
        # )
        nodes_valid = self._sql_to_parquet_file(
            sql_query="""
            SELECT DISTINCT id FROM nodes
            """,
            file_path=Path(tmp_dir_name) / "nodes_valid.parquet",
        )
        # print(
        #     "nodes_valid",
        #     self.connection.sql(f"SELECT DISTINCT id FROM ({nodes_valid.sql_query()}) nodes_valid")
        #     .count("id")
        #     .fetchone(),
        # )
        # NODES - INTERSECTING (NI)
        # - select all from NV which intersect given geometry filter
        nodes_intersecting = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT DISTINCT id FROM nodes n
            SEMI JOIN ({nodes_valid.sql_query()}) nv ON n.id = nv.id
            WHERE {intersection_filter} = true
            """,
            file_path=Path(tmp_dir_name) / "nodes_intersecting.parquet",
        )
        # print(
        #     "nodes_intersecting",
        #     self.connection.sql(
        #         f"SELECT DISTINCT id FROM ({nodes_intersecting.sql_query()}) nodes_intersecting"
        #     )
        #     .count("id")
        #     .fetchone(),
        # )
        # NODES - FILTERED (NF)
        # - select all from NI with tags filter
        nodes_filtered = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT DISTINCT id FROM nodes n
            SEMI JOIN ({nodes_intersecting.sql_query()}) ni ON n.id = ni.id
            WHERE {sql_filter}
            """,
            file_path=Path(tmp_dir_name) / "nodes_filtered.parquet",
        )
        # print(
        #     "nodes_filtered",
        #     self.connection.sql(
        #         f"SELECT DISTINCT id FROM ({nodes_filtered.sql_query()}) nodes_filtered"
        #     )
        #     .count("id")
        #     .fetchone(),
        # )

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
        # print("ways", self.connection.sql("SELECT DISTINCT id FROM ways").count("id").fetchone())
        unnested_way_refs = self._sql_to_parquet_file(
            sql_query="""
            SELECT w.id, UNNEST(refs) as ref
            FROM ways w
            """,
            file_path=Path(tmp_dir_name) / "unnested_way_refs.parquet",
        )
        # print(
        #     "unnested_way_refs",
        #     self.connection.sql(
        #         f"SELECT DISTINCT id FROM ({unnested_way_refs.sql_query()}) unnested_way_refs"
        #     )
        #     .count("id")
        #     .fetchone(),
        # )
        ways_valid = self._sql_to_parquet_file(
            sql_query=f"""
            WITH total_way_refs AS (
                SELECT id, ref
                FROM ({unnested_way_refs.sql_query()}) unnested_way_refs
            ),
            unmatched_way_refs AS (
                SELECT id, ref
                FROM ({unnested_way_refs.sql_query()}) w
                ANTI JOIN ({nodes_valid.sql_query()}) nv ON nv.id = w.ref
            )
            SELECT DISTINCT id
            FROM total_way_refs
            EXCEPT
            SELECT DISTINCT id
            FROM unmatched_way_refs
            """,
            file_path=Path(tmp_dir_name) / "ways_valid.parquet",
        )
        # print(
        #     "ways_valid",
        #     self.connection.sql(f"SELECT DISTINCT id FROM ({ways_valid.sql_query()}) ways_valid")
        #     .count("id")
        #     .fetchone(),
        # )
        # WAYS - INTERSECTING (WI)
        # - select all from WV with joining any from NV on ref
        ways_intersecting = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT DISTINCT uwr.id
            FROM unnested_way_refs uwr
            SEMI JOIN ({ways_valid.sql_query()}) wv ON uwr.id = wv.id
            SEMI JOIN ({nodes_intersecting.sql_query()}) n ON n.id = uwr.ref
            """,
            file_path=Path(tmp_dir_name) / "ways_intersecting.parquet",
        )
        # print(
        #     "ways_intersecting",
        #     self.connection.sql("SELECT DISTINCT id FROM ways_intersecting").count("id").fetchone(),
        # )
        # WAYS - FILTERED (WF)
        # - select all from WI with tags filter
        ways_filtered = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT DISTINCT id FROM ways w
            SEMI JOIN ({ways_intersecting.sql_query()}) wi ON w.id = wi.id
            WHERE {sql_filter}
            """,
            file_path=Path(tmp_dir_name) / "ways_filtered.parquet",
        )
        # print(
        #     "ways_filtered",
        #     self.connection.sql(
        #         f"SELECT DISTINCT id FROM ({ways_filtered.sql_query()}) ways_filtered"
        #     )
        #     .count("id")
        #     .fetchone(),
        # )

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
        # print(
        #     "relations",
        #     self.connection.sql("SELECT DISTINCT id FROM relations").count("id").fetchone(),
        # )
        filtered_relation_refs = self._sql_to_parquet_file(
            sql_query="""
            WITH unnested_relation_refs AS (
                SELECT r.id, UNNEST(refs) as ref, UNNEST(ref_types) as ref_type,
                FROM relations r
            )
            SELECT id, ref
            FROM unnested_relation_refs
            WHERE ref_type = 'way'
            """,
            file_path=Path(tmp_dir_name) / "filtered_relation_refs.parquet",
        )
        # print(
        #     "filtered_relation_refs",
        #     self.connection.sql(
        #         f"SELECT DISTINCT id FROM ({filtered_relation_refs.sql_query()})"
        #         " filtered_relation_refs"
        #     )
        #     .count("id")
        #     .fetchone(),
        # )
        relations_valid = self._sql_to_parquet_file(
            sql_query=f"""
            WITH total_relation_refs AS (
                SELECT id, ref
                FROM ({filtered_relation_refs.sql_query()}) frr
            ),
            unmatched_relation_refs AS (
                SELECT id, ref
                FROM ({filtered_relation_refs.sql_query()}) r
                ANTI JOIN ({ways_valid.sql_query()}) wv ON wv.id = r.ref
            )
            SELECT DISTINCT id
            FROM total_relation_refs
            EXCEPT
            SELECT DISTINCT id
            FROM unmatched_relation_refs
            """,
            file_path=Path(tmp_dir_name) / "relations_valid.parquet",
        )
        # print(
        #     "relations_valid",
        #     self.connection.sql(
        #         f"SELECT DISTINCT id FROM ({relations_valid.sql_query()}) relations_valid"
        #     )
        #     .count("id")
        #     .fetchone(),
        # )
        # RELATIONS - INTERSECTING (RI)
        # - select all from RW with joining any from RV on ref
        relations_intersecting = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT DISTINCT frr.id
            FROM ({filtered_relation_refs.sql_query()}) frr
            SEMI JOIN ({relations_valid.sql_query()}) rv ON frr.id = rv.id
            SEMI JOIN ({ways_intersecting.sql_query()}) wi ON wi.id = frr.ref
            """,
            file_path=Path(tmp_dir_name) / "relations_intersecting.parquet",
        )
        # print(
        #     "relations_intersecting",
        #     self.connection.sql("SELECT DISTINCT id FROM relations_intersecting")
        #     .count("id")
        #     .fetchone(),
        # )
        # RELATIONS - FILTERED (RF)
        # - select all from RI with tags filter
        relations_filtered = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT DISTINCT id FROM relations r
            SEMI JOIN ({relations_intersecting.sql_query()}) ri ON r.id = ri.id
            WHERE {sql_filter}
            """,
            file_path=Path(tmp_dir_name) / "relations_filtered.parquet",
        )
        # print(
        #     "relations_filtered",
        #     self.connection.sql(
        #         f"SELECT DISTINCT id FROM ({relations_filtered.sql_query()}) relations_filtered"
        #     )
        #     .count("id")
        #     .fetchone(),
        # )

        # GET ALL RELATIONS IDS
        # - filtered - all IDs from RF
        filtered_relations_ids = self._sql_to_parquet_file(
            sql_query=f"SELECT DISTINCT id FROM ({relations_filtered.sql_query()}) rf",
            file_path=Path(tmp_dir_name) / "filtered_relations_ids.parquet",
        )

        # GET ALL WAYS IDS
        # - required - all IDs from WF
        #   + all needed to construct relations from RF
        # - filtered - all IDs from WF
        required_ways_ids = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT DISTINCT ref as id
            FROM ({filtered_relation_refs.sql_query()}) frr
            SEMI JOIN ({filtered_relations_ids.sql_query()}) fri ON fri.id = frr.id
            UNION
            SELECT DISTINCT id FROM ({ways_filtered.sql_query()}) wf
            """,
            file_path=Path(tmp_dir_name) / "required_ways_ids.parquet",
        )
        filtered_ways_ids = self._sql_to_parquet_file(
            sql_query=f"SELECT DISTINCT id FROM ({ways_filtered.sql_query()}) wf",
            file_path=Path(tmp_dir_name) / "filtered_ways_ids.parquet",
        )

        # GET ALL NODES IDS
        # - required - all IDs from NF
        #   + all needed to construct ways from WF and relations from RF
        # - filtered - all IDs from NF
        required_nodes_ids = self._sql_to_parquet_file(
            sql_query=f"""
            SELECT DISTINCT ref as id
            FROM ({unnested_way_refs.sql_query()}) uwr
            SEMI JOIN ({required_ways_ids.sql_query()}) rwi ON rwi.id = uwr.id
            UNION
            SELECT DISTINCT ref as id
            FROM ({unnested_way_refs.sql_query()}) uwr
            SEMI JOIN ({filtered_ways_ids.sql_query()}) fwi ON fwi.id = uwr.id
            UNION
            SELECT DISTINCT id FROM ({nodes_filtered.sql_query()}) nf
            """,
            file_path=Path(tmp_dir_name) / "required_nodes_ids.parquet",
        )
        filtered_nodes_ids = self._sql_to_parquet_file(
            sql_query=f"SELECT DISTINCT id FROM ({nodes_filtered.sql_query()}) nf",
            file_path=Path(tmp_dir_name) / "filtered_nodes_ids.parquet",
        )

        return PbfFileHandler.FilteredOSMIds(
            nodes_required=required_nodes_ids,
            nodes_filtered=filtered_nodes_ids,
            ways_required=required_ways_ids,
            ways_filtered=filtered_ways_ids,
            relations_filtered=filtered_relations_ids,
        )

    def _escape(self, value: str) -> str:
        """Escape value for SQL query."""
        return value.replace("'", "''")

    def _generate_osm_tags_sql_filter(self) -> str:
        """Prepare features filter clauses based on tags filter."""
        filter_clauses = ["(1=1)"]

        if self.filter_tags:
            filter_clauses.clear()

            for filter_tag_key, filter_tag_value in self.filter_tags.items():
                if isinstance(filter_tag_value, bool) and filter_tag_value:
                    filter_clauses.append(f"(list_contains(map_keys(tags), '{filter_tag_key}'))")
                elif isinstance(filter_tag_value, str):
                    escaped_value = self._escape(filter_tag_value)
                    filter_clauses.append(
                        f"(list_extract(map_extract(tags, '{filter_tag_key}'), 1) ="
                        f" '{escaped_value}')"
                    )
                elif isinstance(filter_tag_value, list) and filter_tag_value:
                    values_list = [f"'{self._escape(value)}'" for value in filter_tag_value]
                    filter_clauses.append(
                        f"list_has_any(map_extract(tags, '{filter_tag_key}'),"
                        f" [{', '.join(values_list)}])"
                    )

        return " OR ".join(filter_clauses)

    def _generate_osm_tags_sql_select(self, parsed_data: ParsedOSMFeatures) -> str:
        """Prepare features filter clauses based on tags filter."""
        osm_tag_keys = set()
        if not self.filter_tags:
            for elements in [
                parsed_data.nodes,
                parsed_data.ways,
                parsed_data.relations,
            ]:
                found_tag_keys = [row[0] for row in self.connection.sql(f"""
                    SELECT DISTINCT UNNEST(map_keys(tags)) tag_key
                    FROM ({elements.sql_query()})
                """).fetchall()]
                osm_tag_keys.update(found_tag_keys)
        else:
            osm_tag_keys.update(self.filter_tags.keys())

        osm_tag_keys_select_clauses = [
            f"list_extract(map_extract(tags, '{osm_tag_key}'), 1) as \"{osm_tag_key}\""
            for osm_tag_key in sorted(list(osm_tag_keys))
        ]

        if len(osm_tag_keys_select_clauses) > 100:
            warnings.warn(
                "Select clause contains more than 100 columns"
                f" (found {len(osm_tag_keys_select_clauses)} columns)."
                " Query might fail with insufficient memory resources."
                " Consider applying more restrictive OsmTagsFilter for parsing.",
                stacklevel=1,
            )

        return ", ".join(osm_tag_keys_select_clauses)

    def _concatenate_results_to_geoparquet(
        self, parsed_data: ParsedOSMFeatures, tmp_dir_name: str, save_file_path: Path
    ) -> None:
        import geoarrow.pyarrow as ga
        from geoarrow.pyarrow import io

        select_clause = self._generate_osm_tags_sql_select(parsed_data)
        nodes_result_parquet = Path(tmp_dir_name) / "nodes_full.parquet"
        ways_result_parquet = Path(tmp_dir_name) / "ways_full.parquet"
        relations_result_parquet = Path(tmp_dir_name) / "relations_full.parquet"
        self._save_parquet_file_with_geometry(
            self.connection.sql(f"""
            SELECT 'node/' || id as feature_id, {select_clause}, geometry
            FROM ({parsed_data.nodes.sql_query()}) n WHERE is_filtered
        """),
            nodes_result_parquet,
        )
        self._save_parquet_file_with_geometry(
            self.connection.sql(f"""
            SELECT 'way/' || id as feature_id, {select_clause}, geometry
            FROM ({parsed_data.ways.sql_query()}) w
        """),
            ways_result_parquet,
        )
        self._save_parquet_file_with_geometry(
            self.connection.sql(f"""
            SELECT 'relation/' || id as feature_id, {select_clause}, geometry
            FROM ({parsed_data.relations.sql_query()}) r
        """),
            relations_result_parquet,
        )
        print("concatenating results")

        parquet_tables = [
            pq.read_table(parsed_parquet_file)
            for parsed_parquet_file in [
                nodes_result_parquet,
                ways_result_parquet,
                relations_result_parquet,
            ]
        ]
        joined_parquet_table: pa.Table = pa.concat_tables(parquet_tables)
        valid_geometry_column = ga.as_geoarrow(
            ga.to_geopandas(
                ga.with_crs(joined_parquet_table.column("geometry_wkb"), WGS84_CRS)
            ).make_valid()
        )
        joined_parquet_table = joined_parquet_table.append_column(
            GEOMETRY_COLUMN, valid_geometry_column
        )
        joined_parquet_table = joined_parquet_table.drop("geometry_wkb")
        print(joined_parquet_table)
        io.write_geoparquet_table(
            joined_parquet_table, save_file_path, primary_geometry_column=GEOMETRY_COLUMN
        )

    def _generate_geoparquet_result_file_name(
        self, pbf_file_path: Union[str, "os.PathLike[str]"]
    ) -> str:
        pbf_file_name = Path(pbf_file_path).name.removesuffix(".osm.pbf")

        osm_filter_tags_hash_part = "nofilter"
        if self.filter_tags is not None:
            h = hashlib.new("sha256")
            h.update(json.dumps(self.filter_tags).encode())
            osm_filter_tags_hash_part = h.hexdigest()

        clipping_geometry_hash_part = "noclip"
        if self.region_geometry is not None:
            clipping_geometry_hash_part = get_geometry_hash(self.region_geometry)

        result_file_name = (
            f"{pbf_file_name}_{osm_filter_tags_hash_part}_{clipping_geometry_hash_part}.geoparquet"
        )
        return result_file_name

    def _parse_nodes(
        self,
        elements: "duckdb.DuckDBPyRelation",
        filtered_osm_ids: FilteredOSMIds,
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        nodes_with_geometry = self.connection.sql(f"""
            SELECT
                n.id,
                n.tags,
                fn.id IS NOT NULL AS is_filtered,
                ST_Point(n.lon, n.lat) geometry
            FROM ({elements.sql_query()}) n
            SEMI JOIN ({filtered_osm_ids.nodes_required.sql_query()}) rn ON n.id = rn.id
            LEFT JOIN ({filtered_osm_ids.nodes_filtered.sql_query()}) fn ON n.id = fn.id
            WHERE kind = 'node'
        """)
        # print(nodes_with_geometry.sql_query())
        nodes_parquet = self._save_parquet_file_with_geometry(
            elements=nodes_with_geometry, file_path=Path(tmp_dir_name) / "nodes.parquet"
        )
        return nodes_parquet

    def _parse_ways_to_linestrings(
        self,
        elements: "duckdb.DuckDBPyRelation",
        nodes: "duckdb.DuckDBPyRelation",
        filtered_osm_ids: FilteredOSMIds,
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        ways_with_geometry = self.connection.sql(f"""
            WITH ways AS (
                SELECT
                    *
                FROM ({elements.sql_query()}) w
                SEMI JOIN ({filtered_osm_ids.ways_required.sql_query()}) rw ON w.id = rw.id
                WHERE kind = 'way'
            ),
            ways_with_geometry AS (
                SELECT id, ST_MakeLine(list(geometry)) geometry
                FROM (
                    -- Join nodes
                    SELECT w.*, n.geometry
                    FROM (
                        -- Unnest ways
                        SELECT
                            w.id,
                            UNNEST(refs) as ref,
                            UNNEST(range(length(refs))) as idx,
                        FROM ways w
                    ) w
                    JOIN ({nodes.sql_query()}) n
                    ON n.id = w.ref
                    ORDER BY w.id, w.idx
                )
                GROUP BY id
            )
            SELECT
                w_g.id, w.tags, w_g.geometry,
                fw.id IS NOT NULL AS is_filtered
            FROM ways_with_geometry w_g
            JOIN ways w ON w.id = w_g.id
            LEFT JOIN ({filtered_osm_ids.ways_filtered.sql_query()}) fw ON w.id = fw.id
        """)
        # print(ways_with_geometry.sql_query())
        ways_parquet = self._save_parquet_file_with_geometry(
            elements=ways_with_geometry, file_path=Path(tmp_dir_name) / "ways_linestring.parquet"
        )
        return ways_parquet

    def _parse_ways_to_proper_geometry(
        self,
        ways: "duckdb.DuckDBPyRelation",
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        ways_with_geometry = self.connection.sql(f"""
            SELECT
                id,
                tags,
                -- Filter below is based on `_is_closed_way_a_polygon` function from OSMnx
                -- Filter values taken from https://wiki.openstreetmap.org/wiki/Overpass_turbo/Polygon_Features
                CASE WHEN
                    -- if first and last nodes are the same
                    ST_Equals(ST_StartPoint(geometry), ST_EndPoint(geometry))
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
                        -- if the value for that key in the element is in the passlist -> Polygon
                        OR (
                            list_contains(map_keys(tags), 'barrier')
                            AND list_has_any(map_extract(tags, 'barrier'), ['city_wall', 'ditch', 'hedge', 'retaining_wall', 'spikes'])
                        )
                        OR (
                            list_contains(map_keys(tags), 'highway')
                            AND list_has_any(map_extract(tags, 'highway'), ['services', 'rest_area', 'escape', 'elevator'])
                        )
                        OR (
                            list_contains(map_keys(tags), 'power')
                            AND list_has_any(map_extract(tags, 'power'), ['plant', 'substation', 'generator', 'transformer'])
                        )
                        OR (
                            list_contains(map_keys(tags), 'railway')
                            AND list_has_any(map_extract(tags, 'railway'), ['station', 'turntable', 'roundhouse', 'platform'])
                        )
                        OR (
                            list_contains(map_keys(tags), 'waterway')
                            AND list_has_any(map_extract(tags, 'waterway'), ['riverbank', 'dock', 'boatyard', 'dam'])
                        )
                        -- if the value for that key in the element is not in the blocklist -> Polygon
                        OR (
                            list_contains(map_keys(tags), 'aeroway')
                            AND NOT list_has_any(map_extract(tags, 'aeroway'), ['taxiway'])
                        )
                        OR (
                            list_contains(map_keys(tags), 'man_made')
                            AND NOT list_has_any(map_extract(tags, 'man_made'), ['cutline', 'embankment', 'pipeline'])
                        )
                        OR (
                            list_contains(map_keys(tags), 'natural')
                            AND NOT list_has_any(map_extract(tags, 'natural'), ['coastline', 'cliff', 'ridge', 'arete', 'tree_row'])
                        )
                    )
                THEN
                    ST_MakePolygon(geometry)
                ELSE
                    geometry
                END AS geometry
            FROM
                ({ways.sql_query()}) w
            WHERE
                is_filtered
            """)
        # print(ways_with_geometry.sql_query())
        ways_parquet = self._save_parquet_file_with_geometry(
            elements=ways_with_geometry, file_path=Path(tmp_dir_name) / "ways_proper.parquet"
        )
        return ways_parquet

    def _parse_relations(
        self,
        elements: "duckdb.DuckDBPyRelation",
        ways: "duckdb.DuckDBPyRelation",
        filtered_osm_ids: FilteredOSMIds,
        tmp_dir_name: str,
    ) -> "duckdb.DuckDBPyRelation":
        # print(self.connection.sql(f"""
        #     WITH relations AS (
        #         SELECT *
        #         FROM ({elements.sql_query()}) r
        #         SEMI JOIN ({filtered_osm_ids.relations_filtered.sql_query()}) fr ON r.id = fr.id
        #         WHERE kind = 'relation'
        #         and id = 16048460
        #     ),
        #     unnested_relations AS (
        #         SELECT
        #             r.id,
        #             r.ref_type,
        #             COALESCE(r.ref_role, 'outer') ref_role,
        #             r.ref,
        #             w.geometry
        #         FROM (
        #             -- Filter ways
        #             SELECT
        #                 *
        #             FROM (
        #                 -- Unnest relations
        #                 SELECT
        #                     r.id,
        #                     r.tags,
        #                     UNNEST(refs) as ref,
        #                     UNNEST(ref_types) as ref_type,
        #                     UNNEST(ref_roles) as ref_role,
        #                     UNNEST(range(length(refs))) as idx,
        #                 FROM relations r
        #             )
        #             WHERE ref_type = 'way'
        #         ) r
        #         JOIN ({ways.sql_query()}) w
        #         ON w.id = r.ref
        #         ORDER BY r.id, r.idx
        #     ),
        #     relations_with_geometries AS (
        #         SELECT id, ref_role, geom geometry
        #         FROM (
        #             -- Collect geometry
        #             SELECT
        #                 id,
        #                 ref_role,
        #                 UNNEST(ST_Dump(ST_LineMerge(ST_Collect(list(geometry)))), recursive := true),
        #             FROM unnested_relations
        #             GROUP BY id, ref_role
        #         )
        #         WHERE ST_NPoints(geom) >= 4
        #     ),
        #     valid_relations AS (
        #         SELECT id, is_valid
        #         FROM (
        #             SELECT
        #                 id, bool_and(ST_Equals(ST_StartPoint(geometry), ST_EndPoint(geometry))) is_valid
        #             FROM relations_with_geometries
        #             GROUP BY id
        #         )
        #         WHERE is_valid = true
        #     ),
        #     unioned_geometries AS (
        #         SELECT id, ref_role, ST_Union_Agg(ST_MakePolygon(geometry)) geometry
        #         FROM relations_with_geometries
        #         SEMI JOIN valid_relations ON relations_with_geometries.id = valid_relations.id
        #         GROUP BY id, ref_role
        #     ),
        #     final_geometries AS (
        #         SELECT
        #             outers.id,
        #             CASE WHEN inners.id IS NOT NULL THEN
        #                 ST_Difference(outers.geometry, inners.geometry)
        #             ELSE
        #                 outers.geometry
        #             END AS geometry
        #         FROM (
        #             SELECT * FROM
        #             unioned_geometries
        #             WHERE ref_role = 'outer'
        #         ) outers
        #         LEFT JOIN (
        #             SELECT * FROM
        #             unioned_geometries
        #             WHERE ref_role = 'inner'
        #         ) inners
        #         ON outers.id = inners.id
        #     )
        #     SELECT * EXCLUDE(geometry), ST_NPoints(geometry) points FROM relations_with_geometries
        #     ORDER BY ST_NPoints(geometry)
        #     LIMIT 100
        #     --SELECT r.id, r.tags, r_g.geometry
        #     --FROM final_geometries r_g
        #     --JOIN relations r
        #     --ON r.id = r_g.id
        # """).fetchall())
        relations_with_geometry = self.connection.sql(f"""
            WITH relations AS (
                SELECT *
                FROM ({elements.sql_query()}) r
                SEMI JOIN ({filtered_osm_ids.relations_filtered.sql_query()}) fr ON r.id = fr.id
                WHERE kind = 'relation'
                --AND id = 16048460
            ),
            unnested_relations AS (
                SELECT
                    r.id,
                    r.ref_type,
                    COALESCE(r.ref_role, 'outer') ref_role,
                    r.ref,
                    w.geometry
                FROM (
                    -- Filter ways
                    SELECT
                        *
                    FROM (
                        -- Unnest relations
                        SELECT
                            r.id,
                            r.tags,
                            UNNEST(refs) as ref,
                            UNNEST(ref_types) as ref_type,
                            UNNEST(ref_roles) as ref_role,
                            UNNEST(range(length(refs))) as idx,
                        FROM relations r
                    )
                    WHERE ref_type = 'way'
                ) r
                JOIN ({ways.sql_query()}) w
                ON w.id = r.ref
                ORDER BY r.id, r.idx
            ),
            relations_with_geometries AS (
                SELECT id, ref_role, geom geometry
                FROM (
                    -- Collect geometry
                    SELECT
                        id,
                        ref_role,
                        UNNEST(ST_Dump(ST_LineMerge(ST_Collect(list(geometry)))), recursive := true),
                    FROM unnested_relations
                    GROUP BY id, ref_role
                )
                WHERE ST_NPoints(geom) >= 4
            ),
            valid_relations AS (
                SELECT id, is_valid
                FROM (
                    SELECT
                        id, bool_and(ST_Equals(ST_StartPoint(geometry), ST_EndPoint(geometry))) is_valid
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
            SELECT r.id, r.tags, r_g.geometry
            FROM final_geometries r_g
            JOIN relations r
            ON r.id = r_g.id
        """)
        relations_parquet = self._save_parquet_file_with_geometry(
            elements=relations_with_geometry, file_path=Path(tmp_dir_name) / "relations.parquet"
        )
        return relations_parquet
