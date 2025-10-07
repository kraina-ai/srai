"""
H3 regionalizer.

This module exposes Uber's H3 Hexagonal Hierarchical Geospatial Indexing System [1] as
a regionalizer.

Note:
    The default API [2] was chosen (basic_str) to ease the implementation.
    It may be beneficial to try the NumPy API for computationally-heavy work.

References:
    1. https://uber.github.io/h3-py/
    2. https://uber.github.io/h3-py/api_comparison
"""

import tempfile
from pathlib import Path

import duckdb
from rq_geo_toolkit.duckdb import DUCKDB_ABOVE_130

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX
from srai.duckdb import prepare_duckdb_extensions, relation_to_parquet
from srai.geodatatable import VALID_GEO_INPUT, GeoDataTable, prepare_geo_input
from srai.regionalizers import Regionalizer


class H3Regionalizer(Regionalizer):
    """
    H3 Regionalizer.

    H3 Regionalizer allows the given geometries to be divided
    into H3 cells - hexagons with pentagons as a very rare exception
    """

    def __init__(self, resolution: int, buffer: bool = True) -> None:
        """
        Init H3Regionalizer.

        Args:
            resolution (int): Resolution of the cells. See [1] for a full comparison.
            buffer (bool, optional): Whether to fully cover the geometries with
                H3 Cells (visible on the borders). Defaults to True.

        Raises:
            ValueError: If resolution is not between 0 and 15.

        References:
            1. https://h3geo.org/docs/core-library/restable/
        """
        if not (0 <= resolution <= 15):
            raise ValueError(f"Resolution {resolution} is not between 0 and 15.")

        self.resolution = resolution
        self.buffer = buffer

    def transform(self, geo_input: VALID_GEO_INPUT) -> GeoDataTable:
        """
        Regionalize a given GeoDataFrame.

        Transforms given geometries into H3 cells of given resolution
        and optionally applies buffering.

        Args:
            geo_input (VALID_GEO_INPUT): (Multi)Polygons to be regionalized.
                Expected to be in WGS84 CRS.

        Returns:
            GeoDataTable: H3 cells with geometries.
        """
        gdt = prepare_geo_input(geo_input)

        with (
            tempfile.TemporaryDirectory(dir="files") as tmp_dir_name,
            duckdb.connect(
                database=str(Path(tmp_dir_name) / "db.duckdb"),
                config=dict(preserve_insertion_order=True),
            ) as connection,
        ):
            prepare_duckdb_extensions(connection=connection)
            relation = gdt.to_duckdb(connection=connection)

            containment = "CONTAINMENT_OVERLAPPING" if self.buffer else "CONTAINMENT_CENTER"
            parameters_order = (
                f"wkt, {self.resolution}, '{containment}'"
                if DUCKDB_ABOVE_130
                else f"wkt, '{containment}', {self.resolution}"
            )
            h3_coverage_query = f"""
            WITH geometries AS (
                SELECT ST_AsText(geom) as wkt
                FROM (
                    SELECT
                        UNNEST(
                            ST_Dump({GEOMETRY_COLUMN}), recursive := true
                        )
                    FROM ({relation.sql_query()})
                )
            ),
            h3_cells AS (
                SELECT DISTINCT UNNEST(
                    h3_polygon_wkt_to_cells_experimental({parameters_order})
                ) h3_cell
                FROM geometries
            )
            SELECT
                h3_cell as {REGIONS_INDEX},
                ST_GeomFromText(h3_cell_to_boundary_wkt(h3_cell)) AS {GEOMETRY_COLUMN}
            FROM h3_cells
            """

            result_file_name = GeoDataTable.generate_filename()
            result_parquet_path = (
                GeoDataTable.get_directory() / f"{result_file_name}_regions.parquet"
            )

            relation_to_parquet(
                relation=h3_coverage_query,
                result_parquet_path=result_parquet_path,
                connection=connection,
            )

            return GeoDataTable.from_parquet(
                result_parquet_path, index_column_names=REGIONS_INDEX, sort_geometries=True
            )
