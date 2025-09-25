"""
Intersection Joiner.

This module contains intersection joiner implementation.
"""

from typing import Literal, Union, cast, overload

import sedonadb

from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, REGIONS_INDEX
from srai.geodatatable import (
    VALID_GEO_INPUT,
    GeoDataTable,
    ParquetDataTable,
    prepare_geo_input,
)
from srai.joiners import Joiner


class IntersectionJoiner(Joiner):
    """
    Intersection Joiner.

    Intersection Joiner allows to join two GeoDataFrames and find all overlapping geometries. It
    does not apply any grouping or aggregation.
    """

    @overload
    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
    ) -> ParquetDataTable: ...

    @overload
    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
        return_geom: Literal[False],
    ) -> ParquetDataTable: ...

    @overload
    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
        return_geom: Literal[True],
    ) -> GeoDataTable: ...

    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
        return_geom: bool = False,
    ) -> Union[ParquetDataTable, GeoDataTable]:
        """
        Join features to regions based on an 'intersects' predicate.

        Does not apply any grouping to regions.

        Args:
            regions (VALID_GEO_INPUT): regions with which features are joined
            features (VALID_GEO_INPUT): features to be joined
            return_geom (bool): whether to return geometry of the joined features.
                Defaults to False.

        Returns:
            ParquetDataTable or GeoDataTable with an intersection of regions and features,
            which contains a MultiIndex and optionaly a geometry with the intersection.
        """
        regions_gdt = prepare_geo_input(regions, index_name=REGIONS_INDEX)
        features_gdt = prepare_geo_input(features, index_name=FEATURES_INDEX)

        self._validate_indexes(regions_gdt, features_gdt)

        if GEOMETRY_COLUMN not in regions_gdt.columns:
            raise ValueError("Regions must have a geometry column.")
        if GEOMETRY_COLUMN not in features_gdt.columns:
            raise ValueError("Features must have a geometry column.")

        if regions_gdt.empty:
            raise ValueError("Regions must not be empty.")
        if features_gdt.empty:
            raise ValueError("Features must not be empty.")

        return self._join_data(regions=regions_gdt, features=features_gdt, return_geom=return_geom)

    def _join_data(
        self, regions: GeoDataTable, features: GeoDataTable, return_geom: bool
    ) -> Union[ParquetDataTable, GeoDataTable]:
        """
        Join features to regions with returning an intersecting geometry.

        Args:
            regions (GeoDataTable): regions with which features are joined
            features (GeoDataTable): features to be joined
            return_geom (bool): whether to return geometry of the joined features.

        Returns:
            ParquetDataTable or GeoDataTable with an intersection of regions and features,
            which contains a MultiIndex and optionaly a geometry with the intersection.
        """
        base_datatable_class = GeoDataTable if return_geom else ParquetDataTable
        result_file_name = base_datatable_class.generate_filename()
        result_parquet_path = (
            base_datatable_class.get_directory() / f"{result_file_name}_joint.parquet"
        )

        sd = sedonadb.connect()
        regions.to_sedonadb(sd).to_view("regions")
        features.to_sedonadb(sd).to_view("features")

        regions_index_column_names = cast("list[str]", regions.index_column_names)
        features_index_column_names = cast("list[str]", features.index_column_names)

        regions_select_clauses = ",".join(f'regions."{col}"' for col in regions_index_column_names)
        features_select_clauses = ",".join(
            f'features."{col}"' for col in features_index_column_names
        )
        geometry_select_clause = (
            f"""
            , ST_Intersection(
                regions.{GEOMETRY_COLUMN},
                features.{GEOMETRY_COLUMN}
            ) AS {GEOMETRY_COLUMN}
            """
            if return_geom
            else ""
        )

        joined_query = f"""
        SELECT
            {regions_select_clauses},
            {features_select_clauses}
            {geometry_select_clause}
        FROM regions
        JOIN features
        ON ST_Intersects(regions.geometry, features.geometry)
        """

        sd.sql(joined_query).to_parquet(path=result_parquet_path, single_file_output=True)

        return base_datatable_class.from_parquet(
            result_parquet_path,
            index_column_names=regions_index_column_names + features_index_column_names,
        )
