"""
Intersection Joiner.

This module contains intersection joiner implementation.
"""

from typing import TYPE_CHECKING, Union

import geopandas as gpd

from srai.constants import GEOMETRY_COLUMN
from srai.db import count_relation_rows, df_to_duckdb, get_duckdb_connection, relation_to_table
from srai.joiners import Joiner

if TYPE_CHECKING:
    import duckdb


class IntersectionJoiner(Joiner):
    """
    Intersection Joiner.

    Intersection Joiner allows to join two GeoDataFrames and find all overlapping geometries. It
    does not apply any grouping or aggregation.
    """

    def transform(
        self,
        regions: Union["duckdb.DuckDBPyRelation", gpd.GeoDataFrame],
        features: Union["duckdb.DuckDBPyRelation", gpd.GeoDataFrame],
        return_geom: bool = False,
    ) -> "duckdb.DuckDBPyRelation":
        """
        Join features to regions based on an 'intersects' predicate.

        Does not apply any grouping to regions.

        Args:
            regions (Union[duckdb.DuckDBPyRelation, gpd.GeoDataFrame]): regions with which features
                are joined
            features (Union[duckdb.DuckDBPyRelation, gpd.GeoDataFrame]): features to be joined
            return_geom (bool): whether to return geometry of the joined features.
                Defaults to False.

        Returns:
            DuckDB relation with an intersection of regions and features, which contains
            a MultiIndex and optionaly a geometry with the intersection
        """
        if isinstance(regions, gpd.GeoDataFrame):
            regions = df_to_duckdb(regions)

        if isinstance(features, gpd.GeoDataFrame):
            features = df_to_duckdb(features)

        if GEOMETRY_COLUMN not in regions.columns:
            raise ValueError("Regions must have a geometry column.")
        if GEOMETRY_COLUMN not in features.columns:
            raise ValueError("Features must have a geometry column.")

        if count_relation_rows(regions) == 0:
            raise ValueError("Regions must not be empty.")
        if count_relation_rows(features) == 0:
            raise ValueError("Features must not be empty.")

        result_relation: "duckdb.DuckDBPyRelation"

        if return_geom:
            result_relation = self._join_with_geom(regions, features)
        else:
            result_relation = self._join_without_geom(regions, features)

        return relation_to_table(relation=result_relation, prefix="intersection")

    def _join_with_geom(
        self, regions: "duckdb.DuckDBPyRelation", features: "duckdb.DuckDBPyRelation"
    ) -> "duckdb.DuckDBPyRelation":
        """
        Join features to regions with returning an intersecting geometry.

        Args:
            regions (duckdb.DuckDBPyRelation): regions with which features are joined
            features (duckdb.DuckDBPyRelation): features to be joined

        Returns:
            Relation with an intersection of regions and features, which contains
            a MultiIndex and a geometry with the intersection
        """
        intersection_query = """
        SELECT
            regions.region_id,
            features.feature_id,
            ST_Intersection(regions.geometry, features.geometry) geometry
        FROM ({regions_relation}) regions
        JOIN ({features_relation}) features
        ON ST_Intersects(regions.geometry, features.geometry)
        """
        filled_query = intersection_query.format(
            regions_relation=regions.sql_query(), features_relation=features.sql_query()
        )
        joint = get_duckdb_connection().sql(filled_query)
        return joint

    def _join_without_geom(
        self, regions: "duckdb.DuckDBPyRelation", features: "duckdb.DuckDBPyRelation"
    ) -> "duckdb.DuckDBPyRelation":
        """
        Join features to regions without intersection caclulation.

        Args:
            regions (duckdb.DuckDBPyRelation): regions with which features are joined
            features (duckdb.DuckDBPyRelation): features to be joined

        Returns:
            Relation with an intersection of regions and features, which contains
            a MultiIndex
        """
        intersection_query = """
        SELECT
            regions.region_id,
            features.feature_id
        FROM ({regions_relation}) regions
        JOIN ({features_relation}) features
        ON ST_Intersects(regions.geometry, features.geometry)
        """
        filled_query = intersection_query.format(
            regions_relation=regions.sql_query(), features_relation=features.sql_query()
        )
        joint = get_duckdb_connection().sql(filled_query)
        return joint
