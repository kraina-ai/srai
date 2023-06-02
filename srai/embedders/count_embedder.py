"""
Count Embedder.

This module contains count embedder implementation.
"""
from typing import TYPE_CHECKING, List, Optional, Set, Union

import geopandas as gpd
import pandas as pd

from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, REGIONS_INDEX
from srai.db import df_to_duckdb, escape, get_duckdb_connection, relation_to_table
from srai.embedders import Embedder

if TYPE_CHECKING:
    import duckdb


class CountEmbedder(Embedder):
    """Simple Embedder that counts occurences of feature values."""

    def __init__(
        self, expected_output_features: Optional[List[str]] = None, count_subcategories: bool = True
    ) -> None:
        """
        Init CountEmbedder.

        Args:
            expected_output_features (List[str], optional): The features that are expected
                to be found in the resulting embedding. If not None, the missing features are added
                and filled with 0. The unexpected features are removed.
                The resulting columns are sorted accordingly. Defaults to None.
            count_subcategories (bool, optional): Whether to count all subcategories individually
                or count features only on the highest level based on features column name.
                Defaults to True.
        """
        self.expected_output_features: Optional[Set[str]]
        if expected_output_features is not None:
            self.expected_output_features = set(expected_output_features)
        else:
            self.expected_output_features = None

        self.count_subcategories = count_subcategories

    def transform(
        self,
        regions: Union["duckdb.DuckDBPyRelation", gpd.GeoDataFrame],
        features: Union["duckdb.DuckDBPyRelation", gpd.GeoDataFrame],
        joint: Union["duckdb.DuckDBPyRelation", gpd.GeoDataFrame],
    ) -> "duckdb.DuckDBPyRelation":
        """
        Embed a given GeoDataFrame.

        Creates region embeddings by counting the frequencies of each feature value.
        Expects features to be in wide format with each column
        being a separate type of feature (e.g. amenity, leisure)
        and rows to hold values of these features for each object.
        The resulting DataFrame will have columns made by combining
        the feature name (column) and value (row) e.g. amenity_fuel or type_0.
        The rows will hold numbers of this type of feature in each region.

        Args:
            regions (Union[duckdb.DuckDBPyRelation, gpd.GeoDataFrame]): Region indexes and
                geometries.
            features (Union[duckdb.DuckDBPyRelation, gpd.GeoDataFrame]): Feature indexes,
                geometries and feature values.
            joint (Union[duckdb.DuckDBPyRelation, gpd.GeoDataFrame]): Joiner result with
                region-feature multi-index.

        Returns:
            pd.DataFrame: Embedding for each region in regions.

        Raises:
            ValueError: If features is empty and self.expected_output_features is not set.
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        if isinstance(regions, gpd.GeoDataFrame):
            regions = df_to_duckdb(regions)
        if isinstance(features, gpd.GeoDataFrame):
            features = df_to_duckdb(features)
        if isinstance(joint, (pd.DataFrame, gpd.GeoDataFrame)):
            joint = df_to_duckdb(joint)

        self._validate_relations_indexes(regions, features, joint)

        result_relation: "duckdb.DuckDBPyRelation"

        if self.count_subcategories:
            result_relation = self._count_subcategories(regions, features, joint)
        else:
            result_relation = self._count_base_categories(regions, features, joint)

        if self.expected_output_features is not None:
            result_relation = self._filter_to_expected_features(result_relation)

        return relation_to_table(relation=result_relation, prefix="counts_embedding")

    def _count_base_categories(
        self,
        regions_relation: "duckdb.DuckDBPyRelation",
        features_relation: "duckdb.DuckDBPyRelation",
        joint_relation: "duckdb.DuckDBPyRelation",
    ) -> "duckdb.DuckDBPyRelation":
        columns = set(features_relation.columns).difference([FEATURES_INDEX, GEOMETRY_COLUMN])
        sql_query = """
        SELECT
            regions.region_id,
            {counts_clauses}
        FROM ({joint_relation}) joint
        JOIN ({features_relation}) features
        ON features.feature_id = joint.feature_id
        JOIN ({regions_relation}) regions
        ON regions.region_id = joint.region_id
        GROUP BY regions.region_id
        """

        filled_query = sql_query.format(
            joint_relation=joint_relation.sql_query(),
            features_relation=features_relation.sql_query(),
            regions_relation=regions_relation.sql_query(),
            counts_clauses=", ".join(
                f'COUNT(features."{column}") FILTER (features."{column}" IS NOT NULL) AS "{column}"'
                for column in sorted(columns)
                if self.expected_output_features is None or column in self.expected_output_features
            ),
        )
        return get_duckdb_connection().sql(filled_query)

    def _count_subcategories(
        self,
        regions_relation: "duckdb.DuckDBPyRelation",
        features_relation: "duckdb.DuckDBPyRelation",
        joint_relation: "duckdb.DuckDBPyRelation",
    ) -> "duckdb.DuckDBPyRelation":
        columns = sorted(
            set(features_relation.columns).difference([FEATURES_INDEX, GEOMETRY_COLUMN])
        )
        group_clauses = ", ".join(
            f'LIST(DISTINCT "{column}") FILTER (features."{column}" IS NOT NULL) AS "{column}"'
            for column in columns
        )
        values_query = f"SELECT {group_clauses} FROM ({features_relation.sql_query()}) features"
        columns_values = get_duckdb_connection().sql(values_query).fetchone()

        sql_query = """
        SELECT
            regions.region_id,
            {counts_clauses}
        FROM ({joint_relation}) joint
        JOIN ({features_relation}) features
        ON features.feature_id = joint.feature_id
        JOIN ({regions_relation}) regions
        ON regions.region_id = joint.region_id
        GROUP BY regions.region_id
        """

        filled_query = sql_query.format(
            joint_relation=joint_relation.sql_query(),
            features_relation=features_relation.sql_query(),
            regions_relation=regions_relation.sql_query(),
            counts_clauses=", ".join(
                f'COUNT(features."{column}") FILTER (features."{column}" = \'{escape(value)}\')'
                f' AS "{column}_{value}"'
                for column, column_values in zip(columns, columns_values)
                for value in column_values
                if self.expected_output_features is None
                or f"{column}_{value}" in self.expected_output_features
            ),
        )
        return get_duckdb_connection().sql(filled_query)

    def _filter_to_expected_features(
        self, region_embeddings: "duckdb.DuckDBPyRelation"
    ) -> "duckdb.DuckDBPyRelation":
        """
        Add missing and remove excessive columns from embeddings.

        Args:
            region_embeddings (duckdb.DuckDBPyRelation): Counted frequencies of each feature value.

        Returns:
            duckdb.DuckDBPyRelation: Embeddings with expected columns only.
        """
        if not self.expected_output_features:
            return region_embeddings
        existing_columns = set(region_embeddings.columns).difference([REGIONS_INDEX])
        missing_features = self.expected_output_features.difference(existing_columns)
        columns_list = ", ".join(
            f'0 AS "{column}"' if column in missing_features else f'"{column}"'
            for column in sorted(self.expected_output_features)
        )
        sql_query = (
            f"SELECT region_id, {columns_list} FROM ({region_embeddings.sql_query()}) embeddings"
        )
        return get_duckdb_connection().sql(sql_query)
