"""
Count Embedder.

This module contains count embedder implementation.
"""

import hashlib
from typing import Optional, Union, cast

import duckdb
import pandas as pd
import polars as pl

from srai._typing import is_expected_type
from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, REGIONS_INDEX
from srai.embedders import Embedder
from srai.geodatatable import (
    VALID_DATA_INPUT,
    ParquetDataTable,
    prepare_data_input,
)
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter


class CountEmbedder(Embedder):
    """Simple Embedder that counts occurences of feature values."""

    def __init__(
        self,
        expected_output_features: Optional[
            Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter]
        ] = None,
        count_subcategories: bool = True,
    ) -> None:
        """
        Init CountEmbedder.

        Args:
            expected_output_features
                (Union[List[str], OsmTagsFilter, GroupedOsmTagsFilter], optional):
                The features that are expected to be found in the resulting embedding.
                If not None, the missing features are added and filled with 0.
                The unexpected features are removed. The resulting columns are sorted accordingly.
                Defaults to None.
            count_subcategories (bool, optional): Whether to count all subcategories individually
                or count features only on the highest level based on features column name.
                Defaults to True.
        """
        self.count_subcategories = count_subcategories
        self._parse_expected_output_features(expected_output_features)

    def transform(
        self,
        regions: VALID_DATA_INPUT,
        features: VALID_DATA_INPUT,
        joint: VALID_DATA_INPUT,
    ) -> ParquetDataTable:
        """
        Embed a given Geo data.

        Creates region embeddings by counting the frequencies of each feature value.
        Expects features_gdf to be in wide format with each column
        being a separate type of feature (e.g. amenity, leisure)
        and rows to hold values of these features for each object.
        The resulting output will have columns made by combining
        the feature name (column) and value (row) e.g. amenity_fuel or type_0.
        The rows will hold numbers of this type of feature in each region.

        Args:
            regions (VALID_DATA_INPUT): Region indexes.
            features (VALID_DATA_INPUT): Feature indexes and feature values.
            joint (VALID_DATA_INPUT): Joiner result with region-feature multi-index.

        Returns:
            ParquetDataTable: Embedding for each region in regions_gdf.

        Raises:
            ValueError: If features is empty and self.expected_output_features is not set.
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint.index doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
            ValueError: If features contains boolean columns and count_subcategories is True.
        """
        regions_pdt = prepare_data_input(regions).drop_columns([GEOMETRY_COLUMN], missing_ok=True)
        features_pdt = prepare_data_input(features).drop_columns([GEOMETRY_COLUMN], missing_ok=True)
        joint_pdt = prepare_data_input(joint).drop_columns([GEOMETRY_COLUMN], missing_ok=True)

        self._validate_indexes(regions_pdt, features_pdt, joint_pdt)

        if features_pdt.empty:
            if self.expected_output_features is not None:
                paths = list(map(lambda x: f"'{x}'", regions_pdt.parquet_paths))
                select_clauses = [f'0 AS "{column}"' for column in self.expected_output_features]

                prefix_path = ParquetDataTable.generate_filename()
                relation = duckdb.sql(
                    f"""
                    SELECT
                        regions.{regions_pdt.index_name},
                        {",".join(select_clauses)}
                    FROM read_parquet([{",".join(paths)}]) regions
                    """
                )

                h = hashlib.new("sha256")
                h.update(relation.sql_query().encode())
                relation_hash = h.hexdigest()

                save_parquet_path = (
                    ParquetDataTable.get_directory() / f"{prefix_path}_{relation_hash}.parquet"
                )
                relation.to_parquet(str(save_parquet_path))
                return ParquetDataTable.from_parquet(
                    parquet_path=save_parquet_path,
                    index_column_names=regions_pdt.index_column_names,
                )
            else:
                raise ValueError(
                    "Cannot embed with empty features_gdf and no expected_output_features."
                )

        regions_df = pl.scan_parquet(regions_pdt.parquet_paths, hive_partitioning=False)
        features_df = pl.scan_parquet(features_pdt.parquet_paths, hive_partitioning=False)
        joint_df = pl.scan_parquet(joint_pdt.parquet_paths, hive_partitioning=False)

        regions_schema = regions_df.collect_schema()
        features_schema = features_df.collect_schema()
        joint_schema = joint_df.collect_schema()

        original_regions_df_index_type = regions_schema.get(REGIONS_INDEX)

        if (
            regions_schema.get(REGIONS_INDEX).to_python()
            != joint_schema.get(REGIONS_INDEX).to_python()
        ):
            regions_df = regions_df.cast({REGIONS_INDEX: pl.String})
            joint_df = joint_df.cast({REGIONS_INDEX: pl.String})

        if (
            features_schema.get(FEATURES_INDEX).to_python()
            != joint_schema.get(FEATURES_INDEX).to_python()
        ):
            features_df = features_df.cast({FEATURES_INDEX: pl.String})
            joint_df = joint_df.cast({FEATURES_INDEX: pl.String})

        feature_columns = [col for col in features_schema.names() if col != FEATURES_INDEX]
        dtypes = features_schema.dtypes()
        are_all_columns_bool = all(
            dtypes[idx] == pl.Boolean
            for idx, col in enumerate(features_schema.names())
            if col != FEATURES_INDEX
        )

        if self.count_subcategories:
            if are_all_columns_bool:
                raise ValueError("Cannot count subcategories with boolean columns.")

            feature_encodings = (
                features_df.collect(streaming=True).to_dummies(columns=feature_columns).lazy()
            )
            feature_columns = [
                col
                for col in feature_encodings.collect_schema().names()
                if col != FEATURES_INDEX and not col.endswith("_null")
            ]
            feature_encodings = feature_encodings.select([FEATURES_INDEX, *feature_columns])
        elif are_all_columns_bool:
            feature_encodings = features_df.with_columns(
                [
                    pl.col(FEATURES_INDEX),
                    *(pl.col(col).cast(pl.Int32) for col in feature_columns),
                ]
            )
        else:
            feature_encodings = features_df.with_columns(
                [
                    pl.col(FEATURES_INDEX),
                    *(pl.col(col).is_not_null().cast(pl.Int32) for col in feature_columns),
                ]
            )

        joint_with_encodings = joint_df.join(feature_encodings, on=FEATURES_INDEX, how="left")
        region_embeddings = joint_with_encodings.drop(FEATURES_INDEX).group_by(REGIONS_INDEX).sum()
        region_embeddings, feature_columns = self._maybe_filter_to_expected_features(
            region_embeddings, feature_columns
        )

        result_file_name = ParquetDataTable.generate_filename()
        result_parquet_path = (
            ParquetDataTable.get_directory() / f"{result_file_name}_embeddings.parquet"
        )

        (
            regions_df.join(region_embeddings, on=REGIONS_INDEX, how="left")
            .fill_null(0)
            .with_columns(
                [
                    pl.col(REGIONS_INDEX).cast(original_regions_df_index_type),
                    *(pl.col(col).cast(pl.Int32) for col in feature_columns),
                ]
            )
        ).sink_parquet(path=result_parquet_path)

        return ParquetDataTable.from_parquet(result_parquet_path, index_column_names=REGIONS_INDEX)

    def _parse_expected_output_features(
        self,
        expected_output_features: Optional[Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter]],
    ) -> None:
        expected_output_features_list = []

        if is_expected_type(expected_output_features, OsmTagsFilter):
            expected_output_features_list = self._parse_osm_tags_filter_to_expected_features(
                cast("OsmTagsFilter", expected_output_features)
            )
        elif is_expected_type(expected_output_features, GroupedOsmTagsFilter):
            expected_output_features_list = (
                self._parse_grouped_osm_tags_filter_to_expected_features(
                    cast("GroupedOsmTagsFilter", expected_output_features)
                )
            )
        elif isinstance(expected_output_features, list):
            expected_output_features_list = expected_output_features
        elif expected_output_features is not None:
            raise ValueError(
                f"Wrong type of expected_output_features ({type(expected_output_features)})"
            )

        self.expected_output_features = (
            pd.Series(expected_output_features_list) if expected_output_features_list else None
        )

    def _parse_osm_tags_filter_to_expected_features(
        self, osm_filter: OsmTagsFilter, delimiter: str = "_"
    ) -> list[str]:
        expected_output_features: set[str] = set()

        if not self.count_subcategories:
            expected_output_features.update(osm_filter.keys())
        else:
            for osm_tag_key, osm_tag_value in osm_filter.items():
                if isinstance(osm_tag_value, bool) and osm_tag_value:
                    raise ValueError(
                        "Cannot parse bool OSM tag value to expected features list. "
                        "Please use filter without boolean value."
                    )
                elif isinstance(osm_tag_value, str):
                    expected_output_features.add(f"{osm_tag_key}{delimiter}{osm_tag_value}")
                elif isinstance(osm_tag_value, list):
                    expected_output_features.update(
                        f"{osm_tag_key}{delimiter}{tag_value}" for tag_value in osm_tag_value
                    )

        return sorted(list(expected_output_features))

    def _parse_grouped_osm_tags_filter_to_expected_features(
        self, grouped_osm_filter: GroupedOsmTagsFilter
    ) -> list[str]:
        expected_output_features: set[str] = set()

        if not self.count_subcategories:
            expected_output_features.update(grouped_osm_filter.keys())
        else:
            for group_name, osm_filter in grouped_osm_filter.items():
                parsed_osm_tags_filter_features = self._parse_osm_tags_filter_to_expected_features(
                    osm_filter, delimiter="="
                )
                expected_output_features.update(
                    f"{group_name}_{parsed_osm_tags_filter_feature}"
                    for parsed_osm_tags_filter_feature in parsed_osm_tags_filter_features
                )

        return sorted(list(expected_output_features))

    def _maybe_filter_to_expected_features(
        self, region_embeddings: pl.LazyFrame, current_embedding_columns: list[str]
    ) -> tuple[pl.LazyFrame, list[str]]:
        """
        Add missing and remove excessive columns from embeddings.

        Args:
            region_embeddings (pl.LazyFrame): Counted frequencies of each feature value.
            current_embedding_columns (list[str]): List of current embedding columns.

        Returns:
            tuple[pl.LazyFrame, list[str]]: Embeddings with expected columns only and new list
            of columns.
        """
        if self.expected_output_features is None:
            return region_embeddings, current_embedding_columns

        missing_features = self.expected_output_features[
            ~self.expected_output_features.isin(current_embedding_columns)
        ]

        region_embeddings = region_embeddings.with_columns(
            [pl.lit(0, pl.Int32).alias(col) for col in missing_features]
        ).select([REGIONS_INDEX, *self.expected_output_features])
        return region_embeddings, list(self.expected_output_features)
