"""
Contextual Count Embedder.

This module contains contextual count embedder implementation from ARIC@SIGSPATIAL 2021 paper [1].

References:
    1. https://doi.org/10.1145/3486626.3493434
    1. https://arxiv.org/abs/2111.00990
"""

import multiprocessing
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import duckdb
import pandas as pd
import polars as pl
from rq_geo_toolkit.multiprocessing_utils import run_process_with_memory_monitoring
from tqdm import tqdm

from srai.constants import FORCE_TERMINAL
from srai.duckdb import prepare_duckdb_extensions, relation_from_parquet_paths
from srai.embedders.count_embedder import CountEmbedder
from srai.geodatatable import VALID_DATA_INPUT, ParquetDataTable
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter
from srai.neighbourhoods import Neighbourhood
from srai.neighbourhoods._base import IndexType

ctx: multiprocessing.context.SpawnContext = multiprocessing.get_context("spawn")


class WorkerSpawnProcess(  # noqa: D101
    ctx.Process  # type: ignore[name-defined,misc]
):
    def __init__(self, *args: Any, **kwargs: Any):  # noqa: D107
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception: Optional[tuple[Exception, str]] = None

    def run(self) -> None:  # noqa: D102, pragma: no cover
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb: str = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self) -> Optional[tuple[Exception, str]]:  # noqa: D102
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class ContextualCountEmbedder(CountEmbedder):
    """ContextualCountEmbedder."""

    DUCKDB_AGGREGATION_FUNCTION_MAPPING = {
        "average": "AVG",
        "median": "MEDIAN",
        "sum": "SUM",
        "min": "MIN",
        "max": "MAX",
    }

    def __init__(
        self,
        neighbourhood: Neighbourhood[IndexType],
        neighbourhood_distance: int,
        concatenate_vectors: bool = False,
        expected_output_features: Optional[
            Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter]
        ] = None,
        count_subcategories: bool = False,
        aggregation_function: Literal["average", "median", "sum", "min", "max"] = "average",
    ) -> None:
        """
        Init ContextualCountEmbedder.

        Args:
            neighbourhood (Neighbourhood[T]): Neighbourhood object used to get neighbours for
                the contextualization.
            neighbourhood_distance (int): How many neighbours levels should be included in
                the embedding.
            concatenate_vectors (bool, optional): Whether to sum all neighbours into a single vector
                with the same width as `CountEmbedder`, or to concatenate them to the wide format
                and keep all neighbour levels separate. Defaults to False.
            expected_output_features
                (Union[List[str], OsmTagsFilter, GroupedOsmTagsFilter], optional):
                The features that are expected to be found in the resulting embedding.
                If not None, the missing features are added and filled with 0.
                The unexpected features are removed. The resulting columns are sorted accordingly.
                Defaults to None.
            count_subcategories (bool, optional): Whether to count all subcategories individually
                or count features only on the highest level based on features column name.
                Defaults to False.
            aggregation_function (Literal["average", "median", "sum", "min", "max"], optional):
                Function used to aggregate data from the neighbours. Defaults to average.

        Raises:
            ValueError: If `neighbourhood_distance` is negative.
        """
        super().__init__(expected_output_features, count_subcategories)

        self.neighbourhood = neighbourhood
        self.neighbourhood_distance = neighbourhood_distance
        self.concatenate_vectors = concatenate_vectors
        self.aggregation_function = aggregation_function

        if self.neighbourhood_distance < 0:
            raise ValueError("Neighbourhood distance must be positive.")

    def transform(
        self,
        regions: VALID_DATA_INPUT,
        features: VALID_DATA_INPUT,
        joint: VALID_DATA_INPUT,
    ) -> ParquetDataTable:
        """
        Embed a given GeoDataTable.

        Creates region embeddings by counting the frequencies of each feature value and applying
        a contextualization based on neighbours of regions. For each region, features will be
        altered based on the neighbours either by adding aggregated values dimished based on
        distance, or by adding new separate columns with neighbour distance postfix.
        Expects features_gdf to be in wide format with each column being a separate type of
        feature (e.g. amenity, leisure) and rows to hold values of these features for each object.
        The rows will hold numbers of this type of feature in each region. Numbers can be
        fractional because neighbourhoods are aggregated to represent a single value from
        all neighbours on a given level.

        Args:
            regions (VALID_DATA_INPUT): Region indexes.
            features (VALID_DATA_INPUT): Feature indexes and feature values.
            joint (VALID_DATA_INPUT): Joiner result with region-feature multi-index.

        Returns:
            ParquetDataTable: Embedding for each region in regions.

        Raises:
            ValueError: If features is empty and self.expected_output_features is not set.
            ValueError: If any of the input index names is None.
            ValueError: If joint.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in inputs don't overlap correctly.
        """
        counts_pdt = super().transform(regions, features, joint)

        if counts_pdt.empty:
            if self.concatenate_vectors:
                empty_df = pd.DataFrame(
                    [],
                    columns=[
                        counts_pdt.index_name,
                        *(
                            f"{column}_{distance}"
                            for distance, column in product(
                                range(self.neighbourhood_distance + 1), counts_pdt.columns
                            )
                        ),
                    ],
                )
                prefix_path = ParquetDataTable.generate_filename()
                save_parquet_path = (
                    ParquetDataTable.get_directory()
                    / f"{prefix_path}_empty_concatenated_embeddings.parquet"
                )
                empty_df.to_parquet(save_parquet_path)
                return ParquetDataTable.from_parquet(
                    parquet_path=save_parquet_path,
                    index_column_names=counts_pdt.index_column_names,
                )
            else:
                return counts_pdt

        return self._iterate_parquet_data(counts_pdt=counts_pdt)

    def _iterate_parquet_data(self, counts_pdt: ParquetDataTable) -> ParquetDataTable:
        total_rows = counts_pdt.rows
        current_file_idx = 0
        current_offset = 0
        current_limit = 10_000_000

        result_dir_name = ParquetDataTable.generate_filename()
        embedding_type = "concatenated" if self.concatenate_vectors else "squashed"
        result_dir_path = (
            ParquetDataTable.get_directory() / f"{result_dir_name}_{embedding_type}_embeddings"
        )
        result_dir_path.mkdir(parents=True, exist_ok=True)
        saved_result_files = []

        with tqdm(
            total=total_rows,
            desc="Generating embeddings for neighbours",
            disable=FORCE_TERMINAL,
        ) as pbar:
            while current_offset < total_rows:
                try:
                    current_result_file_path = result_dir_path / f"{current_file_idx}.parquet"
                    process = WorkerSpawnProcess(
                        target=_parse_single_batch,
                        kwargs=dict(
                            neighbourhood=self.neighbourhood,
                            neighbourhood_distance=self.neighbourhood_distance,
                            aggregation_function=self.aggregation_function,
                            concatenate_vectors=self.concatenate_vectors,
                            counts_parquet_files=list(counts_pdt.parquet_paths),
                            index_name=cast("str", counts_pdt.index_name),
                            feature_column_names=counts_pdt.columns,
                            limit=current_limit,
                            offset=current_offset,
                            result_file_path=current_result_file_path,
                        ),
                    )
                    run_process_with_memory_monitoring(process)

                    saved_result_files.append(current_result_file_path)

                    current_file_idx += 1
                    current_offset += current_limit
                    pbar.n = min(current_offset, total_rows)
                    pbar.refresh()

                except (duckdb.OutOfMemoryException, MemoryError) as ex:
                    current_limit //= 10
                    if current_limit == 1:
                        raise

                    warnings.warn(
                        f"Encountered {ex.__class__.__name__} during operation."
                        f" Retrying with lower number of rows per batch ({current_limit} rows).",
                        stacklevel=1,
                    )

        return ParquetDataTable.from_parquet(
            parquet_path=saved_result_files, index_column_names=counts_pdt.index_name
        )


def _parse_single_batch(
    neighbourhood: Neighbourhood[IndexType],
    neighbourhood_distance: int,
    aggregation_function: Literal["average", "median", "sum", "min", "max"],
    concatenate_vectors: bool,
    counts_parquet_files: list[Path],
    index_name: str,
    feature_column_names: list[str],
    limit: int,
    offset: int,
    result_file_path: Path,
) -> None:
    num_of_multiprocessing_workers = min(1, cpu_count() - 2)
    with (
        tempfile.TemporaryDirectory(dir="files") as tmp_dir_name,
        duckdb.connect(
            database=str(Path(tmp_dir_name) / "db.duckdb"),
            config=dict(preserve_insertion_order=True),
        ) as connection,
        ProcessPoolExecutor(max_workers=num_of_multiprocessing_workers) as executor,
    ):
        tmp_dir_path = Path(tmp_dir_name)
        prepare_duckdb_extensions(conn=connection)
        full_relation = relation_from_parquet_paths(
            parquet_paths=counts_parquet_files, connection=connection, with_row_number=True
        )
        current_batch_relation = full_relation.limit(n=limit, offset=offset)
        current_regions = (
            current_batch_relation.sort("row_number").select(index_name, "row_number").pl()
        )
        current_region_ids = current_regions[index_name]

        all_neighbours = set((region_id, region_id, 0) for region_id in current_region_ids)

        fn_neighbours = partial(
            _get_existing_neighbours_at_distance,
            neighbourhood=neighbourhood,
            neighbourhood_distance=neighbourhood_distance,
        )
        for result in executor.map(
            fn_neighbours,
            current_region_ids,
            chunksize=ceil(len(current_region_ids) / (4 * num_of_multiprocessing_workers)),
        ):
            all_neighbours.update(result)

        precalculated_neighbours_path = tmp_dir_path / "neighbours.parquet"
        pd.DataFrame(
            all_neighbours,
            columns=["region_id", "neighbour_id", "distance"],
        ).to_parquet(precalculated_neighbours_path)

        current_regions_lf = current_regions.lazy()
        neighbours_lf = pl.scan_parquet(precalculated_neighbours_path)
        embeddings_lf = pl.scan_parquet(counts_parquet_files)

        neighbours_joined_with_embeddings_lf = neighbours_lf.join(
            embeddings_lf, left_on="neighbour_id", right_on=index_name, how="left"
        )

        if concatenate_vectors:
            embeddings = _generate_concatenated_embeddings_lazyframe(
                neighbours_joined_with_embeddings_lf=neighbours_joined_with_embeddings_lf,
                feature_column_names=feature_column_names,
                neighbourhood_distance=neighbourhood_distance,
                aggregation_function=aggregation_function,
            )
        else:
            embeddings = _generate_squashed_embeddings_lazyframe(
                neighbours_joined_with_embeddings_lf=neighbours_joined_with_embeddings_lf,
                feature_column_names=feature_column_names,
                aggregation_function=aggregation_function,
            )

        current_regions_lf.join(
            embeddings, left_on=index_name, right_on="region_id", how="left"
        ).fill_null(0).sort("row_number").drop("row_number").sink_parquet(result_file_path)


def _generate_squashed_embeddings_lazyframe(
    neighbours_joined_with_embeddings_lf: pl.LazyFrame,
    feature_column_names: list[str],
    aggregation_function: Literal["average", "median", "sum", "min", "max"],
) -> pl.LazyFrame:
    return (
        neighbours_joined_with_embeddings_lf.group_by(["region_id", "distance"])
        .agg(
            (_apply_polars_aggregation(pl.col(column), aggregation_function).alias(column))
            for column in feature_column_names
        )
        .with_columns((pl.col("distance") + 1).pow(2).alias("divisor"))
        .with_columns(pl.col(column) / pl.col("divisor") for column in feature_column_names)
        .group_by("region_id")
        .agg((pl.col(column).sum().alias(column)) for column in feature_column_names)
    )


def _generate_concatenated_embeddings_lazyframe(
    neighbours_joined_with_embeddings_lf: pl.LazyFrame,
    feature_column_names: list[str],
    neighbourhood_distance: int,
    aggregation_function: Literal["average", "median", "sum", "min", "max"],
) -> pl.LazyFrame:
    return neighbours_joined_with_embeddings_lf.group_by("region_id").agg(
        (
            _apply_polars_aggregation(
                pl.col(column).filter(pl.col("distance") == distance), aggregation_function
            ).alias(f"{column}_{distance}")
        )
        for distance, column in product(range(neighbourhood_distance + 1), feature_column_names)
    )


def _apply_polars_aggregation(
    expr: pl.Expr, aggregation_function: Literal["average", "median", "sum", "min", "max"]
) -> pl.Expr:
    if aggregation_function == "average":
        return expr.mean()
    elif aggregation_function == "median":
        return expr.median()
    elif aggregation_function == "sum":
        return expr.sum()
    elif aggregation_function == "min":
        return expr.min()
    elif aggregation_function == "max":
        return expr.max()
    else:
        raise ValueError(f"Unknown aggregation function: {aggregation_function}")


def _get_existing_neighbours_at_distance(
    region_id: IndexType,
    neighbourhood: Neighbourhood[IndexType],
    neighbourhood_distance: int,
) -> set[tuple[IndexType, IndexType, int]]:
    return set(
        (region_id, neighbour_region_id, distance)
        for distance in range(1, neighbourhood_distance + 1)
        for neighbour_region_id in neighbourhood.get_neighbours_at_distance(
            region_id, distance, include_center=False
        )
    )
