"""Module for spatially splitting input data."""

from typing import Literal, Optional, Union, cast, overload

import geopandas as gpd
import h3
import pandas as pd
from tqdm import tqdm

from srai.constants import FORCE_TERMINAL

BUCKET_COLUMN_NAME = "bucket"
COUNT_COLUMN_NAME = "count"
H3_COLUMN_NAME = "h3"
POINTS_COLUMN_NAME = "points"

# NOTE: Add option to pass regions_gdf instead of using H3 cells?
# And group based on intersected points.

# NOTE: Add option to create k-folds

__all__ = [
    "train_test_spatial_split",
    "spatial_split_points",
]


def train_test_spatial_split(
    input_gdf: gpd.GeoDataFrame,
    parent_h3_resolution: int,
    geometry_column: str = "geometry",
    target_column: Optional[str] = None,
    n_bins: int = 7,
    categorical: bool = False,
    test_size: Union[float, int] = 0.2,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Split data based on parent H3 cell and stratify the data using specified target.

    Args:
        input_gdf (gpd.GeoDataFrame): GeoDataFrame with point geometries to be splitted.
        parent_h3_resolution (int): H3 resolution used to split the data.
        geometry_column (str, optional): Name of the geometry column.
        target_column: Target column name used to stratify the data distribution.
            If None, split is generated based on number of points within a hex of a given
            resolution. Defaults to None.
        n_bins (int, optional): Bucket number used to stratify target data.
            Only used when categorical is False. Defaults to 7.
        categorical (bool, optional): If True, target column is treated as categorical.
            Target column must be specified in this case. Defaults to False.
        test_size (Union[float, int], optional): Size of the test dataset.
            Can be a fraction (0-1 range) or a total number of rows. Defaults to 0.2.
        random_state (Optional[int], optional): Random state for reproducibility. Defaults to None.
        verbose (bool, optional): Print the progress and report output. Defaults to True.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Train and test GeoDataFrames.
    """
    if test_size <= 0:
        raise ValueError("test_size must be greater than 0.")

    splits = spatial_split_points(
        input_gdf=input_gdf,
        parent_h3_resolution=parent_h3_resolution,
        geometry_column=geometry_column,
        target_column=target_column,
        n_bins=n_bins,
        categorical=categorical,
        test_size=test_size,
        validation_size=0,
        random_state=random_state,
        verbose=verbose,
    )

    return splits["train"], splits["test"]


@overload
def spatial_split_points(
    input_gdf: gpd.GeoDataFrame,
    parent_h3_resolution: int,
    geometry_column: str = "geometry",
    target_column: Optional[str] = None,
    n_bins: int = 7,
    categorical: bool = False,
    test_size: Union[float, int] = 0.2,
    validation_size: Union[float, int] = 0,
    random_state: Optional[int] = None,
    return_split_stats: Literal[False] = False,
    verbose: bool = True,
) -> dict[str, Optional[gpd.GeoDataFrame]]: ...


@overload
def spatial_split_points(
    input_gdf: gpd.GeoDataFrame,
    parent_h3_resolution: int,
    geometry_column: str = "geometry",
    target_column: Optional[str] = None,
    n_bins: int = 7,
    categorical: bool = False,
    test_size: Union[float, int] = 0.2,
    validation_size: Union[float, int] = 0,
    random_state: Optional[int] = None,
    return_split_stats: Literal[True] = True,
    verbose: bool = True,
) -> tuple[dict[str, Optional[gpd.GeoDataFrame]], pd.DataFrame]: ...


def spatial_split_points(
    input_gdf: gpd.GeoDataFrame,
    parent_h3_resolution: int,
    geometry_column: str = "geometry",
    target_column: Optional[str] = None,
    n_bins: int = 7,
    categorical: bool = False,
    test_size: Union[float, int] = 0.2,
    validation_size: Union[float, int] = 0,
    random_state: Optional[int] = None,
    return_split_stats: bool = False,
    verbose: bool = True,
) -> Union[
    dict[str, Optional[gpd.GeoDataFrame]],
    tuple[dict[str, Optional[gpd.GeoDataFrame]], pd.DataFrame],
]:
    """
    Split data based on parent H3 cell and stratify the data using specified target.

    Args:
        input_gdf (gpd.GeoDataFrame): GeoDataFrame with point geometries to be splitted.
        parent_h3_resolution (int): H3 resolution used to split the data.
        geometry_column (str, optional): Name of the geometry column.
        target_column: Target column name used to stratify the data distribution.
            If None, split is generated based on number of points within a hex of a given
            resolution. Defaults to None.
        n_bins (int, optional): Bucket number used to stratify target data.
            Only used when categorical is False. Defaults to 7.
        categorical (bool, optional): If True, target column is treated as categorical.
            Target column must be specified in this case. Defaults to False.
        test_size (Union[float, int], optional): Size of the test dataset.
            Can be a fraction (0-1 range) or a total number of rows. Defaults to 0.2.
        validation_size (Union[float, int], optional): Size of the validation dataset.
            Can be a fraction (0-1 range) or a total number of rows. Defaults to 0.
        random_state (Optional[int], optional): Random state for reproducibility. Defaults to None.
        return_split_stats (bool, optional): Return split statistics. Defaults to False.
        verbose (bool, optional): Print the progress and report output. Defaults to True.

    Returns:
        dict[str, Optional[str]]: _description_
    """
    geom_types = input_gdf[geometry_column].geom_type.unique()
    if len(geom_types) > 1 or geom_types[0] != "Point":
        raise ValueError(
            "Only point geometries can be parsed."
            " Use centroids if you want to split other types of geometries."
        )

    if test_size < 0:
        raise ValueError("test_size cannot be negative.")

    if validation_size < 0:
        raise ValueError("validation_size cannot be negative.")

    total_numer_rows = len(input_gdf)

    if validation_size < 1:
        validation_fraction = validation_size
    else:
        validation_fraction = validation_size / total_numer_rows

    if test_size < 1:
        test_fraction = test_size
    else:
        test_fraction = test_size / total_numer_rows

    if (validation_fraction + test_fraction) >= 1:
        raise ValueError("Test and validation fractions sum up to 1 or more.")

    # Calculate missing train fraction (3 groups adds up to 1)
    train_fraction = 1 - validation_fraction - test_fraction

    # Calculate statistics per H3 parent cell and bucket
    # (number of point per bucket within H3 parent cell)
    if categorical and target_column is None:
        raise ValueError(
            "If categorical is True, target_column must be specified."
            " It should contain categorical values to stratify the data."
        )
    target_column = target_column or COUNT_COLUMN_NAME

    columns_to_keep = [geometry_column]
    if target_column != COUNT_COLUMN_NAME:
        columns_to_keep.append(target_column)

    _gdf = input_gdf[columns_to_keep].copy()
    _gdf[H3_COLUMN_NAME] = _gdf[geometry_column].apply(
        lambda pt: h3.latlng_to_cell(pt.y, pt.x, parent_h3_resolution)
    )

    if target_column == COUNT_COLUMN_NAME:
        h3_cells_stats = _gdf.groupby(H3_COLUMN_NAME).size().reset_index(name=POINTS_COLUMN_NAME)
        h3_cells_stats[BUCKET_COLUMN_NAME] = pd.qcut(
            h3_cells_stats[POINTS_COLUMN_NAME], n_bins, labels=False, duplicates="drop"
        )
    else:
        h3_cells_stats = _gdf.copy()
        if categorical:
            h3_cells_stats.rename(columns={target_column: BUCKET_COLUMN_NAME}, inplace=True)
        else:
            h3_cells_stats[BUCKET_COLUMN_NAME] = pd.qcut(
                h3_cells_stats[target_column], n_bins, labels=False, duplicates="drop"
            )
        h3_cells_stats = (
            h3_cells_stats.groupby([H3_COLUMN_NAME, BUCKET_COLUMN_NAME])
            .size()
            .reset_index(name=POINTS_COLUMN_NAME)
        )

    # Save list of all buckets in the input table
    stratification_buckets = sorted(h3_cells_stats[BUCKET_COLUMN_NAME].unique())
    # Shuffle statistics using random_state
    h3_cells_stats_shuffled = h3_cells_stats.sample(frac=1, random_state=random_state)

    # Define expected ratios for three split datasets
    expected_ratios = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }

    splits = [split for split, ratio in expected_ratios.items() if ratio > 0]

    # Dict for tracking selected parent H3 cells per split
    h3_cell_buckets: dict[str, list[str]] = {split: [] for split in splits}
    # Prepare objects for keeping track of total number of points per bucket and split
    sums = {
        stratification_bucket: {split: 0 for split in splits}
        for stratification_bucket in stratification_buckets
    }

    # Iterate unique H3 cells from the shuffled dataset
    for h3_cell in tqdm(
        h3_cells_stats_shuffled[H3_COLUMN_NAME].unique(),
        desc="Splitting H3 cells",
        disable=FORCE_TERMINAL or not verbose,
    ):
        # Find all statistics per bucket for this parent H3 cell
        rows = h3_cells_stats_shuffled[h3_cells_stats_shuffled[H3_COLUMN_NAME] == h3_cell].to_dict(
            orient="records"
        )

        # Keep track of the smallest found ratio difference
        smallest_ratio_difference = None
        split_to_add_h3_cell = None

        # Check for which group,
        # adding a new entry will result in the best match
        # to the expected ratios for each bucket.
        for current_split in splits:
            if expected_ratios[current_split] == 0:
                continue

            # Keep track of a total difference
            ratio_difference_for_all_buckets = 0
            # Iterate all buckets existing in the current H3 cell
            for row in rows:
                current_stratification_bucket = row[BUCKET_COLUMN_NAME]
                current_number_of_cells = row[POINTS_COLUMN_NAME]

                # Calculate what will be the new total sum of all points so far
                new_total_sum = (
                    sum(sums[current_stratification_bucket].values()) + current_number_of_cells
                )

                # Calculate new ratios after adding this H3 cell to a given split.
                # Simulates what will happen if we increa the sum only for a single split
                # and compare the ratios after.
                new_ratios = {
                    split: (
                        (sums[current_stratification_bucket][split] + current_number_of_cells)
                        / new_total_sum
                        if split == current_split
                        else sums[current_stratification_bucket][split] / new_total_sum
                    )
                    for split in splits
                }

                # Calculate total absolute ratio difference to the expected
                ratio_difference = sum(
                    abs(expected_ratios[split] - new_ratios[split]) for split in splits
                )
                # Increase the difference from all buckets
                ratio_difference_for_all_buckets += ratio_difference

            # If there is a new smallest ratio difference - swap it
            if (
                smallest_ratio_difference is None
                or ratio_difference_for_all_buckets < smallest_ratio_difference
            ):
                smallest_ratio_difference = ratio_difference_for_all_buckets
                split_to_add_h3_cell = current_split

        # Modify list of sums after selecting best matching split
        # We have to add all points for each bucket separately to the dict.
        for row in rows:
            current_stratification_bucket = row[BUCKET_COLUMN_NAME]
            current_number_of_cells = row[POINTS_COLUMN_NAME]
            sums[current_stratification_bucket][cast("str", split_to_add_h3_cell)] += (
                current_number_of_cells
            )

        # Add current H3 cell to the best matching split
        h3_cell_buckets[cast("str", split_to_add_h3_cell)].append(h3_cell)

    # Calculate total sum of points per split
    total_sums = {
        split: sum(sums[bucket][split] for bucket in stratification_buckets) for split in splits
    }
    # Calculate actual ratios base on total sums
    actual_ratios = {
        split: round(
            sum(sums[bucket][split] for bucket in stratification_buckets)
            / sum(total_sums.values()),
            3,
        )
        for split in splits
    }
    # Calculate difference from the expected ratios
    actual_ratios_differences = {
        split: round(expected_ratios[split] - actual_ratios[split], 3) for split in splits
    }

    # Calculate ratio and difference for each bucket
    table_summary_data = []
    for stratification_bucket in stratification_buckets:
        bucket_ratios = {
            split: round(
                sums[stratification_bucket][split] / sum(sums[stratification_bucket].values()),
                5,
            )
            for split in splits
        }
        bucket_ratios_differences = {
            split: round(expected_ratios[split] - bucket_ratios[split], 5) for split in splits
        }
        bucket_points = {split: sums[stratification_bucket][split] for split in splits}

        table_summary_data.append(
            {
                BUCKET_COLUMN_NAME: stratification_bucket,
                **{f"{k}_ratio": v for k, v in bucket_ratios.items()},
                **{f"{k}_ratio_difference": v for k, v in bucket_ratios_differences.items()},
                **{f"{k}_points": v for k, v in bucket_points.items()},
            }
        )

    table_summary_df = pd.DataFrame(table_summary_data)

    # Display splitting results
    if verbose:
        print("Summary of the split:\n")
        train_h3_cells = len(h3_cell_buckets["train"])
        train_points = total_sums["train"]

        val_h3_cells = len(h3_cell_buckets.get("validation", []))
        val_points = total_sums.get("validation", 0)

        test_h3_cells = len(h3_cell_buckets.get("test", []))
        test_points = total_sums.get("test", 0)

        print(f"  Train: {train_h3_cells} H3 cells ({train_points} points)")
        if val_points:
            print(f"  Validation: {val_h3_cells} H3 cells ({val_points} points)")
        if test_points:
            print(f"  Test: {test_h3_cells} H3 cells ({test_points} points)")

        print()
        print("  Expected ratios:", expected_ratios)
        print("  Actual ratios:", actual_ratios)
        print("  Actual ratios difference:", actual_ratios_differences)

        print(table_summary_df)

    # Split input table into three dataframes
    # (Can skip data if the expected ratio is 0 and there are no H3 cells in the bucket)
    splitted_data: dict[str, Optional[str]] = {}
    for split in expected_ratios.keys():
        splitted_data[split] = None
        if split not in h3_cell_buckets or not h3_cell_buckets[split]:
            continue

        matching_indexes = _gdf[_gdf[H3_COLUMN_NAME].isin(h3_cell_buckets[split])].index
        splitted_data[split] = input_gdf.loc[matching_indexes]

    # Return dict with split name and corresponding dataframe
    if return_split_stats:
        return splitted_data, table_summary_df

    return splitted_data
