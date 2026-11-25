"""Spatial split into train, validation and test datasets."""

from typing import Literal, Optional, Union, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd

from srai.constants import GEOMETRY_COLUMN
from srai.spatial_split._engine import split_groups
from srai.spatial_split._inputs import (
    PointsInput,
    TargetInput,
    get_groups,
    parse_points,
    parse_target,
    to_geopandas,
)

COUNT_COLUMN_NAME = "count"


def train_test_spatial_split(
    input_gdf: PointsInput,
    parent_h3_resolution: int,
    geometry_column: str = GEOMETRY_COLUMN,
    target_column: Optional[TargetInput] = None,
    n_bins: int = 7,
    categorical: bool = False,
    test_size: Union[float, int] = 0.2,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Split data based on parent H3 cell and stratify the data using specified target.

    Args:
        input_gdf (PointsInput): Points to be split - GeoDataFrame, GeoSeries or an iterable
            of Shapely points. If CRS is not defined, WGS84 is assumed.
        parent_h3_resolution (int): H3 resolution used to split the data.
        geometry_column (str, optional): Name of the geometry column. If GeoDataFrame doesn't have
            the default column, active geometry is used. Defaults to "geometry".
        target_column (Optional[TargetInput], optional): Target column name, or values in the same
            order as points, used to stratify the data distribution. If None, split is generated
            based on number of points within a hex of a given resolution. Defaults to None.
        n_bins (int, optional): Bucket number used to stratify target data.
            Only used when categorical is False. Defaults to 7.
        categorical (bool, optional): If True, target column is treated as categorical.
            Target column must be specified in this case. Defaults to False.
        test_size (Union[float, int], optional): Size of the test dataset.
            Can be a fraction (0-1 range) or a total number of rows. Defaults to 0.2.
        random_state (Optional[int], optional): Random state for reproducibility. Defaults to None.
        verbose (bool, optional): Print the progress and report output. Defaults to True.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Train and test GeoDataFrames (or GeoSeries, if
            input isn't a GeoDataFrame).
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
    input_gdf: PointsInput,
    parent_h3_resolution: int,
    geometry_column: str = GEOMETRY_COLUMN,
    target_column: Optional[TargetInput] = None,
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
    input_gdf: PointsInput,
    parent_h3_resolution: int,
    geometry_column: str = GEOMETRY_COLUMN,
    target_column: Optional[TargetInput] = None,
    n_bins: int = 7,
    categorical: bool = False,
    test_size: Union[float, int] = 0.2,
    validation_size: Union[float, int] = 0,
    random_state: Optional[int] = None,
    *,
    return_split_stats: Literal[True],
    verbose: bool = True,
) -> tuple[dict[str, Optional[gpd.GeoDataFrame]], pd.DataFrame]: ...


def spatial_split_points(
    input_gdf: PointsInput,
    parent_h3_resolution: int,
    geometry_column: str = GEOMETRY_COLUMN,
    target_column: Optional[TargetInput] = None,
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
        input_gdf (PointsInput): Points to be split - GeoDataFrame, GeoSeries or an iterable
            of Shapely points. If CRS is not defined, WGS84 is assumed.
        parent_h3_resolution (int): H3 resolution used to split the data.
        geometry_column (str, optional): Name of the geometry column. If GeoDataFrame doesn't have
            the default column, active geometry is used. Defaults to "geometry".
        target_column (Optional[TargetInput], optional): Target column name, or values in the same
            order as points, used to stratify the data distribution. If None, split is generated
            based on number of points within a hex of a given resolution. Defaults to None.
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
        Union[dict[str, Optional[gpd.GeoDataFrame]],
            tuple[dict[str, Optional[gpd.GeoDataFrame]], pd.DataFrame]]: Dictionary with `train`,
            `validation` and `test` GeoDataFrames (or GeoSeries, if input isn't a GeoDataFrame).
            If the requested size is 0, value is None. With `return_split_stats`, also a DataFrame
            with the statistics per bucket.
    """
    input_gdf = to_geopandas(input_gdf)
    points = parse_points(_get_geometry(input_gdf, geometry_column))

    test_fraction = _get_fraction(test_size, "test_size", len(points))
    validation_fraction = _get_fraction(validation_size, "validation_size", len(points))

    if (validation_fraction + test_fraction) >= 1:
        raise ValueError("Test and validation fractions sum up to 1 or more.")

    ratios = {
        # Calculate missing train fraction (3 groups adds up to 1)
        "train": 1 - validation_fraction - test_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }

    if (
        isinstance(target_column, str)
        and target_column == COUNT_COLUMN_NAME
        and COUNT_COLUMN_NAME not in getattr(input_gdf, "columns", [])
    ):
        target_column = None

    positions, summary = split_groups(
        groups=get_groups(points, parent_h3_resolution),
        target=parse_target(input_gdf, target_column, categorical),
        ratios={split: ratio for split, ratio in ratios.items() if ratio > 0},
        n_bins=n_bins,
        categorical=categorical,
        random_state=random_state,
        verbose=verbose,
        # Keep the results for a given random state the same as in the previous versions.
        legacy=True,
    )

    splits = {
        split: input_gdf.iloc[positions[split]].copy() if split in positions else None
        for split in ratios
    }

    if return_split_stats:
        return splits, summary

    return splits


def _get_geometry(
    points: Union[gpd.GeoDataFrame, gpd.GeoSeries], geometry_column: str
) -> gpd.GeoSeries:
    if isinstance(points, gpd.GeoSeries):
        return points
    if geometry_column == GEOMETRY_COLUMN and GEOMETRY_COLUMN not in points.columns:
        return points.geometry
    return gpd.GeoSeries(points[geometry_column])


def _get_fraction(size: Union[float, int], name: str, number_of_points: int) -> float:
    if size < 0:
        raise ValueError(f"{name} cannot be negative.")
    if size < 1:
        return size
    if not isinstance(size, (int, np.integer)):
        raise ValueError(f"{name} must be a fraction lower than 1 or an integer number of rows.")
    return cast("float", size / number_of_points)
