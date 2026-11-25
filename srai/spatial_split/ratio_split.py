"""Spatial split into any number of splits."""

from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt

from srai.spatial_split._engine import split_groups
from srai.spatial_split._inputs import (
    PointsInput,
    RegionsInput,
    TargetInput,
    get_groups,
    parse_points,
    parse_target,
    to_geopandas,
)


def spatial_split(
    points: PointsInput,
    ratios: Sequence[float],
    regions: RegionsInput,
    target: Optional[TargetInput] = None,
    n_bins: int = 7,
    categorical: bool = False,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> list[npt.NDArray[np.intp]]:
    """
    Split points into any number of splits, without sharing regions between them.

    Args:
        points (PointsInput): Points to be split - GeoDataFrame (active geometry is used), GeoSeries
            or an iterable of Shapely points. If CRS is not defined, WGS84 is assumed.
        ratios (Sequence[float]): Expected sizes of the splits. Must sum up to 1.
        regions (RegionsInput): Regions used to group the points - H3 resolution or geometries
            (for example, result of any regionalizer). If a point intersects multiple geometries,
            the first one is used. All points must be covered by the geometries.
        target (Optional[TargetInput], optional): Target used to stratify the data distribution.
            Name of the column if points are a GeoDataFrame, or values in the same order as points
            (index is ignored). If None, split is generated based on the number of points within
            a region. Defaults to None.
        n_bins (int, optional): Bucket number used to stratify target data.
            Only used when categorical is False. Defaults to 7.
        categorical (bool, optional): If True, target is treated as categorical.
            Target must be specified in this case. Defaults to False.
        random_state (Optional[int], optional): Random state for reproducibility. Defaults to None.
        verbose (bool, optional): Print the progress and report output. Defaults to True.

    Returns:
        list[npt.NDArray[np.intp]]: Sorted positions of the points in each split.

    Raises:
        ValueError: If input is empty or has geometries other than points, target has missing or
            infinite values, points are outside of the regions, ratios are wrong or there are fewer
            regions than splits.

    Examples:
        Split into train, validation and test datasets using H3 cells:

        >>> import geopandas as gpd
        >>> from srai.spatial_split import spatial_split
        >>> gdf = gpd.GeoDataFrame(
        ...     {"price": range(100)},
        ...     geometry=gpd.points_from_xy(range(100), [0] * 100, crs=4326),
        ... )
        >>> train, validation, test = spatial_split(
        ...     gdf, [0.7, 0.1, 0.2], regions=3, target="price", random_state=0, verbose=False
        ... )
        >>> train_gdf = gdf.iloc[train]
    """
    if len(ratios) < 2 or not all(ratio > 0 for ratio in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("At least two ratios greater than 0 that sum up to 1 must be provided.")

    points = to_geopandas(points)

    splits, _ = split_groups(
        groups=get_groups(parse_points(points.geometry), regions),
        target=parse_target(points, target, categorical),
        ratios={f"split_{i}": ratio for i, ratio in enumerate(ratios)},
        n_bins=n_bins,
        categorical=categorical,
        random_state=random_state,
        verbose=verbose,
    )
    return list(splits.values())
