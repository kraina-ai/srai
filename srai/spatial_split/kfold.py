"""Spatial K-Fold split."""

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


def spatial_kfold(
    points: PointsInput,
    n_splits: int = 5,
    *,
    regions: RegionsInput,
    target: Optional[TargetInput] = None,
    n_bins: int = 7,
    categorical: bool = False,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    """
    Split points into K folds, without sharing regions between train and test datasets.

    Points are split into K equal parts. Each part is used once as a test dataset, while remaining
    ones form a train dataset. Returned folds can be used as a `cv` parameter in the scikit-learn.

    Args:
        points (PointsInput): Points to be split - GeoDataFrame (active geometry is used), GeoSeries
            or an iterable of Shapely points. If CRS is not defined, WGS84 is assumed.
        n_splits (int, optional): Number of folds. Defaults to 5.
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
        list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]: Sorted positions of the points in
            the train and test datasets for each fold.

    Raises:
        ValueError: If input is empty or has geometries other than points, target has missing or
            infinite values, points are outside of the regions or there are fewer regions
            than folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    points = to_geopandas(points)

    splits, _ = split_groups(
        groups=get_groups(parse_points(points.geometry), regions),
        target=parse_target(points, target, categorical),
        ratios={f"fold_{i}": 1 / n_splits for i in range(n_splits)},
        n_bins=n_bins,
        categorical=categorical,
        random_state=random_state,
        verbose=verbose,
        # Each region is tested exactly once, so there is no need for a random order.
        largest_first=True,
    )
    all_positions = np.arange(len(points))
    return [(np.setdiff1d(all_positions, test), test) for test in splits.values()]
