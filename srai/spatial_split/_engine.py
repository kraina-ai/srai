"""Assignment of the groups of rows to the splits."""

from collections.abc import Hashable, Sequence
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from srai.constants import FORCE_TERMINAL

BUCKET_COLUMN_NAME = "bucket"
GROUP_COLUMN_NAME = "group"
POINTS_COLUMN_NAME = "points"
SPLIT_COLUMN_NAME = "split"


def split_groups(
    groups: npt.NDArray[Any],
    target: Optional[pd.Series],
    ratios: dict[str, float],
    n_bins: int,
    categorical: bool,
    random_state: Optional[int],
    verbose: bool,
    largest_first: bool = False,
    legacy: bool = False,
) -> tuple[dict[str, npt.NDArray[np.intp]], pd.DataFrame]:
    """
    Split rows so that each group is in a single split and the target is stratified.

    Args:
        groups (npt.NDArray[Any]): Label of the group for each row.
        target (Optional[pd.Series]): Target values for each row. If None, number of rows in the
            group is used.
        ratios (dict[str, float]): Expected ratio for each split.
        n_bins (int): Number of buckets used for the numerical target.
        categorical (bool): Whether the target is categorical.
        random_state (Optional[int]): Random state for reproducibility.
        verbose (bool): Print the progress and report output.
        largest_first (bool, optional): Assign groups in the order of their size instead of the
            random one. Splits are more equal, but the largest groups are always assigned the same
            way. Defaults to False.
        legacy (bool, optional): Keep the results from the 0.10.2 version for a given random state.
            Defaults to False.

    Returns:
        tuple[dict[str, npt.NDArray[np.intp]], pd.DataFrame]: Sorted positions of the rows in each
            split and the summary per bucket.
    """
    stats = _get_group_stats(groups, target, n_bins, categorical, legacy)
    stats = stats.sample(frac=1, random_state=random_state)

    if largest_first:
        group_sizes = stats.groupby(GROUP_COLUMN_NAME)[POINTS_COLUMN_NAME].transform("sum")
        stats = stats.loc[group_sizes.sort_values(ascending=False, kind="stable").index]

    group_splits = _assign_groups(stats, ratios, verbose, legacy)

    stats[SPLIT_COLUMN_NAME] = stats[GROUP_COLUMN_NAME].map(group_splits)
    summary = _get_summary(stats, ratios)
    if verbose:
        _print_report(stats, summary, ratios)

    row_splits = pd.Series(groups).map(group_splits).to_numpy()
    return {split: np.flatnonzero(row_splits == split) for split in ratios}, summary


def _get_group_stats(
    groups: npt.NDArray[Any],
    target: Optional[pd.Series],
    n_bins: int,
    categorical: bool,
    legacy: bool,
) -> pd.DataFrame:
    """Get number of rows per group and bucket."""
    rows = pd.DataFrame({GROUP_COLUMN_NAME: groups})

    if target is None:
        stats = rows.groupby(GROUP_COLUMN_NAME).size().reset_index(name=POINTS_COLUMN_NAME)
        stats[BUCKET_COLUMN_NAME] = _get_quantile_buckets(
            stats[POINTS_COLUMN_NAME], n_bins, split_single_bucket=not legacy
        )
        return stats

    rows[BUCKET_COLUMN_NAME] = target if categorical else _get_quantile_buckets(target, n_bins)
    stats = (
        # observed - skip unused categories
        rows.groupby([GROUP_COLUMN_NAME, BUCKET_COLUMN_NAME], observed=True)
        .size()
        .reset_index(name=POINTS_COLUMN_NAME)
    )

    # Not all data types report those values as missing
    invalid_values = len(rows) - stats[POINTS_COLUMN_NAME].sum()
    if invalid_values:
        raise ValueError(f"Found {invalid_values} missing or infinite values in the target.")

    return stats


def _get_quantile_buckets(
    values: pd.Series, n_bins: int, split_single_bucket: bool = True
) -> pd.Series:
    if values.nunique() == 1:
        return pd.Series(0, index=values.index)

    buckets = pd.qcut(values, n_bins, labels=False, duplicates="drop")
    if split_single_bucket and n_bins > 1 and buckets.nunique() == 1:
        # Single value dominates other ones and all quantiles are the same.
        unique_values = pd.Series(values.unique())
        unique_buckets = pd.qcut(unique_values, n_bins, labels=False, duplicates="drop")
        buckets = values.map(dict(zip(unique_values, unique_buckets)))

    return buckets


def _assign_groups(
    stats: pd.DataFrame, ratios: dict[str, float], verbose: bool, legacy: bool
) -> dict[Hashable, str]:
    """Assign groups to the splits one by one, using the best match to the expected ratios."""
    group_rows: dict[Hashable, list[tuple[Hashable, int]]] = {}
    for group, bucket, points in zip(
        stats[GROUP_COLUMN_NAME], stats[BUCKET_COLUMN_NAME], stats[POINTS_COLUMN_NAME]
    ):
        group_rows.setdefault(group, []).append((bucket, int(points)))

    splits = list(ratios)
    if len(group_rows) < len(splits):
        raise ValueError(
            f"Cannot create {len(splits)} splits from {len(group_rows)} region(s) with points."
        )

    bucket_sums = {bucket: [0] * len(splits) for bucket in stats[BUCKET_COLUMN_NAME]}
    bucket_sizes: dict[Hashable, int] = dict.fromkeys(bucket_sums, 0)
    for rows in group_rows.values():
        for bucket, points in rows:
            bucket_sizes[bucket] += points
    total_sums = [0] * len(splits)
    total_size = sum(bucket_sizes.values())
    split_ratios = list(ratios.values())
    group_splits: dict[Hashable, str] = {}

    for group, rows in tqdm(
        group_rows.items(), desc="Splitting regions", disable=FORCE_TERMINAL or not verbose
    ):
        group_size = sum(points for _, points in rows)
        if legacy:
            costs = _get_ratio_differences(
                [(bucket_sums[bucket], points) for bucket, points in rows], split_ratios
            )
        else:
            costs = _get_expected_size_differences(
                [
                    *(
                        (bucket_sums[bucket], points, bucket_sizes[bucket])
                        for bucket, points in rows
                    ),
                    (total_sums, group_size, total_size),
                ],
                split_ratios,
            )

        # Leave at least one group for each split
        candidates = list(range(len(splits)))
        empty_splits = [split for split in candidates if not total_sums[split]]
        if len(group_rows) - len(group_splits) <= len(empty_splits):
            candidates = empty_splits

        best_split = min(candidates, key=costs.__getitem__)

        for bucket, points in rows:
            bucket_sums[bucket][best_split] += points
        total_sums[best_split] += group_size
        group_splits[group] = splits[best_split]

    return group_splits


def _get_expected_size_differences(
    sets: list[tuple[list[int], int, int]], ratios: Sequence[float]
) -> list[float]:
    """
    Get change of the chi-squared distance to the expected sizes for each possible split.

    Args:
        sets (list[tuple[list[int], int, int]]): Current sizes of the splits, number of points to be
            added and the size of all splits in the end, for each bucket and for all points.
        ratios (Sequence[float]): Expected ratios of the splits.

    Returns:
        list[float]: Difference after adding the points to each split.
    """
    return [
        sum(
            points * (2 * (sums[split] - ratio * size) + points) / (ratio * size)
            for sums, points, size in sets
        )
        for split, ratio in enumerate(ratios)
    ]


def _get_ratio_differences(
    sets: list[tuple[list[int], int]], ratios: Sequence[float]
) -> list[float]:
    """
    Get total difference between current and expected ratios for each possible split.

    Order of the operations must be kept, because results for a given random state depend on it.

    Args:
        sets (list[tuple[list[int], int]]): Current sizes of the splits and number of points to be
            added for each bucket.
        ratios (Sequence[float]): Expected ratios of the splits.

    Returns:
        list[float]: Difference after adding the points to each split.
    """
    differences = []
    for selected_split in range(len(ratios)):
        total_difference = 0.0
        for sums, points in sets:
            new_total = sum(sums) + points
            total_difference += sum(
                abs(ratio - (sums[split] + points * (split == selected_split)) / new_total)
                for split, ratio in enumerate(ratios)
            )
        differences.append(total_difference)
    return differences


def _get_summary(stats: pd.DataFrame, ratios: dict[str, float]) -> pd.DataFrame:
    """Get number of points and ratios for each bucket."""
    # Type is compared first, because values of different types might not be comparable
    buckets = sorted(stats[BUCKET_COLUMN_NAME].unique(), key=lambda x: (type(x).__name__, x))
    points = stats.groupby([BUCKET_COLUMN_NAME, SPLIT_COLUMN_NAME], observed=True)[
        POINTS_COLUMN_NAME
    ].sum()

    summary_data = []
    for bucket in buckets:
        bucket_points = {split: int(points.get((bucket, split), 0)) for split in ratios}
        bucket_total = sum(bucket_points.values())
        bucket_ratios = {split: round(bucket_points[split] / bucket_total, 5) for split in ratios}
        summary_data.append(
            {
                BUCKET_COLUMN_NAME: bucket,
                **{f"{split}_ratio": bucket_ratios[split] for split in ratios},
                **{
                    f"{split}_ratio_difference": round(ratio - bucket_ratios[split], 5)
                    for split, ratio in ratios.items()
                },
                **{f"{split}_points": bucket_points[split] for split in ratios},
            }
        )

    return pd.DataFrame(summary_data)


def _print_report(stats: pd.DataFrame, summary: pd.DataFrame, ratios: dict[str, float]) -> None:
    points = stats.groupby(SPLIT_COLUMN_NAME)[POINTS_COLUMN_NAME].sum()
    regions = stats.groupby(SPLIT_COLUMN_NAME)[GROUP_COLUMN_NAME].nunique()
    actual_ratios = {split: round(float(points[split] / points.sum()), 3) for split in ratios}

    print("Summary of the split:\n")
    for split in ratios:
        print(f"  {split}: {regions[split]} regions ({points[split]} points)")

    print()
    print("  Expected ratios:", ratios)
    print("  Actual ratios:", actual_ratios)
    print(
        "  Actual ratios difference:",
        {split: round(ratio - actual_ratios[split], 3) for split, ratio in ratios.items()},
    )
    print(summary)
