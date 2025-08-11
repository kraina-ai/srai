"""Spatial splitting tests."""

from typing import cast
from unittest import TestCase

import geopandas as gpd
import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from srai.spatial_split import spatial_split_points

ut = TestCase()


def get_random_points_gdf(number_of_points: int, seed: int) -> gpd.GeoDataFrame:
    """Get random points within WGS84 bounds sampled on a sphere."""
    rng = np.random.default_rng(seed=seed)
    values = rng.standard_normal(number_of_points)
    values_cat = rng.choice(["A", "B", "C", "D"], size=number_of_points, replace=True)
    lon = rng.uniform(-1, 1, number_of_points)
    lat = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
        rng.standard_normal(size=(number_of_points, 1))
    )[:, 0]

    return gpd.GeoDataFrame(
        data={"target": values, "target_cat": values_cat},
        geometry=gpd.points_from_xy(lon, lat, crs=4326),
    )


@pytest.mark.parametrize("n_bins", [3, 7, 10])  # type: ignore
@pytest.mark.parametrize("target", ["count", "target", "target_cat"])  # type: ignore
@pytest.mark.parametrize("test_size", [0.2, 0.5, 25_000])  # type: ignore
@pytest.mark.parametrize("validation_size", [0, 0.2, 15_000])  # type: ignore
def test_spatial_splits(n_bins: int, target: str, test_size: float, validation_size: float) -> None:
    """Test checks if regions are generated correctly."""
    if target == "target_cat" and n_bins > 3:
        pytest.skip("No need to test categorical target for different n_bins.")

    seed = np.random.default_rng().integers(100_000_000)
    points = get_random_points_gdf(100_000, seed)

    splits, table_summary_df = spatial_split_points(
        input_gdf=points,
        parent_h3_resolution=6,
        target_column=target,
        categorical=target == "target_cat",
        n_bins=n_bins,
        test_size=test_size,
        validation_size=validation_size,
        return_split_stats=True,
    )

    assert len(points) == sum(len(_df) for _df in splits.values() if _df is not None), (
        "Returned splits don't sum to original dataframe length."
    )

    ut.assertListEqual(
        points.columns.to_list(),
        cast("gpd.GeoDataFrame", splits["train"]).columns.to_list(),
        "Columns in splits do not match original dataframe.",
    )

    for _, row in table_summary_df.iterrows():
        assert abs(row["train_ratio_difference"]) < 0.01, (
            f"Train ratio above threshold ({row['train_ratio_difference']})"
        )

        assert abs(row["test_ratio_difference"]) < 0.01, (
            f"Test ratio above threshold ({row['test_ratio_difference']})"
        )

        if validation_size > 0:
            assert abs(row["validation_ratio_difference"]) < 0.01, (
                f"Validation ratio above threshold ({row['validation_ratio_difference']})"
            )
