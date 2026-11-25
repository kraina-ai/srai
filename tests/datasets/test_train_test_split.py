"""Tests for splitting datasets without downloading the data."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from srai.datasets import PortoTaxiDataset


@pytest.fixture  # type: ignore
def trajectories_gdf() -> gpd.GeoDataFrame:
    """Create synthetic H3 trajectories."""
    return gpd.GeoDataFrame(
        {
            "trip_id": range(40),
            "duration": [float(i) for i in range(40)],
            "h3_sequence": [[f"cell_{j}" for j in range(i % 8 + 2)] for i in range(40)],
        },
        geometry=[LineString([(0, 0), (1, 1)])] * 40,
        crs=4326,
    )


@pytest.mark.parametrize("task", ["TTE", "HMP"])  # type: ignore
@pytest.mark.parametrize("validation_split", [False, True])  # type: ignore
def test_trajectory_split_drops_helper_columns(
    trajectories_gdf: gpd.GeoDataFrame, task: str, validation_split: bool
) -> None:
    """Test if columns used for stratification are not present in any of the splits."""
    dataset = PortoTaxiDataset()
    dataset.train_gdf = trajectories_gdf

    splits = dataset.train_test_split(
        target_column="trip_id", task=task, validation_split=validation_split, random_state=42
    )

    for split in splits:
        assert not {"x_len", "y_len", "stratification_bin", "stratify_col"} & set(split.columns)
