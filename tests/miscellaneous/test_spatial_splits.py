"""Spatial splitting tests."""

import hashlib
import warnings
from itertools import starmap
from typing import Any, Optional, Union

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pytest
from shapely import Point, box
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from srai.constants import WGS84_CRS
from srai.spatial_split import (
    spatial_kfold,
    spatial_split,
    spatial_split_points,
    train_test_spatial_split,
)
from srai.spatial_split._engine import split_groups

H3_RESOLUTION = 6


def get_clustered_points_gdf(number_of_points: int = 2000, seed: int = 42) -> gpd.GeoDataFrame:
    """Get random points grouped in clusters of different sizes."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.5, 0.5, size=(40, 2))
    weights = 1 / np.arange(1, 41)
    weights /= weights.sum()
    cluster = rng.choice(40, size=number_of_points, p=weights)
    xy = centers[cluster] + rng.normal(0, 0.02, size=(number_of_points, 2))
    target = rng.lognormal(size=number_of_points) * (1 + cluster / 40)

    return gpd.GeoDataFrame(
        {"target": target, "target_cat": np.where(target > 2, "high", "low")},
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1], crs=WGS84_CRS),
    )


def get_grid_regions(cells_per_side: int = 10) -> gpd.GeoSeries:
    """Get square grid covering all clustered points."""
    edges = np.linspace(-1, 1, cells_per_side + 1)
    return gpd.GeoSeries(
        [
            box(x_min, y_min, x_max, y_max)
            for x_min, x_max in zip(edges[:-1], edges[1:])
            for y_min, y_max in zip(edges[:-1], edges[1:])
        ],
        crs=WGS84_CRS,
    )


def get_h3_cells(points: gpd.GeoDataFrame, resolution: int = H3_RESOLUTION) -> np.ndarray:
    """Get H3 cell for each point."""
    return np.array([h3.latlng_to_cell(pt.y, pt.x, resolution) for pt in points.geometry])


def assert_partition(splits: list[np.ndarray], length: int) -> None:
    """Assert that each position is in exactly one split."""
    assert np.array_equal(np.sort(np.concatenate(splits)), np.arange(length))
    for split in splits:
        assert np.array_equal(split, np.sort(split))


def assert_disjoint_groups(splits: list[np.ndarray], groups: np.ndarray) -> None:
    """Assert that no group is shared between the splits."""
    groups_per_split = [set(groups[split]) for split in splits]
    assert sum(map(len, groups_per_split)) == len(set(groups))


def get_folds_or_splits(function_name: str, points: Any, **params: Any) -> list[np.ndarray]:
    """Get 4 equal splits from the selected function."""
    params = {"regions": 7, "random_state": 0, "verbose": False} | params
    if function_name == "spatial_split":
        return spatial_split(points, [0.25] * 4, **params)
    return [test for _, test in spatial_kfold(points, 4, **params)]


def are_equal(first: list[np.ndarray], second: list[np.ndarray]) -> bool:
    """Check if all splits are the same."""
    return all(starmap(np.array_equal, zip(first, second)))


@pytest.fixture  # type: ignore
def points() -> gpd.GeoDataFrame:
    """Get points GeoDataFrame."""
    return get_clustered_points_gdf()


# Splits into named datasets


@pytest.mark.parametrize(
    "target_column,test_size,validation_size,expected_hash",
    [
        (None, 0.2, 0, "4fc7c7e271137a43"),
        (None, 0.2, 0.1, "9f23fc535c01b80a"),
        (None, 400, 200, "9f23fc535c01b80a"),
        (None, 0.5, 0, "68b66a691d4f465b"),
        ("count", 0.2, 0, "4fc7c7e271137a43"),
        ("target", 0.2, 0, "f536293231cbf1a9"),
        ("target", 0.2, 0.1, "a52377f1b6df70e0"),
        ("target", 400, 200, "a52377f1b6df70e0"),
        ("target", 0.5, 0, "390a0642c1370604"),
        ("target_cat", 0.2, 0, "c96405ea83e5d4eb"),
        ("target_cat", 0.2, 0.1, "32e2f9fed4288ba0"),
        ("target_cat", 0.5, 0, "18c61547dc5763e9"),
    ],
)  # type: ignore
def test_results_are_stable(
    points: gpd.GeoDataFrame,
    target_column: Optional[str],
    test_size: float,
    validation_size: float,
    expected_hash: str,
) -> None:
    """Test if a random state gives the same splits as in the version 0.10.2."""
    splits = spatial_split_points(
        points,
        parent_h3_resolution=H3_RESOLUTION,
        target_column=target_column,
        categorical=target_column == "target_cat",
        test_size=test_size,
        validation_size=validation_size,
        random_state=42,
        verbose=False,
    )

    assert list(splits) == ["train", "validation", "test"]
    assert (splits["validation"] is None) == (validation_size == 0)

    labels = pd.Series("", index=points.index)
    for name, split in splits.items():
        if split is not None:
            labels[split.index] = name

    assert hashlib.sha256(",".join(labels).encode()).hexdigest()[:16] == expected_hash


@pytest.mark.parametrize("n_bins", [3, 7, 10])  # type: ignore
@pytest.mark.parametrize("target", [None, "target", "target_cat"])  # type: ignore
@pytest.mark.parametrize("test_size", [0.2, 0.5, 25_000])  # type: ignore
@pytest.mark.parametrize("validation_size", [0, 0.2, 15_000])  # type: ignore
def test_spatial_splits(
    n_bins: int, target: Optional[str], test_size: float, validation_size: float
) -> None:
    """Test if points are split with expected ratios."""
    if target == "target_cat" and n_bins > 3:
        pytest.skip("No need to test categorical target for different n_bins.")

    rng = np.random.default_rng(0)
    uniform_points = gpd.GeoDataFrame(
        {
            "target": rng.standard_normal(100_000),
            "target_cat": rng.choice(["A", "B", "C", "D"], size=100_000),
        },
        geometry=gpd.points_from_xy(*rng.uniform(-1, 1, size=(2, 100_000)), crs=WGS84_CRS),
    )

    splits, table_summary_df = spatial_split_points(
        input_gdf=uniform_points,
        parent_h3_resolution=H3_RESOLUTION,
        target_column=target,
        categorical=target == "target_cat",
        n_bins=n_bins,
        test_size=test_size,
        validation_size=validation_size,
        random_state=0,
        return_split_stats=True,
    )

    existing_splits = [split for split in splits.values() if split is not None]
    assert len(existing_splits) == (3 if validation_size else 2)
    assert_partition([split.index.to_numpy() for split in existing_splits], 100_000)
    assert_disjoint_groups(
        [split.index.to_numpy() for split in existing_splits], get_h3_cells(uniform_points)
    )

    for split in existing_splits:
        assert split.columns.to_list() == uniform_points.columns.to_list()

    names = ["train", "validation", "test"] if validation_size else ["train", "test"]
    assert table_summary_df.columns.to_list() == [
        "bucket",
        *(f"{name}_ratio" for name in names),
        *(f"{name}_ratio_difference" for name in names),
        *(f"{name}_points" for name in names),
    ]
    assert table_summary_df.filter(like="_points").to_numpy().sum() == 100_000
    assert (table_summary_df.filter(like="_ratio_difference").abs() < 0.01).all(axis=None)


@pytest.mark.parametrize("as_list", [True, False])  # type: ignore
def test_named_splits_accept_any_points_and_target(points: gpd.GeoDataFrame, as_list: bool) -> None:
    """Test if the output doesn't depend on the type of the input."""
    params: dict[str, Any] = dict(parent_h3_resolution=6, random_state=42, verbose=False)
    expected_train, expected_test = train_test_spatial_split(
        points, target_column="target", **params
    )

    train, test = train_test_spatial_split(
        points.geometry.to_list() if as_list else points.geometry,
        target_column=points["target"].to_list() if as_list else points["target"].to_numpy(),
        **params,
    )

    assert isinstance(train, gpd.GeoSeries)
    assert train.index.equals(expected_train.index)
    assert test.index.equals(expected_test.index)


# Splits into N datasets


@pytest.mark.parametrize("input_type", ["gdf", "series", "list", "generator"])  # type: ignore
@pytest.mark.parametrize("target_type", [None, "list", "numpy", "series"])  # type: ignore
@pytest.mark.parametrize("ratios", [[0.8, 0.2], [0.7, 0.1, 0.2], [0.25] * 4])  # type: ignore
def test_spatial_split(
    points: gpd.GeoDataFrame, input_type: str, target_type: Optional[str], ratios: list[float]
) -> None:
    """Test splitting into N splits for different inputs."""
    points_input: Any = {
        "gdf": points,
        "series": points.geometry,
        "list": points.geometry.to_list(),
        "generator": iter(points.geometry),
    }[input_type]
    target: Any = {
        None: None,
        "list": points["target"].to_list(),
        "numpy": points["target"].to_numpy(),
        # index not related to the points
        "series": points["target"].set_axis(points.index[::-1]),
    }[target_type]

    params: dict[str, Any] = dict(regions=H3_RESOLUTION, random_state=42, verbose=False)
    splits = spatial_split(points_input, ratios, target=target, **params)

    assert len(splits) == len(ratios)
    assert_partition(splits, len(points))
    assert_disjoint_groups(splits, get_h3_cells(points))
    for split, ratio in zip(splits, ratios):
        assert len(split) / len(points) == pytest.approx(ratio, abs=0.03)

    expected = spatial_split(points, ratios, target="target" if target_type else None, **params)
    for split, expected_split in zip(splits, expected):
        assert np.array_equal(split, expected_split)


def test_random_state(points: gpd.GeoDataFrame) -> None:
    """Test if random state controls the output."""
    first, second, other = (
        spatial_split(points, [0.5, 0.5], H3_RESOLUTION, random_state=seed, verbose=False)
        for seed in (1, 1, 2)
    )
    assert np.array_equal(first[0], second[0])
    assert not np.array_equal(first[0], other[0])


@pytest.mark.parametrize("function_name", ["spatial_split", "spatial_kfold"])  # type: ignore
@pytest.mark.parametrize("categorical", [True, False])  # type: ignore
def test_stratification(points: gpd.GeoDataFrame, function_name: str, categorical: bool) -> None:
    """Test if target has the same distribution in all splits."""
    # Target depends on the location
    target = (points.geometry.x * 100).round().to_numpy()
    if categorical:
        target = np.where(target > 0, "east", "west")
    buckets = pd.Series(target) if categorical else pd.qcut(pd.Series(target), 5, labels=False)

    def _get_max_difference(splits: list[np.ndarray]) -> float:
        expected = buckets.value_counts(normalize=True)
        differences = [
            buckets[split].value_counts(normalize=True).sub(expected, fill_value=0).abs().max()
            for split in splits
        ]
        return float(max(differences))

    stratified = get_folds_or_splits(
        function_name, points, target=target, categorical=categorical, n_bins=5
    )
    not_stratified = get_folds_or_splits(function_name, points)

    assert _get_max_difference(stratified) < 0.03
    assert _get_max_difference(not_stratified) > 2 * _get_max_difference(stratified)


@pytest.mark.parametrize("function_name", ["spatial_split", "spatial_kfold"])  # type: ignore
def test_parameters_are_used(points: gpd.GeoDataFrame, function_name: str) -> None:
    """Test if parameters change the result."""
    results = [
        get_folds_or_splits(function_name, points, target="target", **params)
        for params in (
            {},
            {"n_bins": 2},
            {"random_state": 1},
            # each value is a separate category
            {"categorical": True},
        )
    ]
    for i, result in enumerate(results):
        for other in results[i + 1 :]:
            assert not are_equal(result, other)

    assert are_equal(results[0], get_folds_or_splits(function_name, points, target="target"))


def test_summary(points: gpd.GeoDataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    """Test if returned and printed statistics are correct."""
    splits, table = spatial_split_points(
        points,
        H3_RESOLUTION,
        target_column="target_cat",
        categorical=True,
        test_size=0.3,
        random_state=0,
        return_split_stats=True,
    )

    assert table["bucket"].to_list() == ["high", "low"]
    for name, ratio in (("train", 0.7), ("test", 0.3)):
        split = splits[name]
        assert split is not None
        bucket_points = table["bucket"].map(split["target_cat"].value_counts())
        bucket_ratios = bucket_points / table["bucket"].map(points["target_cat"].value_counts())
        assert table[f"{name}_points"].to_list() == bucket_points.to_list()
        assert table[f"{name}_ratio"].to_list() == pytest.approx(bucket_ratios, abs=1e-5)
        assert table[f"{name}_ratio_difference"].to_list() == pytest.approx(
            ratio - bucket_ratios, abs=2e-5
        )

    output = capsys.readouterr().out
    assert splits["train"] is not None and splits["test"] is not None
    assert f"train: {len(set(get_h3_cells(splits['train'])))} regions" in output
    assert f"({len(splits['test'])} points)" in output
    actual_ratios = {
        name: round(len(splits[name]) / len(points), 3)  # type: ignore[arg-type]
        for name in ("train", "test")
    }
    assert f"Actual ratios: {actual_ratios}" in output


# Custom regions


@pytest.mark.parametrize("as_gdf", [True, False])  # type: ignore
@pytest.mark.parametrize("crs", [WGS84_CRS, 3857, None])  # type: ignore
def test_custom_regions(
    points: gpd.GeoDataFrame, as_gdf: bool, crs: Optional[Union[str, int]]
) -> None:
    """Test splitting by custom geometries."""
    grid = get_grid_regions()
    groups = np.array(
        [next(i for i, g in enumerate(grid) if g.intersects(p)) for p in points.geometry]
    )

    regions = gpd.GeoSeries(grid.to_numpy()) if crs is None else grid.to_crs(crs)
    regions.index = [f"region_{i % 3}" for i in range(len(regions))]
    if as_gdf:
        regions = gpd.GeoDataFrame({"name": "x"}, index=regions.index, geometry=regions)

    splits = spatial_split(
        points, [0.5, 0.3, 0.2], regions, target="target", random_state=0, verbose=False
    )

    assert_partition(splits, len(points))
    assert_disjoint_groups(splits, groups)

    expected = spatial_split(
        points, [0.5, 0.3, 0.2], grid, target="target", random_state=0, verbose=False
    )
    assert are_equal(splits, expected)


def test_overlapping_regions_use_first_match() -> None:
    """Test if point within multiple regions is assigned to the first one."""
    points = [Point(0.5, 0.5), Point(0.5, 0.4), Point(1.5, 0.5), Point(1, 0.5)]
    # Points 0, 1 and 3 intersect both regions
    regions = gpd.GeoSeries([box(0, 0, 1, 1), box(0, 0, 2, 1)], crs=WGS84_CRS)
    splits = spatial_split(points, [0.5, 0.5], regions, random_state=0, verbose=False)
    assert sorted(split.tolist() for split in splits) == [[0, 1, 3], [2]]


def test_points_outside_regions(points: gpd.GeoDataFrame) -> None:
    """Test if points outside of the regions raise an error."""
    with pytest.raises(ValueError, match="outside of the provided regions"):
        spatial_split(points, [0.5, 0.5], get_grid_regions().iloc[:50], verbose=False)


def test_wrong_regions_type(points: gpd.GeoDataFrame) -> None:
    """Test if unknown regions type raises an error."""
    with pytest.raises(TypeError, match="regions"):
        spatial_split(points, [0.5, 0.5], "8", verbose=False)


# K-Folds


@pytest.mark.parametrize("n_splits", [2, 5, 10])  # type: ignore
@pytest.mark.parametrize("target", [None, "target", "target_cat"])  # type: ignore
@pytest.mark.parametrize("custom_regions", [True, False])  # type: ignore
def test_kfold(
    points: gpd.GeoDataFrame, n_splits: int, target: Optional[str], custom_regions: bool
) -> None:
    """Test if each point is tested exactly once."""
    regions = get_grid_regions(100) if custom_regions else 7
    folds = spatial_kfold(
        points,
        n_splits,
        regions=regions,
        target=target,
        categorical=target == "target_cat",
        random_state=0,
        verbose=False,
    )

    assert len(folds) == n_splits
    assert_partition([test for _, test in folds], len(points))
    for train, test in folds:
        assert_partition([train, test], len(points))
        assert len(test) / len(points) == pytest.approx(1 / n_splits, abs=0.02)
        if not custom_regions:
            assert_disjoint_groups([train, test], get_h3_cells(points, 7))


def test_kfold_with_few_regions() -> None:
    """Test if folds are balanced when there are only a few regions per bucket."""
    sizes = [300, 280, 260, 240, 220, 100, 90, 80, 70, 60]
    points = [Point(i, 0) for i, size in enumerate(sizes) for _ in range(size)]
    regions = gpd.GeoSeries([Point(i, 0).buffer(0.4) for i in range(len(sizes))], crs=WGS84_CRS)

    folds = spatial_kfold(points, 5, regions=regions, random_state=0, verbose=False)
    for _, test in folds:
        assert len(test) / len(points) == pytest.approx(0.2, abs=0.05)

    # leave one region out
    folds = spatial_kfold(points, 10, regions=regions, random_state=0, verbose=False)
    assert sorted(len(test) for _, test in folds) == sorted(sizes)


@pytest.mark.parametrize("target", [None, "constant", "rare"])  # type: ignore
def test_kfold_with_different_region_sizes(target: Optional[str]) -> None:
    """Test if the last folds aren't smaller than the other ones."""
    sizes = [2000 // i + 1 for i in range(1, 41)]
    points = [Point(i, 0) for i, size in enumerate(sizes) for _ in range(size)]
    regions = gpd.GeoSeries([Point(i, 0).buffer(0.4) for i in range(len(sizes))], crs=WGS84_CRS)
    values = None
    if target is not None:
        values = [int(target == "rare" and i % 10 == 0) for i in range(len(points))]

    folds = spatial_kfold(
        points,
        10,
        regions=regions,
        target=values,
        categorical=target is not None,
        random_state=0,
        verbose=False,
    )
    # Two largest regions have more points than the expected size of the fold (865)
    assert sorted(len(test) for _, test in folds)[-2:] == [1001, 2001]
    assert min(len(test) for _, test in folds) > 600


def test_kfold_in_scikit_learn(points: gpd.GeoDataFrame) -> None:
    """Test if folds can be used in the scikit-learn."""
    folds = spatial_kfold(points, 4, regions=H3_RESOLUTION, target="target", verbose=False)
    features = np.column_stack([points.geometry.x, points.geometry.y])
    scores = cross_val_score(LinearRegression(), features, points["target"], cv=folds)
    assert len(scores) == 4
    assert np.isfinite(scores).all()


# Input data problems


def test_duplicated_index(points: gpd.GeoDataFrame) -> None:
    """Test if rows with the same index are not duplicated."""
    points["row"] = range(len(points))
    points.index = points.index % 1000
    train, test = train_test_spatial_split(
        points, H3_RESOLUTION, target_column="target", random_state=0, verbose=False
    )
    assert_partition([train["row"].to_numpy(), test["row"].to_numpy()], len(points))


@pytest.mark.parametrize("target", ["target", "target_cat"])  # type: ignore
def test_missing_target_values(points: gpd.GeoDataFrame, target: str) -> None:
    """Test if missing target values raise an error."""
    points.loc[:9, target] = None
    with pytest.raises(ValueError, match="10 missing values"):
        spatial_split(
            points, [0.5, 0.5], 6, target=target, categorical=target == "target_cat", verbose=False
        )


@pytest.mark.parametrize("value", [np.inf, -np.inf])  # type: ignore
@pytest.mark.parametrize("dtype", ["float64", "object", "Float64"])  # type: ignore
def test_infinite_target_values(points: gpd.GeoDataFrame, value: float, dtype: str) -> None:
    """Test if infinite target values raise an error."""
    points.loc[0, "target"] = value
    with pytest.raises(ValueError, match="infinite"):
        spatial_split(points, [0.5, 0.5], 6, target=points["target"].astype(dtype), verbose=False)


def test_invalid_values_in_nullable_types(points: gpd.GeoDataFrame) -> None:
    """Test if NaN values not reported as missing raise an error."""
    zeros = pd.Series([0.0] * 10 + [1.0] * (len(points) - 10), dtype="Float64")
    targets = [zeros / zeros]

    pyarrow = pytest.importorskip("pyarrow")
    values = pyarrow.array(points["target"].where(zeros == 1).to_numpy(), from_pandas=False)
    targets.append(pd.Series(values, dtype=pd.ArrowDtype(pyarrow.float64())))

    for target in targets:
        with pytest.raises(ValueError, match="10 missing"):
            spatial_split(points, [0.5, 0.5], 6, target=target, n_bins=3, verbose=False)

    with pytest.raises(ValueError, match="10 missing"):
        spatial_split(points, [0.5, 0.5], 6, target=targets[1], categorical=True, verbose=False)


def test_rows_are_never_lost() -> None:
    """Test if rows without a bucket raise an error."""
    with pytest.raises(ValueError, match="1 missing"):
        split_groups(
            groups=np.array([0, 0, 1, 1]),
            target=pd.Series([1.0, 2.0, 3.0, np.nan]),
            ratios={"a": 0.5, "b": 0.5},
            n_bins=2,
            categorical=False,
            random_state=0,
            verbose=False,
        )


def test_projected_points(points: gpd.GeoDataFrame) -> None:
    """Test if points are transformed to WGS84 before assigning H3 cells."""
    params: dict[str, Any] = dict(target="target", random_state=0, verbose=False)
    expected = spatial_split(points, [0.8, 0.2], H3_RESOLUTION, **params)
    splits = spatial_split(points.to_crs(3857), [0.8, 0.2], H3_RESOLUTION, **params)
    assert np.array_equal(splits[0], expected[0])


def test_geometry_array_crs(points: gpd.GeoDataFrame) -> None:
    """Test if CRS is read from the array of geometries."""
    params: dict[str, Any] = dict(regions=H3_RESOLUTION, random_state=0, verbose=False)
    expected = spatial_split(points, [0.8, 0.2], **params)
    assert are_equal(
        spatial_split(points.to_crs(3857).geometry.array, [0.8, 0.2], **params), expected
    )


def test_returned_data_can_be_modified(points: gpd.GeoDataFrame) -> None:
    """Test if returned data isn't a view of the input."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train, _ = train_test_spatial_split(points, H3_RESOLUTION, verbose=False)
        train["new"] = 1
    assert "new" not in points.columns


def test_projected_points_without_crs(points: gpd.GeoDataFrame) -> None:
    """Test if coordinates outside of the WGS84 bounds raise an error."""
    coordinates = points.to_crs(3857).geometry.to_list()
    with pytest.raises(ValueError, match="WGS84"):
        spatial_split(coordinates, [0.8, 0.2], H3_RESOLUTION, verbose=False)


def test_constant_target(points: gpd.GeoDataFrame) -> None:
    """Test if constant target doesn't break the split."""
    splits = spatial_split(points, [0.8, 0.2], 6, target=[1.0] * len(points), verbose=False)
    assert_partition(splits, len(points))


def test_single_point_per_cell() -> None:
    """Test if the same number of points per cell doesn't break the split."""
    train, test = train_test_spatial_split(
        [Point(i, 0) for i in range(100)], parent_h3_resolution=6, verbose=False
    )
    assert (len(train), len(test)) == (80, 20)


def test_categorical_dtype(points: gpd.GeoDataFrame) -> None:
    """Test if categorical dtype with unused categories doesn't break the split."""
    params: dict[str, Any] = dict(categorical=True, random_state=0, verbose=False)
    expected = spatial_split(points, [0.8, 0.2], 6, target="target_cat", **params)
    target = pd.Categorical(points["target_cat"], categories=["high", "low", "unused"])
    splits = spatial_split(points, [0.8, 0.2], 6, target=target, **params)
    assert np.array_equal(splits[0], expected[0])


def test_mixed_categories(points: gpd.GeoDataFrame) -> None:
    """Test if categories of different types can be used together."""
    target = [1 if value == "high" else value for value in points["target_cat"]]
    _, table = spatial_split_points(
        points, 6, target_column=target, categorical=True, return_split_stats=True, verbose=False
    )
    assert set(table["bucket"]) == {1, "low"}


@pytest.mark.parametrize("column", ["count", "h3", "bucket", "points", "split", "group"])  # type: ignore
@pytest.mark.parametrize("target", ["target", "target_cat"])  # type: ignore
def test_column_names_are_not_reserved(points: gpd.GeoDataFrame, column: str, target: str) -> None:
    """Test if any column name can be used for the target."""
    params: dict[str, Any] = dict(
        parent_h3_resolution=6, categorical=target == "target_cat", random_state=0, verbose=False
    )
    expected, _ = train_test_spatial_split(points, target_column=target, **params)
    train, _ = train_test_spatial_split(
        points.rename(columns={target: column}), target_column=column, **params
    )
    assert train.index.equals(expected.index)


def test_dominating_target_value(points: gpd.GeoDataFrame) -> None:
    """Test if a target with one dominating value is still stratified."""
    target = np.where(np.arange(len(points)) % 10 == 0, points["target"], 0)
    _, table = spatial_split_points(
        points, 6, target_column=target, return_split_stats=True, verbose=False
    )
    assert len(table) > 1
    assert table.filter(like="_points").sum(axis=1).min() >= 10


def test_number_of_points_stays_in_single_bucket() -> None:
    """Test if buckets of points per cell are the same as in the version 0.10.2."""
    points = [Point(i, 0) for i in range(100)] + [Point(0, 0)] * 5
    _, table = spatial_split_points(points, 6, return_split_stats=True, verbose=False)
    assert len(table) == 1


def test_other_geometry_columns(points: gpd.GeoDataFrame) -> None:
    """Test if active and additional geometry columns can be used."""
    params: dict[str, Any] = dict(parent_h3_resolution=6, random_state=0, verbose=False)
    expected, _ = train_test_spatial_split(points, **params)

    renamed = points.rename_geometry("geom")
    train, _ = train_test_spatial_split(renamed, **params)
    assert train.index.equals(expected.index)
    assert train.geometry.name == "geom"

    additional = points.assign(centroid=points.geometry)
    additional.geometry = additional.buffer(0.001)
    train, _ = train_test_spatial_split(additional, geometry_column="centroid", **params)
    assert train.index.equals(expected.index)

    with pytest.raises(KeyError):
        train_test_spatial_split(points, geometry_column="other", **params)


@pytest.mark.parametrize(
    "wrong_points,match",
    [
        ([], "empty"),
        ([Point(0, 0), None], "missing or empty"),
        ([Point(0, 0), Point()], "missing or empty"),
        ([Point(0, 0), box(0, 0, 1, 1)], "Only point geometries"),
    ],
)  # type: ignore
def test_wrong_points(wrong_points: list[Any], match: str) -> None:
    """Test if wrong points raise an error."""
    with pytest.raises(ValueError, match=match):
        spatial_split(wrong_points, [0.5, 0.5], 6, verbose=False)


def test_wrong_target(points: gpd.GeoDataFrame) -> None:
    """Test if wrong target raises an error."""
    params: dict[str, Any] = dict(ratios=[0.5, 0.5], regions=6, verbose=False)
    with pytest.raises(ValueError, match="length"):
        spatial_split(points, target=[1, 2, 3], **params)
    with pytest.raises(ValueError, match="GeoDataFrame"):
        spatial_split(points.geometry, target="target", **params)
    with pytest.raises(ValueError, match="categorical"):
        spatial_split(points, categorical=True, **params)
    with pytest.raises(ValueError, match="numerical"):
        spatial_split(points, target="target_cat", **params)
    with pytest.raises(KeyError):
        spatial_split(points, target="other", **params)


@pytest.mark.parametrize(
    "ratios", [[], [1.0], [0.5, 0.4], [0.5, 0.6], [1.0, 0.0], [1.5, -0.5], [0.5, float("nan")]]
)  # type: ignore
def test_wrong_ratios(points: gpd.GeoDataFrame, ratios: list[float]) -> None:
    """Test if wrong ratios raise an error."""
    with pytest.raises(ValueError, match="ratios"):
        spatial_split(points, ratios, 6, verbose=False)


@pytest.mark.parametrize("n_splits", [0, 1])  # type: ignore
def test_wrong_number_of_folds(points: gpd.GeoDataFrame, n_splits: int) -> None:
    """Test if wrong number of folds raises an error."""
    with pytest.raises(ValueError, match="n_splits"):
        spatial_kfold(points, n_splits, regions=6, verbose=False)


@pytest.mark.parametrize(
    "test_size,validation_size",
    [(-0.1, 0), (0.1, -0.1), (0.5, 0.5), (2000, 0), (1.0, 0), (0.1, 1.5)],
)  # type: ignore
def test_wrong_sizes(points: gpd.GeoDataFrame, test_size: float, validation_size: float) -> None:
    """Test if wrong sizes raise an error."""
    with pytest.raises(ValueError):
        spatial_split_points(
            points, 6, test_size=test_size, validation_size=validation_size, verbose=False
        )


def test_zero_test_size(points: gpd.GeoDataFrame) -> None:
    """Test if test split can be skipped only in the function with named splits."""
    splits = spatial_split_points(points, 6, test_size=0, verbose=False)
    assert splits["test"] is None
    assert splits["train"] is not None and len(splits["train"]) == len(points)

    with pytest.raises(ValueError, match="test_size must be greater than 0"):
        train_test_spatial_split(points, 6, test_size=0)


def test_not_enough_regions() -> None:
    """Test if a split without any region raises an error."""
    points = [Point(0, 0), Point(0.0001, 0)]
    with pytest.raises(ValueError, match="2 splits from 1 region"):
        train_test_spatial_split(points, parent_h3_resolution=3, verbose=False)
    with pytest.raises(ValueError, match="5 splits from 2 region"):
        spatial_kfold([*points, Point(10, 10)], 5, regions=3, verbose=False)


def test_splits_are_never_empty() -> None:
    """Test if small split gets a region when there are enough of them."""
    for seed in range(20):
        train, test = train_test_spatial_split(
            [Point(i, 0) for i in range(3)],
            parent_h3_resolution=6,
            test_size=0.01,
            random_state=seed,
            verbose=False,
        )
        assert (len(train), len(test)) == (2, 1)


def test_verbose_report(points: gpd.GeoDataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    """Test if report is printed only when requested."""
    spatial_kfold(points, 3, regions=6, verbose=False)
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")

    spatial_kfold(points, 3, regions=6)
    assert "Summary of the split" in capsys.readouterr().out
