"""Test if datasets are loaded properly."""

import inspect
import os
import sys

import pytest

import srai.datasets

RUN_DATASET_TESTS = os.getenv("RUN_DATASET_TESTS", "true").lower() == "true"

dataset_classes = [
    dataset_cls
    for dataset_cls_name, dataset_cls in inspect.getmembers(
        sys.modules[srai.datasets.__name__], inspect.isclass
    )
    if dataset_cls_name not in ("HuggingFaceDataset", "PointDataset", "TrajectoryDataset")
]


@pytest.mark.skipif(not RUN_DATASET_TESTS, reason="Skipping dataset tests")  # type: ignore
@pytest.mark.parametrize("dataset_class", dataset_classes)  # type: ignore
def test_load_dataset(dataset_class: type[srai.datasets.HuggingFaceDataset]) -> None:
    """Test if dataset is loaded properly."""
    dataset_obj = dataset_class()  # type: ignore[call-arg]

    # run default load
    dataset_obj.load()

    assert dataset_obj.train_gdf is not None, "Train gdf is empty"
    assert dataset_obj.test_gdf is not None, "Test gdf is empty"
    assert len(dataset_obj.train_gdf) > len(dataset_obj.test_gdf), "Test split bigger than train"

    # run additional split
    dataset_obj.train_test_split(validation_split=True)

    assert dataset_obj.train_gdf is not None, "Train gdf is empty"
    assert dataset_obj.test_gdf is not None, "Test gdf is empty"
    assert dataset_obj.val_gdf is not None, "Validation gdf is empty"
    assert len(dataset_obj.train_gdf) > len(dataset_obj.val_gdf), (
        "Validation split bigger than train"
    )
