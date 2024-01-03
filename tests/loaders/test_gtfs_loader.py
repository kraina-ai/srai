"""GTFS Loader tests."""

from pathlib import Path
from typing import Any
from unittest import TestCase

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from srai.constants import GEOMETRY_COLUMN
from srai.loaders import GTFSLoader
from srai.loaders.gtfs_loader import GTFS2VEC_DIRECTIONS_PREFIX, GTFS2VEC_TRIPS_PREFIX

ut = TestCase()


def test_validation_ok(mocker: MockerFixture, gtfs_validation_ok: pd.DataFrame) -> None:
    """Test checks if GTFSLoader returns no errors."""
    feed_mock = mocker.MagicMock()
    feed_mock.configure_mock(**{"validate.return_value": gtfs_validation_ok})

    loader = GTFSLoader()
    loader._validate_feed(feed_mock)


def test_validation_error(mocker: MockerFixture, gtfs_validation_error: pd.DataFrame) -> None:
    """Test checks if GTFSLoader raises ValueError on validation error."""
    feed_mock = mocker.MagicMock()
    feed_mock.configure_mock(**{"validate.return_value": gtfs_validation_error})

    warning_mock = mocker.patch("warnings.warn")

    loader = GTFSLoader()
    with pytest.raises(ValueError):
        loader._validate_feed(feed_mock)
        warning_mock.assert_called_once()


def test_validation_warning(mocker: MockerFixture, gtfs_validation_error: pd.DataFrame) -> None:
    """Test checks if GTFSLoader raises ValueError on validation error."""
    feed_mock = mocker.MagicMock()
    feed_mock.configure_mock(**{"validate.return_value": gtfs_validation_error})

    warning_mock = mocker.patch("warnings.warn")

    loader = GTFSLoader()
    loader._validate_feed(feed_mock, fail=False)
    warning_mock.assert_called_once()


def test_gtfs_loader(feed: Any, mocker: MockerFixture, gtfs_validation_ok: pd.DataFrame) -> None:
    """Test GTFSLoader."""
    feed.validate.return_value = gtfs_validation_ok
    mocker.patch("gtfs_kit.read_feed", return_value=feed)

    loader = GTFSLoader()
    features = loader.load(Path("feed.zip").resolve())

    ut.assertCountEqual(features.index, ["42", "76"])
    ut.assertCountEqual(
        features.columns,
        [
            f"{GTFS2VEC_TRIPS_PREFIX}12",
            f"{GTFS2VEC_TRIPS_PREFIX}13",
            f"{GTFS2VEC_DIRECTIONS_PREFIX}12",
            f"{GTFS2VEC_DIRECTIONS_PREFIX}13",
            GEOMETRY_COLUMN,
        ],
    )


def test_gtfs_loader_with_invalid_feed(
    feed: Any, mocker: MockerFixture, gtfs_validation_error: pd.DataFrame
) -> None:
    """Test GTFSLoader with invalid feed."""
    feed.validate.return_value = gtfs_validation_error
    mocker.patch("gtfs_kit.read_feed", return_value=feed)
    warning_mock = mocker.patch("warnings.warn")

    loader = GTFSLoader()
    with pytest.raises(ValueError):
        loader.load(Path("feed.zip").resolve())
        warning_mock.assert_called_once()


def test_gtfs_loader_skip_validation(
    feed: Any, mocker: MockerFixture, gtfs_validation_ok: pd.DataFrame
) -> None:
    """Test GTFSLoader with invalid feed."""
    feed.validate.return_value = gtfs_validation_ok
    mocker.patch("gtfs_kit.read_feed", return_value=feed)

    loader = GTFSLoader()
    loader.load(Path("feed.zip").resolve(), skip_validation=True)

    feed.validate.assert_not_called()
