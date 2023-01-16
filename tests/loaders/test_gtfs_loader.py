"""GTFS Loader tests."""
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from srai.loaders import GTFSLoader


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
