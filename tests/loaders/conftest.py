"""Conftest for loaders."""

from datetime import datetime
from typing import Any

import pandas as pd
import pytest
from pytest_mock import MockerFixture


@pytest.fixture  # type: ignore
def gtfs_validation_ok() -> pd.DataFrame:
    """Get GTFS validation result with no errors."""
    return pd.DataFrame(
        {
            "type": ["warning", "warning"],
            "message": ["test warning", "test warning"],
            "table": ["test_table", "test_table"],
            "rows": [[1, 2], [3, 4]],
        }
    )


@pytest.fixture  # type: ignore
def gtfs_validation_error() -> pd.DataFrame:
    """Get GTFS validation result with errors."""
    return pd.DataFrame(
        {
            "type": ["error", "error", "warning", "warning"],
            "message": ["test error", "test error", "test warning", "test warning"],
            "table": ["test_table", "test_table", "test_table", "test_table"],
            "rows": [[1, 2], [3, 4], [5, 6], [7, 8]],
        }
    )


@pytest.fixture  # type: ignore
def stop_time_series() -> pd.DataFrame:
    """Get mocked stop time series."""
    ts = pd.DataFrame.from_dict(
        {
            "datetime": pd.DatetimeIndex(
                [datetime(2022, 1, 2, 12, 0), datetime(2022, 1, 2, 13, 0)]
            ),
            "42": pd.Series([0, 2]),
            "76": pd.Series([12, 12]),
        }
    ).set_index("datetime")

    ts.columns = pd.MultiIndex.from_tuples(
        [("num_trips", "42"), ("num_trips", "76")], names=["indicator", "stop_id"]
    )

    return ts


@pytest.fixture  # type: ignore
def stop_times() -> pd.DataFrame:
    """Get mocked stop times."""
    return pd.DataFrame(
        {
            "trip_id": ["1", "1", "2", "2"],
            "arrival_time": ["12:00:00", "13:00:00", "12:00:00", "13:00:00"],
            "departure_time": ["12:00:00", "12:00:00", "13:00:00", "13:00:00"],
            "stop_id": ["42", "76", "42", "76"],
        }
    )


@pytest.fixture  # type: ignore
def trips() -> pd.DataFrame:
    """Get mocked trips."""
    return pd.DataFrame(
        {
            "trip_id": ["1", "2"],
            "trip_headsign": ["A", "B"],
        }
    )


@pytest.fixture  # type: ignore
def stops() -> pd.DataFrame:
    """Get mocked stops."""
    return pd.DataFrame(
        {
            "stop_id": ["42", "76"],
            "stop_lat": [51.198083, 51.107133],
            "stop_lon": [16.905892, 17.019394],
        }
    )


@pytest.fixture  # type: ignore
def feed(
    mocker: MockerFixture,
    stop_time_series: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    stops: pd.DataFrame,
) -> Any:
    """Get mocked feed."""
    feed_mock = mocker.MagicMock()
    feed_mock.configure_mock(
        **{
            "get_first_week.return_value": ["", "", "20220102"],
            "compute_stop_time_series.return_value": stop_time_series,
            "stop_times": stop_times,
            "trips": trips,
            "stops": stops,
        }
    )
    return feed_mock
