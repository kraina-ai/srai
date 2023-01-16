"""Conftest for loaders."""

import pandas as pd
import pytest


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
