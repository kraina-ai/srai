import pandas as pd
import pytest
from pytest_mock import MockerFixture

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.datasets import AirbnbMulticityDataset
from srai.loaders import HuggingFaceLoader


@pytest.fixture()  # type: ignore
def loaded_df() -> pd.DataFrame:
    return pd.DataFrame({"f1": ["a", "b"], "latitude": [10, 15], "longitude": [20, 30]})


def test_dataset(mocker: MockerFixture, loaded_df: pd.DataFrame) -> None:
    mocker.patch.object(HuggingFaceLoader, "load").return_value = loaded_df

    dataset = AirbnbMulticityDataset()
    gdf = dataset.load()
    assert not gdf.empty
    assert GEOMETRY_COLUMN in gdf
    assert gdf.crs == WGS84_CRS
