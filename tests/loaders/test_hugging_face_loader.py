import pandas as pd
import pyarrow as pa
import pytest
from datasets import Dataset, DatasetDict
from pytest_mock import MockerFixture

from srai.loaders import HuggingFaceLoader


@pytest.fixture()  # type: ignore
def dataset_dict() -> DatasetDict:
    metric = pa.array([200, 4, 50, 100])
    cities = pa.array(["Wrocław", "Berlin", "Gdańsk", "Amsterdam"])
    names = ["f1", "f2"]
    table = pa.Table.from_arrays([metric, cities], names=names)

    return DatasetDict(train=Dataset(table))


def test_loader(mocker: MockerFixture, dataset_dict: DatasetDict) -> None:
    mocker.patch("datasets.load_dataset").return_value = dataset_dict

    loader = HuggingFaceLoader()
    df = loader.load("test/dataset")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
