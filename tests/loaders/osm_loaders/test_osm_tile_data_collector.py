"""Tests for DataCollector subclasses."""

import os
from pathlib import Path
from typing import Union

import PIL
import pytest
from numpy.random import default_rng
from pytest_mock import MockerFixture

import srai.loaders.osm_loaders.osm_tile_data_collector as collectors

rng = default_rng()
PATH = "path"
FILE_TYPE = "png"


def create_id(x: int, y: int) -> str:
    """Create test id."""
    return f"{x}_{y}_ZOOM"


class TestSavingDataCollector:
    """Tests for class SavingDataCollector."""

    @pytest.fixture  # type: ignore
    def col(self) -> collectors.SavingDataCollector:
        """Fixture for SavingDataCollector."""
        return collectors.SavingDataCollector(PATH, FILE_TYPE)

    def test_save_to_disk(self, mocker: MockerFixture, col: collectors.SavingDataCollector) -> None:
        """Test if save to disk saves image properly."""
        _path_image_save(mocker)
        x, y = 1, 1
        expected = _get_expected_path(x, y)

        path = col.store(
            create_id(x, y), PIL.Image.fromarray(rng.integers(0, 256, size=(3, 3), dtype="uint8"))
        )

        PIL.Image.Image.save.assert_called_once_with(expected)
        assert _get_expected_path(x, y) == path


def _get_expected_path(x: int, y: int) -> Path:
    return Path(os.path.join(PATH, f"{create_id(x,y)}.{FILE_TYPE}"))


def _path_image_save(mocker: MockerFixture) -> None:
    mocker.patch("PIL.Image.Image.save")


class TestInMemoryDataCollector:
    """Test for class InMemoryDataCollector."""

    @pytest.fixture  # type: ignore
    def col(self) -> collectors.InMemoryDataCollector:
        """Fixture for InMemoryDataCollector."""
        return collectors.InMemoryDataCollector()

    def test_should_return_stored(self, col: collectors.InMemoryDataCollector) -> None:
        """Test values of collected images."""
        x, y = 1, 1
        img = PIL.Image.fromarray(rng.integers(0, 256, size=(3, 3), dtype="uint8"))

        stored = col.store(create_id(x, y), img)

        assert stored == img


@pytest.mark.parametrize(  # type: ignore
    "collector_type", [(collectors.DataCollectorType.RETURN), "return"]
)
def test_in_memory_collector_creation(
    collector_type: Union[str, collectors.DataCollectorType],
) -> None:
    """Test if factory creates properly InMemoryDataCollector."""
    created = collectors.get_collector(collector_type)

    assert isinstance(created, collectors.InMemoryDataCollector)


@pytest.mark.parametrize(  # type: ignore
    "collector_type", [(collectors.DataCollectorType.SAVE), "save"]
)
def test_saving_collector_creation(
    collector_type: Union[str, collectors.DataCollectorType],
) -> None:
    """Test if factory creates properly SavingDataCollector."""
    created = collectors.get_collector(collector_type, save_path=PATH, file_extension=FILE_TYPE)

    assert isinstance(created, collectors.SavingDataCollector), f"Invalid type {type(created)}"
    assert created.format == FILE_TYPE
    assert str(created.save_path) == PATH


def test_invalid_type() -> None:
    """Tests if throws on unknown type."""
    with pytest.raises(ValueError):
        collectors.get_collector("Some weird type")
