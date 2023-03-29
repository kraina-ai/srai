import os

import PIL
import pytest
from numpy.random import default_rng
from pytest_mock import MockerFixture

import srai.loaders.osm_loaders.osm_tile_data_collector as collectors

rng = default_rng()
PATH = "path"
FILE_TYPE = "png"


class TestSavingDataCollector:
    """Tests for class SavingDataCollector."""

    @pytest.fixture
    def col(self) -> collectors.SavingDataCollector:
        """Fixture for SavingDataCollector."""
        return collectors.SavingDataCollector(PATH, FILE_TYPE)

    def test_save_to_disk(self, mocker: MockerFixture, col: collectors.SavingDataCollector) -> None:
        """Tests if save to disk saves image properly."""
        _path_image_save(mocker)
        x, y = 1, 1
        expected = _get_expected_path(x, y)

        path = col.store(
            x, y, PIL.Image.fromarray(rng.integers(0, 256, size=(3, 3), dtype="uint8"))
        )

        PIL.Image.Image.save.assert_called_once_with(expected)
        assert path == _get_expected_path(x, y)


def _get_expected_path(x: int, y: int) -> str:
    return os.path.join(PATH, f"{x}_{y}.{FILE_TYPE}")


def _path_image_save(mocker: MockerFixture) -> None:
    mocker.patch("PIL.Image.Image.save")


class TestInMemoryDataCollector:
    """Tests for class InMemoryDataCollector."""

    @pytest.fixture
    def col(self) -> collectors.InMemoryDataCollector:
        """Fixture for InMemoryDataCollector."""
        return collectors.InMemoryDataCollector()

    def test_should_return_stored(self, col: collectors.InMemoryDataCollector) -> None:
        """Tests values of collected images."""
        x, y = 1, 1
        img = PIL.Image.fromarray(rng.integers(0, 256, size=(3, 3), dtype="uint8"))

        stored = col.store(x, y, img)

        assert stored == img
