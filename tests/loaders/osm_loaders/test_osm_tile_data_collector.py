from typing import List

import numpy as np
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
        # col = collectors.SavingDataCollector(PATH, FILE_TYPE)
        x, y = 1, 1
        expected = _get_expected_path(x, y)

        col.store(x, y, PIL.Image.fromarray(rng.integers(0, 256, size=(3, 3), dtype="uint8")))

        PIL.Image.Image.save.assert_called_once_with(expected)

    def test_returns_file_names(
        self, mocker: MockerFixture, col: collectors.SavingDataCollector
    ) -> None:
        """Check returned by collect file names."""
        _path_image_save(mocker)
        col = collectors.SavingDataCollector(PATH, FILE_TYPE)

        for x in range(2):
            for y in range(3):
                col.store(
                    x, y, PIL.Image.fromarray(rng.integers(0, 256, size=(3, 3), dtype="uint8"))
                )

        result = col.collect()

        assert len(result) == 3
        for y in range(3):
            res_y = result[y]
            assert len(res_y) == 2
            for x in range(2):
                assert _get_expected_path(x, y) == res_y[x]

    def test_should_handle_nones(self, col: collectors.SavingDataCollector) -> None:
        """Tests None values."""
        # col = collectors.SavingDataCollector()

        expected: List[List[PIL.Image]] = []
        for y in range(3):
            expected.append([])
            for x in range(2):
                col.store(x, y, None)

        result = col.collect()

        assert len(result) == 3
        for y in range(3):
            res_y = result[y]
            assert len(res_y) == 2
            for x in range(2):
                assert res_y[x] is None


def _get_expected_path(x: int, y: int) -> str:
    return f"{PATH}/{x}_{y}.{FILE_TYPE}"


def _path_image_save(mocker: MockerFixture) -> None:
    mocker.patch("PIL.Image.Image.save")


class TestInMemoryDataCollector:
    """Tests for class InMemoryDataCollector."""

    @pytest.fixture
    def col(self) -> collectors.InMemoryDataCollector:
        """Fixture for InMemoryDataCollector."""
        return collectors.InMemoryDataCollector()

    def test_should_collect(self, col: collectors.InMemoryDataCollector) -> None:
        """Tests values of collected images."""
        # col = collectors.InMemoryDataCollector()

        expected: List[List[PIL.Image]] = []
        for y in range(3):
            expected.append([])
            for x in range(2):
                img = PIL.Image.fromarray(rng.integers(0, 256, size=(3, 3), dtype="uint8"))
                expected[-1].append(img)
                col.store(x, y, img)

        result = col.collect()

        assert len(result) == 3
        for y in range(3):
            res_y = result[y]
            assert len(res_y) == 2
            for x in range(2):
                assert (np.array(expected[y][x]) == np.array(res_y[x])).all(), (
                    np.array(expected[y][x]),
                    np.array(res_y[x]),
                )

    def test_should_handle_nones(self) -> None:
        """Tests None values."""
        col = collectors.InMemoryDataCollector()

        expected: List[List[PIL.Image]] = []
        for y in range(3):
            expected.append([])
            for x in range(2):
                col.store(x, y, None)

        result = col.collect()

        assert len(result) == 3
        for y in range(3):
            res_y = result[y]
            assert len(res_y) == 2
            for x in range(2):
                assert res_y[x] is None
