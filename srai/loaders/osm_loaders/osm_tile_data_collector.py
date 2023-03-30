import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


class DataCollector(ABC):
    """Stores collected images."""

    @abstractmethod
    def store(self, x: int, y: int, data: Image.Image) -> Any:
        """
        Apply action for object storage and returns data of it.

        Args:
            x (int): x coordinate of tile, used as id
            y (int): y coordinate of tile, used as id
            data (Image.Image): tile
        """


class SavingDataCollector(DataCollector):
    """
    Saves in disk collected images.

    Stores paths.
    """

    def __init__(self, save_path: str | Path, f_extension: str) -> None:
        """
        Initialize SavingDataCollector.

        Args:
            save_path (Union[str, Path]): root path for data
            f_extension (str): file name extension
        """
        super().__init__()
        self.save_path = save_path
        self.format = f_extension

    def store(self, x: int, y: int, data: Image.Image) -> str:
        """
        Saves image on disk. Returns path.

        Args:
            x (int): x coordinate of tile, used as id
            y (int): y coordinate of tile, used as id
            data (Image.Image): tile
        """
        path = os.path.join(self.save_path, f"{x}_{y}.png")
        data.save(path)
        return path


class InMemoryDataCollector(DataCollector):
    """Stores data in object memory."""

    def __init__(self) -> None:
        """Initialize InMemoryDataCollector."""
        super().__init__()

    def store(self, x: int, y: int, data: Image.Image) -> Image.Image:
        """
        Simply removes object for usage.

        Args:
            x (int): x coordinate of tile, used as id
            y (int): y coordinate of tile, used as id
            data (Image.Image): tile
        """
        return data
