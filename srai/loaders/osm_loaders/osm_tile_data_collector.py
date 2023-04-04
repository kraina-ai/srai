"""This module contains classes of strategy for handling downloaded tiles."""
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Union

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

    def __init__(self, save_path: Union[str, Path], file_extension: str) -> None:
        """
        Initialize SavingDataCollector.

        Args:
            save_path (Union[str, Path]): root path for data
            file_extension (str): file name extension
        """
        super().__init__()
        if save_path is None or file_extension is None:
            raise ValueError
        self.save_path = Path(save_path)
        self.format = file_extension

    def store(self, x: int, y: int, data: Image.Image) -> Path:
        """
        Saves image on disk. Returns path.

        Args:
            x (int): x coordinate of tile, used as id
            y (int): y coordinate of tile, used as id
            data (Image.Image): tile
        """
        path = self.save_path / f"{x}_{y}.{self.format}"
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


class DataCollectorType(str, Enum):
    """Defines enums to choose one of known DataCollector implementations."""

    SAVE = "save"
    RETURN = "return"


def get_collector(collector_type: Union[DataCollectorType, str], **kwargs: Any) -> DataCollector:
    """
    Returns DataCollector object of type specified by DataCollectorType enum.

    Args:
        collector_type (DataCollectorType): If SAVE returns SavingDataCollector.
        If RETURN returns InMemoryDataCollector.
        **kwargs (Any): Extra arguments used for SavingDataCollector object creation arguments.

    Returns:
        DataCollector: newly created object
    """
    if collector_type == DataCollectorType.SAVE:
        return SavingDataCollector(**kwargs)
    elif collector_type == DataCollectorType.RETURN:
        return InMemoryDataCollector()
    else:
        raise ValueError
