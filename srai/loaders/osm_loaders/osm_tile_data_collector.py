import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from functional import seq
from PIL import Image


class DataCollector(ABC):
    """Stores collected images."""

    @abstractmethod
    def store(self, x: int, y: int, data: Image.Image) -> None:
        """
        Collect and save data in object.

        Args:
            x (int): x coordinate of tile, used as id
            y (int): y coordinate of tile, used as id
            data (Image.Image): tile
        """

    @abstractmethod
    def collect(self) -> List[List[Any]]:
        """
        Returns collected data.

        Returns:
            List[Any]: collected data. Might be images itself or paths
        """


class SavingDataCollector(DataCollector):
    """Saves in disk collected images. Stores paths."""

    def __init__(self, save_path: Union[str, Path], f_extension: str) -> None:
        """
        Initialize SavingDataCollector.

        Args:
            save_path (Union[str, Path]): root path for data
            f_extension (str): file name extension
        """
        super().__init__()
        self.save_path = save_path
        self.format = f_extension
        self.data: Dict[int, List[Optional[str]]] = {}

    def store(self, x: int, y: int, data: Image.Image) -> None:
        """
        Saves image on disk. Keeps path in object memory.

        Args:
            x (int): x coordinate of tile, used as id
            y (int): y coordinate of tile, used as id
            data (Image.Image): tile
        """
        if data is not None:
            path = os.path.join(self.save_path, f"{x}_{y}.png")
            data.save(path)
        else:
            path = None
        if y not in self.data:
            self.data[y] = []
        self.data[y].append(path)

    def collect(self) -> List[List[Optional[str]]]:
        """
        Returns paths of saved images.

        Returns:
            List[List[str]]: 2D list. First dimension is row (by y coordinate),
            second column (by order of store calls)
        """
        return seq(self.data).sorted().map(lambda key: self.data[key]).to_list()


class InMemoryDataCollector(DataCollector):
    """Stores data in object memory."""

    def __init__(self) -> None:
        """Initialize InMemoryDataCollector."""
        super().__init__()
        self.data: Dict[int, List[Image.Image]] = {}

    def store(self, x: int, y: int, data: Image.Image) -> None:
        """
        Keeps image in object state.

        Args:
            x (int): x coordinate of tile, used as id
            y (int): y coordinate of tile, used as id
            data (Image.Image): tile
        """
        if y not in self.data:
            self.data[y] = []
        self.data[y].append(data)

    def collect(self) -> List[List[Image.Image]]:
        """
        Returns Images.

        Returns:
            List[List[Image.Image]]: 2D list. First dimension is row (by y coordinate),
            second column (by order of store calls)
        """
        return seq(self.data).sorted().map(lambda key: self.data[key]).to_list()
