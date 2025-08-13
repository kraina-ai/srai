"""This module contains classes of strategy for handling downloaded tiles."""

from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image


class DataCollector(ABC):
    """Store collected images."""

    @abstractmethod
    def store(self, idx: str, data: "Image.Image") -> Any:
        """
        Apply action for object storage and returns data of it.

        Args:
            idx (str): id of tile
            data (Image.Image): tile
        """


class SavingDataCollector(DataCollector):
    """
    Save in disk collected images.

    Store paths.
    """

    def __init__(self, save_path: Union[str, Path], file_extension: str, **kwargs: Any) -> None:  # noqa: D417
        """
        Initialize SavingDataCollector.

        Args:
            save_path (Union[str, Path]): Root path for saving data.
            file_extension (str): File name extension.
        """
        super().__init__()
        if save_path is None or file_extension is None:
            raise ValueError
        self.save_path = Path(save_path)
        self.format = file_extension

    def store(self, idx: str, data: "Image.Image") -> str:
        """
        Save image on disk. Returns path.

        Args:
            idx (str): id of tile
            data (Image.Image): tile
        """
        path = self.save_path / f"{idx}.{self.format}"
        path.parent.mkdir(exist_ok=True, parents=True)
        data.save(path)
        return str(path.resolve())


class InMemoryDataCollector(DataCollector):
    """Store data in object memory."""

    def __init__(self, file_extension: str, **kwargs: Any) -> None:  # noqa: D417
        """
        Initialize InMemoryDataCollector.

        Args:
            file_extension (str): File name extension. Used to serialize
                images to bytes.
        """
        super().__init__()
        if file_extension is None:
            raise ValueError
        self.format = file_extension

    def store(self, idx: str, data: "Image.Image") -> bytes:
        """
        Simply return object for usage.

        Args:
            idx (str): id of tile
            data (Image.Image): tile
        """
        img_bytes = BytesIO()
        data.save(img_bytes, self.format)
        return img_bytes.getvalue()


class DataCollectorType(str, Enum):
    """Define enums to choose one of known DataCollector implementations."""

    SAVE = "save"
    RETURN = "return"


def get_collector(collector_type: Union[DataCollectorType, str], **kwargs: Any) -> DataCollector:
    """
    Return DataCollector object of type specified by DataCollectorType enum.

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
        return InMemoryDataCollector(**kwargs)
    else:
        raise ValueError
