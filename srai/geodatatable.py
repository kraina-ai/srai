"""ParquetDataTable and GeoDataTable module."""

import hashlib
import inspect
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast

import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds
from geoarrow.pyarrow.io import _geoparquet_guess_geometry_columns

from srai.constants import GEOMETRY_COLUMN

if TYPE_CHECKING:
    from types import FrameType

from typing import TypeVar

_Self = TypeVar("_Self", bound="ParquetDataTable")


class ParquetDataTable:
    """
    ParquetDataTable.

    A SRAI internal internal memory object for keeping data.

    It is a wrapper around parquet files with utility functions.
    """

    FILES_DIRECTORY = Path("files")

    def __init__(self, parquet_paths: Iterable[Path], index_column_name: Optional[str] = None):
        """
        Initialize ParquetDataTable.

        Args:
            parquet_paths (Iterable[Path]): List of parquet files.
            index_column_name (str): Index column name.
        """
        self.index_column_name = index_column_name
        self.parquet_paths = parquet_paths

    @classmethod
    def get_directory(cls) -> Path:
        """Get saving directory."""
        return Path(cls.FILES_DIRECTORY)

    @classmethod
    def set_directory(cls, path: Union[Path, str]) -> None:
        """Set saving directory."""
        cls.FILES_DIRECTORY = Path(path)

    @classmethod
    def from_parquet(
        cls: type[_Self],
        parquet_path: Union[Path, str, Iterable[Union[Path, str]]],
        index_column_name: Optional[str] = None,
    ) -> _Self:
        """
        Create ParquetDataTable object from parquet files.

        Args:
            parquet_path (Union[Path, str, Iterable[Union[Path, str]]]): Path or list of parquet
                paths.
            index_column_name (str): Index column name.

        Returns:
            ParquetDataTable: Created object.
        """
        if isinstance(parquet_path, Iterable):
            return cls(
                parquet_paths=[Path(p) for p in parquet_path],
                index_column_name=index_column_name,
            )
        return cls(parquet_paths=[Path(parquet_path)], index_column_name=index_column_name)

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        parquet_path: Optional[Union[Path, str]] = None,
    ) -> "ParquetDataTable":
        """
        Create ParquetDataTable object from DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame to save.
            parquet_path (Optional[Union[Path, str]], optional): Path where to save the parquet
                file. Defaults to None.

        Returns:
            ParquetDataTable: Created object.
        """
        if not parquet_path:
            prefix_path = cls.generate_filepath(skip_frames=1)
            h = hashlib.new("sha256")
            h.update(dataframe.values.tobytes())
            gdf_hash = h.hexdigest()
            parquet_path = cls.get_directory() / f"{prefix_path}_{gdf_hash}.parquet"

        Path(parquet_path).parent.mkdir(exist_ok=True, parents=True)
        dataframe.to_parquet(parquet_path, index=True)

        return cls.from_parquet(parquet_path=parquet_path, index_column_name=dataframe.index.name)

    @classmethod
    def generate_filepath(cls, skip_frames: int = 0) -> str:
        """
        Generate a filepath based on caller module or class.

        Adds a current timestamp as a hash.

        Returns:
            str: _description_
        """
        frame = cast("FrameType", inspect.currentframe())
        for _ in range(skip_frames + 1):
            if frame.f_back is None:
                break
            frame = frame.f_back

        class_name = None
        module_name = None

        if "self" in frame.f_locals:
            class_name = frame.f_locals["self"].__class__.__name__
        elif "cls" in frame.f_locals:
            class_name = frame.f_locals["cls"].__name__
        else:
            module_name = frame.f_globals["__name__"]

        caller_name = class_name if class_name else module_name

        timestr = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{caller_name}_{timestr}"

    # def to_geodataframe(self):
    #     return gpd.read_parquet(self.parquet_paths)

    # def to_duckdb(self):
    #     return duckdb.read_parquet(self.parquet_paths)


class GeoDataTable(ParquetDataTable):
    """
    GeoDataTable.

    A SRAI internal internal memory object for keeping geo data.

    It is a wrapper around parquet files with utility functions.
    """

    def __init__(self, parquet_paths: Iterable[Path], index_column_name: Optional[str] = None):
        """
        Initialize GeoDataTable.

        Args:
            parquet_paths (Iterable[Path]): List of parquet files.
            index_column_name (str): Index column name.
        """
        super().__init__(parquet_paths, index_column_name)

        schema = ds.dataset(parquet_paths).schema
        geometry_columns = _geoparquet_guess_geometry_columns(schema)

        if GEOMETRY_COLUMN not in geometry_columns:
            raise ValueError(
                "Missing geometry column in the parquet file."
                f" Expecting geoarrow compatible column named '{GEOMETRY_COLUMN}'."
            )

    @classmethod
    def from_geodataframe(
        cls,
        geodataframe: gpd.GeoDataFrame,
        parquet_path: Optional[Union[Path, str]] = None,
    ) -> "GeoDataTable":
        """
        Create GeoDataTable object from GeoDataFrame.

        Args:
            geodataframe (gpd.GeoDataFrame): GeoDataFrame to save.
            parquet_path (Optional[Union[Path, str]], optional): Path where to save the parquet
                file. Defaults to None.

        Returns:
            GeoDataTable: Created object.
        """
        if not parquet_path:
            h = hashlib.new("sha256")
            h.update(geodataframe.values.tobytes())
            gdf_hash = h.hexdigest()
            parquet_path = cls.get_directory() / gdf_hash

        Path(parquet_path).parent.mkdir(exist_ok=True, parents=True)
        geodataframe.to_parquet(parquet_path, index=True)

        return cls.from_parquet(
            parquet_path=parquet_path, index_column_name=geodataframe.index.name
        )
