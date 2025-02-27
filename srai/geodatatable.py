"""
_summary_

Returns:
    _type_: _description_
"""

import hashlib
import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd


class GeoDataTable:
    """
    _summary_

    Returns:
        _type_: _description_
    """

    FILES_DIRECTORY = Path("files")

    def __init__(self, parquet_paths: Iterable[Path]):
        """
        _summary_

        Args:
            parquet_paths (Iterable[Path]): _description_
        """
        self.parquet_paths = parquet_paths

    @classmethod
    def get_directory(cls) -> Path:
        """
        _summary_

        Returns:
            Path: _description_
        """
        return Path(cls.FILES_DIRECTORY)

    @classmethod
    def set_directory(cls, path: Union[Path, str]) -> None:
        """
        _summary_

        Args:
            path (Union[Path, str]): _description_
        """
        cls.FILES_DIRECTORY = Path(path)

    @classmethod
    def from_parquet(
        cls, parquet_path: Union[Path, str, Iterable[Union[Path, str]]]
    ) -> "GeoDataTable":
        """
        _summary_

        Args:
            parquet_path (Union[Path, str, Iterable[Union[Path, str]]]): _description_

        Returns:
            GeoDataTable: _description_
        """
        if isinstance(parquet_path, Iterable):
            return cls(parquet_paths=[Path(p) for p in parquet_path])
        return cls(parquet_paths=[Path(parquet_path)])

    @classmethod
    def from_geodataframe(
        cls, geodataframe: gpd.GeoDataFrame, parquet_path: Optional[Union[Path, str]] = None
    ) -> "GeoDataTable":
        """
        _summary_

        Args:
            geodataframe (gpd.GeoDataFrame): _description_
            parquet_path (Optional[Union[Path, str]], optional): _description_. Defaults to None.

        Returns:
            GeoDataTable: _description_
        """
        if not parquet_path:
            h = hashlib.new("sha256")
            h.update(geodataframe.values.tobytes())
            gdf_hash = h.hexdigest()
            parquet_path = cls.get_directory() / gdf_hash

        Path(parquet_path).parent.mkdir(exist_ok=True, parents=True)
        geodataframe.to_parquet(parquet_path)

        return cls.from_parquet(parquet_path=parquet_path)

    @classmethod
    def generate_filepath(cls, caller: Any) -> str:
        """
        _summary_

        Args:
            caller (Any): _description_

        Returns:
            str: _description_
        """
        is_class = inspect.isclass(caller)
        # TODO: get name / class and generate filepath with current timestamp
        return ""

    # def to_geodataframe(self):
    #     return gpd.read_parquet(self.parquet_paths)

    # def to_duckdb(self):
    #     return duckdb.read_parquet(self.parquet_paths)

    #  @classmethod
    # def from_pandas(cls, df, Schema schema=None, preserve_index=None,
    #                 nthreads=None, columns=None):

    # @classmethod
    # def from_geodataframe():
    #     pass
