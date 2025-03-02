"""ParquetDataTable and GeoDataTable module."""

import hashlib
import inspect
import weakref
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from geoarrow.pyarrow.io import _geoparquet_guess_geometry_columns

from srai.constants import GEOMETRY_COLUMN
from srai.duckdb import prepare_duckdb_extensions

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

    def __init__(
        self,
        parquet_paths: Iterable[Path],
        index_column_names: Optional[Union[str, Iterable[str]]] = None,
    ):
        """
        Initialize ParquetDataTable.

        Args:
            parquet_paths (Iterable[Path]): List of parquet files.
            index_column_names (Optional[Union[str, Iterable[str]]]): Index column names.
        """
        prepare_duckdb_extensions()
        self.index_column_names = (
            (
                [index_column_names]
                if isinstance(index_column_names, str)
                else list(index_column_names)
            )
            if index_column_names is not None
            else None
        )
        self.parquet_paths = parquet_paths
        self._finalizer = weakref.finalize(self, self._cleanup_files, self.parquet_paths)

    @staticmethod
    def _cleanup_files(paths: Iterable[Path]) -> None:
        print("cleanup!")
        for path in paths:
            path.unlink(missing_ok=True)

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
        index_column_names: Optional[Union[str, Iterable[str]]] = None,
    ) -> _Self:
        """
        Create ParquetDataTable object from parquet files.

        Args:
            parquet_path (Union[Path, str, Iterable[Union[Path, str]]]): Path or list of parquet
                paths.
            index_column_names (Optional[Union[str, Iterable[str]]]): Index column name or names.

        Returns:
            ParquetDataTable: Created object.
        """
        if isinstance(parquet_path, (str, Path)):
            return cls(
                parquet_paths=[Path(parquet_path)],
                index_column_names=index_column_names,
            )

        return cls(
            parquet_paths=[Path(p) for p in parquet_path],
            index_column_names=index_column_names,
        )

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
            prefix_path = cls.generate_filename()
            h = hashlib.new("sha256")
            h.update(dataframe.values.tobytes())
            gdf_hash = h.hexdigest()
            parquet_path = cls.get_directory() / f"{prefix_path}_{gdf_hash}.parquet"

        Path(parquet_path).parent.mkdir(exist_ok=True, parents=True)
        dataframe.reset_index().to_parquet(parquet_path, index=False)

        return cls.from_parquet(
            parquet_path=parquet_path,
            index_column_names=dataframe.index.name or dataframe.index.names,
        )

    @classmethod
    def generate_filename(cls) -> str:
        """
        Generate a filepath based on caller module or class.

        Adds a current timestamp as a hash.

        Returns:
            str: Generated file name.
        """
        frame = cast("FrameType", inspect.currentframe())

        caller_name = None

        while caller_name is None or caller_name in (
            "ParquetDataTable",
            "GeoDataTable",
            "srai.geodatatable",
        ):
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

            caller_name = class_name or module_name

        timestr = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{caller_name}_{timestr}"

    def to_dataframe(self) -> pd.DataFrame:
        """Get DataFrame."""
        df = pd.read_parquet(self.parquet_paths)
        if self.index_column_names is not None:
            df.set_index(self.index_column_names, inplace=True)
        return df

    def to_duckdb(
        self, connection: Optional[duckdb.DuckDBPyConnection] = None
    ) -> duckdb.DuckDBPyRelation:
        """Get DuckDB relation."""
        paths = list(map(lambda x: f"'{x}'", self.parquet_paths))
        sql_query = f"SELECT * FROM read_parquet([{','.join(paths)}])"
        if connection is not None:
            return connection.sql(sql_query)

        return duckdb.sql(sql_query)

    @property
    def index_name(self) -> Optional[str]:
        """Get single index name."""
        if self.index_column_names is not None and len(self.index_column_names) == 1:
            return self.index_column_names[0]

        return None

    @property
    def index_names(self) -> Optional[list[str]]:
        """Get all index names."""
        return list(self.index_column_names) if self.index_column_names is not None else None

    @property
    def columns(self) -> list[str]:
        """Get available columns."""
        return list(ds.dataset(self.parquet_paths).schema.names)

    @property
    def empty(self) -> bool:
        """Check if data table is empty."""
        for parquet_path in self.parquet_paths:
            if pq.read_metadata(parquet_path).num_rows > 0:
                return False

        return True

    def drop_columns(
        self: _Self, columns: Union[str, Iterable[str]], missing_ok: bool = False
    ) -> _Self:
        """Drop columns from the data table in place."""
        if isinstance(columns, str):
            columns = [columns]

        existing_columns = self.columns
        missing_columns = set(columns).difference(existing_columns)
        print(f"{missing_columns=}")
        if missing_columns and not missing_ok:
            raise ValueError(f"Columns {missing_columns} are not present in the data table.")

        columns_to_drop = set(columns).intersection(existing_columns)
        print(f"{columns_to_drop=}")
        if not columns_to_drop:
            return self

        columns_to_keep = set(existing_columns).difference(columns)
        print(f"{columns_to_keep=}")
        print(f"{self.parquet_paths=}")

        new_parquet_paths = []
        for parquet_path in self.parquet_paths:
            prefix_path = self.generate_filename()
            relation = duckdb.read_parquet(str(parquet_path)).select(*columns_to_keep)

            h = hashlib.new("sha256")
            h.update(relation.sql_query().encode())
            relation_hash = h.hexdigest()

            new_parquet_path = self.get_directory() / f"{prefix_path}_{relation_hash}.parquet"
            relation.to_parquet(str(new_parquet_path))
            new_parquet_paths.append(new_parquet_path)

        print(f"{new_parquet_paths=}")

        if self.index_column_names is not None:
            new_index_column_names: Optional[list[str]] = list(
                set(self.index_column_names).intersection(columns_to_keep)
            )
            if not new_index_column_names:
                new_index_column_names = None

        return self.from_parquet(
            parquet_path=new_parquet_paths, index_column_names=new_index_column_names
        )

    def __repr__(self) -> str:
        """Create representation string."""
        content = f"{self.__class__.__name__}\n"
        content += f"    Parquet files: {self.parquet_paths}\n"
        content += f"    Index columns: {self.index_column_names}\n"
        content += self.to_duckdb().__repr__()

        return content


class GeoDataTable(ParquetDataTable):
    """
    GeoDataTable.

    A SRAI internal internal memory object for keeping geo data.

    It is a wrapper around parquet files with utility functions.
    """

    def __init__(
        self,
        parquet_paths: Iterable[Path],
        index_column_names: Optional[Union[str, Iterable[str]]] = None,
    ):
        """
        Initialize GeoDataTable.

        Args:
            parquet_paths (Iterable[Path]): List of parquet files.
            index_column_names (Optional[Union[str, Iterable[str]]]): Index column names.
        """
        super().__init__(parquet_paths, index_column_names)

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
            prefix_path = cls.generate_filename()
            h = hashlib.new("sha256")
            h.update(geodataframe.values.tobytes())
            gdf_hash = h.hexdigest()
            parquet_path = cls.get_directory() / f"{prefix_path}_{gdf_hash}.parquet"

        Path(parquet_path).parent.mkdir(exist_ok=True, parents=True)
        geodataframe.reset_index().to_parquet(parquet_path, index=False)

        return cls.from_parquet(
            parquet_path=parquet_path,
            index_column_names=geodataframe.index.name or geodataframe.index.names,
        )

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Get GeoDataFrame."""
        gdf = gpd.read_parquet(self.parquet_paths)
        if self.index_column_names is not None:
            gdf.set_index(self.index_column_names, inplace=True)
        return gdf


VALID_GEO_INPUT = Union[Path, str, Iterable[Union[Path, str]], gpd.GeoDataFrame, GeoDataTable]
VALID_DATA_INPUT = Union[Path, str, Iterable[Union[Path, str]], pd.DataFrame, ParquetDataTable]


def prepare_geo_input(data_input: VALID_GEO_INPUT) -> GeoDataTable:
    """Transform input to GeoDataTable."""
    if isinstance(data_input, GeoDataTable):
        return data_input
    elif isinstance(data_input, gpd.GeoDataFrame):
        return GeoDataTable.from_geodataframe(data_input)

    return GeoDataTable.from_parquet(data_input)


def prepare_data_input(data_input: VALID_DATA_INPUT) -> ParquetDataTable:
    """Transform input to ParquetDataTable."""
    if isinstance(data_input, ParquetDataTable):
        return data_input
    elif isinstance(data_input, pd.DataFrame):
        return ParquetDataTable.from_dataframe(data_input)

    return ParquetDataTable.from_parquet(data_input)
