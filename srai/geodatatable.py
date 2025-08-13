"""ParquetDataTable and GeoDataTable module."""

import hashlib
import inspect
import multiprocessing
import weakref
from collections.abc import Callable, Iterable, Sized
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union, cast, override

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from geoarrow.pyarrow.io import _geoparquet_guess_geometry_columns
from geoarrow.rust.core import GeoArray
from psutil._common import bytes2human
from rq_geo_toolkit.constants import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)
from rq_geo_toolkit.geoparquet_sorting import sort_geoparquet_file_by_geometry
from shapely import coverage_union_all, union_all
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.constants import GEOMETRY_COLUMN
from srai.duckdb import prepare_duckdb_extensions

if TYPE_CHECKING:
    from types import FrameType

from typing import TypeVar

_Self = TypeVar("_Self", bound="ParquetDataTable")


class ParquetDataTable(Sized):
    """
    ParquetDataTable.

    A SRAI internal internal memory object for keeping data.

    It is a wrapper around parquet files with utility functions.
    """

    FILES_DIRECTORY = Path("files/pdt")

    def __init__(
        self,
        parquet_paths: Iterable[Path],
        index_column_names: Optional[Union[str, Iterable[str]]] = None,
        persist_files: bool = False,
    ):
        """
        Initialize ParquetDataTable.

        Args:
            parquet_paths (Iterable[Path]): List of parquet files.
            index_column_names (Optional[Union[str, Iterable[str]]], optional): Index column names.
            persist_files (bool, optional): Whether to keep the files after object removal or
                delete them from disk. Defaults to False.
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
        self.persist_files = persist_files
        self._finalizer: Optional[weakref.finalize[[Iterable[Path]], ParquetDataTable]] = None

        if not self.persist_files:
            self._configure_finalizer()

    @property
    def index_name(self) -> Optional[str]:
        """Get single index name."""
        if self.index_column_names is not None and len(self.index_column_names) == 1:
            return self.index_column_names[0]

        return None

    @property
    def index_names(self) -> Optional[list[str]]:
        """Get all index names."""
        return self.index_column_names

    @property
    def columns(self) -> list[str]:
        """Get available columns."""
        return [
            column_name
            for column_name in ds.dataset(self.parquet_paths).schema.names
            if column_name not in (self.index_column_names or [])
        ]

    @property
    def physical_columns(self) -> list[str]:
        """Get available columns."""
        return [column_name for column_name in ds.dataset(self.parquet_paths).schema.names]

    @property
    def size(self) -> int:
        """Total size in bytes."""
        return sum(parquet_path.stat().st_size for parquet_path in self.parquet_paths)

    @property
    def rows(self) -> int:
        """Number of rows."""
        return sum(pq.read_metadata(parquet_path).num_rows for parquet_path in self.parquet_paths)

    @property
    def empty(self) -> bool:
        """Check if data table is empty."""
        for parquet_path in self.parquet_paths:
            if pq.read_metadata(parquet_path).num_rows > 0:
                return False

        return True

    def __len__(self) -> int:
        """Alias for number of rows."""
        return self.rows

    def __repr__(self) -> str:
        """Create representation string."""
        content = f"{self.__class__.__name__} ({self.rows} rows)\n"
        content += f"  Parquet files ({bytes2human(self.size)}):\n"
        for path in self.parquet_paths:
            content += f"    {path.as_posix()} ({bytes2human(path.stat().st_size)})\n"
        content += "  Index columns:\n"
        for index_column in self.index_column_names or []:
            content += f"    {index_column}\n"
        try:
            duckdb_output = self.to_duckdb().__repr__()
            content += duckdb_output
        except Exception as ex:
            content += str(ex)

        return content

    @staticmethod
    def _cleanup_files(paths: Iterable[Path]) -> None:
        print(f"cleanup! ({paths})")
        for path in paths:
            path.unlink(missing_ok=True)

    def _detach_finalizer(self) -> None:
        if self._finalizer is not None:
            self._finalizer.detach()

    def _configure_finalizer(self) -> None:
        self._detach_finalizer()
        self._finalizer = weakref.finalize(self, self._cleanup_files, self.parquet_paths)

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
        persist_files: bool = False,
    ) -> _Self:
        """
        Create ParquetDataTable object from parquet files.

        Args:
            parquet_path (Union[Path, str, Iterable[Union[Path, str]]]): Path or list of parquet
                paths.
            index_column_names (Optional[Union[str, Iterable[str]]]): Index column name or names.
            persist_files (bool, optional): Whether to keep the files after object removal or
                delete them from disk. Defaults to False.

        Returns:
            ParquetDataTable: Created object.
        """
        if isinstance(parquet_path, (str, Path)):
            return cls(
                parquet_paths=[Path(parquet_path)],
                index_column_names=index_column_names,
                persist_files=persist_files,
            )

        return cls(
            parquet_paths=[Path(p) for p in parquet_path],
            index_column_names=index_column_names,
            persist_files=persist_files,
        )

    @classmethod
    def _dataframe_to_parquet(
        cls,
        dataframe: pd.DataFrame,
        parquet_path: Optional[Union[Path, str]] = None,
    ) -> tuple[Path, Optional[Union[str, Iterable[str]]]]:
        """
        Save DataFrame to parquet file.

        Args:
            dataframe (pd.DataFrame): DataFrame to save.
            parquet_path (Optional[Union[Path, str]], optional): Path where to save the parquet
                file. Defaults to None.

        Returns:
            tuple[Path, Optional[Union[str, Iterable[str]]]]: Path to the saved file and a list of
                index names.
        """
        if not parquet_path:
            prefix_path = cls.generate_filename()
            h = hashlib.new("sha256")
            h.update(dataframe.values.tobytes())
            gdf_hash = h.hexdigest()[:8]
            parquet_path = cls.get_directory() / f"{prefix_path}_{gdf_hash}.parquet"

        Path(parquet_path).parent.mkdir(exist_ok=True, parents=True)
        dataframe.rename(
            columns={column: str(column) for column in dataframe.columns}
        ).reset_index().to_parquet(
            parquet_path,
            index=False,
            row_group_size=PARQUET_ROW_GROUP_SIZE,
            compression=PARQUET_COMPRESSION,
        )

        index_names = dataframe.index.name or dataframe.index.names
        if len(index_names) == 1 and index_names[0] is None:
            index_names = "index"

        return (
            Path(parquet_path),
            cast("Optional[Union[str, Iterable[str]]]", index_names),
        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        parquet_path: Optional[Union[Path, str]] = None,
        persist_files: bool = False,
    ) -> "ParquetDataTable":
        """
        Create ParquetDataTable object from DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame to save.
            parquet_path (Optional[Union[Path, str]], optional): Path where to save the parquet
                file. Defaults to None.
            persist_files (bool, optional): Whether to keep the files after object removal or
                delete them from disk. Defaults to False.

        Returns:
            ParquetDataTable: Created object.
        """
        parquet_path, index_names = cls._dataframe_to_parquet(dataframe, parquet_path)

        return cls.from_parquet(
            parquet_path=parquet_path,
            index_column_names=index_names,
            persist_files=persist_files,
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

    def drop_columns(
        self: _Self, columns: Union[str, Iterable[str]], missing_ok: bool = False
    ) -> "ParquetDataTable":
        """Drop columns from the data table in place."""
        columns_drop_result = self._drop_columns(columns=columns, missing_ok=missing_ok)

        if columns_drop_result is None:
            return self

        new_parquet_paths, new_index_column_names = columns_drop_result

        return self.from_parquet(
            parquet_path=new_parquet_paths,
            index_column_names=new_index_column_names,
            persist_files=False,
        )

    def persist(self) -> None:
        """Disable parquet file removal."""
        self._detach_finalizer()

    def _drop_columns(
        self: _Self, columns: Union[str, Iterable[str]], missing_ok: bool = False
    ) -> Optional[tuple[list[Path], Optional[list[str]]]]:
        """Drop columns from the data table in place."""
        if isinstance(columns, str):
            columns = [columns]

        existing_columns = self.physical_columns
        missing_columns = set(columns).difference(existing_columns)
        print(f"{missing_columns=}")
        if missing_columns and not missing_ok:
            raise ValueError(f"Columns {missing_columns} are not present in the data table.")

        columns_to_drop = set(columns).intersection(existing_columns)
        print(f"{columns_to_drop=}")
        if not columns_to_drop:
            return None

        columns_to_keep = set(existing_columns).difference(columns)
        columns_to_keep_in_order = [c for c in existing_columns if c in columns_to_keep]
        print(f"{columns_to_keep_in_order=}")
        print(f"{self.parquet_paths=}")

        new_parquet_paths = []
        for parquet_path in self.parquet_paths:
            prefix_path = self.generate_filename()
            relation = duckdb.read_parquet(str(parquet_path)).select(*columns_to_keep_in_order)

            h = hashlib.new("sha256")
            h.update(relation.sql_query().encode())
            relation_hash = h.hexdigest()[:8]

            new_parquet_path = self.get_directory() / f"{prefix_path}_{relation_hash}.parquet"
            relation.to_parquet(
                str(new_parquet_path),
                row_group_size=PARQUET_ROW_GROUP_SIZE,
                compression=PARQUET_COMPRESSION,
            )
            new_parquet_paths.append(new_parquet_path)

        print(f"{new_parquet_paths=}")

        if self.index_column_names is not None:
            new_index_column_names: Optional[list[str]] = list(
                set(self.index_column_names).intersection(columns_to_keep)
            )
            if not new_index_column_names:
                new_index_column_names = None

        return new_parquet_paths, new_index_column_names


class GeoDataTable(ParquetDataTable):
    """
    GeoDataTable.

    A SRAI internal internal memory object for keeping geo data.

    It is a wrapper around parquet files with utility functions.
    """

    FILES_DIRECTORY = Path("files/gdt")

    def __init__(
        self,
        parquet_paths: Iterable[Path],
        index_column_names: Optional[Union[str, Iterable[str]]] = None,
        persist_files: bool = False,
        sort_geometries: bool = True,
        sort_extent: Optional[tuple[float, float, float, float]] = None,
    ):
        """
        Initialize GeoDataTable.

        Args:
            parquet_paths (Iterable[Path]): List of parquet files.
            index_column_names (Optional[Union[str, Iterable[str]]]): Index column names.
            persist_files (bool, optional): Whether to keep the files after object removal or
                delete them from disk. Defaults to False.
            sort_geometries (bool, optional): Whether to sort geometries to keep geometries
                organized. Defaults to True.
            sort_extent (Optional[tuple[float, float, float, float]], optional): Extent to use
                in the ST_Hilbert sorting. Defaults to None.
        """
        super().__init__(parquet_paths, index_column_names, persist_files)

        schema = ds.dataset(parquet_paths).schema
        geometry_columns = _geoparquet_guess_geometry_columns(schema)

        if GEOMETRY_COLUMN not in geometry_columns:
            raise ValueError(
                "Missing geometry column in the parquet file."
                f" Expecting geoarrow compatible column named '{GEOMETRY_COLUMN}'."
            )

        if sort_geometries:
            new_parquet_paths = self._sort_geometries(sort_extent)
            self._detach_finalizer()
            self.parquet_paths = new_parquet_paths

            if not self.persist_files:
                self._configure_finalizer()

    @classmethod
    def from_parquet(
        cls,
        parquet_path: Union[Path, str, Iterable[Union[Path, str]]],
        index_column_names: Optional[Union[str, Iterable[str]]] = None,
        persist_files: bool = False,
        sort_geometries: bool = False,
        sort_extent: Optional[tuple[float, float, float, float]] = None,
    ) -> "GeoDataTable":
        """
        Create GeoDataTable object from parquet files.

        Args:
            parquet_path (Union[Path, str, Iterable[Union[Path, str]]]): Path or list of parquet
                paths.
            index_column_names (Optional[Union[str, Iterable[str]]]): Index column name or names.
            persist_files (bool, optional): Whether to keep the files after object removal or
                delete them from disk. Defaults to False.
            sort_geometries (bool, optional): Whether to sort geometries to keep geometries
                organized. Defaults to True.
            sort_extent (Optional[tuple[float, float, float, float]], optional): Extent to use
                in the ST_Hilbert sorting. Defaults to None.

        Returns:
            GeoDataTable: Created object.
        """
        if isinstance(parquet_path, (str, Path)):
            return cls(
                parquet_paths=[Path(parquet_path)],
                index_column_names=index_column_names,
                persist_files=persist_files,
                sort_geometries=sort_geometries,
                sort_extent=sort_extent,
            )

        return cls(
            parquet_paths=[Path(p) for p in parquet_path],
            index_column_names=index_column_names,
            persist_files=persist_files,
            sort_geometries=sort_geometries,
            sort_extent=sort_extent,
        )

    @classmethod
    def from_geodataframe(
        cls,
        geodataframe: gpd.GeoDataFrame,
        parquet_path: Optional[Union[Path, str]] = None,
        persist_files: bool = False,
        sort_geometries: bool = True,
        sort_extent: Optional[tuple[float, float, float, float]] = None,
    ) -> "GeoDataTable":
        """
        Create GeoDataTable object from GeoDataFrame.

        Args:
            geodataframe (gpd.GeoDataFrame): GeoDataFrame to save.
            parquet_path (Optional[Union[Path, str]], optional): Path where to save the parquet
                file. Defaults to None.
            persist_files (bool, optional): Whether to keep the files after object removal or
                delete them from disk. Defaults to False.
            sort_geometries (bool, optional): Whether to sort geometries to keep geometries
                organized. Defaults to True.
            sort_extent (Optional[tuple[float, float, float, float]], optional): Extent to use
                in the ST_Hilbert sorting. Defaults to None.

        Returns:
            GeoDataTable: Created object.
        """
        parquet_path, index_names = cls._dataframe_to_parquet(geodataframe, parquet_path)

        return cls.from_parquet(
            parquet_path=parquet_path,
            index_column_names=index_names,
            persist_files=persist_files,
            sort_geometries=sort_geometries,
            sort_extent=sort_extent,
        )

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Get GeoDataFrame."""
        gdf = gpd.read_parquet(self.parquet_paths)
        if self.index_column_names is not None:
            gdf.set_index(self.index_column_names, inplace=True)
        return gdf

    def _sort_geometries(
        self, sort_extent: Optional[tuple[float, float, float, float]]
    ) -> list[Path]:
        """Order geometries and save parquet files again."""
        sorted_files = []

        for path in self.parquet_paths:
            new_parquet_path = self.get_directory() / f"{path.stem}_sorted.parquet"

            sort_geoparquet_file_by_geometry(
                input_file_path=path,
                output_file_path=new_parquet_path,
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
                row_group_size=PARQUET_ROW_GROUP_SIZE,
                sort_extent=sort_extent,
                verbosity_mode="transient",
                remove_input_file=not self.persist_files,
            )

            sorted_files.append(new_parquet_path)

        return sorted_files

    def union_all(self, method: Literal["coverage", "unary"] = "coverage") -> BaseGeometry:
        """Return union of all geometries in the GeoDataTable."""
        if method == "coverage":
            union_fn = coverage_union_all
        elif method == "unary":
            union_fn = union_all
        else:
            raise ValueError(f"Method '{method}' not recognized. Use 'coverage' or 'unary'.")

        dataset = ds.dataset(self.parquet_paths)

        tuples_to_queue = []
        for pq_file in dataset.files:
            for row_group in range(pq.ParquetFile(pq_file).num_row_groups):
                tuples_to_queue.append((pq_file, row_group))

        no_cpus = multiprocessing.cpu_count()

        total_row_groups = len(tuples_to_queue)

        with ProcessPoolExecutor(max_workers=max(1, min(no_cpus, total_row_groups))) as ex:
            try:
                fn = partial(_union_geometries, union_fn=union_fn)
                geometries = list(
                    tqdm(
                        ex.map(fn, tuples_to_queue, chunksize=1),
                        desc="Unioning geometries",
                        total=total_row_groups,
                    )
                )
            except GEOSException:
                fn = partial(_union_geometries, union_fn=union_all)
                geometries = list(
                    tqdm(
                        ex.map(fn, tuples_to_queue, chunksize=1),
                        desc="Unioning geometries",
                        total=total_row_groups,
                    )
                )

        if len(geometries) == 1:
            return geometries[0]

        try:
            return union_fn(geometries)
        except GEOSException:
            return union_all(geometries)

    @override
    def drop_columns(
        self: "GeoDataTable", columns: Union[str, Iterable[str]], missing_ok: bool = False
    ) -> Union["ParquetDataTable", "GeoDataTable"]:
        """Drop columns from the data table in place."""
        columns_drop_result = self._drop_columns(columns=columns, missing_ok=missing_ok)

        if columns_drop_result is None:
            return self

        new_parquet_paths, new_index_column_names = columns_drop_result

        if GEOMETRY_COLUMN in columns:
            return ParquetDataTable.from_parquet(
                parquet_path=new_parquet_paths,
                index_column_names=new_index_column_names,
                persist_files=False,
            )
        else:
            return GeoDataTable.from_parquet(
                parquet_path=new_parquet_paths,
                index_column_names=new_index_column_names,
                persist_files=False,
            )


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


def _union_geometries(
    parquet_info_tuple: tuple[str, int],
    union_fn: Callable[[Iterable[BaseGeometry]], BaseGeometry],
) -> BaseGeometry:
    parquet_path, row_group = parquet_info_tuple
    return union_fn(
        gpd.GeoSeries.from_arrow(
            GeoArray.from_arrow(
                pq.ParquetFile(parquet_path)
                .read_row_group(row_group, columns=[GEOMETRY_COLUMN])[GEOMETRY_COLUMN]
                .combine_chunks()
            )
        )
    )
