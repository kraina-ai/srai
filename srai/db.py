"""Module with functions required to interact with DuckDB Python objects."""

import secrets
from pathlib import Path
from typing import List, Optional, Union, cast

import duckdb
import geopandas as gpd
import pandas as pd
import psutil
from shapely import wkt as shp_wkt
from shapely.validation import make_valid as shp_make_valid

from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.utils.download import download_file

CONNECTION: Optional[duckdb.DuckDBPyConnection] = None
OFFICIAL_DUCKDB_EXTENSIONS = [
    "json",
    # "spatial",
    # "h3"
]


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Get DuckDB connection.

    Returns:
        duckdb.DuckDBPyConnection: Prepared connection object.
    """
    global CONNECTION  # noqa: PLW0603

    if CONNECTION is None:
        CONNECTION = duckdb.connect(
            database=":memory:",
            config=dict(
                temp_directory="duckdb_temp/",
                allow_unsigned_extensions="true",
                # memory_limit=f"{_get_memory_limit(0.5)}B",
            ),
        )
        for extension in OFFICIAL_DUCKDB_EXTENSIONS:
            CONNECTION.install_extension(extension)
            CONNECTION.load_extension(extension)

        _install_dev_spatial_extension(CONNECTION)

        _create_python_functions(CONNECTION)

    return CONNECTION


def get_new_duckdb_connection(
    db_file: str = ":memory:", read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """
    Get new DuckDB connection.

    Allows for configuration of the connection.

    Returns:
        duckdb.DuckDBPyConnection: Prepared connection object.
    """
    conn = duckdb.connect(
        database=db_file,
        read_only=read_only,
        config=dict(
            temp_directory="duckdb_temp/",
            allow_unsigned_extensions="true",
            # memory_limit=f"{_get_memory_limit(0.25)}B",
        ),
    )
    for extension in OFFICIAL_DUCKDB_EXTENSIONS:
        conn.install_extension(extension)
        conn.load_extension(extension)

    _install_dev_spatial_extension(conn)

    _create_python_functions(conn)

    return conn


def _install_dev_spatial_extension(conn: duckdb.DuckDBPyConnection) -> None:
    spatial_extension_path = Path("spatial.duckdb_extension").resolve().as_posix()
    download_file(
        url="https://drive.google.com/uc?export=download&id=1WMEWVddQiMjnGUbSfcVq6GLs2Zup19hW",
        fname=spatial_extension_path,
        force_download=False,
    )

    conn.execute(f"INSTALL '{spatial_extension_path}'")
    conn.execute(f"LOAD '{spatial_extension_path}'")


def _get_memory_limit(fraction: float = 0.5) -> int:
    available_memory = psutil.virtual_memory().available
    bytes_memory_size = int(available_memory * fraction)
    return bytes_memory_size


def _create_python_functions(connection: duckdb.DuckDBPyConnection) -> None:
    def _make_valid_shapely(wkt_string: str) -> str:
        parsed_geom = shp_wkt.loads(wkt_string)
        valid_geom = shp_make_valid(parsed_geom)
        valid_geom_wkt = cast(str, valid_geom.wkt)
        return valid_geom_wkt

    def _split_geometry_collection(wkt_string: str) -> List[str]:
        parsed_geom = shp_wkt.loads(wkt_string)
        valid_geom_wkts = []
        for member in parsed_geom.geoms:
            valid_geom = shp_make_valid(member)
            valid_geom_wkts.append(valid_geom.wkt)
        return valid_geom_wkts

    connection.create_function("py_make_valid", _make_valid_shapely)
    connection.create_function(
        "py_split_geometry_collection",
        _split_geometry_collection,
        [duckdb.typing.VARCHAR],
        duckdb.list_type(type=duckdb.typing.VARCHAR),
    )

    duckdb_make_valid_macro_query = """
    CREATE OR REPLACE MACRO make_valid_geom(geometry) AS
    CASE WHEN ST_IsValid(geometry)
        THEN geometry
        ELSE ST_GeomFromText(py_make_valid(ST_AsText(geometry)))
    END;
    """
    connection.execute(duckdb_make_valid_macro_query)

    duckdb_split_geometry_collection_macro_query = """
    CREATE OR REPLACE MACRO split_geometry_collection(geometry) AS
    CASE WHEN ST_GeometryType(geometry) = 'GEOMETRYCOLLECTION'
        THEN [
            ST_GeomFromText(geom_member)
            for geom_member in py_split_geometry_collection(ST_AsText(geometry))
        ]
        ELSE [geometry]
    END;
    """
    connection.execute(duckdb_split_geometry_collection_macro_query)


def count_relation_rows(relation: duckdb.DuckDBPyRelation) -> int:
    """Get count of rows in the relation."""
    relation_id = secrets.token_hex(nbytes=16)
    relation_name = f"count_{relation_id}"
    count_query = f"SELECT COUNT(*) FROM {relation_name}"
    count_result = relation.query(
        virtual_table_name=relation_name, sql_query=count_query
    ).fetchone()
    if not count_result:
        return 0

    return int(count_result[0])


def relation_to_table(relation: duckdb.DuckDBPyRelation, prefix: str) -> duckdb.DuckDBPyRelation:
    """Save relation query to table."""
    relation_id = secrets.token_hex(nbytes=16)
    relation_name = f"{prefix}_{relation_id}"
    relation.to_table(relation_name)
    return get_duckdb_connection().table(relation_name)


def delete_table(relation: duckdb.DuckDBPyRelation) -> None:
    """Remove table from the database."""
    get_duckdb_connection().execute(f"DROP TABLE IF EXISTS {relation.alias}")


def duckdb_to_df(relation: duckdb.DuckDBPyRelation) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Transform DuckDB relation to DataFrame or GeoDataFrame.

    Parses data from provided relation to DataFrame or GeoDataFrame if contains a geography column.
    Automatically detects keywords to reconstruct an index for a DataFrame.

    Args:
        relation (duckdb.DuckDBPyRelation): Relation to be transformed.

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: Resulting DataFrame.
    """
    query = "SELECT * FROM {virtual_relation_name}"

    has_geometry = (
        GEOMETRY_COLUMN in relation.columns
        and relation.types[relation.columns.index(GEOMETRY_COLUMN)] == "GEOMETRY"
    )

    if has_geometry:
        query = "SELECT * EXCLUDE (geometry), ST_AsText(geometry) wkt FROM {virtual_relation_name}"

    random_id = f"virtual_{secrets.token_hex(nbytes=16)}"
    filled_query = query.format(virtual_relation_name=random_id)
    df = relation.query(
        virtual_table_name=random_id,
        sql_query=filled_query,
    ).to_df()

    if has_geometry:
        df = gpd.GeoDataFrame(
            data=df.drop(columns=["wkt"]),
            geometry=gpd.GeoSeries().from_wkt(df["wkt"]),
            crs=WGS84_CRS,
        )

    if REGIONS_INDEX in df.columns and FEATURES_INDEX in df.columns:
        df.set_index([REGIONS_INDEX, FEATURES_INDEX], inplace=True)
    elif REGIONS_INDEX in df.columns:
        df.set_index(REGIONS_INDEX, inplace=True)
    elif FEATURES_INDEX in df.columns:
        df.set_index(FEATURES_INDEX, inplace=True)

    return df


def df_to_duckdb(dataframe: Union[pd.DataFrame, gpd.GeoDataFrame]) -> duckdb.DuckDBPyRelation:
    """
    Transforms a DataFrame to a DuckDB relation.

    Parses data from provided DataFrame or GeoDataFrame to relation with optional geography column.

    Args:
        dataframe (Union[pd.DataFrame, gpd.GeoDataFrame]): DataFrame to be transformed.

    Returns:
        duckdb.DuckDBPyRelation: Relation object with data from DataFrame.
    """
    relation_id = secrets.token_hex(nbytes=16)
    temp_dataframe_id = f"temp_df_{relation_id}"

    dataframe = dataframe.reset_index()
    if REGIONS_INDEX not in dataframe.columns and FEATURES_INDEX not in dataframe.columns:
        raise ValueError(
            f"Dataframe must have `{REGIONS_INDEX}` and / or `{FEATURES_INDEX}` index."
        )

    if dataframe.empty:
        table_id = f"parsed_{relation_id}"
        empty_table_query = (
            "CREATE TEMP TABLE"
            f" {table_id} ({', '.join(f'{column} VARCHAR' for column in dataframe.columns)})"
        )
        get_duckdb_connection().execute(empty_table_query)
        return get_duckdb_connection().table(table_id)

    query = f"SELECT * FROM {temp_dataframe_id}"

    has_geometry = isinstance(dataframe, gpd.GeoDataFrame) and not dataframe.empty
    if has_geometry:
        dataframe["wkt"] = dataframe[GEOMETRY_COLUMN].apply(lambda x: x.wkt)
        dataframe.drop(columns=GEOMETRY_COLUMN, inplace=True)
        query = (
            "SELECT * EXCLUDE (wkt), make_valid_geom(ST_GeomFromText(wkt)) geometry FROM"
            f" {temp_dataframe_id}"
        )

    dataframe_sample = _get_nonempty_sample(dataframe)
    dataframe["_is_sample"] = False
    dataframe_sample["_is_sample"] = True

    dataframe = pd.concat([dataframe_sample, dataframe])

    get_duckdb_connection().register(temp_dataframe_id, dataframe)
    final_query = f"SELECT * EXCLUDE (_is_sample) FROM ({query}) WHERE _is_sample = false"

    table_relation = relation_to_table(
        relation=get_duckdb_connection().from_query(final_query), prefix="parsed"
    )
    get_duckdb_connection().unregister(temp_dataframe_id)

    return table_relation


def _get_nonempty_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a dataframe sample with non-null columns.

    DuckDB uses own parsing algorithm to determine column types. Sparse dataframes can result in
    parsing errors. This function returns top 100 non-empty rows from a given dataframe.
    """
    column_samples = {}

    for column_name in df.columns:
        column_samples[column_name] = df.loc[df[column_name].notna()][column_name][
            :100
        ].reset_index(drop=True)

    return pd.DataFrame(data=column_samples)


def escape(value: str) -> str:
    """Escape value for SQL query."""
    return value.replace("'", "''")
