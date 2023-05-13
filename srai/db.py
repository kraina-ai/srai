"""TODO."""

import secrets
from typing import Optional, Union

import duckdb
import geopandas as gpd
import pandas as pd
import psutil

from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS

CONNECTION: Optional[duckdb.DuckDBPyConnection] = None
DUCKDB_EXTENSIONS = ["json", "spatial"]


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
                allow_unsigned_extensions="true",
                memory_limit=f"{int(psutil.virtual_memory().available * 0.25)}b",
                temp_directory="duckdb_temp/",
            ),
        )
        for extension in DUCKDB_EXTENSIONS:
            CONNECTION.install_extension(extension)
            CONNECTION.load_extension(extension)

    return CONNECTION


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
    df = relation.query(
        virtual_table_name=random_id,
        sql_query=query.format(virtual_relation_name=random_id),
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
    relation_name = f"parsed_{relation_id}"
    temp_dataframe_id = f"temp_df_{relation_id}"

    dataframe = dataframe.reset_index()

    query = f"SELECT * FROM {temp_dataframe_id}"

    has_geometry = isinstance(dataframe, gpd.GeoDataFrame)
    if has_geometry:
        dataframe["wkt"] = dataframe[GEOMETRY_COLUMN].apply(lambda x: x.wkt)
        dataframe.drop(columns=GEOMETRY_COLUMN, inplace=True)
        query = f"SELECT * EXCLUDE (wkt), ST_GeomFromText(wkt) geometry FROM {temp_dataframe_id}"

    dataframe_sample = _get_nonempty_sample(dataframe)
    dataframe["_is_sample"] = False
    dataframe_sample["_is_sample"] = True

    dataframe = pd.concat([dataframe_sample, dataframe])

    get_duckdb_connection().register(temp_dataframe_id, dataframe)
    final_query = f"SELECT * EXCLUDE (_is_sample) FROM ({query}) WHERE _is_sample = false"

    relation = get_duckdb_connection().from_query(final_query, alias=relation_name).execute()

    return relation


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
