"""Tests for PbfFileReader."""

import platform
import re
import subprocess
import warnings
from collections.abc import Iterable
from distutils.spawn import find_executable
from pathlib import Path
from typing import Optional, Union, cast
from unittest import TestCase

import duckdb
import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
import six
from parametrization import Parametrization as P
from shapely import hausdorff_distance
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from srai.constants import FEATURES_INDEX
from srai.geometry import remove_interiors
from srai.loaders.download import download_file
from srai.loaders.osm_loaders.filters import GEOFABRIK_LAYERS, HEX2VEC_FILTER, OsmTagsFilter
from srai.loaders.osm_loaders.pbf_file_reader import PbfFileReader

ut = TestCase()
LFS_DIRECTORY_URL = "https://github.com/kraina-ai/srai-test-files/raw/main/files/"


@pytest.mark.parametrize(  # type: ignore
    "test_file_name,query,expected_result_length,expected_features_columns_length",
    [
        (
            "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf",
            None,
            678,
            271,
        ),
        (
            "eb2848d259345ce7dfe8af34fd1ab24503bb0b952e04e872c87c55550fa50fbf.osm.pbf",
            None,
            1,
            22,
        ),
        ("529cdcbb7a3cc103658ef31b39bed24984e421127d319c867edf2f86ff3bb098.osm.pbf", None, 0, 0),
        (
            "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf",
            HEX2VEC_FILTER,
            97,
            10,
        ),
        (
            "eb2848d259345ce7dfe8af34fd1ab24503bb0b952e04e872c87c55550fa50fbf.osm.pbf",
            HEX2VEC_FILTER,
            0,
            0,
        ),
        (
            "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf",
            GEOFABRIK_LAYERS,
            433,
            22,
        ),
        (
            "eb2848d259345ce7dfe8af34fd1ab24503bb0b952e04e872c87c55550fa50fbf.osm.pbf",
            GEOFABRIK_LAYERS,
            0,
            0,
        ),
    ],
)
def test_pbf_reader(
    test_file_name: str,
    query: OsmTagsFilter,
    expected_result_length: int,
    expected_features_columns_length: int,
):
    """Test proper files loading in `PbfFileReader`."""
    handler = PbfFileReader(tags_filter=query)
    features_gdf = handler.get_features_gdf(
        file_paths=[Path(__file__).parent / "test_files" / test_file_name],
        explode_tags=True,
        ignore_cache=True,
    )
    assert (
        len(features_gdf) == expected_result_length
    ), f"Mismatched result length ({len(features_gdf)}, {expected_result_length})"
    assert len(features_gdf.columns) == expected_features_columns_length + 1, (
        f"Mismatched columns length ({len(features_gdf.columns)},"
        f" {expected_features_columns_length + 1})"
    )


def test_pbf_reader_geometry_filtering():  # type: ignore
    """Test proper spatial data filtering in `PbfFileReader`."""
    file_name = "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf"
    handler = PbfFileReader(
        tags_filter=HEX2VEC_FILTER, geometry_filter=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    )
    features_gdf = handler.get_features_gdf(
        file_paths=[Path(__file__).parent / "test_files" / file_name],
        explode_tags=True,
        ignore_cache=True,
    )
    assert len(features_gdf) == 0


# Copyright (C) 2011 by Hong Minhee <http://dahlia.kr/>,
#                       Robert Kajic <http://github.com/kajic>
# Copyright (C) 2020 by Salesforce.com, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
def parse_hstore_tags(tags: str) -> dict[str, Optional[str]]:
    """
    Parse hstore tags to python dict.

    This function has been copied from pghstore library
    https://github.com/heroku/pghstore/blob/main/src/pghstore/_native.py
    since it can't be installed on Windows.
    """
    ESCAPE_RE = re.compile(r"\\(.)")

    PAIR_RE = re.compile(
        r'\s*(?:"(?P<kq>(?:[^\\"]|\\.)*)")\s*=>\s*'
        r'(?:"(?P<vq>(?:[^\\"]|\\.)*)"|(?P<vn>NULL))'
        r"\s*(?:(?P<ts>,)|$)",
        re.IGNORECASE,
    )

    def _unescape(s: str) -> str:
        return ESCAPE_RE.sub(r"\1", s)

    def _parse(string: str, encoding: str = "utf-8") -> Iterable[tuple[str, Optional[str]]]:
        if isinstance(string, six.binary_type):
            string = string.decode(encoding)

        string = string.strip()
        offset = 0
        term_sep = None
        for match in PAIR_RE.finditer(string):
            if match.start() > offset:
                raise ValueError("malformed hstore value: position %d" % offset)

            key = value = None
            kq = match.group("kq")
            if kq:
                key = _unescape(kq)

            if key is None:
                raise ValueError("Malformed hstore value starting at position %d" % offset)

            vq = match.group("vq")
            if vq:
                value = _unescape(vq)
            elif match.group("vn"):
                value = ""
            else:
                value = ""

            yield key, value

            term_sep = match.group("ts")

            offset = match.end()

        if len(string) > offset or term_sep:
            raise ValueError("malformed hstore value: position %d" % offset)

    return dict(_parse(tags, encoding="utf-8"))


def transform_pbf_to_gpkg(extract_name: str, layer_name: str) -> Path:
    """Uses GDAL ogr2ogr to transform PBF file into GPKG."""
    input_file = Path(__file__).parent / "files" / f"{extract_name}.osm.pbf"
    output_file = Path(__file__).parent / "files" / f"{extract_name}_{layer_name}.gpkg"
    config_file = Path(__file__).parent / "test_files" / "osmconf.ini"
    args = [
        "ogr2ogr" if platform.system() != "Windows" else "ogr2ogr.exe",
        str(output_file),
        str(input_file),
        layer_name,
        "-oo",
        f"CONFIG_FILE={config_file}",
    ]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
    _, err = p.communicate()
    rc = p.returncode
    if rc > 0:
        raise RuntimeError(rc, err)

    return output_file


def read_features_with_pyogrio(extract_name: str) -> gpd.GeoDataFrame:
    """Read features from *.osm.pbf file using pyogrio."""
    gdfs = []
    for layer_name in ("points", "lines", "multilinestrings", "multipolygons", "other_relations"):
        gpkg_file_path = transform_pbf_to_gpkg(extract_name, layer_name)
        gdf = pyogrio.read_dataframe(gpkg_file_path)

        if layer_name == "points":
            gdf[FEATURES_INDEX] = "node/" + gdf["osm_id"]
        elif layer_name == "lines":
            gdf[FEATURES_INDEX] = "way/" + gdf["osm_id"]
        elif layer_name in ("multilinestrings", "other_relations"):
            gdf[FEATURES_INDEX] = "relation/" + gdf["osm_id"]
        elif layer_name == "multipolygons":
            gdf[FEATURES_INDEX] = gdf.apply(
                lambda row: (
                    "relation/" + row["osm_id"]
                    if row["osm_id"] is not None
                    else "way/" + row["osm_way_id"]
                ),
                axis=1,
            )

        gdfs.append(gdf)

    final_gdf = gpd.pd.concat(gdfs)
    final_gdf = final_gdf[~final_gdf["all_tags"].isnull()]
    final_gdf["tags"] = final_gdf["all_tags"].apply(parse_hstore_tags)
    non_relations = ~final_gdf[FEATURES_INDEX].str.startswith("relation/")
    relations = final_gdf[FEATURES_INDEX].str.startswith("relation/")
    matching_relations = relations & final_gdf["tags"].apply(
        lambda x: x.get("type") in ("boundary", "multipolygon")
    )
    final_gdf = final_gdf[non_relations | matching_relations]
    final_gdf.geometry = final_gdf.geometry.make_valid()
    return final_gdf[[FEATURES_INDEX, "tags", "geometry"]].set_index(FEATURES_INDEX)


def check_if_relation_in_osm_is_valid_based_on_tags(pbf_file: str, relation_id: str) -> bool:
    """Check if given relation in OSM is valid."""
    duckdb.load_extension("spatial")
    return cast(
        bool,
        duckdb.sql(
            f"SELECT list_contains(ref_roles, 'outer') FROM ST_READOSM('{pbf_file}') "
            "WHERE kind = 'relation' AND len(refs) > 0 AND list_contains(map_keys(tags), 'type') "
            "AND list_has_any(map_extract(tags, 'type'), ['boundary', 'multipolygon']) "
            f"AND id = {relation_id}"
        ).fetchone()[0],
    )


def check_if_relation_in_osm_is_valid_based_on_geometry(pbf_file: str, relation_id: str) -> bool:
    """
    Check if given relation in OSM is valid.

    Reconstructs full geometry for a single ID and check if there is at least one outer geometry.
    Sometimes
    """
    duckdb.load_extension("spatial")
    return cast(
        bool,
        duckdb.sql(f"""
            WITH required_relation AS (
                SELECT
                    r.id
                FROM ST_ReadOsm('{pbf_file}') r
                WHERE r.kind = 'relation'
                    AND len(r.refs) > 0
                    AND list_contains(map_keys(r.tags), 'type')
                    AND list_has_any(
                        map_extract(r.tags, 'type'),
                        ['boundary', 'multipolygon']
                    )
                    AND r.id = {relation_id}
            ),
            unnested_relation_refs AS (
                SELECT
                    r.id,
                    UNNEST(refs) as ref,
                    UNNEST(ref_types) as ref_type,
                    UNNEST(ref_roles) as ref_role,
                    UNNEST(range(length(refs))) as ref_idx
                FROM ST_ReadOsm('{pbf_file}') r
                SEMI JOIN required_relation rr
                ON r.id = rr.id
            ),
            unnested_relation_way_refs AS (
                SELECT id, ref, COALESCE(ref_role, 'outer') as ref_role, ref_idx
                FROM unnested_relation_refs
                WHERE ref_type = 'way'
            ),
            unnested_relations AS (
                SELECT
                    r.id,
                    COALESCE(r.ref_role, 'outer') as ref_role,
                    r.ref,
                FROM unnested_relation_way_refs r
                ORDER BY r.id, r.ref_idx
            ),
            unnested_way_refs AS (
                SELECT
                    w.id,
                    UNNEST(refs) as ref,
                    UNNEST(ref_types) as ref_type,
                    UNNEST(range(length(refs))) as ref_idx
                FROM ST_ReadOsm('{pbf_file}') w
                SEMI JOIN unnested_relation_way_refs urwr
                ON urwr.ref = w.id
                WHERE w.kind = 'way'
            ),
            nodes_geometries AS (
                SELECT
                    n.id,
                    ST_POINT(n.lon, n.lat) geom
                FROM ST_ReadOsm('{pbf_file}') n
                SEMI JOIN unnested_way_refs uwr
                ON uwr.ref = n.id
                WHERE n.kind = 'node'
            ),
            way_geometries AS (
                SELECT uwr.id, ST_MakeLine(LIST(n.geom ORDER BY ref_idx ASC)) linestring
                FROM unnested_way_refs uwr
                JOIN nodes_geometries n
                ON uwr.ref = n.id
                GROUP BY uwr.id
            ),
            any_outer_refs AS (
                SELECT id, bool_or(ref_role == 'outer') any_outer_refs
                FROM unnested_relations
                GROUP BY id
            ),
            relations_with_geometries AS (
                SELECT
                    x.id,
                    CASE WHEN aor.any_outer_refs
                        THEN x.ref_role ELSE 'outer'
                    END as ref_role,
                    x.geom geometry,
                    row_number() OVER (PARTITION BY x.id) as geometry_id
                FROM (
                    SELECT
                        unnested_relations.id,
                        unnested_relations.ref_role,
                        UNNEST(
                            ST_Dump(ST_LineMerge(ST_Collect(list(way_geometries.linestring)))),
                            recursive := true
                        ),
                    FROM unnested_relations
                    JOIN way_geometries ON way_geometries.id = unnested_relations.ref
                    GROUP BY unnested_relations.id, unnested_relations.ref_role
                ) x
                JOIN any_outer_refs aor ON aor.id = x.id
                WHERE ST_NPoints(geom) >= 4
            ),
            valid_relations AS (
                SELECT id, is_valid
                FROM (
                    SELECT
                        id,
                        bool_and(
                            ST_Equals(ST_StartPoint(geometry), ST_EndPoint(geometry))
                        ) is_valid
                    FROM relations_with_geometries
                    WHERE ref_role = 'outer'
                    GROUP BY id
                )
                WHERE is_valid = true
            )
            SELECT COUNT(*) > 0 AS 'any_valid_outer_geometry'
            FROM valid_relations
        """).fetchone()[0],
    )


def get_tags_from_osm_element(pbf_file: str, feature_id: str) -> dict[str, str]:
    """Check if given relation in OSM is valid."""
    duckdb.load_extension("spatial")
    kind, osm_id = feature_id.split("/", 2)
    raw_tags = duckdb.sql(
        f"SELECT tags FROM ST_READOSM('{pbf_file}') WHERE kind = '{kind}' AND id = {osm_id}"
    ).fetchone()[0]
    return dict(zip(raw_tags["key"], raw_tags["value"]))


def extract_polygons_from_geometry(geometry: BaseGeometry) -> list[Union[Polygon, MultiPolygon]]:
    """Extract only Polygons and MultiPolygons from the geometry."""
    polygon_geometries = []
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        polygon_geometries.append(geometry)
    elif geometry.geom_type in ("GeometryCollection"):
        polygon_geometries.extend(
            sub_geom
            for sub_geom in geometry.geoms
            if sub_geom.geom_type in ("Polygon", "MultiPolygon")
        )
    return polygon_geometries


@pytest.mark.skipif(  # type: ignore
    find_executable("ogr2ogr") is None,
    reason="requires ogr2ogr (GDAL) to be installed and available",
)
@P.parameters("extract_name")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Cyprus", "cyprus")  # type: ignore
@P.case("Cambodia", "cambodia")  # type: ignore
@P.case("Maldives", "maldives")  # type: ignore
@P.case("Seychelles", "seychelles")  # type: ignore
@P.case("Mauritius", "mauritius")  # type: ignore
@P.case("Greenland", "greenland")  # type: ignore
@P.case("Bahamas", "bahamas")  # type: ignore
@P.case("Panama", "panama")  # type: ignore
@P.case("Fiji", "fiji")  # type: ignore
@P.case("Kiribati", "kiribati")  # type: ignore
def test_gdal_parity(extract_name: str) -> None:
    """Test if loaded data is similar to GDAL results."""
    pbf_file_download_url = LFS_DIRECTORY_URL + f"{extract_name}-latest.osm.pbf"
    pbf_file_path = Path(__file__).parent / "files" / f"{extract_name}.osm.pbf"
    download_file(pbf_file_download_url, str(pbf_file_path), force_download=True)

    reader = PbfFileReader()
    duckdb_gdf = reader.get_features_gdf([pbf_file_path], explode_tags=False, ignore_cache=True)
    gdal_gdf = read_features_with_pyogrio(extract_name)

    gdal_index = gdal_gdf.index
    duckdb_index = duckdb_gdf.index

    missing_in_duckdb = gdal_index.difference(duckdb_index)
    # Get missing non relation features with at least one non-area tag value
    non_relations_missing_in_duckdb = [
        feature_id
        for feature_id in missing_in_duckdb
        if not feature_id.startswith("relation/")
        and any(True for k in gdal_gdf.loc[feature_id].tags.keys() if k != "area")
    ]
    valid_relations_missing_in_duckdb = [
        feature_id
        for feature_id in missing_in_duckdb
        if feature_id.startswith("relation/")
        and check_if_relation_in_osm_is_valid_based_on_tags(
            str(pbf_file_path), feature_id.replace("relation/", "")
        )
        and check_if_relation_in_osm_is_valid_based_on_geometry(
            str(pbf_file_path), feature_id.replace("relation/", "")
        )
    ]

    invalid_relations_missing_in_duckdb = missing_in_duckdb.difference(
        non_relations_missing_in_duckdb
    ).difference(valid_relations_missing_in_duckdb)

    assert (
        not non_relations_missing_in_duckdb
    ), f"Missing non relation features in PbfFileReader ({non_relations_missing_in_duckdb})"

    assert (
        not valid_relations_missing_in_duckdb
    ), f"Missing valid relation features in PbfFileReader ({valid_relations_missing_in_duckdb})"

    if len(invalid_relations_missing_in_duckdb) > 0:
        warnings.warn(
            "Invalid relations exists in OSM GDAL data extract"
            f" ({invalid_relations_missing_in_duckdb})",
            stacklevel=1,
        )

    invalid_features = []

    common_index = gdal_index.difference(invalid_relations_missing_in_duckdb)
    joined_df = pd.DataFrame(
        dict(
            duckdb_tags=duckdb_gdf.loc[common_index].tags,
            source_tags=gdal_gdf.loc[common_index].tags,
            duckdb_geometry=duckdb_gdf.loc[common_index].geometry,
            gdal_geometry=gdal_gdf.loc[common_index].geometry,
        ),
        index=common_index,
    )

    # Check tags
    joined_df["tags_keys_difference"] = joined_df.apply(
        lambda x: set(x.duckdb_tags.keys())
        .symmetric_difference(x.source_tags.keys())
        .difference(["area"]),
        axis=1,
    )

    # If difference - compare tags with source data.
    # Sometimes GDAL copies tags from members to a parent.
    mismatched_rows = joined_df["tags_keys_difference"].str.len() != 0
    if mismatched_rows.any():
        joined_df.loc[mismatched_rows, "source_tags"] = [
            get_tags_from_osm_element(str(pbf_file_path), row_index)
            for row_index in joined_df.loc[mismatched_rows].index
        ]

        joined_df.loc[mismatched_rows, "tags_keys_difference"] = joined_df.loc[
            mismatched_rows
        ].apply(
            lambda x: set(x.duckdb_tags.keys())
            .symmetric_difference(x.source_tags.keys())
            .difference(["area"]),
            axis=1,
        )

    for row_index in common_index:
        tags_keys_difference = joined_df.loc[row_index, "tags_keys_difference"]
        duckdb_tags = joined_df.loc[row_index, "duckdb_tags"]
        source_tags = joined_df.loc[row_index, "source_tags"]
        assert not tags_keys_difference, (
            f"Tags keys aren't equal. ({row_index}, {tags_keys_difference},"
            f" {duckdb_tags.keys()}, {source_tags.keys()})"
        )
        ut.assertDictEqual(
            {k: v for k, v in duckdb_tags.items() if k != "area"},
            {k: v for k, v in source_tags.items() if k != "area"},
            f"Tags aren't equal. ({row_index})",
        )

    invalid_geometries_df = joined_df

    # Check if both geometries are closed or open
    invalid_geometries_df["duckdb_is_closed"] = invalid_geometries_df["duckdb_geometry"].apply(
        lambda x: x.is_closed
    )
    invalid_geometries_df["gdal_is_closed"] = invalid_geometries_df["gdal_geometry"].apply(
        lambda x: x.is_closed
    )
    invalid_geometries_df["geometry_both_closed_or_not"] = (
        invalid_geometries_df["duckdb_is_closed"] == invalid_geometries_df["gdal_is_closed"]
    )

    tolerance = 0.5 * 10 ** (-6)
    # Check if geometries are almost equal - same geom type, same points
    invalid_geometries_df.loc[
        invalid_geometries_df["geometry_both_closed_or_not"], "geometry_almost_equals"
    ] = gpd.GeoSeries(
        invalid_geometries_df.loc[
            invalid_geometries_df["geometry_both_closed_or_not"], "duckdb_geometry"
        ]
    ).geom_equals_exact(
        gpd.GeoSeries(
            invalid_geometries_df.loc[
                invalid_geometries_df["geometry_both_closed_or_not"], "gdal_geometry"
            ]
        ),
        tolerance=tolerance,
    )
    invalid_geometries_df = invalid_geometries_df.loc[
        ~(
            invalid_geometries_df["geometry_both_closed_or_not"]
            & invalid_geometries_df["geometry_almost_equals"]
        )
    ]
    if invalid_geometries_df.empty:
        return

    # Check geometries equality - same geom type, same points
    invalid_geometries_df.loc[
        invalid_geometries_df["geometry_both_closed_or_not"], "geometry_equals"
    ] = gpd.GeoSeries(
        invalid_geometries_df.loc[
            invalid_geometries_df["geometry_both_closed_or_not"], "duckdb_geometry"
        ]
    ).geom_equals(
        gpd.GeoSeries(
            invalid_geometries_df.loc[
                invalid_geometries_df["geometry_both_closed_or_not"], "gdal_geometry"
            ]
        )
    )
    invalid_geometries_df = invalid_geometries_df.loc[
        ~(
            invalid_geometries_df["geometry_both_closed_or_not"]
            & invalid_geometries_df["geometry_equals"]
        )
    ]
    if invalid_geometries_df.empty:
        return

    # Check geometries overlap if polygons - slight misalingment between points,
    # but marginal
    matching_polygon_geometries_mask = (
        invalid_geometries_df["geometry_both_closed_or_not"]
        & gpd.GeoSeries(invalid_geometries_df["duckdb_geometry"]).geom_type.isin(
            ("Polygon", "MultiPolygon", "GeometryCollection")
        )
        & gpd.GeoSeries(invalid_geometries_df["gdal_geometry"]).geom_type.isin(
            ("Polygon", "MultiPolygon", "GeometryCollection")
        )
    )
    invalid_geometries_df.loc[matching_polygon_geometries_mask, "geometry_intersection_area"] = (
        gpd.GeoSeries(
            invalid_geometries_df.loc[matching_polygon_geometries_mask, "duckdb_geometry"]
        )
        .intersection(
            gpd.GeoSeries(
                invalid_geometries_df.loc[matching_polygon_geometries_mask, "gdal_geometry"]
            )
        )
        .area
    )

    invalid_geometries_df.loc[
        matching_polygon_geometries_mask, "iou_metric"
    ] = invalid_geometries_df.loc[
        matching_polygon_geometries_mask, "geometry_intersection_area"
    ] / (
        gpd.GeoSeries(
            invalid_geometries_df.loc[matching_polygon_geometries_mask, "duckdb_geometry"]
        ).area
        + gpd.GeoSeries(
            invalid_geometries_df.loc[matching_polygon_geometries_mask, "gdal_geometry"]
        ).area
        - invalid_geometries_df.loc[matching_polygon_geometries_mask, "geometry_intersection_area"]
    )

    invalid_geometries_df.loc[matching_polygon_geometries_mask, "geometry_iou_near_one"] = (
        invalid_geometries_df.loc[matching_polygon_geometries_mask, "iou_metric"] >= (1 - tolerance)
    )
    invalid_geometries_df = invalid_geometries_df.loc[
        ~(matching_polygon_geometries_mask & invalid_geometries_df["geometry_iou_near_one"])
    ]
    if invalid_geometries_df.empty:
        return

    # Check if points lay near each other - regardless of geometry type
    # (Polygon vs LineString)
    invalid_geometries_df["hausdorff_distance_value"] = invalid_geometries_df.apply(
        lambda x: hausdorff_distance(x.duckdb_geometry, x.gdal_geometry, densify=0.5), axis=1
    )
    invalid_geometries_df["geometry_close_hausdorff_distance"] = (
        invalid_geometries_df["hausdorff_distance_value"] < 1e-10
    )

    # Check if GDAL geometry is a linestring while DuckDB geometry is a polygon
    invalid_geometries_df.loc[
        invalid_geometries_df["geometry_close_hausdorff_distance"],
        "is_duckdb_polygon_and_gdal_linestring",
    ] = invalid_geometries_df.loc[invalid_geometries_df["geometry_close_hausdorff_distance"]].apply(
        lambda x: x.duckdb_geometry.geom_type
        in (
            "Polygon",
            "MultiPolygon",
        )
        and x.gdal_geometry.geom_type in ("LineString", "MultiLineString"),
        axis=1,
    )

    # Check if DuckDB geometry can be a polygon and not a linestring
    # based on features config
    invalid_geometries_df.loc[
        invalid_geometries_df["geometry_close_hausdorff_distance"]
        & invalid_geometries_df["is_duckdb_polygon_and_gdal_linestring"],
        "is_proper_filter_tag_value",
    ] = invalid_geometries_df.loc[
        invalid_geometries_df["geometry_close_hausdorff_distance"]
        & invalid_geometries_df["is_duckdb_polygon_and_gdal_linestring"],
        "duckdb_tags",
    ].apply(
        lambda x: any(
            (tag in reader.osm_way_polygon_features_config.all)
            or (
                tag in reader.osm_way_polygon_features_config.allowlist
                and value in reader.osm_way_polygon_features_config.allowlist[tag]
            )
            or (
                tag in reader.osm_way_polygon_features_config.denylist
                and value not in reader.osm_way_polygon_features_config.denylist[tag]
            )
            for tag, value in x.items()
        )
    )

    invalid_geometries_df = invalid_geometries_df.loc[
        ~(
            invalid_geometries_df["geometry_close_hausdorff_distance"]
            & invalid_geometries_df["is_duckdb_polygon_and_gdal_linestring"]
            & invalid_geometries_df["is_proper_filter_tag_value"]
        )
    ]
    if invalid_geometries_df.empty:
        return

    # Check if GDAL geometry is a polygon while DuckDB geometry is a linestring
    invalid_geometries_df.loc[
        invalid_geometries_df["geometry_close_hausdorff_distance"],
        "is_duckdb_linestring_and_gdal_polygon",
    ] = invalid_geometries_df.loc[invalid_geometries_df["geometry_close_hausdorff_distance"]].apply(
        lambda x: x.duckdb_geometry.geom_type
        in (
            "LineString",
            "MultiLineString",
        )
        and x.gdal_geometry.geom_type in ("Polygon", "MultiPolygon"),
        axis=1,
    )

    # Check if DuckDB geometry should be a linestring and not a polygon
    # based on features config
    invalid_geometries_df.loc[
        invalid_geometries_df["geometry_close_hausdorff_distance"]
        & invalid_geometries_df["is_duckdb_linestring_and_gdal_polygon"],
        "is_not_in_filter_tag_value",
    ] = invalid_geometries_df.loc[
        invalid_geometries_df["geometry_close_hausdorff_distance"]
        & invalid_geometries_df["is_duckdb_linestring_and_gdal_polygon"],
        "duckdb_tags",
    ].apply(
        lambda x: any(
            (tag not in reader.osm_way_polygon_features_config.all)
            and (
                tag not in reader.osm_way_polygon_features_config.allowlist
                or (
                    tag in reader.osm_way_polygon_features_config.allowlist
                    and value not in reader.osm_way_polygon_features_config.allowlist[tag]
                )
            )
            and (
                tag not in reader.osm_way_polygon_features_config.denylist
                or (
                    tag in reader.osm_way_polygon_features_config.denylist
                    and value in reader.osm_way_polygon_features_config.denylist[tag]
                )
            )
            for tag, value in x.items()
        )
    )

    invalid_geometries_df = invalid_geometries_df.loc[
        ~(
            invalid_geometries_df["geometry_close_hausdorff_distance"]
            & invalid_geometries_df["is_duckdb_linestring_and_gdal_polygon"]
            & invalid_geometries_df["is_not_in_filter_tag_value"]
        )
    ]
    if invalid_geometries_df.empty:
        return

    # Sometimes GDAL parses geometries incorrectly because of errors in OSM data
    # Examples of errors:
    # - overlapping inner ring with outer ring
    # - intersecting outer rings
    # - intersecting inner rings
    # - inner ring outside outer geometry
    # If we detect thattaht the difference between those geometries
    # lie inside the exterior of the geometry, we can assume that the OSM geometry
    # is improperly defined.
    invalid_geometries_df["duckdb_unioned_geometry_without_holes"] = invalid_geometries_df[
        "duckdb_geometry"
    ].apply(
        lambda x: (
            remove_interiors(unary_union(polygons))
            if len(polygons := extract_polygons_from_geometry(x)) > 0
            else None
        )
    )
    invalid_geometries_df["gdal_unioned_geometry_without_holes"] = invalid_geometries_df[
        "gdal_geometry"
    ].apply(
        lambda x: (
            remove_interiors(unary_union(polygons))
            if len(polygons := extract_polygons_from_geometry(x)) > 0
            else None
        )
    )
    invalid_geometries_df["both_polygon_geometries"] = (
        ~pd.isna(invalid_geometries_df["duckdb_unioned_geometry_without_holes"])
    ) & (~pd.isna(invalid_geometries_df["gdal_unioned_geometry_without_holes"]))

    # Check if the differences doesn't extend both geometries,
    # only one sided difference can be accepted
    invalid_geometries_df.loc[
        invalid_geometries_df["both_polygon_geometries"], "duckdb_geometry_fully_covered_by_gdal"
    ] = gpd.GeoSeries(
        invalid_geometries_df.loc[
            invalid_geometries_df["both_polygon_geometries"],
            "duckdb_unioned_geometry_without_holes",
        ]
    ).covered_by(
        gpd.GeoSeries(
            invalid_geometries_df.loc[
                invalid_geometries_df["both_polygon_geometries"],
                "gdal_unioned_geometry_without_holes",
            ]
        )
    )

    invalid_geometries_df.loc[
        invalid_geometries_df["both_polygon_geometries"], "gdal_geometry_fully_covered_by_duckdb"
    ] = gpd.GeoSeries(
        invalid_geometries_df.loc[
            invalid_geometries_df["both_polygon_geometries"], "gdal_unioned_geometry_without_holes"
        ]
    ).covered_by(
        gpd.GeoSeries(
            invalid_geometries_df.loc[
                invalid_geometries_df["both_polygon_geometries"],
                "duckdb_unioned_geometry_without_holes",
            ]
        )
    )

    invalid_geometries_df = invalid_geometries_df.loc[
        ~(
            invalid_geometries_df["duckdb_geometry_fully_covered_by_gdal"]
            | invalid_geometries_df["gdal_geometry_fully_covered_by_duckdb"]
        )
    ]
    if invalid_geometries_df.empty:
        return

    invalid_features = (
        invalid_geometries_df.drop(
            columns=["duckdb_tags", "source_tags", "duckdb_geometry", "gdal_geometry"]
        )
        .reset_index()
        .to_dict(orient="records")
    )

    assert not invalid_features, (
        f"Geometries aren't equal - ({[t[FEATURES_INDEX] for t in invalid_features]}). Full debug"
        f" output: ({invalid_features})"
    )
