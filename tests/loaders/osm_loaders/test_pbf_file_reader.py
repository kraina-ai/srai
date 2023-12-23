"""Tests for PbfFileReader."""

import platform
import re
import subprocess
import warnings
from collections.abc import Hashable, Iterable
from distutils.spawn import find_executable
from pathlib import Path
from typing import Any, Optional, Union, cast
from unittest import TestCase

import duckdb
import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
import six
from parametrization import Parametrization as P
from shapely import get_num_geometries, get_num_points, hausdorff_distance
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon
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
                value = None
            else:
                raise ValueError("Malformed hstore value starting at position %d" % offset)

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


def iou_metric(geom_a: BaseGeometry, geom_b: BaseGeometry) -> float:
    """Calculate IoU metric for geometries."""
    if geom_a.geom_type not in (
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    ) or geom_b.geom_type not in ("Polygon", "MultiPolygon", "GeometryCollection"):
        return 0
    intersection = geom_a.intersection(geom_b).area
    union = geom_a.area + geom_b.area - intersection
    if union == 0:
        return 0
    return float(intersection / union)


def calculate_total_points(geom: BaseGeometry) -> int:
    """Calculate total number of points in a geometry."""
    if isinstance(geom, (Point, MultiPoint)):
        return int(get_num_geometries(geom))

    line_strings = []

    if isinstance(geom, LineString):
        line_strings.append(geom)
    elif isinstance(geom, Polygon):
        line_strings.append(geom.exterior)
        line_strings.extend(geom.interiors)
    else:  # MultiLineString, MultiPolygon, GeometryCollection
        return sum(calculate_total_points(sub_geom) for sub_geom in geom.geoms)

    return sum(get_num_points(line_string) for line_string in line_strings)


def check_if_relation_in_osm_is_valid(pbf_file: str, relation_id: str) -> bool:
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


def check_if_two_geometries_are_similar(
    gdal_row_index: Hashable, duckdb_row: pd.Series, gdal_row: pd.Series, reader: PbfFileReader
) -> tuple[bool, dict[str, Any]]:
    """Check if two goemetries are similar based on multiple critera."""
    duckdb_geometry = duckdb_row.geometry
    gdal_geometry = gdal_row.geometry

    # Check if both geometries are closed or open
    geometry_both_closed_or_not = duckdb_geometry.is_closed == gdal_geometry.is_closed
    # Check geometries equality - same geom type, same points
    geometry_equal = duckdb_geometry.equals(gdal_geometry)

    if geometry_both_closed_or_not and geometry_equal:
        return True, {}

    tolerance = 0.5 * 10 ** (-6)
    # Check if geometries are almost equal - same geom type, same points
    geometry_almost_equal = duckdb_geometry.equals_exact(gdal_geometry, tolerance)

    if geometry_both_closed_or_not and geometry_almost_equal:
        return True, {}

    # Check geometries overlap if polygons - slight misalingment between points,
    # but marginal
    iou_value = iou_metric(duckdb_geometry, gdal_geometry)
    geometry_iou_near_one = iou_value >= (1 - tolerance)

    if geometry_both_closed_or_not and geometry_iou_near_one:
        return True, {}

    # Check if points lay near each other - regardless of geometry type
    # (Polygon vs LineString)
    hausdorff_distance_value = hausdorff_distance(duckdb_geometry, gdal_geometry, densify=0.5)
    geometry_close_hausdorff_distance = hausdorff_distance_value < 1e-10

    # Check if GDAL geometry is a linestring while DuckDB geometry is a polygon
    is_duckdb_polygon_and_gdal_linestring = duckdb_geometry.geom_type in (
        "Polygon",
        "MultiPolygon",
    ) and gdal_geometry.geom_type in ("LineString", "MultiLineString")

    # Check if DuckDB geometry can be a polygon and not a linestring
    # based on features config
    is_proper_filter_tag_value = any(
        (tag in reader.osm_way_polygon_features_config.all)
        or (
            tag in reader.osm_way_polygon_features_config.allowlist
            and value in reader.osm_way_polygon_features_config.allowlist[tag]
        )
        or (
            tag in reader.osm_way_polygon_features_config.denylist
            and value not in reader.osm_way_polygon_features_config.denylist[tag]
        )
        for tag, value in duckdb_row.tags.items()
    )

    # Check if geometries have the same number of points
    duckdb_geometry_points = calculate_total_points(duckdb_geometry)
    gdal_geometry_points = calculate_total_points(gdal_geometry)
    same_number_of_points = duckdb_geometry_points == gdal_geometry_points

    duckdb_polygon_and_gdal_linestring_but_geometried_are_equal = (
        geometry_close_hausdorff_distance
        and is_duckdb_polygon_and_gdal_linestring
        and is_proper_filter_tag_value
        and same_number_of_points
    )

    if duckdb_polygon_and_gdal_linestring_but_geometried_are_equal:
        return True, {}

    # Check if GDAL geometry is a polygon while DuckDB geometry is a linestring
    is_duckdb_linestring_and_gdal_polygon = duckdb_geometry.geom_type in (
        "LineString",
        "MultiLineString",
    ) and gdal_geometry.geom_type in ("Polygon", "MultiPolygon")

    # Check if DuckDB geometry should be a linestring and not a polygon
    # based on features config
    is_not_in_filter_tag_value = any(
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
        for tag, value in duckdb_row.tags.items()
    )

    duckdb_linestring_and_gdal_polygon_but_geometried_are_equal = (
        geometry_close_hausdorff_distance
        and is_duckdb_linestring_and_gdal_polygon
        and is_not_in_filter_tag_value
        and same_number_of_points
    )

    if duckdb_linestring_and_gdal_polygon_but_geometried_are_equal:
        return True, {}

    # Sometimes GDAL parses geometries incorrectly because of errors in OSM data
    # Examples of errors:
    # - overlapping inner ring with outer ring
    # - intersecting outer rings
    # - intersecting inner rings
    # - inner ring outside outer geometry
    # If we detect thattaht the difference between those geometries
    # lie inside the exterior of the geometry, we can assume that the OSM geometry
    # is improperly defined.
    gdal_geometry_fully_covered_by_duckdb = False
    duckdb_geometry_fully_covered_by_gdal = False

    duckdb_polygon_geometries = extract_polygons_from_geometry(duckdb_geometry)
    gdal_polygon_geometries = extract_polygons_from_geometry(gdal_geometry)

    if duckdb_polygon_geometries and gdal_polygon_geometries:
        duckdb_unioned_geometry = unary_union(duckdb_polygon_geometries)
        gdal_unioned_geometry = unary_union(gdal_polygon_geometries)
        duckdb_unioned_geometry_without_holes = remove_interiors(duckdb_unioned_geometry)
        gdal_unioned_geometry_without_holes = remove_interiors(gdal_unioned_geometry)

        # Check if the differences doesn't extend both geometries,
        # only one sided difference can be accepted
        gdal_geometry_fully_covered_by_duckdb = gdal_unioned_geometry_without_holes.covered_by(
            duckdb_unioned_geometry_without_holes
        )
        duckdb_geometry_fully_covered_by_gdal = duckdb_unioned_geometry_without_holes.covered_by(
            gdal_unioned_geometry_without_holes
        )

    duckdb_polygon_geometries = extract_polygons_from_geometry(duckdb_geometry)
    gdal_polygon_geometries = extract_polygons_from_geometry(gdal_geometry)

    if gdal_geometry_fully_covered_by_duckdb or duckdb_geometry_fully_covered_by_gdal:
        warnings.warn(
            f"Detected invalid relation defined in OSM ({gdal_row_index})",
            stacklevel=1,
        )
        return True, {}

    full_debug_dict = {
        FEATURES_INDEX: gdal_row_index,
        "geometry_both_closed_or_not": geometry_both_closed_or_not,
        "geometry_equal": geometry_equal,
        "geometry_almost_equal": geometry_almost_equal,
        "geometry_iou_near_one": geometry_iou_near_one,
        "iou_value": iou_value,
        "geometry_close_hausdorff_distance": geometry_close_hausdorff_distance,
        "hausdorff_distance_value": hausdorff_distance_value,
        "is_duckdb_polygon_and_gdal_linestring": is_duckdb_polygon_and_gdal_linestring,
        "is_duckdb_linestring_and_gdal_polygon": is_duckdb_linestring_and_gdal_polygon,
        "duckdb_geom_type": duckdb_geometry.geom_type,
        "gdal_geom_type": gdal_geometry.geom_type,
        "is_proper_filter_tag_value": is_proper_filter_tag_value,
        "is_not_in_filter_tag_value": is_not_in_filter_tag_value,
        "same_number_of_points": same_number_of_points,
        "duckdb_geometry_points": duckdb_geometry_points,
        "gdal_geometry_points": gdal_geometry_points,
        "duckdb_polygon_and_gdal_linestring_but_geometried_are_equal": (
            duckdb_polygon_and_gdal_linestring_but_geometried_are_equal
        ),
        "duckdb_linestring_and_gdal_polygon_but_geometried_are_equal": (
            duckdb_linestring_and_gdal_polygon_but_geometried_are_equal
        ),
        "duckdb_geometry_fully_covered_by_gdal": duckdb_geometry_fully_covered_by_gdal,
        "gdal_geometry_fully_covered_by_duckdb": gdal_geometry_fully_covered_by_duckdb,
    }

    return False, full_debug_dict


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
@P.case("Sierra Leone", "sierra-leone")  # type: ignore
@P.case("Greenland", "greenland")  # type: ignore
@P.case("El Salvador", "el-salvador")  # type: ignore
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
    non_relations_missing_in_duckdb = [
        feature_id for feature_id in missing_in_duckdb if not feature_id.startswith("relation/")
    ]
    valid_relations_missing_in_duckdb = [
        feature_id
        for feature_id in missing_in_duckdb
        if feature_id.startswith("relation/")
        and check_if_relation_in_osm_is_valid(
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

    if invalid_relations_missing_in_duckdb:
        warnings.warn(
            "Invalid relations exists in OSM GDAL data extract"
            f" ({invalid_relations_missing_in_duckdb})",
            stacklevel=1,
        )

    invalid_features = []

    for gdal_row_index in gdal_index:
        if gdal_row_index in invalid_relations_missing_in_duckdb:
            continue

        duckdb_row = duckdb_gdf.loc[gdal_row_index]
        gdal_row = gdal_gdf.loc[gdal_row_index]
        duckdb_tags = {k: v for k, v in duckdb_row.tags.items() if k != "area"}
        gdal_tags = {k: v for k, v in gdal_row.tags.items() if k != "area"}

        # Check tags
        tags_keys_difference = set(duckdb_tags.keys()).symmetric_difference(gdal_tags.keys())
        assert not tags_keys_difference, (
            f"Tags keys aren't equal. ({gdal_row_index}, {tags_keys_difference},"
            f" {duckdb_tags.keys()}, {gdal_tags.keys()})"
        )
        ut.assertDictEqual(
            duckdb_tags,
            gdal_tags,
            f"Tags aren't equal. ({gdal_row_index})",
        )

        try:
            are_geometries_similar, full_debug_dict = check_if_two_geometries_are_similar(
                gdal_row_index=gdal_row_index,
                duckdb_row=duckdb_row,
                gdal_row=gdal_row,
                reader=reader,
            )

            if not are_geometries_similar:
                invalid_features.append(full_debug_dict)
        except Exception as ex:
            raise RuntimeError(f"Unexpected error for feature: {gdal_row_index}") from ex

    assert not invalid_features, (
        f"Geometries aren't equal - ({[t[FEATURES_INDEX] for t in invalid_features]}). Full debug"
        f" output: ({invalid_features})"
    )
