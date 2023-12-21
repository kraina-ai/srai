"""Tests for PbfFileReader."""

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, cast
from unittest import TestCase

import duckdb
import gdaltools
import geopandas as gpd
import pyogrio
import pytest
import six
from parametrization import Parametrization as P
from shapely import get_num_geometries, get_num_points, hausdorff_distance
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry

from srai.constants import FEATURES_INDEX
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
        gdaltools.ogr2ogr()._get_command(),
        str(output_file),
        str(input_file),
        layer_name,
        "-oo",
        f"CONFIG_FILE={config_file}",
    ]
    print(args)
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
        print(layer_name, len(gdf))

    final_gdf = gpd.pd.concat(gdfs)
    final_gdf = final_gdf[~final_gdf["all_tags"].isnull()]
    final_gdf["tags"] = final_gdf["all_tags"].apply(parse_hstore_tags)
    non_relations = ~final_gdf[FEATURES_INDEX].str.startswith("relation/")
    relations = final_gdf[FEATURES_INDEX].str.startswith("relation/")
    matching_relations = relations & final_gdf["tags"].apply(
        lambda x: x.get("type") in ("boundary", "multipolygon")
    )
    final_gdf = final_gdf[non_relations | matching_relations]
    return final_gdf[[FEATURES_INDEX, "tags", "geometry"]].set_index(FEATURES_INDEX)


def iou_metric(geom_a: BaseGeometry, geom_b: BaseGeometry) -> float:
    """Calculate IoU metric for geometries."""
    if geom_a.geom_type not in ("Polygon", "MultiPolygon") or geom_b.geom_type not in (
        "Polygon",
        "MultiPolygon",
    ):
        return 0
    intersection = geom_a.intersection(geom_b).area
    union = geom_a.area + geom_b.area - intersection
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


@P.parameters("extract_name")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
@P.case("Monaco", "monaco")  # type: ignore
# @P.case("Cyprus", "cyprus")  # type: ignore
# @P.case("Cambodia", "cambodia")  # type: ignore
# @P.case("Maldives", "maldives")  # type: ignore
# @P.case("Seychelles", "seychelles")  # type: ignore
# @P.case("Sierra Leone", "sierra-leone")  # type: ignore
# @P.case("Greenland", "greenland")  # type: ignore
# @P.case("El Salvador", "el-salvador")  # type: ignore
# @P.case("Panama", "panama")  # type: ignore
# @P.case("Fiji", "fiji")  # type: ignore
# @P.case("Kiribati", "kiribati")  # type: ignore
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

    assert (
        not non_relations_missing_in_duckdb
    ), f"Missing non relation features in PbfFileReader ({non_relations_missing_in_duckdb})"

    assert (
        not valid_relations_missing_in_duckdb
    ), f"Missing valid relation features in PbfFileReader ({valid_relations_missing_in_duckdb})"

    invalid_features = []

    for gdal_row_index in gdal_index:
        duckdb_row = duckdb_gdf.loc[gdal_row_index]
        gdal_row = gdal_gdf.loc[gdal_row_index]
        duckdb_tags = duckdb_row.tags
        gdal_tags = duckdb_row.tags

        # Check tags
        tags_keys_difference = set(duckdb_tags.keys()).symmetric_difference(gdal_tags.keys())
        assert (
            not tags_keys_difference
        ), f"Tags keys aren't equal. ({gdal_row_index}, {tags_keys_difference})"
        ut.assertDictEqual(
            duckdb_tags,
            gdal_tags,
            f"Tags aren't equal. ({gdal_row_index})",
        )

        # Check if both geometries are closed or open
        geometry_both_closed_or_not = duckdb_row.geometry.is_closed == gdal_row.geometry.is_closed

        tolerance = 0.5 * 10 ** (-6)
        # Check geometries equality - same geom type, same points
        geometry_equal = duckdb_row.geometry.equals(gdal_row.geometry)
        geometry_almost_equal = duckdb_row.geometry.equals_exact(gdal_row.geometry, tolerance)

        # Check geometries overlap if polygons - slight misalingment between points, but marginal
        iou_value = iou_metric(duckdb_row.geometry, gdal_row.geometry)
        geometry_iou_near_one = iou_value >= (1 - tolerance)

        # Check if points lay near each other - regardless of geometry type (Polygon vs LineString)
        hausdorff_distance_value = hausdorff_distance(
            duckdb_row.geometry, gdal_row.geometry, densify=0.5
        )
        geometry_close_hausdorff_distance = hausdorff_distance_value < 1e-10

        # Check if GDAL geometry is a linestring while DuckDB geometry is a polygon
        is_different_geometry_type = duckdb_row.geometry.geom_type in (
            "Polygon",
            "MultiPolygon",
        ) and gdal_row.geometry.geom_type in ("LineString", "MultiLineString")

        # Check if DuckDB geometry can be a polygon and not a linestring based on features config
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
            for tag, value in duckdb_tags.items()
        )

        # Check if geometries have the same number of points
        duckdb_geometry_points = calculate_total_points(duckdb_row.geometry)
        gdal_geometry_points = calculate_total_points(gdal_row.geometry)
        same_number_of_points = duckdb_geometry_points == gdal_geometry_points

        # Combine conditions
        geometries_are_equal_and_the_same_type = geometry_both_closed_or_not and (
            geometry_equal or geometry_almost_equal or geometry_iou_near_one
        )
        geometries_are_equal_but_different_type = (
            geometry_close_hausdorff_distance
            and is_different_geometry_type
            and is_proper_filter_tag_value
            and same_number_of_points
        )

        full_debug_dict = {
            FEATURES_INDEX: gdal_row_index,
            "geometries_are_equal_and_the_same_type": geometries_are_equal_and_the_same_type,
            "geometries_are_equal_but_different_type": geometries_are_equal_but_different_type,
            "geometry_both_closed_or_not": geometry_both_closed_or_not,
            "geometry_equal": geometry_equal,
            "geometry_almost_equal": geometry_almost_equal,
            "geometry_iou_near_one": geometry_iou_near_one,
            "iou_value": iou_value,
            "geometry_close_hausdorff_distance": geometry_close_hausdorff_distance,
            "hausdorff_distance_value": hausdorff_distance_value,
            "is_different_geometry_type": is_different_geometry_type,
            "duckdb_geom_type": duckdb_row.geometry.geom_type,
            "gdal_geom_type": gdal_row.geometry.geom_type,
            "is_proper_filter_tag_value": is_proper_filter_tag_value,
            "same_number_of_points": same_number_of_points,
            "duckdb_geometry_points": duckdb_geometry_points,
            "gdal_geometry_points": gdal_geometry_points,
        }

        if (
            not geometries_are_equal_and_the_same_type
            and not geometries_are_equal_but_different_type
        ):
            invalid_features.append(full_debug_dict)

    assert not invalid_features, (
        f"Geometries aren't equal - ({[t[FEATURES_INDEX] for t in invalid_features]}). Full debug"
        f" output: ({invalid_features})"
    )
