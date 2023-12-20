"""Tests for PbfFileReader."""

from pathlib import Path
from typing import cast
from unittest import TestCase

import duckdb
import geopandas as gpd
import pghstore
import pyogrio
import pytest
from parametrization import Parametrization as P
from shapely import get_num_geometries, get_num_points, hausdorff_distance
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry

from srai.loaders.download import download_file
from srai.loaders.osm_loaders.filters import GEOFABRIK_LAYERS, HEX2VEC_FILTER, OsmTagsFilter
from srai.loaders.osm_loaders.pbf_file_reader import PbfFileReader

ut = TestCase()


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


def read_features_with_pyogrio(pbf_file: Path) -> gpd.GeoDataFrame:
    """Read features from *.osm.pbf file using pyogrio."""
    gdal_options = dict(
        INTERLEAVED_READING=True,
        CONFIG_FILE=str(Path(__file__).parent / "test_files" / "osmconf.ini"),
        use_arrow=True,
    )
    gdfs = []
    for layer_info in pyogrio.list_layers(pbf_file):
        layer_name = layer_info[0]

        gdf = pyogrio.read_dataframe(
            pbf_file,
            layer=layer_name,
            columns=["osm_id", "osm_way_id", "all_tags", "geometry"],
            **gdal_options,
        )
        assert len(gdf) > 0, f"Layer {layer_name} is empty."

        if layer_name == "points":
            gdf["feature_id"] = "node/" + gdf["osm_id"]
        elif layer_name == "lines":
            gdf["feature_id"] = "way/" + gdf["osm_id"]
        elif layer_name in ("multilinestrings", "other_relations"):
            gdf["feature_id"] = "relation/" + gdf["osm_id"]
        elif layer_name == "multipolygons":
            gdf["feature_id"] = gdf.apply(
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
    final_gdf["tags"] = final_gdf["all_tags"].apply(pghstore.loads)
    non_relations = ~final_gdf["feature_id"].str.startswith("relation/")
    relations = final_gdf["feature_id"].str.startswith("relation/")
    matching_relations = relations & final_gdf["tags"].apply(
        lambda x: x.get("type") in ("boundary", "multipolygon")
    )
    final_gdf = final_gdf[non_relations | matching_relations]
    return final_gdf[["feature_id", "tags", "geometry"]].set_index("feature_id")


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


@P.parameters("pbf_file_name", "pbf_file_download_url")  # type: ignore
@P.case(  # type: ignore
    "Monaco",
    "files/monaco.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/monaco-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Cyprus",
    "files/cyprus.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/cyprus-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Cambodia",
    "files/cambodia.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/cambodia-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Maldives",
    "files/maldives.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/maldives-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Seychelles",
    "files/seychelles.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/seychelles-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Sierra Leone",
    "files/sierra-leone.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/sierra-leone-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Greenland",
    "files/greenland.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/greenland-latest.osm.pbf",
)
@P.case(  # type: ignore
    "El Salvador",
    "files/el-salvador.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/el-salvador-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Panama",
    "files/panama.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/panama-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Fiji",
    "files/fiji.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/fiji-latest.osm.pbf",
)
@P.case(  # type: ignore
    "Kiribati",
    "files/kiribati.osm.pbf",
    "https://github.com/kraina-ai/srai-test-files/raw/main/files/kiribati-latest.osm.pbf",
)
def test_gdal_parity(pbf_file_name: str, pbf_file_download_url: str) -> None:
    """Test if loaded data is similar to GDAL results."""
    pbf_file_path = Path(__file__).parent / pbf_file_name
    download_file(pbf_file_download_url, str(pbf_file_path), force_download=True)

    reader = PbfFileReader()
    duckdb_gdf = reader.get_features_gdf([pbf_file_path], explode_tags=False, ignore_cache=True)
    gdal_gdf = read_features_with_pyogrio(pbf_file_path)

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

        assert (
            geometries_are_equal_and_the_same_type or geometries_are_equal_but_different_type
        ), f"{gdal_row_index} geometries aren't equal. ({full_debug_dict})"
