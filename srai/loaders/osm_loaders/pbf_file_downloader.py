"""
PBF File Downloader.

This module contains a downloader capable of downloading a PBF file from a free Protomaps service.
"""
import hashlib
import json
import warnings
from pathlib import Path
from time import sleep
from typing import Any, Dict, Hashable, Sequence, Union

import geopandas as gpd
import requests
import shapely.wkt as wktlib
import topojson as tp
from shapely.geometry import Polygon, mapping
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.constants import WGS84_CRS
from srai.geometry import buffer_geometry, flatten_geometry, remove_interiors
from srai.loaders import download_file


class PbfFileDownloader:
    """
    PbfFileDownloader.

    PBF(Protocolbuffer Binary Format)[1] file downloader is a downloader
    capable of downloading `*.osm.pbf` files with OSM data for a given area.

    This downloader uses free Protomaps[2] download service to extract a PBF
    file for a given region.

    References:
        1. https://wiki.openstreetmap.org/wiki/PBF_Format
        2. https://protomaps.com/
    """

    PROTOMAPS_API_START_URL = "https://app.protomaps.com/downloads/osm"
    PROTOMAPS_API_DOWNLOAD_URL = "https://app.protomaps.com/downloads/{}/download"

    _PBAR_FORMAT = "[{}] Downloading pbf file #{} ({})"

    SIMPLIFICATION_TOLERANCE_VALUES = [
        1e-07,
        2e-07,
        5e-07,
        1e-06,
        2e-06,
        5e-06,
        1e-05,
        2e-05,
        5e-05,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
    ]

    def __init__(self, download_directory: Union[str, Path] = "files") -> None:
        """
        Initialize PbfFileDownloader.

        Args:
            download_directory (Union[str, Path], optional): Directory where to save
                the downloaded `*.osm.pbf` files. Defaults to "files".
        """
        self.download_directory = download_directory

    def download_pbf_files_for_regions_gdf(
        self, regions_gdf: gpd.GeoDataFrame
    ) -> Dict[Hashable, Sequence[Path]]:
        """
        Download PBF files for regions GeoDataFrame.

        Function will split each multipolygon into single polygons and download PBF files
        for each of them.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.

        Returns:
            Dict[Hashable, Sequence[Path]]: List of Paths to downloaded PBF files per
                each region_id.
        """
        regions_mapping: Dict[Hashable, Sequence[Path]] = {}

        for region_id, row in regions_gdf.iterrows():
            polygons = flatten_geometry(row.geometry)
            regions_mapping[region_id] = [
                self.download_pbf_file_for_polygon(polygon, region_id, polygon_id + 1)
                for polygon_id, polygon in enumerate(polygons)
            ]

        return regions_mapping

    def download_pbf_file_for_polygon(
        self, polygon: Polygon, region_id: str = "OSM", polygon_id: int = 1
    ) -> Path:
        """
        Download PBF file for a single Polygon.

        Function will buffer polygon by 50 meters, simplify exterior boundary to be
        below 1000 points (which is a limit of Protomaps API) and close all holes within it.

        Boundary of the polygon will be sent to Protomaps service and an `*.osm.pbf` file
        will be downloaded with a hash based on WKT representation of the parsed polygon.
        If file exists, it won't be downloaded again.

        Args:
            polygon (Polygon): Polygon boundary of an area to be extracted.
            region_id (str, optional): Region name to be set in progress bar.
                Defaults to "OSM".
            polygon_id (int, optional): Polygon number to be set in progress bar.
                Defaults to 1.

        Returns:
            Path: Path to a downloaded `*.osm.pbf` file.
        """
        geometry_hash = self._get_geometry_hash(polygon)
        pbf_file_path = Path(self.download_directory).resolve() / f"{geometry_hash}.osm.pbf"

        if not pbf_file_path.exists():  # pragma: no cover
            boundary_polygon = self._prepare_polygon_for_download(polygon)
            geometry_geojson = mapping(boundary_polygon)

            s = requests.Session()

            req = s.get(url=self.PROTOMAPS_API_START_URL)

            csrf_token = req.cookies["csrftoken"]
            headers = {
                "Referer": self.PROTOMAPS_API_START_URL,
                "Cookie": f"csrftoken={csrf_token}",
                "X-CSRFToken": csrf_token,
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "SRAI Python package (https://github.com/srai-lab/srai)",
            }
            request_payload = {
                "region": {"type": "geojson", "data": geometry_geojson},
                "name": geometry_hash,
            }

            start_extract_request = s.post(
                url=self.PROTOMAPS_API_START_URL,
                json=request_payload,
                headers=headers,
                cookies=dict(csrftoken=csrf_token),
            )
            start_extract_request.raise_for_status()

            start_extract_result = start_extract_request.json()
            try:
                extraction_uuid = start_extract_result["uuid"]
                status_check_url = start_extract_result["url"]
            except KeyError:
                warnings.warn(json.dumps(start_extract_result), stacklevel=2)
                raise

            with tqdm() as pbar:
                status_response: Dict[str, Any] = {}
                cells_total = 0
                nodes_total = 0
                elems_total = 0
                while not status_response.get("Complete", False):
                    sleep(0.5)
                    status_response = s.get(url=status_check_url).json()
                    cells_total = max(cells_total, status_response.get("CellsTotal", 0))
                    nodes_total = max(nodes_total, status_response.get("NodesTotal", 0))
                    elems_total = max(elems_total, status_response.get("ElemsTotal", 0))

                    cells_prog = status_response.get("CellsProg", None)
                    nodes_prog = status_response.get("NodesProg", None)
                    elems_prog = status_response.get("ElemsProg", None)

                    if cells_total > 0 and cells_prog is not None and cells_prog < cells_total:
                        pbar.set_description(
                            self._PBAR_FORMAT.format(region_id, polygon_id, "Cells")
                        )
                        pbar.total = cells_total + nodes_total + elems_total
                        pbar.n = cells_prog
                    elif nodes_total > 0 and nodes_prog is not None and nodes_prog < nodes_total:
                        pbar.set_description(
                            self._PBAR_FORMAT.format(region_id, polygon_id, "Nodes")
                        )
                        pbar.total = cells_total + nodes_total + elems_total
                        pbar.n = cells_total + nodes_prog
                    elif elems_total > 0 and elems_prog is not None and elems_prog < elems_total:
                        pbar.set_description(
                            self._PBAR_FORMAT.format(region_id, polygon_id, "Elements")
                        )
                        pbar.total = cells_total + nodes_total + elems_total
                        pbar.n = cells_total + nodes_total + elems_prog
                    else:
                        pbar.total = cells_total + nodes_total + elems_total
                        pbar.n = cells_total + nodes_total + elems_total

                    pbar.refresh()

            download_file(
                url=self.PROTOMAPS_API_DOWNLOAD_URL.format(extraction_uuid),
                fname=pbf_file_path.as_posix(),
            )

        return pbf_file_path

    def _prepare_polygon_for_download(self, polygon: Polygon) -> Polygon:
        """
        Prepare polygon for download.

        Function buffers the polygon, closes internal holes and simplifies its boundary to 1000
        points.

        Makes sure that the generated polygon with fully cover the original one by increasing the
        buffer size incrementally. Buffering is applied to the last simplified geometry to speed up
        the process.
        """
        is_fully_covered = False
        buffer_size_meters = 50

        polygon_to_buffer = polygon

        while not is_fully_covered:
            buffered_polygon = buffer_geometry(polygon_to_buffer, meters=buffer_size_meters)
            simplified_polygon = self._simplify_polygon(buffered_polygon, 1000)
            closed_polygon = remove_interiors(simplified_polygon)
            is_fully_covered = polygon.covered_by(closed_polygon)
            buffer_size_meters += 50

            polygon_to_buffer = closed_polygon

        return closed_polygon

    def _simplify_polygon(self, polygon: Polygon, exterior_max_points: int = 1000) -> Polygon:
        """Simplify a polygon boundary to up to provided number of points."""
        simplified_polygon = polygon

        for simplify_tolerance in self.SIMPLIFICATION_TOLERANCE_VALUES:
            simplified_polygon = (
                tp.Topology(
                    polygon,
                    toposimplify=simplify_tolerance,
                    prevent_oversimplify=True,
                )
                .to_gdf(winding_order="CW_CCW", crs=WGS84_CRS, validate=True)
                .geometry[0]
            )

            if len(simplified_polygon.exterior.coords) < exterior_max_points:
                break

        if len(simplified_polygon.exterior.coords) > exterior_max_points:
            simplified_polygon = polygon.convex_hull

        if len(simplified_polygon.exterior.coords) > exterior_max_points:
            simplified_polygon = polygon.minimum_rotated_rectangle

        return simplified_polygon

    def _get_geometry_hash(self, geometry: BaseGeometry) -> str:
        """Generate SHA256 hash based on WKT representation of the polygon."""
        wkt_string = wktlib.dumps(geometry)
        h = hashlib.new("sha256")
        h.update(wkt_string.encode())
        return h.hexdigest()
