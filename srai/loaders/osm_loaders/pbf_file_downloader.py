"""
PBF File Downloader.

This module contains a downloader capable of downloading a PBF files from multiple sources.
"""
import json
import time
import warnings
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, Hashable, List, Literal, Sequence, Union

import geopandas as gpd
import requests
import topojson as tp
from requests import HTTPError
from shapely.geometry import Polygon, mapping
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from tqdm import tqdm

from srai.constants import WGS84_CRS
from srai.geometry import (
    buffer_geometry,
    flatten_geometry,
    flatten_geometry_series,
    get_geometry_hash,
    remove_interiors,
)
from srai.loaders import download_file
from srai.loaders.osm_loaders.openstreetmap_extracts import (
    OpenStreetMapExtract,
    find_smallest_containing_geofabrik_extracts,
    find_smallest_containing_openstreetmap_fr_extracts,
)
from srai.loaders.osm_loaders.pbf_file_clipper import PbfFileClipper

PbfSourceLiteral = Literal["geofabrik", "openstreetmap_fr", "protomaps"]
PbfSourceExtractsFunctions: Dict[
    PbfSourceLiteral,
    Callable[[Union[BaseGeometry, BaseMultipartGeometry]], List[OpenStreetMapExtract]],
] = {
    "geofabrik": find_smallest_containing_geofabrik_extracts,
    "openstreetmap_fr": find_smallest_containing_openstreetmap_fr_extracts,
}


class PbfFileDownloader:
    """
    PbfFileDownloader.

    PBF(Protocolbuffer Binary Format)[1] file downloader is a downloader
    capable of downloading `*.osm.pbf` files with OSM data for a given area.

    This downloader can use multiple sources to extract a PBF file for a given region:
     - Geofabrik - free hosting service with PBF files http://download.geofabrik.de/.
     - OpenStreetMap.fr - free hosting service with PBF files https://download.openstreetmap.fr/.
     - Protomaps - (will be deprecated!) free download service for downloading an extract for
       an area of interest.


    References:
        1. https://wiki.openstreetmap.org/wiki/PBF_Format
    """

    PROTOMAPS_API_START_URL = "https://app.protomaps.com/downloads/osm"
    PROTOMAPS_API_DOWNLOAD_URL = "https://app.protomaps.com/downloads/{}/download"
    PROTOMAPS_MAX_WAIT_TIME_S = 300  # max 5 minutes for protomaps to generate an extract

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

    def __init__(
        self,
        download_source: PbfSourceLiteral = "protomaps",
        download_directory: Union[str, Path] = "files",
        switch_to_geofabrik_on_error: bool = True,
    ) -> None:
        """
        Initialize PbfFileDownloader.

        Args:
            download_source (PbfSourceLiteral, optional): Source to use when downloading PBF files.
                Can be one of: `geofabrik`, `openstreetmap_fr`, `protomaps`.
                Defaults to "protomaps".
            download_directory (Union[str, Path], optional): Directory where to save
                the downloaded `*.osm.pbf` files. Defaults to "files".
            switch_to_geofabrik_on_error (bool, optional): Flag whether to automatically
                switch `download_source` to 'geofabrik' if error occures. Defaults to `True`.
        """
        self.download_source = download_source
        self.download_directory = download_directory
        self.clipper = PbfFileClipper(working_directory=self.download_directory)
        self.switch_to_geofabrik_on_error = switch_to_geofabrik_on_error

    def download_pbf_files_for_regions_gdf(
        self, regions_gdf: gpd.GeoDataFrame
    ) -> Dict[Hashable, Sequence[Path]]:
        """
        Download PBF files for regions GeoDataFrame.

        Function will split each multipolygon into single polygons and download PBF files
        for each of them.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.

        Raises:
            ValueError: If provided geometries aren't shapely.geometry.Polygons.

        Returns:
            Dict[Hashable, Sequence[Path]]: List of Paths to downloaded PBF files per
                each region_id.
        """
        regions_mapping: Dict[Hashable, Sequence[Path]] = {}

        non_polygon_types = set(
            type(geometry)
            for geometry in flatten_geometry_series(regions_gdf.geometry)
            if not isinstance(geometry, Polygon)
        )
        if non_polygon_types:
            raise ValueError(f"Provided geometries aren't Polygons (found: {non_polygon_types})")

        try:
            if self.download_source == "protomaps":
                regions_mapping = self._download_pbf_files_for_polygons_from_protomaps(regions_gdf)
            elif self.download_source in PbfSourceExtractsFunctions:
                regions_mapping = self._download_pbf_files_for_polygons_from_existing_extracts(
                    regions_gdf
                )
        except Exception as err:
            if self.download_source != "geofabrik" and self.switch_to_geofabrik_on_error:
                warnings.warn(
                    f"Error occured ({err}). Auto-switching to 'geofabrik' download source.",
                    stacklevel=1,
                )
                regions_mapping = self._download_pbf_files_for_polygons_from_existing_extracts(
                    regions_gdf, override_to_geofabrik=True
                )
            else:
                error_message = str(err)
                if self.download_source != "geofabrik":
                    error_message += (
                        "\nPlease change the 'download_source' to"
                        " 'geofabrik' or other availablesource:\n"
                        " PbfDownloader(download_source='geofabrik', ...) or"
                        " OsmPbfLoader(download_source='geofabrik', ...)."
                    )
                raise RuntimeError(error_message) from err

        return regions_mapping

    def _download_pbf_files_for_polygons_from_existing_extracts(
        self, regions_gdf: gpd.GeoDataFrame, override_to_geofabrik: bool = False
    ) -> Dict[Hashable, Sequence[Path]]:
        regions_mapping: Dict[Hashable, Sequence[Path]] = {}

        unary_union_geometry = regions_gdf.geometry.unary_union

        if override_to_geofabrik:
            extract_function = PbfSourceExtractsFunctions["geofabrik"]
        else:
            extract_function = PbfSourceExtractsFunctions[self.download_source]
        extracts = extract_function(unary_union_geometry)

        downloaded_pbf_files = []

        for extract in extracts:
            pbf_file_path = Path(self.download_directory).resolve() / f"{extract.id}.osm.pbf"

            download_file(url=extract.url, fname=pbf_file_path.as_posix(), force_download=False)

            downloaded_pbf_files.append(pbf_file_path)

        polygons = flatten_geometry(unary_union_geometry)

        for region_id, row in regions_gdf.iterrows():
            polygons = flatten_geometry(row.geometry)
            regions_mapping[region_id] = [
                self.clipper.clip_pbf_file(polygon, downloaded_pbf_files) for polygon in polygons
            ]

        return regions_mapping

    def _download_pbf_files_for_polygons_from_protomaps(
        self, regions_gdf: gpd.GeoDataFrame
    ) -> Dict[Hashable, Sequence[Path]]:
        regions_mapping: Dict[Hashable, Sequence[Path]] = {}

        for region_id, row in regions_gdf.iterrows():
            polygons = flatten_geometry(row.geometry)
            regions_mapping[region_id] = [
                self._download_pbf_file_for_polygon_from_protomaps(
                    polygon, region_id, polygon_id + 1
                )
                for polygon_id, polygon in enumerate(polygons)
            ]

        return regions_mapping

    def _download_pbf_file_for_polygon_from_protomaps(
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
        geometry_hash = get_geometry_hash(polygon)
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
                "User-Agent": "SRAI Python package (https://github.com/kraina-ai/srai)",
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
            except KeyError as err:
                error_message = (
                    f"Error from the 'Protomaps' service: {json.dumps(start_extract_result)}."
                )
                raise RuntimeError(error_message) from err

            with tqdm() as pbar:
                status_response: Dict[str, Any] = {}
                cells_total = 0
                nodes_total = 0
                elems_total = 0
                start_time = time.time()
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

                    if (time.time() - start_time) > self.PROTOMAPS_MAX_WAIT_TIME_S:
                        error_message = (
                            "'Protomaps' service took too long to generate an extract"
                            f" ({self.PROTOMAPS_MAX_WAIT_TIME_S}s)."
                        )
                        raise RuntimeError(error_message)

            try:
                download_file(
                    url=self.PROTOMAPS_API_DOWNLOAD_URL.format(extraction_uuid),
                    fname=pbf_file_path.as_posix(),
                )
            except HTTPError as err:
                error_message = f"Error from the 'Protomaps' service: {err.response}."
                raise RuntimeError(error_message) from err

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
