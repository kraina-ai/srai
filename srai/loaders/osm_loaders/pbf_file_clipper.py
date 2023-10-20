"""
PBF File Clipper.

This module contains a clipper capable of clipping a PBF file to a smaller size using osmconvert or
osmosis CLI tools.
"""
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from sys import platform
from typing import Sequence, Union

import numpy as np
import tqdm
from shapely.geometry import MultiPolygon, Polygon

from srai.geometry import get_geometry_hash
from srai.loaders.download import download_file


class PbfFileClipper:
    """
    PbfFileClipper.

    PBF(Protocolbuffer Binary Format)[1] file clipper is a tool
    capable of clipping `*.osm.pbf` files with OSM data for a given area to make it smaller.

    Class will automatically download those CLI tools and execute them from Python code.

    This clipper uses two tools to clip a PBF file for a given region:
     - Osmconvert [2] - used on Linux systems
     - Osmosis [3] - used on OS X and Windows systems

    References:
        1. https://wiki.openstreetmap.org/wiki/PBF_Format
        2. https://wiki.openstreetmap.org/wiki/Osmconvert
        3. https://wiki.openstreetmap.org/wiki/Osmosis
    """

    def __init__(self, working_directory: Union[str, Path] = "files") -> None:
        """
        Initialize PbfFileClipper.

        Args:
            working_directory (Union[str, Path], optional): Directory where to save
                the parsed `*.osm.pbf` files. Defaults to "files".
        """
        self.working_directory = Path(working_directory)

        # Tools for clipping OSM files into smaller regions
        # osmconvert is faster, but works on Linux
        self.OSMCONVERT_PATH = (self.working_directory / "osmconvert_tool/osmconvert").as_posix()
        # osmosis is slower, but also works on Mac OS
        self.OSMOSIS_DIRECTORY_PATH = (self.working_directory / "osmosis_tool").as_posix()
        self.OSMOSIS_EXECUTABLE_PATH = (
            self.working_directory / "osmosis_tool/bin/osmosis"
        ).as_posix()

    def clip_pbf_file(
        self,
        geometry: Polygon,
        pbf_files: Sequence[Union[str, "os.PathLike[str]"]],
    ) -> Path:
        """
        Clip a list of PBF files using a given geometry and merge them into a single file.

        Args:
            geometry (Polygon): Geometry to be used for clipping.
            pbf_files (Sequence[Union[str, os.PathLike[str]]]): List of PBF files to be processed.

        Raises:
            RuntimeError: If system is unrecognized (not one of linux, linux2, darwin or win32)

        Returns:
            Path: Location of merged PBF file.
        """
        geometry_hash = get_geometry_hash(geometry)
        final_osm_path = (self.working_directory.resolve() / f"{geometry_hash}.osm.pbf").as_posix()

        if Path(final_osm_path).exists():
            return Path(final_osm_path)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)

            poly_path = (tmp_dir_path.resolve() / f"poly_files/{geometry_hash}.poly").as_posix()

            # create poly file
            self._generate_poly_file(geometry, poly_path)

            if platform in ("linux", "linux2"):
                self._download_osmconvert_tool()
            elif platform in ("darwin", "win32"):
                self._download_osmosis_tool()
            else:
                raise RuntimeError(f"Unrecognized system platform: {platform}")

            # clip all pbf_files in temp directory
            clipped_pbf_files_set = set()
            for pbf_file_path in pbf_files:
                pbf_file_path_posix = Path(pbf_file_path).as_posix()
                pbf_file_name = Path(pbf_file_path).stem.split(".")[0]
                osm_path = (
                    tmp_dir_path.resolve() / f"pbf_files/{pbf_file_name}_{geometry_hash}.osm.pbf"
                ).as_posix()

                clipped_pbf_files_set.add(osm_path)

                if Path(osm_path).exists():
                    continue

                if platform in ("linux", "linux2"):
                    self._clip_pbf_with_osmconvert(pbf_file_path_posix, poly_path, osm_path)
                elif platform in ("darwin", "win32"):
                    self._clip_pbf_with_osmosis(pbf_file_path_posix, poly_path, osm_path)
                else:
                    raise RuntimeError(f"Unrecognized system platform: {platform}")

            clipped_pbf_files_list = list(clipped_pbf_files_set)

            # if single file - copy it
            if len(clipped_pbf_files_list) == 1:
                Path(clipped_pbf_files_list[0]).rename(final_osm_path)
            # merge all of the files and copy it
            else:
                if platform in ("linux", "linux2"):
                    self._merge_pbfs_with_osmconvert(clipped_pbf_files_list, final_osm_path)
                elif platform in ("darwin", "win32"):
                    self._merge_pbfs_with_osmosis(clipped_pbf_files_list, final_osm_path)
                else:
                    raise RuntimeError(f"Unrecognized system platform: {platform}")

        return Path(final_osm_path)

    def _generate_poly_file(self, geometry: Union[Polygon, MultiPolygon], poly_file: str) -> None:
        """
        This function will create the .poly files from the nuts shapefile.

        These.poly files are used to extract data from the openstreetmap files. This function is
        adapted from the OSMPoly function in QGIS.
        """
        Path(poly_file).parent.mkdir(parents=True, exist_ok=True)

        # this will create a list of the different subpolygons
        if geometry.geom_type == "MultiPolygon":
            polygons = geometry.geoms

        # the list will be lenght 1 if it is just one polygon
        elif geometry.geom_type == "Polygon":
            polygons = [geometry]

        # start writing the .poly file
        f = open(poly_file, "w")
        f.write("polygon\n")

        i = 0

        # loop over the different polygons, get their exterior and write the
        # coordinates of the ring to the .poly file
        for polygon in polygons:
            polygon_coords = np.array(polygon.exterior.coords)

            j = 0
            f.write(str(i) + "\n")

            for ring in polygon_coords:
                j = j + 1
                f.write("    " + str(ring[0]) + "     " + str(ring[1]) + "\n")

            i = i + 1
            # close the ring of one subpolygon if done
            f.write("END\n")

        # close the file when done
        f.write("END\n")
        f.close()

    def _download_osmconvert_tool(self) -> None:
        file_path = Path(self.OSMCONVERT_PATH)
        if not file_path.exists():
            tool_url = "http://m.m.i24.cc/osmconvert64"
            download_file(tool_url, self.OSMCONVERT_PATH)

            # make it executable
            file_path.chmod(file_path.stat().st_mode | stat.S_IEXEC)

    def _download_osmosis_tool(self) -> None:
        zip_path = (Path(self.OSMOSIS_DIRECTORY_PATH) / "osmosis.zip").as_posix()
        file_path = Path(self.OSMOSIS_EXECUTABLE_PATH)
        if not file_path.exists():
            tool_url = "https://github.com/openstreetmap/osmosis/releases/download/0.48.3/osmosis-0.48.3.zip"
            download_file(tool_url, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in tqdm(zf.infolist(), desc=""):
                    try:
                        zf.extract(member, self.OSMOSIS_DIRECTORY_PATH)
                    except zipfile.error:
                        pass

            # make it executable
            file_path.chmod(file_path.stat().st_mode | stat.S_IEXEC)

    def _clip_pbf_with_osmconvert(
        self, source_pbf_path: str, poly_file_path: str, output_pbf_path: str
    ) -> None:
        """Wrapper over osmconvert CLI tool."""
        Path(output_pbf_path).parent.mkdir(parents=True, exist_ok=True)
        os.system(
            "{}  {} -B={} --complete-ways --complete-multipolygons -o={}".format(
                self.OSMCONVERT_PATH, source_pbf_path, poly_file_path, output_pbf_path
            )
        )

    def _clip_pbf_with_osmosis(
        self, source_pbf_path: str, poly_file_path: str, output_pbf_path: str
    ) -> None:
        """Wrapper over osmosis CLI tool."""
        Path(output_pbf_path).parent.mkdir(parents=True, exist_ok=True)
        os.system(
            '{} --read-pbf file="{}" --bounding-polygon file="{}" --write-pbf file="{}"'.format(
                self.OSMOSIS_EXECUTABLE_PATH, source_pbf_path, poly_file_path, output_pbf_path
            )
        )

    # osmconvert region1.pbf --out-o5m | osmconvert - region2.pbf -o=all.pbf
    def _merge_pbfs_with_osmconvert(self, pbf_paths: Sequence[str], output_pbf_path: str) -> None:
        Path(output_pbf_path).parent.mkdir(parents=True, exist_ok=True)
        current_first_pbf_path = pbf_paths[0]
        for current_second_pbf_path in pbf_paths[1:]:
            command = (
                f"{self.OSMCONVERT_PATH} {current_first_pbf_path} --out-o5m | "
                f"{self.OSMCONVERT_PATH} - {current_second_pbf_path} -o={output_pbf_path}"
            )
            os.system(command)
            current_first_pbf_path = output_pbf_path

    # osmosis --rb file1.osm --rb file2.osm --merge --wb merged.osm
    def _merge_pbfs_with_osmosis(self, pbf_paths: Sequence[str], output_pbf_path: str) -> None:
        Path(output_pbf_path).parent.mkdir(parents=True, exist_ok=True)
        joined_files = " ".join([f"--rb {pbf_path}" for pbf_path in pbf_paths])
        command = f"{self.OSMOSIS_EXECUTABLE_PATH} {joined_files} --merge --wb {output_pbf_path}"
        os.system(command)
