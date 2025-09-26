"""Utility functions for S2 cells."""

import geopandas as gpd
import numpy as np
import pandas as pd
from s2.s2 import s2_to_geo_boundary
from s2sphere import CellId
from shapely.geometry import Polygon

from srai.constants import REGIONS_INDEX, WGS84_CRS


def get_children_from_token(token: str, target_level: int) -> gpd.GeoDataFrame:
    """
    Given an S2 cell token (string), return a list of its child cells at the specified target level.

    Parameters:
        token (str): The S2 cell token (hex string, e.g., '89c2588').
        target_level (int): The desired resolution level for the child cells.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame of S2 cells representing the children at the target level.
    """
    # Convert the token to a CellId
    cell_id = CellId.from_token(token)

    # Ensure the parent cell's level is no higher than the target level
    if cell_id.level() > target_level:
        raise ValueError("The parent's resolution is higher than the target level.")

    current_cells = [cell_id]

    for _ in range(target_level - cell_id.level()):
        children = []
        for c in current_cells:
            children += list(c.children())
        current_cells = children

    # Directly create a dictionary of region_id to Polygon using a dictionary comprehension
    cells = {
        CellId(c.id()).to_token(): Polygon(
            s2_to_geo_boundary(CellId(c.id()).to_token(), geo_json_conformant=True)
        )
        for c in current_cells
    }

    cells_gdf = gpd.GeoDataFrame(
        None,
        index=cells.keys(),
        geometry=list(cells.values()),
        crs=WGS84_CRS,
    )
    cells_gdf.index.name = REGIONS_INDEX

    sorted_gdf = sort_patches(cells_gdf)

    return sorted_gdf


def sort_patches(patches_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Sort patches from top-left to bottom-right based on their bounding box coordinates.

    Parameters:
        patches_gdf (gpd.GeoDataFrame): GeoDataFrame containing patches with a 'geometry' column.

    Returns:
        gpd.GeoDataFrame: Sorted GeoDataFrame.
    """
    bounds = patches_gdf.geometry.bounds
    sorted_indices = np.lexsort((bounds["minx"].values, -bounds["maxy"].values))
    return patches_gdf.iloc[sorted_indices]


def get_patches_from_img_gdf(
    img_gdf: gpd.GeoDataFrame, target_level: int
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Get patches from an image GeoDataFrame at a specified target level.

    Parameters:
        img_gdf (gpd.GeoDataFrame): GeoDataFrame containing image regions.
        target_level (int): The desired resolution level for the patches.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: GeoDataFrames containing
        patches at the target level and the joint gdf between images and patches.
    """
    patch_s2_regions = []
    img_patch_joint = []

    for img_id, _ in img_gdf.iterrows():
        # Get the children at the target level
        children = get_children_from_token(img_id, target_level=target_level)

        parent_children_joint = gpd.GeoDataFrame(
            {"img_id": img_id, "patch_id": children.index}
        ).set_index(["img_id", "patch_id"])

        patch_s2_regions.append(children)
        img_patch_joint.append(parent_children_joint)

    patch_s2_regions = gpd.GeoDataFrame(pd.concat(patch_s2_regions, axis=0))
    img_patch_joint = gpd.GeoDataFrame(pd.concat(img_patch_joint, axis=0))

    return patch_s2_regions, img_patch_joint
