"""
Predictor module.

This module contains implementation of predictor dedicated to geo data.
"""

from typing import Any, Optional

import geopandas as gpd
import h3
import torch
from shapely.geometry import Point, Polygon
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from srai.models.vectorizer import Vectorizer


class Predictor:
    """Evaluator class."""

    def __init__(
        self,
        task: str = "regression",
        batch_size: Optional[int] = 64,
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Evaluator.

        Args:
            task (str): Evaluation task type, possible values:
                     "trajectory_prediction", "regression", "poi_prediction".
                     Default to regression.
            batch_size (Optional[int], optional): Batch size used for prediction. Default 64.
            device (Optional[str], optional): Type of device used for prediction.
                    Defaults to "cuda' if available.

        Raises:
                ValueError: If task type is not supported.
        """
        if task not in ["trajectory_prediction", "regression", "poi_prediction"]:
            raise ValueError(f"Task {task} not supported in srai library.")
        else:
            self.task = task
        self.device = device
        self.batch_size = batch_size

    def predict(
        self,
        model: nn.Module,
        data: (Dataset | gpd.GeoDataFrame | list[str] | list[Point] | list[tuple[float, float]]),
        resolution: Optional[int] = None,
        embedder_type: Optional[str] = "Hex2VecEmbedder",
        embedder_hidden_sizes: Optional[list[int]] = None,
    ) -> tuple[list[Any], list[str], list[Any]]:
        """
        Predict value for dataset, single hexagon or list of hexagons.

        Args:
            model (nn.Module): Model to predict with
            data (Datset | gpd.GeoDataFrame | list[str] | list[Point] | list[tuple[float, float]]):\
                    Data to predict from: Dataset with vector embeddings "X" and "X_h3_idx" \
                    hex indices, list of hex indices or list of points(shapely Points or x,y tuples)
            resolution (Optional[int]): h3 resolution, default to 9 if not provided
            embedder_type (str): If data is passed as hexagon or point, embedder\
                used to encode it to vector
            embedder_hidden_sizes (Optional[list[int]]): Hidden sizes of embedder, last have\
            to match with model input size
        Returns:
            tuple[list[Any], list[str], list[Any]]: lists of points(if exist), hexagon indexes \
                and matching predictions.

        Raises:
            ValueError: If model input size does not match embedding size
        """
        model.to(self.device)
        model.eval()
        res = resolution if resolution is not None else 9

        if isinstance(data, gpd.GeoDataFrame):
            gdf = data
        else:
            gdf = gpd.GeoDataFrame()

        if isinstance(data, list):
            if isinstance(data[0], str):
                gdf = self._h3_indices_to_geometry(data, res)
            elif isinstance(data[0], tuple):
                h3_indices = self._points_to_hexagon_indices(data, res)
                gdf = self._h3_indices_to_geometry(h3_indices, res)
                gdf["point"] = data
            elif isinstance(data[0], Point):
                gdf = gpd.GeoDataFrame(geometry=data)
                gdf["point"] = data
            gdf["y"] = None
            vectorizer = Vectorizer(
                gdf_dataset=gdf,
                target_column_name="y",
                embedder_type=str(embedder_type),
                h3_resolution=res,
                embedder_hidden_sizes=embedder_hidden_sizes,
            )
            data = vectorizer.get_dataset()

        first_parameter = next(model.parameters())
        input_shape = first_parameter.size()
        if vectorizer.embedder_hidden_sizes[-1] != input_shape[1]:
            raise ValueError(
                f"Model input size {input_shape[1]} does not match \
                    embedding size {vectorizer.embedder_hidden_sizes[-1]}"
            )

        if isinstance(data, Dataset):
            dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False)

        h3_indexes = []
        xy_points = []
        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting...", total=len(dataloader)):
                inputs = batch["X"].to(self.device)
                indexes = batch["X_h3_idx"]
                points = batch["point"] if "point" in batch else ["" for _ in indexes]
                outputs = model(inputs)
                h3_indexes.extend(indexes)
                xy_points.extend(points)
                all_predictions.extend(outputs.cpu().numpy())  # Assuming outputs is a tensor
        return xy_points, h3_indexes, all_predictions

    def _h3_indices_to_geometry(
        self, h3_indexes: list[Any], resolution: int = 9
    ) -> gpd.GeoDataFrame:
        """
        Mapps a list of h3 indexes to geometry in geodataframe.

        Args:
            h3_indexes (list[str]): list of h3 indexes
            resolution (int): h3 resolution

        Returns:
            gpd.GeoDataFrame: Geodataframe with geometries
        """
        polygons = [
            h3.h3_to_geo_boundary(h, geo_json=True, resolution=resolution) for h in h3_indexes
        ]
        gdf = gpd.GeoDataFrame(geometry=[Polygon(polygon) for polygon in polygons])
        gdf.crs = {"init": "epsg:4326"}
        return gdf

    def _points_to_hexagon_indices(self, points: list[Any], resolution: int = 9) -> list[str]:
        """
        Maps a list of (x, y) float tuples to hexagon indices.

        Args:
            points (list[Point]): List of (x, y) float tuples.
            resolution (int): Resolution of the hexagons.

        Returns:
            list: List of hexagon indices corresponding to each point.
        """
        hex_indices = [h3.geo_to_h3(point[1], point[0], resolution) for point in points]
        return hex_indices
