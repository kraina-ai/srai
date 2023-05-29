"""
Hex2Vec model.

This module contains a Hex2Vec[1] model with a regression head.

References:
    [1] https://dl.acm.org/doi/10.1145/3486635.3491076
"""
from pathlib import Path
from typing import TYPE_CHECKING, List

from srai.utils._optional import import_optional_dependencies

if TYPE_CHECKING:  # pragma: no cover
    import torch


try:  # pragma: no cover
    from pytorch_lightning import LightningModule

except ImportError:
    from srai.utils._pytorch_stubs import LightningModule


class Hex2VecModelForRegression(LightningModule):  # type: ignore
    """TODO: Add docstring."""

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        """
        Initialize Hex2VecModel.

        Args:
            layer_sizes (List[int]): List of sizes for the Hex2Vec model layers.
                The first element is the input size (number of features),
                the last element is the (embedding) size.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.


        Raises:
            ValueError: If layer_sizes contains less than 2 elements.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        from torch import nn

        from srai.embedders.hex2vec.model import Hex2VecModel

        super().__init__()
        self.model = Hex2VecModel(layer_sizes, learning_rate)
        self.regression_head = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """TODO: Add docstring."""
        embedding = self.model(x)
        y_pred = self.regression_head(embedding)
        return y_pred

    def training_step(self, batch: List["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        """TODO: Add docstring."""
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import mean_absolute_error

        x, y_true = batch

        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(loss)
        mae = mean_absolute_error(y_pred, y_true)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: List["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        """
        Perform one validation step.

        Args:
            batch (List[torch.Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        import torch
        import torch.nn.functional as F
        from torchmetrics.functional import mean_absolute_error

        x, y_true = batch

        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(loss)
        mae = mean_absolute_error(y_pred, y_true)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Configure optimizer."""
        import torch

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_kwargs(self) -> dict:
        """Get model save kwargs."""
        return {"layer_sizes": self.layer_sizes, "learning_rate": self.learning_rate}

    @classmethod
    def from_pretrained_hex2vec(cls, path: Path, **kwargs: dict) -> "Hex2VecModelForRegression":
        """
        Load model from a file.

        Args:
            path (str): Path to the file.
            **kwargs (dict): Additional kwargs to pass to the model constructor.
        """
        from srai.embedders.hex2vec.model import Hex2VecModel

        hex2vec = Hex2VecModel.load(path, **kwargs)

        model = cls(**kwargs)
        model.model.load_state_dict(hex2vec.state_dict())
        return model


# class Hex2VecRegressor:
#     """Hex2Vec Regressor."""

#     def __init__(
#         self,
#         encoder_sizes: Optional[List[int]] = None,
#         embedder_output_features: Optional[List[str]] = None,
#     ) -> None:
#         """
#         TODO
#         """
#         super().__init__(embedder_output_features)
#         import_optional_dependencies(
#             dependency_group="torch", modules=["torch", "pytorch_lightning"]
#         )
#         from srai.embedders import Hex2VecEmbedder
#         if encoder_sizes is None:
#             encoder_sizes = Hex2VecEmbedder.DEFAULT_ENCODER_SIZES
#         self._assert_encoder_sizes_correct(encoder_sizes)
#         self._encoder_sizes = encoder_sizes
#         self._model: Optional[Hex2VecModelForRegression] = None
#         self._is_fitted = False

#     def transform(
#         self,
#         regions_gdf: gpd.GeoDataFrame,
#         features_gdf: gpd.GeoDataFrame,
#         joint_gdf: gpd.GeoDataFrame,
#     ) -> pd.DataFrame:
#         """
#         Create region embeddings.

#         Args:
#             regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
#             features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
#             joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

#         Returns:
#             pd.DataFrame: Embedding and geometry index for each region in regions_gdf.

#         Raises:
#             ValueError: If features_gdf is empty and self.expected_output_features is not set.
#             ValueError: If any of the gdfs index names is None.
#             ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
#             ValueError: If index levels in gdfs don't overlap correctly.
#         """
#         import torch

#         self._check_is_fitted()
#         counts_df = self._get_raw_counts(regions_gdf, features_gdf, joint_gdf)
#         counts_tensor = torch.from_numpy(counts_df.values)
#         embeddings = self._model(counts_tensor).detach().numpy()  # type: ignore
#         return pd.DataFrame(embeddings, index=counts_df.index)

#     def fit(
#         self,
#         regions_gdf: gpd.GeoDataFrame,
#         features_gdf: gpd.GeoDataFrame,
#         joint_gdf: gpd.GeoDataFrame,
#         neighbourhood: Neighbourhood[T],
#         val_regions_gdf: Optional[gpd.GeoDataFrame] = None,
#         val_features_gdf: Optional[gpd.GeoDataFrame] = None,
#         val_joint_gdf: Optional[gpd.GeoDataFrame] = None,
#         val_neighbourhood: Optional[Neighbourhood[T]] = None,
#         negative_sample_k_distance: int = 2,
#         batch_size: int = 32,
#         learning_rate: float = 0.001,
#         trainer_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         """
#         Fit the model to the data.

#         Args:
#             regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
#             features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
#             joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
#             neighbourhood (Neighbourhood[T]): The neighbourhood to use.
#                 Should be intialized with the same regions.
#             negative_sample_k_distance (int, optional): When sampling negative samples,
#                 sample from a distance > k. Defaults to 2.
#             batch_size (int, optional): Batch size. Defaults to 32.
#             learning_rate (float, optional): Learning rate. Defaults to 0.001.
#             trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.

#         Raises:
#             ValueError: If features_gdf is empty and self.expected_output_features is not set.
#             ValueError: If any of the gdfs index names is None.
#             ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
#             ValueError: If index levels in gdfs don't overlap correctly.
#             ValueError: If negative_sample_k_distance < 2.
#         """
#         import pytorch_lightning as pl
#         from torch.utils.data import DataLoader

#         trainer_kwargs = self._prepare_trainer_kwargs(trainer_kwargs)

#         counts_df = self._get_raw_counts(regions_gdf, features_gdf, joint_gdf)
#         num_features = len(counts_df.columns)
#         self._model = Hex2VecModel(
#             layer_sizes=[num_features, *self._encoder_sizes], learning_rate=learning_rate
#         )
#         dataset = NeighbourDataset(counts_df, neighbourhood, negative_sample_k_distance)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         trainer = pl.Trainer(**trainer_kwargs)
#         if val_regions_gdf is not None and val_features_gdf is not None and val_joint_gdf is not None and val_neighbourhood is not None:
#             val_counts_df = self._get_raw_counts(
#                 val_regions_gdf, val_features_gdf, val_joint_gdf
#             )
#             val_dataset = NeighbourDataset(val_counts_df, val_neighbourhood, negative_sample_k_distance)
#             val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#             trainer_kwargs["val_dataloaders"] = [val_dataloader]
#             trainer.fit(self._model, dataloader, val_dataloader)
#         else:
#             trainer.fit(self._model, dataloader)

#         self._is_fitted = True

#     def fit_transform(
#         self,
#         regions_gdf: gpd.GeoDataFrame,
#         features_gdf: gpd.GeoDataFrame,
#         joint_gdf: gpd.GeoDataFrame,
#         neighbourhood: Neighbourhood[T],
#         val_regions_gdf: Optional[gpd.GeoDataFrame] = None,
#         val_features_gdf: Optional[gpd.GeoDataFrame] = None,
#         val_joint_gdf: Optional[gpd.GeoDataFrame] = None,
#         val_neighbourhood: Optional[Neighbourhood[T]] = None,
#         negative_sample_k_distance: int = 2,
#         batch_size: int = 32,
#         learning_rate: float = 0.001,
#         trainer_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> pd.DataFrame:
#         """
#         Fit the model to the data and return the embeddings.

#         Args:
#             regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
#             features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
#             joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
#             neighbourhood (Neighbourhood[T]): The neighbourhood to use.
#                 Should be intialized with the same regions.
#             negative_sample_k_distance (int, optional): When sampling negative samples,
#                 sample from a distance > k. Defaults to 2.
#             batch_size (int, optional): Batch size. Defaults to 32.
#             learning_rate (float, optional): Learning rate. Defaults to 0.001.
#             trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.

#         Returns:
#             pd.DataFrame: Region embeddings.

#         Raises:
#             ValueError: If features_gdf is empty and self.expected_output_features is not set.
#             ValueError: If any of the gdfs index names is None.
#             ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
#             ValueError: If index levels in gdfs don't overlap correctly.
#             ValueError: If negative_sample_k_distance < 2.
#         """
#         self.fit(
#             regions_gdf,
#             features_gdf,
#             joint_gdf,
#             neighbourhood,
#             val_regions_gdf,
#             val_features_gdf,
#             val_joint_gdf,
#             val_neighbourhood,
#             negative_sample_k_distance,
#             batch_size,
#             learning_rate,
#             trainer_kwargs,
#         )
#         return self.transform(regions_gdf, features_gdf, joint_gdf)

#     def _prepare_trainer_kwargs(self, trainer_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#         if trainer_kwargs is None:
#             trainer_kwargs = {}
#         if "max_epochs" not in trainer_kwargs:
#             trainer_kwargs["max_epochs"] = 3
#         return trainer_kwargs

#     def _get_raw_counts(
#         self, regions_gdf: pd.DataFrame, features_gdf: pd.DataFrame, joint_gdf: pd.DataFrame
#     ) -> pd.DataFrame:
#         return super().transform(regions_gdf, features_gdf, joint_gdf).astype(np.float32)

#     def _check_is_fitted(self) -> None:
#         if not self._is_fitted or self._model is None:
#             raise ModelNotFitException("Model not fitted. Call fit() or fit_transform() first.")

#     def _assert_encoder_sizes_correct(self, encoder_sizes: List[int]) -> None:
#         if len(encoder_sizes) < 1:
#             raise ValueError("Encoder sizes must have at least one element - embedding size.")
#         if any(size <= 0 for size in encoder_sizes):
#             raise ValueError("Encoder sizes must be positive integers.")

#     def save(self, path: Path) -> None:
#         """
#         Save the model to a directory.

#         Args:
#             path (Path): Path to the directory.
#         """
#         import torch
#         self._check_is_fitted()
#         model_kwargs = self._model.get_kwargs()
#         embedder_config = {
#             "model_config": model_kwargs,
#             "embedder_config": {
#                 "encoder_sizes": self._encoder_sizes,
#                 "expected_output_features": self.expected_output_features.tolist() if self.expected_output_features else None,
#             }
#         }

#         path.mkdir(parents=True, exist_ok=True)
#         model_path = path / "model.bin"
#         torch.save(self._model.state_dict(), model_path)
#         config_path = path / "config.json"
#         with open(config_path, "wt") as f:
#             json.dump(embedder_config, f, ensure_ascii=False, indent=4)

#     @classmethod
#     def load(cls, path: Path) -> "Hex2VecEmbedder":
#         """
#         Load the model from a directory.

#         Args:
#             path (Path): Path to the directory.

#         Returns:
#             Hex2VecEmbedder: The loaded embedder.
#         """
#         import torch
#         with open(path / "config.json", "rt") as f:
#             embedder_config = json.load(f)
#         embedder = cls(**embedder_config["embedder_config"])
#         model_path = path / "model.bin"
#         model = Hex2VecModel.load(model_path, **embedder_config["model_config"])
#         embedder._model = model
#         embedder._is_fitted = True
#         return embedder
