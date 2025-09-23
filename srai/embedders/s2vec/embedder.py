"""
S2Vec Embedder.

This module contains embedder from S2Vec paper [1].

References:
    [1] https://arxiv.org/abs/2504.16942
"""

import json
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from srai._optional import import_optional_dependencies
from srai.constants import REGIONS_INDEX
from srai.embedders import CountEmbedder, ModelT
from srai.embedders.s2vec.dataset import S2VecDataset
from srai.embedders.s2vec.model import S2VecModel
from srai.embedders.s2vec.s2_utils import get_patches_from_img_gdf
from srai.exceptions import ModelNotFitException
from srai.joiners.intersection_joiner import IntersectionJoiner
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter

T = TypeVar("T")

try:  # pragma: no cover
    from torch.utils.data import DataLoader

except ImportError:
    from srai.embedders._pytorch_stubs import DataLoader


class S2VecEmbedder(CountEmbedder):
    """S2Vec Embedder."""

    def __init__(
        self,
        target_features: Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter],
        count_subcategories: bool = True,
        batch_size: Optional[int] = 64,
        img_res: int = 8,
        patch_res: int = 12,
        num_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 2,
        embedding_dim: int = 256,
        decoder_dim: int = 128,
        mask_ratio: float = 0.75,
        dropout_prob: float = 0.2,
    ) -> None:
        """
        Initialize S2Vec Embedder.

        Args:
            target_features (Union[List[str], OsmTagsFilter, GroupedOsmTagsFilter]): The features
                that are to be used in the embedding. Should be in "flat" format,
                i.e. "<super-tag>_<sub-tag>", or use OsmTagsFilter object.
            count_subcategories (bool, optional): Whether to count all subcategories individually
                or count features only on the highest level based on features column name.
                Defaults to True.
            batch_size (int, optional): Batch size. Defaults to 64.
            img_res (int, optional): Image resolution. Defaults to 8.
            patch_res (int, optional): Patch resolution. Defaults to 12.
            num_heads (int, optional): Number of heads in the transformer. Defaults to 8.
            encoder_layers (int, optional): Number of encoder layers in the transformer.
                Defaults to 6.
            decoder_layers (int, optional): Number of decoder layers in the transformer.
                Defaults to 2.
            embedding_dim (int, optional): Embedding dimension. Defaults to 256.
            decoder_dim (int, optional): Decoder dimension. Defaults to 128.
            mask_ratio (float, optional): Mask ratio for the transformer. Defaults to 0.75.
            dropout_prob (float, optional): The dropout probability. Defaults to 0.2.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning", "timm"]
        )

        super().__init__(
            expected_output_features=target_features,
            count_subcategories=count_subcategories,
        )

        assert 0.0 <= mask_ratio <= 1.0, "Mask ratio must be between 0 and 1."
        assert 0.0 <= dropout_prob <= 1.0, "Dropout probability must be between 0 and 1."

        self._model: Optional[S2VecModel] = None
        self._is_fitted = False
        self._img_res = img_res
        self._patch_res = patch_res
        self.img_size = 2 ** (patch_res - img_res)
        self._num_heads = num_heads
        self._encoder_layers = encoder_layers
        self._decoder_layers = decoder_layers
        self._embedding_dim = embedding_dim
        self._decoder_dim = decoder_dim
        self._mask_ratio = mask_ratio
        self._dropout_prob = dropout_prob

        self._batch_size = batch_size

        self._dataset: DataLoader = None

    def transform(  # type: ignore[override]
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Create region embeddings.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.

        Returns:
            pd.DataFrame: Region embeddings.
        """
        self._check_is_fitted()

        _, dataloader, self._dataset = self._prepare_dataset(
            regions_gdf,
            features_gdf,
            self._batch_size,
            shuffle=False,
            is_fitting=False,
        )

        return self._transform(dataset=self._dataset, dataloader=dataloader)

    def _transform(
        self,
        dataset: S2VecDataset[T],
        dataloader: Optional[DataLoader] = None,
    ) -> pd.DataFrame:
        import torch

        if dataloader is None:
            dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)

        self._model.encoder.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            embeddings = [
                self._model.encode(batch, mask_ratio=0)[0].detach().numpy()  # type: ignore[union-attr]
                for batch in dataloader
            ]
        self._model.encoder.train()  # type: ignore[union-attr]

        trimmed = [embedding[:, 1:, :] for embedding in embeddings]  # remove cls token

        embeddings = np.concatenate(trimmed, axis=0).reshape(-1, self._embedding_dim)

        df = pd.DataFrame(embeddings, index=dataset.patch_s2_ids)
        df.index.name = REGIONS_INDEX
        return df

    def fit(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        learning_rate: float = 0.001,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Fit the model to the data.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs.
                This is where the number of epochs can be set. Defaults to None.
        """
        import pytorch_lightning as pl

        trainer_kwargs = self._prepare_trainer_kwargs(trainer_kwargs)
        counts_df, dataloader, dataset = self._prepare_dataset(  # type: ignore
            regions_gdf,
            features_gdf,
            self._batch_size,
            shuffle=True,
            is_fitting=True,
        )

        self._prepare_model(counts_df, learning_rate)

        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self._model, dataloader)
        self._is_fitted = True
        self._dataset = dataset

    def _prepare_model(self, counts_df: pd.DataFrame, learning_rate: float) -> None:
        if self._model is None:
            self._model = S2VecModel(
                img_size=self.img_size,
                patch_size=1,
                in_ch=len(counts_df.columns),
                num_heads=self._num_heads,
                encoder_layers=self._encoder_layers,
                decoder_layers=self._decoder_layers,
                embed_dim=self._embedding_dim,
                decoder_dim=self._decoder_dim,
                mask_ratio=self._mask_ratio,
                dropout_prob=self._dropout_prob,
                lr=learning_rate,
            )

    def _prepare_dataset(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        batch_size: Optional[int],
        shuffle: bool = True,
        is_fitting: bool = True,
    ) -> tuple[pd.DataFrame, DataLoader, S2VecDataset[T]]:
        patches_gdf, img_patch_joint_gdf = get_patches_from_img_gdf(
            img_gdf=regions_gdf, target_level=self._patch_res
        )
        joiner = IntersectionJoiner()
        patch_feature_joint_gdf = joiner.transform(patches_gdf, features_gdf)
        counts_df = self._get_raw_counts(patches_gdf, features_gdf, patch_feature_joint_gdf)

        # Calculate mean and std from training dataset
        if is_fitting:
            eps = 1e-8
            self._feature_means = np.mean(counts_df.values, axis=0)
            self._feature_stds = np.std(counts_df.values, axis=0)
            self._empty_features_mask = self._feature_stds < eps
            self._feature_stds[self._empty_features_mask] = 1.0

        # Disable features that were empty in the training dataset
        counts_df.loc[:, self._empty_features_mask] = 0
        # Normalise raw counts
        counts_df = (counts_df - self._feature_means) / self._feature_stds

        dataset: S2VecDataset[T] = S2VecDataset(counts_df, img_patch_joint_gdf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return counts_df, dataloader, dataset

    def fit_transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        learning_rate: float = 0.001,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fit the model to the data and create region embeddings.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. This is where the
                number of epochs can be set. Defaults to None.
        """
        self.fit(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            learning_rate=learning_rate,
            trainer_kwargs=trainer_kwargs,
        )
        assert self._dataset is not None  # for mypy
        return self._transform(dataset=self._dataset)

    def _get_raw_counts(
        self,
        regions_gdf: pd.DataFrame,
        features_gdf: pd.DataFrame,
        joint_gdf: pd.DataFrame,
    ) -> pd.DataFrame:
        return super().transform(regions_gdf, features_gdf, joint_gdf).astype(np.float32)

    def _check_is_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise ModelNotFitException("Model not fitted. Call fit() or fit_transform() first.")

    def _prepare_trainer_kwargs(self, trainer_kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
        if trainer_kwargs is None:
            trainer_kwargs = {}
        if "gradient_clip_val" not in trainer_kwargs:
            trainer_kwargs["gradient_clip_val"] = 1.0
        if "max_epochs" not in trainer_kwargs:
            trainer_kwargs["max_epochs"] = 3
        return trainer_kwargs

    def save(self, path: Union[str, Any]) -> None:
        """
        Save the S2VecEmbedder model to a directory.

        Args:
            path (Union[str, Any]): Path to the directory.
        """
        embedder_config = {
            "target_features": cast("pd.Series", self.expected_output_features).to_json(
                orient="records"
            ),
            "count_subcategories": self.count_subcategories,
            "batch_size": self._batch_size,
            "img_res": self._img_res,
            "patch_res": self._patch_res,
            "num_heads": self._num_heads,
            "encoder_layers": self._encoder_layers,
            "decoder_layers": self._decoder_layers,
            "embedding_dim": self._embedding_dim,
            "decoder_dim": self._decoder_dim,
            "mask_ratio": self._mask_ratio,
            "dropout_prob": self._dropout_prob,
        }

        normalisation_config = {
            "feature_means": self._feature_means.tolist(),
            "feature_stds": self._feature_stds.tolist(),
            "empty_features_mask": self._empty_features_mask.tolist(),
        }

        self._save(path, embedder_config, normalisation_config)

    def _save(
        self,
        path: Union[str, Any],
        embedder_config: dict[str, Any],
        normalisation_config: dict[str, Any],
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._check_is_fitted()

        path.mkdir(parents=True, exist_ok=True)

        # save model and config
        self._model.save(path / "model.pt")  # type: ignore
        # combine model config and embedder config
        model_config = self._model.get_config()  # type: ignore

        config = {
            "model_config": model_config,
            "embedder_config": embedder_config,
            "normalisation_config": normalisation_config,
        }

        with (path / "config.json").open("w") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "S2VecEmbedder":
        """
        Load the model from a directory.

        Args:
            path (Union[Path, str]): Path to the directory.
            model_module (type[ModelT]): Model class.

        Returns:
            S2VecEmbedder: S2VecEmbedder object.
        """
        return cls._load(path, S2VecModel)

    @classmethod
    def _load(cls, path: Union[Path, str], model_module: type[ModelT]) -> "S2VecEmbedder":
        if isinstance(path, str):
            path = Path(path)
        with (path / "config.json").open("r") as f:
            config = json.load(f)

        config["embedder_config"]["target_features"] = json.loads(
            config["embedder_config"]["target_features"]
        )
        embedder = cls(**config["embedder_config"])
        model_path = path / "model.pt"
        model = model_module.load(model_path, **config["model_config"])
        embedder._model = model
        embedder._is_fitted = True
        embedder._feature_means = np.array(config["normalisation_config"]["feature_means"])
        embedder._feature_stds = np.array(config["normalisation_config"]["feature_stds"])
        embedder._empty_features_mask = np.array(
            config["normalisation_config"]["empty_features_mask"]
        )

        return embedder
