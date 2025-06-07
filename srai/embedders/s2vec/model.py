"""
S2VecModel.

This pytorch lightning module implements the S2Vec masked autoencoder model.

References:
    [1] https://arxiv.org/abs/2504.16942
    [2] https://arxiv.org/abs/2111.06377
"""

from typing import TYPE_CHECKING, Union

from srai._optional import import_optional_dependencies
from srai.embedders import Model
from srai.embedders.s2vec.positional_encoding import get_2d_sincos_pos_embed

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn

try:  # pragma: no cover
    import torch  # noqa: F811
    from torch import nn  # noqa: F811

except ImportError:
    from srai.embedders._pytorch_stubs import nn, torch


class MAEEncoder(nn.Module):
    """Masked Autoencoder Encoder."""

    def __init__(self, embed_dim: int, depth: int, num_heads: int):
        """
        Initialize the MAEEncoder.

        Args:
            embed_dim (int): The dimension of the embedding.
            depth (int): The number of encoder layers.
            num_heads (int): The number of attention heads.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=norm)

    def forward(self, x):
        """
        Forward pass of the MAEEncoder.

        Args:
            x (torch.Tensor): The input tensor. The dimensions are
                (batch_size, num_patches, embed_dim).

        Returns:
            torch.Tensor: The output tensor from the encoder.
        """
        return self.encoder(x)


class MAEDecoder(nn.Module):
    """Masked Autoencoder Decoder."""

    def __init__(self, decoder_dim: int, patch_dim: int, depth: int, num_heads: int):
        """
        Initialize the MAEDecoder.

        Args:
            decoder_dim (int): The dimension of the decoder.
            patch_dim (int): The dimension of the patches.
            depth (int): The number of decoder layers.
            num_heads (int): The number of attention heads.
        """
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(decoder_dim, num_heads, batch_first=True)
        norm = nn.LayerNorm(decoder_dim)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth, norm=norm)
        self.output = nn.Linear(decoder_dim, patch_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass of the MAEDecoder.

        Args:
            x (torch.Tensor): The input tensor. The dimensions are
                (batch_size, num_patches, decoder_dim).

        Returns:
            torch.Tensor: The output tensor from the decoder.
        """
        x = self.decoder(x)
        x = self.output(x)
        return x


class S2VecModel(Model):
    """
    S2Vec Model.

    This class implements the S2Vec model. It is based on the masked autoencoder architecture.
    The model is described in [1]. It takes a rasterized image as input (counts of
    features per region) and outputs dense embeddings.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_ch: int,
        num_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 2,
        embed_dim: int = 256,
        decoder_dim: int = 128,
        mask_ratio: float = 0.75,
        lr: float = 5e-4,
        weight_decay: float = 1e-3,
    ):
        """
        Initialize the S2Vec model.

        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of the patches.
            in_ch (int): The number of input channels.
            num_heads (int): The number of attention heads.
            encoder_layers (int): The number of encoder layers. Defaults to 6.
            decoder_layers (int): The number of decoder layers. Defaults to 2.
            embed_dim (int): The dimension of the encoder. Defaults to 256.
            decoder_dim (int): The dimension of the decoder. Defaults to 128.
            mask_ratio (float): The ratio of masked patches. Defaults to 0.75.
            lr (float): The learning rate. Defaults to 5e-4.
            weight_decay (float): The weight decay. Defaults to 1e-3.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        from torch import nn

        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        patch_dim = patch_size * patch_size * in_ch
        self.grid_size = img_size // patch_size
        self.patch_embed = nn.Linear(in_ch, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder = MAEEncoder(embed_dim, encoder_layers, num_heads)
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder = MAEDecoder(
            decoder_dim,
            patch_dim,
            decoder_layers,
            num_heads,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.mask_ratio = mask_ratio
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size, cls_token=True)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_dim, self.grid_size, cls_token=True)
        self.decoder_pos_embed = nn.Parameter(decoder_pos_embed, requires_grad=False)
        self.patch_dim = patch_dim
        self.lr = lr
        self.weight_decay = weight_decay

    def random_masking(
        self, x: "torch.Tensor", mask_ratio: float
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Randomly mask patches in the input tensor.

        This function randomly selects a subset of patches to mask and returns the masked
        tensor, the mask, and the indices to restore the original order.
        The mask is a binary tensor indicating which patches are masked (1) and which are not (0).

        Args:
            x (torch.Tensor): The input tensor. The dimensions are
                (batch_size, num_patches, embed_dim).
            mask_ratio (float): The ratio of masked patches.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The masked tensor, the mask, and the
            indices to restore the original order.

        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def encode(
        self, x: "torch.Tensor", mask_ratio: float
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): The input tensor. The dimensions are
                (batch_size, num_patches, embed_dim).
            mask_ratio (float): The ratio of masked patches.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The encoded tensor, the mask, and the
            indices to restore the original order.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]  # Add positional embedding, excluding class token

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # Class token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # Expand class token to batch size

        x = torch.cat([cls_tokens, x], dim=1)  # Concatenate class token

        return self.encoder(x), mask, ids_restore

    def decode(self, x: "torch.Tensor", ids_restore: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): The input tensor. The dimensions are
                (batch_size, num_patches, embed_dim).
            ids_restore (torch.Tensor): The indices to restore the original order.

        Returns:
            torch.Tensor: The output tensor from the decoder.
        """
        x = self.decoder_embed(x)  # Project to decoder dimension
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        x = self.decoder(x)

        x = x[:, 1:, :]  # Exclude class token
        return x

    def forward(
        self, inputs: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Forward pass of the S2Vec model.

        Args:
            inputs (torch.Tensor): The input tensor. The dimensions are
                (batch_size, num_patches, num_features).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The reconstructed tensor,
            the target tensor, and the mask.
        """
        latent, mask, ids_restore = self.encode(inputs, self.mask_ratio)
        pred = self.decode(latent, ids_restore)
        target = inputs

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()  # Only on masked patches

        return loss, pred, mask

    def training_step(self, batch: list["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        """
        Perform a training step. This is called by PyTorch Lightning.

        One training step consists of a forward pass, a loss calculation, and a backward pass.

        Args:
            batch (List[torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        rec, target, mask = self(batch)

        B, N, D = target.shape
        loss = (rec - target).pow(2).mean(dim=-1)  # MSE per patch
        loss = (loss * mask).sum() / mask.sum()  # Only on masked patches

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: list["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        """
        Perform a validation step. This is called by PyTorch Lightning.

        Args:
            batch (List[torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        imgs, _ = batch
        rec, target, mask = self(imgs)

        B, N, D = target.shape
        loss = (rec - target).pow(2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> list["torch.optim.Optimizer"]:
        """
        Configure the optimizers. This is called by PyTorch Lightning.

        Returns:
            List[torch.optim.Optimizer]: The optimizers.
        """
        opt: torch.optim.Optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=100
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_config(self) -> dict[str, Union[int, float]]:
        """
        Get the model configuration.

        Returns:
            Dict[str, Union[int, float]]: The model configuration.
        """
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_ch": self.in_ch,
            "embed_dim": self.embed_dim,
            "mask_ratio": self.mask_ratio,
            "lr": self.lr,
        }
