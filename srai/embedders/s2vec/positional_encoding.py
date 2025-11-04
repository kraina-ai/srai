"""Utility functions for generating sinusoidal positional embeddings in 1D and 2D."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def get_1d_sincos_pos_embed(embed_dim: int, pos: "torch.Tensor") -> "torch.Tensor":
    """
    Generate 1D sin-cos positional embeddings.

    Args:
        embed_dim (int): Dimension of the embeddings (must be even).
        pos (torch.Tensor): a 1D tensor of positions.

    Returns:
        torch.Tensor: Positional embeddings of shape (len(pos), embed_dim)
    """
    import torch

    assert embed_dim % 2 == 0, "Embed dimension must be even"
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    pos_embed = torch.cat([emb_sin, emb_cos], dim=1)
    return pos_embed


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> "torch.Tensor":
    """
    Generate 2D sin-cos positional embeddings.

    Args:
        embed_dim (int): Dimension of the embeddings.
        grid_size (int): The height (and width) of the grid.
        cls_token (bool): Whether to add an extra class token embedding.

    Returns:
        torch.Tensor: Positional embeddings of shape
        (1, grid_size*grid_size (+1 if cls_token), embed_dim)
    """
    import torch

    # Generate grid of positions
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")  # each shape: (grid_size, grid_size)
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)
    grid = grid.reshape(2, -1).transpose(0, 1)  # (grid_size*grid_size, 2)

    # For each dimension, generate sin-cos embedding and then concatenate
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[:, 0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[:, 1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (grid_size*grid_size, embed_dim)
    if cls_token:
        cls_token_embed = torch.zeros([1, embed_dim], dtype=torch.float32)
        pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)
    return pos_embed.unsqueeze(0)
