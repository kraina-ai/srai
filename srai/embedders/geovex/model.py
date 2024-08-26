"""
GeoVexModel.

This pytorch lightning module implements the GeoVeX hexagonal autoencoder model.

References:
    [1] https://openreview.net/forum?id=7bvWopYY1H
"""

import math
from typing import TYPE_CHECKING, Callable, Union

from srai._optional import import_optional_dependencies
from srai.embedders import Model

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn

try:  # pragma: no cover
    import torch  # noqa: F811
    from torch import nn  # noqa: F811

except ImportError:
    from srai.embedders._pytorch_stubs import nn, torch


def get_radius(i: int, j: int) -> int:
    """
    Calculates the radius of a cube given its coordinates.

    Ref https://www.redblobgames.com/grids/hexagons/#distances-axial

    Args:
        i (int): The x-coordinate of the cube.
        j (int): The y-coordinate of the cube.

    Returns:
        int: The radius of the cube.
    """
    origin = (0, 0, 0)
    target = (i, j, -i - j)
    return cube_distance(origin, target)


def cube_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    """
    Calculates the maximum distance between two points in a cube.

    Args:
        a (Tuple[int, int, int]): The first point.
        b (Tuple[int, int, int]): The second point.
    """
    vec = cube_subtract(a, b)
    return max(abs(vec[0]), abs(vec[1]), abs(vec[2]))


def cube_subtract(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Subtracts the coordinates of two 3D points and returns the resulting tuple.

    Args:
        a (Tuple[int, int, int]): The first point to subtract from.
        b (Tuple[int, int, int]): The second point to subtract.

    Returns:
        Tuple[int, int, int]: The resulting tuple of subtracted coordinates.
    """
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def get_shape(r: int) -> int:
    """
    Get the shape of the embedding.

    Args:
        r (int): The radius of the hexagonal region.

    Returns:
        int: The shape of the embedding.
    """
    return 2 * r + 2


def build_mask_funcs(R: int) -> tuple[Callable[[int, int], float], ...]:
    """
    Build the mask functions for the loss function. These functions depend on the radius of the
    hexagonal region. They weight the loss function to give more importance to the center of the
    region.

    Args:
        R (int): Radius of the hexagonal region.

    Returns:
        Tuple[callable, callable]: The mask functions.
    """  # noqa: D205

    def w_dist(i: int, j: int) -> float:
        """
        The Distance Weighting Kernel. Equation (6) in [1].

        Args:
            i (int): row index of the first point
            j (int): column index of the first point

        Returns:
            float: The weight of the loss function.
        """
        q = j - R
        r = i - R
        r = get_radius(q, r)
        return 1 / (1 + r) if r <= R else 0

    def w_num(i: int, j: int) -> float:
        """
        The Numerosity Weighting Kernel. Equation (6) in [1].

        The 6 is the number of hexagons in a ring,
        which is multiplied by the ring number to get the total number of hexagons in the ring.

        Args:
            i (int): row index of the first point
            j (int): column index of the first point

        Returns:
            float: The weight of the loss function.
        """
        # r represents the ring number, which can be found using the ij index
        q = j - R
        r = i - R
        r = get_radius(q, r)
        return 1 / (6 * r) if r <= R and r > 0 else 1 if r == 0 else 0

    return w_dist, w_num


class GeoVeXLoss(nn.Module):  # type: ignore
    """
    The loss function for the GeoVeX model.

    Defined in [1]. Equations (4) and (7).
    """

    def __init__(self, R: int):
        """
        Initialize the GeoVeXLoss.

        Args:
            R (int): The radius of the hexagonal region.
        """
        super().__init__()
        self.R = R

        # strip out the padding from y,
        # so that the loss function is only calculated on the valid regions
        self.M = get_shape(self.R) - 1

        # register the mask functions as buffers
        # so that they are saved with the model
        w_dist_func, w_num_func = build_mask_funcs(self.R)

        self.register_buffer(
            "_w_dist_matrix",
            torch.tensor(
                [[w_dist_func(i, j) for j in range(self.M)] for i in range(self.M)],
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "_w_num_matrix",
            torch.tensor(
                [[w_num_func(i, j) for j in range(self.M)] for i in range(self.M)],
                dtype=torch.float32,
            ),
        )

    def forward(
        self, pi: "torch.Tensor", lambda_: "torch.Tensor", y: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Forward pass of the loss function.

        Args:
            pi (torch.Tensor): The predicted pi tensor.
            lambda_ (torch.Tensor): The predicted lambda tensor.
            y (torch.Tensor): The target tensor.

        Returns:
            float: The loss value.
        """
        # trim the padding from y, pi, and lambda_
        y = y[:, :, : self.M, : self.M]
        pi = pi[:, :, : self.M, : self.M]
        lambda_ = lambda_[:, :, : self.M, : self.M]

        I0 = (y == 0).float()
        I_greater_0 = (y > 0).float()

        # torch.exp(-1 * lambda_) instead of torch.exp(lambda_). the paper has a typo, I think...
        log_likelihood_0 = I0 * torch.log(pi + (1 - pi) * torch.exp(-1 * lambda_))
        log_likelihood_greater_0 = I_greater_0 * (
            torch.log(1 - pi)
            - lambda_
            + y * torch.log(lambda_)
            - torch.lgamma(y + 1)  # this is the ln(factorial(y))
        )

        log_likelihood = log_likelihood_0 + log_likelihood_greater_0
        loss: torch.Tensor = -torch.sum(
            log_likelihood * self._w_dist_matrix * self._w_num_matrix
        ) / (torch.sum(self._w_dist_matrix) * torch.sum(self._w_num_matrix))
        return loss


class HexagonalConv2d(nn.Module):  # type: ignore
    """Hexagonal Convolutional Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        bias: bool = True,
        groups: int = 1,
    ):
        """
        Initialize the HexagonalConv2d. This is a convolutional layer with a hexagonal kernel.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of the kernel. Defaults to 3.
            stride (int, optional): The stride of the convolution. Defaults to 2.
            padding (int, optional): The padding of the convolution. Defaults to 0.
            bias (bool, optional): Whether to use bias. Defaults to True.
            groups (int, optional): The number of groups. Defaults to 1.
        """
        from torch import nn

        super().__init__()

        if kernel_size != 3:
            raise NotImplementedError("kernel_size must be 3. Hexagonal kernel is 3x3.")

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups
        )
        self.register_buffer("hexagonal_mask", self._create_hexagonal_mask())

    @staticmethod
    def _create_hexagonal_mask() -> "torch.Tensor":
        """Create the hexagonal mask."""
        return torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass of the HexagonalConv2d.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        self.conv.weight = nn.Parameter(self.conv.weight * self.hexagonal_mask)
        return self.conv(x)


class HexagonalConvTranspose2d(HexagonalConv2d):
    """Hexagonal Transpose Convolutional Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = True,
    ):
        """
        Initialize the HexagonalConvTranspose2d.

        This is a transpose convolutional layer with a hexagonal kernel.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of the kernel. Defaults to 3.
            stride (int, optional): The stride of the convolution. Defaults to 2.
            padding (int, optional): The padding of the convolution. Defaults to 0.
            output_padding (int, optional): The output padding of the convolution. Defaults to 0.
            bias (bool, optional): Whether to use bias. Defaults to True.
        """
        from torch import nn

        super(HexagonalConv2d, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=bias,
        )
        self.register_buffer("hexagonal_mask", self._create_hexagonal_mask())


class Reshape(nn.Module):  # type: ignore
    """Reshape layer."""

    def __init__(self, shape: tuple[int, ...]):
        """
        Initialize the Reshape layer.

        Args:
            shape (Tuple[int, ...]): The shape of the output tensor.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass of the Reshape layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reshaped tensor.
        """
        return x.view(self.shape)


class GeoVeXZIP(nn.Module):  # type: ignore
    """GeoVeX Zero-Inflated Poisson Layer."""

    def __init__(self, in_dim: int, m: int, out_dim: int):
        """
        Initialize the GeoVeXZIP layer.

        Args:
            in_dim (int): The input dimension.
            m (int): The height and width of the tensor
            r (int): The radius of the hexagonal region.
            out_dim (int): The output dimension.
        """
        from torch import nn

        super().__init__()
        self.in_dim = in_dim
        self.M = m
        self.out_dim = out_dim
        self.pi = nn.Linear(in_dim, out_dim)
        self.lambda_ = nn.Linear(in_dim, out_dim)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Forward pass of the GeoVeXZIP layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted pi and lambda tensors.
        """
        _x = x.view(-1, self.M, self.M, self.in_dim)
        pi = torch.sigmoid(self.pi(_x)).view(
            -1,
            self.out_dim,
            self.M,
            self.M,
        )
        lambda_ = torch.exp(self.lambda_(_x)).view(
            -1,
            self.out_dim,
            self.M,
            self.M,
        )
        # clamp pi to avoid nan's
        clamped_pi = torch.clamp(pi, 1e-6, 1 - 1e-6)
        return clamped_pi, lambda_


class GeoVexModel(Model):
    """
    GeoVeX Model.

    This class implements the GeoVeX model. It is based on a convolutional autoencoder with a Zero-
    Inflated Poisson layer. The model is described in [1]. It takes a 3d tensor as input (counts of
    features per region) and outputs dense embeddings. The 3d tensor consists of the target region
    at the center and radius R neighbors around it.
    """

    def __init__(
        self,
        k_dim: int,
        radius: int,
        conv_layers: int = 2,
        emb_size: int = 32,
        learning_rate: float = 1e-5,
        conv_layer_size: int = 256,
    ):
        """
        Initialize the GeoVeX model.

        Args:
            k_dim (int): the number of input channels
            radius (int): the radius of the hexagonal region
            conv_layers (int, optional): The number of convolutional layers. Defaults to 2.
            emb_size (int, optional): The dimension of the inner embedding. Defaults to 32.
            learning_rate (float, optional): The learning rate. Defaults to 1e-5.
            conv_layer_size (int, optional): The size of the initial convolutional layer.
        """
        if k_dim < conv_layer_size:
            raise ValueError(f"k_dim must be greater than {conv_layer_size}")

        if conv_layers < 2:
            raise ValueError("conv_layers must be greater than 1")

        if radius < 2:
            raise ValueError("R must be greater than 1")

        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        from torch import nn

        super().__init__()

        self.k_dim = k_dim

        self.R = radius
        self.lr = learning_rate
        self.emb_size = emb_size
        self.conv_layer_size = conv_layer_size
        self.conv_layers = conv_layers

        # input size is 2R + 2
        self.M = get_shape(self.R)

        # calculate the padding to preserve the input size
        #  equation for output size with stride is
        #  out_size = (in_size - kernel_size + padding + stride) / stride
        stride = 2
        kernel_size = 3
        padding = math.ceil(((stride - 1) * self.M - stride + kernel_size) / 2)

        # find the size of the linear layer
        # equation is the output size of the conv layers
        in_size = self.M
        ll_padding = 0 if self.R < 5 else 1
        for _ in range(conv_layers - 1):
            out_size = math.floor(
                (in_size - kernel_size + ll_padding + stride) / stride,
            )
            in_size = out_size

        conv_sizes = [conv_layer_size * 2**i for i in range(conv_layers)]
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(self.k_dim),
            nn.ReLU(),
            # have to add padding to preserve the input size
            HexagonalConv2d(self.k_dim, conv_sizes[0], kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm2d(conv_layer_size),
            nn.ReLU(),
            *(
                # second conv block
                nn.Sequential(
                    HexagonalConv2d(
                        conv_sizes[i - 1],
                        conv_sizes[i],
                        kernel_size=3,
                        stride=2,
                        padding=ll_padding,
                    ),
                    nn.BatchNorm2d(conv_sizes[i]),
                    nn.ReLU(),
                )
                for i in range(1, conv_layers)
            ),
            # flatten
            nn.Flatten(),
            nn.Linear(out_size**2 * conv_sizes[-1], self.emb_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_size, out_size**2 * conv_sizes[-1]),
            # maintain the batch size, but reshape the rest
            Reshape((-1, conv_sizes[-1], out_size, out_size)),
            # decoder has conv transpose layers - 1,
            # as the reshape layer is the first "transpose layer"
            *(
                nn.Sequential(
                    HexagonalConvTranspose2d(
                        conv_sizes[-1 * (i + 1)],
                        conv_sizes[-1 * (i + 2)],
                        kernel_size=3,
                        stride=2,
                        output_padding=1,
                        padding=ll_padding,
                    ),
                    nn.BatchNorm2d(conv_sizes[-1 * (i + 2)]),
                    nn.ReLU(),
                )
                for i in range(conv_layers - 1)
            ),
            GeoVeXZIP(conv_sizes[0], self.M, self.k_dim),
        )

        self._loss = GeoVeXLoss(self.R)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Forward pass of the GeoVeX model.

        Args:
            x (torch.Tensor): The input tensor. The dimensions are
                (batch_size, k_dim, R * 2 + 1, R * 2 + 1).

        Returns:
            torch.Tensor: The output tensor.
        """
        res: tuple[torch.Tensor, torch.Tensor] = self.decoder(self.encoder(x))
        return res[0], res[1]

    def training_step(self, batch: list["torch.Tensor"], batch_idx: int) -> "torch.Tensor":
        # sourcery skip: class-extract-method
        """
        Perform a training step. This is called by PyTorch Lightning.

        One training step consists of a forward pass, a loss calculation, and a backward pass.

        Args:
            batch (List[torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        loss = self._loss.forward(*self.forward(batch), batch)
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
        loss = self._loss.forward(*self.forward(batch), batch)
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> list["torch.optim.Optimizer"]:
        """
        Configure the optimizers. This is called by PyTorch Lightning.

        Returns:
            List[torch.optim.Optimizer]: The optimizers.
        """
        opt: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        return [opt]

    # override get_config to return the model configuration
    def get_config(self) -> dict[str, Union[int, float]]:
        """
        Get the model configuration.

        Returns:
            Dict[str, Union[int, float]]: The model configuration.
        """
        return {
            "k_dim": self.k_dim,
            "radius": self.R,
            "conv_layers": self.conv_layers,
            "emb_size": self.emb_size,
            "learning_rate": self.lr,
            "conv_layer_size": self.conv_layer_size,
        }
