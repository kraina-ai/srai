from typing import Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(Generic[T_co]):
    """Dataset class stub."""


class DataLoader(Generic[T_co]):
    """DataLoader class stub."""


class LightningModule:  # pragma: no cover
    """LightningModule class stub."""


class nn:  # pragma: no cover
    """Pytorch nn class stub."""

    class Module:
        """Pytorch nn.Module class stub."""


class torch:  # pragma: no cover
    """Pytorch class stub."""
