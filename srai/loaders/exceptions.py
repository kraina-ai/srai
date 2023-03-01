"""Exceptions for loaders."""

from srai.utils.exceptions import SRAIException


class LoadedDataIsEmptyException(SRAIException):
    """Exception when the loaded data returned by the loader is empty."""

    def __init__(self, message: str):  # noqa: D107
        super().__init__()

        self.message = message

    def __str__(self) -> str:  # noqa: D105
        return f"The loaded data returned by the loader is empty. {self.message}"
