"""Custom exceptions for the SRAI package."""


class SRAIException(Exception):
    """Base class for all SRAI exceptions."""


class ModelNotFitException(SRAIException):
    """Exception raised when a model is not fit."""


class LoadedDataIsEmptyException(SRAIException):
    """Exception when the loaded data returned by the loader is empty."""

    def __init__(self, message: str):  # noqa: D107
        super().__init__()

        self.message = message

    def __str__(self) -> str:  # noqa: D105
        return f"The data returned by the loader is empty. {self.message}"
