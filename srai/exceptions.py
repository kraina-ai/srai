"""Custom exceptions for the SRAI package."""


class SRAIException(Exception):
    """Base class for all SRAI exceptions."""

    pass


class ModelNotFitException(SRAIException):
    """Exception raised when a model is not fit."""

    pass
