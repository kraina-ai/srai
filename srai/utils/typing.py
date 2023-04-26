"""Utility function for typing purposes."""

from contextlib import suppress
from typing import Any

from typeguard import TypeCheckError, check_type


def is_expected_type(value: object, expected_type: Any) -> bool:
    """
    Check if an object is a given type.

    Uses `typeguard` library to check objects using `typing` definitions.

    Args:
        value (object): Value to be checked against `expected_type`.
        expected_type (Any): A class or generic type instance.

    Returns:
        bool: Flag whether the object is an instance of the required type.
    """
    result = False

    with suppress(TypeCheckError):
        check_type(value, expected_type)
        result = True

    return result
