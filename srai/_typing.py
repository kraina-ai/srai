"""Utility function for typing purposes."""

from collections.abc import Iterable
from contextlib import suppress
from typing import Any

from typeguard import (
    CollectionCheckStrategy,
    TypeCheckerCallable,
    TypeCheckError,
    check_type,
    checker_lookup_functions,
)
from typeguard._checkers import check_list


def _iterable_checker_lookup(
    origin_type: Any, args: tuple[Any, ...], extras: tuple[Any, ...]
) -> TypeCheckerCallable | None:
    if origin_type == Iterable:
        return check_list

    return None


checker_lookup_functions.append(_iterable_checker_lookup)


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
        check_type(
            value, expected_type, collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS
        )
        result = True

    return result
