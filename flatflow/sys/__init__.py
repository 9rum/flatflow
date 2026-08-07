# SPDX-License-Identifier: Apache-2.0

import sys

__all__ = ["getsizeof"]


def getsizeof(obj: object, index: int | None = None) -> int:
    """Returns the user-defined size of an object at position :param:`index`.

    :func:`getsizeof` calls the object's :meth:`__sizeof__` method.
    If the object does not provide means to retrieve the size, a TypeError will be
    raised.

    Args:
        obj (object): An object to get the size of.
        index (int, optional): Index of the object.

    Returns:
        int: The user-defined size of the object.
    """
    if index is None:
        return sys.getsizeof(obj)
    return obj.__sizeof__(index)  # type: ignore[call-arg]
