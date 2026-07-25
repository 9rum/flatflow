# SPDX-License-Identifier: Apache-2.0

__all__ = ["getsizeof"]


def getsizeof(o: object, index: int) -> int:
    """Returns the user-defined size of an object at position :param:`index`.

    :func:`getsizeof` calls the object's :meth:`__sizeof__` method.
    If the object does not provide means to retrieve the size, a TypeError will be
    raised.

    Args:
        o (object): An object to get the size of.
        index (int): Index of the object.

    Returns:
        int: The user-defined size of the object.
    """
    return o.__sizeof__(index)  # type: ignore[call-arg]
