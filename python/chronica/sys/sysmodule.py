# Copyright 2022 Sogang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["getsizeof"]

def getsizeof(obj: object, index: int) -> int:
    r"""Returns the relative size of an object at position ``index``.

    :func:`getsizeof` calls the object's :meth:`__sizeof__` method. If the object
    does not provide means to retrieve the size, a TypeError will be raised.

    Args:
        obj (object): an object to get the relative size of
        index (int): index of the object
    """
    return obj.__sizeof__(index)  # type: ignore[call-arg]
