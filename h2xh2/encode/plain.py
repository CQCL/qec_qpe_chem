# Copyright 2025 Quantinuum (www.quantinuum.com)
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

"""Encoding module that does nothing. It is just for compatibility."""

from typing import NamedTuple
from pytket.circuit import Circuit
from pytket.backends.backendresult import BackendResult

__all__ = [
    "EncodeOptions",
    "InterpretOptions",
    "encode",
    "interpret",
]


class EncodeOptions(NamedTuple):
    pass


class InterpretOptions(NamedTuple):
    pass


def encode(
    circ: Circuit,
    options: EncodeOptions | None = None,
) -> Circuit:
    """Do nothing, just for compatibility."""
    if options is None:
        options = EncodeOptions()
    return circ


def interpret(
    result: BackendResult,
    options: InterpretOptions | None = None,
) -> BackendResult:
    """Do nothing, just for compatibility."""
    if options is None:
        options = InterpretOptions()
    return result
