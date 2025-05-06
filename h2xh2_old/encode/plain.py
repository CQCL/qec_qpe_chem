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
