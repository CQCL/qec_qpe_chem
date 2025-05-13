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

from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray
from pytket import Bit, Qubit
from typing import NamedTuple, Counter
from enum import Enum
from .steane_corrections import syndrome_from_readout, readout_correction
import re


class ReadoutMode(Enum):
    """Readout interpretation mode.

    Raw:
        Interpret the raw measurement outcomes as (-1) ** sum(bits).
    Detect:
        Post-select the measurement outcomes that remain in the code space.
    Correct:
        Perform the error correction based on the lookup table.
    """

    Raw = 0
    Detect = 1
    Correct = 2


class InterpretOptions(NamedTuple):
    """Options for the interpret function to be used by the workflow driver.

    Args:
        readout_mode:
            Specify the readout mode.
    """

    # Readout mode.
    readout_mode: ReadoutMode = ReadoutMode.Correct


# def decoder(
#     syndrome: tuple[int, int, int],
# ) -> int | None:
#     """Steane decoder based on the lookup table.

#     Args:
#         syndrome:
#             Syndrome measurement outcomes.

#     Returns:
#         Identified error position if the syndrome is non-trivial.
#     """
#     assert len(syndrome) == 3
#     lookup_table = {
#         (0, 0, 0): None,
#         (1, 0, 0): 0,
#         (1, 1, 0): 1,
#         (1, 1, 1): 2,
#         (1, 0, 1): 3,
#         (0, 1, 0): 4,
#         (0, 1, 1): 5,
#         (0, 0, 1): 6,
#     }
#     return lookup_table[syndrome]


# def ro_correction(
#     readout: list[int],
# ) -> list[int]:
#     """Readout error correction.

#     Args:
#         readout:
#             Measurement outcome (physical).
#     Returns:
#         Error corrected measurement outcome.
#     """
#     readout_ = list(readout[:])
#     syndrome = syndrome_from_readout(readout)
#     flip = decoder(syndrome)
#     if flip is not None:
#         readout_[flip] = (readout_[flip] + 1) % 2
#     return readout_


def l2p(i: int | Qubit | Bit) -> list[int]:
    """Index convertor for the qubit register.

    Args:
        i:
            Logical qubit index.

    Returns:
        List of indices of the corresponding physical qubits/bits.
    """
    i_ = i
    if isinstance(i, (Qubit, Bit)):
        i_ = i.index[0]
    r = slice(7 * i_, 7 * (i_ + 1))
    return r


def get_decoded_result(
    result: BackendResult,
    readout_mode: ReadoutMode = ReadoutMode.Raw,
) -> BackendResult:
    assert readout_mode.value in [0, 1, 2]
    rng = "c"
    bitlist = result.get_bitlist()
    # Chose the data bit register.
    cbits = [b for b in bitlist if b.reg_name == rng]
    l_data = len(cbits)
    n_logical_qubits = l_data // 7
    # Error detection bits.
    cbits += [b for b in bitlist if re.match("iceberg_discard_b", b.reg_name)]
    # Interpret the physical results.
    counts = result.get_counts(cbits=cbits)
    logical_counts = Counter()
    for readout0, val in counts.items():
        # Post selection by the error detection.
        if sum(readout0[l_data:]) > 0:
            continue
        # Use the readout as it is.
        if readout_mode == ReadoutMode.Raw:
            readout = readout0[:l_data]
        # Readout error detection.
        elif readout_mode == ReadoutMode.Detect:
            error_detected = False
            for il in range(n_logical_qubits):
                syndrome = syndrome_from_readout(readout0[il * 7 : il * 7 + 7])
                if sum(syndrome) > 0:
                    error_detected = True
                    break
            if error_detected:
                continue
            else:
                readout = readout0[:l_data]
        # Readout error correction.
        elif readout_mode == ReadoutMode.Correct:
            readout: list[int] = []
            for il in range(n_logical_qubits):
                readout += list(readout_correction(readout0[il * 7 : il * 7 + 7]))
        else:
            raise RuntimeError()
        lreadout: list[int] = []
        for i in range(len(cbits[:l_data]) // 7):
            parity: int = int((-1) ** int(sum(readout[l2p(i)])))
            lreadout.append((1 - parity) // 2)
        logical_readout = tuple(lreadout)
        logical_counts[OutcomeArray.from_readouts([logical_readout])] += int(val)
    logical_result = BackendResult(counts=logical_counts)
    return logical_result


def interpret(
    result: BackendResult, options: InterpretOptions = InterpretOptions()
) -> BackendResult:
    """An interpret function to be used by the workflow driver.

    Args:
        Result:
            Backend result in the physical space.
        options:
            Interpret options.

    Returns:
        Backend result in the logical space.
    """
    return get_decoded_result(result, readout_mode=options.readout_mode)
