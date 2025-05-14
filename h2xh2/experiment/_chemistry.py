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

from typing import (
    NamedTuple,
    Callable,
)
import numpy as np
from pytket.circuit import (
    Circuit,
    PauliExpBox,
    Pauli,
)
from ..encode import (
    steane_z_correct,
    steane_x_correct,
    iceberg_w_0_detect,
    iceberg_w_1_detect,
)

_APPROX_ENERGY: float = -1.13629792
_DELTAT = 0.986620 / np.pi


class ChemData(NamedTuple):
    # H2 with R_HH = 1.5 A, qubit tapering with (-ZIZI, -IZIZ, IZZI).
    CZ: float = 0.7960489286466914
    CX: float = -0.1809233927385484
    CI: float = -0.3209561440881913
    FCI_ENERGY: float = -1.13730605

    # Energy and ansatz parameters for this specific condition.
    MAX_BITS: int = 5
    DELTAT: float = _DELTAT
    APPROX_ENERGY: float = _APPROX_ENERGY
    APPROX_PHASE: float = -_APPROX_ENERGY * _DELTAT
    ANSATZ_PARAM: float = [-0.08728706, -0.25]


_chem_data = ChemData()


def resolve_phase(phase: float, max_bits: int = 10) -> list[int]:
    phase_ = phase % 2.0
    bits = []
    atol = 2 ** (-max_bits)
    phase_ += atol
    if phase_ > 2.0:
        return bits
    for i in range(max_bits):
        val = 2**-i
        if phase_ >= val:
            bits.append(1)
            phase_ -= val
        else:
            bits.append(0)
        if phase_ < atol:
            break
    return bits


def _add_ctrlu_0(
    circ: Circuit,
    k: int,
    angle_z: float,
    angle_x: float,
) -> Callable[[int], Circuit]:
    """Plain ctrl-U circuit without any QPE circuit included."""
    for ii in range(k):
        # circ.CRz(2 * cz * deltat, 0, 1)
        circ.Rz(angle_z, 1)
        circ.CX(0, 1)
        circ.Rz(-angle_z, 1)
        circ.CX(0, 1)
        # circ.CRx(2 * cx * deltat, 0, 1)
        circ.H(1)
        circ.Rz(angle_x, 1)
        circ.CX(0, 1)
        circ.Rz(-angle_x, 1)
        circ.CX(0, 1)
        circ.H(1)
    return circ


def _add_ctrlu_1(
    circ: Circuit,
    k: int,
    angle_z: float,
    angle_x: float,
) -> Callable[[int], Circuit]:
    """Add Steane QEC for X syndrome of the QPE ancilla qubit."""
    for ii in range(k):
        # circ.CRz(2 * cz * deltat, 0, 1)
        circ.Rz(angle_z, 1)
        circ.CX(0, 1)
        circ.Rz(-angle_z, 1)
        circ.CX(0, 1)
        # circ.CRx(2 * cx * deltat, 0, 1)
        circ.H(1)
        circ.Rz(angle_x, 1)
        circ.CX(0, 1)
        circ.Rz(-angle_x, 1)
        circ.CX(0, 1)
        circ.H(1)
        if ii + 1 >= k:
            break
        circ.add_barrier(circ.qubits)
        circ.add_custom_gate(steane_x_correct, [], [0])
        circ.add_barrier(circ.qubits)
    return circ


def _add_ctrlu_2(
    circ: Circuit,
    k: int,
    angle_z: float,
    angle_x: float,
) -> Callable[[int], Circuit]:
    """Add Steane QEC for each Trotter step and QEC_X in the middle."""
    for ii in range(k):
        # circ.CRz(2 * cz * deltat, 0, 1)
        circ.Rz(angle_z, 1)
        circ.CX(0, 1)
        circ.Rz(-angle_z, 1)
        circ.CX(0, 1)
        # Steane QEC for X.
        circ.add_barrier(circ.qubits)
        circ.add_custom_gate(steane_x_correct, [], [0])
        circ.add_barrier(circ.qubits)
        circ.add_custom_gate(steane_x_correct, [], [1])
        circ.add_barrier(circ.qubits)
        # circ.CRx(2 * cx * deltat, 0, 1)
        circ.H(1)
        circ.Rz(angle_x, 1)
        # Spin echo.
        circ.CX(0, 1)
        circ.Rz(-angle_x, 1)
        circ.CX(0, 1)
        # Spin echo.
        circ.H(1)
        if ii + 1 >= k:
            break
        circ.add_barrier(circ.qubits)

        circ.add_custom_gate(steane_x_correct, [], [0])
        circ.add_custom_gate(steane_z_correct, [], [0])
        circ.add_barrier(circ.qubits)
        circ.add_custom_gate(steane_x_correct, [], [1])
        circ.add_custom_gate(steane_z_correct, [], [1])
        circ.add_barrier(circ.qubits)
    return circ


def get_ctrl_func(
    benchmark: bool = False,
    pft_rz: bool = False,
    qec_level: int = 0,
) -> Callable[[int], Circuit]:
    """Get a function to return a circuit representing ctrl-U.

    Args:
        benchmark: Add Rz(-kEt) if True.
        pft_rz: Add the iceberg-style QED for Rz(beta).
        qec_level: 0 -> No QEC, 1 -> QEC for QPE ancilla, 2 -> for all logical qubits.

    Returns:
        A function to return a circuit.
    """

    def get_ctrlu(k: int) -> Circuit:
        circ = Circuit(2)
        # Round the rotation angle. This is done for the compatibility between plain and Stean.
        bits = resolve_phase(
            _chem_data.CZ * _chem_data.DELTAT,
            max_bits=_chem_data.MAX_BITS,
        )
        angle_z = sum([a * 2**-i for i, a in enumerate(bits)])
        bits = resolve_phase(
            _chem_data.CX * _chem_data.DELTAT,
            max_bits=_chem_data.MAX_BITS,
        )
        angle_x = sum([a * 2**-i for i, a in enumerate(bits)])
        match qec_level:
            case 0:
                _add_ctrlu_0(circ, k, angle_z, angle_x)
            case 1:
                _add_ctrlu_1(circ, k, angle_z, angle_x)
            case 2:
                _add_ctrlu_2(circ, k, angle_z, angle_x)
            case _:
                raise ValueError("qec_level")
        if benchmark:
            phase_factor = _chem_data.CI - _chem_data.APPROX_ENERGY
        else:
            phase_factor = _chem_data.CI
        phase = -1 * k * phase_factor * _chem_data.DELTAT
        circ.add_pauliexpbox(
            PauliExpBox([Pauli.Z], phase),
            circ.qubits[:1],
        )
        if pft_rz:
            circ.add_custom_gate(iceberg_w_0_detect, [], [0])
            circ.add_custom_gate(iceberg_w_1_detect, [], [0])
        return circ

    return get_ctrlu


def get_state(
    benchmark: bool = False,
    pft_rz: bool = False,
) -> Circuit:
    """Get the initial state.

    Args:
        benchmark: Return HF state if False else the approximate FCI state.
        pft_rz: Partially FT Rz by adding the iceberg-style QED.

    Returns:
        State preparation circuit.
    """
    state = Circuit(1)
    state.X(0)
    if benchmark:
        state.add_pauliexpbox(
            PauliExpBox([Pauli.Y], _chem_data.ANSATZ_PARAM[0]),
            state.qubits,
        )
        if abs(_chem_data.ANSATZ_PARAM[1]) > 0:
            state.add_pauliexpbox(
                PauliExpBox([Pauli.Z], _chem_data.ANSATZ_PARAM[1]),
                state.qubits,
            )
        if pft_rz:
            # Iceberg-style error detection.
            state.add_custom_gate(iceberg_w_0_detect, [], [0])
            state.add_custom_gate(iceberg_w_1_detect, [], [0])
    return state
