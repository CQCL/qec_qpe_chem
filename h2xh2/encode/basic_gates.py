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

from pytket.circuit import Bit, Circuit, Qubit, Pauli
from typing import List
from itertools import pairwise


# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.041058


def get_H(data_qubits: List[Qubit]) -> Circuit:
    assert len(data_qubits) == 7
    c: Circuit = Circuit()
    for q in data_qubits:
        c.add_qubit(q)
        c.H(q)

    return c


def get_X(data_qubits: List[Qubit]) -> Circuit:
    assert len(data_qubits) == 7
    c: Circuit = Circuit()
    for q in [data_qubits[1], data_qubits[3], data_qubits[5]]:
        c.add_qubit(q)
        c.X(q)

    return c


def get_Y(data_qubits: List[Qubit]) -> Circuit:
    assert len(data_qubits) == 7
    c: Circuit = Circuit()
    for q in [data_qubits[1], data_qubits[3], data_qubits[5]]:
        c.add_qubit(q)
        c.Y(q)

    return c


def get_Z(data_qubits: List[Qubit]) -> Circuit:
    assert len(data_qubits) == 7
    c: Circuit = Circuit()
    for q in [data_qubits[1], data_qubits[3], data_qubits[5]]:
        c.add_qubit(q)
        c.Z(q)

    return c


def get_S(data_qubits: List[Qubit]) -> Circuit:
    assert len(data_qubits) == 7
    c: Circuit = Circuit()
    for q in data_qubits:
        c.add_qubit(q)
        c.Sdg(q)

    return c


def get_Sdg(data_qubits: List[Qubit]) -> Circuit:
    assert len(data_qubits) == 7
    c: Circuit = Circuit()
    for q in data_qubits:
        c.add_qubit(q)
        c.S(q)

    return c


def get_V(data_qubits: List[Qubit]) -> Circuit:
    assert len(data_qubits) == 7
    c: Circuit = Circuit()
    for q in data_qubits:
        c.add_qubit(q)
        c.Vdg(q)

    return c


def get_Vdg(data_qubits: List[Qubit]) -> Circuit:
    assert len(data_qubits) == 7
    c: Circuit = Circuit()
    for q in data_qubits:
        c.add_qubit(q)
        c.V(q)

    return c


def get_CX(control_qubits: List[Qubit], target_qubits: List[Qubit]) -> Circuit:
    c: Circuit = Circuit()
    assert len(control_qubits) == 7
    assert len(target_qubits) == 7
    for control, target in zip(control_qubits, target_qubits):
        c.add_qubit(control)
        c.add_qubit(target)
        c.CX(control, target)

    return c


def get_Measure(qubits: List[Qubit], bits: List[Bit]) -> Circuit:
    assert len(qubits) == 7
    assert len(bits) == 7
    c: Circuit = Circuit()
    for q, b in zip(qubits, bits):
        c.add_qubit(q)
        c.add_bit(b)
        c.Measure(q, b)

    return c


def get_Pauli_exponential(
    all_data_qubits: List[List[Qubit]], pauli_letters: List[Pauli], phase: float
):
    assert len(pauli_letters) == len(all_data_qubits)
    paulis_collected: List[Pauli] = []
    qubits_collected: List[Qubit] = []
    # we might flip the angle
    phase_corrected: float = phase
    for pauli, data_qubits in zip(pauli_letters, all_data_qubits):
        if pauli == Pauli.Y:
            phase_corrected *= -1
        paulis_collected += [pauli] * 3
        qubits_collected += [data_qubits[1], data_qubits[3], data_qubits[5]]

    assert len(qubits_collected) > 1

    basis_change: Circuit = Circuit()
    c: Circuit = Circuit()
    for q, p in zip(qubits_collected, paulis_collected):
        assert p != Pauli.I
        c.add_qubit(q)
        basis_change.add_qubit(q)
        if p == Pauli.X:
            basis_change.H(q)
        if p == Pauli.Y:
            basis_change.V(q)

    for q_pair in pairwise(qubits_collected[:-1]):
        basis_change.CX(q_pair[0], q_pair[1])

    c.append(basis_change)
    c.ZZPhase(phase_corrected, qubits_collected[-1], qubits_collected[-2])
    c.append(basis_change.dagger())

    return c
