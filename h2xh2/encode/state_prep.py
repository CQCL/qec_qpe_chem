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

from pytket.circuit import (
    Circuit,
    Bit,
    Qubit,
)

from typing import List


def get_non_ft_prep(data_qubits: List[Qubit]) -> Circuit:
    non_ft_prep_circ: Circuit = Circuit()
    for q in data_qubits:
        non_ft_prep_circ.add_qubit(q)
        non_ft_prep_circ.Reset(q)
    non_ft_prep_circ.H(data_qubits[0]).H(data_qubits[4]).H(data_qubits[6])
    non_ft_prep_circ.CX(data_qubits[0], data_qubits[1]).CX(
        data_qubits[4], data_qubits[5]
    ).CX(data_qubits[6], data_qubits[3]).CX(data_qubits[6], data_qubits[5]).CX(
        data_qubits[4], data_qubits[2]
    ).CX(
        data_qubits[0], data_qubits[3]
    ).CX(
        data_qubits[4], data_qubits[1]
    ).CX(
        data_qubits[3], data_qubits[2]
    )
    return non_ft_prep_circ


def get_non_ft_rz_plus_prep(phase: float, data_qubits: List[Qubit]) -> Circuit:
    non_ft_rz_plus_prep_circ: Circuit = Circuit()
    for q in data_qubits:
        non_ft_rz_plus_prep_circ.add_qubit(q)
        non_ft_rz_plus_prep_circ.Reset(q)

    non_ft_rz_plus_prep_circ.H(data_qubits[0]).H(data_qubits[4]).H(data_qubits[6])
    non_ft_rz_plus_prep_circ.CX(data_qubits[0], data_qubits[1])
    non_ft_rz_plus_prep_circ.CX(data_qubits[4], data_qubits[5])
    non_ft_rz_plus_prep_circ.CX(data_qubits[6], data_qubits[3])
    non_ft_rz_plus_prep_circ.CX(data_qubits[6], data_qubits[5])
    non_ft_rz_plus_prep_circ.CX(data_qubits[4], data_qubits[2])
    non_ft_rz_plus_prep_circ.CX(data_qubits[0], data_qubits[3])
    non_ft_rz_plus_prep_circ.CX(data_qubits[4], data_qubits[1])
    non_ft_rz_plus_prep_circ.XXPhase(phase, data_qubits[3], data_qubits[4])
    non_ft_rz_plus_prep_circ.CX(data_qubits[3], data_qubits[2])
    for q in data_qubits:
        non_ft_rz_plus_prep_circ.H(q)
    return non_ft_rz_plus_prep_circ


def get_ft_prep(data_qubits: List[Qubit], goto_qubit: Qubit, goto_bit: Bit) -> Circuit:
    ft_prep_circ: Circuit = Circuit()
    for q in data_qubits + [goto_qubit]:
        ft_prep_circ.add_qubit(q)
        ft_prep_circ.Reset(q)
    ft_prep_circ.add_bit(goto_bit)

    ft_prep_circ.H(data_qubits[0]).H(data_qubits[4]).H(data_qubits[6])
    ft_prep_circ.CX(data_qubits[0], data_qubits[1]).CX(
        data_qubits[4], data_qubits[5]
    ).CX(data_qubits[6], data_qubits[3]).CX(data_qubits[6], data_qubits[5]).CX(
        data_qubits[4], data_qubits[2]
    ).CX(
        data_qubits[0], data_qubits[3]
    ).CX(
        data_qubits[4], data_qubits[1]
    ).CX(
        data_qubits[3], data_qubits[2]
    )
    ft_prep_circ.add_barrier([data_qubits[1], data_qubits[3], data_qubits[5]])
    ft_prep_circ.CX(data_qubits[1], goto_qubit).CX(data_qubits[3], goto_qubit).CX(
        data_qubits[5], goto_qubit
    )
    ft_prep_circ.Measure(goto_qubit, goto_bit)
    return ft_prep_circ
