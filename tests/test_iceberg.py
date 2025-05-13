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

from h2xh2.encode import (  # type: ignore
    iceberg_detect_zx,
    iceberg_detect_z,
    iceberg_detect_x,
    get_H,
    get_non_ft_prep,
    get_Measure,
)
from pytket.backends.backendresult import BackendResult
from utils import compile_and_run
from pytket import Circuit, Qubit, Bit
from typing import List


def test_iceberg_zx_zerror_detection():
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(2)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(2)]
    discard_bit: Bit = Bit("discard", 0)

    for error_index in range(7):
        for syndrome_index, syndrome in enumerate(
            [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
        ):
            c: Circuit = get_non_ft_prep(data_qubits)
            c.append(get_H(data_qubits))
            c.add_barrier(data_qubits)

            # Apply Z error
            c.Z(data_qubits[error_index])
            # Apply iceberg ZX detection
            iceberg_zx_circuit: Circuit = iceberg_detect_zx(
                syndrome_index,
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                discard_bit,
            )
            c.append(iceberg_zx_circuit)

            c.append(get_H(data_qubits))
            c.append(get_Measure(data_qubits, data_bits))

            r: BackendResult = compile_and_run(c, 20)
            for ancilla_result in r.get_counts(cbits=ancilla_bits):
                if error_index in syndrome:
                    assert ancilla_result == (0, 1)
                else:
                    assert ancilla_result == (0, 0)


def test_iceberg_zx_xerror_detection():
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(2)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(2)]
    discard_bit: Bit = Bit("discard", 0)

    for error_index in range(7):
        for syndrome_index, syndrome in enumerate(
            [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
        ):
            c: Circuit = get_non_ft_prep(data_qubits)
            c.append(get_H(data_qubits))
            c.add_barrier(data_qubits)

            # Apply X error
            c.X(data_qubits[error_index])
            # Apply iceberg ZX detection
            iceberg_zx_circuit: Circuit = iceberg_detect_zx(
                syndrome_index,
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                discard_bit,
            )
            c.append(iceberg_zx_circuit)

            c.append(get_Measure(data_qubits, data_bits))

            result: BackendResult = compile_and_run(c, n_shots=10)
            for ancilla_result in result.get_counts(cbits=ancilla_bits):
                if error_index in syndrome:
                    assert ancilla_result == (1, 0)
                else:
                    assert ancilla_result == (0, 0)


def test_iceberg_z_xerror_detection():
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(2)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(2)]
    discard_bit: Bit = Bit("discard", 0)

    for error_index in range(7):
        for syndrome_index, syndrome in enumerate(
            [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
        ):
            c: Circuit = get_non_ft_prep(data_qubits)
            c.add_barrier(data_qubits)

            # Apply X error
            c.X(data_qubits[error_index])
            # Apply iceberg Z detection
            iceberg_z_circuit: Circuit = iceberg_detect_z(
                syndrome_index, data_qubits, ancilla_qubits, ancilla_bits, discard_bit
            )
            c.append(iceberg_z_circuit)

            c.add_barrier(data_qubits)
            c.append(get_Measure(data_qubits, data_bits))

            result: BackendResult = compile_and_run(c, n_shots=10)
            for ancilla_result in result.get_counts(cbits=ancilla_bits):
                if error_index in syndrome:
                    assert ancilla_result == (1, 0)
                else:
                    assert ancilla_result == (0, 0)


def test_iceberg_x_zerror_detection():
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(2)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(2)]
    discard_bit: Bit = Bit("discard", 0)

    for error_index in range(7):
        for syndrome_index, syndrome in enumerate(
            [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
        ):
            c: Circuit = get_non_ft_prep(data_qubits)
            c.add_barrier(data_qubits)
            c.append(get_H(data_qubits))

            # Apply Z error
            c.Z(data_qubits[error_index])
            # Apply iceberg X detection
            iceberg_x_circuit: Circuit = iceberg_detect_x(
                syndrome_index, data_qubits, ancilla_qubits, ancilla_bits, discard_bit
            )
            c.append(iceberg_x_circuit)

            c.add_barrier(data_qubits)
            c.append(get_Measure(data_qubits, data_bits))

            result: BackendResult = compile_and_run(c, n_shots=10)
            for ancilla_result in result.get_counts(cbits=ancilla_bits):
                if error_index in syndrome:
                    assert ancilla_result == (1, 0)
                else:
                    assert ancilla_result == (0, 0)
