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

from pytket.circuit import Qubit, Bit, Circuit, ClBitVar, ClExpr, ClOp, WiredClExpr
from typing import List, Tuple


def iceberg_detect_x(
    index: int,
    data_qubits: List[Qubit],
    ancilla_qubits: List[Qubit],
    ancilla_bits: List[Bit],
    discard_bit: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancilla_qubits) == 2
    assert len(ancilla_bits) == 2
    detection: Circuit = Circuit()
    scratch_bits: List[Bit] = [Bit("scratch", 0), Bit("scratch", 1)]
    for q in data_qubits + ancilla_qubits:
        detection.add_qubit(q)
    for b in ancilla_bits + scratch_bits + [discard_bit]:
        detection.add_bit(b)

    detection.add_barrier(data_qubits + ancilla_qubits)
    for q in ancilla_qubits:
        detection.Reset(q)
    # XXXXIII -> 0, 1, 2, 3
    # IXXIXXI -> 1, 2, 4, 5
    # IIXXIXX -> 2, 3, 5, 6
    stabilizer_indices: Tuple[Tuple[int, int, int]] = (
        (0, 1, 2, 3),
        (1, 2, 4, 5),
        (2, 3, 5, 6),
    )
    # The X detection circuit is written to 4 qubits
    # These 4 qubits depend on the chosen "index"
    acting_qubits: List[Qubit] = [data_qubits[i] for i in stabilizer_indices[index]]
    assert len(acting_qubits) == 4
    detection.H(ancilla_qubits[0])
    detection.CX(ancilla_qubits[0], acting_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[0], ancilla_qubits[1])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[0], acting_qubits[1])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[0], acting_qubits[2])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[0], ancilla_qubits[1])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[0], acting_qubits[3])
    detection.H(ancilla_qubits[0])
    detection.Measure(ancilla_qubits[0], ancilla_bits[0])
    detection.Measure(ancilla_qubits[1], ancilla_bits[1])

    detection.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[0], ancilla_bits[1], scratch_bits[0]],
    )
    detection.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [scratch_bits[0], discard_bit, scratch_bits[1]],
    )
    detection.add_c_copybits([scratch_bits[1]], [discard_bit])
    return detection


def iceberg_detect_z(
    index: int,
    data_qubits: List[Qubit],
    ancilla_qubits: List[Qubit],
    ancilla_bits: List[Bit],
    discard_bit: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancilla_qubits) == 2
    assert len(ancilla_bits) == 2
    detection: Circuit = Circuit()
    scratch_bits: List[Bit] = [Bit("scratch", 0), Bit("scratch", 1)]
    for q in data_qubits + ancilla_qubits:
        detection.add_qubit(q)
    for b in ancilla_bits + scratch_bits + [discard_bit]:
        detection.add_bit(b)

    detection.add_barrier(data_qubits + ancilla_qubits)
    for q in ancilla_qubits:
        detection.Reset(q)

    # ZZZZIII -> 0, 1, 2, 3
    # IZZIZZI -> 1, 2, 4, 5
    # IIZZIZZ -> 2, 3, 5, 6
    stabilizer_indices: Tuple[Tuple[int, int, int]] = (
        (0, 1, 2, 3),
        (1, 2, 4, 5),
        (2, 3, 5, 6),
    )

    acting_qubits: List[Qubit] = [data_qubits[i] for i in stabilizer_indices[index]]
    assert len(acting_qubits) == 4
    detection.H(ancilla_qubits[1])
    detection.CX(acting_qubits[0], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[1], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(acting_qubits[1], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(acting_qubits[2], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[1], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(acting_qubits[3], ancilla_qubits[0])
    detection.H(ancilla_qubits[1])
    detection.Measure(ancilla_qubits[0], ancilla_bits[0])
    detection.Measure(ancilla_qubits[1], ancilla_bits[1])

    detection.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[0], ancilla_bits[1], scratch_bits[0]],
    )
    detection.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [scratch_bits[0], discard_bit, scratch_bits[1]],
    )
    detection.add_c_copybits([scratch_bits[1]], [discard_bit])
    return detection


def iceberg_detect_zx(
    index: int,
    data_qubits: List[Qubit],
    ancilla_qubits: List[Qubit],
    ancilla_bits: List[Bit],
    discard_bit: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancilla_qubits) == 2
    assert len(ancilla_bits) == 2
    detection: Circuit = Circuit()
    scratch_bits: List[Bit] = [Bit("scratch", 0), Bit("scratch", 1)]
    for q in data_qubits + ancilla_qubits:
        detection.add_qubit(q)
    for b in ancilla_bits + scratch_bits + [discard_bit]:
        detection.add_bit(b)

    detection.add_barrier(data_qubits + ancilla_qubits)
    for q in ancilla_qubits:
        detection.Reset(q)

    stabilizer_indices: Tuple[Tuple[int, int, int]] = (
        (0, 1, 2, 3),
        (1, 2, 4, 5),
        (2, 3, 5, 6),
    )
    # The ZX detection circuit is written to 4 qubits
    # These 4 qubits depend on the chosen "index"

    acting_qubits: List[Qubit] = [data_qubits[i] for i in stabilizer_indices[index]]
    assert len(acting_qubits) == 4
    detection.H(ancilla_qubits[1])
    detection.CX(ancilla_qubits[1], acting_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(acting_qubits[0], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(acting_qubits[1], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[1], acting_qubits[1])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[1], acting_qubits[2])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(acting_qubits[2], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(acting_qubits[3], ancilla_qubits[0])
    detection.add_barrier(data_qubits + ancilla_qubits)
    detection.CX(ancilla_qubits[1], acting_qubits[3])
    detection.H(ancilla_qubits[1])
    detection.Measure(ancilla_qubits[0], ancilla_bits[0])
    detection.Measure(ancilla_qubits[1], ancilla_bits[1])

    detection.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[0], ancilla_bits[1], scratch_bits[0]],
    )
    detection.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [scratch_bits[0], discard_bit, scratch_bits[1]],
    )
    detection.add_c_copybits([scratch_bits[1]], [discard_bit])

    return detection
