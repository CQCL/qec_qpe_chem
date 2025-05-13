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

from h2xh2.encode import get_non_ft_prep, get_ft_prep  # type: ignore
from pytket import Bit, Circuit, Qubit
from pytket.backends.backendresult import BackendResult
from pytket.circuit import CircBox, UnitID
from typing import List
from utils import compile_and_run


def test_non_ft_prep_identity() -> None:
    data_qubits: List[Qubit] = [Qubit("dq", i) for i in range(7)]
    bits: List[Bit] = [Bit("b", i) for i in range(7)]
    c: Circuit = get_non_ft_prep(data_qubits)
    for q, b in zip(data_qubits, bits):
        c.add_bit(b)
        c.Measure(q, b)

    # should always return even parity strings
    bs_set = set()
    for bitstring in compile_and_run(c, 100).get_counts(cbits=bits):
        bs_set.add(bitstring)
        assert sum(bitstring) % 2 == 0


def test_non_ft_prep_x() -> None:
    data_qubits: List[Qubit] = [Qubit("dq", i) for i in range(7)]
    bits: List[Bit] = [Bit("b", i) for i in range(7)]
    c: Circuit = get_non_ft_prep(data_qubits)
    for q, b in zip(data_qubits, bits):
        c.add_bit(b)
        c.X(q)
        c.Measure(q, b)

    # should always return odd parity strings
    for bitstring in compile_and_run(c, 100).get_counts(cbits=bits):
        assert sum(bitstring) % 2 == 1


def test_ft_prep_identity() -> None:
    data_qubits: List[Qubit] = [Qubit("dq", i) for i in range(7)]
    bits: List[Bit] = [Bit("b", i) for i in range(7)]
    goto_qubit: Qubit = Qubit("goto", 0)
    goto_bit: Bit = Bit("goto_bit", 0)

    c: Circuit = get_ft_prep(data_qubits, goto_qubit, goto_bit)
    for q, b in zip(data_qubits, bits):
        c.add_bit(b)
        c.Measure(q, b)

    result: BackendResult = compile_and_run(c, 100)
    # noiseless simulation => always even parity
    for bitstring in result.get_counts(cbits=bits):
        assert sum(bitstring) % 2 == 0
    # noiseless simulation => goto_bit stays off
    assert list(result.get_counts(cbits=[goto_bit]).keys()) == [(0,)]


def test_ft_prep_cond_x_off() -> None:
    data_qubits: List[Qubit] = [Qubit("dq", i) for i in range(7)]
    bits: List[Bit] = [Bit("b", i) for i in range(7)]
    goto_qubit: Qubit = Qubit("goto", 0)
    goto_bit: Bit = Bit("goto_bit", 0)
    condition_bit: Bit = Bit("cond", 0)

    c: Circuit = get_ft_prep(data_qubits, goto_qubit, goto_bit)
    c.add_bit(condition_bit)
    c.add_c_setbits([False], [condition_bit])
    logical_x: Circuit = Circuit(7)
    for i in range(7):
        logical_x.X(i)
    c.add_circbox(CircBox(logical_x), data_qubits, condition=condition_bit)

    for q, b in zip(data_qubits, bits):
        c.add_bit(b)
        c.Measure(q, b)

    result: BackendResult = compile_and_run(c, 100)
    # noiseless simulation => always even parity
    for bitstring in result.get_counts(cbits=bits):
        assert sum(bitstring) % 2 == 0
    # noiseless simulation => goto_bit stays off
    assert list(result.get_counts(cbits=[goto_bit]).keys()) == [(0,)]


def test_ft_prep_cond_x_on() -> None:
    data_qubits: List[Qubit] = [Qubit("dq", i) for i in range(7)]
    bits: List[Bit] = [Bit("b", i) for i in range(7)]
    goto_qubit: Qubit = Qubit("goto", 0)
    goto_bit: Bit = Bit("goto_bit", 0)
    condition_bit: Bit = Bit("cond", 0)

    c: Circuit = get_ft_prep(data_qubits, goto_qubit, goto_bit)
    c.add_bit(condition_bit)
    c.add_c_setbits([True], [condition_bit])
    logical_x: Circuit = Circuit(7)
    for i in range(7):
        logical_x.X(i)
    c.add_circbox(CircBox(logical_x), data_qubits, condition=condition_bit)

    for q, b in zip(data_qubits, bits):
        c.add_bit(b)
        c.Measure(q, b)

    result: BackendResult = compile_and_run(c, 100)
    # noiseless simulation => always odd parity
    for bitstring in result.get_counts(cbits=bits):
        assert sum(bitstring) % 2 == 1
    # noiseless simulation => goto_bit stays off
    assert list(result.get_counts(cbits=[goto_bit]).keys()) == [(0,)]


def test_ft_cond_prep_on() -> None:
    data_qubits: List[Qubit] = [Qubit("dq", i) for i in range(7)]
    bits: List[Bit] = [Bit("b", i) for i in range(7)]
    goto_qubit: Qubit = Qubit("goto", 0)
    goto_bit: Bit = Bit("goto_bit", 0)
    condition_bit: Bit = Bit("cond", 0)
    c: Circuit = Circuit()
    for q in data_qubits + [goto_qubit]:
        c.add_qubit(q)
    for b in bits + [goto_bit, condition_bit]:
        c.add_bit(b)

    prep: Circuit = get_ft_prep(data_qubits, goto_qubit, goto_bit)

    c.add_c_setbits([True], [condition_bit])
    args: List[UnitID] = prep.qubits + prep.bits  # type: ignore

    cbox: CircBox = CircBox(prep)
    c.add_circbox(cbox, args, condition=condition_bit)
    for q, b in zip(data_qubits, bits):
        c.Measure(q, b)

    result: BackendResult = compile_and_run(c, 10)
    # noiseless simulation => always even parity
    for bitstring in result.get_counts(cbits=bits):
        assert sum(bitstring) % 2 == 0
    # noiseless simulation => goto_bit stays off
    assert list(result.get_counts(cbits=[goto_bit]).keys()) == [(0,)]


def test_ft_cond_prep_off() -> None:
    data_qubits: List[Qubit] = [Qubit("dq", i) for i in range(7)]
    bits: List[Bit] = [Bit("b", i) for i in range(7)]
    goto_qubit: Qubit = Qubit("goto", 0)
    goto_bit: Bit = Bit("goto_bit", 0)
    condition_bit: Bit = Bit("cond", 0)
    c: Circuit = Circuit()
    for q in data_qubits + [goto_qubit]:
        c.add_qubit(q)
    for b in bits + [goto_bit, condition_bit]:
        c.add_bit(b)

    prep: Circuit = get_ft_prep(data_qubits, goto_qubit, goto_bit)

    c.add_c_setbits([False], [condition_bit])
    args: List[UnitID] = prep.qubits + prep.bits  # type: ignore

    cbox: CircBox = CircBox(prep)
    c.add_circbox(cbox, args, condition=condition_bit)
    for q, b in zip(data_qubits, bits):
        c.Measure(q, b)

    result: BackendResult = compile_and_run(c, 10)
    assert list(result.get_counts(cbits=bits + [goto_bit]).keys()) == [
        (0, 0, 0, 0, 0, 0, 0, 0)
    ]
