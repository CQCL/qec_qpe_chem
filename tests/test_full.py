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
    steane_z_correct,
    iceberg_x_0_detect,
    iceberg_w_0_detect,
    iceberg_z_0_detect,
    get_encoded_circuit,
    get_decoded_result,
    RzMode,
    RzOptionsBinFracNonFT,
    RzOptionsBinFracPartFT,
)

from pytket.circuit import Circuit, Bit, Qubit, Pauli, PauliExpBox
from pytket.backends.backendresult import BackendResult
from utils import compile_and_run


def test_basic():
    encoded: Circuit = get_encoded_circuit(Circuit(2, 2).X(1).measure_all())
    logical_result: BackendResult = get_decoded_result(compile_and_run(encoded, 10))
    assert list(logical_result.get_counts().keys()) == [(0, 1)]


def test_barriers():
    encoded: Circuit = get_encoded_circuit(
        Circuit(1, 1)
        .add_barrier([Qubit(0)])
        .H(0)
        .add_barrier([Qubit(0)])
        .H(0)
        .measure_all()
    )
    logical_result: BackendResult = get_decoded_result(compile_and_run(encoded, 10))
    assert list(logical_result.get_counts().keys()) == [(0,)]


def test_bin_frac_rz_peb():
    n_bits: int = 5
    phase: float = 3 * 2 ** -(n_bits - 1)
    logical: Circuit = (
        Circuit(1, 1)
        .H(0)
        .Rz(phase, 0)
        .add_barrier([Qubit(0)])
        .add_pauliexpbox(
            PauliExpBox([Pauli.Z], -phase),
            [0],
        )
        .H(0)
        .add_barrier([Qubit(0)])
        .add_custom_gate(steane_z_correct, [], [0])
        .measure_all()
    )
    encoded: Circuit = get_encoded_circuit(
        logical,
        rz_mode=RzMode.BIN_FRAC_NON_FT,  # rz_mode=RzMode.BIN_FRAC_PART_FT
        rz_options=RzOptionsBinFracNonFT(max_bits=n_bits),
    )

    raw_result: BackendResult = compile_and_run(encoded, 10)
    logical_result: BackendResult = get_decoded_result(raw_result)
    assert list(logical_result.get_counts().keys()) == [(0,)]


def test_bin_frac_rz_tdg():
    phase: float = 0.25
    logical: Circuit = (
        Circuit(1, 1)
        .H(0)
        .Rz(phase, 0)
        .add_barrier([Qubit(0)])
        .Tdg(0)
        .H(0)
        .add_barrier([Qubit(0)])
        .add_custom_gate(steane_z_correct, [], [0])
        .measure_all()
    )
    encoded: Circuit = get_encoded_circuit(
        logical,
        rz_mode=RzMode.BIN_FRAC_NON_FT,  # rz_mode=RzMode.BIN_FRAC_PART_FT
        rz_options=RzOptionsBinFracNonFT(max_bits=5),
    )
    logical_result: BackendResult = get_decoded_result(compile_and_run(encoded, 10))
    assert list(logical_result.get_counts().keys()) == [(0,)]


def test_part_goto_rz_tdg():
    phase: float = 0.25
    logical: Circuit = (
        Circuit(1, 1)
        .H(0)
        .Rz(phase, 0)
        .add_barrier([Qubit(0)])
        .Tdg(0)
        .H(0)
        .add_barrier([Qubit(0)])
        .add_custom_gate(steane_z_correct, [], [0])
        .measure_all()
    )
    encoded: Circuit = get_encoded_circuit(
        logical,
        rz_mode=RzMode.BIN_FRAC_PART_FT,
        rz_options=RzOptionsBinFracPartFT(10, 10),
    )
    result: BackendResult = compile_and_run(encoded, 10)
    logical_result: BackendResult = get_decoded_result(result)
    assert list(logical_result.get_counts().keys()) == [(0,)]


def test_discard():
    encoded: Circuit = get_encoded_circuit(
        Circuit(1, 1).add_custom_gate(iceberg_x_0_detect, [], [0]).measure_all()
    )
    logical_result: BackendResult = get_decoded_result(compile_and_run(encoded, 10))
    assert list(logical_result.get_counts().keys()) == [(0,)]
    encoded.add_c_setbits([True], [Bit("iceberg_discard_b", 0)])
    assert len(get_decoded_result(compile_and_run(encoded, 10)).get_counts()) == 0


def test_iceberg_w0():
    encoded: Circuit = get_encoded_circuit(
        Circuit(1, 1).add_custom_gate(iceberg_w_0_detect, [], [0]).measure_all()
    )
    logical_result: BackendResult = get_decoded_result(compile_and_run(encoded, 10))
    assert list(logical_result.get_counts().keys()) == [(0,)]


def test_iceberg_x0():
    encoded: Circuit = get_encoded_circuit(
        Circuit(1, 1).add_custom_gate(iceberg_x_0_detect, [], [0]).measure_all()
    )
    logical_result: BackendResult = get_decoded_result(compile_and_run(encoded, 10))
    assert list(logical_result.get_counts().keys()) == [(0,)]


def test_iceberg_z0():
    encoded: Circuit = get_encoded_circuit(
        Circuit(1, 1).add_custom_gate(iceberg_z_0_detect, [], [0]).measure_all()
    )
    logical_result: BackendResult = get_decoded_result(compile_and_run(encoded, 10))
    assert list(logical_result.get_counts().keys()) == [(0,)]
