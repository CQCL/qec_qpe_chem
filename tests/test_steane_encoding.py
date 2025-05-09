from h2xh2.encode import (  # type: ignore
    classical_steane_decoding,
    steane_z_correction,
    steane_x_correction,
    steane_lookup_table,
    get_non_ft_prep,
    get_H,
    get_Measure,
    syndrome_from_readout,
)
from pytket import Bit, Circuit, Qubit
from pytket.backends.backendresult import BackendResult
from typing import List
from itertools import product
from utils import compile_and_run


def test_classical_steane_decoding() -> None:
    ancilla_bits: List[Bit] = [Bit("ancilla", i) for i in range(7)]
    syndrome_bits: List[Bit] = [Bit("syndrome", i) for i in range(3)]

    for syndrome in list(product(*[range(2)] * 3))[1:]:
        c: Circuit = Circuit(7)
        c.X(steane_lookup_table[syndrome])
        for q, b in zip(c.qubits, ancilla_bits):
            c.add_bit(b)
            c.Measure(q, b)
        c.append(classical_steane_decoding(ancilla_bits, syndrome_bits))
        r: BackendResult = compile_and_run(c, 20)
        assert list(r.get_counts(cbits=syndrome_bits).keys()) == [syndrome[::-1]]


def test_steane_z_correction() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    syndrome_bits: List[Bit] = [Bit("syndrome", i) for i in range(3)]
    goto_qubit: Qubit = Qubit("goto_q", 0)
    goto_bit: Bit = Bit("goto_b", 0)
    register_bit: Bit = Bit("reg_b", 0)

    # the steane_correct_z method should be able to fix single X errors on
    # any data qubit - emulate it and confirm this is true
    for error_index in range(1):
        c: Circuit = get_non_ft_prep(data_qubits)
        c.add_barrier(data_qubits)
        c.X(data_qubits[error_index])
        c.append(
            steane_z_correction(
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                syndrome_bits,
                goto_qubit,
                goto_bit,
                register_bit,
                2,
            )
        )
        c.append(get_Measure(data_qubits, data_bits))

        r: BackendResult = compile_and_run(c, 10)
        for k in r.get_counts(cbits=data_bits + syndrome_bits):  # type: ignore
            # check artifical error is detected
            assert sum(k[7:]) > 0
            # check artifical error has been corrected
            assert syndrome_from_readout(k[:7]) == (0, 0, 0)
        assert list(r.get_counts(cbits=[goto_bit]).keys()) == [(0,)]


def test_steane_x_correction() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    syndrome_bits: List[Bit] = [Bit("syndrome", i) for i in range(3)]
    goto_qubit: Qubit = Qubit("goto_q", 0)
    goto_bit: Bit = Bit("goto_b", 0)
    register_bit: Bit = Bit("reg_b", 0)

    # the steane_correct_x method should be able to fix single Z errors on
    # any data qubit - emulate it and confirm this is true
    for error_index in range(7):
        c: Circuit = get_non_ft_prep(data_qubits)
        c.append(get_H(data_qubits))
        c.add_barrier(data_qubits)
        c.Z(data_qubits[error_index])
        c.append(
            steane_x_correction(
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                syndrome_bits,
                goto_qubit,
                goto_bit,
                register_bit,
                2,
            )
        )
        c.append(get_Measure(data_qubits, data_bits))
        r: BackendResult = compile_and_run(c, 10)
        for k in r.get_counts(cbits=data_bits + syndrome_bits):  # type: ignore
            # check artifical error is detected
            assert sum(k[7:]) > 0
            # check artifical error has been corrected
            assert syndrome_from_readout(k[:7]) == (0, 0, 0)
        assert list(r.get_counts(cbits=[goto_bit]).keys()) == [(0,)]


def test_steane_xz_correction() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    syndrome_bits: List[Bit] = [Bit("syndrome", i) for i in range(3)]
    goto_qubit: Qubit = Qubit("goto_q", 0)
    goto_bit: Bit = Bit("goto_b", 0)
    register_bit: Bit = Bit("reg_b", 0)

    # the steane_correct_z and steane_correct_z methods
    # together should be able to fix single X and Z errors
    for error_index_pair in [(i, j) for i in range(7) for j in range(7)]:
        c: Circuit = get_non_ft_prep(data_qubits)
        c.append(get_H(data_qubits))
        c.add_barrier(data_qubits)
        c.Z(data_qubits[error_index_pair[0]])
        c.X(data_qubits[error_index_pair[1]])
        c.append(
            steane_x_correction(
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                syndrome_bits,
                goto_qubit,
                goto_bit,
                register_bit,
                2,
            )
        )
        c.append(
            steane_z_correction(
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                syndrome_bits,
                goto_qubit,
                goto_bit,
                register_bit,
                2,
            )
        )
        c.append(get_Measure(data_qubits, data_bits))
        r: BackendResult = compile_and_run(c, 5)
        for k in r.get_counts(cbits=data_bits + syndrome_bits):  # type: ignore
            # check artifical error is detected
            assert sum(k[7:]) > 0
            # check artifical error has been corrected
            assert syndrome_from_readout(k[:7]) == (0, 0, 0)
        assert list(r.get_counts(cbits=[goto_bit]).keys()) == [(0,)]
