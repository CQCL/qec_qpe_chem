from h2xh2.encode import (
    RzDirect,
    RzNonFt,
    RzFtPrep,
    RzKNonFt,
    RzPartFt,
    RzMeasFt,
    RzKMeasFt,
    get_non_ft_prep,
    get_H,
    get_S,
    get_Sdg,
    get_Measure,
)
from pytket import Circuit, Qubit, Bit
from pytket.passes import DecomposeBoxes
from pytket.circuit import CircBox
from pytket.backends.backendresult import BackendResult
from utils import compile_and_run
from typing import List


def test_rz_direct() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]

    phase: float = 0.25
    # Non FT plus state prep
    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))

    # Sequence of Rz, S and H that give the identity
    c.append(RzDirect.get_circuit(phase, data_qubits))
    c.add_barrier(data_qubits)
    c.append(RzDirect.get_circuit(phase, data_qubits))
    c.add_barrier(data_qubits)
    c.append(RzDirect.get_circuit(-phase, data_qubits))
    c.add_barrier(data_qubits)
    c.append(RzDirect.get_circuit(phase, data_qubits))
    c.add_barrier(data_qubits)
    c.append(RzDirect.get_circuit(phase, data_qubits))
    c.add_barrier(data_qubits)
    c.append(RzDirect.get_circuit(phase, data_qubits))
    c.append(get_S(data_qubits))
    c.append(get_S(data_qubits))

    # Measure, check
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0


def test_non_ft_rz_T() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    condition_bit: Bit = Bit("cond_b", 0)

    # Non FT plus state prep
    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))
    # Non FT Magic state injection
    phase: float = 0.25
    c.append(
        RzNonFt.get_circuit(
            phase, data_qubits, ancilla_qubits, ancilla_bits, condition_bit
        )
    )
    c.add_circbox(CircBox(get_S(data_qubits)), data_qubits, condition=condition_bit)
    # Non-FT direct Rz to cancel out previous Rz, meaning we can check parity
    c.append(RzDirect.get_circuit(-phase, data_qubits))

    # X Measurement
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    DecomposeBoxes().apply(c)
    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0


def test_non_ft_rz_Tdg() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    condition_bit: Bit = Bit("cond_b", 0)

    # Non FT plus state prep
    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))
    # Non FT Magic state injection
    phase: float = -0.25
    c.append(
        RzNonFt.get_circuit(
            phase, data_qubits, ancilla_qubits, ancilla_bits, condition_bit
        )
    )
    c.add_circbox(CircBox(get_Sdg(data_qubits)), data_qubits, condition=condition_bit)
    # Non-FT direct Rz to cancel out previous Rz, meaning we can check parity
    c.append(RzDirect.get_circuit(-phase, data_qubits))

    # X Measurement
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0


def test_rz_ft_prep_T() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(3)]
    flag_bit: Bit = Bit("flag_b", 0)
    goto_qubit: Qubit = Qubit("goto_q", 0)
    goto_bit: Bit = Bit("goto_b", 0)
    # Non FT plus state prep
    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))
    # Non FT Magic state injection
    phase: float = 0.25
    c.append(
        RzFtPrep(2).get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            flag_bit,
            goto_qubit,
            goto_bit,
        )
    )
    c.add_circbox(CircBox(get_S(data_qubits)), data_qubits, condition=flag_bit)
    c.add_barrier(data_qubits)
    # Non-FT direct Rz to cancel out previous Rz, meaning we can check parity
    c.append(RzDirect.get_circuit(-phase, data_qubits))

    # X Measurement
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 50)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0
    assert list(r.get_counts(cbits=[goto_bit]).keys()) == [(0,)]


def test_rz_ft_prep_Tdg() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(3)]
    flag_bit: Bit = Bit("flag_b", 0)
    goto_qubit: Qubit = Qubit("goto_q", 0)
    goto_bit: Bit = Bit("goto_b", 0)
    # Non FT plus state prep
    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))
    # Non FT Magic state injection
    phase: float = -0.25
    c.append(
        RzFtPrep(2).get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            flag_bit,
            goto_qubit,
            goto_bit,
        )
    )
    c.add_circbox(CircBox(get_Sdg(data_qubits)), data_qubits, condition=flag_bit)
    c.add_barrier(data_qubits)
    # Non-FT direct Rz to cancel out previous Rz, meaning we can check parity
    c.append(RzDirect.get_circuit(-phase, data_qubits))

    # X Measurement
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0
    assert list(r.get_counts(cbits=[goto_bit]).keys()) == [(0,)]


def test_rzk_non_ft() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    condition_bit: Bit = Bit("condition_b", 0)

    c: Circuit = Circuit()
    # Non-FT |+> prep
    c.append(get_non_ft_prep(data_qubits))
    c.append(get_H(data_qubits))

    phase: float = 1.65625
    assert RzKNonFt.resolve_phase(phase, 6) == [True, True, False, True, False, True]

    c.append(
        RzKNonFt(6).get_circuit(
            phase, data_qubits, ancilla_qubits, ancilla_bits, condition_bit, True
        )
    )

    # Non-FT direct Rz operation to leave an identity
    c.append(RzDirect.get_circuit(-phase, data_qubits))
    # Measure
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0


def test_rz_part_ft_prep() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    prep_qubits: List[Qubit] = [Qubit("prep_q", i) for i in range(2)]
    syndrome_bits: List[Bit] = [Bit("x_synd", i) for i in range(5)]
    flag_bit: Bit = Bit("flag_b", 0)

    phase: float = 0.31
    # FT e-i*phase|+> prep
    c: Circuit = RzPartFt(1).get_prep(
        phase, data_qubits, prep_qubits, syndrome_bits, flag_bit
    )
    c.add_barrier(data_qubits)
    # undo phase
    c.append(RzDirect.get_circuit(-phase, data_qubits))
    c.add_barrier(data_qubits)
    # |+> -> |0>
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))
    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0
    assert list(r.get_counts(cbits=syndrome_bits).keys()) == [(0, 0, 0, 0, 0)]


def test_rz_part_ft():
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    prep_qubits: List[Qubit] = [Qubit("prep_q", i) for i in range(2)]
    condition_bit: Bit = Bit("condition_b", 0)
    syndrome_bits: List[Bit] = [Bit("x_synd", i) for i in range(5)]
    flag_bit: Bit = Bit("flag_b", 0)

    # Non FT plus state
    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))
    c.add_bit(condition_bit)
    c.add_c_setbits([True], [condition_bit])
    # Test magic state injection
    phase: float = 0.33
    n_rus: int = 1
    c.append(
        RzPartFt(n_rus).get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            prep_qubits,
            syndrome_bits,
            flag_bit,
            condition_bit,
        )
    )

    rz_circ: Circuit = RzDirect.get_circuit(2 * phase, data_qubits)
    # Condition Rz to undo previous Rz
    c.add_circbox(CircBox(rz_circ), rz_circ.qubits, condition=condition_bit)
    c.add_barrier(data_qubits)
    # Else, if none, undo phase
    c.append(RzDirect.get_circuit(-phase, data_qubits))
    c.add_barrier(data_qubits)
    # Measure
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0
    assert list(r.get_counts(cbits=[flag_bit]).keys()) == [(0,)]


def test_rz_meas_ft() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    condition_bit: Bit = Bit("condition_b", 0)
    syndrome_bits: List[Bit] = [Bit("synd_b", i) for i in range(3)]

    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))
    c.add_bit(condition_bit)
    c.add_c_setbits([True], [condition_bit])
    phase: float = 0.25
    # Inject magic state
    c.append(
        RzMeasFt().get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            syndrome_bits,
            condition_bit,
        )
    )
    c.add_circbox(CircBox(get_S(data_qubits)), data_qubits, condition=condition_bit)
    c.append(RzDirect.get_circuit(-phase, data_qubits))

    # X Measurement
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0


def test_rz_k_meas_ft() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    condition_bit: Bit = Bit("condition_b", 0)
    syndrome_bits: List[Bit] = [Bit("synd_b", i) for i in range(3)]

    c: Circuit = Circuit()
    # Non-FT |+> prep
    c.append(get_non_ft_prep(data_qubits))
    c.append(get_H(data_qubits))

    phase: float = 1.65625
    assert RzKNonFt.resolve_phase(phase, 6) == [True, True, False, True, False, True]

    c.append(
        RzKMeasFt(6).get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            syndrome_bits,
            condition_bit,
            True,
        )
    )
    c.add_barrier(data_qubits)
    c.append(RzDirect.get_circuit(-phase, data_qubits))

    # X Measurement
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)

    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0

def test_rz_part_ft() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    prep_qubits: List[Qubit] = [Qubit("prep_q", i) for i in range(2)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    flag_bit: Bit = Bit("flag_b", 0)
    condition_bit: Bit = Bit("condition_b", 0)
    syndrome_bits: List[Bit] = [Bit("synd_b", i) for i in range(5

    )]

    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))
    c.add_bit(condition_bit)
    c.add_c_setbits([True], [condition_bit])
    phase: float = 0.25
    # Inject magic state
    c.append(
        RzPartFt(8).get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            prep_qubits,
            syndrome_bits,
            flag_bit,
            condition_bit,
        )
    )
    c.add_circbox(CircBox(get_S(data_qubits)), data_qubits, condition=condition_bit)
    c.append(RzDirect.get_circuit(-phase, data_qubits))

    # X Measurement
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0

def test_rz_k_part_ft() -> None:
    data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
    data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    prep_qubits: List[Qubit] = [Qubit("prep_q", i) for i in range(2)]
    ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
    flag_bit: Bit = Bit("flag_b", 0)
    condition_bit: Bit = Bit("condition_b", 0)
    syndrome_bits: List[Bit] = [Bit("synd_b", i) for i in range(5

    )]

    c: Circuit = get_non_ft_prep(data_qubits)
    c.append(get_H(data_qubits))
    c.add_bit(condition_bit)
    c.add_c_setbits([True], [condition_bit])
    phase: float = 0.25
    # Inject magic state
    c.append(
        RzKPartFt(8).get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            prep_qubits,
            syndrome_bits,
            flag_bit,
            condition_bit,
        )
    )
    c.add_circbox(CircBox(get_S(data_qubits)), data_qubits, condition=condition_bit)
    c.append(RzDirect.get_circuit(-phase, data_qubits))

    # X Measurement
    c.append(get_H(data_qubits))
    c.append(get_Measure(data_qubits, data_bits))

    r: BackendResult = compile_and_run(c, 10)
    for bitstring in r.get_counts(cbits=data_bits):
        assert sum(bitstring) % 2 == 0


if __name__ == "__main__":
    test_rz_part_ft_goto()


# def test_rz_k_part_ft_goto() -> None:    
#     data_qubits: List[Qubit] = [Qubit("data_q", i) for i in range(7)]
#     data_bits: List[Bit] = [Bit("data_b", i) for i in range(7)]
#     ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
#     ancilla_bits: List[Bit] = [Bit("ancilla_b", i) for i in range(7)]
#     condition_bit: Bit = Bit("condition_b", 0)
#     syndrome_bits: List[Bit] = [Bit("synd_b", i) for i in range(3)]

#     c: Circuit = Circuit()
#     # Non-FT |+> prep
#     c.append(get_non_ft_prep(data_qubits))
#     c.append(get_H(data_qubits))

#     phase: float = 1.65625
#     assert RzKNonFt.resolve_phase(phase, 6) == [True, True, False, True, False, True]

#     c.append(
#         RzKMeasFt(6).get_circuit(
#             phase,
#             data_qubits,
#             ancilla_qubits,
#             ancilla_bits,
#             syndrome_bits,
#             condition_bit,
#             True,
#         )
#     )
#     c.add_barrier(data_qubits)
#     c.append(RzDirect.get_circuit(-phase, data_qubits))

#     # X Measurement
#     c.append(get_H(data_qubits))
#     c.append(get_Measure(data_qubits, data_bits))

#     r: BackendResult = compile_and_run(c, 10)

#     for bitstring in r.get_counts(cbits=data_bits):
#         assert sum(bitstring) % 2 == 0