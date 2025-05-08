from pytket.circuit import Circuit, Bit, Qubit, OpType
from typing import List
from .iceberg_detections import iceberg_detect_zx


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
    ).CX(data_qubits[0], data_qubits[3]).CX(data_qubits[4], data_qubits[1]).CX(
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
    ).CX(data_qubits[0], data_qubits[3]).CX(data_qubits[4], data_qubits[1]).CX(
        data_qubits[3], data_qubits[2]
    )
    ft_prep_circ.add_barrier([data_qubits[1], data_qubits[3], data_qubits[5]])
    ft_prep_circ.CX(data_qubits[1], goto_qubit).CX(data_qubits[3], goto_qubit).CX(
        data_qubits[5], goto_qubit
    )
    ft_prep_circ.Measure(goto_qubit, goto_bit)
    return ft_prep_circ


def get_non_ft_rz_plus_prep(phase: float, data_qubits: List[Qubit]) -> Circuit:
    c: Circuit = Circuit()
    assert len(data_qubits) == 7
    for q in data_qubits:
        c.add_qubit(q)
        c.add_gate(OpType.Reset, [q])

    c.H(data_qubits[0]).H(data_qubits[4]).H(data_qubits[6]).CX(
        data_qubits[0], data_qubits[1]
    ).CX(data_qubits[4], data_qubits[5]).CX(data_qubits[6], data_qubits[3]).CX(
        data_qubits[6], data_qubits[5]
    ).CX(data_qubits[4], data_qubits[2]).CX(data_qubits[0], data_qubits[3]).CX(
        data_qubits[4], data_qubits[1]
    ).XXPhase(phase, data_qubits[3], data_qubits[4]).CX(data_qubits[3], data_qubits[2])
    for q in data_qubits:
        c.H(q)
    return c


def get_prep_rz_part_ft_goto(phase: float, data_qubits: List[Qubit], ancilla_qubits: List[Qubit], syndrome_bits: List[Bit], flag_bit: Bit, n_rus: int) -> Circuit:
    c: Circuit = Circuit()
    assert len(data_qubits) == 7
    assert len(ancilla_qubits) == 2
    assert len(syndrome_bits) == 5

    discard_bits: List[Bit] = [Bit("discard", 0), Bit("discard", 1)]
    for q in data_qubits + ancilla_qubits:
        c.add_qubit(q)

    for b in syndrome_bits + [flag_bit] + discard_bits:
        c.add_bit(b)

    
    # FT |+> state prep
    c.append(get_ft_prep(data_qubits, ancilla_qubits[0], syndrome_bits[0]))
    for q in data_qubits:
        c.H(q)

    # non FT zero
    c.CX(data_qubits[5], data_qubits[3])
    c.ZZPhase(phase, data_qubits[3], data_qubits[1])
    c.CX(data_qubits[5], data_qubits[3])


    # detect errors with Iceberg gadget
    c.append(iceberg_detect_zx(0, data_qubits, ancilla_qubits, syndrome_bits[1:3], discard_bits[0]))
    c.append(iceberg_detect_zx(1, data_qubits, ancilla_qubits, syndrome_bits[3:5], discard_bits[1]))
    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
            bit_posn = {i:i for i in range(2)},
            output_posn = [2]
        ),
        discard_bits + [flag_bit],
    )
    goto_c: Circuit = c.copy()
    # repeat construction while flag_bit is false, i.e. while iceberg detection finds errors
    for _ in range(n_rus - 1):
        c.add_circbox(
            CircBox(goto_c),
            goto_c.qubits + goto_c.bits,
            condition = flag_bit
        )  
    return c

    