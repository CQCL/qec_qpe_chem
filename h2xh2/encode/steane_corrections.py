from pytket import Qubit, Bit, Circuit
from pytket.circuit import ClBitVar, ClExpr, ClOp, WiredClExpr, CircBox
from pytket.passes import DecomposeBoxes
from typing import Dict, List, Tuple
from .state_prep import get_non_ft_prep, get_ft_prep
from .basic_gates import get_H, get_CX, get_Measure
from itertools import product
from functools import reduce
from operator import xor


def classical_steane_decoding(
    ancilla_bits: List[Bit],
    syndrome_bits: List[Bit],
) -> Circuit:
    assert len(ancilla_bits) == 7
    assert len(syndrome_bits) == 3
    c: Circuit = Circuit()
    scratch_bits: List[Bit] = [Bit("scratch", i) for i in range(2)]
    for b in ancilla_bits + syndrome_bits + scratch_bits:
        c.add_bit(b)

    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[0], ancilla_bits[1], scratch_bits[0]],
    )
    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[2], scratch_bits[0], scratch_bits[1]],
    )
    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[3], scratch_bits[1], syndrome_bits[0]],
    )

    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[1], ancilla_bits[2], scratch_bits[0]],
    )
    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[4], scratch_bits[0], scratch_bits[1]],
    )
    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[5], scratch_bits[1], syndrome_bits[1]],
    )

    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[2], ancilla_bits[3], scratch_bits[0]],
    )
    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[5], scratch_bits[0], scratch_bits[1]],
    )
    c.add_clexpr(
        WiredClExpr(
            expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
            bit_posn={i: i for i in range(2)},
            output_posn=[2],
        ),
        [ancilla_bits[6], scratch_bits[1], syndrome_bits[2]],
    )
    return c


steane_lookup_table: Dict[Tuple[bool, bool, bool], int] = {
    (0, 0, 1): 0,
    (0, 1, 1): 1,
    (1, 1, 1): 2,
    (1, 0, 1): 3,
    (0, 1, 0): 4,
    (1, 1, 0): 5,
    (1, 0, 0): 6,
}


# Non-FT state prep provided if max_repeats = 0
def steane_z_correction(
    data_qubits: List[Qubit],
    ancilla_qubits: List[Qubit],
    ancilla_bits: List[Bit],
    syndrome_bits: List[Bit],
    goto_qubit: Qubit,
    goto_bit: Bit,
    register_bit: Bit,
    max_repeats: int = 0,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancilla_qubits) == 7
    assert len(ancilla_bits) == 7
    assert len(syndrome_bits) == 3

    correction: Circuit = Circuit()
    for q in data_qubits + ancilla_qubits + [goto_qubit]:
        correction.add_qubit(q)
    for b in ancilla_bits + syndrome_bits + [goto_bit, register_bit]:
        correction.add_bit(b)

    correction.add_barrier(data_qubits + ancilla_qubits + [goto_qubit])
    # FT plus state preparation.
    if max_repeats == 0:
        # non-FT
        correction.append(get_non_ft_prep(ancilla_qubits))
    else:
        # FT
        assert max_repeats >= 1
        correction.add_c_setbits([True], [goto_bit])
        ft_prep_circ: Circuit = get_ft_prep(ancilla_qubits, goto_qubit, goto_bit)
        for _ in range(max_repeats):
            # N.B. max_repeats == 1 => one guaranteed correction
            correction.add_circbox(
                CircBox(ft_prep_circ),
                ft_prep_circ.qubits + ft_prep_circ.bits,
                condition=goto_bit,
            )

    DecomposeBoxes().apply(correction)
    # Tranvsersal H
    correction.append(get_H(ancilla_qubits))

    # Logical (transversal) CX
    correction.append(get_CX(data_qubits, ancilla_qubits))

    # Measure Ancilla qubits
    correction.append(get_Measure(ancilla_qubits, ancilla_bits))

    correction.append(classical_steane_decoding(ancilla_bits, syndrome_bits))

    for syndrome in list(product(*[range(2)] * 3))[1:]:
        assert syndrome in steane_lookup_table
        correction.add_c_setbits([True], [register_bit])
        for index, b in enumerate(syndrome):
            correction.add_c_setbits(
                [False],
                [register_bit],
                condition_bits=[syndrome_bits[index]],
                condition_value=int(b) ^ 1,
            )
        correction.X(
            data_qubits[steane_lookup_table[syndrome[::-1]]],
            condition_bits=[register_bit],
            condition_value=1,
        )
    return correction


# Non-FT state prep provided if max_repeats = 0
def steane_x_correction(
    data_qubits: List[Qubit],
    ancilla_qubits: List[Qubit],
    ancilla_bits: List[Qubit],
    syndrome_bits: List[Bit],
    goto_qubit: Qubit,
    goto_bit: Bit,
    register_bit: Bit,
    max_repeats: int = 0,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancilla_qubits) == 7
    assert len(ancilla_bits) == 7
    assert len(syndrome_bits) == 3

    correction: Circuit = Circuit()
    for q in data_qubits + ancilla_qubits + [goto_qubit]:
        correction.add_qubit(q)
    for b in ancilla_bits + syndrome_bits + [goto_bit, register_bit]:
        correction.add_bit(b)

    correction.add_barrier(data_qubits + ancilla_qubits + [goto_qubit])
    # FT 0 state preparation.
    if max_repeats == 0:
        # non-FT
        correction.append(get_non_ft_prep(ancilla_qubits))
    else:
        # FT
        assert max_repeats >= 1
        correction.add_c_setbits([True], [goto_bit])
        ft_prep_circ: Circuit = get_ft_prep(ancilla_qubits, goto_qubit, goto_bit)
        for _ in range(max_repeats):
            # N.B. max_repeats == 1 => one guaranteed correction

            correction.add_circbox(
                CircBox(ft_prep_circ),
                ft_prep_circ.qubits + ft_prep_circ.bits,
                condition=goto_bit,
            )

    DecomposeBoxes().apply(correction)
    # Logical (transversal) CX
    correction.append(get_CX(ancilla_qubits, data_qubits))
    # Logical (tranversal) H
    correction.append(get_H(ancilla_qubits))

    # Measure Ancilla qubits
    for q, b in zip(ancilla_qubits, ancilla_bits):
        correction.Measure(q, b)

    correction.append(classical_steane_decoding(ancilla_bits, syndrome_bits))

    for syndrome in list(product(*[range(2)] * 3))[1:]:
        assert syndrome in steane_lookup_table
        correction.add_c_setbits([True], [register_bit])
        for index, b in enumerate(syndrome):
            correction.add_c_setbits(
                [False],
                [register_bit],
                condition_bits=[syndrome_bits[index]],
                condition_value=int(b) ^ 1,
            )
        correction.Z(
            data_qubits[steane_lookup_table[syndrome[::-1]]],
            condition_bits=[register_bit],
            condition_value=1,
        )

    return correction


def syndrome_from_readout(
    readout: Tuple[int, int, int, int, int, int, int],
) -> Tuple[int, int, int]:
    assert len(readout) == 7
    return (
        sum([readout[i] for i in [0, 1, 2, 3]]) % 2,
        sum([readout[i] for i in [1, 2, 4, 5]]) % 2,
        sum([readout[i] for i in [2, 3, 5, 6]]) % 2,
    )


def readout_correction(
    readout: Tuple[int, int, int, int, int, int, int],
) -> Tuple[int, int, int, int, int, int, int]:
    # n.b. this does not edit the input readout, instead copying and editing a new object
    readout_copy: List[int] = list(readout)[:]
    syndrome: Tuple[int, int, int] = syndrome_from_readout(readout_copy)
    if syndrome == (0, 0, 0):
        return tuple(readout_copy)
    assert syndrome in steane_lookup_table
    flip: int = steane_lookup_table[syndrome[::-1]]
    readout_copy[flip] = int(not readout_copy[flip])
    return tuple(readout_copy)
