"""
Different Circuit constructions for adding encoded Rz instructions.
"""

from pytket import Bit, Circuit, Qubit
from pytket.circuit import CircBox, ClBitVar, ClExpr, ClOp, WiredClExpr
from typing import List

from .state_prep import (
    get_non_ft_rz_plus_prep,
    get_ft_prep,
)
from .basic_gates import get_S, get_Z, get_Sdg, get_H, get_CX
from .iceberg_detections import iceberg_detect_zx
from .steane_corrections import classical_steane_decoding


class RzEncoding:
    """
    Base class that constructs circuits for implementing encoded Rz gates in
    a QEC circuit.

    This class is designed to be used in conjunction with the `SteaneCode` class.

    When the `SteaneCode.get_encoded_circuit` method encounters either an Rz, T or Tdg
    gate, it will request an appropriate circuit from an `RzEncoding` child class object,
    """

    def __init__(self) -> None:
        pass

    def get_circuit(self) -> Circuit:
        pass


class RzDirect(RzEncoding):
    def get_circuit(phase: float, data_qubits: List[Qubit]) -> Circuit:
        assert len(data_qubits) == 7
        c: Circuit = Circuit()
        for q in data_qubits:
            c.add_qubit(q)
        c.CX(data_qubits[5], data_qubits[3])
        c.ZZPhase(phase, data_qubits[3], data_qubits[1])
        c.CX(data_qubits[5], data_qubits[3])
        return c


class RzNonFt(RzEncoding):
    def get_circuit(
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        ancilla_bits: List[Bit],
        flag_bit: Bit,
    ) -> Circuit:
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 7
        assert len(ancilla_bits) == 7
        c: Circuit = Circuit()
        for q in data_qubits + ancilla_qubits:
            c.add_qubit(q)
        for b in ancilla_bits + [flag_bit]:
            c.add_bit(b)

        c.add_barrier(data_qubits + ancilla_qubits)

        # Non-Ft Rz|+> state preparation
        c.append(get_non_ft_rz_plus_prep(phase, ancilla_qubits))

        # Gate Teleportation
        c.add_barrier(data_qubits + ancilla_qubits)
        for ctrl, trgt in zip(data_qubits, ancilla_qubits):
            c.CX(ctrl, trgt)
        c.add_barrier(data_qubits + ancilla_qubits)

        c.Measure(ancilla_qubits[1], ancilla_bits[0])
        c.Measure(ancilla_qubits[3], ancilla_bits[1])
        c.Measure(ancilla_qubits[5], ancilla_bits[2])

        # use ancilla_bits[3] as a scratch bit
        # we know from assertion it exists
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[0], ancilla_bits[1], ancilla_bits[3]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[3], ancilla_bits[2], flag_bit],
        )
        return c


class RzFtPrep(RzEncoding):
    def __init__(self, _max_rus: int):
        self.max_rus_ = _max_rus

    def get_circuit(
        self,
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        ancilla_bits: List[Bit],
        flag_bit: Bit,
        goto_qubit: Qubit,
        goto_bit: Bit,
    ) -> Circuit:
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 7
        assert len(ancilla_bits) == 3
        c: Circuit = Circuit()
        scratch_bit: Bit = Bit("scratch", 0)
        for q in data_qubits + ancilla_qubits + [goto_qubit]:
            c.add_qubit(q)
        for b in ancilla_bits + [flag_bit, goto_bit, scratch_bit]:
            c.add_bit(b)

        c.add_barrier(data_qubits + ancilla_qubits + [goto_qubit])
        c.add_c_setbits([True], [goto_bit])
        # RUS Ft |+> Prep
        ft_prep_cbox: CircBox = CircBox(
            get_ft_prep(ancilla_qubits, goto_qubit, goto_bit)
        )
        for _ in range(self.max_rus_):
            c.add_circbox(
                ft_prep_cbox,
                ancilla_qubits + [goto_qubit, goto_bit],
                condition=goto_bit,
            )
        # Non-Ft Rz
        c.append(get_H(ancilla_qubits))
        c.append(RzDirect.get_circuit(phase, ancilla_qubits))
        # Ancilla Measurement for gate teleportation
        c.append(get_CX(data_qubits, ancilla_qubits))

        c.Measure(ancilla_qubits[1], ancilla_bits[0])
        c.Measure(ancilla_qubits[3], ancilla_bits[1])
        c.Measure(ancilla_qubits[5], ancilla_bits[2])

        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[0], ancilla_bits[1], scratch_bit],
        )

        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[2], scratch_bit, flag_bit],
        )
        return c


class RzRusNonFt(RzEncoding):
    def __init__(self, _max_rus: int):
        self.max_rus_ = _max_rus

    def get_circuit(
        self,
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        ancilla_bits: List[Bit],
        condition_bit: Bit,
        discard_bit: Bit,
    ) -> Circuit:
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 7
        assert len(ancilla_bits) == 3

        c: Circuit = Circuit()
        scratch_bit: Bit = Bit("scratch", 0)
        for q in data_qubits + ancilla_qubits:
            c.add_qubit(q)
        for b in ancilla_bits + [condition_bit, discard_bit]:
            c.add_bit(b)

        # we use condition_bit to flag whether an the RUS subcircuit has been successful
        c.add_c_setbits([True], condition_bit)
        # we add each subcircuit as a classically conditioned Circuit, which can help lower level compilation
        # use conditional blocks in QIR
        # we double the phase value with each
        # TODO: if phase_i gets multiplied into a Clifford angle then we should just add a classically conditioned Clifford and move on?
        phase_i: float = phase
        for _ in range(self.max_rus_):
            c.add_circuit(
                RzNonFt.get_circuit(
                    phase_i, data_qubits, ancilla_qubits, ancilla_bits, condition_bit
                ),
                condition=condition_bit,
            )
            phase_i *= 2
        # If RUS is not succesful then discard_bit is false

        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [discard_bit, condition_bit, scratch_bit],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [scratch_bit, condition_bit, discard_bit],
        )
        return c


class RzMeasFt(RzEncoding):
    def get_circuit(
        self,
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        ancilla_bits: List[Bit],
        syndrome_bits: List[Bit],
        condition_bit: Bit,
    ) -> Circuit:
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 7
        assert len(ancilla_bits) == 7
        assert len(syndrome_bits) == 3
        scratch_bits: List[Bit] = [Bit("scratch", i) for i in range(6)]
        c: Circuit = Circuit()
        for q in data_qubits + ancilla_qubits:
            c.add_qubit(q)
        for b in ancilla_bits + syndrome_bits + scratch_bits + [condition_bit]:
            c.add_bit(b)
        c.add_barrier(data_qubits + ancilla_qubits)

        c.append(get_non_ft_rz_plus_prep(phase, ancilla_qubits))

        c.append(get_CX(data_qubits, ancilla_qubits))
        for q, b in zip(ancilla_qubits, ancilla_bits):
            c.Measure(q, b)

        # Write parities to syndrome bits
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

        # Check error
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[0], syndrome_bits[1], scratch_bits[0]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[2], scratch_bits[0], syndrome_bits[0]],
        )

        # Correct error
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
            [ancilla_bits[3], scratch_bits[1], scratch_bits[2]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[4], scratch_bits[2], scratch_bits[3]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[5], scratch_bits[3], scratch_bits[4]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[6], scratch_bits[4], scratch_bits[5]],
        )

        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[0], scratch_bits[5], condition_bit],
        )

        return c


class RzPartFt(RzEncoding):
    def __init__(self, _max_rus: int):
        self.max_rus_ = _max_rus

    def get_prep(
        self,
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        syndrome_bits: List[Bit],
        flag_bit: Bit,
    ) -> Circuit:
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 2
        assert len(syndrome_bits) == 5

        scratch_bits: List[Bit] = [Bit("scratch", i) for i in range(3)]

        c: Circuit = Circuit()
        for q in data_qubits + ancilla_qubits:
            c.add_qubit(q)
        for b in syndrome_bits + scratch_bits + [flag_bit]:
            c.add_bit(b)

        # Make a repeat circuit
        repeat: Circuit = c.copy()

        # Ft |+> Prep
        repeat.append(get_ft_prep(data_qubits, ancilla_qubits[0], syndrome_bits[0]))
        repeat.append(get_H(data_qubits))
        # Rz Gate
        repeat.append(RzDirect.get_circuit(phase, data_qubits))
        # Check for errors

        repeat.append(
            iceberg_detect_zx(
                0, data_qubits, ancilla_qubits, syndrome_bits[1:3], Bit("dummy", 0)
            )
        )
        repeat.append(
            iceberg_detect_zx(
                1, data_qubits, ancilla_qubits, syndrome_bits[3:5], Bit("dummy", 0)
            )
        )

        # Write error output to flag_bit
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[0], syndrome_bits[1], scratch_bits[0]],
        )

        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[2], scratch_bits[0], scratch_bits[1]],
        )

        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[3], scratch_bits[1], scratch_bits[2]],
        )

        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[4], scratch_bits[2], flag_bit],
        )

        c.append(repeat)
        # Repeat this until success/max repeats value is hit
        for _ in range(self.max_rus_):
            c.add_circbox(
                CircBox(repeat), repeat.qubits + repeat.bits, condition=flag_bit
            )

        return c

    def get_circuit(
        self,
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        ancilla_bits: List[Bit],
        prep_qubits: List[Qubit],
        syndrome_bits: List[Bit],
        flag_bit: Bit,
        condition_bit: Bit,
    ):
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 7
        assert len(ancilla_bits) == 7
        assert len(prep_qubits) == 2
        assert len(syndrome_bits) == 5

        scratch_bits: List[Bit] = [Bit("scratch", i) for i in range(5)]

        c: Circuit = Circuit()
        for q in data_qubits + ancilla_qubits + prep_qubits:
            c.add_qubit(q)
        for b in (
            ancilla_bits + syndrome_bits + scratch_bits + [flag_bit, condition_bit]
        ):
            c.add_bit(b)

        c.add_barrier(data_qubits + ancilla_qubits)
        # Ft |+> state preparation with repeat until success.
        c.append(
            self.get_prep(phase, ancilla_qubits, prep_qubits, syndrome_bits, flag_bit)
        )

        # Gate teleportation
        c.append(get_CX(data_qubits, ancilla_qubits))
        # Ft Measure
        for q, b in zip(ancilla_qubits, ancilla_bits):
            c.Measure(q, b)

        # Write parity to syndrome_bits[3]
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
            [ancilla_bits[3], scratch_bits[1], scratch_bits[2]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[4], scratch_bits[2], scratch_bits[3]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[5], scratch_bits[3], scratch_bits[4]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [ancilla_bits[6], scratch_bits[4], syndrome_bits[3]],
        )

        c.append(classical_steane_decoding(ancilla_bits, syndrome_bits[:3]))

        # Check error
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[0], syndrome_bits[1], scratch_bits[0]],
        )
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitOr, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[2], scratch_bits[0], syndrome_bits[0]],
        )

        # write error to condition bit
        c.add_clexpr(
            WiredClExpr(
                expr=ClExpr(op=ClOp.BitXor, args=[ClBitVar(i) for i in range(2)]),
                bit_posn={i: i for i in range(2)},
                output_posn=[2],
            ),
            [syndrome_bits[0], syndrome_bits[3], condition_bit],
        )
        return c


class RzKNonFt(RzEncoding):
    def __init__(self, _max_bits: int):
        self.max_bits_ = _max_bits

    def resolve_phase(phase: float, max_bits: int) -> List[bool]:
        phase_: float = phase % 2
        binary_expansion: List[bool] = [False for _ in range(max_bits)]
        atol = 2**-max_bits
        for i in range(max_bits):
            val: float = 2**-i
            if phase_ >= val:
                binary_expansion[i] = True
                phase_ -= val
            if phase_ < atol:
                break
        return binary_expansion

    def get_circuit(
        self,
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        ancilla_bits: List[Bit],
        condition_bit: Bit,
        head: bool,
    ) -> Circuit:
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 7
        assert len(ancilla_bits) == 7
        #  we require recursion as we classically condition all added phase gates on the measurement of the previous step
        binary_expansion: List[Bit] = RzKNonFt.resolve_phase(phase, self.max_bits_)
        # skim binary expansion to remove last n zero terms
        while binary_expansion[-1] == False:
            assert binary_expansion.pop() == False

        c = Circuit()
        for q in data_qubits + ancilla_qubits:
            c.add_qubit(q)
        for b in ancilla_bits + [condition_bit]:
            c.add_bit(b)
        if head:
            c.add_c_setbits([True], [condition_bit])

        match binary_expansion:
            # => I
            case ():
                return c
            # => Z
            case (1,):
                c.add_circbox(
                    CircBox(get_Z(data_qubits)), data_qubits, condition=condition_bit
                )
                return c
            # => S
            case (0, 1):
                c.add_circbox(
                    CircBox(get_S(data_qubits)), data_qubits, condition=condition_bit
                )
                return c
            # => Sdg
            case (1, 1):
                c.add_circbox(
                    CircBox(get_Sdg(data_qubits)), data_qubits, condition=condition_bit
                )
                return c

        internal_circ: Circuit = RzNonFt.get_circuit(
            phase, data_qubits, ancilla_qubits, ancilla_bits, condition_bit
        )
        c.add_circbox(
            CircBox(
                internal_circ,
            ),
            internal_circ.qubits + internal_circ.bits,
            condition=condition_bit,
        )

        # recursion
        updated_phase: float = sum(
            [float(kval) * 2**-i for i, kval in enumerate(binary_expansion[1:])]
        )
        c.append(
            self.get_circuit(
                updated_phase,
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                condition_bit,
                head=False,
            ),
        )
        return c


class RzKMeasFt(RzKNonFt):
    def __init__(self, _max_bits: int):
        self.max_bits_ = _max_bits

    def get_circuit(
        self,
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        ancilla_bits: List[Bit],
        syndrome_bits: List[Bit],
        condition_bit: Bit,
        head: bool,
    ) -> Circuit:
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 7
        assert len(ancilla_bits) == 7
        assert len(syndrome_bits) == 3
        #  we require recursion as we classically condition all added phase gates on the measurement of the previous step
        binary_expansion: List[Bit] = RzKNonFt.resolve_phase(phase, self.max_bits_)
        # skim binary expansion to remove last n zero terms
        while binary_expansion[-1] == False:
            assert binary_expansion.pop() == False

        c = Circuit()
        for q in data_qubits + ancilla_qubits:
            c.add_qubit(q)
        for b in ancilla_bits + syndrome_bits + [condition_bit]:
            c.add_bit(b)
        if head:
            c.add_c_setbits([True], [condition_bit])

        match binary_expansion:
            # => I
            case ():
                return c
            # => Z
            case (1,):
                c.add_circbox(
                    CircBox(get_Z(data_qubits)), data_qubits, condition=condition_bit
                )
                return c
            # => S
            case (0, 1):
                c.add_circbox(
                    CircBox(get_S(data_qubits)), data_qubits, condition=condition_bit
                )
                return c
            # => Sdg
            case (1, 1):
                c.add_circbox(
                    CircBox(get_Sdg(data_qubits)), data_qubits, condition=condition_bit
                )
                return c

        rz_meas_c: Circuit = RzMeasFt().get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            syndrome_bits,
            condition_bit,
        )
        for b in [Bit("scratch", i) for i in range(6)]:
            c.add_bit(b)

        c.add_circbox(
            CircBox(rz_meas_c),
            rz_meas_c.qubits + rz_meas_c.bits,
            condition=condition_bit,
        )

        updated_phase: float = sum(
            [float(kval) * 2**-i for i, kval in enumerate(binary_expansion[1:])]
        )
        c.append(
            RzKMeasFt(self.max_bits_).get_circuit(
                updated_phase,
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                syndrome_bits,
                condition_bit,
                False,
            )
        )
        return c


class RzKPartFt(RzKNonFt):
    def __init__(self, _max_rus: int, _max_bits: int):
        self.max_bits_ = _max_bits
        self.max_rus_ = _max_rus

    def get_circuit(
        self,
        phase: float,
        data_qubits: List[Qubit],
        ancilla_qubits: List[Qubit],
        ancilla_bits: List[Bit],
        prep_qubits: List[Qubit],
        syndrome_bits: List[Bit],
        flag_bit: Bit,
        condition_bit: Bit,
        head: bool,
    ) -> Circuit:
        assert len(data_qubits) == 7
        assert len(ancilla_qubits) == 7
        assert len(ancilla_bits) == 7
        assert len(prep_qubits) == 2
        assert len(syndrome_bits) == 5
        #  we require recursion as we classically condition all added phase gates on the measurement of the previous step
        binary_expansion: List[Bit] = RzKNonFt.resolve_phase(phase, self.max_bits_)
        # skim binary expansion to remove last n zero terms
        while binary_expansion[-1] == False:
            assert binary_expansion.pop() == False

        c = Circuit()
        for q in data_qubits + ancilla_qubits + prep_qubits:
            c.add_qubit(q)
        for b in (
            ancilla_bits + syndrome_bits + [flag_bit, condition_bit, Bit("dummy", 0)]
        ):
            c.add_bit(b)
        if head:
            c.add_c_setbits([True], [condition_bit])

        match binary_expansion:
            # => I
            case ():
                return c
            # => Z
            case (1,):
                c.add_circbox(
                    CircBox(get_Z(data_qubits)), data_qubits, condition=condition_bit
                )
                return c
            # => S
            case (0, 1):
                c.add_circbox(
                    CircBox(get_S(data_qubits)), data_qubits, condition=condition_bit
                )
                return c
            # => Sdg
            case (1, 1):
                c.add_circbox(
                    CircBox(get_Sdg(data_qubits)), data_qubits, condition=condition_bit
                )
                return c

        rz_part_c: Circuit = RzPartFt(self.max_rus_).get_circuit(
            phase,
            data_qubits,
            ancilla_qubits,
            ancilla_bits,
            prep_qubits,
            syndrome_bits,
            flag_bit,
            condition_bit,
        )
        for b in [Bit("scratch", i) for i in range(6)]:
            c.add_bit(b)

        c.add_circbox(
            CircBox(rz_part_c),
            rz_part_c.qubits + rz_part_c.bits,
            condition=condition_bit,
        )

        updated_phase: float = sum(
            [float(kval) * 2**-i for i, kval in enumerate(binary_expansion[1:])]
        )
        c.append(
            RzKPartFt(self.max_rus_, self.max_bits_).get_circuit(
                updated_phase,
                data_qubits,
                ancilla_qubits,
                ancilla_bits,
                prep_qubits,
                syndrome_bits,
                flag_bit,
                condition_bit,
                False,
            )
        )
        return c
