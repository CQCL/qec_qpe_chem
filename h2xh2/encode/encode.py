from pytket.circuit import Circuit, OpType, Qubit, Bit
from typing import List, NamedTuple, Dict
from enum import Enum
from itertools import chain
from .basic_gates import (
    get_H,
    get_X,
    get_Y,
    get_Z,
    get_S,
    get_Sdg,
    get_V,
    get_Vdg,
    get_CX,
    get_Measure,
    get_Pauli_exponential,
)

from .steane_corrections import steane_z_correction, steane_x_correction
from .iceberg_detections import iceberg_detect_x, iceberg_detect_z, iceberg_detect_zx
from .rz_encoding import RzDirect, RzKNonFt, RzKMeasFt, RzKPartFt
from .state_prep import get_non_ft_prep


class RzMode(Enum):
    """Rz encoding mode.

    DIRECT:
        Direct (iceberg-code-style) operation, which is low-cost but not scalable.
    BIN_FRAC_NON_FT:
        Indirect (gate teleportation) and deterministic operation with the angle in binary fraction (non-FT).
    BIN_FRAC_MEAS_FT:
        Indirect (gate teleportation) and deterministic operation with the angle in binary fraction (MEAS-FT).
    BBIN_FRAC_PART_FT:
        Indirect (gate teleportation) and deterministic operation with the angle in binary fraction (partially FT).
    BIN_FRAC_PART_FT_GOTO:
        Indirect (gate teleportation) and deterministic operation with the angle in binary fraction (partially FT with Goto's state prep).
    """

    DIRECT = "direct"
    BIN_FRAC_NON_FT = "bin_frac_non_ft"
    BIN_FRAC_MEAS_FT = "bin_frac_meas_ft"
    BIN_FRAC_PART_FT = "bin_frac_part_ft"
    BIN_FRAC_PART_FT_GOTO = "bin_frac_part_ft_goto"


class RzOptionsRUS(NamedTuple):
    """Options for the RzMode.RUS.

    Args:
        max_rus:
            Maximum number of repeat-until-success.
    """

    max_rus: int = 8


class RzOptionsBinFracNonFT(NamedTuple):
    """Options for the RzMode.BIN_FRAC_NON_FT.

    Args:
        max_bits:
            Maximum number of bits (i.e., max_bits - 2 = max_rus)
    """

    max_bits: int = 10


class RzOptionsBinFracMeasFT(NamedTuple):
    """Options for the RzMode.BIN_FRAC_MEAS_FT.

    Args:
        max_bits:
            Maximum number of bits (i.e., max_bits - 2 = max_rus)
    """

    max_bits: int = 10


class RzOptionsBinFracPartFT(NamedTuple):
    """Options for the RzMode.BIN_FRAC_PART_FT and
    RzMode.BIN_FRAC_PART_FT_GOTO.

    Args:
        max_bits:
            Maximum number of bits (i.e., max_bits - 2 = max_rus)
        n_rus:
            Number of RUS if PFT implementation is used.
    """

    max_bits: int = 10
    max_rus: int = 1


class EncodeData(NamedTuple):
    """Encode data to be used by the automated workflow driver.

    Args:
        n_qubits:
            Number of logical qubits.
        n_bits:
            Number of logical bits.
    """

    n_qubits: int
    n_bits: int


class EncodeOptions(NamedTuple):
    """Encdode options for the Steane code.

    Args:
        rz_mode:
            Rz operation mode. Default is the iceberg direct mode.
        rz_options:
            Options for the Rz operation.
        ft_prep:
            Enable FT state preparation (i.e. Goto state preparation) for the data register.
        n_rus_prep:
            Number of RUS for the FT state preparation.
        ft_prep_synd:
            Enable FT state preparation for the ancilla register for the Steane QEC.
        ft_rus_synd:
            Number of RUS for the FT state preparation for the Steane QEC.
        ft_prep_gate:
            Enable FT state preparation for the Rz gate teleportation gadget.
        ft_rus_gate:
            Number of RUS for the FT state preparation for the Rz gate teleportation.
    """

    # Non-FT (Rz including T) operation.
    rz_mode: RzMode = RzMode.DIRECT
    rz_options: RzOptionsRUS | RzOptionsBinFracNonFT | RzOptionsBinFracMeasFT | None = (
        None
    )

    # Use Goto state preparation (applied to the state preparation of the data register)
    ft_prep: bool = False
    n_rus_prep: int = 1

    # Use Goto state preparation (applied to the ancilla register for syndrome measurements)
    ft_prep_synd: bool = False
    n_rus_synd: int = 1

    # Use Goto state preparation (applied to the gate teleportation including the magic state injection)
    ft_prep_gate: bool = False
    n_rus_gate: int = 1


def get_encoded_circuit(
    circuit: Circuit,
    rz_mode: RzMode = RzMode.DIRECT,
    rz_options: (
        RzOptionsRUS | RzOptionsBinFracNonFT | RzOptionsBinFracPartFT | None
    ) = None,
    ft_prep: bool = False,
    n_rus_prep: int = 1,
    ft_prep_synd: bool = False,
    n_rus_synd: int = 1,
    ft_prep_gate: bool = False,
    n_rus_gate: int = 1,
) -> Circuit:
    """
    Given a pytket circuit and an Rz gate implementation, returns a new
    encoded circuit. Error detection/correction cycles are pre-assigned
    using CustomGate to define their locations.
    """
    # each qubit in the circuit requires 7 data qubits to be encoded, plus additional ancilla bits
    get_data_qubits: Dict[Qubit, List[Qubit]] = {
        q: [Qubit(str(q).replace("[", "").replace("]", ""), i) for i in range(7)]
        for q in circuit.qubits
    }
    # we just use the default "c" register for cmopatibility with workflow code
    # ideally we return this dictionary object so the user knows what to use
    get_data_bits: Dict[Bit, List[Bit]] = {
        b: [Bit("c", i + 7 * index) for i in range(7)]
        for index, b in enumerate(circuit.bits)
    }

    # we also need 7 ancilla qubits for running Steane/Iceberg cycles and for Rz gates
    # (TODO: we can provide shorter circuit depth by filling out the full device later)
    ancilla_qubits: List[Qubit] = [Qubit("ancilla_q", i) for i in range(7)]
    goto_qubit: Qubit = Qubit("goto_q", 0)
    # finally we need 2 additional qubits for Iceberg Checks if using RzPartFT
    prep_qubits: List[Qubit] = [Qubit("part_prep", i) for i in range(2)]
    # and for ease, we assign various bit registers for types of correction/z gates
    rz_ancilla_bits: List[Bit] = [Bit("rz_ancilla_b", i) for i in range(7)]
    rz_syndrome_bits: List[Bit] = [Bit("rz_syndrome_b", i) for i in range(3)]
    steane_ancilla_bits: List[Bit] = [Bit("steane_ancilla_b", i) for i in range(7)]
    steane_syndrome_bits: List[Bit] = [Bit("steane_syndrome_b", i) for i in range(3)]
    iceberg_syndrome_bits: List[Bit] = [Bit("iceberg_syndrome_b", i) for i in range(2)]
    part_ft_syndrome_bits: List[Bit] = [Bit("part_ft_syndrome_b", i) for i in range(5)]
    iceberg_discard_bit: Bit = Bit("iceberg_discard_b", 0)
    no_detected_error_bit: Bit = Bit("no_detected_error", 0)
    goto_bit: Bit = Bit("goto_b", 0)
    register_bit: Bit = Bit("register_b", 0)
    flag_bit: Bit = Bit("flag", 0)
    condition_bit: Bit = Bit("condition", 0)

    # add suitable qubits/bits to some circuit
    encoded_circuit: Circuit = Circuit()
    for q in list(chain(*get_data_qubits.values())) + ancilla_qubits:
        encoded_circuit.add_qubit(q)
    for b in (
        rz_ancilla_bits
        + rz_syndrome_bits
        + steane_ancilla_bits
        + steane_syndrome_bits
        + iceberg_syndrome_bits
        + [
            iceberg_discard_bit,
            no_detected_error_bit,
            goto_bit,
            register_bit,
            flag_bit,
            condition_bit,
        ]
    ):
        encoded_circuit.add_bit(b)

    # non-FT prep for each qubit
    # TODO: add option for FT (don't need it for immediate runs)
    for qs in get_data_qubits.values():
        encoded_circuit.append(get_non_ft_prep(qs))

    for command in circuit.get_commands():
        match command.op.type:
            case OpType.Barrier:
                # TODO: Include bits
                assert len(command.bits) == 0
                all_qubits: List[Qubit] = []
                for q in command.qubits:
                    assert q in get_data_qubits
                    all_qubits.extend(get_data_qubits[q])
                encoded_circuit.add_barrier(all_qubits)

            # all custom gates should correspond to detection/correction cycles
            # any others are rejected
            case OpType.CustomGate:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                data_qubits: List[Qubit] = get_data_qubits[command.qubits[0]]
                # match on name of custom gate to add correct correction/detection cycle
                match command.op.name:
                    case "steane_z_correct":
                        encoded_circuit.append(
                            steane_z_correction(
                                data_qubits,
                                ancilla_qubits,
                                steane_ancilla_bits,
                                steane_syndrome_bits,
                                goto_qubit,
                                goto_bit,
                                register_bit,
                                n_rus_synd - 1,
                            )
                        )

                    case "steane_x_correct":
                        encoded_circuit.append(
                            steane_x_correction(
                                data_qubits,
                                ancilla_qubits,
                                steane_ancilla_bits,
                                steane_syndrome_bits,
                                goto_qubit,
                                goto_bit,
                                register_bit,
                                n_rus_synd - 1,
                            )
                        )

                    case "iceberg_w_0_detect":
                        encoded_circuit.append(
                            iceberg_detect_zx(
                                0,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "iceberg_w_1_detect":
                        encoded_circuit.append(
                            iceberg_detect_zx(
                                1,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "iceberg_w_2_detect":
                        encoded_circuit.append(
                            iceberg_detect_zx(
                                2,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "iceberg_x_0_detect":
                        encoded_circuit.append(
                            iceberg_detect_x(
                                0,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "iceberg_x_1_detect":
                        encoded_circuit.append(
                            iceberg_detect_x(
                                1,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "iceberg_x_2_detect":
                        encoded_circuit.append(
                            iceberg_detect_x(
                                2,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "iceberg_z_0_detect":
                        encoded_circuit.append(
                            iceberg_detect_z(
                                0,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "iceberg_z_1_detect":
                        encoded_circuit.append(
                            iceberg_detect_z(
                                1,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "iceberg_z_2_detect":
                        encoded_circuit.append(
                            iceberg_detect_z(
                                2,
                                data_qubits,
                                ancilla_qubits[:2],
                                iceberg_syndrome_bits,
                                iceberg_discard_bit,
                            )
                        )
                    case "x_dynamical_decoupling":
                        for q in data_qubits:
                            encoded_circuit.X(q)
                    case _:
                        assert False

            # end cycle adding based on custom gate choice
            case OpType.Rz:
                assert len(command.op.params) == 1
                phase: float = command.op.params[0]
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                data_qubits: List[Qubit] = get_data_qubits[command.qubits[0]]
                match rz_mode:
                    case RzMode.DIRECT:
                        encoded_circuit.append(RzDirect.get_circuit(phase, data_qubits))
                    case RzMode.BIN_FRAC_NON_FT:
                        encoded_circuit.append(
                            RzKNonFt(rz_options.max_bits).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                no_detected_error_bit,
                                True,
                            )
                        )
                    case RzMode.BIN_FRAC_MEAS_FT:
                        encoded_circuit.append(
                            RzKMeasFt(rz_options.max_bits).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                rz_syndrome_bits,
                                no_detected_error_bit,
                                True,
                            )
                        )
                    case RzMode.BIN_FRAC_PART_FT:
                        encoded_circuit.append(
                            RzKPartFt(
                                rz_options.max_rus, rz_options.max_bits
                            ).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                prep_qubits,
                                part_ft_syndrome_bits,
                                flag_bit,
                                condition_bit,
                                True,
                            )
                        )
                    case _:
                        assert False

            case OpType.T:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                phase: float = 0.25
                data_qubits: List[Qubit] = get_data_qubits[command.qubits[0]]
                match rz_mode:
                    case RzMode.DIRECT:
                        encoded_circuit.append(
                            RzDirect().get_circuit(phase, data_qubits)
                        )
                    case RzMode.BIN_FRAC_NON_FT:
                        encoded_circuit.append(
                            RzKNonFt(rz_options.max_bits).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                no_detected_error_bit,
                                True,
                            )
                        )
                    case RzMode.BIN_FRAC_MEAS_FT:
                        encoded_circuit.append(
                            RzKMeasFt(rz_options.max_bits).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                rz_syndrome_bits,
                                no_detected_error_bit,
                                True,
                            )
                        )
                    case RzMode.BIN_FRAC_PART_FT:
                        encoded_circuit.append(
                            RzKPartFt(
                                rz_options.max_rus, rz_options.max_bits
                            ).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                prep_qubits,
                                part_ft_syndrome_bits,
                                flag_bit,
                                condition_bit,
                                True,
                            )
                        )
                    case _:
                        assert False

            case OpType.Tdg:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                phase: float = -0.25
                data_qubits: List[Qubit] = get_data_qubits[command.qubits[0]]
                match rz_mode:
                    case RzMode.DIRECT:
                        encoded_circuit.append(
                            RzDirect().get_circuit(phase, data_qubits)
                        )

                    case RzMode.BIN_FRAC_NON_FT:
                        encoded_circuit.append(
                            RzKNonFt(rz_options.max_bits).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                no_detected_error_bit,
                                True,
                            )
                        )
                    case RzMode.BIN_FRAC_MEAS_FT:
                        encoded_circuit.append(
                            RzKMeasFt(rz_options.max_bits).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                rz_syndrome_bits,
                                no_detected_error_bit,
                                True,
                            )
                        )
                    case RzMode.BIN_FRAC_PART_FT:
                        encoded_circuit.append(
                            RzKPartFt(
                                rz_options.max_rus, rz_options.max_bits
                            ).get_circuit(
                                phase,
                                data_qubits,
                                ancilla_qubits,
                                rz_ancilla_bits,
                                prep_qubits,
                                part_ft_syndrome_bits,
                                flag_bit,
                                condition_bit,
                                True,
                            )
                        )
                    case _:
                        assert False

            case OpType.PauliExpBox:
                all_qubits: List[Qubit] = []
                for q in command.qubits:
                    assert q in get_data_qubits
                    all_qubits.append(get_data_qubits[q])
                encoded_circuit.append(
                    get_Pauli_exponential(
                        all_qubits, command.op.get_paulis(), command.op.get_phase()
                    )
                )

            case OpType.X:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                encoded_circuit.append(get_X(get_data_qubits[command.qubits[0]]))

            case OpType.Y:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                encoded_circuit.append(get_Y(get_data_qubits[command.qubits[0]]))

            case OpType.Z:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                encoded_circuit.append(get_Z(get_data_qubits[command.qubits[0]]))

            case OpType.H:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                encoded_circuit.append(get_H(get_data_qubits[command.qubits[0]]))

            case OpType.S:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                encoded_circuit.append(get_S(get_data_qubits[command.qubits[0]]))

            case OpType.Sdg:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                encoded_circuit.append(get_Sdg(get_data_qubits[command.qubits[0]]))

            case OpType.V:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                encoded_circuit.append(get_V(get_data_qubits[command.qubits[0]]))

            case OpType.Vdg:
                assert len(command.qubits) == 1
                assert command.qubits[0] in get_data_qubits
                encoded_circuit.append(get_Vdg(get_data_qubits[command.qubits[0]]))

            case OpType.CX:
                assert len(command.qubits) == 2
                assert command.qubits[0] in get_data_qubits
                assert command.qubits[1] in get_data_qubits
                encoded_circuit.append(
                    get_CX(
                        get_data_qubits[command.qubits[0]],
                        get_data_qubits[command.qubits[1]],
                    )
                )

            case OpType.Measure:
                assert len(command.qubits) == 1
                assert len(command.bits) == 1
                assert command.qubits[0] in get_data_qubits
                assert command.bits[0] in get_data_bits
                qbs: List[Qubit] = get_data_qubits[command.qubits[0]]
                encoded_circuit.add_barrier(qbs)
                encoded_circuit.append(
                    get_Measure(
                        qbs,
                        get_data_bits[command.bits[0]],
                    )
                )
            case _:
                assert False
    encoded_circuit.remove_blank_wires()
    return encoded_circuit


def encode(
    circ: Circuit,
    options: EncodeOptions = EncodeOptions(),
) -> Circuit:
    """
    A function to be used by the workflow driver.

    Args:
        circ:
            Input logical circuit.
        n_shots:
            The number of shots to be taken.
        options:
            Encode options.

    Returns:
        (Encoded circuit, number of shots, and encoding data)

    """

    return get_encoded_circuit(
        circ,
        rz_mode=options.rz_mode,
        rz_options=options.rz_options,
        ft_prep=options.ft_prep,
        n_rus_prep=options.n_rus_prep,
        ft_prep_synd=options.ft_prep_synd,
        n_rus_synd=options.n_rus_synd,
        ft_prep_gate=options.ft_prep_gate,
        n_rus_gate=options.n_rus_gate,
    )
