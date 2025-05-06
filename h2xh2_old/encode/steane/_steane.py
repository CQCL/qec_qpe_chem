"""Prototype impolementation of the Steane code.

Stabilizers.
-------
0123456
-------
XXXXIII
IXXIXXI
IIXXIXX
ZZZZIII
IZZIZZI
IIZZIZZ
-------

Logical operators.
Z_L =  XXIIXII
X_L =  ZZIIZII
Y_L = -YYIIYII
"""

import re
import itertools
from enum import Enum
from typing import (
    NamedTuple,
    Counter,
    cast,
)
from pytket.circuit import (
    Circuit,
    Qubit,
    Bit,
    OpType,
    PauliExpBox,
    CircBox,
    UnitID,
    Pauli,
    CXConfigType,
)
from pytket.passes import RemoveRedundancies
from pytket.utils.outcomearray import OutcomeArray
from pytket.backends.backendresult import BackendResult

N_ANCL_QUBITS = 9
N_ANCL_BITS = 13
LOGICAL_Z = [0, 1, 4]
LOGICAL_X = [0, 1, 4]
LOGICAL_Y = [0, 1, 4]
DISCARD = "xDISCARD"
CORRECT = "xCORRECT"


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


class ReadoutMode(Enum):
    """Readout interpretation mode.

    Raw:
        Interpret the raw measurement outcomes as (-1) ** sum(bits).
    Detect:
        Post-select the measurement outcomes that remain in the code space.
    Correct:
        Perform the error correction based on the lookup table.
    """

    Raw = 0
    Detect = 1
    Correct = 2


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


class RzOptionsBinFracSpamFT(NamedTuple):
    """Options for the RzMode.BIN_FRAC_SPAM_FT.

    Args:
        max_bits:
            Maximum number of bits (i.e., max_bits - 2 = max_rus)
        n_rus:
            Number of RUS if PFT implementation is used.
    """

    max_bits: int = 10
    n_rus: int = 1


class RzOptionsBinFracPartFT(NamedTuple):
    """Options for the RzMode.BIN_FRAC_PART_FT and RzMode.BIN_FRAC_PART_FT_GOTO.

    Args:
        max_bits:
            Maximum number of bits (i.e., max_bits - 2 = max_rus)
        n_rus:
            Number of RUS if PFT implementation is used.
    """

    max_bits: int = 10
    n_rus: int = 1


class EncodeOptions(NamedTuple):
    """Encdode options for the Steane code.

    Args:
        rz_mode:
            Rz operation mode. Default is the iceberg-style direct mode.
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
    """

    # Non-FT (Rz including T) operation.
    rz_mode: RzMode = RzMode.DIRECT
    rz_options: (
        RzOptionsBinFracNonFT
        | RzOptionsBinFracMeasFT
        | RzOptionsBinFracSpamFT
        | RzOptionsBinFracPartFT
        | None
    ) = None

    # Use Goto state preparation (applied to the state preparation of the data register)
    ft_prep: bool = False
    n_rus_prep: int = 1

    # Use Goto state preparation (applied to the ancilla register for syndrome measurements)
    ft_prep_synd: bool = False
    n_rus_synd: int = 1


def encode(
    circ: Circuit,
    options: EncodeOptions | None = None,
) -> Circuit:
    """A function to be used by the workflow driver.

    Args:
        circ:
            Input logical circuit.
        options:
            Encode options.

    Returns:
        Encoded circuit

    """
    ec = SteaneCode()
    if options is None:
        options = EncodeOptions()
    circ_enc = ec.get_encoded_circuit(
        circ,
        rz_mode=options.rz_mode,
        rz_options=options.rz_options,
        ft_prep=options.ft_prep,
        n_rus_prep=options.n_rus_prep,
        ft_prep_synd=options.ft_prep_synd,
        n_rus_synd=options.n_rus_synd,
    )
    RemoveRedundancies().apply(circ_enc)
    return circ_enc


class InterpretOptions(NamedTuple):
    """Options for the interpret function to be used by the workflow driver.

    Args:
        readout_mode:
            Specify the readout mode.
    """

    # Readout mode.
    readout_mode: ReadoutMode = ReadoutMode.Correct


def interpret(
    result: BackendResult,
    options: InterpretOptions | None = None,
) -> BackendResult:
    """An interpret function to be used by the workflow driver.

    Args:
        Result:
            Backend result in the physical space.
        options:
            Interpret options.

    Returns:
        Backend result in the logical space.
    """
    if options is None:
        options = InterpretOptions()
    rnm = RegNameMapping()
    n_qubits = len([r for r in result.get_qbitlist() if r.reg_name == rnm.data_qubits])
    n_bits = len([r for r in result.get_bitlist() if r.reg_name == rnm.data_bits])
    primitives = SteanePrimitives(
        n_qubits=n_qubits,
        n_bits=n_bits,
    )
    ec = SteaneCode(primitives=primitives)
    result_enc = ec.get_decoded_result(
        result,
        readout_mode=options.readout_mode,
    )
    return result_enc


class RegNameMapping(NamedTuple):
    """Register name mapping.

    Args:
        data_qubits:
            Data qubits to encode the logical qubits.
        data_bits:
            Data bits to encode the logical bits.
        ancl_qubits:
            Ancilla qubits to be used for Steane QEC, Iceberg QED, and gate teleportation.
        ancl_bits:
            Ancilla bits to be used with the ancl_qubits.
        synd_bits:
            Bit register for the syndrome measurements.
        condition:
            Bit register for the gate teleportation condition.
        discard:
            Bit register that serves as a flag for post-selection.
        correct:
            Bit register that indicates error correction happens.

    Note:
        Do not change the default values.
    """

    data_qubits: str = "q"
    data_bits: str = "c"
    ancl_qubits: str = "xq"
    ancl_bits: str = "xc"
    synd_bits: str = "xSYND"
    condition: str = "xCOND"
    discard: str = DISCARD
    correct: str = CORRECT


class AcceptedCircBoxes(Enum):
    STEANE_Z = "|SZ|"
    STEANE_X = "|SX|"
    STELEPO_X = "|TX|"
    ICEBERG_W0 = "|IW0|"
    ICEBERG_W1 = "|IW1|"
    ICEBERG_W2 = "|IW2|"
    ICEBERG_X0 = "|IX0|"
    ICEBERG_X1 = "|IX1|"
    ICEBERG_X2 = "|IX2|"
    ICEBERG_Z0 = "|IZ0|"
    ICEBERG_Z1 = "|IZ1|"
    ICEBERG_Z2 = "|IZ2|"
    X_TRANSV = "|XT|"
    X_DD = "|XD|"


def pauli_exp_zzphase(
    paulis: list[Pauli],
    phase: float,
) -> Circuit:
    """Pauli gadget using ZZPhase gates instead of Rz.

    Args:
        paulis:
            List of Pauli operators.
        phase:
            phase factor.

    Returns:
        Circuit that is equivalent to :code:`PauliExpBox(paulis, phase)`.
    """
    circ = Circuit(len(paulis))
    basis_change = Circuit(len(paulis))
    # Basis change.
    for i, p in enumerate(paulis):
        if p == Pauli.X:
            basis_change.H(i)
        elif p == Pauli.Y:
            basis_change.V(i)
    # CX cascade.
    ls_cx = [i for i, p in enumerate(paulis) if p != Pauli.I][::-1]
    for i, k in enumerate(ls_cx[:-2]):
        basis_change.CX(k, ls_cx[i + 1])
    circ.append(basis_change)
    # Add the rotation gate.
    if len(ls_cx) == 1:
        circ.Rz(phase, ls_cx[0])
    else:
        circ.ZZPhase(phase, ls_cx[-1], ls_cx[-2])
    # CX cascade + Basis change.
    circ.append(basis_change.dagger())
    return circ


def decoder(
    syndrome: tuple[int, int, int],
) -> int | None:
    """Steane decoder based on the lookup table.

    Args:
        syndrome:
            Syndrome measurement outcomes.

    Returns:
        Identified error position if the syndrome is non-trivial.
    """
    assert len(syndrome) == 3
    lookup_table = {
        (0, 0, 0): None,
        (1, 0, 0): 0,
        (1, 1, 0): 1,
        (1, 1, 1): 2,
        (1, 0, 1): 3,
        (0, 1, 0): 4,
        (0, 1, 1): 5,
        (0, 0, 1): 6,
    }
    return lookup_table[syndrome]


def ro_syndrome(
    readout: list[int],
) -> tuple[int, int, int]:
    """Syndrome from the readout.

    Args:
        readout:
            Measurement outcome (physical).

    Returns:
        Syndrome measurement outcomes.
    """
    assert len(readout) == 7
    s0 = (-1) ** sum(readout[i] for i in (0, 1, 2, 3))
    s1 = (-1) ** sum(readout[i] for i in (1, 2, 4, 5))
    s2 = (-1) ** sum(readout[i] for i in (2, 3, 5, 6))
    s0 = (1 - s0) // 2
    s1 = (1 - s1) // 2
    s2 = (1 - s2) // 2
    return (s0, s1, s2)


def ro_correction(
    readout: list[int],
) -> list[int]:
    """Readout error correction.

    Args:
        readout:
            Measurement outcome (physical).
    Returns:
        Error corrected measurement outcome.
    """
    readout_ = list(readout[:])
    syndrome = ro_syndrome(readout)
    flip = decoder(syndrome)
    if flip is not None:
        readout_[flip] = (readout_[flip] + 1) % 2
    return readout_


def l2p(i: int | Qubit | Bit) -> list[int]:
    """Index convertor for the qubit register.

    Args:
        i:
            Logical qubit index.

    Returns:
        List of indices of the corresponding physical qubits/bits.
    """
    i_ = i
    if isinstance(i, (Qubit, Bit)):
        i_ = i.index[0]
    r = slice(7 * i_, 7 * (i_ + 1))
    return r


def add_x_transv(circ: Circuit) -> None:
    """Indicator to add full transversal X gates for the dynamical decoupling."""
    c = Circuit(circ.n_qubits, name=AcceptedCircBoxes.X_TRANSV.value)
    c.add_barrier(circ.qubits)
    for q in c.qubits:
        c.X(q)
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], circ.qubits))


def add_x_dd(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add a transversal X gate for the dynamical decoupling."""
    c = Circuit(1, name=AcceptedCircBoxes.X_DD.value)
    c.X(cast(UnitID, 0))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_steane_z(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Steane error correction gadget for the Z syndromes."""
    c = Circuit(1, name=AcceptedCircBoxes.STEANE_Z.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_stelepo_x(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the State teleportation error correction gadget for the X syndromes."""
    c = Circuit(1, name=AcceptedCircBoxes.STELEPO_X.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_steane_x(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Steane error correction gadget for the X syndromes."""
    c = Circuit(1, name=AcceptedCircBoxes.STEANE_X.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_w0(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for ZX (0123)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_W0.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_w1(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for ZX (1245)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_W1.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_w2(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for ZX (2356)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_W2.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_x0(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for X (0123)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_X0.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_x1(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for X (1245)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_X1.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_x2(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for X (2356)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_X2.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_z0(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for Z (0123)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_Z0.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_z1(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for Z (1245)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_Z1.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def add_iceberg_z2(circ: Circuit, qubit: Qubit) -> None:
    """Indicator to add the Iceberg error detection gadget for Z (2356)."""
    c = Circuit(1, name=AcceptedCircBoxes.ICEBERG_Z2.value)
    c.add_barrier(cast(list[UnitID], [0]))
    box = CircBox(circ=c)
    circ.add_circbox(box, cast(list[UnitID], [qubit]))


def resolve_phase(phase: float, max_bits: int = 10) -> list[int]:
    """Indicator to add the Iceberg error detection gadget for Z (2356)."""
    phase_ = phase % 2.0
    bits = []
    atol = 2 ** (-max_bits)
    phase_ += atol
    if phase_ > 2.0:
        return bits
    for i in range(max_bits):
        val = 2**-i
        if phase_ >= val:
            bits.append(1)
            phase_ -= val
        else:
            bits.append(0)
        if phase_ < atol:
            break
    return bits


class SteanePrimitives(object):
    """Steane code primitives."""

    def __init__(
        self,
        n_qubits: int,
        n_bits: int,
        reg_name_mapping: RegNameMapping | None = None,
    ):
        self._n_qubits = n_qubits
        self._n_bits = n_bits
        if reg_name_mapping is None:
            reg_name_mapping = RegNameMapping()
        self._reg_name_mapping = RegNameMapping()
        self._init_register()

    def _init_register(self) -> Circuit:
        """Prepare the empty circuit for the Steane code."""
        # Initialize the data qubit/bit register.
        rnm = self._reg_name_mapping
        circ = Circuit()
        _data_qubits: list[Qubit] = []
        _data_bits: list[Bit] = []
        ii = 0
        for i in range(self._n_qubits):
            for _ in range(7):
                qubit = Qubit(rnm.data_qubits, ii)
                _data_qubits.append(qubit)
                bit = Bit(rnm.data_bits, ii)
                _data_bits.append(bit)
                circ.add_qubit(qubit)
                circ.add_bit(bit)
                ii += 1
        # Initialize the ancilla qubit register.
        _ancl_qubits = [Qubit(rnm.ancl_qubits, i) for i in range(N_ANCL_QUBITS)]
        for q in _ancl_qubits:
            circ.add_qubit(q)
        # Initialize the ancilla bits for the Steane QEC.
        _ancl_bits = [Bit(rnm.ancl_bits, i) for i in range(N_ANCL_BITS)]
        for b in _ancl_bits:
            circ.add_bit(b)
        # Initialize the general bit for post-selection.
        _discart_bit = Bit(rnm.discard, 0)
        circ.add_bit(_discart_bit)
        # Initialize the condition bit.
        _condition_bit = Bit(rnm.condition, 0)
        circ.add_bit(_condition_bit)
        # Initialize the correction bit.
        _correct_bit = Bit(rnm.correct, 0)
        circ.add_bit(_correct_bit)
        # Save the qubits.
        self._data_qubits = _data_qubits
        self._ancl_qubits = _ancl_qubits
        # Save the bits.
        self._data_bits = _data_bits
        self._ancl_bits = _ancl_bits
        self._discard_bit = _discart_bit
        self._correct_bit = _correct_bit
        self._condition_bit = _condition_bit
        # Error detection / correction index.
        self._synd_index = 0
        # Default circuit register.
        self._circ0 = circ

    @property
    def circ0(self) -> Circuit:
        return self._circ0

    @property
    def data_qubits(self) -> list[Qubit]:
        return self._data_qubits

    @property
    def data_bits(self) -> list[Bit]:
        return self._data_bits

    @property
    def ancila_qubits(self) -> list[Qubit]:
        return self._ancl_qubits

    @property
    def ancila_bits(self) -> list[Bit]:
        return self._ancl_bits

    @property
    def discard_bit(self) -> Bit:
        return self._discard_bit

    def prep_non_ft(self) -> Circuit:
        """Non fault-tolerant state preparation."""
        circ = self.circ0.copy()
        for i in range(self._n_qubits):
            qubits = self._data_qubits[l2p(i)]
            circ.append(get_prep_non_ft(qubits))
        circ.remove_blank_wires()
        return circ

    def prep(self, n_rus: int = 1) -> Circuit:
        """Non fault-tolerant state preparation."""
        circ = self.circ0.copy()
        goto_qubit = self._ancl_qubits[0]
        goto_bit = self._ancl_bits[0]
        discard = self._discard_bit
        for i in range(self._n_qubits):
            qubits = self._data_qubits[l2p(i)]
            if i > 0:
                circ.add_barrier(self._data_qubits)
            circ.append(get_prep_ft(qubits, goto_qubit, goto_bit))
            for _ in range(n_rus - 1):
                circ.append(
                    get_prep_ft(qubits, goto_qubit, goto_bit, condition=goto_bit)
                )
            circ.add_classicalexpbox_bit(
                goto_bit | discard,
                [discard],
            )
        circ.remove_blank_wires()
        return circ

    def add_barrier(self, qubits: list[int]) -> Circuit:
        circ = self.circ0.copy()
        data_qubits = []
        for q in qubits:
            data_qubits += self._data_qubits[l2p(q)]
        circ.add_barrier(data_qubits)
        return circ

    def Measure(self, qubit: int | Qubit, bit: int | Bit) -> Circuit:
        if isinstance(qubit, int):
            iq = qubit
            ib = bit
        else:
            iq = qubit.index[0]
            ib = bit.index[0]
        circ = self.circ0.copy()
        qls = self._data_qubits[l2p(iq)]
        cls = self._data_bits[l2p(ib)]
        circ.add_barrier(qls)
        for q, b in zip(qls, cls):
            circ.Measure(q, b)
        circ.remove_blank_wires()
        return circ

    def Reset(self, qubit: int) -> Circuit:
        circ = self.circ0.copy()
        qubits = self._data_qubits[l2p(qubit)]
        for q in qubits:
            circ.Reset(q)
        return circ

    def X(self, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_X(qubits)
        return circ

    def Y(self, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_Y(qubits)
        return circ

    def Z(self, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_Z(qubits)
        return circ

    def S(self, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_S(qubits)
        return circ

    def Sdg(self, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_Sdg(qubits)
        return circ

    def V(self, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_V(qubits)
        return circ

    def Vdg(self, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_Vdg(qubits)
        return circ

    def H(self, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_H(qubits)
        return circ

    def CX(self, ctrl: int, targ: int) -> Circuit:
        ctrl_qubits = self._data_qubits[l2p(ctrl)]
        targ_qubits = self._data_qubits[l2p(targ)]
        circ = get_CX(ctrl_qubits, targ_qubits)
        return circ

    def PauliExpBox(
        self,
        paulis: list[Pauli],
        phase: float,
        qubits: list[int],
        use_rzz: bool = True,
    ) -> Circuit:
        paulis_p: list[Pauli] = []
        qubits_p: list[Qubit] = []
        phase_p = phase
        for pauli, qubit in zip(paulis, qubits):
            if pauli == Pauli.Y:
                phase_p *= -1
            paulis_p += [pauli] * 3
            dataq = self._data_qubits[l2p(qubit)]
            if pauli == Pauli.X:
                qubits_p += [dataq[i] for i in LOGICAL_X]
            elif pauli == Pauli.Y:
                qubits_p += [dataq[i] for i in LOGICAL_Y]
            elif pauli == Pauli.Z:
                qubits_p += [dataq[i] for i in LOGICAL_Z]
        circ = self.circ0.copy()
        if use_rzz:
            circ.add_circuit(
                pauli_exp_zzphase(paulis_p, phase_p),
                qubits_p,
            )
        else:
            circ.add_pauliexpbox(
                PauliExpBox(
                    paulis_p,
                    phase_p,
                    cx_config_type=CXConfigType.Snake,
                ),
                qubits_p,
            )
        return circ

    def Rz_direct(self, phase: float, qubit: int) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_Rz_direct(phase, qubits)
        return circ

    def Rz_k_non_ft(
        self,
        phase: float,
        qubit: int,
        rz_options: RzOptionsBinFracNonFT | None = None,
    ) -> Circuit:
        if rz_options is None:
            rz_options = RzOptionsBinFracNonFT()
        assert isinstance(rz_options, RzOptionsBinFracNonFT)
        if phase == 0.0:
            return Circuit()
        angle_bits = resolve_phase(phase, max_bits=rz_options.max_bits)
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:7]
        ancl_bits = self._ancl_bits[:3]
        # flag_bit = self._ancl_bits[3]
        cond_bit = self._condition_bit
        circ = get_rzk_non_ft(angle_bits, data_qubits, ancl_qubits, ancl_bits, cond_bit)
        return circ

    def Rz_k_meas_ft(
        self,
        phase: float,
        qubit: int,
        rz_options: RzOptionsBinFracMeasFT | None = None,
    ) -> Circuit:
        if rz_options is None:
            rz_options = RzOptionsBinFracMeasFT()
        assert isinstance(rz_options, RzOptionsBinFracMeasFT)
        if phase == 0.0:
            return Circuit()
        angle_bits = resolve_phase(phase, max_bits=rz_options.max_bits)
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:7]
        ancl_bits = self._ancl_bits[:11]
        cond_bit = self._condition_bit
        circ = get_rzk_meas_ft(
            angle_bits, data_qubits, ancl_qubits, ancl_bits, cond_bit
        )
        return circ

    def Rz_k_part_ft(
        self,
        phase: float,
        qubit: int,
        rz_options: RzOptionsBinFracPartFT | None = None,
    ) -> Circuit:
        if rz_options is None:
            rz_options = RzOptionsBinFracPartFT()
        assert isinstance(rz_options, RzOptionsBinFracPartFT)
        if phase == 0.0:
            return Circuit()
        angle_bits = resolve_phase(phase, max_bits=rz_options.max_bits)
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:9]
        ancl_bits = self._ancl_bits[:13]
        # condition = self._ancl_bits[10]
        condition = self._condition_bit
        discard_bit = self._discard_bit
        circ = get_rzk_part_ft(
            angle_bits,
            data_qubits,
            ancl_qubits,
            ancl_bits,
            discard_bit,
            condition,
            rz_options.n_rus,
        )
        return circ

    def Rz_k_part_ft_goto(
        self,
        phase: float,
        qubit: int,
        rz_options: RzOptionsBinFracPartFT | None = None,
    ) -> Circuit:
        if rz_options is None:
            rz_options = RzOptionsBinFracPartFT()
        assert isinstance(rz_options, RzOptionsBinFracPartFT)
        if phase == 0.0:
            return Circuit()
        angle_bits = resolve_phase(phase, max_bits=rz_options.max_bits)
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:9]
        ancl_bits = self._ancl_bits[:13]
        # condition = self._ancl_bits[10]
        condition = self._condition_bit
        discard_bit = self._discard_bit
        circ = get_rzk_part_ft_goto(
            angle_bits,
            data_qubits,
            ancl_qubits,
            ancl_bits,
            discard_bit,
            condition,
            rz_options.n_rus,
        )
        return circ

    def _Rz_k_ftprep(self, phase: float, qubit: int, n_rus: int) -> Circuit:
        raise NotImplementedError()
        # if phase == 0.0:
        #     return Circuit()
        # data_qubits = self._data_qubits[l2p(qubit)]
        # ancl_qubits = self._ancl_qubits[:7]
        # goto_qubit = self._ancl_qubits[8]
        # ancl_bits = self._ancl_bits[:3]
        # goto_bit = self._ancl_bits[3]
        # flag_bit = self._ancl_bits[4]
        # disc_bit = self._discard_bit
        # kval, sign = phase_to_binary_frac(phase)
        # circ = get_rzk_ftprep(
        #     kval,
        #     sign,
        #     data_qubits,
        #     ancl_qubits,
        #     ancl_bits,
        #     flag_bit,
        #     goto_qubit,
        #     goto_bit,
        #     n_rus,
        #     discard_bit=disc_bit,
        # )
        return circ

    def X_transversal(
        self,
        qubit: int,
    ) -> Circuit:
        qubits = self._data_qubits[l2p(qubit)]
        circ = get_X_transveral(qubits)
        return circ

    def X_full_transversal(
        self,
    ) -> Circuit:
        circ = self._circ0.copy()
        for q in range(self._n_qubits):
            qubits = self._data_qubits[l2p(q)]
            circ.append(get_X_transveral(qubits))
        return circ

    def steane_correct_x_non_ft(
        self,
        qubit: int,
    ) -> Circuit:
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:7]
        ancl_bits = self._ancl_bits[:7]
        # reg_name = self._reg_name_mapping.synd_bits + f"x{self._synd_index:03d}"
        reg_name = self._reg_name_mapping.synd_bits
        synd_bits = [Bit(reg_name, i) for i in range(3)]
        # self._synd_index += 1
        circ = get_steane_correct_x_non_ft(
            data_qubits,
            ancl_qubits,
            ancl_bits,
            synd_bits,
        )
        circ.add_bit(self._correct_bit)
        circ.add_classicalexpbox_bit(
            synd_bits[0] | synd_bits[1] | synd_bits[2] | self._correct_bit,
            [self._correct_bit],
        )
        return circ

    def steane_correct_z_non_ft(
        self,
        qubit: int,
    ) -> Circuit:
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:7]
        ancl_bits = self._ancl_bits[:7]
        # reg_name = self._reg_name_mapping.synd_bits + f"z{self._synd_index:03d}"
        reg_name = self._reg_name_mapping.synd_bits
        synd_bits = [Bit(reg_name, i) for i in range(3)]
        # self._synd_index += 1
        circ = get_steane_correct_z_non_ft(
            data_qubits,
            ancl_qubits,
            ancl_bits,
            synd_bits,
        )
        circ.add_bit(self._correct_bit)
        circ.add_classicalexpbox_bit(
            synd_bits[0] | synd_bits[1] | synd_bits[2] | self._correct_bit,
            [self._correct_bit],
        )
        return circ

    def steane_correct_x(
        self,
        qubit: int,
        n_rus: int = 1,
    ) -> Circuit:
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:7]
        ancl_bits = self._ancl_bits[:7]
        # reg_name = self._reg_name_mapping.synd_bits + f"x{self._synd_index:03d}"
        reg_name = self._reg_name_mapping.synd_bits
        synd_bits = [Bit(reg_name, i) for i in range(3)]
        # self._synd_index += 1
        goto_qubit = self._ancl_qubits[7]
        goto_bit = self._ancl_bits[7]
        circ = get_steane_correct_x(
            data_qubits,
            ancl_qubits,
            ancl_bits,
            synd_bits,
            goto_qubit,
            goto_bit,
            n_rus=n_rus,
        )
        circ.add_bit(self._discard_bit)
        circ.add_classicalexpbox_bit(
            goto_bit | self._discard_bit,
            [self._discard_bit],
        )
        circ.add_bit(self._correct_bit)
        circ.add_classicalexpbox_bit(
            synd_bits[0] | synd_bits[1] | synd_bits[2] | self._correct_bit,
            [self._correct_bit],
        )
        return circ

    def steane_correct_z(
        self,
        qubit: int,
        n_rus: int = 1,
    ) -> Circuit:
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:7]
        ancl_bits = self._ancl_bits[:7]
        # reg_name = self._reg_name_mapping.synd_bits + f"x{self._synd_index:03d}"
        reg_name = self._reg_name_mapping.synd_bits
        synd_bits = [Bit(reg_name, i) for i in range(3)]
        # self._synd_index += 1
        goto_qubit = self._ancl_qubits[7]
        goto_bit = self._ancl_bits[7]
        circ = get_steane_correct_z(
            data_qubits,
            ancl_qubits,
            ancl_bits,
            synd_bits,
            goto_qubit,
            goto_bit,
            n_rus=n_rus,
        )
        circ.add_bit(self._discard_bit)
        circ.add_classicalexpbox_bit(
            goto_bit | self._discard_bit,
            [self._discard_bit],
        )
        circ.add_bit(self._correct_bit)
        circ.add_classicalexpbox_bit(
            synd_bits[0] | synd_bits[1] | synd_bits[2] | self._correct_bit,
            [self._correct_bit],
        )
        return circ

    def stelepo_correct_x_nonft(
        self,
        qubit: int,
    ) -> Circuit:
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:7]
        ancl_bits = self._ancl_bits[:11]
        circ = get_stelepo_correct_x_non_ft(
            data_qubits,
            ancl_qubits,
            ancl_bits,
        )
        return circ

    def iceberg_detect_zx(
        self,
        index: int,
        qubit: int,
    ) -> Circuit:
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:2]
        ancl_bits = self._ancl_bits[:2]
        circ = get_iceberg_zx(
            index,
            data_qubits,
            ancl_qubits,
            ancl_bits,
        )
        circ.add_bit(self._discard_bit)
        circ.add_classicalexpbox_bit(
            ancl_bits[0] | ancl_bits[1] | self._discard_bit,
            [self._discard_bit],
        )
        return circ

    def iceberg_detect_z(
        self,
        index: int,
        qubit: int,
    ) -> Circuit:
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:2]
        ancl_bits = self._ancl_bits[:2]
        circ = get_iceberg_z(
            index,
            data_qubits,
            ancl_qubits,
            ancl_bits,
        )
        circ.add_bit(self._discard_bit)
        circ.add_classicalexpbox_bit(
            ancl_bits[0] | ancl_bits[1] | self._discard_bit,
            [self._discard_bit],
        )
        return circ

    def iceberg_detect_x(
        self,
        index: int,
        qubit: int,
    ) -> Circuit:
        data_qubits = self._data_qubits[l2p(qubit)]
        ancl_qubits = self._ancl_qubits[:2]
        ancl_bits = self._ancl_bits[:2]
        circ = get_iceberg_x(
            index,
            data_qubits,
            ancl_qubits,
            ancl_bits,
        )
        circ.add_bit(self._discard_bit)
        circ.add_classicalexpbox_bit(
            ancl_bits[0] | ancl_bits[1] | self._discard_bit,
            [self._discard_bit],
        )
        return circ


class SteaneCode(object):
    def __init__(
        self,
        primitives: SteanePrimitives | None = None,
    ):
        self._primitives = primitives

    def get_encoded_circuit(
        self,
        circ: Circuit,
        rz_mode: RzMode = RzMode.DIRECT,
        rz_options: (
            RzOptionsBinFracNonFT
            | RzOptionsBinFracMeasFT
            | RzOptionsBinFracSpamFT
            | RzOptionsBinFracPartFT
            | None
        ) = None,
        ft_prep: bool = False,
        n_rus_prep: int = 1,
        ft_prep_synd: bool = False,
        n_rus_synd: int = 1,
    ) -> Circuit:
        if self._primitives is None:
            self._primitives = SteanePrimitives(
                n_qubits=circ.n_qubits,
                n_bits=circ.n_bits,
            )
        if ft_prep:
            circe = self._primitives.prep(n_rus=n_rus_prep)
        else:
            circe = self._primitives.prep_non_ft()
        for cmd in circ:
            optype = cmd.op.type
            optype_s = str(optype).split(".")[1]
            # Add barrier.
            if optype == OpType.Barrier:
                circe.append(self._primitives.add_barrier(cmd.args))
            # Operations indicated by the special CircBox.
            elif optype == OpType.CircBox:
                tmp_circ = cmd.op.get_circuit()
                match AcceptedCircBoxes(tmp_circ.name):
                    case AcceptedCircBoxes.STEANE_X:
                        if ft_prep_synd:
                            circe.append(
                                self._primitives.steane_correct_x(
                                    cmd.qubits[0],
                                    n_rus=n_rus_synd,
                                ),
                            )
                        else:
                            circe.append(
                                self._primitives.steane_correct_x_non_ft(
                                    cmd.qubits[0],
                                ),
                            )
                    case AcceptedCircBoxes.STEANE_Z:
                        if ft_prep_synd:
                            circe.append(
                                self._primitives.steane_correct_z(
                                    cmd.qubits[0],
                                    n_rus=n_rus_synd,
                                ),
                            )
                        else:
                            circe.append(
                                self._primitives.steane_correct_z_non_ft(
                                    cmd.qubits[0],
                                ),
                            )
                    case AcceptedCircBoxes.STELEPO_X:
                        circe.append(
                            self._primitives.stelepo_correct_x_nonft(
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_W0:
                        circe.append(
                            self._primitives.iceberg_detect_zx(
                                0,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_W1:
                        circe.append(
                            self._primitives.iceberg_detect_zx(
                                1,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_W2:
                        circe.append(
                            self._primitives.iceberg_detect_zx(
                                2,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_X0:
                        circe.append(
                            self._primitives.iceberg_detect_x(
                                0,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_X1:
                        circe.append(
                            self._primitives.iceberg_detect_x(
                                1,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_X2:
                        circe.append(
                            self._primitives.iceberg_detect_x(
                                2,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_Z0:
                        circe.append(
                            self._primitives.iceberg_detect_z(
                                0,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_Z1:
                        circe.append(
                            self._primitives.iceberg_detect_z(
                                1,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.ICEBERG_Z2:
                        circe.append(
                            self._primitives.iceberg_detect_z(
                                2,
                                cmd.qubits[0],
                            ),
                        )
                    case AcceptedCircBoxes.X_TRANSV:
                        circe.add_barrier(self._primitives._data_qubits)
                        circe.append(self._primitives.X_full_transversal())
                        circe.add_barrier(self._primitives._data_qubits)
                    case AcceptedCircBoxes.X_DD:
                        circe.append(self._primitives.X_transversal(cmd.qubits[0]))
                    case _:
                        raise NotImplementedError(
                            f"CircBox {tmp_circ.name} is not supported"
                        )
            # Add encoded PauliExpBox (Direct implementation only):
            elif cmd.op.type == OpType.PauliExpBox:
                circe.append(
                    self._primitives.PauliExpBox(
                        cmd.op.get_paulis(),
                        cmd.op.get_phase(),
                        cmd.args,
                        use_rzz=True,
                    )
                )
            # Add Rz gate.
            elif optype in (OpType.Rz, OpType.T, OpType.Tdg):
                if optype == OpType.Rz:
                    args = cmd.op.params + cmd.args
                elif optype == OpType.T:
                    args = [0.25] + cmd.args
                elif optype == OpType.Tdg:
                    args = [-0.25] + cmd.args
                else:
                    raise RuntimeError()
                match rz_mode:
                    case RzMode.DIRECT:
                        circe.append(self._primitives.Rz_direct(*args))
                    case RzMode.BIN_FRAC_NON_FT:
                        circe.append(
                            self._primitives.Rz_k_non_ft(*args, rz_options=rz_options)
                        )
                    case RzMode.BIN_FRAC_MEAS_FT:
                        circe.append(
                            self._primitives.Rz_k_meas_ft(*args, rz_options=rz_options)
                        )
                    case RzMode.BIN_FRAC_PART_FT:
                        circe.append(
                            self._primitives.Rz_k_part_ft(*args, rz_options=rz_options)
                        )
                    case RzMode.BIN_FRAC_PART_FT_GOTO:
                        circe.append(
                            self._primitives.Rz_k_part_ft_goto(
                                *args, rz_options=rz_options
                            )
                        )
                    case _:
                        raise KeyError()
            elif hasattr(self._primitives, optype_s):
                func = getattr(self._primitives, optype_s)
                args = cmd.op.params + cmd.args
                circe.append(func(*args))
            else:
                raise RuntimeError(f"{optype} not recognized")
        RemoveRedundancies().apply(circe)
        circe.remove_blank_wires()
        return circe

    def get_discard_rate(
        self,
        result: BackendResult,
    ) -> float:
        total_n_shots = sum(result.get_counts().values())
        counts = self.get_decoded_result(result).get_counts()
        n_shots = sum(counts.values())
        rate = 1 - n_shots / total_n_shots
        return rate

    def get_decoded_result(
        self,
        result: BackendResult,
        readout_mode: ReadoutMode = ReadoutMode.Raw,
    ) -> BackendResult:
        assert readout_mode.value in [0, 1, 2]
        rng = self._primitives._reg_name_mapping
        bitlist = result.get_bitlist()
        # Chose the data bit register.
        cbits = [b for b in bitlist if b.reg_name == rng.data_bits]
        l_data = len(cbits)
        n_logical_qubits = l_data // 7
        # Error detection bits.
        cbits += [b for b in bitlist if re.match(DISCARD, b.reg_name)]
        # Interpret the physical results.
        counts = result.get_counts(cbits=cbits)
        logical_counts = Counter()
        for readout0, val in counts.items():
            # Post selection by the error detection.
            if sum(readout0[l_data:]) > 0:
                continue
            # Use the readout as it is.
            if readout_mode == ReadoutMode.Raw:
                readout = readout0[:l_data]
            # Readout error detection.
            elif readout_mode == ReadoutMode.Detect:
                error_detected = False
                for il in range(n_logical_qubits):
                    syndrome = ro_syndrome(readout0[il * 7 : il * 7 + 7])
                    if sum(syndrome) > 0:
                        error_detected = True
                        break
                if error_detected:
                    continue
                else:
                    readout = readout0[:l_data]
            # Readout error correction.
            elif readout_mode == ReadoutMode.Correct:
                readout: list[int] = []
                for il in range(n_logical_qubits):
                    readout += list(ro_correction(readout0[il * 7 : il * 7 + 7]))
            else:
                raise RuntimeError()
            lreadout: list[int] = []
            for i in range(len(cbits[:l_data]) // 7):
                parity = (-1) ** sum(readout[l2p(i)])
                lreadout.append((1 - parity) // 2)
            logical_readout = tuple(lreadout)
            # print(logical_readout)
            logical_counts[OutcomeArray.from_readouts([logical_readout])] += int(val)
        logical_result = BackendResult(counts=logical_counts)
        return logical_result


# ======================================================================
# Basic operations
# ======================================================================


def get_Rz_direct(
    phase: float,
    qubits: list,
    condition: Bit | None = None,
) -> Circuit:
    assert len(qubits) == 7
    circ = Circuit()
    for q in qubits:
        circ.add_qubit(q)
    if condition is None:
        circ.CX(qubits[LOGICAL_Z[2]], qubits[LOGICAL_Z[1]])
        circ.ZZPhase(phase, qubits[LOGICAL_Z[1]], qubits[LOGICAL_Z[0]])
        circ.CX(qubits[LOGICAL_Z[2]], qubits[LOGICAL_Z[1]])
    else:
        circ.add_bit(condition)
        circ.CX(qubits[LOGICAL_Z[2]], qubits[LOGICAL_Z[1]], condition=condition)
        circ.ZZPhase(
            phase, qubits[LOGICAL_Z[1]], qubits[LOGICAL_Z[0]], condition=condition
        )
        circ.CX(qubits[LOGICAL_Z[2]], qubits[LOGICAL_Z[1]], condition=condition)
    return circ


def get_iceberg_zx(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        return _get_iceberg_zx(index, data_qubits, ancl_qubits, ancl_bits)
    else:
        return _get_iceberg_zx_cond(
            index, data_qubits, ancl_qubits, ancl_bits, condition
        )


def get_iceberg_z(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        return _get_iceberg_z(index, data_qubits, ancl_qubits, ancl_bits)
    else:
        return _get_iceberg_z_cond(
            index, data_qubits, ancl_qubits, ancl_bits, condition
        )


def get_iceberg_x(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        return _get_iceberg_x(index, data_qubits, ancl_qubits, ancl_bits)
    else:
        return _get_iceberg_x_cond(
            index, data_qubits, ancl_qubits, ancl_bits, condition
        )


def _get_iceberg_zx(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 2
    assert len(ancl_bits) == 2
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    ls = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    qubits = [data_qubits[i] for i in ls[index]]
    for q in ancl_qubits:
        circ.Reset(q)
    circ.H(ancl_qubits[1])
    circ.CX(ancl_qubits[1], qubits[0])
    circ.add_barrier(ancl_qubits + data_qubits)
    circ.CX(qubits[0], ancl_qubits[0])
    circ.add_barrier(ancl_qubits + data_qubits)
    circ.CX(qubits[1], ancl_qubits[0])
    circ.add_barrier(ancl_qubits + data_qubits)
    circ.CX(ancl_qubits[1], qubits[1])
    circ.add_barrier(ancl_qubits + data_qubits)
    circ.CX(ancl_qubits[1], qubits[2])
    circ.add_barrier(ancl_qubits + data_qubits)
    circ.CX(qubits[2], ancl_qubits[0])
    circ.add_barrier(ancl_qubits + data_qubits)
    circ.CX(qubits[3], ancl_qubits[0])
    circ.add_barrier(ancl_qubits + data_qubits)
    circ.CX(ancl_qubits[1], qubits[3])
    circ.H(ancl_qubits[1])
    circ.Measure(ancl_qubits[0], ancl_bits[0])
    circ.Measure(ancl_qubits[1], ancl_bits[1])
    return circ


def _get_iceberg_z(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 2
    assert len(ancl_bits) == 2
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    for q in ancl_qubits:
        circ.Reset(q)
    ls = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    qubits = [data_qubits[i] for i in ls[index]]
    circ.H(ancl_qubits[1])
    circ.CX(qubits[0], ancl_qubits[0])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[1], ancl_qubits[0])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[1], ancl_qubits[0])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[2], ancl_qubits[0])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[1], ancl_qubits[0])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[3], ancl_qubits[0])
    circ.H(ancl_qubits[1])
    circ.Measure(ancl_qubits[0], ancl_bits[0])
    circ.Measure(ancl_qubits[1], ancl_bits[1])
    return circ


def _get_iceberg_z_cond(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 2
    assert len(ancl_bits) == 2
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits + [condition]:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    for q in ancl_qubits:
        circ.Reset(q, condition=condition)
    ls = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    qubits = [data_qubits[i] for i in ls[index]]
    circ.H(ancl_qubits[1], condition=condition)
    circ.CX(qubits[0], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[1], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[1], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[2], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[1], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[3], ancl_qubits[0], condition=condition)
    circ.H(ancl_qubits[1], condition=condition)
    circ.Measure(ancl_qubits[0], ancl_bits[0], condition=condition)
    circ.Measure(ancl_qubits[1], ancl_bits[1], condition=condition)
    return circ


def _get_iceberg_x(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 2
    assert len(ancl_bits) == 2
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    for q in ancl_qubits:
        circ.Reset(q)
    ls = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    qubits = [data_qubits[i] for i in ls[index]]
    circ.H(ancl_qubits[0])
    circ.CX(ancl_qubits[0], qubits[0])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], ancl_qubits[1])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], qubits[1])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], qubits[2])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], ancl_qubits[1])
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], qubits[3])
    circ.H(ancl_qubits[0])
    circ.Measure(ancl_qubits[0], ancl_bits[0])
    circ.Measure(ancl_qubits[1], ancl_bits[1])
    return circ


def _get_iceberg_x_cond(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 2
    assert len(ancl_bits) == 2
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits + [condition]:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    for q in ancl_qubits:
        circ.Reset(q, condition=condition)
    ls = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    qubits = [data_qubits[i] for i in ls[index]]
    circ.H(ancl_qubits[0], condition=condition)
    circ.CX(ancl_qubits[0], qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], ancl_qubits[1], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], qubits[1], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], qubits[2], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], ancl_qubits[1], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[0], qubits[3], condition=condition)
    circ.H(ancl_qubits[0], condition=condition)
    circ.Measure(ancl_qubits[0], ancl_bits[0], condition=condition)
    circ.Measure(ancl_qubits[1], ancl_bits[1], condition=condition)
    return circ


def _get_iceberg_zx_cond(
    index: int,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 2
    assert len(ancl_bits) == 2
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits + [condition]:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    for q in ancl_qubits:
        circ.Reset(q, condition=condition)
    ls = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    qubits = [data_qubits[i] for i in ls[index]]
    circ.H(ancl_qubits[1], condition=condition)
    circ.CX(ancl_qubits[1], qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[0], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[1], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[1], qubits[1], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[1], qubits[2], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[2], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(qubits[3], ancl_qubits[0], condition=condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    circ.CX(ancl_qubits[1], qubits[3], condition=condition)
    circ.H(ancl_qubits[1], condition=condition)
    circ.Measure(ancl_qubits[0], ancl_bits[0], condition=condition)
    circ.Measure(ancl_qubits[1], ancl_bits[1], condition=condition)
    return circ


def get_prep_non_ft(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        return _get_prep_non_ft(qubits)
    else:
        return _get_prep_non_ft_cond(qubits, condition)


def _get_prep_non_ft(qubits: list[Qubit]) -> Circuit:
    circ = Circuit()
    for q in qubits:
        circ.add_qubit(q)
        circ.add_gate(OpType.Reset, [q])
    # Add Hadamard gates
    circ.H(qubits[0])
    circ.H(qubits[4])
    circ.H(qubits[6])
    # Add CX gates
    circ.CX(qubits[0], qubits[1])
    circ.CX(qubits[4], qubits[5])
    circ.CX(qubits[6], qubits[3])
    circ.CX(qubits[6], qubits[5])
    circ.CX(qubits[4], qubits[2])
    circ.CX(qubits[0], qubits[3])
    circ.CX(qubits[4], qubits[1])
    circ.CX(qubits[3], qubits[2])
    return circ


def _get_prep_non_ft_cond(qubits: list[Qubit], condition=Bit) -> Circuit:
    circ = Circuit()
    circ.add_bit(condition)
    for q in qubits:
        circ.add_qubit(q)
        circ.add_gate(OpType.Reset, [q], condition=condition)
    # Add Hadamard gates
    circ.H(qubits[0], condition=condition)
    circ.H(qubits[4], condition=condition)
    circ.H(qubits[6], condition=condition)
    # Add CX gates
    circ.CX(qubits[0], qubits[1], condition=condition)
    circ.CX(qubits[4], qubits[5], condition=condition)
    circ.CX(qubits[6], qubits[3], condition=condition)
    circ.CX(qubits[6], qubits[5], condition=condition)
    circ.CX(qubits[4], qubits[2], condition=condition)
    circ.CX(qubits[0], qubits[3], condition=condition)
    circ.CX(qubits[4], qubits[1], condition=condition)
    circ.CX(qubits[3], qubits[2], condition=condition)
    return circ


def get_prep_ft(
    qubits: list[Qubit],
    goto_qubit: Qubit,
    goto_bit: Bit,
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        return _get_prep_ft(qubits, goto_qubit, goto_bit)
    else:
        return _get_prep_ft_cond(qubits, goto_qubit, goto_bit, condition)


def _get_prep_ft(
    qubits: list[Qubit],
    goto_qubit: Qubit,
    goto_bit: Bit,
) -> Circuit:
    circ = Circuit()
    for q in qubits + [goto_qubit]:
        circ.add_qubit(q)
        circ.Reset(q)
    circ.add_bit(goto_bit)
    # Add Hadamard gates.
    circ.H(qubits[0])
    circ.H(qubits[4])
    circ.H(qubits[6])
    # Add CX gates.
    circ.CX(qubits[0], qubits[1])
    circ.CX(qubits[4], qubits[5])
    circ.CX(qubits[6], qubits[3])
    circ.CX(qubits[6], qubits[5])
    circ.CX(qubits[4], qubits[2])
    circ.CX(qubits[0], qubits[3])
    circ.CX(qubits[4], qubits[1])
    circ.CX(qubits[3], qubits[2])
    # Mid-circuit measurement.
    circ.add_barrier([qubits[i] for i in [1, 3, 5]])
    circ.CX(qubits[1], goto_qubit)
    circ.CX(qubits[3], goto_qubit)
    circ.CX(qubits[5], goto_qubit)
    circ.Measure(goto_qubit, goto_bit)
    return circ


def _get_prep_ft_cond(
    qubits: list[Qubit],
    goto_qubit: Qubit,
    goto_bit: Bit,
    condition: Bit,
) -> Circuit:
    assert len(qubits) == 7
    circ = Circuit()
    circ.add_bit(goto_bit)
    if goto_bit != condition:
        circ.add_bit(condition)
    for q in qubits + [goto_qubit]:
        circ.add_qubit(q)
        circ.Reset(q, condition=condition)
    # Add Hadamard gates.
    circ.H(qubits[0], condition=condition)
    circ.H(qubits[4], condition=condition)
    circ.H(qubits[6], condition=condition)
    # Add CX gates.
    circ.CX(qubits[0], qubits[1], condition=condition)
    circ.CX(qubits[4], qubits[5], condition=condition)
    circ.CX(qubits[6], qubits[3], condition=condition)
    circ.CX(qubits[6], qubits[5], condition=condition)
    circ.CX(qubits[4], qubits[2], condition=condition)
    circ.CX(qubits[0], qubits[3], condition=condition)
    circ.CX(qubits[4], qubits[1], condition=condition)
    circ.CX(qubits[3], qubits[2], condition=condition)
    # Mid-circuit measurement.
    circ.add_barrier([qubits[i] for i in [1, 3, 5]])
    circ.CX(qubits[1], goto_qubit, condition=condition)
    circ.CX(qubits[3], goto_qubit, condition=condition)
    circ.CX(qubits[5], goto_qubit, condition=condition)
    circ.Measure(goto_qubit, goto_bit, condition=condition)
    return circ


def get_CX(
    ctrl_qubits: list[Qubit],
    targ_qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        return _get_CX(ctrl_qubits, targ_qubits)
    else:
        return _get_CX_cond(ctrl_qubits, targ_qubits, condition)


def _get_CX(
    ctrl_qubits: list[Qubit],
    targ_qubits: list[Qubit],
) -> Circuit:
    assert len(ctrl_qubits) == 7
    assert len(targ_qubits) == 7
    circ = Circuit()
    for qc, qt in zip(ctrl_qubits, targ_qubits):
        circ.add_qubit(qc)
        circ.add_qubit(qt)
    circ.add_barrier(ctrl_qubits + targ_qubits)
    for qc, qt in zip(ctrl_qubits, targ_qubits):
        circ.CX(qc, qt)
    circ.add_barrier(ctrl_qubits + targ_qubits)
    return circ


def _get_CX_cond(
    ctrl_qubits: list[Qubit],
    targ_qubits: list[Qubit],
    condition: Bit,
) -> Circuit:
    """Transversal CNOT."""
    assert len(ctrl_qubits) == 7
    assert len(targ_qubits) == 7
    circ = Circuit()
    for qc, qt in zip(ctrl_qubits, targ_qubits):
        circ.add_qubit(qc)
        circ.add_qubit(qt)
    circ.add_bit(condition)
    circ.add_barrier(ctrl_qubits + targ_qubits)
    for qc, qt in zip(ctrl_qubits, targ_qubits):
        circ.CX(qc, qt, condition=condition)
    circ.add_barrier(ctrl_qubits + targ_qubits)
    return circ


def get_steane_decoder(
    ancl_bits: list[Bit],
    synd_bits: list[Bit],
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        return _get_steane_decoder(ancl_bits, synd_bits)
    else:
        return _get_steane_decoder_cond(ancl_bits, synd_bits, condition)


def _get_steane_decoder(
    ancl_bits: list[Bit],
    synd_bits: list[Bit],
) -> Circuit:
    assert len(ancl_bits) == 7
    assert len(synd_bits) == 3
    circ = Circuit()
    for b in ancl_bits + synd_bits:
        circ.add_bit(b)
    circ.add_classicalexpbox_bit(
        ancl_bits[0] ^ ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[3],
        [synd_bits[0]],
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[4] ^ ancl_bits[5],
        [synd_bits[1]],
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[2] ^ ancl_bits[3] ^ ancl_bits[5] ^ ancl_bits[6],
        [synd_bits[2]],
    )
    return circ


def _get_steane_decoder_cond(
    ancl_bits: list[Bit],
    synd_bits: list[Bit],
    condition: Bit,
) -> Circuit:
    assert len(ancl_bits) == 7
    assert len(synd_bits) == 3
    circ = Circuit()
    for b in ancl_bits + synd_bits + [condition]:
        circ.add_bit(b)
    circ.add_classicalexpbox_bit(
        ancl_bits[0] ^ ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[3],
        [synd_bits[0]],
        condition=condition,
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[4] ^ ancl_bits[5],
        [synd_bits[1]],
        condition=condition,
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[2] ^ ancl_bits[3] ^ ancl_bits[5] ^ ancl_bits[6],
        [synd_bits[2]],
        condition=condition,
    )
    return circ


def get_measure(
    qubits: list[Qubit],
    bits: list[Bit],
    condition: Bit | None = None,
) -> Circuit:
    # Measurements of the ancillae.
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q, b in zip(qubits, bits):
        circ.add_qubit(q)
        circ.add_bit(b)
        if condition is None:
            circ.Measure(q, b)
        else:
            circ.Measure(q, b, condition=condition)
    return circ


def get_Y(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in [qubits[i] for i in LOGICAL_Y]:
        circ.add_qubit(q)
        if condition is None:
            circ.Y(q)
        else:
            circ.Y(q, condition=condition)
    return circ


def get_X(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in [qubits[i] for i in LOGICAL_X]:
        circ.add_qubit(q)
        if condition is None:
            circ.X(q)
        else:
            circ.X(q, condition=condition)
    return circ


def get_X_transveral(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in qubits:
        circ.add_qubit(q)
        if condition is None:
            circ.X(q)
        else:
            circ.X(q, condition=condition)
    return circ


def get_Z(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in [qubits[i] for i in LOGICAL_Z]:
        circ.add_qubit(q)
        if condition is None:
            circ.Z(q)
        else:
            circ.Z(q, condition=condition)
    return circ


def get_H(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in qubits:
        circ.add_qubit(q)
        if condition is None:
            circ.H(q)
        else:
            circ.H(q, condition=condition)
    return circ


def get_S(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in qubits:
        circ.add_qubit(q)
        if condition is None:
            circ.Sdg(q)
        else:
            circ.Sdg(q, condition=condition)
    return circ


def get_Sdg(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in qubits:
        circ.add_qubit(q)
        if condition is None:
            circ.S(q)
        else:
            circ.S(q, condition=condition)
    return circ


def get_V(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in qubits:
        circ.add_qubit(q)
        if condition is None:
            circ.Vdg(q)
        else:
            circ.Vdg(q, condition=condition)
    return circ


def get_Vdg(
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    circ = Circuit()
    if condition is not None:
        circ.add_bit(condition)
    for q in qubits:
        circ.add_qubit(q)
        if condition is None:
            circ.V(q)
        else:
            circ.V(q, condition=condition)
    return circ


def get_steane_correct_x_non_ft(
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Qubit],
    synd_bits: list[Qubit],
) -> Circuit:
    circ = Circuit()
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 7
    assert len(ancl_bits) == 7
    assert len(synd_bits) == 3
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    circ.add_barrier(data_qubits + ancl_qubits)
    # FT 0 state preparation.
    circ.append(get_prep_non_ft(ancl_qubits))
    # Logical CX (transversal CXs).
    circ.append(get_CX(ancl_qubits, data_qubits))
    circ.append(get_H(ancl_qubits))
    # Measurements of the ancillae.
    circ.append(get_measure(ancl_qubits, ancl_bits))
    # Add the Steane decorder.
    circ.append(get_steane_decoder(ancl_bits, synd_bits))
    for ii, syndrome in enumerate(itertools.product(*[range(2)] * 3)):
        qec_target = decoder(syndrome[::-1])
        if qec_target is not None:
            circ.Z(
                data_qubits[qec_target],
                condition_bits=synd_bits,
                condition_value=ii,
            )
    return circ


def get_steane_correct_x(
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Qubit],
    synd_bits: list[Qubit],
    goto_qubit: Qubit,
    goto_bit: Bit,
    n_rus: int = 1,
) -> Circuit:
    circ = Circuit()
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 7
    assert len(ancl_bits) == 7
    assert len(synd_bits) == 3
    for q in data_qubits + ancl_qubits + [goto_qubit]:
        circ.add_qubit(q)
    circ.add_barrier(data_qubits + ancl_qubits + [goto_qubit])
    # FT 0 state preparation.
    circ.append(get_prep_ft(ancl_qubits, goto_qubit, goto_bit))
    # Repeat until success.
    for _ in range(n_rus - 1):
        circ.append(get_prep_ft(ancl_qubits, goto_qubit, goto_bit, condition=goto_bit))
    # Logical CX (transversal CXs).
    circ.append(get_CX(ancl_qubits, data_qubits))
    circ.append(get_H(ancl_qubits))
    # Measurements of the ancillae.
    circ.append(get_measure(ancl_qubits, ancl_bits))
    # Add the Steane decorder.
    circ.append(get_steane_decoder(ancl_bits, synd_bits))
    for ii, syndrome in enumerate(itertools.product(*[range(2)] * 3)):
        qec_target = decoder(syndrome[::-1])
        if qec_target is not None:
            circ.Z(
                data_qubits[qec_target],
                condition_bits=synd_bits,
                condition_value=ii,
            )
    return circ


def get_steane_correct_z_non_ft(
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Qubit],
    synd_bits: list[Qubit],
) -> Circuit:
    circ = Circuit()
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 7
    assert len(ancl_bits) == 7
    assert len(synd_bits) == 3
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    circ.add_barrier(data_qubits + ancl_qubits)
    # FT plus state preparation.
    circ.append(get_prep_non_ft(ancl_qubits))
    circ.append(get_H(ancl_qubits))
    # Logical CX (transversal CXs).
    circ.append(get_CX(data_qubits, ancl_qubits))
    # Measurements of the ancillae.
    circ.append(get_measure(ancl_qubits, ancl_bits))
    # Add the Steane decorder.
    circ.append(get_steane_decoder(ancl_bits, synd_bits))
    for ii, syndrome in enumerate(itertools.product(*[range(2)] * 3)):
        qec_target = decoder(syndrome[::-1])
        if qec_target is not None:
            circ.X(
                data_qubits[qec_target],
                condition_bits=synd_bits,
                condition_value=ii,
            )
    return circ


def get_steane_correct_z(
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Qubit],
    synd_bits: list[Qubit],
    goto_qubit: Qubit,
    goto_bit: Bit,
    n_rus: int = 1,
) -> Circuit:
    circ = Circuit()
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 7
    assert len(ancl_bits) == 7
    assert len(synd_bits) == 3
    for q in data_qubits + ancl_qubits + [goto_qubit]:
        circ.add_qubit(q)
    circ.add_barrier(data_qubits + ancl_qubits + [goto_qubit])
    # FT plus state preparation.
    circ.append(get_prep_ft(ancl_qubits, goto_qubit, goto_bit))
    # Repeat until success.
    for _ in range(n_rus - 1):
        circ.append(get_prep_ft(ancl_qubits, goto_qubit, goto_bit, condition=goto_bit))
    circ.append(get_H(ancl_qubits))
    # Logical CX (transversal CXs).
    circ.append(get_CX(data_qubits, ancl_qubits))
    # Measurements of the ancillae.
    circ.append(get_measure(ancl_qubits, ancl_bits))
    # Add the Steane decorder.
    circ.append(get_steane_decoder(ancl_bits, synd_bits))
    for ii, syndrome in enumerate(itertools.product(*[range(2)] * 3)):
        qec_target = decoder(syndrome[::-1])
        if qec_target is not None:
            circ.X(
                data_qubits[qec_target],
                condition_bits=synd_bits,
                condition_value=ii,
            )
    return circ


def get_stelepo_correct_x_non_ft(
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Qubit],
) -> Circuit:
    circ = Circuit()
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 7
    assert len(ancl_bits) == 11
    synd_bits = ancl_bits[7:]
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for c in ancl_bits:
        circ.add_bit(c)
    circ.add_barrier(data_qubits + ancl_qubits)
    # Non-FT plus state preparation.
    circ.append(get_prep_non_ft(ancl_qubits))
    # Logical CX (transversal CXs).
    circ.append(get_CX(data_qubits, ancl_qubits))
    # Measurements of the ancillae in the X basis.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, ancl_bits))
    # Add the Steane decorder.
    # circ.append(get_steane_decoder(ancl_bits, synd_bits))
    # Measurement outcome.
    circ.add_classicalexpbox_bit(
        ancl_bits[0]
        ^ ancl_bits[1]
        ^ ancl_bits[2]
        ^ ancl_bits[3]
        ^ ancl_bits[4]
        ^ ancl_bits[5]
        ^ ancl_bits[6],
        [synd_bits[3]],
    )
    # Syndrome checks.
    circ.add_classicalexpbox_bit(
        ancl_bits[0] ^ ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[3],
        [synd_bits[0]],
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[4] ^ ancl_bits[5],
        [synd_bits[1]],
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[2] ^ ancl_bits[3] ^ ancl_bits[5] ^ ancl_bits[6],
        [synd_bits[2]],
    )
    # Check if error occurs or not.
    circ.add_classicalexpbox_bit(
        synd_bits[0] | synd_bits[1] | synd_bits[2],
        [synd_bits[0]],
    )
    # Error corrected measurement outcome.
    circ.add_classicalexpbox_bit(
        synd_bits[3] ^ synd_bits[0],
        [synd_bits[3]],
    )
    circ.append(get_Z(ancl_qubits, condition=synd_bits[3]))
    # Swap operations.
    circ.add_barrier(data_qubits + ancl_qubits)
    for dataq, anclq in zip(data_qubits, ancl_qubits):
        # circ.Reset(dataq)
        circ.SWAP(dataq, anclq)
    return circ


def get_rzk_non_ft(
    angle_bits: list[int],
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit,
    _head: bool = True,
) -> Circuit:
    bits = []
    flag = False
    for ii in angle_bits[::-1]:
        if ii > 0:
            flag = True
        if flag:
            bits = [ii] + bits
    bits = tuple(bits)
    circ = Circuit()
    if _head:
        circ.add_bit(condition)
        circ.add_c_setbits([True], [condition])
    # Logical I.
    if bits == tuple([]):
        return circ
    # Logical Z.
    elif bits == (1,):
        circ.append(get_Z(data_qubits))
        return circ
    # Logical S/Sdg.
    elif bits == (0, 1):
        circ.append(get_S(data_qubits, condition=condition))
        return circ
    elif bits == (1, 1):
        circ.append(get_Sdg(data_qubits, condition=condition))
        return circ
    # Logical T or higher precision Rzk.
    phase = sum([kval * 2**-i for i, kval in enumerate(angle_bits)])
    circ.append(
        get_rz_non_ft(
            phase, data_qubits, ancl_qubits, ancl_bits, condition, condition=condition
        )
    )
    # Recursive implementation.
    circ.append(
        get_rzk_non_ft(
            angle_bits[1:], data_qubits, ancl_qubits, ancl_bits, condition, _head=False
        )
    )
    return circ


def get_rzk_meas_ft(
    angle_bits: list[int],
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit,
    _head: bool = True,
) -> Circuit:
    bits = []
    flag = False
    for ii in angle_bits[::-1]:
        if ii > 0:
            flag = True
        if flag:
            bits = [ii] + bits
    bits = tuple(bits)
    circ = Circuit()
    if _head:
        circ.add_bit(condition)
        circ.add_c_setbits([True], [condition])
    # Logical I.
    if bits == tuple([]):
        return circ
    # Logical Z.
    elif bits == (1,):
        circ.append(get_Z(data_qubits))
        return circ
    # Logical S/Sdg.
    elif bits == (0, 1):
        circ.append(get_S(data_qubits, condition=condition))
        return circ
    elif bits == (1, 1):
        circ.append(get_Sdg(data_qubits, condition=condition))
        return circ
    # Logical T or higher precision Rzk.
    phase = sum([kval * 2**-i for i, kval in enumerate(angle_bits)])
    circ.append(
        get_rz_meas_ft(
            phase,
            data_qubits,
            ancl_qubits,
            ancl_bits,
            condition,
        )
    )
    # Recursive implementation.
    circ.append(
        get_rzk_meas_ft(
            angle_bits[1:], data_qubits, ancl_qubits, ancl_bits, condition, _head=False
        )
    )
    return circ


def get_rzk_part_ft(
    angle_bits: list[int],
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    discard_bit: Bit,
    condition: Bit,
    n_rus: int,
    _head: bool = True,
) -> Circuit:
    bits = []
    flag = False
    for ii in angle_bits[::-1]:
        if ii > 0:
            flag = True
        if flag:
            bits = [ii] + bits
    bits = tuple(bits)
    circ = Circuit()
    circ.add_bit(condition)
    circ.add_bit(discard_bit)
    if _head:
        circ.add_c_setbits([True], [condition])
    # Logical I.
    if bits == tuple([]):
        return circ
    # Logical Z.
    elif bits == (1,):
        circ.append(get_Z(data_qubits))
        return circ
    # Logical S/Sdg.
    elif bits == (0, 1):
        circ.append(get_S(data_qubits, condition=condition))
        return circ
    elif bits == (1, 1):
        circ.append(get_Sdg(data_qubits, condition=condition))
        return circ
    # Logical T or higher precision Rzk.
    phase = sum([kval * 2**-i for i, kval in enumerate(angle_bits)])
    synd_bits = ancl_bits[7:11]
    flag_bit = ancl_bits[11]
    circ.append(
        get_rz_part_ft(
            phase,
            data_qubits,
            ancl_qubits,
            ancl_bits[:7],
            synd_bits,
            flag_bit,
            n_rus,
            condition,
        )
    )
    # Post selection.
    circ.add_classicalexpbox_bit(
        flag_bit | discard_bit,
        [discard_bit],
    )
    # Recursive implementation.
    circ.append(
        get_rzk_part_ft(
            angle_bits[1:],
            data_qubits,
            ancl_qubits,
            ancl_bits,
            discard_bit,
            condition,
            n_rus,
            _head=False,
        )
    )
    return circ


def get_rzk_part_ft_goto(
    angle_bits: list[int],
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    discard_bit: Bit,
    condition: Bit,
    n_rus: int,
    _head: bool = True,
) -> Circuit:
    bits = []
    flag = False
    for ii in angle_bits[::-1]:
        if ii > 0:
            flag = True
        if flag:
            bits = [ii] + bits
    bits = tuple(bits)
    circ = Circuit()
    circ.add_bit(condition)
    circ.add_bit(discard_bit)
    if _head:
        circ.add_c_setbits([True], [condition])
    # Logical I.
    if bits == tuple([]):
        return circ
    # Logical Z.
    elif bits == (1,):
        circ.append(get_Z(data_qubits))
        return circ
    # Logical S/Sdg.
    elif bits == (0, 1):
        circ.append(get_S(data_qubits, condition=condition))
        return circ
    elif bits == (1, 1):
        circ.append(get_Sdg(data_qubits, condition=condition))
        return circ
    # Logical T or higher precision Rzk.
    phase = sum([kval * 2**-i for i, kval in enumerate(angle_bits)])
    synd_bits = ancl_bits[7:12]
    flag_bit = ancl_bits[12]
    circ.append(
        get_rz_part_ft_goto(
            phase,
            data_qubits,
            ancl_qubits,
            ancl_bits[:7],
            synd_bits,
            flag_bit,
            n_rus,
            condition,
        )
    )
    # Post selection.
    circ.add_classicalexpbox_bit(
        flag_bit | discard_bit,
        [discard_bit],
    )
    # Recursive implementation.
    circ.append(
        get_rzk_part_ft_goto(
            angle_bits[1:],
            data_qubits,
            ancl_qubits,
            ancl_bits,
            discard_bit,
            condition,
            n_rus,
            _head=False,
        )
    )
    return circ


def get_rz_non_ft(
    phase: float,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    flag_bit: Bit,
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        return _get_rz_non_ft(phase, data_qubits, ancl_qubits, ancl_bits, flag_bit)
    else:
        return _get_rz_non_ft_cond(
            phase, data_qubits, ancl_qubits, ancl_bits, flag_bit, condition
        )


def get_rz_meas_ft(
    phase: float,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 7
    assert len(ancl_bits) == 7 + 4
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits + [condition]:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    # Non-FT Rz|+> state preparation.
    circ.append(get_prep_rzplus_non_ft(phase, ancl_qubits, condition=condition))
    # Ancilla measurement for the gate teleportation.
    circ.append(get_CX(data_qubits, ancl_qubits, condition=condition))
    # FT Measurement.
    for i in range(7):
        circ.Measure(ancl_qubits[i], ancl_bits[i], condition=condition)
    # Save the parity in ancl_bit[0].
    synd_bits = ancl_bits[7:]
    circ.add_classicalexpbox_bit(
        ancl_bits[0]
        ^ ancl_bits[1]
        ^ ancl_bits[2]
        ^ ancl_bits[3]
        ^ ancl_bits[4]
        ^ ancl_bits[5]
        ^ ancl_bits[6],
        [synd_bits[3]],
        condition=condition,
    )
    # Syndrome checks.
    circ.add_classicalexpbox_bit(
        ancl_bits[0] ^ ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[3],
        [synd_bits[0]],
        condition=condition,
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[4] ^ ancl_bits[5],
        [synd_bits[1]],
        condition=condition,
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[2] ^ ancl_bits[3] ^ ancl_bits[5] ^ ancl_bits[6],
        [synd_bits[2]],
        condition=condition,
    )
    # Check if error occurs or not.
    circ.add_classicalexpbox_bit(
        synd_bits[0] | synd_bits[1] | synd_bits[2],
        [synd_bits[0]],
        condition=condition,
    )
    # Error corrected measurement outcome.
    circ.add_classicalexpbox_bit(
        synd_bits[3] ^ synd_bits[0],
        [condition],
        condition=condition,
    )
    return circ


def get_prep_rzplus_non_ft(
    phase: float,
    qubits: list[Qubit],
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        circ = _get_prep_rzplus_non_ft(phase, qubits)
    else:
        circ = _get_prep_rzplus_non_ft_cond(phase, qubits, condition)
    return circ


def _get_prep_rzplus_non_ft(
    phase: float,
    qubits: list[Qubit],
) -> Circuit:
    assert len(qubits) == 7
    circ = Circuit()
    for q in qubits:
        circ.add_qubit(q)
    for q in qubits:
        circ.add_gate(OpType.Reset, [q])
    circ.H(qubits[0])
    circ.H(qubits[4])
    circ.H(qubits[6])
    circ.CX(qubits[0], qubits[1])
    circ.CX(qubits[4], qubits[5])
    circ.CX(qubits[6], qubits[3])
    circ.CX(qubits[6], qubits[5])
    circ.CX(qubits[4], qubits[2])
    circ.CX(qubits[0], qubits[3])
    circ.CX(qubits[4], qubits[1])
    circ.XXPhase(phase, qubits[3], qubits[4])
    circ.CX(qubits[3], qubits[2])
    circ.append(get_H(qubits))
    return circ


def _get_prep_rzplus_non_ft_cond(
    phase: float,
    qubits: list[Qubit],
    condition: Bit,
) -> Circuit:
    assert len(qubits) == 7
    circ = Circuit()
    for q in qubits:
        circ.add_qubit(q)
    circ.add_bit(condition)
    for q in qubits:
        circ.add_gate(OpType.Reset, [q], condition=condition)
    circ.H(qubits[0], condition=condition)
    circ.H(qubits[4], condition=condition)
    circ.H(qubits[6], condition=condition)
    circ.CX(qubits[0], qubits[1], condition=condition)
    circ.CX(qubits[4], qubits[5], condition=condition)
    circ.CX(qubits[6], qubits[3], condition=condition)
    circ.CX(qubits[6], qubits[5], condition=condition)
    circ.CX(qubits[4], qubits[2], condition=condition)
    circ.CX(qubits[0], qubits[3], condition=condition)
    circ.CX(qubits[4], qubits[1], condition=condition)
    circ.XXPhase(phase, qubits[3], qubits[4], condition=condition)
    circ.CX(qubits[3], qubits[2], condition=condition)
    circ.append(get_H(qubits, condition=condition))
    return circ


def _get_rz_non_ft(
    phase: float,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    flag_bit: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 7
    assert len(ancl_bits) == 3
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits + [flag_bit]:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    # Non-FT Rz|+> state preparation.
    circ.append(get_prep_rzplus_non_ft(phase, ancl_qubits))
    # Gate teleportation.
    circ.append(get_CX(data_qubits, ancl_qubits))
    for i, j in enumerate(LOGICAL_Z):
        circ.Measure(ancl_qubits[j], ancl_bits[i])
    circ.add_classicalexpbox_bit(
        ancl_bits[0] ^ ancl_bits[1] ^ ancl_bits[2],
        [flag_bit],
    )
    return circ


def _get_rz_non_ft_cond(
    phase: float,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    flag_bit: Bit,
    condition: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 7
    assert len(ancl_bits) == 3
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits + [flag_bit]:
        circ.add_bit(b)
    if flag_bit != condition:
        circ.add_bit(condition)
    circ.add_barrier(data_qubits + ancl_qubits)
    # Non-FT Rz|+> state preparation.
    circ.append(get_prep_rzplus_non_ft(phase, ancl_qubits, condition=condition))
    # Gate teleportation.
    circ.append(get_CX(data_qubits, ancl_qubits, condition=condition))
    for i, j in enumerate(LOGICAL_Z):
        circ.Measure(ancl_qubits[j], ancl_bits[i], condition=condition)
    circ.add_classicalexpbox_bit(
        ancl_bits[0] ^ ancl_bits[1] ^ ancl_bits[2],
        [flag_bit],
        condition=condition,
    )
    return circ


def get_rz_part_ft(
    phase: float,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    synd_bits: list[Bit],
    flag_bit: Bit,
    n_rus: int,
    condition: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 9
    assert len(ancl_bits) == 7
    assert len(synd_bits) == 4
    # Prepare the qubit/bit registers.
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits + synd_bits + [condition, flag_bit]:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    # # FT |+> state preparation with repeat until success.
    circ.append(
        get_prep_rz_part_ft(
            phase,
            ancl_qubits,
            synd_bits + [flag_bit],
            n_rus,
            condition=condition,
        )
    )
    # Gate teleportation.
    circ.append(get_CX(data_qubits, ancl_qubits[:7], condition=condition))
    # FT Measurement.
    for i in range(7):
        circ.Measure(ancl_qubits[i], ancl_bits[i], condition=condition)
    # Save the parity in ancl_bit[2].
    circ.add_classicalexpbox_bit(
        ancl_bits[0]
        ^ ancl_bits[1]
        ^ ancl_bits[2]
        ^ ancl_bits[3]
        ^ ancl_bits[4]
        ^ ancl_bits[5]
        ^ ancl_bits[6],
        [synd_bits[3]],
        condition=condition,
    )
    # Syndrome checks.
    circ.add_classicalexpbox_bit(
        ancl_bits[0] ^ ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[3],
        [synd_bits[0]],
        condition=condition,
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[4] ^ ancl_bits[5],
        [synd_bits[1]],
        condition=condition,
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[2] ^ ancl_bits[3] ^ ancl_bits[5] ^ ancl_bits[6],
        [synd_bits[2]],
        condition=condition,
    )
    # Check if error occurs or not.
    circ.add_classicalexpbox_bit(
        synd_bits[0] | synd_bits[1] | synd_bits[2],
        [synd_bits[0]],
        condition=condition,
    )
    # Error corrected measurement outcome.
    circ.add_classicalexpbox_bit(
        synd_bits[3] ^ synd_bits[0],
        [condition],
        condition=condition,
    )
    return circ


def get_rz_part_ft_goto(
    phase: float,
    data_qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    synd_bits: list[Bit],
    flag_bit: Bit,
    n_rus: int,
    condition: Bit,
) -> Circuit:
    assert len(data_qubits) == 7
    assert len(ancl_qubits) == 9
    assert len(ancl_bits) == 7
    assert len(synd_bits) == 5
    # Prepare the qubit/bit registers.
    circ = Circuit()
    for q in data_qubits + ancl_qubits:
        circ.add_qubit(q)
    for b in ancl_bits + synd_bits + [condition, flag_bit]:
        circ.add_bit(b)
    circ.add_barrier(data_qubits + ancl_qubits)
    # # FT |+> state preparation with repeat until success.
    circ.append(
        get_prep_rz_part_ft_goto(
            phase,
            ancl_qubits,
            synd_bits + [flag_bit],
            n_rus,
            condition=condition,
        )
    )
    # Gate teleportation.
    circ.append(get_CX(data_qubits, ancl_qubits[:7], condition=condition))
    # FT Measurement.
    for i in range(7):
        circ.Measure(ancl_qubits[i], ancl_bits[i], condition=condition)
    # Save the parity in ancl_bit[2].
    circ.add_classicalexpbox_bit(
        ancl_bits[0]
        ^ ancl_bits[1]
        ^ ancl_bits[2]
        ^ ancl_bits[3]
        ^ ancl_bits[4]
        ^ ancl_bits[5]
        ^ ancl_bits[6],
        [synd_bits[3]],
        condition=condition,
    )
    # Syndrome checks.
    circ.add_classicalexpbox_bit(
        ancl_bits[0] ^ ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[3],
        [synd_bits[0]],
        condition=condition,
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[1] ^ ancl_bits[2] ^ ancl_bits[4] ^ ancl_bits[5],
        [synd_bits[1]],
        condition=condition,
    )
    circ.add_classicalexpbox_bit(
        ancl_bits[2] ^ ancl_bits[3] ^ ancl_bits[5] ^ ancl_bits[6],
        [synd_bits[2]],
        condition=condition,
    )
    # Check if error occurs or not.
    circ.add_classicalexpbox_bit(
        synd_bits[0] | synd_bits[1] | synd_bits[2],
        [synd_bits[0]],
        condition=condition,
    )
    # Error corrected measurement outcome.
    circ.add_classicalexpbox_bit(
        synd_bits[3] ^ synd_bits[0],
        [condition],
        condition=condition,
    )
    return circ


def get_iceberg_zzx(
    qubits: list[Qubit],
    ancl_qubits: list[Qubit],
    ancl_bits: list[Bit],
    condition: Bit | None = None,
) -> Circuit:
    assert len(qubits) == 7
    assert len(ancl_qubits) == 2
    assert len(ancl_bits) == 2
    circ = Circuit()
    # XXXXIII and ZZZZIII.
    circ.append(
        get_iceberg_zx(
            0,
            qubits,
            ancl_qubits,
            ancl_bits,
            condition=condition,
        )
    )
    # IXXIXXI.
    circ.append(
        get_iceberg_z(
            1,
            qubits,
            ancl_qubits,
            ancl_bits,
            condition=condition,
        )
    )
    return circ


def get_prep_rz_part_ft(
    phase: float,
    qubits: list[Qubit],
    bits: list[Bit],
    n_rus: int,
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        circ = _get_prep_rz_part_ft(phase, qubits, bits, n_rus)
    else:
        circ = _get_prep_rz_part_ft_cond(phase, qubits, bits, n_rus, condition)
    return circ


def get_prep_rz_part_ft_goto(
    phase: float,
    qubits: list[Qubit],
    bits: list[Bit],
    n_rus: int,
    condition: Bit | None = None,
) -> Circuit:
    if condition is None:
        circ = _get_prep_rz_part_ft_goto(phase, qubits, bits, n_rus)
    else:
        circ = _get_prep_rz_part_ft_goto_cond(phase, qubits, bits, n_rus, condition)
    return circ


def _get_prep_rz_part_ft_goto(
    phase: float,
    qubits: list[Qubit],
    bits: list[Bit],
    n_rus: int,
) -> Circuit:
    assert len(qubits) == 9
    assert len(bits) == 6
    circ = Circuit()
    for q in qubits:
        circ.add_qubit(q)
    for b in bits:
        circ.add_bit(b)
    synd_bits = bits[:5]
    flag_bit = bits[5]
    # FT |+> state preparation.
    circ.append(get_prep_ft(qubits[:7], qubits[7], synd_bits[0]))
    circ.append(get_H(qubits[:7]))
    # Non-FT Rz.
    circ.append(get_Rz_direct(phase, qubits[:7]))
    circ.append(get_iceberg_zx(0, qubits[:7], qubits[7:9], synd_bits[1:3]))
    circ.append(get_iceberg_zx(1, qubits[:7], qubits[7:9], synd_bits[3:5]))
    circ.add_classicalexpbox_bit(
        synd_bits[0] | synd_bits[1] | synd_bits[2] | synd_bits[3] | synd_bits[4],
        [flag_bit],
    )
    for _ in range(n_rus - 1):
        # FT |+> state preparation.
        circ.append(
            get_prep_ft(qubits[:7], qubits[7], synd_bits[0], condition=flag_bit)
        )
        circ.append(get_H(qubits[:7], condition=flag_bit))
        # Non-FT Rz.
        circ.append(get_Rz_direct(phase, qubits[:7], condition=flag_bit))
        # Iceberg-style error detections.
        circ.append(
            get_iceberg_zx(
                0, qubits[:7], qubits[7:9], synd_bits[1:3], condition=flag_bit
            )
        )
        circ.append(
            get_iceberg_zx(
                1, qubits[:7], qubits[7:9], synd_bits[3:5], condition=flag_bit
            )
        )
        circ.add_classicalexpbox_bit(
            synd_bits[0] | synd_bits[1] | synd_bits[2] | synd_bits[3] | synd_bits[4],
            [flag_bit],
            condition=flag_bit,
        )
    return circ


def _get_prep_rz_part_ft(
    phase: float,
    qubits: list[Qubit],
    bits: list[Bit],
    n_rus: int,
) -> Circuit:
    assert len(qubits) == 9
    assert len(bits) == 5
    circ = Circuit()
    for q in qubits:
        circ.add_qubit(q)
    for b in bits:
        circ.add_bit(b)
    synd_bits = bits[:4]
    flag_bit = bits[4]
    # circ.append(get_prep_ft(qubits[:7], qubits[7], synd_bits[0]))
    # circ.append(get_H(qubits[:7]))
    # circ.append(get_Rz_direct(phase, qubits[:7]))
    circ.append(get_prep_rzplus_non_ft(phase, qubits[:7]))
    circ.append(get_iceberg_zx(0, qubits[:7], qubits[7:9], synd_bits[0:2]))
    circ.append(get_iceberg_zx(1, qubits[:7], qubits[7:9], synd_bits[2:4]))
    circ.add_classicalexpbox_bit(
        synd_bits[0] | synd_bits[1] | synd_bits[2] | synd_bits[3],
        [flag_bit],
    )
    for _ in range(n_rus - 1):
        # Non-FT Rz(x)|+> state preparation.
        circ.append(get_prep_rzplus_non_ft(phase, qubits[:7], condition=flag_bit))
        # Iceberg-style error detections.
        circ.append(
            get_iceberg_zx(
                0, qubits[:7], qubits[7:9], synd_bits[0:2], condition=flag_bit
            )
        )
        circ.append(
            get_iceberg_zx(
                1, qubits[:7], qubits[7:9], synd_bits[2:4], condition=flag_bit
            )
        )
        circ.add_classicalexpbox_bit(
            synd_bits[0] | synd_bits[1] | synd_bits[2] | synd_bits[3],
            [flag_bit],
            condition=flag_bit,
        )
    return circ


def _get_prep_rz_part_ft_cond(
    phase: float,
    qubits: list[Qubit],
    bits: list[Bit],
    n_rus: int,
    condition: Bit,
) -> Circuit:
    assert len(qubits) == 9
    assert len(bits) == 5
    circ = Circuit()
    for q in qubits:
        circ.add_qubit(q)
    for b in bits + [condition]:
        circ.add_bit(b)
    synd_bits = bits[:4]
    flag_bit = bits[4]
    circ.add_c_setbits([False], [flag_bit])
    circ.add_c_setbits([True], [flag_bit], condition=condition)
    for _ in range(n_rus):
        # Non-FT Rz(x)|+> state preparation.
        circ.append(get_prep_rzplus_non_ft(phase, qubits[:7], condition=flag_bit))
        # Iceberg-style error detection.
        circ.append(
            get_iceberg_zx(
                0, qubits[:7], qubits[7:9], synd_bits[0:2], condition=flag_bit
            )
        )
        circ.append(
            get_iceberg_zx(
                1, qubits[:7], qubits[7:9], synd_bits[2:4], condition=flag_bit
            )
        )
        circ.add_classicalexpbox_bit(
            synd_bits[0] | synd_bits[1] | synd_bits[2] | synd_bits[3],
            [flag_bit],
            condition=flag_bit,
        )
    return circ


def _get_prep_rz_part_ft_goto_cond(
    phase: float,
    qubits: list[Qubit],
    bits: list[Bit],
    n_rus: int,
    condition: Bit,
) -> Circuit:
    assert len(qubits) == 9
    assert len(bits) == 6
    circ = Circuit()
    for q in qubits:
        circ.add_qubit(q)
    for b in bits + [condition]:
        circ.add_bit(b)
    synd_bits = bits[:5]
    flag_bit = bits[5]
    circ.add_c_setbits([False], [flag_bit])
    circ.add_c_setbits([True], [flag_bit], condition=condition)
    for _ in range(n_rus):
        # FT |+> state preparation.
        circ.append(
            get_prep_ft(qubits[:7], qubits[7], synd_bits[0], condition=flag_bit)
        )
        circ.append(get_H(qubits[:7], condition=flag_bit))
        # FT Rz(x).
        circ.append(get_Rz_direct(phase, qubits[:7], condition=flag_bit))
        # Non-FT Rz(x)|+> state preparation.
        # Iceberg-style error detection.
        circ.append(
            get_iceberg_zx(
                0, qubits[:7], qubits[7:9], synd_bits[1:3], condition=flag_bit
            )
        )
        circ.append(
            get_iceberg_zx(
                1, qubits[:7], qubits[7:9], synd_bits[3:5], condition=flag_bit
            )
        )
        circ.add_classicalexpbox_bit(
            synd_bits[0] | synd_bits[1] | synd_bits[2] | synd_bits[3] | synd_bits[4],
            [flag_bit],
            condition=flag_bit,
        )
    return circ
