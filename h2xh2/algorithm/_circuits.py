"""Circuit builder collection covering both canonical and iterative QP(D)E circuits."""

from typing import (
    Callable,
)
from pytket.circuit import (
    Circuit,
    OpType,
    Qubit,
    Pauli,
    PauliExpBox,
)


def add_rz(
    circ: Circuit,
    qubit: Qubit | int,
    angle: float,
) -> None:
    """Add Rz gate in a bit more clever way to reduce the cost.

    Rz in replaced with S, Z, Sdg if possible, otherwise it is
    implemented in the form of PauliExpBox to be replaced with the
    iceberg-style Rz in the Steane code by convention.

    Args:
        circ: Circuit.
        qubit: Target qubit.
        angle: the Rz rotation angle.
    """
    if (angle % 2.0) == 0.0:
        return
    if int(angle * 2) == angle * 2:
        x = int((angle * 2) % 4.0)
        if x == 1:
            circ.S(qubit)
        elif x == 2:
            circ.Z(qubit)
        elif x == 3:
            circ.Sdg(qubit)
        else:
            raise RuntimeError()
    else:
        circ.add_pauliexpbox(
            PauliExpBox([Pauli.Z], angle),
            [qubit],
        )


def get_qpe_func(
    state: Circuit,
    get_ctrlu: Callable[[int], Circuit],
    n_rounds: int = 1,
) -> Callable[[int, float], Circuit]:
    """Return a function to build Kitaev's (standard) QPE circuit.

    Args:
        state: State prearation circuit.
        get_ctrlu: A function to return the circuit representing the controlled unitary.
        n_rounds: Number of rounds.

    Returns:
        A function to build the QPE circuit.
    """

    def get_circuit(k: int, beta: float) -> Circuit:
        """Build a regular QPE circuit."""
        # Set quantum/classical registers.
        circ = Circuit(state.n_qubits + n_rounds, n_rounds)
        system_register = circ.qubits[-state.n_qubits :]
        ancilla_register = circ.qubits[:n_rounds]
        classical_register = circ.bits
        # Now build the circuit.
        circ.add_circuit(state, system_register)
        for j, q in enumerate(ancilla_register[::-1]):
            circ.H(q)
            current_register = [q] + system_register
            multiple = 2**j
            ctrlu = get_ctrlu(multiple * k)
            circ.add_circuit(ctrlu, current_register)
            # if abs(multiple * beta) > 0:
            #     circ.Rz(multiple * beta, q)
            add_rz(circ, q, multiple * beta)
        iqft_circ = iqft(n_rounds)
        circ.add_circuit(iqft_circ, ancilla_register)
        # Add measurement.
        for q, c in zip(ancilla_register, classical_register):
            circ.Measure(q, c)
        return circ

    return get_circuit


def iqft(n_qubits: int) -> Circuit:
    """Inverse quantum Fourier transform (iQFT) circuit.

    Args:
        n_qubits:  Number of qubits.

    Returns:
        iQFT circuit.
    """
    circuit = Circuit(n_qubits, name="IQFT")
    for q in range(int(n_qubits / 2)):
        circuit.SWAP(q, n_qubits - q - 1)
    for q in range(n_qubits)[::-1]:
        for p in range(q + 1, n_qubits)[::-1]:
            angle = -1 * ((1 / 2) ** (p - q))
            circuit.add_gate(OpType.CU1, angle, [Qubit(p), Qubit(q)])
        circuit.H(q)
    return circuit


def qft(n_qubits: int) -> Circuit:
    """Quantum Fourier transform (QFT) circuit.

    Args:
        n_qubits: Number of qubits.

    Returns:
        QFT circuit.
    """
    circuit = Circuit(n_qubits, name="QFT")
    circuit.add_circuit(iqft(n_qubits=n_qubits).dagger(), circuit.qubits)
    return circuit
