"""
Utility functions common to all tests.
"""

from pytket import Circuit
from pytket.backends.backendresult import BackendResult
from pytket.extensions.quantinuum import (
    QuantinuumBackend,
    QuantinuumAPIOffline,
    Language,
)
from pytket.qasm import circuit_to_qasm_str

def compile_and_run(circuit: Circuit, n_shots: int) -> BackendResult:
    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
    )
    compiled = backend.get_compiled_circuit(circuit, optimisation_level=0)

    # print(circuit_to_qasm_str(compiled,header="hqslib1"))
    handle = backend.process_circuit(
        compiled, n_shots, noisy_simulation=False, language=Language.QASM
    )
    return backend.get_result(handle)
