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


def compile_and_run(circuit: Circuit, n_shots: int) -> BackendResult:
    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()  # type: ignore
    )
    compiled = backend.get_compiled_circuit(circuit, optimisation_level=0)

    # print(circuit_to_qasm_str(compiled,header="hqslib1"))
    handle = backend.process_circuit(
        compiled, n_shots, noisy_simulation=False, language=Language.QASM
    )
    return backend.get_result(handle)
