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

from typing import (
    Callable,
    NamedTuple,
)
import numpy as np
from pytket.circuit import Circuit
from pytket.backends.backendresult import BackendResult
from ..encode import steane
from ..algorithm import (
    get_qpe_func,
    get_ms,
)
from ._chemistry import (
    get_ctrl_func,
    get_state,
)


class IqpeInput(NamedTuple):
    k_list: list[int]
    beta_list: list[int]
    encode: Callable[[Circuit, steane.EncodeOptions | None], Circuit]
    interpret: Callable[[BackendResult, steane.InterpretOptions | None], BackendResult]
    encode_options: steane.EncodeOptions | None = None
    intepret_options: steane.InterpretOptions | None = None
    qec_level: int = 0
    pft_rz: bool = False


def build_encode_iqpe_circuits(params: IqpeInput) -> list[Circuit]:
    """Build the cirucits."""
    logical_circuits = build_iqpe_circuits(
        k_list=params.k_list,
        beta_list=params.beta_list,
        pft_rz=params.pft_rz,
        qec_level=params.qec_level,
    )
    encoded_circuits = [
        params.encode(c, params.encode_options) for c in logical_circuits
    ]
    return encoded_circuits


def interpret_process_iqpe_results(
    results: list[BackendResult],
    params: IqpeInput,
) -> tuple[list[int], list[float], list[int]]:
    logical_results = [params.interpret(r, params.intepret_options) for r in results]
    iqpe_result = process_iqpe_results(
        logical_results,
        params.k_list,
        params.beta_list,
    )
    return iqpe_result


def build_iqpe_circuits(
    k_list: list[int],
    beta_list: list[float],
    pft_rz: bool,
    qec_level: int,
) -> list[Circuit]:
    state = get_state(
        benchmark=False,
        pft_rz=pft_rz,
    )
    get_ctrlu = get_ctrl_func(
        benchmark=False,
        pft_rz=pft_rz,
        qec_level=qec_level,
    )
    get_circuit = get_qpe_func(
        state,
        get_ctrlu,
    )
    circuits: list[Circuit] = []
    for k, beta in zip(k_list, beta_list):
        circuits.append(get_circuit(k, beta))
    return circuits


def process_iqpe_results(
    results: list[BackendResult],
    k_list: list[int],
    beta_list: list[float],
) -> tuple[list[int], list[float], list[int]]:
    ks, betas, ms = get_ms(k_list, beta_list, results)
    return ks, betas, ms
