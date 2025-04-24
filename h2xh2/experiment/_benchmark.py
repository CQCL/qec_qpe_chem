from typing import (
    Callable,
    NamedTuple,
)
from pytket.circuit import Circuit
from pytket.passes import RemoveBarriers
from pytket.backends.backendresult import BackendResult
from h2xh2.encode import steane
from ._chemistry import (
    get_state,
    get_ctrl_func,
)
from ..algorithm import get_qpe_func


class BenchmarkInput(NamedTuple):
    k_list: list[int]
    encode: Callable[[Circuit, steane.EncodeOptions | None], Circuit]
    interpret: Callable[[BackendResult, steane.InterpretOptions | None], BackendResult]
    encode_options: steane.EncodeOptions | None = None
    intepret_options: steane.InterpretOptions | None = None
    qec_level: int = 0
    pft_rz: bool = False


class BenchmarkResult(NamedTuple):
    p0: list[float]
    p0_unc: list[float]
    n_shots: list[int]


def build_benchmark_circuits(
    k_list: list[int],
    pft_rz: bool,
    qec_level: int,
) -> list[Circuit]:
    state = get_state(
        benchmark=True,
        pft_rz=pft_rz,
    )
    get_ctrlu = get_ctrl_func(
        benchmark=True,
        pft_rz=pft_rz,
        qec_level=qec_level,
    )
    get_circuit = get_qpe_func(
        state,
        get_ctrlu,
    )
    circuits: list[Circuit] = []
    for k in k_list:
        circuits.append(get_circuit(k, beta=0.0))
    return circuits


def process_benchmark_results(
    results: list[BackendResult],
) -> BenchmarkResult:
    ls_p0: list[float] = []
    ls_p0_unc: list[float] = []
    ls_n_shots: list[int] = []
    for r in results:
        counts = r.get_counts()
        n_shots = int(sum(counts.values()))
        ls_n_shots.append(n_shots)
        distribution = r.get_distribution()
        p0 = distribution.get((0,), 0.0)
        p0_unc = (p0 * (1 - p0) / n_shots) ** 0.5
        ls_p0.append(p0)
        ls_p0_unc.append(p0_unc)
    benchmark_result = BenchmarkResult(
        p0=ls_p0,
        n_shots=ls_n_shots,
        p0_unc=ls_p0_unc,
    )
    return benchmark_result


def build_encode_benchmark_circuits(params: BenchmarkInput) -> list[Circuit]:
    """Build the cirucits."""
    logical_circuits = build_benchmark_circuits(
        k_list=params.k_list,
        pft_rz=params.pft_rz,
        qec_level=params.qec_level,
    )
    encoded_circuits = [
        params.encode(c, params.encode_options) for c in logical_circuits
    ]
    return encoded_circuits


def build_encode_benchmark_circuits_no_barriers(
    params: BenchmarkInput,
) -> list[Circuit]:
    """Build the cirucits."""
    encoded_circuits = build_encode_benchmark_circuits(params)
    rb = RemoveBarriers()
    for circ in encoded_circuits:
        rb.apply(circ)
    return encoded_circuits


def interpret_process_benchmark_results(
    results: list[BackendResult],
    params: BenchmarkInput,
) -> BenchmarkResult:
    logical_results = [params.interpret(r, params.intepret_options) for r in results]
    benchmark_result = process_benchmark_results(logical_results)
    return benchmark_result
