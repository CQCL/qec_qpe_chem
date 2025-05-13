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

"""Collection of the experimental setup.

Note:
    Most of the codes in this module are specific to this particular project.
    Some of the chemistry setup is even hard coded with a little flexibility.
"""

from ._chemistry import ChemData
from ._benchmark import (
    BenchmarkInput,
    BenchmarkResult,
    build_encode_benchmark_circuits,
    interpret_process_benchmark_results,
    build_benchmark_circuits,
    build_encode_benchmark_circuits_no_barriers,
    process_benchmark_results,
)
from ._iqpe import (
    IqpeInput,
    build_encode_iqpe_circuits,
    build_iqpe_circuits,
    interpret_process_iqpe_results,
    process_iqpe_results,
)

__all__ = [
    "ChemData",
    "BenchmarkInput",
    "BenchmarkResult",
    "build_benchmark_circuits",
    "process_benchmark_results",
    "build_encode_benchmark_circuits",
    "interpret_process_benchmark_results",
    "IqpeInput",
    "process_iqpe_results",
    "build_iqpe_circuits",
    "build_encode_iqpe_circuits",
    "interpret_process_iqpe_results",
]
