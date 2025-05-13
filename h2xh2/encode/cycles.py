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

from pytket.circuit import Circuit, CustomGateDef


# each `CustomGateDef` is an empty single qubit gate. When converting to an encoded circuit
# the automatic encoding picks up the name of the custom gates and adds the desired cycle
# Steane QEC cycles
steane_z_correct: CustomGateDef = CustomGateDef.define(
    "steane_z_correct", Circuit(1), []
)
steane_x_correct: CustomGateDef = CustomGateDef.define(
    "steane_x_correct", Circuit(1), []
)
iceberg_w_0_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_w_0_detect", Circuit(1), []
)
iceberg_w_1_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_w_1_detect", Circuit(1), []
)
iceberg_w_2_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_w_2_detect", Circuit(1), []
)

iceberg_x_0_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_x_0_detect", Circuit(1), []
)
iceberg_x_1_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_x_1_detect", Circuit(1), []
)
iceberg_x_2_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_x_2_detect", Circuit(1), []
)

iceberg_z_0_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_z_0_detect", Circuit(1), []
)
iceberg_z_1_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_z_1_detect", Circuit(1), []
)
iceberg_z_2_detect: CustomGateDef = CustomGateDef.define(
    "iceberg_z_2_detect", Circuit(1), []
)

x_dynamical_decoupling: CustomGateDef = CustomGateDef.define(
    "x_dynamical_decoupling", Circuit(1), []
)

x_transversal: CustomGateDef = CustomGateDef.define("x_transversal", Circuit(1), [])
