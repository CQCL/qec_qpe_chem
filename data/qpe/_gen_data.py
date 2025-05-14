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


# %%
import sys
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from h2xh2.algorithm import (
    update_log,
)
from h2xh2.algorithm._bayesian_qpe import bootstrap_sampling
from h2xh2.experiment import ChemData

matplotlib.rcParams["font.size"] = 12

LAMBDA_D = 0.10806
# N_SHOTS = 100
# N_SHOTS = 500
# N_SHOTS = 1000
# N_SHOTS = 2300
N_SHOTS = int(sys.argv[1])
RESOLUTION = 2**12
PLOT_POINTS = 2**10
INTERVAL = RESOLUTION // PLOT_POINTS

# Load the data.
with open("qpe_data.json") as f:
    data = json.load(f)
ks, betas, ms = data


# Prepare the error_rate used in the noise-aware likelihood function.
def error_rate(k: int) -> float:
    lambda_d = LAMBDA_D
    val = 1 - (1 - lambda_d) ** k
    return val


ks = ks[:N_SHOTS]
betas = betas[:N_SHOTS]
ms = ms[:N_SHOTS]
n_shots = min(N_SHOTS, len(ms))

# Grid points.
phi = np.linspace(-1, 1, RESOLUTION + 1)[:-1]
posterior_log = update_log(
    phi,
    np.ones_like(phi),
    ks,
    betas,
    ms,
    error_rate=error_rate,
)
ls = []
ii = 0
for phix, posterior_logx in zip(phi, posterior_log):
    if ii % INTERVAL == 0:
        ls.append([phix, posterior_logx])
    ii += 1
with open(f"data_{n_shots:04d}.txt", "w") as f:
    json.dump(ls, f)
