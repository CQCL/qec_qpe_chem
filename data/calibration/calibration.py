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
import numpy as np
from matplotlib import rcParams
from scipy.optimize import curve_fit
from typing import NamedTuple
import matplotlib.pyplot as plt

rcParams["font.size"] = 12

# Parameter.
formats = ["o", "^", "v", "s", "<", ">", "p", "D", "*", "X"]


class Property(NamedTuple):
    color: str
    label: str


LSDATA = [
    ("no_qec_emulator", Property("green", "NoQEC (emulator)")),
    ("exp_emulator", Property("red", "Exp (emulator)")),
    ("exp_hardware", Property("black", "Exp (hardware)")),
]


def fun(x, error_rate):
    y = (1 + (1 - error_rate) ** x) / 2
    return y


for i, (target, prop) in enumerate(LSDATA):
    with open(f"{target}/_benchmark_result.json") as f:
        benchmark_result = json.load(f)
    ks = [3, 6, 9, 12]
    p0 = np.array(benchmark_result["p0"])
    n_shots = np.array(benchmark_result["n_shots"])

    # Calculate the circuit-wise standard error.
    se = np.sqrt(p0 * (1 - p0) / n_shots + 1e-6)
    popt, pcov = curve_fit(
        fun,
        ks,
        p0,
        sigma=se,
        p0=[0.1],
        bounds=[0.0, 1.0],
    )
    print(f"{prop.label:16} error_rate = {popt[0]:.5f} +/- {pcov[0,0]**0.5:.5f})")
    xfit = np.linspace(0, ks[-1] + 50, 1000)
    plt.errorbar(
        ks,
        2 * (1 - p0),
        yerr=2 * se,
        capsize=5,
        fmt=formats[i],
        markersize=6,
        ecolor=prop.color,
        markeredgecolor=prop.color,
        color="w",
        label=prop.label,
    )
    yfit = 2 * (1 - fun(xfit, popt[0]))
    if target not in ["no_qec_emulator"]:
        plt.plot(xfit, yfit, "-", color=prop.color, alpha=0.6)
        coeff = 1.0
        yfit_u = 2 * (1 - fun(xfit, popt[0] + coeff * pcov[0, 0] ** 0.5))
        yfit_d = 2 * (1 - fun(xfit, popt[0] - coeff * pcov[0, 0] ** 0.5))
        plt.fill_between(xfit, yfit_u, yfit_d, alpha=0.1, color=prop.color)
plt.xlabel(r"$k$")
plt.grid()
plt.ylabel(r"$q$")
plt.legend(fontsize=10)
plt.xlim(0.8, ks[-1] + ks[0] - 0.8)
# plt.show()
plt.savefig("_fig_calibration.pdf", dpi=300)
