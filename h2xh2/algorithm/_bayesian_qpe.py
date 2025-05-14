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

"""Bayesian QPE (including information theory QPE) utilities.

Grid-based bayesian QPE, not based on the more efficient Fourier respresentation
for the sake of simplicity.
"""

from typing import Callable
from pytket.backends.backendresult import BackendResult
import numpy as np
from ._utils import noise_aware_likelihood

PRECISION = 15
ATOL = 10**-PRECISION


def bayesian_update(
    phi: np.ndarray[float],
    prior: np.ndarray[float],
    k_list: list[int],
    beta_list: list[float],
    results: list[BackendResult],
    error_rate: Callable[[int], float] | None = None,
) -> np.ndarray[float]:
    """Bayesian update (high-level interface taking the BackendResults objects).

    Args:
        phi: Equi-distant grid points of the phase [0, 2), i.e., pytket conventions.
        prior: Prior distribution (not necesaliry normalized).
        k_list: list of k, the number of repeat of ctrl-U.
        beta_list: list of beta, the Rz rotation angle before the X measurement.
        results: list of Backend results
        error_rate: Error rate to be passed to the noise-aware likelihood.

    Returns:
        Posterior distribution (normalized).
    """
    (ks, betas, ms) = get_ms(k_list, beta_list, results)
    posterior = update(phi, prior, ks, betas, ms, error_rate=error_rate)
    return posterior


def get_ms(
    k_list: list[int],
    beta_list: list[float],
    results: list[BackendResult],
) -> tuple[list[int], list[float], list[int]]:
    """Create a list of (k, beta, m).

    Args:
        k_list: list of k, the number of repeat of ctrl-U.
        beta_list: list of beta, the Rz rotation angle before the X measurement.
        results: list of Backend results

    Returns:
        List of number-of-shots-aware list of (k, beta, m).
    """
    ks: list[int] = []
    betas: list[int] = []
    ms: list[int] = []
    for k, beta, result in zip(k_list, beta_list, results):
        counts = result.get_counts()
        for readout, count in counts.items():
            for _ in range(count):
                if readout == (0,):
                    m = 0
                else:
                    m = 1
                ks.append(k)
                betas.append(beta)
                ms.append(m)
    return (ks, betas, ms)


def bootstrap_sampling(
    phi: list[float],
    ks: list[int],
    betas: list[float],
    ms: list[int],
    error_rate: Callable[[int], float] | None = None,
    b: int = 1000,
) -> tuple[float, float]:
    """Bootstrap resampling method."""
    phi_tilde: list[float] = []
    ilist = list(range(len(ks)))
    prior = np.ones_like(phi)
    for ib in range(b):
        indices = np.random.choice(ilist, size=len(ilist))
        ks_b = [ks[i] for i in indices]
        betas_b = [betas[i] for i in indices]
        ms_b = [ms[i] for i in indices]
        posterior = update(
            phi,
            prior,
            ks_b,
            betas_b,
            ms_b,
            error_rate=error_rate,
        )
        ii = np.argmax(posterior)
        phi_tilde.append(phi[ii])
    phi_tilde = np.array(phi_tilde)
    # Use circular mean and Holevo variance for the numerical stability.
    phi_mu = np.angle(np.average(np.exp(1j * np.pi * phi_tilde))) / np.pi
    holevo = np.abs(np.average(np.exp(1j * np.pi * phi_tilde))) ** -2 - 1
    phi_sigma = np.sqrt(holevo) / np.pi
    return phi_mu, phi_sigma


def update_log(
    phi: np.ndarray[float],
    prior: np.ndarray[float],
    ks: list[int],
    betas: list[float],
    ms: list[int | None],
    error_rate: Callable[[int], float] | None = None,
) -> np.ndarray[float]:
    """Update the probability distribution.

    Args:
        phi: Equi-distant grid points of the phase [0, 2), i.e., pytket conventions.
        prior: Prior distribution (not necesaliry normalized).
        k_list: list of k, the number of repeat of ctrl-U.
        beta_list: list of beta, the Rz rotation angle before the X measurement.
        ms: list of measurement outcomes {0, 1}.
        error_rate: Error rate to be passed to the noise-aware likelihood.

    Returns:
        Posterior distribution (normalized).
    """
    log_prior = np.log(prior)
    for k, beta, m in zip(ks, betas, ms):
        if m is None:
            continue
        likelihood = noise_aware_likelihood(
            k=k,
            beta=beta,
            m=m,
            phi=phi,
            error_rate=error_rate,
        )
        log_prior += np.log(
            np.maximum(likelihood, ATOL),
        )
    return log_prior


def update(
    phi: np.ndarray[float],
    prior: np.ndarray[float],
    ks: list[int],
    betas: list[float],
    ms: list[int | None],
    error_rate: Callable[[int], float] | None = None,
) -> np.ndarray[float]:
    """Update the probability distribution.

    Args:
        phi: Equi-distant grid points of the phase [0, 2), i.e., pytket conventions.
        prior: Prior distribution (not necesaliry normalized).
        k_list: list of k, the number of repeat of ctrl-U.
        beta_list: list of beta, the Rz rotation angle before the X measurement.
        ms: list of measurement outcomes {0, 1}.
        error_rate: Error rate to be passed to the noise-aware likelihood.

    Returns:
        Posterior distribution (normalized).
    """
    log_prior = update_log(phi, prior, ks, betas, ms, error_rate)
    log_prior -= np.max(log_prior)
    posterior = np.exp(log_prior)
    # Normalize the posterior distribution.
    dphi = phi[1] - phi[0]
    posterior /= np.sum(posterior) * dphi
    posterior = np.round(posterior, PRECISION)
    return posterior


def generate_ks(
    k_max: int,
    n_samples: int,
    error_rate: Callable[[int], float] | None = None,
    discard_rate: Callable[[int], float] | None = None,
) -> list[int]:
    ks = list(range(1, k_max + 1))
    if error_rate is None:
        dist = np.ones(len(ks), dtype=float)
    else:
        dist = np.array([1 / (1 - error_rate(k)) for k in ks])
    if discard_rate is None:
        dist *= np.ones(len(ks), dtype=float)
    else:
        dist *= np.array([1 / (1 - discard_rate(k)) for k in ks])
    dist /= np.sum(dist)
    k_list = np.random.choice(ks, size=n_samples, replace=True, p=dist)
    k_list = [int(k) for k in k_list]
    return k_list
