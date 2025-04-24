# %%
"""QPE utility collection."""

from typing import (
    Optional,
    Union,
    Callable,
    Counter,
)
import numpy as np


def counts_to_lists(
    counts: Counter,
) -> tuple[list[list[int]], list[int]]:
    """Convert the shot counts to the lists of keys and values."""
    keys = []
    for k in counts.keys():
        v = tuple(int(i) for i in k)
        keys.append(v)
    vals = []
    for v in counts.values():
        vals.append(int(v))
    return (keys, vals)


def get_mu_and_sigma(
    prior: np.ndarray,
    phi: np.ndarray | None = None,
) -> tuple[float, float]:
    """Calculate the circular mean mu and square of Holevo variance sigma."""
    if phi is None:
        phi = np.linspace(0, 2, len(prior) + 1)[:-1]
    dphi = 2 / len(phi)
    # Circular mean.
    mu = np.angle(np.sum(np.exp(1.0j * np.pi * phi) * prior) * dphi * np.pi)
    mu /= np.pi
    # Holevo variance sigma^2.
    sigma2 = np.abs(np.sum(np.exp(1.0j * np.pi * phi) * prior) * dphi) ** -2 - 1
    sigma = np.sqrt(sigma2) / np.pi
    return mu, sigma


def binary_fraction(readout: list[int]) -> float:
    r"""Converts (BigEndian) bits into a fraction number [0, 2).

    Args:
       Readout: Bit string.

    Returns:
       Phase factor :math:`\phi` as a binary fraction.

    Examples:
        >>> binary_fraction([1, 0, 0])
        1.0
        >>> binary_fraction([0, 0, 1])
        0.25
    """
    phase = 0.0
    for i, r in enumerate(readout):
        phase += r * (0.5**i)
    return phase


def noise_aware_likelihood(
    k: int,
    beta: float,
    m: int,
    phi: Union[float, list],
    error_rate: Callable[[int], float] | None = None,
    phase_shift: Callable[[int], float] | None = None,
) -> Union[float, list]:
    """Likelihood function for the noiseless simulation.

    Args:
        k:
            Multiple of the controlled unitary.
        beta:
            Rotation angle (in half turn) applied before the X measurement.
        m:
            Measurement outcome in {0, 1}.
        phi:
            phase.
        error_rate:
            Function to return error rate [0, 1].

    Returns:
        Likelihood function as a function of :code:`phi`.
    """
    q = 0.0
    if error_rate is not None:
        q = error_rate(k)
    omega = 0.0
    if phase_shift is not None:
        omega = phase_shift(k)
    phi = np.array(phi)
    val = 1 + (1 - q) * (-1) ** m * np.cos(np.pi * (k * phi + beta - omega))
    val *= 0.5
    val = val.tolist()
    return val


def get_mock_backend(
    phis: list[float],
    amps: list[float],
    error_rate: Optional[Callable[[int], float]] = None,
    phase_shift: Optional[Callable[[int], float]] = None,
) -> Callable[[int, float], int]:
    """Return a mock backend function."""

    def mock_backend(k: int, beta: float) -> int:
        """Mock backend."""
        p0 = 0.0
        for phi, amp in zip(phis, amps):
            # Probability to measure 0.
            p0t = noise_aware_likelihood(
                k,
                beta,
                0,
                phi,
                error_rate=error_rate,
                phase_shift=phase_shift,
            )
            p0 += amp * p0t
        # Draw the measurement outcome m.
        m = 0
        if np.random.random() > p0:
            m = 1
        return m

    return mock_backend
