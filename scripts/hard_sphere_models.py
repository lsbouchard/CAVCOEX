"""Deterministic thermodynamic models used by the manuscript benchmarks.

The three-dimensional finite-size calculations in this module are explicitly
EOS-reweighted calculations, not particle-resolved Monte Carlo simulations.
They use a Percus--Yevick compressibility-route canonical free energy and the
consistent scaled-particle insertion polynomial.  This makes the numerical
probability identities testable at finite volume while keeping the modelling
assumption visible in one place.

Reduced units are used throughout: beta = sigma = Lambda = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

PI_OVER_SIX = np.pi / 6.0
ETA_CLOSE_PACKED = np.pi / (3.0 * np.sqrt(2.0))

# Speedy's empirical fcc-solid EOS,
# Z = 3/(1-x) - A (x-B)/(x-C), x = eta/eta_cp.  The first
# coefficients are those published by Speedy; the second set is the
# simulation-data refit reported by Bannerman et al.  Keeping the sets named
# prevents a refit from being mistaken for an independent functional form.
SPEEDY_1998_COEFFICIENTS = (0.5921, 0.7072, 0.601)
BANNERMAN_2010_COEFFICIENTS = (0.620735, 0.708194, 0.591663)


def number_density(eta: np.ndarray | float) -> np.ndarray | float:
    """Return rho sigma^3 for a packing fraction eta."""

    return np.asarray(eta) / PI_OVER_SIX


def packing_fraction(rho: np.ndarray | float) -> np.ndarray | float:
    """Return eta for a reduced number density rho sigma^3."""

    return PI_OVER_SIX * np.asarray(rho)


def cs_z(eta: np.ndarray | float) -> np.ndarray:
    """Carnahan--Starling compressibility factor."""

    eta = np.asarray(eta, dtype=float)
    return (1.0 + eta + eta**2 - eta**3) / (1.0 - eta) ** 3


def cs_a_ex(eta: np.ndarray | float) -> np.ndarray:
    """Carnahan--Starling excess Helmholtz free energy per particle."""

    eta = np.asarray(eta, dtype=float)
    return (4.0 * eta - 3.0 * eta**2) / (1.0 - eta) ** 2


def cs_mu_ex(eta: np.ndarray | float) -> np.ndarray:
    """Carnahan--Starling excess chemical potential."""

    eta = np.asarray(eta, dtype=float)
    return (8.0 * eta - 9.0 * eta**2 + 3.0 * eta**3) / (1.0 - eta) ** 3


def cs_s0(eta: np.ndarray | float) -> np.ndarray:
    """Carnahan--Starling zero-wavevector structure factor."""

    eta = np.asarray(eta, dtype=float)
    numerator = (1.0 - eta) ** 4
    denominator = 1.0 + 4.0 * eta + 4.0 * eta**2 - 4.0 * eta**3 + eta**4
    return numerator / denominator


def speedy_solid_z(
    eta: np.ndarray | float,
    coefficients: tuple[float, float, float] = SPEEDY_1998_COEFFICIENTS,
) -> np.ndarray:
    """Return the Speedy empirical fcc-solid compressibility factor.

    Parameters are ``(A, B, C)`` in

    ``Z = 3/(1-x) - A (x-B)/(x-C)``, ``x = eta/eta_cp``.

    The expression is intended for the crystalline branch, not for the fluid
    or the mechanically unstable continuation of the crystal.
    """

    eta = np.asarray(eta, dtype=float)
    x = eta / ETA_CLOSE_PACKED
    if np.any(x >= 1.0):
        raise ValueError("The fcc-solid EOS requires eta < eta_close_packed")
    a, b, c = coefficients
    if np.any(np.isclose(x, c)):
        raise ValueError("The requested density lies on the empirical EOS pole")
    return 3.0 / (1.0 - x) - a * (x - b) / (x - c)


def speedy_solid_s0(
    eta: np.ndarray | float,
    coefficients: tuple[float, float, float] = SPEEDY_1998_COEFFICIENTS,
) -> np.ndarray:
    """Return ``[d(beta p)/d rho]^{-1}`` from the Speedy fcc EOS."""

    eta = np.asarray(eta, dtype=float)
    x = eta / ETA_CLOSE_PACKED
    z = speedy_solid_z(eta, coefficients)
    a, b, c = coefficients
    dz_dx = 3.0 / (1.0 - x) ** 2 - a * (b - c) / (x - c) ** 2
    return 1.0 / (z + x * dz_dx)


def pyc_z(eta: np.ndarray | float) -> np.ndarray:
    """Percus--Yevick compressibility-route compressibility factor."""

    eta = np.asarray(eta, dtype=float)
    return (1.0 + eta + eta**2) / (1.0 - eta) ** 3


def pyc_a_ex(eta: np.ndarray | float) -> np.ndarray:
    """PY-compressibility excess Helmholtz free energy per particle."""

    eta = np.asarray(eta, dtype=float)
    return -np.log1p(-eta) + 1.5 * eta * (2.0 - eta) / (1.0 - eta) ** 2


def pyc_mu_ex(eta: np.ndarray | float) -> np.ndarray:
    """PY-compressibility excess chemical potential."""

    eta = np.asarray(eta, dtype=float)
    return pyc_a_ex(eta) + pyc_z(eta) - 1.0


def pyc_s0(eta: np.ndarray | float) -> np.ndarray:
    """PY-compressibility S(0), differentiated from eta Z(eta)."""

    eta = np.asarray(eta, dtype=float)
    # d[eta Z]/d eta = (1 + 4 eta + 4 eta^2) / (1 - eta)^4.
    return (1.0 - eta) ** 4 / (1.0 + 4.0 * eta + 4.0 * eta**2)


def spt_cavity_work(eta: np.ndarray | float, r_over_sigma: float) -> np.ndarray:
    """Scaled-particle work for a center-exclusion radius R.

    The solute-to-host diameter ratio is q = 2 R/sigma - 1.  The expression
    has the exact point-particle coefficients c0 and c1 and uses the
    Percus--Yevick compressibility-route pressure for c3.  At R/sigma = 1 it
    reduces to the PY-compressibility excess chemical potential.
    """

    eta = np.asarray(eta, dtype=float)
    q = 2.0 * float(r_over_sigma) - 1.0
    if q < 0.0:
        raise ValueError("The scaled-particle expression requires R/sigma >= 0.5")
    t = eta / (1.0 - eta)
    c0 = -np.log1p(-eta)
    c1 = 3.0 * t
    c2 = 3.0 * t + 4.5 * t**2
    c3 = eta * pyc_z(eta)
    return c0 + c1 * q + c2 * q**2 + c3 * q**3


def spt_density_derivatives(
    eta: float, r_over_sigma: float, step: float = 2.0e-4
) -> tuple[float, float]:
    """Return d w/d rho and d^2 w/d rho^2 by a five-point stencil."""

    if eta <= 2.0 * step or eta >= 1.0 - 2.0 * step:
        raise ValueError("eta is too close to a stencil boundary")
    offsets = np.arange(-2, 3, dtype=float)
    values = spt_cavity_work(eta + step * offsets, r_over_sigma)
    d_eta = (values[0] - 8.0 * values[1] + 8.0 * values[3] - values[4]) / (
        12.0 * step
    )
    d2_eta = (
        -values[0]
        + 16.0 * values[1]
        - 30.0 * values[2]
        + 16.0 * values[3]
        - values[4]
    ) / (12.0 * step**2)
    return PI_OVER_SIX * d_eta, PI_OVER_SIX**2 * d2_eta


def finite_difference_weights(offsets: np.ndarray, derivative_order: int) -> np.ndarray:
    """Return polynomial finite-difference weights for unit-spaced offsets."""

    offsets = np.asarray(offsets, dtype=float)
    if derivative_order < 0 or derivative_order >= offsets.size:
        raise ValueError("derivative_order must be smaller than the stencil size")
    matrix = np.vstack([offsets**power for power in range(offsets.size)])
    rhs = np.zeros(offsets.size)
    rhs[derivative_order] = factorial(derivative_order)
    return np.linalg.solve(matrix, rhs)


def cumulants_from_distribution(
    values: np.ndarray, probability: np.ndarray, max_order: int = 4
) -> np.ndarray:
    """Return cumulants kappa_1,...,kappa_max_order for a discrete law."""

    values = np.asarray(values, dtype=float)
    probability = np.asarray(probability, dtype=float)
    probability = probability / probability.sum()
    mean = float(np.dot(probability, values))
    centered = values - mean
    central = {
        order: float(np.dot(probability, centered**order))
        for order in range(2, max(4, max_order) + 1)
    }
    cumulants = [mean]
    if max_order >= 2:
        cumulants.append(central[2])
    if max_order >= 3:
        cumulants.append(central[3])
    if max_order >= 4:
        cumulants.append(central[4] - 3.0 * central[2] ** 2)
    if max_order > 4:
        raise NotImplementedError("Cumulants above fourth order are not needed here")
    return np.asarray(cumulants, dtype=float)


@dataclass(frozen=True)
class GrandDistribution:
    n: np.ndarray
    probability: np.ndarray
    log_partition: float
    mean: float
    variance: float


def _three_dimensional_support(volume: float, target_n: int | None = None) -> np.ndarray:
    """Choose a conservative support below eta=0.68 for EOS reweighting."""

    upper_from_eta = int(np.floor(0.68 * volume / PI_OVER_SIX))
    if target_n is None:
        upper = upper_from_eta
    else:
        upper = min(upper_from_eta, max(int(1.80 * target_n) + 200, target_n + 300))
    return np.arange(0, max(upper, 1) + 1, dtype=int)


def pyc_canonical_log_q(n: np.ndarray, volume: float) -> np.ndarray:
    """Approximate ln Q_N from the thermodynamic PY-c free energy."""

    n = np.asarray(n, dtype=int)
    result = np.full(n.shape, -np.inf, dtype=float)
    result[n == 0] = 0.0
    positive = n > 0
    eta = PI_OVER_SIX * n[positive] / volume
    valid = eta < 0.70
    pos_indices = np.flatnonzero(positive)
    selected = pos_indices[valid]
    n_selected = n[selected].astype(float)
    eta_selected = PI_OVER_SIX * n_selected / volume
    result[selected] = (
        n_selected * np.log(volume)
        - gammaln(n_selected + 1.0)
        - n_selected * pyc_a_ex(eta_selected)
    )
    return result


def grand_distribution_3d(
    x: float, volume: float, target_n: int | None = None
) -> GrandDistribution:
    """Return the EOS-reweighted grand particle-number distribution."""

    n = _three_dimensional_support(volume, target_n)
    log_weight = x * n + pyc_canonical_log_q(n, volume)
    log_partition = float(logsumexp(log_weight))
    probability = np.exp(log_weight - log_partition)
    mean = float(np.dot(probability, n))
    variance = float(np.dot(probability, (n - mean) ** 2))
    tail_mass = float(probability[-8:].sum())
    if tail_mass > 1.0e-12:
        raise RuntimeError(
            f"Particle-number support is too short (upper-tail mass={tail_mass:.3e})"
        )
    return GrandDistribution(n, probability, log_partition, mean, variance)


def tune_activity_for_mean(target_n: int, volume: float, eta_guess: float) -> float:
    """Find x=beta mu such that the EOS-reweighted mean equals target_n."""

    rho_guess = float(number_density(eta_guess))
    center = np.log(rho_guess) + float(pyc_mu_ex(eta_guess))

    def residual(x: float) -> float:
        return grand_distribution_3d(x, volume, target_n).mean - target_n

    # The thermodynamic estimate is already close; a narrow bracket also keeps
    # trial activities away from the artificial high-density support cutoff.
    return float(brentq(residual, center - 0.25, center + 0.25, xtol=2.0e-13))


def cavity_conditioned_distribution(
    base: GrandDistribution, volume: float, r_over_sigma: float
) -> tuple[np.ndarray, float]:
    """Return P_R(N) and beta W_R from a base grand distribution."""

    eta_n = PI_OVER_SIX * base.n / volume
    work_n = spt_cavity_work(eta_n, r_over_sigma)
    log_probability = np.full(base.probability.shape, -np.inf, dtype=float)
    positive = base.probability > 0.0
    log_probability[positive] = np.log(base.probability[positive])
    log_terms = log_probability - work_n
    log_cavity_probability = float(logsumexp(log_terms))
    conditioned = np.exp(log_terms - log_cavity_probability)
    return conditioned, -log_cavity_probability


def cavity_work_at_activity(
    x: float, volume: float, r_over_sigma: float, target_n: int | None = None
) -> float:
    """Return the finite-volume EOS-reweighted cavity work at activity x."""

    base = grand_distribution_3d(x, volume, target_n)
    _, work = cavity_conditioned_distribution(base, volume, r_over_sigma)
    return work


def tonks_log_q(n: np.ndarray, length: float) -> np.ndarray:
    """Exact configurational ln Q_N for unit hard rods on a periodic ring."""

    n = np.asarray(n, dtype=int)
    result = np.full(n.shape, -np.inf, dtype=float)
    result[n == 0] = 0.0
    positive = (n > 0) & (length - n > 0.0)
    n_positive = n[positive].astype(float)
    result[positive] = (
        np.log(length)
        + (n_positive - 1.0) * np.log(length - n_positive)
        - gammaln(n_positive + 1.0)
    )
    return result


def tonks_insertion_probability(n: int, length: float) -> float:
    """Exact canonical insertion probability for one more unit hard rod."""

    if n < 0:
        raise ValueError("n must be nonnegative")
    log_q = tonks_log_q(np.asarray([n, n + 1]), length)
    if not np.isfinite(log_q[1]):
        return 0.0
    return float((n + 1.0) / length * np.exp(log_q[1] - log_q[0]))


def tonks_grand_distribution(x: float, length: float) -> GrandDistribution:
    """Exact grand distribution of periodic unit hard rods."""

    n = np.arange(0, int(np.floor(length - 1.0e-12)) + 1, dtype=int)
    log_weight = x * n + tonks_log_q(n, length)
    log_partition = float(logsumexp(log_weight))
    probability = np.exp(log_weight - log_partition)
    mean = float(np.dot(probability, n))
    variance = float(np.dot(probability, (n - mean) ** 2))
    return GrandDistribution(n, probability, log_partition, mean, variance)
