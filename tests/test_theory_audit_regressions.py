"""Small exact models for the mathematical qualifications in the 2026 audit."""

import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hard_sphere_models import (  # noqa: E402
    tonks_grand_distribution,
    tonks_insertion_probability,
)


def test_tonks_insertion_enforces_next_sector_support() -> None:
    # The present sector exists, but another rod cannot be inserted.
    for length, number in ((5.5, 5), (6.0, 5), (9.25, 9)):
        assert number < length
        assert tonks_insertion_probability(number, length) == 0.0

    # On an insertion-supported sector the positive-part formula agrees
    # with the independent partition-function-ratio implementation.
    for length, number in ((5.5, 1), (5.5, 4), (9.25, 8)):
        available = max(length - number - 1.0, 0.0) ** number
        expected = available / (length * (length - number) ** (number - 1))
        np.testing.assert_allclose(
            tonks_insertion_probability(number, length), expected, rtol=2e-14
        )


def test_soft_bias_temperature_derivatives_use_each_ensemble_energy() -> None:
    host_energy = np.array([0.0, 0.4, 1.6, 2.3])
    bias_energy = np.array([0.1, 0.8, 0.3, 0.9])
    number = np.array([0.0, 1.0, 2.0, 3.0])
    chemical_potential = 0.2
    beta = 1.2
    grand_energy_0 = host_energy - chemical_potential * number
    grand_energy_c = host_energy + bias_energy - chemical_potential * number

    def work(at_beta: float) -> float:
        return float(
            logsumexp(-at_beta * grand_energy_0)
            - logsumexp(-at_beta * grand_energy_c)
        )

    p0 = np.exp(-beta * grand_energy_0 - logsumexp(-beta * grand_energy_0))
    pc = np.exp(-beta * grand_energy_c - logsumexp(-beta * grand_energy_c))
    mean0 = np.dot(p0, grand_energy_0)
    meanc = np.dot(pc, grand_energy_c)
    h = 1e-4
    first = (work(beta + h) - work(beta - h)) / (2 * h)
    second = (work(beta + h) - 2 * work(beta) + work(beta - h)) / h**2
    np.testing.assert_allclose(first, meanc - mean0, rtol=2e-8, atol=2e-10)
    expected_second = np.dot(p0, (grand_energy_0 - mean0) ** 2) - np.dot(
        pc, (grand_energy_c - meanc) ** 2
    )
    np.testing.assert_allclose(second, expected_second, rtol=2e-5, atol=1e-7)

    # Omitting the physical bias energy is a distinguishable wrong answer.
    host_only = np.dot(pc - p0, grand_energy_0)
    assert abs(first - host_only) > 0.1


def test_finite_basin_insertion_derivative_keeps_discrepancy() -> None:
    # An even-N basin deliberately changes label under every insertion.
    # The exact finite-basin formula must work without compatibility.
    length = 7.25
    x = 1.4

    def basin(at_x: float) -> tuple[float, float, float, float, float]:
        grand = tonks_grand_distribution(at_x, length)
        selected = grand.n.astype(int) % 2 == 0
        n = grand.n[selected]
        probability = grand.probability[selected]
        probability = probability / probability.sum()
        insertion = np.array(
            [tonks_insertion_probability(int(value), length) for value in n]
        )
        p = float(np.dot(probability, insertion))
        mean = float(np.dot(probability, n))
        variance = float(np.dot(probability, (n - mean) ** 2))
        direct_response = float(np.dot(probability * insertion, n) / p - mean)
        rho = mean / length
        discrepancy = p - rho * np.exp(-at_x)
        return rho, variance / length, p, discrepancy, direct_response

    rho, rho_prime, _, discrepancy, direct = basin(x)
    step = 1e-5
    discrepancy_prime = (basin(x + step)[3] - basin(x - step)[3]) / (2 * step)
    corrected = (rho_prime - rho + np.exp(x) * discrepancy_prime) / (
        rho + np.exp(x) * discrepancy
    )
    np.testing.assert_allclose(corrected, direct, rtol=2e-8, atol=2e-9)
    assert abs(direct - (rho_prime / rho - 1.0)) > 1e-3
