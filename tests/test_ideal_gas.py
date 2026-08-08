import math

import numpy as np


def test_ideal_gas_cavity_mixture_and_nucleation_theorem() -> None:
    """A fixed empty subvolume of an ideal gas has exactly Poisson work."""

    activity = 0.37
    volume = 19.0
    cavity_volume = 2.3
    mean_n = activity * volume
    n = np.arange(0, 100)
    probability = np.exp(
        -mean_n
        + n * np.log(mean_n)
        - np.asarray([math.lgamma(int(value) + 1) for value in n])
    )
    canonical_cavity_probability = (1.0 - cavity_volume / volume) ** n
    grand_cavity_probability = float(
        np.dot(probability, canonical_cavity_probability)
    )

    np.testing.assert_allclose(
        grand_cavity_probability,
        np.exp(-activity * cavity_volume),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    beta_work = -np.log(grand_cavity_probability)
    delta_n = -activity * cavity_volume
    np.testing.assert_allclose(beta_work, -delta_n, rtol=2.0e-14, atol=2.0e-14)


def test_zero_particle_sector_always_satisfies_a_cavity() -> None:
    for fraction in (0.0, 0.1, 0.9, 1.0):
        assert (1.0 - fraction) ** 0 == 1.0
