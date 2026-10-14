"""Finite periodic hard rods independently check the general 2d force factor.

Ring partition functions supply pressure. The independently ordered host
integral in the complementary open interval supplies the cavity work. No
sampler, cavity analysis script, or numerical EOS is imported.
"""

import math

import numpy as np
import pytest


@pytest.mark.parametrize("activity", [.2, 1., 7.3])
def test_finite_grand_pressure_predicts_particle_cavity_radius_force(activity):
    length = 5.3
    diameter = 1.
    maximum = math.floor(length / diameter)
    numbers = np.arange(1, maximum + 1)
    # Lambda=1. Ordered periodic rods have one unrestricted origin and
    # a simplex of N free interparticle gaps of total L-N*sigma.
    partition = np.array([length * (length - n * diameter)**(n - 1)
                          / math.factorial(n) for n in numbers])
    weights = activity**numbers * partition
    normalization = 1 + weights.sum()
    mean_number = weights @ numbers / normalization
    density = mean_number / length
    pressure = weights @ (1 / length + (numbers - 1) / (length - numbers * diameter))
    pressure /= normalization

    def cavity_normalization(radius):
        # Excluding a center interval of length 2R turns the allowed host
        # into an open interval. At R>=sigma/2 no wrap bond crosses the
        # empty interval, and subtracting ordered separations gives this
        # independent simplex integral.
        return 1 + sum(activity**n * max(length - 2 * radius - (n - 1) * diameter, 0.)**n
                       / math.factorial(n) for n in numbers)

    radius = diameter
    step = 1e-5
    derivative = -(math.log(cavity_normalization(radius + step))
                   - math.log(cavity_normalization(radius - step))) / (2 * step)
    # In dimension d, the angularly integrated hard virial has factor
    # 1/(2d). The SI three-dimensional result specializes to 2d=6.
    expected = 2 / diameter * (pressure / density - 1)
    assert derivative == pytest.approx(expected, rel=2e-9, abs=2e-10)
    assert derivative > 0  # positive expansion work, negative inward force
