"""Independent exact geometry/algebra controls for the cavity-radius corollary.

No Markov chain or coexistence model is used.  The one-dimensional controls
are Haar translates of explicit hard point configurations; they check only
the dimension-independent small-ball counting identity.  Three-dimensional
controls check open-ball geometry and the thin-shell bound.  The tracer
test is a chain-rule identity, not evidence of fluid--solid coexistence.
"""

from fractions import Fraction as F
from itertools import combinations, product

import pytest


def circle_distance(a, b, length):
    delta = abs(a - b) % length
    return min(delta, length - delta)


def interval_union_length(points, radius, length):
    """Exact covered length of periodic intervals, independent of N*2R."""
    assert 0 < 2 * radius < length
    pieces = []
    for point in points:
        left, right = (point - radius) % length, (point + radius) % length
        if left < right:
            pieces.append((left, right))
        else:
            pieces.extend(((F(0), right), (left, length)))
    covered = F(0)
    current_left = current_right = None
    for left, right in sorted(pieces):
        if current_left is None:
            current_left, current_right = left, right
        elif left <= current_right:
            current_right = max(current_right, right)
        else:
            covered += current_right - current_left
            current_left, current_right = left, right
    if current_left is not None:
        covered += current_right - current_left
    return covered


@pytest.mark.parametrize("radius", [F(1, 10), F(1, 4), F(1, 2)])
def test_exact_haar_translate_void_probability_for_hard_circle_configurations(radius):
    length, sigma = F(5), F(1)
    configurations = [(), (F(0),), (F(0), F(1)),
                      tuple(F(i) for i in range(5)),
                      (F(1, 4), F(7, 4), F(13, 4)),
                      (F(1, 2), F(9, 2))]
    for points in configurations:
        assert all(circle_distance(a, b, length) >= sigma
                   for a, b in combinations(points, 2))
        covered = interval_union_length(points, radius, length)
        # Uniform common translation of all hosts is a stationary law.
        # Its cavity probability is one minus the exact covered fraction.
        probability = 1 - covered / length
        assert probability == 1 - F(len(points)) * 2 * radius / length


def test_exact_number_mixture_needs_expectation_not_an_empirical_average():
    length, radius = F(5), F(1, 2)
    configurations = [(), (F(0),), (F(0), F(1))]
    weights = [F(1, 7), F(2, 7), F(4, 7)]
    assert sum(weights) == 1
    expected_number = sum(weight * len(points)
                          for weight, points in zip(weights, configurations))
    probability = sum(weight * (1 - interval_union_length(points, radius, length) / length)
                      for weight, points in zip(weights, configurations))
    assert probability == 1 - expected_number * 2 * radius / length
    assert expected_number == F(10, 7)


def torus_distance_squared(a, b, length):
    return sum(circle_distance(x, y, length) ** 2 for x, y in zip(a, b))


def test_three_dimensional_open_ball_bound_is_sharp_at_half_diameter():
    length, sigma, radius = F(6), F(2), F(1)
    center = (F(0), F(0), F(0))
    points = [(F(1), F(0), F(0)), (F(5), F(0), F(0))]
    assert torus_distance_squared(*points, length) == sigma ** 2
    # Two touching hosts meet the closed-ball boundary, not its interior.
    assert sum(torus_distance_squared(point, center, length) < radius ** 2
               for point in points) == 0
    larger_radius = radius + F(1, 1000)
    assert sum(torus_distance_squared(point, center, length) < larger_radius ** 2
               for point in points) == 2


def test_every_pair_in_a_three_dimensional_half_diameter_open_ball_is_incompatible():
    radius, sigma = F(1, 2), F(1)
    coordinates = [F(i, 6) for i in range(-2, 3)]
    interior = [point for point in product(coordinates, repeat=3)
                if sum(x * x for x in point) < radius ** 2]
    for a, b in combinations(interior, 2):
        distance_squared = sum((x - y) ** 2 for x, y in zip(a, b))
        # Cauchy--Schwarz is an independent algebraic triangle bound.
        upper = 2 * sum(x * x + y * y for x, y in zip(a, b))
        assert distance_squared <= upper < 4 * radius ** 2 == sigma ** 2


def test_indicator_to_shell_count_error_has_the_factorial_second_moment_bound():
    # Pointwise inequality: n-1{n>0} <= n(n-1)/2.
    for n in range(101):
        assert n - int(n > 0) <= F(n * (n - 1), 2)
    probabilities = [F(1, 7), F(2, 7), F(3, 7), F(1, 7)]
    error = sum(p * (n - int(n > 0)) for n, p in enumerate(probabilities))
    bound = sum(p * F(n * (n - 1), 2) for n, p in enumerate(probabilities))
    assert error <= bound
    # v_h = (4*pi/3)*h*(3R^2+3Rh+h^2): after division by h,
    # the factorial bound is O(h).  Remove the common positive constants
    # to verify the polynomial identity exactly, without floating pi.
    radius = F(1)
    for k in range(1, 9):
        h = F(1, 10 ** k)
        polynomial = 3 * radius ** 2 + 3 * radius * h + h ** 2
        assert (radius + h) ** 3 - radius ** 3 == h * polynomial
        normalized_bound = h * polynomial ** 2
        assert normalized_bound <= 49 * h


@pytest.mark.parametrize("sigma", [F(1, 2), F(1), F(2)])
def test_additive_tracer_chain_rule_and_mole_fraction_normalization(sigma):
    # Algebraic contact data only: these numbers do not define a simulated
    # or experimentally realized coexistence model.
    rho_f, rho_s, beta_pressure = F(9, 10), F(11, 10), F(58, 5)
    contact_f = 6 * (beta_pressure - rho_f) / sigma
    contact_s = 6 * (beta_pressure - rho_s) / sigma
    a_prime = contact_s / rho_s - contact_f / rho_f
    assert a_prime == 6 * beta_pressure / sigma * (1 / rho_s - 1 / rho_f)
    assert a_prime < 0
    # R=(sigma+d)/2 and log K=log(rho_f/rho_s)-a(R).
    chain_rule_slope = -a_prime * F(1, 2)
    expected = 3 * beta_pressure / sigma * (1 / rho_f - 1 / rho_s)
    assert chain_rule_slope == expected > 0
    assert (rho_f / rho_s) * (rho_s / rho_f) == 1
    assert -a_prime != expected  # Missing the additive-contact 1/2 fails.
