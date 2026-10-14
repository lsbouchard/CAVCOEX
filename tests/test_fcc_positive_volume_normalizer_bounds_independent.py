"""Pure exact controls for OPTIONAL positive-volume normalizer lower bounds.

There is no simulation, phase classification or production projection here.
The FCC reference defines a measurable subset of an unrestricted target, not
a fixed-phase model. Rational geometry and Taylor bounds certify safe decimals.
"""
import itertools
import math
from fractions import Fraction as F


def exponential_bounds(value, terms=45):
    """Rigorous rational lower/upper bounds for exp(value), value>=0."""
    value = F(value)
    assert 0 <= value < terms + 2
    term = F(1)
    total = term
    for index in range(1, terms + 1):
        term *= value / index
        total += term
    next_term = term * value / (terms + 1)
    # Subsequent positive-term ratios are no larger than value/(terms+2).
    return total, total + next_term / (1 - value / (terms + 2))


def test_exact_periodic_fcc_geometry_has_positive_margin_for_delta_one_twentieth():
    volume, sigma, delta = F(19281, 625), F(1), F(1, 20)
    basis = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    positions = [tuple(F(2 * cell[axis] + offset[axis], 4) for axis in range(3))
                 for cell in itertools.product(range(2), repeat=3) for offset in basis]
    normalized_minimum = min(sum((min(abs(first[k] - second[k]),
                                      1 - abs(first[k] - second[k])) ** 2
                                for k in range(3)), F(0))
                             for first, second in itertools.combinations(positions, 2))
    assert len(positions) == len(set(positions)) == 32
    assert normalized_minimum == F(1, 8)  # d_min^2=L^2/8
    # d_min>sigma+2delta, certified without numerical cube roots or sqrt(2).
    assert volume ** 2 > 512 * (sigma + 2 * delta) ** 6
    # Balls are embedded/disjoint on the torus, also certified rationally.
    assert (2 * delta) ** 3 < volume


def test_disjoint_site_cells_cancel_factorial_and_cavity_tags_include_normalized_center_measure():
    sites, cells_per_site, activity = 3, 2, F(7, 5)
    bins = [set(range(index * cells_per_site, (index + 1) * cells_per_site))
            for index in range(sites)]
    points = list(range(sites * cells_per_site))
    volume = len(points)
    ordered_full = [coordinates for coordinates in itertools.permutations(points, sites)
                    if all(sum(point in cell for point in coordinates) == 1 for cell in bins)]
    assert len(ordered_full) == math.factorial(sites) * cells_per_site ** sites
    subset_weight = F(len(ordered_full), math.factorial(sites)) * activity ** sites
    assert subset_weight == (activity * cells_per_site) ** sites
    marked_count = 0
    for tag_cell in bins:
        remaining = [cell for cell in bins if cell != tag_cell]
        for center in tag_cell:
            marked_count += sum(1 for _ in itertools.product(*remaining))
    cavity_subset_weight = F(marked_count, volume) * activity ** (sites - 1)
    assert cavity_subset_weight == sites * activity ** (sites - 1) * cells_per_site ** sites / volume
    # Missing the Haar 1/V or inserting an extra sector factorial is incorrect.
    assert cavity_subset_weight != marked_count * activity ** (sites - 1)
    assert subset_weight != subset_weight / math.factorial(sites)


def test_safe_decimal_lower_bounds_are_certified_by_rational_directed_inequalities():
    # pi>3.14159 is a conservative established lower bound. With delta=1/20,
    # v_delta=pi/6000>314159/600000000. No optimistic binary log is needed.
    ball_lower = F(314159, 600000000)
    assert exponential_bounds(F(75548, 10000))[0] > 1 / ball_lower
    # Thus log(v_delta)>-7.5548 and log Xi0>32*(16-7.5548)=270.2464.
    log_base_lower = 32 * (F(16) - F(75548, 10000))
    assert log_base_lower > F(270246, 1000)
    volume = F(19281, 625)
    assert exponential_bounds(F(342913, 100000))[0] > volume  # log V<3.42913
    assert exponential_bounds(F(346573, 100000))[1] < 32     # log32>3.46573
    log_cavity_lower = log_base_lower - 16 - F(342913, 100000) + F(346573, 100000)
    assert log_cavity_lower == F(254283, 1000)
    # All component inequalities are strict, so log Xi_sigma>254.283.


def test_exact_ideal_reference_supplies_conservative_absolute_root_bounds_not_phase_odds():
    # Full ideal log Xi=zeta_I V bounds its cap80 counterpart from above.
    ideal_density = 1 / F(6653, 6250)  # 1/1.06448
    volume, quota_ideal, quota_physical = F(19281, 625), F(1, 50), F(3, 10)
    ideal_log_upper = ideal_density * volume
    assert ideal_log_upper < F(28980911, 1000000)
    assert exponential_bounds(F(3912024, 1000000))[0] > 1 / quota_ideal
    root_gauge_upper = ideal_log_upper + F(3912024, 1000000)
    assert root_gauge_upper < F(32893, 1000)
    assert exponential_bounds(F(12039, 10000))[1] < 1 / quota_physical
    # b_I=0: C=log Xi_I-log tau_I, and b_j=C-log Xi_j+log tau_j.
    base_weight_upper = F(32893, 1000) - F(270246, 1000) - F(12039, 10000)
    cavity_weight_upper = F(32893, 1000) - F(254283, 1000) - F(12039, 10000)
    assert base_weight_upper == F(-2385569, 10000)
    assert cavity_weight_upper == F(-2225939, 10000)
    assert F(-226234, 1000) > base_weight_upper
    assert F(-211145, 1000) > cavity_weight_upper
    # These exclude an efficiency guess as a desired quota-normalizing root.
    # They do not reject the arbitrary finite weights as legal frozen targets.
