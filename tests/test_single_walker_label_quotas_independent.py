"""Exact finite controls: label quotas never prescribe within-label phases.

This finite hard graph is a probability/control fixture, not a continuum
fluid--solid surrogate. No compiled kernel or Monte Carlo is run. The final
controls bind the ordinary Python quota/calibration helpers to this reference.
"""
from fractions import Fraction as F
from itertools import combinations
import math

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as tempering


VERTICES = tuple(range(4))
VOLUME = len(VERTICES)
HARD_EDGES = {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 3))}
NEIGHBORS = {0: (1, 3), 1: (0, 2, 3), 2: (1, 3), 3: (0, 1, 2)}
ACTIVITY = {0: F(2, 3), 1: F(2, 3), 2: F(1, 4), 3: F(3, 5)}
QUOTA = {0: F(3, 10), 1: F(3, 10), 2: F(1, 50), 3: F(19, 50)}


def hard(host):
    return not any(edge.issubset(host) for edge in HARD_EDGES)


def reference_weight(node, host, center):
    if node in (0, 1) and not hard(host):
        return F(0)
    if node == 1 and (center in host or not hard(host | {center})):
        return F(0)
    weight = ACTIVITY[node] ** len(host) / VOLUME
    if node == 3:
        overlaps = sum(edge.issubset(host) for edge in HARD_EDGES)
        # Rational penetrable-pair, category-order and number costs give a
        # nontrivial auxiliary law without any physical phase restriction.
        weight *= F(1, 5) ** overlaps
        weight *= F(1, 3) ** int(0 in host)
        weight *= F(1, 2) ** ((len(host) - 2) ** 2)
    return weight


def states_and_normalizers(cap=2):
    hosts = [frozenset(items) for n in range(cap + 1)
             for items in combinations(VERTICES, n)]
    states = [(node, host, center) for node in NEIGHBORS for host in hosts
              for center in VERTICES if reference_weight(node, host, center)]
    normalizers = {node: sum(reference_weight(*state) for state in states if state[0] == node)
                   for node in NEIGHBORS}
    assert all(value > 0 for value in normalizers.values())
    return states, normalizers


def normalized_law(states, node_weight):
    masses = {state: node_weight[state[0]] * reference_weight(*state) for state in states}
    total = sum(masses.values())
    return {state: mass / total for state, mass in masses.items()}


def gauge_update(weights, visited, quota, gain):
    updated = {node: weights[node] + gain * quota[node] - gain * (node == visited)
               for node in quota}
    origin = updated[2]
    return {node: value - origin for node, value in updated.items()}


@pytest.mark.parametrize('cap', [1, 2, 3])
def test_positive_quota_mean_field_root_and_unchanged_physical_conditionals(cap):
    states, z = states_and_normalizers(cap)
    optimum = {node: QUOTA[node] / z[node] for node in QUOTA}
    law = normalized_law(states, optimum)
    marginal = {node: sum(mass for state, mass in law.items() if state[0] == node)
                for node in QUOTA}
    assert marginal == QUOTA
    assert {node: QUOTA[node] - marginal[node] for node in QUOTA} == {node: F(0) for node in QUOTA}
    other_quota = {0: F(1, 10), 1: F(2, 5), 2: F(1, 10), 3: F(2, 5)}
    other_law = normalized_law(states, {node: other_quota[node] / z[node] for node in QUOTA})
    for state in states:
        node = state[0]
        conditional = reference_weight(*state) / z[node]
        assert law[state] / QUOTA[node] == conditional
        assert other_law[state] / other_quota[node] == conditional


def test_expected_gauge_fixed_update_is_zero_exactly_at_the_quota_root():
    weights = {0: F(7), 1: F(-3), 2: F(0), 3: F(5)}
    gain = F(3, 10)
    for node in QUOTA:
        expected = sum(QUOTA[visited] * gauge_update(weights, visited, QUOTA, gain)[node]
                       for visited in QUOTA)
        assert expected == weights[node]
    states, _ = states_and_normalizers()
    imperfect = normalized_law(states, {0: F(2), 1: F(3), 2: F(5), 3: F(7)})
    marginal = {node: sum(value for state, value in imperfect.items() if state[0] == node)
                for node in QUOTA}
    assert marginal != QUOTA
    drift = {node: gain * (QUOTA[node] - marginal[node]) for node in QUOTA}
    assert sum(drift.values()) == 0 and any(value != 0 for value in drift.values())


def test_uniform_quota_is_legacy_gauge_equivalent_before_intermediate_clipping():
    quota = {node: F(1, 4) for node in QUOTA}
    weights = {0: F(7, 10), 1: F(-3), 2: F(0), 3: F(5)}
    gain = F(3, 10)
    for visited in QUOTA:
        legacy = {node: value - gain * (node == visited) for node, value in weights.items()}
        origin = legacy[2]
        legacy = {node: value - origin for node, value in legacy.items()}
        assert gauge_update(weights, visited, quota, gain) == legacy


def test_hypothetical_premature_clipping_would_destroy_uniform_gauge_equivalence():
    # This is NOT the current driver: it gauges THEN clips at the declared
    # interval. It shows why a hypothetical clipping/gauge-order change cannot
    # be justified solely by uniform-quota gauge algebra. The real legacy
    # branch remains necessary for bitwise floating-point compatibility.
    weights, gain = [F(99, 100), F(0), F(0)], F(3, 10)
    clip = lambda value: max(F(-1), min(F(1), value))
    legacy = [clip(value - gain * (node == 1)) for node, value in enumerate(weights)]
    new = [clip(value + gain / 3 - gain * (node == 1)) for node, value in enumerate(weights)]
    new = [value - new[2] for value in new]
    assert legacy == [F(99, 100), F(-3, 10), F(0)]
    assert new == [F(9, 10), F(-3, 10), F(0)] and new != legacy


def compound_neighbor_rows(states, node_weight, cap):
    valid, rows = set(states), {}
    for state in states:
        node, host, center = state
        row = {}
        for other in NEIGHBORS[node]:
            chance = F(1, 2 * len(NEIGHBORS[node]))
            degree_ratio = F(len(NEIGHBORS[node]), len(NEIGHBORS[other]))
            if node == 0 and other == 1:
                if not host:
                    continue
                ratio = node_weight[1] / node_weight[0] * len(host) / (ACTIVITY[0] * VOLUME) * degree_ratio
                for tag in host:
                    destination = (1, host - {tag}, tag)
                    assert destination in valid
                    row[destination] = row.get(destination, F(0)) + chance / len(host) * min(F(1), ratio)
            elif node == 1 and other == 0:
                if len(host) >= cap:
                    continue
                ratio = node_weight[0] / node_weight[1] * ACTIVITY[0] * VOLUME / (len(host) + 1) * degree_ratio
                for newcenter in VERTICES:
                    destination = (0, host | {center}, newcenter)
                    assert destination in valid
                    row[destination] = row.get(destination, F(0)) + chance / VOLUME * min(F(1), ratio)
            else:
                destination = (other, host, center)
                if destination not in valid:
                    continue
                ratio = (node_weight[other] * reference_weight(*destination)
                         / (node_weight[node] * reference_weight(*state)) * degree_ratio)
                row[destination] = row.get(destination, F(0)) + chance * min(F(1), ratio)
        row[state] = F(1) - sum(row.values())
        rows[state] = row
    return rows


def test_quota_root_is_stationary_under_actual_kind_of_gibbs_tagged_neighbor_heatbath_composition():
    states, z = states_and_normalizers()
    node_weight = {node: QUOTA[node] / z[node] for node in QUOTA}
    law = normalized_law(states, node_weight)
    gibbs = {}
    for state in states:
        _, host, center = state
        scores = {(node, host, center): node_weight[node] * reference_weight(node, host, center)
                  for node in QUOTA if reference_weight(node, host, center)}
        total = sum(scores.values())
        gibbs[state] = {destination: mass / total for destination, mass in scores.items()}
    neighbor = compound_neighbor_rows(states, node_weight, 2)
    ideal_law = {state: reference_weight(*state) / z[2] for state in states if state[0] == 2}
    heatbath = {state: ideal_law if state[0] == 2 else {state: F(1)} for state in states}
    for transition in (gibbs, neighbor, heatbath):
        assert all(sum(row.values()) == 1 and all(value >= 0 for value in row.values())
                   for row in transition.values())
        propagated = {state: F(0) for state in states}
        for origin, mass in law.items():
            for destination, probability in transition[origin].items():
                propagated[destination] += mass * probability
        assert propagated == law
        law = propagated


def test_work_uses_observed_frozen_dwell_odds_not_nominal_quotas_when_weights_are_imperfect():
    states, z = states_and_normalizers()
    imperfect_weight = {0: F(2), 1: F(3), 2: F(5), 3: F(7)}
    law = normalized_law(states, imperfect_weight)
    base = sum(value for state, value in law.items() if state[0] == 0)
    cavity = sum(value for state, value in law.items() if state[0] == 1)
    assert base / cavity != QUOTA[0] / QUOTA[1]
    corrected = base / cavity * imperfect_weight[1] / imperfect_weight[0]
    assert corrected == z[0] / z[1]
    quota_substitution = QUOTA[0] / QUOTA[1] * imperfect_weight[1] / imperfect_weight[0]
    assert quota_substitution != z[0] / z[1]
    assert math.log(float(corrected)) == pytest.approx(math.log(float(z[0] / z[1])))
    # Expected full-clock cycle rewards obey renewal reward/Kac, not an
    # unweighted average of per-cycle work or conditional phase ratios.
    expected_base_cycle = base / sum(value for state, value in law.items() if state[0] == 2)
    expected_cavity_cycle = cavity / sum(value for state, value in law.items() if state[0] == 2)
    assert expected_base_cycle / expected_cavity_cycle == base / cavity


def test_label_quota_is_not_a_within_physical_node_category_probability():
    states, z = states_and_normalizers()
    category = sum(reference_weight(*state) for state in states
                   if state[0] == 0 and len(state[1]) >= 2) / z[0]
    assert category == F(4, 15) and category != QUOTA[0]
    for quota in (QUOTA, {0: F(1, 10), 1: F(2, 5), 2: F(1, 10), 3: F(2, 5)}):
        law = normalized_law(states, {node: quota[node] / z[node] for node in quota})
        numerator = sum(value for state, value in law.items() if state[0] == 0 and len(state[1]) >= 2)
        denominator = sum(value for state, value in law.items() if state[0] == 0)
        assert numerator / denominator == category


def weighted_flatness(counts, quota):
    scaled = [F(counts[node]) / quota[node] for node in quota]
    return min(scaled) / (sum(scaled) / len(scaled))


def test_weighted_flatness_retains_minimum_actual_visits_not_scaled_pseudocounts():
    desired = {0: 1500, 1: 1500, 2: 100, 3: 1900}
    assert weighted_flatness(desired, QUOTA) == 1
    assert min(desired.values()) == 100
    assert F(min(desired.values()), sum(desired.values()) // 4) < F(1, 2)
    under_sampled = {node: value // 10 for node, value in desired.items()}
    assert weighted_flatness(under_sampled, QUOTA) == 1
    assert min(under_sampled.values()) < 100  # Must still fail the stage.
    uniformly_flat = {node: 1000 for node in QUOTA}
    assert weighted_flatness(uniformly_flat, QUOTA) < F(1, 2)


def test_equal_half_means_alone_do_not_certify_conditional_distribution_stationarity():
    first, second = [(32, F(1, 5)), (32, F(4, 5))], [(32, F(1, 2))] * 2
    assert sum(n for n, _ in first) == sum(n for n, _ in second)
    assert sum(b for _, b in first) == sum(b for _, b in second)
    assert sum(b > F(45, 100) for _, b in first) != sum(b > F(45, 100) for _, b in second)
    # These are efficiency diagnostics only, under either quota policy.


def test_actual_endpoint_priority_quota_matches_declared_positive_label_law():
    actual = tempering.resolve_label_quota(162, 70, 71, 0, 'endpoint_priority')
    expected = np.full(162, .38 / 159)
    expected[[70, 71, 0]] = [.30, .30, .02]
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual > 0.)
    assert abs(actual.sum() - 1.) < 1e-14
    with pytest.raises(ValueError):
        tempering.resolve_label_quota(3, 0, 1, 2, 'endpoint_priority')


def actual_calibration_half(counts, drift=0.):
    labels = np.repeat(np.arange(4), counts)
    rows = np.zeros((len(labels), 12))
    rows[:, 1] = labels
    rows[:, 2] = np.where(labels == 0, 32. + drift, np.where(labels == 1, 31., 29.))
    rows[:, 3] = np.where(labels == 0, .7, np.where(labels == 1, .72, .2))
    return rows, labels.astype(np.int64)


@pytest.mark.parametrize('rare_actual_visits', [False, True])
def test_actual_weighted_flatness_keeps_raw_visits_and_paired_half_stability(rare_actual_visits):
    quota = np.array([float(QUOTA[node]) for node in range(4)])
    controller = tempering.StagedFlatnessCalibration(4, np.array([0, 1]), label_quota=quota)
    counts = [150, 150, 10, 190] if rare_actual_visits else [1500, 1500, 100, 1900]
    first, updated = actual_calibration_half(counts)
    weights = np.zeros(4)
    controller.observe(first, updated, len(first), weights)
    second, updated = actual_calibration_half(counts, drift=0. if rare_actual_visits else 1.)
    record = controller.observe(second, updated, len(first) + len(second), weights)
    assert record['flatness_minimum_to_mean'] == pytest.approx(1., abs=1e-14)
    assert not record['stage_passed']
    if rare_actual_visits:
        assert not record['minimum_visits_met']
        assert record['dense_half_stability_passed']
    else:
        assert record['minimum_visits_met'] and record['flatness_passed']
        assert not record['dense_half_stability_passed']
        comparison = next(item for item in record['dense_half_profiles'] if item['node'] == 0)
        assert comparison['mean_N_difference'] == 1.
