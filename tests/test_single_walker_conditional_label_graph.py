"""Exact finite-state controls for a conditional node-label Gibbs update.

This is an independently enumerated probability model, not continuum sampling
or coexistence evidence. Rational order/number penalties make every stationary
mass calculation exact. Tagged moves alter host number, order and center, so
the Gibbs conditional must use the post-move configuration.
"""
from fractions import Fraction as F
from itertools import combinations

import pytest


VERTICES = tuple(range(4))
VOLUME = len(VERTICES)
CAP = 2
NODES = tuple(range(5))
ACTIVITY = (F(2, 3), F(2, 3), F(1, 4), F(3, 2), F(1, 5))
NODE_WEIGHT = (F(2), F(3), F(5), F(7), F(11))
# 0: hard base; 1: matched cavity; 2: soft/order/number auxiliary;
# 3: hard node at a different activity; 4: ideal auxiliary.
GRAPH = {0: (1, 2, 3), 1: (0, 4), 2: (0, 4), 3: (0,), 4: (1, 2)}


def _features(host, center):
    n = len(host)
    order = abs(sum(1 if site % 2 == 0 else -1 for site in host))
    pairs = sum(abs(first - second) == 1 for first, second in combinations(host, 2))
    minimum = min((abs(site - center) for site in host), default=VOLUME)
    return n, order, pairs, minimum


def _feature_weight(node, features):
    n, order, pairs, minimum = features
    if node in (0, 1, 3) and pairs:
        return F(0)
    if node == 1 and n and minimum <= 1:
        return F(0)
    value = NODE_WEIGHT[node] * ACTIVITY[node] ** n / VOLUME
    if node == 2:
        value *= F(1, 3) ** pairs
        value *= F(1, 2) ** (order - 1) ** 2
        value *= F(2, 3) ** (n - 2) ** 2
    return value


def _weight(state):
    node, host, center = state
    return _feature_weight(node, _features(host, center))


def _states():
    hosts = [frozenset(points) for n in range(CAP + 1)
             for points in combinations(VERTICES, n)]
    return [(node, host, center) for node in NODES for host in hosts
            for center in VERTICES if _weight((node, host, center))]


def _gibbs_row(state, feature_override=None, degree_bias=False):
    _, host, center = state
    features = _features(host, center) if feature_override is None else feature_override
    weights = { (node, host, center): _feature_weight(node, features)
                for node in NODES }
    if degree_bias:
        weights = {destination: mass * len(GRAPH[destination[0]])
                   for destination, mass in weights.items()}
    total = sum(weights.values())
    assert total > 0
    return {destination: mass / total for destination, mass in weights.items() if mass}


def _gibbs_rows(states, degree_bias=False):
    return {state: _gibbs_row(state, degree_bias=degree_bias) for state in states}


def _hop_rows(states):
    valid = set(states)
    rows = {}
    for origin in states:
        node, host, center = origin
        row = {}
        for other in GRAPH[node]:
            proposal = F(1, 2 * len(GRAPH[node]))
            degree_ratio = F(len(GRAPH[node]), len(GRAPH[other]))
            if (node, other) == (0, 1) and host:
                ratio = (NODE_WEIGHT[other] / NODE_WEIGHT[node] * len(host)
                         / (ACTIVITY[node] * VOLUME) * degree_ratio)
                for tag in host:
                    destination = (other, host - {tag}, tag)
                    assert destination in valid
                    row[destination] = row.get(destination, F(0)) + proposal / len(host) * min(F(1), ratio)
            elif (node, other) == (1, 0) and len(host) < CAP:
                restored = host | {center}
                assert len(restored) == len(host) + 1
                ratio = (NODE_WEIGHT[other] / NODE_WEIGHT[node] * ACTIVITY[node]
                         * VOLUME / len(restored) * degree_ratio)
                for newcenter in VERTICES:
                    destination = (other, restored, newcenter)
                    assert destination in valid
                    row[destination] = row.get(destination, F(0)) + proposal / VOLUME * min(F(1), ratio)
            elif {node, other} != {0, 1}:
                destination = (other, host, center)
                if destination in valid:
                    ratio = _weight(destination) / _weight(origin) * degree_ratio
                    row[destination] = row.get(destination, F(0)) + proposal * min(F(1), ratio)
        row[origin] = row.get(origin, F(0)) + 1 - sum(row.values())
        rows[origin] = row
    return rows


def _propagate(states, rows, masses=None):
    masses = {_state: _weight(_state) for _state in states} if masses is None else masses
    result = {state: F(0) for state in states}
    for origin, mass in masses.items():
        assert sum(rows[origin].values()) == 1
        assert all(chance >= 0 for chance in rows[origin].values())
        for destination, chance in rows[origin].items():
            result[destination] = result.get(destination, F(0)) + mass * chance
    return result


def test_conditional_label_heatbath_has_exact_full_stationary_law_and_support():
    states = _states()
    rows = _gibbs_rows(states)
    masses = {state: _weight(state) for state in states}
    assert _propagate(states, rows) == masses
    for origin, row in rows.items():
        for destination, chance in row.items():
            assert _weight(destination) > 0
            assert destination[1:] == origin[1:]
            assert _weight(origin) * chance == _weight(destination) * rows[destination][origin]
    # The conditional includes distinct activities, hard and soft supports,
    # and an actual order/number penalty. All factors are bounded by the cap.
    occupied_overlap = (4, frozenset((0, 1)), 3)
    assert set(destination[0] for destination in rows[occupied_overlap]) == {2, 4}


def test_gibbs_label_is_degree_independent_and_an_added_degree_factor_is_wrong():
    states = _states()
    masses = {state: _weight(state) for state in states}
    assert _propagate(states, _gibbs_rows(states)) == masses
    assert _propagate(states, _gibbs_rows(states, degree_bias=True)) != masses
    # No graph is supplied to the correct conditional: topology only affects
    # the separate neighbor-MH proposal, not the Gibbs probabilities.
    first = _gibbs_row((0, frozenset((0, 2)), 3))
    second = _gibbs_row((3, frozenset((0, 2)), 3))
    assert first == second


def test_local_then_tagged_neighbor_then_gibbs_composition_is_stationary():
    states = _states()
    masses = {state: _weight(state) for state in states}
    bynode = {node: [state for state in states if state[0] == node] for node in NODES}
    # Exact within-node host/center heatbath is an independent local invariant
    # control; invariance of its composition is the point, not speed-up.
    local = {}
    for origin in states:
        selected = bynode[origin[0]]
        normalizer = sum(_weight(state) for state in selected)
        local[origin] = {state: _weight(state) / normalizer for state in selected}
    neighbor = _hop_rows(states)
    assert _propagate(states, neighbor) == masses
    for rows in (local, neighbor, _gibbs_rows(states)):
        masses = _propagate(states, rows, masses)
    assert masses == {state: _weight(state) for state in states}


@pytest.mark.parametrize("origin,destination", [
    ((0, frozenset((0, 2)), 3), (1, frozenset((2,)), 0)),
    ((0, frozenset((0,)), 3), (1, frozenset(), 0)),
    ((1, frozenset((2,)), 0), (0, frozenset((0, 2)), 0)),
])
def test_post_tagged_number_order_and_center_features_must_be_reconstructed(origin, destination):
    states = _states()
    assert destination in _hop_rows(states)[origin]
    before, after = _features(origin[1], origin[2]), _features(destination[1], destination[2])
    assert before[0] != after[0] and before[1] != after[1] and before[3] != after[3]
    correct = _gibbs_row(destination)
    stale = _gibbs_row(destination, feature_override=before)
    assert correct != stale
    assert set(correct) <= set(states)
    # The genuine Palm endpoints remain hard, so both pair counts are zero.
    # N/order/minimum-distance can nevertheless not be copied from pre-hop stat.
    assert before[2] == after[2] == 0


def test_stale_features_after_tagged_moves_destroy_complete_sweep_stationarity():
    states = _states()
    masses = {state: _weight(state) for state in states}
    neighbors = _hop_rows(states)
    result = {}
    for origin, mass in masses.items():
        for intermediate, chance in neighbors[origin].items():
            changed_host = intermediate[1:] != origin[1:]
            stale = _features(origin[1], origin[2]) if changed_host else None
            for destination, gibbs_chance in _gibbs_row(intermediate, stale).items():
                result[destination] = result.get(destination, F(0)) + mass * chance * gibbs_chance
    assert any(result.get(state, F(0)) != weight for state, weight in masses.items())


def test_gibbs_then_lazy_neighbor_has_uniform_positive_reservoir_minorization():
    states = _states()
    ideal = 4
    # Nonnegative costs give this conservative bound, independently of host,
    # center and current node. It is a proof control, not a useful mixing rate.
    inverse_bound = sum(NODE_WEIGHT[node] / NODE_WEIGHT[ideal]
                        * max(F(1), ACTIVITY[node] / ACTIVITY[ideal]) ** CAP
                        for node in NODES)
    bound = 1 / inverse_bound
    assert bound > 0
    neighbors = _hop_rows(states)
    reservoir = [state for state in states if state[0] == ideal]
    norm = sum(_weight(state) for state in reservoir)
    refresh = {state: _weight(state) / norm for state in reservoir}
    for origin in states:
        conditional = _gibbs_row(origin)
        ideal_state = (ideal, origin[1], origin[2])
        assert conditional[ideal_state] >= bound
        assert neighbors[ideal_state][ideal_state] >= F(1, 2)
        ideal_arrival = sum(probability * sum(chance for destination, chance
                                              in neighbors[intermediate].items()
                                              if destination[0] == ideal)
                            for intermediate, probability in conditional.items())
        assert ideal_arrival >= bound / 2
        # The next incoming-ideal refresh removes all host/center memory;
        # an invariant ideal local kernel leaves this same conditional law.
        for destination, probability in refresh.items():
            assert ideal_arrival * probability >= bound / 2 * probability
