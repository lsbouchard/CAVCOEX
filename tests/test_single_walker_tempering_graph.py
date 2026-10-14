"""Exact finite-graph controls for the separate single-host continuum design.

No Monte Carlo, compiled kernel, or fluid--solid coexistence assumption occurs
here.  Unlabelled lattice subsets have weights zeta**N; auxiliary centers are
uniform over vertices, so these controls independently expose the marked-center
volume and proposal-degree factors.
"""
from fractions import Fraction as F
from itertools import combinations

import pytest


VERTICES = tuple(range(4))
VOLUME = len(VERTICES)
PAIR_EDGES = {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 3))}
GRAPH = {0: (1,), 1: (0, 2), 2: (1,)}
ACTIVITY = {0: F(2, 3), 1: F(2, 3), 2: F(1, 4)}
NODE_WEIGHT = {0: F(2), 1: F(3), 2: F(5)}


def _subsets(cap):
    return [frozenset(subset) for n in range(cap + 1)
            for subset in combinations(VERTICES, n)]


def _hard(host):
    return not any(edge.issubset(host) for edge in PAIR_EDGES)


def _allowed(node, host, center):
    if node == 2:
        return True  # Finite-lattice ideal reservoir, not a continuum Poisson law.
    if not _hard(host):
        return False
    return node == 0 or (center not in host and _hard(host | {center}))


def _states(cap=2):
    return [(node, host, center) for node in GRAPH for host in _subsets(cap)
            for center in VERTICES if _allowed(node, host, center)]


def _weight(state, weights=NODE_WEIGHT):
    node, host, _ = state
    return weights[node] * ACTIVITY[node] ** len(host) / VOLUME


def _hop_rows(states, corrected=True):
    """Explicit lazy neighbor MH proposal on an irregular three-node path."""
    valid = set(states)
    rows = {}
    for state in states:
        node, host, center = state
        row = {}
        for newnode in GRAPH[node]:
            newstate = (newnode, host, center)
            if newstate not in valid:
                continue
            ratio = _weight(newstate) / _weight(state)
            if corrected:
                ratio *= F(len(GRAPH[node]), len(GRAPH[newnode]))
            mass = F(1, 2 * len(GRAPH[node])) * min(F(1), ratio)
            row[newstate] = row.get(newstate, F(0)) + mass
        row[state] = F(1) - sum(row.values())
        rows[state] = row
    return rows


def _tagged_rows(states, cap=2, formula="correct", refresh=True):
    valid = set(states)
    choose = {0: F(1, 3), 1: F(2, 5)}  # Unequal move-type selection probabilities.
    rows = {}
    for state in states:
        node, host, center = state
        row = {}
        if node == 0 and host:
            n = len(host)
            for tag in host:
                newstate = (1, host - {tag}, tag)
                assert newstate in valid
                numerator = n if formula != "squared" else n * n
                volume = VOLUME if formula != "missing_volume" else 1
                ratio = (NODE_WEIGHT[1] / NODE_WEIGHT[0] * numerator
                         / (ACTIVITY[0] * volume) * choose[1] / choose[0])
                row[newstate] = choose[0] / n * min(F(1), ratio)
        elif node == 1 and len(host) < cap:
            n = len(host) + 1
            restored = host | {center}
            assert _hard(restored) and len(restored) == n
            numerator = n if formula != "squared" else n * n
            volume = VOLUME if formula != "missing_volume" else 1
            ratio = (NODE_WEIGHT[0] / NODE_WEIGHT[1] * ACTIVITY[0]
                     * volume / numerator * choose[0] / choose[1])
            centers = VERTICES if refresh else (center,)
            for base_center in centers:
                newstate = (0, restored, base_center)
                assert newstate in valid
                row[newstate] = choose[1] / len(centers) * min(F(1), ratio)
        row[state] = F(1) - sum(row.values())
        rows[state] = row
    return rows


def _balance_defects(states, rows):
    return [(first, second, _weight(first) * mass
             - _weight(second) * rows[second].get(first, F(0)))
            for first in states for second, mass in rows[first].items()
            if _weight(first) * mass != _weight(second) * rows[second].get(first, F(0))]


def _assert_stationary(states, rows):
    for state in states:
        assert sum(rows[state].values()) == 1
        assert all(mass >= 0 for mass in rows[state].values())
        incoming = sum(_weight(other) * rows[other].get(state, F(0)) for other in states)
        assert incoming == _weight(state)


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_tagged_jump_has_exact_full_stationary_law_with_unequal_selection_and_cap(cap):
    states = _states(cap)
    rows = _tagged_rows(states, cap)
    assert not _balance_defects(states, rows)
    _assert_stationary(states, rows)
    # N=0 is an explicit forward self-loop, not division by zero.
    for state in states:
        node, host, _ = state
        if node == 0 and not host:
            assert rows[state] == {state: F(1)}
        if node == 1 and len(host) == cap:
            assert rows[state] == {state: F(1)}  # Capped reverse restoration rejected.


@pytest.mark.parametrize("wrong_formula", ["squared", "missing_volume"])
def test_tempting_wrong_tagged_ratios_fail_detailed_balance(wrong_formula):
    states = _states()
    defects = _balance_defects(states, _tagged_rows(states, formula=wrong_formula))
    assert defects
    assert any(len(first[1]) == 2 or len(second[1]) == 2
               for first, second, _ in defects)


def test_reverse_base_center_must_be_refreshed_not_retained():
    states = _states()
    rows = _tagged_rows(states, refresh=False)
    defects = _balance_defects(states, rows)
    assert defects
    # A base center different from its tagged particle would otherwise have a
    # nonzero forward move but no reverse proposal to its original state.
    assert any(mass and rows[second].get(first, F(0)) == 0
               for first in states for second, mass in rows[first].items())


def test_lazy_degree_corrected_hops_preserve_exact_extended_law():
    states = _states()
    rows = _hop_rows(states)
    assert not _balance_defects(states, rows)
    _assert_stationary(states, rows)
    assert all(rows[state][state] >= F(1, 2) for state in states)


def test_missing_neighbor_degree_factor_changes_target_on_irregular_graph():
    states = _states()
    assert _balance_defects(states, _hop_rows(states, corrected=False))


def test_finite_tempering_weights_change_node_frequency_not_any_conditional_target():
    states = _states()
    first = {state: _weight(state) for state in states}
    alternate = {state: _weight(state, {0: F(17), 1: F(1, 19), 2: F(23)})
                 for state in states}
    first_norm, alternate_norm = sum(first.values()), sum(alternate.values())
    for node in GRAPH:
        selected = [state for state in states if state[0] == node]
        first_node = sum(first[state] for state in selected)
        alternate_node = sum(alternate[state] for state in selected)
        assert first_node / first_norm != alternate_node / alternate_norm
        for state in selected:
            assert first[state] / first_node == alternate[state] / alternate_node


def test_exact_ideal_reservoir_heatbath_preserves_extended_law_without_host_memory():
    states = _states()
    ideal = [state for state in states if state[0] == 2]
    norm = sum(_weight(state) for state in ideal)
    heatbath = {state: _weight(state) / norm for state in ideal}
    rows = {state: heatbath.copy() if state[0] == 2 else {state: F(1)}
            for state in states}
    _assert_stationary(states, rows)
    assert rows[ideal[0]] == rows[ideal[-1]]
    # Conditional on any host subset, the auxiliary center remains uniform.
    for host in _subsets(2):
        weights = [heatbath[(2, host, center)] for center in VERTICES]
        assert len(set(weights)) == 1


def test_exact_sweep_composition_of_hops_tagged_and_heatbath_is_stationary():
    states = _states()
    hops, tagged = _hop_rows(states), _tagged_rows(states)
    ideal = [state for state in states if state[0] == 2]
    norm = sum(_weight(state) for state in ideal)
    reservoir = {state: _weight(state) / norm for state in ideal}
    # Compute the complete ordered sweep law by propagation of exact masses,
    # without claiming the composed transition matrix itself is reversible.
    masses = {state: _weight(state) for state in states}
    for kernel in (hops, tagged, {state: reservoir if state[0] == 2 else {state: F(1)}
                                for state in states}):
        new = {state: F(0) for state in states}
        for origin, mass in masses.items():
            for destination, chance in kernel[origin].items():
                new[destination] += mass * chance
        masses = new
    assert masses == {state: _weight(state) for state in states}


def test_regenerative_phase_estimator_weights_physical_dwells_not_cycles():
    # Three hypothetical independent cycle rewards: the second cycle has nine
    # physical dwells, and the third has none. This is algebra, not sampled data.
    rewards = [F(1), F(0), F(0)]
    dwells = [F(1), F(9), F(0)]
    correct = sum(rewards) / sum(dwells)
    wrong = sum(r / d for r, d in zip(rewards, dwells) if d) / 2
    assert correct == F(1, 10)
    assert wrong == F(1, 2)
    assert sum(r - correct * d for r, d in zip(rewards, dwells)) == 0
