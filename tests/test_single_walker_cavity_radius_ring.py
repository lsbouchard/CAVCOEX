"""Exact cavity-radius ring: stationary law and genuine number transport.

Finite lattice supports are nested cavity events. This is a kernel/probability
control, not a continuum phase-coexistence or efficiency calculation.
"""
from fractions import Fraction as F
from itertools import combinations

import pytest


VERTICES = tuple(range(4))
VOLUME = len(VERTICES)
ACTIVITY = (F(2, 3), F(2, 3), F(2, 3), F(1, 5))
WEIGHT = (F(2), F(3), F(5), F(7))
# 0: base; 1: matched cavity; 2: smaller cavity; 3: ideal reservoir.
GRAPH = {0: (1, 2, 3), 1: (0, 2), 2: (0, 1), 3: (0,)}


def _allowed(node, host, center):
    if node == 3:
        return True
    if any(abs(first - second) <= 1 for first, second in combinations(host, 2)):
        return False
    if node == 1:
        return all(abs(site - center) >= 2 for site in host)
    if node == 2:
        return center not in host
    return True


def _weight(state):
    node, host, center = state
    return WEIGHT[node] * ACTIVITY[node] ** len(host) / VOLUME if _allowed(node, host, center) else F(0)


def _states(cap):
    return [(node, frozenset(host), center) for node in GRAPH for n in range(cap + 1)
            for host in combinations(VERTICES, n) for center in VERTICES
            if _allowed(node, frozenset(host), center)]


def _rows(states, cap):
    valid, result = set(states), {}
    for origin in states:
        node, host, center = origin
        row = {}
        for other in GRAPH[node]:
            chance = F(1, 2 * len(GRAPH[node]))
            degree = F(len(GRAPH[node]), len(GRAPH[other]))
            if (node, other) == (0, 1) and host:
                ratio = WEIGHT[1] / WEIGHT[0] * len(host) / (ACTIVITY[0] * VOLUME) * degree
                for tag in host:
                    destination = (1, host - {tag}, tag)
                    assert destination in valid
                    row[destination] = row.get(destination, F(0)) + chance / len(host) * min(F(1), ratio)
            elif (node, other) == (1, 0) and len(host) < cap:
                restored = host | {center}
                assert len(restored) == len(host) + 1
                ratio = WEIGHT[0] / WEIGHT[1] * ACTIVITY[0] * VOLUME / len(restored) * degree
                for newcenter in VERTICES:
                    destination = (0, restored, newcenter)
                    assert destination in valid
                    row[destination] = row.get(destination, F(0)) + chance / VOLUME * min(F(1), ratio)
            elif {node, other} != {0, 1}:
                destination = (other, host, center)
                if destination in valid:
                    ratio = _weight(destination) / _weight(origin) * degree
                    row[destination] = row.get(destination, F(0)) + chance * min(F(1), ratio)
        row[origin] = row.get(origin, F(0)) + 1 - sum(row.values())
        result[origin] = row
    return result


@pytest.mark.parametrize("cap", [1, 2])
def test_radius_ring_preserves_complete_capped_joint_target(cap):
    states = _states(cap)
    rows = _rows(states, cap)
    for origin in states:
        assert sum(rows[origin].values()) == 1
        assert rows[origin][origin] >= F(1, 2)
        incoming = sum(_weight(other) * rows[other].get(origin, F(0)) for other in states)
        assert incoming == _weight(origin)
        for destination, chance in rows[origin].items():
            assert _weight(origin) * chance == _weight(destination) * rows[destination].get(origin, F(0))


@pytest.mark.parametrize("cap", [1, 2])
def test_shrinking_radius_loop_transports_one_particle_with_exact_reverse_flux(cap):
    states = _states(cap)
    rows = _rows(states, cap)
    host = frozenset((0,)) if cap == 1 else frozenset((0, 2))
    reduced = host - {0}
    forward = [(0, host, 3), (1, reduced, 0), (2, reduced, 0), (0, reduced, 0)]
    reverse = [forward[-1], forward[-2], forward[-3], forward[0]]
    assert len(forward[-1][1]) == len(forward[0][1]) - 1
    assert forward[0][0] == forward[-1][0] == 0
    forward_mass, reverse_mass = _weight(forward[0]), _weight(reverse[0])
    for first, second in zip(forward, forward[1:]):
        assert rows[first][second] > 0
        forward_mass *= rows[first][second]
    for first, second in zip(reverse, reverse[1:]):
        assert rows[first][second] > 0
        reverse_mass *= rows[first][second]
    assert forward_mass == reverse_mass
    # Radius changes preserve host number and center. Only the tagged edge
    # deletes/restores, so a closed loop—not a birth/death proposal—carries N.
    assert forward[1][1:] == forward[2][1:] == forward[3][1:]


def test_global_label_gibbs_already_allows_number_preserving_exit_from_matched_cavity():
    # The radius ring is not the first possible escape once global Gibbs is
    # enabled. It provides extra legitimate relaxation targets, not a proof
    # that vacancy free energies or phase barriers disappear.
    origin = (1, frozenset((2,)), 0)
    base = (0, origin[1], origin[2])
    conditional_mass = _weight(base) / sum(_weight((node, origin[1], origin[2])) for node in GRAPH)
    assert conditional_mass > 0 and len(base[1]) == len(origin[1])
