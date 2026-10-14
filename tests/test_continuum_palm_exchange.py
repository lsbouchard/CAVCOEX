"""Exact finite-state checks of coupled unconstrained/particle-cavity exchange.

The independent-set graph analogue retains the complete cavity-center state.
Uniformly tag a point in A, exchange A minus that point with B, insert B's
cavity center into the new A, and use the old tagged point as the new center.
The exact Metropolis ratio is |A|/(|B|+1), not its square.
"""

import math

import numpy as np
import pytest


def independent_sets(vertex_count, edges):
    return [
        mask for mask in range(1 << vertex_count)
        if all(not ((mask >> a) & 1 and (mask >> b) & 1) for a, b in edges)
    ]


def insertion_allowed(mask, center, edges):
    if (mask >> center) & 1:
        return False
    return all(
        not ((a == center and (mask >> b) & 1)
             or (b == center and (mask >> a) & 1))
        for a, b in edges
    )


def exchange_transition(vertex_count, edges, acceptance_power=1):
    hosts = independent_sets(vertex_count, edges)
    states = [
        (a, b, center)
        for a in hosts for b in hosts for center in range(vertex_count)
        if insertion_allowed(b, center, edges)
    ]
    indices = {state: i for i, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)))
    for row, (a, b, center) in enumerate(states):
        n, m = a.bit_count(), b.bit_count()
        if not n:
            transition[row, row] = 1.
            continue
        acceptance = min(1., (n / (m + 1))**acceptance_power)
        for tagged in range(vertex_count):
            if (a >> tagged) & 1:
                updated = (b | (1 << center), a ^ (1 << tagged), tagged)
                # This lookup independently checks every proposed support.
                col = indices[updated]
                assert updated[0].bit_count() + updated[1].bit_count() == n + m
                transition[row, col] += acceptance / n
                transition[row, row] += (1 - acceptance) / n
    return states, transition


@pytest.mark.parametrize("activity", [.2, 1., 7.3])
def test_exact_hard_core_graph_palm_exchange_detailed_balance(activity):
    vertex_count = 6
    edges = [(i, (i + 1) % vertex_count) for i in range(vertex_count)]
    states, transition = exchange_transition(vertex_count, edges)
    weights = np.array([activity**(a.bit_count() + b.bit_count())
                        for a, b, _ in states])
    probability = weights / weights.sum()
    np.testing.assert_allclose(transition.sum(axis=1), 1., atol=2e-15, rtol=0)
    flux = probability[:, None] * transition
    np.testing.assert_allclose(flux, flux.T, atol=2e-16, rtol=0)
    np.testing.assert_allclose(probability @ transition, probability, atol=2e-16, rtol=0)


def test_squared_palm_exchange_ratio_fails_graph_detailed_balance():
    vertex_count = 6
    edges = [(i, (i + 1) % vertex_count) for i in range(vertex_count)]
    states, wrong_transition = exchange_transition(vertex_count, edges, acceptance_power=2)
    # Equal activity gives the uniform measure on all permitted joint states.
    flux = wrong_transition / len(states)
    assert np.max(abs(flux - flux.T)) > 1e-4


@pytest.mark.parametrize("total", range(1, 13))
def test_ideal_gas_factorial_normalization_gives_same_count_ratio(total):
    # Independent equal-activity Poisson hosts, conditioned on their total,
    # have this binomial law. The move maps (n,m) to (m+1,n-1).
    probability = np.array([math.comb(total, n) / 2**total
                            for n in range(total + 1)])
    transition = np.zeros((total + 1, total + 1))
    transition[0, 0] = 1.
    for n in range(1, total + 1):
        m = total - n
        updated_n = m + 1
        acceptance = min(1., n / (m + 1))
        transition[n, updated_n] += acceptance
        transition[n, n] += 1 - acceptance
    flux = probability[:, None] * transition
    np.testing.assert_allclose(flux, flux.T, atol=2e-16, rtol=0)
    np.testing.assert_allclose(probability @ transition, probability, atol=2e-16, rtol=0)
