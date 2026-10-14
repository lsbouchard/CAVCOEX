"""Independent finite-graph control of a proposed coordinate-biased GCMC move.

No production sampler implements this optional proposal in this test. It
verifies the complete mixture density and, crucially, the deletion reverse
density computed on the residual host rather than the original host.
"""

import math

import numpy as np
import pytest


def occupied_vertices(mask, size):
    return [vertex for vertex in range(size) if (mask >> vertex) & 1]


def proposal_density(mask, size, epsilon, box_half_width):
    particles = occupied_vertices(mask, size)
    n = len(particles)
    if n < 2:
        return np.full(size, 1 / size)
    covering = np.zeros(size)
    for first in particles:
        for second in particles:
            if first != second:
                center = (2 * first - second) % size
                for offset in range(-box_half_width, box_half_width + 1):
                    covering[(center + offset) % size] += 1
    volume = 2 * box_half_width + 1
    return epsilon / size + (1 - epsilon) * covering / (n * (n - 1) * volume)


def graph_hosts(size):
    return [mask for mask in range(1 << size)
            if all(not ((mask >> vertex) & 1 and (mask >> ((vertex + 1) % size)) & 1)
                   for vertex in range(size))]


def graph_order_cost(mask, size):
    particles = occupied_vertices(mask, size)
    order = abs(sum(np.exp(4j * np.pi * vertex / size) for vertex in particles)) / len(particles) if particles else 0.
    return 1.7 * (order - .4)**2


def transition_matrix(size, activity, epsilon, wrong_death_density=False):
    hosts = graph_hosts(size)
    index = {mask: row for row, mask in enumerate(hosts)}
    costs = np.array([graph_order_cost(mask, size) for mask in hosts])
    transition = np.zeros((len(hosts), len(hosts)))
    for row, mask in enumerate(hosts):
        n = mask.bit_count()
        density = proposal_density(mask, size, epsilon, 1)
        for vertex in range(size):
            insertion_probability = .5 * density[vertex]
            inserted = mask | (1 << vertex)
            if (mask >> vertex) & 1 or inserted not in index:
                transition[row, row] += insertion_probability
            else:
                col = index[inserted]
                acceptance = min(1., activity / ((n + 1) * density[vertex])
                                 * math.exp(costs[row] - costs[col]))
                transition[row, col] += insertion_probability * acceptance
                transition[row, row] += insertion_probability * (1 - acceptance)
        if not n:
            transition[row, row] += .5
        else:
            for vertex in occupied_vertices(mask, size):
                deleted = mask ^ (1 << vertex)
                col = index[deleted]
                reverse_density = (density if wrong_death_density else
                                   proposal_density(deleted, size, epsilon, 1))
                acceptance = min(1., n * reverse_density[vertex] / activity
                                 * math.exp(costs[row] - costs[col]))
                transition[row, col] += .5 / n * acceptance
                transition[row, row] += .5 / n * (1 - acceptance)
    weights = np.array([activity**mask.bit_count() for mask in hosts]) * np.exp(-costs)
    return transition, weights / weights.sum()


@pytest.mark.parametrize("activity", [.2, 1., 7.3])
@pytest.mark.parametrize("epsilon", [.1, .7])
def test_state_dependent_birth_death_preserves_full_graph_target(activity, epsilon):
    transition, probability = transition_matrix(8, activity, epsilon)
    np.testing.assert_allclose(transition.sum(axis=1), 1., atol=2e-15, rtol=0)
    flux = probability[:, None] * transition
    np.testing.assert_allclose(flux, flux.T, atol=2e-16, rtol=0)
    np.testing.assert_allclose(probability @ transition, probability, atol=2e-16, rtol=0)


@pytest.mark.parametrize("half_width", [0, 1, 2, 3])
def test_complete_mixture_is_normalized_positive_and_translation_covariant(half_width):
    size, epsilon = 8, .13
    for mask in range(1 << size):
        density = proposal_density(mask, size, epsilon, half_width)
        np.testing.assert_allclose(density.sum(), 1., atol=2e-15, rtol=0)
        assert np.all(density >= epsilon / size)
        shifted = sum(1 << ((vertex + 3) % size) for vertex in occupied_vertices(mask, size))
        np.testing.assert_allclose(proposal_density(shifted, size, epsilon, half_width),
                                   np.roll(density, 3), atol=2e-16, rtol=0)
        if mask.bit_count() < 2:
            np.testing.assert_array_equal(density, np.full(size, 1 / size))


def test_using_original_host_density_for_death_breaks_detailed_balance():
    transition, probability = transition_matrix(8, 1., .1, wrong_death_density=True)
    flux = probability[:, None] * transition
    assert np.max(abs(flux - flux.T)) > 1e-4
