"""Independent pure controls: number bias is transport, not crystal occupancy.

No Monte Carlo/JIT calls are made here. Sector sums retain the full-host 1/N!
measure. Geometry below is reconstructed without sampler registry helpers.
"""
import math

import numpy as np
import pytest

from scripts import continuum_tempering_graph as graph


VOLUME = 32 * .96405
SITES = 32
CAP = 80


def sector_law(field, volume, cap, nu=0., preferred=0.):
    numbers = np.arange(cap + 1, dtype=float)
    logs = np.array([n * (field + math.log(volume)) - math.lgamma(n + 1)
                     - nu * (n - preferred) ** 2 for n in numbers])
    peak = float(logs.max())
    log_partition = peak + math.log(np.exp(logs - peak).sum())
    return numbers, logs, np.exp(logs - log_partition), log_partition


def test_dense_ideal_reservoir_has_a_number_distribution_not_a_fixed_crystal():
    target, weights = graph.build_compact_graph(VOLUME, SITES, CAP, base_field=16.)
    node = int(target['ideal_node'])
    assert node == 0
    field = float(target['activity'][node])
    assert field == pytest.approx(math.log(1 / 1.06448), abs=1e-15)
    assert target['radius'][node] == target['pair_penalty'][node] == 0
    assert target['number_stiffness'][node] == target['umbrella_stiffness'][node] == 0
    numbers, _, probability, _ = sector_law(field, VOLUME, CAP)
    mean = float(probability @ numbers)
    variance = float(probability @ ((numbers - mean) ** 2))
    expected = VOLUME / 1.06448
    assert mean == pytest.approx(expected, abs=1e-10)
    assert variance == pytest.approx(expected, abs=1e-8)
    assert 28 < mean < 30
    assert probability[32] > .02
    for n in range(25, 36):
        assert probability[n + 1] / probability[n] == pytest.approx(expected / (n + 1), rel=1e-13)
    assert CAP >= math.ceil(6 * VOLUME / math.pi)
    assert np.isfinite(weights).all() and weights[node] == 0


@pytest.mark.parametrize('nu,preferred', [(0., 0.), (.25, 32.), (4., 32.)])
def test_ideal_number_normalizer_keeps_full_host_factorials(nu, preferred):
    field = math.log(1 / 1.06448)
    numbers, logs, _, normalizer = sector_law(field, VOLUME, CAP, nu, preferred)
    actual = graph.compact_weight_guess(field, VOLUME, CAP, 0., 0., .88, nu, preferred)
    assert actual == pytest.approx(-normalizer, abs=2e-13)
    # No phase-fragment factor or occupancy-sector restriction can be inserted
    # into this exact ideal-plus-number target. All finite-cap log masses exist.
    assert np.isfinite(logs).all()
    differences = field + math.log(VOLUME) - np.log(numbers[1:]) - nu * (2 * (numbers[:-1] - preferred) + 1)
    np.testing.assert_allclose(np.diff(logs), differences, atol=2e-12, rtol=1e-13)


def test_ordered_number_cost_is_removed_before_the_shared_physical_endpoint():
    target, _ = graph.build_compact_graph(VOLUME, SITES, CAP, base_field=16., cavity_radius_stages=16)
    nu, preferred = target['number_stiffness'], target['number_preferred']
    np.testing.assert_array_equal(nu[21:70], np.full(49, 4.))
    np.testing.assert_array_equal(preferred[21:70], np.full(49, 32.))
    np.testing.assert_array_equal(nu[134:139], [.25, .5, 1., 2., 4.])
    np.testing.assert_array_equal(nu[139:145], [2., 1., .5, .25, .1, 0.])
    assert np.all(preferred[134:145] == SITES)
    physical = target['physical_nodes']
    assert np.all(nu[physical] == 0)
    assert np.all(target['umbrella_stiffness'][physical] == 0)
    assert np.isposinf(target['pair_penalty'][physical]).all()
    assert np.all(target['activity'][physical] == 16.)
    assert np.all(target['diameter'][physical] == 1.)
    path = target['radius_path_nodes']
    np.testing.assert_array_equal(path, np.r_[70, np.arange(145, 160), 71])
    assert np.all(np.diff(target['radius'][path]) > 0)
    assert target['radius'][path[0]] == 0 and target['radius'][path[-1]] == 1
    edges = [tuple(sorted(map(int, edge))) for edge in target['exchange_edges']]
    assert len(set(edges)) == len(edges) and all(a != b for a, b in edges)
    assert (20, 21) not in edges and (69, 70) not in edges
    for route in ([20, 134, 135, 136, 137, 138, 21], [69, 139, 140, 141, 142, 143, 144, 70], path):
        assert all(tuple(sorted(map(int, pair))) in edges for pair in zip(route[:-1], route[1:]))
    assert (70, 71) in edges
    reached = {0}
    while True:
        expanded = reached | {b for a, b in edges if a in reached} | {a for a, b in edges if b in reached}
        if expanded == reached:
            break
        reached = expanded
    assert reached == set(range(len(target['activity'])))


def independent_fcc_and_order(length):
    basis = np.array([[0., 0., 0.], [0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])
    points = np.array([(np.array([ix, iy, iz]) + b) * length / 2
                       for ix in range(2) for iy in range(2) for iz in range(2) for b in basis])
    modes = 2 * np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                           [2, 0, 0], [0, 2, 0], [0, 0, 2]])

    def order(coords):
        phases = 2 * math.pi * coords @ modes.T / length
        return float(np.abs(np.exp(1j * phases).sum(axis=0)).mean() / len(coords))

    return points, order


def test_preferred_number_and_large_bragg_do_not_certify_crystal_occupancy():
    length = VOLUME ** (1 / 3)
    perfect, order = independent_fcc_and_order(length)
    defective = perfect.copy()
    defective[1] = (perfect[0] + [1e-3, 0., 0.]) % length
    # Distinct coordinates stay in the simple-point configuration space.
    assert len(np.unique(defective, axis=0)) == len(defective) == SITES
    assert order(perfect) == pytest.approx(1., abs=1e-14)
    assert order(defective) > .99999
    assert 4 * (len(perfect) - SITES) ** 2 == 4 * (len(defective) - SITES) ** 2 == 0

    def distances(first, second):
        difference = first[:, None, :] - second[None, :, :]
        difference -= length * np.rint(difference / length)
        return np.sum(difference ** 2, axis=2)

    regular_distances = distances(perfect, perfect)
    bad_distances = distances(defective, defective)
    np.fill_diagonal(regular_distances, np.inf)
    np.fill_diagonal(bad_distances, np.inf)
    assert regular_distances.min() > 1
    assert bad_distances.min() == pytest.approx(1e-6, abs=1e-18)
    regular_counts = np.bincount(np.argmin(distances(perfect, perfect), axis=1), minlength=SITES)
    defective_counts = np.bincount(np.argmin(distances(defective, perfect), axis=1), minlength=SITES)
    assert np.all(regular_counts == 1)
    assert np.count_nonzero(defective_counts == 0) == 1
    assert np.count_nonzero(defective_counts == 2) == 1
    # Soft penalties give finite weight, but the hard physical target rejects
    # the paired-site configuration. Number bias cannot replace that support.
    target, _ = graph.build_compact_graph(VOLUME, SITES, CAP, base_field=16.)
    node = 21
    overlaps = int(np.count_nonzero(np.triu(bad_distances < 1, k=1)))
    energy = (target['pair_penalty'][node] * overlaps
              + target['umbrella_stiffness'][node] * (order(defective) - target['umbrella_preferred'][node]) ** 2)
    assert overlaps >= 1 and np.isfinite(energy)
