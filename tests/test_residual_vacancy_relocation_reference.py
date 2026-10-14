"""Independent mathematical controls for a proposed constant-N relocation.

These are reference-law tests, NOT implemented-kernel or dense mixing evidence.
No sampler is imported. Actual-kernel binding is in the separate
test_continuum_vacancy_relocation_equilibrium_independent.py control.

Implementation-binding controls required once an option exists:
* actual N<=3 continuum partition quadrature with nonzero pair/order/number
  costs, cavity support, and at least one nonuniform residual proposal;
* reconstruct every accepted/rejected move's Fourier cache and hard support;
* seeded option-off regression and rejection leaves coordinates/cache unchanged;
* particle/coordinate permutation, periodic translation, overlapping cubes,
  residual low-N/no-empty-site fallbacks and all covering-box marginal density;
* evaluate old and new positions against the SAME residual host, with a
  uniformly selected tag and no birth/death, activity or factorial multiplier.
"""
from itertools import combinations, product
import math

import numpy as np
import pytest


def reference_cube_density(point, centers, length, width, epsilon):
    if len(centers) == 0:
        return 1 / length ** 3
    offsets = np.asarray(point) - np.asarray(centers)
    offsets -= length * np.rint(offsets / length)
    covering = np.count_nonzero(np.all(np.abs(offsets) < width, axis=1))
    return epsilon / length ** 3 + (1 - epsilon) * covering / (len(centers) * (2 * width) ** 3)


def exact_piecewise_density_integral(centers, length, width, epsilon):
    cuts = []
    for axis in range(3):
        values = [0., length]
        values.extend((np.asarray(centers)[:, axis] - width) % length)
        values.extend((np.asarray(centers)[:, axis] + width) % length)
        cuts.append(np.unique(values))
    total = 0.
    for intervals in product(*(list(zip(axis[:-1], axis[1:])) for axis in cuts)):
        point = np.array([.5 * (left + right) for left, right in intervals])
        volume = math.prod(right - left for left, right in intervals)
        total += volume * reference_cube_density(point, centers, length, width, epsilon)
    return total


def test_periodic_overlapping_cube_density_is_a_normalized_mixture_not_a_union():
    centers = np.array([[.95, .5, .5], [.05, .5, .5], [.05, .6, .5]])
    epsilon, length, width = .2, 1., .2
    assert exact_piecewise_density_integral(centers, length, width, epsilon) == pytest.approx(1., abs=2e-14)
    expected = epsilon + (1 - epsilon) / (2 * width) ** 3
    assert reference_cube_density([0., .55, .5], centers, length, width, epsilon) == pytest.approx(expected)
    assert reference_cube_density([1., .55, .5], centers, length, width, epsilon) == pytest.approx(expected)
    assert reference_cube_density([.5, .1, .1], centers, length, width, epsilon) == epsilon


def graph_density(point, residual, vertices=6, epsilon=.2):
    # Two distinct candidate positions per reference site permit an auxiliary
    # paired-site host without putting coincident points in configuration space.
    occupied = {r // 2 for r in residual}
    empty = [site for site in range(vertices // 2) if site not in occupied]
    if len(residual) < 2 or not empty:
        return 1 / vertices
    return epsilon / vertices + (1 - epsilon) * (point // 2 in empty) / (2 * len(empty))


def graph_cost(host, gamma, kappa, preferred, nu, number_preferred, forbidden):
    if any(point in forbidden for point in host):
        return np.inf
    overlaps = sum(a // 2 == b // 2 for a, b in combinations(host, 2))
    if overlaps and np.isposinf(gamma):
        return np.inf
    pair_cost = 0. if not overlaps else gamma * overlaps
    order = abs(sum(np.exp(2j * math.pi * point / 6) for point in host)) / len(host) if host else 0.
    return pair_cost + kappa * (order - preferred) ** 2 + nu * (len(host) - number_preferred) ** 2


def exact_graph_transition(gamma, kappa, preferred, nu, number_preferred, forbidden=(), wrong=False):
    hosts = [tuple(items) for n in range(4) for items in combinations(range(6), n)]
    energies = np.array([graph_cost(h, gamma, kappa, preferred, nu, number_preferred, forbidden) for h in hosts])
    valid = np.isfinite(energies)
    hosts, energies = [h for h, keep in zip(hosts, valid) if keep], energies[valid]
    lookup = {host: i for i, host in enumerate(hosts)}
    field = math.log(.3)
    # Counting unordered subsets has already supplied the one full-host
    # combinatorial factor: no extra n! or phase-fragment weight is inserted.
    logs = np.array([field * len(h) - cost for h, cost in zip(hosts, energies)])
    weight = np.exp(logs - logs.max())
    weight /= weight.sum()
    transition = np.zeros((len(hosts), len(hosts)))
    for index, host in enumerate(hosts):
        if not host:
            transition[index, index] = 1.
            continue
        for old in host:
            residual = tuple(point for point in host if point != old)
            for new in range(6):
                proposal = graph_density(new, residual) / len(host)
                replacement = tuple(sorted((*residual, new)))
                if new in residual or replacement not in lookup:
                    transition[index, index] += proposal
                    continue
                other = lookup[replacement]
                log_ratio = energies[index] - energies[other]
                if not wrong:
                    log_ratio += math.log(graph_density(old, residual) / graph_density(new, residual))
                acceptance = math.exp(min(0., log_ratio))
                transition[index, other] += proposal * acceptance
                transition[index, index] += proposal * (1 - acceptance)
    return hosts, weight, transition


@pytest.mark.parametrize('gamma,forbidden', [(.7, ()), (8., (0,)), (np.inf, (0, 1))])
def test_residual_tag_relocation_preserves_every_number_sector_and_full_target(gamma, forbidden):
    hosts, probability, transition = exact_graph_transition(gamma, 2.5, .6, 1.8, 2.4, forbidden)
    np.testing.assert_allclose(transition.sum(axis=1), 1., atol=2e-15)
    np.testing.assert_allclose(probability @ transition, probability, atol=2e-15)
    flux = probability[:, None] * transition
    np.testing.assert_allclose(flux, flux.T, atol=2e-15)
    for i, before in enumerate(hosts):
        for j, after in enumerate(hosts):
            if len(before) != len(after):
                assert transition[i, j] == 0


def test_missing_reverse_residual_density_breaks_stationarity():
    _, probability, transition = exact_graph_transition(.7, 2.5, .6, 1.8, 2.4, wrong=True)
    assert np.max(np.abs(probability @ transition - probability)) > 1e-5


def test_fcc_paired_site_to_vacancy_relocation_is_number_neutral_and_reversible():
    volume = 32 * .96405
    length = volume ** (1 / 3)
    basis = np.array([[0., 0., 0.], [0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])
    perfect = np.array([(np.array([i, j, k]) + b) * length / 2
                        for i in range(2) for j in range(2) for k in range(2) for b in basis])
    defective = perfect.copy()
    defective[1] = (perfect[0] + [1e-3, 0., 0.]) % length
    old, new = defective[1].copy(), perfect[1].copy()
    residual = np.delete(defective, 1, axis=0)
    restored = defective.copy()
    restored[1] = new
    np.testing.assert_array_equal(np.delete(restored, 1, axis=0), residual)
    assert len(np.unique(defective, axis=0)) == len(defective) == 32
    assert 4 * (len(defective) - 32) ** 2 == 4 * (len(restored) - 32) ** 2 == 0
    # The residual has precisely the missing site used to draw a repair.
    offsets = residual[:, None, :] - perfect[None, :, :]
    offsets -= length * np.rint(offsets / length)
    counts = np.bincount(np.argmin((offsets ** 2).sum(axis=2), axis=1), minlength=32)
    centers = perfect[counts == 0]
    np.testing.assert_array_equal(centers, perfect[1:2])
    old_density = reference_cube_density(old, centers, length, .06, .2)
    new_density = reference_cube_density(new, centers, length, .06, .2)
    assert 0 < old_density < new_density
    # A narrow vacancy proposal has an explicit reverse-density price; merely
    # repairing an overlap does not justify unconditional acceptance.
    pair_gain = 8.
    forward = min(1., math.exp(pair_gain) * old_density / new_density)
    reverse = min(1., math.exp(-pair_gain) * new_density / old_density)
    assert forward > 0 and reverse > 0
    assert math.exp(-pair_gain) * new_density * forward == pytest.approx(old_density * reverse, rel=1e-13)
