"""Independent finite-target laws for an optional auxiliary number umbrella.

The ideal gas has a closed capped number law.  Hard/cavity and soft/order
targets are integrated in nine dimensions with no registry in the target.
All physical nodes remain number-unbiased.  These tests verify implemented
equilibrium laws; they do not establish dense fluid--solid mixing.
"""

import itertools
import math
import os
import sys
from fractions import Fraction as F
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import fcc_reference, run_chunk, seed_rng

from test_continuum_vacancy_birth_equilibrium import (
    VECTORS, verify_fourier_caches, verify_product_number_law,
)


def test_number_umbrella_death_increment_has_a_uniform_finite_bound():
    for cap in (1, 3, 8):
        for nu, preferred in itertools.product(
                (F(0), F(3, 10), F(4)),
                (F(-2), F(1, 4), F(1), F(cap), F(cap + 2))):
            safe_penalty = max(F(0), nu * (2 * preferred - 1))
            for n in range(1, cap + 1):
                before = nu * (n - preferred) ** 2
                after = nu * (n - 1 - preferred) ** 2
                delta = after - before
                assert delta == nu * (1 - 2 * (n - preferred))
                assert delta <= safe_penalty
            # Combining a bounded Bragg umbrella does not change the total
            # number draining argument: its additional cost is at most kappa.
            kappa = F(5, 2)
            for b_old, b_new, b_star in itertools.product(
                    (F(0), F(1, 2), F(1)), repeat=3):
                order_delta = kappa * ((b_new - b_star) ** 2 - (b_old - b_star) ** 2)
                assert order_delta <= kappa


def test_auxiliary_number_umbrella_preserves_capped_ideal_product_law():
    length, cap = 3., 5
    means = np.array([3.5, 2.3])
    number_stiffness = np.array([0., .8])
    number_preferred = np.array([0., 3.7])
    probabilities = []
    for replica, mean in enumerate(means):
        weights = np.array([
            mean ** n / math.factorial(n)
            * math.exp(-number_stiffness[replica] * (n - number_preferred[replica]) ** 2)
            for n in range(cap + 1)
        ])
        probabilities.append(weights / weights.sum())
    positions = np.zeros((2, cap, 3))
    counts = np.zeros(2, dtype=np.int64)
    centers = np.zeros((2, 3))
    identities = np.arange(2)
    diameter, radius = np.zeros(2), np.zeros(2)
    fields = np.log(means / length ** 3)
    penalty, stiffness, preferred = np.full(2, np.inf), np.zeros(2), np.zeros(2)
    edges = np.array([[0, 1]], dtype=np.int64)
    cavity_nodes, snapshots = np.array([0], dtype=np.int64), np.arange(2)
    reference = fcc_reference(4, length)
    seed_rng(189211)
    arguments = positions, counts, centers, diameter, radius, fields, identities, length
    tail = (0, 3, reference, np.ones(2), penalty, stiffness, preferred,
            edges, cavity_nodes, snapshots, False, True, .35, .2, 1)
    kwargs = dict(number_stiffness=number_stiffness, number_preferred=number_preferred)
    run_chunk(*arguments, 3000, *tail, **kwargs)
    samples, accept, swaps, configurations = run_chunk(*arguments, 51000, *tail, **kwargs)
    verify_product_number_law(samples, probabilities)
    for replica, target in enumerate(probabilities):
        np.testing.assert_allclose(samples[:, replica, 0].mean(),
                                   np.dot(np.arange(cap + 1), target),
                                   atol=.08, rtol=0)
        assert np.count_nonzero(samples[:, replica, 0] >= 3) > 1000
        assert accept[replica, 3] > 1000
    assert swaps[0, 1] > 1000
    verify_fourier_caches(samples, configurations, length)


def test_number_and_order_umbrellas_match_independent_nine_dimensional_product_integrals():
    length, activity, cap, cells = 3., .06, 3, 2
    volume = length ** 3
    penalties = np.array([np.inf, np.inf, .7, 8.])
    stiffness = np.array([0., 0., 2.5, 0.])
    preferred = np.array([0., 0., .7, 0.])
    number_stiffness = np.array([0., 0., .8, .5])
    number_preferred = np.array([0., 0., 2.2, .6])
    vectors = cells * VECTORS
    points = ((qmc.Sobol(9, scramble=True, seed=189212).random_base2(17)
               .reshape((-1, 3, 3)) - .5) * length)
    clear = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        difference = points[:, first] - points[:, second]
        difference -= length * np.rint(difference / length)
        clear.append(np.linalg.norm(difference, axis=1) >= 1.)
    clear = np.array(clear).T
    cavity_clear = np.linalg.norm(points, axis=2) >= 1.
    weights = [[1., activity * volume],
               [1., activity * (volume - 4 * np.pi / 3)]]
    order_numerators = []
    for replica in (2, 3):
        empty = math.exp(-stiffness[replica] * preferred[replica] ** 2
                         - number_stiffness[replica] * number_preferred[replica] ** 2)
        single = activity * volume * math.exp(
            -stiffness[replica] * (1 - preferred[replica]) ** 2
            - number_stiffness[replica] * (1 - number_preferred[replica]) ** 2)
        weights.append([empty, single])
        order_numerators.append(single)
    mode_sum = np.exp(2j * np.pi / length * (points[:, 0] @ vectors.T))
    for n in (2, 3):
        mode_sum += np.exp(2j * np.pi / length * (points[:, n - 1] @ vectors.T))
        order = abs(mode_sum).mean(axis=1) / n
        hard_mask = clear[:, 0] if n == 2 else clear.all(axis=1)
        overlaps = (~clear[:, 0]).astype(float) if n == 2 else (~clear).sum(axis=1)
        cavity_mask = cavity_clear[:, :n].all(axis=1)
        factor = (activity * volume) ** n / math.factorial(n)
        weights[0].append(factor * hard_mask.mean())
        weights[1].append(factor * (hard_mask & cavity_mask).mean())
        for slot, replica in enumerate((2, 3)):
            density = np.exp(-penalties[replica] * overlaps
                             - stiffness[replica] * (order - preferred[replica]) ** 2
                             - number_stiffness[replica] * (n - number_preferred[replica]) ** 2)
            weights[replica].append(factor * density.mean())
            order_numerators[slot] += factor * (order * density).mean()
    probabilities = np.array([np.array(weight) / sum(weight) for weight in weights])
    expected_order = np.array([order_numerators[slot] / sum(weights[replica])
                               for slot, replica in enumerate((2, 3))])

    positions = np.zeros((4, cap, 3))
    positions[:, 0] = [1.2, 1.1, 1.3]
    counts = np.ones(4, dtype=np.int64)
    centers = np.zeros((4, 3))
    identities = np.arange(4)
    diameter, radius = np.ones(4), np.array([0., 1., 0., 0.])
    fields = np.full(4, math.log(activity))
    reference = fcc_reference(4 * cells ** 3, length)
    edges = np.array([[0, 2], [2, 3], [3, 1]], dtype=np.int64)
    cavity_nodes, snapshots = np.array([0, 1], dtype=np.int64), np.arange(4)
    seed_rng(189213)
    arguments = positions, counts, centers, diameter, radius, fields, identities, length
    tail = (0, 2, reference, np.ones(4), penalties, stiffness, preferred,
            edges, cavity_nodes, snapshots, True, True, .30, .2, 1, 1)
    kwargs = dict(number_stiffness=number_stiffness, number_preferred=number_preferred)
    run_chunk(*arguments, 3000, *tail, **kwargs)
    samples, accept, swaps, configurations = run_chunk(*arguments, 40000, *tail, **kwargs)
    for pair in itertools.combinations(range(4), 2):
        indices = list(pair)
        verify_product_number_law(samples[:, indices], probabilities[indices])
    np.testing.assert_allclose(samples[:, 2:, 1].mean(axis=0), expected_order,
                               atol=.025, rtol=0)
    assert accept[0, 7] > 1000
    assert (swaps[:, 1] > 1000).all()
    for replica in range(4):
        assert accept[replica, 3] > 500
    for replica in (0, 1):
        assert np.count_nonzero(samples[:, replica, 0] == 3) > 1000
        for first, second in ((0, 1), (0, 2), (1, 2)):
            selected = samples[:, replica, 0] > max(first, second)
            separation = configurations[selected, replica, first] - configurations[selected, replica, second]
            separation -= length * np.rint(separation / length)
            assert np.all(np.linalg.norm(separation, axis=1) >= 1.)
    assert np.all(samples[:, 1, 6] >= 1.)
    verify_fourier_caches(samples, configurations, length, vectors)
