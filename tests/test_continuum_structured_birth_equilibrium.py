"""Independent finite-target control of vacancy proposals on auxiliaries.

The newly eligible branches are a soft pair/order-biased target and a
strong pair-penalty target with no order bias.  Both are compared against
nine-dimensional partition integrals, not against an occupancy model.
"""

import itertools
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pytest
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import fcc_reference, run_chunk, seed_rng

from test_continuum_vacancy_birth_equilibrium import (
    VECTORS, verify_fourier_caches, verify_product_number_law,
)


@pytest.mark.parametrize("cells", [1, 2])
def test_structured_vacancy_births_match_soft_order_and_penetrable_pair_targets(cells):
    length, activity, cap = 3., .06, 3
    volume = length**3
    penalties = np.array([np.inf, np.inf, .7, 8.])
    stiffness = np.array([0., 0., 2.5, 0.])
    preferred = np.array([0., 0., .7, 0.])
    vectors = cells * VECTORS
    points = ((qmc.Sobol(9, scramble=True, seed=188712 + cells).random_base2(17)
               .reshape((-1, 3, 3)) - .5) * length)
    clear = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        difference = points[:, first] - points[:, second]
        difference -= length * np.rint(difference / length)
        clear.append(np.linalg.norm(difference, axis=1) >= 1.)
    clear = np.array(clear).T
    cavity_clear = np.linalg.norm(points, axis=2) >= 1.
    weights = [
        [1., activity * volume],
        [1., activity * (volume - 4 * np.pi / 3)],
    ]
    order_numerators = []
    for replica in (2, 3):
        empty = math.exp(-stiffness[replica] * preferred[replica]**2)
        single = activity * volume * math.exp(-stiffness[replica] * (1 - preferred[replica])**2)
        weights.append([empty, single])
        order_numerators.append(single)
    mode_sum = np.exp(2j * np.pi / length * (points[:, 0] @ vectors.T))
    for n in (2, 3):
        mode_sum += np.exp(2j * np.pi / length * (points[:, n - 1] @ vectors.T))
        order = abs(mode_sum).mean(axis=1) / n
        hard_mask = clear[:, 0] if n == 2 else clear.all(axis=1)
        overlaps = (~clear[:, 0]).astype(float) if n == 2 else (~clear).sum(axis=1)
        cavity_mask = cavity_clear[:, :n].all(axis=1)
        factor = (activity * volume)**n / math.factorial(n)
        weights[0].append(factor * hard_mask.mean())
        weights[1].append(factor * (hard_mask & cavity_mask).mean())
        for slot, replica in enumerate((2, 3)):
            density = np.exp(-penalties[replica] * overlaps
                             - stiffness[replica] * (order - preferred[replica])**2)
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
    activity_fields = np.full(4, math.log(activity))
    reference = fcc_reference(4 * cells**3, length)
    edges = np.array([[0, 2], [2, 3], [3, 1]], dtype=np.int64)
    cavity_nodes, snapshots = np.array([0, 1], dtype=np.int64), np.arange(4)
    seed_rng(188713 + cells)
    arguments = positions, counts, centers, diameter, radius, activity_fields, identities, length
    tail = (0, 2, reference, np.ones(4), penalties, stiffness, preferred,
            edges, cavity_nodes, snapshots, True, True, .30, .2, 1, 1)
    run_chunk(*arguments, 3000, *tail)
    samples, accept, swaps, configurations = run_chunk(*arguments, 60000 if cells == 1 else 40000, *tail)
    for pair in itertools.combinations(range(4), 2):
        indices = list(pair)
        verify_product_number_law(samples[:, indices], probabilities[indices])
    np.testing.assert_allclose(samples[:, 2:, 1].mean(axis=0), expected_order,
                               atol=.025, rtol=0)
    assert accept[0, 7] > 1000
    assert (swaps[:, 1] > 1000).all()
    for replica in range(4):
        assert accept[replica, 3] > 1000
        assert np.count_nonzero(samples[:, replica, 0] == 3) > 1000
    verify_fourier_caches(samples, configurations, length, vectors)
    for replica in (0, 1):
        for first, second in ((0, 1), (0, 2), (1, 2)):
            selected = samples[:, replica, 0] > max(first, second)
            separation = configurations[selected, replica, first] - configurations[selected, replica, second]
            separation -= length * np.rint(separation / length)
            assert np.all(np.linalg.norm(separation, axis=1) >= 1.)
    assert np.all(samples[:, 1, 6] >= 1.)
