"""Independent equilibrium laws for the FCC-vacancy *proposal*, not target.

The ideal capped N<=5 law exercises registry changes and N>=2 births.
The capped hard/cavity N<=3 law is integrated independently in nine
dimensions, alongside a soft/order-biased auxiliary target.  No registry or
occupancy enters the reference partition integrals.  These finite controls
test kernel correctness; they do not certify dense phase mixing.
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import fcc_reference, run_chunk, seed_rng


VECTORS = np.array(
    [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
     [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float,
)


def verify_product_number_law(samples, probability, tolerance=.025):
    for replica, expected in enumerate(probability):
        observed = np.bincount(samples[:, replica, 0].astype(int),
                               minlength=len(expected)) / len(samples)
        np.testing.assert_allclose(observed, expected, atol=tolerance, rtol=0)
    joint = np.zeros((len(probability[0]), len(probability[1])))
    np.add.at(joint, (samples[:, 0, 0].astype(int),
                     samples[:, 1, 0].astype(int)), 1)
    np.testing.assert_allclose(joint / len(samples),
                               np.outer(probability[0], probability[1]),
                               atol=tolerance, rtol=0)
    assert abs(np.cov(samples[:, 0, 0], samples[:, 1, 0], ddof=0)[0, 1]) < .06


def verify_fourier_caches(samples, configurations, length, vectors=VECTORS):
    modes = np.zeros(samples.shape[:2] + (7,), dtype=complex)
    for slot in range(configurations.shape[2]):
        phases = 2 * np.pi / length * (configurations[:, :, slot] @ vectors.T)
        modes += np.exp(1j * phases) * (samples[:, :, 0, None] > slot)
    expected = np.divide(abs(modes).mean(axis=2), samples[:, :, 0],
                         out=np.zeros_like(samples[:, :, 0]),
                         where=samples[:, :, 0] > 0)
    np.testing.assert_allclose(samples[:, :, 1], expected, atol=1e-10, rtol=0)


def test_vacancy_birth_preserves_capped_poisson_and_product_laws():
    length, cap = 3., 5
    means = np.array([3.5, 2.3])
    probabilities = []
    for mean in means:
        weights = np.array([mean**n / math.factorial(n) for n in range(cap + 1)])
        probabilities.append(weights / sum(weights))
    positions = np.zeros((2, cap, 3))
    counts = np.zeros(2, dtype=np.int64)
    centers = np.zeros((2, 3))
    identities = np.arange(2)
    diameter, radius = np.zeros(2), np.zeros(2)
    activity = np.log(means / length**3)
    penalty = np.full(2, np.inf)
    stiffness, preferred = np.zeros(2), np.zeros(2)
    reference = fcc_reference(4, length)
    edges = np.array([[0, 1]], dtype=np.int64)
    cavity_nodes, snapshots = np.array([0], dtype=np.int64), np.arange(2)
    seed_rng(188611)
    arguments = positions, counts, centers, diameter, radius, activity, identities, length
    tail = (0, 3, reference, np.ones(2), penalty, stiffness, preferred,
            edges, cavity_nodes, snapshots, False, True, .35, .2, 1)
    run_chunk(*arguments, 3000, *tail)
    samples, accept, _, configurations = run_chunk(*arguments, 60000, *tail)
    verify_product_number_law(samples, probabilities)
    for replica, probability in enumerate(probabilities):
        np.testing.assert_allclose(samples[:, replica, 0].mean(),
                                   np.dot(np.arange(cap + 1), probability),
                                   atol=.08, rtol=0)
        assert np.count_nonzero(samples[:, replica, 0] >= 3) > 1000
        assert accept[replica, 3] > 1000
    verify_fourier_caches(samples, configurations, length)


def test_vacancy_birth_matches_hard_cavity_and_auxiliary_partition_integrals():
    length, activity, cap = 3., .06, 3
    volume = length**3
    penalty, kappa, preferred = .7, 1.7, .65
    points = ((qmc.Sobol(9, scramble=True, seed=188612).random_base2(17)
               .reshape((-1, 3, 3)) - .5) * length)
    pair_clear = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        difference = points[:, first] - points[:, second]
        difference -= length * np.rint(difference / length)
        pair_clear.append(np.linalg.norm(difference, axis=1) >= 1.)
    pair_clear = np.array(pair_clear).T
    cavity_clear = np.linalg.norm(points, axis=2) >= 1.
    weights = [
        [1., activity * volume],
        [1., activity * (volume - 4 * np.pi / 3)],
        [math.exp(-kappa * preferred**2),
         activity * volume * math.exp(-kappa * (1 - preferred)**2)],
    ]
    mode_sum = np.exp(2j * np.pi / length * (points[:, 0] @ VECTORS.T))
    order_numerator = weights[2][1]
    for n in (2, 3):
        mode_sum += np.exp(2j * np.pi / length * (points[:, n - 1] @ VECTORS.T))
        order = abs(mode_sum).mean(axis=1) / n
        hard_mask = pair_clear[:, 0] if n == 2 else pair_clear.all(axis=1)
        overlaps = (~pair_clear[:, 0]).astype(float) if n == 2 else (~pair_clear).sum(axis=1)
        cavity_mask = cavity_clear[:, :n].all(axis=1)
        soft_weight = np.exp(-penalty * overlaps - kappa * (order - preferred)**2)
        factor = (activity * volume)**n / math.factorial(n)
        weights[0].append(factor * hard_mask.mean())
        weights[1].append(factor * (hard_mask & cavity_mask).mean())
        weights[2].append(factor * soft_weight.mean())
        order_numerator += factor * (order * soft_weight).mean()
    probabilities = np.array([np.array(weight) / sum(weight) for weight in weights])
    expected_order = order_numerator / sum(weights[2])

    positions = np.zeros((3, cap, 3))
    positions[:, 0] = [1.2, 1.1, 1.3]
    counts = np.ones(3, dtype=np.int64)
    centers = np.zeros((3, 3))
    identities = np.arange(3)
    diameter, radius = np.ones(3), np.array([0., 1., 0.])
    fields = np.full(3, math.log(activity))
    penalties = np.array([np.inf, np.inf, penalty])
    stiffness, preferences = np.array([0., 0., kappa]), np.array([0., 0., preferred])
    reference = fcc_reference(4, length)
    edges = np.array([[0, 2], [2, 1]], dtype=np.int64)
    cavity_nodes, snapshots = np.array([0, 1], dtype=np.int64), np.arange(3)
    seed_rng(188613)
    arguments = positions, counts, centers, diameter, radius, fields, identities, length
    tail = (0, 2, reference, np.ones(3), penalties, stiffness, preferences,
            edges, cavity_nodes, snapshots, True, True, .30, .2, 1)
    run_chunk(*arguments, 3000, *tail)
    samples, accept, _, configurations = run_chunk(*arguments, 50000, *tail)
    verify_product_number_law(samples, probabilities)
    np.testing.assert_allclose(samples[:, 2, 1].mean(), expected_order, atol=.025, rtol=0)
    assert accept[0, 7] > 1000
    for replica in (0, 1):
        assert np.count_nonzero(samples[:, replica, 0] == 3) > 1000
        for first, second in ((0, 1), (0, 2), (1, 2)):
            selected = samples[:, replica, 0] > max(first, second)
            difference = configurations[selected, replica, first] - configurations[selected, replica, second]
            difference -= length * np.rint(difference / length)
            assert np.all(np.linalg.norm(difference, axis=1) >= 1.)
    assert np.all(samples[:, 1, 6] >= 1.)
    verify_fourier_caches(samples, configurations, length)
