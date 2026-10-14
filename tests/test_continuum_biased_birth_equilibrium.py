"""Independent equilibrium controls of the implemented coordinate-biased GCMC.

An ideal N<=5 target exercises nonuniform births from N>=2. A hard/cavity
N<=3 target plus a biased soft replica is integrated independently in nine
dimensions. Both cases test joint product laws and coordinate cache updates.
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import fcc_reference, run_chunk, seed_rng


VECTORS = np.array(
    [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
     [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float,
)


def check_reference_fourier_modes(samples, snapshots, length):
    cap = snapshots.shape[2]
    modes = np.zeros(samples.shape[:2] + (7,), dtype=complex)
    # Accumulate one coordinate slot at a time to keep temporary memory bounded.
    for particle in range(cap):
        phase = 2 * np.pi / length * (snapshots[:, :, particle] @ VECTORS.T)
        modes += np.exp(1j * phase) * (samples[:, :, 0, None] > particle)
    expected = np.divide(abs(modes).mean(axis=2), samples[:, :, 0],
                         out=np.zeros_like(samples[:, :, 0]),
                         where=samples[:, :, 0] > 0)
    np.testing.assert_allclose(samples[:, :, 1], expected, atol=1e-10, rtol=0)


def check_product_number_law(samples, probability, tolerance=.025):
    for replica, target in enumerate(probability):
        empirical = np.bincount(samples[:, replica, 0].astype(int), minlength=len(target)) / len(samples)
        np.testing.assert_allclose(empirical, target, atol=tolerance, rtol=0)
    empirical_joint = np.zeros((len(probability[0]), len(probability[1])))
    np.add.at(empirical_joint,
              (samples[:, 0, 0].astype(int), samples[:, 1, 0].astype(int)), 1)
    empirical_joint /= len(samples)
    np.testing.assert_allclose(empirical_joint, np.outer(probability[0], probability[1]),
                               atol=tolerance, rtol=0)
    assert abs(np.cov(samples[:, 0, 0], samples[:, 1, 0], ddof=0)[0, 1]) < .06


def test_biased_birth_preserves_capped_ideal_gas_and_product_number_laws():
    length, cap = 3., 5
    poisson_means = np.array([3.5, 2.3])
    fields = np.log(poisson_means / length**3)
    probability = []
    for target_mean in poisson_means:
        weights = np.array([target_mean**n / math.factorial(n) for n in range(cap + 1)])
        probability.append(weights / weights.sum())
    positions = np.zeros((2, cap, 3))
    counts = np.zeros(2, dtype=np.int64)
    centers = np.zeros((2, 3))
    identities = np.arange(2)
    diameter, radius = np.zeros(2), np.zeros(2)
    penalty, stiffness, preferred = np.full(2, np.inf), np.zeros(2), np.zeros(2)
    edges = np.array([[0, 1]], dtype=np.int64)
    cavity_nodes = np.array([0], dtype=np.int64)
    snapshots = np.arange(2)
    reference = fcc_reference(4, length)
    seed_rng(188511)
    arguments = positions, counts, centers, diameter, radius, fields, identities, length
    tail = (0, 3, reference, np.ones(2), penalty, stiffness, preferred,
            edges, cavity_nodes, snapshots, False, True, .35, .2)
    run_chunk(*arguments, 3000, *tail)
    samples, accept, _, coordinates = run_chunk(*arguments, 60000, *tail)
    check_product_number_law(samples, probability)
    for replica, target in enumerate(probability):
        expected_mean = np.dot(np.arange(cap + 1), target)
        np.testing.assert_allclose(samples[:, replica, 0].mean(), expected_mean, atol=.08, rtol=0)
        assert np.count_nonzero(samples[:, replica, 0] >= 3) > 1000
        assert accept[replica, 3] > 1000
    check_reference_fourier_modes(samples, coordinates, length)


def test_biased_birth_hard_cavity_and_soft_auxiliary_match_nine_dimensional_quadrature():
    length, activity = 3., .06
    volume, cap = length**3, 3
    penalty, kappa, preferred = .7, 1.7, .65
    points = (qmc.Sobol(9, scramble=True, seed=188512).random_base2(17)
              .reshape((-1, 3, 3)) - .5) * length
    clear = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        separation = points[:, first] - points[:, second]
        separation -= length * np.rint(separation / length)
        clear.append(np.linalg.norm(separation, axis=1) >= 1.)
    clear = np.array(clear).T
    cavity_clear = np.linalg.norm(points, axis=2) >= 1.
    weights = [
        [1., activity * volume],
        [1., activity * (volume - 4 * np.pi / 3)],
        [math.exp(-kappa * preferred**2),
         activity * volume * math.exp(-kappa * (1 - preferred)**2)],
    ]
    mode_sum = np.exp(2j * np.pi / length * (points[:, 0] @ VECTORS.T))
    expected_soft_order_numerator = weights[2][1]
    for n in (2, 3):
        mode_sum += np.exp(2j * np.pi / length * (points[:, n - 1] @ VECTORS.T))
        order = abs(mode_sum).mean(axis=1) / n
        pair_mask = clear[:, 0] if n == 2 else clear.all(axis=1)
        overlaps = (~clear[:, 0]).astype(float) if n == 2 else (~clear).sum(axis=1)
        cavity_mask = cavity_clear[:, :n].all(axis=1)
        soft_factor = np.exp(-penalty * overlaps - kappa * (order - preferred)**2)
        prefactor = (activity * volume)**n / math.factorial(n)
        weights[0].append(prefactor * pair_mask.mean())
        weights[1].append(prefactor * (pair_mask & cavity_mask).mean())
        weights[2].append(prefactor * soft_factor.mean())
        expected_soft_order_numerator += prefactor * (order * soft_factor).mean()
    probability = np.array([np.array(w) / sum(w) for w in weights])
    expected_soft_order = expected_soft_order_numerator / sum(weights[2])

    positions = np.zeros((3, cap, 3))
    positions[:, 0] = [1.2, 1.1, 1.3]
    counts = np.ones(3, dtype=np.int64)
    centers = np.zeros((3, 3))
    identities = np.arange(3)
    diameter = np.ones(3)
    radius = np.array([0., 1., 0.])
    fields = np.full(3, math.log(activity))
    penalties = np.array([np.inf, np.inf, penalty])
    stiffness = np.array([0., 0., kappa])
    preferences = np.array([0., 0., preferred])
    edges = np.array([[0, 2], [2, 1]], dtype=np.int64)
    cavity_nodes, snapshot_nodes = np.array([0, 1], dtype=np.int64), np.arange(3)
    reference = fcc_reference(4, length)
    seed_rng(188513)
    arguments = positions, counts, centers, diameter, radius, fields, identities, length
    tail = (0, 2, reference, np.ones(3), penalties, stiffness, preferences,
            edges, cavity_nodes, snapshot_nodes, True, True, .30, .2)
    run_chunk(*arguments, 3000, *tail)
    samples, accept, _, coordinates = run_chunk(*arguments, 50000, *tail)
    check_product_number_law(samples, probability)
    np.testing.assert_allclose(samples[:, 2, 1].mean(), expected_soft_order, atol=.025, rtol=0)
    assert accept[0, 7] > 1000
    assert np.count_nonzero(samples[:, 0, 0] == 3) > 1000
    assert np.count_nonzero(samples[:, 1, 0] == 3) > 1000
    check_reference_fourier_modes(samples, coordinates, length)
    for replica in (0, 1):
        for first, second in ((0, 1), (0, 2), (1, 2)):
            selected = samples[:, replica, 0] > max(first, second)
            separation = coordinates[selected, replica, first] - coordinates[selected, replica, second]
            separation -= length * np.rint(separation / length)
            assert np.all(np.linalg.norm(separation, axis=1) >= 1.)
    assert np.all(samples[:, 1, 6] >= 1.)
