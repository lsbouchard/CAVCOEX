"""Independent small-system normalization control for the auxiliary umbrella.

For N <= 2 the partition sum reduces to one uniform relative-position
integral.  This exercises current-N Bragg normalization and all variable-N
cache updates without relying on a dense-fluid sampling result.
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import fcc_reference, run_chunk, seed_rng


def test_two_particle_umbrella_matches_independent_partition_quadrature():
    length = 3.0
    volume = length**3
    activity = 0.06
    penalties = np.array([0.4, 2.7, 1.2])
    stiffness = np.array([0.8, 3.2, 1.8])
    preferred = np.array([0.35, 0.70, 0.50])
    vectors = np.array(
        [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    relative = (qmc.Sobol(3, scramble=True, seed=81621).random_base2(17) - 0.5) * length
    bragg_two = np.abs(np.cos(np.pi / length * (relative @ vectors.T))).mean(axis=1)
    overlaps = (np.linalg.norm(relative, axis=1) < 1).astype(float)
    expected = []
    expected_order = []
    for penalty, kappa, target in zip(penalties, stiffness, preferred):
        two_particle_factor = np.exp(-penalty * overlaps - kappa * (bragg_two - target)**2)
        weights = np.array(
            [
                math.exp(-kappa * target**2),
                activity * volume * math.exp(-kappa * (1 - target)**2),
                (activity * volume)**2 / 2
                * two_particle_factor.mean(),
            ]
        )
        expected.append(weights / weights.sum())
        expected_order.append(
            (weights[1] + (activity * volume)**2 / 2
             * (bragg_two * two_particle_factor).mean()) / weights.sum()
        )

    positions = np.zeros((3, 2, 3))
    positions[:, 0] = 0.7
    counts = np.ones(3, dtype=np.int64)
    centers = np.zeros((3, 3))
    identities = np.arange(3)
    reference = fcc_reference(4, length)
    sigma = np.ones(3)
    radius = np.zeros(3)
    chemical_field = np.full(3, math.log(activity))
    exchange_edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    cavity_nodes = np.array([1], dtype=np.int64)
    seed_rng(188502)
    run_chunk(
        positions, counts, centers, sigma, radius, chemical_field, identities,
        length, 2000, 1, 2, reference, np.ones(3), penalties, stiffness, preferred,
        exchange_edges, cavity_nodes,
    )
    samples, _, swaps, _ = run_chunk(
        positions, counts, centers, sigma, radius, chemical_field, identities,
        length, 40000, 1, 2, reference, np.ones(3), penalties, stiffness, preferred,
        exchange_edges, cavity_nodes,
    )
    for replica, target_probability in enumerate(expected):
        observed = np.bincount(samples[:, replica, 0].astype(int), minlength=3) / len(samples)
        np.testing.assert_allclose(observed, target_probability, atol=0.025, rtol=0)
        np.testing.assert_allclose(samples[:, replica, 1].mean(), expected_order[replica],
                                   atol=0.025, rtol=0)
    assert np.all(swaps[:, 1] > 100)
