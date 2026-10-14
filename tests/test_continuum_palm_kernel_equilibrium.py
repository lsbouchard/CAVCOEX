"""Independent capped two-particle quadrature control of the actual Palm kernel.

This is a deliberately capped toy target, not the physical production cap.
Both hard endpoints and one biased soft node are tested against their exact
partition sums; the joint endpoint distribution must be their product.
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import fcc_reference, run_chunk, seed_rng


VECTORS = np.array(
    [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
     [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float,
)


def test_actual_palm_exchange_matches_independent_endpoint_partition_integrals():
    length, activity = 3., .06
    volume = length**3
    penalty, kappa, preferred = .7, 2.5, .65
    # The cavity center can be fixed at the origin in these partition
    # integrals because all pair/order weights are translation invariant.
    coordinates = qmc.Sobol(6, scramble=True, seed=188507).random_base2(18)
    coordinates = (coordinates.reshape((-1, 2, 3)) - .5) * length
    difference = coordinates[:, 0] - coordinates[:, 1]
    difference -= length * np.rint(difference / length)
    pair_clear = np.linalg.norm(difference, axis=1) >= 1.
    cavity_clear = (np.linalg.norm(coordinates, axis=2) >= 1.).all(axis=1)
    two_order = abs(np.cos(np.pi / length * (difference @ VECTORS.T))).mean(axis=1)
    soft_factor = np.exp(-penalty * ~pair_clear - kappa * (two_order - preferred)**2)
    prefactor = (activity * volume)**2 / 2
    sector_weights = [
        np.array([1., activity * volume, prefactor * pair_clear.mean()]),
        np.array([1., activity * (volume - 4 * np.pi / 3),
                  prefactor * (pair_clear & cavity_clear).mean()]),
        np.array([math.exp(-kappa * preferred**2),
                  activity * volume * math.exp(-kappa * (1 - preferred)**2),
                  prefactor * soft_factor.mean()]),
    ]
    probability = np.array([weight / weight.sum() for weight in sector_weights])
    expected_soft_order = (
        sector_weights[2][1] + prefactor * (two_order * soft_factor).mean()
    ) / sector_weights[2].sum()

    positions = np.zeros((3, 2, 3))
    positions[:, 0] = [1.2, 1.1, 1.3]
    counts = np.ones(3, dtype=np.int64)
    centers = np.zeros((3, 3))
    identities = np.arange(3)
    diameter = np.ones(3)
    radius = np.array([0., 1., 0.])
    activity_fields = np.full(3, math.log(activity))
    penalties = np.array([np.inf, np.inf, penalty])
    stiffness = np.array([0., 0., kappa])
    preferences = np.array([0., 0., preferred])
    edges = np.array([[0, 2], [2, 1]], dtype=np.int64)
    cavity_nodes = np.array([0, 1], dtype=np.int64)
    snapshot_nodes = np.arange(3)
    reference = fcc_reference(4, length)
    seed_rng(188508)
    arguments = (
        positions, counts, centers, diameter, radius, activity_fields, identities,
        length,
    )
    tail = (
        0, 2, reference, np.ones(3), penalties, stiffness, preferences,
        edges, cavity_nodes, snapshot_nodes, True,
    )
    run_chunk(*arguments, 2000, *tail)
    samples, accept, _, snapshots = run_chunk(*arguments, 40000, *tail)
    for replica in range(3):
        empirical = np.bincount(samples[:, replica, 0].astype(int), minlength=3) / len(samples)
        np.testing.assert_allclose(empirical, probability[replica], atol=.025, rtol=0)
    empirical_joint = np.zeros((3, 3))
    np.add.at(empirical_joint,
              (samples[:, 0, 0].astype(int), samples[:, 1, 0].astype(int)), 1)
    empirical_joint /= len(samples)
    np.testing.assert_allclose(empirical_joint, np.outer(probability[0], probability[1]),
                               atol=.025, rtol=0)
    assert abs(np.cov(samples[:, 0, 0], samples[:, 1, 0], ddof=0)[0, 1]) < .04
    np.testing.assert_allclose(samples[:, 2, 1].mean(), expected_soft_order,
                               atol=.025, rtol=0)
    assert accept[0, 7] > 1000
    assert accept[0, 6] == 40000

    # Direct complex sums from all stored snapshots validate every cache,
    # including those that later enter the nonzero auxiliary order energy.
    occupied = np.arange(2)[None, None, :] < samples[:, :, 0, None]
    phases = 2 * np.pi / length * (snapshots @ VECTORS.T)
    modes = (np.exp(1j * phases) * occupied[:, :, :, None]).sum(axis=2)
    rebuilt_order = np.divide(abs(modes).mean(axis=2), samples[:, :, 0],
                              out=np.zeros_like(samples[:, :, 0]),
                              where=samples[:, :, 0] > 0)
    np.testing.assert_allclose(samples[:, :, 1], rebuilt_order, atol=1e-10, rtol=0)
    for replica in (0, 1):
        pair_frames = samples[:, replica, 0] == 2
        separation = snapshots[pair_frames, replica, 0] - snapshots[pair_frames, replica, 1]
        separation -= length * np.rint(separation / length)
        assert np.all(np.linalg.norm(separation, axis=1) >= 1.)
    assert np.all(samples[:, 1, 6] >= 1.)
    if counts[1]:
        displacement = positions[1, :counts[1]] - centers[1]
        displacement -= length * np.rint(displacement / length)
        assert np.all(np.linalg.norm(displacement, axis=1) >= 1.)
