"""Independent hard-activity-path law; explicitly truncated N<=3 target.

This uses unequal chemical potentials, a dilute hard/ideal regeneration link,
and a matched cavity. It is not a dense fluid--solid equilibrium calculation.
"""
import math
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from scripts import continuum_simulated_tempering as tempering
from test_single_walker_continuum_equilibrium import _independent_partition_table


def test_unequal_activity_hard_path_matches_joint_conditional_and_work_laws():
    length, cap, cells, reference_activity = 3., 3, 2, .06
    activities = np.array([.06, .06, .006, .11, .006])
    fields = np.log(activities)
    penalties = np.array([np.inf, np.inf, np.inf, np.inf, 0.])
    zero = np.zeros(5)
    reference_sectors, _ = _independent_partition_table(
        length, reference_activity, cap, cells, penalties, zero, zero, zero, zero)
    sectors = reference_sectors * (activities[:, None] / reference_activity) ** np.arange(cap + 1)
    z = sectors.sum(axis=1)
    number_laws = sectors / z[:, None]
    proportions = np.array([1.1, .8, 1.3, .7, 1.2])
    log_weights = np.log(proportions) - np.log(z)
    expected_joint = proportions[:, None] / proportions.sum() * number_laws
    # A native-shaped activity path plus two endpoint leaves, hence 1/2/3
    # neighbor degrees. The ideal and dilute hard node have identical activity.
    edges = np.array([[4, 2], [2, 0], [0, 3], [0, 1]], dtype=np.int64)
    neighbors, edge_table, degrees = tempering.adjacency(edges, 5)
    assert set(degrees) == {1, 2, 3}
    positions = np.zeros((1, cap, 3))
    counts = np.zeros(1, dtype=np.int64)
    centers = np.zeros((1, 3))
    reference = tempering.kernel.fcc_reference(32, length)
    cdf = tempering.capped_poisson_cdf(fields[4] + math.log(length ** 3), cap)
    tail = (reference, length, np.ones(5), np.array([0., 1., 0., 0., 0.]),
            fields, penalties, zero, zero, zero, zero, np.full(5, .8),
            np.array([True, True, True, True, False]), 0, 1, 4,
            neighbors, edge_table, degrees, log_weights, cdf)
    tempering.kernel.seed_rng(260930411)
    warm = tempering.run_tempering_chunk(positions, counts, centers, 4, 0, 0, 5000, *tail,
                                         coordinate_stride=7, biased_birth=True,
                                         birth_width=.3, birth_uniform_fraction=.2)
    node, cycle = warm[-2:]
    steps = 160000
    coordinate_centers = np.empty((len(np.arange(7 - (5000 % 7) - 1, steps, 7)), 3))
    result = tempering.run_tempering_chunk(positions, counts, centers, node, cycle, 5000,
                                           steps, *tail, coordinate_stride=7,
                                           biased_birth=True, birth_width=.3,
                                           birth_uniform_fraction=.2,
                                           coordinate_centers=coordinate_centers)
    trace, local, moves, configurations, clocks, coordinate_labels, _, _ = result
    labels, numbers = trace[:, 1].astype(int), trace[:, 2].astype(int)
    joint = np.bincount(labels * (cap + 1) + numbers, minlength=5 * (cap + 1))
    joint = joint.reshape((5, cap + 1)) / len(trace)
    np.testing.assert_allclose(joint, expected_joint, atol=.012, rtol=0)
    np.testing.assert_allclose(joint.sum(axis=1), proportions / proportions.sum(), atol=.015, rtol=0)
    for node in range(5):
        selected = labels == node
        assert selected.sum() > 15000
        observed = np.bincount(numbers[selected], minlength=cap + 1) / selected.sum()
        np.testing.assert_allclose(observed, number_laws[node], atol=.025, rtol=0)
        np.testing.assert_allclose(trace[selected, 2].mean(),
                                   number_laws[node] @ np.arange(cap + 1), atol=.04, rtol=0)
    assert (moves[:, 1] > 1000).all()
    assert local[3] > 10000
    assert np.count_nonzero((labels == 1) & (numbers == cap)) > 500
    # Verify work independently from all full-clock dwells in completed cycles,
    # never from transition counts or means of separate cycle fractions.
    events = np.flatnonzero(trace[:, 11])
    assert len(events) > 20000
    assert np.all(labels[events] == 4)
    complete = trace[events[0]:events[-1]]
    db, dc = np.count_nonzero(complete[:, 1] == 0), np.count_nonzero(complete[:, 1] == 1)
    work = math.log(db / dc) + log_weights[1] - log_weights[0]
    expected_work = -math.log(z[1] / z[0])
    np.testing.assert_allclose(work, expected_work, atol=.07, rtol=0)
    assert np.array_equal(np.diff(trace[:, 10]), trace[1:, 11])
    rows = clocks.astype(int) - 5001
    assert np.array_equal(coordinate_labels, labels[rows])
    for index, row in enumerate(rows):
        n = numbers[row]
        host = configurations[index, :n]
        difference = host - coordinate_centers[index]
        difference -= length * np.rint(difference / length)
        minimum = np.linalg.norm(difference, axis=1).min() if n else length
        np.testing.assert_allclose(minimum, trace[row, 7], atol=1e-10, rtol=0)
        if coordinate_labels[index] != 4:
            for first in range(n):
                for second in range(first):
                    delta = host[first] - host[second]
                    delta -= length * np.rint(delta / length)
                    assert np.linalg.norm(delta) >= 1.
        if coordinate_labels[index] == 1:
            assert minimum >= 1.
