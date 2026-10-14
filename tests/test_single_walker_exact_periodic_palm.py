"""Analytic, untruncated physical Palm control on a very small torus.

With L=sigma=1, every minimum-image pair separation is at most sqrt(3)/2
< sigma. Thus an unrestricted hard-sphere host has N<=1, and a matched
cavity has N=0. K=2 does not truncate either physical law. The ideal
auxiliary is deliberately capped and compared with its own capped law.
This is not a phase-coexistence model. No contact/virial theorem is used:
the geometric hypothesis sigma<L/2 of that theorem fails here.
"""
import math
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from scripts import continuum_simulated_tempering as tempering


def test_exact_periodic_palm_number_fano_singleton_and_cycle_work_laws():
    length, sigma, volume, cap = 1., 1., 1., 2
    assert math.sqrt(3) * length / 2 < sigma
    activities = np.array([.8, .8, .2])
    fields = np.log(activities)
    # Analytic grand sectors, not a numerical partition calculation.
    sectors = np.array([[1., activities[0] * volume, 0.],
                        [1., 0., 0.],
                        [1., activities[2] * volume,
                         (activities[2] * volume) ** 2 / 2]])
    partition = sectors.sum(axis=1)
    conditional = sectors / partition[:, None]
    proportions = np.array([1.1, .8, 1.3])
    log_weights = np.log(proportions) - np.log(partition)
    expected_joint = proportions[:, None] / proportions.sum() * conditional
    # The tagged physical edge and an ideal link have unequal node degrees.
    edges = np.array([[0, 1], [0, 2]], dtype=np.int64)
    neighbors, edge_table, degrees = tempering.adjacency(edges, 3)
    assert np.array_equal(degrees, [2, 1, 1])
    zero = np.zeros(3)
    positions = np.zeros((1, cap, 3))
    counts = np.zeros(1, dtype=np.int64)
    centers = np.zeros((1, 3))
    reference = tempering.kernel.fcc_reference(4, length)
    cdf = tempering.capped_poisson_cdf(math.log(activities[2] * volume), cap)
    tail = (reference, length, np.full(3, sigma), np.array([0., sigma, 0.]),
            fields, np.array([np.inf, np.inf, 0.]), zero, zero, zero, zero,
            np.full(3, .25), np.array([True, True, False]), 0, 1, 2,
            neighbors, edge_table, degrees, log_weights, cdf)
    tempering.kernel.seed_rng(260930501)
    warm_steps, steps, stride = 2000, 80000, 11
    warm = tempering.run_tempering_chunk(
        positions, counts, centers, 2, 0, 0, warm_steps, *tail,
        coordinate_stride=stride, biased_birth=True,
        birth_width=.12, birth_uniform_fraction=.2)
    node, cycle = warm[-2:]
    coordinate_count = np.count_nonzero(np.arange(warm_steps + 1,
                                                 warm_steps + steps + 1) % stride == 0)
    coordinate_centers = np.empty((coordinate_count, 3))
    result = tempering.run_tempering_chunk(
        positions, counts, centers, node, cycle, warm_steps, steps, *tail,
        coordinate_stride=stride, biased_birth=True, birth_width=.12,
        birth_uniform_fraction=.2, coordinate_centers=coordinate_centers)
    trace, local, moves, configurations, clocks, labels_at_coordinates, _, _ = result
    labels, numbers = trace[:, 1].astype(int), trace[:, 2].astype(int)
    joint = np.bincount(labels * (cap + 1) + numbers, minlength=3 * (cap + 1))
    joint = joint.reshape((3, cap + 1)) / steps
    np.testing.assert_allclose(joint, expected_joint, atol=.012, rtol=0)
    for label in range(3):
        selected = labels == label
        assert selected.sum() > 15000
        observed = np.bincount(numbers[selected], minlength=cap + 1) / selected.sum()
        np.testing.assert_allclose(observed, conditional[label], atol=.018, rtol=0)
    base, cavity = labels == 0, labels == 1
    assert set(numbers[base]) == {0, 1}
    assert (numbers[cavity] == 0).all()
    assert np.count_nonzero((labels == 2) & (numbers == 2)) > 300
    assert (moves[:, 1] > 1000).all() and local[3] > 1000
    # The binary implementation label B>.45 is 1 for a singleton and 0 for
    # an empty host; it is not interpreted as a fluid/solid classification.
    for selected in (base, cavity):
        np.testing.assert_allclose(trace[selected, 3], numbers[selected], atol=1e-12, rtol=0)
        assert np.array_equal(trace[selected, 8], -numbers[selected])
    mean_base = numbers[base].mean()
    mean_cavity = numbers[cavity].mean()
    expected_mean = activities[0] * volume / partition[0]
    np.testing.assert_allclose(mean_base, expected_mean, atol=.018, rtol=0)
    # Exact unrestricted reduced-Palm number law and Fano correction.
    fano = numbers[base].var() / mean_base
    np.testing.assert_allclose(mean_cavity - mean_base, fano - 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(fano, 1 / partition[0], atol=.018, rtol=0)
    palm_label = (np.mean(numbers[base] * (trace[base, 3] > .45))
                  + np.mean(trace[base, 8])) / mean_base
    assert palm_label == 0. and np.mean(trace[cavity, 3] > .45) == 0.
    events = np.flatnonzero(trace[:, 11])
    assert len(events) > 20000 and (labels[events] == 2).all()
    assert np.array_equal(np.diff(trace[:, 10]), trace[1:, 11])
    complete = trace[events[0]:events[-1]]
    db, dc = np.count_nonzero(complete[:, 1] == 0), np.count_nonzero(complete[:, 1] == 1)
    dwell_work = math.log(db / dc) + log_weights[1] - log_weights[0]
    exact_work = math.log(partition[0] / partition[1])
    palm_work = -math.log(mean_base / (activities[0] * volume))
    np.testing.assert_allclose([dwell_work, palm_work], exact_work, atol=.05, rtol=0)
    # Independent sparse-coordinate alignment, including the cavity center.
    rows = clocks.astype(int) - warm_steps - 1
    assert np.array_equal(labels_at_coordinates, labels[rows])
    for index, row in enumerate(rows):
        n = numbers[row]
        host = configurations[index, :n]
        difference = host - coordinate_centers[index]
        difference -= length * np.rint(difference / length)
        minimum = np.linalg.norm(difference, axis=1).min() if n else length
        np.testing.assert_allclose(minimum, trace[row, 7], atol=1e-12, rtol=0)
        if labels[row] in (0, 1):
            assert n <= 1
            assert trace[row, 8] == -n
        if labels[row] == 1:
            assert n == 0 and minimum >= sigma
