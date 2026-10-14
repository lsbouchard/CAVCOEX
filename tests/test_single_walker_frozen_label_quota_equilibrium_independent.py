"""Actual frozen nonuniform-b continuum law with a positive label quota.

All expectations come from the independent N<=3 partition integrals, not from
calibrated dense histories or assumed label flatness. This toy cap truncates
the physical laws; no unrestricted Palm/GNZ or coexistence claim is made.
"""
import math
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import numpy as np

from scripts import continuum_simulated_tempering as tempering
from test_single_walker_continuum_equilibrium import VECTORS, _independent_partition_table
from test_single_walker_gibbs_continuum_equilibrium import _enable_gibbs_with_independent_protocol_checks


def test_frozen_endpoint_priority_weights_preserve_full_conditional_partition_and_cycle_work_laws(monkeypatch):
    monkeypatch.delenv('CAVCOEX_ST_CONTROL_EXPORT', raising=False)
    length, activity, cap, cells = 3., .06, 3, 2
    penalty = np.array([np.inf, np.inf, .7, 8., 0.])
    kappa = np.array([0., 0., 2.5, 0., 0.])
    preferred = np.array([0., 0., .7, 0., 0.])
    nu = np.array([0., 0., .8, .5, 0.])
    nstar = np.array([0., 0., 2.2, .6, 0.])
    sectors, expected_order = _independent_partition_table(
        length, activity, cap, cells, penalty, kappa, preferred, nu, nstar)
    normalizer = sectors.sum(axis=1)
    conditional_number = sectors / normalizer[:, None]
    quota = np.array([.30, .30, .19, .19, .02])
    # Independent normalizers define arbitrary finite frozen weights. This is
    # not a claim that adaptive calibration obtained them in a dense run.
    weights = np.log(quota) - np.log(normalizer)
    weights -= weights[4]
    immutable_weights = weights.copy()
    expected_joint = quota[:, None] * conditional_number
    edges = np.array([[0, 2], [2, 3], [3, 4], [4, 1], [0, 1], [2, 4]], dtype=np.int64)
    neighbors, edge_table, degrees = tempering.adjacency(edges, 5)
    positions = np.zeros((1, cap, 3))
    counts = np.zeros(1, dtype=np.int64)
    centers = np.zeros((1, 3))
    reference = tempering.kernel.fcc_reference(32, length)
    diameter, radius = np.ones(5), np.array([0., 1., 0., 0., 0.])
    fields = np.full(5, math.log(activity))
    displacement = np.full(5, .8)
    physical = np.array([True, True, False, False, False])
    cdf = tempering.capped_poisson_cdf(math.log(activity * length ** 3), cap)
    tail = (reference, length, diameter, radius, fields, penalty, kappa, preferred,
            nu, nstar, displacement, physical, 0, 1, 4, neighbors, edge_table,
            degrees, weights, cdf)
    checked_chunks = _enable_gibbs_with_independent_protocol_checks(monkeypatch)
    tempering.kernel.seed_rng(260930441)
    counters = np.zeros(2, dtype=np.int64)
    warm = tempering.run_tempering_chunk(positions, counts, centers, 4, 0, 0, 5000,
        *tail, coordinate_stride=11, biased_birth=True, birth_width=.3,
        birth_uniform_fraction=.2, calibration_label_quota=quota,
        vacancy_relocation_probability=1., vacancy_relocation_counts=counters)
    node, cycle = warm[-2:]
    steps, clock = 200000, 5000
    coordinate_centers = np.empty((len(np.arange(11 - clock % 11 - 1, steps, 11)), 3))
    counters[:] = 0
    result = tempering.run_tempering_chunk(positions, counts, centers, node, cycle,
        clock, steps, *tail, coordinate_stride=11, biased_birth=True,
        birth_width=.3, birth_uniform_fraction=.2, coordinate_centers=coordinate_centers,
        calibration_label_quota=quota, vacancy_relocation_probability=1.,
        vacancy_relocation_counts=counters)
    rows, _, moves, configurations, clocks, saved_labels, _, _ = result
    assert weights.tobytes() == immutable_weights.tobytes()  # gain=0: quota inert.
    labels, numbers = rows[:, 1].astype(int), rows[:, 2].astype(int)
    observed_joint = np.bincount(labels * 4 + numbers, minlength=20).reshape(5, 4) / steps
    np.testing.assert_allclose(observed_joint, expected_joint, atol=.012, rtol=0)
    np.testing.assert_allclose(observed_joint.sum(axis=1), quota, atol=.015, rtol=0)
    for label in range(5):
        selected = labels == label
        assert selected.sum() > (1500 if label == 4 else 10000)
        observed = np.bincount(numbers[selected], minlength=4) / selected.sum()
        np.testing.assert_allclose(observed, conditional_number[label], atol=.025, rtol=0)
        np.testing.assert_allclose(rows[selected, 3].mean(), expected_order[label], atol=.025, rtol=0)
    eligible = ((penalty[labels] >= 8) | (kappa[labels] > 0)) & (numbers > 0)
    assert counters[0] == np.count_nonzero(eligible)
    assert 1000 < counters[1] < counters[0]
    assert np.all(rows[(labels == 0) | (labels == 1), 4] == 0)
    assert np.all(rows[(labels == 1) & (numbers > 0), 7] >= 1.)
    assert [n for n, _ in checked_chunks] == [5000, 200000]
    # The ideal quota is .02 with degree 3: its expected outgoing proposals per
    # edge are only T*.02*.5/3, about 667 BEFORE MH rejection. A >1000 cutoff
    # borrowed from a higher-ideal-mass control is not a meaningful precision
    # requirement. These bidirectional counters require weak repeated execution
    # coverage only; equilibrium precision is tested by independent laws above.
    assert np.all(moves[:, 1] > 500)
    events = np.flatnonzero(rows[:, 11])
    assert len(events) > 1000 and np.all(labels[events] == 4)
    assert np.array_equal(np.diff(rows[:, 10]), rows[1:, 11])
    completed = labels[events[0]:events[-1]]
    base, cavity = np.count_nonzero(completed == 0), np.count_nonzero(completed == 1)
    assert base > 10000 and cavity > 10000
    work = math.log(base / cavity) + weights[1] - weights[0]
    assert abs(work - math.log(normalizer[0] / normalizer[1])) < .07
    # All sparse snapshots must use the corresponding full-clock observation,
    # not an accepted-hop chain or a center saved after the label/Palm step.
    indices = clocks.astype(int) - (clock + 1)
    np.testing.assert_array_equal(saved_labels, labels[indices])
    vectors = cells * VECTORS
    for frame, index in enumerate(indices):
        n, label = numbers[index], labels[index]
        active = configurations[frame, :n]
        assert np.isnan(configurations[frame, n:]).all()
        assert np.all((active >= 0.) & (active < length))
        order = np.abs(np.exp(2j * np.pi / length * (active @ vectors.T)).mean(axis=0)).mean() if n else 0.
        np.testing.assert_allclose(order, rows[index, 3], atol=1e-10, rtol=0)
        delta = active - coordinate_centers[frame]
        delta -= length * np.rint(delta / length)
        minimum = np.linalg.norm(delta, axis=1).min() if n else length
        np.testing.assert_allclose(minimum, rows[index, 7], atol=1e-10, rtol=0)
        overlaps = 0
        for first in range(n):
            for second in range(first):
                separation = active[first] - active[second]
                separation -= length * np.rint(separation / length)
                overlaps += np.dot(separation, separation) < 1.
        assert overlaps == rows[index, 4]
        if label in (0, 1):
            assert overlaps == 0
        if label == 1:
            assert minimum >= 1. or n == 0
