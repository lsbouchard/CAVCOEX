"""Actual continuum radius ring against exact, untruncated physical laws.

On L=sigma=1, hard hosts have N<=1. Thus K=2 leaves all physical radii
untruncated, whereas the ideal auxiliary has its explicitly capped law.
No contact-force or phase-coexistence interpretation is used on this torus.
"""
import math
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from scripts import continuum_simulated_tempering as tempering


def _closed_empty_host_radius_loops(trace, source, final):
    """Count measured substep paths, never use these as equilibrium rewards.

An accepted tag deletes the only host. Subsequent recorded local states stay
empty, and the two label substeps visit sigma,.5,.25,0 in that order without
skipping a radius. Gibbs and ordinary-radius label steps preserve that empty
host. The final base has one fewer particle than the initial tagged base.
"""
    rank = {3: 0, 2: 1, 1: 2, 0: 3}
    paths = []
    for start in np.flatnonzero((source == 0) & (final == 3) & (trace[:, 2] == 1)):
        progress = 0
        for step in range(start + 1, min(start + 40, len(trace))):
            if trace[step, 2] != 0 or int(trace[step, 1]) not in rank:
                break
            if rank[int(trace[step, 1])] != progress:
                break
            for label in (int(source[step]), int(final[step])):
                if label not in rank or rank[label] not in (progress, progress + 1):
                    progress = -1
                    break
                progress = rank[label]
            if progress < 0:
                break
            if progress == 3:
                assert final[step] == 0
                paths.append((int(start), step))
                break
    return paths


def test_actual_continuum_radius_ring_preserves_exact_laws_and_transports_number():
    length, sigma, activity, cap = 1., 1., .2, 2
    assert math.sqrt(3) * length / 2 < sigma
    radius = np.array([0., .25, .5, 1., 0.])
    excluded = 4 * math.pi / 3 * radius[:3] ** 3
    sectors = np.array([[1., activity, 0.],
                        [1., activity * (1 - excluded[1]), 0.],
                        [1., activity * (1 - excluded[2]), 0.],
                        [1., 0., 0.],
                        [1., activity, activity ** 2 / 2]])
    partition = sectors.sum(axis=1)
    conditional = sectors / partition[:, None]
    proportions = np.array([1.1, 1.4, 1.3, .9, 1.5])
    log_weights = np.log(proportions) - np.log(partition)
    expected_joint = proportions[:, None] / proportions.sum() * conditional
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4]], dtype=np.int64)
    neighbors, edge_table, degrees = tempering.adjacency(edges, 5)
    assert set(degrees) == {1, 2, 3}
    zero = np.zeros(5)
    fields = np.full(5, math.log(activity))
    physical = np.array([True, True, True, True, False])
    positions = np.zeros((1, cap, 3))
    counts = np.zeros(1, dtype=np.int64)
    centers = np.zeros((1, 3))
    reference = tempering.kernel.fcc_reference(4, length)
    cdf = tempering.capped_poisson_cdf(math.log(activity), cap)
    tail = (reference, length, np.ones(5), radius, fields,
            np.array([np.inf, np.inf, np.inf, np.inf, 0.]),
            zero, zero, zero, zero, np.full(5, .25), physical,
            0, 3, 4, neighbors, edge_table, degrees, log_weights, cdf)
    warm_steps, steps, stride = 3000, 120000, 13
    tempering.kernel.seed_rng(260930601)
    warm = tempering.run_tempering_chunk(
        positions, counts, centers, 4, 0, 0, warm_steps, *tail,
        coordinate_stride=stride, biased_birth=True, birth_width=.12,
        birth_uniform_fraction=.2, gibbs_label=True)
    node, cycle = warm[-2:]
    coordinate_count = np.count_nonzero(np.arange(warm_steps + 1,
                                                 warm_steps + steps + 1) % stride == 0)
    coordinate_centers = np.empty((coordinate_count, 3))
    source, final = np.full(steps, -1, dtype=np.int64), np.full(steps, -1, dtype=np.int64)
    gibbs = np.zeros(2, dtype=np.int64)
    result = tempering.run_tempering_chunk(
        positions, counts, centers, node, cycle, warm_steps, steps, *tail,
        coordinate_stride=stride, biased_birth=True, birth_width=.12,
        birth_uniform_fraction=.2, coordinate_centers=coordinate_centers,
        gibbs_label=True, neighbor_source_labels=source,
        weight_update_labels=final, gibbs_counters=gibbs)
    trace, local, moves, configurations, clocks, saved_labels, _, _ = result
    labels, numbers = trace[:, 1].astype(int), trace[:, 2].astype(int)
    joint = np.bincount(labels * (cap + 1) + numbers, minlength=5 * (cap + 1))
    joint = joint.reshape((5, cap + 1)) / steps
    np.testing.assert_allclose(joint, expected_joint, atol=.009, rtol=0)
    for label in range(5):
        selected = labels == label
        assert selected.sum() > 10000
        observed = np.bincount(numbers[selected], minlength=cap + 1) / selected.sum()
        np.testing.assert_allclose(observed, conditional[label], atol=.018, rtol=0)
    assert (numbers[labels != 4] <= 1).all()
    assert (numbers[labels == 3] == 0).all()
    assert np.count_nonzero((labels == 4) & (numbers == 2)) > 300
    assert gibbs[0] == steps
    assert gibbs[1] == np.count_nonzero(source != labels)
    assert np.array_equal(labels[1:], final[:-1]) and final[-1] == result[-2]
    assert ((source >= 0) & (source < 5) & (final >= 0) & (final < 5)).all()
    accepted = np.zeros(len(edges), dtype=np.int64)
    for old, new in zip(source, final):
        if old != new:
            locations = np.flatnonzero(neighbors[old] == new)
            assert len(locations) == 1
            accepted[edge_table[old, locations[0]]] += 1
    assert np.array_equal(accepted, moves[:, 1])
    assert (accepted > 1000).all() and local[3] > 1000
    for label in range(4):
        occupied = (source == label) & (numbers > 0)
        assert (trace[occupied, 7] >= radius[label]).all()
        assert (trace[source == label, 4] == 0).all()
    # A second independent geometric observable for the small cavities.
    base_singleton = (labels == 0) & (numbers == 1)
    quarter_singleton = (labels == 1) & (numbers == 1)
    np.testing.assert_allclose(np.mean(trace[base_singleton, 7] < .5), excluded[2], atol=.035, rtol=0)
    annulus = (excluded[2] - excluded[1]) / (1 - excluded[1])
    np.testing.assert_allclose(np.mean(trace[quarter_singleton, 7] < .5), annulus, atol=.035, rtol=0)
    events = np.flatnonzero(trace[:, 11])
    assert len(events) > 15000 and (labels[events] == 4).all()
    assert np.array_equal(np.diff(trace[:, 10]), trace[1:, 11])
    complete = labels[events[0]:events[-1]]
    base_dwell = np.count_nonzero(complete == 0)
    for label in (1, 2, 3):
        dwell = np.count_nonzero(complete == label)
        work = math.log(base_dwell / dwell) + log_weights[label] - log_weights[0]
        np.testing.assert_allclose(work, math.log(partition[0] / partition[label]), atol=.05, rtol=0)
    mean_base = numbers[labels == 0].mean()
    fano = numbers[labels == 0].var() / mean_base
    np.testing.assert_allclose(-mean_base, fano - 1, atol=1e-12, rtol=0)
    assert np.array_equal(trace[labels != 4, 8], -numbers[labels != 4])
    rows = clocks.astype(int) - warm_steps - 1
    assert np.array_equal(saved_labels, labels[rows])
    for index, row in enumerate(rows):
        host = configurations[index, :numbers[row]]
        difference = host - coordinate_centers[index]
        difference -= length * np.rint(difference / length)
        distance = np.linalg.norm(difference, axis=1).min() if len(host) else length
        np.testing.assert_allclose(distance, trace[row, 7], atol=1e-12, rtol=0)
        if labels[row] != 4:
            assert len(host) <= 1 and (not len(host) or distance >= radius[labels[row]])
    tagged_deletions = np.count_nonzero((source == 0) & (final == 3) & (numbers == 1))
    assert tagged_deletions > 300
    loops = _closed_empty_host_radius_loops(trace, source, final)
    assert loops  # A measured complete radius-ring number-transport path.
    for first, last in loops:
        assert numbers[first] == 1 and source[first] == 0 and final[first] == 3
        assert (numbers[first + 1:last + 1] == 0).all() and final[last] == 0
