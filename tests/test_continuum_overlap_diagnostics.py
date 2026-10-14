"""Optional overlap traces are exact observables and pathwise inert."""

import numpy as np
import pytest

from scripts.continuum_replica_exchange import fcc_reference, run_chunk, seed_rng


def fixture():
    length = 3.
    positions = np.zeros((3, 5, 3))
    positions[:, 0] = [.4, .7, 1.1]
    return dict(positions=positions, counts=np.ones(3, dtype=np.int64),
                centers=np.zeros((3, 3)), diameter=np.ones(3),
                radius=np.array([0., 1., 0.]), activity=np.full(3, -2.2),
                identities=np.arange(3), length=length, sweeps=400,
                endpoint=0, sample_stride=5,
                reference=fcc_reference(4, length), displacement=np.full(3, .8),
                pair_penalty=np.array([np.inf, np.inf, .7]),
                umbrella_stiffness=np.array([0., 0., 2.5]),
                umbrella_preferred=np.array([0., 0., .4]),
                exchange_edges=np.array([[0, 1], [0, 2], [1, 2]]),
                cavity_nodes=np.array([0, 1]), snapshot_nodes=np.arange(3),
                palm_transfer=True, biased_birth=True, birth_proposal=1,
                birth_scope=1, birth_width=.3)


def test_overlap_diagnostics_do_not_change_a_single_random_draw_or_state():
    first = fixture()
    second = fixture()
    seed_rng(940028)
    original = run_chunk(**first)
    overlaps = np.full((80, 3), -1, dtype=np.int64)
    seed_rng(940028)
    observed = run_chunk(**second, overlap_samples=overlaps)
    for a, b in zip(original, observed):
        np.testing.assert_array_equal(a, b)
    for name in ("positions", "counts", "centers", "identities"):
        np.testing.assert_array_equal(first[name], second[name])
    samples, _, _, snapshots = observed
    expected = np.zeros_like(overlaps)
    for frame in range(len(samples)):
        for node in range(3):
            n = int(samples[frame, node, 0])
            coordinates = snapshots[frame, node, :n]
            for i in range(n):
                displacement = coordinates[i] - coordinates[:i]
                displacement -= 3*np.rint(displacement/3)
                expected[frame, node] += np.count_nonzero(np.sum(displacement**2, axis=1) < 1)
    np.testing.assert_array_equal(overlaps, expected)
    assert not overlaps[:, :2].any()
    assert overlaps[:, 2].max() > 0


def test_overlap_array_shape_is_validated_before_sampling():
    settings = fixture()
    with pytest.raises(ValueError, match="Pair-overlap diagnostic array"):
        run_chunk.py_func(**settings, overlap_samples=np.zeros((79, 3)))


@pytest.mark.parametrize("sweeps,stride", [(1, 0), (-1, 1), (11, 2)])
def test_low_level_chunk_cadence_is_validated(sweeps, stride):
    settings = fixture()
    settings.update(sweeps=sweeps, sample_stride=stride)
    with pytest.raises(ValueError, match="Chunk sweeps"):
        run_chunk.py_func(**settings)
