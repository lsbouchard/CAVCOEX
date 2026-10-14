"""Default-off trajectories match the immutable pre-Gibbs executed source."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as current


LEGACY_HASH = '8de3389f084fd0d9494b37caccf6784dce652f3e3e385ebd52627c601920f23f'


def legacy_module():
    path = current.ROOT / 'data_raw' / 'source_snapshots' / f'continuum_simulated_tempering_{LEGACY_HASH}.py'
    if not path.exists():
        pytest.skip('Immutable audit source is supplied in the full audit archive')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LEGACY_HASH
    name = 'scripts._single_walker_pre_Gibbs_audit_oracle'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def run(module, gain, buffers):
    length, cap, steps = 3., 3, 3000
    positions = np.zeros((1, cap, 3))
    counts = np.zeros(1, dtype=np.int64)
    centers = np.zeros((1, 3))
    neighbors, edge_table, degrees = current.adjacency(np.array([[0, 1], [1, 2], [2, 0]]), 3)
    field = np.full(3, math.log(.06))
    weights = np.array([0., .3, -.4])
    updated = np.empty(steps, dtype=np.int64)
    intermediate = np.empty(steps, dtype=np.int64)
    counters = np.zeros(2, dtype=np.int64)
    optional = dict(gibbs_label=False, constant_calibration_gain=False,
                    neighbor_source_labels=intermediate, weight_update_labels=updated,
                    gibbs_counters=counters) if buffers else {}
    current.kernel.seed_rng(2026093007)
    result = module.run_tempering_chunk(positions, counts, centers, 2, 0, 13, steps,
        current.kernel.fcc_reference(32, length), length, np.ones(3), np.array([0., 1., 0.]),
        field, np.array([np.inf, np.inf, 0.]), np.zeros(3), np.zeros(3), np.zeros(3),
        np.zeros(3), np.full(3, .8), np.array([True, True, False]), 0, 1, 2,
        neighbors, edge_table, degrees, weights,
        current.capped_poisson_cdf(field[2] + math.log(length ** 3), cap),
        calibration_gain=gain, coordinate_stride=11, birth_width=.3, **optional)
    return result, positions, counts, centers, weights, intermediate, updated, counters


@pytest.mark.parametrize('gain', [0., .2])
def test_default_off_preserves_old_RNG_and_state_bit_for_bit(gain):
    old = run(legacy_module(), gain, False)
    new = run(current, gain, True)
    for old_item, new_item in zip(old[0][:6], new[0][:6]):
        np.testing.assert_array_equal(old_item, new_item)
    assert old[0][-2:] == new[0][-2:]
    for old_item, new_item in zip(old[1:5], new[1:5]):
        np.testing.assert_array_equal(old_item, new_item)
    np.testing.assert_array_equal(new[5], new[0][0][:, 1])
    np.testing.assert_array_equal(new[6][:-1], new[0][0][1:, 1])
    assert new[6][-1] == new[0][-2]
    assert not new[7].any()
