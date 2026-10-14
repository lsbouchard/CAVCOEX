"""Independent variant controls from the immutable original native artifact.

Auxiliary number biases are transport targets, not occupancy constraints or
phase weights. No continuum MC or JIT kernel is run by these tests.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import continuum_tempering_graph as graph
from test_native_compact_number_target_independent import (
    independent_fcc_and_order, sector_law,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'data_raw/controls/native_compact_number_radius_graph_v1.npz'
ORIGINAL_SHA = 'a24ef9533ad94d794b0bd0b22da6c86d0bba1036db708538e61e41fe94a3e50b'
VOLUME, SITES, CAP = 32 * .96405, 32, 80
FIELDS = ('diameter', 'radius', 'activity', 'pair_penalty', 'umbrella_stiffness',
          'umbrella_preferred', 'number_stiffness', 'number_preferred', 'displacement')


def original_arrays():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA
    with np.load(ORIGINAL, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def construct(strength):
    return graph.build_compact_graph(VOLUME, SITES, CAP, base_field=16.,
        cavity_radius_stages=16, ordered_hardening_number_strength=strength)


def test_default_four_preserves_the_immutable_original_target_and_weight_bytes():
    old = original_arrays()
    target, weights = construct(4.)
    for name, value in target.items():
        actual, expected = np.asarray(value), old[name]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), name
    assert weights.dtype == old['initial_log_weights'].dtype
    assert weights.tobytes() == old['initial_log_weights'].tobytes()


@pytest.mark.parametrize('strength', [.25, 1., 2.])
def test_variant_reconstructs_full_targets_and_graph_from_the_original_artifact(strength):
    old = original_arrays()
    target, weights = construct(strength)
    # The original 145 non-radius targets retain identities. Insert two fixed
    # gamma/B nodes before APPENDING the original physical radii, now shifted2.
    for name in FIELDS:
        expected = np.r_[old[name][:145], old[name][[38, 38]], old[name][145:]]
        if name == 'number_stiffness':
            expected[39:70] = strength
            expected[139:145] = [strength / 2, strength / 4, strength / 8,
                                  strength / 16, strength / 40, 0.]
            expected[145:147] = [math.sqrt(4 * strength), strength]
        np.testing.assert_array_equal(target[name], expected, err_msg=name)
    assert len(weights) == 162 and np.isfinite(weights).all()
    assert weights[0] == 0.
    # Preserve BOTH original low-order routes: 72..133 never acquire a new
    # number preference, changed density, or order bias.
    for name in FIELDS:
        np.testing.assert_array_equal(target[name][72:134], old[name][72:134])
    np.testing.assert_array_equal(target['number_stiffness'][21:39], np.full(18, 4.))
    np.testing.assert_array_equal(target['number_preferred'][21:70], np.full(49, SITES))
    for name in ('pair_penalty', 'umbrella_stiffness', 'umbrella_preferred', 'activity'):
        np.testing.assert_array_equal(target[name][145:147], np.repeat(old[name][38], 2))
    physical = np.r_[70, 71, np.arange(147, 162)]
    np.testing.assert_array_equal(target['physical_nodes'], physical)
    np.testing.assert_array_equal(target['radius_path_nodes'], np.r_[70, np.arange(147, 162), 71])
    assert np.all(target['number_stiffness'][physical] == 0.)
    assert np.all(target['umbrella_stiffness'][physical] == 0.)
    assert np.isposinf(target['pair_penalty'][physical]).all()
    assert np.all(target['activity'][physical] == 16.)
    assert target['volume'] == VOLUME
    expected_edges = set()
    for edge in old['exchange_edges']:
        if tuple(edge) == (38, 39):
            continue
        mapped = [int(node) + (2 if node >= 145 else 0) for node in edge]
        expected_edges.add(tuple(sorted(mapped)))
    expected_edges.update({(38, 145), (145, 146), (39, 146)})
    actual_edges = [tuple(sorted(map(int, edge))) for edge in target['exchange_edges']]
    assert len(set(actual_edges)) == len(actual_edges)
    assert set(actual_edges) == expected_edges
    reached = {0}
    while True:
        expanded = reached | {b for a, b in actual_edges if a in reached} | {a for a, b in actual_edges if b in reached}
        if expanded == reached:
            break
        reached = expanded
    assert reached == set(range(162))


@pytest.mark.parametrize('strength', [.25, 1., 2., 4.])
def test_strength_changes_the_exact_number_cost_not_factorials_or_allowed_sectors(strength):
    field = math.log(1 / 1.06448)
    numbers, logs, _, normalizer = sector_law(field, VOLUME, CAP, strength, SITES)
    assert np.isfinite(logs).all()
    model_ideal = graph.compact_weight_guess(field, VOLUME, CAP, 0., 0., .88, strength, SITES)
    assert model_ideal == pytest.approx(-normalizer, abs=2e-13)
    expected_ratio_logs = field + math.log(VOLUME) - np.log(numbers[1:]) - strength * (2 * (numbers[:-1] - SITES) + 1)
    np.testing.assert_allclose(np.diff(logs), expected_ratio_logs, atol=2e-12, rtol=1e-13)
    # The exact death term is +U_N(N)-U_N(N-1), irrespective of registry.
    assert strength * ((32 - SITES) ** 2 - (31 - SITES) ** 2) == -strength
    assert strength * ((31 - SITES) ** 2 - (30 - SITES) ** 2) == -3 * strength


def test_weaker_number_cost_can_admit_a_high_bragg_interstitial_cloud_not_a_crystal():
    length = VOLUME ** (1 / 3)
    perfect, order = independent_fcc_and_order(length)
    interstitial = np.r_[perfect, [(perfect[0] + [1e-3, 0., 0.]) % length]]
    assert len(np.unique(interstitial, axis=0)) == len(interstitial) == 33
    assert order(interstitial) > .99999
    difference = interstitial[:, None, :] - interstitial[None, :, :]
    difference -= length * np.rint(difference / length)
    distances = np.sum(difference ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    assert np.count_nonzero(np.triu(distances < 1., k=1)) == 1
    assert distances.min() == pytest.approx(1e-6, abs=1e-18)
    # H1 gives the extra soft particle a finite number cost1 rather than4.
    # It never makes the overlapping configuration physically admissible.
    assert (len(interstitial) - SITES) ** 2 == 1
    assert np.isfinite(2. + 1. * (len(interstitial) - SITES) ** 2)
