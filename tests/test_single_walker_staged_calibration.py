import json

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as st


def trace(labels=(0, 0, 1, 1, 2, 2), number=25., bragg=.2):
    result = np.zeros((len(labels), 12))
    result[:, 0] = np.arange(1, len(labels) + 1)
    result[:, 1] = labels
    result[:, 2] = number
    result[:, 3] = bragg
    result[:, 11] = result[:, 1] == 0
    return result


def controller(**kwargs):
    settings = dict(nodes=3, dense_nodes=np.array([1]), gain=.2, minimum_gain=.1,
                    minimum_visits=2, half_minimum_visits=2)
    settings.update(kwargs)
    return st.StagedFlatnessCalibration(**settings)


def test_recent_epoch_half_samples_and_gain_halving_are_required():
    calibration = controller()
    values = trace()
    row = calibration.observe(values, values[:, 1].astype(int), 6, np.zeros(3))
    assert not row['stage_passed']
    assert not row['dense_half_profiles'][0]['sufficient_half_observations']
    row = calibration.observe(values, values[:, 1].astype(int), 12, np.zeros(3))
    assert row['stage_passed'] and calibration.gain == .1
    assert not calibration.completed and calibration.stage == 1
    assert not calibration.chunks
    calibration.observe(values, values[:, 1].astype(int), 18, np.zeros(3))
    calibration.observe(values, values[:, 1].astype(int), 24, np.zeros(3))
    assert calibration.completed
    outcome = calibration.outcome(24, 100)
    assert outcome['completed'] and not outcome['budget_exhausted']
    assert [r['gain'] for r in outcome['stage_history']] == [.2, .2, .1, .1]
    json.dumps(outcome, allow_nan=False)


def test_flatness_counts_updated_label_not_pre_observation_label():
    calibration = controller()
    values = trace()
    updated = np.zeros(len(values), dtype=int)
    calibration.observe(values, updated, 6, np.zeros(3))
    row = calibration.observe(values, updated, 12, np.zeros(3))
    assert row['weight_update_visits'] == [12, 0, 0]
    assert row['dense_half_stability_passed']
    assert not row['flatness_passed'] and not row['stage_passed']


def test_stability_never_pairs_post_tag_label_with_stale_N_or_B():
    calibration = controller()
    # Dense label 1 appears only in updated labels, never in the physical
    # observation trace: matching flat counts cannot create half-window data.
    values = trace(labels=(0, 0, 2, 2, 2, 2))
    updated = np.array([0, 0, 1, 1, 2, 2])
    calibration.observe(values, updated, 6, np.zeros(3))
    row = calibration.observe(values, updated, 12, np.zeros(3))
    assert row['flatness_passed']
    profile = row['dense_half_profiles'][0]
    assert not profile['sufficient_half_observations']
    assert profile['first_half']['mean_N'] is None
    assert not row['stage_passed']


@pytest.mark.parametrize('number,bragg', [(27., .2), (25., .5)])
def test_recent_epoch_N_or_order_drift_prevents_gain_reduction(number, bragg):
    calibration = controller()
    first, second = trace(), trace(number=number, bragg=bragg)
    calibration.observe(first, first[:, 1].astype(int), 6, np.zeros(3))
    row = calibration.observe(second, second[:, 1].astype(int), 12, np.zeros(3))
    assert row['flatness_passed'] and not row['dense_half_stability_passed']
    assert calibration.gain == .2 and calibration.stage == 0


def test_budget_exhaustion_is_explicit_and_does_not_extend_or_pass_stage():
    calibration = controller()
    values = trace()
    calibration.observe(values, values[:, 1].astype(int), 6, np.zeros(3))
    outcome = calibration.outcome(6, 6)
    assert outcome['budget_exhausted'] and not outcome['completed']
    assert outcome['actual_clock_steps'] == outcome['requested_budget'] == 6


@pytest.mark.parametrize('kwargs', [dict(gain=0), dict(gain=np.inf), dict(minimum_gain=-1),
    dict(minimum_gain=.3), dict(flatness=0), dict(flatness=1.1), dict(minimum_visits=0),
    dict(half_minimum_visits=0), dict(number_tolerance=np.nan), dict(bragg_tolerance=-1)])
def test_nonfinite_or_invalid_criteria_are_rejected(kwargs):
    with pytest.raises(ValueError, match='criteria'):
        controller(**kwargs)


def test_malformed_update_labels_weights_and_dense_nodes_are_rejected():
    with pytest.raises(ValueError, match='Dense'):
        controller(dense_nodes=np.array([], dtype=int))
    calibration = controller()
    values = trace()
    with pytest.raises(ValueError, match='statistics'):
        calibration.observe(values, values[:, 1], 6, np.zeros(3))
    with pytest.raises(ValueError, match='statistics'):
        calibration.observe(values, values[:, 1].astype(int), 6, np.array([0., np.inf, 0.]))


def test_complete_stage_rejects_further_adaptive_observations():
    calibration = controller(minimum_gain=.2)
    values = trace()
    for clock in (6, 12):
        calibration.observe(values, values[:, 1].astype(int), clock, np.zeros(3))
    assert calibration.completed
    with pytest.raises(ValueError, match='completed'):
        calibration.observe(values, values[:, 1].astype(int), 18, np.zeros(3))
