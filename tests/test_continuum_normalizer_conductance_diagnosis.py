import numpy as np

from scripts import continuum_normalizer_conductance_diagnosis as diagnosis


def fixture():
    return {'activity': np.ones(4), 'diameter': np.ones(4), 'radius': np.zeros(4),
            'pair_penalty': np.array([0., 0., 2., np.inf]),
            'umbrella_stiffness': np.zeros(4), 'umbrella_preferred': np.array([0., .88, 0., 0.]),
            'number_stiffness': np.zeros(4), 'number_preferred': np.array([0., 32., 0., 0.]),
            'calibration_label_quota': np.array([.1, .2, .3, .4]),
            'log_weights': np.log([.1, .2, .3, .4])+np.array([1., 1., 2., 3.])}


def test_zero_strength_preferences_do_not_split_exact_target_duplicates():
    d = fixture()
    assert diagnosis.target_signature(d, 0) == diagnosis.target_signature(d, 1)
    result = diagnosis.necessary_normalizer_root_checks(d)
    assert result['exact_duplicate_target_groups'][0]['nodes'] == [0, 1]
    assert result['exact_duplicate_target_groups'][0]['span'] == 0
    assert result['zero_umbrella_pair_penalty_groups'][0]['violations'] == []


def test_nonzero_strength_preferences_are_part_of_target():
    d = fixture()
    d['umbrella_stiffness'][:2] = 1
    assert diagnosis.target_signature(d, 0) != diagnosis.target_signature(d, 1)
    d['umbrella_stiffness'][:2] = 0
    d['number_stiffness'][:2] = 1
    assert diagnosis.target_signature(d, 0) != diagnosis.target_signature(d, 1)


def test_duplicate_weight_mismatch_is_quota_corrected():
    d = fixture()
    d['log_weights'][1] += 7
    rows = diagnosis.necessary_normalizer_root_checks(d)['exact_duplicate_target_groups']
    assert np.isclose(rows[0]['span'], 7)


def test_pair_penalty_normalizer_order_has_correct_direction():
    d = fixture()
    d['log_weights'][3] -= 5
    result = diagnosis.necessary_normalizer_root_checks(d)
    bad = result['zero_umbrella_pair_penalty_groups'][0]['violations']
    assert len(bad) == 3
    assert {row['higher_penalty_node'] for row in bad} == {3}
    assert all(row['adjusted_higher_minus_lower'] < 0 for row in bad)
