"""Independent strict-record controls for label quotas, not phase weights."""
import copy

import numpy as np
import pytest

from scripts import continuum_single_walker_analyze as st


def quota_record(mode='endpoint_priority'):
    quota = np.array([.02,.30,.30,.19,.19]) if mode == 'endpoint_priority' else np.full(5,.2)
    counts = np.array([2,30,30,19,19]) if mode == 'endpoint_priority' else np.full(5,20)
    measure = ('minimum(counts/quota)/mean(counts/quota)' if mode == 'endpoint_priority'
               else 'minimum(counts)/mean(counts)')
    profile = {'node':1,'first_half':{'observations':int(counts[1]/2),'mean_N':3.,'mean_Bragg':.2},
               'second_half':{'observations':int(counts[1]/2),'mean_N':3.,'mean_Bragg':.2},
               'sufficient_half_observations':True,'mean_N_difference':0.,
               'mean_Bragg_difference':0.,'passed':True}
    row = {'stage':0,'gain':.1,'end_clock':100,'epoch_steps':100,
           'first_half_steps':50,'second_half_steps':50,'weight_update_visits':counts.tolist(),
           'flatness_minimum_to_mean':1.,'minimum_visits_met':True,'flatness_passed':True,
           'dense_half_stability_passed':True,'stage_passed':True,
           'ideal_heatbath_events':int(counts[0]),'dense_half_profiles':[profile],
           'log_weights':[0.]*5}
    if mode == 'endpoint_priority':row['quota_scaled_weight_update_visits'] = (counts/quota).tolist()
    outcome = {'method':'staged_wang_landau','completed':True,'budget_exhausted':False,
               'actual_clock_steps':100,'requested_budget':200,'initial_gain':.1,
               'minimum_gain':.1,'final_gain':.1,'stage_history':[row],
               'label_quota':quota.tolist(),'flatness_measure':measure,
               'criteria':{'flatness_minimum_to_mean':.5,'minimum_weight_update_visits_per_label':1,
                   'minimum_observations_per_dense_label_per_half':1,
                   'maximum_half_mean_N_difference':.25,'maximum_half_mean_Bragg_difference':.1,
                   'dense_diagnostic_nodes':[1]}}
    data = {'activity':np.zeros(5),'cap':5,'base_node':1,'cavity_node':2,'ideal_node':0,
            'log_weights':np.zeros(5),'initial_log_weights':np.zeros(5),'calibration_visits':counts.copy(),
            'calibration_weight_update_visits':counts.copy(),'calibration_label_quota':quota.copy()}
    metadata = {'calibration_method':'staged_wang_landau','calibration_cutoff_clock':100,
        'settings':{'calibration_method':'staged_wang_landau','calibration_steps':200,
                    'calibration_gain':.1,'calibration_label_quota':mode},
        'calibration_outcome':outcome,'calibration_label_quota':quota.tolist(),
        'calibration_label_quota_mode':mode,'calibration_flatness_measure':measure,
        'calibration_gauge_interval_clock_steps':1000,'calibration_log_weight_clip':[-1e5,1e5],
        'calibration_weight_update_rule':('b += gain*quota; b[final_label] -= gain; periodic ideal-node gauge'
              if mode == 'endpoint_priority' else 'legacy uniform decrement-only; periodic ideal-node gauge')}
    return data,metadata


@pytest.mark.parametrize('mode',['uniform','endpoint_priority'])
def test_declared_label_quota_replays_staged_counts_without_changing_conditional_estimator(mode):
    data,metadata = quota_record(mode)
    assert st.validate_calibration_record(data,metadata) == 'staged_wang_landau'
    expected = data['calibration_label_quota'].copy()
    quota,actual_mode = st.validate_calibration_label_quota(data,metadata)
    np.testing.assert_array_equal(quota,expected)
    assert actual_mode == mode
    # The thermodynamic work uses frozen label weights and dwell, not quotas.
    values = st.estimate_totals([30,10,90,270,20,0,0,0,0,0],1.,2.,3.,10.)
    assert values['work_from_label_dwell'] == pytest.approx(np.log(3)+1.)


def test_fully_absent_quota_retains_legacy_uniform_semantics():
    data = {'activity':np.zeros(5)}
    quota,mode = st.validate_calibration_label_quota(data,{})
    np.testing.assert_array_equal(quota,np.full(5,.2))
    assert mode == 'legacy_uniform'
    assert st.validate_calibration_record(data,{'calibration_cutoff_clock':0}) == 'legacy_one_over_t'


@pytest.mark.parametrize('mode',['uniform','endpoint_priority'])
def test_declared_quota_is_bound_for_one_over_t_without_staged_history(mode):
    data,metadata = quota_record(mode)
    metadata['calibration_method'] = metadata['settings']['calibration_method'] = 'one_over_t'
    metadata['settings']['calibration_steps'] = 100
    out = metadata['calibration_outcome']
    out.update(method='one_over_t',stage_history=[],requested_budget=100)
    assert st.validate_calibration_record(data,metadata) == 'one_over_t'


def test_quota_flatness_does_not_replace_minimum_actual_visit_requirement():
    data,metadata = quota_record()
    out = metadata['calibration_outcome']
    out['criteria']['minimum_weight_update_visits_per_label'] = 3
    row = out['stage_history'][0]
    row.update(minimum_visits_met=False,flatness_passed=False,stage_passed=False)
    out.update(completed=False,budget_exhausted=True,requested_budget=100)
    metadata['settings']['calibration_steps'] = 100
    assert row['flatness_minimum_to_mean'] == 1.
    assert st.validate_calibration_record(data,metadata) == 'staged_wang_landau'
    row.update(minimum_visits_met=True,flatness_passed=True,stage_passed=True)
    with pytest.raises(ValueError,match='pass flags'):st.validate_calibration_record(data,metadata)


@pytest.mark.parametrize('value',[
    [0.,.3,.3,.2,.2],[-.02,.3,.3,.21,.21],[np.nan,.3,.3,.19,.19],
    [np.inf,.3,.3,.19,.19],[.02,.3,.3,.19,.20],[.02,.3,.3],
    [[.02,.3,.3,.19,.19]], [.2,.2,.2,.2,.2],
])
def test_quota_raw_shape_normalization_positivity_and_declared_allocation_are_bound(value):
    data,metadata = quota_record()
    data['calibration_label_quota'] = np.asarray(value,dtype=float)
    with pytest.raises(ValueError,match='quota'):st.validate_calibration_record(data,metadata)


def test_quota_requires_float64_raw_not_coercible_integer_or_float32():
    for dtype in (int,np.float32,bool):
        data,metadata = quota_record('uniform')
        data['calibration_label_quota'] = data['calibration_label_quota'].astype(dtype)
        with pytest.raises(ValueError,match='quota'):st.validate_calibration_record(data,metadata)


@pytest.mark.parametrize('missing',[
    'calibration_label_quota','calibration_label_quota_mode','calibration_flatness_measure',
    'calibration_weight_update_rule','calibration_gauge_interval_clock_steps','calibration_log_weight_clip',
])
def test_partial_quota_metadata_cannot_fall_back_to_legacy(missing):
    data,metadata = quota_record()
    metadata.pop(missing)
    with pytest.raises(ValueError,match='quota'):st.validate_calibration_record(data,metadata)


@pytest.mark.parametrize('mutation',[
    lambda d,m:d.pop('calibration_label_quota'),
    lambda d,m:m['settings'].pop('calibration_label_quota'),
    lambda d,m:m['settings'].__setitem__('calibration_label_quota','custom'),
    lambda d,m:m.__setitem__('calibration_label_quota_mode','uniform'),
    lambda d,m:m.__setitem__('calibration_label_quota',[.2]*5),
    lambda d,m:m['calibration_outcome'].pop('label_quota'),
    lambda d,m:m['calibration_outcome'].__setitem__('label_quota',[.2]*5),
    lambda d,m:m['calibration_outcome'].__setitem__('flatness_measure','minimum(counts)/mean(counts)'),
    lambda d,m:m.__setitem__('calibration_weight_update_rule','physical phase preference'),
    lambda d,m:m.__setitem__('calibration_gauge_interval_clock_steps',100),
    lambda d,m:m.__setitem__('calibration_log_weight_clip',[-100.,100.]),
    lambda d,m:m['calibration_outcome']['stage_history'][0].pop('quota_scaled_weight_update_visits'),
    lambda d,m:m['calibration_outcome']['stage_history'][0].__setitem__('quota_scaled_weight_update_visits',[2,30,30,19,19]),
    lambda d,m:m['calibration_outcome']['stage_history'][0].__setitem__('flatness_minimum_to_mean',.1),
])
def test_quota_protocol_scaled_counts_and_efficiency_flags_cannot_be_relabelled(mutation):
    data,metadata = quota_record()
    mutation(data,metadata)
    with pytest.raises(ValueError):st.validate_calibration_record(data,metadata)


def test_uniform_history_does_not_accept_undeclared_scaled_flatness():
    data,metadata = quota_record('uniform')
    metadata['calibration_outcome']['stage_history'][0]['quota_scaled_weight_update_visits'] = [100.]*5
    with pytest.raises(ValueError,match='Uniform'):st.validate_calibration_record(data,metadata)


def test_staged_quota_cumulative_counts_retain_each_stage_terminal_tally_once():
    data,metadata = quota_record()
    out = metadata['calibration_outcome']
    first = out['stage_history'][0]
    first['gain'] = out['initial_gain'] = .2
    second = copy.deepcopy(first)
    second.update(stage=1,gain=.1,end_clock=200)
    out['stage_history'].append(second)
    data['calibration_visits'] *= 2
    data['calibration_weight_update_visits'] *= 2
    out['actual_clock_steps'] = metadata['calibration_cutoff_clock'] = 200
    out['requested_budget'] = metadata['settings']['calibration_steps'] = 300
    assert st.validate_calibration_record(data,metadata) == 'staged_wang_landau'


def weight_ledger(mode='endpoint_priority'):
    data,metadata = quota_record(mode)
    quota = data['calibration_label_quota']
    weights = np.array([0.,2.,-3.,.7,1.])
    data['initial_log_weights'] = weights.copy()
    clocks = 0
    history = []
    # Actual individual scalar updates, not the validator's aggregate formula.
    for stage,gain,chunks in ((0,.2,([1,4,4,0,3],[2,1,4,3])),(1,.1,([0,2,2,1],))):
        counts = np.zeros(5,dtype=int)
        for sequence in chunks:
            for label in sequence:
                if mode == 'endpoint_priority':weights += gain*quota
                weights[label] -= gain
                clocks += 1
                counts[label] += 1
            history.append({'stage':stage,'gain':gain,'end_clock':clocks,
                            'weight_update_visits':counts.tolist(),'log_weights':weights.tolist()})
    metadata['calibration_outcome']['stage_history'] = history
    data['log_weights'] = weights-weights[0]
    return data,metadata,quota


@pytest.mark.parametrize('mode',['uniform','endpoint_priority'])
def test_intermediate_weight_replay_matches_individual_updates_and_stage_reset(mode):
    data,metadata,quota = weight_ledger(mode)
    result = st.replay_quota_stage_weights(data,metadata,quota,mode)
    assert result['checkpoint_intervals_checked'] == 3
    assert result['status'] == 'gauge_invariant_increments_verified_without_clipping'
    for row in metadata['calibration_outcome']['stage_history']:
        row['log_weights'] = (np.asarray(row['log_weights'])+123.).tolist()
    assert st.replay_quota_stage_weights(data,metadata,quota,mode)['checkpoint_intervals_checked'] == 3


@pytest.mark.parametrize('mutation',[
    lambda d,m:m['calibration_outcome']['stage_history'][0]['log_weights'].__setitem__(1,2.),
    lambda d,m:m['calibration_outcome']['stage_history'][-1]['log_weights'].__setitem__(3,-1.),
    lambda d,m:m['calibration_outcome']['stage_history'][1]['weight_update_visits'].__setitem__(2,0),
    lambda d,m:d['initial_log_weights'].__setitem__(4,2.),
    lambda d,m:m['calibration_outcome']['stage_history'][1].__setitem__('gain',.3),
])
def test_intermediate_weight_replay_rejects_counts_gains_or_weight_tampering(mutation):
    data,metadata,quota = weight_ledger()
    mutation(data,metadata)
    with pytest.raises(ValueError,match='Intermediate quota log weights'):
        st.replay_quota_stage_weights(data,metadata,quota,'endpoint_priority')


def test_final_matching_weight_does_not_hide_tampered_intermediate_staged_record():
    data,metadata = quota_record()
    out = metadata['calibration_outcome']
    first = out['stage_history'][0]
    first['gain'] = out['initial_gain'] = .2
    second = copy.deepcopy(first)
    second.update(stage=1,gain=.1,end_clock=200)
    out['stage_history'].append(second)
    data['calibration_visits'] *= 2
    data['calibration_weight_update_visits'] *= 2
    out['actual_clock_steps'] = metadata['calibration_cutoff_clock'] = 200
    out['requested_budget'] = metadata['settings']['calibration_steps'] = 300
    first['log_weights'][1] = .001
    with pytest.raises(ValueError,match='Intermediate quota log weights'):
        st.validate_calibration_record(data,metadata)


@pytest.mark.parametrize('end_clock',[1000,1500])
def test_unexcluded_clipping_rejects_count_only_replay_without_claiming_target_error(end_clock):
    data,metadata,quota = weight_ledger()
    data['initial_log_weights'][1] = 99999.
    row = metadata['calibration_outcome']['stage_history'][0]
    row.update(end_clock=end_clock,weight_update_visits=[0,end_clock,0,0,0])
    metadata['calibration_outcome']['stage_history'] = [row]
    with pytest.raises(ValueError,match='Insufficient ordered-update evidence.*clipping'):
        st.replay_quota_stage_weights(data,metadata,quota,'endpoint_priority')


def test_no_scheduled_clamp_allows_exact_replay_even_if_a_conservative_value_exceeds_clip():
    data,metadata,quota = weight_ledger()
    data['initial_log_weights'][1] += 100001.
    for row in metadata['calibration_outcome']['stage_history']:
        row['log_weights'][1] += 100001.
    assert st.replay_quota_stage_weights(data,metadata,quota,'endpoint_priority')['checkpoint_intervals_checked'] == 3


def test_empty_declared_staged_calibration_must_preserve_initial_relative_weights():
    data,metadata,quota = weight_ledger()
    metadata['calibration_outcome']['stage_history'] = []
    data['log_weights'] = data['initial_log_weights']+4.
    assert st.replay_quota_stage_weights(data,metadata,quota,'endpoint_priority')['checkpoint_intervals_checked'] == 0
    data['log_weights'][1] += .1
    with pytest.raises(ValueError,match='Zero-clock'):
        st.replay_quota_stage_weights(data,metadata,quota,'endpoint_priority')
