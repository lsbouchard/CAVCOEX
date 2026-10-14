"""Independent renewal-reward controls, without executing a Monte Carlo driver."""
import numpy as np
import pytest
import json
import math

from scripts import continuum_single_walker_analyze as st


def clock_trace():
    # Full clock: one long physical dwell with repeated rejected states, and
    # a short visit. Regeneration rows themselves are at the ideal label 0.
    labels = [1,0,*([1]*9),0,1,0,0,2,2,0,1]
    data = np.zeros((len(labels),12))
    data[:,0] = np.arange(1,len(labels)+1)
    data[:,1] = labels
    data[:,2] = 3
    data[:,3] = .2
    data[12,3] = .8
    data[:,11] = data[:,1]==0
    data[:,10] = np.cumsum(data[:,11])
    return data


def parameters():
    return dict(log_weight_base=1.,log_weight_cavity=2.,activity=3.,volume=10.)


def test_complete_clock_cycle_ratio_keeps_rejections_and_zero_dwell_cycles():
    rewards,bounds,coverage = st.complete_cycle_rewards(clock_trace(),1,2,0)
    assert rewards[:,0].tolist() == [9,1,0,0]
    assert rewards[:,1].tolist() == [0,0,0,2]
    assert coverage['zero_physical_dwell_cycles'] == 1
    assert coverage['leading_rows_excluded'] == 1
    assert coverage['unfinished_trailing_rows_excluded'] == 2
    assert coverage['complete_cycle_clock_steps'] == 16
    result = st.estimate_totals(rewards.sum(axis=0),**parameters())
    assert result['q_solid_base'] == .1
    # Averaging the two nonempty cycle fractions gives .5, the wrong target.
    assert np.mean(rewards[:2,5]/rewards[:2,0]) == .5
    assert result['work_from_label_dwell'] == pytest.approx(np.log(10/2)+1)
    assert result['partition_ratio_cavity_to_base'] == pytest.approx(.2*np.exp(-1))
    assert result['log_partition_ratio_cavity_to_base'] == pytest.approx(np.log(.2)-1)


def test_ratio_residual_standard_error_matches_independent_hand_calculation():
    numerator,denominator = np.array([0.,1.,0.]),np.array([9.,1.,0.])
    q = .1
    residual = np.array([-.9,.9,0.])
    expected = np.sqrt(np.sum(residual**2)/2)/(np.sqrt(3)*(10/3))
    assert st.ratio_cycle_standard_error(numerator,denominator) == pytest.approx(expected)
    assert np.isnan(st.ratio_cycle_standard_error([0],[0]))


def test_each_activity_profile_uses_complete_cycle_dwell_without_pooling():
    samples = clock_trace()
    _,bounds,_ = st.complete_cycle_rewards(samples,1,2,0)
    data = {'samples':samples,'physical_nodes':np.array([1,2,3]),
            'activity':np.array([-2.,3.,3.,4.]),'radius':np.array([0.,0.,1.,0.]),'volume':10.}
    profile = st.physical_node_cycle_profiles(data,bounds)
    assert profile[0]['complete_cycle_clock_visits'] == 10
    assert profile[0]['operational_B_fraction'] == .1
    assert profile[0]['dense_operational_B_fraction'] == 0
    assert profile[1]['complete_cycle_clock_visits'] == 2
    assert profile[2]['complete_cycle_clock_visits'] == 0
    assert np.isnan(profile[2]['mean_N'])
    assert profile[2]['B_boundary_fraction_uncertainty'] is None
    assert profile[0]['maximum_cycle_fraction_of_dwell'] == .9


def test_no_completed_cycle_gives_no_node_specific_estimate():
    data={'samples':clock_trace(),'physical_nodes':np.array([1]),
          'activity':np.array([-2.,3.]),'radius':np.zeros(2),'volume':10.}
    profile = st.physical_node_cycle_profiles(data,np.empty((0,2),int))[0]
    assert profile['complete_cycle_clock_visits'] == 0
    assert np.isnan(profile['operational_B_fraction'])
    assert profile['B_fraction_uncertainty_display']['SE_for_display'] is None


@pytest.mark.parametrize('point,high,low,dominant,incomplete',[
    (0.,0,100,.02,0.),(1.,100,0,.02,0.),(.2,3,100,.02,0.),
    (.2,30,100,.9,0.),(.2,30,100,.02,.8),
])
def test_boundary_or_sparse_basin_empirical_error_is_not_displayed_as_physical_precision(point,high,low,dominant,incomplete):
    result = st.phase_uncertainty_display(point,0.,100,high,low,dominant,incomplete)
    assert result['SE_for_display'] is None
    assert not result['display_checks_passed']


def test_conditional_phase_error_display_is_not_an_equilibrium_certificate():
    result = st.phase_uncertainty_display(.5,.02,100,70,70,.05,.01)
    assert result['SE_for_display'] == .02
    assert 'not a convergence or equilibrium certificate' in result['scope']


def compound_label_fixture():
    samples = clock_trace()
    samples[:,7] = 2.
    source = samples[:,1].astype(int)
    source[0] = 2
    return {'samples':samples,'activity':np.zeros(3),'diameter':np.ones(3),
            'radius':np.array([0.,0.,1.]),'pair_penalty':np.array([0.,np.inf,np.inf]),
            'neighbor_source_labels':source,'gibbs_attempt_change_counts':np.array([len(samples),1])}


def test_compound_label_trace_is_bound_without_reconstructing_neighbor_hops():
    data = compound_label_fixture()
    protocol = {'label_update_protocol':'gibbs_then_lazy_neighbor_tagged'}
    assert st.validate_label_update_protocol(data,protocol) == protocol['label_update_protocol']
    legacy = {key:value for key,value in data.items()
              if key not in ('neighbor_source_labels','gibbs_attempt_change_counts')}
    assert st.validate_label_update_protocol(legacy,{}) == 'lazy_neighbor_tagged'


@pytest.mark.parametrize('mutation',[
    lambda d:d.pop('neighbor_source_labels'),
    lambda d:d.__setitem__('neighbor_source_labels',d['neighbor_source_labels'][:-1]),
    lambda d:d['neighbor_source_labels'].__setitem__(0,3),
    lambda d:d['gibbs_attempt_change_counts'].__setitem__(1,2),
    lambda d:d['gibbs_attempt_change_counts'].__setitem__(0,999),
    lambda d:d['samples'].__setitem__((0,7),.5),
    lambda d:d['samples'].__setitem__((0,4),1),
])
def test_compound_label_protocol_rejects_missing_or_inconsistent_diagnostics(mutation):
    data = compound_label_fixture()
    mutation(data)
    with pytest.raises(ValueError):
        st.validate_label_update_protocol(data,{'label_update_protocol':'gibbs_then_lazy_neighbor_tagged'})


def test_neighbor_only_declaration_cannot_hide_gibbs_updates():
    with pytest.raises(ValueError,match='Neighbor-only'):
        st.validate_label_update_protocol(compound_label_fixture(),{})
    with pytest.raises(ValueError,match='Unrecognized'):
        st.validate_label_update_protocol(compound_label_fixture(),{'label_update_protocol':'unknown'})


def test_empty_host_has_no_cavity_support_restriction_after_gibbs():
    data = compound_label_fixture()
    data['samples'][0,2] = 0
    data['samples'][0,7] = .1
    assert st.validate_label_update_protocol(data,{'label_update_protocol':'gibbs_then_lazy_neighbor_tagged'})


def test_gibbs_setting_cannot_silently_fall_back_to_legacy_protocol():
    data = compound_label_fixture()
    with pytest.raises(ValueError,match='setting disagrees'):
        st.validate_label_update_protocol(data,{'settings':{'gibbs_label':True}})


def staged_record_fixture():
    data = {'activity':np.zeros(3),'cap':5,'ideal_node':0,'log_weights':np.zeros(3),
            'calibration_visits':np.array([2,2,2]),'calibration_weight_update_visits':np.array([2,2,2])}
    profile = {'node':1,'first_half':{'observations':1,'mean_N':3.,'mean_Bragg':.2},
               'second_half':{'observations':1,'mean_N':3.,'mean_Bragg':.2},
               'sufficient_half_observations':True,'mean_N_difference':0.,
               'mean_Bragg_difference':0.,'passed':True}
    row = {'stage':0,'gain':.1,'end_clock':6,'epoch_steps':6,'first_half_steps':3,'second_half_steps':3,
           'weight_update_visits':[2,2,2],'flatness_minimum_to_mean':1.,'minimum_visits_met':True,
           'flatness_passed':True,'dense_half_stability_passed':True,'stage_passed':True,
           'ideal_heatbath_events':2,'dense_half_profiles':[profile],'log_weights':[0.,0.,0.]}
    outcome = {'method':'staged_wang_landau','completed':True,'budget_exhausted':False,
               'actual_clock_steps':6,'requested_budget':12,'initial_gain':.1,'minimum_gain':.1,
               'final_gain':.1,'stage_history':[row],
               'criteria':{'flatness_minimum_to_mean':.5,'minimum_weight_update_visits_per_label':1,
                           'minimum_observations_per_dense_label_per_half':1,
                           'maximum_half_mean_N_difference':.25,'maximum_half_mean_Bragg_difference':.1,
                           'dense_diagnostic_nodes':[1]}}
    meta = {'calibration_method':'staged_wang_landau','calibration_cutoff_clock':6,
            'settings':{'calibration_method':'staged_wang_landau','calibration_steps':12,'calibration_gain':.1},
            'calibration_outcome':outcome}
    return data,meta


def test_staged_calibration_flags_are_reconstructed_as_efficiency_only():
    data,meta = staged_record_fixture()
    assert st.validate_calibration_record(data,meta) == 'staged_wang_landau'
    assert st.validate_calibration_record({'activity':np.zeros(3)}, {'calibration_cutoff_clock':0}) == 'legacy_one_over_t'


@pytest.mark.parametrize('finish_second_stage',[False,True])
def test_staged_counts_sum_only_terminal_epochs_and_allow_budget_at_stage_boundary(finish_second_stage):
    data,meta = staged_record_fixture()
    outcome = meta['calibration_outcome']
    first = outcome['stage_history'][0]
    first['gain'] = outcome['initial_gain'] = .2
    if finish_second_stage:
        second = json.loads(json.dumps(first))
        second.update(stage=1,gain=.1,end_clock=12)
        outcome['stage_history'].append(second)
        data['calibration_visits'] *= 2
        data['calibration_weight_update_visits'] *= 2
        outcome['actual_clock_steps'] = meta['calibration_cutoff_clock'] = 12
        outcome['requested_budget'] = meta['settings']['calibration_steps'] = 18
    else:
        outcome.update(completed=False,budget_exhausted=True)
        outcome['requested_budget'] = meta['settings']['calibration_steps'] = 6
    assert st.validate_calibration_record(data,meta) == 'staged_wang_landau'


@pytest.mark.parametrize('mutation',[
    lambda d,m:m.pop('calibration_method'),
    lambda d,m:m['settings'].__setitem__('calibration_method','one_over_t'),
    lambda d,m:m['calibration_outcome'].__setitem__('actual_clock_steps',5),
    lambda d,m:d['calibration_weight_update_visits'].__setitem__(0,1),
    lambda d,m:m['calibration_outcome']['stage_history'][0].__setitem__('stage_passed',False),
    lambda d,m:m['calibration_outcome']['stage_history'][0]['dense_half_profiles'][0]['first_half'].__setitem__('mean_N',4.),
    lambda d,m:m['calibration_outcome']['stage_history'][0].__setitem__('second_half_steps',2),
    lambda d,m:m['calibration_outcome']['stage_history'][0]['log_weights'].__setitem__(1,.01),
    lambda d,m:m['calibration_outcome'].__setitem__('completed',False),
    lambda d,m:m['calibration_outcome'].__setitem__('stage_history',[]),
])
def test_declared_staged_calibration_rejects_missing_or_inconsistent_records(mutation):
    data,meta = staged_record_fixture()
    mutation(data,meta)
    with pytest.raises(ValueError):
        st.validate_calibration_record(data,meta)


def test_fixed_one_over_t_zero_gain_has_no_actual_weight_decrements():
    data,meta = staged_record_fixture()
    meta['calibration_method'] = meta['settings']['calibration_method'] = 'one_over_t'
    meta['settings']['calibration_steps'] = 6
    meta['settings']['calibration_gain'] = 0.
    data['calibration_weight_update_visits'][:] = 0
    meta['calibration_outcome'] = {'method':'one_over_t','actual_clock_steps':6,'requested_budget':6,
                                   'completed':True,'budget_exhausted':False,'stage_history':[]}
    assert st.validate_calibration_record(data,meta) == 'one_over_t'
    data['calibration_weight_update_visits'][0] = 1
    with pytest.raises(ValueError):st.validate_calibration_record(data,meta)


def test_dense_half_counts_across_labels_cannot_exceed_clock_length():
    data,meta = staged_record_fixture()
    out = meta['calibration_outcome']
    out['criteria']['dense_diagnostic_nodes'] = [1,2]
    profiles = out['stage_history'][0]['dense_half_profiles']
    profiles.append(json.loads(json.dumps(profiles[0])))
    profiles[1]['node'] = 2
    for profile in profiles:
        profile['first_half']['observations'] = profile['second_half']['observations'] = 2
    with pytest.raises(ValueError,match='total observation clock'):
        st.validate_calibration_record(data,meta)


@pytest.mark.parametrize('field',['dense','heatbath'])
def test_terminal_stage_observations_reproduce_raw_calibration_histogram(field):
    data,meta = staged_record_fixture()
    if field == 'dense':
        data['calibration_visits'] = np.array([2,1,3])
    else:
        meta['calibration_outcome']['stage_history'][0]['ideal_heatbath_events'] = 3
    with pytest.raises(ValueError,match='Final staged outcome'):
        st.validate_calibration_record(data,meta)


def test_same_stage_cumulative_heatbath_count_cannot_decrease():
    data,meta = staged_record_fixture()
    final = meta['calibration_outcome']['stage_history'][0]
    first = json.loads(json.dumps(final))
    first.update(end_clock=3,epoch_steps=3,first_half_steps=0,second_half_steps=3,
                 weight_update_visits=[1,1,1],dense_half_stability_passed=False,stage_passed=False,
                 ideal_heatbath_events=1)
    profile = first['dense_half_profiles'][0]
    profile.update(sufficient_half_observations=False,mean_N_difference=None,mean_Bragg_difference=None,passed=False)
    profile['first_half'] = {'observations':0,'mean_N':None,'mean_Bragg':None}
    final['ideal_heatbath_events'] = 0
    meta['calibration_outcome']['stage_history'].insert(0,first)
    with pytest.raises(ValueError,match='cannot decrease'):
        st.validate_calibration_record(data,meta)


def radius_path_fixture():
    return {'activity':np.array([-2.,16.,16.,16.]),'radius':np.array([0.,0.,1.,.75]),
            'base_node':1,'cavity_node':2,'physical_nodes':np.array([1,2,3]),
            'radius_path_nodes':np.array([1,3,2]),
            'exchange_edges':np.array([[0,1],[1,2],[1,3],[3,2]])}


def test_ordinary_radius_path_is_separate_from_tagged_endpoint_edge():
    st.validate_radius_path(radius_path_fixture())
    st.validate_radius_path({'activity':np.zeros(3)})


@pytest.mark.parametrize('mutation',[
    lambda d:d.__setitem__('radius_path_nodes',np.array([1,2])),
    lambda d:d.__setitem__('radius_path_nodes',np.array([1,3,3,2])),
    lambda d:d['activity'].__setitem__(3,15.9),
    lambda d:d['radius'].__setitem__(3,1.1),
    lambda d:d.__setitem__('exchange_edges',np.array([[0,1],[1,2],[1,3]])),
    lambda d:d.__setitem__('physical_nodes',np.array([1,2])),
])
def test_radius_path_rejects_missing_edge_or_mixed_physical_target(mutation):
    data = radius_path_fixture()
    mutation(data)
    with pytest.raises(ValueError):st.validate_radius_path(data)


@pytest.mark.parametrize('mutation',[
    lambda a:a.__setitem__((3,0),a[2,0]+2),
    lambda a:a.__setitem__((3,10),99),
    lambda a:a.__setitem__((3,11),1),
    lambda a:a.__setitem__((3,8),4),
])
def test_rejects_sparse_clock_false_regeneration_or_invalid_deletion(mutation):
    data = clock_trace()
    mutation(data)
    with pytest.raises(ValueError):
        st.complete_cycle_rewards(data,1,2,0)


def test_truncating_physical_cap_does_not_assert_untruncated_Palm_identity():
    rewards,_,_ = st.complete_cycle_rewards(clock_trace(),1,2,0)
    result = st.estimate_totals(rewards.sum(axis=0),**parameters(),physical_cap_certified=False)
    assert result['mean_N_base'] == 3
    assert not any('Palm' in field for field in result)


def test_whole_cycle_bootstrap_reports_undefined_rare_visit_draws_without_discarding():
    rewards,_,_ = st.complete_cycle_rewards(clock_trace(),1,2,0)
    point,errors,fraction = st.bootstrap_complete_cycles(rewards,100,7,**parameters())
    assert point['q_solid_base'] == .1
    assert fraction['mean_N_cavity'] < 1
    assert errors['mean_N_cavity'] is None
    assert errors['work_from_label_dwell'] is None


def test_whole_cycle_bootstrap_uses_common_resampling_and_preserves_fixed_relations():
    # Equal endpoint dwell per complete cycle and N_c=N_b-1 pathwise.
    rewards = np.array([[3,3,9,27,6,0,0,0,0,0],
                        [2,2,8,32,6,2,2,8,0,0],
                        [0,0,0,0,0,0,0,0,0,0]],float)
    point,errors,fraction = st.bootstrap_complete_cycles(rewards,200,2,**parameters())
    assert point['number_response_direct'] == -1
    # Some bootstrap draws select only the zero-dwell cycle: they must remain
    # visibly undefined rather than being removed to fabricate a zero error.
    assert fraction['number_response_direct'] < 1
    assert errors['number_response_direct'] is None


def test_exposed_whole_cycle_covariance_uses_common_draws_and_exact_sign():
    rewards = np.array([[3,2,9,27,6,0,0,0,0,0],
                        [2,3,8,32,6,2,2,8,0,0]],float)
    point,errors,fraction,cov = st.bootstrap_complete_cycles(
        rewards,200,2,return_covariance=True,**parameters())
    index = {name:i for i,name in enumerate(cov['columns'])}
    work,logz = index['work_from_label_dwell'],index['log_partition_ratio_cavity_to_base']
    assert cov['matrix'][work][work] == pytest.approx(errors['work_from_label_dwell']**2)
    assert cov['matrix'][work][logz] == pytest.approx(-errors['work_from_label_dwell']**2)
    assert fraction['work_from_label_dwell'] == 1


def raw_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(st,'ROOT',tmp_path)
    source = tmp_path/'synthetic_source.py'
    source.write_text('# Integrity fixture only; not an executed simulation.\n')
    samples = clock_trace()
    samples[:,5] = 1
    samples[:,7] = 2
    arrays = dict(schema=st.SCHEMA,samples=samples,log_weights=np.array([0.,1.,2.]),
        volume=8.,base_node=1,cavity_node=2,ideal_node=0,physical_nodes=np.array([1,2]),
        diameter=np.ones(3),radius=np.array([0.,0.,1.]),activity=np.array([-2.,3.,3.]),
        pair_penalty=np.array([0.,np.inf,np.inf]),umbrella_stiffness=np.zeros(3),
        umbrella_preferred=np.zeros(3),number_stiffness=np.zeros(3),number_preferred=np.zeros(3),
        displacement=np.full(3,.1),
        exchange_edges=np.array([[0,1],[1,2]]),sites=4,cap=20,
        coordinate_configurations=np.zeros((2,20,3)),coordinate_clock_steps=np.array([2,3]),
        coordinate_labels=samples[[1,2],1].astype(int),coordinate_centers=np.ones((2,3)))
    metadata = dict(schema=st.SCHEMA,weights_frozen_before_production=True,
        columns=list(st.COLUMNS),
        sample_cadence_clock_steps=1,calibration_cutoff_clock=0,
        settings={'production_steps':len(samples)},source_snapshots={
            name:['synthetic_source.py',st.sha256(source)] for name in ('driver','kernel','graph_validation')},
        input_sha256={})
    path = tmp_path/'synthetic.npz'
    return path,arrays,metadata,source


def save_fixture(path,arrays,metadata):
    np.savez_compressed(path,**arrays)
    metadata['raw_sha256'] = st.sha256(path)
    path.with_suffix('.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('mutation',[None,'missing_raw','shape','outcome','mode'])
@pytest.mark.parametrize('declare_projection_none',[False,True])
def test_loader_binds_new_uniform_quota_even_without_any_adaptive_clock(tmp_path,monkeypatch,mutation,declare_projection_none):
    path,arrays,metadata,_ = raw_fixture(tmp_path,monkeypatch)
    quota = np.full(3,1/3)
    arrays.update(calibration_label_quota=quota,calibration_visits=np.zeros(3,dtype=int),
                  calibration_weight_update_visits=np.zeros(3,dtype=int))
    metadata['settings'].update(calibration_method='one_over_t',calibration_label_quota='uniform',
                                calibration_steps=0,calibration_gain=0.)
    metadata.update(calibration_method='one_over_t',calibration_label_quota=quota.tolist(),
        calibration_label_quota_mode='uniform',calibration_flatness_measure='minimum(counts)/mean(counts)',
        calibration_weight_update_rule='legacy uniform decrement-only; periodic ideal-node gauge',
        calibration_gauge_interval_clock_steps=1000,calibration_log_weight_clip=[-1e5,1e5],
        calibration_outcome={'method':'one_over_t','actual_clock_steps':0,'requested_budget':0,
            'completed':True,'budget_exhausted':False,'stage_history':[],
            'label_quota':quota.tolist(),'flatness_measure':'minimum(counts)/mean(counts)'})
    if declare_projection_none:
        arrays['calibration_normalizer_projection'] = np.array('none')
        metadata['calibration_normalizer_projection'] = 'none'
        metadata['settings']['calibration_normalizer_projection'] = 'none'
        metadata['calibration_outcome']['normalizer_projection_mode'] = 'none'
    if mutation == 'missing_raw':arrays.pop('calibration_label_quota')
    elif mutation == 'shape':arrays['calibration_label_quota'] = np.ones((3,1))/3
    elif mutation == 'outcome':metadata['calibration_outcome']['label_quota'] = [.2,.3,.5]
    elif mutation == 'mode':metadata['calibration_label_quota_mode'] = 'endpoint_priority'
    save_fixture(path,arrays,metadata)
    if mutation is None:
        data,_,_ = st.load_validated_run(path)
        np.testing.assert_array_equal(data['calibration_label_quota'],quota)
    else:
        with pytest.raises(ValueError):st.load_validated_run(path)


def relocation_metadata(metadata,probability=.25):
    metadata['settings']['vacancy_relocation_probability'] = probability
    metadata.update(
        vacancy_relocation_proposal='uniform_tag_residual_vacancy_mh_v1',
        vacancy_relocation_scope='same-diameter structured nodes (pair_penalty >= 8 or umbrella_stiffness > 0)',
        vacancy_relocation_timing='after GCMC/local translations, before fixed-clock observation and label updates')
    return metadata


@pytest.mark.parametrize('probability',[0.,.25,1.])
def test_relocation_counter_contract_and_legacy_zero_preserve_trace_observations(tmp_path,monkeypatch,probability):
    path,arrays,metadata,_ = raw_fixture(tmp_path,monkeypatch)
    original = arrays['samples'].copy()
    relocation_metadata(metadata,probability)
    eligible = int(np.count_nonzero(original[:,1] != 0))
    attempts = 0 if probability == 0 else eligible if probability == 1 else 2
    arrays['vacancy_relocation_counts'] = np.array([attempts,attempts//2],dtype=np.int64)
    save_fixture(path,arrays,metadata)
    data,_,_ = st.load_validated_run(path)
    np.testing.assert_array_equal(data['samples'],original)
    assert data['vacancy_relocation']['eligible_nonempty_clock_steps'] == eligible
    assert data['vacancy_relocation']['attempted'] == attempts
    assert data['label_update_protocol'] == 'lazy_neighbor_tagged'
    if probability == 0:
        arrays.pop('vacancy_relocation_counts')
        for key in ('vacancy_relocation_proposal','vacancy_relocation_scope','vacancy_relocation_timing'):
            metadata.pop(key)
        metadata['settings'].pop('vacancy_relocation_probability')
        save_fixture(path,arrays,metadata)
        data,_,_ = st.load_validated_run(path)
        assert data['vacancy_relocation']['attempted'] == 0


@pytest.mark.parametrize('case',[
    'negative_probability','large_probability','nan_probability','boolean_probability','vector_probability',
    'missing_counts','shape','fractional_count','negative_count','more_acceptances',
    'too_many_attempts','unit_probability_missing_attempts','disabled_nonzero',
    'proposal','scope','timing','missing_timing',
])
def test_loader_rejects_invalid_relocation_clock_scope_and_protocol(tmp_path,monkeypatch,case):
    path,arrays,metadata,_ = raw_fixture(tmp_path,monkeypatch)
    relocation_metadata(metadata)
    arrays['vacancy_relocation_counts'] = np.array([2,1])
    if case == 'negative_probability':metadata['settings']['vacancy_relocation_probability'] = -.1
    elif case == 'large_probability':metadata['settings']['vacancy_relocation_probability'] = 1.1
    elif case == 'nan_probability':metadata['settings']['vacancy_relocation_probability'] = float('nan')
    elif case == 'boolean_probability':metadata['settings']['vacancy_relocation_probability'] = True
    elif case == 'vector_probability':metadata['settings']['vacancy_relocation_probability'] = [.25]
    elif case == 'missing_counts':arrays.pop('vacancy_relocation_counts')
    elif case == 'shape':arrays['vacancy_relocation_counts'] = np.array([[2,1]])
    elif case == 'fractional_count':arrays['vacancy_relocation_counts'] = np.array([2.,.5])
    elif case == 'negative_count':arrays['vacancy_relocation_counts'] = np.array([-2,0])
    elif case == 'more_acceptances':arrays['vacancy_relocation_counts'] = np.array([2,3])
    elif case == 'too_many_attempts':arrays['vacancy_relocation_counts'] = np.array([len(arrays['samples']),0])
    elif case == 'unit_probability_missing_attempts':metadata['settings']['vacancy_relocation_probability'] = 1.
    elif case == 'disabled_nonzero':metadata['settings']['vacancy_relocation_probability'] = 0.
    elif case in ('proposal','scope','timing'):metadata['vacancy_relocation_'+case] = 'wrong'
    elif case == 'missing_timing':metadata.pop('vacancy_relocation_timing')
    save_fixture(path,arrays,metadata)
    with pytest.raises(ValueError):st.load_validated_run(path)


def test_relocation_eligibility_excludes_empty_unstructured_and_different_diameter_rows():
    data = dict(samples=np.array([[0,0,1],[0,1,1],[0,2,1],[0,3,1],[0,1,0]],float),
                base_node=1,diameter=np.array([1.,1.,1.,.5]),
                pair_penalty=np.array([0.,np.inf,2.,np.inf]),
                umbrella_stiffness=np.array([0.,0.,1.,0.]),
                vacancy_relocation_counts=np.array([2,1]))
    metadata = relocation_metadata({'settings':{}},1.)
    result = st.validate_vacancy_relocation(data,metadata)
    assert result['eligible_nonempty_clock_steps'] == 2
    data['vacancy_relocation_counts'][0] = 3
    with pytest.raises(ValueError,match='eligible clocks'):
        st.validate_vacancy_relocation(data,metadata)


def test_loader_and_analysis_keep_separate_schema_and_exact_input_provenance(tmp_path,monkeypatch):
    path,arrays,metadata,source = raw_fixture(tmp_path,monkeypatch)
    save_fixture(path,arrays,metadata)
    data,record,inputs = st.load_validated_run(path)
    assert data['physical_cap_certified']
    assert data['coordinate_sample_indices'].tolist() == [1,2]
    assert source in inputs
    output = tmp_path/'summary.json'
    result = st.write_analysis([path],output,20,1)['synthetic']
    assert result['coverage']['zero_physical_dwell_cycles'] == 1
    assert result['estimates_from_complete_cycle_reward_ratios']['q_solid_base'] == .1
    assert result['adequacy_warnings']
    assert any('incomplete cycles' in warning for warning in result['adequacy_warnings'])
    assert any('more than20%' in warning for warning in result['adequacy_warnings'])
    assert 'duration-dependent' in result['uncertainty_scope']
    assert 'product_target' not in result
    provenance = json.loads(output.with_suffix('.metadata.json').read_text())
    assert provenance['input_sha256']['synthetic.npz'] == st.sha256(path)
    assert provenance['output_sha256']['summary.json'] == st.sha256(output)
    snapshot = tmp_path/provenance['source_snapshot']
    assert snapshot.read_bytes() == st.EXECUTION_SOURCE_BYTES
    assert st.sha256(snapshot) == provenance['script_sha256']


def test_analyzer_source_snapshot_uses_startup_bytes_and_refuses_replacement(tmp_path,monkeypatch):
    monkeypatch.setattr(st,'ROOT',tmp_path)
    source = tmp_path/'edited_later.py'
    source.write_text('# Not the imported source bytes.\n')
    monkeypatch.setattr(st,'__file__',str(source))
    snapshot = st.preserve_execution_source()
    assert snapshot.read_bytes() == st.EXECUTION_SOURCE_BYTES
    assert st.preserve_execution_source() == snapshot
    snapshot.write_text('# Corrupted immutable snapshot.\n')
    with pytest.raises(ValueError,match='executed bytes'):
        st.preserve_execution_source()


def test_analyzer_snapshot_directory_cannot_escape_project(tmp_path,monkeypatch):
    root = tmp_path/'project'
    root.mkdir()
    (root/'data_raw').mkdir()
    (root/'data_raw'/'source_snapshots').symlink_to(tmp_path,target_is_directory=True)
    monkeypatch.setattr(st,'ROOT',root)
    with pytest.raises(ValueError,match='inside the project'):
        st.preserve_execution_source()


@pytest.mark.parametrize('existing',['output','sidecar','both'])
def test_analysis_rejects_existing_derivative_prefix_before_loading_inputs(tmp_path,monkeypatch,existing):
    output = tmp_path/'preserved.json'
    sidecar = output.with_suffix('.metadata.json')
    protected = [output,sidecar] if existing == 'both' else [output if existing == 'output' else sidecar]
    for path in protected:path.write_bytes(b'Historical derivative evidence\n')
    def forbidden_analysis(*args,**kwargs):
        raise AssertionError('Collision must be detected before reading any raw input')
    monkeypatch.setattr(st,'analyze',forbidden_analysis)
    with pytest.raises(ValueError,match='prefix must be new'):
        st.write_analysis([tmp_path/'not_even_loaded.npz'],output)
    for path in protected:assert path.read_bytes() == b'Historical derivative evidence\n'


@pytest.mark.parametrize('case',[
    'replica_schema','wrong_raw_hash','unfrozen_weights','sparse_clock','source_hash',
    'weight_nan','physical_number_bias','physical_order_bias','hard_overlap',
    'cavity_overlap','occupancy_identity','calibration_rows','coordinates_label',
    'duplicate_graph_edge','missing_tag_edge','ideal_not_ideal',
    'column_order','missing_graph_source','cap_infinite','label_nan','coordinate_center',
    'coordinate_missing_center','coordinate_time_nan','unverified_graph_sidecar',
    'partial_cycle_deletion',
    'singleton_deletion','empty_order',
])
def test_loader_rejects_invalid_target_clock_regeneration_and_provenance(tmp_path,monkeypatch,case):
    path,arrays,metadata,source = raw_fixture(tmp_path,monkeypatch)
    if case == 'replica_schema':metadata['schema'] = 'replica product'
    elif case == 'unfrozen_weights':metadata['weights_frozen_before_production'] = False
    elif case == 'sparse_clock':metadata['sample_cadence_clock_steps'] = 10
    elif case == 'source_hash':source.write_text('# changed fixture bytes\n')
    elif case == 'weight_nan':arrays['log_weights'][1] = np.nan
    elif case == 'physical_number_bias':arrays['number_stiffness'][1] = .1
    elif case == 'physical_order_bias':arrays['umbrella_stiffness'][2] = .1
    elif case == 'hard_overlap':arrays['samples'][0,4] = 1
    elif case == 'cavity_overlap':arrays['samples'][15,7] = .2
    elif case == 'occupancy_identity':arrays['samples'][0,5] = 2
    elif case == 'calibration_rows':metadata['calibration_cutoff_clock'] = 1
    elif case == 'coordinates_label':arrays['coordinate_labels'][0] = 2
    elif case == 'duplicate_graph_edge':arrays['exchange_edges'] = np.array([[0,1],[1,2],[1,0]])
    elif case == 'missing_tag_edge':arrays['exchange_edges'] = np.array([[0,1],[0,2]])
    elif case == 'ideal_not_ideal':arrays['pair_penalty'][0] = .1
    elif case == 'column_order':metadata['columns'][1:3] = metadata['columns'][2:0:-1]
    elif case == 'missing_graph_source':metadata['source_snapshots'].pop('graph_validation')
    elif case == 'cap_infinite':arrays['cap'] = np.inf
    elif case == 'label_nan':arrays['base_node'] = np.nan
    elif case == 'coordinate_center':arrays['coordinate_centers'][0,0] = 2
    elif case == 'coordinate_missing_center':arrays.pop('coordinate_centers')
    elif case == 'coordinate_time_nan':arrays['coordinate_clock_steps'] = np.array([2.,np.nan])
    elif case == 'unverified_graph_sidecar':metadata['settings']['graph_from'] = 'unverified.npz'
    elif case == 'partial_cycle_deletion':arrays['samples'][0,8] = 4
    elif case == 'singleton_deletion':
        arrays['samples'][0,2] = 1
        arrays['samples'][0,3] = 1
        arrays['samples'][0,5] = 3
    elif case == 'empty_order':
        arrays['samples'][0,2] = 0
        arrays['samples'][0,5] = 4
    save_fixture(path,arrays,metadata)
    if case == 'wrong_raw_hash':
        metadata['raw_sha256'] = 'incorrect'
        path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        st.load_validated_run(path)


def test_analysis_does_not_overwrite_raw_or_source(tmp_path,monkeypatch):
    path,arrays,metadata,source = raw_fixture(tmp_path,monkeypatch)
    save_fixture(path,arrays,metadata)
    before = path.with_suffix('.json').read_bytes()
    with pytest.raises(ValueError):
        st.write_analysis([path],path.with_suffix('.json'))
    assert path.with_suffix('.json').read_bytes() == before


def test_no_regeneration_never_substitutes_physical_dwell_for_cycle_estimates(tmp_path,monkeypatch):
    path,arrays,metadata,_ = raw_fixture(tmp_path,monkeypatch)
    arrays['samples'][:,1] = 1
    arrays['samples'][:,10:] = 0
    arrays['coordinate_labels'][:] = 1
    save_fixture(path,arrays,metadata)
    result,_ = st.analyze(path,20,1)
    assert result['coverage']['complete_cycles'] == 0
    assert result['coverage']['complete_cycle_clock_steps'] == 0
    assert all(not np.isfinite(v) for v in result['estimates_from_complete_cycle_reward_ratios'].values())
    assert all(v is None for v in result['whole_cycle_bootstrap_SE'].values())
    assert all(v is None for v in result['boundary_fraction_uncertainty'].values())
    assert result['fraction_of_clock_in_incomplete_cycles'] == 1
    assert result['label_clock_profile'][1]['clock_visits'] == len(arrays['samples'])


def imported_native_fixture(tmp_path,monkeypatch):
    path,arrays,metadata,source = raw_fixture(tmp_path,monkeypatch)
    graph = tmp_path/'native.npz'
    names = ('volume','diameter','radius','activity','pair_penalty','umbrella_stiffness',
             'umbrella_preferred','number_stiffness','number_preferred','displacement',
             'exchange_edges','base_node','cavity_node','ideal_node','physical_nodes','sites','cap')
    arrays['initial_log_weights'] = arrays['log_weights'].copy()
    np.savez_compressed(graph,schema='cavcoex.native_tempering_graph.v1',
                        initial_log_weights=arrays['log_weights'],**{k:arrays[k] for k in names})
    record = {'schema':'cavcoex.native_tempering_graph.v1','raw_sha256':st.sha256(graph),
              'source_snapshots':dict(metadata['source_snapshots'],generator=metadata['source_snapshots']['driver']),
              'input_sha256':{source.name:st.sha256(source)}}
    graph.with_suffix('.json').write_text(json.dumps(record))
    metadata['settings']['graph_from'] = graph.name
    metadata['input_sha256'] = {p.name:st.sha256(p) for p in (graph,graph.with_suffix('.json'))}
    return path,arrays,metadata,graph


def test_imported_native_graph_is_bound_to_recorded_targets_and_proposals(tmp_path,monkeypatch):
    path,arrays,metadata,graph = imported_native_fixture(tmp_path,monkeypatch)
    save_fixture(path,arrays,metadata)
    _,_,inputs = st.load_validated_run(path)
    assert {graph,graph.with_suffix('.json')} <= inputs


def test_native_missing_parameter_provenance_cannot_implicitly_become_explicit(tmp_path,monkeypatch):
    path,arrays,metadata,graph = imported_native_fixture(tmp_path,monkeypatch)
    record = json.loads(graph.with_suffix('.json').read_text())
    record['input_sha256'] = {}
    graph.with_suffix('.json').write_text(json.dumps(record))
    metadata['input_sha256'][graph.with_suffix('.json').name] = st.sha256(graph.with_suffix('.json'))
    save_fixture(path,arrays,metadata)
    with pytest.raises(ValueError,match='nonempty input provenance'):st.load_validated_run(path)


def explicit_settings(stages=1):
    return dict(volume=30.8496,sites=32,cap=80,sigma=1.,base_field=16.,
                displacement=.15,cavity_radius_stages=stages)


def test_explicit_native_reconstruction_is_self_contained_and_keeps_physical_radii_separate():
    target,weights = st.reconstruct_explicit_native_graph(explicit_settings(16))
    assert len(weights) == 56
    path = target['radius_path_nodes']
    np.testing.assert_allclose(target['radius'][path],1-(1-np.arange(17)/16)**2)
    np.testing.assert_array_equal(target['activity'][path],16.)
    assert weights[target['ideal_node']] == 0
    assert target['cap'] == 80 and target['sites'] == 32
    st.validate_radius_path(target)


@pytest.mark.parametrize('mutation',[
    lambda p:p.pop('volume'),lambda p:p.__setitem__('cap',100001),
    lambda p:p.__setitem__('sites',100001),
    lambda p:p.__setitem__('cavity_radius_stages',257),
    lambda p:p.__setitem__('base_field',18),lambda p:p.__setitem__('sigma',.9),
])
def test_explicit_reconstruction_requires_bound_scalars_and_resource_limits(mutation):
    parameters = explicit_settings()
    mutation(parameters)
    with pytest.raises(ValueError):st.reconstruct_explicit_native_graph(parameters)


@pytest.mark.parametrize('stages',[1,4,16])
def test_compact_reconstruction_binds_number_branches_and_unbiased_physical_nodes(stages):
    parameters = dict(explicit_settings(stages),graph_kind='compact')
    target,weights = st.reconstruct_explicit_native_graph(parameters)
    nodes = 145+stages-1
    assert len(weights) == nodes and np.isfinite(weights).all() and weights[0] == 0
    assert (target['base_node'],target['cavity_node'],target['ideal_node']) == (70,71,0)
    assert target['activity'][0] == math.log(1/1.06448)
    physical = target['physical_nodes']
    np.testing.assert_array_equal(physical,np.r_[70,71,np.arange(145,nodes)])
    assert np.isposinf(target['pair_penalty'][physical]).all()
    assert np.all(target['umbrella_stiffness'][physical] == 0)
    assert np.all(target['number_stiffness'][physical] == 0)
    assert np.all(target['activity'][physical] == parameters['base_field'])
    np.testing.assert_array_equal(target['number_stiffness'][21:70],4.)
    np.testing.assert_array_equal(target['number_preferred'][21:70],parameters['sites'])
    np.testing.assert_array_equal(target['number_stiffness'][89:120],4.)
    np.testing.assert_array_equal(target['number_preferred'][89:120],parameters['volume']/1.06448)
    np.testing.assert_array_equal(target['number_stiffness'][134:139],[.25,.5,1.,2.,4.])
    np.testing.assert_array_equal(target['number_stiffness'][139:145],[2.,1.,.5,.25,.1,0.])
    edges = set(map(tuple,target['exchange_edges']))
    assert {(20,134),(138,21),(69,139),(144,70),(20,120),(123,89),(119,124),(133,70)} <= edges
    assert not {(20,21),(69,70),(20,89),(119,70)} & edges
    st.validate_radius_path(target)
    # The physical matched-cavity work guess still has the specified model
    # contrast; no branch restraint is allowed to enter a physical target.
    _,hard_weights = st.reconstruct_explicit_native_graph(explicit_settings(stages))
    hard,_ = st.reconstruct_explicit_native_graph(explicit_settings(stages))
    assert weights[71]-weights[70] == pytest.approx(
        hard_weights[hard['cavity_node']]-hard_weights[hard['base_node']],abs=1e-12)


@pytest.mark.parametrize('stages',[1,16])
def test_compact_historical_default_and_explicit_four_are_exactly_equal(stages):
    parameters = dict(explicit_settings(stages),graph_kind='compact')
    legacy,legacy_weights = st.reconstruct_explicit_native_graph(parameters)
    explicit,explicit_weights = st.reconstruct_explicit_native_graph(
        dict(parameters,ordered_hardening_number_strength=4.))
    assert legacy.keys() == explicit.keys()
    for name in legacy:np.testing.assert_array_equal(legacy[name],explicit[name])
    np.testing.assert_array_equal(legacy_weights,explicit_weights)


@pytest.mark.parametrize('strength',[.125,1.,2.5])
@pytest.mark.parametrize('stages',[1,16])
def test_compact_hardening_relaxation_keeps_ordering_and_physical_laws_fixed(strength,stages):
    parameters = dict(explicit_settings(stages),graph_kind='compact')
    old,old_weights = st.reconstruct_explicit_native_graph(parameters)
    target,weights = st.reconstruct_explicit_native_graph(
        dict(parameters,ordered_hardening_number_strength=strength))
    assert len(weights) == 147+stages-1
    np.testing.assert_array_equal(target['number_stiffness'][21:39],4.)
    np.testing.assert_array_equal(target['number_stiffness'][39:70],strength)
    np.testing.assert_array_equal(target['number_stiffness'][139:145],
                                  [strength/2,strength/4,strength/8,strength/16,strength/40,0.])
    np.testing.assert_array_equal(target['number_stiffness'][145:147],
                                  [math.sqrt(4*strength),strength])
    for name in ('pair_penalty','umbrella_stiffness','umbrella_preferred','activity','radius','number_preferred'):
        np.testing.assert_array_equal(target[name][145:147],old[name][38])
    for name in ('diameter','radius','activity','pair_penalty','umbrella_stiffness',
                 'umbrella_preferred','number_stiffness','number_preferred','displacement'):
        np.testing.assert_array_equal(target[name][72:139],old[name][72:139])
        np.testing.assert_array_equal(target[name][target['physical_nodes']],old[name][old['physical_nodes']])
    np.testing.assert_array_equal(weights[target['physical_nodes']],old_weights[old['physical_nodes']])
    edges = set(map(tuple,target['exchange_edges']))
    assert {(38,145),(145,146),(146,39)} <= edges and (38,39) not in edges
    st.validate_radius_path(target)
    # A death at N*=sites changes only the number-restraint contribution by H.
    nstar = target['sites']
    assert strength*((nstar-1-nstar)**2-(nstar-nstar)**2) == strength
    assert target['number_stiffness'][39] < old['number_stiffness'][39]


@pytest.mark.parametrize('value',[True,False,None,'1',[],[1],0.,-1.,4.01,np.nan,np.inf,-np.inf])
def test_compact_hardening_strength_rejects_invalid_declared_setting(value):
    parameters = dict(explicit_settings(),graph_kind='compact',ordered_hardening_number_strength=value)
    with pytest.raises(ValueError,match='hardening'):st.reconstruct_explicit_native_graph(parameters)


def test_nondefault_hardening_strength_cannot_be_silently_ignored_by_hard_graph():
    with pytest.raises(ValueError,match='compact graph'):
        st.reconstruct_explicit_native_graph(dict(explicit_settings(),ordered_hardening_number_strength=1.))


@pytest.mark.parametrize('mutation',[
    lambda p:p.__setitem__('graph_kind','unknown'),
    lambda p:p.__setitem__('graph_kind',None),
    lambda p:p.__setitem__('sites',12),
    lambda p:p.__setitem__('sites',108),
    lambda p:p.__setitem__('sites',32.5),
])
def test_compact_reconstruction_rejects_unknown_design_nonfcc_and_reference_above_cap(mutation):
    parameters = dict(explicit_settings(),graph_kind='compact')
    mutation(parameters)
    with pytest.raises(ValueError):st.reconstruct_explicit_native_graph(parameters)


@pytest.mark.parametrize('stages',[1,16])
def test_explicit_legacy_graph_kind_default_matches_declared_hard_activity(stages):
    old,old_weights = st.reconstruct_explicit_native_graph(explicit_settings(stages))
    new,new_weights = st.reconstruct_explicit_native_graph(dict(explicit_settings(stages),graph_kind='hard_activity'))
    assert old.keys() == new.keys()
    for name in old:np.testing.assert_array_equal(old[name],new[name])
    np.testing.assert_array_equal(old_weights,new_weights)


@pytest.mark.parametrize('volume,sites,cap,field,stages',[
    (30.8496,32,80,16.,1),
    (30.8496,32,80,16.,16),
    (108*.96405+.00123456789,108,256,16.00000012345,7),
    (2.,4,6,.3,3),
])
@pytest.mark.parametrize('strength',[4.,1.,2.5])
def test_independent_compact_reconstruction_matches_generator_at_nonbinary_volumes(volume,sites,cap,field,stages,strength):
    # Importing the graph builder does not run or compile a simulation kernel.
    # The analyzer above has its own declarative schedule and partition sum.
    from scripts import continuum_tempering_graph as graph
    parameters = dict(volume=volume,sites=sites,cap=cap,sigma=1.,base_field=field,
                      displacement=.137,cavity_radius_stages=stages,graph_kind='compact',
                      ordered_hardening_number_strength=strength)
    expected,guess = st.reconstruct_explicit_native_graph(parameters)
    actual,actual_guess = graph.build_compact_graph(volume,sites,cap,1.,field,.137,stages,strength)
    assert actual['volume'] == volume
    for name,value in actual.items():np.testing.assert_array_equal(value,expected[name])
    np.testing.assert_allclose(actual_guess,guess,rtol=0,atol=1e-10)


@pytest.mark.parametrize('strength',[4.,1.])
@pytest.mark.parametrize('mutation',[None,'kind','model','constants','imported','number_target','number_strength','weights','hardening_setting'])
def test_compact_native_provenance_binds_auxiliary_design_and_efficiency_model(tmp_path,monkeypatch,mutation,strength):
    monkeypatch.setattr(st,'ROOT',tmp_path)
    parameters = dict(explicit_settings(4),graph_kind='compact',ordered_hardening_number_strength=strength)
    target,weights = st.reconstruct_explicit_native_graph(parameters)
    source = tmp_path/'source.py'
    source.write_text('# Synthetic source record; no simulation claimed.\n')
    record = {'schema':'cavcoex.native_tempering_graph.v1','construction_mode':'explicit_box',
        'settings':parameters,'input_sha256':{},
        'initial_weight_model':'soft_cs_number_gaussian_bragg_v1',
        'initial_weight_model_constants':{'effective_packing_fraction_cutoff':.74,'Bragg_mean':.2,'Bragg_variance':.01},
        'source_snapshots':{key:[source.name,st.sha256(source)]
                            for key in ('generator','kernel','graph_validation','target_validation')}}
    if mutation == 'kind':record['settings'].pop('graph_kind')
    elif mutation == 'model':record['initial_weight_model'] = 'unspecified'
    elif mutation == 'constants':record['initial_weight_model_constants']['Bragg_variance'] = .02
    elif mutation == 'imported':
        record['construction_mode'] = 'validated_parameter_archive'
        record['input_sha256'][source.name] = st.sha256(source)
    elif mutation == 'number_target':target['number_preferred'][30] -= .25
    elif mutation == 'number_strength':target['number_stiffness'][89] = 0.
    elif mutation == 'weights':weights[22] += .01
    elif mutation == 'hardening_setting':record['settings']['ordered_hardening_number_strength'] = 2.
    graph = tmp_path/'compact.npz'
    np.savez_compressed(graph,schema=record['schema'],initial_log_weights=weights,**target)
    record['raw_sha256'] = st.sha256(graph)
    graph.with_suffix('.json').write_text(json.dumps(record))
    data = dict(target,initial_log_weights=weights.copy())
    verified = {graph,graph.with_suffix('.json')}
    if mutation is None:
        st.verify_imported_targets(data,{'settings':{'graph_from':graph.name}},verified)
        assert source in verified
    else:
        with pytest.raises(ValueError):
            st.verify_imported_targets(data,{'settings':{'graph_from':graph.name}},verified)


@pytest.mark.parametrize('mutation',[None,'mode','origin','validator','volume','weights'])
def test_explicit_native_graph_provenance_and_arrays_are_independently_rebuilt(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(st,'ROOT',tmp_path)
    parameters = explicit_settings(4)
    target,weights = st.reconstruct_explicit_native_graph(parameters)
    data = dict(target,initial_log_weights=weights.copy())
    source = tmp_path/'source.py'
    source.write_text('# synthetic fixture producer\n')
    graph = tmp_path/'explicit.npz'
    record = {'schema':'cavcoex.native_tempering_graph.v1','construction_mode':'explicit_box',
              'settings':parameters,'input_sha256':{},
              'source_snapshots':{key:[source.name,st.sha256(source)]
                                  for key in ('generator','kernel','graph_validation','target_validation')}}
    if mutation == 'mode':record.pop('construction_mode')
    elif mutation == 'origin':record['input_sha256'][source.name] = st.sha256(source)
    elif mutation == 'validator':record['source_snapshots'].pop('target_validation')
    elif mutation == 'volume':record['settings']['volume'] += .1
    elif mutation == 'weights':weights[1] += .01
    np.savez_compressed(graph,schema=record['schema'],initial_log_weights=weights,**target)
    record['raw_sha256'] = st.sha256(graph)
    graph.with_suffix('.json').write_text(json.dumps(record))
    metadata = {'settings':{'graph_from':graph.name}}
    verified = {graph,graph.with_suffix('.json')}
    if mutation is None:
        st.verify_imported_targets(data,metadata,verified)
        assert source in verified
    else:
        with pytest.raises(ValueError):st.verify_imported_targets(data,metadata,verified)


@pytest.mark.parametrize('mutation',['activity','displacement','initial_weight','site_count'])
def test_imported_native_graph_rejects_hash_valid_but_mismatched_output(tmp_path,monkeypatch,mutation):
    path,arrays,metadata,_ = imported_native_fixture(tmp_path,monkeypatch)
    if mutation == 'activity':arrays['activity'][1:] += .1
    elif mutation == 'displacement':arrays['displacement'][1] += .1
    elif mutation == 'initial_weight':arrays['initial_log_weights'][1] += .1
    elif mutation == 'site_count':arrays['sites'] += 1
    save_fixture(path,arrays,metadata)
    with pytest.raises(ValueError,match='differ'):
        st.load_validated_run(path)


@pytest.mark.parametrize('mutation',[None,'unfrozen','proposal_mismatch','initial_weight_mismatch'])
def test_imported_frozen_weight_archive_integrity_and_gauge(tmp_path,monkeypatch,mutation):
    path,arrays,metadata,_ = imported_native_fixture(tmp_path,monkeypatch)
    weights_path = tmp_path/'weights.npz'
    source_arrays = {k:np.array(v,copy=True) for k,v in arrays.items()}
    source_arrays['log_weights'] += 5.  # Arbitrary common gauge must cancel.
    source_metadata = json.loads(json.dumps(metadata))
    source_metadata['settings']['graph_from'] = None
    if mutation == 'unfrozen':source_metadata['weights_frozen_before_production'] = False
    elif mutation == 'proposal_mismatch':source_arrays['displacement'][0] += .01
    elif mutation == 'initial_weight_mismatch':arrays['initial_log_weights'][1] += .01
    save_fixture(weights_path,source_arrays,source_metadata)
    metadata['settings']['weights_from'] = weights_path.name
    metadata['input_sha256'].update({p.name:st.sha256(p) for p in (weights_path,weights_path.with_suffix('.json'))})
    save_fixture(path,arrays,metadata)
    if mutation is None:
        data,_,inputs = st.load_validated_run(path)
        np.testing.assert_array_equal(data['initial_log_weights'],[0.,1.,2.])
        assert weights_path in inputs
    else:
        with pytest.raises(ValueError):
            st.load_validated_run(path)
