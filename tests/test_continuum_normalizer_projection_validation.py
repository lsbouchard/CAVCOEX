"""Independent physical-domain, convex KKT and pre/post replay controls."""
import copy
import math

import numpy as np
import pytest

from scripts import continuum_single_walker_analyze as audit


def protocol(field=0.):
    q = np.array([.02,.30,.30,.19,.19])
    data = {'volume':8.,'cap':16,'activity':np.array([-2.,field,field,field,0.]),
        'diameter':np.ones(5),'radius':np.array([0.,0.,1.,.5,0.]),
        'pair_penalty':np.array([0.,np.inf,np.inf,np.inf,2.]),
        'umbrella_stiffness':np.zeros(5),'number_stiffness':np.zeros(5),
        'base_node':1,'cavity_node':2,'ideal_node':0,'physical_nodes':np.array([1,2,3]),
        'radius_path_nodes':np.array([1,3,2]),
        'calibration_normalizer_projection':np.array('nested_physical_radius')}
    meta = {'calibration_normalizer_projection':'nested_physical_radius',
        'settings':{'calibration_normalizer_projection':'nested_physical_radius'},
        'calibration_method':'staged_wang_landau','calibration_cutoff_clock':100,
        'calibration_gauge_interval_clock_steps':1000,
        'calibration_normalizer_projection_timing':
            'after each discarded chunk, before staged calibration observation; never frozen production',
        'source_snapshots':{'normalizer_projection':['synthetic_source_not_executed.py','0'*64]},
        'calibration_outcome':{'normalizer_projection_mode':'nested_physical_radius',
                               'stage_history':[{'normalizer_projection':{}}]}}
    return data,meta,q


def manual_row(data,q,before,after):
    path = data['radius_path_nodes']
    pre = np.zeros(5)
    pre[path] = np.asarray(before)+np.log(q[path])
    post = pre.copy()
    post[path] = np.asarray(after)+np.log(q[path])
    logv = data['activity'][data['base_node']]+math.log(data['volume'])
    bound = math.ceil(6*data['volume']/math.pi)
    lower,upper = max(0.,logv-math.log(bound)),float(np.logaddexp(0.,logv))
    before,after = np.asarray(before,dtype=float),np.asarray(after,dtype=float)
    violation = max(0.,float(-np.diff(after).min()),lower-after[-1]+after[0],after[-1]-after[0]-upper)
    feasible = bool(np.all(np.diff(before) >= 0) and lower <= before[-1]-before[0] <= upper)
    return {'stage':0,'gain':.1,'end_clock':100,
        'weight_update_visits':[2,30,30,19,19],'log_weights':post.tolist(),
        'normalizer_projection':{'method':'unweighted_euclidean_nested_normalizer_slsqp_v1',
            'path_nodes':path.tolist(),'pre_projection_log_weights':pre.tolist(),
            'quota_adjusted_pre':before.tolist(),'quota_adjusted_post':after.tolist(),
            'lower_span':lower,'upper_span':upper,'geometric_particle_bound':bound,
            'geometric_bound_rounding':'ceiling(6V/(pi sigma^3)); conservative floating-bound guard',
            'solver_success':True,'solver_status':0,'solver_message':'Independent algebraic fixture',
            'solver_iterations':0,'objective_squared_distance':float(np.sum((after-before)**2)),
            'maximum_constraint_violation':float(violation),
            'constraint_tolerance':1e-7,'solver_ftol':1e-13,'solver_maxiter':1000,
            'solver_variable_scale':1. if feasible else max(1.,float(np.max(np.abs(before-before.mean()))))}}


@pytest.mark.parametrize('case',['isotonic','joint_upper','lower','feasible'])
def test_closed_form_unique_euclidean_projections_pass_independent_kkt(case):
    data,meta,q = protocol(16. if case == 'lower' else 0.)
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    if case == 'isotonic':before,after = [3.,0.,2.],[1.5,1.5,2.]
    elif case == 'joint_upper':
        upper = math.log(9.)
        before,after = [0.,0.,6.],[(6-upper)/3,(6-upper)/3,(6+2*upper)/3]
    elif case == 'lower':
        lower = 16+math.log(.5)
        before,after = [0.,0.,0.],[-lower/2,0.,lower/2]
    else:before = after = [0.,.5,1.]
    row = manual_row(data,q,before,after)
    pre,summary = audit.verify_normalizer_projection_row(data,q,row,config)
    assert summary['maximum_stationarity_residual'] < 1e-10
    np.testing.assert_allclose(pre[data['radius_path_nodes']]-np.log(q[data['radius_path_nodes']]),before)


def test_feasible_centroid_preserving_nonprojection_fails_kkt_despite_solver_success():
    data,meta,q = protocol()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    row = manual_row(data,q,[3.,0.,2.],[1.4,1.6,2.])
    with pytest.raises(ValueError,match='KKT optimality'):
        audit.verify_normalizer_projection_row(data,q,row,config)


def test_endpoint_clipping_after_isotonic_fit_is_not_joint_projection():
    data,meta,q = protocol()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    upper = math.log(9.)
    row = manual_row(data,q,[0.,0.,6.],[2-upper/2,2.,2+upper/2])
    with pytest.raises(ValueError,match='KKT optimality'):
        audit.verify_normalizer_projection_row(data,q,row,config)


@pytest.mark.parametrize('mutation',[
    lambda d,m:d['activity'].__setitem__(3,.1),
    lambda d,m:d['diameter'].__setitem__(3,.9),
    lambda d,m:d['pair_penalty'].__setitem__(3,25.),
    lambda d,m:d['umbrella_stiffness'].__setitem__(3,.1),
    lambda d,m:d['number_stiffness'].__setitem__(3,.1),
    lambda d,m:d['radius'].__setitem__(3,1.),
    lambda d,m:d.__setitem__('physical_nodes',np.array([1,2])),
    lambda d,m:d.__setitem__('cap',15),
    lambda d,m:d.__setitem__('volume',.9),
    lambda d,m:d.__setitem__('radius_path_nodes',np.array([1,3,3,2])),
    lambda d,m:m.__setitem__('calibration_method','one_over_t'),
    lambda d,m:m.__setitem__('calibration_cutoff_clock',0),
    lambda d,m:m['source_snapshots'].pop('normalizer_projection'),
    lambda d,m:m.__setitem__('calibration_normalizer_projection_timing','during production'),
    lambda d,m:m['calibration_outcome'].pop('normalizer_projection_mode'),
    lambda d,m:d.__setitem__('calibration_normalizer_projection',np.array(['nested_physical_radius'])),
])
def test_projection_requires_exact_physical_domain_and_discarded_execution_contract(mutation):
    data,meta,q = protocol()
    mutation(data,meta)
    with pytest.raises(ValueError):audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')


@pytest.mark.parametrize('mutation',[
    lambda r:r['normalizer_projection'].__setitem__('lower_span',.1),
    lambda r:r['normalizer_projection'].__setitem__('geometric_particle_bound',15),
    lambda r:r['normalizer_projection'].__setitem__('geometric_bound_rounding','floor'),
    lambda r:r['normalizer_projection'].__setitem__('path_nodes',[1,2,3]),
    lambda r:r['normalizer_projection']['quota_adjusted_pre'].__setitem__(0,2.9),
    lambda r:r['log_weights'].__setitem__(4,.1),
    lambda r:r['normalizer_projection'].__setitem__('objective_squared_distance',0.),
    lambda r:r['normalizer_projection'].__setitem__('maximum_constraint_violation',.1),
    lambda r:r['normalizer_projection'].__setitem__('solver_success',False),
    lambda r:r['normalizer_projection'].__setitem__('constraint_tolerance',1.),
    lambda r:r['normalizer_projection'].__setitem__('solver_variable_scale',100.),
    lambda r:r['normalizer_projection'].__setitem__('solver_ftol',1e-5),
    lambda r:r['normalizer_projection'].__setitem__('solver_maxiter',10),
])
def test_projection_record_cannot_relabel_bounds_objective_or_unprojected_targets(mutation):
    data,meta,q = protocol()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    row = manual_row(data,q,[3.,0.,2.],[1.5,1.5,2.])
    mutation(row)
    with pytest.raises(ValueError):audit.verify_normalizer_projection_row(data,q,row,config)


def test_post_projection_has_no_undeclared_gauge_and_preserves_path_mean():
    data,meta,q = protocol()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    row = manual_row(data,q,[3.,0.,2.],[1.5,1.5,2.])
    row['log_weights'] = (np.asarray(row['log_weights'])+7.).tolist()
    with pytest.raises(ValueError,match='unprojected relative weight'):
        audit.verify_normalizer_projection_row(data,q,row,config)
    wrong = manual_row(data,q,[3.,0.,2.],[2.,2.,2.5])
    with pytest.raises(ValueError,match='centroid'):
        audit.verify_normalizer_projection_row(data,q,wrong,config)


def test_count_replay_uses_prior_post_projection_to_next_pre_projection():
    data,meta,q = protocol()
    first = manual_row(data,q,[3.,0.,2.],[1.5,1.5,2.])
    second = manual_row(data,q,[1.5,1.5,2.],[1.5,1.5,2.])
    second.update(end_clock=200,weight_update_visits=[4,60,60,38,38])
    meta['calibration_outcome']['stage_history'] = [first,second]
    meta['calibration_cutoff_clock'] = 200
    data['initial_log_weights'] = np.asarray(first['normalizer_projection']['pre_projection_log_weights'])
    data['log_weights'] = np.asarray(second['log_weights'])
    summary = audit.replay_quota_stage_weights(data,meta,q,'endpoint_priority')
    assert summary['checkpoint_intervals_checked'] == 2
    assert len(summary['normalizer_projection_checks']) == 2
    second['normalizer_projection']['pre_projection_log_weights'][4] += .01
    with pytest.raises(ValueError):audit.replay_quota_stage_weights(data,meta,q,'endpoint_priority')


def test_projection_none_and_fully_absent_protocol_preserve_old_replay_semantics():
    data,meta,q = protocol()
    assert audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')['mode'] != 'none'
    meta['calibration_normalizer_projection'] = meta['settings']['calibration_normalizer_projection'] = 'none'
    data['calibration_normalizer_projection'] = np.array('none')
    meta['calibration_outcome']['normalizer_projection_mode'] = 'none'
    with pytest.raises(ValueError,match='Projection-none'):
        audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    meta['calibration_outcome']['stage_history'] = []
    assert audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority') == {'mode':'none'}
    assert audit.validate_normalizer_projection_protocol({'activity':np.zeros(5)},{},q,'legacy_uniform') == {'mode':'none'}


@pytest.mark.parametrize('count',[257,258])
def test_independent_projection_workspace_has_a_declared_bounded_path(count):
    data,meta,_ = protocol()
    nodes = count+1
    for key in ('activity','radius','umbrella_stiffness','number_stiffness'):
        data[key] = np.zeros(nodes)
    data['diameter'] = np.ones(nodes)
    data['pair_penalty'] = np.full(nodes,np.inf)
    data['pair_penalty'][0] = 0.
    data['radius'][1:] = np.linspace(0.,1.,count)
    data['physical_nodes'] = data['radius_path_nodes'] = np.arange(1,nodes)
    data['base_node'],data['cavity_node'] = 1,nodes-1
    q = np.full(nodes,1/nodes)
    if count == 257:
        result = audit.validate_normalizer_projection_protocol(data,meta,q,'uniform')
        assert result['matrix'].shape == (258,257)
    else:
        with pytest.raises(ValueError,match='bounded'):
            audit.validate_normalizer_projection_protocol(data,meta,q,'uniform')


@pytest.mark.parametrize('field',['raw','settings','top','outcome'])
def test_partial_none_declaration_or_conflicting_outcome_is_not_legacy(field):
    data,meta,q = protocol()
    meta['calibration_normalizer_projection'] = meta['settings']['calibration_normalizer_projection'] = 'none'
    data['calibration_normalizer_projection'] = np.array('none')
    meta['calibration_outcome']['normalizer_projection_mode'] = 'none'
    meta['calibration_outcome']['stage_history'] = []
    meta['calibration_cutoff_clock'] = 0
    if field == 'raw':data.pop('calibration_normalizer_projection')
    elif field == 'settings':meta['settings'].pop('calibration_normalizer_projection')
    elif field == 'top':meta.pop('calibration_normalizer_projection')
    else:meta['calibration_outcome']['normalizer_projection_mode'] = 'nested_physical_radius'
    with pytest.raises(ValueError):audit.validate_normalizer_projection_protocol(data,meta,q,'uniform')


@pytest.mark.parametrize('before',[[3.,0.,2.],[0.,0.,6.],[-10.,1.,9.],[0.,.5,1.]])
def test_actual_slsqp_output_passes_independent_algebraic_constraints_and_nnls_kkt(before):
    from scripts import continuum_normalizer_projection as producer
    data,meta,q = protocol()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    pre = np.zeros(5)
    pre[data['radius_path_nodes']] = np.asarray(before)+np.log(q[data['radius_path_nodes']])
    post,record = producer.project_physical_normalizer_weights(pre,q,data,data['cap'])
    row = {'log_weights':post.tolist(),'normalizer_projection':record}
    _,check = audit.verify_normalizer_projection_row(data,q,row,config)
    assert check['maximum_stationarity_residual'] <= check['stationarity_tolerance']


def test_large_gap_seventeen_node_projection_has_independent_kkt_certificate():
    from scripts import continuum_normalizer_projection as producer
    data,meta,_ = protocol(16.)
    nodes = 20
    for key in ('activity','radius','umbrella_stiffness','number_stiffness'):
        data[key] = np.zeros(nodes)
    path = np.arange(1,18)
    data['activity'][path] = 16.
    data['volume'],data['cap'] = 30.8496,80
    data['diameter'] = np.ones(nodes)
    data['pair_penalty'] = np.full(nodes,2.)
    data['pair_penalty'][path] = np.inf
    data['pair_penalty'][0] = 0.
    data['radius'][path] = 1-(1-np.arange(17)/16)**2
    data['physical_nodes'] = data['radius_path_nodes'] = path
    data['base_node'],data['cavity_node'] = 1,17
    q = np.full(nodes,.38/(nodes-3))
    q[[0,1,17]] = [.02,.3,.3]
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    before = np.zeros(nodes)
    before[path] = np.linspace(-120.,0.,17)+np.log(q[path])
    after,record = producer.project_physical_normalizer_weights(before,q,data,data['cap'])
    _,check = audit.verify_normalizer_projection_row(data,q,{'log_weights':after.tolist(),
        'normalizer_projection':record},config)
    assert check['maximum_stationarity_residual'] <= check['stationarity_tolerance']
