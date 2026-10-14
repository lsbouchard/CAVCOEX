"""Independent target reconstruction and signed-equality projection auditing."""
import copy
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from scripts import continuum_single_walker_analyze as audit


ROUNDING = 'decimal80_ln_nextplus_ceiling_arithmetic_exp_nextplus_float_ceiling_positive_subnormal_v1'


def certified_fixture():
    from test_continuum_normalizer_projection_validation import protocol
    data,meta,_ = protocol()
    data.update(activity=np.array([-2.,0.,0.,0.,-2.,0.,0.]),
        diameter=np.array([1.,1.,1.,1.,1.3,1.,1.]),
        radius=np.array([0.,0.,1.,.5,0.,0.,0.]),
        pair_penalty=np.array([0.,np.inf,np.inf,np.inf,0.,5.,np.inf]),
        umbrella_stiffness=np.zeros(7),umbrella_preferred=np.arange(7,dtype=float)/10,
        number_stiffness=np.zeros(7),number_preferred=np.arange(7,dtype=float),
        calibration_normalizer_projection=np.array('certified_normalizers'))
    q = np.full(7,.38/4)
    q[[0,1,2]] = [.02,.30,.30]
    meta['settings']['calibration_normalizer_projection'] = 'certified_normalizers'
    meta['calibration_normalizer_projection'] = 'certified_normalizers'
    meta['calibration_outcome']['normalizer_projection_mode'] = 'certified_normalizers'
    return data,meta,q


def algebraic_record(data,q,config,before,after):
    """Encode a hand-derived projection; do not invoke the producer optimizer."""
    selected,path = config['selected'],config['path']
    y,z = np.asarray(before,dtype=float),np.asarray(after,dtype=float)
    pre = np.zeros(len(q)); pre[selected] = y+np.log(q[selected])
    post = pre.copy(); post[selected] = z+np.log(q[selected])
    feasible = np.all(config['matrix']@y <= config['rhs'])
    violation = max(0.,float(np.max(config['matrix']@z-config['rhs'])))
    record = {
        'method':'unweighted_euclidean_certified_normalizer_slsqp_v1',
        'path_nodes':path.tolist(),'selected_nodes':selected.tolist(),
        'pre_projection_log_weights':pre.tolist(),
        'quota_adjusted_pre':(pre[path]-np.log(q[path])).tolist(),
        'quota_adjusted_post':(post[path]-np.log(q[path])).tolist(),
        'selected_quota_adjusted_pre':y.tolist(),'selected_quota_adjusted_post':z.tolist(),
        'duplicate_groups':copy.deepcopy(config['duplicate_groups']),
        'soft_hard_certificates':copy.deepcopy(config['soft_hard_certificates']),
        'epsilon_rounding':ROUNDING,'epsilon_decimal_precision':80,
        'scaled_bound_rounding':
            'binary64_division_nextafter_outward_nonzero_finite_bounds_exact_zero_infinity_preserved_v1',
        'lower_span':config['lower'],'upper_span':config['upper'],
        'geometric_particle_bound':config['particle_bound'],
        'geometric_bound_rounding':'ceiling(6V/(pi sigma^3)); conservative floating-bound guard',
        'solver_success':True,'solver_status':0,'solver_message':'Independent closed-form projection',
        'solver_iterations':0,'solver_variable_scale':1. if feasible else max(1.,float(np.max(np.abs(y-y.mean()))),config['lower']),
        'solver_ftol':1e-13,'solver_maxiter':1000,'constraint_tolerance':1e-7,
        'maximum_constraint_violation':violation,
        'objective_squared_distance':float(np.dot(z-y,z-y)),
    }
    return {'normalizer_projection':record,'log_weights':post.tolist(),
            'stage':0,'gain':.1,'end_clock':1000,'weight_update_visits':[20,300,300,95,95,95,95]}


def test_independent_target_canonicalization_ignores_only_zero_cost_preferences():
    data,meta,q = certified_fixture()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    assert config['duplicate_groups'] == [{'anchor_node':0,'nodes':[0,4]}, {'anchor_node':1,'nodes':[1,6]}]
    assert config['selected'].tolist() == list(range(7))
    certificates = config['soft_hard_certificates']
    assert len(certificates) == 1
    assert certificates[0]['soft_node'] == 5 and certificates[0]['hard_node'] == 1
    assert certificates[0]['epsilon_upper'] >= 8*math.exp(-5)
    assert config['matrix'].shape == (10,7)
    # Once a cost is active, its preference really changes the target.
    data['umbrella_stiffness'][4] = .1
    changed = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    assert changed['duplicate_groups'] == [{'anchor_node':1,'nodes':[1,6]}]


@pytest.mark.parametrize('sign',[-1.,1.])
def test_duplicate_equality_accepts_both_multiplier_signs_and_includes_ideal(sign):
    data,meta,q = certified_fixture()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    before = [2*sign,0.,1.,.5,-2*sign,0.,0.]
    after = [0.,0.,1.,.5,0.,0.,0.]
    row = algebraic_record(data,q,config,before,after)
    _,check = audit.verify_normalizer_projection_row(data,q,row,config)
    assert check['maximum_stationarity_residual'] < 1e-12
    # For sign=+1 the single member-minus-anchor equality multiplier is -2.
    # Nonnegative equality multipliers would incorrectly reject this optimum.
    assert config['matrix'].shape[0] > len(config['path'])+1


def test_feasible_but_nonoptimal_cross_component_shift_fails_kkt():
    data,meta,q = certified_fixture()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    before = [2.,0.,1.,.5,-2.,0.,0.]
    # Equality/interval constraints and GLOBAL centroid are all preserved, but
    # moving the independent ideal component cannot minimize squared distance.
    after = [.1,-.04,.96,.46,.1,-.04,-.04]
    row = algebraic_record(data,q,config,before,after)
    assert max(config['matrix']@np.array(after)-config['rhs']) < 1e-12
    assert abs(sum(before)-sum(after)) < 1e-12
    with pytest.raises(ValueError,match='KKT optimality'):
        audit.verify_normalizer_projection_row(data,q,row,config)


@pytest.mark.parametrize('x,volume,gamma',[(16.,30.8496,25.),(-10.,8.,0.),(-1000.,8.,0.),
                                          (-1e308,8.,1e308)])
def test_decimal_certificate_bounds_exact_float_inputs_without_zero_underflow(x,volume,gamma):
    epsilon,log_upper,decimal_upper = audit._certified_soft_hard_epsilon(x,volume,gamma)
    assert 0 < epsilon <= 1 and math.isfinite(log_upper)
    with localcontext() as context:
        context.prec = 120
        exact_log = Decimal.from_float(x)+Decimal.from_float(volume).ln()-Decimal.from_float(gamma)
        assert Decimal(decimal_upper) >= exact_log
        assert Decimal.from_float(log_upper) >= Decimal(decimal_upper)
        if exact_log > -10000:
            assert Decimal.from_float(epsilon) >= exact_log.exp()
    if x <= -1000:
        assert epsilon == math.nextafter(0.,math.inf)


def test_nontight_bound_is_not_selected_and_gamma_zero_can_be_certified():
    assert audit._certified_soft_hard_epsilon(5.,8.,0.) is None
    data,meta,q = certified_fixture()
    data['activity'][[0,1,2,3,4,5,6]] = -10.
    data['pair_penalty'][5] = 0.
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    certificates = config['soft_hard_certificates']
    assert len(certificates) == 1 and certificates[0]['pair_penalty'] == 0.
    assert certificates[0]['soft_node'] == 0 and certificates[0]['hard_node'] == 1


@pytest.mark.parametrize('mutation',[
    lambda d:d['number_stiffness'].__setitem__(5,.1),
    lambda d:d['umbrella_stiffness'].__setitem__(5,.1),
    lambda d:d['activity'].__setitem__(5,.01),
    lambda d:d['radius'].__setitem__(5,.01),
    lambda d:d['diameter'].__setitem__(5,.99),
])
def test_soft_hard_certificate_requires_exact_shared_unbiased_target(mutation):
    data,meta,q = certified_fixture()
    mutation(data)
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    assert config['soft_hard_certificates'] == []


@pytest.mark.parametrize('mutation',[
    lambda r:r.__setitem__('epsilon_rounding','ordinary exp'),
    lambda r:r.__setitem__('scaled_bound_rounding','nearest division'),
    lambda r:r.__setitem__('epsilon_decimal_precision',50),
    lambda r:r['duplicate_groups'][0].__setitem__('anchor_node',4),
    lambda r:r['duplicate_groups'][1]['nodes'].pop(),
    lambda r:r['selected_nodes'].reverse(),
    lambda r:r['soft_hard_certificates'][0].__setitem__('epsilon_upper',.9),
    lambda r:r['soft_hard_certificates'][0].__setitem__('hard_node',6),
    lambda r:r['soft_hard_certificates'][0].__setitem__('log_epsilon_upper_decimal','-5'),
    lambda r:r['quota_adjusted_pre'].__setitem__(0,.1),
    lambda r:r['selected_quota_adjusted_pre'].__setitem__(0,1.9),
])
def test_certified_projection_record_cannot_omit_or_relabel_exact_constraints(mutation):
    data,meta,q = certified_fixture()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    row = algebraic_record(data,q,config,[2.,0.,1.,.5,-2.,0.,0.],[0.,0.,1.,.5,0.,0.,0.])
    mutation(row['normalizer_projection'])
    with pytest.raises(ValueError):audit.verify_normalizer_projection_row(data,q,row,config)


def test_certified_count_replay_uses_previous_union_post_to_current_pre():
    data,meta,q = certified_fixture()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    before,after = [2.,0.,1.,.5,-2.,0.,0.],[0.,0.,1.,.5,0.,0.,0.]
    first = algebraic_record(data,q,config,before,after)
    second = algebraic_record(data,q,config,after,after)
    second.update(end_clock=2000,weight_update_visits=[40,600,600,190,190,190,190])
    meta['calibration_outcome']['stage_history'] = [first,second]
    meta['calibration_cutoff_clock'] = 2000
    data['initial_log_weights'] = np.asarray(first['normalizer_projection']['pre_projection_log_weights'])
    data['log_weights'] = np.asarray(second['log_weights'])
    summary = audit.replay_quota_stage_weights(data,meta,q,'endpoint_priority')
    assert summary['checkpoint_intervals_checked'] == 2
    assert len(summary['normalizer_projection_checks']) == 2
    assert summary['normalizer_projection_mode'] == 'certified_normalizers'


@pytest.mark.parametrize('name',['umbrella_preferred','number_preferred'])
def test_certified_identity_requires_complete_preference_arrays(name):
    data,meta,q = certified_fixture()
    data.pop(name)
    with pytest.raises(ValueError,match='complete real'):
        audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')


@pytest.mark.parametrize('before',[
    [2.,0.,1.,.5,-2.,0.,0.],
    [-2.,0.,1.,.5,2.,0.,0.],
    [8.,-10.,6.,-2.,-3.,9.,-2.],
    [0.,0.,0.,0.,0.,0.,0.],
])
def test_actual_joint_optimizer_passes_independent_signed_kkt(before):
    from scripts import continuum_normalizer_projection as producer
    data,meta,q = certified_fixture()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    pre = np.asarray(before)+np.log(q)
    post,record = producer.project_certified_normalizer_weights(pre,q,data,data['cap'])
    _,check = audit.verify_normalizer_projection_row(data,q,{'log_weights':post.tolist(),
        'normalizer_projection':record},config)
    assert check['maximum_stationarity_residual'] <= check['stationarity_tolerance']


def test_certified_solver_scale_also_covers_large_required_endpoint_span():
    from scripts import continuum_normalizer_projection as producer
    data,meta,q = certified_fixture()
    data['activity'][[1,2,3,5,6]] = 16.
    data['pair_penalty'][5] = 25.
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    pre = np.log(q)
    post,record = producer.project_certified_normalizer_weights(pre,q,data,data['cap'])
    assert record['solver_variable_scale'] == config['lower']
    _,check = audit.verify_normalizer_projection_row(data,q,{'log_weights':post.tolist(),
        'normalizer_projection':record},config)
    assert check['maximum_stationarity_residual'] <= check['stationarity_tolerance']


def test_constraint_reconstruction_does_not_import_producer_matching_or_bounds(monkeypatch):
    from scripts import continuum_normalizer_projection as producer
    def forbidden(*args,**kwargs):
        pytest.fail('Auditor reused producer certificate construction')
    for name in ('certified_normalizer_constraints','canonical_target_signature','upward_soft_hard_epsilon'):
        monkeypatch.setattr(producer,name,forbidden)
    data,meta,q = certified_fixture()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    assert len(config['soft_hard_certificates']) == 1


@pytest.mark.parametrize('count,match',[(258,'bounded selected'),(4097,'4096')])
def test_certified_scan_and_selected_workspace_guards_precede_matrix_allocation(count,match):
    data,meta,_ = certified_fixture()
    for name in ('activity','diameter','radius','pair_penalty','umbrella_stiffness',
                 'umbrella_preferred','number_stiffness','number_preferred'):
        previous = data[name]
        extended = np.full(count,previous[0])
        extended[:len(previous)] = previous
        data[name] = extended
    q = np.full(count,1/count)
    with pytest.raises(ValueError,match=match):
        audit.validate_normalizer_projection_protocol(data,meta,q,'uniform')


def test_actual_archived_severe_weights_have_independent_joint_kkt_certificate():
    from scripts import continuum_normalizer_projection as producer
    from test_certified_normalizer_projection import severe_actual_target
    # The immutable exploratory trajectory is an optimizer regression input,
    # not equilibrium evidence. np.load reads target/weight arrays only.
    data,q,weights,cap = severe_actual_target()
    data['cap'] = cap
    data['calibration_normalizer_projection'] = np.array('certified_normalizers')
    _,meta,_ = certified_fixture()
    config = audit.validate_normalizer_projection_protocol(data,meta,q,'endpoint_priority')
    post,record = producer.project_certified_normalizer_weights(weights,q,data,cap)
    _,check = audit.verify_normalizer_projection_row(data,q,{'log_weights':post.tolist(),
        'normalizer_projection':record},config)
    assert check['maximum_stationarity_residual'] <= check['stationarity_tolerance']
    assert any(group['nodes'] == [0,88] for group in config['duplicate_groups'])
    assert any(row['soft_node'] == 72 and row['hard_node'] == 70
               for row in config['soft_hard_certificates'])
