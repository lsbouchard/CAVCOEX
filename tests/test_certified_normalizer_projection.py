"""Direct certified helper, exact default compatibility and bounded workspace."""
import ast
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import continuum_normalizer_projection as projection

ROOT = Path(__file__).resolve().parents[1]
OLD_HASH = 'c6d1e4898e9a8f0295670848d839864fd85e8237097cb3e2d6ccc10828d02b86'


def target():
    return {'activity':np.ones(6),'diameter':np.ones(6),
        'radius':np.array([0.,1.,0.,0.,0.,0.]),
        'pair_penalty':np.array([np.inf,np.inf,2.,2.,0.,0.]),
        'umbrella_stiffness':np.zeros(6),'umbrella_preferred':np.arange(6,dtype=float),
        'number_stiffness':np.zeros(6),'number_preferred':np.arange(6,dtype=float),
        'radius_path_nodes':np.array([0,1]),'physical_nodes':np.array([0,1]),
        'base_node':0,'cavity_node':1,'volume':1.}


def old_module():
    path = ROOT/'data_raw/source_snapshots'/f'continuum_normalizer_projection_{OLD_HASH}.py'
    if not path.exists():
        pytest.skip('Immutable old projection helper is in the full audit archive')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == OLD_HASH
    spec = importlib.util.spec_from_file_location('_old_nested_normalizer_oracle',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def severe_actual_target():
    path = ROOT/'data_raw/continuum_single_host_native_compact_h1_projected_quota_pilot.npz'
    if not path.exists():
        pytest.skip('Closed severe weight fixture is in the full audit archive')
    names = ['activity','diameter','radius','pair_penalty','umbrella_stiffness','umbrella_preferred',
             'number_stiffness','number_preferred','radius_path_nodes','physical_nodes']
    with np.load(path,allow_pickle=False) as archive:
        data = {name:archive[name] for name in names}
        data.update({name:archive[name].item() for name in ('volume','base_node','cavity_node')})
        return data,archive['calibration_label_quota'],archive['log_weights'],int(archive['cap'])


def assert_joint_feasibility(config, adjusted, tolerance=1e-7):
    assert np.max(config['inequality_lower']-config['inequality_matrix']@adjusted) <= tolerance
    assert np.max(config['inequality_matrix']@adjusted-config['inequality_upper']) <= tolerance
    assert np.max(np.abs(config['equality_matrix']@adjusted)) <= tolerance


def test_default_nested_functions_are_ast_identical_to_immutable_executed_source():
    old = ast.parse(old_module().EXECUTION_SOURCE_BYTES)
    current = ast.parse(projection.EXECUTION_SOURCE_BYTES)
    for name in ('nested_physical_constraints','project_physical_normalizer_weights'):
        first = next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name==name)
        second = next(node for node in current.body if isinstance(node,ast.FunctionDef) and node.name==name)
        assert ast.dump(first,include_attributes=False) == ast.dump(second,include_attributes=False)


def test_old_nested_actual_severe_fixture_stays_bitwise_identical():
    t,q,b,cap = severe_actual_target()
    old,old_record = old_module().project_physical_normalizer_weights(b,q,t,cap)
    current,record = projection.project_physical_normalizer_weights(b,q,t,cap)
    np.testing.assert_array_equal(current,old)
    assert record == old_record


def test_joint_closed_form_with_soft_span_duplicate_equality_and_radius_span():
    t = target(); t['activity'][4:] = -2.
    q = np.full(6,1/6)
    before = np.array([2.,3.,-2.,-2.,10.,0.])
    config = projection.certified_normalizer_constraints(t,q,2)
    epsilon = config['soft_hard_certificates'][0]['epsilon_upper']
    upper = config['upper_span']
    u = (2+3-upper-2+epsilon-2+epsilon)/4
    expected = np.array([u,u+upper,u-epsilon,u-epsilon,5.,5.])
    result,record = projection.project_certified_normalizer_weights(before+np.log(q),q,t,2)
    np.testing.assert_allclose(result-np.log(q),expected,rtol=0,atol=1e-8)
    assert_joint_feasibility(config,result-np.log(q))
    assert record['selected_nodes'] == list(range(6))
    assert record['duplicate_groups'] == [{'anchor_node':2,'nodes':[2,3]}, {'anchor_node':4,'nodes':[4,5]}]
    assert record['objective_squared_distance'] == pytest.approx(np.sum((expected-before)**2))
    # The former path vectors remain path-only for archive compatibility.
    assert len(record['quota_adjusted_post']) == 2
    assert len(record['selected_quota_adjusted_post']) == 6


def test_canonical_zero_cost_preferences_and_zero_pair_diameter_are_ignored():
    t = target(); t['diameter'][5] = 2.
    assert projection.canonical_target_signature(t,4) == projection.canonical_target_signature(t,5)
    assert projection.canonical_target_signature(t,2) == projection.canonical_target_signature(t,3)
    t['umbrella_stiffness'][3] = .1
    assert projection.canonical_target_signature(t,2) != projection.canonical_target_signature(t,3)
    t['umbrella_stiffness'][3] = 0; t['number_stiffness'][3] = .1
    assert projection.canonical_target_signature(t,2) != projection.canonical_target_signature(t,3)


@pytest.mark.parametrize('x,volume,gamma',[(16.,30.8496,25.),(-10.,8.,0.),(-1000.,8.,0.),(-1e308,8.,1e308)])
def test_upward_epsilon_certificate_and_positive_underflow(x,volume,gamma):
    bound = projection.upward_soft_hard_epsilon(x,volume,gamma)
    assert 0 < bound['epsilon_upper'] <= 1
    assert math.isfinite(bound['log_epsilon_upper'])
    with localcontext() as context:
        context.prec = 120
        exact_log = Decimal.from_float(x)+Decimal.from_float(volume).ln()-Decimal.from_float(gamma)
        assert Decimal(bound['log_epsilon_upper_decimal']) >= exact_log
        assert Decimal.from_float(bound['log_epsilon_upper']) >= Decimal(bound['log_epsilon_upper_decimal'])
        if exact_log > -10000:
            assert Decimal.from_float(bound['epsilon_upper']) >= exact_log.exp()
    if x <= -1000:
        assert bound['epsilon_upper'] == math.nextafter(0.,math.inf)


def test_only_tight_exactly_matched_unbiased_certificates_are_selected():
    t = target(); q = np.full(6,1/6)
    assert projection.upward_soft_hard_epsilon(5.,8.,0.) is None
    certificates = projection.certified_normalizer_constraints(t,q,2)['soft_hard_certificates']
    assert [(row['soft_node'],row['hard_node']) for row in certificates] == [(2,0)]
    t['number_stiffness'][2:4] = .1
    assert projection.certified_normalizer_constraints(t,q,2)['soft_hard_certificates'] == []


def test_gamma_zero_bound_and_literal_diameter_matching_rep():
    t = target(); t['activity'][:] = -2.
    t['diameter'][4] = 2.
    q = np.full(6,1/6)
    rows = projection.certified_normalizer_constraints(t,q,2)['soft_hard_certificates']
    assert [(row['soft_node'],row['hard_node']) for row in rows] == [(2,0),(5,0)]
    assert rows[1]['pair_penalty'] == 0


def test_actual_severe_full_target_is_projected_jointly_with_off_union_unchanged():
    t,q,b,cap = severe_actual_target()
    config = projection.certified_normalizer_constraints(t,q,cap)
    result,record = projection.project_certified_normalizer_weights(b,q,t,cap)
    selected = config['selected_nodes']
    after = result[selected]-np.log(q[selected])
    assert_joint_feasibility(config,after)
    assert [0,88] in [group['nodes'] for group in config['duplicate_groups']]
    assert [72,133,144] in [group['nodes'] for group in config['duplicate_groups']]
    assert (72,70) in [(row['soft_node'],row['hard_node']) for row in config['soft_hard_certificates']]
    assert record['solver_success']
    assert after.mean() == pytest.approx(np.mean(b[selected]-np.log(q[selected])),abs=1e-8)
    mask = np.ones(len(b),dtype=bool); mask[selected] = False
    np.testing.assert_array_equal(result[mask],b[mask])
    shifted,_ = projection.project_certified_normalizer_weights(b+2000,q,t,cap)
    np.testing.assert_allclose(shifted-2000,result,rtol=0,atol=2e-7)


def test_feasible_certified_input_is_exact_noop():
    t = target(); t['activity'][4:] = -2.
    q = np.full(6,1/6)
    before = np.array([0.,1.,0.,0.,3.,3.])+np.log(q)
    result,record = projection.project_certified_normalizer_weights(before,q,t,2)
    np.testing.assert_array_equal(result,before)
    assert record['solver_iterations'] == 0
    assert record['objective_squared_distance'] == 0


def test_certified_solver_nonsuccess_is_fatal(monkeypatch):
    from scipy import optimize
    monkeypatch.setattr(optimize,'minimize',lambda *a,**k:SimpleNamespace(
        status=8,success=False,message='controlled joint failure',nit=1,x=np.zeros(6)))
    with pytest.raises(ValueError,match='controlled joint failure'):
        projection.project_certified_normalizer_weights(np.arange(6,dtype=float),np.full(6,1/6),target(),2)


def test_solver_receives_positive_subnormal_upper_bound_for_gamma1100(monkeypatch):
    from scipy import optimize
    captured = {}
    actual = optimize.minimize
    def capture(*args,**kwargs):
        captured['constraint'] = kwargs['constraints'][0]
        return actual(*args,**kwargs)
    monkeypatch.setattr(optimize,'minimize',capture)
    t = target(); t['pair_penalty'][2:4] = 1100.; t['activity'][4:] = -2.
    q = np.full(6,1/6)
    before = np.array([99900.,-99900.,100.,-200.,20.,-50.])+np.log(q)
    output,record = projection.project_certified_normalizer_weights(before,q,t,2)
    config = projection.certified_normalizer_constraints(t,q,2)
    assert config['soft_hard_certificates'][0]['epsilon_upper'] == math.nextafter(0.,math.inf)
    soft_row = len(config['path_nodes'])
    assert captured['constraint'].ub[soft_row] == math.nextafter(0.,math.inf)
    assert captured['constraint'].lb[soft_row] == 0.
    assert record['scaled_bound_rounding'] == projection.SCALED_BOUND_ROUNDING
    assert record['solver_variable_scale'] > 1000
    assert_joint_feasibility(config,output[config['selected_nodes']]-np.log(q[config['selected_nodes']]))


def test_scaled_bound_rounding_is_outward_with_exact_zeros_and_infinities():
    tiny = math.nextafter(0.,math.inf)
    lower = np.array([0.,-0.,tiny,-tiny,-np.inf,1.,-1.])
    upper = np.array([0.,-0.,tiny,-tiny,np.inf,1.,-1.])
    lo,hi = projection.outward_scaled_inequality_bounds(lower,upper,1e5)
    assert np.array_equal(lo[:2],lower[:2]) and np.array_equal(hi[:2],upper[:2])
    assert lo[4] == -np.inf and hi[4] == np.inf
    assert hi[2] == tiny and lo[3] == -tiny
    for index in (2,3,5,6):
        with localcontext() as context:
            context.prec = 100
            assert Decimal.from_float(lo[index]) <= Decimal.from_float(lower[index])/Decimal.from_float(1e5)
            assert Decimal.from_float(hi[index]) >= Decimal.from_float(upper[index])/Decimal.from_float(1e5)


@pytest.mark.parametrize('key,value',[
    ('pair_penalty',-1.),('pair_penalty',-np.inf),('diameter',0.),('radius',-.1),
    ('umbrella_stiffness',-.1),('number_stiffness',-.1),('number_preferred',np.nan)])
def test_invalid_canonical_target_guard(key,value):
    t = target(); t[key][4] = value
    with pytest.raises(ValueError):
        projection.certified_normalizer_constraints(t,np.full(6,1/6),2)


def many_nodes(count):
    t = target()
    for key,value in list(t.items()):
        if isinstance(value,np.ndarray) and key not in ('radius_path_nodes','physical_nodes'):
            t[key] = np.r_[value,np.full(count-len(value),value[-1])]
    return t


def test_certified_union_and_target_scan_are_bounded_before_solver():
    t = many_nodes(258)
    with pytest.raises(ValueError,match='257 selected'):
        projection.certified_normalizer_constraints(t,np.full(258,1/258),2)
    t = many_nodes(4097)
    with pytest.raises(ValueError,match='4096'):
        projection.certified_normalizer_constraints(t,np.full(4097,1/4097),2)
