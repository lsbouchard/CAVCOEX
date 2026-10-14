"""Deterministic simultaneous-projection controls; no sampling engine."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import continuum_normalizer_projection as projection


def target(nodes=3):
    return {'activity':np.full(nodes,1.),'diameter':np.ones(nodes),
        'radius':np.linspace(0.,1.,nodes),'pair_penalty':np.full(nodes,np.inf),
        'umbrella_stiffness':np.zeros(nodes),'number_stiffness':np.zeros(nodes),
        'radius_path_nodes':np.arange(nodes,dtype=np.int64),'physical_nodes':np.arange(nodes),
        'base_node':0,'cavity_node':nodes-1,'volume':1.}


@pytest.mark.parametrize('a',[[-10.,10.],[10.,-10.]])
def test_two_node_projection_has_exact_closed_form(a):
    t=target(2); q=np.array([.4,.6]); a=np.array(a)
    constraints=projection.nested_physical_constraints(t,q,2)
    span=np.clip(a[-1]-a[0],constraints['lower_span'],constraints['upper_span'])
    expected=np.array([a.mean()-span/2,a.mean()+span/2])
    output,record=projection.project_physical_normalizer_weights(a+np.log(q),q,t,2)
    np.testing.assert_allclose(output-np.log(q),expected,rtol=0,atol=1e-9)
    assert record['solver_success']
    assert record['objective_squared_distance']==pytest.approx(np.sum((expected-a)**2))


def test_three_node_isotonic_case_and_gauge_covariance():
    t=target(); q=np.array([.3,.4,.3]); a=np.array([-.1,-.3,1.])
    output,record=projection.project_physical_normalizer_weights(a+np.log(q),q,t,2)
    np.testing.assert_allclose(output-np.log(q),[-.2,-.2,1.],atol=1e-9,rtol=0)
    shifted,_=projection.project_physical_normalizer_weights(a+np.log(q)+1234.,q,t,2)
    np.testing.assert_allclose(shifted-1234.,output,atol=1e-8,rtol=0)
    assert record['objective_squared_distance']==pytest.approx(.02)


def test_feasible_input_is_exact_no_op():
    t=target(); q=np.array([.3,.4,.3]); before=np.array([0.,.5,1.])+np.log(q)
    output,record=projection.project_physical_normalizer_weights(before,q,t,2)
    np.testing.assert_array_equal(output,before)
    assert record['objective_squared_distance']==0
    assert record['solver_iterations']==0


def test_full_path_joint_projection_preserves_mean_not_endpoint_span_guess():
    t=target(17); t['volume']=30.8496; t['activity'][:]=16.
    q=np.full(17,.4/15); q[[0,-1]]=.3
    a=np.linspace(-345.,-329.,17); a[0],a[-1]=-225.,-210.
    before=a+np.log(q)
    output,record=projection.project_physical_normalizer_weights(before,q,t,80)
    actual=output-np.log(q)
    assert np.diff(actual).min()>=-1e-7
    assert record['lower_span']-1e-7<=actual[-1]-actual[0]<=record['upper_span']+1e-7
    assert actual.mean()==pytest.approx(a.mean(),abs=1e-8)
    assert record['geometric_particle_bound']==59
    assert record['lower_span']==pytest.approx(16+math.log(30.8496)-math.log(59))
    np.testing.assert_array_equal(record['pre_projection_log_weights'],before)


@pytest.mark.parametrize('mutation',[
    lambda t:t['activity'].__setitem__(1,2.),
    lambda t:t['diameter'].__setitem__(1,.9),
    lambda t:t['pair_penalty'].__setitem__(1,8.),
    lambda t:t['umbrella_stiffness'].__setitem__(1,1.),
    lambda t:t['number_stiffness'].__setitem__(1,1.),
    lambda t:t['radius'].__setitem__(1,1.),
    lambda t:t.update(physical_nodes=np.array([0,2])),
    lambda t:t.update(radius_path_nodes=np.array([0.,1.,2.])),
    lambda t:t.update(volume=.5),
])
def test_invalid_exact_bound_domain_rejected(mutation):
    t=target(); mutation(t)
    with pytest.raises(ValueError):
        projection.nested_physical_constraints(t,np.full(3,1/3),2)


@pytest.mark.parametrize('cap',[1,2.5,True])
def test_truncating_or_noninteger_cap_rejected(cap):
    with pytest.raises(ValueError):
        projection.nested_physical_constraints(target(),np.full(3,1/3),cap)


def test_nonpath_weights_unchanged_exactly():
    t=target(4); t['radius_path_nodes']=np.array([1,2,3]); t['base_node']=1
    t['radius'][:]=[0.,0.,.5,1.]
    q=np.array([.02,.30,.38,.30]); before=np.array([77.,2.,-3.,-1.])+np.log(q)
    output,_=projection.project_physical_normalizer_weights(before,q,t,2)
    assert output[0]==before[0]


def test_projection_path_allocation_bound():
    accepted=projection.nested_physical_constraints(target(257),np.full(257,1/257),2)
    assert accepted['matrix'].shape==(257,257)
    with pytest.raises(ValueError,match='2..257'):
        projection.nested_physical_constraints(target(258),np.full(258,1/258),2)


def test_near_clip_stress_and_legal_gauge_shift():
    t=target(17); t['volume']=30.8496; t['activity'][:]=16.
    q=np.full(17,.4/15);q[[0,-1]]=.3
    a=np.linspace(-99970.,99970.,17);a[0],a[-1]=30000.,-30000.
    before=a+np.log(q)
    assert np.max(np.abs(before+2000.))<1e5
    output,record=projection.project_physical_normalizer_weights(before,q,t,80)
    shifted,_=projection.project_physical_normalizer_weights(before+2000.,q,t,80)
    np.testing.assert_allclose(shifted-2000.,output,rtol=0,atol=2e-7)
    actual=output-np.log(q)
    assert np.diff(actual).min()>=-1e-7
    assert record['lower_span']-1e-7<=actual[-1]-actual[0]<=record['upper_span']+1e-7
    assert record['solver_variable_scale']>10000
    assert record['solver_ftol']==1e-13


def test_solver_nonsuccess_never_silently_becomes_projection(monkeypatch):
    from scipy import optimize
    monkeypatch.setattr(optimize,'minimize',lambda *a,**k:SimpleNamespace(
        status=8,success=False,message='controlled line-search failure',nit=1,x=np.zeros(3)))
    with pytest.raises(ValueError,match='controlled line-search failure'):
        projection.project_physical_normalizer_weights(np.array([3.,-3.,0.]),np.full(3,1/3),target(),2)
