"""Exact necessary-normalizer and historical-checkpoint controls, no MC."""
import math

import numpy as np

from scripts import continuum_endpoint_quota_diagnosis as diagnosis


def target(weights):
    return {'radius_path_nodes':np.array([0,1,2]),
        'calibration_label_quota':np.array([.3,.002,.3]),
        'log_weights':np.array(weights), 'base_node':0,'cavity_node':2,
        'volume':30.8496,'diameter':np.ones(3),'cap':80,'activity':np.full(3,16.)}


def test_exact_quota_adjusted_nested_constraint_and_endpoint_bounds():
    q=np.array([.3,.002,.3])
    # Quota allocation itself is not a monotonicity violation.
    a=np.array([-100.,-92.,-84.5])
    result=diagnosis.normalizer_constraints(target(a+np.log(q)))
    assert result['monotonicity_violations']==0
    assert result['endpoint_adjusted_span']==15.5
    assert result['endpoint_span_within_exact_bounds']
    assert result['geometric_particle_upper_bound']==math.floor(6*30.8496/math.pi)


def test_bad_radius_relative_weight_fails_necessary_normalizer_conditions():
    q=np.array([.3,.002,.3])
    a=np.array([-225.,-329.,-210.])
    result=diagnosis.normalizer_constraints(target(a+np.log(q)))
    assert result['monotonicity_violations']==1
    assert result['largest_negative_adjacent_difference']==-104.
    assert not result['endpoint_span_within_exact_bounds']
    # Passing conditional target kernels need not pass efficiency consistency.
    assert 'arbitrary frozen b still preserves' in result['scope']


def test_checkpoint_contrasts_preserve_discarded_last_iterate_oscillations():
    history=[]
    for t,gap in [(10000,4.),(20000,300.),(30000,123.)]:
        history.append({'end_clock':t,'gain':.15,'stage':1,
            'log_weights':[gap,0.,gap+15.], 'flatness_minimum_to_mean':.97,
            'dense_half_stability_passed':False,'dense_half_profiles':[]})
    result=diagnosis.checkpoint_contrasts({'calibration_outcome':{'stage_history':history}},[0,1,2],0,2)
    assert result['last_120k_cavity_radius_gap_range']==[19.,315.]
    assert result['last_stage_gain']==.15
    assert not result['last_stage_half_stability_passed']
    assert result['all_checkpoint_contrasts'][-1]['cavity_minus_last_radius']==138.
