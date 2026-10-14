"""Quota bookkeeping and exact unchanged uniform/default trajectories."""
import hashlib
import importlib.util
import math
import sys

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as st

LEGACY_HASH = 'fc230a608039d6166042eb004d02c8fd4e8c32da8e85ff042e61c01a1bd9b789'


def legacy_module():
    path = st.ROOT/'data_raw/source_snapshots'/f'continuum_simulated_tempering_{LEGACY_HASH}.py'
    if not path.exists():
        pytest.skip('Immutable pre-quota source is distributed in the full audit archive')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LEGACY_HASH
    name = 'scripts._single_host_pre_quota_oracle'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def run_toy(module,quota=None,gain=.2,steps=2000,gibbs=True):
    length,cap,nodes = 3.,3,4
    positions = np.zeros((1,cap,3))
    counts = np.zeros(1,dtype=np.int64)
    centers = np.zeros((1,3))
    neighbors,edge_table,degrees = st.adjacency(np.array([[0,1],[0,2],[2,3],[3,0]]),nodes)
    activity = np.full(nodes,math.log(.06))
    weights = np.array([.3,-.4,.5,0.])
    updated = np.empty(steps,dtype=np.int64)
    intermediate = np.empty(steps,dtype=np.int64)
    gibbs_counts = np.zeros(2,dtype=np.int64)
    relocation_counts = np.zeros(2,dtype=np.int64)
    optional = {} if quota is None else {'calibration_label_quota':quota}
    st.kernel.seed_rng(2026093044)
    result = module.run_tempering_chunk(positions,counts,centers,3,0,0,steps,
        st.kernel.fcc_reference(32,length),length,np.ones(nodes),np.array([0.,1.,0.,0.]),
        activity,np.array([np.inf,np.inf,8.,0.]),np.array([0.,0.,2.5,0.]),
        np.array([0.,0.,.3,0.]),np.array([0.,0.,.8,0.]),np.array([0.,0.,1.8,0.]),
        np.full(nodes,.4),np.array([True,True,False,False]),0,1,3,
        neighbors,edge_table,degrees,weights,
        st.capped_poisson_cdf(activity[3]+math.log(length**3),cap),
        calibration_gain=gain,constant_calibration_gain=True,coordinate_stride=19,
        gibbs_label=gibbs,neighbor_source_labels=intermediate,weight_update_labels=updated,
        gibbs_counters=gibbs_counts,vacancy_relocation_probability=1.,
        vacancy_relocation_counts=relocation_counts,**optional)
    return result,positions,counts,centers,weights,updated,intermediate,gibbs_counts,relocation_counts


def assert_same(left,right):
    for a,b in zip(left[0][:6],right[0][:6]):
        np.testing.assert_array_equal(a,b)
    assert left[0][-2:] == right[0][-2:]
    for a,b in zip(left[1:],right[1:]):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('gain,gibbs',[(0.,False),(.2,False),(0.,True),(.2,True)])
def test_uniform_default_and_explicit_uniform_match_immutable_source(gain,gibbs):
    old = run_toy(legacy_module(),gain=gain,gibbs=gibbs)
    assert_same(old,run_toy(st,gain=gain,gibbs=gibbs))
    assert_same(old,run_toy(st,quota=np.full(4,.25),gain=gain,gibbs=gibbs))


def test_nonuniform_quota_update_uses_final_touched_labels_not_trace_labels():
    quota = st.resolve_label_quota(4,0,1,3,'endpoint_priority')
    result = run_toy(st,quota=quota,steps=17)
    initial = np.array([.3,-.4,.5,0.])
    counts = np.bincount(result[5],minlength=4)
    np.testing.assert_allclose(result[4],initial+.2*17*quota-.2*counts,rtol=0,atol=2e-15)
    assert not np.array_equal(result[5],result[0][0][:,1])


def test_frozen_nonuniform_quota_is_pathwise_inert():
    quota = st.resolve_label_quota(4,0,1,3,'endpoint_priority')
    assert_same(run_toy(st,gain=0.),run_toy(st,quota=quota,gain=0.))


def test_endpoint_priority_is_positive_normalized_and_has_no_phase_inputs():
    quota = st.resolve_label_quota(162,70,71,0,'endpoint_priority')
    assert quota[70] == quota[71] == .3 and quota[0] == .02
    others = np.delete(quota,[70,71,0])
    np.testing.assert_array_equal(others,np.full(159,.38/159))
    assert abs(quota.sum()-1) < 1e-15


@pytest.mark.parametrize('quota',[[0.,.5,.5],[-.1,.5,.6],[np.nan,.5,.5],
    [np.inf,.5,.5],[.2,.2,.2],[[.5,.5]],[]])
def test_invalid_quota_vectors_rejected(quota):
    with pytest.raises(ValueError,match='quota'):
        st.validate_label_quota(quota,3)


@pytest.mark.parametrize('args',[(3,0,1,2),(4,0,0,2),(4,0,1,4),(4,False,1,2)])
def test_endpoint_priority_requires_three_distinct_labels_and_auxiliaries(args):
    with pytest.raises(ValueError,match='priority'):
        st.resolve_label_quota(*args,mode='endpoint_priority')


def calibration_trace(labels):
    values = np.zeros((len(labels),12))
    values[:,0] = np.arange(1,len(labels)+1)
    values[:,1] = labels
    values[:,2:4] = [2.,.3]
    return values


def test_weighted_flatness_is_targeted_but_raw_minimum_counts_still_required():
    quota = np.array([.2,.3,.5])
    labels = np.repeat(np.arange(3),[2,3,5])
    trace = calibration_trace(labels)
    calibration = st.StagedFlatnessCalibration(3,np.array([1]),gain=.2,minimum_gain=.2,
        flatness=.9,minimum_visits=3,half_minimum_visits=1,label_quota=quota)
    first = calibration.observe(trace,labels,10,np.zeros(3))
    assert first['flatness_minimum_to_mean'] == 1.
    assert not first['minimum_visits_met'] and not first['flatness_passed']
    second = calibration.observe(trace,labels,20,np.zeros(3))
    assert second['stage_passed'] and calibration.completed
    np.testing.assert_allclose(second['quota_scaled_weight_update_visits'],[20,20,20])
    assert calibration.outcome(20,30)['label_quota'] == quota.tolist()


def test_flat_raw_counts_do_not_pass_nonuniform_targeted_flatness():
    quota = np.array([.1,.1,.8])
    labels = np.repeat(np.arange(3),4)
    trace = calibration_trace(labels)
    calibration = st.StagedFlatnessCalibration(3,np.array([1]),minimum_visits=1,
        half_minimum_visits=1,label_quota=quota)
    calibration.observe(trace,labels,12,np.zeros(3))
    row = calibration.observe(trace,labels,24,np.zeros(3))
    assert row['dense_half_stability_passed']
    assert not row['flatness_passed']


def test_quota_memory_history_is_explicit_and_default_budget_unchanged():
    args = (400000,3000000,10000,100,80,162,17,True)
    default = st.projected_memory_budget(*args)
    weighted = st.projected_memory_budget(*args,nonuniform_label_quota=True)
    assert weighted['calibration_history_envelope_bytes']-default['calibration_history_envelope_bytes'] == 300*16*162
    assert weighted['projected_peak_bytes'] < weighted['peak_limit_bytes']
