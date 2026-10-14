"""Pure one-host target/graph/heatbath-law controls, not dense simulations."""
import math
import json

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as st


def valid_target():
    return {'diameter':np.ones(3),'radius':np.array([0.,0.,1.]),
        'activity':np.array([-2.,1.,1.]),'pair_penalty':np.array([0.,np.inf,np.inf]),
        'umbrella_stiffness':np.zeros(3),'umbrella_preferred':np.zeros(3),
        'number_stiffness':np.zeros(3),'number_preferred':np.zeros(3),
        'displacement':np.full(3,.1),'volume':8.,'base_node':1,'cavity_node':2,
        'ideal_node':0,'physical_nodes':np.array([1,2]),'exchange_edges':np.array([[0,1],[1,2]])}


@pytest.mark.parametrize('log_mean',[-20.,0.,4.,20.])
def test_exact_capped_poisson_normalization(log_mean):
    cap = 5
    cdf = st.capped_poisson_cdf(log_mean,cap)
    probabilities = np.diff(np.r_[0.,cdf])
    independently = np.array([math.exp(n*log_mean-math.lgamma(n+1)-max(0.,5*log_mean-math.lgamma(6)))
                              for n in range(cap+1)])
    independently /= independently.sum()
    assert probabilities == pytest.approx(independently,abs=5e-16)
    assert cdf[-1] == 1.
    assert np.all(np.diff(cdf)>=0)


@pytest.mark.parametrize('log_mean,cap',[(np.nan,2),(np.inf,2),(0.,-1),(0.,1.5)])
def test_bad_heatbath_parameters_rejected(log_mean,cap):
    with pytest.raises(ValueError):
        st.capped_poisson_cdf(log_mean,cap)


def test_irregular_degree_graph_contract():
    neighbors,edges,degrees = st.adjacency(np.array([[0,1],[1,2],[1,3]]),4)
    assert degrees.tolist() == [1,3,1,1]
    assert neighbors[1].tolist() == [0,2,3]
    assert edges[1].tolist() == [0,1,2]


@pytest.mark.parametrize('edges,nodes',[(np.array([[0,1],[1,0]]),2),
    (np.array([[0,0]]),1),(np.array([[0,1]]),3),(np.array([[0,2]]),2)])
def test_bad_graphs_rejected(edges,nodes):
    with pytest.raises(ValueError):
        st.adjacency(edges,nodes)


def test_valid_physical_targets_and_geometric_cap():
    st.validate_targets(valid_target(),16)
    with pytest.raises(ValueError,match='packing'):
        st.validate_targets(valid_target(),15)


@pytest.mark.parametrize('key,value',[
    ('pair_penalty',np.array([0.,-np.inf,np.inf])),
    ('number_stiffness',np.array([0.,.1,0.])),
    ('umbrella_stiffness',np.array([0.,0.,1.])),
    ('activity',np.array([-2.,np.nan,1.])),
    ('displacement',np.array([.1,-.1,.1])),
])
def test_invalid_costs_and_physical_biases_rejected(key,value):
    target = valid_target()
    target[key] = value
    with pytest.raises(ValueError):
        st.validate_targets(target,16)


def test_ideal_bias_and_tagged_activity_mismatch_rejected():
    target = valid_target()
    target['number_stiffness'][0] = .1
    with pytest.raises(ValueError,match='ideal'):
        st.validate_targets(target,16)
    target = valid_target()
    target['activity'][2] = 2.
    with pytest.raises(ValueError,match='matched activity'):
        st.validate_targets(target,16)


def frozen_archive(tmp_path,monkeypatch,**changes):
    monkeypatch.setattr(st,'ROOT',tmp_path)
    source = tmp_path/'source.py'
    source.write_text('verified test dependency\n')
    incoming = tmp_path/'input.bin'
    incoming.write_bytes(b'input')
    path = tmp_path/'weights.npz'
    arrays = {**valid_target(),'schema':st.SCHEMA,'sites':4,'cap':16,'log_weights':np.zeros(3)}
    arrays.update(changes)
    np.savez(path,**arrays)
    metadata = {'schema':st.SCHEMA,'weights_frozen_before_production':True,
        'raw_sha256':st.kernel.file_sha256(path),
        'source_snapshots':{name:['source.py',st.kernel.file_sha256(source)]
            for name in ('driver','kernel','graph_validation')},
        'input_sha256':{'input.bin':st.kernel.file_sha256(incoming)}}
    path.with_suffix('.json').write_text(json.dumps(metadata))
    return path


def test_frozen_weights_require_complete_matching_graph(tmp_path,monkeypatch):
    path = frozen_archive(tmp_path,monkeypatch)
    assert np.array_equal(st.load_frozen_weights(path,valid_target(),4,16),np.zeros(3))


@pytest.mark.parametrize('name,value',[
    ('volume',8.1),('sites',32),('cap',80),('base_node',0),('cavity_node',1),
    ('ideal_node',1),('physical_nodes',np.array([0,1,2])),
    ('displacement',np.full(3,.2)),('log_weights',np.array([0.,np.nan,0.])),
])
def test_frozen_weight_incompatibilities_rejected(tmp_path,monkeypatch,name,value):
    path = frozen_archive(tmp_path,monkeypatch,**{name:value})
    with pytest.raises(ValueError):
        st.load_frozen_weights(path,valid_target(),4,16)


@pytest.mark.parametrize('file',['source.py','input.bin','weights.npz'])
def test_frozen_weights_verify_source_inputs_and_raw(tmp_path,monkeypatch,file):
    path = frozen_archive(tmp_path,monkeypatch)
    with (tmp_path/file).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError):
        st.load_frozen_weights(path,valid_target(),4,16)
