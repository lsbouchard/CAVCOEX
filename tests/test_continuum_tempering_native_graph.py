"""Pure native-graph/EOS/provenance controls; no manufactured trajectory."""
import json
import math

import numpy as np
import pytest

from scripts import continuum_tempering_graph as graph
from scripts import continuum_simulated_tempering as sampler


def test_default_native_graph_is42_unbiased_nodes_with_deep_gas_and_tagged_cavity():
    target,weights = graph.build_activity_graph(32*.96405,32,80)
    assert len(weights)==42
    sampler.validate_targets(target,80)
    assert target['activity'][0] == target['activity'][1] == -2.
    assert target['pair_penalty'][0] == 0
    assert np.isposinf(target['pair_penalty'][1:]).all()
    assert np.all(target['umbrella_stiffness']==0)
    assert np.all(target['number_stiffness']==0)
    assert target['activity'][target['base_node']] == 14.8
    assert target['radius'][target['cavity_node']] == 1
    assert np.all(np.diff(target['activity'][1:-1])>0)
    assert weights[0] == 0
    assert np.isfinite(weights).all()


def test_physical_base_activity_retains_more_than12_fractional_digits():
    field = 14.812345678901234
    target,_ = graph.build_activity_graph(30.,32,80,base_field=field)
    assert target['activity'][target['base_node']] == field
    assert target['activity'][target['cavity_node']] == field
    sampler.validate_targets(target,80)


def test_initial_guess_has_correct_cs_activity_and_dilute_limit():
    for field in (-2.,8.,14.8,17.):
        rho,z = graph.fluid_cs_density(field)
        eta = rho*np.pi/6
        assert math.log(rho)+graph.kernel.excess_mu_cs(eta)==pytest.approx(field,abs=1e-12)
        assert z>1.
    rho,z = graph.fluid_cs_density(-10.)
    assert rho == pytest.approx(math.exp(-10.),rel=.001)
    assert z == pytest.approx(1.,rel=.001)


def test_capped_ideal_log_partition_is_analytically_normalized():
    field,volume,cap = -2.,30.,5
    expected = sum((math.exp(field)*volume)**n/math.factorial(n) for n in range(cap+1))
    assert graph.log_capped_ideal_partition(field,volume,cap)==pytest.approx(math.log(expected))


def test_cavity_initial_weight_difference_is_efficiency_guess_not_phase_odds():
    target,weights = graph.build_activity_graph(30.,32,80)
    rho,_ = graph.fluid_cs_density(14.8)
    assert weights[target['cavity_node']]-weights[target['base_node']] == pytest.approx(14.8-math.log(rho))


@pytest.mark.parametrize('changes',[{'sigma':.9},{'base_field':np.nan},{'base_field':18.},
    {'displacement':0.},{'volume':-1.},{'sites':0},{'cap':0}])
def test_bad_native_construction_rejected(changes):
    arguments = {'volume':30.,'sites':32,'cap':80}
    arguments.update(changes)
    with pytest.raises(ValueError):
        graph.build_activity_graph(**arguments)


def native_fixture(tmp_path,monkeypatch,fabricated=False,**changes):
    monkeypatch.setattr(graph,'ROOT',tmp_path)
    target,weights = graph.build_activity_graph(30.,32,80)
    path = tmp_path/'native.npz'
    arrays = {'schema':graph.SCHEMA,'sites':32,'cap':80,'initial_log_weights':weights,**target}
    arrays.update(changes)
    if fabricated:
        arrays['samples'] = np.zeros((1,42,10))
    np.savez(path,**arrays)
    source = tmp_path/'source.py'
    source.write_text('controlled dependency\n')
    parameter = tmp_path/'parameter.bin'
    parameter.write_bytes(b'geometry origin')
    metadata = {'schema':graph.SCHEMA,'raw_sha256':graph.kernel.file_sha256(path),
        'settings':{'sites':32,'cap':80},'source_snapshots':{name:['source.py',graph.kernel.file_sha256(source)]
            for name in ('generator','kernel','graph_validation')},
        'input_sha256':{'parameter.bin':graph.kernel.file_sha256(parameter)}}
    path.with_suffix('.json').write_text(json.dumps(metadata))
    return path,target,weights


def test_native_loader_verifies_provenance_and_casts_scalar_ids(tmp_path,monkeypatch):
    path,expected,weights = native_fixture(tmp_path,monkeypatch)
    target,data,metadata = graph.load_native_graph(path)
    assert isinstance(target['volume'],float)
    assert isinstance(target['base_node'],int)
    assert np.array_equal(data['initial_log_weights'],weights)
    assert 'samples' not in data
    assert np.array_equal(target['physical_nodes'],expected['physical_nodes'])


def test_native_loader_rejects_fabricated_trajectory(tmp_path,monkeypatch):
    path,_,_ = native_fixture(tmp_path,monkeypatch,fabricated=True)
    with pytest.raises(ValueError,match='fabricated'):
        graph.load_native_graph(path)


@pytest.mark.parametrize('name,value',[('sites',32.1),('sites',np.nan),('sites',np.array([32])),
    ('cap',80.2),('cap',np.inf),('base_node',31.2),('cavity_node',np.nan),('ideal_node',-.5)])
def test_native_loader_rejects_lossy_integer_scalars(tmp_path,monkeypatch,name,value):
    path,_,_ = native_fixture(tmp_path,monkeypatch,**{name:value})
    with pytest.raises(ValueError,match='integer scalar'):
        graph.load_native_graph(path)


@pytest.mark.parametrize('file',['source.py','parameter.bin','native.npz'])
def test_native_loader_checks_all_bytes(tmp_path,monkeypatch,file):
    path,_,_ = native_fixture(tmp_path,monkeypatch)
    with (tmp_path/file).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError):
        graph.load_native_graph(path)
