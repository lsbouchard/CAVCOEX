"""Pure native compact construction and efficiency-model controls."""
import json
import math

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as st
from scripts import continuum_tempering_graph as graph


@pytest.mark.parametrize('stages,nodes', [(1,145),(16,160)])
def test_compact_routes_share_one_unbiased_physical_endpoint(stages,nodes):
    target,weights = graph.build_compact_graph(30.8496,32,80,base_field=16.,cavity_radius_stages=stages)
    neighbors,edge_table,degrees = st.validate_targets(target,80)
    assert len(weights)==nodes
    assert target['base_node']==70 and target['cavity_node']==71 and target['ideal_node']==0
    physical = target['physical_nodes']
    assert np.all(target['activity'][physical]==16.)
    assert np.isposinf(target['pair_penalty'][physical]).all()
    assert not target['number_stiffness'][physical].any()
    assert not target['umbrella_stiffness'][physical].any()
    assert target['activity'][0]==np.log(1/1.06448)
    assert target['pair_penalty'][0]==target['radius'][0]==0
    assert target['number_stiffness'][0]==target['umbrella_stiffness'][0]==0
    assert np.isfinite(weights).all() and weights[0]==0
    assert np.all(np.asarray(degrees)>0)
    assert np.all(target['diameter']==1.)


def test_ordered_number_entry_and_removal_are_finite_and_reversible():
    target,weights = graph.build_compact_graph(30.8496,32,80,base_field=16.)
    nu,nstar = target['number_stiffness'],target['number_preferred']
    np.testing.assert_array_equal(nu[134:139],[.25,.5,1.,2.,4.])
    np.testing.assert_array_equal(nu[139:145],[2.,1.,.5,.25,.1,0.])
    assert np.all(nstar[134:]==32.)
    assert np.all(nu[21:70]==4.) and np.all(nstar[21:70]==32.)
    assert np.all(target['pair_penalty'][134:139]==2)
    assert np.all(target['umbrella_stiffness'][134:139]==120)
    assert np.all(target['umbrella_preferred'][134:139]==.2)
    assert np.all(target['pair_penalty'][139:145]==25)
    assert not target['umbrella_stiffness'][139:145].any()
    edges = {tuple(sorted(map(int,e))) for e in target['exchange_edges']}
    assert (20,21) not in edges and (69,70) not in edges
    route = [20,134,135,136,137,138,21]
    tail = [69,139,140,141,142,143,144,70]
    assert all(tuple(sorted(pair)) in edges for pair in zip(route[:-1],route[1:]))
    assert all(tuple(sorted(pair)) in edges for pair in zip(tail[:-1],tail[1:]))
    # Even far from either preferred number the ACTUAL auxiliary cost is
    # finite, not a projection or occupation-sector constraint.
    assert np.isfinite(4.*(np.arange(81.)-32.)**2).all()


def test_low_order_number_target_uses_the_exact_explicit_volume():
    volume = 108*.96405+.00123456789
    target,_ = graph.build_compact_graph(volume,108,220,base_field=16.)
    assert target['volume']==volume
    np.testing.assert_array_equal(target['number_preferred'][89:134],np.full(45,volume/1.06448))
    assert np.all(target['number_preferred'][21:70]==108.)


def test_model_ideal_partition_is_exact_but_gaussian_order_is_only_a_guess():
    x,volume,cap = np.log(1/1.06448),30.8496,80
    exact = graph.log_capped_ideal_partition(x,volume,cap)
    assert graph.compact_weight_guess(x,volume,cap,0.,0.,.2,0.,0.)==-exact
    for kappa,preferred in ((0.,.88),(120.,.2),(120.,.88)):
        actual = graph.compact_weight_guess(x,volume,cap,0.,kappa,preferred,0.,0.)
        denominator = 1+2*kappa*.01
        expected = -exact+.5*np.log(denominator)+kappa*(.2-preferred)**2/denominator
        assert actual==pytest.approx(expected,abs=1e-12)


def test_efficiency_eta_cutoff_does_not_change_actual_particle_cap_or_target():
    target,_ = graph.build_compact_graph(30.8496,32,80,base_field=16.)
    # The initial model omits eta>=.74 terms, but the target still has a finite
    # soft interaction energy at EVERY allowed N and its cap stays80.
    node = 21
    assert np.isfinite(target['pair_penalty'][node])
    n = 80
    cost = (target['pair_penalty'][node]*n*(n-1)/2
            +target['umbrella_stiffness'][node]*(0.-target['umbrella_preferred'][node])**2
            +target['number_stiffness'][node]*(n-target['number_preferred'][node])**2)
    assert np.isfinite(cost)
    assert graph.COMPACT_WEIGHT_CONSTANTS['effective_packing_fraction_cutoff']==.74


@pytest.mark.parametrize('change', [{'sites':31},{'sites':33},{'sites':32,'cap':31},
                                  {'sites':100001},{'cap':100001},{'volume':100.},
                                  {'volume':np.inf},{'base_field':np.nan},
                                  {'cavity_radius_stages':0},{'displacement':0.}])
def test_malformed_unbounded_or_infeasible_auxiliary_settings_rejected_before_model_allocation(change):
    kwargs = dict(volume=30.8496,sites=32,cap=80,base_field=16.)
    kwargs.update(change)
    with pytest.raises(ValueError):
        graph.build_compact_graph(**kwargs)


def compact_fixture(tmp_path,monkeypatch,change=None):
    monkeypatch.setattr(graph,'ROOT',tmp_path)
    settings = dict(volume=30.8496,sites=32,cap=80,sigma=1.,base_field=16.,
                    displacement=.15,cavity_radius_stages=16,graph_kind='compact')
    target,weights = graph.build_native_graph(**settings)
    path = tmp_path/'compact.npz'
    source = tmp_path/'source.py'
    source.write_text('controlled source fixture\n')
    if change=='physical_bias':
        target['number_stiffness'][target['base_node']] = .01
    if change=='model_weights':
        weights[21] += .01
    np.savez(path,schema=graph.SCHEMA,sites=32,cap=80,initial_log_weights=weights,**target)
    metadata = dict(schema=graph.SCHEMA,construction_mode='explicit_box',settings=settings,
        source_snapshots={name:['source.py',graph.kernel.file_sha256(source)]
            for name in ('generator','kernel','graph_validation','target_validation')},
        input_sha256={},raw_sha256=graph.kernel.file_sha256(path),
        initial_weight_model=graph.COMPACT_WEIGHT_MODEL,
        initial_weight_model_constants=dict(graph.COMPACT_WEIGHT_CONSTANTS))
    if change=='model_constants':
        metadata['initial_weight_model_constants']['Bragg_variance'] = .02
    if change=='model_version':
        metadata['initial_weight_model'] = 'unrecorded_model'
    if change=='imported_mode':
        metadata['construction_mode'] = 'validated_parameter_archive'
    path.with_suffix('.json').write_text(json.dumps(metadata))
    return path


def test_explicit_compact_loader_binds_every_scalar_target_and_efficiency_weight(tmp_path,monkeypatch):
    path = compact_fixture(tmp_path,monkeypatch)
    target,data,metadata = graph.load_native_graph(path)
    assert target['volume']==30.8496 and len(target['activity'])==160
    assert metadata['input_sha256']=={}


@pytest.mark.parametrize('change',['physical_bias','model_weights','model_constants','model_version','imported_mode'])
def test_compact_loader_rejects_unbound_or_implicitly_imported_construction(tmp_path,monkeypatch,change):
    path = compact_fixture(tmp_path,monkeypatch,change)
    with pytest.raises(ValueError):
        graph.load_native_graph(path)


def test_default_dispatch_preserves_hard_activity_graph_bytes():
    original,oldweights = graph.build_activity_graph(30.8496,32,80,base_field=16.,cavity_radius_stages=16)
    target,weights = graph.build_native_graph(30.8496,32,80,base_field=16.,cavity_radius_stages=16)
    assert original.keys()==target.keys()
    assert all(np.asarray(original[name]).tobytes()==np.asarray(target[name]).tobytes() for name in original)
    assert oldweights.tobytes()==weights.tobytes()
