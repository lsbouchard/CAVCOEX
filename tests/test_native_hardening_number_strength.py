"""Direct construction guards and immutable default-array oracles."""
from pathlib import Path

import numpy as np
import pytest

from scripts import continuum_tempering_graph as graph


def build(**options):
    return graph.build_compact_graph(30.8496,32,80,base_field=16.,
                                    cavity_radius_stages=16,**options)


def test_default_four_preserves_immutable_v1_raw_array_bytes():
    path=graph.ROOT/'data_raw/continuum_native_compact_number_radius_graph_v1.npz'
    if not path.exists():
        pytest.skip('Immutable original graph is supplied in the full audit archive')
    target,weights=build()
    explicit,explicit_weights=build(ordered_hardening_number_strength=4.)
    with np.load(path,allow_pickle=False) as original:
        for key,value in target.items():
            value=np.asarray(value)
            np.testing.assert_array_equal(value,original[key])
            assert value.tobytes()==np.asarray(original[key]).tobytes()
            assert value.tobytes()==np.asarray(explicit[key]).tobytes()
        assert weights.tobytes()==original['initial_log_weights'].tobytes()
        assert weights.tobytes()==explicit_weights.tobytes()


def test_one_changes_only_declared_auxiliary_number_route_and_radius_indices():
    old,_=build()
    target,weights=build(ordered_hardening_number_strength=1.)
    assert len(weights)==162 and np.isfinite(weights).all()
    np.testing.assert_array_equal(target['number_stiffness'][21:39],4.)
    np.testing.assert_array_equal(target['number_stiffness'][39:70],1.)
    np.testing.assert_array_equal(target['number_stiffness'][139:145],[.5,.25,.125,.0625,.025,0.])
    np.testing.assert_array_equal(target['number_stiffness'][145:147],[2.,1.])
    for key in ('diameter','radius','activity','pair_penalty','umbrella_stiffness',
                'umbrella_preferred','number_preferred','displacement'):
        np.testing.assert_array_equal(target[key][:145],old[key][:145])
        np.testing.assert_array_equal(target[key][145:147],np.full(2,old[key][38]))
    # The low-order number bridge retains its original target, not H.
    np.testing.assert_array_equal(target['number_stiffness'][89:134],old['number_stiffness'][89:134])
    assert target['base_node']==70 and target['cavity_node']==71
    np.testing.assert_array_equal(target['radius_path_nodes'],np.r_[70,np.arange(147,162),71])
    physical=target['physical_nodes']
    assert np.all(np.isposinf(target['pair_penalty'][physical]))
    assert np.all(target['number_stiffness'][physical]==0)
    assert np.all(target['umbrella_stiffness'][physical]==0)
    edges={tuple(e) for e in target['exchange_edges']}
    assert (38,39) not in edges
    assert {(38,145),(145,146),(146,39)}<=edges


@pytest.mark.parametrize('strength',[.01,.3,2.5,3.999])
def test_general_strength_uses_geometric_midpoint_without_changing_number_target(strength):
    target,_=build(ordered_hardening_number_strength=strength)
    np.testing.assert_array_equal(target['number_stiffness'][145:147],
                                  [np.sqrt(4*strength),strength])
    np.testing.assert_array_equal(target['number_preferred'][145:147],32.)


@pytest.mark.parametrize('strength',[True,np.bool_(False),0.,-1.,4.001,np.nan,np.inf,
                                     -np.inf,'1',None,[1.],np.array([1.])])
def test_malformed_strength_is_rejected_before_graph_allocation(strength):
    with pytest.raises(ValueError,match='finite real scalar'):
        build(ordered_hardening_number_strength=strength)


def test_hard_activity_rejects_ignored_nondefault_auxiliary_option():
    with pytest.raises(ValueError,match='only to compact'):
        graph.build_native_graph(30.8496,32,80,graph_kind='hard_activity',
                                 ordered_hardening_number_strength=1.)
