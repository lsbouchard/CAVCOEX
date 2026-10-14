"""Unbiased native radius-ring construction and efficiency-guess controls."""
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as st
from scripts import continuum_tempering_graph as graph


def test_disabled_radius_option_preserves_every_old_target_byte():
    digest = '25b9d1d314160fc3161ecbdcefe7f2069cc255fc64e46c544f92e20d09a1ca3a'
    path = st.ROOT / 'data_raw' / 'source_snapshots' / f'continuum_tempering_graph_{digest}.py'
    if not path.exists():
        pytest.skip('Full audit source oracle not included in the lean package')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    name = 'scripts._pre_radius_native_graph_oracle'
    spec = importlib.util.spec_from_file_location(name, path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy
    spec.loader.exec_module(legacy)
    for field in (14.8, 16., 14.812345678901234):
        old, old_weights = legacy.build_activity_graph(32 * .96405, 32, 80, base_field=field)
        new, new_weights = graph.build_activity_graph(32 * .96405, 32, 80, base_field=field,
                                                     cavity_radius_stages=1)
        assert old.keys() == new.keys()
        for key in old:
            assert np.asarray(old[key]).dtype == np.asarray(new[key]).dtype
            assert np.asarray(old[key]).tobytes() == np.asarray(new[key]).tobytes()
        assert old_weights.tobytes() == new_weights.tobytes()


@pytest.mark.parametrize('stages', [2, 4, 16])
def test_physical_radius_ring_retains_tagged_closure_and_unbiased_targets(stages):
    original, original_weights = graph.build_activity_graph(32 * .96405, 32, 80, base_field=16.)
    target, weights = graph.build_activity_graph(32 * .96405, 32, 80, base_field=16.,
                                               cavity_radius_stages=stages)
    neighbors, edge_table, degrees = st.validate_targets(target, 80)
    path = target['radius_path_nodes']
    assert len(weights) == len(original_weights) + stages - 1
    assert len(path) == stages + 1
    assert path[0] == target['base_node'] and path[-1] == target['cavity_node']
    assert np.all(np.diff(target['radius'][path]) > 0)
    assert target['radius'][path[0]] == 0 and target['radius'][path[-1]] == 1
    assert np.all(target['activity'][path] == 16.)
    assert np.isposinf(target['pair_penalty'][path]).all()
    assert not target['number_stiffness'][path].any()
    assert not target['umbrella_stiffness'][path].any()
    assert degrees[target['base_node']] == 4
    assert degrees[target['cavity_node']] == 2
    # One genuine graph cycle; neither radius growth nor marked closure erased.
    assert len(target['exchange_edges']) - len(weights) + 1 == 1
    canonical = {tuple(sorted(map(int, e))) for e in target['exchange_edges']}
    assert tuple(sorted((int(path[0]), int(path[-1])))) in canonical
    assert all(tuple(sorted(map(int, e))) in canonical for e in zip(path[:-1], path[1:]))
    for key in ('diameter', 'radius', 'activity', 'pair_penalty', 'umbrella_stiffness',
                'number_stiffness', 'displacement'):
        np.testing.assert_array_equal(target[key][:len(original_weights)], original[key])
    np.testing.assert_array_equal(weights[:len(original_weights)], original_weights)
    assert np.isfinite(weights).all()


@pytest.mark.parametrize('field', [-2., 8., 14.8, 16., 17.])
def test_cavity_weight_guesses_match_geometry_and_endpoint_but_are_only_guesses(field):
    radii = np.linspace(0., 1., 101)
    work = np.array([graph.cavity_weight_guess(r, field) for r in radii])
    rho, z = graph.fluid_cs_density(field)
    eta = rho * np.pi / 6
    assert np.isfinite(work).all() and (np.diff(work) >= 0).all()
    np.testing.assert_allclose(work[radii <= .5], -np.log1p(-8 * eta * radii[radii <= .5] ** 3))
    assert work[0] == 0.
    assert work[-1] == pytest.approx(field - np.log(rho))
    left_slope = (graph.cavity_weight_guess(.5, field) - graph.cavity_weight_guess(.5 - 1e-7, field)) / 1e-7
    right_slope = (graph.cavity_weight_guess(.5 + 1e-7, field) - graph.cavity_weight_guess(.5, field)) / 1e-7
    assert left_slope == pytest.approx(right_slope, rel=1e-5)


@pytest.mark.parametrize('stages', [0, 1.5, 257, np.inf])
def test_malformed_or_unbounded_stage_counts_are_rejected(stages):
    with pytest.raises(ValueError):
        graph.build_activity_graph(30., 32, 80, cavity_radius_stages=stages)


@pytest.mark.parametrize('change', ['reverse', 'fractional', 'activity', 'missing_edge'])
def test_radius_path_metadata_cannot_silently_describe_another_target(change):
    target, _ = graph.build_activity_graph(30., 32, 80, cavity_radius_stages=4)
    path = target['radius_path_nodes']
    if change == 'reverse':
        target['radius_path_nodes'] = path[::-1]
    elif change == 'fractional':
        target['radius_path_nodes'] = path.astype(float)
    elif change == 'activity':
        target['activity'][path[1]] += .01
    elif change == 'missing_edge':
        target['exchange_edges'] = target['exchange_edges'][:-1]
    with pytest.raises(ValueError, match='radius path'):
        st.validate_targets(target, 80)


def explicit_fixture(tmp_path,monkeypatch,change=None):
    monkeypatch.setattr(graph,'ROOT',tmp_path)
    settings = dict(volume=30.8496,sites=32,cap=80,sigma=1.,base_field=16.,
                    displacement=.15,cavity_radius_stages=16)
    target,weights = graph.build_activity_graph(settings['volume'],settings['sites'],settings['cap'],
        settings['sigma'],settings['base_field'],settings['displacement'],settings['cavity_radius_stages'])
    if change=='target':
        target['radius'][target['radius_path_nodes'][1]] += .001
    elif change=='weights':
        weights[10] += .01
    path = tmp_path/'explicit.npz'
    np.savez(path,schema=graph.SCHEMA,sites=32,cap=80,initial_log_weights=weights,**target)
    source = tmp_path/'source.py'
    source.write_text('controlled executed-source fixture\n')
    snapshots = {name:['source.py',graph.kernel.file_sha256(source)]
                 for name in ('generator','kernel','graph_validation','target_validation')}
    metadata = dict(schema=graph.SCHEMA,construction_mode='explicit_box',settings=settings,
                    source_snapshots=snapshots,input_sha256={},raw_sha256=graph.kernel.file_sha256(path))
    if change=='missing_validator':
        del snapshots['target_validation']
    elif change=='missing_scalar':
        del settings['sigma']
    elif change=='fractional_stages':
        settings['cavity_radius_stages'] = 16.1
    elif change=='hidden_input':
        metadata['input_sha256'] = {'source.py':graph.kernel.file_sha256(source)}
    elif change=='undeclared_mode':
        del metadata['construction_mode']
    path.with_suffix('.json').write_text(json.dumps(metadata))
    return path


def test_explicit_native_graph_needs_no_historical_archive_and_binds_every_array(tmp_path,monkeypatch):
    path = explicit_fixture(tmp_path,monkeypatch)
    target,data,metadata = graph.load_native_graph(path)
    assert metadata['input_sha256']=={}
    assert target['volume']==30.8496 and len(target['activity'])==56
    assert len(target['radius_path_nodes'])==17
    assert np.isfinite(data['initial_log_weights']).all()


@pytest.mark.parametrize('change',['target','weights','missing_validator','missing_scalar',
                                 'fractional_stages','hidden_input','undeclared_mode'])
def test_explicit_native_mode_rejects_unbound_settings_arrays_or_inputs(tmp_path,monkeypatch,change):
    path = explicit_fixture(tmp_path,monkeypatch,change)
    with pytest.raises(ValueError):
        graph.load_native_graph(path)


def test_explicit_cap_cannot_allocate_an_unbounded_partition_array():
    for cap in (100001,10**12):
        with pytest.raises(ValueError,match='100000'):
            graph.log_capped_ideal_partition(-2.,30.,cap)
        with pytest.raises(ValueError,match='100000'):
            graph.build_activity_graph(30.,32,cap)
    with pytest.raises(ValueError,match='100000'):
        graph.build_activity_graph(30.,10**12,80)
