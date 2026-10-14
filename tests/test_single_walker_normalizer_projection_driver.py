"""Projection mode binding and unchanged default calibration bookkeeping."""
import hashlib
import importlib.util
import sys

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as st

LEGACY_HASH='25d42f556d2bc62f8b44f66fb2b64015a9ae0c2c72dd2b5317542f95478d0cde'


def legacy_module():
    path=st.ROOT/'data_raw/source_snapshots'/f'continuum_simulated_tempering_{LEGACY_HASH}.py'
    if not path.exists(): pytest.skip('Immutable prior source is in the full audit archive')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==LEGACY_HASH
    name='scripts._single_host_pre_projection_oracle'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]


def trace():
    rows=np.zeros((12,12));rows[:,0]=np.arange(1,13)
    rows[:,1]=np.arange(12)%3;rows[:,2]=2.;rows[:,3]=.2
    return rows


@pytest.mark.parametrize('quota',[None,np.array([.3,.4,.3])])
def test_default_none_stage_output_matches_immutable_pre_projection_source(quota):
    classes=[legacy_module().StagedFlatnessCalibration,st.StagedFlatnessCalibration]
    results=[]
    for cls in classes:
        controller=cls(3,np.array([0,2]),gain=.2,minimum_gain=.1,minimum_visits=1,
            half_minimum_visits=1,label_quota=quota)
        for clock in (12,24,36,48):
            controller.observe(trace(),np.arange(12,dtype=np.int64)%3,clock,np.array([0.,1.,2.]))
        results.append(controller.outcome(48,48))
    assert results[0]==results[1]
    assert all('normalizer_projection' not in row for row in results[1]['stage_history'])


@pytest.mark.parametrize('mode',['nested_physical_radius','certified_normalizers'])
def test_projection_mode_requires_record_and_checkpoint_outcome(mode):
    controller=st.StagedFlatnessCalibration(3,np.array([0,2]),gain=.2,minimum_gain=.1,
        normalizer_projection_mode=mode)
    with pytest.raises(ValueError,match='Projection history'):
        controller.observe(trace(),np.arange(12,dtype=np.int64)%3,12,np.zeros(3))
    record={'method':'control','pre_projection_log_weights':[0.,0.,0.]}
    row=controller.observe(trace(),np.arange(12,dtype=np.int64)%3,12,np.ones(3),record)
    assert row['normalizer_projection'] is record
    assert row['log_weights']==[1.,1.,1.]
    assert controller.outcome(12,12)['normalizer_projection_mode']==mode


def test_none_mode_rejects_injected_projection_record():
    controller=st.StagedFlatnessCalibration(3,np.array([0,2]))
    with pytest.raises(ValueError,match='Projection history'):
        controller.observe(trace(),np.arange(12,dtype=np.int64)%3,12,np.zeros(3),{})


def test_memory_guard_accounts_for_pre_and_post_projection_history():
    args=(300000,2000000,5000,100,80,162,17,True)
    old=st.projected_memory_budget(*args,nonuniform_label_quota=True)
    new=st.projected_memory_budget(*args,nonuniform_label_quota=True,normalizer_projection_nodes=17)
    assert new['calibration_history_envelope_bytes']>old['calibration_history_envelope_bytes']
    assert new['projected_peak_bytes']>old['projected_peak_bytes']
    assert new['raw_array_bytes']==old['raw_array_bytes']
    assert new['projected_peak_bytes']<=new['peak_limit_bytes']


@pytest.mark.parametrize('flags',[[],['--calibration-method','staged_wang_landau','--calibration-steps','0']])
@pytest.mark.parametrize('mode',['nested_physical_radius','certified_normalizers'])
def test_projection_cli_requires_discarded_staged_calibration_before_graph_read(monkeypatch,tmp_path,flags,mode):
    monkeypatch.setattr(sys,'argv',['st','--graph-from',str(tmp_path/'missing.npz'),
        '--output',str(tmp_path/'unused'),'--calibration-normalizer-projection',mode,*flags])
    with pytest.raises(SystemExit) as stop:
        st.main()
    assert stop.value.code==2
    assert not (tmp_path/'unused.npz').exists()


def test_certified_memory_guard_adds_union_history_and_solver_workspace():
    args=(300000,2000000,5000,100,80,162,17,True)
    old=st.projected_memory_budget(*args,nonuniform_label_quota=True,normalizer_projection_nodes=17)
    new=st.projected_memory_budget(*args,nonuniform_label_quota=True,
        normalizer_projection_nodes=31,certified_projection_linear_rows=26)
    assert new['raw_array_bytes']==old['raw_array_bytes']
    assert new['projected_peak_bytes']>old['projected_peak_bytes']
    assert new['certified_projection_solver_workspace_bytes']>0
    assert new['projected_peak_bytes']<=new['peak_limit_bytes']
