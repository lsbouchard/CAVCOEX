"""Pure optional-protocol/resource guards; target-law tests are independent."""
import json
import sys

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as sampler


def test_refined_budget_includes_runtime_history_and_array_copies():
    budget = sampler.projected_memory_budget(200000,2400000,5000,100,80,160,17,True)
    assert budget['raw_array_bytes'] < budget['raw_limit_bytes']
    assert budget['runtime_allowance_bytes'] > 1668661248
    assert budget['projected_peak_bytes'] > budget['runtime_allowance_bytes']+4*budget['raw_array_bytes']
    assert budget['projected_peak_bytes'] < budget['peak_limit_bytes']
    long = sampler.projected_memory_budget(200000,12000000,5000,100,80,160,17,True)
    assert long['raw_array_bytes'] == budget['raw_array_bytes']
    assert long['projected_peak_bytes'] > long['peak_limit_bytes']


def test_utf8_checkpoint_roundtrip_has_no_unicode_array_expansion(tmp_path):
    original = {'completed':False,'scope':'Finite discarded calibration; \u03bd is not a constraint',
                'stage_history':[{'stage':0,'log_weights':[0.,-300.]}]}
    encoded = sampler.checkpoint_outcome_bytes(original)
    assert encoded.dtype.kind == 'S'
    path = tmp_path/'checkpoint.npz'
    np.savez_compressed(path,calibration_outcome=encoded)
    with np.load(path,allow_pickle=False) as archive:
        value = archive['calibration_outcome']
        assert value.dtype.kind == 'S'
        assert json.loads(value.item().decode('utf-8')) == original
        assert value.nbytes == len(value.item())


@pytest.mark.parametrize('probability',['-0.1','1.1','nan','inf'])
def test_cli_rejects_invalid_relocation_probability_before_loading_graph(monkeypatch,probability):
    monkeypatch.setattr(sys,'argv',['continuum_simulated_tempering',
        '--graph-from','does-not-exist.npz','--output','unused',
        '--vacancy-relocation-probability',probability])
    with pytest.raises(SystemExit) as error:
        sampler.main()
    assert error.value.code == 2


@pytest.mark.parametrize('field,value',[(0,0),(1,-1),(2,0),(3,0),(4,0),(5,0),(6,161)])
def test_peak_workspace_rejects_invalid_dimensions(field,value):
    args=[200000,2400000,5000,100,80,160,17,True]
    args[field]=value
    with pytest.raises(ValueError,match='dimensions'):
        sampler.projected_memory_budget(*args)
