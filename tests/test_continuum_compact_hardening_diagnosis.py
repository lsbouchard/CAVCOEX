import json
import math

import numpy as np
import pytest

from scripts.continuum_compact_hardening_diagnosis import death_logratio,fit_registry,reverse_density


def reference(length=3.2):
    basis = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    return np.array([(np.array([i,j,k])+b)*length/2
        for i in range(2) for j in range(2) for k in range(2) for b in basis])


@pytest.mark.parametrize('n,number_term',[(31,-12.),(32,-4.),(33,4.)])
def test_number_bias_death_sign_and_metropolis_odds(n,number_term):
    actual,unbiased,term = death_logratio(n,.006,10.,7.,.3,4.,32.)
    assert term==number_term
    assert actual==unbiased+term
    assert math.exp(actual-unbiased)==pytest.approx(math.exp(number_term))


def test_independent_registry_reconstructs_a_translated_perfect_FCC():
    ref = reference()
    positions = (ref+np.array([.73,.11,.19]))%3.2
    centers,occupied,assigned,score = fit_registry(positions,ref,3.2,2)
    assert np.all(occupied==1)
    assert score<1e-25
    assert len(np.unique(assigned))==32


def test_reverse_refit_finds_a_removed_single_site_but_not_a_paired_site():
    ref = reference()
    width,uniform = .06,.2
    density = reverse_density(ref[0],ref[1:],ref,3.2,2,width,uniform)
    assert density==pytest.approx(uniform/3.2**3+(1-uniform)/(2*width)**3)
    paired = ref.copy()
    paired[-1] = ref[0]+.02
    residual = paired[1:]
    centers,occupancy,_,_ = fit_registry(residual,ref,3.2,2)
    assert np.count_nonzero(occupancy==0)==1
    # After removing one member, the other still occupies that site; the
    # deleted point is not covered by the remaining vacancy's proposal box.
    paired_density = reverse_density(paired[0],residual,ref,3.2,2,width,uniform)
    assert paired_density==pytest.approx(uniform/3.2**3)


def test_hard_target_profile_is_strict_JSON_not_float_infinity(monkeypatch):
    from scripts import continuum_compact_hardening_diagnosis as diagnosis
    rows = np.zeros((2,12))
    rows[:,2]=32
    data={'samples':rows,'pair_penalty':np.array([np.inf]),
          'umbrella_stiffness':np.zeros(1),'number_stiffness':np.zeros(1),
          'base_node':0}
    monkeypatch.setattr(diagnosis.gates,'conditional_label_diagnostics',
        lambda *args,**kwargs:{'mean_conditional_probability_by_node':[1.]})
    monkeypatch.setattr(diagnosis,'saved_deletion_diagnostics',lambda *args:{})
    metadata={'calibration_outcome':{'stage_history':[{}]}}
    result=diagnosis.diagnose(data,metadata,None)
    profile=result['node_profiles'][0]
    assert profile['gamma'] is None and profile['hard_pair_support'] is True
    json.dumps(result,allow_nan=False)
