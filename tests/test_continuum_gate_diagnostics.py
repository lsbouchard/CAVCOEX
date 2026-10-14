"""Pure transport/support controls; no Monte Carlo, JIT, or phase-weight fit."""
import numpy as np
import pytest

from scripts import continuum_gate_diagnostics as gate


def target():
    return {"diameter":np.ones(2),"radius":np.zeros(2),
        "pair_penalty":np.array([np.inf,2.]),"activity":np.array([3.,1.]),
        "umbrella_stiffness":np.array([0.,4.]),"umbrella_preferred":np.array([0.,.8]),
        "number_stiffness":np.array([0.,2.]),"number_preferred":np.array([0.,3.]),
        "exchange_edges":np.array([[0,1]])}


def test_structural_strata_never_name_auxiliary_phases():
    assert gate.structural_state(32,.8) == "high_Bragg"
    assert gate.structural_state(29,.2) == "low_Bragg"
    assert gate.structural_state(2,.9) == "intermediate_or_dilute"


def test_transport_reservoir_before_return_and_phase_change():
    ids = np.array([[0,1,2],[1,0,2],[1,2,0],[0,2,1]])
    n = np.full_like(ids,29)
    b = np.full(ids.shape,.2)
    b[0,0] = .8
    result = gate.passage_diagnostics(ids,n,b,np.array([1,0,0],bool),
        np.array([0,0,1],bool),{"reservoir":np.array([2])},10)
    row = result["physical_departure_excursions"]["high_Bragg:N29"]
    assert row["departures"] == 1
    assert row["reservoir_before_return"] == 1
    assert row["successful_transit_median_sweeps"] == 20
    assert result["observed_changed_physical_Bragg_stratum_after_source_visit"][
        "reservoir:high_Bragg->low_Bragg"] == 1


def test_transport_return_first_and_censored():
    ids = np.array([[0,1,2],[1,0,2],[0,1,2],[1,0,2]])
    n = np.full_like(ids,29)
    b = np.full(ids.shape,.2)
    result = gate.passage_diagnostics(ids,n,b,np.array([1,0,0],bool),
        np.array([0,0,1],bool),{},10)
    row = result["physical_departure_excursions"]["low_Bragg:N29"]
    assert row["reservoir_before_return"] == 0
    assert row["physical_return_first"] == 2
    assert row["censored"] == 1
    assert row["successful_transit_median_sweeps"] is None


@pytest.mark.parametrize("bad_ids",[np.empty((0,2),int),np.array([[0,0]])])
def test_invalid_walker_rows_rejected(bad_ids):
    with pytest.raises(ValueError,match="permutations"):
        gate.passage_diagnostics(bad_ids,bad_ids,bad_ids,
            np.array([1,0],bool),np.array([0,1],bool),{},10)


def test_overlapping_masks_rejected():
    with pytest.raises(ValueError,match="disjoint"):
        gate.passage_diagnostics(np.array([[0,1]]),np.array([[29,29]]),
            np.array([[.2,.2]]),np.array([1,0],bool),np.array([1,0],bool),{},10)


def test_hard_support_rejection_avoids_infinity_times_zero():
    with np.errstate(invalid="raise",over="raise"):
        probabilities,valid = gate.hypothetical_swap_acceptance(target(),
            np.array([[3,4],[3,4]]),np.array([[.2,.8],[.2,.8]]),
            np.array([[0,1],[0,0]]),np.full((2,2),2.),0,1)
    assert probabilities[0] == 0
    assert not valid[0]
    assert np.isfinite(probabilities).all()


def test_full_cross_target_swap_cost_matches_direct_algebra():
    t = target()
    n,b,p,d = np.array([[3,4]]),np.array([[.2,.8]]),np.zeros((1,2)),np.full((1,2),2.)
    result,valid = gate.hypothetical_swap_acceptance(t,n,b,p,d,0,1)
    # activity part (3-1)*(4-3)=2; auxiliary B cost rises 1.44;
    # auxiliary number cost falls from 2 to 0: log ratio 2-1.44+2.
    assert valid[0]
    assert result[0] == 1.
    reverse,_ = gate.hypothetical_swap_acceptance(t,n[:,::-1],b[:,::-1],p,d,0,1)
    assert reverse[0] == pytest.approx(np.exp(-2.56))


def test_cavity_cross_support_uses_center_distance():
    t = target()
    t["pair_penalty"][:] = np.inf
    t["radius"][1] = 1.
    probabilities,valid = gate.hypothetical_swap_acceptance(t,np.array([[3,4]]),
        np.array([[.2,.8]]),np.zeros((1,2)),np.array([[.5,1.2]]),0,1)
    assert not valid[0] and probabilities[0] == 0


def test_current_state_support_violation_rejected():
    with pytest.raises(ValueError,match="own hard support"):
        gate.hypothetical_swap_acceptance(target(),np.array([[3,4]]),
            np.array([[.2,.8]]),np.array([[1,0]]),np.full((1,2),2.),0,1)


def test_stratified_pair_counts_are_complete():
    result = gate.swap_gate_diagnostics(target(),np.array([[3,4],[3,4]]),
        np.array([[.2,.8],[.2,.8]]),np.zeros((2,2)),np.full((2,2),2.))
    assert len(result) == 1
    assert sum(row["saved_pairs"] for row in result[0]["strata"].values()) == 2


@pytest.mark.parametrize("suffix",[".npz",".json"])
def test_output_cannot_overwrite_raw_or_sidecar(tmp_path,suffix):
    raw = tmp_path/'input.npz'
    with pytest.raises(ValueError,match="new JSON"):
        gate.write_diagnostics([raw],raw.with_suffix(suffix))


def test_non_json_output_rejected_before_input_loading(tmp_path):
    with pytest.raises(ValueError,match="new JSON"):
        gate.write_diagnostics([tmp_path/'missing.npz'],tmp_path/'result.txt')
