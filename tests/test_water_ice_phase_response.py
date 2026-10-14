"""Independent finite-law checks; no water simulation or fitted parameters."""
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from water_ice_phase_response import (
    phase_probability_tilt, thermal_response_per_molecule,
    equal_weight_temperature_shift, cluster_profile_reweight,
)


def test_exact_bayes_phase_tilt():
    q = np.array([0.1, 0.5, 0.9])
    p_ice, p_liquid = 0.04, 0.01
    y = np.log(q / (1-q))
    work_contrast = -2.5 * np.log(p_ice/p_liquid)
    ref, biased = phase_probability_tilt(y, work_contrast, 2.5)
    expected = q*p_ice/(q*p_ice+(1-q)*p_liquid)
    np.testing.assert_allclose(ref, q)
    np.testing.assert_allclose(biased, expected)


def test_ice_favored_means_expansion_and_enthalpy_reduction():
    dq, dv, dh = thermal_response_per_molecule(0.0, -1.2,
                                              volume_gap=0.08, latent_heat=2.0)
    assert dq > 0 and dv > 0 and dh < 0
    assert dv == pytest.approx(0.08*dq)
    assert dh == pytest.approx(-2*dq)


def test_zero_contrast_and_stable_extreme_odds():
    z = np.array([-1000, 0, 1000.0])
    dq, dv, dh = thermal_response_per_molecule(z, 0,
                                              volume_gap=1, latent_heat=1)
    np.testing.assert_array_equal(dq, 0)
    np.testing.assert_array_equal(dv, 0)
    np.testing.assert_array_equal(dh, 0)


def test_temperature_shift_matches_exact_constant_enthalpy_model():
    # In reduced kB=1 units, y=N*ell*(beta-beta_eq) and a(beta)=beta*dw.
    n, ell, tm, dw = 1000, 2.0, 3.0, -4.0
    beta_crossing = (1/tm)/(1-dw/(n*ell))
    exact_shift = 1/beta_crossing - tm
    predicted = equal_weight_temperature_shift(melting_temperature=tm,
        work_contrast=dw, molecule_count=n, latent_heat=ell)
    assert predicted == pytest.approx(exact_shift)
    assert predicted > 0
    assert equal_weight_temperature_shift(melting_temperature=tm,
        work_contrast=dw, molecule_count=2*n, latent_heat=ell) == pytest.approx(predicted/2)


def test_cluster_mass_and_profile_identity_against_direct_microstates():
    micro_mass = np.array([0.1, 0.3, 0.2, 0.4])
    cluster = np.array([0, 1, 1, 2])
    chi = np.array([1, 0, 1, 1])
    pi0 = np.bincount(cluster, weights=micro_mass, minlength=3)
    weighted = np.bincount(cluster, weights=micro_mass*chi, minlength=3)
    compatibility = weighted/pi0
    kt = 1.7
    pic, gc, wav = cluster_profile_reweight(pi0, compatibility, thermal_energy=kt)
    np.testing.assert_allclose(pic, weighted/weighted.sum())
    g0 = -kt*np.log(pi0)
    w = -kt*np.log(compatibility)
    np.testing.assert_allclose(gc-g0, w-wav)
    assert (gc[2]-gc[0])-(g0[2]-g0[0]) == pytest.approx(w[2]-w[0])


def test_zero_mass_and_forbidden_cluster_bins():
    pic, gc, _ = cluster_profile_reweight([0.2, 0.8, 0], [1, 0, 0], thermal_energy=1)
    np.testing.assert_array_equal(pic, [1, 0, 0])
    assert np.isinf(gc[1]) and np.isinf(gc[2])
    with pytest.raises(ValueError, match="normalization"):
        cluster_profile_reweight([0.2, 0.8], [0, 0], thermal_energy=1)


def test_common_compatibility_scale_does_not_change_profile():
    pi0, p = [0.1, 0.3, 0.6], np.array([0.9, 0.3, 0.5])
    a = cluster_profile_reweight(pi0, p, thermal_energy=1)
    b = cluster_profile_reweight(pi0, p/2, thermal_energy=1)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    assert b[2]-a[2] == pytest.approx(np.log(2))


@pytest.mark.parametrize("mass,p", [([0.2,0.4],[1,1]), ([-0.1,1.1],[1,1]),
                                    ([0.5,0.5],[1,1.1]), ([1],[np.nan])])
def test_invalid_cluster_laws_are_rejected(mass, p):
    with pytest.raises(ValueError):
        cluster_profile_reweight(mass, p, thermal_energy=1)
