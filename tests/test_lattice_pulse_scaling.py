import sys
import json
import hashlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lattice_analysis_common import (  # noqa: E402
    local_quadratic_extremum,
    weighted_linear_fit,
)
from lattice_gas_phase_switch import logistic  # noqa: E402
from lattice_pulse_scaling import (  # noqa: E402
    analytic_pulse_limits,
    continuous_pulse_observables,
    weighted_polynomial_fit,
)


def test_plot_only_preserves_numerical_execution_and_inputs(tmp_path,monkeypatch):
    import lattice_pulse_scaling as pulse
    for directory in ('data_processed','generated','figures'):
        (tmp_path/directory).mkdir()
    names = ('lattice_peak_center_amplitude.csv','lattice_peak_scaling_fits.csv',
             'lattice_width_hierarchical.csv','lattice_width_hierarchical_fit.csv',
             'lattice_pulse_curves.csv','lattice_scalar_joint_validation.csv')
    for name in names:
        (tmp_path/'data_processed'/name).write_text('value\n1\n')
    original = {'generated_utc':'original time','bootstrap_draws':5000,
                'script_sha256':'original execution source'}
    metadata = tmp_path/'generated/lattice_pulse_scaling_metadata.json'
    metadata.write_text(json.dumps(original))
    monkeypatch.setattr(pulse,'ROOT',tmp_path)
    monkeypatch.setattr(pulse,'parse_args',lambda:SimpleNamespace(plot_only=True))
    def render(*args):
        args[-1].write_bytes(b'new figure')
    monkeypatch.setattr(pulse,'make_figure',render)
    pulse.main()
    actual = json.loads(metadata.read_text())
    assert {key:actual[key] for key in original} == original
    assert actual['figure_render']['scope'].startswith('figure only')
    for name in names:
        path = tmp_path/'data_processed'/name
        assert path.read_text() == 'value\n1\n'
        assert actual['figure_render']['input_sha256'][f'data_processed/{name}'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_analytic_pulse_center_magnitude_and_width() -> None:
    delta_rho = 0.83
    delta_beta_w = 2.7
    expected = analytic_pulse_limits(delta_rho, delta_beta_w)
    h = np.linspace(-5.0, 8.0, 26_001)
    y = delta_rho * (
        logistic(delta_rho * h - delta_beta_w)
        - logistic(delta_rho * h)
    )
    center, magnitude = local_quadratic_extremum(h, y)
    assert abs(center - expected["peak_center_h"]) < 1.0e-8
    assert abs(magnitude - expected["peak_magnitude"]) < 1.0e-8
    continuous = continuous_pulse_observables(h[::100], y[::100])
    assert abs(continuous["peak_center_h"] - expected["peak_center_h"]) < 2.0e-5
    assert abs(continuous["peak_magnitude"] - expected["peak_magnitude"]) < 2.0e-6
    assert abs(continuous["fwhm_h"] - expected["fwhm_h"]) < 3.0e-5


def test_continuous_extraction_improves_on_sampled_grid_peak() -> None:
    center_exact = 0.373
    sigma = 0.61
    h = np.linspace(-2.0, 2.0, 41)
    response = -np.exp(-0.5 * ((h - center_exact) / sigma) ** 2)
    extracted = continuous_pulse_observables(h, response)
    sampled_center = float(h[np.argmax(np.abs(response))])
    exact_width = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert abs(extracted["peak_center_h"] - center_exact) < 1.0e-4
    assert abs(extracted["peak_center_h"] - center_exact) < abs(
        sampled_center - center_exact
    )
    assert abs(extracted["peak_magnitude"] - 1.0) < 5.0e-6
    assert abs(extracted["fwhm_h"] - exact_width) < 3.0e-5


def test_weighted_linear_fit_recovers_synthetic_finite_size_correction() -> None:
    volumes = np.array([256.0, 576.0, 1024.0, 2304.0, 4096.0])
    exact_intercept = 2.125
    correction = -3.75
    values = exact_intercept + correction / volumes
    standard_errors = np.linspace(0.001, 0.002, len(volumes))
    fit = weighted_linear_fit(1.0 / volumes, values, standard_errors)
    assert abs(fit["A"] - exact_intercept) < 1.0e-12
    assert abs(fit["B"] - correction) < 1.0e-10
    assert fit["chi2"] < 1.0e-20
    assert fit["dof"] == 3


def test_weighted_quadratic_fit_recovers_synthetic_correction() -> None:
    volumes = np.array([256.0, 576.0, 1024.0, 2304.0, 4096.0])
    inverse_volume = 1.0 / volumes
    exact_intercept = 2.125
    values = exact_intercept - 3.75 * inverse_volume + 290.0 * inverse_volume**2
    standard_errors = np.linspace(0.001, 0.002, len(volumes))
    fit = weighted_polynomial_fit(
        inverse_volume, values, standard_errors, degree=2
    )
    assert abs(fit["A"] - exact_intercept) < 1.0e-12
    assert abs(fit["B"] + 3.75) < 1.0e-9
    assert abs(fit["C"] - 290.0) < 1.0e-7
    assert fit["chi2"] < 1.0e-20
    assert fit["dof"] == 2


def test_pulse_fits_report_complete_exact_limit_comparisons() -> None:
    peak_observables = pd.read_csv(
        ROOT / "data_processed" / "lattice_peak_center_amplitude.csv"
    )
    width_observables = pd.read_csv(
        ROOT / "data_processed" / "lattice_width_hierarchical.csv"
    )
    assert set(peak_observables["replicas"]) == {12}
    assert set(width_observables["replicas"]) == {12}

    peak = pd.read_csv(
        ROOT / "data_processed" / "lattice_peak_scaling_fits.csv"
    )
    width = pd.read_csv(
        ROOT / "data_processed" / "lattice_width_hierarchical_fit.csv"
    )
    fits = pd.concat((peak, width), ignore_index=True)
    assert set(fits["observable"]) == {
        "peak_center_h",
        "peak_magnitude",
        "fwhm_h",
    }
    assert np.all(fits["dof"] == 3)
    assert np.all(fits[["cov_AA", "cov_BB"]].to_numpy() > 0.0)

    fixed = pd.read_csv(
        ROOT / "data_processed" / "lattice_exact_limit_fixed_fits.csv"
    )
    assert len(fixed) == 6
    assert set(fixed["mode"]) == {"paired", "unpaired"}
    assert set(fixed["observable"]) == {
        "peak_center_h",
        "peak_magnitude",
        "fwhm_h",
    }
    assert set(fixed["sizes"]) == {"16 24 32 48 64"}
    assert set(fixed["replicas_per_size"]) == {12}
    assert set(fixed["dof"]) == {4}
    numerical_columns = [
        "exact_limit",
        "B",
        "se_B",
        "chi2",
        "chi2_p_value",
        "centered_hierarchical_bootstrap_p_value",
        "max_abs_standardized_residual",
    ]
    assert np.all(np.isfinite(fixed[numerical_columns].to_numpy()))
    assert np.all(fixed["se_B"] > 0.0)
    for p_value in (
        "chi2_p_value",
        "centered_hierarchical_bootstrap_p_value",
    ):
        assert np.all(fixed[p_value].between(0.0, 1.0))
    from scipy.stats import chi2
    np.testing.assert_allclose(fixed["chi2_p_value"],
                               chi2.sf(fixed["chi2"], fixed["dof"]), rtol=1e-12)
    assert np.all(fixed["max_abs_standardized_residual"] >= 0.0)

    exact_from_free_fit = fits.set_index("observable")["exact_limit"]
    for observable, group in fixed.groupby("observable"):
        assert np.allclose(
            group["exact_limit"], exact_from_free_fit.loc[observable]
        )


def test_pulse_sensitivity_outputs_are_complete_and_stable() -> None:
    finite_size = pd.read_csv(
        ROOT / "data_processed" / "lattice_pulse_finite_size_sensitivity.csv"
    )
    assert set(finite_size["observable"]) == {
        "peak_center_h",
        "peak_magnitude",
        "fwhm_h",
    }
    assert set(finite_size["model"]) == {
        "linear_all",
        "quadratic_all",
        "linear_drop_L16",
    }
    assert len(finite_size) == 9
    assert np.all(np.isfinite(finite_size[["A", "se_A", "chi2"]]))
    assert np.all(finite_size["se_A"] > 0.0)

    grid = pd.read_csv(
        ROOT / "data_processed" / "lattice_pulse_grid_sensitivity.csv"
    )
    intercepts = grid[grid["scope"] == "thermodynamic_intercept"]
    assert set(np.round(intercepts["grid_spacing_h"], 6)) == {0.025, 0.05, 0.1}
    assert len(intercepts) == 9
    normalized_shift = np.abs(
        intercepts["difference_from_finest"] / intercepts["se_estimate"]
    )
    assert np.all(normalized_shift < 0.02)
