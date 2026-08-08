import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from derive_pulse_finite_size_corrections import (  # noqa: E402
    PulseParameters,
    fwhm_corrections,
    peak_corrections,
)
from lattice_pulse_correction_prediction import (  # noqa: E402
    ANALYSIS_CONFIGS,
    COEFFICIENTS,
    EXPECTED_LENGTHS,
    H_GRID,
    OBSERVABLES,
    TAYLOR_SPECS,
    extract_primitive_coefficients,
    fixed_exact_slope_many,
    predict_slopes,
    sha256,
    weighted_fit_many,
)


def test_local_phase_primitives_recover_all_four_coefficients() -> None:
    volume = 2304
    delta_rho = 0.81
    a0 = 2.35
    expected = np.array([0.27, -0.16, 0.42, -1.30])
    d0, dchi, a1, dn = expected
    phase_gap = (
        volume * delta_rho
        + d0
        + dchi * H_GRID
        + 0.07 * H_GRID**2
    )
    delta_work = (
        a0
        + (a1 - dn * H_GRID) / volume
        - 0.03 * H_GRID**2 / volume
    )
    observed = {
        "meanN0_liquid": 0.5 * phase_gap,
        "meanN0_gas": -0.5 * phase_gap,
        "delta_betaW": delta_work,
        "deltaN_liquid": np.full_like(H_GRID, 0.5 * dn),
        "deltaN_gas": np.full_like(H_GRID, -0.5 * dn),
    }
    extracted, diagnostics = extract_primitive_coefficients(
        observed,
        volume,
        delta_rho,
        a0,
        TAYLOR_SPECS["quadratic_h0p5"],
    )
    assert np.allclose(extracted, expected, atol=2.0e-10)
    assert abs(diagnostics["dn_derivative_minus_identity"]) < 2.0e-10


def test_size_fits_recover_intercept_slope_and_quadratic_nuisance() -> None:
    inverse_volume = 1.0 / np.square(np.asarray(EXPECTED_LENGTHS, dtype=float))
    coefficient = 0.37 - 12.5 * inverse_volume
    coefficient_draws = np.tile(coefficient, (17, 1))
    errors = np.linspace(0.02, 0.04, len(inverse_volume))
    central, draws, mask = weighted_fit_many(
        inverse_volume,
        coefficient,
        coefficient_draws,
        errors,
        "linear_all",
    )
    assert np.all(mask)
    assert np.allclose(central, [0.37, -12.5], atol=2.0e-11)
    assert np.allclose(draws, central[None, :], atol=2.0e-11)

    exact_limit = 2.1
    direct = exact_limit + 4.7 * inverse_volume - 310.0 * inverse_volume**2
    direct_draws = np.tile(direct, (17, 1))
    slope, slopes, mask, nuisance, nuisances = fixed_exact_slope_many(
        inverse_volume,
        direct,
        direct_draws,
        errors,
        exact_limit,
        "quadratic_all",
    )
    assert np.all(mask)
    assert abs(slope - 4.7) < 2.0e-11
    assert abs(nuisance + 310.0) < 2.0e-8
    assert np.allclose(slopes, slope, atol=2.0e-11)
    assert np.allclose(nuisances, nuisance, atol=2.0e-8)


def test_prediction_is_exactly_the_generated_formula_mapping() -> None:
    coefficients = np.array([0.22, -0.04, 0.31, -1.7])
    predicted = predict_slopes(coefficients)
    from lattice_gas_phase_switch import (  # noqa: WPS433
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
    )

    parameters = PulseParameters(
        delta_rho=exact_spontaneous_magnetization(K_ISING),
        d0=coefficients[0],
        dchi=coefficients[1],
        a0=exact_pair_cavity_parameters(K_ISING)[2],
        a1=coefficients[2],
        dn=coefficients[3],
    )
    peak = peak_corrections(parameters)
    width = fwhm_corrections(parameters)
    expected = np.array(
        [peak["center_h_1"], peak["magnitude_peak_1"], width["fwhm_h_1"]]
    )
    assert np.array_equal(predicted, expected)


def test_committed_coefficient_prediction_artifacts_are_complete() -> None:
    coefficient_path = (
        ROOT / "data_processed" / "lattice_pulse_correction_coefficients.csv"
    )
    slope_path = (
        ROOT
        / "data_processed"
        / "lattice_pulse_correction_slope_predictions.csv"
    )
    metadata_path = (
        ROOT
        / "generated"
        / "lattice_pulse_correction_prediction_metadata.json"
    )
    coefficients = pd.read_csv(coefficient_path)
    slopes = pd.read_csv(slope_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    finite = coefficients[
        coefficients["record_type"] == "finite_volume_primitive"
    ]
    assert set(finite["L"].astype(int)) == set(EXPECTED_LENGTHS)
    assert set(finite["coefficient"]) == set(COEFFICIENTS)
    assert set(finite["replicas_per_size"]) == {12}
    assert len(finite) == len(TAYLOR_SPECS) * len(EXPECTED_LENGTHS) * len(
        COEFFICIENTS
    )

    asymptotic = coefficients[
        coefficients["record_type"] == "asymptotic_coefficient"
    ]
    assert set(asymptotic["analysis_config"]) == {
        config.name for config in ANALYSIS_CONFIGS
    }
    primary = asymptotic[asymptotic["analysis_config"] == "primary"].set_index(
        "coefficient"
    )
    assert set(primary.index) == set(COEFFICIENTS)
    assert primary.loc["dchi", "estimate"] == 0.0
    assert (
        primary.loc["dchi", "identifiability"]
        == "identified_zero_by_particle_hole_symmetry"
    )
    assert "unresolved_from_zero" in primary.loc["d0", "identifiability"]
    assert "unresolved_from_zero" in primary.loc["a1", "identifiability"]
    assert primary.loc["dn", "identifiability"] == "nonzero_and_sensitivity_stable"

    comparisons = slopes[slopes["record_type"] == "observable_comparison"]
    assert len(comparisons) == len(ANALYSIS_CONFIGS) * len(OBSERVABLES)
    assert set(comparisons["observable"]) == set(OBSERVABLES)
    primary_comparison = comparisons[
        comparisons["analysis_config"] == "primary"
    ]
    assert set(primary_comparison["direct_fit_model"]) == {"quadratic_all"}
    assert np.all(
        np.isfinite(
            primary_comparison[
                [
                    "predicted_B",
                    "direct_B",
                    "direct_C_1_over_V2",
                    "difference_bootstrap_se",
                    "difference_z",
                    "corr_predicted_direct",
                ]
            ].to_numpy()
        )
    )
    assert np.max(np.abs(primary_comparison["difference_z"])) < 2.0
    covariance = slopes[slopes["record_type"] == "primary_covariance"]
    assert len(covariance) == 36
    assert np.all(np.isfinite(covariance["covariance"]))

    assert metadata["settings"]["bootstrap_draws"] == 2000
    assert metadata["settings"]["replicas_by_length"] == {
        str(length): 12 for length in EXPECTED_LENGTHS
    }
    assert "no fitted pulse slope" in metadata["coefficient_source"]
    assert metadata["symmetry"]["dchi_identified_and_set_to_zero"] is True
    assert metadata["output_sha256"] == {
        "data_processed/lattice_pulse_correction_coefficients.csv": sha256(
            coefficient_path
        ),
        "data_processed/lattice_pulse_correction_slope_predictions.csv": sha256(
            slope_path
        ),
    }
