import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from derive_pulse_finite_size_corrections import (  # noqa: E402
    PulseParameters,
    first_order_response,
)
from lattice_leave_lmax_prediction import (  # noqa: E402
    COVARIANCE_THRESHOLDS,
    FIT_CONFIGS,
    H_GRID,
    PRIMITIVE_NAMES,
    TARGETS,
    TARGET_LENGTH,
    TRAINING_LENGTHS,
    first_order_curve,
    sha256,
    weighted_sequence_fit,
)


def test_weighted_sequence_fit_recovers_polynomial_limits() -> None:
    inverse_volume = 1.0 / np.square(np.asarray(TRAINING_LENGTHS, dtype=float))
    values = 0.37 - 12.5 * inverse_volume + 800.0 * inverse_volume**2
    draws = np.tile(values, (23, 1))
    errors = np.linspace(0.02, 0.05, len(values))
    parameters, draw_parameters = weighted_sequence_fit(
        inverse_volume, values, draws, errors, degree=2
    )
    assert np.allclose(parameters, [0.37, -12.5, 800.0], atol=1.0e-9)
    assert np.allclose(draw_parameters, parameters[None, :], atol=1.0e-9)


def test_vectorized_curve_is_the_derived_first_order_formula() -> None:
    exact_delta_rho = 0.81
    exact_a0 = 2.35
    coefficients = np.asarray([0.27, -0.16, 0.42, -1.30])
    phase_responses = np.asarray([-4.1, -0.25])
    volume = TARGET_LENGTH**2
    curves = first_order_curve(
        H_GRID,
        volume,
        coefficients,
        phase_responses,
        exact_delta_rho,
        exact_a0,
    )[0]
    parameters = PulseParameters(
        delta_rho=exact_delta_rho,
        d0=coefficients[0],
        dchi=coefficients[1],
        a0=exact_a0,
        a1=coefficients[2],
        dn=coefficients[3],
    )
    expected_reweighting = np.asarray(
        [first_order_response(h, volume, parameters) for h in H_GRID]
    )
    cavity_weight = 1.0 / (
        1.0 + np.exp(-(exact_delta_rho * H_GRID - exact_a0))
    )
    expected_total = expected_reweighting + (
        cavity_weight * phase_responses[0]
        + (1.0 - cavity_weight) * phase_responses[1]
    ) / volume
    assert np.allclose(curves[1], expected_reweighting, rtol=0.0, atol=2.0e-16)
    assert np.allclose(curves[0], expected_total, rtol=0.0, atol=2.0e-16)


def test_committed_leave_lmax_artifacts_are_complete_and_nonleaking() -> None:
    coefficient_path = (
        ROOT
        / "data_processed"
        / "lattice_leave_lmax_prediction_coefficients.csv"
    )
    curve_path = (
        ROOT / "data_processed" / "lattice_leave_lmax_prediction_curves.csv"
    )
    diagnostic_path = (
        ROOT
        / "data_processed"
        / "lattice_leave_lmax_prediction_diagnostics.csv"
    )
    covariance_path = (
        ROOT
        / "data_processed"
        / "lattice_leave_lmax_prediction_covariance.npz"
    )
    figure_path = ROOT / "figures" / "lattice_leave_lmax_prediction.pdf"
    metadata_path = (
        ROOT / "generated" / "lattice_leave_lmax_prediction_metadata.json"
    )
    for path in (
        coefficient_path,
        curve_path,
        diagnostic_path,
        covariance_path,
        figure_path,
        metadata_path,
    ):
        assert path.exists() and path.stat().st_size > 0

    coefficients = pd.read_csv(coefficient_path)
    curves = pd.read_csv(curve_path)
    diagnostics = pd.read_csv(diagnostic_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    finite = coefficients[
        coefficients["record_type"] == "finite_volume_training_primitive"
    ]
    assert set(finite["L"].astype(int)) == set(TRAINING_LENGTHS)
    assert TARGET_LENGTH not in set(finite["L"].astype(int))
    assert set(finite["quantity"]) == set(PRIMITIVE_NAMES)
    assert len(finite) == len(TRAINING_LENGTHS) * len(PRIMITIVE_NAMES)

    fitted = coefficients[
        coefficients["record_type"] == "thermodynamic_limit_fit"
    ]
    assert set(fitted["analysis_config"]) == {
        config.name for config in FIT_CONFIGS
    }
    assert len(fitted) == len(FIT_CONFIGS) * len(PRIMITIVE_NAMES)
    primary_dchi = fitted[
        (fitted["analysis_config"] == "primary")
        & (fitted["quantity"] == "dchi")
    ].iloc[0]
    assert primary_dchi["estimate"] == 0.0
    assert primary_dchi["bootstrap_se"] == 0.0

    assert len(curves) == len(FIT_CONFIGS) * len(TARGETS) * len(H_GRID)
    assert set(curves["target"]) == set(TARGETS)
    assert set(curves["target_L"]) == {TARGET_LENGTH}
    assert set(curves["replicas_heldout"]) == {12}
    assert set(curves["replicas_training_per_size"]) == {12}
    assert np.all(np.isfinite(curves["pointwise_z"]))
    assert np.all(curves["simultaneous_critical_95"] > 1.959)

    assert len(diagnostics) == (
        len(FIT_CONFIGS) * (len(TARGETS) + 1) * len(COVARIANCE_THRESHOLDS)
    )
    assert set(diagnostics["target"]) == {
        *TARGETS,
        "joint_total_reweighting",
    }
    assert np.all(np.isfinite(diagnostics["covariance_Q"]))
    assert np.all(
        (diagnostics["covariance_centered_bootstrap_p_value"] > 0.0)
        & (diagnostics["covariance_centered_bootstrap_p_value"] <= 1.0)
    )

    with np.load(covariance_path) as covariance:
        assert np.array_equal(covariance["h_grid"], H_GRID)
        assert covariance["covariance_total"].shape == (len(H_GRID), len(H_GRID))
        assert covariance["covariance_reweighting"].shape == (
            len(H_GRID),
            len(H_GRID),
        )
        assert covariance["covariance_joint_total_reweighting"].shape == (
            2 * len(H_GRID),
            2 * len(H_GRID),
        )

    assert metadata["settings"]["training_lengths"] == list(TRAINING_LENGTHS)
    assert metadata["settings"]["target_length"] == TARGET_LENGTH
    assert metadata["target_nonleakage"]["fit_uses_target_length"] is False
    assert metadata["settings"]["replica_ids_by_length"][str(TARGET_LENGTH)] == list(
        range(12)
    )
    assert "no claim of independent points" in metadata["dependence_statement"]
    expected_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            coefficient_path,
            curve_path,
            diagnostic_path,
            covariance_path,
            figure_path,
        )
    }
    assert metadata["output_sha256"] == expected_hashes
