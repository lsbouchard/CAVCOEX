from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lattice_gas_phase_switch import (  # noqa: E402
    K_ISING,
    exact_pair_cavity_parameters,
    exact_spontaneous_magnetization,
)
from lattice_pulse_scaling import analytic_pulse_limits  # noqa: E402
from lattice_validation_statistics import (  # noqa: E402
    OBSERVABLES,
    deterministic_three_folds,
    finite_volume_reconstruction,
    fixed_exact_linear_fit,
    quadratic_from_whitener,
    regularized_precision,
)


def test_fixed_exact_fit_recovers_synthetic_slope_and_is_order_invariant() -> None:
    inverse_volume = 1.0 / np.array([256.0, 576.0, 1024.0, 2304.0, 4096.0])
    exact_limit = 1.731
    slope = -8.25
    standard_errors = np.linspace(0.001, 0.002, len(inverse_volume))
    values = exact_limit + slope * inverse_volume

    fit = fixed_exact_linear_fit(
        inverse_volume, values, standard_errors, exact_limit
    )
    assert abs(fit["B"] - slope) < 1.0e-12
    assert fit["chi2"] < 1.0e-20
    assert fit["dof"] == len(values) - 1

    order = np.array([3, 0, 4, 1, 2])
    permuted = fixed_exact_linear_fit(
        inverse_volume[order],
        values[order],
        standard_errors[order],
        exact_limit,
    )
    for key in ("B", "se_B", "chi2", "chi2_p_value"):
        assert np.isclose(fit[key], permuted[key], rtol=1.0e-13, atol=1.0e-15)


def test_correlation_eigenmode_regularization_is_stable_and_unit_invariant() -> None:
    covariance = np.array(
        [
            [4.0, 1.999999999, 0.1],
            [1.999999999, 1.0, 0.05],
            [0.1, 0.05, 9.0],
        ]
    )
    residual = np.array([0.3, 0.15, -0.2])
    regularized = regularized_precision(covariance, 1.0e-8)
    assert regularized["rank"] == 2
    assert np.all(np.isfinite(regularized["precision"]))
    statistic = quadratic_from_whitener(
        residual[None, :], regularized["whitener"]
    )[0]

    unit_change = np.diag([10.0, 0.01, 3.0])
    transformed = regularized_precision(
        unit_change @ covariance @ unit_change, 1.0e-8
    )
    transformed_statistic = quadratic_from_whitener(
        (unit_change @ residual)[None, :], transformed["whitener"]
    )[0]
    assert transformed["rank"] == regularized["rank"]
    assert np.isclose(statistic, transformed_statistic, rtol=2.0e-8)


def test_deterministic_folds_are_disjoint_and_exhaustive() -> None:
    folds = deterministic_three_folds(np.arange(12)[::-1])
    heldout_counts = {replica: 0 for replica in range(12)}
    assert len(folds) == 3
    for training, heldout in folds:
        assert len(training) == 8
        assert len(heldout) == 4
        assert set(training).isdisjoint(heldout)
        assert set(training) | set(heldout) == set(range(12))
        for replica in heldout:
            heldout_counts[replica] += 1
    assert set(heldout_counts.values()) == {1}


def test_exact_phase_decomposition_closes_for_constructed_observables() -> None:
    q0 = np.array([0.18, 0.41, 0.77])
    qc = np.array([0.09, 0.27, 0.63])
    mean_liquid = np.array([75.0, 76.0, 77.0])
    mean_gas = np.array([22.0, 23.0, 24.0])
    liquid_response = np.array([-1.4, -1.2, -1.0])
    gas_response = np.array([-0.3, -0.2, -0.1])
    reweighting = (qc - q0) * (mean_liquid - mean_gas)
    intraphase = qc * liquid_response + (1.0 - qc) * gas_response
    observed = {
        "q0_liquid": q0,
        "qC_liquid": qc,
        "meanN0_liquid": mean_liquid,
        "meanN0_gas": mean_gas,
        "deltaN_liquid": liquid_response,
        "deltaN_gas": gas_response,
        "deltaN_phase_reweighting": reweighting,
        "deltaN": reweighting + intraphase,
    }
    reconstruction = finite_volume_reconstruction(observed)
    assert np.allclose(reconstruction["reweighting"], reweighting)
    assert np.allclose(reconstruction["intraphase"], intraphase)
    assert np.max(np.abs(reconstruction["total_closure"])) < 1.0e-14
    assert np.max(np.abs(reconstruction["reweighting_closure"])) < 1.0e-14


def test_validation_artifacts_are_complete_and_self_consistent() -> None:
    fixed = pd.read_csv(
        ROOT / "data_processed" / "lattice_exact_limit_fixed_fits.csv"
    )
    joint = pd.read_csv(
        ROOT / "data_processed" / "lattice_scalar_joint_validation.csv"
    )
    covariance = pd.read_csv(
        ROOT / "data_processed" / "lattice_scalar_covariance.csv"
    )
    holdout = pd.read_csv(
        ROOT / "data_processed" / "lattice_curve_holdout_test.csv"
    )
    metadata = json.loads(
        (ROOT / "generated" / "lattice_validation_statistics_metadata.json").read_text()
    )

    modes = {"paired", "unpaired"}
    sizes = {16, 24, 32, 48, 64}
    exact = analytic_pulse_limits(
        exact_spontaneous_magnetization(K_ISING),
        exact_pair_cavity_parameters(K_ISING)[2],
    )
    assert len(fixed) == len(modes) * len(OBSERVABLES)
    assert set(fixed["mode"]) == modes
    assert set(fixed["observable"]) == set(OBSERVABLES)
    assert np.all(fixed["dof"] == 4)
    assert np.all(fixed["replicas_per_size"] == 12)
    assert np.all(np.isfinite(fixed[["B", "se_B", "chi2"]]))
    assert np.all(fixed["se_B"] > 0.0)
    for row in fixed.itertuples():
        assert np.isclose(row.exact_limit, exact[row.observable], rtol=0.0, atol=1e-12)
        assert 0.0 <= row.chi2_p_value <= 1.0
        assert 0.0 <= row.centered_hierarchical_bootstrap_p_value <= 1.0

    central = joint[joint["record_type"] == "central"]
    bootstrap = joint[joint["record_type"] == "bootstrap"]
    joint_fits = joint[joint["record_type"] == "joint_fit"]
    expected_draws = int(metadata["scalar_bootstrap_draws"])
    assert len(central) == len(modes) * len(sizes)
    assert len(bootstrap) == len(modes) * len(sizes) * expected_draws
    assert set(central["mode"]) == modes
    assert set(central["L"]) == sizes
    assert np.all(central["replicas"] == 12)
    assert np.all(central["joint_dof"] > 0)
    assert np.all(np.isfinite(central[list(OBSERVABLES)]))
    assert np.all(np.isfinite(bootstrap[list(OBSERVABLES)]))
    assert len(joint_fits) == len(modes) * 3
    assert set(joint_fits["fit_model"]) == {
        "linear_all",
        "quadratic_all",
        "linear_drop_L16",
    }
    assert np.all(joint_fits["joint_parameter_count"] < joint_fits["joint_rank"])
    assert np.all(
        joint_fits["joint_dof"]
        == joint_fits["joint_rank"] - joint_fits["joint_parameter_count"]
    )
    assert np.all(
        np.isfinite(
            joint_fits[
                [
                    "joint_Q",
                    "joint_chi2_p_value",
                    "joint_centered_bootstrap_p_value",
                ]
            ]
        )
    )
    paired_models = joint_fits[joint_fits["mode"] == "paired"].set_index(
        "fit_model"
    )
    assert (
        paired_models.loc[
            "linear_all", "joint_centered_bootstrap_p_value"
        ]
        < 0.01
    )
    assert (
        paired_models.loc[
            "quadratic_all", "joint_centered_bootstrap_p_value"
        ]
        > 0.05
    )
    assert (
        paired_models.loc[
            "linear_drop_L16", "joint_centered_bootstrap_p_value"
        ]
        < 0.01
    )

    assert len(covariance) == len(modes) * len(sizes) * len(OBSERVABLES) ** 2
    assert set(covariance["mode"]) == modes
    assert set(covariance["L"]) == sizes
    for (_, _), group in covariance.groupby(["mode", "L"]):
        matrix = group.pivot(
            index="row_observable",
            columns="column_observable",
            values="covariance",
        ).loc[list(OBSERVABLES), list(OBSERVABLES)].to_numpy()
        assert np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-15)
        assert np.all(np.linalg.eigvalsh(matrix) >= -1.0e-12)

    assert len(holdout) == 2 * 3 * 3 * 2
    assert set(holdout["L"]) == {48, 64}
    assert set(holdout["prediction_kind"]) == {
        "exact_crossfit",
        "leading_logistic",
        "first_corrected_logistic",
    }
    assert set(holdout["target"]) == {"total", "reweighting"}
    assert np.all(holdout["training_replica_count"] == 8)
    assert np.all(holdout["heldout_replica_count"] == 4)
    assert np.all(holdout["covariance_retained_rank"] > 0)
    assert np.all(holdout["maximum_training_decomposition_closure"] < 1.0e-10)
    exact_crossfit = holdout[holdout["prediction_kind"] == "exact_crossfit"]
    corrected = holdout[
        holdout["prediction_kind"] == "first_corrected_logistic"
    ]
    leading = holdout[holdout["prediction_kind"] == "leading_logistic"]
    assert np.all(
        exact_crossfit["centered_hierarchical_bootstrap_p_value"] > 0.05
    )
    assert np.all(exact_crossfit["inside_simultaneous_95_band"])
    assert np.all(corrected["inside_simultaneous_95_band"])
    assert not np.any(leading["inside_simultaneous_95_band"])
    keys = ["L", "fold", "target"]
    corrected_rmse = corrected.set_index(keys)["rmse_response_over_V"]
    leading_rmse = leading.set_index(keys)["rmse_response_over_V"]
    rmse_ratio = corrected_rmse / leading_rmse
    assert np.median(rmse_ratio) < 0.20
    assert np.max(rmse_ratio) < 0.50
    for row in holdout.itertuples():
        training = {int(value) for value in row.training_replica_ids.split()}
        heldout_ids = {int(value) for value in row.heldout_replica_ids.split()}
        assert training.isdisjoint(heldout_ids)
        assert training | heldout_ids == set(range(12))
