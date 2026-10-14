from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lattice_free_intercept_joint_quadratic import (  # noqa: E402
    ANALYSIS_VERSION,
    EIGENMODE_SENSITIVITY_THRESHOLDS,
    EXPECTED_LENGTHS,
    gls_fit,
    joint_quadratic_design,
    nested_intercept_statistic,
)
from lattice_validation_statistics import (  # noqa: E402
    COVARIANCE_THRESHOLDS,
    OBSERVABLES,
    PRIMARY_COVARIANCE_THRESHOLD,
    regularized_precision,
)


SUMMARY_PATH = (
    ROOT
    / "data_processed"
    / "lattice_free_intercept_joint_quadratic_summary.csv"
)
PARAMETER_PATH = (
    ROOT
    / "data_processed"
    / "lattice_free_intercept_joint_quadratic_parameters.csv"
)
INTERCEPT_COVARIANCE_PATH = (
    ROOT
    / "data_processed"
    / "lattice_free_intercept_joint_quadratic_intercept_covariance.csv"
)
EIGENMODE_PATH = (
    ROOT
    / "data_processed"
    / "lattice_free_intercept_joint_quadratic_eigenmodes.csv"
)
BOOTSTRAP_PATH = (
    ROOT
    / "data_processed"
    / "lattice_free_intercept_joint_quadratic_bootstrap.csv"
)
METADATA_PATH = (
    ROOT
    / "generated"
    / "lattice_free_intercept_joint_quadratic_metadata.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_free_design_recovers_nine_parameters_and_nested_null() -> None:
    volumes = np.asarray(EXPECTED_LENGTHS, dtype=float) ** 2
    design, names = joint_quadratic_design(volumes)
    assert design.shape == (15, 9)
    assert len(names) == 9
    exact = np.asarray([2.3, 0.7, 5.5])
    coefficients = np.asarray(
        [2.3, 0.1, -0.03, 0.7, 0.02, 0.01, 5.5, 0.4, -0.08]
    )
    values = design @ coefficients
    covariance = np.diag(np.linspace(0.8, 1.2, len(values)))
    whitener = np.asarray(regularized_precision(covariance, 1.0e-8)["whitener"])
    fitted = gls_fit(values, np.zeros_like(values), design, whitener)
    assert fitted["rank"] == 9
    assert np.allclose(fitted["coefficients"], coefficients, atol=1.0e-12)
    assert fitted["Q"] < 1.0e-24

    nested = nested_intercept_statistic(values, exact, design, whitener)
    assert nested["restriction_count"] == 3
    assert nested["delta_Q"] < 1.0e-24

    shifted = coefficients.copy()
    shifted[[0, 3, 6]] += np.asarray([0.08, -0.03, 0.12])
    alternative = nested_intercept_statistic(
        design @ shifted, exact, design, whitener
    )
    assert alternative["unrestricted"]["Q"] < 1.0e-24
    assert alternative["delta_Q"] > 0.0


def test_production_outputs_have_declared_counts_ranks_and_calibration() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    parameters = pd.read_csv(PARAMETER_PATH)
    intercept_covariance = pd.read_csv(INTERCEPT_COVARIANCE_PATH)
    eigenmodes = pd.read_csv(EIGENMODE_PATH)
    bootstrap = pd.read_csv(BOOTSTRAP_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert len(summary) == len(EIGENMODE_SENSITIVITY_THRESHOLDS)
    assert set(summary["relative_eigenvalue_threshold"]) == set(
        EIGENMODE_SENSITIVITY_THRESHOLDS
    )
    assert summary["is_primary_threshold"].sum() == 1
    assert np.all(summary["observation_count"] == 15)
    assert np.all(summary["unrestricted_parameter_count"] == 9)
    assert np.all(summary["restricted_nuisance_parameter_count"] == 6)
    assert np.all(summary["intercept_restriction_count"] == 3)
    assert np.all(
        summary["global_lack_of_fit_dof"]
        == summary["covariance_retained_rank"]
        - summary["unrestricted_whitened_design_rank"]
    )
    assert np.all(summary["global_lack_of_fit_dof"] > 0)
    assert np.all(summary["intercept_test_dof"] == 3)
    p_columns = [
        "global_lack_of_fit_chi2_p_value",
        "global_lack_of_fit_bootstrap_p_value",
        "intercept_test_chi2_p_value",
        "intercept_test_bootstrap_p_value",
    ]
    assert np.all((summary[p_columns] >= 0.0) & (summary[p_columns] <= 1.0))
    assert np.all(summary["intercept_nested_delta_Q"] >= -1.0e-12)

    assert len(parameters) == len(EIGENMODE_SENSITIVITY_THRESHOLDS) * 9
    assert set(parameters["observable"]) == set(OBSERVABLES)
    assert set(parameters["coefficient"]) == {"A", "B", "C"}
    assert np.all(parameters["physical_bootstrap_standard_error"] > 0.0)
    assert len(intercept_covariance) == len(EIGENMODE_SENSITIVITY_THRESHOLDS) * 9
    assert len(eigenmodes) == len(EIGENMODE_SENSITIVITY_THRESHOLDS) * 15
    assert np.all(eigenmodes.groupby("relative_eigenvalue_threshold").size() == 15)

    expected_draws = int(metadata["data"]["bootstrap_draws"])
    assert (
        len(bootstrap)
        == len(EIGENMODE_SENSITIVITY_THRESHOLDS) * expected_draws
    )
    assert np.all(
        bootstrap.groupby("relative_eigenvalue_threshold").size()
        == expected_draws
    )
    for row in summary.itertuples():
        draws = bootstrap[
            np.isclose(
                bootstrap["relative_eigenvalue_threshold"],
                row.relative_eigenvalue_threshold,
                rtol=0.0,
                atol=1.0e-16,
            )
        ]
        global_p = (
            1
            + np.count_nonzero(
                draws["global_lack_of_fit_Q_under_fitted_free_null"]
                >= row.global_lack_of_fit_Q
            )
        ) / (expected_draws + 1)
        intercept_p = (
            1
            + np.count_nonzero(
                draws[
                    "intercept_nested_delta_Q_under_exact_intercept_null"
                ]
                >= row.intercept_nested_delta_Q
            )
        ) / (expected_draws + 1)
        assert np.isclose(global_p, row.global_lack_of_fit_bootstrap_p_value)
        assert np.isclose(intercept_p, row.intercept_test_bootstrap_p_value)

    for (_, _), group in intercept_covariance.groupby(
        ["relative_eigenvalue_threshold", "is_primary_threshold"]
    ):
        matrix = group.pivot(
            index="row_observable",
            columns="column_observable",
            values="bootstrap_covariance",
        ).loc[list(OBSERVABLES), list(OBSERVABLES)].to_numpy()
        assert np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-15)
        assert np.min(np.linalg.eigvalsh(matrix)) >= -1.0e-12


def test_primary_model_dimensions_and_interval_ordering() -> None:
    summary = pd.read_csv(SUMMARY_PATH).sort_values(
        "relative_eigenvalue_threshold"
    )
    primary = summary[
        np.isclose(
            summary["relative_eigenvalue_threshold"],
            PRIMARY_COVARIANCE_THRESHOLD,
            rtol=0.0,
            atol=1.0e-16,
        )
    ].iloc[0]
    assert primary["covariance_retained_rank"] == 15
    assert primary["unrestricted_whitened_design_rank"] == 9
    assert primary["restricted_whitened_design_rank"] == 6
    assert primary["global_lack_of_fit_dof"] == 6
    inherited = summary[
        summary["relative_eigenvalue_threshold"].isin(COVARIANCE_THRESHOLDS)
    ]
    assert inherited["covariance_retained_rank"].nunique() == 1
    assert inherited["covariance_retained_rank"].iloc[0] == 15
    stress = summary[
        np.isclose(
            summary["relative_eigenvalue_threshold"],
            1.0e-4,
            rtol=0.0,
            atol=1.0e-16,
        )
    ].iloc[0]
    assert stress["covariance_retained_rank"] == 14

    primary_intercepts = pd.read_csv(PARAMETER_PATH)
    primary_intercepts = primary_intercepts[
        primary_intercepts["is_primary_threshold"]
        & (primary_intercepts["coefficient"] == "A")
    ]
    assert np.all(
        primary_intercepts["physical_bootstrap_percentile_2p5"]
        < primary_intercepts["physical_bootstrap_percentile_97p5"]
    )


def test_metadata_hashes_and_model_accounting_are_current() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["analysis_version"] == ANALYSIS_VERSION
    assert metadata["model"]["unrestricted_parameter_count"] == 9
    assert metadata["model"]["restricted_parameter_count"] == 6
    assert metadata["model"]["intercept_restriction_count"] == 3
    assert metadata["covariance_regularization"][
        "primary_relative_threshold"
    ] == PRIMARY_COVARIANCE_THRESHOLD
    assert metadata["bootstrap_calibration"]["tail_probability"] == (
        "(1 + count[T_boot >= T_obs])/(B + 1)"
    )
    assert metadata["script_sha256"] == _sha256(
        ROOT / "scripts" / "lattice_free_intercept_joint_quadratic.py"
    )
    assert metadata["dependency_sha256"] == {
        "scripts/lattice_validation_statistics.py": _sha256(
            ROOT / "scripts" / "lattice_validation_statistics.py"
        )
    }
    for relative_path, digest in metadata["input_sha256"].items():
        assert digest == _sha256(ROOT / relative_path)
    for relative_path, digest in metadata["output_sha256"].items():
        assert digest == _sha256(ROOT / relative_path)
