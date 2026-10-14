#!/usr/bin/env python3
"""Covariance-aware free-intercept test of the lattice pulse limits.

The upstream lattice validation stores the central and stratified block-bootstrap
vectors for three pulse observables at five independently simulated lattice
sizes.  This module uses those records without regenerating or altering them.
It fits the joint model

    X_L = A_X + B_X / V + C_X / V**2

with all three intercepts free.  It then reports (i) a nine-parameter global
lack-of-fit test and (ii) a nested, three-restriction test of the joint null
that the free intercept vector equals the exact thermodynamic-limit vector.
Both primary p values are calibrated by adding the stored centered
stratified block-bootstrap noise to the appropriate fitted null model and
refitting every replicate.  Correlation-matrix eigenmode thresholds are
reported as a declared sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.linalg import block_diag
from scipy.stats import chi2

try:
    from .lattice_validation_statistics import (
        COVARIANCE_THRESHOLDS,
        OBSERVABLES,
        PRIMARY_COVARIANCE_THRESHOLD,
        REFERENCE_VOLUME,
        _calibrated_tail_probability,
        regularized_precision,
        sha256,
    )
except ImportError:
    from lattice_validation_statistics import (
        COVARIANCE_THRESHOLDS,
        OBSERVABLES,
        PRIMARY_COVARIANCE_THRESHOLD,
        REFERENCE_VOLUME,
        _calibrated_tail_probability,
        regularized_precision,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_processed" / "lattice_scalar_joint_validation.csv"
EXPECTED_LENGTHS = (16, 24, 32, 48, 64)
ANALYSIS_VERSION = "lattice-free-intercept-joint-quadratic-v1"
# The three inherited thresholds leave all modes intact for these data.  The
# additional 1e-4 stress threshold deliberately tests the conclusion after
# truncating the single weakest correlation mode; it is not the primary fit.
EIGENMODE_SENSITIVITY_THRESHOLDS = (
    1.0e-4,
    *COVARIANCE_THRESHOLDS,
)


def joint_quadratic_design(
    volumes: np.ndarray,
    reference_volume: float = float(REFERENCE_VOLUME),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the stable nine-column free quadratic design.

    The fitted columns are ``1``, ``u=V_ref/V``, and ``u**2`` separately for
    each observable.  Thus the fitted slope coefficients are numerically
    scaled: ``B_physical = B_scaled * V_ref`` and
    ``C_physical = C_scaled * V_ref**2``.
    """

    volumes = np.asarray(volumes, dtype=float)
    if volumes.ndim != 1 or len(volumes) < 3:
        raise ValueError("volumes must be a one-dimensional vector")
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("volumes must be finite and positive")
    if not np.isfinite(reference_volume) or reference_volume <= 0.0:
        raise ValueError("reference_volume must be finite and positive")

    observable_count = len(OBSERVABLES)
    design = np.zeros((len(volumes) * observable_count, 3 * observable_count))
    names: list[str] = []
    for observable in OBSERVABLES:
        names.extend(
            (
                f"A_{observable}",
                f"B_scaled_{observable}",
                f"C_scaled_{observable}",
            )
        )
    for size_index, volume in enumerate(volumes):
        u = reference_volume / volume
        for observable_index in range(observable_count):
            row = size_index * observable_count + observable_index
            column = 3 * observable_index
            design[row, column : column + 3] = (1.0, u, u * u)
    return design, tuple(names)


def restricted_quadratic_design(
    free_design: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Remove the three intercept columns from a free quadratic design."""

    free_design = np.asarray(free_design, dtype=float)
    expected_columns = 3 * len(OBSERVABLES)
    if free_design.ndim != 2 or free_design.shape[1] != expected_columns:
        raise ValueError("free_design must have three columns per observable")
    nuisance_columns = tuple(
        column
        for observable_index in range(len(OBSERVABLES))
        for column in (3 * observable_index + 1, 3 * observable_index + 2)
    )
    return free_design[:, nuisance_columns], nuisance_columns


def gls_fit(
    values: np.ndarray,
    offset: np.ndarray,
    design: np.ndarray,
    whitener: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """Fit a linear GLS model in the retained covariance eigenmodes."""

    values = np.asarray(values, dtype=float)
    offset = np.asarray(offset, dtype=float)
    design = np.asarray(design, dtype=float)
    whitener = np.asarray(whitener, dtype=float)
    if values.ndim != 1 or offset.shape != values.shape:
        raise ValueError("values and offset must be one-dimensional peers")
    if design.shape[0] != len(values):
        raise ValueError("design row count must match values")
    if whitener.ndim != 2 or whitener.shape[1] != len(values):
        raise ValueError("whitener column count must match values")

    whitened_design = whitener @ design
    whitened_values = whitener @ (values - offset)
    coefficients, _, rank, _ = np.linalg.lstsq(
        whitened_design, whitened_values, rcond=None
    )
    if rank != design.shape[1]:
        raise ValueError(
            f"whitened design rank {rank} is below {design.shape[1]} parameters"
        )
    fitted = offset + design @ coefficients
    residual = values - fitted
    statistic = float(np.sum((whitener @ residual) ** 2))
    return {
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
        "Q": statistic,
        "rank": int(rank),
    }


def batch_gls_fit(
    samples: np.ndarray,
    offset: np.ndarray,
    design: np.ndarray,
    whitener: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit the same GLS model to every row in ``samples``."""

    samples = np.asarray(samples, dtype=float)
    offset = np.asarray(offset, dtype=float)
    design = np.asarray(design, dtype=float)
    whitener = np.asarray(whitener, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(offset):
        raise ValueError("samples must have one row per draw and one column per datum")
    whitened_design = whitener @ design
    design_inverse = np.linalg.pinv(whitened_design)
    rank = int(np.linalg.matrix_rank(whitened_design))
    if rank != design.shape[1]:
        raise ValueError(
            f"whitened design rank {rank} is below {design.shape[1]} parameters"
        )
    whitened_samples = (samples - offset[None, :]) @ whitener.T
    coefficients = whitened_samples @ design_inverse.T
    whitened_residual = whitened_samples - coefficients @ whitened_design.T
    statistics = np.sum(whitened_residual**2, axis=1)
    return coefficients, statistics, rank


def nested_intercept_statistic(
    values: np.ndarray,
    exact_intercepts: np.ndarray,
    design: np.ndarray,
    whitener: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """Compare free and exact-intercept quadratic fits by their GLS Q values."""

    exact_intercepts = np.asarray(exact_intercepts, dtype=float)
    if exact_intercepts.shape != (len(OBSERVABLES),):
        raise ValueError("exact_intercepts must contain one value per observable")
    repeats = len(values) // len(OBSERVABLES)
    if repeats * len(OBSERVABLES) != len(values):
        raise ValueError("values do not contain complete observable triplets")
    exact_vector = np.tile(exact_intercepts, repeats)
    restricted_design, _ = restricted_quadratic_design(design)
    unrestricted = gls_fit(values, np.zeros_like(values), design, whitener)
    restricted = gls_fit(values, exact_vector, restricted_design, whitener)
    delta = float(restricted["Q"] - unrestricted["Q"])
    tolerance = 1.0e-10 * max(1.0, abs(float(restricted["Q"])))
    if delta < -tolerance:
        raise ArithmeticError("nested GLS statistic is negative beyond roundoff")
    return {
        "unrestricted": unrestricted,
        "restricted": restricted,
        "delta_Q": max(0.0, delta),
        "restriction_count": int(
            unrestricted["rank"] - restricted["rank"]
        ),
        "exact_vector": exact_vector,
    }


def load_paired_vectors(path: Path) -> dict[str, np.ndarray | list[int] | int]:
    """Load and validate the 15-vector and its stored bootstrap replicates."""

    frame = pd.read_csv(path)
    required = {
        "record_type",
        "mode",
        "L",
        "draw",
        *OBSERVABLES,
        *(f"exact_{observable}" for observable in OBSERVABLES),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    paired = frame[frame["mode"] == "paired"].copy()
    central = paired[paired["record_type"] == "central"].copy()
    central["L"] = central["L"].astype(int)
    central = central.sort_values("L")
    lengths = central["L"].tolist()
    if tuple(lengths) != EXPECTED_LENGTHS:
        raise ValueError(
            f"expected one paired central row for sizes {EXPECTED_LENGTHS}; got {lengths}"
        )
    if central["L"].duplicated().any():
        raise ValueError("paired central rows contain duplicate sizes")

    exact_intercepts = np.empty(len(OBSERVABLES), dtype=float)
    for observable_index, observable in enumerate(OBSERVABLES):
        exact_values = central[f"exact_{observable}"].to_numpy(dtype=float)
        if not np.allclose(exact_values, exact_values[0], rtol=0.0, atol=1.0e-13):
            raise ValueError(f"exact limit for {observable} varies across sizes")
        exact_intercepts[observable_index] = exact_values[0]

    central_matrix = central[list(OBSERVABLES)].to_numpy(dtype=float)
    bootstrap_groups: list[np.ndarray] = []
    covariance_blocks: list[np.ndarray] = []
    reference_draws: np.ndarray | None = None
    for length in lengths:
        group = paired[
            (paired["record_type"] == "bootstrap")
            & (paired["L"] == float(length))
        ].copy()
        group = group.sort_values("draw")
        draws = group["draw"].to_numpy(dtype=int)
        if len(draws) < 2 or not np.array_equal(draws, np.arange(len(draws))):
            raise ValueError(f"L={length} bootstrap draw indices are incomplete")
        if reference_draws is None:
            reference_draws = draws
        elif not np.array_equal(draws, reference_draws):
            raise ValueError("bootstrap draw indices differ across lattice sizes")
        samples = group[list(OBSERVABLES)].to_numpy(dtype=float)
        if np.any(~np.isfinite(samples)):
            raise ValueError(f"L={length} bootstrap values are nonfinite")
        bootstrap_groups.append(samples)
        covariance_blocks.append(np.cov(samples, rowvar=False, ddof=1))
    assert reference_draws is not None
    bootstrap_cube = np.stack(bootstrap_groups, axis=1)
    noise_draws = bootstrap_cube - central_matrix[None, :, :]
    covariance = block_diag(*covariance_blocks)
    return {
        "lengths": lengths,
        "volumes": np.asarray(lengths, dtype=float) ** 2,
        "values": central_matrix.reshape(-1),
        "central_matrix": central_matrix,
        "noise_draws": noise_draws.reshape(len(reference_draws), -1),
        "covariance": covariance,
        "exact_intercepts": exact_intercepts,
        "bootstrap_draws": int(len(reference_draws)),
    }


def analyze_threshold(
    values: np.ndarray,
    noise_draws: np.ndarray,
    covariance: np.ndarray,
    exact_intercepts: np.ndarray,
    design: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Fit and calibrate both tests at one eigenmode threshold."""

    regularized = regularized_precision(covariance, threshold)
    whitener = np.asarray(regularized["whitener"])
    nested = nested_intercept_statistic(
        values, exact_intercepts, design, whitener
    )
    unrestricted = nested["unrestricted"]
    restricted = nested["restricted"]
    exact_vector = np.asarray(nested["exact_vector"])
    restricted_design, _ = restricted_quadratic_design(design)

    free_null = np.asarray(unrestricted["fitted"])
    _, bootstrap_global_q, unrestricted_rank = batch_gls_fit(
        free_null[None, :] + noise_draws,
        np.zeros_like(values),
        design,
        whitener,
    )
    restricted_null = np.asarray(restricted["fitted"])
    _, bootstrap_restricted_q, restricted_rank = batch_gls_fit(
        restricted_null[None, :] + noise_draws,
        exact_vector,
        restricted_design,
        whitener,
    )
    _, bootstrap_unrestricted_q, _ = batch_gls_fit(
        restricted_null[None, :] + noise_draws,
        np.zeros_like(values),
        design,
        whitener,
    )
    bootstrap_delta_q = bootstrap_restricted_q - bootstrap_unrestricted_q
    tolerance = 1.0e-10 * np.maximum(1.0, np.abs(bootstrap_restricted_q))
    if np.any(bootstrap_delta_q < -tolerance):
        raise ArithmeticError("bootstrap nested statistics are negative beyond roundoff")
    bootstrap_delta_q = np.maximum(bootstrap_delta_q, 0.0)

    parameter_draws, _, _ = batch_gls_fit(
        values[None, :] + noise_draws,
        np.zeros_like(values),
        design,
        whitener,
    )
    intercept_columns = np.arange(0, design.shape[1], 3)
    intercept_draws = parameter_draws[:, intercept_columns]
    intercept_covariance = np.cov(intercept_draws, rowvar=False, ddof=1)
    intercept_correlation = intercept_covariance / np.outer(
        np.sqrt(np.diag(intercept_covariance)),
        np.sqrt(np.diag(intercept_covariance)),
    )

    covariance_rank = int(regularized["rank"])
    global_dof = covariance_rank - unrestricted_rank
    restriction_count = unrestricted_rank - restricted_rank
    if global_dof <= 0:
        raise ValueError("no covariance degrees of freedom remain for lack of fit")
    if restriction_count != len(OBSERVABLES):
        raise ValueError("the nested test does not impose three estimable restrictions")
    observed_global_q = float(unrestricted["Q"])
    observed_delta_q = float(nested["delta_Q"])
    eigenvalues = np.asarray(regularized["eigenvalues"])
    maximum_eigenvalue = float(regularized["maximum_eigenvalue"])
    retained = eigenvalues > threshold * maximum_eigenvalue
    return {
        "regularized": regularized,
        "retained": retained,
        "unrestricted": unrestricted,
        "restricted": restricted,
        "parameter_draws": parameter_draws,
        "intercept_covariance": intercept_covariance,
        "intercept_correlation": intercept_correlation,
        "bootstrap_global_q": bootstrap_global_q,
        "bootstrap_delta_q": bootstrap_delta_q,
        "summary": {
            "relative_eigenvalue_threshold": threshold,
            "is_primary_threshold": threshold == PRIMARY_COVARIANCE_THRESHOLD,
            "observation_count": len(values),
            "covariance_retained_rank": covariance_rank,
            "covariance_condition_number_retained": regularized[
                "condition_number_retained"
            ],
            "smallest_retained_correlation_eigenvalue": regularized[
                "minimum_retained_eigenvalue"
            ],
            "largest_correlation_eigenvalue": maximum_eigenvalue,
            "unrestricted_parameter_count": design.shape[1],
            "unrestricted_whitened_design_rank": unrestricted_rank,
            "global_lack_of_fit_dof": global_dof,
            "global_lack_of_fit_Q": observed_global_q,
            "global_lack_of_fit_chi2_p_value": float(
                chi2.sf(observed_global_q, global_dof)
            ),
            "global_lack_of_fit_bootstrap_p_value": _calibrated_tail_probability(
                observed_global_q, bootstrap_global_q
            ),
            "restricted_nuisance_parameter_count": restricted_design.shape[1],
            "restricted_whitened_design_rank": restricted_rank,
            "intercept_restriction_count": restriction_count,
            "intercept_test_dof": restriction_count,
            "restricted_Q_exact_intercepts": float(restricted["Q"]),
            "unrestricted_Q_free_intercepts": observed_global_q,
            "intercept_nested_delta_Q": observed_delta_q,
            "intercept_test_chi2_p_value": float(
                chi2.sf(observed_delta_q, restriction_count)
            ),
            "intercept_test_bootstrap_p_value": _calibrated_tail_probability(
                observed_delta_q, bootstrap_delta_q
            ),
            "bootstrap_draws": len(noise_draws),
            "minimum_reportable_bootstrap_p_value": 1.0 / (
                len(noise_draws) + 1
            ),
        },
    }


def parameter_records(
    threshold: float,
    result: dict[str, object],
    exact_intercepts: np.ndarray,
) -> list[dict[str, object]]:
    """Build physical-unit coefficient and bootstrap-uncertainty records."""

    coefficients = np.asarray(result["unrestricted"]["coefficients"])
    draws = np.asarray(result["parameter_draws"])
    records: list[dict[str, object]] = []
    scale_by_coefficient = {
        "A": 1.0,
        "B": float(REFERENCE_VOLUME),
        "C": float(REFERENCE_VOLUME**2),
    }
    for observable_index, observable in enumerate(OBSERVABLES):
        for local_index, coefficient_name in enumerate(("A", "B", "C")):
            column = 3 * observable_index + local_index
            scale = scale_by_coefficient[coefficient_name]
            physical_draws = scale * draws[:, column]
            physical_value = scale * coefficients[column]
            lower, upper = np.quantile(physical_draws, (0.025, 0.975))
            exact_limit = (
                float(exact_intercepts[observable_index])
                if coefficient_name == "A"
                else np.nan
            )
            records.append(
                {
                    "relative_eigenvalue_threshold": threshold,
                    "is_primary_threshold": threshold
                    == PRIMARY_COVARIANCE_THRESHOLD,
                    "observable": observable,
                    "coefficient": coefficient_name,
                    "parameter_column": column,
                    "scaled_fit_value": coefficients[column],
                    "physical_fit_value": physical_value,
                    "physical_bootstrap_standard_error": np.std(
                        physical_draws, ddof=1
                    ),
                    "physical_bootstrap_mean_minus_fit": np.mean(physical_draws)
                    - physical_value,
                    "physical_bootstrap_percentile_2p5": lower,
                    "physical_bootstrap_percentile_97p5": upper,
                    "exact_limit": exact_limit,
                    "difference_from_exact_limit": (
                        physical_value - exact_limit
                        if coefficient_name == "A"
                        else np.nan
                    ),
                    "reference_volume": REFERENCE_VOLUME,
                }
            )
    return records


def intercept_covariance_records(
    threshold: float, result: dict[str, object]
) -> list[dict[str, object]]:
    covariance = np.asarray(result["intercept_covariance"])
    correlation = np.asarray(result["intercept_correlation"])
    return [
        {
            "relative_eigenvalue_threshold": threshold,
            "is_primary_threshold": threshold == PRIMARY_COVARIANCE_THRESHOLD,
            "row_observable": row_observable,
            "column_observable": column_observable,
            "bootstrap_covariance": covariance[row_index, column_index],
            "bootstrap_correlation": correlation[row_index, column_index],
        }
        for row_index, row_observable in enumerate(OBSERVABLES)
        for column_index, column_observable in enumerate(OBSERVABLES)
    ]


def eigenmode_records(
    threshold: float, result: dict[str, object]
) -> list[dict[str, object]]:
    eigenvalues = np.asarray(result["regularized"]["eigenvalues"])
    retained = np.asarray(result["retained"], dtype=bool)
    maximum = float(np.max(eigenvalues))
    order = np.argsort(eigenvalues)[::-1]
    return [
        {
            "relative_eigenvalue_threshold": threshold,
            "descending_mode_index": mode_index,
            "correlation_eigenvalue": eigenvalues[eigen_index],
            "eigenvalue_over_maximum": eigenvalues[eigen_index] / maximum,
            "retained": bool(retained[eigen_index]),
        }
        for mode_index, eigen_index in enumerate(order, start=1)
    ]


def bootstrap_records(
    threshold: float, result: dict[str, object]
) -> list[dict[str, object]]:
    global_q = np.asarray(result["bootstrap_global_q"])
    delta_q = np.asarray(result["bootstrap_delta_q"])
    return [
        {
            "relative_eigenvalue_threshold": threshold,
            "draw": draw,
            "global_lack_of_fit_Q_under_fitted_free_null": global_q[draw],
            "intercept_nested_delta_Q_under_exact_intercept_null": delta_q[draw],
        }
        for draw in range(len(global_q))
    ]


def analyze(input_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Run all declared eigenmode thresholds and assemble output tables."""

    loaded = load_paired_vectors(input_path)
    volumes = np.asarray(loaded["volumes"])
    values = np.asarray(loaded["values"])
    noise_draws = np.asarray(loaded["noise_draws"])
    covariance = np.asarray(loaded["covariance"])
    exact_intercepts = np.asarray(loaded["exact_intercepts"])
    design, parameter_names = joint_quadratic_design(volumes)

    summaries: list[dict[str, object]] = []
    parameters: list[dict[str, object]] = []
    intercept_covariances: list[dict[str, object]] = []
    eigenmodes: list[dict[str, object]] = []
    bootstrap: list[dict[str, object]] = []
    for threshold in EIGENMODE_SENSITIVITY_THRESHOLDS:
        result = analyze_threshold(
            values,
            noise_draws,
            covariance,
            exact_intercepts,
            design,
            threshold,
        )
        summaries.append(result["summary"])
        parameters.extend(parameter_records(threshold, result, exact_intercepts))
        intercept_covariances.extend(
            intercept_covariance_records(threshold, result)
        )
        eigenmodes.extend(eigenmode_records(threshold, result))
        bootstrap.extend(bootstrap_records(threshold, result))

    tables = {
        "summary": pd.DataFrame(summaries),
        "parameters": pd.DataFrame(parameters),
        "intercept_covariance": pd.DataFrame(intercept_covariances),
        "eigenmodes": pd.DataFrame(eigenmodes),
        "bootstrap": pd.DataFrame(bootstrap),
    }
    analysis_details = {
        "lengths": loaded["lengths"],
        "volumes": volumes.astype(int).tolist(),
        "bootstrap_draws": loaded["bootstrap_draws"],
        "exact_intercepts": dict(zip(OBSERVABLES, exact_intercepts.tolist())),
        "parameter_names_scaled": list(parameter_names),
    }
    return tables, analysis_details


def write_outputs(
    input_path: Path,
    output_root: Path,
    tables: dict[str, pd.DataFrame],
    analysis_details: dict[str, object],
) -> dict[str, Path]:
    """Write processed tables and complete, hash-addressed metadata."""

    data_directory = output_root / "data_processed"
    generated_directory = output_root / "generated"
    data_directory.mkdir(parents=True, exist_ok=True)
    generated_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary": data_directory
        / "lattice_free_intercept_joint_quadratic_summary.csv",
        "parameters": data_directory
        / "lattice_free_intercept_joint_quadratic_parameters.csv",
        "intercept_covariance": data_directory
        / "lattice_free_intercept_joint_quadratic_intercept_covariance.csv",
        "eigenmodes": data_directory
        / "lattice_free_intercept_joint_quadratic_eigenmodes.csv",
        "bootstrap": data_directory
        / "lattice_free_intercept_joint_quadratic_bootstrap.csv",
    }
    for name, path in output_paths.items():
        tables[name].to_csv(path, index=False)

    script_path = Path(__file__).resolve()
    upstream_script = ROOT / "scripts" / "lattice_validation_statistics.py"
    upstream_metadata = ROOT / "generated" / "lattice_validation_statistics_metadata.json"
    metadata = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_free_intercept_joint_quadratic.py",
        "script_sha256": sha256(script_path),
        "dependency_sha256": {
            "scripts/lattice_validation_statistics.py": sha256(upstream_script),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_version,
        "input_sha256": {
            str(input_path.resolve().relative_to(ROOT)): sha256(input_path),
            str(upstream_metadata.resolve().relative_to(ROOT)): sha256(
                upstream_metadata
            ),
        },
        "output_sha256": {
            str(path.resolve().relative_to(output_root)): sha256(path)
            for path in output_paths.values()
        },
        "data": {
            **analysis_details,
            "mode": "paired_antithetic",
            "observable_order_within_size": list(OBSERVABLES),
            "joint_observation_order": (
                "increasing L, then peak_center_h, peak_magnitude, fwhm_h"
            ),
            "observation_count": len(EXPECTED_LENGTHS) * len(OBSERVABLES),
            "independence_structure": (
                "independent simulations at different lattice sizes; full "
                "three-observable bootstrap covariance retained within each size"
            ),
        },
        "model": {
            "physical_form": "X_L = A_X + B_X/V + C_X/V^2",
            "fitted_form": "X_L = A_X + b_X*u + c_X*u^2, u=V_ref/V",
            "reference_volume": REFERENCE_VOLUME,
            "physical_conversion": {
                "A_X": "A_X",
                "B_X": "b_X * V_ref",
                "C_X": "c_X * V_ref^2",
            },
            "unrestricted_parameter_count": 9,
            "restricted_parameter_count": 6,
            "restricted_null": (
                "the three A_X are fixed jointly to the exact limiting vector; "
                "all three B_X and C_X remain free"
            ),
            "intercept_restriction_count": 3,
        },
        "covariance_regularization": {
            "method": (
                "eigendecomposition of the 15x15 correlation matrix; retain "
                "lambda > relative_threshold * lambda_max and evaluate GLS in "
                "the retained whitened modes"
            ),
            "sensitivity_thresholds": list(EIGENMODE_SENSITIVITY_THRESHOLDS),
            "primary_relative_threshold": PRIMARY_COVARIANCE_THRESHOLD,
            "stress_threshold": {
                "relative_threshold": 1.0e-4,
                "purpose": (
                    "conservative sensitivity check that truncates the weakest "
                    "correlation eigenmode; it is not used for the primary result"
                ),
            },
            "covariance_construction": (
                "block diagonal across lattice sizes from the sample covariance "
                "of the stored three-observable stratified block-bootstrap vectors"
            ),
        },
        "bootstrap_calibration": {
            "source": (
                "stored upstream stratified time-block "
                "bootstrap vectors"
            ),
            "centered_noise_definition": "bootstrap vector minus central vector",
            "global_lack_of_fit": (
                "add each centered noise vector to the fitted nine-parameter free "
                "quadratic null, refit all nine parameters, and recompute Q"
            ),
            "joint_intercept_test": (
                "add each centered noise vector to the fitted exact-intercept "
                "quadratic null, refit both six-parameter restricted and "
                "nine-parameter unrestricted models, and recompute Delta Q"
            ),
            "test_statistic": (
                "Delta Q = Q_restricted(A=A_exact) - Q_unrestricted(A free)"
            ),
            "tail_probability": "(1 + count[T_boot >= T_obs])/(B + 1)",
            "parameter_uncertainty": (
                "refit the unrestricted model to central vector plus each stored "
                "centered bootstrap noise vector"
            ),
            "draw_count": analysis_details["bootstrap_draws"],
            "covariance_treatment": (
                "the covariance estimate and its retained eigenmodes are held "
                "fixed while every bootstrap null replicate is refitted"
            ),
        },
        "degrees_of_freedom": {
            "global_lack_of_fit": (
                "retained covariance rank minus unrestricted whitened-design rank"
            ),
            "joint_intercept_test": (
                "unrestricted whitened-design rank minus restricted "
                "whitened-design rank; three when all intercept restrictions are "
                "estimable"
            ),
            "asymptotic_chi_square_p_values": (
                "reported as secondary diagnostics; bootstrap-calibrated p values "
                "are primary"
            ),
        },
        "reproduce_command": (
            "python3 scripts/lattice_free_intercept_joint_quadratic.py"
        ),
        "limitations": [
            "The test conditions on the upstream scalar-extraction and bootstrap procedure.",
            "The finite-size ansatz is quadratic in 1/V over the five simulated sizes; the global lack-of-fit test checks this ansatz against the retained covariance modes.",
            "The bootstrap calibrates finite-sample noise but does not replace an analytic proof of the thermodynamic limit.",
        ],
    }
    metadata_path = (
        generated_directory / "lattice_free_intercept_joint_quadratic_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_paths["metadata"] = metadata_path
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    input_path = arguments.input.resolve()
    output_root = arguments.output_root.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    tables, details = analyze(input_path)
    output_paths = write_outputs(input_path, output_root, tables, details)
    print("Free-intercept joint quadratic lattice test")
    print(tables["summary"].to_string(index=False))
    primary_parameters = tables["parameters"][
        tables["parameters"]["is_primary_threshold"]
        & (tables["parameters"]["coefficient"] == "A")
    ]
    print("\nPrimary-threshold intercept estimates")
    print(
        primary_parameters[
            [
                "observable",
                "physical_fit_value",
                "physical_bootstrap_standard_error",
                "exact_limit",
                "difference_from_exact_limit",
            ]
        ].to_string(index=False)
    )
    print("\nWrote")
    for path in output_paths.values():
        print(path)


if __name__ == "__main__":
    main()
