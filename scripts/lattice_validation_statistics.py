#!/usr/bin/env python3
"""Independent validation statistics for the lattice-gas coexistence pulse.

This module deliberately lives beside, rather than inside, the production
analysis.  It provides four publication-facing checks:

1. exact-limit-fixed ``X_L = X_inf + B/V`` fits with both chi-square and
   centered stratified block-bootstrap calibration;
2. a joint three-observable compatibility test retaining every scalar
   bootstrap vector and its covariance;
3. complete paired and raw-unpaired pulse sequences at every lattice size;
4. deterministic three-fold, reserved-replica curve tests at L=48 and 64.

The reserved-replica analysis distinguishes three predictions.  ``exact_crossfit`` uses
the exact finite-volume response decomposition evaluated with phase primitives
estimated only from the training replicas.  ``leading_logistic`` uses only
the training-replica h=0 density gap, cavity-work difference, phase weight,
and intraphase responses.  ``first_corrected_logistic`` projects training
phase primitives onto the independently derived first-order finite-volume
curve and predicts reserved replicas without using their data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.linalg import block_diag
from scipy.special import expit, logit
from scipy.stats import chi2

try:
    from .lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        load_trajectory_archives,
        replicas_for_size,
        stratified_block_indices,
    )
    from .lattice_gas_phase_switch import (
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        observables_from_contributions,
    )
    from .lattice_pulse_scaling import (
        analytic_pulse_limits,
        continuous_pulse_observables,
    )
    from .derive_pulse_finite_size_corrections import (
        PulseParameters,
        first_order_response,
    )
except ImportError:
    from lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        load_trajectory_archives,
        replicas_for_size,
        stratified_block_indices,
    )
    from lattice_gas_phase_switch import (
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        observables_from_contributions,
    )
    from lattice_pulse_scaling import (
        analytic_pulse_limits,
        continuous_pulse_observables,
    )
    from derive_pulse_finite_size_corrections import (
        PulseParameters,
        first_order_response,
    )


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RAW = ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz"
ADDITIONAL_RAW = ROOT / "data_raw" / "lattice_gas_additional_replicas.npz"
DEFAULT_RAW = [BASELINE_RAW, ADDITIONAL_RAW]
DEFAULT_SEED = 26_080_607
DEFAULT_BLOCKS = 40
DEFAULT_SCALAR_DRAWS = 2_000
DEFAULT_HOLDOUT_DRAWS = 1_000

# Chosen from the exact pulse before examining finite-size residuals.  The
# exact half-height crossings are approximately -0.423 and 5.141.
H_GRID = np.linspace(-1.0, 6.0, 141)
OBSERVABLES = ("peak_center_h", "peak_magnitude", "fwhm_h")
CURVE_TARGETS = ("total", "reweighting")
PREDICTION_KINDS = (
    "exact_crossfit",
    "leading_logistic",
    "first_corrected_logistic",
)
JOINT_FIT_MODELS = (
    "linear_all",
    "quadratic_all",
    "linear_drop_L16",
)
COVARIANCE_THRESHOLDS = (1.0e-6, 1.0e-8, 1.0e-10)
PRIMARY_COVARIANCE_THRESHOLD = 1.0e-8
REFERENCE_VOLUME = 16**2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_exact_linear_fit(
    inverse_volume: np.ndarray,
    values: np.ndarray,
    standard_errors: np.ndarray,
    exact_limit: float,
) -> dict[str, float | int | np.ndarray]:
    """Fit ``values = exact_limit + B * inverse_volume`` by WLS."""

    inverse_volume = np.asarray(inverse_volume, dtype=float)
    values = np.asarray(values, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    if not (
        inverse_volume.ndim == values.ndim == standard_errors.ndim == 1
        and len(inverse_volume) == len(values) == len(standard_errors)
    ):
        raise ValueError("fit inputs must be one-dimensional peers")
    if len(values) < 2 or np.any(standard_errors <= 0.0):
        raise ValueError("fit requires two points and positive standard errors")
    precision = 1.0 / standard_errors**2
    denominator = float(np.sum(precision * inverse_volume**2))
    if denominator <= 0.0:
        raise ValueError("inverse-volume design is singular")
    slope = float(
        np.sum(
            precision * inverse_volume * (values - float(exact_limit))
        )
        / denominator
    )
    se_slope = float(np.sqrt(1.0 / denominator))
    fitted = float(exact_limit) + slope * inverse_volume
    pulls = (values - fitted) / standard_errors
    statistic = float(np.dot(pulls, pulls))
    dof = int(len(values) - 1)
    return {
        "B": slope,
        "se_B": se_slope,
        "chi2": statistic,
        "dof": dof,
        "chi2_p_value": float(chi2.sf(statistic, dof)),
        "max_abs_standardized_residual": float(np.max(np.abs(pulls))),
        "fitted": fitted,
        "standardized_residuals": pulls,
    }


def regularized_precision(
    covariance: np.ndarray, relative_threshold: float
) -> dict[str, np.ndarray | float | int]:
    """Return a scale-invariant truncated-eigenmode covariance inverse.

    Regularization is applied to the correlation matrix, not the raw
    covariance, so a change of observable units cannot alter the retained
    eigenmodes.  The returned whitener satisfies ``Q=||W residual||^2``.
    """

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if relative_threshold <= 0.0:
        raise ValueError("relative threshold must be positive")
    covariance = 0.5 * (covariance + covariance.T)
    scales = np.sqrt(np.diag(covariance))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("covariance diagonal must be finite and positive")
    correlation = covariance / np.outer(scales, scales)
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    maximum = float(np.max(eigenvalues))
    keep = eigenvalues > relative_threshold * maximum
    if not np.any(keep):
        raise ValueError("regularization removed every covariance mode")
    retained_values = eigenvalues[keep]
    retained_vectors = eigenvectors[:, keep]
    whitener = (
        retained_vectors.T / np.sqrt(retained_values)[:, None]
    ) / scales[None, :]
    precision = whitener.T @ whitener
    return {
        "precision": precision,
        "whitener": whitener,
        "rank": int(np.count_nonzero(keep)),
        "minimum_retained_eigenvalue": float(np.min(retained_values)),
        "maximum_eigenvalue": maximum,
        "condition_number_retained": float(
            maximum / np.min(retained_values)
        ),
        "correlation": correlation,
        "eigenvalues": eigenvalues,
        "retained_eigenvalues": retained_values,
        "retained_eigenvectors": retained_vectors,
    }


def quadratic_from_whitener(
    vectors: np.ndarray, whitener: np.ndarray
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    whitener = np.asarray(whitener, dtype=float)
    transformed = vectors @ whitener.T
    return np.sum(transformed**2, axis=-1)


def deterministic_three_folds(
    replicas: list[int] | np.ndarray,
) -> list[tuple[list[int], list[int]]]:
    """Return three four-replica holdouts and their eight-replica complements."""

    ordered = sorted(int(value) for value in replicas)
    if len(ordered) != 12 or len(set(ordered)) != 12:
        raise ValueError("three-fold validation requires 12 distinct replicas")
    folds: list[tuple[list[int], list[int]]] = []
    for start in (0, 4, 8):
        heldout = ordered[start : start + 4]
        training = [replica for replica in ordered if replica not in heldout]
        folds.append((training, heldout))
    return folds


def finite_volume_reconstruction(
    observed: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reconstruct total and reweighting responses from exact phase primitives."""

    q0 = np.asarray(observed["q0_liquid"], dtype=float)
    qc = np.asarray(observed["qC_liquid"], dtype=float)
    gap = np.asarray(observed["meanN0_liquid"], dtype=float) - np.asarray(
        observed["meanN0_gas"], dtype=float
    )
    reweighting = (qc - q0) * gap
    intraphase = qc * np.asarray(
        observed["deltaN_liquid"], dtype=float
    ) + (1.0 - qc) * np.asarray(observed["deltaN_gas"], dtype=float)
    total = intraphase + reweighting
    direct = np.asarray(observed["deltaN"], dtype=float)
    direct_reweighting = np.asarray(
        observed["deltaN_phase_reweighting"], dtype=float
    )
    return {
        "total": total,
        "reweighting": reweighting,
        "intraphase": intraphase,
        "total_closure": direct - total,
        "reweighting_closure": direct_reweighting - reweighting,
    }


def leading_logistic_prediction(
    h_grid: np.ndarray,
    training_observed: dict[str, np.ndarray],
    volume: int,
) -> dict[str, np.ndarray]:
    """Leading two-state curve from training-replica primitives at h=0."""

    h_grid = np.asarray(h_grid, dtype=float)
    zero_index = int(np.argmin(np.abs(h_grid)))
    if abs(h_grid[zero_index]) > 1.0e-12:
        raise ValueError("the leading prediction requires h=0 on the grid")
    q0_zero = float(training_observed["q0_liquid"][zero_index])
    if not 0.0 < q0_zero < 1.0:
        raise ValueError("training phase weight must be strictly between zero and one")
    delta_rho = float(training_observed["delta_rho"][zero_index])
    delta_beta_w = float(training_observed["delta_betaW"][zero_index])
    phase_gap = float(
        training_observed["meanN0_liquid"][zero_index]
        - training_observed["meanN0_gas"][zero_index]
    )
    if abs(phase_gap / volume - delta_rho) > 1.0e-12:
        raise AssertionError("inconsistent training density gap")
    log_odds = float(logit(q0_zero)) + delta_rho * h_grid
    q0 = expit(log_odds)
    qc = expit(log_odds - delta_beta_w)
    reweighting = phase_gap * (qc - q0)
    liquid_response = float(training_observed["deltaN_liquid"][zero_index])
    gas_response = float(training_observed["deltaN_gas"][zero_index])
    intraphase = qc * liquid_response + (1.0 - qc) * gas_response
    return {
        "total": reweighting + intraphase,
        "reweighting": reweighting,
        "intraphase": intraphase,
    }


def first_corrected_logistic_prediction(
    h_grid: np.ndarray,
    training_observed: dict[str, np.ndarray],
    volume: int,
) -> dict[str, np.ndarray | float]:
    """Predict the derived first-corrected pulse from training data only.

    The exact thermodynamic ``delta_rho`` and ``a0`` are fixed inputs.  The
    four finite-volume coefficients are estimated by least squares from the
    training phase primitives over the specified field interval:

    ``D_V - V delta_rho = d0 + dchi h`` and
    ``V(a_V-a0) = a1 - dn h``.

    The total-response prediction adds the leading intraphase term, which is
    already of order ``1/V`` after scaling the response by volume.
    """

    h_grid = np.asarray(h_grid, dtype=float)
    exact_delta_rho = exact_spontaneous_magnetization(K_ISING)
    exact_a0 = exact_pair_cavity_parameters(K_ISING)[2]
    phase_gap = np.asarray(training_observed["meanN0_liquid"]) - np.asarray(
        training_observed["meanN0_gas"]
    )
    delta_beta_w = np.asarray(training_observed["delta_betaW"])
    design = np.column_stack((np.ones_like(h_grid), h_grid))
    d0, dchi = np.linalg.lstsq(
        design,
        phase_gap - volume * exact_delta_rho,
        rcond=None,
    )[0]
    a1, negative_dn = np.linalg.lstsq(
        design,
        volume * (delta_beta_w - exact_a0),
        rcond=None,
    )[0]
    dn = -negative_dn
    parameters = PulseParameters(
        delta_rho=float(exact_delta_rho),
        d0=float(d0),
        dchi=float(dchi),
        a0=float(exact_a0),
        a1=float(a1),
        dn=float(dn),
    )
    reweighting_scaled = np.asarray(
        [
            first_order_response(float(h), float(volume), parameters)
            for h in h_grid
        ]
    )
    zero_index = int(np.argmin(np.abs(h_grid)))
    if abs(h_grid[zero_index]) > 1.0e-12:
        raise ValueError("the corrected prediction requires h=0 on the grid")
    qc_leading = expit(exact_delta_rho * h_grid - exact_a0)
    liquid_response = float(training_observed["deltaN_liquid"][zero_index])
    gas_response = float(training_observed["deltaN_gas"][zero_index])
    intraphase_scaled = (
        qc_leading * liquid_response
        + (1.0 - qc_leading) * gas_response
    ) / volume
    return {
        "total": reweighting_scaled + intraphase_scaled,
        "reweighting": reweighting_scaled,
        "intraphase": intraphase_scaled,
        "d0": float(d0),
        "dchi": float(dchi),
        "a1": float(a1),
        "dn": float(dn),
    }


def _pulse_vector(contributions: np.ndarray, volume: int) -> np.ndarray:
    observed = observables_from_contributions(contributions, volume)
    scalar = continuous_pulse_observables(
        H_GRID, observed["deltaN_phase_reweighting"] / volume
    )
    return np.asarray([scalar[name] for name in OBSERVABLES], dtype=float)


def _hierarchical_totals_common_modes(
    arrays: dict[str, np.ndarray],
    draws: int,
    rng: np.random.Generator,
):
    modes = tuple(arrays)
    replica_count, block_count = arrays[modes[0]].shape[:2]
    for mode in modes[1:]:
        if arrays[mode].shape[:2] != (replica_count, block_count):
            raise ValueError("paired and unpaired block hierarchies differ")
    for _ in range(draws):
        indices = stratified_block_indices(replica_count, block_count, rng)
        totals = {
            mode: np.zeros(arrays[mode].shape[2:], dtype=float)
            for mode in modes
        }
        for replica, selected_blocks in enumerate(indices):
            for mode in modes:
                totals[mode] += np.sum(
                    arrays[mode][replica, selected_blocks], axis=0
                )
        yield totals


def _subset_total(blocks: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.sum(blocks[np.asarray(indices, dtype=int)], axis=(0, 1))


def _resample_subset_total(
    blocks: np.ndarray,
    indices: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    replica_count = len(indices)
    block_count = blocks.shape[1]
    total = np.zeros(blocks.shape[2:], dtype=float)
    block_indices = stratified_block_indices(replica_count, block_count, rng)
    for replica_index, selected_blocks in zip(indices, block_indices):
        total += np.sum(blocks[replica_index, selected_blocks], axis=0)
    return total


def _calibrated_tail_probability(
    observed_statistic: float, bootstrap_statistics: np.ndarray
) -> float:
    return float(
        (1 + np.count_nonzero(bootstrap_statistics >= observed_statistic))
        / (len(bootstrap_statistics) + 1)
    )


def _gls_fixed_exact_fit(
    values: np.ndarray,
    exact_vector: np.ndarray,
    design: np.ndarray,
    whitener: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit an exact-intercept GLS model in retained covariance modes."""

    whitened_design = whitener @ design
    whitened_values = whitener @ (values - exact_vector)
    coefficients, _, design_rank, _ = np.linalg.lstsq(
        whitened_design, whitened_values, rcond=None
    )
    if design_rank != design.shape[1]:
        raise ValueError("joint fixed-limit design is rank deficient")
    residual = values - exact_vector - design @ coefficients
    statistic = float(np.sum((whitener @ residual) ** 2))
    return coefficients, residual, statistic


def _joint_model_design(
    fit_model: str, lengths: list[int]
) -> tuple[list[int], np.ndarray, tuple[int, ...]]:
    """Build a stably scaled, observable-specific joint design."""

    if fit_model == "linear_all":
        selected_lengths = list(lengths)
        powers = (1,)
    elif fit_model == "quadratic_all":
        selected_lengths = list(lengths)
        powers = (1, 2)
    elif fit_model == "linear_drop_L16":
        selected_lengths = [length for length in lengths if length >= 24]
        powers = (1,)
    else:
        raise ValueError(f"unknown joint fit model: {fit_model}")
    rows = len(selected_lengths) * len(OBSERVABLES)
    columns = len(powers) * len(OBSERVABLES)
    design = np.zeros((rows, columns), dtype=float)
    for length_index, length in enumerate(selected_lengths):
        scaled_inverse_volume = REFERENCE_VOLUME / (length * length)
        for power_index, power in enumerate(powers):
            for observable_index in range(len(OBSERVABLES)):
                design[
                    length_index * len(OBSERVABLES) + observable_index,
                    power_index * len(OBSERVABLES) + observable_index,
                ] = scaled_inverse_volume**power
    return selected_lengths, design, powers


def _dominant_residual_diagnostics(
    residual: np.ndarray,
    covariance: np.ndarray,
    regularized: dict[str, np.ndarray | float | int],
    selected_lengths: list[int],
) -> dict[str, float | int | str]:
    pointwise = residual / np.sqrt(np.diag(covariance))
    pointwise_index = int(np.argmax(np.abs(pointwise)))
    pointwise_length = selected_lengths[pointwise_index // len(OBSERVABLES)]
    pointwise_observable = OBSERVABLES[pointwise_index % len(OBSERVABLES)]

    whitener = np.asarray(regularized["whitener"])
    whitened_residual = whitener @ residual
    mode_index = int(np.argmax(np.abs(whitened_residual)))
    eigenvectors = np.asarray(regularized["retained_eigenvectors"])
    eigenvalues = np.asarray(regularized["retained_eigenvalues"])
    loadings = eigenvectors[:, mode_index]
    labels = [
        f"L{length}:{observable}"
        for length in selected_lengths
        for observable in OBSERVABLES
    ]
    largest_loadings = np.argsort(np.abs(loadings))[-3:][::-1]
    loading_summary = "; ".join(
        f"{labels[index]}={loadings[index]:+.3f}"
        for index in largest_loadings
    )
    return {
        "max_abs_pointwise_standardized_residual": float(
            np.max(np.abs(pointwise))
        ),
        "largest_pointwise_residual_L": pointwise_length,
        "largest_pointwise_residual_observable": pointwise_observable,
        "max_abs_whitened_mode_residual": float(
            np.max(np.abs(whitened_residual))
        ),
        "dominant_mode_retained_eigenvalue": float(eigenvalues[mode_index]),
        "dominant_mode_top_standardized_loadings": loading_summary,
    }


def run_scalar_validation(
    trajectories: dict[tuple[int, int], dict[str, np.ndarray]],
    block_count: int,
    bootstrap_draws: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[int, dict[str, np.ndarray | list[int] | int]]],
    dict[int, np.ndarray],
]:
    """Run paired/unpaired scalar sequences and exact-limit validation."""

    exact_magnetization = exact_spontaneous_magnetization(K_ISING)
    exact_delta_w = exact_pair_cavity_parameters(K_ISING)[2]
    exact = analytic_pulse_limits(exact_magnetization, exact_delta_w)
    scalar_data: dict[
        str, dict[int, dict[str, np.ndarray | list[int] | int]]
    ] = {"paired": {}, "unpaired": {}}
    paired_blocks: dict[int, np.ndarray] = {}

    for length in available_sizes(trajectories):
        volume = length * length
        arrays: dict[str, np.ndarray] = {}
        replicas_reference: list[int] | None = None
        for mode in ("paired", "unpaired"):
            blocks, replicas = contribution_blocks(
                trajectories, length, H_GRID, block_count, mode=mode
            )
            if replicas_reference is None:
                replicas_reference = replicas
            elif replicas != replicas_reference:
                raise AssertionError("paired/unpaired replica identities differ")
            arrays[mode] = blocks
        assert replicas_reference is not None
        if replicas_reference != list(range(12)):
            raise ValueError(
                f"L={length} requires replica IDs 0--11; got {replicas_reference}"
            )
        paired_blocks[length] = arrays["paired"]
        central = {
            mode: _pulse_vector(np.sum(blocks, axis=(0, 1)), volume)
            for mode, blocks in arrays.items()
        }
        bootstraps = {
            mode: np.empty((bootstrap_draws, len(OBSERVABLES)), dtype=float)
            for mode in arrays
        }
        rng = np.random.default_rng(seed + 10_003 * length)
        for draw, totals in enumerate(
            _hierarchical_totals_common_modes(arrays, bootstrap_draws, rng)
        ):
            for mode, total in totals.items():
                bootstraps[mode][draw] = _pulse_vector(total, volume)
        for mode in arrays:
            scalar_data[mode][length] = {
                "central": central[mode],
                "bootstrap": bootstraps[mode],
                "standard_errors": np.std(bootstraps[mode], axis=0, ddof=1),
                "replicas": replicas_reference,
                "volume": volume,
            }

    lengths = sorted(scalar_data["paired"])
    inverse_volume = np.asarray([1.0 / (length * length) for length in lengths])
    fixed_rows: list[dict] = []
    for mode in ("paired", "unpaired"):
        for observable_index, observable in enumerate(OBSERVABLES):
            values = np.asarray(
                [scalar_data[mode][length]["central"][observable_index] for length in lengths]
            )
            errors = np.asarray(
                [
                    scalar_data[mode][length]["standard_errors"][observable_index]
                    for length in lengths
                ]
            )
            fit = fixed_exact_linear_fit(
                inverse_volume, values, errors, exact[observable]
            )
            model = np.asarray(fit["fitted"], dtype=float)
            bootstrap_statistics = np.empty(bootstrap_draws, dtype=float)
            for draw in range(bootstrap_draws):
                noise = np.asarray(
                    [
                        scalar_data[mode][length]["bootstrap"][draw, observable_index]
                        - scalar_data[mode][length]["central"][observable_index]
                        for length in lengths
                    ]
                )
                bootstrap_fit = fixed_exact_linear_fit(
                    inverse_volume, model + noise, errors, exact[observable]
                )
                bootstrap_statistics[draw] = float(bootstrap_fit["chi2"])
            calibrated = _calibrated_tail_probability(
                float(fit["chi2"]), bootstrap_statistics
            )
            row = {
                "mode": mode,
                "observable": observable,
                "exact_limit": exact[observable],
                "B": fit["B"],
                "se_B": fit["se_B"],
                "chi2": fit["chi2"],
                "dof": fit["dof"],
                "chi2_p_value": fit["chi2_p_value"],
                "centered_hierarchical_bootstrap_p_value": calibrated,
                "max_abs_standardized_residual": fit[
                    "max_abs_standardized_residual"
                ],
                "sizes": " ".join(str(length) for length in lengths),
                "replicas_per_size": 12,
                "blocks_per_replica": block_count,
                "bootstrap_draws": bootstrap_draws,
            }
            fixed_rows.append(row)

    covariance_rows: list[dict] = []
    joint_results: dict[str, dict[str, dict]] = {}
    for mode in ("paired", "unpaired"):
        covariance_by_length: dict[int, np.ndarray] = {}
        for length in lengths:
            covariance = np.cov(
                scalar_data[mode][length]["bootstrap"], rowvar=False, ddof=1
            )
            regularized = regularized_precision(
                covariance, PRIMARY_COVARIANCE_THRESHOLD
            )
            correlation = np.asarray(regularized["correlation"])
            covariance_by_length[length] = covariance
            for row_index, row_observable in enumerate(OBSERVABLES):
                for column_index, column_observable in enumerate(OBSERVABLES):
                    covariance_rows.append(
                        {
                            "mode": mode,
                            "L": length,
                            "V": length * length,
                            "row_observable": row_observable,
                            "column_observable": column_observable,
                            "covariance": covariance[row_index, column_index],
                            "correlation": correlation[row_index, column_index],
                            "relative_eigenvalue_threshold": PRIMARY_COVARIANCE_THRESHOLD,
                            "retained_rank": regularized["rank"],
                            "condition_number_retained": regularized[
                                "condition_number_retained"
                            ],
                        }
                    )
        joint_results[mode] = {}
        for fit_model in JOINT_FIT_MODELS:
            selected_lengths, design, powers = _joint_model_design(
                fit_model, lengths
            )
            covariance_joint = block_diag(
                *[covariance_by_length[length] for length in selected_lengths]
            )
            values = np.concatenate(
                [
                    scalar_data[mode][length]["central"]
                    for length in selected_lengths
                ]
            )
            exact_vector = np.tile(
                np.asarray([exact[name] for name in OBSERVABLES]),
                len(selected_lengths),
            )
            noise_draws = np.asarray(
                [
                    np.concatenate(
                        [
                            scalar_data[mode][length]["bootstrap"][draw]
                            - scalar_data[mode][length]["central"]
                            for length in selected_lengths
                        ]
                    )
                    for draw in range(bootstrap_draws)
                ]
            )
            threshold_results: dict[float, dict] = {}
            for threshold in COVARIANCE_THRESHOLDS:
                regularized_joint = regularized_precision(
                    covariance_joint, threshold
                )
                whitener = np.asarray(regularized_joint["whitener"])
                coefficients, residual, statistic = _gls_fixed_exact_fit(
                    values, exact_vector, design, whitener
                )
                model = exact_vector + design @ coefficients
                bootstrap_statistics = np.empty(bootstrap_draws, dtype=float)
                for draw in range(bootstrap_draws):
                    _, _, bootstrap_statistics[draw] = _gls_fixed_exact_fit(
                        model + noise_draws[draw],
                        exact_vector,
                        design,
                        whitener,
                    )
                rank = int(regularized_joint["rank"])
                parameter_count = int(design.shape[1])
                dof = rank - parameter_count
                if dof <= 0:
                    raise ValueError(
                        f"{fit_model} leaves no joint goodness-of-fit degrees of freedom"
                    )
                threshold_results[threshold] = {
                    "coefficients_scaled": coefficients,
                    "residual": residual,
                    "Q": statistic,
                    "rank": rank,
                    "parameter_count": parameter_count,
                    "dof": dof,
                    "chi2_p_value": float(chi2.sf(statistic, dof)),
                    "centered_bootstrap_p_value": _calibrated_tail_probability(
                        statistic, bootstrap_statistics
                    ),
                    "condition_number_retained": regularized_joint[
                        "condition_number_retained"
                    ],
                    "regularized": regularized_joint,
                }
            primary = threshold_results[PRIMARY_COVARIANCE_THRESHOLD]
            physical_coefficients = {
                power: np.asarray(primary["coefficients_scaled"])[
                    power_index
                    * len(OBSERVABLES) : (power_index + 1)
                    * len(OBSERVABLES)
                ]
                * REFERENCE_VOLUME**power
                for power_index, power in enumerate(powers)
            }
            diagnostics = _dominant_residual_diagnostics(
                np.asarray(primary["residual"]),
                covariance_joint,
                primary["regularized"],
                selected_lengths,
            )
            joint_results[mode][fit_model] = {
                **primary,
                **diagnostics,
                "fit_model": fit_model,
                "selected_lengths": selected_lengths,
                "powers": powers,
                "physical_coefficients": physical_coefficients,
                "relative_eigenvalue_threshold": PRIMARY_COVARIANCE_THRESHOLD,
                "threshold_results": threshold_results,
            }

    joint_rows: list[dict] = []
    for mode in ("paired", "unpaired"):
        joint = joint_results[mode]["linear_all"]
        for length in lengths:
            entry = scalar_data[mode][length]
            row = {
                "record_type": "central",
                "fit_model": "linear_all",
                "mode": mode,
                "L": length,
                "V": entry["volume"],
                "replicas": 12,
                "replica_ids": " ".join(str(value) for value in entry["replicas"]),
                "blocks_per_replica": block_count,
                "bootstrap_draws": bootstrap_draws,
                "draw": np.nan,
                "joint_Q": joint["Q"],
                "joint_rank": joint["rank"],
                "joint_dof": joint["dof"],
                "joint_chi2_p_value": joint["chi2_p_value"],
                "joint_centered_bootstrap_p_value": joint[
                    "centered_bootstrap_p_value"
                ],
                "joint_relative_eigenvalue_threshold": joint[
                    "relative_eigenvalue_threshold"
                ],
                "joint_parameter_count": joint["parameter_count"],
                "joint_condition_number_retained": joint[
                    "condition_number_retained"
                ],
            }
            for index, observable in enumerate(OBSERVABLES):
                row[observable] = entry["central"][index]
                row[f"se_{observable}"] = entry["standard_errors"][index]
                row[f"exact_{observable}"] = exact[observable]
                row[f"joint_B_{observable}"] = joint[
                    "physical_coefficients"
                ][1][index]
                row[f"joint_C_{observable}"] = np.nan
            joint_rows.append(row)
            for draw in range(bootstrap_draws):
                bootstrap_row = {
                    "record_type": "bootstrap",
                    "fit_model": "raw_bootstrap_vector",
                    "mode": mode,
                    "L": length,
                    "V": entry["volume"],
                    "replicas": 12,
                    "replica_ids": row["replica_ids"],
                    "blocks_per_replica": block_count,
                    "bootstrap_draws": bootstrap_draws,
                    "draw": draw,
                }
                for index, observable in enumerate(OBSERVABLES):
                    bootstrap_row[observable] = entry["bootstrap"][draw, index]
                joint_rows.append(bootstrap_row)

        replica_ids = " ".join(str(value) for value in range(12))
        for fit_model in JOINT_FIT_MODELS:
            result = joint_results[mode][fit_model]
            thresholds = result["threshold_results"]
            summary_row = {
                "record_type": "joint_fit",
                "fit_model": fit_model,
                "mode": mode,
                "L": np.nan,
                "V": np.nan,
                "replicas": 12,
                "replica_ids": replica_ids,
                "blocks_per_replica": block_count,
                "bootstrap_draws": bootstrap_draws,
                "draw": np.nan,
                "fit_sizes": " ".join(
                    str(length) for length in result["selected_lengths"]
                ),
                "joint_Q": result["Q"],
                "joint_rank": result["rank"],
                "joint_parameter_count": result["parameter_count"],
                "joint_dof": result["dof"],
                "joint_chi2_p_value": result["chi2_p_value"],
                "joint_centered_bootstrap_p_value": result[
                    "centered_bootstrap_p_value"
                ],
                "joint_model_passes_centered_0p05": (
                    result["centered_bootstrap_p_value"] >= 0.05
                ),
                "joint_relative_eigenvalue_threshold": result[
                    "relative_eigenvalue_threshold"
                ],
                "joint_condition_number_retained": result[
                    "condition_number_retained"
                ],
                "rank_threshold_1e-6": thresholds[1.0e-6]["rank"],
                "p_value_threshold_1e-6": thresholds[1.0e-6][
                    "centered_bootstrap_p_value"
                ],
                "rank_threshold_1e-8": thresholds[1.0e-8]["rank"],
                "p_value_threshold_1e-8": thresholds[1.0e-8][
                    "centered_bootstrap_p_value"
                ],
                "rank_threshold_1e-10": thresholds[1.0e-10]["rank"],
                "p_value_threshold_1e-10": thresholds[1.0e-10][
                    "centered_bootstrap_p_value"
                ],
                "max_abs_pointwise_standardized_residual": result[
                    "max_abs_pointwise_standardized_residual"
                ],
                "largest_pointwise_residual_L": result[
                    "largest_pointwise_residual_L"
                ],
                "largest_pointwise_residual_observable": result[
                    "largest_pointwise_residual_observable"
                ],
                "max_abs_whitened_mode_residual": result[
                    "max_abs_whitened_mode_residual"
                ],
                "dominant_mode_retained_eigenvalue": result[
                    "dominant_mode_retained_eigenvalue"
                ],
                "dominant_mode_top_standardized_loadings": result[
                    "dominant_mode_top_standardized_loadings"
                ],
            }
            for index, observable in enumerate(OBSERVABLES):
                summary_row[f"exact_{observable}"] = exact[observable]
                summary_row[f"joint_B_{observable}"] = result[
                    "physical_coefficients"
                ][1][index]
                summary_row[f"joint_C_{observable}"] = (
                    result["physical_coefficients"].get(
                        2, np.full(len(OBSERVABLES), np.nan)
                    )[index]
                )
            joint_rows.append(summary_row)

    return (
        pd.DataFrame(fixed_rows),
        pd.DataFrame(joint_rows),
        pd.DataFrame(covariance_rows),
        scalar_data,
        paired_blocks,
    )


def _prediction_curves(
    observed: dict[str, np.ndarray], volume: int
) -> dict[str, dict[str, np.ndarray | float]]:
    reconstruction = finite_volume_reconstruction(observed)
    leading = leading_logistic_prediction(H_GRID, observed, volume)
    first_corrected = first_corrected_logistic_prediction(
        H_GRID, observed, volume
    )
    return {
        "exact_crossfit": {
            "total": reconstruction["total"] / volume,
            "reweighting": reconstruction["reweighting"] / volume,
        },
        "leading_logistic": {
            "total": leading["total"] / volume,
            "reweighting": leading["reweighting"] / volume,
        },
        "first_corrected_logistic": first_corrected,
    }


def run_holdout_validation(
    trajectories: dict[tuple[int, int], dict[str, np.ndarray]],
    paired_blocks: dict[int, np.ndarray],
    bootstrap_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run deterministic three-fold cross-fitted curve validation."""

    rows: list[dict] = []
    plot_records: list[dict] = []
    for length in (48, 64):
        volume = length * length
        replicas = replicas_for_size(trajectories, length)
        folds = deterministic_three_folds(replicas)
        replica_to_index = {replica: index for index, replica in enumerate(replicas)}
        blocks = paired_blocks[length]
        for fold_index, (training_ids, heldout_ids) in enumerate(folds):
            training_indices = [replica_to_index[value] for value in training_ids]
            heldout_indices = [replica_to_index[value] for value in heldout_ids]
            training_total = _subset_total(blocks, training_indices)
            heldout_total = _subset_total(blocks, heldout_indices)
            training_observed = observables_from_contributions(
                training_total, volume
            )
            heldout_observed = observables_from_contributions(
                heldout_total, volume
            )
            reconstruction = finite_volume_reconstruction(training_observed)
            maximum_closure = float(
                max(
                    np.max(np.abs(reconstruction["total_closure"])),
                    np.max(np.abs(reconstruction["reweighting_closure"])),
                )
            )
            central_predictions = _prediction_curves(training_observed, volume)
            central_targets = {
                "total": heldout_observed["deltaN"] / volume,
                "reweighting": (
                    heldout_observed["deltaN_phase_reweighting"] / volume
                ),
            }
            residual_draws = {
                kind: {
                    target: np.empty((bootstrap_draws, len(H_GRID)), dtype=float)
                    for target in CURVE_TARGETS
                }
                for kind in PREDICTION_KINDS
            }
            rng = np.random.default_rng(
                seed + 100_003 * length + 10_007 * fold_index
            )
            for draw in range(bootstrap_draws):
                training_draw = _resample_subset_total(
                    blocks, training_indices, rng
                )
                heldout_draw = _resample_subset_total(
                    blocks, heldout_indices, rng
                )
                train_observed_draw = observables_from_contributions(
                    training_draw, volume
                )
                heldout_observed_draw = observables_from_contributions(
                    heldout_draw, volume
                )
                predictions_draw = _prediction_curves(
                    train_observed_draw, volume
                )
                targets_draw = {
                    "total": heldout_observed_draw["deltaN"] / volume,
                    "reweighting": (
                        heldout_observed_draw["deltaN_phase_reweighting"]
                        / volume
                    ),
                }
                for kind in PREDICTION_KINDS:
                    for target in CURVE_TARGETS:
                        residual_draws[kind][target][draw] = (
                            targets_draw[target] - predictions_draw[kind][target]
                        )

            for kind in PREDICTION_KINDS:
                for target in CURVE_TARGETS:
                    central_residual = (
                        central_targets[target] - central_predictions[kind][target]
                    )
                    draws = residual_draws[kind][target]
                    covariance = np.cov(draws, rowvar=False, ddof=1)
                    centered_noise = draws - np.mean(draws, axis=0)
                    threshold_results: dict[float, dict] = {}
                    for threshold in COVARIANCE_THRESHOLDS:
                        regularized = regularized_precision(covariance, threshold)
                        whitener = np.asarray(regularized["whitener"])
                        observed_q = float(
                            quadratic_from_whitener(
                                central_residual[None, :], whitener
                            )[0]
                        )
                        bootstrap_q = quadratic_from_whitener(
                            centered_noise, whitener
                        )
                        threshold_results[threshold] = {
                            "Q": observed_q,
                            "rank": regularized["rank"],
                            "p": _calibrated_tail_probability(
                                observed_q, bootstrap_q
                            ),
                            "condition": regularized[
                                "condition_number_retained"
                            ],
                        }
                    pointwise_se = np.std(draws, axis=0, ddof=1)
                    valid = pointwise_se > 0.0
                    standardized = np.zeros_like(central_residual)
                    standardized[valid] = central_residual[valid] / pointwise_se[valid]
                    centered_standardized = np.zeros_like(centered_noise)
                    centered_standardized[:, valid] = (
                        centered_noise[:, valid] / pointwise_se[valid]
                    )
                    bootstrap_maximum = np.max(
                        np.abs(centered_standardized[:, valid]), axis=1
                    )
                    observed_maximum = float(np.max(np.abs(standardized[valid])))
                    simultaneous_critical = float(
                        np.quantile(bootstrap_maximum, 0.95)
                    )
                    primary = threshold_results[PRIMARY_COVARIANCE_THRESHOLD]
                    row = {
                        "L": length,
                        "V": volume,
                        "fold": fold_index,
                        "mode": "paired_antithetic",
                        "training_replica_ids": " ".join(
                            str(value) for value in training_ids
                        ),
                        "heldout_replica_ids": " ".join(
                            str(value) for value in heldout_ids
                        ),
                        "training_replica_count": len(training_ids),
                        "heldout_replica_count": len(heldout_ids),
                        "prediction_kind": kind,
                        "prediction_scope": {
                            "exact_crossfit": (
                                "exact finite-volume decomposition evaluated "
                                "with training-replica phase primitives"
                            ),
                            "leading_logistic": (
                                "leading logistic approximation from training "
                                "h=0 primitives"
                            ),
                            "first_corrected_logistic": (
                                "derived O(1/V) curve with finite-volume "
                                "coefficients fitted only to training phase primitives"
                            ),
                        }[kind],
                        "target": target,
                        "h_min": float(H_GRID[0]),
                        "h_max": float(H_GRID[-1]),
                        "h_points": len(H_GRID),
                        "bootstrap_draws": bootstrap_draws,
                        "covariance_relative_threshold": PRIMARY_COVARIANCE_THRESHOLD,
                        "covariance_retained_rank": primary["rank"],
                        "covariance_condition_number_retained": primary["condition"],
                        "generalized_Q": primary["Q"],
                        "centered_hierarchical_bootstrap_p_value": primary["p"],
                        "rank_threshold_1e-6": threshold_results[1.0e-6]["rank"],
                        "p_value_threshold_1e-6": threshold_results[1.0e-6]["p"],
                        "rank_threshold_1e-8": threshold_results[1.0e-8]["rank"],
                        "p_value_threshold_1e-8": threshold_results[1.0e-8]["p"],
                        "rank_threshold_1e-10": threshold_results[1.0e-10]["rank"],
                        "p_value_threshold_1e-10": threshold_results[1.0e-10]["p"],
                        "rmse_response_over_V": float(
                            np.sqrt(np.mean(central_residual**2))
                        ),
                        "max_abs_response_over_V": float(
                            np.max(np.abs(central_residual))
                        ),
                        "max_abs_pointwise_standardized_residual": observed_maximum,
                        "simultaneous_95_multiplier": simultaneous_critical,
                        "inside_simultaneous_95_band": (
                            observed_maximum <= simultaneous_critical
                        ),
                        "maximum_training_decomposition_closure": maximum_closure,
                        "training_d0": central_predictions[kind].get(
                            "d0", np.nan
                        ),
                        "training_dchi": central_predictions[kind].get(
                            "dchi", np.nan
                        ),
                        "training_a1": central_predictions[kind].get(
                            "a1", np.nan
                        ),
                        "training_dn": central_predictions[kind].get(
                            "dn", np.nan
                        ),
                    }
                    rows.append(row)
                    plot_records.append(
                        {
                            "L": length,
                            "fold": fold_index,
                            "prediction_kind": kind,
                            "target": target,
                            "standardized_residual": standardized,
                        }
                    )
    return pd.DataFrame(rows), plot_records


def make_exact_limit_figure(
    fixed_fits: pd.DataFrame,
    joint_vectors: pd.DataFrame,
    output: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    titles = {
        "peak_center_h": "Peak center",
        "peak_magnitude": "Peak magnitude",
        "fwhm_h": "Pulse width",
    }
    central = joint_vectors[joint_vectors["record_type"] == "central"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
    markers = {"paired": "o", "unpaired": "s"}
    colors = {"paired": "#0072B2", "unpaired": "#D55E00"}
    for axis, observable in zip(axes, OBSERVABLES):
        for mode in ("paired", "unpaired"):
            selected = central[central["mode"] == mode].sort_values("L")
            fit = fixed_fits[
                (fixed_fits["mode"] == mode)
                & (fixed_fits["observable"] == observable)
            ].iloc[0]
            fitted = fit["exact_limit"] + fit["B"] / selected["V"].to_numpy()
            pulls = (
                selected[observable].to_numpy() - fitted
            ) / selected[f"se_{observable}"].to_numpy()
            axis.plot(
                1.0 / selected["V"],
                pulls,
                marker=markers[mode],
                color=colors[mode],
                linestyle="none",
                markersize=4.2,
                label=mode,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhspan(-2.0, 2.0, color="0.92", zorder=-1)
        axis.set_title(titles[observable])
        axis.set_xlabel(r"$1/V$")
        axis.set_ylabel("standardized residual")
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axis.grid(alpha=0.16, linewidth=0.5)
    axes[0].legend(frameon=False)
    for label, axis in zip("abc", axes):
        axis.text(
            -0.14,
            1.04,
            f"({label})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.4,
        )
    fig.savefig(output)
    plt.close(fig)


def make_holdout_figure(plot_records: list[dict], output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.3,
            "axes.labelsize": 8.9,
            "axes.titlesize": 9.0,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for row_index, length in enumerate((48, 64)):
        for column_index, target in enumerate(CURVE_TARGETS):
            axis = axes[row_index, column_index]
            for fold in range(3):
                for kind, linestyle in (
                    ("exact_crossfit", "-"),
                    ("first_corrected_logistic", "-."),
                    ("leading_logistic", "--"),
                ):
                    selected = [
                        record
                        for record in plot_records
                        if record["L"] == length
                        and record["fold"] == fold
                        and record["target"] == target
                        and record["prediction_kind"] == kind
                    ]
                    if len(selected) != 1:
                        raise AssertionError("missing unique holdout plot record")
                    axis.plot(
                        H_GRID,
                        selected[0]["standardized_residual"],
                        color=colors[fold],
                        linestyle=linestyle,
                        linewidth=1.0,
                    )
            axis.axhline(0.0, color="black", linewidth=0.75)
            axis.axhspan(-2.0, 2.0, color="0.93", zorder=-1)
            axis.set_title(
                rf"$L={length}$, "
                + ("total response" if target == "total" else "reweighting term")
            )
            axis.set_xlabel(r"$h=V(x-x_{\rm coex})$")
            axis.set_ylabel("pointwise standardized residual")
            axis.grid(alpha=0.15, linewidth=0.5)
    legend_handles = [
        Line2D([0], [0], color=color, linewidth=1.2, label=f"fold {index + 1}")
        for index, color in enumerate(colors)
    ] + [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.2,
            linestyle="-",
            label="finite-volume crossfit",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.2,
            linestyle="-.",
            label=r"corrected $1/V$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="leading",
        ),
    ]
    axes[0, 0].legend(
        handles=legend_handles,
        frameon=False,
        ncol=2,
        loc="lower left",
    )
    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.10,
            1.035,
            f"({label})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.3,
        )
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", default=DEFAULT_RAW)
    parser.add_argument("--blocks-per-replica", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--scalar-bootstrap-draws", type=int, default=DEFAULT_SCALAR_DRAWS
    )
    parser.add_argument(
        "--holdout-bootstrap-draws", type=int, default=DEFAULT_HOLDOUT_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_paths = [Path(path) for path in args.raw]
    for path in raw_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    trajectories = load_trajectory_archives(raw_paths)
    expected_sizes = [16, 24, 32, 48, 64]
    if available_sizes(trajectories) != expected_sizes:
        raise ValueError(
            f"expected sizes {expected_sizes}; got {available_sizes(trajectories)}"
        )
    for length in expected_sizes:
        replicas = replicas_for_size(trajectories, length)
        if replicas != list(range(12)):
            raise ValueError(f"L={length} does not contain replica IDs 0--11")

    fixed, joint, covariance, _, paired_blocks = run_scalar_validation(
        trajectories,
        args.blocks_per_replica,
        args.scalar_bootstrap_draws,
        args.seed,
    )
    holdout, plot_records = run_holdout_validation(
        trajectories,
        paired_blocks,
        args.holdout_bootstrap_draws,
        args.seed + 1_000_003,
    )

    data_directory = ROOT / "data_processed"
    figure_directory = ROOT / "figures"
    generated_directory = ROOT / "generated"
    data_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    generated_directory.mkdir(parents=True, exist_ok=True)
    fixed_path = data_directory / "lattice_exact_limit_fixed_fits.csv"
    joint_path = data_directory / "lattice_scalar_joint_validation.csv"
    covariance_path = data_directory / "lattice_scalar_covariance.csv"
    holdout_path = data_directory / "lattice_curve_holdout_test.csv"
    residual_figure_path = figure_directory / "lattice_exact_limit_residuals.pdf"
    holdout_figure_path = figure_directory / "lattice_holdout_validation.pdf"
    metadata_path = generated_directory / "lattice_validation_statistics_metadata.json"

    fixed.to_csv(fixed_path, index=False)
    joint.to_csv(joint_path, index=False)
    covariance.to_csv(covariance_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    make_exact_limit_figure(fixed, joint, residual_figure_path)
    make_holdout_figure(plot_records, holdout_figure_path)

    exact_magnetization = exact_spontaneous_magnetization(K_ISING)
    exact_delta_w = exact_pair_cavity_parameters(K_ISING)[2]
    exact = analytic_pulse_limits(exact_magnetization, exact_delta_w)
    output_paths = [
        fixed_path,
        joint_path,
        covariance_path,
        holdout_path,
        residual_figure_path,
        holdout_figure_path,
    ]
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_validation_statistics.py",
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "scipy": scipy_version,
        "input_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path) for path in raw_paths
        },
        "output_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path)
            for path in output_paths
        },
        "seed": args.seed,
        "blocks_per_replica": args.blocks_per_replica,
        "scalar_bootstrap_draws": args.scalar_bootstrap_draws,
        "holdout_bootstrap_draws": args.holdout_bootstrap_draws,
        "replica_ids_by_size": {
            str(length): replicas_for_size(trajectories, length)
            for length in expected_sizes
        },
        "analysis_mode": {
            "scalar": ["paired_antithetic", "raw_unpaired"],
            "holdout": "paired_antithetic with disjoint training/heldout replicas",
        },
        "bootstrap_hierarchy": (
            "all replicas retained as fixed strata; time blocks sampled with "
            "replacement independently within each replica"
        ),
        "independence_structure": (
            "lattice sizes come from independent simulations and are treated as "
            "independent covariance blocks; observables within a size retain their "
            "full bootstrap covariance"
        ),
        "field_interval": {
            "definition": (
                "specified from the exact half-height crossings before examining "
                "finite-size residuals"
            ),
            "h_min": float(H_GRID[0]),
            "h_max": float(H_GRID[-1]),
            "points": len(H_GRID),
            "spacing": float(H_GRID[1] - H_GRID[0]),
        },
        "exact_limits": exact,
        "exact_inputs": {
            "K": K_ISING,
            "spontaneous_magnetization": exact_magnetization,
            "delta_beta_w": exact_delta_w,
        },
        "fixed_limit_fit": (
            "X_L = X_infinity_exact + B/V; B is the only fitted parameter"
        ),
        "joint_fixed_limit_models": {
            "primary_baseline": "linear_all",
            "linear_all": "X_L = X_infinity_exact + B/V for all five sizes",
            "quadratic_all": (
                "X_L = X_infinity_exact + B/V + C/V^2 for all five sizes"
            ),
            "linear_drop_L16": (
                "X_L = X_infinity_exact + B/V for L=24,32,48,64"
            ),
            "reference_volume_used_for_numerical_conditioning": REFERENCE_VOLUME,
        },
        "bootstrap_calibration": (
            "centered stratified block-bootstrap noise is added to the fitted null "
            "model and every replicate is refitted before computing its statistic"
        ),
        "monte_carlo_p_value": (
            "(1 + number of centered-bootstrap statistics at least as large as "
            "observed)/(B+1); the minimum reportable values are therefore 1/(B+1)"
        ),
        "joint_scalar_records": {
            "central_rows": 10,
            "bootstrap_rows": int(
                2 * len(expected_sizes) * args.scalar_bootstrap_draws
            ),
            "joint_fit_rows": 2 * len(JOINT_FIT_MODELS),
            "fit_models": list(JOINT_FIT_MODELS),
            "observables": list(OBSERVABLES),
        },
        "covariance_regularization": {
            "method": (
                "eigenmode truncation of the correlation matrix; retained modes "
                "have lambda > relative_threshold * lambda_max"
            ),
            "primary_relative_threshold": PRIMARY_COVARIANCE_THRESHOLD,
            "sensitivity_thresholds": list(COVARIANCE_THRESHOLDS),
        },
        "holdout_folds": {
            str(length): [
                {"training": training, "heldout": heldout}
                for training, heldout in deterministic_three_folds(
                    replicas_for_size(trajectories, length)
                )
            ]
            for length in (48, 64)
        },
        "prediction_distinction": {
            "exact_crossfit": (
                "the finite-volume decomposition is algebraically exact for its "
                "training primitives; transfer to heldout replicas is a statistical "
                "cross-fitted prediction"
            ),
            "leading_logistic": (
                "uses only h=0 training primitives and additionally assumes their "
                "leading two-state field dependence"
            ),
            "first_corrected_logistic": (
                "fixes the exact thermodynamic density gap and cavity-work "
                "difference, fits d0,dchi,a1,dn only to training phase primitives "
                "over the specified field interval, and evaluates the derived "
                "O(1/V) curve; the total response adds its leading intraphase term"
            ),
        },
        "limitations": [
            "The centered bootstrap calibrates finite-sample noise but is not a proof of asymptotic chi-square behavior.",
            "Covariance modes below the declared relative threshold are not assigned inverse weight.",
            "The exact-crossfit label refers to the decomposition, not to equality of finite training and heldout estimates.",
            "The first-corrected prediction tests the derived truncated expansion on a fixed field interval; it is not an all-orders identity.",
        ],
        "reproduce_command": (
            "python3 scripts/lattice_validation_statistics.py "
            f"--blocks-per-replica {args.blocks_per_replica} "
            f"--scalar-bootstrap-draws {args.scalar_bootstrap_draws} "
            f"--holdout-bootstrap-draws {args.holdout_bootstrap_draws} "
            f"--seed {args.seed}"
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("Exact-limit-fixed scalar fits")
    print(
        fixed[
            [
                "mode",
                "observable",
                "B",
                "se_B",
                "chi2",
                "dof",
                "chi2_p_value",
                "centered_hierarchical_bootstrap_p_value",
            ]
        ].to_string(index=False)
    )
    print("\nJoint scalar compatibility")
    print(
        joint[joint["record_type"] == "joint_fit"][
            [
                "mode",
                "fit_model",
                "fit_sizes",
                "joint_Q",
                "joint_rank",
                "joint_parameter_count",
                "joint_dof",
                "joint_chi2_p_value",
                "joint_centered_bootstrap_p_value",
                "largest_pointwise_residual_L",
                "largest_pointwise_residual_observable",
            ]
        ]
        .to_string(index=False)
    )
    print("\nValidation summaries")
    print(
        holdout.groupby(["prediction_kind", "target"])[
            [
                "centered_hierarchical_bootstrap_p_value",
                "max_abs_pointwise_standardized_residual",
            ]
        ]
        .agg(["min", "median", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()
