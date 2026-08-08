#!/usr/bin/env python3
"""Leave-largest-size-out prediction of the complete coexistence response.

This analysis is deliberately separate from the production lattice analysis.
It uses all twelve paired-antithetic replicas at ``L=16,24,32,48`` to fit
the coefficients of the independently derived first-order finite-volume
curve.  It then predicts the complete total and phase-reweighting curves at
``L=64`` without using any ``L=64`` quantity in the fit.  Only after the
prediction is fixed are all twelve ``L=64`` replicas used for validation.

Uncertainty is propagated by a two-level bootstrap (replicas, then time
blocks within each selected replica).  Sizes are resampled with distinct
random streams.  Pointwise, simultaneous-curve, and regularized full-
covariance diagnostics therefore include uncertainty from both coefficient
training and the reserved target curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.special import expit
from scipy.stats import chi2, norm

try:
    from .lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        load_trajectory_archives,
        replicas_for_size,
    )
    from .lattice_gas_phase_switch import (
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        observables_from_contributions,
    )
    from .lattice_pulse_correction_prediction import (
        TAYLOR_SPECS,
        extract_primitive_coefficients,
        resample_total,
    )
    from .lattice_validation_statistics import (
        quadratic_from_whitener,
        regularized_precision,
    )
except ImportError:
    from lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        load_trajectory_archives,
        replicas_for_size,
    )
    from lattice_gas_phase_switch import (
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        observables_from_contributions,
    )
    from lattice_pulse_correction_prediction import (
        TAYLOR_SPECS,
        extract_primitive_coefficients,
        resample_total,
    )
    from lattice_validation_statistics import (
        quadratic_from_whitener,
        regularized_precision,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = (
    ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz",
    ROOT / "data_raw" / "lattice_gas_additional_replicas.npz",
)
DEFAULT_SEED = 26_080_611
DEFAULT_BOOTSTRAP_DRAWS = 2_000
DEFAULT_BLOCKS = 40

H_GRID = np.linspace(-1.0, 6.0, 141)
EXPECTED_LENGTHS = (16, 24, 32, 48, 64)
TRAINING_LENGTHS = (16, 24, 32, 48)
TARGET_LENGTH = 64
TARGETS = ("total", "reweighting")
PRIMITIVE_NAMES = (
    "d0",
    "dchi",
    "a1",
    "dn",
    "intraphase_liquid",
    "intraphase_gas",
)
TAYLOR_SPEC_NAME = "quadratic_h0p5"
SYMMETRY_TOLERANCE = 1.0e-8
COVARIANCE_THRESHOLDS = (1.0e-6, 1.0e-8, 1.0e-10)
PRIMARY_COVARIANCE_THRESHOLD = 1.0e-8


@dataclass(frozen=True)
class FitConfig:
    """Specified size fit for each finite-volume primitive sequence."""

    name: str
    included_lengths: tuple[int, ...]
    polynomial_degree_in_inverse_volume: int
    role: str


FIT_CONFIGS = (
    FitConfig("primary", TRAINING_LENGTHS, 1, "primary"),
    FitConfig("drop_L16", (24, 32, 48), 1, "size_cut_sensitivity"),
    FitConfig("constant_all", TRAINING_LENGTHS, 0, "fit_form_sensitivity"),
    FitConfig("quadratic_all", TRAINING_LENGTHS, 2, "fit_form_sensitivity"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(np.asarray(values, dtype=float), (0.025, 0.975))
    return float(lower), float(upper)


def weighted_sequence_fit(
    inverse_volume: np.ndarray,
    central_values: np.ndarray,
    bootstrap_values: np.ndarray,
    standard_errors: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a polynomial in ``1/V`` with fixed bootstrap-derived WLS weights."""

    inverse_volume = np.asarray(inverse_volume, dtype=float)
    central_values = np.asarray(central_values, dtype=float)
    bootstrap_values = np.asarray(bootstrap_values, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    if degree < 0 or len(inverse_volume) <= degree:
        raise ValueError("the size fit must have positive residual degrees of freedom")
    if central_values.shape != inverse_volume.shape:
        raise ValueError("central sequence and inverse-volume design differ")
    if bootstrap_values.ndim != 2 or bootstrap_values.shape[1] != len(inverse_volume):
        raise ValueError("bootstrap sequences have the wrong shape")
    if standard_errors.shape != inverse_volume.shape:
        raise ValueError("standard errors have the wrong shape")
    if np.any(~np.isfinite(standard_errors)) or np.any(standard_errors < 0.0):
        raise ValueError("standard errors must be finite and nonnegative")

    design = np.column_stack(
        [inverse_volume**power for power in range(degree + 1)]
    )
    weights = (
        1.0 / standard_errors**2
        if np.all(standard_errors > 1.0e-14)
        else np.ones_like(standard_errors)
    )
    column_scales = np.max(np.abs(design), axis=0)
    scaled_design = design / column_scales[None, :]
    square_root_weights = np.sqrt(weights)
    weighted_design = square_root_weights[:, None] * scaled_design
    operator = (
        np.diag(1.0 / column_scales)
        @ np.linalg.pinv(weighted_design)
        @ np.diag(square_root_weights)
    )
    return operator @ central_values, bootstrap_values @ operator.T


def first_order_curve(
    h_grid: np.ndarray,
    volume: int,
    coefficient_vectors: np.ndarray,
    phase_response_vectors: np.ndarray,
    exact_delta_rho: float,
    exact_a0: float,
) -> np.ndarray:
    """Return total and reweighting curves for one or many parameter vectors.

    The returned axes are ``draw, target, field``.  The phase-response inputs
    are the thermodynamic-limit conditional responses in the liquid and gas
    basins.  Their contribution to the scaled total curve is already order
    ``1/V``; finite-size changes in those inputs enter only at order ``1/V^2``.
    """

    h_grid = np.asarray(h_grid, dtype=float)
    coefficients = np.atleast_2d(np.asarray(coefficient_vectors, dtype=float))
    phase_responses = np.atleast_2d(
        np.asarray(phase_response_vectors, dtype=float)
    )
    if coefficients.shape[1] != 4:
        raise ValueError("coefficient vectors must contain d0,dchi,a1,dn")
    if phase_responses.shape != (len(coefficients), 2):
        raise ValueError("phase responses must contain liquid and gas values")
    if volume <= 0:
        raise ValueError("volume must be positive")

    h = h_grid[None, :]
    z = float(exact_delta_rho) * h
    leading_cavity_weight = expit(z - float(exact_a0))
    leading_free_weight = expit(z)
    kernel = leading_cavity_weight - leading_free_weight
    cavity_prime = leading_cavity_weight * (1.0 - leading_cavity_weight)
    free_prime = leading_free_weight * (1.0 - leading_free_weight)
    kernel_z = cavity_prime - free_prime
    kernel_a = -cavity_prime

    d0 = coefficients[:, 0, None]
    dchi = coefficients[:, 1, None]
    a1 = coefficients[:, 2, None]
    dn = coefficients[:, 3, None]
    u = d0 + dchi * h
    v = d0 * h + 0.5 * dchi * h**2
    w = a1 - dn * h
    reweighting = float(exact_delta_rho) * kernel + (
        u * kernel
        + float(exact_delta_rho) * (v * kernel_z + w * kernel_a)
    ) / float(volume)

    liquid = phase_responses[:, 0, None]
    gas = phase_responses[:, 1, None]
    intraphase = (
        leading_cavity_weight * liquid
        + (1.0 - leading_cavity_weight) * gas
    ) / float(volume)
    return np.stack((reweighting + intraphase, reweighting), axis=1)


def extract_training_vector(
    observed: dict[str, np.ndarray],
    volume: int,
    exact_delta_rho: float,
    exact_a0: float,
) -> np.ndarray:
    """Extract only quantities permitted to train the reserved prediction."""

    coefficients, _ = extract_primitive_coefficients(
        observed,
        volume,
        exact_delta_rho,
        exact_a0,
        TAYLOR_SPECS[TAYLOR_SPEC_NAME],
    )
    zero_index = int(np.argmin(np.abs(H_GRID)))
    if abs(H_GRID[zero_index]) > 1.0e-14:
        raise AssertionError("H_GRID must contain zero")
    return np.concatenate(
        (
            coefficients,
            np.asarray(
                [
                    observed["deltaN_liquid"][zero_index],
                    observed["deltaN_gas"][zero_index],
                ],
                dtype=float,
            ),
        )
    )


def extract_target_curves(
    observed: dict[str, np.ndarray], volume: int
) -> np.ndarray:
    """Extract only the two reserved curves used for prediction assessment."""

    return np.asarray(
        [
            np.asarray(observed["deltaN"], dtype=float) / volume,
            np.asarray(observed["deltaN_phase_reweighting"], dtype=float) / volume,
        ]
    )


def fit_configuration(
    config: FitConfig,
    central_training: np.ndarray,
    bootstrap_training: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit all primitive limits while retaining within-draw covariance."""

    training_lengths = np.asarray(TRAINING_LENGTHS)
    selected = np.asarray(
        [length in config.included_lengths for length in TRAINING_LENGTHS]
    )
    if tuple(training_lengths[selected]) != config.included_lengths:
        raise ValueError(f"invalid size selection for {config.name}")
    inverse_volume = 1.0 / np.square(training_lengths[selected].astype(float))
    standard_errors = np.std(bootstrap_training, axis=0, ddof=1)
    central_limits = np.empty(len(PRIMITIVE_NAMES), dtype=float)
    bootstrap_limits = np.empty(
        (len(bootstrap_training), len(PRIMITIVE_NAMES)), dtype=float
    )
    central_nuisance = np.full((len(PRIMITIVE_NAMES), 2), np.nan)

    for index, name in enumerate(PRIMITIVE_NAMES):
        if name == "dchi":
            central_limits[index] = 0.0
            bootstrap_limits[:, index] = 0.0
            central_nuisance[index] = 0.0
            continue
        parameters, draw_parameters = weighted_sequence_fit(
            inverse_volume,
            central_training[selected, index],
            bootstrap_training[:, selected, index],
            standard_errors[selected, index],
            config.polynomial_degree_in_inverse_volume,
        )
        central_limits[index] = parameters[0]
        bootstrap_limits[:, index] = draw_parameters[:, 0]
        if len(parameters) > 1:
            central_nuisance[index, 0] = parameters[1]
        if len(parameters) > 2:
            central_nuisance[index, 1] = parameters[2]

    return {
        "central_limits": central_limits,
        "bootstrap_limits": bootstrap_limits,
        "central_nuisance": central_nuisance,
        "selected": selected,
        "standard_errors": standard_errors,
    }


def curve_diagnostics(
    config: FitConfig,
    prediction_central: np.ndarray,
    prediction_draws: np.ndarray,
    target_central: np.ndarray,
    target_draws: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Build pointwise/simultaneous and full-covariance diagnostics."""

    difference_central = target_central - prediction_central
    difference_draws = target_draws - prediction_draws
    centered = difference_draws - np.mean(difference_draws, axis=0)
    standard_errors = np.std(difference_draws, axis=0, ddof=1)
    if np.any(~np.isfinite(standard_errors)) or np.any(standard_errors <= 0.0):
        raise ValueError("curve-difference standard errors must be positive")
    standardized = difference_central / standard_errors
    standardized_noise = centered / standard_errors[None, ...]

    curve_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    covariances: dict[str, np.ndarray] = {}
    selections: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for target_index, target in enumerate(TARGETS):
        selections.append(
            (
                target,
                difference_central[target_index],
                difference_draws[:, target_index],
                centered[:, target_index],
            )
        )
    selections.append(
        (
            "joint_total_reweighting",
            difference_central.reshape(-1),
            difference_draws.reshape(len(difference_draws), -1),
            centered.reshape(len(centered), -1),
        )
    )

    simultaneous_by_target: dict[str, tuple[float, float, float, bool]] = {}
    for target_index, target in enumerate(TARGETS):
        bootstrap_maximum = np.max(
            np.abs(standardized_noise[:, target_index]), axis=1
        )
        critical = float(np.quantile(bootstrap_maximum, 0.95))
        observed_maximum = float(np.max(np.abs(standardized[target_index])))
        simultaneous_p = float(
            (1 + np.count_nonzero(bootstrap_maximum >= observed_maximum))
            / (len(bootstrap_maximum) + 1)
        )
        simultaneous_by_target[target] = (
            critical,
            observed_maximum,
            simultaneous_p,
            bool(observed_maximum <= critical),
        )
        for field_index, h_value in enumerate(H_GRID):
            se = standard_errors[target_index, field_index]
            z_value = standardized[target_index, field_index]
            predicted = prediction_central[target_index, field_index]
            curve_rows.append(
                {
                    "analysis_config": config.name,
                    "role": config.role,
                    "training_L": " ".join(map(str, config.included_lengths)),
                    "target_L": TARGET_LENGTH,
                    "target": target,
                    "h": h_value,
                    "prediction": predicted,
                    "prediction_bootstrap_se": np.std(
                        prediction_draws[:, target_index, field_index], ddof=1
                    ),
                    "prediction_ci95_low": np.quantile(
                        prediction_draws[:, target_index, field_index], 0.025
                    ),
                    "prediction_ci95_high": np.quantile(
                        prediction_draws[:, target_index, field_index], 0.975
                    ),
                    "heldout_observation": target_central[
                        target_index, field_index
                    ],
                    "heldout_bootstrap_se": np.std(
                        target_draws[:, target_index, field_index], ddof=1
                    ),
                    "heldout_minus_prediction": difference_central[
                        target_index, field_index
                    ],
                    "difference_bootstrap_se": se,
                    "pointwise_z": z_value,
                    "pointwise_two_sided_normal_p": 2.0 * norm.sf(abs(z_value)),
                    "pointwise_95_low": predicted - 1.959963984540054 * se,
                    "pointwise_95_high": predicted + 1.959963984540054 * se,
                    "simultaneous_95_low": predicted - critical * se,
                    "simultaneous_95_high": predicted + critical * se,
                    "simultaneous_critical_95": critical,
                    "inside_simultaneous_95": abs(z_value) <= critical,
                    "bootstrap_draws": len(difference_draws),
                    "replicas_training_per_size": 12,
                    "replicas_heldout": 12,
                    "blocks_per_replica": DEFAULT_BLOCKS,
                }
            )

    for name, central, draws, noise in selections:
        covariance = np.cov(draws, rowvar=False, ddof=1)
        covariances[name] = covariance
        if name in TARGETS:
            target_index = TARGETS.index(name)
            critical, maximum_z, simultaneous_p, simultaneous_pass = (
                simultaneous_by_target[name]
            )
            argmax = int(np.argmax(np.abs(standardized[target_index])))
            largest_h = float(H_GRID[argmax])
            rmse = float(np.sqrt(np.mean(central**2)))
            maximum_absolute = float(np.max(np.abs(central)))
            reference_magnitude = float(
                np.max(np.abs(target_central[target_index]))
            )
            fraction_pointwise = float(
                np.mean(np.abs(standardized[target_index]) <= 1.959963984540054)
            )
        else:
            joint_standard_errors = np.sqrt(np.diag(covariance))
            joint_z = central / joint_standard_errors
            bootstrap_maximum = np.max(
                np.abs(noise / joint_standard_errors[None, :]), axis=1
            )
            critical = float(np.quantile(bootstrap_maximum, 0.95))
            maximum_z = float(np.max(np.abs(joint_z)))
            simultaneous_p = float(
                (1 + np.count_nonzero(bootstrap_maximum >= maximum_z))
                / (len(bootstrap_maximum) + 1)
            )
            simultaneous_pass = bool(maximum_z <= critical)
            argmax = int(np.argmax(np.abs(joint_z)))
            target_index, field_index = divmod(argmax, len(H_GRID))
            largest_h = float(H_GRID[field_index])
            rmse = float(np.sqrt(np.mean(central**2)))
            maximum_absolute = float(np.max(np.abs(central)))
            reference_magnitude = float(np.max(np.abs(target_central)))
            fraction_pointwise = float(
                np.mean(np.abs(joint_z) <= 1.959963984540054)
            )

        for threshold in COVARIANCE_THRESHOLDS:
            regularized = regularized_precision(covariance, threshold)
            whitener = np.asarray(regularized["whitener"])
            statistic = float(quadratic_from_whitener(central, whitener))
            bootstrap_statistics = quadratic_from_whitener(noise, whitener)
            bootstrap_p = float(
                (1 + np.count_nonzero(bootstrap_statistics >= statistic))
                / (len(bootstrap_statistics) + 1)
            )
            diagnostic_rows.append(
                {
                    "analysis_config": config.name,
                    "role": config.role,
                    "training_L": " ".join(map(str, config.included_lengths)),
                    "target_L": TARGET_LENGTH,
                    "target": name,
                    "bootstrap_draws": len(draws),
                    "replicas_training_per_size": 12,
                    "replicas_heldout": 12,
                    "blocks_per_replica": DEFAULT_BLOCKS,
                    "rmse": rmse,
                    "maximum_absolute_difference": maximum_absolute,
                    "maximum_absolute_difference_times_V2": (
                        maximum_absolute * TARGET_LENGTH**4
                    ),
                    "maximum_relative_difference_to_heldout_peak": (
                        maximum_absolute / reference_magnitude
                    ),
                    "max_abs_pointwise_z": maximum_z,
                    "largest_pointwise_residual_h": largest_h,
                    "fraction_inside_pointwise_95": fraction_pointwise,
                    "simultaneous_critical_95": critical,
                    "simultaneous_centered_bootstrap_p_value": simultaneous_p,
                    "all_points_inside_simultaneous_95": simultaneous_pass,
                    "covariance_relative_eigenvalue_threshold": threshold,
                    "covariance_rank": regularized["rank"],
                    "covariance_Q": statistic,
                    "covariance_chi2_p_value": chi2.sf(
                        statistic, int(regularized["rank"])
                    ),
                    "covariance_centered_bootstrap_p_value": bootstrap_p,
                    "covariance_condition_number_retained": regularized[
                        "condition_number_retained"
                    ],
                }
            )
    return curve_rows, diagnostic_rows, covariances


def make_figure(curves: pd.DataFrame, output: Path) -> None:
    """Render the primary evaluation curves and simultaneous residual check."""

    primary = curves[curves["analysis_config"] == "primary"]
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.2,
            "axes.labelsize": 8.7,
            "axes.titlesize": 8.9,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.9), constrained_layout=True)
    colors = {"total": "#0072B2", "reweighting": "#D55E00"}
    titles = {"total": "total response", "reweighting": "phase reweighting"}
    for column, target in enumerate(TARGETS):
        selected = primary[primary["target"] == target].sort_values("h")
        if len(selected) != len(H_GRID):
            raise ValueError(f"missing primary {target} curve")
        h = selected["h"].to_numpy()
        predicted = selected["prediction"].to_numpy()
        observed = selected["heldout_observation"].to_numpy()
        lower = selected["simultaneous_95_low"].to_numpy()
        upper = selected["simultaneous_95_high"].to_numpy()
        z = selected["pointwise_z"].to_numpy()
        critical = float(selected["simultaneous_critical_95"].iloc[0])

        axis = axes[0, column]
        axis.fill_between(h, lower, upper, color="0.88", linewidth=0.0)
        axis.plot(h, predicted, color="black", linewidth=1.25, label=r"$L\leq48$ prediction")
        axis.plot(
            h[::5],
            observed[::5],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=colors[target],
            markeredgewidth=0.8,
            markersize=3.2,
            label=r"evaluation data $L=64$",
        )
        axis.set_title(titles[target])
        axis.set_xlabel(r"$h=V(x-x_{\rm coex})$")
        axis.set_ylabel(r"scaled response $\Delta N/V$")
        axis.grid(alpha=0.15, linewidth=0.5)
        axis.legend(frameon=False, loc="upper center")

        residual_axis = axes[1, column]
        residual_axis.axhspan(-critical, critical, color="0.92", zorder=-2)
        residual_axis.axhline(0.0, color="black", linewidth=0.75)
        residual_axis.axhline(critical, color="0.35", linestyle="--", linewidth=0.8)
        residual_axis.axhline(-critical, color="0.35", linestyle="--", linewidth=0.8)
        residual_axis.plot(
            h,
            z,
            color=colors[target],
            linewidth=1.0,
            marker="o",
            markersize=2.0,
            markevery=4,
            markerfacecolor="white",
            markeredgewidth=0.55,
        )
        residual_axis.set_title(f"{titles[target]}: standardized residual")
        residual_axis.set_xlabel(r"$h=V(x-x_{\rm coex})$")
        residual_axis.set_ylabel("evaluation minus prediction (SE)")
        residual_axis.text(
            0.03,
            0.93,
            rf"$\max|z|={np.max(np.abs(z)):.2f}$; $c_{{0.95}}={critical:.2f}$",
            transform=residual_axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        residual_axis.grid(alpha=0.15, linewidth=0.5)
    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.11,
            1.035,
            f"({label})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.2,
        )
    fig.savefig(output)
    plt.close(fig)


def analyze(
    raw_paths: tuple[Path, ...],
    block_count: int,
    bootstrap_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    if block_count <= 1 or bootstrap_draws < 100:
        raise ValueError("analysis requires at least two blocks and 100 bootstrap draws")
    trajectories = load_trajectory_archives(raw_paths)
    if tuple(available_sizes(trajectories)) != EXPECTED_LENGTHS:
        raise ValueError(
            f"expected sizes {EXPECTED_LENGTHS}; got {available_sizes(trajectories)}"
        )
    for length in EXPECTED_LENGTHS:
        replicas = replicas_for_size(trajectories, length)
        if replicas != list(range(12)):
            raise ValueError(f"L={length} requires replica IDs 0--11; got {replicas}")

    exact_delta_rho = exact_spontaneous_magnetization(K_ISING)
    exact_a0 = exact_pair_cavity_parameters(K_ISING)[2]
    central_training = np.empty((len(TRAINING_LENGTHS), len(PRIMITIVE_NAMES)))
    bootstrap_training = np.empty(
        (bootstrap_draws, len(TRAINING_LENGTHS), len(PRIMITIVE_NAMES))
    )

    # Every training size has its own random stream.  No L=64 result is read
    # in this loop, so the prediction is fixed before target evaluation.
    for length_index, length in enumerate(TRAINING_LENGTHS):
        volume = length * length
        blocks, _ = contribution_blocks(
            trajectories, length, H_GRID, block_count, mode="paired"
        )
        central_observed = observables_from_contributions(
            np.sum(blocks, axis=(0, 1)), volume
        )
        central_training[length_index] = extract_training_vector(
            central_observed, volume, exact_delta_rho, exact_a0
        )
        rng = np.random.default_rng(seed + 10_003 * length)
        for draw in range(bootstrap_draws):
            observed = observables_from_contributions(
                resample_total(blocks, rng), volume
            )
            bootstrap_training[draw, length_index] = extract_training_vector(
                observed, volume, exact_delta_rho, exact_a0
            )

    maximum_dchi = max(
        float(np.max(np.abs(central_training[:, 1]))),
        float(np.max(np.abs(bootstrap_training[:, :, 1]))),
    )
    if maximum_dchi >= SYMMETRY_TOLERANCE:
        raise ValueError(
            "paired-antithetic dchi is not symmetry-zero; do not force the fit"
        )
    central_training[:, 1] = 0.0
    bootstrap_training[:, :, 1] = 0.0

    fit_results: dict[str, dict[str, np.ndarray]] = {}
    predictions_central: dict[str, np.ndarray] = {}
    predictions_draws: dict[str, np.ndarray] = {}
    for config in FIT_CONFIGS:
        fit = fit_configuration(config, central_training, bootstrap_training)
        fit_results[config.name] = fit
        limits = fit["central_limits"]
        draw_limits = fit["bootstrap_limits"]
        predictions_central[config.name] = first_order_curve(
            H_GRID,
            TARGET_LENGTH**2,
            limits[:4],
            limits[4:],
            exact_delta_rho,
            exact_a0,
        )[0]
        predictions_draws[config.name] = first_order_curve(
            H_GRID,
            TARGET_LENGTH**2,
            draw_limits[:, :4],
            draw_limits[:, 4:],
            exact_delta_rho,
            exact_a0,
        )

    # The reserved size enters only here, after every prediction above exists.
    target_blocks, target_replicas = contribution_blocks(
        trajectories, TARGET_LENGTH, H_GRID, block_count, mode="paired"
    )
    target_volume = TARGET_LENGTH**2
    target_central = extract_target_curves(
        observables_from_contributions(
            np.sum(target_blocks, axis=(0, 1)), target_volume
        ),
        target_volume,
    )
    target_draws = np.empty((bootstrap_draws, len(TARGETS), len(H_GRID)))
    target_rng = np.random.default_rng(seed + 1_000_003 + 10_003 * TARGET_LENGTH)
    for draw in range(bootstrap_draws):
        target_draws[draw] = extract_target_curves(
            observables_from_contributions(
                resample_total(target_blocks, target_rng), target_volume
            ),
            target_volume,
        )

    standard_errors = np.std(bootstrap_training, axis=0, ddof=1)
    coefficient_rows: list[dict[str, Any]] = []
    for length_index, length in enumerate(TRAINING_LENGTHS):
        for quantity_index, quantity in enumerate(PRIMITIVE_NAMES):
            draws = bootstrap_training[:, length_index, quantity_index]
            lower, upper = interval(draws)
            coefficient_rows.append(
                {
                    "record_type": "finite_volume_training_primitive",
                    "analysis_config": TAYLOR_SPEC_NAME,
                    "role": "training_input_only",
                    "quantity": quantity,
                    "L": length,
                    "V": length * length,
                    "inverse_V": 1.0 / (length * length),
                    "estimate": central_training[length_index, quantity_index],
                    "bootstrap_se": standard_errors[length_index, quantity_index],
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "polynomial_degree_in_inverse_V": np.nan,
                    "included_L": str(length),
                    "nuisance_1_over_V": np.nan,
                    "nuisance_1_over_V2": np.nan,
                    "replicas_per_size": 12,
                    "blocks_per_replica": block_count,
                    "bootstrap_draws": bootstrap_draws,
                }
            )
    for config in FIT_CONFIGS:
        fit = fit_results[config.name]
        for quantity_index, quantity in enumerate(PRIMITIVE_NAMES):
            draws = fit["bootstrap_limits"][:, quantity_index]
            lower, upper = interval(draws)
            coefficient_rows.append(
                {
                    "record_type": "thermodynamic_limit_fit",
                    "analysis_config": config.name,
                    "role": config.role,
                    "quantity": quantity,
                    "L": np.nan,
                    "V": np.nan,
                    "inverse_V": 0.0,
                    "estimate": fit["central_limits"][quantity_index],
                    "bootstrap_se": np.std(draws, ddof=1),
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "polynomial_degree_in_inverse_V": config.polynomial_degree_in_inverse_volume,
                    "included_L": " ".join(map(str, config.included_lengths)),
                    "nuisance_1_over_V": fit["central_nuisance"][quantity_index, 0],
                    "nuisance_1_over_V2": fit["central_nuisance"][quantity_index, 1],
                    "replicas_per_size": 12,
                    "blocks_per_replica": block_count,
                    "bootstrap_draws": bootstrap_draws,
                }
            )

    curve_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    primary_covariances: dict[str, np.ndarray] = {}
    for config in FIT_CONFIGS:
        rows, diagnostics, covariances = curve_diagnostics(
            config,
            predictions_central[config.name],
            predictions_draws[config.name],
            target_central,
            target_draws,
        )
        # Record the actual block count rather than relying on the default in
        # the helper's portable row schema.
        for row in rows:
            row["blocks_per_replica"] = block_count
        for row in diagnostics:
            row["blocks_per_replica"] = block_count
        curve_rows.extend(rows)
        diagnostic_rows.extend(diagnostics)
        if config.role == "primary":
            primary_covariances = covariances

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_leave_lmax_prediction.py",
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "scipy": scipy_version,
        },
        "raw_paths": [str(path.resolve().relative_to(ROOT)) for path in raw_paths],
        "raw_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path) for path in raw_paths
        },
        "settings": {
            "training_lengths": list(TRAINING_LENGTHS),
            "target_length": TARGET_LENGTH,
            "replica_ids_by_length": {
                str(length): replicas_for_size(trajectories, length)
                for length in EXPECTED_LENGTHS
            },
            "blocks_per_replica": block_count,
            "bootstrap_draws": bootstrap_draws,
            "seed": seed,
            "h_min": float(H_GRID[0]),
            "h_max": float(H_GRID[-1]),
            "h_points": len(H_GRID),
            "taylor_specification": TAYLOR_SPEC_NAME,
            "taylor_half_window_h": TAYLOR_SPECS[TAYLOR_SPEC_NAME].half_window,
            "taylor_polynomial_degree": TAYLOR_SPECS[TAYLOR_SPEC_NAME].degree,
            "fit_configurations": [
                {
                    "name": config.name,
                    "role": config.role,
                    "included_lengths": list(config.included_lengths),
                    "polynomial_degree_in_inverse_volume": config.polynomial_degree_in_inverse_volume,
                }
                for config in FIT_CONFIGS
            ],
            "covariance_relative_eigenvalue_thresholds": list(
                COVARIANCE_THRESHOLDS
            ),
            "primary_covariance_relative_eigenvalue_threshold": PRIMARY_COVARIANCE_THRESHOLD,
        },
        "exact_fixed_inputs": {
            "K": K_ISING,
            "delta_rho": exact_delta_rho,
            "a0": exact_a0,
        },
        "prediction_definition": (
            "The exact thermodynamic delta_rho and a0 are fixed. The finite-V "
            "training sequences d0(V), dchi(V), a1(V), dn(V), and the two h=0 "
            "conditional intraphase responses are fitted only at L<=48. Their "
            "1/V polynomial intercepts enter the derived first-order curve at "
            "V=64^2. Particle-hole symmetry identifies dchi=0."
        ),
        "target_nonleakage": {
            "fit_uses_target_length": False,
            "target_quantities_accessed": [
                "deltaN/V",
                "deltaN_phase_reweighting/V",
            ],
            "target_access_timing": (
                "all central and bootstrap predictions are computed before "
                "L=64 blocks are constructed"
            ),
        },
        "bootstrap_hierarchy": (
            "Within each size, replicas are sampled with replacement and time "
            "blocks are sampled with replacement within each selected replica."
        ),
        "dependence_statement": (
            "All observables and coefficients from one resample share indices, "
            "retaining their covariance. Distinct lattice sizes use distinct "
            "simulation and bootstrap streams; no claim of independent points "
            "within a curve is made."
        ),
        "simultaneous_calibration": (
            "The 95% critical value is the percentile of the maximum absolute "
            "pointwise-standardized centered bootstrap curve."
        ),
        "covariance_calibration": (
            "The correlation matrix is eigenvalue-truncated at the specified "
            "relative thresholds. The observed quadratic statistic is calibrated "
            "against centered hierarchical-bootstrap difference curves."
        ),
        "statistical_null_scope": (
            "The pointwise, simultaneous, and covariance p-values test exact "
            "equality of the finite L=64 curve to the truncated first-order "
            "predictor. The analytic result itself permits an O(V^-2) remainder. "
            "A high-precision contrast can therefore resolve a small higher-order "
            "remainder without contradicting the first-order expansion."
        ),
        "interpretive_scope": (
            "This is an out-of-size prediction of L=64 from smaller sizes, not "
            "an independent experimental replication. The paired-antithetic "
            "estimator uses the exact particle-hole symmetry of this benchmark."
        ),
        "target_replicas": target_replicas,
        "maximum_pre_forced_abs_dchi": maximum_dchi,
    }
    return (
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(curve_rows),
        pd.DataFrame(diagnostic_rows),
        metadata,
        primary_covariances,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", default=DEFAULT_RAW)
    parser.add_argument("--blocks-per-replica", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_paths = tuple(Path(path) for path in args.raw)
    for path in raw_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    coefficients, curves, diagnostics, metadata, covariances = analyze(
        raw_paths,
        args.blocks_per_replica,
        args.bootstrap_draws,
        args.seed,
    )

    data_directory = ROOT / "data_processed"
    generated_directory = ROOT / "generated"
    figure_directory = ROOT / "figures"
    data_directory.mkdir(parents=True, exist_ok=True)
    generated_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    coefficient_path = data_directory / "lattice_leave_lmax_prediction_coefficients.csv"
    curve_path = data_directory / "lattice_leave_lmax_prediction_curves.csv"
    diagnostic_path = data_directory / "lattice_leave_lmax_prediction_diagnostics.csv"
    covariance_path = data_directory / "lattice_leave_lmax_prediction_covariance.npz"
    figure_path = figure_directory / "lattice_leave_lmax_prediction.pdf"
    metadata_path = generated_directory / "lattice_leave_lmax_prediction_metadata.json"

    coefficients.to_csv(coefficient_path, index=False)
    curves.to_csv(curve_path, index=False)
    diagnostics.to_csv(diagnostic_path, index=False)
    np.savez_compressed(
        covariance_path,
        h_grid=H_GRID,
        target_order=np.asarray(TARGETS),
        covariance_total=covariances["total"],
        covariance_reweighting=covariances["reweighting"],
        covariance_joint_total_reweighting=covariances[
            "joint_total_reweighting"
        ],
    )
    make_figure(curves, figure_path)

    output_paths = (
        coefficient_path,
        curve_path,
        diagnostic_path,
        covariance_path,
        figure_path,
    )
    metadata["code_sha256"] = {
        "scripts/lattice_leave_lmax_prediction.py": sha256(Path(__file__).resolve()),
        "scripts/lattice_pulse_correction_prediction.py": sha256(
            ROOT / "scripts" / "lattice_pulse_correction_prediction.py"
        ),
        "scripts/lattice_validation_statistics.py": sha256(
            ROOT / "scripts" / "lattice_validation_statistics.py"
        ),
        "scripts/lattice_analysis_common.py": sha256(
            ROOT / "scripts" / "lattice_analysis_common.py"
        ),
        "scripts/lattice_gas_phase_switch.py": sha256(
            ROOT / "scripts" / "lattice_gas_phase_switch.py"
        ),
    }
    metadata["output_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in output_paths
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    primary = diagnostics[
        (diagnostics["analysis_config"] == "primary")
        & (
            diagnostics["covariance_relative_eigenvalue_threshold"]
            == PRIMARY_COVARIANCE_THRESHOLD
        )
    ]
    print("Leave-L=64-out prediction (trained only on L<=48)")
    for _, row in primary.iterrows():
        print(
            f"  {row['target']}: max|z|={row['max_abs_pointwise_z']:.3f}, "
            f"simultaneous p={row['simultaneous_centered_bootstrap_p_value']:.4f}, "
            f"covariance p={row['covariance_centered_bootstrap_p_value']:.4f}, "
            f"rank={int(row['covariance_rank'])}"
        )


if __name__ == "__main__":
    main()
