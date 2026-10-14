#!/usr/bin/env python3
"""Replica-respecting tests of lattice coexistence-pulse center, height, width."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import __version__ as scipy_version
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

try:
    from .lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        stratified_block_bootstrap_totals,
        load_trajectory_archives,
        weighted_linear_fit,
    )
    from .lattice_gas_phase_switch import (
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        observables_from_contributions,
        phase_theory,
    )
except ImportError:
    from lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        stratified_block_bootstrap_totals,
        load_trajectory_archives,
        weighted_linear_fit,
    )
    from lattice_gas_phase_switch import (
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        observables_from_contributions,
        phase_theory,
    )


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RAW = ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz"
ADDITIONAL_RAW = ROOT / "data_raw" / "lattice_gas_additional_replicas.npz"
DEFAULT_RAW = [BASELINE_RAW] + ([ADDITIONAL_RAW] if ADDITIONAL_RAW.exists() else [])
DEFAULT_SEED = 26_080_602
ANALYSIS_H_GRID = np.linspace(-6.0, 9.0, 601)
GRID_SENSITIVITY_STRIDES = (1, 2, 4)


def analytic_pulse_limits(delta_rho: float, delta_beta_w: float) -> dict[str, float]:
    """Return the exact scaled-field center, magnitude, and FWHM."""

    return {
        "peak_center_h": float(delta_beta_w / (2.0 * delta_rho)),
        "peak_magnitude": float(
            delta_rho * np.tanh(delta_beta_w / 4.0)
        ),
        "fwhm_h": float(
            2.0
            * np.arccosh(2.0 + np.cosh(0.5 * delta_beta_w))
            / delta_rho
        ),
    }


def continuous_pulse_observables(
    h_grid: np.ndarray, response: np.ndarray
) -> dict[str, float]:
    """Extract pulse center, magnitude, and FWHM from a cubic interpolant.

    The peak is obtained by bounded continuous optimization.  The two
    half-height crossings are then obtained by bracketing adjacent sampled
    values and applying Brent root finding to the same interpolant.  Thus none
    of the reported observables is restricted to a sampled field value.
    """

    h_grid = np.asarray(h_grid, dtype=float)
    response = np.asarray(response, dtype=float)
    if h_grid.ndim != 1 or response.shape != h_grid.shape:
        raise ValueError("h_grid and response must be one-dimensional peers")
    if len(h_grid) < 7 or np.any(np.diff(h_grid) <= 0.0):
        raise ValueError(
            "at least seven strictly increasing grid points are required"
        )
    magnitude = np.abs(response)
    peak_index = int(np.argmax(magnitude))
    if peak_index < 2 or peak_index > len(h_grid) - 3:
        raise ValueError("pulse maximum is too close to the grid boundary")

    interpolant = CubicSpline(h_grid, magnitude, extrapolate=False)
    peak_result = minimize_scalar(
        lambda field: -float(interpolant(field)),
        bounds=(h_grid[peak_index - 2], h_grid[peak_index + 2]),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not peak_result.success:
        raise RuntimeError(
            f"continuous peak optimization failed: {peak_result.message}"
        )
    center = float(peak_result.x)
    peak_magnitude = float(interpolant(center))
    half_height = 0.5 * peak_magnitude
    sampled_offsets = magnitude - half_height

    left_candidates = np.where(
        sampled_offsets[:peak_index] * sampled_offsets[1 : peak_index + 1]
        <= 0.0
    )[0]
    right_candidates = np.where(
        sampled_offsets[peak_index:-1] * sampled_offsets[peak_index + 1 :]
        <= 0.0
    )[0]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ValueError("pulse half-height crossings are not bracketed")
    left_index = int(left_candidates[-1])
    right_index = int(peak_index + right_candidates[0])
    root_function = lambda field: float(interpolant(field) - half_height)
    left_root = brentq(
        root_function,
        h_grid[left_index],
        h_grid[left_index + 1],
        xtol=1.0e-13,
    )
    right_root = brentq(
        root_function,
        h_grid[right_index],
        h_grid[right_index + 1],
        xtol=1.0e-13,
    )
    return {
        "peak_center_h": center,
        "peak_magnitude": peak_magnitude,
        "fwhm_h": float(right_root - left_root),
    }


def pulse_observables(
    h_grid: np.ndarray, contributions: np.ndarray, volume: int
) -> dict[str, float]:
    observed = observables_from_contributions(contributions, volume)
    response = observed["deltaN_phase_reweighting"] / volume
    return continuous_pulse_observables(h_grid, response)


def weighted_polynomial_fit(
    inverse_volume: np.ndarray,
    values: np.ndarray,
    standard_errors: np.ndarray,
    degree: int,
) -> dict[str, float | int]:
    """Weighted polynomial fit with a numerically scaled inverse-volume axis."""

    inverse_volume = np.asarray(inverse_volume, dtype=float)
    values = np.asarray(values, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    if degree < 0 or len(values) <= degree:
        raise ValueError("fit requires more observations than polynomial degree")
    if np.any(standard_errors <= 0.0):
        raise ValueError("all standard errors must be positive")
    scale = float(np.max(np.abs(inverse_volume)))
    if scale <= 0.0:
        raise ValueError("inverse-volume axis has zero scale")
    scaled = inverse_volume / scale
    design = np.column_stack([scaled**power for power in range(degree + 1)])
    precision = 1.0 / standard_errors**2
    normal = design.T @ (precision[:, None] * design)
    covariance_scaled = np.linalg.inv(normal)
    coefficients_scaled = covariance_scaled @ design.T @ (precision * values)
    transformation = np.diag([scale ** (-power) for power in range(degree + 1)])
    coefficients = transformation @ coefficients_scaled
    covariance = transformation @ covariance_scaled @ transformation
    residual = (values - design @ coefficients_scaled) / standard_errors
    result: dict[str, float | int] = {
        "A": float(coefficients[0]),
        "se_A": float(np.sqrt(covariance[0, 0])),
        "chi2": float(np.dot(residual, residual)),
        "dof": int(len(values) - degree - 1),
        "degree": int(degree),
    }
    coefficient_names = "ABCDEF"
    for index in range(1, degree + 1):
        result[coefficient_names[index]] = float(coefficients[index])
        result[f"se_{coefficient_names[index]}"] = float(
            np.sqrt(covariance[index, index])
        )
    return result


def correction_model_sensitivity(
    all_observables: pd.DataFrame, exact: dict[str, float]
) -> pd.DataFrame:
    """Compare nominal and alternative finite-size correction models."""

    model_specs = (
        ("linear_all", 1, np.ones(len(all_observables), dtype=bool)),
        ("quadratic_all", 2, np.ones(len(all_observables), dtype=bool)),
        ("linear_drop_L16", 1, all_observables["L"].to_numpy() >= 24),
    )
    records: list[dict[str, float | int | str]] = []
    for observable in ("peak_center_h", "peak_magnitude", "fwhm_h"):
        nominal_intercept: float | None = None
        nominal_se: float | None = None
        pending: list[dict[str, float | int | str]] = []
        for model, degree, mask in model_specs:
            selected = all_observables.loc[mask]
            fit = weighted_polynomial_fit(
                selected["inverse_V"].to_numpy(),
                selected[observable].to_numpy(),
                selected[f"se_{observable}"].to_numpy(),
                degree,
            )
            record: dict[str, float | int | str] = {
                "observable": observable,
                "model": model,
                "included_L": " ".join(str(value) for value in selected["L"]),
                "n_sizes": len(selected),
                **fit,
                "exact_limit": exact[observable],
                "difference_from_exact": float(fit["A"] - exact[observable]),
                "difference_in_se_A": float(
                    (fit["A"] - exact[observable]) / fit["se_A"]
                ),
            }
            pending.append(record)
            if model == "linear_all":
                nominal_intercept = float(fit["A"])
                nominal_se = float(fit["se_A"])
        assert nominal_intercept is not None and nominal_se is not None
        for record in pending:
            difference = float(record["A"]) - nominal_intercept
            record["difference_from_nominal"] = difference
            record["difference_from_nominal_in_nominal_se"] = difference / nominal_se
            records.append(record)
    return pd.DataFrame(records)


def analyze(
    raw_paths: list[Path],
    block_count: int,
    bootstrap_draws: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    trajectories = load_trajectory_archives(raw_paths)
    h_grid = np.asarray(ANALYSIS_H_GRID, dtype=float)
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int]] = []
    curve_records: list[dict[str, float | int]] = []
    grid_records: list[dict[str, float | int | str]] = []

    for length in available_sizes(trajectories):
        volume = length * length
        blocks, replicas = contribution_blocks(
            trajectories, length, h_grid, block_count, mode="paired"
        )
        central_contributions = np.sum(blocks, axis=(0, 1))
        central = pulse_observables(h_grid, central_contributions, volume)
        central_observed = observables_from_contributions(
            central_contributions, volume
        )
        central_response = central_observed["deltaN_phase_reweighting"] / volume
        central_total = central_observed["deltaN"] / volume
        central_intraphase = central_observed["deltaN_intraphase"] / volume
        for index, (h_value, response) in enumerate(zip(h_grid, central_response)):
            curve_records.append(
                {
                    "L": length,
                    "V": volume,
                    "h": h_value,
                    "pulse_magnitude_over_V": -response,
                    "total_response_magnitude_over_V": -central_total[index],
                    "intraphase_response_over_V": central_intraphase[index],
                    "decomposition_residual_over_V": (
                        central_observed["decomposition_residual"][index] / volume
                    ),
                }
            )
        for stride in GRID_SENSITIVITY_STRIDES:
            selected_h = h_grid[::stride]
            selected_response = central_response[::stride]
            if selected_h[-1] != h_grid[-1]:
                raise AssertionError("grid-sensitivity stride must retain endpoint")
            estimates = continuous_pulse_observables(
                selected_h, selected_response
            )
            for observable, estimate in estimates.items():
                grid_records.append(
                    {
                        "scope": f"L{length}",
                        "L": length,
                        "V": volume,
                        "observable": observable,
                        "grid_stride": stride,
                        "grid_spacing_h": round(
                            float(np.diff(selected_h)[0]), 12
                        ),
                        "n_grid_points": len(selected_h),
                        "estimate": estimate,
                    }
                )
        bootstrapped = {
            name: np.empty(bootstrap_draws, dtype=float) for name in central
        }
        for draw, total in enumerate(
            stratified_block_bootstrap_totals(blocks, bootstrap_draws, rng)
        ):
            result = pulse_observables(h_grid, total, volume)
            for name, value in result.items():
                bootstrapped[name][draw] = value
        record: dict[str, float | int] = {
            "L": length,
            "V": volume,
            "inverse_V": 1.0 / volume,
            "replicas": len(replicas),
            "blocks_per_replica": block_count,
            "bootstrap_draws": bootstrap_draws,
        }
        for name, value in central.items():
            finite_values = bootstrapped[name][np.isfinite(bootstrapped[name])]
            if len(finite_values) != bootstrap_draws:
                raise RuntimeError(f"nonfinite bootstrap {name} at L={length}")
            record[name] = value
            record[f"se_{name}"] = float(np.std(finite_values, ddof=1))
        records.append(record)

    all_observables = pd.DataFrame(records).sort_values("L").reset_index(drop=True)
    peak_frame = all_observables[
        [
            "L",
            "V",
            "inverse_V",
            "replicas",
            "blocks_per_replica",
            "bootstrap_draws",
            "peak_center_h",
            "se_peak_center_h",
            "peak_magnitude",
            "se_peak_magnitude",
        ]
    ].copy()
    width_frame = all_observables[
        [
            "L",
            "V",
            "inverse_V",
            "replicas",
            "blocks_per_replica",
            "bootstrap_draws",
            "fwhm_h",
            "se_fwhm_h",
        ]
    ].copy()

    exact_magnetization = exact_spontaneous_magnetization(K_ISING)
    exact_delta_w = exact_pair_cavity_parameters(K_ISING)[2]
    exact = analytic_pulse_limits(exact_magnetization, exact_delta_w)
    fit_records = []
    for observable in ("peak_center_h", "peak_magnitude", "fwhm_h"):
        fit = weighted_linear_fit(
            all_observables["inverse_V"].to_numpy(),
            all_observables[observable].to_numpy(),
            all_observables[f"se_{observable}"].to_numpy(),
        )
        fit_records.append(
            {
                "observable": observable,
                **fit,
                "exact_limit": exact[observable],
                "difference_from_exact": fit["A"] - exact[observable],
                "difference_in_se_A": (
                    (fit["A"] - exact[observable]) / fit["se_A"]
                ),
            }
        )
    fits = pd.DataFrame(fit_records)
    peak_fits = fits[fits["observable"] != "fwhm_h"].reset_index(drop=True)
    width_fit = fits[fits["observable"] == "fwhm_h"].reset_index(drop=True)
    correction_sensitivity = correction_model_sensitivity(all_observables, exact)

    grid_sensitivity = pd.DataFrame(grid_records)
    fine_estimates = grid_sensitivity[grid_sensitivity["grid_stride"] == 1][
        ["scope", "observable", "estimate"]
    ].rename(columns={"estimate": "finest_grid_estimate"})
    grid_sensitivity = grid_sensitivity.merge(
        fine_estimates, on=["scope", "observable"], validate="many_to_one"
    )
    grid_sensitivity["difference_from_finest"] = (
        grid_sensitivity["estimate"] - grid_sensitivity["finest_grid_estimate"]
    )

    # Propagate each extraction grid through the nominal A+B/V extrapolation.
    intercept_records: list[dict[str, float | int | str]] = []
    for observable in ("peak_center_h", "peak_magnitude", "fwhm_h"):
        standard_errors = all_observables[f"se_{observable}"].to_numpy()
        for stride in GRID_SENSITIVITY_STRIDES:
            selected = grid_sensitivity[
                (grid_sensitivity["observable"] == observable)
                & (grid_sensitivity["grid_stride"] == stride)
            ].sort_values("L")
            fit = weighted_linear_fit(
                all_observables["inverse_V"].to_numpy(),
                selected["estimate"].to_numpy(),
                standard_errors,
            )
            intercept_records.append(
                {
                    "scope": "thermodynamic_intercept",
                    "L": np.nan,
                    "V": np.nan,
                    "observable": observable,
                    "grid_stride": stride,
                    "grid_spacing_h": float(
                        round(
                            grid_sensitivity.loc[
                                grid_sensitivity["grid_stride"] == stride,
                                "grid_spacing_h",
                            ].iloc[0],
                            12,
                        )
                    ),
                    "n_grid_points": int(
                        grid_sensitivity.loc[
                            grid_sensitivity["grid_stride"] == stride,
                            "n_grid_points",
                        ].iloc[0]
                    ),
                    "estimate": fit["A"],
                    "se_estimate": fit["se_A"],
                    "chi2": fit["chi2"],
                    "dof": fit["dof"],
                }
            )
    intercept_frame = pd.DataFrame(intercept_records)
    fine_intercepts = intercept_frame[intercept_frame["grid_stride"] == 1][
        ["scope", "observable", "estimate"]
    ].rename(columns={"estimate": "finest_grid_estimate"})
    intercept_frame = intercept_frame.merge(
        fine_intercepts, on=["scope", "observable"], validate="many_to_one"
    )
    intercept_frame["difference_from_finest"] = (
        intercept_frame["estimate"] - intercept_frame["finest_grid_estimate"]
    )
    grid_sensitivity = pd.concat(
        [grid_sensitivity, intercept_frame], ignore_index=True, sort=False
    )
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_pulse_scaling.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "scipy": scipy_version,
        "raw_paths": [str(path.resolve().relative_to(ROOT)) for path in raw_paths],
        "seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "resampling": (
            "all replicas retained as fixed strata; time blocks sampled with "
            "replacement independently within each replica"
        ),
        "blocks_per_replica": block_count,
        "analysis_grid": {
            "minimum_h": float(h_grid[0]),
            "maximum_h": float(h_grid[-1]),
            "points": len(h_grid),
            "spacing_h": round(float(np.diff(h_grid)[0]), 12),
        },
        "peak_estimator": (
            "bounded continuous minimization of a cubic-spline interpolant "
            "of the pulse magnitude"
        ),
        "width_estimator": (
            "Brent root finding for both half-height crossings of the same "
            "cubic-spline interpolant"
        ),
        "fit": "weighted least squares A+B/V using stratified block standard errors",
        "finite_size_sensitivity_models": [
            "A+B/V using all sizes",
            "A+B/V+C/V^2 using all sizes",
            "A+B/V after excluding L=16",
        ],
        "grid_sensitivity_spacings_h": sorted(
            grid_sensitivity["grid_spacing_h"].dropna().unique().tolist()
        ),
        "finite_size_sensitivity_max_abs_shift_from_nominal": {
            observable: float(group["difference_from_nominal"].abs().max())
            for observable, group in correction_sensitivity.groupby("observable")
        },
        "grid_sensitivity_max_abs_intercept_shift": {
            observable: float(group["difference_from_finest"].abs().max())
            for observable, group in grid_sensitivity[
                grid_sensitivity["scope"] == "thermodynamic_intercept"
            ].groupby("observable")
        },
        "sensitivity_outputs": [
            "data_processed/lattice_pulse_finite_size_sensitivity.csv",
            "data_processed/lattice_pulse_grid_sensitivity.csv",
        ],
        "exact_limits": exact,
    }
    curves = pd.DataFrame(curve_records)
    return (
        peak_frame,
        peak_fits,
        width_frame,
        width_fit,
        curves,
        correction_sensitivity,
        grid_sensitivity,
        metadata,
    )


def make_figure(
    peak: pd.DataFrame,
    peak_fits: pd.DataFrame,
    width: pd.DataFrame,
    width_fit: pd.DataFrame,
    curves: pd.DataFrame,
    output: Path,
) -> None:
    """Create the four-panel PRE-scale pulse and finite-size figure.

    The scalar panels use the covariance-aware all-size quadratic model whose
    exact thermodynamic limits are fixed by the analytic solution.  The joint
    validation script supplies the correlated B and C coefficients; displaying
    separate marginal linear fits here would foreground a model rejected by
    the joint lack-of-fit test.
    """

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.7,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.9,
            "ytick.labelsize": 7.9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    palette = ("#56B4E9", "#009E73", "#E69F00", "#D55E00", "#7B3294")
    colors = {
        int(length): palette[index % len(palette)]
        for index, length in enumerate(sorted(curves["L"].unique()))
    }
    exact_magnetization = exact_spontaneous_magnetization(K_ISING)
    exact_delta_w = exact_pair_cavity_parameters(K_ISING)[2]
    exact_limits = analytic_pulse_limits(exact_magnetization, exact_delta_w)
    joint_path = ROOT / "data_processed" / "lattice_scalar_joint_validation.csv"
    if not joint_path.exists():
        raise FileNotFoundError(
            "run scripts/lattice_validation_statistics.py before making the figure"
        )
    joint = pd.read_csv(joint_path)
    selected_joint = joint[
        (joint["record_type"] == "joint_fit")
        & (joint["mode"] == "paired")
        & (joint["fit_model"] == "quadratic_all")
    ]
    if len(selected_joint) != 1:
        raise ValueError("missing unique paired quadratic_all joint fit")
    quadratic_joint = selected_joint.iloc[0]
    smooth_h = np.linspace(curves["h"].min(), curves["h"].max(), 1201)
    exact_curve = -phase_theory(smooth_h, exact_magnetization, exact_delta_w)

    # A dedicated legend row spans both top panels; no legend lies over data.
    fig = plt.figure(figsize=(7.1, 5.55), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(0.17, 1, 1))
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.set_axis_off()
    axes = np.array([[fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
                     [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]])
    pulse_axis, center_axis, magnitude_axis, width_axis = axes.flat
    for length, group in curves.groupby("L"):
        pulse_axis.plot(
            group["h"],
            group["total_response_magnitude_over_V"],
            color=colors[int(length)],
            linewidth=1.0,
            marker="o",
            markevery=48,
            markersize=2.0,
            label=rf"$L={int(length)}$",
        )
        pulse_axis.plot(
            group["h"],
            group["pulse_magnitude_over_V"],
            color=colors[int(length)],
            linewidth=0.65,
            linestyle=":",
            alpha=0.9,
        )
    pulse_axis.plot(
        smooth_h,
        exact_curve,
        color="black",
        linestyle="--",
        linewidth=1.25,
        label="exact limit",
    )
    pulse_axis.set_title("Total response and phase reweighting")
    pulse_axis.set_xlabel(r"$h=V(x-x_{\rm coex})$")
    pulse_axis.set_ylabel(r"$-\Delta N_{\mathcal{C}}/V$")
    handles, labels = pulse_axis.get_legend_handles_labels()
    legend_axis.legend(
        handles, labels,
        frameon=False,
        ncol=len(labels),
        fontsize=8.5,
        loc="center",
        borderaxespad=0.0,
    )
    pulse_axis.grid(alpha=0.16, linewidth=0.5)
    inset = pulse_axis.inset_axes([0.08, 0.66, 0.30, 0.20])
    for length in (int(curves["L"].min()), int(curves["L"].max())):
        group = curves[curves["L"] == length]
        inset.plot(
            group["h"],
            group["intraphase_response_over_V"] * group["V"],
            color=colors[length],
            linewidth=0.75,
            label=rf"$L={length}$",
        )
    inset.set_title(r"finite $\Delta N_{\rm intra}$", fontsize=6.8)
    inset.tick_params(labelsize=5.8, length=2)
    inset.legend(frameon=False, fontsize=5.4, loc="best")
    inset.grid(alpha=0.14, linewidth=0.4)

    plot_specs = (
        (
            center_axis,
            peak,
            peak_fits[peak_fits["observable"] == "peak_center_h"].iloc[0],
            "peak_center_h",
            "se_peak_center_h",
            r"$h_{\rm peak}$",
            "Peak center",
        ),
        (
            magnitude_axis,
            peak,
            peak_fits[peak_fits["observable"] == "peak_magnitude"].iloc[0],
            "peak_magnitude",
            "se_peak_magnitude",
            r"$|\Delta N_{\rm rw}|_{\rm peak}/V$",
            "Peak magnitude",
        ),
        (
            width_axis,
            width,
            width_fit.iloc[0],
            "fwhm_h",
            "se_fwhm_h",
            r"$V\Delta x_{\rm FWHM}$",
            "Pulse width",
        ),
    )
    for axis, frame, fit, value_column, error_column, ylabel, title in plot_specs:
        inverse_volume = frame["inverse_V"].to_numpy()
        axis.errorbar(
            inverse_volume,
            frame[value_column],
            yerr=frame[error_column],
            fmt="o",
            color="#0072B2",
            markersize=4.2,
            capsize=2.0,
            label="block-bootstrap estimate",
        )
        fit_x = np.linspace(0.0, 1.05 * np.max(inverse_volume), 200)
        exact_limit = float(fit["exact_limit"])
        coefficient_b = float(quadratic_joint[f"joint_B_{value_column}"])
        coefficient_c = float(quadratic_joint[f"joint_C_{value_column}"])
        axis.plot(
            fit_x,
            exact_limit + coefficient_b * fit_x + coefficient_c * fit_x**2,
            color="#D55E00",
            linewidth=1.05,
            label=r"joint $X_\infty^{\rm exact}+B/V+C/V^2$",
        )
        axis.axhline(
            exact_limit,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="exact limit",
        )
        axis.set_title(title)
        axis.set_xlabel(r"$1/V$")
        axis.set_ylabel(ylabel)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axis.grid(alpha=0.16, linewidth=0.5)
    center_axis.legend(frameon=False)

    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.10,
            1.035,
            f"({label})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", default=DEFAULT_RAW)
    parser.add_argument("--blocks-per-replica", type=int, default=40)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw saved numerical tables without resampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        names = ("lattice_peak_center_amplitude.csv", "lattice_peak_scaling_fits.csv",
                 "lattice_width_hierarchical.csv", "lattice_width_hierarchical_fit.csv",
                 "lattice_pulse_curves.csv")
        inputs = [ROOT / "data_processed" / name for name in names]
        joint = ROOT / "data_processed/lattice_scalar_joint_validation.csv"
        output = ROOT / "figures/lattice_pulse_scaling.pdf"
        metadata_path = ROOT / "generated/lattice_pulse_scaling_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        make_figure(*(pd.read_csv(path) for path in inputs), output)
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        metadata["figure_render"] = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "figure only; original numerical execution record preserved",
            "command": "python3 scripts/lattice_pulse_scaling.py --plot-only",
            "script_sha256": digest(Path(__file__).resolve()),
            "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (*inputs, joint)},
            "output_sha256": {str(output.relative_to(ROOT)): digest(output)},
            "layout": "dedicated external legend row aligned across both top panels",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print("Redrew saved pulse figure with an external legend band; numerical data unchanged")
        return
    (
        peak,
        peak_fits,
        width,
        width_fit,
        curves,
        correction_sensitivity,
        grid_sensitivity,
        metadata,
    ) = analyze(
        args.raw, args.blocks_per_replica, args.bootstrap_draws, args.seed
    )
    peak.to_csv(
        ROOT / "data_processed" / "lattice_peak_center_amplitude.csv", index=False
    )
    peak_fits.to_csv(
        ROOT / "data_processed" / "lattice_peak_scaling_fits.csv", index=False
    )
    width.to_csv(
        ROOT / "data_processed" / "lattice_width_hierarchical.csv", index=False
    )
    width_fit.to_csv(
        ROOT / "data_processed" / "lattice_width_hierarchical_fit.csv", index=False
    )
    curves.to_csv(
        ROOT / "data_processed" / "lattice_pulse_curves.csv", index=False
    )
    correction_sensitivity.to_csv(
        ROOT
        / "data_processed"
        / "lattice_pulse_finite_size_sensitivity.csv",
        index=False,
    )
    grid_sensitivity.to_csv(
        ROOT / "data_processed" / "lattice_pulse_grid_sensitivity.csv",
        index=False,
    )
    make_figure(
        peak,
        peak_fits,
        width,
        width_fit,
        curves,
        ROOT / "figures" / "lattice_pulse_scaling.pdf",
    )
    (ROOT / "generated" / "lattice_pulse_scaling_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for _, row in pd.concat((peak_fits, width_fit)).iterrows():
        print(
            f"{row['observable']}: A={row['A']:.9f} +/- {row['se_A']:.9f}; "
            f"exact={row['exact_limit']:.9f}; chi2/dof={row['chi2']:.3f}/{int(row['dof'])}"
        )
    print("finite-size correction-model sensitivity:")
    for _, row in correction_sensitivity.iterrows():
        print(
            f"  {row['observable']} [{row['model']}]: "
            f"A={row['A']:.9f} +/- {row['se_A']:.9f}; "
            f"shift/nominal-SE={row['difference_from_nominal_in_nominal_se']:.3f}"
        )
    intercepts = grid_sensitivity[
        grid_sensitivity["scope"] == "thermodynamic_intercept"
    ]
    print("extraction-grid sensitivity of thermodynamic intercepts:")
    for _, row in intercepts.iterrows():
        print(
            f"  {row['observable']} [dh={row['grid_spacing_h']:.3f}]: "
            f"A={row['estimate']:.9f}; "
            f"shift={row['difference_from_finest']:+.3e}"
        )


if __name__ == "__main__":
    main()
