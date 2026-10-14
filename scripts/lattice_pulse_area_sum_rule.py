#!/usr/bin/env python3
"""Signed-area sum rule for the lattice coexistence reweighting pulse.

For the exact thermodynamic two-state response

    R(h) = Delta rho [sigma(Delta rho h - a) - sigma(Delta rho h)],

the signed area is independent of the density gap:

    integral_{-infinity}^{infinity} R(h) dh = -a.

Here ``a`` is the liquid-minus-gas cavity-work difference.  This script
integrates the *signed* finite-volume phase-reweighting response directly from
the archived trajectories, estimates its uncertainty with the production
fixed replica strata, and compares the finite-size sequence with both the
exact finite-window target and the full analytic sum rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.integrate import simpson
from scipy.stats import chi2

try:
    from .lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        stratified_block_bootstrap_totals,
        load_trajectory_archives,
        replicas_for_size,
    )
    from .lattice_gas_phase_switch import (
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        observables_from_contributions,
    )
except ImportError:
    from lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        stratified_block_bootstrap_totals,
        load_trajectory_archives,
        replicas_for_size,
    )
    from lattice_gas_phase_switch import (
        K_ISING,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        observables_from_contributions,
    )


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RAW = ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz"
ADDITIONAL_RAW = ROOT / "data_raw" / "lattice_gas_additional_replicas.npz"
DEFAULT_RAW = [BASELINE_RAW, ADDITIONAL_RAW]
DEFAULT_SEED = 26_080_609
DEFAULT_BLOCKS = 40
DEFAULT_BOOTSTRAP_DRAWS = 2_000
EXPECTED_SIZES = (16, 24, 32, 48, 64)

# This interval and grid match the production pulse curves and were fixed
# before evaluating the area.  Its exact omitted tail is reported explicitly.
H_MIN = -6.0
H_MAX = 9.0
GRID_POINTS = 601
H_GRID = np.linspace(H_MIN, H_MAX, GRID_POINTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_area_targets(
    delta_rho: float,
    delta_beta_w: float,
    h_min: float = H_MIN,
    h_max: float = H_MAX,
) -> dict[str, float]:
    """Return the exact truncated area, full area, and omitted signed tail.

    An antiderivative of the exact response is

    ``log(1+exp(Delta rho h-a)) - log(1+exp(Delta rho h))``.

    ``numpy.logaddexp`` evaluates both softplus terms without overflow.
    """

    delta_rho = float(delta_rho)
    delta_beta_w = float(delta_beta_w)
    h_min = float(h_min)
    h_max = float(h_max)
    if delta_rho <= 0.0 or delta_beta_w <= 0.0:
        raise ValueError("delta_rho and delta_beta_w must be positive")
    if not h_min < h_max:
        raise ValueError("h_min must be smaller than h_max")

    def antiderivative(field: float) -> float:
        scaled = delta_rho * field
        return float(
            np.logaddexp(0.0, scaled - delta_beta_w)
            - np.logaddexp(0.0, scaled)
        )

    truncated = antiderivative(h_max) - antiderivative(h_min)
    full = -delta_beta_w
    tail = full - truncated
    return {
        "truncated_area": float(truncated),
        "full_area": float(full),
        "omitted_tail_area": float(tail),
        "omitted_tail_fraction_of_full_magnitude": float(abs(tail / full)),
    }


def signed_phase_reweighting_area(
    h_grid: np.ndarray,
    contributions: np.ndarray,
    volume: int,
) -> float:
    """Integrate the signed phase-reweighting response divided by volume."""

    h_grid = np.asarray(h_grid, dtype=float)
    if h_grid.ndim != 1 or len(h_grid) < 3:
        raise ValueError("h_grid must contain at least three points")
    if np.any(np.diff(h_grid) <= 0.0):
        raise ValueError("h_grid must be strictly increasing")
    observed = observables_from_contributions(contributions, int(volume))
    response = np.asarray(observed["deltaN_phase_reweighting"], dtype=float)
    response /= float(volume)
    if response.shape != h_grid.shape or not np.all(np.isfinite(response)):
        raise ValueError("phase-reweighting response is nonfinite or mis-shaped")
    return float(simpson(response, x=h_grid))


def weighted_polynomial_fit(
    inverse_volume: np.ndarray,
    values: np.ndarray,
    standard_errors: np.ndarray,
    degree: int,
    exact_limit: float | None,
) -> dict[str, object]:
    """Fit a polynomial in ``1/V`` with an exact or freely fitted intercept."""

    x = np.asarray(inverse_volume, dtype=float)
    y = np.asarray(values, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    if not (x.ndim == y.ndim == se.ndim == 1 and len(x) == len(y) == len(se)):
        raise ValueError("fit inputs must be one-dimensional peers")
    if degree < 1 or np.any(x <= 0.0) or np.any(se <= 0.0):
        raise ValueError("fit requires degree >= 1 and positive x and errors")
    parameter_count = degree + (exact_limit is None)
    if len(y) <= parameter_count:
        raise ValueError("fit leaves no goodness-of-fit degrees of freedom")

    scale = float(np.max(x))
    scaled_x = x / scale
    first_power = 0 if exact_limit is None else 1
    powers = np.arange(first_power, degree + 1)
    design = np.column_stack([scaled_x**power for power in powers])
    fit_values = y if exact_limit is None else y - float(exact_limit)
    precision = 1.0 / se**2
    normal = design.T @ (precision[:, None] * design)
    covariance_scaled = np.linalg.inv(normal)
    coefficients_scaled = covariance_scaled @ design.T @ (precision * fit_values)
    fitted = design @ coefficients_scaled
    if exact_limit is not None:
        fitted = fitted + float(exact_limit)
    residuals = y - fitted
    pulls = residuals / se
    statistic = float(np.dot(pulls, pulls))
    dof = int(len(y) - len(powers))

    physical_coefficients = np.asarray(
        [coefficient / scale**power for coefficient, power in zip(coefficients_scaled, powers)]
    )
    transformation = np.diag([scale ** (-power) for power in powers])
    covariance = transformation @ covariance_scaled @ transformation
    if exact_limit is None:
        limit = float(physical_coefficients[0])
        se_limit = float(np.sqrt(covariance[0, 0]))
    else:
        limit = float(exact_limit)
        se_limit = 0.0

    named_coefficients: dict[str, float] = {}
    coefficient_names = {1: "B", 2: "C", 3: "D"}
    for index, power in enumerate(powers):
        if power == 0:
            continue
        name = coefficient_names.get(int(power), f"coefficient_{int(power)}")
        named_coefficients[name] = float(physical_coefficients[index])
        named_coefficients[f"se_{name}"] = float(np.sqrt(covariance[index, index]))

    return {
        "thermodynamic_limit": limit,
        "se_thermodynamic_limit": se_limit,
        **named_coefficients,
        "chi2": statistic,
        "dof": dof,
        "chi2_p_value": float(chi2.sf(statistic, dof)),
        "max_abs_standardized_residual": float(np.max(np.abs(pulls))),
        "fitted": np.asarray(fitted, dtype=float),
        "residuals": np.asarray(residuals, dtype=float),
    }


def _fit_specifications(lengths: np.ndarray) -> tuple[dict[str, object], ...]:
    all_sizes = np.ones(len(lengths), dtype=bool)
    return (
        {
            "model": "fixed_linear_all",
            "degree": 1,
            "fixed": True,
            "mask": all_sizes,
        },
        {
            "model": "fixed_quadratic_all",
            "degree": 2,
            "fixed": True,
            "mask": all_sizes,
        },
        {
            "model": "fixed_linear_drop_L16",
            "degree": 1,
            "fixed": True,
            "mask": lengths >= 24,
        },
        {
            "model": "free_linear_all",
            "degree": 1,
            "fixed": False,
            "mask": all_sizes,
        },
        {
            "model": "free_quadratic_all",
            "degree": 2,
            "fixed": False,
            "mask": all_sizes,
        },
    )


def finite_size_comparison(
    areas: pd.DataFrame,
    bootstrap_areas: np.ndarray,
    exact_truncated_area: float,
) -> pd.DataFrame:
    """Compare the finite-size sequence with the exact finite-window limit."""

    lengths = areas["L"].to_numpy(dtype=int)
    inverse_volume = areas["inverse_V"].to_numpy(dtype=float)
    values = areas["signed_area"].to_numpy(dtype=float)
    errors = areas["se_signed_area"].to_numpy(dtype=float)
    bootstrap_areas = np.asarray(bootstrap_areas, dtype=float)
    if bootstrap_areas.ndim != 2 or bootstrap_areas.shape[1] != len(areas):
        raise ValueError("bootstrap_areas must have shape draws by sizes")

    records: list[dict[str, object]] = []
    draws = bootstrap_areas.shape[0]
    for spec in _fit_specifications(lengths):
        mask = np.asarray(spec["mask"], dtype=bool)
        x = inverse_volume[mask]
        y = values[mask]
        se = errors[mask]
        exact_limit = exact_truncated_area if bool(spec["fixed"]) else None
        fit = weighted_polynomial_fit(
            x, y, se, int(spec["degree"]), exact_limit
        )
        fitted = np.asarray(fit.pop("fitted"), dtype=float)
        fit.pop("residuals")

        selected_bootstrap = bootstrap_areas[:, mask]
        noise = selected_bootstrap - y[None, :]
        null_statistics = np.empty(draws, dtype=float)
        bootstrap_limits = np.empty(draws, dtype=float)
        bootstrap_coefficients: dict[str, np.ndarray] = {
            name: np.empty(draws, dtype=float)
            for name in ("B", "C", "D")
            if name in fit
        }
        for draw in range(draws):
            null_fit = weighted_polynomial_fit(
                x,
                fitted + noise[draw],
                se,
                int(spec["degree"]),
                exact_limit,
            )
            null_statistics[draw] = float(null_fit["chi2"])
            direct_fit = weighted_polynomial_fit(
                x,
                selected_bootstrap[draw],
                se,
                int(spec["degree"]),
                exact_limit,
            )
            bootstrap_limits[draw] = float(direct_fit["thermodynamic_limit"])
            for name, samples in bootstrap_coefficients.items():
                samples[draw] = float(direct_fit[name])

        observed_statistic = float(fit["chi2"])
        calibrated_p = float(
            (1 + np.count_nonzero(null_statistics >= observed_statistic))
            / (draws + 1)
        )
        record: dict[str, object] = {
            "model": str(spec["model"]),
            "exact_limit_fixed": bool(spec["fixed"]),
            "degree_in_inverse_V": int(spec["degree"]),
            "included_L": " ".join(str(value) for value in lengths[mask]),
            "n_sizes": int(np.count_nonzero(mask)),
            **fit,
            "exact_truncated_area": exact_truncated_area,
            "difference_from_exact": (
                float(fit["thermodynamic_limit"]) - exact_truncated_area
            ),
            "difference_from_exact_in_se": (
                np.nan
                if float(fit["se_thermodynamic_limit"]) == 0.0
                else (
                    float(fit["thermodynamic_limit"]) - exact_truncated_area
                )
                / float(fit["se_thermodynamic_limit"])
            ),
            "centered_hierarchical_bootstrap_p_value": calibrated_p,
            "bootstrap_se_thermodynamic_limit": float(
                np.std(bootstrap_limits, ddof=1)
            ),
            "bootstrap_ci95_low_thermodynamic_limit": float(
                np.quantile(bootstrap_limits, 0.025)
            ),
            "bootstrap_ci95_high_thermodynamic_limit": float(
                np.quantile(bootstrap_limits, 0.975)
            ),
        }
        for name, samples in bootstrap_coefficients.items():
            record[f"bootstrap_se_{name}"] = float(np.std(samples, ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def analyze(
    raw_paths: list[Path],
    block_count: int,
    bootstrap_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, object]]:
    """Run the signed-area analysis from raw trajectory archives."""

    if block_count < 2 or bootstrap_draws < 20:
        raise ValueError("use at least two blocks and 20 bootstrap draws")
    trajectories = load_trajectory_archives(raw_paths)
    lengths = available_sizes(trajectories)
    if lengths != list(EXPECTED_SIZES):
        raise ValueError(f"expected sizes {EXPECTED_SIZES}; got {lengths}")
    for length in lengths:
        replicas = replicas_for_size(trajectories, length)
        if replicas != list(range(12)):
            raise ValueError(f"L={length} does not contain replica IDs 0--11")

    exact_delta_rho = exact_spontaneous_magnetization(K_ISING)
    exact_delta_w = exact_pair_cavity_parameters(K_ISING)[2]
    targets = exact_area_targets(exact_delta_rho, exact_delta_w)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    bootstrap_by_size: list[np.ndarray] = []

    for length in lengths:
        volume = length * length
        blocks, replicas = contribution_blocks(
            trajectories, length, H_GRID, block_count, mode="paired"
        )
        central_total = np.sum(blocks, axis=(0, 1))
        central_area = signed_phase_reweighting_area(
            H_GRID, central_total, volume
        )
        bootstrap_values = np.empty(bootstrap_draws, dtype=float)
        for draw, total in enumerate(
            stratified_block_bootstrap_totals(blocks, bootstrap_draws, rng)
        ):
            bootstrap_values[draw] = signed_phase_reweighting_area(
                H_GRID, total, volume
            )
        if not np.all(np.isfinite(bootstrap_values)):
            raise RuntimeError(f"nonfinite bootstrap area at L={length}")
        standard_error = float(np.std(bootstrap_values, ddof=1))
        records.append(
            {
                "L": length,
                "V": volume,
                "inverse_V": 1.0 / volume,
                "replicas": len(replicas),
                "blocks_per_replica": block_count,
                "bootstrap_draws": bootstrap_draws,
                "h_min": H_MIN,
                "h_max": H_MAX,
                "grid_points": GRID_POINTS,
                "grid_spacing_h": float(H_GRID[1] - H_GRID[0]),
                "quadrature": "composite Simpson",
                "signed_area": central_area,
                "se_signed_area": standard_error,
                "bootstrap_mean_signed_area": float(np.mean(bootstrap_values)),
                "bootstrap_bias_signed_area": float(
                    np.mean(bootstrap_values) - central_area
                ),
                "bootstrap_ci95_low_signed_area": float(
                    np.quantile(bootstrap_values, 0.025)
                ),
                "bootstrap_ci95_high_signed_area": float(
                    np.quantile(bootstrap_values, 0.975)
                ),
                "exact_truncated_area": targets["truncated_area"],
                "exact_full_area": targets["full_area"],
                "exact_omitted_tail_area": targets["omitted_tail_area"],
                "residual_from_exact_truncated_area": (
                    central_area - targets["truncated_area"]
                ),
                "standardized_residual_from_exact_truncated_area": (
                    central_area - targets["truncated_area"]
                )
                / standard_error,
                "residual_from_exact_full_area": (
                    central_area - targets["full_area"]
                ),
            }
        )
        bootstrap_by_size.append(bootstrap_values)

    area_frame = pd.DataFrame(records).sort_values("L").reset_index(drop=True)
    bootstrap_matrix = np.column_stack(bootstrap_by_size)
    fits = finite_size_comparison(
        area_frame, bootstrap_matrix, float(targets["truncated_area"])
    )
    metadata: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_pulse_area_sum_rule.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "scipy": scipy_version,
        "seed": seed,
        "blocks_per_replica": block_count,
        "bootstrap_draws": bootstrap_draws,
        "replica_ids_by_size": {
            str(length): replicas_for_size(trajectories, length)
            for length in lengths
        },
        "bootstrap_hierarchy": (
            "all replicas retained as fixed strata; time blocks sampled with "
            "replacement independently within each replica"
        ),
        "analysis_mode": "paired spin-inversion antithetic contributions",
        "field_interval": {
            "selection": (
                "fixed production pulse interval, declared before evaluating "
                "the area"
            ),
            "h_min": H_MIN,
            "h_max": H_MAX,
            "points": GRID_POINTS,
            "spacing_h": float(H_GRID[1] - H_GRID[0]),
        },
        "quadrature": "composite Simpson rule on the fixed uniform h grid",
        "response_integrated": "deltaN_phase_reweighting(h)/V, with its sign retained",
        "exact_response": (
            "R_infinity(h)=Delta_rho[sigma(Delta_rho h-a)-sigma(Delta_rho h)]"
        ),
        "exact_antiderivative": (
            "log(1+exp(Delta_rho h-a))-log(1+exp(Delta_rho h))"
        ),
        "full_sum_rule": "integral from -infinity to infinity R_infinity(h) dh=-a",
        "exact_inputs": {
            "K": K_ISING,
            "delta_rho": exact_delta_rho,
            "a_delta_beta_w": exact_delta_w,
        },
        "exact_area_targets": targets,
        "finite_size_models": [
            "A_exact+B/V",
            "A_exact+B/V+C/V^2",
            "A_exact+B/V after excluding L=16",
            "A+B/V with A free",
            "A+B/V+C/V^2 with A free",
        ],
        "bootstrap_calibration": (
            "centered stratified block-bootstrap noise is added to each fitted null "
            "curve and every replicate is refitted before computing its statistic"
        ),
        "input_sha256": {str(path): sha256(path) for path in raw_paths},
    }
    return area_frame, fits, bootstrap_matrix, metadata


def _fit_curve(
    inverse_volume: np.ndarray,
    row: pd.Series,
) -> np.ndarray:
    values = np.full_like(
        inverse_volume, float(row["thermodynamic_limit"]), dtype=float
    )
    for power, name in ((1, "B"), (2, "C"), (3, "D")):
        if name in row.index and pd.notna(row[name]):
            values += float(row[name]) * inverse_volume**power
    return values


def make_figure(areas: pd.DataFrame, fits: pd.DataFrame, output: Path) -> None:
    """Plot the finite-size signed areas and their exact-limit comparisons."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(4.45, 3.35), constrained_layout=True)
    plotted_x = 1.0e3 * areas["inverse_V"].to_numpy()
    axis.errorbar(
        plotted_x,
        areas["signed_area"],
        yerr=areas["se_signed_area"],
        fmt="o",
        color="#0072B2",
        markeredgecolor="black",
        markeredgewidth=0.45,
        capsize=2.3,
        linewidth=0.8,
        markersize=4.4,
        label="trajectory estimate",
        zorder=4,
    )
    smooth_inverse_volume = np.linspace(0.0, areas["inverse_V"].max() * 1.04, 300)
    line_styles = {
        "fixed_linear_all": ("--", "#D55E00", r"$A_\infty+B/V$"),
        "fixed_quadratic_all": ("-", "#009E73", r"$A_\infty+B/V+C/V^2$"),
    }
    for model, (style, color, label) in line_styles.items():
        selected = fits[fits["model"] == model]
        if len(selected) != 1:
            raise ValueError(f"missing unique fit {model}")
        axis.plot(
            1.0e3 * smooth_inverse_volume,
            _fit_curve(smooth_inverse_volume, selected.iloc[0]),
            linestyle=style,
            color=color,
            linewidth=1.1,
            label=label,
            zorder=2,
        )
    truncated = float(areas["exact_truncated_area"].iloc[0])
    full = float(areas["exact_full_area"].iloc[0])
    axis.axhline(
        truncated,
        color="black",
        linewidth=0.9,
        label="exact finite-window limit",
        zorder=1,
    )
    axis.axhline(
        full,
        color="#7B3294",
        linewidth=0.9,
        linestyle=":",
        label=r"full sum rule $-a$",
        zorder=1,
    )
    for _, row in areas.iterrows():
        axis.annotate(
            rf"$L={int(row['L'])}$",
            (1.0e3 * float(row["inverse_V"]), float(row["signed_area"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.0,
        )
    axis.set_xlabel(r"$10^3/V$")
    axis.set_ylabel(r"$\int_{-6}^{9}\Delta N_{\rm rw}(h)/V\,dh$")
    axis.grid(alpha=0.18, linewidth=0.5)
    axis.legend(frameon=False, loc="best", ncol=1)
    fig.savefig(output)
    plt.close(fig)


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
    raw_paths = [Path(path) for path in args.raw]
    for path in raw_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    started = time.perf_counter()
    areas, fits, _, metadata = analyze(
        raw_paths,
        args.blocks_per_replica,
        args.bootstrap_draws,
        args.seed,
    )

    data_directory = ROOT / "data_processed"
    figure_directory = ROOT / "figures"
    generated_directory = ROOT / "generated"
    data_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    generated_directory.mkdir(parents=True, exist_ok=True)
    area_path = data_directory / "lattice_pulse_area_sum_rule.csv"
    fit_path = data_directory / "lattice_pulse_area_sum_rule_fits.csv"
    figure_path = figure_directory / "lattice_pulse_area_sum_rule.pdf"
    metadata_path = generated_directory / "lattice_pulse_area_sum_rule_metadata.json"

    areas.to_csv(area_path, index=False)
    fits.to_csv(fit_path, index=False)
    make_figure(areas, fits, figure_path)
    output_paths = [area_path, fit_path, figure_path]
    metadata.update(
        {
            "script_sha256": sha256(Path(__file__).resolve()),
            "output_sha256": {
                str(path.resolve().relative_to(ROOT)): sha256(path)
                for path in output_paths
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "reproduce_command": (
                "python3 scripts/lattice_pulse_area_sum_rule.py "
                f"--blocks-per-replica {args.blocks_per_replica} "
                f"--bootstrap-draws {args.bootstrap_draws} --seed {args.seed}"
            ),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("Signed phase-reweighting area on -6 <= h <= 9")
    print(
        areas[
            [
                "L",
                "signed_area",
                "se_signed_area",
                "residual_from_exact_truncated_area",
                "standardized_residual_from_exact_truncated_area",
            ]
        ].to_string(index=False)
    )
    print(
        "\nExact finite-window area "
        f"{areas['exact_truncated_area'].iloc[0]:.12f}; "
        f"full area {areas['exact_full_area'].iloc[0]:.12f}; "
        f"omitted tail {areas['exact_omitted_tail_area'].iloc[0]:.12f}"
    )
    print("\nFinite-size exact-limit comparisons")
    print(
        fits[
            [
                "model",
                "thermodynamic_limit",
                "se_thermodynamic_limit",
                "chi2",
                "dof",
                "chi2_p_value",
                "centered_hierarchical_bootstrap_p_value",
            ]
        ].to_string(index=False)
    )
    print(f"\nRuntime: {metadata['runtime_seconds']:.1f} s")


if __name__ == "__main__":
    main()
