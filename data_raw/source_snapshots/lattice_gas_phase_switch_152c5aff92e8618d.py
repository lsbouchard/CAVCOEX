#!/usr/bin/env python3
"""Two-phase Wolff test of the coexistence-window reweighting law.

The microscopic model is the square-lattice gas

    beta H = -beta*epsilon sum_<ij> n_i n_j - x sum_i n_i,

at beta*epsilon=2 and coexistence activity x=-2*beta*epsilon=-4.
With s_i=2*n_i-1 this is the zero-field square-lattice Ising model at
K=beta*epsilon/4=1/2. Wolff cluster updates sample both magnetization
sectors on a common normalization.

The constraint is an empty nearest-neighbor pair whose bond position is
averaged uniformly over the periodic lattice. Its probability depends on
both the phase density and the nearest-neighbor correlation, so it cannot be
reconstructed from the particle-number histogram alone.

Data collected at coexistence are reweighted to x=x_coex+h/V.  No logistic
curve is fitted: the finite-size prediction uses the separately estimated
phase density gap and phase-conditioned cavity-work difference at h=0; the
thermodynamic prediction uses Yang's exact spontaneous magnetization.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from scipy.special import ellipk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = (16, 24, 32, 48, 64)
DEFAULT_H_GRID = np.linspace(-6.0, 9.0, 301)
K_ISING = 0.5
BETA_EPSILON = 4.0 * K_ISING
X_COEX = -2.0 * BETA_EPSILON


@njit(cache=True)
def _neighbors_square_lattice(length: int) -> np.ndarray:
    volume = length * length
    neighbors = np.empty((volume, 4), dtype=np.int64)
    for row in range(length):
        for col in range(length):
            site = row * length + col
            neighbors[site, 0] = row * length + (col + 1) % length
            neighbors[site, 1] = row * length + (col - 1 + length) % length
            neighbors[site, 2] = ((row + 1) % length) * length + col
            neighbors[site, 3] = ((row - 1 + length) % length) * length + col
    return neighbors


@njit(cache=True)
def _wolff_series(
    length: int,
    coupling: float,
    seed: int,
    equilibration_steps: int,
    production_steps: int,
    initial_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return magnetization, bond sum, two fixed spins, and cluster sizes."""

    np.random.seed(seed)
    volume = length * length
    spins = np.full(volume, initial_sign, dtype=np.int8)
    magnetization = initial_sign * volume
    bond_sum = 2 * volume
    neighbors = _neighbors_square_lattice(length)
    stack = np.empty(volume, dtype=np.int64)
    cluster = np.empty(volume, dtype=np.int64)
    marks = np.zeros(volume, dtype=np.int64)
    mark = 0
    add_probability = 1.0 - np.exp(-2.0 * coupling)

    magnetizations = np.empty(production_steps, dtype=np.int32)
    bond_sums = np.empty(production_steps, dtype=np.int32)
    site_zero = np.empty(production_steps, dtype=np.int8)
    site_right = np.empty(production_steps, dtype=np.int8)
    cluster_sizes = np.empty(production_steps, dtype=np.int32)
    total_steps = equilibration_steps + production_steps

    for step in range(total_steps):
        mark += 1
        seed_site = np.random.randint(volume)
        old_spin = spins[seed_site]
        stack[0] = seed_site
        marks[seed_site] = mark
        stack_size = 1
        cluster_size = 0

        while stack_size > 0:
            stack_size -= 1
            site = stack[stack_size]
            cluster[cluster_size] = site
            cluster_size += 1
            for direction in range(4):
                candidate = neighbors[site, direction]
                if (
                    marks[candidate] != mark
                    and spins[candidate] == old_spin
                    and np.random.random() < add_probability
                ):
                    marks[candidate] = mark
                    stack[stack_size] = candidate
                    stack_size += 1

        for index in range(cluster_size):
            site = cluster[index]
            for direction in range(4):
                candidate = neighbors[site, direction]
                if marks[candidate] != mark:
                    bond_sum -= 2 * spins[site] * spins[candidate]

        for index in range(cluster_size):
            site = cluster[index]
            spins[site] = -spins[site]
        magnetization -= 2 * old_spin * cluster_size

        if step >= equilibration_steps:
            out = step - equilibration_steps
            magnetizations[out] = magnetization
            bond_sums[out] = bond_sum
            site_zero[out] = spins[0]
            site_right[out] = spins[1]
            cluster_sizes[out] = cluster_size

    return magnetizations, bond_sums, site_zero, site_right, cluster_sizes


def logistic(value: np.ndarray | float) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def exact_spontaneous_magnetization(coupling: float) -> float:
    """Yang's infinite-volume square-lattice Ising magnetization."""

    return float((1.0 - np.sinh(2.0 * coupling) ** -4) ** 0.125)


def exact_nearest_neighbor_correlation(coupling: float) -> float:
    """Onsager nearest-neighbor correlation for the isotropic square lattice."""

    twice = 2.0 * coupling
    modulus = 2.0 * np.sinh(twice) / np.cosh(twice) ** 2
    bracket = 1.0 + (2.0 / np.pi) * (
        2.0 * np.tanh(twice) ** 2 - 1.0
    ) * ellipk(modulus**2)
    return float(0.5 * bracket / np.tanh(twice))


def exact_pair_cavity_parameters(coupling: float) -> tuple[float, float, float]:
    """Return liquid/gas empty-bond probabilities and their work difference."""

    magnetization = exact_spontaneous_magnetization(coupling)
    correlation = exact_nearest_neighbor_correlation(coupling)
    p_liquid = (1.0 - 2.0 * magnetization + correlation) / 4.0
    p_gas = (1.0 + 2.0 * magnetization + correlation) / 4.0
    return p_liquid, p_gas, float(np.log(p_gas / p_liquid))


def orient_symmetry_pairs(
    magnetization: np.ndarray,
    site_zero: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient each sampled configuration so the first member is liquid-like.

    The spin-inverted configuration is included analytically as an antithetic
    partner.  For the measure-zero finite-lattice tie M=0, the member with
    s_0=+1 is assigned to the liquid basin.  This is a fixed exhaustive tie
    rule and makes global spin inversion exchange the two basins exactly.
    """

    liquid_is_original = (magnetization > 0) | (
        (magnetization == 0) & (site_zero > 0)
    )
    oriented_m = np.where(liquid_is_original, magnetization, -magnetization)
    oriented_site_zero = np.where(liquid_is_original, site_zero, -site_zero)
    return oriented_m, oriented_site_zero


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """Estimate the integrated autocorrelation time by the initial-positive sum."""

    centered = np.asarray(values, dtype=float) - np.mean(values)
    variance = float(np.dot(centered, centered) / len(centered))
    if variance == 0.0:
        return 0.5
    transform = np.fft.rfft(centered, n=2 * len(centered))
    correlation = np.fft.irfft(transform * np.conjugate(transform))[
        : len(centered)
    ]
    correlation /= variance * np.arange(len(centered), 0, -1)
    tau = 0.5
    for lag in range(1, len(correlation) - 1, 2):
        pair_sum = correlation[lag] + correlation[lag + 1]
        if pair_sum <= 0.0:
            break
        tau += pair_sum
    return float(tau)


@dataclass
class BlockData:
    replica: int
    block: int
    n_liquid: np.ndarray
    pair_empty_liquid: np.ndarray
    pair_empty_gas: np.ndarray
    fixed_empty_liquid: np.ndarray
    fixed_empty_gas: np.ndarray


def make_blocks(
    magnetization: np.ndarray,
    bond_sum: np.ndarray,
    site_zero: np.ndarray,
    site_right: np.ndarray,
    volume: int,
    replica: int,
    block_count: int,
) -> list[BlockData]:
    oriented_m, _ = orient_symmetry_pairs(magnetization, site_zero)
    liquid_is_original = (magnetization > 0) | (
        (magnetization == 0) & (site_zero > 0)
    )
    n_liquid = (volume + oriented_m.astype(np.int64)) // 2
    pair_empty_liquid = (
        2.0 * volume - 4.0 * oriented_m + bond_sum
    ) / (8.0 * volume)
    pair_empty_gas = (
        2.0 * volume + 4.0 * oriented_m + bond_sum
    ) / (8.0 * volume)
    original_empty = (site_zero == -1) & (site_right == -1)
    original_filled = (site_zero == 1) & (site_right == 1)
    fixed_empty_liquid = np.where(
        liquid_is_original, original_empty, original_filled
    ).astype(float)
    fixed_empty_gas = np.where(
        liquid_is_original, original_filled, original_empty
    ).astype(float)
    indices = np.array_split(np.arange(len(n_liquid)), block_count)
    return [
        BlockData(
            replica=replica,
            block=block,
            n_liquid=n_liquid[index],
            pair_empty_liquid=pair_empty_liquid[index],
            pair_empty_gas=pair_empty_gas[index],
            fixed_empty_liquid=fixed_empty_liquid[index],
            fixed_empty_gas=fixed_empty_gas[index],
        )
        for block, index in enumerate(indices)
        if len(index) > 0
    ]


CONTRIBUTION_NAMES = (
    "z",
    "n",
    "c",
    "cn",
    "z_liquid",
    "z_gas",
    "n_liquid",
    "n_gas",
    "c_liquid",
    "c_gas",
    "cn_liquid",
    "cn_gas",
    "fixed_c_liquid",
    "fixed_c_gas",
)


def block_contributions(
    block: BlockData, volume: int, h_grid: np.ndarray
) -> np.ndarray:
    n_liquid = block.n_liquid.astype(float)
    n_gas = volume - n_liquid
    centered_density = n_liquid / volume - 0.5
    u_liquid = np.exp(np.outer(h_grid, centered_density))
    u_gas = np.exp(-np.outer(h_grid, centered_density))

    c_liquid = block.pair_empty_liquid
    c_gas = block.pair_empty_gas
    fixed_c_liquid = block.fixed_empty_liquid.astype(float)
    fixed_c_gas = block.fixed_empty_gas.astype(float)

    def weighted_sum(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
        return weights @ values

    z_liquid = np.sum(u_liquid, axis=1)
    z_gas = np.sum(u_gas, axis=1)
    n_liquid_sum = weighted_sum(u_liquid, n_liquid)
    n_gas_sum = weighted_sum(u_gas, n_gas)
    c_liquid_sum = weighted_sum(u_liquid, c_liquid)
    c_gas_sum = weighted_sum(u_gas, c_gas)
    cn_liquid_sum = weighted_sum(u_liquid, c_liquid * n_liquid)
    cn_gas_sum = weighted_sum(u_gas, c_gas * n_gas)
    fixed_liquid_sum = weighted_sum(u_liquid, fixed_c_liquid)
    fixed_gas_sum = weighted_sum(u_gas, fixed_c_gas)

    return np.column_stack(
        (
            z_liquid + z_gas,
            n_liquid_sum + n_gas_sum,
            c_liquid_sum + c_gas_sum,
            cn_liquid_sum + cn_gas_sum,
            z_liquid,
            z_gas,
            n_liquid_sum,
            n_gas_sum,
            c_liquid_sum,
            c_gas_sum,
            cn_liquid_sum,
            cn_gas_sum,
            fixed_liquid_sum,
            fixed_gas_sum,
        )
    )


def observables_from_contributions(
    contributions: np.ndarray, volume: int
) -> dict[str, np.ndarray]:
    values = {
        name: contributions[:, index]
        for index, name in enumerate(CONTRIBUTION_NAMES)
    }
    mean_n = values["n"] / values["z"]
    mean_n_cavity = values["cn"] / values["c"]
    q_liquid = values["z_liquid"] / values["z"]
    q_cavity_liquid = values["c_liquid"] / values["c"]
    mean_n_liquid = values["n_liquid"] / values["z_liquid"]
    mean_n_gas = values["n_gas"] / values["z_gas"]
    mean_n_cavity_liquid = values["cn_liquid"] / values["c_liquid"]
    mean_n_cavity_gas = values["cn_gas"] / values["c_gas"]
    delta_n_liquid = mean_n_cavity_liquid - mean_n_liquid
    delta_n_gas = mean_n_cavity_gas - mean_n_gas
    intraphase = (
        q_cavity_liquid * delta_n_liquid
        + (1.0 - q_cavity_liquid) * delta_n_gas
    )
    phase_reweighting = (
        (q_cavity_liquid - q_liquid) * (mean_n_liquid - mean_n_gas)
    )
    delta_n = mean_n_cavity - mean_n
    work = -np.log(values["c"] / values["z"])
    work_liquid = -np.log(values["c_liquid"] / values["z_liquid"])
    work_gas = -np.log(values["c_gas"] / values["z_gas"])
    fixed_probability_liquid = (
        values["fixed_c_liquid"] / values["z_liquid"]
    )
    fixed_probability_gas = values["fixed_c_gas"] / values["z_gas"]

    return {
        "meanN0": mean_n,
        "meanNC": mean_n_cavity,
        "deltaN": delta_n,
        "q0_liquid": q_liquid,
        "qC_liquid": q_cavity_liquid,
        "meanN0_liquid": mean_n_liquid,
        "meanN0_gas": mean_n_gas,
        "deltaN_liquid": delta_n_liquid,
        "deltaN_gas": delta_n_gas,
        "deltaN_intraphase": intraphase,
        "deltaN_phase_reweighting": phase_reweighting,
        "decomposition_residual": delta_n - intraphase - phase_reweighting,
        "betaW": work,
        "betaW_liquid": work_liquid,
        "betaW_gas": work_gas,
        "delta_betaW": work_liquid - work_gas,
        "P_fixed_empty_liquid": fixed_probability_liquid,
        "P_fixed_empty_gas": fixed_probability_gas,
        "P_spatial_empty_liquid": values["c_liquid"] / values["z_liquid"],
        "P_spatial_empty_gas": values["c_gas"] / values["z_gas"],
        "delta_rho": (mean_n_liquid - mean_n_gas) / volume,
    }


def fwhm_from_grid(h_grid: np.ndarray, response: np.ndarray) -> float:
    magnitude = np.abs(np.asarray(response, dtype=float))
    peak_index = int(np.argmax(magnitude))
    half_height = 0.5 * magnitude[peak_index]

    left_values = magnitude[: peak_index + 1] - half_height
    right_values = magnitude[peak_index:] - half_height
    left_candidates = np.where(left_values[:-1] * left_values[1:] <= 0.0)[0]
    right_candidates = np.where(
        right_values[:-1] * right_values[1:] <= 0.0
    )[0]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return float("nan")

    left_index = int(left_candidates[-1])
    right_index = int(peak_index + right_candidates[0])

    def interpolate(index: int) -> float:
        x0, x1 = h_grid[index], h_grid[index + 1]
        y0, y1 = magnitude[index], magnitude[index + 1]
        return float(x0 + (half_height - y0) * (x1 - x0) / (y1 - y0))

    return interpolate(right_index) - interpolate(left_index)


def phase_theory(
    h_grid: np.ndarray, delta_rho: float, delta_beta_w: float
) -> np.ndarray:
    y = delta_rho * h_grid
    return delta_rho * (
        logistic(y - delta_beta_w) - logistic(y)
    )


def summarize_one_size(
    length: int,
    blocks: list[BlockData],
    h_grid: np.ndarray,
    bootstrap_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    volume = length * length
    block_arrays = np.asarray(
        [block_contributions(block, volume, h_grid) for block in blocks]
    )
    total = np.sum(block_arrays, axis=0)
    observed = observables_from_contributions(total, volume)
    center = int(np.argmin(np.abs(h_grid)))
    delta_rho_zero = float(observed["delta_rho"][center])
    delta_w_zero = float(observed["delta_betaW"][center])
    predicted = phase_theory(h_grid, delta_rho_zero, delta_w_zero)
    observed_scaled = observed["deltaN_phase_reweighting"] / volume

    rng = np.random.default_rng(seed)
    bootstrap_keys = (
        "deltaN",
        "deltaN_intraphase",
        "deltaN_phase_reweighting",
        "q0_liquid",
        "qC_liquid",
        "delta_betaW",
        "delta_rho",
        "P_fixed_empty_liquid",
        "P_fixed_empty_gas",
    )
    bootstrap = {
        key: np.empty((bootstrap_draws, len(h_grid)), dtype=float)
        for key in bootstrap_keys
    }
    bootstrap_fwhm = np.empty(bootstrap_draws, dtype=float)
    block_count = len(blocks)
    for draw in range(bootstrap_draws):
        indices = rng.integers(0, block_count, block_count)
        draw_observed = observables_from_contributions(
            np.sum(block_arrays[indices], axis=0), volume
        )
        for key in bootstrap_keys:
            bootstrap[key][draw] = draw_observed[key]
        bootstrap_fwhm[draw] = fwhm_from_grid(
            h_grid, draw_observed["deltaN_phase_reweighting"] / volume
        )

    standard_errors = {
        key: np.std(values, axis=0, ddof=1)
        for key, values in bootstrap.items()
    }
    rows = []
    for index, h_value in enumerate(h_grid):
        rows.append(
            {
                "L": length,
                "V": volume,
                "h": h_value,
                "delta_x": h_value / volume,
                "q0_liquid": observed["q0_liquid"][index],
                "se_q0_liquid": standard_errors["q0_liquid"][index],
                "qC_liquid": observed["qC_liquid"][index],
                "se_qC_liquid": standard_errors["qC_liquid"][index],
                "meanN0": observed["meanN0"][index],
                "meanNC": observed["meanNC"][index],
                "deltaN": observed["deltaN"][index],
                "se_deltaN": standard_errors["deltaN"][index],
                "deltaN_intraphase": observed["deltaN_intraphase"][index],
                "se_deltaN_intraphase": standard_errors[
                    "deltaN_intraphase"
                ][index],
                "deltaN_phase_reweighting": observed[
                    "deltaN_phase_reweighting"
                ][index],
                "se_deltaN_phase_reweighting": standard_errors[
                    "deltaN_phase_reweighting"
                ][index],
                "deltaN_phase_reweighting_over_V": observed_scaled[index],
                "se_deltaN_phase_reweighting_over_V": standard_errors[
                    "deltaN_phase_reweighting"
                ][index]
                / volume,
                "delta_rho": observed["delta_rho"][index],
                "se_delta_rho": standard_errors["delta_rho"][index],
                "delta_betaW": observed["delta_betaW"][index],
                "se_delta_betaW": standard_errors["delta_betaW"][index],
                "finite_size_prediction": predicted[index],
                "decomposition_residual": observed[
                    "decomposition_residual"
                ][index],
                "betaW": observed["betaW"][index],
                "source_type": "mc_new",
                "run_id": "lattice_gas_phase_switch_v1",
                "units_convention": (
                    "beta=1; beta_epsilon=2; "
                    "x=beta_mu; x_coex=-4; h=V*(x-x_coex)"
                ),
                "raw_data_path": (
                    "data_raw/lattice_gas_phase_switch_trajectories.npz"
                ),
                "analysis_version": "1",
            }
        )

    exact_magnetization = exact_spontaneous_magnetization(K_ISING)
    exact_correlation = exact_nearest_neighbor_correlation(K_ISING)
    exact_p_liquid, exact_p_gas, exact_delta_w = (
        exact_pair_cavity_parameters(K_ISING)
    )
    exact_width = float(
        2.0
        * np.arccosh(2.0 + np.cosh(0.5 * exact_delta_w))
        / exact_magnetization
    )
    finite_width = float(
        2.0
        * np.arccosh(2.0 + np.cosh(0.5 * delta_w_zero))
        / delta_rho_zero
    )
    direct_width = fwhm_from_grid(h_grid, observed_scaled)
    coarse_width = fwhm_from_grid(h_grid[::2], observed_scaled[::2])
    grid_width_error = abs(direct_width - coarse_width)
    theory_scale = max(np.max(np.abs(predicted)), 1.0e-15)
    rms_curve_error = float(
        np.sqrt(np.mean((observed_scaled - predicted) ** 2)) / theory_scale
    )

    frame = pd.DataFrame(rows)
    step_h = float(h_grid[1] - h_grid[0])
    derivative_h = np.full(len(h_grid), np.nan)
    derivative_h[2:-2] = (
        observed["betaW"][:-4]
        - 8.0 * observed["betaW"][1:-3]
        + 8.0 * observed["betaW"][3:-1]
        - observed["betaW"][4:]
    ) / (12.0 * step_h)
    derivative_x = volume * derivative_h
    nucleation_residual = derivative_x + observed["deltaN"]
    frame["d_betaW_dx_five_point"] = derivative_x
    frame["nucleation_theorem_residual"] = nucleation_residual
    interior = np.isfinite(nucleation_residual)

    summary = {
        "L": length,
        "V": volume,
        "delta_rho_h0": delta_rho_zero,
        "se_delta_rho_h0": float(standard_errors["delta_rho"][center]),
        "delta_betaW_h0": delta_w_zero,
        "se_delta_betaW_h0": float(
            standard_errors["delta_betaW"][center]
        ),
        "qC_liquid_h0": float(observed["qC_liquid"][center]),
        "se_qC_liquid_h0": float(
            standard_errors["qC_liquid"][center]
        ),
        "deltaN_h0": float(observed["deltaN"][center]),
        "se_deltaN_h0": float(standard_errors["deltaN"][center]),
        "intraphase_h0": float(observed["deltaN_intraphase"][center]),
        "se_intraphase_h0": float(
            standard_errors["deltaN_intraphase"][center]
        ),
        "phase_reweighting_h0": float(
            observed["deltaN_phase_reweighting"][center]
        ),
        "se_phase_reweighting_h0": float(
            standard_errors["deltaN_phase_reweighting"][center]
        ),
        "phase_reweighting_over_V_h0": float(observed_scaled[center]),
        "se_phase_reweighting_over_V_h0": float(
            standard_errors["deltaN_phase_reweighting"][center] / volume
        ),
        "P_spatial_empty_liquid_h0": float(
            observed["P_spatial_empty_liquid"][center]
        ),
        "P_fixed_empty_liquid_h0": float(
            observed["P_fixed_empty_liquid"][center]
        ),
        "se_P_fixed_empty_liquid_h0": float(
            standard_errors["P_fixed_empty_liquid"][center]
        ),
        "P_spatial_empty_gas_h0": float(
            observed["P_spatial_empty_gas"][center]
        ),
        "P_fixed_empty_gas_h0": float(
            observed["P_fixed_empty_gas"][center]
        ),
        "se_P_fixed_empty_gas_h0": float(
            standard_errors["P_fixed_empty_gas"][center]
        ),
        "direct_fwhm_h": direct_width,
        "se_direct_fwhm_h": float(
            np.nanstd(bootstrap_fwhm, ddof=1)
        ),
        "fwhm_grid_discretization_estimate_h": grid_width_error,
        "finite_size_predicted_fwhm_h": finite_width,
        "exact_fwhm_h": exact_width,
        "normalized_curve_rmse": rms_curve_error,
        "maximum_abs_nucleation_theorem_residual": float(
            np.max(np.abs(nucleation_residual[interior]))
        ),
        "normalized_nucleation_theorem_rmse": float(
            np.sqrt(np.mean(nucleation_residual[interior] ** 2))
            / np.max(np.abs(observed["deltaN"][interior]))
        ),
        "maximum_decomposition_residual": float(
            np.max(np.abs(observed["decomposition_residual"]))
        ),
        "exact_magnetization": exact_magnetization,
        "exact_nearest_neighbor_correlation": exact_correlation,
        "exact_pair_empty_liquid": exact_p_liquid,
        "exact_pair_empty_gas": exact_p_gas,
        "exact_delta_betaW": exact_delta_w,
    }
    return frame, summary


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.7,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def make_figure(
    data: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> None:
    configure_plotting()
    palette = ("#56B4E9", "#009E73", "#E69F00", "#D55E00", "#7B3294")
    colors = {
        int(length): palette[index % len(palette)]
        for index, length in enumerate(sorted(data["L"].unique()))
    }
    exact_m = exact_spontaneous_magnetization(K_ISING)
    _, _, exact_delta_w = exact_pair_cavity_parameters(K_ISING)
    h_smooth = np.linspace(data["h"].min(), data["h"].max(), 1001)
    exact_curve = phase_theory(h_smooth, exact_m, exact_delta_w)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), constrained_layout=True)
    ax_weights, ax_collapse, ax_scaling, ax_width = axes.flat
    for length, group in data.groupby("L"):
        group = group.sort_values("h")
        color = colors[int(length)]
        if int(length) in (16, 32, 64):
            ax_weights.plot(
                group["h"],
                group["q0_liquid"],
                color=color,
                linewidth=0.9,
                linestyle="--",
                marker="o",
                markevery=30,
                markersize=2.0,
                label=rf"$L={int(length)}$, $q_\ell$",
            )
            ax_weights.plot(
                group["h"],
                group["qC_liquid"],
                color=color,
                linewidth=1.1,
                marker="s",
                markevery=30,
                markersize=2.0,
                label=rf"$L={int(length)}$, $\widetilde q_\ell$",
            )
        ax_collapse.plot(
            group["h"],
            group["deltaN_phase_reweighting_over_V"],
            color=color,
            linewidth=0.9,
            marker="o",
            markevery=30,
            markersize=1.8,
            label=rf"$L={int(length)}$",
        )

    ax_collapse.plot(
        h_smooth,
        exact_curve,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label="exact thermodynamic limit",
    )
    ax_weights.set_title("Direct phase weights")
    ax_weights.set_xlabel(r"$h=V(x-x_{\rm coex})$")
    ax_weights.set_ylabel("liquid weight")
    ax_weights.legend(frameon=False, ncol=2, fontsize=7.0)
    ax_weights.grid(alpha=0.16, linewidth=0.5)

    ax_collapse.set_title("No-fit coexistence collapse")
    ax_collapse.set_xlabel(r"$h=V(x-x_{\rm coex})$")
    ax_collapse.set_ylabel(r"$\Delta N_{\rm sel}/V$")
    ax_collapse.legend(frameon=False, ncol=2)
    ax_collapse.grid(alpha=0.16, linewidth=0.5)

    ordered = summary.sort_values("V")
    volume = ordered["V"].to_numpy()
    ax_scaling.errorbar(
        volume,
        np.abs(ordered["phase_reweighting_h0"]),
        yerr=ordered["se_phase_reweighting_h0"],
        fmt="o",
        color="#D55E00",
        markersize=4,
        capsize=2,
        label=r"$|\Delta N_{\rm rw}|$",
    )
    exact_equal_weight = exact_m * (
        0.5 - float(logistic(-exact_delta_w))
    )
    ax_scaling.plot(
        volume,
        exact_equal_weight * volume,
        "k--",
        linewidth=1.0,
        label=r"exact $V$ coefficient",
    )
    ax_scaling.errorbar(
        volume,
        np.abs(ordered["intraphase_h0"]),
        yerr=ordered["se_intraphase_h0"],
        fmt="s",
        color="#0072B2",
        markersize=3.5,
        capsize=2,
        label=r"$|\Delta N_{\rm intra}|$",
    )
    ax_scaling.set_title("Extensive reweighting, finite background")
    ax_scaling.set_xlabel(r"$V=L^2$")
    ax_scaling.set_ylabel("equal-weight response magnitude")
    ax_scaling.legend(frameon=False)
    ax_scaling.grid(alpha=0.16, linewidth=0.5)

    ax_width.errorbar(
        1.0 / volume,
        ordered["direct_fwhm_h"],
        yerr=ordered["se_direct_fwhm_h"],
        fmt="o",
        color="#009E73",
        markersize=4,
        capsize=2,
        label=r"direct $V\Delta x_{\rm FWHM}$",
    )
    ax_width.plot(
        1.0 / volume,
        ordered["finite_size_predicted_fwhm_h"],
        color="#D55E00",
        marker="s",
        markersize=3,
        linewidth=0.8,
        label="measured-parameter prediction",
    )
    ax_width.axhline(
        ordered["exact_fwhm_h"].iloc[0],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="exact thermodynamic limit",
    )
    ax_width.set_title(r"Width coefficient: $\Delta x_{\rm FWHM}\propto V^{-1}$")
    ax_width.set_xlabel(r"$1/V$")
    ax_width.set_ylabel(r"$V\Delta x_{\rm FWHM}$")
    ax_width.legend(frameon=False)
    ax_width.grid(alpha=0.16, linewidth=0.5)

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
    parser.add_argument("--seed", type=int, default=25_072_026)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--equil-steps", type=int, default=2_000)
    parser.add_argument("--production-steps", type=int, default=20_000)
    parser.add_argument("--blocks-per-replica", type=int, default=40)
    parser.add_argument("--bootstrap-draws", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, float | int]] = []

    # Trigger compilation before timing/progress output.
    _wolff_series(4, K_ISING, 1, 2, 2, 1)

    for length in args.sizes:
        for replica in range(args.replicas):
            replica_seed = int(args.seed + 1009 * length + 104_729 * replica)
            initial_sign = 1 if replica % 2 == 0 else -1
            (
                magnetization,
                bond_sum,
                site_zero,
                site_right,
                cluster_sizes,
            ) = _wolff_series(
                length,
                K_ISING,
                replica_seed,
                args.equil_steps,
                args.production_steps,
                initial_sign,
            )
            raw[f"L{length}_replica{replica}_magnetization"] = magnetization
            raw[f"L{length}_replica{replica}_bond_sum"] = bond_sum
            raw[f"L{length}_replica{replica}_site_zero"] = site_zero
            raw[f"L{length}_replica{replica}_site_right"] = site_right
            raw[f"L{length}_replica{replica}_cluster_size"] = cluster_sizes
            oriented_m, _ = orient_symmetry_pairs(magnetization, site_zero)
            pair_empty_liquid = (
                2.0 * length * length - 4.0 * oriented_m + bond_sum
            ) / (8.0 * length * length)
            signs = np.sign(magnetization)
            nonzero = signs[signs != 0]
            switches = int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
            diagnostics.append(
                {
                    "L": length,
                    "replica": replica,
                    "seed": replica_seed,
                    "initial_sign": initial_sign,
                    "magnetization_sector_reversals": switches,
                    "mean_cluster_fraction": float(
                        np.mean(cluster_sizes) / (length * length)
                    ),
                    "mean_abs_magnetization": float(
                        np.mean(np.abs(magnetization)) / (length * length)
                    ),
                    "tau_int_abs_magnetization": (
                        integrated_autocorrelation_time(
                            np.abs(magnetization) / (length * length)
                        )
                    ),
                    "tau_int_pair_empty_liquid": (
                        integrated_autocorrelation_time(pair_empty_liquid)
                    ),
                    "bootstrap_block_length_steps": (
                        args.production_steps / args.blocks_per_replica
                    ),
                }
            )
        print(
            f"L={length}: baseline replicas={args.replicas}, sector_reversals="
            f"{sum(d['magnetization_sector_reversals'] for d in diagnostics if d['L'] == length)}"
        )

    diagnostics_frame = pd.DataFrame(diagnostics)

    raw_path = ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz"
    np.savez_compressed(raw_path, **raw)
    diagnostics_frame.to_csv(
        ROOT / "data_raw" / "lattice_gas_phase_switch_diagnostics.csv",
        index=False,
    )

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_gas_phase_switch.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "settings": vars(args),
        "model": {
            "name": "periodic square-lattice gas / square-lattice Ising model",
            "ising_coupling_K": K_ISING,
            "beta_epsilon": BETA_EPSILON,
            "coexistence_log_activity_x": X_COEX,
            "constraint": (
                "two empty nearest-neighbor sites, uniformly averaged over "
                "lattice bonds"
            ),
            "basin": (
                "liquid if M>0; gas if M<0; for M=0, "
                "liquid iff spin at site zero is +1"
            ),
            "sampler": "Wolff single-cluster sampling of both sectors at zero field",
            "downstream_reweighting": (
                "performed only by merged hierarchical analysis scripts after "
                "the additional-replica archive has been generated"
            ),
            "symmetry_variance_reduction": (
                "each configuration averaged with its global spin inverse"
            ),
        },
        "exact_thermodynamic_values": {
            "spontaneous_magnetization": exact_spontaneous_magnetization(
                K_ISING
            ),
            "nearest_neighbor_correlation": (
                exact_nearest_neighbor_correlation(K_ISING)
            ),
            "pair_empty_liquid": exact_pair_cavity_parameters(K_ISING)[0],
            "pair_empty_gas": exact_pair_cavity_parameters(K_ISING)[1],
            "delta_betaW": exact_pair_cavity_parameters(K_ISING)[2],
        },
        "quality_summary": {
            "minimum_sector_reversals_per_replica": int(
                diagnostics_frame["magnetization_sector_reversals"].min()
            ),
            "maximum_tau_int_steps": float(
                diagnostics_frame[
                    [
                        "tau_int_abs_magnetization",
                        "tau_int_pair_empty_liquid",
                    ]
                ].to_numpy().max()
            ),
            "minimum_block_length_over_tau": float(
                np.min(
                    diagnostics_frame["bootstrap_block_length_steps"]
                    / diagnostics_frame[
                        [
                            "tau_int_abs_magnetization",
                            "tau_int_pair_empty_liquid",
                        ]
                    ].max(axis=1)
                )
            ),
        },
        "provenance": {
            "source_type": "mc_new",
            "raw_trajectory_path": str(raw_path.relative_to(ROOT)),
            "diagnostics_path": (
                "data_raw/lattice_gas_phase_switch_diagnostics.csv"
            ),
            "analysis_note": (
                "this script is trajectory production only; all "
                "manuscript-facing analysis consumes both raw archives through "
                "scripts/lattice_analysis_common.py"
            ),
        },
    }
    (ROOT / "generated" / "lattice_gas_phase_switch_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote baseline lattice-gas trajectories, diagnostics, and metadata")


if __name__ == "__main__":
    main()
