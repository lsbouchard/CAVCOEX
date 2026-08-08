#!/usr/bin/env python3
"""Independent exact controls for the square-lattice cavity calculation.

This module provides two controls that do not use Monte Carlo trajectories:

1. Exact enumeration of every configuration on periodic L=2, 3, and 4
   square lattices.  The bond list contains the same right/down periodic
   bonds, including the parallel edges at L=2, as the production sampler.
   The phase convention is liquid for M>0 and, when M=0, liquid precisely
   when the spin at site zero is +1.
2. The exact axial square-Ising interface tension
   beta*tau = 2 K + log(tanh K), combined with the periodic two-interface
   Boltzmann factor exp(-2 beta*tau L).  Prefactors 1, L, and V are shown as
   sensitivity estimates, not rigorous probability or confidence bounds.

The enumerator is checked in three independent ways: vectorized microscopic
sums, a scalar direct sum written in lattice-gas variables, and the existing
production contribution/observable helpers applied to exact sufficient
statistics grouped by their zero-field Boltzmann factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Imported as scripts.lattice_exact_controls.
    from .lattice_gas_phase_switch import (
        BETA_EPSILON,
        K_ISING,
        X_COEX,
        block_contributions,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        make_blocks,
        observables_from_contributions,
        phase_theory,
    )
except ImportError:  # Executed directly from scripts/.
    from lattice_gas_phase_switch import (
        BETA_EPSILON,
        K_ISING,
        X_COEX,
        block_contributions,
        exact_pair_cavity_parameters,
        exact_spontaneous_magnetization,
        make_blocks,
        observables_from_contributions,
        phase_theory,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENUMERATION_SIZES = (2, 3, 4)
DEFAULT_PRODUCTION_SIZES = (16, 24, 32, 48, 64)
DEFAULT_H_MIN = -6.0
DEFAULT_H_MAX = 9.0
DEFAULT_H_STEP = 0.25
DIRECT_SUM_H_VALUES = (-3.0, 0.0, 4.0)
ANALYSIS_VERSION = "lattice-exact-controls-v1"


@dataclass(frozen=True)
class Microstates:
    """Microscopic sufficient statistics for all configurations at one L."""

    length: int
    codes: np.ndarray
    spins: np.ndarray
    number: np.ndarray
    magnetization: np.ndarray
    bond_sum: np.ndarray
    occupied_bonds: np.ndarray
    liquid: np.ndarray
    spatial_empty: np.ndarray
    spatial_full: np.ndarray
    fixed_empty: np.ndarray
    fixed_full: np.ndarray

    @property
    def volume(self) -> int:
        return self.length * self.length


def periodic_bonds(length: int) -> np.ndarray:
    """Return the production right/down bond list, including L=2 multiedges."""

    if length < 2:
        raise ValueError("length must be at least two")
    bonds: list[tuple[int, int]] = []
    for row in range(length):
        for col in range(length):
            site = row * length + col
            right = row * length + (col + 1) % length
            down = ((row + 1) % length) * length + col
            bonds.extend(((site, right), (site, down)))
    result = np.asarray(bonds, dtype=np.int64)
    if result.shape != (2 * length * length, 2):
        raise AssertionError("unexpected periodic bond count")
    return result


@lru_cache(maxsize=None)
def enumerate_microstates(length: int) -> Microstates:
    """Enumerate all spin configurations and exact microscopic statistics."""

    volume = length * length
    if volume > 20:
        raise ValueError("brute-force enumeration is restricted to V<=20")
    codes = np.arange(1 << volume, dtype=np.uint64)
    bit_positions = np.arange(volume, dtype=np.uint64)
    occupancy = ((codes[:, None] >> bit_positions) & 1).astype(np.int8)
    spins = (2 * occupancy - 1).astype(np.int8)
    number = np.sum(occupancy, axis=1, dtype=np.int64)
    magnetization = 2 * number - volume

    bonds = periodic_bonds(length)
    left = bonds[:, 0]
    right = bonds[:, 1]
    bond_products = spins[:, left] * spins[:, right]
    bond_sum = np.sum(bond_products, axis=1, dtype=np.int64)
    occupied_bonds = np.sum(
        occupancy[:, left] * occupancy[:, right], axis=1, dtype=np.int64
    )
    empty_bonds = np.sum(
        (1 - occupancy[:, left]) * (1 - occupancy[:, right]),
        axis=1,
        dtype=np.int64,
    )
    full_bonds = occupied_bonds
    spatial_empty = empty_bonds.astype(float) / (2.0 * volume)
    spatial_full = full_bonds.astype(float) / (2.0 * volume)
    fixed_empty = ((spins[:, 0] < 0) & (spins[:, 1] < 0)).astype(float)
    fixed_full = ((spins[:, 0] > 0) & (spins[:, 1] > 0)).astype(float)
    liquid = (magnetization > 0) | (
        (magnetization == 0) & (spins[:, 0] > 0)
    )

    # Exact microscopic identities independently audit the bond convention.
    formula_empty = (2.0 * volume - 4.0 * magnetization + bond_sum) / (
        8.0 * volume
    )
    formula_full = (2.0 * volume + 4.0 * magnetization + bond_sum) / (
        8.0 * volume
    )
    if not np.array_equal(bond_sum, 4 * occupied_bonds - 8 * number + 2 * volume):
        raise AssertionError("spin and lattice-gas bond energies disagree")
    if not np.allclose(spatial_empty, formula_empty, rtol=0.0, atol=0.0):
        raise AssertionError("empty-bond formula disagrees with direct counting")
    if not np.allclose(spatial_full, formula_full, rtol=0.0, atol=0.0):
        raise AssertionError("full-bond formula disagrees with direct counting")

    return Microstates(
        length=length,
        codes=codes,
        spins=spins,
        number=number,
        magnetization=magnetization,
        bond_sum=bond_sum,
        occupied_bonds=occupied_bonds,
        liquid=liquid,
        spatial_empty=spatial_empty,
        spatial_full=spatial_full,
        fixed_empty=fixed_empty,
        fixed_full=fixed_full,
    )


def _standard_contributions(states: Microstates, h_grid: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Form exact production-schema sums directly from microscopic arrays."""

    h_grid = np.asarray(h_grid, dtype=float)
    volume = states.volume
    base_weight = np.exp(K_ISING * states.bond_sum.astype(float))
    field_weight = np.exp(
        np.outer(h_grid, states.magnetization.astype(float) / (2.0 * volume))
    )
    weight = field_weight * base_weight[None, :]
    liquid = states.liquid
    gas = ~liquid
    number = states.number.astype(float)
    cavity = states.spatial_empty
    fixed = states.fixed_empty

    def weighted(values: np.ndarray) -> np.ndarray:
        return weight @ np.asarray(values, dtype=float)

    z = np.sum(weight, axis=1)
    n = weighted(number)
    c = weighted(cavity)
    cn = weighted(cavity * number)
    z_liquid = np.sum(weight[:, liquid], axis=1)
    z_gas = np.sum(weight[:, gas], axis=1)
    n_liquid = weight[:, liquid] @ number[liquid]
    n_gas = weight[:, gas] @ number[gas]
    c_liquid = weight[:, liquid] @ cavity[liquid]
    c_gas = weight[:, gas] @ cavity[gas]
    cn_liquid = weight[:, liquid] @ (cavity[liquid] * number[liquid])
    cn_gas = weight[:, gas] @ (cavity[gas] * number[gas])
    fixed_liquid = weight[:, liquid] @ fixed[liquid]
    fixed_gas = weight[:, gas] @ fixed[gas]
    contributions = np.column_stack(
        (
            z,
            n,
            c,
            cn,
            z_liquid,
            z_gas,
            n_liquid,
            n_gas,
            c_liquid,
            c_gas,
            cn_liquid,
            cn_gas,
            fixed_liquid,
            fixed_gas,
        )
    )
    auxiliary = {
        "weight": weight,
        "n2": weighted(number**2),
        "cn2": weighted(cavity * number**2),
        "fixed_total": weighted(fixed),
    }
    return contributions, auxiliary


def _direct_observables(
    contributions: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    volume: int,
) -> dict[str, np.ndarray]:
    """Evaluate observables without calling the production observable helper."""

    (
        z,
        n,
        c,
        cn,
        z_liquid,
        z_gas,
        n_liquid,
        n_gas,
        c_liquid,
        c_gas,
        cn_liquid,
        cn_gas,
        fixed_liquid,
        fixed_gas,
    ) = contributions.T
    mean_n = n / z
    mean_nc = cn / c
    q0_liquid = z_liquid / z
    q0_gas = z_gas / z
    q_c_liquid = c_liquid / c
    q_c_gas = c_gas / c
    mean_n_liquid = n_liquid / z_liquid
    mean_n_gas = n_gas / z_gas
    mean_nc_liquid = cn_liquid / c_liquid
    mean_nc_gas = cn_gas / c_gas
    delta_liquid = mean_nc_liquid - mean_n_liquid
    delta_gas = mean_nc_gas - mean_n_gas
    intraphase = q_c_liquid * delta_liquid + q_c_gas * delta_gas
    reweighting = (q_c_liquid - q0_liquid) * (
        mean_n_liquid - mean_n_gas
    )
    delta_n = mean_nc - mean_n
    p_cavity = c / z
    p_liquid = c_liquid / z_liquid
    p_gas = c_gas / z_gas
    beta_w = -np.log(p_cavity)
    beta_w_liquid = -np.log(p_liquid)
    beta_w_gas = -np.log(p_gas)
    delta_beta_w = beta_w_liquid - beta_w_gas
    covariance = cn / z - mean_n * p_cavity
    delta_from_covariance = covariance / p_cavity
    var_n0 = auxiliary["n2"] / z - mean_n**2
    var_nc = auxiliary["cn2"] / c - mean_nc**2
    fixed_total = auxiliary["fixed_total"] / z
    fixed_phase_liquid = fixed_liquid / z_liquid
    fixed_phase_gas = fixed_gas / z_gas

    return {
        "partition_sum_centered": z,
        "probability_sum": np.sum(auxiliary["weight"] / z[:, None], axis=1),
        "q0_liquid": q0_liquid,
        "q0_gas": q0_gas,
        "q0_sum": q0_liquid + q0_gas,
        "qC_liquid": q_c_liquid,
        "qC_gas": q_c_gas,
        "qC_sum": q_c_liquid + q_c_gas,
        "P_spatial_empty": p_cavity,
        "P_fixed_empty": fixed_total,
        "P_spatial_empty_liquid": p_liquid,
        "P_spatial_empty_gas": p_gas,
        "P_fixed_empty_liquid": fixed_phase_liquid,
        "P_fixed_empty_gas": fixed_phase_gas,
        "betaW": beta_w,
        "betaW_liquid": beta_w_liquid,
        "betaW_gas": beta_w_gas,
        "delta_betaW": delta_beta_w,
        "meanN0": mean_n,
        "meanNC": mean_nc,
        "meanN0_liquid": mean_n_liquid,
        "meanN0_gas": mean_n_gas,
        "deltaN_liquid": delta_liquid,
        "deltaN_gas": delta_gas,
        "deltaN": delta_n,
        "deltaN_intraphase": intraphase,
        "deltaN_phase_reweighting": reweighting,
        "decomposition_residual": delta_n - intraphase - reweighting,
        "deltaN_from_covariance": delta_from_covariance,
        "covariance_identity_residual": delta_n - delta_from_covariance,
        "d_betaW_dx_analytic": -delta_from_covariance,
        "d_betaW_dh_analytic": -delta_from_covariance / volume,
        "varN0": var_n0,
        "varNC": var_nc,
        "phase_log_odds": np.log(z_liquid / z_gas),
        "constrained_phase_log_odds": np.log(c_liquid / c_gas),
        "log_odds_shift_residual": (
            np.log((c_liquid / c_gas) / (z_liquid / z_gas)) + delta_beta_w
        ),
    }


def _production_helper_contributions(states: Microstates, h_grid: np.ndarray) -> np.ndarray:
    """Apply production block helpers to one representative per spin orbit."""

    representative = states.liquid
    magnetization = states.magnetization[representative].astype(np.int32)
    bond_sum = states.bond_sum[representative].astype(np.int32)
    site_zero = states.spins[representative, 0].astype(np.int8)
    site_right = states.spins[representative, 1].astype(np.int8)
    total = np.zeros((len(h_grid), 14), dtype=float)

    # K*bond_sum is constant within each group, so the production helper's
    # unweighted observation sum can be multiplied by the exact base weight.
    for bond_value in np.unique(bond_sum):
        selected = bond_sum == bond_value
        block = make_blocks(
            magnetization[selected],
            bond_sum[selected],
            site_zero[selected],
            site_right[selected],
            states.volume,
            replica=0,
            block_count=1,
        )[0]
        total += np.exp(K_ISING * float(bond_value)) * block_contributions(
            block, states.volume, np.asarray(h_grid, dtype=float)
        )
    return total


def exact_enumeration(length: int, h_grid: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return exact finite-size observables and cross-implementation diagnostics."""

    states = enumerate_microstates(length)
    h_grid = np.asarray(h_grid, dtype=float)
    contributions, auxiliary = _standard_contributions(states, h_grid)
    observed = _direct_observables(contributions, auxiliary, states.volume)

    helper_contributions = _production_helper_contributions(states, h_grid)
    relative_contribution_error = np.max(
        np.abs(helper_contributions - contributions)
        / np.maximum(1.0, np.abs(contributions))
    )
    helper_observed = observables_from_contributions(
        helper_contributions, states.volume
    )
    shared_keys = (
        "meanN0",
        "meanNC",
        "deltaN",
        "q0_liquid",
        "qC_liquid",
        "deltaN_liquid",
        "deltaN_gas",
        "deltaN_intraphase",
        "deltaN_phase_reweighting",
        "betaW",
        "delta_betaW",
        "P_fixed_empty_liquid",
        "P_fixed_empty_gas",
        "P_spatial_empty_liquid",
        "P_spatial_empty_gas",
    )
    helper_observable_error = max(
        float(np.max(np.abs(helper_observed[key] - observed[key])))
        for key in shared_keys
    )

    rows: list[dict[str, float | int | str]] = []
    for index, h_value in enumerate(h_grid):
        row: dict[str, float | int | str] = {
            "L": length,
            "V": states.volume,
            "configurations": len(states.codes),
            "h": float(h_value),
            "delta_x": float(h_value / states.volume),
            "x": float(X_COEX + h_value / states.volume),
            "source_type": "exact_enumeration",
            "analysis_version": ANALYSIS_VERSION,
            "production_helper_max_relative_contribution_error": float(
                relative_contribution_error
            ),
            "production_helper_max_abs_observable_error": float(
                helper_observable_error
            ),
        }
        row.update({key: float(value[index]) for key, value in observed.items()})
        row["fixed_minus_spatial_empty"] = (
            row["P_fixed_empty"] - row["P_spatial_empty"]
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    diagnostics = {
        "maximum_abs_decomposition_residual": float(
            np.max(np.abs(frame["decomposition_residual"]))
        ),
        "maximum_abs_covariance_identity_residual": float(
            np.max(np.abs(frame["covariance_identity_residual"]))
        ),
        "maximum_abs_log_odds_shift_residual": float(
            np.max(np.abs(frame["log_odds_shift_residual"]))
        ),
        "maximum_abs_probability_normalization_residual": float(
            np.max(np.abs(frame["probability_sum"] - 1.0))
        ),
        "maximum_abs_fixed_minus_spatial_empty": float(
            np.max(np.abs(frame["fixed_minus_spatial_empty"]))
        ),
        "production_helper_max_relative_contribution_error": float(
            relative_contribution_error
        ),
        "production_helper_max_abs_observable_error": float(
            helper_observable_error
        ),
    }
    return frame, diagnostics


def exact_observables_at_h(length: int, h_value: float) -> dict[str, float]:
    """Convenience evaluator for derivative and independent-sum tests."""

    frame, _ = exact_enumeration(length, np.asarray([h_value], dtype=float))
    row = frame.iloc[0]
    return {
        key: float(value)
        for key, value in row.items()
        if isinstance(value, (float, int, np.floating, np.integer))
    }


def scalar_lattice_gas_direct_sum(length: int, h_value: float) -> dict[str, float]:
    """Independent scalar sum using beta*epsilon bonds + x*N directly.

    This intentionally does not call ``enumerate_microstates`` or use the
    Ising sufficient-statistic formula for the empty-pair fraction.
    """

    volume = length * length
    bonds = periodic_bonds(length)
    x_value = X_COEX + h_value / volume
    accumulators = np.zeros(14, dtype=float)
    probability_weighted_n2 = 0.0
    cavity_weighted_n2 = 0.0
    fixed_total = 0.0

    for code in range(1 << volume):
        occupancy = np.asarray(
            [(code >> site) & 1 for site in range(volume)], dtype=np.int8
        )
        number = int(np.sum(occupancy))
        magnetization = 2 * number - volume
        occupied_bonds = 0
        empty_bonds = 0
        for left, right in bonds:
            occupied_bonds += int(occupancy[left] * occupancy[right])
            empty_bonds += int((1 - occupancy[left]) * (1 - occupancy[right]))
        cavity = empty_bonds / (2.0 * volume)
        fixed = float(occupancy[0] == 0 and occupancy[1] == 0)
        liquid = magnetization > 0 or (
            magnetization == 0 and occupancy[0] == 1
        )
        weight = float(
            np.exp(BETA_EPSILON * occupied_bonds + x_value * number)
        )
        c_weight = weight * cavity
        accumulators[0] += weight
        accumulators[1] += weight * number
        accumulators[2] += c_weight
        accumulators[3] += c_weight * number
        branch = 4 if liquid else 5
        accumulators[branch] += weight
        accumulators[branch + 2] += weight * number
        accumulators[branch + 4] += c_weight
        accumulators[branch + 6] += c_weight * number
        accumulators[12 if liquid else 13] += weight * fixed
        probability_weighted_n2 += weight * number**2
        cavity_weighted_n2 += c_weight * number**2
        fixed_total += weight * fixed

    contributions = accumulators[None, :]
    auxiliary = {
        "weight": np.ones((1, 1), dtype=float),
        "n2": np.asarray([probability_weighted_n2]),
        "cn2": np.asarray([cavity_weighted_n2]),
        "fixed_total": np.asarray([fixed_total]),
    }
    observed = _direct_observables(contributions, auxiliary, volume)
    # ``probability_sum`` above cannot be obtained from the compressed scalar
    # accumulator, so replace its placeholder by the exact normalized total.
    observed["probability_sum"] = np.asarray([1.0])
    return {key: float(value[0]) for key, value in observed.items()}


def independent_direct_sum_error() -> float:
    """Maximum observable discrepancy for documented independent spot checks."""

    keys = (
        "q0_liquid",
        "qC_liquid",
        "P_spatial_empty",
        "P_fixed_empty",
        "P_spatial_empty_liquid",
        "P_spatial_empty_gas",
        "delta_betaW",
        "meanN0",
        "meanNC",
        "deltaN",
        "deltaN_intraphase",
        "deltaN_phase_reweighting",
    )
    maximum = 0.0
    for length in (2, 3):
        for h_value in DIRECT_SUM_H_VALUES:
            vectorized = exact_observables_at_h(length, h_value)
            scalar = scalar_lattice_gas_direct_sum(length, h_value)
            maximum = max(
                maximum,
                *(abs(vectorized[key] - scalar[key]) for key in keys),
            )
    return float(maximum)


def interface_weight_estimates(lengths: tuple[int, ...]) -> pd.DataFrame:
    """Return exact exponential scale and transparent prefactor sensitivity."""

    beta_tau = float(2.0 * K_ISING + np.log(np.tanh(K_ISING)))
    records: list[dict[str, float | int | str]] = []
    for length in lengths:
        volume = length * length
        exponent = 2.0 * beta_tau * length
        bare = float(np.exp(-exponent))
        for label, prefactor in (("1", 1.0), ("L", float(length)), ("V", float(volume))):
            relative = prefactor * bare
            probability = relative / (1.0 + relative)
            response_scale = min(1.0, 2.0 * probability)
            records.append(
                {
                    "L": length,
                    "V": volume,
                    "K": K_ISING,
                    "beta_tau_axial": beta_tau,
                    "two_interface_exponent": exponent,
                    "bare_two_interface_factor": bare,
                    "prefactor_label": label,
                    "prefactor_value": prefactor,
                    "relative_interface_weight_estimate": relative,
                    "interface_probability_scale_estimate": probability,
                    "worst_scale_normalized_response_estimate": response_scale,
                    "estimate_status": (
                        "prefactor-sensitivity scale; not a rigorous bound or "
                        "confidence interval"
                    ),
                    "analysis_version": ANALYSIS_VERSION,
                }
            )
    return pd.DataFrame(records)


def make_figure(
    exact_frame: pd.DataFrame,
    interface_frame: pd.DataFrame,
    output: Path,
) -> None:
    """Create a compact visual summary of the two independent controls."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    colors = {2: "#0072B2", 3: "#009E73", 4: "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), constrained_layout=True)

    response_axis, interface_axis = axes
    for length, group in exact_frame.groupby("L"):
        response_axis.plot(
            group["h"],
            -group["deltaN_phase_reweighting"] / group["V"],
            color=colors[int(length)],
            linewidth=1.25,
            marker="o",
            markersize=2.5,
            markevery=6,
            label=rf"exact $L={int(length)}$",
        )
    h_dense = np.linspace(DEFAULT_H_MIN, DEFAULT_H_MAX, 600)
    exact_magnetization = exact_spontaneous_magnetization(K_ISING)
    exact_delta_w = exact_pair_cavity_parameters(K_ISING)[2]
    response_axis.plot(
        h_dense,
        -phase_theory(h_dense, exact_magnetization, exact_delta_w),
        color="black",
        linestyle="--",
        linewidth=1.05,
        label="thermodynamic limit",
    )
    response_axis.set_xlabel(r"$h=V(x-x_{\rm coex})$")
    response_axis.set_ylabel(r"$-\Delta N_{\rm rw}/V$")
    response_axis.set_title("Exact finite-lattice response")
    response_axis.set_xlim(DEFAULT_H_MIN, DEFAULT_H_MAX)
    response_axis.set_ylim(bottom=-0.015)
    response_axis.grid(alpha=0.18, linewidth=0.5)
    response_axis.legend(frameon=False, loc="upper left")

    prefactor_colors = {"1": "#0072B2", "L": "#E69F00", "V": "#CC79A7"}
    for label, group in interface_frame.groupby("prefactor_label", sort=False):
        group = group.sort_values("L")
        color = prefactor_colors[str(label)]
        interface_axis.semilogy(
            group["L"],
            group["relative_interface_weight_estimate"],
            color=color,
            marker="o",
            linewidth=1.15,
            label=rf"$A={label}$",
        )
        interface_axis.semilogy(
            group["L"],
            group["worst_scale_normalized_response_estimate"],
            color=color,
            marker="s",
            fillstyle="none",
            linestyle="--",
            linewidth=0.9,
        )
    interface_axis.plot(
        [],
        [],
        color="0.25",
        marker="s",
        fillstyle="none",
        linestyle="--",
        linewidth=0.9,
        label="response scale",
    )
    interface_axis.set_xlabel(r"$L$")
    interface_axis.set_ylabel("estimated scale")
    interface_axis.set_title("Periodic two-interface estimate")
    interface_axis.grid(alpha=0.18, linewidth=0.5, which="both")
    interface_axis.legend(frameon=False, ncol=2, loc="upper right")

    for label, axis in zip("ab", axes):
        axis.text(
            -0.12,
            1.035,
            f"({label})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )
    fig.savefig(
        output,
        metadata={
            "Title": "Exact lattice-gas and interface-weight controls",
            "Author": "Louis-S. Bouchard",
            "Creator": "scripts/lattice_exact_controls.py",
            "Subject": "Independent exact controls for cavity-induced phase reweighting",
        },
    )
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _h_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    if step <= 0.0 or maximum <= minimum:
        raise ValueError("invalid h-grid bounds")
    intervals = int(round((maximum - minimum) / step))
    if not np.isclose(minimum + intervals * step, maximum):
        raise ValueError("h range must be an integer multiple of h step")
    return np.linspace(minimum, maximum, intervals + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-min", type=float, default=DEFAULT_H_MIN)
    parser.add_argument("--h-max", type=float, default=DEFAULT_H_MAX)
    parser.add_argument("--h-step", type=float, default=DEFAULT_H_STEP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h_grid = _h_grid(args.h_min, args.h_max, args.h_step)
    exact_frames: list[pd.DataFrame] = []
    enumeration_diagnostics: dict[str, dict[str, float]] = {}
    for length in DEFAULT_ENUMERATION_SIZES:
        frame, diagnostics = exact_enumeration(length, h_grid)
        exact_frames.append(frame)
        enumeration_diagnostics[str(length)] = diagnostics
    exact_frame = pd.concat(exact_frames, ignore_index=True)
    interface_frame = interface_weight_estimates(DEFAULT_PRODUCTION_SIZES)

    exact_path = ROOT / "data_processed" / "lattice_exact_finite_size.csv"
    interface_path = ROOT / "data_processed" / "lattice_interface_weight_estimate.csv"
    figure_path = ROOT / "figures" / "lattice_exact_finite_size_control.pdf"
    metadata_path = ROOT / "generated" / "lattice_exact_controls_metadata.json"
    exact_frame.to_csv(exact_path, index=False)
    interface_frame.to_csv(interface_path, index=False)
    make_figure(exact_frame, interface_frame, figure_path)

    scalar_error = independent_direct_sum_error()
    beta_tau = float(2.0 * K_ISING + np.log(np.tanh(K_ISING)))
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "script": "scripts/lattice_exact_controls.py",
        "script_sha256": file_sha256(Path(__file__)),
        "dependency_sha256": {
            "scripts/lattice_gas_phase_switch.py": file_sha256(
                ROOT / "scripts" / "lattice_gas_phase_switch.py"
            )
        },
        "reproduce_command": "python3 scripts/lattice_exact_controls.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "model": {
            "name": "periodic square-lattice gas / square-lattice Ising model",
            "K": K_ISING,
            "beta_epsilon": BETA_EPSILON,
            "x_coexistence": X_COEX,
            "bond_counting": (
                "right and down bonds from every site; 2V bonds including "
                "parallel periodic edges at L=2"
            ),
            "basin_rule": "liquid iff M>0 or (M=0 and s_0=+1)",
            "cavity": "empty nearest-neighbor bond uniformly averaged over 2V bonds",
            "reweighting": "x=x_coex+h/V; centered factor exp[h M/(2V)]",
        },
        "enumeration": {
            "sizes": list(DEFAULT_ENUMERATION_SIZES),
            "configuration_counts": {
                str(length): 1 << (length * length)
                for length in DEFAULT_ENUMERATION_SIZES
            },
            "h_grid": {
                "minimum": float(h_grid[0]),
                "maximum": float(h_grid[-1]),
                "step": float(h_grid[1] - h_grid[0]),
                "points": len(h_grid),
            },
            "independent_scalar_direct_sum": {
                "sizes": [2, 3],
                "h_values": list(DIRECT_SUM_H_VALUES),
                "maximum_abs_observable_error": scalar_error,
                "implementation": (
                    "scalar lattice-gas sum exp[beta*epsilon*n_bonds+x*N], "
                    "with explicit bond and empty-bond loops"
                ),
            },
            "production_helper_comparison": (
                "one representative per spin-inversion orbit, grouped by "
                "bond sum and passed through make_blocks, block_contributions, "
                "and observables_from_contributions"
            ),
            "diagnostics_by_L": enumeration_diagnostics,
        },
        "interface_estimate": {
            "production_sizes": list(DEFAULT_PRODUCTION_SIZES),
            "beta_tau_axial": beta_tau,
            "formula": "beta*tau=2K+ln(tanh K)",
            "periodic_factor": "exp(-2*beta*tau*L)",
            "prefactors": ["1", "L", "V"],
            "response_scale_formula": (
                "if r=A exp(-2 beta tau L), report min[1,2r/(1+r)]"
            ),
            "status": (
                "transparent exponential and prefactor-sensitivity estimates; "
                "not rigorous bounds or confidence intervals"
            ),
        },
        "outputs_sha256": {
            str(exact_path.relative_to(ROOT)): file_sha256(exact_path),
            str(interface_path.relative_to(ROOT)): file_sha256(interface_path),
            str(figure_path.relative_to(ROOT)): file_sha256(figure_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    maximum_closure = max(
        value["maximum_abs_decomposition_residual"]
        for value in enumeration_diagnostics.values()
    )
    maximum_helper = max(
        value["production_helper_max_abs_observable_error"]
        for value in enumeration_diagnostics.values()
    )
    print(
        f"enumerated L={DEFAULT_ENUMERATION_SIZES}; "
        f"max closure={maximum_closure:.3e}; "
        f"max helper error={maximum_helper:.3e}; "
        f"independent scalar error={scalar_error:.3e}; beta_tau={beta_tau:.12f}"
    )


if __name__ == "__main__":
    main()
