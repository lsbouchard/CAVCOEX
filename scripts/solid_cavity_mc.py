#!/usr/bin/env python3
"""Canonical hard-sphere cavity probabilities in a defect-free FCC basin.

This calculation complements ``cavity_mc.py`` with genuine particle-resolved
trajectories initialized in a commensurate FCC crystal.  The density is the
thermodynamic-limit solid coexistence density transcribed from Fernandez
et al., Phys. Rev. Lett. 108, 165701 (2012).  Single-particle Metropolis moves
are performed at fixed N and V; every production block is audited with global
Q6, FCC Bragg amplitude, and lattice-referenced Lindemann displacement.

Scope is deliberately narrow.  These are phase-conditioned canonical solid
data.  There is no grand-canonical sampling, phase switch, cross-basin
normalization, or chemical-potential reweighting, so these outputs do not
complete manuscript task N4.

Reduced units are beta = sigma = Lambda = 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig")
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numba
import numpy as np
import pandas as pd
import scipy

from cavity_mc import (
    EXACT_CONTROL_Z_LIMIT,
    advance_sweeps,
    box_length,
    fcc_positions,
    global_q6,
    maximum_fcc_bragg_amplitude,
    minimum_pair_distance,
    production_blocks,
    summarize_block_probability,
)


DEFAULT_SIZES = (108, 256)
BASE_SEED = 92_026_072
ANALYSIS_VERSION = "solid-cavity-mc-v2"
SOURCE_RELATIVE_PATH = "data_raw/fernandez2012_coexistence.csv"
SOURCE_DOI = "10.1103/PhysRevLett.108.165701"
SOLID_RADII = np.asarray(
    [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    dtype=np.float64,
)

# Prespecified crystal-basin and statistical reporting gates.
Q6_CRYSTAL_MINIMUM = 0.35
FCC_BRAGG_CRYSTAL_MINIMUM = 0.35
LINDEMANN_MAXIMUM = 0.25
MINIMUM_PRODUCTION_ACCEPTANCE = 0.08
MAXIMUM_PRODUCTION_ACCEPTANCE = 0.65
MAX_DISPLACEMENT_CAP = 0.16
CONTACT_SHELL_WIDTH = 0.02
CONTACT_SHELL_SENSITIVITY_WIDTHS = (0.01, 0.02, 0.03)


@dataclass(frozen=True)
class SolidRunSettings:
    equil_sweeps: int
    tune_sweeps: int
    tune_chunk_sweeps: int
    audit_chunk_sweeps: int
    production_blocks: int
    sweeps_per_block: int
    sample_stride: int
    probes_per_sample: int

    @property
    def production_sweeps(self) -> int:
        return self.production_blocks * self.sweeps_per_block

    @property
    def samples_per_block(self) -> int:
        return self.sweeps_per_block // self.sample_stride

    @property
    def trials_per_block(self) -> int:
        return self.samples_per_block * self.probes_per_sample

    @property
    def probe_trials_per_state(self) -> int:
        return self.production_blocks * self.trials_per_block


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_solid_coexistence_density(source_path: Path) -> dict[str, Any]:
    """Read the published thermodynamic-limit solid specific volume."""

    table = pd.read_csv(source_path, dtype={"N": str})
    selected = table.loc[table["N"].str.lower() == "infinity"]
    if len(selected) != 1:
        raise ValueError("Expected one thermodynamic-limit source row.")
    row = selected.iloc[0]
    if str(row["doi"]) != SOURCE_DOI:
        raise ValueError("Unexpected coexistence-density source DOI.")
    specific_volume = float(row["v_solid"])
    se_specific_volume = float(row["se_v_solid"])
    eta = float(np.pi / (6.0 * specific_volume))
    se_eta = float(eta * se_specific_volume / specific_volume)
    return {
        "eta": eta,
        "se_eta": se_eta,
        "specific_volume": specific_volume,
        "se_specific_volume": se_specific_volume,
        "pressure": float(row["p_equal_weight"]),
        "se_pressure": float(row["se_p_equal_weight"]),
        "doi": str(row["doi"]),
        "citation": str(row["source"]),
    }


def lattice_displacements(
    positions: np.ndarray, reference: np.ndarray, length: float
) -> np.ndarray:
    """Site-referenced displacements after removing periodic global drift."""

    displacement = positions - reference
    displacement -= length * np.rint(displacement / length)
    global_shift = np.empty(3, dtype=np.float64)
    for axis in range(3):
        angle = 2.0 * np.pi * displacement[:, axis] / length
        global_shift[axis] = (
            length
            / (2.0 * np.pi)
            * math.atan2(float(np.mean(np.sin(angle))), float(np.mean(np.cos(angle))))
        )
    displacement -= global_shift
    displacement -= length * np.rint(displacement / length)
    return displacement


def lindemann_ratio(
    positions: np.ndarray,
    reference: np.ndarray,
    length: float,
    n_particles: int,
) -> tuple[float, float]:
    """Return RMS and maximum site displacement divided by FCC nearest spacing."""

    cells = int(round((n_particles / 4.0) ** (1.0 / 3.0)))
    nearest_neighbor_spacing = length / (cells * np.sqrt(2.0))
    displacements = lattice_displacements(positions, reference, length)
    magnitudes = np.sqrt(np.sum(displacements * displacements, axis=1))
    rms = float(np.sqrt(np.mean(magnitudes * magnitudes)))
    maximum = float(np.max(magnitudes))
    return rms / nearest_neighbor_spacing, maximum / nearest_neighbor_spacing


def contact_shell_pressure(
    positions: np.ndarray,
    length: float,
    shell_width: float = CONTACT_SHELL_WIDTH,
) -> tuple[float, float, int]:
    """Return a near-contact hard-sphere pressure diagnostic.

    The estimator averages the radial distribution function over the shell
    ``1 <= r < 1 + shell_width`` and substitutes that shell average for
    ``g(1+)`` in the hard-sphere virial relation

        beta p = rho [1 + 4 eta g(1+)].

    A finite shell is deliberately used instead of presenting a noisy
    zero-width extrapolation as an exact pressure measurement.  Replica
    outputs therefore label this quantity as a contact-shell diagnostic and
    report its shell-width sensitivity.
    """

    if shell_width <= 0.0:
        raise ValueError("shell_width must be positive")
    n_particles = int(positions.shape[0])
    volume = float(length**3)
    rho = n_particles / volume
    eta = np.pi * rho / 6.0
    lower = 1.0
    upper = 1.0 + float(shell_width)
    shell_pairs = 0
    for index in range(n_particles - 1):
        delta = positions[index + 1 :] - positions[index]
        delta -= length * np.rint(delta / length)
        distances_squared = np.sum(delta * delta, axis=1)
        shell_pairs += int(
            np.count_nonzero(
                (distances_squared >= lower**2)
                & (distances_squared < upper**2)
            )
        )
    shell_volume = 4.0 * np.pi / 3.0 * (upper**3 - lower**3)
    ideal_pair_count = 0.5 * n_particles * rho * shell_volume
    shell_g = float(shell_pairs / ideal_pair_count)
    beta_pressure = float(rho * (1.0 + 4.0 * eta * shell_g))
    return beta_pressure, shell_g, shell_pairs


def structure_snapshot(
    positions: np.ndarray,
    reference: np.ndarray,
    length: float,
    n_particles: int,
) -> dict[str, float | int]:
    q6, bonds = global_q6(positions, length)
    bragg = maximum_fcc_bragg_amplitude(positions, length, n_particles)
    lindemann, maximum_site_displacement = lindemann_ratio(
        positions, reference, length, n_particles
    )
    pressure_by_width = {
        width: contact_shell_pressure(positions, length, width)
        for width in CONTACT_SHELL_SENSITIVITY_WIDTHS
    }
    pressure, contact_g, contact_pairs = pressure_by_width[CONTACT_SHELL_WIDTH]
    return {
        "global_Q6": q6,
        "Q6_bonds": bonds,
        "max_FCC_Bragg": bragg,
        "lindemann_ratio": lindemann,
        "max_site_displacement_over_nn": maximum_site_displacement,
        "minimum_pair_distance": minimum_pair_distance(positions, length),
        "beta_pressure_contact_shell": pressure,
        "contact_shell_g": contact_g,
        "contact_shell_pair_count": contact_pairs,
        "beta_pressure_contact_shell_delta_0p01": pressure_by_width[0.01][0],
        "beta_pressure_contact_shell_delta_0p02": pressure_by_width[0.02][0],
        "beta_pressure_contact_shell_delta_0p03": pressure_by_width[0.03][0],
    }


def structure_gate(frame: pd.DataFrame) -> tuple[bool, list[str]]:
    """Apply crystal-basin gates to every post-initial audited snapshot."""

    reasons: list[str] = []
    if float(frame["global_Q6"].min()) < Q6_CRYSTAL_MINIMUM:
        reasons.append("Q6_below_crystal_threshold")
    if float(frame["max_FCC_Bragg"].min()) < FCC_BRAGG_CRYSTAL_MINIMUM:
        reasons.append("Bragg_below_crystal_threshold")
    if float(frame["lindemann_ratio"].max()) > LINDEMANN_MAXIMUM:
        reasons.append("Lindemann_above_crystal_threshold")
    if float(frame["minimum_pair_distance"].min()) < 1.0 - 2.0e-12:
        reasons.append("hard_sphere_overlap")
    return not reasons, reasons


def equilibrate_solid(
    positions: np.ndarray,
    reference: np.ndarray,
    length: float,
    n_particles: int,
    n_cells: int,
    settings: SolidRunSettings,
    seed: int,
) -> tuple[float, list[dict[str, Any]], int, int]:
    """Conservative burn-in with a frozen post-tuning displacement."""

    if settings.tune_sweeps > settings.equil_sweeps:
        raise ValueError("tune_sweeps cannot exceed equil_sweeps")
    max_displacement = 0.075
    completed = 0
    accepted_total = 0
    attempted_total = 0
    trace: list[dict[str, Any]] = []
    chunk_index = 0
    while completed < settings.equil_sweeps:
        if completed < settings.tune_sweeps:
            chunk = min(
                settings.tune_chunk_sweeps,
                settings.tune_sweeps - completed,
            )
            stage = "adaptive_burn_in"
        else:
            chunk = min(
                settings.audit_chunk_sweeps,
                settings.equil_sweeps - completed,
            )
            stage = "fixed_step_burn_in"
        accepted, attempted = advance_sweeps(
            positions,
            length,
            n_cells,
            max_displacement,
            chunk,
            seed + 17_171 * chunk_index,
        )
        acceptance = accepted / attempted
        completed += chunk
        accepted_total += accepted
        attempted_total += attempted
        snapshot = structure_snapshot(
            positions, reference, length, n_particles
        )
        trace.append(
            {
                "stage": stage,
                "chunk": chunk_index,
                "sweeps": chunk,
                "cumulative_sweeps": completed,
                "max_displacement_used": max_displacement,
                "move_acceptance": acceptance,
                **snapshot,
            }
        )
        if stage == "adaptive_burn_in":
            target_acceptance = 0.30
            max_displacement *= math.exp(0.65 * (acceptance - target_acceptance))
            max_displacement = float(
                np.clip(max_displacement, 0.012, MAX_DISPLACEMENT_CAP)
            )
        chunk_index += 1
    return max_displacement, trace, accepted_total, attempted_total


def state_seeds(base_seed: int, size_index: int) -> tuple[int, int]:
    offset = 1_000_003 * size_index
    return base_seed + offset + 10_000_000, base_seed + offset + 20_000_000


def run_state(
    n_particles: int,
    eta: float,
    eta_source: dict[str, Any],
    size_index: int,
    settings: SolidRunSettings,
    base_seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
]:
    """Generate one FCC-basin trajectory and its block-level cavity data."""

    length = box_length(n_particles, eta)
    volume = length**3
    n_cells = int(math.floor(length))
    if n_cells < 3 or length / n_cells < float(np.max(SOLID_RADII)):
        raise ValueError("Linked cells are not safe for the requested radii.")
    reference = fcc_positions(n_particles, length)
    positions = reference.copy()
    initial = structure_snapshot(positions, reference, length, n_particles)
    equil_seed, production_seed = state_seeds(base_seed, size_index)
    run_id = (
        f"{ANALYSIS_VERSION}_N{n_particles}_eta{eta:.6f}_seed{production_seed}"
    )
    start = time.perf_counter()
    (
        max_displacement,
        equil_trace,
        equil_accepted,
        equil_attempted,
    ) = equilibrate_solid(
        positions,
        reference,
        length,
        n_particles,
        n_cells,
        settings,
        equil_seed,
    )

    block_hits = np.zeros(
        (settings.production_blocks, SOLID_RADII.size), dtype=np.int64
    )
    block_accepts = np.zeros(settings.production_blocks, dtype=np.int64)
    block_attempts = np.zeros(settings.production_blocks, dtype=np.int64)
    raw_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    for block in range(settings.production_blocks):
        hits, accepts, attempts = production_blocks(
            positions,
            length,
            n_cells,
            max_displacement,
            1,
            settings.sweeps_per_block,
            settings.sample_stride,
            settings.probes_per_sample,
            SOLID_RADII**2,
            production_seed + 104_729 * block,
        )
        block_hits[block] = hits[0]
        block_accepts[block] = accepts[0]
        block_attempts[block] = attempts[0]
        snapshot = structure_snapshot(
            positions, reference, length, n_particles
        )
        structure_rows.append(
            {
                "source_type": "mc_new",
                "run_id": run_id,
                "seed": production_seed,
                "N": n_particles,
                "eta": eta,
                "stage": "production",
                "block": block,
                "cumulative_sweeps": (block + 1) * settings.sweeps_per_block,
                **snapshot,
            }
        )
        raw_row: dict[str, Any] = {
            "source_type": "mc_new",
            "run_id": run_id,
            "seed": production_seed,
            "N": n_particles,
            "V": volume,
            "L": length,
            "eta": eta,
            "block": block,
            "probe_trials": settings.trials_per_block,
            "move_accepts": int(accepts[0]),
            "move_attempts": int(attempts[0]),
            "move_acceptance": float(accepts[0] / attempts[0]),
        }
        for radius_index, radius in enumerate(SOLID_RADII):
            label = f"R{radius:.2f}".replace(".", "p")
            raw_row[f"empty_events_{label}"] = int(hits[0, radius_index])
            raw_row[f"P_empty_{label}"] = float(
                hits[0, radius_index] / settings.trials_per_block
            )
        raw_rows.append(raw_row)

    elapsed = time.perf_counter() - start
    structure_frame = pd.DataFrame(structure_rows)
    # The basin gate covers both the fixed/tuned burn-in trace and every
    # production block; the analytic t=0 initializer is reported separately.
    all_audited_structure = pd.DataFrame(equil_trace + structure_rows)
    crystal_pass, crystal_reasons = structure_gate(all_audited_structure)
    production_acceptance = float(
        np.sum(block_accepts) / np.sum(block_attempts)
    )
    acceptance_pass = bool(
        MINIMUM_PRODUCTION_ACCEPTANCE
        <= production_acceptance
        <= MAXIMUM_PRODUCTION_ACCEPTANCE
    )
    processed_rows: list[dict[str, Any]] = []
    for radius_index, radius in enumerate(SOLID_RADII):
        summary = summarize_block_probability(
            block_hits[:, radius_index], settings.trials_per_block
        )
        statistical_reliable = bool(summary["reliable_direct"])
        statistical_status = str(summary["reliability_status"])
        exact_probability = 1.0 - eta if np.isclose(radius, 0.50) else np.nan
        exact_residual = (
            float(summary["P_empty"] - exact_probability)
            if np.isfinite(exact_probability)
            else np.nan
        )
        exact_z = (
            float(exact_residual / summary["se_P_empty"])
            if np.isfinite(exact_probability) and summary["se_P_empty"] > 0.0
            else np.nan
        )
        exact_control_pass = bool(
            not np.isfinite(exact_probability)
            or abs(exact_z) <= EXACT_CONTROL_Z_LIMIT
        )
        combined_reasons: list[str] = []
        if not statistical_reliable:
            combined_reasons.append(statistical_status)
        if not crystal_pass:
            combined_reasons.extend(crystal_reasons)
        if not acceptance_pass:
            combined_reasons.append("production_acceptance_outside_gate")
        if not exact_control_pass:
            combined_reasons.append("exact_R0p5_control_failed")
        reliable = not combined_reasons
        summary["reliable_direct"] = reliable
        summary["reliability_status"] = (
            "reliable_direct" if reliable else ";".join(combined_reasons)
        )
        processed_rows.append(
            {
                "source_type": "mc_new",
                "run_id": run_id,
                "analysis_version": ANALYSIS_VERSION,
                "seed": production_seed,
                "equil_seed": equil_seed,
                "units_convention": "beta=sigma=Lambda=1",
                "raw_data_path": "data_raw/solid_cavity_mc_blocks.csv",
                "ensemble": "canonical_NVT",
                "phase_scope": "defect_free_commensurate_FCC_solid_basin",
                "N4_phase_switch_data": False,
                "cross_basin_normalization": False,
                "N": n_particles,
                "V": volume,
                "L": length,
                "eta": eta,
                "se_eta_source": eta_source["se_eta"],
                "eta_source_specific_volume": eta_source["specific_volume"],
                "eta_source_doi": eta_source["doi"],
                "R_over_sigma": radius,
                "n_cells_per_dimension": n_cells,
                "cell_width": length / n_cells,
                "equil_sweeps": settings.equil_sweeps,
                "production_sweeps": settings.production_sweeps,
                "sample_stride_sweeps": settings.sample_stride,
                "production_samples": (
                    settings.production_blocks * settings.samples_per_block
                ),
                "probes_per_sample": settings.probes_per_sample,
                "n_blocks": settings.production_blocks,
                **summary,
                "statistical_reliable_direct": statistical_reliable,
                "statistical_reliability_status": statistical_status,
                "equil_acceptance": equil_accepted / equil_attempted,
                "production_acceptance": production_acceptance,
                "max_displacement": max_displacement,
                "crystal_structure_gate": crystal_pass,
                "crystal_structure_status": (
                    "crystal_retained"
                    if crystal_pass
                    else ";".join(crystal_reasons)
                ),
                "minimum_audited_Q6": float(
                    all_audited_structure["global_Q6"].min()
                ),
                "minimum_audited_FCC_Bragg": float(
                    all_audited_structure["max_FCC_Bragg"].min()
                ),
                "maximum_audited_lindemann": float(
                    all_audited_structure["lindemann_ratio"].max()
                ),
                "minimum_audited_pair_distance": float(
                    all_audited_structure["minimum_pair_distance"].min()
                ),
                "acceptance_gate": acceptance_pass,
                "exact_P_empty": exact_probability,
                "exact_residual": exact_residual,
                "exact_z": exact_z,
                "exact_control_pass": exact_control_pass,
                "elapsed_seconds_state": elapsed,
            }
        )

    diagnostic_rows: list[dict[str, Any]] = [
        {
            "source_type": "analytic_initializer",
            "run_id": run_id,
            "seed": equil_seed,
            "N": n_particles,
            "eta": eta,
            "stage": "initial_FCC",
            "chunk": -1,
            "sweeps": 0,
            "cumulative_sweeps": 0,
            "max_displacement_used": 0.0,
            "move_acceptance": np.nan,
            **initial,
        }
    ]
    diagnostic_rows.extend(
        {
            "source_type": "mc_new",
            "run_id": run_id,
            "seed": equil_seed,
            "N": n_particles,
            "eta": eta,
            **row,
        }
        for row in equil_trace
    )
    diagnostic_rows.extend(structure_rows)
    return processed_rows, raw_rows, diagnostic_rows, positions.copy()


def make_figure(
    processed: pd.DataFrame,
    raw: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output: Path,
) -> None:
    """Write a manuscript-ready vector figure with statistical/phase audits."""

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
    colors = {108: "#0072B2", 256: "#D55E00"}
    markers = {108: "o", 256: "s"}
    horizontal_offsets = {108: -0.0035, 256: 0.0035}
    fig = plt.figure(figsize=(7.15, 5.05), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.42, 1.0))
    work_axis = fig.add_subplot(grid[0, :])
    convergence_axis = fig.add_subplot(grid[1, 0])
    structure_axis = fig.add_subplot(grid[1, 1])

    for n_particles in sorted(processed["N"].unique()):
        group = processed[processed["N"] == n_particles].sort_values(
            "R_over_sigma"
        )
        reliable = group["reliable_direct"].astype(bool)
        finite = np.isfinite(group["betaW_N"])
        selected = reliable & finite
        radius_values = (
            group.loc[selected, "R_over_sigma"]
            + horizontal_offsets[int(n_particles)]
        )
        work_axis.errorbar(
            radius_values,
            group.loc[selected, "betaW_N"],
            yerr=group.loc[selected, "se_betaW_N"],
            color=colors[int(n_particles)],
            marker=markers[int(n_particles)],
            markersize=5.0,
            linewidth=0.95,
            capsize=2.0,
            label="_nolegend_",
        )
        rejected = (~reliable) & finite
        if rejected.any():
            work_axis.errorbar(
                group.loc[rejected, "R_over_sigma"]
                + horizontal_offsets[int(n_particles)],
                group.loc[rejected, "betaW_N"],
                yerr=group.loc[rejected, "se_betaW_N"],
                color=colors[int(n_particles)],
                marker=markers[int(n_particles)],
                markerfacecolor="none",
                markersize=5.4,
                linewidth=0.0,
                capsize=2.0,
                label="_nolegend_",
            )
    work_axis.set_xlabel(r"cavity radius $R/\sigma$")
    work_axis.set_ylabel(r"$\beta W_N(R)=-\ln P_{0,N}(R)$")
    work_axis.set_xticks(SOLID_RADII[::2])
    work_axis.set_xlim(0.47, 1.03)
    work_axis.grid(alpha=0.16, linewidth=0.5)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[108],
            marker=markers[108],
            linewidth=0.95,
            markersize=4.8,
            label=r"$N=108$",
        ),
        Line2D(
            [0],
            [0],
            color=colors[256],
            marker=markers[256],
            linewidth=0.95,
            markersize=4.8,
            label=r"$N=256$",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            markerfacecolor="white",
            linewidth=0.0,
            markersize=5.0,
            label="rare-event estimate",
        ),
    ]
    work_axis.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        ncol=2,
        loc="upper left",
        fontsize=7.0,
    )
    work_axis.set_title("Cavity work versus radius")

    for n_particles in sorted(raw["N"].unique()):
        group = raw[raw["N"] == n_particles].sort_values("block")
        cumulative_events = np.cumsum(group["empty_events_R0p75"].to_numpy())
        cumulative_trials = np.cumsum(group["probe_trials"].to_numpy())
        cumulative_probability = cumulative_events / cumulative_trials
        cumulative_work = -np.log(cumulative_probability)
        convergence_axis.plot(
            group["block"] + 1,
            cumulative_work,
            color=colors[int(n_particles)],
            linewidth=0.95,
            label=rf"$N={n_particles}$",
        )
        final = processed[
            (processed["N"] == n_particles)
            & np.isclose(processed["R_over_sigma"], 0.75)
        ].iloc[0]
        convergence_axis.axhspan(
            final["betaW_N"] - 1.96 * final["se_betaW_N"],
            final["betaW_N"] + 1.96 * final["se_betaW_N"],
            color=colors[int(n_particles)],
            alpha=0.10,
            linewidth=0.0,
        )
    convergence_axis.set_xlabel("production block")
    convergence_axis.set_ylabel(r"cumulative $\beta W_N(0.75\sigma)$")
    convergence_axis.set_title(r"Running convergence at $R/\sigma=0.75$")
    convergence_axis.grid(alpha=0.14, linewidth=0.5)
    convergence_axis.legend(frameon=False)

    production = diagnostics[diagnostics["stage"] == "production"]
    metric_specs = [
        ("global_Q6", r"$Q_6$", Q6_CRYSTAL_MINIMUM, "minimum"),
        (
            "max_FCC_Bragg",
            "fcc Bragg",
            FCC_BRAGG_CRYSTAL_MINIMUM,
            "minimum",
        ),
        ("lindemann_ratio", "Lindemann", LINDEMANN_MAXIMUM, "maximum"),
    ]
    metric_positions = np.arange(len(metric_specs), dtype=float)
    for n_particles in sorted(production["N"].unique()):
        group = production[production["N"] == n_particles]
        offset = -0.08 if int(n_particles) == 108 else 0.08
        for index, (column, _, _, _) in enumerate(metric_specs):
            values = group[column].to_numpy()
            center = float(np.mean(values))
            structure_axis.errorbar(
                metric_positions[index] + offset,
                center,
                yerr=np.asarray(
                    [[center - float(np.min(values))], [float(np.max(values)) - center]]
                ),
                color=colors[int(n_particles)],
                marker=markers[int(n_particles)],
                markersize=4.5,
                linewidth=0.9,
                capsize=2.0,
                label=rf"$N={n_particles}$" if index == 0 else "_nolegend_",
            )
    structure_axis.set_xticks(
        metric_positions,
        [r"$Q_6$", "fcc Bragg", "Lindemann"],
    )
    structure_axis.set_ylabel("production-block value")
    structure_axis.set_ylim(0.0, 1.0)
    structure_axis.set_title("Crystalline-order diagnostics")
    structure_axis.grid(alpha=0.14, linewidth=0.5, axis="y")
    structure_axis.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 0.62),
    )

    for label, axis, x_position in (
        ("a", work_axis, -0.055),
        ("b", convergence_axis, -0.11),
        ("c", structure_axis, -0.11),
    ):
        axis.text(
            x_position,
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
            "Title": "Canonical hard-sphere FCC solid cavity benchmark",
            "Subject": (
                "Particle-resolved solid-basin MC; explicitly not a "
                "phase-switch calculation"
            ),
            "Creator": "scripts/solid_cavity_mc.py",
        },
    )
    plt.close(fig)


def write_table(processed: pd.DataFrame, output: Path) -> None:
    """Create a small LaTeX table generated directly from processed results."""

    rows = [
        r"\begin{tabular}{rrrrrrl}",
        r"\hline",
        (
            r"$N$ & $R/\sigma$ & $P_{0,N}$ & $\mathrm{SE}(P_{0,N})$ "
            r"& $\beta W_N$ & empty events & status \\"
        ),
        r"\hline",
    ]
    for row in processed.itertuples(index=False):
        if row.empty_events == 0:
            probability = "0 observed"
            se_probability = "--"
            beta_work = "--"
            status = "raw exposure only"
        else:
            probability = f"{row.P_empty:.7g}"
            se_probability = f"{row.se_P_empty:.2g}"
            beta_work = (
                f"{row.betaW_N:.5f}" if np.isfinite(row.betaW_N) else "--"
            )
            status = "direct" if row.reliable_direct else "gated"
        rows.append(
            f"{row.N:d} & {row.R_over_sigma:.2f} & {probability} & "
            f"{se_probability} & {beta_work} & {row.empty_events:d} & "
            f"{status} \\\\"
        )
    rows.extend([r"\hline", r"\end{tabular}"])
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_results_note(
    processed: pd.DataFrame,
    diagnostics: pd.DataFrame,
    source: dict[str, Any],
    output: Path,
) -> None:
    lines = [
        "# Solid cavity MC: manuscript-ready numerical summary",
        "",
        (
            f"The runs use eta_s = {source['eta']:.8f} +/- "
            f"{source['se_eta']:.8f}, derived from the published "
            f"thermodynamic-limit solid specific volume "
            f"{source['specific_volume']:.5f} +/- "
            f"{source['se_specific_volume']:.5f} sigma^3 "
            f"(DOI {source['doi']})."
        ),
        "",
    ]
    for n_particles in sorted(processed["N"].unique()):
        group = processed[processed["N"] == n_particles]
        half = group[np.isclose(group["R_over_sigma"], 0.50)].iloc[0]
        three_quarters = group[
            np.isclose(group["R_over_sigma"], 0.75)
        ].iloc[0]
        one = group[np.isclose(group["R_over_sigma"], 1.00)].iloc[0]
        prod = diagnostics[
            (diagnostics["N"] == n_particles)
            & (diagnostics["stage"] == "production")
        ]
        lines.extend(
            [
                f"## N = {n_particles}",
                "",
                (
                    f"- R/sigma = 0.50: P0 = {half.P_empty:.8f} +/- "
                    f"{half.se_P_empty:.8f}, beta W = {half.betaW_N:.6f} "
                    f"+/- {half.se_betaW_N:.6f}; exact-control "
                    f"z = {half.exact_z:+.2f}."
                ),
                (
                    f"- R/sigma = 0.75: P0 = "
                    f"{three_quarters.P_empty:.8g} +/- "
                    f"{three_quarters.se_P_empty:.2g}, beta W = "
                    f"{three_quarters.betaW_N:.6f} +/- "
                    f"{three_quarters.se_betaW_N:.6f}; "
                    f"{three_quarters.reliability_status}."
                ),
                (
                    f"- R/sigma = 1.00 audit: {one.empty_events:d} empty "
                    f"events in {one.probe_trials:d} correlated spatial "
                    f"probes; {one.reliability_status}. Only this raw exposure "
                    "is reported, with no independent-trial confidence bound "
                    "and no finite work estimate."
                ),
                (
                    f"- Crystal gates over every production block: "
                    f"min Q6 = {prod.global_Q6.min():.4f}, min FCC Bragg = "
                    f"{prod.max_FCC_Bragg.min():.4f}, max Lindemann ratio = "
                    f"{prod.lindemann_ratio.max():.4f}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Scope limitation",
            "",
            (
                "These are fixed-density, canonical, phase-conditioned FCC "
                "measurements. They provide no fluid/solid relative weight, "
                "no cross-basin normalization, and no phase-switch data."
            ),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    root: Path,
    settings: SolidRunSettings,
    sizes: tuple[int, ...],
    base_seed: int,
    source: dict[str, Any],
    processed: pd.DataFrame,
    diagnostics: pd.DataFrame,
    elapsed: float,
) -> None:
    script_path = Path(__file__).resolve()
    source_path = root / SOURCE_RELATIVE_PATH
    exact = processed[np.isclose(processed["R_over_sigma"], 0.50)]
    direct_three_quarters = processed[
        np.isclose(processed["R_over_sigma"], 0.75)
    ]
    direct_one = processed[np.isclose(processed["R_over_sigma"], 1.00)]
    production = diagnostics[diagnostics["stage"] == "production"]
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "source_type": "mc_new",
        "script": "scripts/solid_cavity_mc.py",
        "script_sha256": sha256(script_path),
        "code_commit_at_generation": git_commit(root),
        "reproduce_command": (
            "python3 scripts/solid_cavity_mc.py "
            f"--seed {base_seed} --sizes {' '.join(map(str, sizes))} "
            f"--equil-sweeps {settings.equil_sweeps} "
            f"--tune-sweeps {settings.tune_sweeps} "
            f"--production-blocks {settings.production_blocks} "
            f"--sweeps-per-block {settings.sweeps_per_block} "
            f"--sample-stride {settings.sample_stride} "
            f"--probes-per-sample {settings.probes_per_sample}"
        ),
        "scope": {
            "ensemble": "canonical NVT",
            "phase": "defect-free commensurate FCC solid basin",
            "potential": "additive hard spheres of diameter sigma=1",
            "box": "periodic cube",
            "moves": "symmetric single-particle Metropolis translations",
            "cavity_centers": "independent uniform random spatial probes",
            "radii_over_sigma": SOLID_RADII.tolist(),
            "not_N4": (
                "No phase switch, grand-canonical sampling, chemical-potential "
                "reweighting, fluid/solid relative weight, or cross-basin "
                "normalization was performed."
            ),
        },
        "density_source": {
            **source,
            "raw_table": SOURCE_RELATIVE_PATH,
            "raw_table_sha256": sha256(source_path),
            "derived_relation": "eta_s = pi / (6 v_solid)",
        },
        "units_convention": "beta=sigma=Lambda=1",
        "base_seed": base_seed,
        "sizes": list(sizes),
        "settings": asdict(settings),
        "derived_settings": {
            "production_sweeps": settings.production_sweeps,
            "samples_per_block": settings.samples_per_block,
            "probe_trials_per_state": settings.probe_trials_per_state,
        },
        "uncertainty": {
            "method": (
                "maximum standard error over 1-, 2-, 5-, and 10-block "
                "batchings; beta-W uncertainty uses the delta method"
            ),
            "blocks": settings.production_blocks,
            "sweeps_per_block": settings.sweeps_per_block,
            "stationarity_checks": [
                "first-half versus second-half block means",
                "single-block versus paired-block standard errors",
                "multiscale blocking through ten adjacent base blocks",
            ],
            "zero_event_reporting": "raw exposure only; no confidence bound",
        },
        "quality_thresholds": {
            "minimum_empty_events": 100,
            "minimum_positive_blocks": 20,
            "maximum_relative_probability_se": 0.20,
            "maximum_half_difference_z": 3.5,
            "acceptable_double_block_se_ratio": [0.67, 1.50],
            "exact_R0p5_maximum_abs_z": EXACT_CONTROL_Z_LIMIT,
            "all_block_global_Q6_minimum": Q6_CRYSTAL_MINIMUM,
            "all_block_FCC_Bragg_minimum": FCC_BRAGG_CRYSTAL_MINIMUM,
            "all_block_Lindemann_maximum": LINDEMANN_MAXIMUM,
            "production_acceptance_interval": [
                MINIMUM_PRODUCTION_ACCEPTANCE,
                MAXIMUM_PRODUCTION_ACCEPTANCE,
            ],
        },
        "quality_summary": {
            "all_R0p5_exact_controls_pass": bool(
                exact["exact_control_pass"].all()
            ),
            "maximum_abs_R0p5_exact_z": float(
                np.max(np.abs(exact["exact_z"]))
            ),
            "all_states_crystal_structure_gate": bool(
                processed.groupby("N")["crystal_structure_gate"].first().all()
            ),
            "all_states_acceptance_gate": bool(
                processed.groupby("N")["acceptance_gate"].first().all()
            ),
            "minimum_production_Q6": float(production["global_Q6"].min()),
            "minimum_production_FCC_Bragg": float(
                production["max_FCC_Bragg"].min()
            ),
            "maximum_production_Lindemann": float(
                production["lindemann_ratio"].max()
            ),
            "R0p75_direct_rows_reliable": int(
                direct_three_quarters["reliable_direct"].sum()
            ),
            "R0p75_direct_rows_total": int(direct_three_quarters.shape[0]),
            "R1_direct_rows_reliable": int(
                direct_one["reliable_direct"].sum()
            ),
            "R1_direct_rows_total": int(direct_one.shape[0]),
            "R1_event_audit": direct_one[
                [
                    "N",
                    "probe_trials",
                    "empty_events",
                    "reliability_status",
                ]
            ].to_dict(orient="records"),
        },
        "artifacts": {
            "processed": "data_processed/solid_cavity_mc.csv",
            "raw_blocks": "data_raw/solid_cavity_mc_blocks.csv",
            "structure_diagnostics": (
                "data_raw/solid_cavity_mc_structure.csv"
            ),
            "final_states": "data_raw/solid_cavity_mc_final_states.npz",
            "figure": "figures/solid_cavity_mc_benchmark.pdf",
            "latex_table": "generated/solid_cavity_mc_table.tex",
            "results_note": "generated/solid_cavity_mc_results.md",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
            "numba_threads": numba.get_num_threads(),
        },
        "wall_seconds_total": elapsed,
    }
    output = root / "generated" / "solid_cavity_mc_metadata.json"
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a canonical hard-sphere FCC-basin cavity benchmark."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--equil-sweeps", type=int, default=24_000)
    parser.add_argument("--tune-sweeps", type=int, default=4_000)
    parser.add_argument("--tune-chunk-sweeps", type=int, default=400)
    parser.add_argument("--audit-chunk-sweeps", type=int, default=2_000)
    parser.add_argument("--production-blocks", type=int, default=100)
    parser.add_argument("--sweeps-per-block", type=int, default=400)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--probes-per-sample", type=int, default=2_048)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short smoke run; output is explicitly not production quality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    sizes = tuple(int(value) for value in args.sizes)
    for n_particles in sizes:
        cells = int(round((n_particles / 4.0) ** (1.0 / 3.0)))
        if 4 * cells**3 != n_particles:
            raise ValueError(f"N={n_particles} is not commensurate FCC.")
    source_path = root / SOURCE_RELATIVE_PATH
    source = load_solid_coexistence_density(source_path)
    if args.quick:
        settings = SolidRunSettings(
            equil_sweeps=400,
            tune_sweeps=200,
            tune_chunk_sweeps=100,
            audit_chunk_sweeps=200,
            production_blocks=5,
            sweeps_per_block=40,
            sample_stride=5,
            probes_per_sample=512,
        )
    else:
        settings = SolidRunSettings(
            equil_sweeps=args.equil_sweeps,
            tune_sweeps=args.tune_sweeps,
            tune_chunk_sweeps=args.tune_chunk_sweeps,
            audit_chunk_sweeps=args.audit_chunk_sweeps,
            production_blocks=args.production_blocks,
            sweeps_per_block=args.sweeps_per_block,
            sample_stride=args.sample_stride,
            probes_per_sample=args.probes_per_sample,
        )
    if settings.sweeps_per_block % settings.sample_stride:
        raise ValueError("sweeps_per_block must be divisible by sample_stride.")
    if settings.production_blocks < 2:
        raise ValueError("At least two production blocks are required.")
    for directory in ("data_processed", "data_raw", "figures", "generated"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    print(
        "published solid density: "
        f"eta_s={source['eta']:.8f} +/- {source['se_eta']:.8f} "
        f"(DOI {source['doi']})",
        flush=True,
    )
    all_processed: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    final_states: dict[str, np.ndarray] = {}
    start = time.perf_counter()
    for size_index, n_particles in enumerate(sizes):
        print(
            f"running solid N={n_particles}, eta={source['eta']:.6f}, "
            f"{settings.probe_trials_per_state:,} probes",
            flush=True,
        )
        processed, raw, diagnostics, final_positions = run_state(
            n_particles,
            source["eta"],
            source,
            size_index,
            settings,
            args.seed,
        )
        all_processed.extend(processed)
        all_raw.extend(raw)
        all_diagnostics.extend(diagnostics)
        final_states[f"positions_N{n_particles}"] = final_positions
        final_states[f"box_length_N{n_particles}"] = np.asarray(
            processed[0]["L"]
        )
        by_radius = {
            round(float(row["R_over_sigma"]), 2): row for row in processed
        }
        half = by_radius[0.50]
        three_quarters = by_radius[0.75]
        one = by_radius[1.00]
        print(
            "  "
            f"accept={half['production_acceptance']:.3f}, "
            f"minQ6={half['minimum_audited_Q6']:.3f}, "
            f"minBragg={half['minimum_audited_FCC_Bragg']:.3f}, "
            f"maxLind={half['maximum_audited_lindemann']:.3f}, "
            f"R=0.5 z={half['exact_z']:+.2f}, "
            f"R=0.75 events={three_quarters['empty_events']}, "
            f"R=1 events={one['empty_events']}",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    processed = pd.DataFrame(all_processed).sort_values(["N", "R_over_sigma"])
    raw = pd.DataFrame(all_raw).sort_values(["N", "block"])
    diagnostics = pd.DataFrame(all_diagnostics).sort_values(
        ["N", "stage", "cumulative_sweeps"]
    )
    commit = git_commit(root)
    script_hash = sha256(Path(__file__).resolve())
    for frame in (processed, raw, diagnostics):
        frame.insert(3, "code_commit", commit)
        frame.insert(4, "script_sha256", script_hash)
    processed.to_csv(
        root / "data_processed" / "solid_cavity_mc.csv", index=False
    )
    raw.to_csv(root / "data_raw" / "solid_cavity_mc_blocks.csv", index=False)
    diagnostics.to_csv(
        root / "data_raw" / "solid_cavity_mc_structure.csv", index=False
    )
    np.savez_compressed(
        root / "data_raw" / "solid_cavity_mc_final_states.npz",
        **final_states,
    )
    make_figure(
        processed,
        raw,
        diagnostics,
        root / "figures" / "solid_cavity_mc_benchmark.pdf",
    )
    write_table(processed, root / "generated" / "solid_cavity_mc_table.tex")
    write_results_note(
        processed,
        diagnostics,
        source,
        root / "generated" / "solid_cavity_mc_results.md",
    )
    write_metadata(
        root,
        settings,
        sizes,
        args.seed,
        source,
        processed,
        diagnostics,
        elapsed,
    )
    print(f"completed in {elapsed:.1f} s")
    print(
        "quality: "
        f"exact={processed[np.isclose(processed.R_over_sigma, 0.5)].exact_control_pass.all()}, "
        f"crystal={processed.groupby('N').crystal_structure_gate.first().all()}, "
        f"R0.75 direct="
        f"{processed[np.isclose(processed.R_over_sigma, 0.75)].reliable_direct.sum()}/"
        f"{len(sizes)}, "
        f"R1 direct="
        f"{processed[np.isclose(processed.R_over_sigma, 1.0)].reliable_direct.sum()}/"
        f"{len(sizes)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
