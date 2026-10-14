#!/usr/bin/env python3
"""Particle-resolved canonical Monte Carlo for hard-sphere cavities.

This program generates new fixed-N, fixed-V hard-sphere trajectories in
periodic cubic boxes.  It uses single-particle Metropolis translations and a
Numba-compiled linked-cell list for both overlap checks and uniformly random
spatial cavity probes.

Scope
-----
The default state points are homogeneous stable fluids (eta <= 0.40) at
N = 108 and 256.  This is a direct canonical cavity benchmark.  It is not a
grand-canonical calculation, does not sample a fluid/solid phase switch, and
must not be described as N4 coexistence data.

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

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy
from numba import njit
from scipy.special import sph_harm_y
from scipy.stats import t as student_t


RADII = np.asarray([0.50, 0.75, 1.00], dtype=np.float64)
DEFAULT_SIZES = (108, 256)
DEFAULT_ETAS = (0.05, 0.10, 0.20, 0.30, 0.40)
BASE_SEED = 20_260_724
ANALYSIS_VERSION = "cavity-mc-v1"
MIN_EMPTY_EVENTS = 100
MIN_POSITIVE_BLOCKS = 20
MAX_RELATIVE_SE = 0.20
MAX_HALF_DIFFERENCE_Z = 3.5
DOUBLE_BLOCK_SE_RATIO_INTERVAL = (0.67, 1.50)
EXACT_CONTROL_Z_LIMIT = 3.0
Q6_FLUID_LIMIT = 0.25
FCC_BRAGG_FLUID_LIMIT = 0.35
MAX_CELL_LIST_BYTES = 128 * 1024**2


@dataclass(frozen=True)
class RunSettings:
    equil_sweeps: int
    tune_sweeps: int
    tune_chunk_sweeps: int
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
    def probe_trials_per_state(self) -> int:
        return (
            self.production_blocks
            * self.samples_per_block
            * self.probes_per_sample
        )


def box_length(n_particles: int, eta: float) -> float:
    """Cubic box length for unit-diameter spheres at packing fraction eta."""

    return float((n_particles * np.pi / (6.0 * eta)) ** (1.0 / 3.0))


def fcc_positions(n_particles: int, length: float) -> np.ndarray:
    """Return a commensurate nonoverlapping FCC starting configuration."""

    cells_float = (n_particles / 4.0) ** (1.0 / 3.0)
    cells = int(round(cells_float))
    if 4 * cells**3 != n_particles:
        raise ValueError(
            "The FCC initializer requires N=4*m^3; "
            f"received N={n_particles}."
        )
    basis = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ],
        dtype=np.float64,
    )
    lattice_constant = length / cells
    positions = np.empty((n_particles, 3), dtype=np.float64)
    index = 0
    for ix in range(cells):
        for iy in range(cells):
            for iz in range(cells):
                origin = np.asarray((ix, iy, iz), dtype=np.float64)
                for offset in basis:
                    positions[index] = (origin + offset) * lattice_constant
                    index += 1
    return positions


@njit(cache=True)
def _flat_cell_index(cx: int, cy: int, cz: int, n_cells: int) -> int:
    return (cx * n_cells + cy) * n_cells + cz


@njit(cache=True)
def _build_cell_list(
    positions: np.ndarray, length: float, n_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("box length must be finite and positive")
    n_particles = positions.shape[0]
    # counts and dense members are int64 arrays. Check the budget before
    # cubing n_cells so even a malformed enormous input cannot overflow.
    allowed_cells = MAX_CELL_LIST_BYTES // 8 // (n_particles + 1)
    if n_cells > allowed_cells // n_cells // n_cells:
        raise ValueError(
            "dense cell-list allocation exceeds 128 MiB; reduce the particle "
            "count or cell count, or use a sparse cell-list implementation"
        )
    total_cells = n_cells**3
    counts = np.zeros(total_cells, dtype=np.int64)
    members = np.full((total_cells, n_particles), -1, dtype=np.int64)
    cell_width = length / n_cells
    for particle in range(n_particles):
        cx = min(int(positions[particle, 0] / cell_width), n_cells - 1)
        cy = min(int(positions[particle, 1] / cell_width), n_cells - 1)
        cz = min(int(positions[particle, 2] / cell_width), n_cells - 1)
        cell = _flat_cell_index(cx, cy, cz, n_cells)
        slot = counts[cell]
        members[cell, slot] = particle
        counts[cell] += 1
    return counts, members


@njit(cache=True)
def _attempt_single_move(
    positions: np.ndarray,
    length: float,
    n_cells: int,
    cell_width: float,
    counts: np.ndarray,
    members: np.ndarray,
    max_displacement: float,
) -> int:
    """Attempt one symmetric hard-sphere Metropolis translation."""

    n_particles = positions.shape[0]
    particle = np.random.randint(0, n_particles)
    old_x = positions[particle, 0]
    old_y = positions[particle, 1]
    old_z = positions[particle, 2]

    new_x = old_x + (2.0 * np.random.random() - 1.0) * max_displacement
    new_y = old_y + (2.0 * np.random.random() - 1.0) * max_displacement
    new_z = old_z + (2.0 * np.random.random() - 1.0) * max_displacement
    new_x -= length * math.floor(new_x / length)
    new_y -= length * math.floor(new_y / length)
    new_z -= length * math.floor(new_z / length)

    new_cx = min(int(new_x / cell_width), n_cells - 1)
    new_cy = min(int(new_y / cell_width), n_cells - 1)
    new_cz = min(int(new_z / cell_width), n_cells - 1)

    for offset_x in range(-1, 2):
        cx = (new_cx + offset_x) % n_cells
        for offset_y in range(-1, 2):
            cy = (new_cy + offset_y) % n_cells
            for offset_z in range(-1, 2):
                cz = (new_cz + offset_z) % n_cells
                cell = _flat_cell_index(cx, cy, cz, n_cells)
                for slot in range(counts[cell]):
                    other = members[cell, slot]
                    if other == particle:
                        continue
                    dx = new_x - positions[other, 0]
                    dy = new_y - positions[other, 1]
                    dz = new_z - positions[other, 2]
                    dx -= length * np.rint(dx / length)
                    dy -= length * np.rint(dy / length)
                    dz -= length * np.rint(dz / length)
                    if dx * dx + dy * dy + dz * dz < 1.0:
                        return 0

    old_cx = min(int(old_x / cell_width), n_cells - 1)
    old_cy = min(int(old_y / cell_width), n_cells - 1)
    old_cz = min(int(old_z / cell_width), n_cells - 1)
    old_cell = _flat_cell_index(old_cx, old_cy, old_cz, n_cells)
    new_cell = _flat_cell_index(new_cx, new_cy, new_cz, n_cells)
    if old_cell != new_cell:
        found_slot = -1
        for slot in range(counts[old_cell]):
            if members[old_cell, slot] == particle:
                found_slot = slot
                break
        if found_slot < 0:
            return -1
        last_slot = counts[old_cell] - 1
        members[old_cell, found_slot] = members[old_cell, last_slot]
        members[old_cell, last_slot] = -1
        counts[old_cell] -= 1

        destination_slot = counts[new_cell]
        members[new_cell, destination_slot] = particle
        counts[new_cell] += 1

    positions[particle, 0] = new_x
    positions[particle, 1] = new_y
    positions[particle, 2] = new_z
    return 1


@njit(cache=True)
def advance_sweeps(
    positions: np.ndarray,
    length: float,
    n_cells: int,
    max_displacement: float,
    sweeps: int,
    seed: int,
) -> tuple[int, int]:
    """Advance a state with single-particle moves and return accepted, attempted."""

    np.random.seed(seed)
    counts, members = _build_cell_list(positions, length, n_cells)
    cell_width = length / n_cells
    n_particles = positions.shape[0]
    accepted = 0
    attempted = sweeps * n_particles
    for _ in range(attempted):
        outcome = _attempt_single_move(
            positions,
            length,
            n_cells,
            cell_width,
            counts,
            members,
            max_displacement,
        )
        if outcome < 0:
            raise RuntimeError("Linked-cell membership invariant failed.")
        accepted += outcome
    return accepted, attempted


@njit(cache=True)
def _probe_cavities(
    positions: np.ndarray,
    length: float,
    n_cells: int,
    cell_width: float,
    counts: np.ndarray,
    members: np.ndarray,
    radii_squared: np.ndarray,
    n_probes: int,
) -> np.ndarray:
    """Count empty cavities for each radius at uniform random probe centers."""

    hits = np.zeros(radii_squared.size, dtype=np.int64)
    maximum_radius_squared = radii_squared[-1]
    for _ in range(n_probes):
        probe_x = length * np.random.random()
        probe_y = length * np.random.random()
        probe_z = length * np.random.random()
        probe_cx = min(int(probe_x / cell_width), n_cells - 1)
        probe_cy = min(int(probe_y / cell_width), n_cells - 1)
        probe_cz = min(int(probe_z / cell_width), n_cells - 1)
        minimum_squared_distance = maximum_radius_squared

        found_inside_smallest = False
        for offset_x in range(-1, 2):
            if found_inside_smallest:
                break
            cx = (probe_cx + offset_x) % n_cells
            for offset_y in range(-1, 2):
                if found_inside_smallest:
                    break
                cy = (probe_cy + offset_y) % n_cells
                for offset_z in range(-1, 2):
                    cz = (probe_cz + offset_z) % n_cells
                    cell = _flat_cell_index(cx, cy, cz, n_cells)
                    for slot in range(counts[cell]):
                        particle = members[cell, slot]
                        dx = probe_x - positions[particle, 0]
                        dy = probe_y - positions[particle, 1]
                        dz = probe_z - positions[particle, 2]
                        dx -= length * np.rint(dx / length)
                        dy -= length * np.rint(dy / length)
                        dz -= length * np.rint(dz / length)
                        squared_distance = dx * dx + dy * dy + dz * dz
                        if squared_distance < minimum_squared_distance:
                            minimum_squared_distance = squared_distance
                            if minimum_squared_distance < radii_squared[0]:
                                found_inside_smallest = True
                                break
                    if found_inside_smallest:
                        break

        for radius_index in range(radii_squared.size):
            if minimum_squared_distance >= radii_squared[radius_index]:
                hits[radius_index] += 1
    return hits


@njit(cache=True)
def production_blocks(
    positions: np.ndarray,
    length: float,
    n_cells: int,
    max_displacement: float,
    n_blocks: int,
    sweeps_per_block: int,
    sample_stride: int,
    probes_per_sample: int,
    radii_squared: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run fixed-step production and retain block-level probe and move counts."""

    np.random.seed(seed)
    counts, members = _build_cell_list(positions, length, n_cells)
    cell_width = length / n_cells
    n_particles = positions.shape[0]
    samples_per_block = sweeps_per_block // sample_stride
    block_hits = np.zeros((n_blocks, radii_squared.size), dtype=np.int64)
    block_move_accepts = np.zeros(n_blocks, dtype=np.int64)
    block_move_attempts = np.zeros(n_blocks, dtype=np.int64)

    for block in range(n_blocks):
        for _ in range(samples_per_block):
            attempts = sample_stride * n_particles
            for _ in range(attempts):
                outcome = _attempt_single_move(
                    positions,
                    length,
                    n_cells,
                    cell_width,
                    counts,
                    members,
                    max_displacement,
                )
                if outcome < 0:
                    raise RuntimeError("Linked-cell membership invariant failed.")
                block_move_accepts[block] += outcome
            block_move_attempts[block] += attempts
            block_hits[block] += _probe_cavities(
                positions,
                length,
                n_cells,
                cell_width,
                counts,
                members,
                radii_squared,
                probes_per_sample,
            )
    return block_hits, block_move_accepts, block_move_attempts


def minimum_pair_distance(positions: np.ndarray, length: float) -> float:
    """Brute-force audit of the smallest periodic pair distance."""

    minimum = np.inf
    for index in range(positions.shape[0] - 1):
        delta = positions[index + 1 :] - positions[index]
        delta -= length * np.rint(delta / length)
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        minimum = min(minimum, float(np.min(distances)))
    return minimum


def global_q6(
    positions: np.ndarray, length: float, neighbor_cutoff: float = 1.50
) -> tuple[float, int]:
    """Rotation-invariant global Steinhardt Q6 from bonds below a fixed cutoff."""

    bond_vectors: list[np.ndarray] = []
    for index in range(positions.shape[0] - 1):
        delta = positions[index + 1 :] - positions[index]
        delta -= length * np.rint(delta / length)
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        selected = delta[distances < neighbor_cutoff]
        if selected.size:
            bond_vectors.extend(selected)
    if not bond_vectors:
        return float("nan"), 0
    vectors = np.asarray(bond_vectors, dtype=np.float64)
    radii = np.sqrt(np.sum(vectors * vectors, axis=1))
    theta = np.arccos(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    phi = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi)
    qlm = np.asarray(
        [np.mean(sph_harm_y(6, order, theta, phi)) for order in range(-6, 7)]
    )
    q6 = np.sqrt(4.0 * np.pi / 13.0 * np.sum(np.abs(qlm) ** 2))
    return float(q6), int(vectors.shape[0])


def maximum_fcc_bragg_amplitude(
    positions: np.ndarray, length: float, n_particles: int
) -> float:
    """Maximum amplitude at reciprocal vectors of the initial FCC orientation."""

    cells = int(round((n_particles / 4.0) ** (1.0 / 3.0)))
    indices: list[tuple[int, int, int]] = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                indices.append((sx * cells, sy * cells, sz * cells))
    for axis in range(3):
        for sign in (-1, 1):
            vector = [0, 0, 0]
            vector[axis] = sign * 2 * cells
            indices.append(tuple(vector))
    fractional = positions / length
    amplitudes = []
    for reciprocal_index in indices:
        phase = 2.0 * np.pi * (
            reciprocal_index[0] * fractional[:, 0]
            + reciprocal_index[1] * fractional[:, 1]
            + reciprocal_index[2] * fractional[:, 2]
        )
        amplitudes.append(abs(np.mean(np.exp(1j * phase))))
    return float(max(amplitudes))


def _standard_error(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def summarize_block_probability(
    block_hits: np.ndarray,
    trials_per_block: int,
) -> dict[str, Any]:
    """Block estimate, transformed work uncertainty, and rare-event gates."""

    hits = np.asarray(block_hits, dtype=np.int64)
    blocks = hits.size
    trials = int(blocks * trials_per_block)
    events = int(np.sum(hits))
    positive_blocks = int(np.count_nonzero(hits))
    block_probabilities = hits.astype(np.float64) / trials_per_block
    probability = float(events / trials)
    se_probability_single_block = _standard_error(block_probabilities)
    blocking_standard_errors: dict[int, float] = {}
    for factor in (1, 2, 5, 10):
        grouped_count = blocks // factor
        if grouped_count < 4:
            continue
        grouped = block_probabilities[: grouped_count * factor].reshape(
            grouped_count, factor
        ).mean(axis=1)
        blocking_standard_errors[factor] = _standard_error(grouped)
    finite_blocking = {
        factor: value
        for factor, value in blocking_standard_errors.items()
        if np.isfinite(value)
    }
    if finite_blocking:
        selected_block_factor = max(
            finite_blocking, key=lambda factor: finite_blocking[factor]
        )
        se_probability = float(finite_blocking[selected_block_factor])
    else:
        selected_block_factor = 1
        se_probability = se_probability_single_block
    paired_block_count = blocks // 2
    if paired_block_count >= 2:
        paired_probabilities = block_probabilities[
            : 2 * paired_block_count
        ].reshape(paired_block_count, 2).mean(axis=1)
        se_probability_double_block = _standard_error(paired_probabilities)
        double_block_se_ratio = (
            float(se_probability_double_block / se_probability_single_block)
            if se_probability_single_block > 0.0
            else float("nan")
        )
    else:
        se_probability_double_block = float("nan")
        double_block_se_ratio = float("nan")
    max_blocking_se_ratio = (
        float(se_probability / se_probability_single_block)
        if se_probability_single_block > 0.0
        else float("nan")
    )
    relative_se = (
        float(se_probability / probability)
        if probability > 0.0 and np.isfinite(se_probability)
        else float("inf")
    )

    half = blocks // 2
    first = block_probabilities[:half]
    second = block_probabilities[half:]
    difference_se = math.sqrt(_standard_error(first) ** 2 + _standard_error(second) ** 2)
    if difference_se > 0.0 and np.isfinite(difference_se):
        half_difference_z = float(abs(np.mean(first) - np.mean(second)) / difference_se)
    elif difference_se == 0.0 and np.mean(first) != np.mean(second):
        half_difference_z = float("inf")
    else:
        half_difference_z = 0.0

    if probability > 0.0:
        beta_work = float(-np.log(probability))
        se_beta_work = float(se_probability / probability)
    else:
        beta_work = float("nan")
        se_beta_work = float("nan")

    effective_blocks = max(2, blocks // selected_block_factor)
    t_critical = float(student_t.ppf(0.975, effective_blocks - 1))
    probability_low = max(0.0, probability - t_critical * se_probability)
    probability_high = min(1.0, probability + t_critical * se_probability)
    beta_work_low = (
        float(-np.log(probability_high)) if probability_high > 0.0 else float("nan")
    )
    beta_work_high = (
        float(-np.log(probability_low)) if probability_low > 0.0 else float("nan")
    )
    reasons: list[str] = []
    if events == 0:
        reasons.append("zero_events")
    elif events < MIN_EMPTY_EVENTS:
        reasons.append("too_few_events")
    if positive_blocks < MIN_POSITIVE_BLOCKS:
        reasons.append("too_few_positive_blocks")
    if relative_se > MAX_RELATIVE_SE:
        reasons.append("relative_error_too_large")
    if half_difference_z > MAX_HALF_DIFFERENCE_Z:
        reasons.append("block_half_difference")
    if (
        np.isfinite(double_block_se_ratio)
        and not (
            DOUBLE_BLOCK_SE_RATIO_INTERVAL[0]
            <= double_block_se_ratio
            <= DOUBLE_BLOCK_SE_RATIO_INTERVAL[1]
        )
    ):
        reasons.append("double_block_se_instability")
    reliable = not reasons

    return {
        "probe_trials": trials,
        "empty_events": events,
        "positive_blocks": positive_blocks,
        "P_empty": probability,
        "se_P_empty": se_probability,
        "se_P_empty_single_block": se_probability_single_block,
        "se_P_empty_double_block": se_probability_double_block,
        "double_block_se_ratio": double_block_se_ratio,
        "max_blocking_se_ratio": max_blocking_se_ratio,
        "selected_block_factor": selected_block_factor,
        "effective_blocks_for_ci": effective_blocks,
        "relative_se_P_empty": relative_se,
        "P_empty_ci95_low": probability_low,
        "P_empty_ci95_high": probability_high,
        "betaW_N": beta_work,
        "se_betaW_N": se_beta_work,
        "betaW_ci95_low": beta_work_low,
        "betaW_ci95_high": beta_work_high,
        "half_difference_z": half_difference_z,
        "reliable_direct": bool(reliable),
        "reliability_status": "reliable_direct" if reliable else ";".join(reasons),
    }


def tune_and_equilibrate(
    positions: np.ndarray,
    length: float,
    n_cells: int,
    settings: RunSettings,
    seed: int,
) -> tuple[float, int, int, list[dict[str, Any]]]:
    """Adapt a displacement during burn-in, freeze it, and finish equilibration."""

    if settings.tune_sweeps > settings.equil_sweeps:
        raise ValueError("tune_sweeps cannot exceed equil_sweeps")
    max_displacement = min(0.25, 0.08 * length)
    completed = 0
    accepted_total = 0
    attempted_total = 0
    trace: list[dict[str, Any]] = []
    chunk_index = 0
    while completed < settings.tune_sweeps:
        chunk = min(settings.tune_chunk_sweeps, settings.tune_sweeps - completed)
        accepted, attempted = advance_sweeps(
            positions,
            length,
            n_cells,
            max_displacement,
            chunk,
            seed + chunk_index,
        )
        acceptance = accepted / attempted
        completed += chunk
        accepted_total += accepted
        attempted_total += attempted
        trace.append(
            {
                "stage": "adaptive_burn_in",
                "chunk": chunk_index,
                "sweeps": chunk,
                "cumulative_sweeps": completed,
                "max_displacement_used": max_displacement,
                "acceptance": acceptance,
            }
        )
        target_acceptance = 0.35
        max_displacement *= math.exp(0.70 * (acceptance - target_acceptance))
        max_displacement = float(np.clip(max_displacement, 0.015, 0.50 * length))
        chunk_index += 1

    remaining = settings.equil_sweeps - completed
    if remaining:
        accepted, attempted = advance_sweeps(
            positions,
            length,
            n_cells,
            max_displacement,
            remaining,
            seed + 100_000,
        )
        accepted_total += accepted
        attempted_total += attempted
        trace.append(
            {
                "stage": "fixed_step_burn_in",
                "chunk": chunk_index,
                "sweeps": remaining,
                "cumulative_sweeps": settings.equil_sweeps,
                "max_displacement_used": max_displacement,
                "acceptance": accepted / attempted,
            }
        )
    return max_displacement, accepted_total, attempted_total, trace


def state_seeds(base_seed: int, size_index: int, eta_index: int) -> tuple[int, int]:
    """Stable, nonoverlapping seed allocation for equilibration and production."""

    state_offset = 100_003 * size_index + 7_919 * eta_index
    return base_seed + state_offset + 1_000_000, base_seed + state_offset + 2_000_000


def run_state(
    n_particles: int,
    eta: float,
    size_index: int,
    eta_index: int,
    settings: RunSettings,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    """Run one canonical state point and return processed, raw, diagnostics, state."""

    length = box_length(n_particles, eta)
    volume = length**3
    n_cells = int(math.floor(length))
    if n_cells < 3:
        raise ValueError("At least three linked cells per dimension are required.")
    cell_width = length / n_cells
    positions = fcc_positions(n_particles, length)
    initial_minimum_distance = minimum_pair_distance(positions, length)
    initial_q6, initial_bonds = global_q6(positions, length)
    initial_bragg = maximum_fcc_bragg_amplitude(positions, length, n_particles)
    equil_seed, production_seed = state_seeds(base_seed, size_index, eta_index)
    run_id = (
        f"{ANALYSIS_VERSION}_N{n_particles}_eta{eta:.3f}_seed{production_seed}"
    )

    start = time.perf_counter()
    (
        max_displacement,
        equil_accepted,
        equil_attempted,
        equil_trace,
    ) = tune_and_equilibrate(
        positions,
        length,
        n_cells,
        settings,
        equil_seed,
    )
    minimum_after_equil = minimum_pair_distance(positions, length)
    q6_after_equil, q6_bonds_after_equil = global_q6(positions, length)
    bragg_after_equil = maximum_fcc_bragg_amplitude(
        positions, length, n_particles
    )

    block_hits, block_accepts, block_attempts = production_blocks(
        positions,
        length,
        n_cells,
        max_displacement,
        settings.production_blocks,
        settings.sweeps_per_block,
        settings.sample_stride,
        settings.probes_per_sample,
        RADII**2,
        production_seed,
    )
    elapsed = time.perf_counter() - start
    minimum_final = minimum_pair_distance(positions, length)
    q6_final, q6_bonds_final = global_q6(positions, length)
    bragg_final = maximum_fcc_bragg_amplitude(positions, length, n_particles)
    production_acceptance = float(np.sum(block_accepts) / np.sum(block_attempts))
    equil_acceptance = float(equil_accepted / equil_attempted)

    fluid_structure_gate = bool(
        (not np.isfinite(q6_after_equil) or q6_after_equil < Q6_FLUID_LIMIT)
        and (not np.isfinite(q6_final) or q6_final < Q6_FLUID_LIMIT)
        and bragg_after_equil < FCC_BRAGG_FLUID_LIMIT
        and bragg_final < FCC_BRAGG_FLUID_LIMIT
    )
    overlap_gate = bool(
        minimum_after_equil >= 1.0 - 2.0e-12
        and minimum_final >= 1.0 - 2.0e-12
    )
    acceptance_gate = bool(0.10 <= production_acceptance <= 0.90)
    trials_per_block = settings.samples_per_block * settings.probes_per_sample

    processed_rows: list[dict[str, Any]] = []
    exact_control_pass = True
    for radius_index, radius in enumerate(RADII):
        summary = summarize_block_probability(
            block_hits[:, radius_index], trials_per_block
        )
        exact_probability = 1.0 - eta if np.isclose(radius, 0.50) else float("nan")
        exact_residual = (
            summary["P_empty"] - exact_probability
            if np.isfinite(exact_probability)
            else float("nan")
        )
        exact_z = (
            exact_residual / summary["se_P_empty"]
            if np.isfinite(exact_probability) and summary["se_P_empty"] > 0.0
            else float("nan")
        )
        radius_exact_pass = bool(
            not np.isfinite(exact_probability)
            or abs(exact_z) <= EXACT_CONTROL_Z_LIMIT
        )
        exact_control_pass = exact_control_pass and radius_exact_pass
        processed_rows.append(
            {
                "source_type": "mc_new",
                "run_id": run_id,
                "analysis_version": ANALYSIS_VERSION,
                "seed": production_seed,
                "equil_seed": equil_seed,
                "units_convention": "beta=sigma=Lambda=1",
                "raw_data_path": "data_raw/cavity_mc_blocks.csv",
                "ensemble": "canonical_NVT",
                "phase_scope": "stable_homogeneous_fluid_eta_le_0.40",
                "N4_phase_switch_data": False,
                "N": n_particles,
                "V": volume,
                "L": length,
                "eta": eta,
                "R_over_sigma": radius,
                "n_cells_per_dimension": n_cells,
                "cell_width": cell_width,
                "equil_sweeps": settings.equil_sweeps,
                "production_sweeps": settings.production_sweeps,
                "sample_stride_sweeps": settings.sample_stride,
                "production_samples": (
                    settings.production_blocks * settings.samples_per_block
                ),
                "probes_per_sample": settings.probes_per_sample,
                "n_blocks": settings.production_blocks,
                **summary,
                "equil_acceptance": equil_acceptance,
                "production_acceptance": production_acceptance,
                "max_displacement": max_displacement,
                "minimum_pair_distance_after_equil": minimum_after_equil,
                "minimum_pair_distance_final": minimum_final,
                "global_Q6_after_equil": q6_after_equil,
                "global_Q6_final": q6_final,
                "Q6_bonds_after_equil": q6_bonds_after_equil,
                "Q6_bonds_final": q6_bonds_final,
                "max_FCC_Bragg_after_equil": bragg_after_equil,
                "max_FCC_Bragg_final": bragg_final,
                "fluid_structure_gate": fluid_structure_gate,
                "overlap_gate": overlap_gate,
                "acceptance_gate": acceptance_gate,
                "exact_P_empty": exact_probability,
                "exact_residual": exact_residual,
                "exact_z": exact_z,
                "exact_control_pass": radius_exact_pass,
                "elapsed_seconds_state": elapsed,
            }
        )

    raw_rows: list[dict[str, Any]] = []
    for block in range(settings.production_blocks):
        row: dict[str, Any] = {
            "source_type": "mc_new",
            "run_id": run_id,
            "seed": production_seed,
            "N": n_particles,
            "V": volume,
            "L": length,
            "eta": eta,
            "block": block,
            "probe_trials": trials_per_block,
            "move_accepts": int(block_accepts[block]),
            "move_attempts": int(block_attempts[block]),
            "move_acceptance": float(block_accepts[block] / block_attempts[block]),
        }
        for radius_index, radius in enumerate(RADII):
            label = f"R{radius:.2f}".replace(".", "p")
            row[f"empty_events_{label}"] = int(block_hits[block, radius_index])
            row[f"P_empty_{label}"] = float(
                block_hits[block, radius_index] / trials_per_block
            )
        raw_rows.append(row)

    diagnostic_rows: list[dict[str, Any]] = []
    for trace_row in equil_trace:
        diagnostic_rows.append(
            {
                "run_id": run_id,
                "seed": equil_seed,
                "N": n_particles,
                "eta": eta,
                **trace_row,
            }
        )
    diagnostic_rows.append(
        {
            "run_id": run_id,
            "seed": production_seed,
            "N": n_particles,
            "eta": eta,
            "stage": "state_summary",
            "chunk": -1,
            "sweeps": settings.production_sweeps,
            "cumulative_sweeps": settings.production_sweeps,
            "max_displacement_used": max_displacement,
            "acceptance": production_acceptance,
            "initial_minimum_pair_distance": initial_minimum_distance,
            "initial_global_Q6": initial_q6,
            "initial_Q6_bonds": initial_bonds,
            "initial_max_FCC_Bragg": initial_bragg,
            "minimum_pair_distance_after_equil": minimum_after_equil,
            "minimum_pair_distance_final": minimum_final,
            "global_Q6_after_equil": q6_after_equil,
            "global_Q6_final": q6_final,
            "max_FCC_Bragg_after_equil": bragg_after_equil,
            "max_FCC_Bragg_final": bragg_final,
            "fluid_structure_gate": fluid_structure_gate,
            "overlap_gate": overlap_gate,
            "acceptance_gate": acceptance_gate,
            "exact_control_pass": exact_control_pass,
            "elapsed_seconds": elapsed,
        }
    )
    return processed_rows, raw_rows, diagnostic_rows, positions.copy()


def make_figure(data: pd.DataFrame, output: Path) -> None:
    """Create the vector benchmark and expose statistical quality gates."""

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
    colors = {0.50: "#0072B2", 0.75: "#E69F00", 1.00: "#009E73"}
    markers = {108: "o", 256: "s"}
    fig = plt.figure(figsize=(7.0, 5.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.75, 1.0))
    work_axis = fig.add_subplot(grid[0, :])
    exact_axis = fig.add_subplot(grid[1, 0])
    events_axis = fig.add_subplot(grid[1, 1])

    for radius in RADII:
        for n_particles in sorted(data["N"].unique()):
            group = data[
                np.isclose(data["R_over_sigma"], radius)
                & (data["N"] == n_particles)
            ].sort_values("eta")
            reliable = group["reliable_direct"].astype(bool).to_numpy()
            finite = np.isfinite(group["betaW_N"].to_numpy())
            selected = reliable & finite
            if np.any(selected):
                work_axis.errorbar(
                    group.loc[selected, "eta"],
                    group.loc[selected, "betaW_N"],
                    yerr=group.loc[selected, "se_betaW_N"],
                    color=colors[float(radius)],
                    marker=markers[int(n_particles)],
                    markersize=4.1,
                    linewidth=0.9,
                    capsize=1.8,
                    label=(
                        rf"$R/\sigma={radius:.2f}$, $N={n_particles}$"
                    ),
                )
            rejected = (~reliable) & finite
            if np.any(rejected):
                work_axis.errorbar(
                    group.loc[rejected, "eta"],
                    group.loc[rejected, "betaW_N"],
                    yerr=group.loc[rejected, "se_betaW_N"],
                    color=colors[float(radius)],
                    marker=markers[int(n_particles)],
                    markerfacecolor="none",
                    markersize=5.0,
                    linewidth=0.0,
                    capsize=1.8,
                    label="_nolegend_",
                )

    eta_curve = np.linspace(0.0, 0.405, 400)
    work_axis.plot(
        eta_curve,
        -np.log1p(-eta_curve),
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=r"exact $R/\sigma=0.50$: $-\ln(1-\eta)$",
    )
    work_axis.set_ylabel(r"$\beta W_N(R)=-\ln P_{0,N}(R)$")
    work_axis.set_xlim(0.035, 0.415)
    work_axis.set_ylim(bottom=0.0)
    work_axis.grid(alpha=0.16, linewidth=0.5)
    work_axis.legend(frameon=False, ncol=2, loc="upper left")
    work_axis.set_title("Cavity work")

    exact = data[np.isclose(data["R_over_sigma"], 0.50)].sort_values(["N", "eta"])
    for n_particles in sorted(exact["N"].unique()):
        group = exact[exact["N"] == n_particles]
        exact_axis.scatter(
            group["eta"],
            group["exact_z"],
            color="#4D4D4D",
            marker=markers[int(n_particles)],
            s=18,
            label=rf"$N={n_particles}$",
        )
    exact_axis.axhspan(
        -EXACT_CONTROL_Z_LIMIT,
        EXACT_CONTROL_Z_LIMIT,
        color="#56B4E9",
        alpha=0.13,
        linewidth=0.0,
    )
    exact_axis.axhline(0.0, color="black", linewidth=0.7)
    exact_axis.axhline(
        EXACT_CONTROL_Z_LIMIT, color="#0072B2", linestyle=":", linewidth=0.7
    )
    exact_axis.axhline(
        -EXACT_CONTROL_Z_LIMIT, color="#0072B2", linestyle=":", linewidth=0.7
    )
    exact_axis.set_xlabel(r"packing fraction $\eta$")
    exact_axis.set_ylabel(r"$[P(0.5)-(1-\eta)]/\mathrm{SE}$")
    exact_axis.set_title("Exact small-cavity control")
    exact_axis.set_ylim(-3.25, 3.25)
    exact_axis.grid(alpha=0.14, linewidth=0.5)
    exact_axis.legend(frameon=False, fontsize=7.0)

    rare = data[np.isclose(data["R_over_sigma"], 1.00)].sort_values(["N", "eta"])
    for n_particles in sorted(rare["N"].unique()):
        group = rare[rare["N"] == n_particles]
        facecolors = [
            "#009E73" if bool(value) else "none"
            for value in group["reliable_direct"]
        ]
        events_axis.plot(
            group["eta"],
            group["empty_events"],
            color="#009E73",
            linewidth=0.8,
            alpha=0.7,
        )
        events_axis.scatter(
            group["eta"],
            group["empty_events"],
            marker=markers[int(n_particles)],
            facecolors=facecolors,
            edgecolors="#009E73",
            s=22,
            label=rf"$N={n_particles}$",
            zorder=3,
        )
    events_axis.axhline(
        MIN_EMPTY_EVENTS,
        color="#D55E00",
        linestyle="--",
        linewidth=0.8,
        label=f"{MIN_EMPTY_EVENTS}-event criterion",
    )
    events_axis.set_yscale("log")
    events_axis.set_xlabel(r"packing fraction $\eta$")
    events_axis.set_ylabel(r"empty events, $R/\sigma=1$")
    events_axis.set_title("Direct-insertion statistics")
    events_axis.grid(alpha=0.14, linewidth=0.5, which="both")
    events_axis.legend(frameon=False, fontsize=7.0)

    for label, axis, x_position in (
        ("a", work_axis, -0.045),
        ("b", exact_axis, -0.105),
        ("c", events_axis, -0.105),
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
            "Title": "Stable-fluid canonical hard-sphere cavity benchmark",
            "Subject": "Particle-resolved canonical fluid MC",
            "Creator": "scripts/cavity_mc.py",
        },
    )
    plt.close(fig)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(
    root: Path,
    settings: RunSettings,
    base_seed: int,
    sizes: tuple[int, ...],
    etas: tuple[float, ...],
    processed: pd.DataFrame,
    total_elapsed: float,
    final_state_path: Path,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    exact_rows = processed[np.isclose(processed["R_over_sigma"], 0.50)]
    state_rows = (
        processed.sort_values("R_over_sigma")
        .groupby(["N", "eta"], as_index=False)
        .first()
    )
    metadata: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "source_type": "mc_new",
        "script": "scripts/cavity_mc.py",
        "script_sha256": _sha256(script_path),
        "code_commit_at_generation": _git_commit(root),
        "reproduce_command": (
            "python3 scripts/cavity_mc.py "
            f"--seed {base_seed} --sizes {' '.join(map(str, sizes))} "
            f"--etas {' '.join(f'{eta:g}' for eta in etas)} "
            f"--equil-sweeps {settings.equil_sweeps} "
            f"--tune-sweeps {settings.tune_sweeps} "
            f"--production-blocks {settings.production_blocks} "
            f"--sweeps-per-block {settings.sweeps_per_block} "
            f"--sample-stride {settings.sample_stride} "
            f"--probes-per-sample {settings.probes_per_sample}"
        ),
        "scope": {
            "ensemble": "canonical NVT",
            "potential": "additive hard spheres of diameter sigma=1",
            "box": "periodic cube",
            "moves": "symmetric single-particle Metropolis translations",
            "state_points": "stable homogeneous fluid only, eta <= 0.40",
            "cavity_centers": "independent uniform random spatial probes",
            "radii_over_sigma": RADII.tolist(),
            "not_N4": (
                "No phase switch, basin weighting, chemical-potential "
                "reweighting, crystal branch, or coexistence free-energy "
                "normalization was performed. These data do not satisfy N4."
            ),
        },
        "units_convention": "beta=sigma=Lambda=1",
        "base_seed": base_seed,
        "seed_scheme": (
            "state_seeds(base_seed,size_index,eta_index), with distinct "
            "equilibration and production seeds"
        ),
        "sizes": list(sizes),
        "packing_fractions": list(etas),
        "settings": asdict(settings),
        "derived_settings": {
            "production_sweeps": settings.production_sweeps,
            "samples_per_block": settings.samples_per_block,
            "probe_trials_per_state": settings.probe_trials_per_state,
        },
        "uncertainty": {
            "method": (
                "standard error of equal-length production-block cavity "
                "probabilities; beta-W uncertainty uses the delta method"
            ),
            "blocks": settings.production_blocks,
            "sweeps_per_block": settings.sweeps_per_block,
            "confidence_interval": (
                "two-sided Student-t interval on block probabilities, "
                "monotonically transformed for beta W"
            ),
            "stationarity_check": "first-half versus second-half block means",
        },
        "quality_thresholds": {
            "minimum_empty_events": MIN_EMPTY_EVENTS,
            "minimum_positive_blocks": MIN_POSITIVE_BLOCKS,
            "maximum_relative_probability_se": MAX_RELATIVE_SE,
            "maximum_half_difference_z": MAX_HALF_DIFFERENCE_Z,
            "acceptable_double_block_se_ratio": list(
                DOUBLE_BLOCK_SE_RATIO_INTERVAL
            ),
            "exact_R0p5_maximum_abs_z": EXACT_CONTROL_Z_LIMIT,
            "fluid_global_Q6_maximum": Q6_FLUID_LIMIT,
            "fluid_initial_orientation_Bragg_maximum": FCC_BRAGG_FLUID_LIMIT,
            "production_acceptance_interval": [0.10, 0.90],
        },
        "quality_summary": {
            "all_R0p5_exact_controls_pass": bool(
                exact_rows["exact_control_pass"].all()
            ),
            "maximum_abs_R0p5_exact_z": float(
                np.max(np.abs(exact_rows["exact_z"]))
            ),
            "all_states_overlap_free": bool(state_rows["overlap_gate"].all()),
            "all_states_fluid_structure_gate": bool(
                state_rows["fluid_structure_gate"].all()
            ),
            "all_states_acceptance_gate": bool(
                state_rows["acceptance_gate"].all()
            ),
            "all_rows_double_block_se_stability_gate": bool(
                processed["double_block_se_ratio"].between(
                    *DOUBLE_BLOCK_SE_RATIO_INTERVAL
                ).all()
            ),
            "double_block_se_ratio_range": [
                float(processed["double_block_se_ratio"].min()),
                float(processed["double_block_se_ratio"].max()),
            ],
            "direct_rows_reliable": int(processed["reliable_direct"].sum()),
            "direct_rows_total": int(processed.shape[0]),
            "unreliable_rows": processed.loc[
                ~processed["reliable_direct"].astype(bool),
                [
                    "N",
                    "eta",
                    "R_over_sigma",
                    "empty_events",
                    "reliability_status",
                ],
            ].to_dict(orient="records"),
        },
        "artifacts": {
            "processed": "data_processed/cavity_mc.csv",
            "raw_blocks": "data_raw/cavity_mc_blocks.csv",
            "equilibration_diagnostics": "data_raw/cavity_mc_equilibration.csv",
            "final_states": str(final_state_path.relative_to(root)),
            "figure": "figures/cavity_mc_benchmark.pdf",
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
        "wall_seconds_total": total_elapsed,
    }
    output = root / "generated" / "cavity_mc_metadata.json"
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic particle-resolved canonical hard-sphere cavity MC."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--etas", type=float, nargs="+", default=DEFAULT_ETAS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--equil-sweeps", type=int, default=16_000)
    parser.add_argument("--tune-sweeps", type=int, default=4_000)
    parser.add_argument("--tune-chunk-sweeps", type=int, default=400)
    parser.add_argument("--production-blocks", type=int, default=50)
    parser.add_argument("--sweeps-per-block", type=int, default=400)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--probes-per-sample", type=int, default=2_048)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short smoke run; its output is not production-quality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    sizes = tuple(int(value) for value in args.sizes)
    etas = tuple(float(value) for value in args.etas)
    if any(eta <= 0.0 or eta > 0.40 for eta in etas):
        raise ValueError("This benchmark is restricted to 0 < eta <= 0.40.")

    if args.quick:
        settings = RunSettings(
            equil_sweeps=200,
            tune_sweeps=100,
            tune_chunk_sweeps=50,
            production_blocks=4,
            sweeps_per_block=40,
            sample_stride=5,
            probes_per_sample=128,
        )
    else:
        settings = RunSettings(
            equil_sweeps=args.equil_sweeps,
            tune_sweeps=args.tune_sweeps,
            tune_chunk_sweeps=args.tune_chunk_sweeps,
            production_blocks=args.production_blocks,
            sweeps_per_block=args.sweeps_per_block,
            sample_stride=args.sample_stride,
            probes_per_sample=args.probes_per_sample,
        )
    if settings.sweeps_per_block % settings.sample_stride:
        raise ValueError("sweeps_per_block must be divisible by sample_stride")
    if settings.production_blocks < 2:
        raise ValueError("At least two production blocks are required.")

    for directory in ("data_processed", "data_raw", "figures", "generated"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    all_processed: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    final_states: dict[str, np.ndarray] = {}
    start = time.perf_counter()
    for size_index, n_particles in enumerate(sizes):
        for eta_index, eta in enumerate(etas):
            print(
                f"running N={n_particles}, eta={eta:.3f}, "
                f"{settings.probe_trials_per_state:,} probes",
                flush=True,
            )
            processed, raw, diagnostics, final_positions = run_state(
                n_particles,
                eta,
                size_index,
                eta_index,
                settings,
                args.seed,
            )
            all_processed.extend(processed)
            all_raw.extend(raw)
            all_diagnostics.extend(diagnostics)
            key = f"positions_N{n_particles}_eta{eta:.3f}".replace(".", "p")
            final_states[key] = final_positions
            final_states[f"box_length_N{n_particles}_eta{eta:.3f}".replace(".", "p")] = (
                np.asarray(processed[0]["L"])
            )
            exact_row = processed[0]
            rare_row = processed[-1]
            print(
                "  "
                f"accept={exact_row['production_acceptance']:.3f}, "
                f"Q6={exact_row['global_Q6_final']:.3f}, "
                f"R=0.5 z={exact_row['exact_z']:+.2f}, "
                f"R=1 events={rare_row['empty_events']}, "
                f"status={rare_row['reliability_status']}",
                flush=True,
            )

    total_elapsed = time.perf_counter() - start
    processed_frame = pd.DataFrame(all_processed).sort_values(
        ["N", "eta", "R_over_sigma"]
    )
    raw_frame = pd.DataFrame(all_raw).sort_values(["N", "eta", "block"])
    diagnostic_frame = pd.DataFrame(all_diagnostics).sort_values(
        ["N", "eta", "cumulative_sweeps"]
    )
    code_commit = _git_commit(root)
    script_sha256 = _sha256(Path(__file__).resolve())
    processed_frame.insert(3, "code_commit", code_commit)
    processed_frame.insert(4, "script_sha256", script_sha256)
    raw_frame.insert(3, "code_commit", code_commit)
    raw_frame.insert(4, "script_sha256", script_sha256)
    diagnostic_frame.insert(3, "code_commit", code_commit)
    diagnostic_frame.insert(4, "script_sha256", script_sha256)
    processed_frame.to_csv(root / "data_processed" / "cavity_mc.csv", index=False)
    raw_frame.to_csv(root / "data_raw" / "cavity_mc_blocks.csv", index=False)
    diagnostic_frame.to_csv(
        root / "data_raw" / "cavity_mc_equilibration.csv", index=False
    )
    final_state_path = root / "data_raw" / "cavity_mc_final_states.npz"
    np.savez_compressed(final_state_path, **final_states)
    make_figure(processed_frame, root / "figures" / "cavity_mc_benchmark.pdf")
    metadata = write_metadata(
        root,
        settings,
        args.seed,
        sizes,
        etas,
        processed_frame,
        total_elapsed,
        final_state_path,
    )
    print(f"completed in {total_elapsed:.1f} s")
    print(
        "quality: "
        f"exact_control={metadata['quality_summary']['all_R0p5_exact_controls_pass']}, "
        f"overlap={metadata['quality_summary']['all_states_overlap_free']}, "
        f"fluid={metadata['quality_summary']['all_states_fluid_structure_gate']}, "
        f"direct_reliable="
        f"{metadata['quality_summary']['direct_rows_reliable']}/"
        f"{metadata['quality_summary']['direct_rows_total']}"
    )


if __name__ == "__main__":
    main()
