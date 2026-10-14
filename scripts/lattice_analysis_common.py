#!/usr/bin/env python3
"""Shared, read-only analysis helpers for the lattice-gas trajectories.

The production sampler intentionally remains separate.  These helpers load one
or more NPZ trajectory archives, preserve replica identity, construct paired
(spin-inversion antithetic) or raw unpaired block contributions, and implement
the stratified time-block bootstrap used by the revision analyses.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

try:  # Imported as ``scripts.lattice_analysis_common`` by tests.
    from .lattice_gas_phase_switch import (
        BlockData,
        CONTRIBUTION_NAMES,
        block_contributions,
        make_blocks,
    )
except ImportError:  # Executed by a script in this directory.
    from lattice_gas_phase_switch import (
        BlockData,
        CONTRIBUTION_NAMES,
        block_contributions,
        make_blocks,
    )


TRAJECTORY_PATTERN = re.compile(
    r"^L(?P<L>\d+)_replica(?P<replica>\d+)_(?P<field>.+)$"
)
REQUIRED_FIELDS = (
    "magnetization",
    "bond_sum",
    "site_zero",
    "site_right",
    "cluster_size",
)


def load_trajectory_archives(
    paths: Iterable[str | Path],
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Load NPZ archives while rejecting duplicate or incomplete replicas."""

    trajectories: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for path_like in paths:
        path = Path(path_like)
        with np.load(path) as archive:
            for key in archive.files:
                match = TRAJECTORY_PATTERN.match(key)
                if match is None:
                    continue
                identity = (int(match["L"]), int(match["replica"]))
                field = match["field"]
                entry = trajectories.setdefault(identity, {})
                if field in entry:
                    raise ValueError(
                        f"duplicate {field!r} for L={identity[0]}, "
                        f"replica={identity[1]}"
                    )
                entry[field] = np.asarray(archive[key])

    if not trajectories:
        raise ValueError("no lattice-gas trajectories found")
    for identity, fields in trajectories.items():
        missing = sorted(set(REQUIRED_FIELDS) - set(fields))
        if missing:
            raise ValueError(f"{identity} is missing fields: {missing}")
        lengths = {len(fields[name]) for name in REQUIRED_FIELDS}
        if len(lengths) != 1:
            raise ValueError(f"inconsistent trajectory lengths for {identity}")
    return trajectories


def available_sizes(
    trajectories: dict[tuple[int, int], dict[str, np.ndarray]],
) -> list[int]:
    return sorted({identity[0] for identity in trajectories})


def replicas_for_size(
    trajectories: dict[tuple[int, int], dict[str, np.ndarray]], length: int
) -> list[int]:
    return sorted(replica for size, replica in trajectories if size == length)


def _unpaired_block_contributions(
    magnetization: np.ndarray,
    bond_sum: np.ndarray,
    site_zero: np.ndarray,
    site_right: np.ndarray,
    volume: int,
    h_grid: np.ndarray,
) -> np.ndarray:
    """Contribution sums from the raw chain, without spin-inversion pairing."""

    magnetization = np.asarray(magnetization, dtype=float)
    bond_sum = np.asarray(bond_sum, dtype=float)
    site_zero = np.asarray(site_zero)
    site_right = np.asarray(site_right)
    number = 0.5 * (volume + magnetization)
    centered_density = magnetization / (2.0 * volume)
    weights = np.exp(np.outer(np.asarray(h_grid, dtype=float), centered_density))
    cavity = (2.0 * volume - 4.0 * magnetization + bond_sum) / (8.0 * volume)
    liquid = (magnetization > 0.0) | (
        (magnetization == 0.0) & (site_zero > 0)
    )
    gas = ~liquid
    fixed_cavity = ((site_zero < 0) & (site_right < 0)).astype(float)

    def total(values: np.ndarray) -> np.ndarray:
        return weights @ np.asarray(values, dtype=float)

    z = np.sum(weights, axis=1)
    n = total(number)
    c = total(cavity)
    cn = total(cavity * number)
    z_liquid = np.sum(weights[:, liquid], axis=1)
    z_gas = np.sum(weights[:, gas], axis=1)
    n_liquid = weights[:, liquid] @ number[liquid]
    n_gas = weights[:, gas] @ number[gas]
    c_liquid = weights[:, liquid] @ cavity[liquid]
    c_gas = weights[:, gas] @ cavity[gas]
    cn_liquid = weights[:, liquid] @ (cavity[liquid] * number[liquid])
    cn_gas = weights[:, gas] @ (cavity[gas] * number[gas])
    fixed_liquid = weights[:, liquid] @ fixed_cavity[liquid]
    fixed_gas = weights[:, gas] @ fixed_cavity[gas]
    return np.column_stack(
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


def contribution_blocks(
    trajectories: dict[tuple[int, int], dict[str, np.ndarray]],
    length: int,
    h_grid: np.ndarray,
    block_count: int,
    mode: str = "paired",
) -> tuple[np.ndarray, list[int]]:
    """Return arrays with axes replica, block, field, contribution."""

    if mode not in {"paired", "unpaired"}:
        raise ValueError("mode must be 'paired' or 'unpaired'")
    replicas = replicas_for_size(trajectories, length)
    if not replicas:
        raise ValueError(f"no replicas found for L={length}")
    volume = length * length
    per_replica: list[np.ndarray] = []
    for replica in replicas:
        fields = trajectories[(length, replica)]
        magnetization = fields["magnetization"]
        bond_sum = fields["bond_sum"]
        site_zero = fields["site_zero"]
        site_right = fields["site_right"]
        if mode == "paired":
            blocks: list[BlockData] = make_blocks(
                magnetization,
                bond_sum,
                site_zero,
                site_right,
                volume,
                replica,
                block_count,
            )
            arrays = np.asarray(
                [block_contributions(block, volume, h_grid) for block in blocks]
            )
        else:
            indices = np.array_split(np.arange(len(magnetization)), block_count)
            arrays = np.asarray(
                [
                    _unpaired_block_contributions(
                        magnetization[index],
                        bond_sum[index],
                        site_zero[index],
                        site_right[index],
                        volume,
                        h_grid,
                    )
                    for index in indices
                    if len(index) > 0
                ]
            )
        if len(arrays) != block_count:
            raise ValueError(
                f"L={length}, replica={replica} produced {len(arrays)} blocks"
            )
        per_replica.append(arrays)
    result = np.asarray(per_replica)
    if result.shape[-1] != len(CONTRIBUTION_NAMES):
        raise AssertionError("unexpected contribution schema")
    return result, replicas


def stratified_block_indices(
    replica_count: int, block_count: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample time blocks independently inside every retained chain.

    The chains sample the same equilibrium law and are not random physical
    clusters. Resampling chains as well as their blocks counts within-chain
    Monte Carlo noise twice. Chain identity is therefore a fixed stratum.
    The caller must choose blocks longer than the autocorrelation time.
    """
    if replica_count < 1 or block_count < 2:
        raise ValueError("at least one replica and two time blocks are required")
    return rng.integers(0, block_count, size=(replica_count, block_count))


def stratified_block_bootstrap_totals(
    block_arrays: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    """Yield totals after independently resampling blocks within each chain."""

    replica_count, block_count = block_arrays.shape[:2]
    output_shape = block_arrays.shape[2:]
    for _ in range(draws):
        total = np.zeros(output_shape, dtype=float)
        indices = stratified_block_indices(replica_count, block_count, rng)
        for replica, selected_blocks in enumerate(indices):
            total += np.sum(block_arrays[replica, selected_blocks], axis=0)
        yield total


def local_quadratic_extremum(
    x: np.ndarray,
    y: np.ndarray,
    half_window: int = 2,
    magnitude: bool = True,
) -> tuple[float, float]:
    """Estimate a smooth grid extremum from a local quadratic fit."""

    x = np.asarray(x, dtype=float)
    values = np.abs(np.asarray(y, dtype=float)) if magnitude else np.asarray(y, dtype=float)
    peak = int(np.argmax(values))
    if peak - half_window < 0 or peak + half_window >= len(x):
        raise ValueError("extremum is too close to the grid boundary")
    selection = slice(peak - half_window, peak + half_window + 1)
    coefficients = np.polyfit(x[selection], values[selection], 2)
    if coefficients[0] >= 0.0:
        raise ValueError("local quadratic does not have a maximum")
    center = -coefficients[1] / (2.0 * coefficients[0])
    if not (x[peak - half_window] <= center <= x[peak + half_window]):
        raise ValueError("quadratic extremum lies outside the fit window")
    height = np.polyval(coefficients, center)
    return float(center), float(height)


def weighted_linear_fit(
    inverse_volume: np.ndarray,
    values: np.ndarray,
    standard_errors: np.ndarray,
) -> dict[str, float]:
    """Fit ``A + B/V`` with independent, supplied standard errors."""

    inverse_volume = np.asarray(inverse_volume, dtype=float)
    values = np.asarray(values, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    if np.any(standard_errors <= 0.0):
        raise ValueError("all standard errors must be positive")
    design = np.column_stack((np.ones(len(values)), inverse_volume))
    precision = 1.0 / standard_errors**2
    normal = design.T @ (precision[:, None] * design)
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ design.T @ (precision * values)
    residual = (values - design @ coefficients) / standard_errors
    return {
        "A": float(coefficients[0]),
        "B": float(coefficients[1]),
        "se_A": float(np.sqrt(covariance[0, 0])),
        "se_B": float(np.sqrt(covariance[1, 1])),
        "cov_AA": float(covariance[0, 0]),
        "cov_AB": float(covariance[0, 1]),
        "cov_BB": float(covariance[1, 1]),
        "chi2": float(np.dot(residual, residual)),
        "dof": int(len(values) - 2),
    }


def paired_observation_arrays(
    trajectories: dict[tuple[int, int], dict[str, np.ndarray]], length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return oriented M, bond sum, and the original-sign tie variable."""

    volume = length * length
    oriented_magnetization: list[np.ndarray] = []
    bond_sums: list[np.ndarray] = []
    site_zeros: list[np.ndarray] = []
    for replica in replicas_for_size(trajectories, length):
        fields = trajectories[(length, replica)]
        magnetization = fields["magnetization"]
        site_zero = fields["site_zero"]
        liquid = (magnetization > 0) | (
            (magnetization == 0) & (site_zero > 0)
        )
        oriented_magnetization.append(np.where(liquid, magnetization, -magnetization))
        bond_sums.append(fields["bond_sum"])
        site_zeros.append(site_zero)
    oriented = np.concatenate(oriented_magnetization).astype(float)
    bonds = np.concatenate(bond_sums).astype(float)
    sites = np.concatenate(site_zeros)
    if np.any(np.abs(oriented) > volume):
        raise ValueError("magnetization outside physical range")
    return oriented, bonds, sites


def kish_effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.dot(weights, weights))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)
