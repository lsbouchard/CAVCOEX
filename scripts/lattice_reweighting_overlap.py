#!/usr/bin/env python3
"""Orbit- and block-level weighted-support diagnostics for lattice reweighting.

Kish's concentration functional is used only as a support diagnostic.  A
sampled configuration and its deterministic spin inverse form one antithetic
orbit, not two independent observations.  We therefore combine both branches
before computing orbit-level concentration and also aggregate orbit weights
over the replica--time blocks used by the hierarchical uncertainty analysis.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        load_trajectory_archives,
        replicas_for_size,
    )
    from .lattice_gas_phase_switch import (
        DEFAULT_H_GRID,
        observables_from_contributions,
        orient_symmetry_pairs,
    )
    from .lattice_pulse_scaling import continuous_pulse_observables
except ImportError:
    from lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        load_trajectory_archives,
        replicas_for_size,
    )
    from lattice_gas_phase_switch import (
        DEFAULT_H_GRID,
        observables_from_contributions,
        orient_symmetry_pairs,
    )
    from lattice_pulse_scaling import continuous_pulse_observables


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RAW = ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz"
ADDITIONAL_RAW = ROOT / "data_raw" / "lattice_gas_additional_replicas.npz"
DEFAULT_RAW = [BASELINE_RAW] + ([ADDITIONAL_RAW] if ADDITIONAL_RAW.exists() else [])
SUPPORT_THRESHOLDS = (0.05, 0.10, 0.20)
DEFAULT_BLOCK_COUNT = 40
ESTIMATORS = (
    "unconstrained_orbit",
    "constrained_orbit",
    "liquid_cavity_branch",
    "gas_cavity_branch",
)


def support_summary(weights: np.ndarray) -> dict[str, float | int]:
    """Return scale-free concentration measures for nonnegative weights."""

    weights = np.asarray(weights, dtype=float).ravel()
    if len(weights) == 0 or np.any(weights < 0.0):
        raise ValueError("support weights must be a nonempty nonnegative array")
    total = float(np.sum(weights))
    if total <= 0.0:
        return {
            "units": len(weights),
            "kish_count": 0.0,
            "kish_fraction": 0.0,
            "entropy_count": 0.0,
            "perplexity_fraction": 0.0,
            "maximum_normalized_weight": 0.0,
        }
    normalized = weights / total
    positive = normalized > 0.0
    kish_count = float(1.0 / np.dot(normalized, normalized))
    entropy_count = float(
        np.exp(-np.sum(normalized[positive] * np.log(normalized[positive])))
    )
    return {
        "units": len(weights),
        "kish_count": kish_count,
        "kish_fraction": kish_count / len(weights),
        "entropy_count": entropy_count,
        "perplexity_fraction": entropy_count / len(weights),
        "maximum_normalized_weight": float(np.max(normalized)),
    }


def _replica_weight_arrays(
    fields: dict[str, np.ndarray], volume: int, h_grid: np.ndarray
) -> dict[str, np.ndarray]:
    magnetization = np.asarray(fields["magnetization"])
    bond_sum = np.asarray(fields["bond_sum"], dtype=float)
    site_zero = np.asarray(fields["site_zero"])
    oriented_m, _ = orient_symmetry_pairs(magnetization, site_zero)
    centered_density = oriented_m.astype(float) / (2.0 * volume)
    liquid_weight = np.exp(np.outer(h_grid, centered_density))
    gas_weight = np.exp(-np.outer(h_grid, centered_density))
    pair_empty_liquid = (
        2.0 * volume - 4.0 * oriented_m + bond_sum
    ) / (8.0 * volume)
    pair_empty_gas = (
        2.0 * volume + 4.0 * oriented_m + bond_sum
    ) / (8.0 * volume)
    return {
        "unconstrained_orbit": liquid_weight + gas_weight,
        "constrained_orbit": (
            liquid_weight * pair_empty_liquid[None, :]
            + gas_weight * pair_empty_gas[None, :]
        ),
        "liquid_cavity_branch": (
            liquid_weight * pair_empty_liquid[None, :]
        ),
        "gas_cavity_branch": gas_weight * pair_empty_gas[None, :],
    }


def _contiguous_interval(mask: np.ndarray, center: int) -> tuple[int, int] | None:
    if not bool(mask[center]):
        return None
    left = center
    right = center
    while left > 0 and bool(mask[left - 1]):
        left -= 1
    while right + 1 < len(mask) and bool(mask[right + 1]):
        right += 1
    return left, right


def analyze(
    raw_paths: list[Path], block_count: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    trajectories = load_trajectory_archives(raw_paths)
    h_grid = np.asarray(DEFAULT_H_GRID, dtype=float)
    records: list[dict] = []
    replica_records: list[dict] = []

    for length in available_sizes(trajectories):
        volume = length * length
        replicas = replicas_for_size(trajectories, length)
        per_estimator_orbits: dict[str, list[np.ndarray]] = {
            estimator: [] for estimator in ESTIMATORS
        }
        per_estimator_blocks: dict[str, list[np.ndarray]] = {
            estimator: [] for estimator in ESTIMATORS
        }
        per_estimator_replica: dict[str, list[np.ndarray]] = {
            estimator: [] for estimator in ESTIMATORS
        }
        for replica in replicas:
            arrays = _replica_weight_arrays(
                trajectories[(length, replica)], volume, h_grid
            )
            for estimator, weights in arrays.items():
                indices = np.array_split(np.arange(weights.shape[1]), block_count)
                block_weights = np.column_stack(
                    [np.sum(weights[:, index], axis=1) for index in indices]
                )
                per_estimator_orbits[estimator].append(weights)
                per_estimator_blocks[estimator].append(block_weights)
                per_estimator_replica[estimator].append(
                    np.sum(weights, axis=1)
                )

        for estimator in ESTIMATORS:
            orbit_weights = np.concatenate(
                per_estimator_orbits[estimator], axis=1
            )
            block_weights = np.concatenate(
                per_estimator_blocks[estimator], axis=1
            )
            replica_weights = np.column_stack(
                per_estimator_replica[estimator]
            )
            replica_normalized = replica_weights / np.sum(
                replica_weights, axis=1, keepdims=True
            )
            for index, h_value in enumerate(h_grid):
                for analysis_unit, weights in (
                    ("antithetic_orbit", orbit_weights[index]),
                    ("replica_time_block", block_weights[index]),
                ):
                    row = support_summary(weights)
                    records.append(
                        {
                            "L": length,
                            "V": volume,
                            "h": h_value,
                            "delta_x": h_value / volume,
                            "estimator": estimator,
                            "analysis_unit": analysis_unit,
                            **row,
                            "replicas": len(replicas),
                            "blocks_per_replica": block_count,
                            "maximum_replica_weight_fraction": float(
                                np.max(replica_normalized[index])
                            ),
                            "replicas_above_one_percent": int(
                                np.count_nonzero(replica_normalized[index] >= 0.01)
                            ),
                        }
                    )
                for replica_index, replica in enumerate(replicas):
                    replica_records.append(
                        {
                            "L": length,
                            "V": volume,
                            "h": h_value,
                            "estimator": estimator,
                            "replica": replica,
                            "normalized_weight_fraction": float(
                                replica_normalized[index, replica_index]
                            ),
                        }
                    )

    curves = pd.DataFrame(records)
    replica_frame = pd.DataFrame(replica_records)

    sensitivity_rows: list[dict] = []
    orbit_support = curves[curves["analysis_unit"] == "antithetic_orbit"]
    for length in available_sizes(trajectories):
        volume = length * length
        blocks, replicas = contribution_blocks(
            trajectories, length, h_grid, block_count, mode="paired"
        )
        observed = observables_from_contributions(
            np.sum(blocks, axis=(0, 1)), volume
        )
        response = observed["deltaN_phase_reweighting"] / volume
        full = continuous_pulse_observables(h_grid, response)
        support_by_field = (
            orbit_support[orbit_support["L"] == length]
            .groupby("h")["kish_fraction"]
            .min()
            .reindex(h_grid)
            .to_numpy()
        )
        peak_index = int(np.argmin(np.abs(h_grid - full["peak_center_h"])))
        for threshold in SUPPORT_THRESHOLDS:
            interval = _contiguous_interval(
                support_by_field >= threshold, peak_index
            )
            if interval is None:
                estimates = {name: np.nan for name in full}
                left = right = np.nan
                status = "peak_not_supported"
            else:
                left_index, right_index = interval
                left = float(h_grid[left_index])
                right = float(h_grid[right_index])
                try:
                    estimates = continuous_pulse_observables(
                        h_grid[left_index : right_index + 1],
                        response[left_index : right_index + 1],
                    )
                    status = "estimated"
                except ValueError:
                    estimates = {name: np.nan for name in full}
                    status = "half_height_not_bracketed"
            for observable, full_value in full.items():
                estimate = estimates[observable]
                sensitivity_rows.append(
                    {
                        "L": length,
                        "V": volume,
                        "replicas": len(replicas),
                        "support_threshold": threshold,
                        "supported_h_min": left,
                        "supported_h_max": right,
                        "status": status,
                        "observable": observable,
                        "estimate": estimate,
                        "full_interval_estimate": full_value,
                        "shift_from_full_interval": estimate - full_value,
                    }
                )
    sensitivity = pd.DataFrame(sensitivity_rows)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_reweighting_overlap.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "raw_paths": [str(path.resolve().relative_to(ROOT)) for path in raw_paths],
        "replica_ids": {
            str(length): replicas_for_size(trajectories, length)
            for length in available_sizes(trajectories)
        },
        "block_count_per_replica": block_count,
        "support_thresholds_for_sensitivity": list(SUPPORT_THRESHOLDS),
        "definition": (
            "Kish weighted-support fraction=(sum w)^2/[n sum(w^2)]; "
            "weight entropy is reported as exp[-sum p log p]/n"
        ),
        "antithetic_treatment": (
            "each sampled configuration and its deterministic global spin "
            "inverse are combined into one orbit weight before concentration "
            "is evaluated"
        ),
        "interpretation": (
            "weight concentration and reweighting support only; neither value "
            "is an independent-sample count or a substitute for the "
            "replica-respecting hierarchical uncertainty calculation"
        ),
    }
    return curves, replica_frame, sensitivity, metadata


def make_figure(curves: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    titles = {
        "unconstrained_orbit": "Unconstrained orbit weight",
        "constrained_orbit": "Constrained orbit weight",
        "liquid_cavity_branch": "Liquid cavity branch",
        "gas_cavity_branch": "Gas cavity branch",
    }
    palette = ("#56B4E9", "#009E73", "#E69F00", "#D55E00", "#7B3294")
    block_curves = curves[curves["analysis_unit"] == "replica_time_block"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), constrained_layout=True)
    for axis, estimator in zip(axes.flat, titles):
        selected = block_curves[block_curves["estimator"] == estimator]
        for color, (length, group) in zip(palette, selected.groupby("L")):
            axis.plot(
                group["h"],
                group["kish_fraction"],
                color=color,
                linewidth=1.0,
                label=rf"$L={int(length)}$",
            )
        for threshold, style in ((0.10, "--"), (0.20, ":")):
            axis.axhline(
                threshold,
                color="black",
                linestyle=style,
                linewidth=0.75,
                label=rf"support {threshold:.2f}" if estimator == "unconstrained_orbit" else None,
            )
        axis.set_title(titles[estimator])
        axis.set_xlabel(r"$h=V(x-x_{\rm coex})$")
        axis.set_ylabel("Kish weighted-support fraction")
        axis.set_ylim(0.0, 1.03)
        axis.grid(alpha=0.16, linewidth=0.5)
    axes.flat[0].legend(frameon=False, ncol=2)
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
    parser.add_argument("--blocks-per-replica", type=int, default=DEFAULT_BLOCK_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves, replica_weights, sensitivity, metadata = analyze(
        args.raw, args.blocks_per_replica
    )
    curves.to_csv(
        ROOT / "data_processed" / "lattice_reweighting_overlap.csv", index=False
    )
    replica_weights.to_csv(
        ROOT / "data_processed" / "lattice_reweighting_replica_weights.csv",
        index=False,
    )
    sensitivity.to_csv(
        ROOT / "data_processed" / "lattice_support_interval_sensitivity.csv",
        index=False,
    )
    minima = (
        curves.loc[
            lambda frame: frame.groupby(
                ["L", "estimator", "analysis_unit"]
            )["kish_fraction"].idxmin()
        ]
        .sort_values(["analysis_unit", "L", "estimator"])
        .reset_index(drop=True)
    )
    minima.to_csv(
        ROOT / "generated" / "lattice_reweighting_overlap_minima.csv",
        index=False,
    )
    make_figure(curves, ROOT / "figures" / "lattice_reweighting_overlap.pdf")
    (ROOT / "generated" / "lattice_reweighting_overlap_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    weakest = minima.loc[minima["kish_fraction"].idxmin()]
    print(
        "weakest weighted support: "
        f"L={int(weakest['L'])}, {weakest['estimator']}, "
        f"unit={weakest['analysis_unit']}, "
        f"Kish fraction={weakest['kish_fraction']:.3f}, "
        f"perplexity fraction={weakest['perplexity_fraction']:.3f}, "
        f"max normalized weight={weakest['maximum_normalized_weight']:.3f}, "
        f"h={weakest['h']:.2f}"
    )


if __name__ == "__main__":
    main()
