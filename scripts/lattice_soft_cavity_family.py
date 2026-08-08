#!/usr/bin/env python3
"""Tunable soft two-site cavity benchmark from the archived lattice chains.

For a uniformly sampled nearest-neighbor bond, let k be the number of occupied
sites and use the fixed bias chi_lambda = exp(-lambda k).  Lambda=0 is the
identity weight and lambda=infinity is the hard empty-pair cavity.  The stored
magnetization and bond sum determine the spatial fractions with k=0,1,2 for
every configuration, so the complete family can be evaluated without a new
Monte Carlo campaign.
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

try:
    from .lattice_analysis_common import (
        hierarchical_bootstrap_totals,
        load_trajectory_archives,
        replicas_for_size,
    )
    from .lattice_gas_phase_switch import (
        CONTRIBUTION_NAMES,
        K_ISING,
        exact_nearest_neighbor_correlation,
        exact_spontaneous_magnetization,
        observables_from_contributions,
        orient_symmetry_pairs,
        phase_theory,
    )
    from .lattice_pulse_scaling import (
        analytic_pulse_limits,
        continuous_pulse_observables,
    )
except ImportError:
    from lattice_analysis_common import (
        hierarchical_bootstrap_totals,
        load_trajectory_archives,
        replicas_for_size,
    )
    from lattice_gas_phase_switch import (
        CONTRIBUTION_NAMES,
        K_ISING,
        exact_nearest_neighbor_correlation,
        exact_spontaneous_magnetization,
        observables_from_contributions,
        orient_symmetry_pairs,
        phase_theory,
    )
    from lattice_pulse_scaling import (
        analytic_pulse_limits,
        continuous_pulse_observables,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = [
    ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz",
    ROOT / "data_raw" / "lattice_gas_additional_replicas.npz",
]
SOFT_LAMBDAS = (0.25, 0.5, 1.0, 2.0, float("inf"))
H_GRID = np.linspace(-3.0, 8.0, 221)
DEFAULT_SEED = 26_080_611


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lambda_label(value: float) -> str:
    return "infinity" if np.isinf(value) else f"{value:g}"


def coupling_coordinate(value: float) -> float:
    """Map lambda in [0,infinity] to the compact staging coordinate [0,1]."""

    if value < 0.0:
        raise ValueError("lambda must be nonnegative")
    return 1.0 if np.isinf(value) else float(1.0 - np.exp(-value))


def soft_pair_weights(
    pair_empty_liquid: np.ndarray,
    pair_empty_gas: np.ndarray,
    coupling_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spatially averaged soft-bond weights in a symmetry pair."""

    if coupling_lambda < 0.0:
        raise ValueError("lambda must be nonnegative")
    p00_liquid = np.asarray(pair_empty_liquid, dtype=float)
    p00_gas = np.asarray(pair_empty_gas, dtype=float)
    if p00_liquid.shape != p00_gas.shape:
        raise ValueError("paired empty-bond arrays must have the same shape")
    one_occupied = 1.0 - p00_liquid - p00_gas
    stage = 0.0 if np.isinf(coupling_lambda) else float(np.exp(-coupling_lambda))
    liquid = p00_liquid + stage * one_occupied + stage**2 * p00_gas
    gas = p00_gas + stage * one_occupied + stage**2 * p00_liquid
    return liquid, gas


def exact_soft_pair_parameters(
    coupling: float, coupling_lambda: float
) -> tuple[float, float, float]:
    """Return exact liquid/gas soft weights and their log-work contrast."""

    magnetization = exact_spontaneous_magnetization(coupling)
    correlation = exact_nearest_neighbor_correlation(coupling)
    p00_liquid = (1.0 - 2.0 * magnetization + correlation) / 4.0
    p00_gas = (1.0 + 2.0 * magnetization + correlation) / 4.0
    one_occupied = (1.0 - correlation) / 2.0
    stage = 0.0 if np.isinf(coupling_lambda) else float(np.exp(-coupling_lambda))
    p_liquid = p00_liquid + stage * one_occupied + stage**2 * p00_gas
    p_gas = p00_gas + stage * one_occupied + stage**2 * p00_liquid
    return p_liquid, p_gas, float(np.log(p_gas / p_liquid))


def replica_soft_contributions(
    fields: dict[str, np.ndarray],
    volume: int,
    h_grid: np.ndarray,
    coupling_lambda: float,
    block_count: int,
) -> np.ndarray:
    """Return block contributions with axes block, field, contribution."""

    if block_count < 2:
        raise ValueError("at least two time blocks are required")

    magnetization = np.asarray(fields["magnetization"])
    bond_sum = np.asarray(fields["bond_sum"])
    site_zero = np.asarray(fields["site_zero"])
    oriented_m, _ = orient_symmetry_pairs(magnetization, site_zero)
    n_liquid = (volume + oriented_m.astype(np.int64)) // 2
    n_gas = volume - n_liquid
    p00_liquid = (2.0 * volume - 4.0 * oriented_m + bond_sum) / (8.0 * volume)
    p00_gas = (2.0 * volume + 4.0 * oriented_m + bond_sum) / (8.0 * volume)
    c_liquid, c_gas = soft_pair_weights(
        p00_liquid, p00_gas, coupling_lambda
    )
    centered_density = n_liquid / volume - 0.5
    u_liquid = np.exp(np.outer(np.asarray(h_grid, dtype=float), centered_density))
    u_gas = np.exp(-np.outer(np.asarray(h_grid, dtype=float), centered_density))

    blocks = [index for index in np.array_split(np.arange(len(n_liquid)), block_count) if len(index)]
    if len(blocks) != block_count:
        raise ValueError("trajectory cannot be divided into the requested blocks")
    starts = np.asarray([int(index[0]) for index in blocks], dtype=int)

    def reduce_blocks(weights: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
        weighted = weights if values is None else weights * values[None, :]
        return np.add.reduceat(weighted, starts, axis=1).T

    z_liquid = reduce_blocks(u_liquid)
    z_gas = reduce_blocks(u_gas)
    n_liquid_sum = reduce_blocks(u_liquid, n_liquid)
    n_gas_sum = reduce_blocks(u_gas, n_gas)
    c_liquid_sum = reduce_blocks(u_liquid, c_liquid)
    c_gas_sum = reduce_blocks(u_gas, c_gas)
    cn_liquid_sum = reduce_blocks(u_liquid, c_liquid * n_liquid)
    cn_gas_sum = reduce_blocks(u_gas, c_gas * n_gas)
    zeros = np.zeros_like(z_liquid)
    output = np.stack(
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
            zeros,
            zeros,
        ),
        axis=-1,
    )
    if output.shape[-1] != len(CONTRIBUTION_NAMES):
        raise AssertionError("unexpected contribution schema")
    return output


def analyze(
    raw_paths: list[Path],
    length: int,
    block_count: int,
    bootstrap_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if block_count < 2 or bootstrap_draws < 20:
        raise ValueError("use at least two blocks and 20 bootstrap draws")
    trajectories = load_trajectory_archives(raw_paths)
    replicas = replicas_for_size(trajectories, length)
    if replicas != list(range(12)):
        raise ValueError(
            f"L={length} must contain replica IDs 0--11; got {replicas}"
        )
    volume = length * length
    rng = np.random.default_rng(seed)
    exact_magnetization = exact_spontaneous_magnetization(K_ISING)
    feature_records: list[dict[str, float | int | str]] = []
    curve_records: list[dict[str, float | int | str]] = []

    for coupling_lambda in SOFT_LAMBDAS:
        block_arrays = np.asarray(
            [
                replica_soft_contributions(
                    trajectories[(length, replica)],
                    volume,
                    H_GRID,
                    coupling_lambda,
                    block_count,
                )
                for replica in replicas
            ]
        )
        central = np.sum(block_arrays, axis=(0, 1))
        observed = observables_from_contributions(central, volume)
        response = observed["deltaN_phase_reweighting"] / volume
        measured = continuous_pulse_observables(H_GRID, response)
        center_index = int(np.argmin(np.abs(H_GRID)))

        bootstrap = {
            "delta_betaW": np.empty(bootstrap_draws, dtype=float),
            **{
                name: np.empty(bootstrap_draws, dtype=float)
                for name in ("peak_center_h", "peak_magnitude", "fwhm_h")
            },
        }
        for draw, total in enumerate(
            hierarchical_bootstrap_totals(block_arrays, bootstrap_draws, rng)
        ):
            draw_observed = observables_from_contributions(total, volume)
            draw_features = continuous_pulse_observables(
                H_GRID,
                draw_observed["deltaN_phase_reweighting"] / volume,
            )
            bootstrap["delta_betaW"][draw] = draw_observed["delta_betaW"][
                center_index
            ]
            for name, value in draw_features.items():
                bootstrap[name][draw] = value

        _, _, exact_a = exact_soft_pair_parameters(K_ISING, coupling_lambda)
        exact = analytic_pulse_limits(exact_magnetization, exact_a)
        record: dict[str, float | int | str] = {
            "lambda_label": lambda_label(coupling_lambda),
            "lambda": coupling_lambda,
            "staging_coordinate": coupling_coordinate(coupling_lambda),
            "L": length,
            "V": volume,
            "replicas": len(replicas),
            "blocks_per_replica": block_count,
            "bootstrap_draws": bootstrap_draws,
            "exact_delta_betaW": exact_a,
            "measured_delta_betaW": float(observed["delta_betaW"][center_index]),
            "se_delta_betaW": float(np.std(bootstrap["delta_betaW"], ddof=1)),
        }
        record["difference_z_delta_betaW"] = (
            float(record["measured_delta_betaW"]) - exact_a
        ) / float(record["se_delta_betaW"])
        for name in ("peak_center_h", "peak_magnitude", "fwhm_h"):
            standard_error = float(np.std(bootstrap[name], ddof=1))
            record[f"exact_{name}"] = exact[name]
            record[f"measured_{name}"] = measured[name]
            record[f"se_{name}"] = standard_error
            record[f"difference_z_{name}"] = (
                measured[name] - exact[name]
            ) / standard_error
        feature_records.append(record)

        exact_curve = -phase_theory(H_GRID, exact_magnetization, exact_a)
        for h_value, measured_value, exact_value in zip(
            H_GRID, -response, exact_curve
        ):
            curve_records.append(
                {
                    "lambda_label": lambda_label(coupling_lambda),
                    "lambda": coupling_lambda,
                    "staging_coordinate": coupling_coordinate(coupling_lambda),
                    "L": length,
                    "V": volume,
                    "h": h_value,
                    "measured_negative_reweighting_over_V": measured_value,
                    "exact_negative_reweighting_over_V": exact_value,
                }
            )

    features = pd.DataFrame(feature_records)
    curves = pd.DataFrame(curve_records)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_soft_cavity_family.py",
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
        "seed": seed,
        "L": length,
        "replica_ids": replicas,
        "blocks_per_replica": block_count,
        "bootstrap_draws": bootstrap_draws,
        "lambdas": [lambda_label(value) for value in SOFT_LAMBDAS],
        "bias": "chi_lambda = exp[-lambda * (n_i+n_j)] for a uniformly sampled nearest-neighbor bond",
        "reuse_scope": "same archived unconstrained trajectories as the hard-cavity benchmark; this is a staged-weight family, not an independent simulation ensemble",
        "comparison_scope": (
            "reported z scores compare the finite L measurement directly with "
            "the exact thermodynamic limit; they diagnose finite-size displacement "
            "and are not thermodynamic-limit extrapolation tests"
        ),
        "exact_soft_weight": (
            "p_lambda=p00+exp(-lambda)*p01+exp(-2 lambda)*p11"
        ),
        "exact_work_contrast": "Delta beta W=log(p_lambda,gas/p_lambda,liquid)",
    }
    return features, curves, metadata


def make_figure(features: pd.DataFrame, curves: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.6,
            "axes.labelsize": 9.1,
            "axes.titlesize": 9.1,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    colors = ("#56B4E9", "#009E73", "#E69F00", "#D55E00", "#7B3294")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 5.15),
        gridspec_kw={"wspace": 0.24, "hspace": 0.16},
        constrained_layout=True,
    )
    curve_axis, work_axis, magnitude_axis, width_axis = axes.flat

    ordered = features.sort_values("staging_coordinate")
    for color, (_, row) in zip(colors, ordered.iterrows()):
        selected = curves[curves["lambda_label"] == row["lambda_label"]]
        label = r"$\lambda=\infty$" if row["lambda_label"] == "infinity" else rf"$\lambda={row['lambda_label']}$"
        curve_axis.plot(
            selected["h"],
            selected["measured_negative_reweighting_over_V"],
            color=color,
            linewidth=1.15,
            label=label,
        )
        curve_axis.plot(
            selected["h"],
            selected["exact_negative_reweighting_over_V"],
            color=color,
            linewidth=0.9,
            linestyle="--",
        )
    curve_axis.set_title(r"Measured $L=64$ curves (solid); exact limits (dashed)")
    curve_axis.set_xlabel(r"$h=V(x-x_{\rm coex})$")
    curve_axis.set_ylabel(r"$-\Delta N_{\rm rw}/V$")
    curve_axis.legend(
        frameon=False,
        ncol=1,
        fontsize=6.8,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
    )
    curve_axis.grid(alpha=0.16, linewidth=0.5)

    x = ordered["staging_coordinate"].to_numpy()
    panel_specs = (
        (
            work_axis,
            "measured_delta_betaW",
            "se_delta_betaW",
            "exact_delta_betaW",
            r"$\beta\Delta W$",
            "Phase-work contrast",
        ),
        (
            magnitude_axis,
            "measured_peak_magnitude",
            "se_peak_magnitude",
            "exact_peak_magnitude",
            r"$|\Delta N_{\rm rw}|_{\rm peak}/V$",
            "Peak magnitude",
        ),
        (
            width_axis,
            "measured_fwhm_h",
            "se_fwhm_h",
            "exact_fwhm_h",
            r"$V\Delta x_{\rm FWHM}$",
            "Pulse width",
        ),
    )
    for axis, measured, error, exact, ylabel, title in panel_specs:
        axis.plot(x, ordered[exact], color="black", linewidth=1.1, label="exact limit")
        axis.errorbar(
            x,
            ordered[measured],
            yerr=ordered[error],
            fmt="o",
            color="#0072B2",
            markersize=4.0,
            capsize=2.0,
            label=r"$L=64$",
        )
        axis.set_title(title)
        axis.set_xlabel(r"staging coordinate $1-e^{-\lambda}$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.16, linewidth=0.5)
    work_axis.legend(frameon=False)
    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.105,
            1.14,
            f"({label})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.4,
            clip_on=False,
        )
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", default=DEFAULT_RAW)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--blocks-per-replica", type=int, default=40)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    raw_paths = [Path(path) for path in args.raw]
    for path in raw_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    features, curves, metadata = analyze(
        raw_paths,
        args.length,
        args.blocks_per_replica,
        args.bootstrap_draws,
        args.seed,
    )
    feature_path = ROOT / "data_processed" / "lattice_soft_cavity_features.csv"
    curve_path = ROOT / "data_processed" / "lattice_soft_cavity_curves.csv"
    figure_path = ROOT / "figures" / "lattice_soft_cavity_family.pdf"
    metadata_path = ROOT / "generated" / "lattice_soft_cavity_metadata.json"
    for directory in (
        feature_path.parent,
        figure_path.parent,
        metadata_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_path, index=False)
    curves.to_csv(curve_path, index=False)
    make_figure(features, curves, figure_path)
    metadata["output_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (feature_path, curve_path, figure_path)
    }
    metadata["runtime_seconds"] = float(time.perf_counter() - started)
    metadata["reproduce_command"] = (
        "python3 scripts/lattice_soft_cavity_family.py "
        f"--length {args.length} --blocks-per-replica {args.blocks_per_replica} "
        f"--bootstrap-draws {args.bootstrap_draws} --seed {args.seed}"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        features[
            [
                "lambda_label",
                "exact_delta_betaW",
                "measured_delta_betaW",
                "se_delta_betaW",
                "difference_z_delta_betaW",
                "difference_z_peak_center_h",
                "difference_z_peak_magnitude",
                "difference_z_fwhm_h",
            ]
        ].to_string(index=False)
    )
    print(f"runtime_seconds={metadata['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
