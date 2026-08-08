#!/usr/bin/env python3
"""Finite-volume cumulant and cavity benchmarks by EOS reweighting.

This script generates two deterministic numerical checks:

1. A finite-volume verification of the constraint cumulant hierarchy using
   independent high-order finite differences of the log-sum cavity work.
2. A convergence test of the canonical--grand-canonical mixing correction.

The calculation is a discrete particle-number reweighting based on the
Percus--Yevick compressibility-route free energy and its consistent
scaled-particle cavity polynomial.  It is not labelled or used as direct
particle-resolved Monte Carlo data.
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

from hard_sphere_models import (
    cavity_conditioned_distribution,
    cavity_work_at_activity,
    cumulants_from_distribution,
    finite_difference_weights,
    grand_distribution_3d,
    number_density,
    pyc_s0,
    spt_cavity_work,
    spt_density_derivatives,
    tune_activity_for_mean,
)

TARGET_N_CUMULANT = (64, 128, 256, 512)
TARGET_N_FINITE = (64, 108, 256, 500, 864, 1372)
ETA_CUMULANT = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
ETA_FINITE = (0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.48, 0.49)
RADII_CUMULANT = (0.50, 0.75, 1.00)
RADII_FINITE = (0.50, 0.75, 1.00, 1.25, 1.50)


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


def add_panel_labels(axes: np.ndarray) -> None:
    """Add publication-style labels without competing with data or legends."""
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", axes.flat):
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


def derivative_stencil(
    x: float, volume: float, radius: float, target_n: int, step: float
) -> np.ndarray:
    offsets = np.arange(-5, 6, dtype=float)
    values = np.asarray(
        [
            cavity_work_at_activity(x + step * offset, volume, radius, target_n)
            for offset in offsets
        ]
    )
    return np.asarray(
        [
            np.dot(finite_difference_weights(offsets, order), values) / step**order
            for order in range(1, 5)
        ]
    )


def make_cumulant_data() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    derivative_floor = np.asarray([2.0e-10, 2.0e-9, 2.0e-8, 2.0e-7])
    for target_n in TARGET_N_CUMULANT:
        for eta in ETA_CUMULANT:
            rho = float(number_density(eta))
            volume = target_n / rho
            x = tune_activity_for_mean(target_n, volume, eta)
            base = grand_distribution_3d(x, volume, target_n)
            base_cumulants = cumulants_from_distribution(
                base.n, base.probability, max_order=4
            )
            for radius in RADII_CUMULANT:
                conditioned, work = cavity_conditioned_distribution(
                    base, volume, radius
                )
                conditioned_cumulants = cumulants_from_distribution(
                    base.n, conditioned, max_order=4
                )
                cumulant_difference = base_cumulants - conditioned_cumulants
                derivative_a = derivative_stencil(
                    x, volume, radius, target_n, step=0.10
                )
                derivative_b = derivative_stencil(
                    x, volume, radius, target_n, step=0.12
                )
                derivative_error = np.maximum(
                    np.abs(derivative_a - derivative_b), derivative_floor
                )
                cumulant_error = 5.0e-11 * (
                    1.0 + np.abs(base_cumulants) + np.abs(conditioned_cumulants)
                )
                for order in range(1, 5):
                    combined = np.hypot(
                        derivative_error[order - 1], cumulant_error[order - 1]
                    )
                    residual = derivative_a[order - 1] - cumulant_difference[order - 1]
                    rows.append(
                        {
                            "V": volume,
                            "meanN": target_n,
                            "x": x,
                            "eta": eta,
                            "R_over_sigma": radius,
                            "m": order,
                            "betaW": work,
                            "derivative": derivative_a[order - 1],
                            "se_derivative": derivative_error[order - 1],
                            "cumulant_difference": cumulant_difference[order - 1],
                            "se_cumulant_difference": cumulant_error[order - 1],
                            "standardized_residual": residual / combined,
                            "estimate_type": "PYc_EOS_discrete_reweighting",
                            "source_type": "eos_model",
                            "run_id": "grand_reweight_v1",
                            "seed": np.nan,
                            "units_convention": "beta=sigma=Lambda=1",
                            "raw_data_path": "none_deterministic",
                            "analysis_version": "1",
                        }
                    )
    return pd.DataFrame(rows)


def make_finite_size_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for eta in ETA_FINITE:
        rho = float(number_density(eta))
        s0 = float(pyc_s0(eta))
        chi_n = rho * s0
        for radius in RADII_FINITE:
            w_prime, w_second = spt_density_derivatives(eta, radius)
            predicted = 0.5 * chi_n * (w_second - w_prime**2)
            if radius == 0.50:
                predicted = 0.0
            canonical_work = float(spt_cavity_work(eta, radius))
            for target_n in TARGET_N_FINITE:
                volume = target_n / rho
                x = tune_activity_for_mean(target_n, volume, eta)
                base = grand_distribution_3d(x, volume, target_n)
                _, grand_work = cavity_conditioned_distribution(base, volume, radius)
                if radius == 0.50:
                    # P(empty point-particle cavity)=1-eta_N is affine in N,
                    # so the mixture equality at the mean density is exact.
                    grand_work = canonical_work
                rows.append(
                    {
                        "N": target_n,
                        "V": volume,
                        "x": x,
                        "eta": eta,
                        "R_over_sigma": radius,
                        "betaW_N": canonical_work,
                        "se_betaW_N": np.nan,
                        "betaW_mu": grand_work,
                        "se_betaW_mu": np.nan,
                        "predicted_coefficient": predicted,
                        "se_predicted_coefficient": np.nan,
                        "V_delta_betaW": volume * (grand_work - canonical_work),
                        "estimate_type": "PYc_SPT_EOS_discrete_reweighting",
                        "source_type": "eos_model",
                        "run_id": "grand_reweight_v1",
                        "seed": np.nan,
                        "units_convention": "beta=sigma=Lambda=1",
                        "raw_data_path": "none_deterministic",
                        "analysis_version": "1",
                    }
                )
    data = pd.DataFrame(rows)

    fits: list[dict[str, float]] = []
    for (eta, radius), group in data.groupby(["eta", "R_over_sigma"]):
        inverse_v = 1.0 / group["V"].to_numpy()
        scaled_difference = group["V_delta_betaW"].to_numpy()
        if radius == 0.50:
            intercept = 0.0
            slope = 0.0
            quadratic = 0.0
        else:
            quadratic, slope, intercept = np.polyfit(
                inverse_v, scaled_difference, deg=2
            )
        predicted = float(group["predicted_coefficient"].iloc[0])
        fits.append(
            {
                "eta": eta,
                "R_over_sigma": radius,
                "fitted_intercept": intercept,
                "predicted_coefficient": predicted,
                "absolute_difference": intercept - predicted,
                "relative_difference": (
                    (intercept - predicted) / predicted
                    if abs(predicted) > 1.0e-10
                    else np.nan
                ),
                "linear_coefficient": slope,
                "quadratic_coefficient": quadratic,
            }
        )
    return data, pd.DataFrame(fits)


def plot_cumulants(data: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    colors = {0.50: "#0072B2", 0.75: "#D55E00", 1.00: "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
    representative = data[data["meanN"] == 256]
    for order, axis in enumerate(axes.flat, start=1):
        subset = representative[representative["m"] == order]
        for radius, group in subset.groupby("R_over_sigma"):
            group = group.sort_values("eta")
            axis.plot(
                group["eta"],
                group["cumulant_difference"],
                color=colors[radius],
                linewidth=1.0,
                label=rf"$R/\sigma={radius:.2f}$",
            )
            axis.errorbar(
                group["eta"],
                group["derivative"],
                yerr=group["se_derivative"],
                linestyle="none",
                marker="o",
                markersize=3.2,
                markerfacecolor="white",
                markeredgewidth=0.75,
                color=colors[radius],
                rasterized=False,
            )
        axis.axhline(0.0, color="0.35", linewidth=0.55, zorder=0)
        axis.set_title(rf"Order $m={order}$")
        axis.set_xlabel(r"packing fraction $\eta$")
        axis.set_ylabel(rf"order-{order} response")
        axis.grid(alpha=0.16, linewidth=0.5)
    axes[0, 0].legend(frameon=False, loc="best")
    add_panel_labels(axes)
    fig.savefig(output)
    plt.close(fig)


def plot_finite_size(data: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    selected_eta = (0.10, 0.30, 0.45, 0.49)
    palette = {
        0.10: "#56B4E9",
        0.30: "#009E73",
        0.45: "#E69F00",
        0.49: "#CC79A7",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), constrained_layout=True)
    for axis, radius in zip(axes.flat, RADII_FINITE[1:]):
        subset_r = data[data["R_over_sigma"] == radius]
        for eta in selected_eta:
            group = subset_r[np.isclose(subset_r["eta"], eta)].sort_values("V")
            inverse_v = 1.0 / group["V"].to_numpy()
            values = group["V_delta_betaW"].to_numpy()
            predicted = float(group["predicted_coefficient"].iloc[0])
            axis.plot(
                inverse_v,
                100.0 * (values / predicted - 1.0),
                "o-",
                color=palette[eta],
                markersize=3.0,
                linewidth=0.7,
                label=rf"$\eta={eta:.2f}$",
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        axis.set_title(rf"$R/\sigma={radius:.2f}$")
        axis.set_xlabel(r"$1/V$")
        axis.grid(alpha=0.16, linewidth=0.5)
    axes.flat[0].legend(frameon=False, ncol=2, loc="best")
    fig.supylabel("deviation from leading coefficient (percent)", fontsize=9)
    add_panel_labels(axes)
    fig.savefig(output)
    plt.close(fig)


def write_summary_table(fits: pd.DataFrame, output: Path) -> None:
    selected = fits[
        fits["eta"].isin([0.10, 0.30, 0.45, 0.49])
        & fits["R_over_sigma"].isin([0.75, 1.00, 1.25, 1.50])
    ]
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$\eta$ & $R/\sigma$ & fitted $V\Delta(\beta W)$ & prediction \\",
        r"\midrule",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"{row.eta:.2f} & {row.R_over_sigma:.2f} & "
            f"{row.fitted_intercept:.5g} & {row.predicted_coefficient:.5g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(output: Path, cumulants: pd.DataFrame, finite: pd.DataFrame) -> None:
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/grand_reweight.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "random_seed": None,
        "method": "deterministic discrete EOS reweighting",
        "eos": "Percus-Yevick compressibility route",
        "cavity_model": "consistent scaled-particle cubic insertion polynomial",
        "cumulant_rows": int(len(cumulants)),
        "finite_size_rows": int(len(finite)),
        "statistical_standard_errors": False,
        "note": (
            "CSV columns named se_* are retained for schema compatibility; "
            "for deterministic finite-size values they are blank. Cumulant "
            "se_* columns are conservative numerical differentiation/roundoff "
            "error estimates, not Monte Carlo sampling errors."
        ),
    }
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    data_dir = root / "data_processed"
    figure_dir = root / "figures"
    generated_dir = root / "generated"
    for directory in (data_dir, figure_dir, generated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cumulants = make_cumulant_data()
    finite, fits = make_finite_size_data()
    cumulants.to_csv(data_dir / "cumulant_test.csv", index=False)
    finite.to_csv(data_dir / "finite_size_cavity.csv", index=False)
    fits.to_csv(generated_dir / "finite_size_fits.csv", index=False)
    plot_cumulants(cumulants, figure_dir / "cumulant_hierarchy.pdf")
    plot_finite_size(finite, figure_dir / "finite_size_collapse.pdf")
    write_summary_table(fits, generated_dir / "finite_size_table.tex")
    write_metadata(generated_dir / "grand_reweight_metadata.json", cumulants, finite)

    standardized = np.abs(cumulants["standardized_residual"].to_numpy())
    within_three = float(np.mean(standardized <= 3.0))
    print(f"wrote {len(cumulants)} cumulant rows; fraction within 3 errors={within_three:.4f}")
    print(f"wrote {len(finite)} finite-size rows and {len(fits)} asymptotic fits")


if __name__ == "__main__":
    main()
