#!/usr/bin/env python3
"""Published-data-anchored coexistence reweighting calculation.

For a spatially averaged particle-sized cavity in an infinite-volume
thermodynamic pure Gibbs phase, the Georgii--Nguyen--Zessin identity gives

    exp(-beta W_alpha) = rho_alpha Lambda^3 exp(-beta mu).

For a finite basin-restricted sequence, convergence to this relation requires
the explicit insertion-discrepancy term described in the manuscript to vanish.
Consequently, at equal unconstrained phase weights the constrained solid
weight is rho_s/(rho_f + rho_s), and the leading extensive reweighting term is
fully determined asymptotically by the two coexistence densities.  This script
combines that thermodynamic-limit relation with the finite-size coexistence
densities reported by Fernandez et al. (PRL 108, 165701, 2012), producing a
data-anchored finite-size proxy and power analysis.  It is a new calculation
from published summary statistics, not a reanalysis of their trajectories and
not a new phase-switch simulation.
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
from scipy.optimize import brentq

from hard_sphere_models import PI_OVER_SIX, cs_s0

SEED = 24072026
BOOTSTRAP_DRAWS = 20_000
H_GRID = np.linspace(-40.0, 40.0, 161)
ETA_CLOSE_PACKED = np.pi / (3.0 * np.sqrt(2.0))


def logistic(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def solid_s0_cell(eta_s: np.ndarray, pressure: np.ndarray, rho_s: np.ndarray) -> np.ndarray:
    """Pressure-anchored free-volume estimate of the solid compressibility.

    Locally write Z=A/(1-eta/eta_cp), with A selected so the reported
    coexistence pressure is reproduced at the reported solid density.  Then
    S(0)=[d(beta p)/d rho]^{-1}=(1-r)^2/A.
    """

    ratio = eta_s / ETA_CLOSE_PACKED
    amplitude = (pressure / rho_s) * (1.0 - ratio)
    return (1.0 - ratio) ** 2 / amplitude


def finite_size_observables(
    rho_f: np.ndarray,
    rho_s: np.ndarray,
    pressure: np.ndarray,
    volume: float,
    h_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the exact two-phase mixture for arrays of density samples."""

    rho_f = np.asarray(rho_f, dtype=float).reshape(-1, 1)
    rho_s = np.asarray(rho_s, dtype=float).reshape(-1, 1)
    pressure = np.asarray(pressure, dtype=float).reshape(-1, 1)
    h_values = np.asarray(h_values, dtype=float).reshape(1, -1)

    delta_rho = rho_s - rho_f
    delta_w = np.log(rho_f / rho_s)
    y = delta_rho * h_values
    q0_s = logistic(y)
    q_c_s = logistic(y - delta_w)
    q0_f = 1.0 - q0_s
    q_c_f = 1.0 - q_c_s

    eta_f = PI_OVER_SIX * rho_f
    eta_s = PI_OVER_SIX * rho_s
    d_f = cs_s0(eta_f) - 1.0
    d_s = solid_s0_cell(eta_s, pressure, rho_s) - 1.0
    intraphase = q_c_f * d_f + q_c_s * d_s
    reweight = (q_c_s - q0_s) * volume * delta_rho
    total = intraphase + reweight
    scaled_prediction = delta_rho * (q_c_s - q0_s)

    return {
        "delta_rho": np.broadcast_to(delta_rho, q0_s.shape),
        "delta_w": np.broadcast_to(delta_w, q0_s.shape),
        "q0_f": q0_f,
        "q0_s": q0_s,
        "qC_f": q_c_f,
        "qC_s": q_c_s,
        "d_f": np.broadcast_to(d_f, q0_s.shape),
        "d_s": np.broadcast_to(d_s, q0_s.shape),
        "intraphase": intraphase,
        "reweight": reweight,
        "total": total,
        "scaled_prediction": scaled_prediction,
    }


def summarize_bootstrap(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(samples, axis=0), np.std(samples, axis=0, ddof=1)


def make_phase_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    detailed_rows: list[dict[str, float | int | str]] = []
    equal_weight_rows: list[dict[str, float | int | str]] = []
    finite = raw[raw["N"] != "infinity"].copy()

    for source in finite.itertuples(index=False):
        n_fluid = int(source.N)
        v_f = float(source.v_fluid)
        v_s = float(source.v_solid)
        rho_f = 1.0 / v_f
        rho_s = 1.0 / v_s
        volume = n_fluid * v_f
        pressure = float(source.p_equal_weight)

        obs = finite_size_observables(
            np.asarray([rho_f]),
            np.asarray([rho_s]),
            np.asarray([pressure]),
            volume,
            H_GRID,
        )

        v_f_draw = rng.normal(v_f, float(source.se_v_fluid), BOOTSTRAP_DRAWS)
        v_s_draw = rng.normal(v_s, float(source.se_v_solid), BOOTSTRAP_DRAWS)
        p_draw = rng.normal(
            pressure, float(source.se_p_equal_weight), BOOTSTRAP_DRAWS
        )
        bootstrap = finite_size_observables(
            1.0 / v_f_draw, 1.0 / v_s_draw, p_draw, volume, H_GRID
        )
        means: dict[str, np.ndarray] = {}
        errors: dict[str, np.ndarray] = {}
        for key, values in bootstrap.items():
            means[key], errors[key] = summarize_bootstrap(values)

        mean_n0_f = rho_f * volume
        mean_n0_s = rho_s * volume
        for index, h_value in enumerate(H_GRID):
            mu_relative = h_value / volume
            for phase in ("fluid", "solid"):
                if phase == "fluid":
                    q0 = float(obs["q0_f"][0, index])
                    qc = float(obs["qC_f"][0, index])
                    d_phase = float(obs["d_f"][0, index])
                    mean_n0 = mean_n0_f
                    q0_error = float(errors["q0_f"][index])
                    qc_error = float(errors["qC_f"][index])
                    d_error = float(errors["d_f"][index])
                else:
                    q0 = float(obs["q0_s"][0, index])
                    qc = float(obs["qC_s"][0, index])
                    d_phase = float(obs["d_s"][0, index])
                    mean_n0 = mean_n0_s
                    q0_error = float(errors["q0_s"][index])
                    qc_error = float(errors["qC_s"][index])
                    d_error = float(errors["d_s"][index])
                detailed_rows.append(
                    {
                        "V": volume,
                        "N_fluid_reference": n_fluid,
                        "mu": mu_relative,
                        "mu_equal_weight": 0.0,
                        "h": h_value,
                        "phase": phase,
                        "basin_threshold": "published_tethered_phase_minimum",
                        "R_over_sigma": 1.0,
                        "q0": q0,
                        "se_q0": q0_error,
                        "qC": qc,
                        "se_qC": qc_error,
                        "meanN0": mean_n0,
                        "meanNC": mean_n0 + d_phase,
                        "deltaN_phase": d_phase,
                        "se_deltaN_phase": d_error,
                        "deltaN_intra": float(obs["intraphase"][0, index]),
                        "se_deltaN_intra": float(errors["intraphase"][index]),
                        "deltaN_reweight": float(obs["reweight"][0, index]),
                        "se_deltaN_reweight": float(errors["reweight"][index]),
                        "deltaN_total": float(obs["total"][0, index]),
                        "se_deltaN_total": float(errors["total"][index]),
                        "delta_rho": float(obs["delta_rho"][0, index]),
                        "se_delta_rho": float(errors["delta_rho"][index]),
                        "delta_w": float(obs["delta_w"][0, index]),
                        "se_delta_w": float(errors["delta_w"][index]),
                        "scaled_prediction": float(
                            obs["scaled_prediction"][0, index]
                        ),
                        "se_scaled_prediction": float(
                            errors["scaled_prediction"][index]
                        ),
                        "estimate_type": "asymptotic_pure_phase_relation_plus_published_densities",
                        "source_type": "external_reference_plus_analytic_asymptotic",
                        "units_convention": "beta=sigma=Lambda=1; mu relative to equal weight",
                    }
                )

        center = int(np.argmin(np.abs(H_GRID)))
        equal_weight_rows.append(
            {
                "N_fluid_reference": n_fluid,
                "V": volume,
                "rho_f": rho_f,
                "rho_s": rho_s,
                "eta_f": PI_OVER_SIX * rho_f,
                "eta_s": PI_OVER_SIX * rho_s,
                "qC_s": float(obs["qC_s"][0, center]),
                "se_qC_s": float(errors["qC_s"][center]),
                "delta_w": float(obs["delta_w"][0, center]),
                "se_delta_w": float(errors["delta_w"][center]),
                "deltaN_intra": float(obs["intraphase"][0, center]),
                "se_deltaN_intra": float(errors["intraphase"][center]),
                "deltaN_reweight": float(obs["reweight"][0, center]),
                "se_deltaN_reweight": float(errors["reweight"][center]),
                "deltaN_total": float(obs["total"][0, center]),
                "se_deltaN_total": float(errors["total"][center]),
                "deltaN_reweight_over_V": float(
                    obs["scaled_prediction"][0, center]
                ),
                "se_deltaN_reweight_over_V": float(
                    errors["scaled_prediction"][center]
                ),
            }
        )
    return pd.DataFrame(detailed_rows), pd.DataFrame(equal_weight_rows)


def infinite_prediction(raw: pd.DataFrame, h_values: np.ndarray) -> dict[str, np.ndarray]:
    row = raw[raw["N"] == "infinity"].iloc[0]
    rho_f = 1.0 / float(row["v_fluid"])
    rho_s = 1.0 / float(row["v_solid"])
    delta_rho = rho_s - rho_f
    delta_w = np.log(rho_f / rho_s)
    y = delta_rho * h_values
    q0_s = logistic(y)
    q_c_s = logistic(y - delta_w)
    return {
        "h": h_values,
        "rho_f": np.full_like(h_values, rho_f),
        "rho_s": np.full_like(h_values, rho_s),
        "delta_rho": np.full_like(h_values, delta_rho),
        "delta_w": np.full_like(h_values, delta_w),
        "q0_s": q0_s,
        "qC_s": q_c_s,
        "scaled_prediction": delta_rho * (q_c_s - q0_s),
    }


def fwhm_rows(equal_weight: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in equal_weight.itertuples(index=False):
        amplitude = np.log(row.rho_s / row.rho_f)
        center = -0.5 * amplitude

        def difference(y: float) -> float:
            return float(logistic(np.asarray([y + amplitude]))[0] - logistic(np.asarray([y]))[0])

        half = 0.5 * difference(center)
        left = brentq(lambda y: difference(y) - half, -30.0, center)
        right = brentq(lambda y: difference(y) - half, center, 30.0)
        width_y = right - left
        delta_rho = row.rho_s - row.rho_f
        width_mu = width_y / (delta_rho * row.V)
        rows.append(
            {
                "N_fluid_reference": row.N_fluid_reference,
                "V": row.V,
                "width_y": width_y,
                "fwhm_mu": width_mu,
                "V_times_fwhm_mu": row.V * width_mu,
            }
        )
    return pd.DataFrame(rows)


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


def plot_equal_weight(
    equal_weight: pd.DataFrame, raw: pd.DataFrame, output: Path
) -> None:
    configure_plotting()
    table = equal_weight.sort_values("V")
    volume = table["V"].to_numpy()
    positions = np.arange(len(table), dtype=float)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.1, 3.05), constrained_layout=True
    )
    width = 0.32
    left.bar(
        positions - width / 2,
        table["deltaN_intra"],
        width,
        yerr=table["se_deltaN_intra"],
        color="#56B4E9",
        edgecolor="none",
        capsize=2.0,
        label="intraphase",
    )
    left.bar(
        positions + width / 2,
        table["deltaN_reweight"],
        width,
        yerr=table["se_deltaN_reweight"],
        color="#E69F00",
        edgecolor="none",
        capsize=2.0,
        label="phase reweighting",
    )
    left.errorbar(
        positions,
        table["deltaN_total"],
        yerr=table["se_deltaN_total"],
        marker="D",
        markerfacecolor="#009E73",
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=4.5,
        linewidth=0.0,
        color="#009E73",
        capsize=2.0,
        label="total",
        zorder=4,
    )
    left.axhline(0.0, color="black", linewidth=0.6)
    left.set_xticks(
        positions,
        [rf"{value:.0f}" for value in volume],
    )
    left.set_xlabel(r"$V/\sigma^3$ (published finite-size states)")
    left.set_ylabel(r"$\Delta N_\sigma$ at equal phase weight")
    left.set_title("Local depletion and phase reweighting")
    left.grid(alpha=0.16, linewidth=0.5)
    left.legend(frameon=False, fontsize=7.1)

    infinite = raw[raw["N"] == "infinity"].iloc[0]
    rho_f_infinite = 1.0 / float(infinite["v_fluid"])
    rho_s_infinite = 1.0 / float(infinite["v_solid"])
    coefficient_infinite = (rho_s_infinite - rho_f_infinite) ** 2 / (
        2.0 * (rho_s_infinite + rho_f_infinite)
    )
    relative = 100.0 * (
        table["deltaN_reweight_over_V"] / coefficient_infinite - 1.0
    )
    relative_error = (
        100.0 * table["se_deltaN_reweight_over_V"] / coefficient_infinite
    )
    source_positions = np.arange(len(table), dtype=float)
    right.errorbar(
        source_positions,
        relative,
        yerr=relative_error,
        color="#D55E00",
        marker="s",
        markerfacecolor="white",
        markersize=4.5,
        linewidth=0.0,
        capsize=2.0,
    )
    right.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    right.set_xticks(
        source_positions,
        [str(int(value)) for value in table["N_fluid_reference"]],
        rotation=30,
    )
    right.set_xlabel(r"fluid reference size $N_f$")
    right.set_ylabel("deviation from bulk coefficient (%)")
    right.set_title("Sensitivity to finite-size density inputs")
    right.grid(alpha=0.16, linewidth=0.5)
    for label, axis in zip(("a", "b"), (left, right)):
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


def plot_collapse(
    detailed: pd.DataFrame,
    raw: pd.DataFrame,
    widths: pd.DataFrame,
    output: Path,
) -> None:
    configure_plotting()
    mixture = detailed[detailed["phase"] == "fluid"].copy()
    prediction = infinite_prediction(raw, H_GRID)
    volumes = np.asarray(sorted(mixture["V"].unique()), dtype=float)
    normalizer = mpl.colors.LogNorm(vmin=float(volumes.min()), vmax=float(volumes.max()))
    color_map = mpl.colormaps["viridis"]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.1, 3.05), constrained_layout=True
    )
    for volume, group in mixture.groupby("V", sort=True):
        color = color_map(normalizer(float(volume)))
        group = group.sort_values("h")
        left.plot(
            group["h"],
            group["deltaN_reweight"] / volume,
            color=color,
            linewidth=0.95,
        )
    left.plot(
        prediction["h"],
        prediction["scaled_prediction"],
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=r"$V\to\infty$ prediction",
    )
    left.set_xlabel(r"$h=V(\mu-\mu_V^{\rm eq})$")
    left.set_ylabel(r"$\Delta N_{\rm reweight}/V$")
    left.set_title("Scaled phase-reweighting pulse")
    left.grid(alpha=0.16, linewidth=0.5)
    left.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        loc="upper right",
    )

    mu_grid = np.linspace(-0.36, 0.36, 721)
    finite_raw = raw[raw["N"] != "infinity"].copy()
    for source in finite_raw.itertuples(index=False):
        volume = int(source.N) * float(source.v_fluid)
        color = color_map(normalizer(float(volume)))
        observables = finite_size_observables(
            np.asarray([1.0 / float(source.v_fluid)]),
            np.asarray([1.0 / float(source.v_solid)]),
            np.asarray([float(source.p_equal_weight)]),
            volume,
            volume * mu_grid,
        )
        right.plot(
            mu_grid,
            observables["total"][0],
            color=color,
            linewidth=0.95,
        )
    right.axhline(0.0, color="black", linewidth=0.55)
    right.set_xlabel(r"$\mu-\mu_V^{\rm eq}$")
    right.set_ylabel(r"full predicted $\Delta N_\sigma$")
    right.set_title("Unscaled growth and narrowing")
    right.grid(alpha=0.16, linewidth=0.5)

    scalar_map = mpl.cm.ScalarMappable(norm=normalizer, cmap=color_map)
    colorbar = fig.colorbar(scalar_map, ax=[left, right], pad=0.02, shrink=0.92)
    colorbar.set_label(r"$V/\sigma^3$")
    for label, axis in zip(("a", "b"), (left, right)):
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


def plot_source_uncertainty(
    equal_weight: pd.DataFrame, raw: pd.DataFrame, output: Path
) -> None:
    configure_plotting()
    finite = equal_weight.sort_values("V")
    infinite = raw[raw["N"] == "infinity"].iloc[0]
    rho_f = 1.0 / float(infinite["v_fluid"])
    rho_s = 1.0 / float(infinite["v_solid"])
    coefficient = (rho_s - rho_f) * (
        rho_s / (rho_s + rho_f) - 0.5
    )
    fig, axis = plt.subplots(figsize=(4.1, 2.8), constrained_layout=True)
    axis.errorbar(
        finite["N_fluid_reference"],
        finite["deltaN_reweight_over_V"],
        yerr=finite["se_deltaN_reweight_over_V"],
        marker="o",
        color="#D55E00",
        linewidth=0.9,
        capsize=2.0,
        label="propagated published errors",
    )
    axis.axhline(
        coefficient,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label=rf"infinite-volume densities ({coefficient:.5f})",
    )
    axis.set_xscale("log")
    axis.set_xlabel("fluid reference particle number")
    axis.set_ylabel(r"$\Delta N_{\rm reweight}/V$")
    axis.set_title("Sensitivity to coexistence-density uncertainty")
    axis.grid(alpha=0.16, which="both", linewidth=0.5)
    axis.legend(frameon=False)
    fig.savefig(output)
    plt.close(fig)


def write_latex_table(equal_weight: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$N_f$ & $V/\sigma^3$ & $\widetilde q_s$ & $\Delta N_{\rm intra}$ & $\Delta N_{\rm rew}$ \\",
        r"\midrule",
    ]
    for row in equal_weight.itertuples(index=False):
        lines.append(
            f"{int(row.N_fluid_reference)} & {row.V:.2f} & {row.qC_s:.5f} & "
            f"{row.deltaN_intra:.4f} & {row.deltaN_reweight:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = pd.read_csv(
        root / "data_raw" / "fernandez2012_coexistence.csv", dtype={"N": str}
    )
    detailed, equal_weight = make_phase_data(raw)
    widths = fwhm_rows(equal_weight)

    detailed.to_csv(root / "data_processed" / "phase_conditioned_cavity.csv", index=False)
    equal_weight.to_csv(root / "generated" / "phase_equal_weight_summary.csv", index=False)
    widths.to_csv(root / "generated" / "coexistence_widths.csv", index=False)
    plot_equal_weight(
        equal_weight, raw, root / "figures" / "phase_reweighting.pdf"
    )
    plot_collapse(
        detailed, raw, widths, root / "figures" / "coexistence_collapse.pdf"
    )
    plot_source_uncertainty(
        equal_weight, raw, root / "figures" / "phase_source_uncertainty.pdf"
    )
    write_latex_table(equal_weight, root / "generated" / "phase_table.tex")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/phase_conditioned.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "random_seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "source": "Fernandez et al., Phys. Rev. Lett. 108, 165701 (2012)",
        "doi": "10.1103/PhysRevLett.108.165701",
        "source_data": "published Table I summary statistics",
        "method": (
            "thermodynamic pure-phase R=sigma insertion relation plus "
            "two-phase asymptotics evaluated with external coexistence densities"
        ),
        "limitations": [
            "not a reanalysis of particle trajectories",
            "not a new tethered or phase-switch simulation",
            "finite-N curves are a data-anchored proxy, not an exact finite-basin insertion identity",
            "the pure-phase insertion relation assumes insertion/removal-stable basins in the thermodynamic limit",
            "reported source covariances were unavailable; tabulated errors were sampled independently",
            "solid intraphase background uses an occupancy-constrained, pressure-anchored free-volume proxy rather than a defect-equilibrated grand-canonical susceptibility",
            "the leading phase-reweighting term does not depend on the intraphase compressibility estimate",
        ],
    }
    (root / "generated" / "phase_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(detailed)} phase-conditioned rows and "
        f"{len(equal_weight)} equal-weight summaries"
    )


if __name__ == "__main__":
    main()
