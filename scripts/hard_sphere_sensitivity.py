#!/usr/bin/env python3
"""Occupancy-constrained solid-compressibility proxy sensitivity.

The phase-reweighting term for a particle-sized cavity is fixed by the two
coexistence densities.  Only the nonextensive intraphase term requires a
solid compressibility proxy in the present data-anchored calculation.  This
script therefore evaluates that term with three explicitly named
descriptions:

* the pressure-anchored local free-volume derivative used in the main
  calculation;
* Speedy's empirical 1998 fcc equation of state; and
* the same Speedy functional form with the 2010 MD-refit coefficients.

The resulting spread is a model-sensitivity diagnostic, not a statistical
confidence interval and not a substitute for a direct grand-canonical
fluid--crystal calculation.
"""

from __future__ import annotations

import argparse
import hashlib
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

from hard_sphere_models import (
    BANNERMAN_2010_COEFFICIENTS,
    PI_OVER_SIX,
    SPEEDY_1998_COEFFICIENTS,
    cs_s0,
    speedy_solid_s0,
)
from phase_conditioned import logistic, solid_s0_cell


SEED = 6082026
BOOTSTRAP_DRAWS = 20_000
H_GRID = np.linspace(-40.0, 40.0, 161)


@dataclass(frozen=True)
class SolidCompressibilityModel:
    key: str
    label: str
    functional_family: str
    color: str


SOLID_MODELS = (
    SolidCompressibilityModel(
        "pressure_anchored_free_volume",
        "pressure-anchored free volume",
        "local_free_volume_derivative",
        "#0072B2",
    ),
    SolidCompressibilityModel(
        "speedy_1998",
        "Speedy (1998)",
        "speedy_fcc_eos_original_coefficients",
        "#D55E00",
    ),
    SolidCompressibilityModel(
        "bannerman_2010_refit",
        "Speedy EOS, 2010 refit",
        "speedy_fcc_eos_refitted_coefficients",
        "#009E73",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solid_s0_for_model(
    model_key: str,
    eta_s: np.ndarray,
    pressure: np.ndarray,
    rho_s: np.ndarray,
) -> np.ndarray:
    """Return one occupancy-constrained solid EOS proxy."""

    if model_key == "pressure_anchored_free_volume":
        return solid_s0_cell(eta_s, pressure, rho_s)
    if model_key == "speedy_1998":
        return speedy_solid_s0(eta_s, SPEEDY_1998_COEFFICIENTS)
    if model_key == "bannerman_2010_refit":
        return speedy_solid_s0(eta_s, BANNERMAN_2010_COEFFICIENTS)
    raise KeyError(f"Unknown solid compressibility model: {model_key}")


def model_observables(
    rho_f: np.ndarray,
    rho_s: np.ndarray,
    pressure: np.ndarray,
    volume: float,
    h_values: np.ndarray,
    model_key: str,
) -> dict[str, np.ndarray]:
    """Evaluate the two-phase response for one solid compressibility model."""

    rho_f = np.asarray(rho_f, dtype=float).reshape(-1, 1)
    rho_s = np.asarray(rho_s, dtype=float).reshape(-1, 1)
    pressure = np.asarray(pressure, dtype=float).reshape(-1, 1)
    h_values = np.asarray(h_values, dtype=float).reshape(1, -1)
    delta_rho = rho_s - rho_f
    delta_w = np.log(rho_f / rho_s)
    y = delta_rho * h_values
    q0_s = logistic(y)
    qc_s = logistic(y - delta_w)
    qc_f = 1.0 - qc_s

    fluid_s0 = cs_s0(PI_OVER_SIX * rho_f)
    solid_s0 = solid_s0_for_model(
        model_key, PI_OVER_SIX * rho_s, pressure, rho_s
    )
    intraphase = qc_f * (fluid_s0 - 1.0) + qc_s * (solid_s0 - 1.0)
    scaled_reweight = delta_rho * (qc_s - q0_s)
    reweight = volume * scaled_reweight
    total = intraphase + reweight
    return {
        "q0_s": q0_s,
        "qC_s": qc_s,
        "fluid_s0": np.broadcast_to(fluid_s0, q0_s.shape),
        "solid_s0": np.broadcast_to(solid_s0, q0_s.shape),
        "intraphase": intraphase,
        "reweight": reweight,
        "scaled_reweight": scaled_reweight,
        "total": total,
    }


def bootstrap_standard_errors(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.std(np.asarray(value, dtype=float), axis=0, ddof=1)
        for key, value in samples.items()
    }


def build_sensitivity_data(
    raw: pd.DataFrame,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return finite-source sensitivity curves and propagated input errors."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    finite = raw[raw["N"] != "infinity"].copy()
    for source in finite.itertuples(index=False):
        n_fluid = int(source.N)
        v_f = float(source.v_fluid)
        v_s = float(source.v_solid)
        pressure = float(source.p_equal_weight)
        rho_f = 1.0 / v_f
        rho_s = 1.0 / v_s
        volume = n_fluid * v_f

        v_f_draw = rng.normal(v_f, float(source.se_v_fluid), draws)
        v_s_draw = rng.normal(v_s, float(source.se_v_solid), draws)
        pressure_draw = rng.normal(
            pressure, float(source.se_p_equal_weight), draws
        )
        rho_f_draw = 1.0 / v_f_draw
        rho_s_draw = 1.0 / v_s_draw

        for model in SOLID_MODELS:
            nominal = model_observables(
                np.asarray([rho_f]),
                np.asarray([rho_s]),
                np.asarray([pressure]),
                volume,
                H_GRID,
                model.key,
            )
            bootstrap = model_observables(
                rho_f_draw,
                rho_s_draw,
                pressure_draw,
                volume,
                H_GRID,
                model.key,
            )
            errors = bootstrap_standard_errors(bootstrap)
            for index, h_value in enumerate(H_GRID):
                rows.append(
                    {
                        "N_fluid_reference": n_fluid,
                        "V": volume,
                        "h": h_value,
                        "mu_minus_equal_weight": h_value / volume,
                        "model": model.key,
                        "model_label": model.label,
                        "functional_family": model.functional_family,
                        "q0_s": float(nominal["q0_s"][0, index]),
                        "se_q0_s": float(errors["q0_s"][index]),
                        "qC_s": float(nominal["qC_s"][0, index]),
                        "se_qC_s": float(errors["qC_s"][index]),
                        "fluid_s0": float(nominal["fluid_s0"][0, index]),
                        "se_fluid_s0": float(errors["fluid_s0"][index]),
                        "solid_s0": float(nominal["solid_s0"][0, index]),
                        "se_solid_s0": float(errors["solid_s0"][index]),
                        "deltaN_intraphase": float(
                            nominal["intraphase"][0, index]
                        ),
                        "se_deltaN_intraphase": float(
                            errors["intraphase"][index]
                        ),
                        "deltaN_reweight": float(nominal["reweight"][0, index]),
                        "se_deltaN_reweight": float(errors["reweight"][index]),
                        "deltaN_reweight_over_V": float(
                            nominal["scaled_reweight"][0, index]
                        ),
                        "se_deltaN_reweight_over_V": float(
                            errors["scaled_reweight"][index]
                        ),
                        "deltaN_total": float(nominal["total"][0, index]),
                        "se_deltaN_total": float(errors["total"][index]),
                        "source_type": "published_summary_plus_model_sensitivity",
                    }
                )
    return pd.DataFrame(rows)


def build_model_envelope(curves: pd.DataFrame) -> pd.DataFrame:
    """Construct the outer model-plus-input-error envelope at each state."""

    rows = []
    keys = ["N_fluid_reference", "V", "h", "mu_minus_equal_weight"]
    for group_key, group in curves.groupby(keys, sort=True):
        row = dict(zip(keys, group_key))
        for value, error in (
            ("solid_s0", "se_solid_s0"),
            ("deltaN_intraphase", "se_deltaN_intraphase"),
            ("deltaN_total", "se_deltaN_total"),
        ):
            row[f"{value}_model_min"] = float(group[value].min())
            row[f"{value}_model_max"] = float(group[value].max())
            row[f"{value}_outer_lower"] = float(
                (group[value] - group[error]).min()
            )
            row[f"{value}_outer_upper"] = float(
                (group[value] + group[error]).max()
            )
        coefficient_range = float(group["deltaN_reweight_over_V"].max()) - float(
            group["deltaN_reweight_over_V"].min()
        )
        row["density_coefficient"] = float(
            group["deltaN_reweight_over_V"].iloc[0]
        )
        row["se_density_coefficient"] = float(
            group["se_deltaN_reweight_over_V"].iloc[0]
        )
        row["density_coefficient_model_range"] = coefficient_range
        rows.append(row)
    return pd.DataFrame(rows)


def build_crossing_data(
    raw: pd.DataFrame,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED + 1,
) -> pd.DataFrame:
    """Return the equal-weight sign-crossing volume under each model."""

    source = raw[raw["N"] == "infinity"].iloc[0]
    rng = np.random.default_rng(seed)
    v_f = float(source["v_fluid"])
    v_s = float(source["v_solid"])
    pressure = float(source["p_equal_weight"])
    v_f_draw = rng.normal(v_f, float(source["se_v_fluid"]), draws)
    v_s_draw = rng.normal(v_s, float(source["se_v_solid"]), draws)
    pressure_draw = rng.normal(
        pressure, float(source["se_p_equal_weight"]), draws
    )
    rows = []
    for model in SOLID_MODELS:
        nominal = model_observables(
            np.asarray([1.0 / v_f]),
            np.asarray([1.0 / v_s]),
            np.asarray([pressure]),
            1.0,
            np.asarray([0.0]),
            model.key,
        )
        bootstrap = model_observables(
            1.0 / v_f_draw,
            1.0 / v_s_draw,
            pressure_draw,
            1.0,
            np.asarray([0.0]),
            model.key,
        )
        intra = float(nominal["intraphase"][0, 0])
        coefficient = float(nominal["scaled_reweight"][0, 0])
        crossing = -intra / coefficient
        crossing_draw = -bootstrap["intraphase"][:, 0] / bootstrap[
            "scaled_reweight"
        ][:, 0]
        rows.append(
            {
                "model": model.key,
                "model_label": model.label,
                "functional_family": model.functional_family,
                "bulk_solid_s0": float(nominal["solid_s0"][0, 0]),
                "se_bulk_solid_s0": float(
                    np.std(bootstrap["solid_s0"][:, 0], ddof=1)
                ),
                "equal_weight_intraphase": intra,
                "se_equal_weight_intraphase": float(
                    np.std(bootstrap["intraphase"][:, 0], ddof=1)
                ),
                "density_coefficient": coefficient,
                "se_density_coefficient": float(
                    np.std(bootstrap["scaled_reweight"][:, 0], ddof=1)
                ),
                "zero_crossing_V": crossing,
                "se_zero_crossing_V": float(np.std(crossing_draw, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def plot_sensitivity(
    curves: pd.DataFrame,
    envelope: pd.DataFrame,
    crossings: pd.DataFrame,
    output: Path,
) -> None:
    """Write the vector supplemental sensitivity figure."""

    configure_plotting()
    center = curves[np.isclose(curves["h"], 0.0)].copy()
    fig, axes = plt.subplots(
        2, 2, figsize=(7.15, 5.45), constrained_layout=True
    )
    ax_s0, ax_intra, ax_volume, ax_pulse = axes.ravel()
    marker_cycle = ("o", "s", "^")
    color_by_model = {model.key: model.color for model in SOLID_MODELS}

    for marker, model in zip(marker_cycle, SOLID_MODELS):
        group = center[center["model"] == model.key].sort_values(
            "N_fluid_reference"
        )
        ax_s0.errorbar(
            group["N_fluid_reference"],
            group["solid_s0"],
            yerr=group["se_solid_s0"],
            marker=marker,
            markersize=3.8,
            linewidth=0.85,
            capsize=1.6,
            color=model.color,
            label=model.label,
        )
        ax_intra.errorbar(
            group["V"],
            group["deltaN_intraphase"],
            yerr=group["se_deltaN_intraphase"],
            marker=marker,
            markersize=3.8,
            linewidth=0.85,
            capsize=1.6,
            color=model.color,
            label=model.label,
        )

    ax_s0.set_xscale("log")
    ax_s0.set_xlabel(r"published fluid reference size $N_f$")
    ax_s0.set_ylabel(r"solid proxy $S_{s,\mathrm{occ}}^{\rm EOS}$")
    ax_s0.set_title("Occupancy-constrained EOS proxies")
    ax_s0.grid(alpha=0.16, linewidth=0.5)
    ax_s0.legend(frameon=False, loc="best")

    ax_intra.set_xscale("log")
    ax_intra.set_xlabel(r"$V/\sigma^3$")
    ax_intra.set_ylabel(r"$\Delta N_{\rm intra}$ at $h=0$")
    ax_intra.set_title("Nonextensive term")
    ax_intra.grid(alpha=0.16, linewidth=0.5)

    coefficient = float(crossings["density_coefficient"].iloc[0])
    coefficient_se = float(crossings["se_density_coefficient"].iloc[0])
    volume_grid = np.linspace(0.0, 700.0, 351)
    intra_lower = float(
        (
            crossings["equal_weight_intraphase"]
            - crossings["se_equal_weight_intraphase"]
        ).min()
    )
    intra_upper = float(
        (
            crossings["equal_weight_intraphase"]
            + crossings["se_equal_weight_intraphase"]
        ).max()
    )
    outer_lower = intra_lower + volume_grid * (coefficient - coefficient_se)
    outer_upper = intra_upper + volume_grid * (coefficient + coefficient_se)
    ax_volume.fill_between(
        volume_grid,
        outer_lower,
        outer_upper,
        color="#999999",
        alpha=0.32,
        linewidth=0.0,
        label="model + input-error envelope",
    )
    for model in SOLID_MODELS:
        row = crossings[crossings["model"] == model.key].iloc[0]
        ax_volume.plot(
            volume_grid,
            float(row["equal_weight_intraphase"]) + coefficient * volume_grid,
            color=model.color,
            linewidth=0.85,
        )
    ax_volume.plot(
        volume_grid,
        coefficient * volume_grid,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="density-only reweighting",
    )
    ax_volume.axhline(0.0, color="black", linewidth=0.55)
    ax_volume.set_xlabel(r"$V/\sigma^3$")
    ax_volume.set_ylabel(r"$\Delta N_\sigma$ at $h=0$")
    ax_volume.set_title("Full response and sign crossing")
    ax_volume.grid(alpha=0.16, linewidth=0.5)
    ax_volume.legend(frameon=False, loc="upper left")
    crossing_inset = ax_volume.inset_axes([0.54, 0.08, 0.42, 0.34])
    crossing_inset.fill_between(
        volume_grid,
        outer_lower,
        outer_upper,
        color="#999999",
        alpha=0.32,
        linewidth=0.0,
    )
    for model in SOLID_MODELS:
        row = crossings[crossings["model"] == model.key].iloc[0]
        crossing_inset.plot(
            volume_grid,
            float(row["equal_weight_intraphase"]) + coefficient * volume_grid,
            color=model.color,
            linewidth=1.0,
        )
        crossing_inset.errorbar(
            float(row["zero_crossing_V"]),
            0.0,
            xerr=float(row["se_zero_crossing_V"]),
            marker="o",
            markersize=2.5,
            color=model.color,
            capsize=1.3,
            linewidth=0.65,
        )
    crossing_inset.axhline(0.0, color="black", linewidth=0.45)
    crossing_inset.set_xlim(399.0, 408.0)
    crossing_inset.set_ylim(-0.012, 0.012)
    crossing_inset.set_xlabel(r"$V/\sigma^3$", fontsize=6.4, labelpad=1.0)
    crossing_inset.set_ylabel(r"$\Delta N_\sigma$", fontsize=6.4, labelpad=1.0)
    crossing_inset.tick_params(labelsize=5.8, pad=1.0)
    crossing_inset.grid(alpha=0.12, linewidth=0.4)

    representative_n = 500
    pulse = curves[curves["N_fluid_reference"] == representative_n].copy()
    pulse_envelope = envelope[
        envelope["N_fluid_reference"] == representative_n
    ].sort_values("h")
    ax_pulse.fill_between(
        pulse_envelope["h"].to_numpy(),
        pulse_envelope["deltaN_total_outer_lower"].to_numpy(),
        pulse_envelope["deltaN_total_outer_upper"].to_numpy(),
        color="#999999",
        alpha=0.32,
        linewidth=0.0,
        label="outer envelope",
    )
    for model in SOLID_MODELS:
        group = pulse[pulse["model"] == model.key].sort_values("h")
        ax_pulse.plot(
            group["h"],
            group["deltaN_total"],
            color=color_by_model[model.key],
            linewidth=0.9,
            label=model.label,
        )
    ax_pulse.axhline(0.0, color="black", linewidth=0.55)
    ax_pulse.set_xlabel(r"$h=V(\mu-\mu_V^{\rm eq})$")
    ax_pulse.set_ylabel(r"proxy total $\Delta N_\sigma$")
    ax_pulse.set_title(r"Representative proxy pulse ($N_f=500$ source)")
    ax_pulse.grid(alpha=0.16, linewidth=0.5)
    residual_inset = ax_pulse.inset_axes([0.55, 0.57, 0.41, 0.36])
    baseline = pulse[
        pulse["model"] == "pressure_anchored_free_volume"
    ].sort_values("h")
    for model in SOLID_MODELS[1:]:
        group = pulse[pulse["model"] == model.key].sort_values("h")
        residual_inset.plot(
            group["h"],
            1.0e3
            * (
                group["deltaN_total"].to_numpy()
                - baseline["deltaN_total"].to_numpy()
            ),
            color=model.color,
            linewidth=0.85,
        )
    residual_inset.axhline(0.0, color="black", linewidth=0.4)
    residual_inset.set_xlabel(r"$h$", fontsize=6.4, labelpad=1.0)
    residual_inset.set_ylabel(r"$10^3\,\delta\Delta N_\sigma$", fontsize=6.4, labelpad=1.0)
    residual_inset.tick_params(labelsize=5.8, pad=1.0)
    residual_inset.grid(alpha=0.12, linewidth=0.4)

    for label, axis in zip(("a", "b", "c", "d"), axes.ravel()):
        axis.text(
            -0.12,
            1.035,
            f"({label})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.4,
            fontweight="bold",
            clip_on=False,
        )
    fig.savefig(
        output,
        metadata={
            "Title": "Hard-sphere occupancy-constrained EOS-proxy sensitivity",
            "Subject": "Model dependence of the nonextensive coexistence response",
            "Creator": "scripts/hard_sphere_sensitivity.py",
        },
    )
    plt.close(fig)


def validate_outputs(curves: pd.DataFrame, envelope: pd.DataFrame) -> None:
    """Fail loudly if model dependence leaks into the density-only term."""

    maximum_range = float(envelope["density_coefficient_model_range"].max())
    if maximum_range > 5.0e-15:
        raise RuntimeError(
            "The nominal density-only coefficient changed across solid models: "
            f"range={maximum_range:.3e}"
        )
    residual = curves["deltaN_total"] - (
        curves["deltaN_intraphase"] + curves["deltaN_reweight"]
    )
    if float(np.max(np.abs(residual))) > 2.0e-13:
        raise RuntimeError("Response decomposition failed its numerical identity")
    if not np.all(curves["solid_s0"] > 0.0):
        raise RuntimeError("A solid occupancy-constrained EOS proxy became nonpositive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    root = args.project_root.resolve()
    input_path = root / "data_raw" / "fernandez2012_coexistence.csv"
    raw = pd.read_csv(input_path, dtype={"N": str})
    curves = build_sensitivity_data(raw, draws=args.bootstrap_draws)
    envelope = build_model_envelope(curves)
    crossings = build_crossing_data(raw, draws=args.bootstrap_draws)
    validate_outputs(curves, envelope)

    curve_path = root / "data_processed" / "hard_sphere_model_sensitivity.csv"
    envelope_path = root / "data_processed" / "hard_sphere_model_envelope.csv"
    crossing_path = root / "data_processed" / "hard_sphere_model_crossings.csv"
    figure_path = root / "figures" / "hard_sphere_model_sensitivity.pdf"
    curves.to_csv(curve_path, index=False)
    envelope.to_csv(envelope_path, index=False)
    crossings.to_csv(crossing_path, index=False)
    plot_sensitivity(curves, envelope, crossings, figure_path)

    code_path = Path(__file__).resolve()
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/hard_sphere_sensitivity.py",
        "reproduction_command": (
            "python scripts/hard_sphere_sensitivity.py "
            f"--bootstrap-draws {args.bootstrap_draws}"
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "random_seed": SEED,
        "crossing_random_seed": SEED + 1,
        "bootstrap_draws": args.bootstrap_draws,
        "input": str(input_path.relative_to(root)),
        "input_sha256": file_sha256(input_path),
        "script_sha256": file_sha256(code_path),
        "models": {
            "pressure_anchored_free_volume": {
                "description": (
                    "local Z=A/(1-eta/eta_cp), with A chosen separately at "
                    "each source state to reproduce its tabulated pressure"
                ),
                "status": "baseline modelling assumption, not a fitted global EOS",
            },
            "speedy_1998": {
                "description": (
                    "Z=3/(1-x)-A(x-B)/(x-C), x=eta/eta_cp, "
                    "A=0.5921, B=0.7072, C=0.601"
                ),
                "citation": (
                    "R. J. Speedy, J. Phys.: Condens. Matter 10, 4387 (1998)"
                ),
                "doi": "10.1088/0953-8984/10/20/006",
            },
            "bannerman_2010_refit": {
                "description": (
                    "Speedy functional form with A=0.620735, B=0.708194, "
                    "C=0.591663"
                ),
                "citation": (
                    "M. N. Bannerman et al., J. Chem. Phys. 132, 084507 (2010)"
                ),
                "doi": "10.1063/1.3328823",
                "status": "coefficient refit, not an independent functional family",
            },
        },
        "source": {
            "citation": (
                "L. A. Fernandez et al., Phys. Rev. Lett. 108, 165701 (2012)"
            ),
            "doi": "10.1103/PhysRevLett.108.165701",
            "data_used": "published Table I summary statistics",
        },
        "invariant": (
            "deltaN_reweight/V is computed only from the two densities and "
            "phase weights; its model range is numerically zero"
        ),
        "maximum_density_coefficient_model_range": float(
            envelope["density_coefficient_model_range"].max()
        ),
        "limitations": [
            "the model envelope is a sensitivity range, not a confidence interval",
            "the two Speedy curves share one functional family",
            "published input covariances were unavailable and errors were sampled independently",
            "finite-size volumes are held at their nominal tabulated fluid values during error propagation",
            "this is not a direct common-normalization fluid-crystal simulation",
            "the sign crossing combines an exact density-only leading term with a model-dependent O(1) term",
        ],
        "outputs": [
            str(curve_path.relative_to(root)),
            str(envelope_path.relative_to(root)),
            str(crossing_path.relative_to(root)),
            str(figure_path.relative_to(root)),
        ],
    }
    metadata_path = root / "generated" / "hard_sphere_model_sensitivity_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(curves)} curve rows, {len(envelope)} envelope rows, "
        f"and {len(crossings)} model summaries"
    )


if __name__ == "__main__":
    main()
