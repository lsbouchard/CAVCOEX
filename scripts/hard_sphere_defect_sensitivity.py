#!/usr/bin/env python3
"""Vacancy-scale sensitivity of the density-only hard-sphere result.

The high-precision coexistence densities used by the manuscript come from a
commensurate FCC branch.  A grand-canonical equilibrium crystal can also vary
its number of lattice sites and hence its vacancy population.  This script
quantifies one deliberately limited perturbation: at fixed lattice-site
density and fixed fluid density, a representative vacancy fraction

    x_v = (M - N) / M

changes the solid *particle* density according to

    rho_s -> (1 - x_v) rho_s.

The calculation is a scale diagnostic, not a defect-corrected coexistence
calculation and not a total uncertainty.  It omits relaxation of the lattice-
site density, the coexistence pressure and chemical potential, and the fluid
density.  Source-density standard errors and the vacancy-scale perturbation
are reported separately; no probability distribution is assigned to x_v.
The sensitivity of the baseline coefficient's propagated source error to
the unreported fluid--solid specific-volume correlation is also tabulated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_ORDER = (
    "rho_fluid",
    "rho_solid",
    "beta_delta_w",
    "constrained_solid_weight",
    "reweight_coefficient",
    "relative_mean_number_fraction",
    "particles_per_1000",
    "one_particle_mean_number",
)

COMPARISON_METRICS = (
    "beta_delta_w",
    "constrained_solid_weight",
    "reweight_coefficient",
    "relative_mean_number_fraction",
    "particles_per_1000",
    "one_particle_mean_number",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observables(
    v_fluid: float | complex,
    v_solid: float | complex,
    vacancy_fraction: float,
) -> dict[str, float | complex]:
    """Return equal-weight density-only observables for one perturbation."""

    rho_fluid = 1.0 / v_fluid
    rho_solid = (1.0 - vacancy_fraction) / v_solid
    density_gap = rho_solid - rho_fluid
    density_sum = rho_solid + rho_fluid
    relative_fraction = density_gap**2 / density_sum**2
    return {
        "rho_fluid": rho_fluid,
        "rho_solid": rho_solid,
        "beta_delta_w": np.log(rho_fluid / rho_solid),
        "constrained_solid_weight": rho_solid / density_sum,
        "reweight_coefficient": density_gap**2 / (2.0 * density_sum),
        "relative_mean_number_fraction": relative_fraction,
        "particles_per_1000": 1000.0 * relative_fraction,
        "one_particle_mean_number": 1.0 / relative_fraction,
    }


def metric_vector(
    v_fluid: float | complex,
    v_solid: float | complex,
    vacancy_fraction: float,
    metrics: tuple[str, ...] = METRIC_ORDER,
) -> np.ndarray:
    values = observables(v_fluid, v_solid, vacancy_fraction)
    return np.asarray([values[name] for name in metrics])


def complex_step_jacobian(
    function,
    point: np.ndarray,
    step: float = 1.0e-30,
) -> np.ndarray:
    """Return a deterministic first-derivative Jacobian."""

    baseline = np.asarray(function(*point), dtype=float)
    jacobian = np.empty((baseline.size, point.size), dtype=float)
    for column in range(point.size):
        perturbed = point.astype(complex)
        perturbed[column] += 1j * step
        jacobian[:, column] = np.imag(function(*perturbed)) / step
    return jacobian


def source_standard_errors(
    v_fluid: float,
    v_solid: float,
    se_v_fluid: float,
    se_v_solid: float,
    vacancy_fraction: float,
    metrics: tuple[str, ...] = METRIC_ORDER,
) -> np.ndarray:
    """Propagate independent published specific-volume standard errors."""

    point = np.asarray([v_fluid, v_solid], dtype=float)
    jacobian = complex_step_jacobian(
        lambda vf, vs: metric_vector(vf, vs, vacancy_fraction, metrics),
        point,
    )
    variances = np.sum(
        (jacobian * np.asarray([se_v_fluid, se_v_solid])) ** 2,
        axis=1,
    )
    return np.sqrt(variances)


def density_coefficient_covariance_sensitivity(
    v_fluid: float,
    v_solid: float,
    se_v_fluid: float,
    se_v_solid: float,
    correlations: np.ndarray,
) -> pd.DataFrame:
    """Propagate source errors for every supplied correlation in [-1, 1].

    This first-order delta-method calculation holds the published marginal
    errors fixed; it does not estimate their unknown correlation.  If a and
    b are the signed derivative-times-error contributions, the sharp range
    over all admissible correlations is [||a|-|b||, |a|+|b|].
    Specific volumes and their errors are in units of sigma**3, so the
    returned coefficient and standard errors are in units of sigma**-3.
    """

    inputs = np.asarray([v_fluid, v_solid, se_v_fluid, se_v_solid])
    correlations = np.atleast_1d(np.asarray(correlations, dtype=float))
    if (
        not np.all(np.isfinite(inputs))
        or np.any(inputs[:2] <= 0.0)
        or np.any(inputs[2:] < 0.0)
        or correlations.ndim != 1
        or not np.all(np.isfinite(correlations))
        or np.any(np.abs(correlations) > 1.0)
    ):
        raise ValueError("require positive volumes, nonnegative errors, and correlations in [-1, 1]")
    rho_fluid, rho_solid = 1.0 / v_fluid, 1.0 / v_solid
    gap, total = rho_solid - rho_fluid, rho_solid + rho_fluid
    coefficient = gap**2 / (2.0 * total)
    derivative_fluid = (gap / total + gap**2 / (2.0 * total**2)) / v_fluid**2
    derivative_solid = (-gap / total + gap**2 / (2.0 * total**2)) / v_solid**2
    a, b = derivative_fluid * se_v_fluid, derivative_solid * se_v_solid
    # This convex combination of endpoint variances avoids cancellation
    # when a = +/- b and the limiting propagated variance is exactly zero.
    variance = (
        (1.0 + correlations) * (a + b)**2
        + (1.0 - correlations) * (a - b)**2
    ) / 2.0
    return pd.DataFrame({
        "specific_volume_correlation": correlations,
        "reweight_coefficient": coefficient,
        "se_reweight_coefficient_source": np.sqrt(variance),
        "d_coefficient_d_v_fluid": derivative_fluid,
        "d_coefficient_d_v_solid": derivative_solid,
        "fluid_signed_se_contribution": a,
        "solid_signed_se_contribution": b,
    })


def comparison_vector(
    v_fluid: float | complex,
    v_solid: float | complex,
    vacancy_fraction: float,
) -> np.ndarray:
    baseline = metric_vector(v_fluid, v_solid, 0.0, COMPARISON_METRICS)
    perturbed = metric_vector(
        v_fluid, v_solid, vacancy_fraction, COMPARISON_METRICS
    )
    shift = perturbed - baseline
    relative_percent = 100.0 * shift / baseline
    return np.concatenate((shift, relative_percent))


def build_outputs(
    coexistence: pd.Series,
    vacancy_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v_fluid = float(coexistence["v_fluid"])
    v_solid = float(coexistence["v_solid"])
    se_v_fluid = float(coexistence["se_v_fluid"])
    se_v_solid = float(coexistence["se_v_solid"])

    scenario_rows: list[dict[str, float | str]] = []
    scenario_values: dict[str, np.ndarray] = {}
    scenario_errors: dict[str, np.ndarray] = {}
    for scenario, fraction in (
        ("commensurate_fcc_baseline", 0.0),
        ("fixed_site_density_vacancy_scale", vacancy_fraction),
    ):
        values = np.real(
            metric_vector(v_fluid, v_solid, fraction, METRIC_ORDER)
        ).astype(float)
        errors = source_standard_errors(
            v_fluid,
            v_solid,
            se_v_fluid,
            se_v_solid,
            fraction,
            METRIC_ORDER,
        )
        scenario_values[scenario] = values
        scenario_errors[scenario] = errors
        row: dict[str, float | str] = {
            "scenario": scenario,
            "vacancy_fraction": fraction,
            "source_uncertainty_scope": (
                "published specific-volume standard errors only; "
                "zero covariance assumed because none was reported"
            ),
            "defect_scope": (
                "fixed lattice-site density and fixed fluid density; "
                "not a re-equilibrated coexistence calculation or total uncertainty"
            ),
        }
        for name, value, error in zip(METRIC_ORDER, values, errors):
            row[name] = value
            row[f"se_{name}_source"] = error
        scenario_rows.append(row)

    point = np.asarray([v_fluid, v_solid], dtype=float)
    comparison = np.real(
        comparison_vector(v_fluid, v_solid, vacancy_fraction)
    ).astype(float)
    comparison_jacobian = complex_step_jacobian(
        lambda vf, vs: comparison_vector(vf, vs, vacancy_fraction),
        point,
    )
    comparison_errors = np.sqrt(
        np.sum(
            (
                comparison_jacobian
                * np.asarray([se_v_fluid, se_v_solid])
            )
            ** 2,
            axis=1,
        )
    )

    baseline_values = scenario_values["commensurate_fcc_baseline"]
    perturbed_values = scenario_values["fixed_site_density_vacancy_scale"]
    baseline_errors = scenario_errors["commensurate_fcc_baseline"]
    metric_index = {name: index for index, name in enumerate(METRIC_ORDER)}
    comparison_rows = []
    n_metrics = len(COMPARISON_METRICS)
    for index, name in enumerate(COMPARISON_METRICS):
        full_index = metric_index[name]
        source_error = baseline_errors[full_index]
        shift = comparison[index]
        comparison_rows.append(
            {
                "metric": name,
                "baseline": baseline_values[full_index],
                "source_standard_error_baseline": source_error,
                "fixed_site_density_vacancy_scale": perturbed_values[
                    full_index
                ],
                "absolute_shift": shift,
                "source_standard_error_absolute_shift": comparison_errors[
                    index
                ],
                "relative_shift_percent": comparison[n_metrics + index],
                "source_standard_error_relative_shift_percent": (
                    comparison_errors[n_metrics + index]
                ),
                "absolute_shift_over_baseline_source_standard_error": (
                    abs(shift) / source_error
                ),
                "interpretation": (
                    "one-sided occupancy-scale perturbation; not a confidence "
                    "interval and not a total defect uncertainty"
                ),
            }
        )
    return pd.DataFrame(scenario_rows), pd.DataFrame(comparison_rows)


def tex_scientific(value: float, significant_digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return (
        f"{mantissa:.{significant_digits - 1}f}"
        rf"\times10^{{{exponent}}}"
    )


def write_tex_macros(
    vacancy_fraction: float,
    scenarios: pd.DataFrame,
    comparison: pd.DataFrame,
    output: Path,
) -> None:
    perturbed = scenarios[
        scenarios["scenario"] == "fixed_site_density_vacancy_scale"
    ].iloc[0]
    by_metric = comparison.set_index("metric")
    coefficient = by_metric.loc["reweight_coefficient"]
    relative = by_metric.loc["relative_mean_number_fraction"]
    delta_w = by_metric.loc["beta_delta_w"]
    q_s = by_metric.loc["constrained_solid_weight"]
    lines = [
        "% Generated by scripts/hard_sphere_defect_sensitivity.py.",
        "% The vacancy scenario is a scale perturbation, not a total uncertainty.",
        rf"\newcommand{{\HardSphereVacancyScaleFraction}}{{{tex_scientific(vacancy_fraction, 2)}}}",
        rf"\newcommand{{\HardSphereVacancyScaleCoefficient}}{{{float(perturbed.reweight_coefficient):.5f}}}",
        rf"\newcommand{{\HardSphereVacancyScaleCoefficientSourceSE}}{{{tex_scientific(float(perturbed.se_reweight_coefficient_source), 2)}}}",
        rf"\newcommand{{\HardSphereVacancyScaleCoefficientShift}}{{{tex_scientific(float(coefficient.absolute_shift), 2)}}}",
        rf"\newcommand{{\HardSphereVacancyScaleCoefficientRelativeShift}}{{{float(coefficient.relative_shift_percent):.2f}\%}}",
        rf"\newcommand{{\HardSphereVacancyScaleCoefficientShiftInSourceSE}}{{{float(coefficient.absolute_shift_over_baseline_source_standard_error):.2f}}}",
        rf"\newcommand{{\HardSphereVacancyScaleRelativeNumberShift}}{{{float(relative.relative_shift_percent):.2f}\%}}",
        rf"\newcommand{{\HardSphereVacancyScaleDeltaWShift}}{{{tex_scientific(float(delta_w.absolute_shift), 2)}}}",
        rf"\newcommand{{\HardSphereVacancyScaleSolidWeightShift}}{{{tex_scientific(float(q_s.absolute_shift), 2)}}}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(
    vacancy_fraction: float,
    scenarios: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    baseline = scenarios[
        scenarios["scenario"] == "commensurate_fcc_baseline"
    ].iloc[0]
    perturbed = scenarios[
        scenarios["scenario"] == "fixed_site_density_vacancy_scale"
    ].iloc[0]
    expected_delta_w_shift = -np.log1p(-vacancy_fraction)
    actual_delta_w_shift = (
        perturbed.beta_delta_w - baseline.beta_delta_w
    )
    if not np.isclose(
        actual_delta_w_shift, expected_delta_w_shift, rtol=2.0e-14
    ):
        raise RuntimeError("vacancy-work shift failed its exact identity")
    if not perturbed.reweight_coefficient < baseline.reweight_coefficient:
        raise RuntimeError("representative vacancy scale did not reduce the gap")
    if not np.all(np.isfinite(comparison.select_dtypes("number"))):
        raise RuntimeError("nonfinite defect-sensitivity output")
    if vacancy_fraction <= 0.0 or vacancy_fraction >= 1.0:
        raise RuntimeError("vacancy fraction is outside (0,1)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    coexistence_path = root / "data_raw" / "fernandez2012_coexistence.csv"
    vacancy_path = root / "data_raw" / "hard_sphere_vacancy_scale.csv"
    coexistence_raw = pd.read_csv(coexistence_path, dtype={"N": str})
    coexistence = coexistence_raw[
        coexistence_raw["N"] == "infinity"
    ].iloc[0]
    vacancy_raw = pd.read_csv(vacancy_path)
    vacancy_fraction = float(
        vacancy_raw[
            vacancy_raw["quantity"] == "vacancy_fraction_near_melting"
        ].iloc[0]["value"]
    )
    source_pressure_shift = float(
        vacancy_raw[
            vacancy_raw["quantity"]
            == "vacancy_coexistence_pressure_shift"
        ].iloc[0]["value"]
    )
    source_chemical_potential_shift = float(
        vacancy_raw[
            vacancy_raw["quantity"]
            == "vacancy_coexistence_chemical_potential_shift"
        ].iloc[0]["value"]
    )

    scenarios, comparison = build_outputs(coexistence, vacancy_fraction)
    validate(vacancy_fraction, scenarios, comparison)
    covariance_sensitivity = density_coefficient_covariance_sensitivity(
        float(coexistence["v_fluid"]),
        float(coexistence["v_solid"]),
        float(coexistence["se_v_fluid"]),
        float(coexistence["se_v_solid"]),
        np.linspace(-1.0, 1.0, 21),
    )

    scenario_path = (
        root / "data_processed" / "hard_sphere_defect_sensitivity.csv"
    )
    comparison_path = (
        root / "generated" / "hard_sphere_defect_sensitivity_summary.csv"
    )
    macro_path = (
        root / "generated" / "hard_sphere_defect_sensitivity_macros.tex"
    )
    metadata_path = (
        root / "generated" / "hard_sphere_defect_sensitivity_metadata.json"
    )
    covariance_path = (
        root / "data_processed" / "hard_sphere_density_covariance_sensitivity.csv"
    )
    scenarios.to_csv(scenario_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    covariance_sensitivity.to_csv(covariance_path, index=False)
    write_tex_macros(vacancy_fraction, scenarios, comparison, macro_path)

    coefficient = comparison.set_index("metric").loc[
        "reweight_coefficient"
    ]
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/hard_sphere_defect_sensitivity.py",
        "reproduction_command": "python scripts/hard_sphere_defect_sensitivity.py",
        "input_sha256": {
            "data_raw/fernandez2012_coexistence.csv": file_sha256(
                coexistence_path
            ),
            "data_raw/hard_sphere_vacancy_scale.csv": file_sha256(
                vacancy_path
            ),
        },
        "sources": {
            "coexistence_specific_volumes": {
                "doi": "10.1103/PhysRevLett.108.165701",
                "uncertainty": "published standard errors; covariance not reported",
                "verified_author_version": {
                    "url": "https://arxiv.org/pdf/1103.2599v3",
                    "version": "arXiv:1103.2599v3, 12 March 2012",
                    "location": "Table I, page 4, infinite-volume row",
                    "finite_size_duplicate": (
                        "Table I gives identical specific volumes and errors "
                        "for N=500 and N=864; retained as in the source, "
                        "without an inferred correction"
                    ),
                },
            },
            "representative_vacancy_fraction": {
                "value": vacancy_fraction,
                "doi": "10.1021/jp010779e",
                "verified_author_version_url": "https://arxiv.org/pdf/cond-mat/0105060v1",
                "verified_location": "page 4, paragraph following Eq. (23)",
                "uncertainty": (
                    "source reports an approximate scale; no statistical "
                    "uncertainty is assigned here"
                ),
            },
            "reported_first_order_defect_shifts": {
                "coexistence_pressure_kBT_per_sigma_cubed": (
                    source_pressure_shift
                ),
                "coexistence_chemical_potential_kBT": (
                    source_chemical_potential_shift
                ),
                "doi": "10.1021/jp010779e",
                "role": (
                    "provenance only; these values are not folded into the "
                    "fixed-site-density scale perturbation"
                ),
            },
        },
        "scenario_definition": {
            "vacancy_fraction": "x_v=(M-N)/M",
            "solid_particle_density": "rho_s(x_v)=(1-x_v)rho_s(0)",
            "held_fixed": ["lattice-site density", "fluid density"],
            "omitted": [
                "lattice relaxation",
                "coexistence-pressure shift",
                "coexistence-chemical-potential shift",
                "fluid-density re-equilibration",
                "uncertainty in vacancy fraction",
            ],
            "status": (
                "one-sided scale perturbation, not a defect-corrected "
                "coexistence result, confidence interval, or total uncertainty"
            ),
        },
        "uncertainty_propagation": {
            "method": "first-order delta method with complex-step Jacobian",
            "specific_volume_covariance": 0.0,
            "reported_se_scope": "Fernandez specific-volume errors only",
            "covariance_sensitivity": {
                "output": str(covariance_path.relative_to(root)),
                "correlation_range": [-1.0, 1.0],
                "standard_error_minimum": float(
                    covariance_sensitivity.se_reweight_coefficient_source.min()
                ),
                "standard_error_maximum": float(
                    covariance_sensitivity.se_reweight_coefficient_source.max()
                ),
                "scope": "sharp first-order range at fixed published marginal errors; not a total uncertainty",
            },
        },
        "headline_result": {
            "coefficient_absolute_shift": float(coefficient.absolute_shift),
            "coefficient_relative_shift_percent": float(
                coefficient.relative_shift_percent
            ),
            "shift_over_published_density_source_se": float(
                coefficient.absolute_shift_over_baseline_source_standard_error
            ),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (
        scenario_path,
        covariance_path,
        comparison_path,
        macro_path,
        metadata_path,
    ):
        print(f"wrote {path.relative_to(root)}")


if __name__ == "__main__":
    main()
