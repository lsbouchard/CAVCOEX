#!/usr/bin/env python3
"""Generate manuscript numerical macros directly from validated artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def grouped_integer(value: int) -> str:
    """Return a TeX integer with thin-space thousands separators."""

    return f"{int(value):,}".replace(",", r"\,")


def scientific(value: float, significant: int = 2) -> str:
    """Return a compact TeX scientific-notation value."""

    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    if abs(round(mantissa, significant - 1)) >= 10.0:
        exponent += 1
        mantissa /= 10.0
    return rf"{mantissa:.{significant - 1}f}\times10^{{{exponent}}}"


def measurement(value: float, standard_error: float, uncertainty_digits: int = 2) -> str:
    """Return conventional ``value(parenthetical uncertainty)`` text."""

    if standard_error <= 0.0:
        return f"{value:g}"
    exponent = int(np.floor(np.log10(abs(standard_error))))
    decimal_places = max(0, -exponent + uncertainty_digits - 1)
    scale = 10.0**decimal_places
    uncertainty_integer = int(round(standard_error * scale))
    return f"{value:.{decimal_places}f}({uncertainty_integer})"


def probability(value: float) -> str:
    """Round empirical probabilities without overstating resampling precision."""

    if value >= 1.0e-2:
        return f"{value:.2f}"
    if value >= 1.0e-3:
        return f"{value:.3f}"
    return scientific(value, 1)


def main() -> None:
    cumulants = pd.read_csv(ROOT / "data_processed" / "cumulant_test.csv")
    finite = pd.read_csv(ROOT / "generated" / "finite_size_fits.csv")
    tonks_meta = json.loads(
        (ROOT / "generated" / "tonks_metadata.json").read_text(encoding="utf-8")
    )
    mc = pd.read_csv(ROOT / "data_processed" / "cavity_mc.csv")
    mc_meta = json.loads(
        (ROOT / "generated" / "cavity_mc_metadata.json").read_text(encoding="utf-8")
    )
    solid = pd.read_csv(ROOT / "data_processed" / "solid_cavity_mc.csv")
    solid_meta = json.loads(
        (ROOT / "generated" / "solid_cavity_mc_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    solid_replica = pd.read_csv(
        ROOT / "data_processed" / "solid_cavity_mc_replica_comparison.csv"
    )
    solid_replica_meta = json.loads(
        (
            ROOT / "generated" / "solid_cavity_mc_replica_metadata.json"
        ).read_text(encoding="utf-8")
    )
    eos = pd.read_csv(ROOT / "generated" / "cs_benchmark.csv")
    eos_meta = json.loads(
        (ROOT / "generated" / "eos_benchmark_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    phase_finite = pd.read_csv(
        ROOT / "generated" / "phase_equal_weight_summary.csv"
    )
    phase_raw = pd.read_csv(
        ROOT / "data_raw" / "fernandez2012_coexistence.csv", dtype={"N": str}
    )
    lattice_meta = json.loads(
        (
            ROOT / "generated" / "lattice_gas_phase_switch_metadata.json"
        ).read_text(encoding="utf-8")
    )
    lattice_exact = lattice_meta["exact_thermodynamic_values"]
    lattice_diagnostics = pd.read_csv(
        ROOT / "data_raw" / "lattice_gas_phase_switch_diagnostics.csv"
    )
    additional_diagnostics_path = (
        ROOT / "data_raw" / "lattice_gas_additional_replicas_diagnostics.csv"
    )
    if additional_diagnostics_path.exists():
        lattice_diagnostics = pd.concat(
            (lattice_diagnostics, pd.read_csv(additional_diagnostics_path)),
            ignore_index=True,
        )
    lattice_maximum_tau = float(
        lattice_diagnostics[
            ["tau_int_abs_magnetization", "tau_int_pair_empty_liquid"]
        ].to_numpy().max()
    )
    lattice_minimum_switches = int(
        lattice_diagnostics["magnetization_sector_reversals"].min()
    )
    lattice_minimum_block_tau_ratio = float(500.0 / lattice_maximum_tau)
    lattice_robust = pd.read_csv(
        ROOT / "data_processed" / "lattice_paired_unpaired_hierarchical.csv"
    )
    lattice_peak_fits = pd.read_csv(
        ROOT / "data_processed" / "lattice_peak_scaling_fits.csv"
    ).set_index("observable")
    lattice_width_fit = pd.read_csv(
        ROOT / "data_processed" / "lattice_width_hierarchical_fit.csv"
    ).iloc[0]
    lattice_widths = pd.read_csv(
        ROOT / "data_processed" / "lattice_width_hierarchical.csv"
    ).sort_values("L")
    lattice_overlap = pd.read_csv(
        ROOT / "generated" / "lattice_reweighting_overlap_minima.csv"
    )
    lattice_fixed_fits = pd.read_csv(
        ROOT / "data_processed" / "lattice_exact_limit_fixed_fits.csv"
    )
    lattice_joint = pd.read_csv(
        ROOT / "data_processed" / "lattice_scalar_joint_validation.csv"
    )
    lattice_holdout = pd.read_csv(
        ROOT / "data_processed" / "lattice_curve_holdout_test.csv"
    )
    lattice_correction_coefficients = pd.read_csv(
        ROOT / "data_processed" / "lattice_pulse_correction_coefficients.csv"
    )
    lattice_correction_slopes = pd.read_csv(
        ROOT / "data_processed" / "lattice_pulse_correction_slope_predictions.csv"
    )
    lattice_free_quadratic = pd.read_csv(
        ROOT
        / "data_processed"
        / "lattice_free_intercept_joint_quadratic_summary.csv"
    )
    lattice_area = pd.read_csv(
        ROOT / "data_processed" / "lattice_pulse_area_sum_rule.csv"
    )
    lattice_area_fits = pd.read_csv(
        ROOT / "data_processed" / "lattice_pulse_area_sum_rule_fits.csv"
    )
    lattice_leave_lmax = pd.read_csv(
        ROOT / "data_processed" / "lattice_leave_lmax_prediction_diagnostics.csv"
    )
    lattice_soft_family = pd.read_csv(
        ROOT / "data_processed" / "lattice_soft_cavity_features.csv"
    )
    lattice_large_L = int(lattice_robust["L"].max())

    def robust_value(
        mode: str, observable: str, length: int | None = None
    ) -> pd.Series:
        selected_length = lattice_large_L if length is None else length
        selected = lattice_robust[
            (lattice_robust["L"] == selected_length)
            & (lattice_robust["mode"] == mode)
            & (lattice_robust["observable"] == observable)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"missing unique robust value for {mode=}, {observable=}"
            )
        return selected.iloc[0]

    paired_delta_w = robust_value("paired", "delta_betaW")
    unpaired_delta_w = robust_value("unpaired", "delta_betaW")
    paired_response = robust_value("paired", "deltaN_phase_reweighting")
    unpaired_response = robust_value("unpaired", "deltaN_phase_reweighting")
    paired_intraphase = robust_value("paired", "deltaN_intraphase")
    paired_gap = robust_value("paired", "delta_rho")
    paired_qc = robust_value("paired", "qC_liquid")
    unpaired_q0 = robust_value("unpaired", "q0_liquid")
    lattice_width_largest = lattice_widths.iloc[-1]
    overlap_orbit = lattice_overlap[
        lattice_overlap["analysis_unit"] == "antithetic_orbit"
    ]
    overlap_block = lattice_overlap[
        lattice_overlap["analysis_unit"] == "replica_time_block"
    ]
    overlap_weakest_orbit = overlap_orbit.loc[
        overlap_orbit["kish_fraction"].idxmin()
    ]
    overlap_weakest_block = overlap_block.loc[
        overlap_block["kish_fraction"].idxmin()
    ]

    def fixed_fit(mode: str, observable: str) -> pd.Series:
        selected = lattice_fixed_fits[
            (lattice_fixed_fits["mode"] == mode)
            & (lattice_fixed_fits["observable"] == observable)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"missing unique exact-limit fit for {mode=}, {observable=}"
            )
        return selected.iloc[0]

    def joint_fit(mode: str, fit_model: str) -> pd.Series:
        selected = lattice_joint[
            (lattice_joint["mode"] == mode)
            & (lattice_joint["record_type"] == "joint_fit")
            & (lattice_joint["fit_model"] == fit_model)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"missing unique joint fit for {mode=}, {fit_model=}"
            )
        return selected.iloc[0]

    paired_linear_joint = joint_fit("paired", "linear_all")
    paired_quadratic_joint = joint_fit("paired", "quadratic_all")
    paired_size_cut_joint = joint_fit("paired", "linear_drop_L16")
    free_quadratic_primary = lattice_free_quadratic[
        lattice_free_quadratic["is_primary_threshold"]
    ].iloc[0]
    area_fixed_quadratic = lattice_area_fits[
        lattice_area_fits["model"] == "fixed_quadratic_all"
    ].iloc[0]
    area_free_quadratic = lattice_area_fits[
        lattice_area_fits["model"] == "free_quadratic_all"
    ].iloc[0]
    area_reference = lattice_area.iloc[0]

    def leave_lmax_result(target: str) -> pd.Series:
        selected = lattice_leave_lmax[
            (lattice_leave_lmax["analysis_config"] == "primary")
            & (lattice_leave_lmax["target"] == target)
            & np.isclose(
                lattice_leave_lmax["covariance_relative_eigenvalue_threshold"],
                1.0e-8,
                rtol=0.0,
                atol=0.0,
            )
        ]
        if len(selected) != 1:
            raise ValueError(f"missing unique primary leave-Lmax result for {target=}")
        return selected.iloc[0]

    leave_lmax_total = leave_lmax_result("total")
    leave_lmax_reweighting = leave_lmax_result("reweighting")
    soft_feature_relative_errors = []
    for observable in (
        "delta_betaW",
        "peak_center_h",
        "peak_magnitude",
        "fwhm_h",
    ):
        soft_feature_relative_errors.extend(
            (
                np.abs(
                    lattice_soft_family[f"measured_{observable}"]
                    - lattice_soft_family[f"exact_{observable}"]
                )
                / np.abs(lattice_soft_family[f"exact_{observable}"])
            ).tolist()
        )
    exact_holdout = lattice_holdout[
        lattice_holdout["prediction_kind"] == "exact_crossfit"
    ]
    logistic_holdout = lattice_holdout[
        lattice_holdout["prediction_kind"] == "leading_logistic"
    ]
    corrected_holdout = lattice_holdout[
        lattice_holdout["prediction_kind"] == "first_corrected_logistic"
    ]

    def correction_coefficient(name: str) -> pd.Series:
        selected = lattice_correction_coefficients[
            (lattice_correction_coefficients["record_type"] == "asymptotic_coefficient")
            & (lattice_correction_coefficients["analysis_config"] == "primary")
            & (lattice_correction_coefficients["coefficient"] == name)
        ]
        if len(selected) != 1:
            raise ValueError(f"missing unique primary correction coefficient {name=}")
        return selected.iloc[0]

    def correction_slope(config: str, observable: str) -> pd.Series:
        selected = lattice_correction_slopes[
            (lattice_correction_slopes["record_type"] == "observable_comparison")
            & (lattice_correction_slopes["analysis_config"] == config)
            & (lattice_correction_slopes["observable"] == observable)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"missing unique correction slope for {config=}, {observable=}"
            )
        return selected.iloc[0]

    def correction_joint(config: str) -> pd.Series:
        selected = lattice_correction_slopes[
            (lattice_correction_slopes["record_type"] == "joint_comparison")
            & (lattice_correction_slopes["analysis_config"] == config)
        ]
        if len(selected) != 1:
            raise ValueError(f"missing unique correction joint result for {config=}")
        return selected.iloc[0]

    primary_correction_slopes = {
        observable: correction_slope("primary", observable)
        for observable in ("peak_center_h", "peak_magnitude", "fwhm_h")
    }
    primary_correction_joint = correction_joint("primary")
    size_cut_correction_joint = correction_joint("drop_L16")
    fixed_spatial_difference_rows = lattice_robust[
        (lattice_robust["mode"] == "paired")
        & lattice_robust["observable"].isin(
            [
                "P_fixed_minus_spatial_liquid",
                "P_fixed_minus_spatial_gas",
                "delta_betaW_fixed_minus_spatial",
            ]
        )
    ].copy()
    fixed_spatial_difference_rows["absolute_z"] = (
        fixed_spatial_difference_rows["value"].abs()
        / fixed_spatial_difference_rows["standard_error"]
    )

    # Reproduce the stated infinite-volume uncertainty propagation from the
    # published specific-volume standard errors. The source reports no
    # covariance, so the two tabulated errors are sampled independently.
    source = phase_raw.loc[phase_raw["N"] == "infinity"].iloc[0]
    rng = np.random.default_rng(24_072_026)
    draws = 20_000
    vf = rng.normal(source.v_fluid, source.se_v_fluid, draws)
    vs = rng.normal(source.v_solid, source.se_v_solid, draws)
    rho_f_draw = 1.0 / vf
    rho_s_draw = 1.0 / vs
    eta_f_draw = np.pi * rho_f_draw / 6.0
    eta_s_draw = np.pi * rho_s_draw / 6.0
    delta_rho_draw = rho_s_draw - rho_f_draw
    delta_w_draw = np.log(rho_f_draw / rho_s_draw)
    qs_draw = rho_s_draw / (rho_f_draw + rho_s_draw)
    coefficient_draw = delta_rho_draw**2 / (
        2.0 * (rho_f_draw + rho_s_draw)
    )
    relative_number_draw = (
        delta_rho_draw**2 / (rho_f_draw + rho_s_draw) ** 2
    )
    particles_per_thousand_draw = 1000.0 * relative_number_draw
    one_particle_n_draw = 1.0 / relative_number_draw
    width_draw = (
        2.0
        * np.arccosh(2.0 + np.cosh(0.5 * delta_w_draw))
        / np.abs(delta_rho_draw)
    )

    rho_f = 1.0 / float(source.v_fluid)
    rho_s = 1.0 / float(source.v_solid)
    eta_f = np.pi * rho_f / 6.0
    eta_s = np.pi * rho_s / 6.0
    delta_w = float(np.log(rho_f / rho_s))
    qs = float(rho_s / (rho_f + rho_s))
    coefficient = float((rho_s - rho_f) ** 2 / (2.0 * (rho_f + rho_s)))
    relative_number = float((rho_s - rho_f) ** 2 / (rho_f + rho_s) ** 2)
    particles_per_thousand = 1000.0 * relative_number
    one_particle_n = 1.0 / relative_number
    width = float(
        2.0 * np.arccosh(2.0 + np.cosh(0.5 * delta_w)) / abs(rho_s - rho_f)
    )

    direct_pilot = mc[
        (mc["N"] == 108)
        & np.isclose(mc["eta"], 0.40)
        & np.isclose(mc["R_over_sigma"], 1.0)
    ].iloc[0]
    direct_large = mc[
        (mc["N"] == 256)
        & np.isclose(mc["eta"], 0.40)
        & np.isclose(mc["R_over_sigma"], 1.0)
    ].iloc[0]
    solid_small = solid[
        (solid["N"] == 108) & np.isclose(solid["R_over_sigma"], 0.75)
    ].iloc[0]
    solid_large = solid[
        (solid["N"] == 256) & np.isclose(solid["R_over_sigma"], 0.75)
    ].iloc[0]
    solid_small_point_eight = solid[
        (solid["N"] == 108) & np.isclose(solid["R_over_sigma"], 0.80)
    ].iloc[0]
    solid_large_point_eight = solid[
        (solid["N"] == 256) & np.isclose(solid["R_over_sigma"], 0.80)
    ].iloc[0]

    def solid_replica_row(size: int, metric: str) -> pd.Series:
        selected = solid_replica[
            (solid_replica["N"] == size)
            & (solid_replica["metric"] == metric)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"missing unique solid-replica row for {size=}, {metric=}"
            )
        return selected.iloc[0]

    solid_replica_small = solid_replica_row(108, "betaW_R0p80")
    solid_replica_large = solid_replica_row(256, "betaW_R0p80")
    solid_replica_small_pressure = solid_replica_row(
        108, "beta_pressure_contact_shell_delta_0p02"
    )
    solid_replica_large_pressure = solid_replica_row(
        256, "beta_pressure_contact_shell_delta_0p02"
    )
    replica_settings = solid_replica_meta["settings"]
    solid_replica_probes = int(
        replica_settings["production_blocks"]
        * replica_settings["sweeps_per_block"]
        / replica_settings["sample_stride"]
        * replica_settings["probes_per_sample"]
    )
    eos_reference_rows = []
    for model, group in eos.groupby("model"):
        ordered = group.sort_values("eta")
        eos_reference_rows.append(
            {
                "model": model,
                "S0": float(
                    np.interp(0.492, ordered["eta"], ordered["S0"])
                ),
                "dW_dmu": float(
                    np.interp(0.492, ordered["eta"], ordered["dW_dmu"])
                ),
            }
        )
    eos_at_reference = pd.DataFrame(eos_reference_rows).set_index("model")
    eos_s0_min = float(eos_at_reference["S0"].min())
    eos_s0_max = float(eos_at_reference["S0"].max())
    eos_response_min = float(eos_at_reference["dW_dmu"].min())
    eos_response_max = float(eos_at_reference["dW_dmu"].max())
    eos_fd_max = max(
        eos_meta["checks"][
            "random_five_point_finite_difference_max_relative_errors"
        ].values()
    )
    lattice_exact_width = float(lattice_width_fit.exact_limit)
    lattice_width_percent_error = 100.0 * (
        lattice_width_largest.fwhm_h / lattice_exact_width - 1.0
    )
    lattice_width_percent_se = (
        100.0
        * lattice_width_largest.se_fwhm_h
        / lattice_exact_width
    )

    relative_by_radius = {}
    for radius in (0.75, 1.00, 1.25, 1.50):
        selected = finite[np.isclose(finite["R_over_sigma"], radius)]
        relative_by_radius[radius] = float(
            np.max(np.abs(selected["relative_difference"]))
        )

    contribution_text = []
    for row in phase_finite.sort_values("N_fluid_reference").itertuples():
        contribution_text.append(
            f"{row.deltaN_reweight:.3f}({int(round(row.se_deltaN_reweight * 1000))})"
        )

    settings = mc_meta["derived_settings"]
    n_states = len(mc[["N", "eta"]].drop_duplicates())
    total_probes = n_states * int(settings["probe_trials_per_state"])
    lines = [
        "% Generated by scripts/make_numerical_macros.py; do not edit.",
        rf"\newcommand{{\CumulantComparisonCount}}{{{len(cumulants)}}}",
        rf"\newcommand{{\CumulantMaximumZ}}{{{np.max(np.abs(cumulants['standardized_residual'])):.3f}}}",
        rf"\newcommand{{\TonksMaximumIdentityError}}{{{scientific(float(tonks_meta['maximum_identity_absolute_error']), 3)}}}",
        rf"\newcommand{{\FiniteErrorRThreeQuarter}}{{{scientific(relative_by_radius[0.75], 2)}}}",
        rf"\newcommand{{\FiniteErrorROne}}{{{scientific(relative_by_radius[1.00], 2)}}}",
        rf"\newcommand{{\FiniteErrorRFiveQuarter}}{{{scientific(relative_by_radius[1.25], 2)}}}",
        rf"\newcommand{{\FiniteErrorRThreeHalf}}{{{scientific(relative_by_radius[1.50], 2)}}}",
        rf"\newcommand{{\MCProbeCountPerState}}{{{grouped_integer(settings['probe_trials_per_state'])}}}",
        rf"\newcommand{{\MCTotalProbeCount}}{{{grouped_integer(total_probes)}}}",
        rf"\newcommand{{\MCMaximumControlZ}}{{{mc_meta['quality_summary']['maximum_abs_R0p5_exact_z']:.2f}}}",
        rf"\newcommand{{\MCReliableRows}}{{{mc_meta['quality_summary']['direct_rows_reliable']}}}",
        rf"\newcommand{{\MCTotalRows}}{{{mc_meta['quality_summary']['direct_rows_total']}}}",
        rf"\newcommand{{\MCPilotEmptyEvents}}{{{int(direct_pilot.empty_events)}}}",
        rf"\newcommand{{\MCPilotProbability}}{{{direct_pilot.P_empty * 1e5:.2f}({int(round(direct_pilot.se_P_empty * 1e7)):02d})\times10^{{-5}}}}",
        rf"\newcommand{{\MCLargeEmptyEvents}}{{{int(direct_large.empty_events)}}}",
        rf"\newcommand{{\MCLargeProbability}}{{{direct_large.P_empty * 1e4:.3f}({int(round(direct_large.se_P_empty * 1e7)):02d})\times10^{{-4}}}}",
        rf"\newcommand{{\SolidProbeCountPerSize}}{{{grouped_integer(solid_meta['derived_settings']['probe_trials_per_state'])}}}",
        rf"\newcommand{{\SolidTotalProbeCount}}{{{grouped_integer(2 * solid_meta['derived_settings']['probe_trials_per_state'])}}}",
        rf"\newcommand{{\SolidProductionSweepCount}}{{{grouped_integer(solid_meta['derived_settings']['production_sweeps'])}}}",
        rf"\newcommand{{\SolidProductionBlockCount}}{{{solid_meta['settings']['production_blocks']}}}",
        rf"\newcommand{{\SolidRadiusCount}}{{{solid['R_over_sigma'].nunique()}}}",
        rf"\newcommand{{\SolidMaximumControlZ}}{{{solid_meta['quality_summary']['maximum_abs_R0p5_exact_z']:.2f}}}",
        rf"\newcommand{{\SolidSmallRThreeQuarterWork}}{{{solid_small.betaW_N:.3f}({int(round(solid_small.se_betaW_N * 1000)):02d})}}",
        rf"\newcommand{{\SolidLargeRThreeQuarterWork}}{{{solid_large.betaW_N:.3f}({int(round(solid_large.se_betaW_N * 1000)):02d})}}",
        rf"\newcommand{{\SolidSmallRThreeQuarterEvents}}{{{int(solid_small.empty_events)}}}",
        rf"\newcommand{{\SolidLargeRThreeQuarterEvents}}{{{int(solid_large.empty_events)}}}",
        rf"\newcommand{{\SolidSmallRPointEightWork}}{{{solid_small_point_eight.betaW_N:.3f}({int(round(solid_small_point_eight.se_betaW_N * 1000)):02d})}}",
        rf"\newcommand{{\SolidLargeRPointEightWork}}{{{solid_large_point_eight.betaW_N:.2f}({int(round(solid_large_point_eight.se_betaW_N * 100)):02d})}}",
        rf"\newcommand{{\SolidSmallRPointEightEvents}}{{{int(solid_small_point_eight.empty_events)}}}",
        rf"\newcommand{{\SolidLargeRPointEightEvents}}{{{int(solid_large_point_eight.empty_events)}}}",
        rf"\newcommand{{\SolidMinimumQSix}}{{{solid_meta['quality_summary']['minimum_production_Q6']:.3f}}}",
        rf"\newcommand{{\SolidMinimumBragg}}{{{solid_meta['quality_summary']['minimum_production_FCC_Bragg']:.3f}}}",
        rf"\newcommand{{\SolidMaximumLindemann}}{{{solid_meta['quality_summary']['maximum_production_Lindemann']:.3f}}}",
        rf"\newcommand{{\SolidReplicaCountPerSize}}{{{solid_replica_meta['replicas_per_size']}}}",
        rf"\newcommand{{\SolidReplicaTrajectoryCount}}{{{solid_replica_meta['quality_summary']['independent_state_trajectories']}}}",
        rf"\newcommand{{\SolidReplicaProbeCountPerTrajectory}}{{{grouped_integer(solid_replica_probes)}}}",
        rf"\newcommand{{\SolidReplicaSmallWork}}{{{measurement(solid_replica_small.additional_mean, solid_replica_small.hierarchical_se)}}}",
        rf"\newcommand{{\SolidReplicaLargeWork}}{{{measurement(solid_replica_large.additional_mean, solid_replica_large.hierarchical_se)}}}",
        rf"\newcommand{{\SolidReplicaSmallBaselineZ}}{{{abs(solid_replica_small.additional_minus_baseline_z):.2f}}}",
        rf"\newcommand{{\SolidReplicaLargeBaselineZ}}{{{abs(solid_replica_large.additional_minus_baseline_z):.2f}}}",
        rf"\newcommand{{\SolidReplicaMinimumEvents}}{{{solid_replica_meta['quality_summary']['minimum_R0p80_events_per_replica']}}}",
        rf"\newcommand{{\SolidReplicaMaximumStructuralZ}}{{{solid_replica_meta['quality_summary']['maximum_abs_structural_baseline_difference_z']:.2f}}}",
        rf"\newcommand{{\SolidReplicaMaximumWorkQPerDof}}{{{max(solid_replica_small.cochran_q_per_dof, solid_replica_large.cochran_q_per_dof):.2f}}}",
        rf"\newcommand{{\SolidReplicaMaximumWorkDeleteOneZ}}{{{max(solid_replica_small.max_delete_one_shift_over_hierarchical_se, solid_replica_large.max_delete_one_shift_over_hierarchical_se):.2f}}}",
        rf"\newcommand{{\SolidReplicaSmallPressure}}{{{measurement(solid_replica_small_pressure.additional_mean, solid_replica_small_pressure.hierarchical_se)}}}",
        rf"\newcommand{{\SolidReplicaLargePressure}}{{{measurement(solid_replica_large_pressure.additional_mean, solid_replica_large_pressure.hierarchical_se)}}}",
        rf"\newcommand{{\EOSFluidEtaMarker}}{{{eos_meta['coexistence_marker']['eta_fluid']:.6f}({int(round(eos_meta['coexistence_marker']['eta_fluid_standard_uncertainty'] * 1e6)):02d})}}",
        rf"\newcommand{{\EOSSZeroRange}}{{{eos_s0_min:.4f}\text{{--}}{eos_s0_max:.4f}}}",
        rf"\newcommand{{\EOSResponseRange}}{{{eos_response_min:.4f}\text{{--}}{eos_response_max:.4f}}}",
        rf"\newcommand{{\EOSMaximumDerivativeError}}{{{scientific(float(eos_fd_max), 2)}}}",
        rf"\newcommand{{\CoexistenceEtaSolid}}{{{eta_s:.6f}({int(round(np.std(eta_s_draw, ddof=1) * 1e6)):02d})}}",
        rf"\newcommand{{\CoexistenceEtaFluid}}{{{eta_f:.6f}({int(round(np.std(eta_f_draw, ddof=1) * 1e6)):02d})}}",
        rf"\newcommand{{\CoexistenceDeltaW}}{{{delta_w:.5f}({int(round(np.std(delta_w_draw, ddof=1) * 1e5)):02d})}}",
        rf"\newcommand{{\CoexistenceConstrainedSolidWeight}}{{{qs:.6f}({int(round(np.std(qs_draw, ddof=1) * 1e6)):02d})}}",
        rf"\newcommand{{\CoexistenceReweightCoefficient}}{{{coefficient:.7f}({int(round(np.std(coefficient_draw, ddof=1) * 1e7)):02d})}}",
        rf"\newcommand{{\CoexistenceFWHMCoefficient}}{{{width:.3f}({int(round(np.std(width_draw, ddof=1) * 1e3)):02d})}}",
        rf"\newcommand{{\CoexistenceRelativeNumberFraction}}{{{measurement(relative_number, float(np.std(relative_number_draw, ddof=1)))}}}",
        rf"\newcommand{{\CoexistenceParticlesPerThousand}}{{{measurement(particles_per_thousand, float(np.std(particles_per_thousand_draw, ddof=1)))}}}",
        rf"\newcommand{{\CoexistenceOneParticleN}}{{{measurement(one_particle_n, float(np.std(one_particle_n_draw, ddof=1)))}}}",
        rf"\newcommand{{\CoexistenceFiniteSizeCount}}{{{len(phase_finite)}}}",
        rf"\newcommand{{\CoexistenceFiniteContributions}}{{{', '.join(contribution_text)}}}",
        rf"\newcommand{{\LatticeGasLargestL}}{{{lattice_large_L}}}",
        rf"\newcommand{{\LatticeGasReplicaCount}}{{{int(paired_delta_w.replicas)}}}",
        rf"\newcommand{{\LatticeGasStepsPerReplica}}{{{grouped_integer(lattice_meta['settings']['production_steps'])}}}",
        rf"\newcommand{{\LatticeGasMinimumSwitches}}{{{grouped_integer(lattice_minimum_switches)}}}",
        rf"\newcommand{{\LatticeGasExactMagnetization}}{{{lattice_exact['spontaneous_magnetization']:.5f}}}",
        rf"\newcommand{{\LatticeGasExactPairCorrelation}}{{{lattice_exact['nearest_neighbor_correlation']:.5f}}}",
        rf"\newcommand{{\LatticeGasExactPairEmptyLiquid}}{{{lattice_exact['pair_empty_liquid']:.5f}}}",
        rf"\newcommand{{\LatticeGasExactPairEmptyGas}}{{{lattice_exact['pair_empty_gas']:.5f}}}",
        rf"\newcommand{{\LatticeGasMeasuredGap}}{{{measurement(paired_gap.value, paired_gap.standard_error)}}}",
        rf"\newcommand{{\LatticeGasExactDeltaW}}{{{lattice_exact['delta_betaW']:.4f}}}",
        rf"\newcommand{{\LatticeGasMeasuredDeltaW}}{{{measurement(paired_delta_w.value, paired_delta_w.standard_error)}}}",
        rf"\newcommand{{\LatticeGasConstrainedLiquidWeight}}{{{measurement(paired_qc.value, paired_qc.standard_error)}}}",
        rf"\newcommand{{\LatticeGasReweightingResponse}}{{{measurement(paired_response.value, paired_response.standard_error)}}}",
        rf"\newcommand{{\LatticeGasIntraphaseResponse}}{{{measurement(paired_intraphase.value, paired_intraphase.standard_error)}}}",
        rf"\newcommand{{\LatticeGasPairedDeltaW}}{{{measurement(paired_delta_w.value, paired_delta_w.standard_error)}}}",
        rf"\newcommand{{\LatticeGasUnpairedDeltaW}}{{{measurement(unpaired_delta_w.value, unpaired_delta_w.standard_error)}}}",
        rf"\newcommand{{\LatticeGasPairedReweightingResponse}}{{{measurement(paired_response.value, paired_response.standard_error)}}}",
        rf"\newcommand{{\LatticeGasUnpairedReweightingResponse}}{{{measurement(unpaired_response.value, unpaired_response.standard_error)}}}",
        rf"\newcommand{{\LatticeGasUnpairedPhaseWeight}}{{{measurement(unpaired_q0.value, unpaired_q0.standard_error)}}}",
        rf"\newcommand{{\LatticeGasMeasuredWidth}}{{{measurement(lattice_width_largest.fwhm_h, lattice_width_largest.se_fwhm_h)}}}",
        rf"\newcommand{{\LatticeGasExactWidth}}{{{lattice_exact_width:.4f}}}",
        rf"\newcommand{{\LatticeGasWidthPercentError}}{{{lattice_width_percent_error:.3f}({int(round(lattice_width_percent_se * 1e3))})}}",
        rf"\newcommand{{\LatticeGasMaximumTau}}{{{lattice_maximum_tau:.2f}}}",
        rf"\newcommand{{\LatticeGasMinimumBlockTauRatio}}{{{lattice_minimum_block_tau_ratio:.0f}}}",
        rf"\newcommand{{\LatticeGasPeakCenterIntercept}}{{{measurement(lattice_peak_fits.loc['peak_center_h', 'A'], lattice_peak_fits.loc['peak_center_h', 'se_A'])}}}",
        rf"\newcommand{{\LatticeGasExactPeakCenter}}{{{lattice_peak_fits.loc['peak_center_h', 'exact_limit']:.4f}}}",
        rf"\newcommand{{\LatticeGasPeakCenterChiSquare}}{{{lattice_peak_fits.loc['peak_center_h', 'chi2']:.3f}}}",
        rf"\newcommand{{\LatticeGasPeakMagnitudeIntercept}}{{{measurement(lattice_peak_fits.loc['peak_magnitude', 'A'], lattice_peak_fits.loc['peak_magnitude', 'se_A'])}}}",
        rf"\newcommand{{\LatticeGasExactPeakMagnitude}}{{{lattice_peak_fits.loc['peak_magnitude', 'exact_limit']:.5f}}}",
        rf"\newcommand{{\LatticeGasPeakMagnitudeChiSquare}}{{{lattice_peak_fits.loc['peak_magnitude', 'chi2']:.3f}}}",
        rf"\newcommand{{\LatticeGasWidthIntercept}}{{{measurement(lattice_width_fit.A, lattice_width_fit.se_A)}}}",
        rf"\newcommand{{\LatticeGasWidthInterceptExact}}{{{lattice_width_fit.exact_limit:.4f}}}",
        rf"\newcommand{{\LatticeGasWidthFitChiSquare}}{{{lattice_width_fit.chi2:.3f}}}",
        rf"\newcommand{{\LatticeGasPulseFitDOF}}{{{int(lattice_width_fit.dof)}}}",
        rf"\newcommand{{\LatticeGasMinimumOrbitSupportFraction}}{{{overlap_weakest_orbit.kish_fraction:.3f}}}",
        rf"\newcommand{{\LatticeGasMinimumOrbitPerplexityFraction}}{{{overlap_weakest_orbit.perplexity_fraction:.3f}}}",
        rf"\newcommand{{\LatticeGasMaximumOrbitWeightFraction}}{{{overlap_weakest_orbit.maximum_normalized_weight:.4f}}}",
        rf"\newcommand{{\LatticeGasMinimumBlockSupportFraction}}{{{overlap_weakest_block.kish_fraction:.3f}}}",
        rf"\newcommand{{\LatticeGasMaximumBlockWeightFraction}}{{{overlap_weakest_block.maximum_normalized_weight:.4f}}}",
        rf"\newcommand{{\LatticeGasCenterFixedPValue}}{{{probability(float(fixed_fit('paired', 'peak_center_h').centered_hierarchical_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasMagnitudeFixedPValue}}{{{probability(float(fixed_fit('paired', 'peak_magnitude').centered_hierarchical_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasWidthFixedPValue}}{{{probability(float(fixed_fit('paired', 'fwhm_h').centered_hierarchical_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasLinearJointPValue}}{{{probability(float(paired_linear_joint.joint_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasQuadraticJointPValue}}{{{probability(float(paired_quadratic_joint.joint_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasSizeCutJointPValue}}{{{probability(float(paired_size_cut_joint.joint_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasFreeQuadraticGlobalPValue}}{{{probability(float(free_quadratic_primary.global_lack_of_fit_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasExactInterceptJointPValue}}{{{probability(float(free_quadratic_primary.intercept_test_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasAreaExactFull}}{{{float(area_reference.exact_full_area):.4f}}}",
        rf"\newcommand{{\LatticeGasAreaExactWindow}}{{{float(area_reference.exact_truncated_area):.4f}}}",
        rf"\newcommand{{\LatticeGasAreaTailPercent}}{{{100.0 * abs(float(area_reference.exact_omitted_tail_area) / float(area_reference.exact_full_area)):.3f}\%}}",
        rf"\newcommand{{\LatticeGasAreaFreeQuadratic}}{{{measurement(float(area_free_quadratic.thermodynamic_limit), float(area_free_quadratic.bootstrap_se_thermodynamic_limit))}}}",
        rf"\newcommand{{\LatticeGasAreaFreeQuadraticZ}}{{{abs(float(area_free_quadratic.difference_from_exact_in_se)):.3f}}}",
        rf"\newcommand{{\LatticeGasAreaFixedQuadraticPValue}}{{{probability(float(area_fixed_quadratic.centered_hierarchical_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasLeaveLmaxTotalMaximumZ}}{{{float(leave_lmax_total.max_abs_pointwise_z):.3f}}}",
        rf"\newcommand{{\LatticeGasLeaveLmaxReweightMaximumZ}}{{{float(leave_lmax_reweighting.max_abs_pointwise_z):.3f}}}",
        rf"\newcommand{{\LatticeGasLeaveLmaxTotalSimultaneousPValue}}{{{probability(float(leave_lmax_total.simultaneous_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasLeaveLmaxReweightSimultaneousPValue}}{{{probability(float(leave_lmax_reweighting.simultaneous_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasLeaveLmaxTotalCovariancePValue}}{{{probability(float(leave_lmax_total.covariance_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasLeaveLmaxReweightCovariancePValue}}{{{probability(float(leave_lmax_reweighting.covariance_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasLeaveLmaxMaximumRelativeError}}{{{100.0 * max(float(leave_lmax_total.maximum_relative_difference_to_heldout_peak), float(leave_lmax_reweighting.maximum_relative_difference_to_heldout_peak)):.4f}\%}}",
        rf"\newcommand{{\LatticeGasSoftFamilyCount}}{{{len(lattice_soft_family)}}}",
        rf"\newcommand{{\LatticeGasSoftFamilyMinimumWork}}{{{float(lattice_soft_family.exact_delta_betaW.min()):.5f}}}",
        rf"\newcommand{{\LatticeGasSoftFamilyMaximumWork}}{{{float(lattice_soft_family.exact_delta_betaW.max()):.4f}}}",
        rf"\newcommand{{\LatticeGasSoftFamilyMaximumWorkZ}}{{{float(np.max(np.abs(lattice_soft_family.difference_z_delta_betaW))):.3f}}}",
        rf"\newcommand{{\LatticeGasSoftFamilyMaximumRelativeFeatureError}}{{{100.0 * max(soft_feature_relative_errors):.3f}\%}}",
        rf"\newcommand{{\LatticeGasHoldoutMinimumExactPValue}}{{{probability(float(exact_holdout.centered_hierarchical_bootstrap_p_value.min()))}}}",
        rf"\newcommand{{\LatticeGasHoldoutMinimumLogisticPValue}}{{{probability(float(logistic_holdout.centered_hierarchical_bootstrap_p_value.min()))}}}",
        rf"\newcommand{{\LatticeGasHoldoutMinimumCorrectedPValue}}{{{probability(float(corrected_holdout.centered_hierarchical_bootstrap_p_value.min()))}}}",
        rf"\newcommand{{\LatticeGasHoldoutCorrectedRMSEImprovement}}{{{100.0 * (1.0 - corrected_holdout.rmse_response_over_V.mean() / logistic_holdout.rmse_response_over_V.mean()):.1f}\%}}",
        rf"\newcommand{{\LatticeGasCorrectionDZero}}{{{correction_coefficient('d0').estimate:.3f}\pm{correction_coefficient('d0').bootstrap_se:.3f}}}",
        rf"\newcommand{{\LatticeGasCorrectionAOne}}{{{correction_coefficient('a1').estimate:.2f}\pm{correction_coefficient('a1').bootstrap_se:.2f}}}",
        rf"\newcommand{{\LatticeGasCorrectionDn}}{{{correction_coefficient('dn').estimate:.3f}\pm{correction_coefficient('dn').bootstrap_se:.3f}}}",
        rf"\newcommand{{\LatticeGasCorrectionMaximumMarginalZ}}{{{max(abs(row.difference_z) for row in primary_correction_slopes.values()):.2f}}}",
        rf"\newcommand{{\LatticeGasCorrectionAllSizeJointPValue}}{{{probability(float(primary_correction_joint.joint_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasCorrectionSizeCutJointPValue}}{{{probability(float(size_cut_correction_joint.joint_centered_bootstrap_p_value))}}}",
        rf"\newcommand{{\LatticeGasFixedSpatialMaximumZ}}{{{fixed_spatial_difference_rows.absolute_z.max():.2f}}}",
        "",
    ]
    output = ROOT / "generated" / "numerical_results.tex"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output.relative_to(ROOT)}")

    table_lines = [
        "% Generated by scripts/make_numerical_macros.py; do not edit.",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{rcccc}",
        r"$L$ & $\beta\Delta W_L$ paired & $\beta\Delta W_L$ raw & "
        r"$|\Delta N_{\rm rw}|/V$ paired & $|\Delta N_{\rm rw}|/V$ raw \\",
        r"\colrule",
    ]
    for length in sorted(lattice_robust["L"].unique()):
        volume = int(length) ** 2
        delta_w_paired = robust_value("paired", "delta_betaW", int(length))
        delta_w_unpaired = robust_value("unpaired", "delta_betaW", int(length))
        response_paired = robust_value(
            "paired", "deltaN_phase_reweighting", int(length)
        )
        response_unpaired = robust_value(
            "unpaired", "deltaN_phase_reweighting", int(length)
        )
        table_lines.append(
            f"{int(length)} & "
            f"{measurement(delta_w_paired.value, delta_w_paired.standard_error)} & "
            f"{measurement(delta_w_unpaired.value, delta_w_unpaired.standard_error)} & "
            f"{measurement(abs(response_paired.value) / volume, response_paired.standard_error / volume)} & "
            f"{measurement(abs(response_unpaired.value) / volume, response_unpaired.standard_error / volume)} "
            r"\\"
        )
    table_lines.extend((r"\end{tabular}", r"\end{ruledtabular}", ""))
    table_output = ROOT / "generated" / "lattice_paired_unpaired_table.tex"
    table_output.write_text("\n".join(table_lines), encoding="utf-8")
    print(f"wrote {table_output.relative_to(ROOT)}")

    correction_table_lines = [
        "% Generated by scripts/make_numerical_macros.py; do not edit.",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lccc}",
        r"Pulse quantity & Predicted $B_X$ & Fitted $B_X$ & Difference/$\mathrm{SE}$ \\",
        r"\colrule",
    ]
    correction_labels = {
        "peak_center_h": r"Center $h_{\rm peak}$",
        "peak_magnitude": r"Magnitude $|\Delta N_{\rm rw}|_{\rm peak}/V$",
        "fwhm_h": r"Width $V\Delta x_{\rm FWHM}$",
    }
    for observable in ("peak_center_h", "peak_magnitude", "fwhm_h"):
        row = primary_correction_slopes[observable]
        correction_table_lines.append(
            f"{correction_labels[observable]} & "
            f"{measurement(row.predicted_B, row.predicted_bootstrap_se)} & "
            f"{measurement(row.direct_B, row.direct_bootstrap_se)} & "
            f"{row.difference_z:+.2f} "
            r"\\"
        )
    correction_table_lines.extend((r"\end{tabular}", r"\end{ruledtabular}", ""))
    correction_table_output = (
        ROOT / "generated" / "lattice_pulse_correction_slope_table.tex"
    )
    correction_table_output.write_text(
        "\n".join(correction_table_lines), encoding="utf-8"
    )
    print(f"wrote {correction_table_output.relative_to(ROOT)}")

    fixed_spatial_table_lines = [
        "% Generated by scripts/make_numerical_macros.py; do not edit.",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{rcccccc}",
        r"$L$ & $p_\ell^{\rm fixed}$ & $p_\ell^{\rm avg}$ & "
        r"$p_g^{\rm fixed}$ & $p_g^{\rm avg}$ & "
        r"$\beta\Delta W^{\rm fixed}$ & $\beta\Delta W^{\rm avg}$ \\",
        r"\colrule",
    ]
    for length in sorted(lattice_robust["L"].unique()):
        selected_length = int(length)
        p_liquid_fixed = robust_value(
            "paired", "P_fixed_empty_liquid", selected_length
        )
        p_liquid_spatial = robust_value(
            "paired", "P_spatial_empty_liquid", selected_length
        )
        p_gas_fixed = robust_value(
            "paired", "P_fixed_empty_gas", selected_length
        )
        p_gas_spatial = robust_value(
            "paired", "P_spatial_empty_gas", selected_length
        )
        work_fixed = robust_value(
            "paired", "delta_betaW_fixed", selected_length
        )
        work_spatial = robust_value("paired", "delta_betaW", selected_length)
        fixed_spatial_table_lines.append(
            f"{selected_length} & "
            f"{measurement(p_liquid_fixed.value, p_liquid_fixed.standard_error)} & "
            f"{measurement(p_liquid_spatial.value, p_liquid_spatial.standard_error)} & "
            f"{measurement(p_gas_fixed.value, p_gas_fixed.standard_error)} & "
            f"{measurement(p_gas_spatial.value, p_gas_spatial.standard_error)} & "
            f"{measurement(work_fixed.value, work_fixed.standard_error)} & "
            f"{measurement(work_spatial.value, work_spatial.standard_error)} "
            r"\\"
        )
    fixed_spatial_table_lines.extend((r"\end{tabular}", r"\end{ruledtabular}", ""))
    fixed_spatial_table_output = (
        ROOT / "generated" / "lattice_fixed_spatial_table.tex"
    )
    fixed_spatial_table_output.write_text(
        "\n".join(fixed_spatial_table_lines), encoding="utf-8"
    )
    print(f"wrote {fixed_spatial_table_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
