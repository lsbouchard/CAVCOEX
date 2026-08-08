#!/usr/bin/env python3
"""Independent-replica audit of the fixed-N commensurate FCC references.

This campaign reuses the production machinery in ``solid_cavity_mc.py`` but
writes to a separate raw archive.  Its purpose is to test independent-seed
consistency of the rare-event estimate at R/sigma=0.80 and of crystalline
structural diagnostics.  It remains a canonical, phase-restricted reference
calculation: no fluid/solid common normalization or phase switch is inferred.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig")
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy

from cavity_mc import (
    DOUBLE_BLOCK_SE_RATIO_INTERVAL,
    MAX_HALF_DIFFERENCE_Z,
    MAX_RELATIVE_SE,
    MIN_EMPTY_EVENTS,
    MIN_POSITIVE_BLOCKS,
)
from solid_cavity_mc import (
    CONTACT_SHELL_SENSITIVITY_WIDTHS,
    SolidRunSettings,
    load_solid_coexistence_density,
    run_state,
)


ANALYSIS_VERSION = "solid-cavity-independent-replicas-v1"
DEFAULT_SIZES = (108, 256)
DEFAULT_REPLICAS = 4
BASE_SEED = 631_026_801
REPLICA_SEED_STRIDE = 10_000_019
TARGET_RADIUS = 0.80
RAW_BLOCKS_PATH = "data_raw/solid_cavity_mc_additional_replica_blocks.csv"
RAW_STRUCTURE_PATH = "data_raw/solid_cavity_mc_additional_replica_structure.csv"
RAW_STATES_PATH = "data_raw/solid_cavity_mc_additional_replica_states.npz"
PROCESSED_PATH = "data_processed/solid_cavity_mc_additional_replicas.csv"
STRUCTURE_SUMMARY_PATH = (
    "data_processed/solid_cavity_mc_replica_structure_summary.csv"
)
COMPARISON_PATH = "data_processed/solid_cavity_mc_replica_comparison.csv"
FIGURE_PATH = "figures/solid_cavity_mc_replica_audit.pdf"
METADATA_PATH = "generated/solid_cavity_mc_replica_metadata.json"
RESULTS_PATH = "generated/solid_cavity_mc_replica_results.md"
ZERO_EVENT_BOUND_COLUMNS = (
    "zero_event_P_upper95",
    "zero_event_P_upper95_independent_block",
    "zero_event_bound_scope",
    "zero_event_betaW_lower95",
    "zero_event_betaW_lower95_independent_block",
)


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


def replica_base_seed(base_seed: int, replica: int) -> int:
    """Map a replica index to a disjoint deterministic seed stream."""

    if replica < 0:
        raise ValueError("replica must be nonnegative")
    return int(base_seed + REPLICA_SEED_STRIDE * replica)


def conservative_block_se(values: np.ndarray) -> tuple[float, int]:
    """Largest adjacent-batch standard error over 1, 2, 5, and 10 blocks."""

    vector = np.asarray(values, dtype=np.float64)
    candidates: dict[int, float] = {}
    for factor in (1, 2, 5, 10):
        count = vector.size // factor
        if count < 4:
            continue
        grouped = vector[: count * factor].reshape(count, factor).mean(axis=1)
        candidates[factor] = float(
            np.std(grouped, ddof=1) / np.sqrt(grouped.size)
        )
    finite = {key: value for key, value in candidates.items() if np.isfinite(value)}
    if not finite:
        return float("nan"), 1
    selected = max(finite, key=finite.get)
    return finite[selected], selected


def hierarchical_summary(
    values: np.ndarray, within_standard_errors: np.ndarray
) -> dict[str, float]:
    """Conservative equal-replica mean and hierarchical uncertainty."""

    vector = np.asarray(values, dtype=np.float64)
    within = np.asarray(within_standard_errors, dtype=np.float64)
    if vector.size < 2 or vector.size != within.size:
        raise ValueError("At least two values and one SE per value are required.")
    mean = float(np.mean(vector))
    within_se = float(np.sqrt(np.sum(within**2)) / vector.size)
    between_se = float(np.std(vector, ddof=1) / np.sqrt(vector.size))
    hierarchical_se = float(max(within_se, between_se))
    delete_one = np.asarray(
        [np.mean(np.delete(vector, index)) for index in range(vector.size)]
    )
    max_delete_one_shift = float(np.max(np.abs(delete_one - mean)))
    inverse_variance = np.where(within > 0.0, 1.0 / within**2, 0.0)
    if np.sum(inverse_variance) > 0.0:
        weighted_mean = float(np.sum(inverse_variance * vector) / np.sum(inverse_variance))
        cochran_q = float(np.sum(inverse_variance * (vector - weighted_mean) ** 2))
        reduced_q = cochran_q / (vector.size - 1)
    else:
        weighted_mean = float("nan")
        cochran_q = float("nan")
        reduced_q = float("nan")
    return {
        "mean": mean,
        "within_replica_se": within_se,
        "between_replica_se": between_se,
        "hierarchical_se": hierarchical_se,
        "weighted_mean": weighted_mean,
        "cochran_q": cochran_q,
        "cochran_q_per_dof": reduced_q,
        "max_delete_one_shift": max_delete_one_shift,
        "max_delete_one_shift_over_hierarchical_se": (
            max_delete_one_shift / hierarchical_se
            if hierarchical_se > 0.0
            else float("nan")
        ),
    }


def apply_replica_event_classification(processed: pd.DataFrame) -> pd.DataFrame:
    """Replace the legacy probability cutoff with observable diagnostics.

    A nonzero estimate is eligible when it has enough events and positive
    blocks, controlled relative error, stationary block halves, stable
    double-block errors, and passes all physical/control gates.  Eligibility
    is then combined with an independent-replica consistency audit.  Rows
    with no events report exposure only; no nominal zero-event confidence
    bound is retained.
    """

    frame = processed.drop(
        columns=[column for column in ZERO_EVENT_BOUND_COLUMNS if column in processed],
        errors="ignore",
    ).copy()
    base_passes: list[bool] = []
    base_statuses: list[str] = []
    for row in frame.itertuples(index=False):
        reasons: list[str] = []
        if int(row.empty_events) == 0:
            reasons.append("raw_exposure_only_zero_events")
        else:
            if int(row.empty_events) < MIN_EMPTY_EVENTS:
                reasons.append("too_few_events")
            if int(row.positive_blocks) < MIN_POSITIVE_BLOCKS:
                reasons.append("too_few_positive_blocks")
            if float(row.relative_se_P_empty) > MAX_RELATIVE_SE:
                reasons.append("relative_error_too_large")
            if float(row.half_difference_z) > MAX_HALF_DIFFERENCE_Z:
                reasons.append("block_half_difference")
            ratio = float(row.double_block_se_ratio)
            if np.isfinite(ratio) and not (
                DOUBLE_BLOCK_SE_RATIO_INTERVAL[0]
                <= ratio
                <= DOUBLE_BLOCK_SE_RATIO_INTERVAL[1]
            ):
                reasons.append("double_block_se_instability")
        if not bool(row.crystal_structure_gate):
            reasons.append("crystal_structure_gate_failed")
        if not bool(row.acceptance_gate):
            reasons.append("production_acceptance_outside_gate")
        if not bool(row.exact_control_pass):
            reasons.append("exact_R0p5_control_failed")
        base_passes.append(not reasons)
        base_statuses.append("event_diagnostics_pass" if not reasons else ";".join(reasons))
    frame["event_diagnostics_pass"] = base_passes
    frame["event_diagnostics_status"] = base_statuses

    frame["replica_consistency_available"] = False
    frame["replica_cochran_q_per_dof"] = np.nan
    frame["replica_max_delete_one_shift_over_hierarchical_se"] = np.nan
    frame["replica_consistency_pass"] = False
    for (_, _), indices in frame.groupby(["N", "R_over_sigma"]).groups.items():
        group = frame.loc[indices]
        positive = group[
            (group["empty_events"] > 0)
            & np.isfinite(group["P_empty"])
            & np.isfinite(group["se_P_empty"])
            & (group["se_P_empty"] > 0.0)
        ]
        if len(positive) < 2:
            continue
        summary = hierarchical_summary(
            positive["P_empty"].to_numpy(), positive["se_P_empty"].to_numpy()
        )
        consistency_pass = bool(
            summary["cochran_q_per_dof"] <= 4.0
            and summary["max_delete_one_shift_over_hierarchical_se"] <= 1.5
        )
        frame.loc[indices, "replica_consistency_available"] = True
        frame.loc[indices, "replica_cochran_q_per_dof"] = summary[
            "cochran_q_per_dof"
        ]
        frame.loc[
            indices, "replica_max_delete_one_shift_over_hierarchical_se"
        ] = summary["max_delete_one_shift_over_hierarchical_se"]
        frame.loc[indices, "replica_consistency_pass"] = consistency_pass

    audited_pass = frame["event_diagnostics_pass"] & frame[
        "replica_consistency_pass"
    ]
    final_status: list[str] = []
    for event_status, event_pass, available, consistency_pass in zip(
        frame["event_diagnostics_status"],
        frame["event_diagnostics_pass"],
        frame["replica_consistency_available"],
        frame["replica_consistency_pass"],
    ):
        if not event_pass:
            final_status.append(str(event_status))
        elif not available:
            final_status.append("replica_consistency_unavailable")
        elif not consistency_pass:
            final_status.append("replica_consistency_failed")
        else:
            final_status.append("replica_audited_estimate")
    frame["statistical_reliable_direct"] = frame["event_diagnostics_pass"]
    frame["statistical_reliability_status"] = frame["event_diagnostics_status"]
    frame["reliable_direct"] = audited_pass
    frame["replica_audited_reliable"] = audited_pass
    frame["reliability_status"] = final_status
    return frame


def _structure_replica_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "beta_pressure_contact_shell",
        "beta_pressure_contact_shell_delta_0p01",
        "beta_pressure_contact_shell_delta_0p02",
        "beta_pressure_contact_shell_delta_0p03",
        "global_Q6",
        "max_FCC_Bragg",
        "lindemann_ratio",
    ]
    production = diagnostics[diagnostics["stage"] == "production"]
    rows: list[dict[str, Any]] = []
    for (n_particles, replica), group in production.groupby(["N", "replica"]):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            se, factor = conservative_block_se(values)
            rows.append(
                {
                    "analysis_level": "replica",
                    "N": int(n_particles),
                    "replica": int(replica),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "se": se,
                    "selected_block_factor": factor,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "n_blocks": int(values.size),
                }
            )
    replica_frame = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    for (n_particles, metric), group in replica_frame.groupby(["N", "metric"]):
        summary = hierarchical_summary(group["mean"].to_numpy(), group["se"].to_numpy())
        aggregate_rows.append(
            {
                "analysis_level": "aggregate",
                "N": int(n_particles),
                "replica": -1,
                "metric": metric,
                "mean": summary["mean"],
                "se": summary["hierarchical_se"],
                "selected_block_factor": -1,
                "minimum": float(group["minimum"].min()),
                "maximum": float(group["maximum"].max()),
                "n_blocks": int(group["n_blocks"].sum()),
                **summary,
            }
        )
    return pd.concat(
        [replica_frame, pd.DataFrame(aggregate_rows)], ignore_index=True
    )


def make_comparison(
    processed: pd.DataFrame,
    diagnostics: pd.DataFrame,
    baseline_processed: pd.DataFrame,
    baseline_diagnostics: pd.DataFrame,
    source: dict[str, Any],
) -> pd.DataFrame:
    """Summarize rare-event and structural agreement with the baseline."""

    rows: list[dict[str, Any]] = []
    target = processed[np.isclose(processed["R_over_sigma"], TARGET_RADIUS)]
    for n_particles, group in target.groupby("N"):
        probability = hierarchical_summary(
            group["P_empty"].to_numpy(), group["se_P_empty"].to_numpy()
        )
        beta_work_values = group["betaW_N"].to_numpy(dtype=float)
        beta_work_ses = group["se_betaW_N"].to_numpy(dtype=float)
        beta_work = hierarchical_summary(beta_work_values, beta_work_ses)
        baseline = baseline_processed[
            (baseline_processed["N"] == n_particles)
            & np.isclose(baseline_processed["R_over_sigma"], TARGET_RADIUS)
        ].iloc[0]
        for metric, summary, baseline_mean, baseline_se in (
            (
                "P_empty_R0p80",
                probability,
                float(baseline["P_empty"]),
                float(baseline["se_P_empty"]),
            ),
            (
                "betaW_R0p80",
                beta_work,
                float(baseline["betaW_N"]),
                float(baseline["se_betaW_N"]),
            ),
        ):
            comparison_se = math.hypot(summary["hierarchical_se"], baseline_se)
            rows.append(
                {
                    "N": int(n_particles),
                    "metric": metric,
                    "additional_mean": summary["mean"],
                    "hierarchical_se": summary["hierarchical_se"],
                    "within_replica_se": summary["within_replica_se"],
                    "between_replica_se": summary["between_replica_se"],
                    "cochran_q_per_dof": summary["cochran_q_per_dof"],
                    "max_delete_one_shift_over_hierarchical_se": summary[
                        "max_delete_one_shift_over_hierarchical_se"
                    ],
                    "baseline_mean": baseline_mean,
                    "baseline_se": baseline_se,
                    "additional_minus_baseline_z": (
                        (summary["mean"] - baseline_mean) / comparison_se
                        if comparison_se > 0.0
                        else np.nan
                    ),
                    "published_reference": np.nan,
                    "published_reference_se": np.nan,
                    "interpretation": "independent fixed-N reference replicas",
                }
            )

    structure = _structure_replica_summary(diagnostics)
    replica_structure = structure[structure["analysis_level"] == "replica"]
    baseline_production = baseline_diagnostics[
        baseline_diagnostics["stage"] == "production"
    ]
    for n_particles in sorted(processed["N"].unique()):
        for metric in ("global_Q6", "max_FCC_Bragg", "lindemann_ratio"):
            group = replica_structure[
                (replica_structure["N"] == n_particles)
                & (replica_structure["metric"] == metric)
            ]
            summary = hierarchical_summary(group["mean"].to_numpy(), group["se"].to_numpy())
            baseline_values = baseline_production[
                baseline_production["N"] == n_particles
            ][metric].to_numpy(dtype=float)
            baseline_se, _ = conservative_block_se(baseline_values)
            baseline_mean = float(np.mean(baseline_values))
            comparison_se = math.hypot(summary["hierarchical_se"], baseline_se)
            rows.append(
                {
                    "N": int(n_particles),
                    "metric": metric,
                    "additional_mean": summary["mean"],
                    "hierarchical_se": summary["hierarchical_se"],
                    "within_replica_se": summary["within_replica_se"],
                    "between_replica_se": summary["between_replica_se"],
                    "cochran_q_per_dof": summary["cochran_q_per_dof"],
                    "max_delete_one_shift_over_hierarchical_se": summary[
                        "max_delete_one_shift_over_hierarchical_se"
                    ],
                    "baseline_mean": baseline_mean,
                    "baseline_se": baseline_se,
                    "additional_minus_baseline_z": (
                        (summary["mean"] - baseline_mean) / comparison_se
                        if comparison_se > 0.0
                        else np.nan
                    ),
                    "published_reference": np.nan,
                    "published_reference_se": np.nan,
                    "interpretation": "crystal-retention distribution",
                }
            )

        pressure_group = replica_structure[
            (replica_structure["N"] == n_particles)
            & (replica_structure["metric"] == "beta_pressure_contact_shell")
        ]
        pressure = hierarchical_summary(
            pressure_group["mean"].to_numpy(), pressure_group["se"].to_numpy()
        )
        reference = float(source["pressure"])
        reference_se = float(source["se_pressure"])
        if "beta_pressure_contact_shell" in baseline_production.columns:
            baseline_pressure_values = baseline_production[
                baseline_production["N"] == n_particles
            ]["beta_pressure_contact_shell"].to_numpy(dtype=float)
            baseline_pressure_se, _ = conservative_block_se(
                baseline_pressure_values
            )
            baseline_pressure_mean = float(np.mean(baseline_pressure_values))
            baseline_pressure_comparison_se = math.hypot(
                pressure["hierarchical_se"], baseline_pressure_se
            )
            baseline_pressure_z = (
                (pressure["mean"] - baseline_pressure_mean)
                / baseline_pressure_comparison_se
                if baseline_pressure_comparison_se > 0.0
                else np.nan
            )
        else:
            baseline_pressure_mean = np.nan
            baseline_pressure_se = np.nan
            baseline_pressure_z = np.nan
        rows.append(
            {
                "N": int(n_particles),
                "metric": "beta_pressure_contact_shell_delta_0p02",
                "additional_mean": pressure["mean"],
                "hierarchical_se": pressure["hierarchical_se"],
                "within_replica_se": pressure["within_replica_se"],
                "between_replica_se": pressure["between_replica_se"],
                "cochran_q_per_dof": pressure["cochran_q_per_dof"],
                "max_delete_one_shift_over_hierarchical_se": pressure[
                    "max_delete_one_shift_over_hierarchical_se"
                ],
                "baseline_mean": baseline_pressure_mean,
                "baseline_se": baseline_pressure_se,
                "additional_minus_baseline_z": baseline_pressure_z,
                "published_reference": reference,
                "published_reference_se": reference_se,
                "finite_shell_minus_published": pressure["mean"] - reference,
                "additional_minus_published_z": np.nan,
                "interpretation": (
                    "finite-width contact-shell diagnostic; not an exact "
                    "zero-width pressure estimator, so no significance test "
                    "against the published pressure is assigned"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["N", "metric"]).reset_index(drop=True)


def make_figure(
    processed: pd.DataFrame,
    diagnostics: pd.DataFrame,
    baseline_processed: pd.DataFrame,
    baseline_diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    source: dict[str, Any],
    output: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), constrained_layout=True)
    colors = {108: "#4C78A8", 256: "#B14A68"}

    target = processed[np.isclose(processed["R_over_sigma"], TARGET_RADIUS)]
    axis = axes[0, 0]
    target_sizes = sorted(target["N"].unique())
    for x_index, n_particles in enumerate(target_sizes):
        group = target[target["N"] == n_particles].sort_values("replica")
        offsets = np.linspace(-0.16, 0.16, len(group))
        axis.errorbar(
            x_index + offsets,
            group["betaW_N"],
            yerr=group["se_betaW_N"],
            fmt="o",
            ms=4.0,
            capsize=2,
            color=colors[int(n_particles)],
            alpha=0.85,
            label=f"N={int(n_particles)} replicas",
        )
        baseline = baseline_processed[
            (baseline_processed["N"] == n_particles)
            & np.isclose(baseline_processed["R_over_sigma"], TARGET_RADIUS)
        ].iloc[0]
        axis.errorbar(
            [x_index + 0.29],
            [baseline["betaW_N"]],
            yerr=[baseline["se_betaW_N"]],
            fmt="s",
            mfc="white",
            mec="black",
            color="black",
            capsize=2,
            ms=4.2,
            label="original trajectory" if x_index == 0 else None,
        )
        row = comparison[
            (comparison["N"] == n_particles)
            & (comparison["metric"] == "betaW_R0p80")
        ].iloc[0]
        axis.errorbar(
            [x_index - 0.29],
            [row["additional_mean"]],
            yerr=[row["hierarchical_se"]],
            fmt="D",
            color="#222222",
            capsize=2.5,
            ms=4.2,
            label="replica aggregate" if x_index == 0 else None,
        )
    axis.set_xticks(
        np.arange(len(target_sizes)), [str(int(value)) for value in target_sizes]
    )
    axis.set_xlabel(r"$N$")
    axis.set_ylabel(r"$\beta W_N(R/\sigma=0.80)$")
    axis.set_title("Rare-event replica consistency")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="best", frameon=False)
    axis.grid(alpha=0.16, linewidth=0.5, axis="y")

    production = diagnostics[diagnostics["stage"] == "production"]
    axis = axes[0, 1]
    production_sizes = sorted(production["N"].unique())
    positions = list(range(1, len(production_sizes) + 1))
    pressure_sets = [
        production[production["N"] == n]["beta_pressure_contact_shell"].to_numpy()
        for n in production_sizes
    ]
    violin = axis.violinplot(
        pressure_sets,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showextrema=False,
    )
    for body, n_particles in zip(violin["bodies"], production_sizes):
        body.set_facecolor(colors[int(n_particles)])
        body.set_alpha(0.45)
    violin["cmeans"].set_color("#222222")
    axis.axhline(
        source["pressure"], color="#222222", linestyle="--", linewidth=1.0,
        label="published bulk pressure",
    )
    axis.set_xticks(positions, [str(int(value)) for value in production_sizes])
    axis.set_xlabel(r"$N$")
    axis.set_ylabel(r"contact-shell $\beta p\sigma^3$")
    axis.set_title(r"Finite-shell pressure proxy ($\Delta r/\sigma=0.02$)")
    axis.legend(frameon=False, loc="best")
    axis.grid(alpha=0.16, linewidth=0.5, axis="y")

    axis = axes[1, 0]
    for n_particles in sorted(production["N"].unique()):
        group = production[production["N"] == n_particles]
        axis.scatter(
            group["global_Q6"],
            group["max_FCC_Bragg"],
            s=8,
            alpha=0.28,
            color=colors[int(n_particles)],
            edgecolors="none",
            label=f"N={int(n_particles)} additional",
        )
        baseline = baseline_diagnostics[
            (baseline_diagnostics["N"] == n_particles)
            & (baseline_diagnostics["stage"] == "production")
        ]
        axis.scatter(
            baseline["global_Q6"],
            baseline["max_FCC_Bragg"],
            s=13,
            facecolors="none",
            edgecolors=colors[int(n_particles)],
            linewidths=0.55,
            alpha=0.55,
            label=f"N={int(n_particles)} original",
        )
    axis.set_xlabel(r"global $Q_6$")
    axis.set_ylabel("fcc Bragg amplitude")
    axis.set_title("Ordered-state distributions")
    axis.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8)
    axis.grid(alpha=0.14, linewidth=0.5)

    axis = axes[1, 1]
    lindemann_sets: list[np.ndarray] = []
    lindemann_labels: list[str] = []
    lindemann_colors: list[str] = []
    for n_particles in sorted(production["N"].unique()):
        lindemann_sets.append(
            production[production["N"] == n_particles]["lindemann_ratio"].to_numpy()
        )
        lindemann_labels.append(f"{int(n_particles)}\nnew")
        lindemann_colors.append(colors[int(n_particles)])
        lindemann_sets.append(
            baseline_diagnostics[
                (baseline_diagnostics["N"] == n_particles)
                & (baseline_diagnostics["stage"] == "production")
            ]["lindemann_ratio"].to_numpy()
        )
        lindemann_labels.append(f"{int(n_particles)}\norig.")
        lindemann_colors.append("#A7A7A7")
    box = axis.boxplot(
        lindemann_sets,
        tick_labels=lindemann_labels,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], lindemann_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    all_lindemann = np.concatenate(lindemann_sets)
    padding = 0.12 * (float(np.max(all_lindemann)) - float(np.min(all_lindemann)))
    axis.set_ylim(
        float(np.min(all_lindemann)) - padding,
        float(np.max(all_lindemann)) + padding,
    )
    axis.set_ylabel("Lindemann ratio")
    axis.set_title("Replica/original structural overlap")
    axis.grid(alpha=0.16, linewidth=0.5, axis="y")

    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.13,
            1.04,
            f"({label})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            clip_on=False,
        )
    fig.savefig(
        output,
        metadata={
            "Title": "Independent-replica audit of canonical FCC references",
            "Subject": "Fixed-N reference-basin checks; no phase normalization",
            "Creator": "scripts/solid_cavity_replica_campaign.py",
        },
    )
    plt.close(fig)


def write_results_note(
    processed: pd.DataFrame,
    comparison: pd.DataFrame,
    structure_summary: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Independent-replica FCC reference audit",
        "",
        (
            "These results are fixed-N, defect-free commensurate fcc "
            "reference-basin checks. They do not provide a fluid/solid common "
            "normalization, a phase switch, or a direct coexistence bias."
        ),
        "",
    ]
    target = processed[np.isclose(processed["R_over_sigma"], TARGET_RADIUS)]
    for n_particles in sorted(target["N"].unique()):
        group = target[target["N"] == n_particles].sort_values("replica")
        work_row = comparison[
            (comparison["N"] == n_particles)
            & (comparison["metric"] == "betaW_R0p80")
        ].iloc[0]
        pressure_row = comparison[
            (comparison["N"] == n_particles)
            & (comparison["metric"] == "beta_pressure_contact_shell_delta_0p02")
        ].iloc[0]
        lines.extend(
            [
                f"## N = {int(n_particles)}",
                "",
                (
                    "- Replica works at R/sigma=0.80: "
                    + ", ".join(
                        f"{row.betaW_N:.4f} +/- {row.se_betaW_N:.4f} "
                        f"({int(row.empty_events)} events)"
                        for row in group.itertuples(index=False)
                    )
                    + "."
                ),
                (
                    f"- Equal-replica aggregate: beta W = "
                    f"{work_row.additional_mean:.4f} +/- "
                    f"{work_row.hierarchical_se:.4f}; difference from the "
                    f"original trajectory = "
                    f"{work_row.additional_minus_baseline_z:+.2f} combined SE."
                ),
                (
                    f"- Contact-shell pressure diagnostic: beta p sigma^3 = "
                    f"{pressure_row.additional_mean:.3f} +/- "
                    f"{pressure_row.hierarchical_se:.3f}. This finite-shell "
                    "quantity is a structural diagnostic, not an exact "
                    "zero-width pressure extrapolation; its difference from "
                    "the published pressure is therefore not assigned a "
                    "statistical significance."
                ),
            ]
        )
        for metric, label in (
            ("global_Q6", "Q6"),
            ("max_FCC_Bragg", "fcc Bragg"),
            ("lindemann_ratio", "Lindemann"),
        ):
            row = comparison[
                (comparison["N"] == n_particles)
                & (comparison["metric"] == metric)
            ].iloc[0]
            lines.append(
                f"- {label}: {row.additional_mean:.5f} +/- "
                f"{row.hierarchical_se:.5f}; difference from the original "
                f"trajectory = {row.additional_minus_baseline_z:+.2f} combined SE."
            )
        lines.append("")
    aggregate = structure_summary[
        structure_summary["analysis_level"] == "aggregate"
    ]
    lines.extend(
        [
            "## Audit summary",
            "",
            (
                f"All {processed[['N', 'replica']].drop_duplicates().shape[0]} "
                "independent state trajectories retained the predeclared "
                "crystalline structure gates."
            ),
            (
                "The largest delete-one-replica shift among the reported "
                "structural aggregates was "
                f"{aggregate.max_delete_one_shift_over_hierarchical_se.max():.2f} "
                "hierarchical standard errors."
            ),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent replicas of the canonical FCC reference states."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--equil-sweeps", type=int, default=16_000)
    parser.add_argument("--tune-sweeps", type=int, default=4_000)
    parser.add_argument("--production-blocks", type=int, default=100)
    parser.add_argument("--sweeps-per-block", type=int, default=400)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--probes-per-sample", type=int, default=2_048)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace only the additional-replica artifacts."
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    sizes = tuple(int(value) for value in args.sizes)
    if args.replicas < (2 if args.quick else 4):
        raise ValueError(
            "The quick audit requires at least two replicas; the production "
            "campaign requires at least four."
        )
    for n_particles in sizes:
        cells = int(round((n_particles / 4.0) ** (1.0 / 3.0)))
        if 4 * cells**3 != n_particles:
            raise ValueError(f"N={n_particles} is not commensurate FCC.")
    for relative in (RAW_BLOCKS_PATH, RAW_STRUCTURE_PATH, RAW_STATES_PATH):
        if (root / relative).exists() and not args.overwrite:
            raise FileExistsError(
                f"{relative} already exists; pass --overwrite to replace only campaign outputs."
            )
    if args.quick:
        settings = SolidRunSettings(
            equil_sweeps=400,
            tune_sweeps=200,
            tune_chunk_sweeps=100,
            audit_chunk_sweeps=200,
            production_blocks=20,
            sweeps_per_block=100,
            sample_stride=5,
            probes_per_sample=2_048,
        )
    else:
        settings = SolidRunSettings(
            equil_sweeps=args.equil_sweeps,
            tune_sweeps=args.tune_sweeps,
            tune_chunk_sweeps=400,
            audit_chunk_sweeps=2_000,
            production_blocks=args.production_blocks,
            sweeps_per_block=args.sweeps_per_block,
            sample_stride=args.sample_stride,
            probes_per_sample=args.probes_per_sample,
        )
    if settings.sweeps_per_block % settings.sample_stride:
        raise ValueError("sweeps_per_block must be divisible by sample_stride.")
    source = load_solid_coexistence_density(
        root / "data_raw" / "fernandez2012_coexistence.csv"
    )
    baseline_processed_path = root / "data_processed" / "solid_cavity_mc.csv"
    baseline_diagnostics_path = root / "data_raw" / "solid_cavity_mc_structure.csv"
    baseline_processed = pd.read_csv(baseline_processed_path)
    baseline_diagnostics = pd.read_csv(baseline_diagnostics_path)
    for directory in ("data_processed", "data_raw", "figures", "generated"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    all_processed: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    final_states: dict[str, np.ndarray] = {}
    seed_records: list[dict[str, int]] = []
    start = time.perf_counter()
    for replica in range(args.replicas):
        base_seed = replica_base_seed(args.seed, replica)
        for size_index, n_particles in enumerate(sizes):
            print(
                f"replica {replica + 1}/{args.replicas}, N={n_particles}, "
                f"base seed={base_seed}",
                flush=True,
            )
            processed_rows, raw_rows, diagnostics_rows, final_positions = run_state(
                n_particles,
                source["eta"],
                source,
                size_index,
                settings,
                base_seed,
            )
            equil_seed = int(processed_rows[0]["equil_seed"])
            production_seed = int(processed_rows[0]["seed"])
            run_id = (
                f"{ANALYSIS_VERSION}_rep{replica}_N{n_particles}_"
                f"seed{production_seed}"
            )
            seed_records.append(
                {
                    "replica": replica,
                    "N": n_particles,
                    "base_seed": base_seed,
                    "equil_seed": equil_seed,
                    "production_seed": production_seed,
                }
            )
            for collection in (processed_rows, raw_rows, diagnostics_rows):
                for row in collection:
                    row["source_type"] = "mc_additional_independent_replica"
                    row["run_id"] = run_id
                    row["replica"] = replica
                    row["replica_base_seed"] = base_seed
            for row in processed_rows:
                row["analysis_version"] = ANALYSIS_VERSION
                row["raw_data_path"] = RAW_BLOCKS_PATH
            all_processed.extend(processed_rows)
            all_raw.extend(raw_rows)
            all_diagnostics.extend(diagnostics_rows)
            final_states[f"positions_rep{replica}_N{n_particles}"] = final_positions
            final_states[f"box_length_rep{replica}_N{n_particles}"] = np.asarray(
                processed_rows[0]["L"]
            )
            target = next(
                row for row in processed_rows if np.isclose(row["R_over_sigma"], TARGET_RADIUS)
            )
            print(
                f"  R=0.80 betaW={target['betaW_N']:.4f} +/- "
                f"{target['se_betaW_N']:.4f}, events={target['empty_events']}, "
                f"crystal={target['crystal_structure_gate']}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    processed = apply_replica_event_classification(
        pd.DataFrame(all_processed).sort_values(["N", "replica", "R_over_sigma"])
    )
    raw = pd.DataFrame(all_raw).sort_values(["N", "replica", "block"])
    diagnostics = pd.DataFrame(all_diagnostics).sort_values(
        ["N", "replica", "stage", "cumulative_sweeps"]
    )
    commit = git_commit(root)
    script_hash = sha256(Path(__file__).resolve())
    for frame in (processed, raw, diagnostics):
        frame["campaign_code_commit"] = commit
        frame["campaign_script_sha256"] = script_hash
    structure_summary = _structure_replica_summary(diagnostics)
    comparison = make_comparison(
        processed,
        diagnostics,
        baseline_processed,
        baseline_diagnostics,
        source,
    )
    processed.to_csv(root / PROCESSED_PATH, index=False)
    raw.to_csv(root / RAW_BLOCKS_PATH, index=False)
    diagnostics.to_csv(root / RAW_STRUCTURE_PATH, index=False)
    structure_summary.to_csv(root / STRUCTURE_SUMMARY_PATH, index=False)
    comparison.to_csv(root / COMPARISON_PATH, index=False)
    np.savez_compressed(root / RAW_STATES_PATH, **final_states)
    make_figure(
        processed,
        diagnostics,
        baseline_processed,
        baseline_diagnostics,
        comparison,
        source,
        root / FIGURE_PATH,
    )
    write_results_note(processed, comparison, structure_summary, root / RESULTS_PATH)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "script": "scripts/solid_cavity_replica_campaign.py",
        "script_sha256": script_hash,
        "solid_cavity_mc_script_sha256": sha256(root / "scripts" / "solid_cavity_mc.py"),
        "code_commit_at_generation": commit,
        "reproduce_command": (
            "python3 scripts/solid_cavity_replica_campaign.py "
            f"--seed {args.seed} --sizes {' '.join(map(str, sizes))} "
            f"--replicas {args.replicas} --equil-sweeps {settings.equil_sweeps} "
            f"--tune-sweeps {settings.tune_sweeps} "
            f"--production-blocks {settings.production_blocks} "
            f"--sweeps-per-block {settings.sweeps_per_block} "
            f"--sample-stride {settings.sample_stride} "
            f"--probes-per-sample {settings.probes_per_sample} --overwrite"
        ),
        "scope": {
            "ensemble": "canonical NVT",
            "phase": "defect-free commensurate FCC reference basin",
            "purpose": "independent-replica consistency and rare-event audit",
            "not_direct_coexistence": (
                "No phase switch, grand-canonical relative weight, chemical-"
                "potential reweighting, or common fluid/solid normalization."
            ),
        },
        "base_seed": args.seed,
        "replica_seed_stride": REPLICA_SEED_STRIDE,
        "seed_records": seed_records,
        "sizes": list(sizes),
        "replicas_per_size": args.replicas,
        "settings": asdict(settings),
        "target_rare_event_radius_over_sigma": TARGET_RADIUS,
        "uncertainty": {
            "within_trajectory": (
                "largest adjacent-batch SE over 1-, 2-, 5-, and 10-block aggregations"
            ),
            "aggregate": (
                "equal-replica mean with max(within-replica propagated SE, "
                "between-replica SE)"
            ),
            "robustness": "Cochran Q and delete-one-replica shifts",
        },
        "classification": {
            "inputs": [
                "empty-event count",
                "positive-block count",
                "relative probability standard error",
                "first-half versus second-half stationarity",
                "double-block standard-error stability",
                "crystal/acceptance/exact-control gates",
                "independent-replica Cochran Q and delete-one stability",
            ],
            "probability_floor_used": False,
            "zero_event_reporting": (
                "raw probe exposure only; no rule-of-three or nominal 95% "
                "zero-event confidence bound"
            ),
            "replica_consistency_thresholds": {
                "maximum_cochran_q_per_dof": 4.0,
                "maximum_delete_one_shift_over_hierarchical_se": 1.5,
            },
        },
        "pressure_diagnostic": {
            "definition": (
                "finite-shell average g over 1 <= r/sigma < 1 + Delta, "
                "inserted into beta p = rho[1 + 4 eta g(1+)]"
            ),
            "central_shell_width_over_sigma": 0.02,
            "sensitivity_widths_over_sigma": list(CONTACT_SHELL_SENSITIVITY_WIDTHS),
            "limitation": (
                "diagnostic only; finite-shell result is not represented as "
                "an exact zero-width pressure extrapolation"
            ),
        },
        "density_source": source,
        "baseline_inputs": {
            "processed": str(baseline_processed_path.relative_to(root)),
            "processed_sha256": sha256(baseline_processed_path),
            "structure": str(baseline_diagnostics_path.relative_to(root)),
            "structure_sha256": sha256(baseline_diagnostics_path),
        },
        "quality_summary": {
            "independent_state_trajectories": int(
                processed[["N", "replica"]].drop_duplicates().shape[0]
            ),
            "all_crystal_gates_pass": bool(processed["crystal_structure_gate"].all()),
            "all_exact_R0p5_controls_pass": bool(
                processed[np.isclose(processed["R_over_sigma"], 0.50)][
                    "exact_control_pass"
                ].all()
            ),
            "minimum_R0p80_events_per_replica": int(
                processed[np.isclose(processed["R_over_sigma"], TARGET_RADIUS)][
                    "empty_events"
                ].min()
            ),
            "all_R0p80_replica_audited": bool(
                processed[np.isclose(processed["R_over_sigma"], TARGET_RADIUS)][
                    "replica_audited_reliable"
                ].all()
            ),
            "maximum_abs_R0p80_baseline_difference_z": float(
                comparison[comparison["metric"].isin(["P_empty_R0p80", "betaW_R0p80"])][
                    "additional_minus_baseline_z"
                ].abs().max()
            ),
            "maximum_abs_structural_baseline_difference_z": float(
                comparison[comparison["metric"].isin(["global_Q6", "max_FCC_Bragg", "lindemann_ratio"])][
                    "additional_minus_baseline_z"
                ].abs().max()
            ),
        },
        "artifacts": {
            "processed_replicas": PROCESSED_PATH,
            "raw_blocks": RAW_BLOCKS_PATH,
            "raw_structure": RAW_STRUCTURE_PATH,
            "final_states": RAW_STATES_PATH,
            "structure_summary": STRUCTURE_SUMMARY_PATH,
            "comparison": COMPARISON_PATH,
            "figure": FIGURE_PATH,
            "results_note": RESULTS_PATH,
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
    (root / METADATA_PATH).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"completed {args.replicas} replicas at {len(sizes)} sizes in {elapsed:.1f} s")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
