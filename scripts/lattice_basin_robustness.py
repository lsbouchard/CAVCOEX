#!/usr/bin/env python3
"""Observed narrow-interfacial-basin contribution in lattice trajectories.

This is deliberately a finite-sample diagnostic, not an asymptotic fit.  The
interfacial basin is declared by ``|M|/V < cutoff`` before inspecting its
counts.  Zero observations are reported as zero observed contribution with an
explicit non-estimable conditional flag; no independent-trial confidence bound
is attached to the correlated Markov-chain samples.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .lattice_analysis_common import (
        available_sizes,
        load_trajectory_archives,
        replicas_for_size,
    )
except ImportError:
    from lattice_analysis_common import (
        available_sizes,
        load_trajectory_archives,
        replicas_for_size,
    )


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RAW = ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz"
ADDITIONAL_RAW = ROOT / "data_raw" / "lattice_gas_additional_replicas.npz"
DEFAULT_RAW = [BASELINE_RAW] + ([ADDITIONAL_RAW] if ADDITIONAL_RAW.exists() else [])
DEFAULT_CUTOFFS = (0.05, 0.10, 0.15)


def analyze(raw_paths: list[Path], cutoffs: list[float]) -> tuple[pd.DataFrame, dict]:
    trajectories = load_trajectory_archives(raw_paths)
    rows: list[dict] = []
    for length in available_sizes(trajectories):
        volume = length * length
        replicas = replicas_for_size(trajectories, length)
        magnetization = np.concatenate(
            [trajectories[(length, replica)]["magnetization"] for replica in replicas]
        ).astype(float)
        bond_sum = np.concatenate(
            [trajectories[(length, replica)]["bond_sum"] for replica in replicas]
        ).astype(float)
        number = 0.5 * (volume + magnetization)
        cavity = (2.0 * volume - 4.0 * magnetization + bond_sum) / (8.0 * volume)
        total_count = len(magnetization)
        total_cavity = float(np.sum(cavity))
        mean_n = float(np.mean(number))
        mean_n_cavity = float(np.sum(cavity * number) / total_cavity)
        total_response = mean_n_cavity - mean_n

        for cutoff in cutoffs:
            interfacial = np.abs(magnetization) / volume < cutoff
            count = int(np.count_nonzero(interfacial))
            cavity_sum = float(np.sum(cavity[interfacial]))
            q0 = count / total_count
            q_cavity = cavity_sum / total_cavity
            estimable = count > 0 and cavity_sum > 0.0
            if estimable:
                mean_n_interface = float(np.mean(number[interfacial]))
                mean_n_cavity_interface = float(
                    np.sum(cavity[interfacial] * number[interfacial]) / cavity_sum
                )
                intraphase = q_cavity * (
                    mean_n_cavity_interface - mean_n_interface
                )
                reallocation = (q_cavity - q0) * mean_n_interface
            else:
                mean_n_interface = np.nan
                mean_n_cavity_interface = np.nan
                intraphase = 0.0
                reallocation = 0.0
            contribution = intraphase + reallocation
            rows.append(
                {
                    "L": length,
                    "V": volume,
                    "magnetization_fraction_cutoff": cutoff,
                    "replicas": len(replicas),
                    "raw_configurations": total_count,
                    "interfacial_observations": count,
                    "observed_q0_interface": q0,
                    "observed_qC_interface": q_cavity,
                    "one_observation_fraction": 1.0 / total_count,
                    "conditional_interface_estimable": estimable,
                    "meanN0_interface": mean_n_interface,
                    "meanNC_interface": mean_n_cavity_interface,
                    "observed_intraphase_contribution": intraphase,
                    "observed_reallocation_contribution": reallocation,
                    "observed_total_interface_contribution": contribution,
                    "full_unpaired_deltaN": total_response,
                    "absolute_interface_contribution_over_V": (
                        abs(contribution) / volume
                    ),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["magnetization_fraction_cutoff", "L"]
    )
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_basin_robustness.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "raw_paths": [str(path.resolve().relative_to(ROOT)) for path in raw_paths],
        "predeclared_interfacial_basin": "|M|/V < cutoff",
        "cutoffs": cutoffs,
        "interpretation": (
            "finite-sample sensitivity diagnostic only; zeros are observed zeros, "
            "not independent-trial confidence bounds or an asymptotic law"
        ),
    }
    return frame, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", default=DEFAULT_RAW)
    parser.add_argument(
        "--cutoffs", type=float, nargs="+", default=list(DEFAULT_CUTOFFS)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, metadata = analyze(args.raw, args.cutoffs)
    frame.to_csv(
        ROOT / "data_processed" / "lattice_basin_robustness.csv", index=False
    )
    (ROOT / "generated" / "lattice_basin_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for _, row in frame.iterrows():
        print(
            f"L={int(row['L'])}, cutoff={row['magnetization_fraction_cutoff']:.2f}: "
            f"{int(row['interfacial_observations'])}/"
            f"{int(row['raw_configurations'])} observed"
        )


if __name__ == "__main__":
    main()
