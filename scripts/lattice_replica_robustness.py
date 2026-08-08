#!/usr/bin/env python3
"""Hierarchical paired-versus-unpaired lattice-gas robustness analysis."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

try:
    from .lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        load_trajectory_archives,
    )
    from .lattice_gas_phase_switch import observables_from_contributions
except ImportError:
    from lattice_analysis_common import (
        available_sizes,
        contribution_blocks,
        load_trajectory_archives,
    )
    from lattice_gas_phase_switch import observables_from_contributions


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RAW = ROOT / "data_raw" / "lattice_gas_phase_switch_trajectories.npz"
ADDITIONAL_RAW = ROOT / "data_raw" / "lattice_gas_additional_replicas.npz"
DEFAULT_RAW = [BASELINE_RAW] + ([ADDITIONAL_RAW] if ADDITIONAL_RAW.exists() else [])
DEFAULT_SEED = 26_080_603
OBSERVABLES = (
    "delta_betaW",
    "deltaN_phase_reweighting",
    "deltaN_intraphase",
    "deltaN",
    "q0_liquid",
    "qC_liquid",
    "delta_rho",
    "P_fixed_empty_liquid",
    "P_fixed_empty_gas",
    "P_spatial_empty_liquid",
    "P_spatial_empty_gas",
    "P_fixed_minus_spatial_liquid",
    "P_fixed_minus_spatial_gas",
    "delta_betaW_fixed",
    "delta_betaW_fixed_minus_spatial",
)


def scalar_observables(contributions: np.ndarray, volume: int) -> dict[str, float]:
    observed = observables_from_contributions(contributions, volume)
    fixed_delta_work = float(
        np.log(
            observed["P_fixed_empty_gas"][0]
            / observed["P_fixed_empty_liquid"][0]
        )
    )
    observed["delta_betaW_fixed"] = np.asarray([fixed_delta_work])
    observed["P_fixed_minus_spatial_liquid"] = (
        observed["P_fixed_empty_liquid"] - observed["P_spatial_empty_liquid"]
    )
    observed["P_fixed_minus_spatial_gas"] = (
        observed["P_fixed_empty_gas"] - observed["P_spatial_empty_gas"]
    )
    observed["delta_betaW_fixed_minus_spatial"] = np.asarray(
        [fixed_delta_work - observed["delta_betaW"][0]]
    )
    return {name: float(observed[name][0]) for name in OBSERVABLES}


def analyze(
    raw_paths: list[Path],
    block_count: int,
    bootstrap_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    trajectories = load_trajectory_archives(raw_paths)
    rng = np.random.default_rng(seed)
    result_rows: list[dict] = []
    difference_rows: list[dict] = []
    replica_rows: list[dict] = []
    delete_one_rows: list[dict] = []

    for length in available_sizes(trajectories):
        volume = length * length
        paired, replicas = contribution_blocks(
            trajectories, length, np.array([0.0]), block_count, "paired"
        )
        unpaired, unpaired_replicas = contribution_blocks(
            trajectories, length, np.array([0.0]), block_count, "unpaired"
        )
        if replicas != unpaired_replicas:
            raise AssertionError("paired and unpaired replica sets differ")
        arrays = {"paired": paired, "unpaired": unpaired}
        central = {
            mode: scalar_observables(np.sum(blocks, axis=(0, 1)), volume)
            for mode, blocks in arrays.items()
        }
        bootstrap = {
            mode: {
                observable: np.empty(bootstrap_draws, dtype=float)
                for observable in OBSERVABLES
            }
            for mode in arrays
        }
        difference_bootstrap = {
            observable: np.empty(bootstrap_draws, dtype=float)
            for observable in OBSERVABLES
        }

        replica_count = len(replicas)
        for draw in range(bootstrap_draws):
            selected_replicas = rng.integers(0, replica_count, replica_count)
            totals = {
                mode: np.zeros(blocks.shape[2:], dtype=float)
                for mode, blocks in arrays.items()
            }
            for selected in selected_replicas:
                selected_blocks = rng.integers(0, block_count, block_count)
                for mode, blocks in arrays.items():
                    totals[mode] += np.sum(
                        blocks[selected, selected_blocks], axis=0
                    )
            draw_values = {
                mode: scalar_observables(total, volume)
                for mode, total in totals.items()
            }
            for observable in OBSERVABLES:
                for mode in arrays:
                    bootstrap[mode][observable][draw] = draw_values[mode][
                        observable
                    ]
                difference_bootstrap[observable][draw] = (
                    draw_values["paired"][observable]
                    - draw_values["unpaired"][observable]
                )

        t_multiplier = float(t.ppf(0.975, df=replica_count - 1))
        for mode in arrays:
            for observable in OBSERVABLES:
                standard_error = float(
                    np.std(bootstrap[mode][observable], ddof=1)
                )
                value = central[mode][observable]
                result_rows.append(
                    {
                        "L": length,
                        "V": volume,
                        "mode": mode,
                        "observable": observable,
                        "value": value,
                        "standard_error": standard_error,
                        "replicas": replica_count,
                        "blocks_per_replica": block_count,
                        "bootstrap_draws": bootstrap_draws,
                        "small_sample_t_multiplier_95": t_multiplier,
                        "small_sample_ci95_low": value
                        - t_multiplier * standard_error,
                        "small_sample_ci95_high": value
                        + t_multiplier * standard_error,
                    }
                )
        for observable in OBSERVABLES:
            difference = (
                central["paired"][observable]
                - central["unpaired"][observable]
            )
            difference_error = float(
                np.std(difference_bootstrap[observable], ddof=1)
            )
            unpaired_error = float(
                np.std(bootstrap["unpaired"][observable], ddof=1)
            )
            difference_rows.append(
                {
                    "L": length,
                    "V": volume,
                    "observable": observable,
                    "paired_minus_unpaired": difference,
                    "standard_error_of_difference": difference_error,
                    "unpaired_standard_error": unpaired_error,
                    "absolute_difference_over_unpaired_se": (
                        abs(difference) / unpaired_error
                        if unpaired_error > 0.0
                        else np.nan
                    ),
                }
            )

        for mode, blocks in arrays.items():
            per_replica = [
                scalar_observables(np.sum(blocks[index], axis=0), volume)
                for index in range(replica_count)
            ]
            for replica_index, replica in enumerate(replicas):
                values = per_replica[replica_index]
                for observable, value in values.items():
                    replica_rows.append(
                        {
                            "L": length,
                            "V": volume,
                            "mode": mode,
                            "replica": replica,
                            "observable": observable,
                            "value": value,
                        }
                    )
            for omitted_index, omitted_replica in enumerate(replicas):
                keep = np.arange(replica_count) != omitted_index
                values = scalar_observables(
                    np.sum(blocks[keep], axis=(0, 1)), volume
                )
                for observable, value in values.items():
                    replica_estimates = np.asarray(
                        [entry[observable] for entry in per_replica], dtype=float
                    )
                    all_replica_se = float(
                        np.std(replica_estimates, ddof=1)
                        / np.sqrt(replica_count)
                    )
                    retained_estimates = replica_estimates[keep]
                    deleted_replica_se = float(
                        np.std(retained_estimates, ddof=1)
                        / np.sqrt(replica_count - 1)
                    )
                    delete_one_rows.append(
                        {
                            "L": length,
                            "V": volume,
                            "mode": mode,
                            "omitted_replica": omitted_replica,
                            "observable": observable,
                            "value": value,
                            "shift_from_all_replicas": (
                                value - central[mode][observable]
                            ),
                            "all_replica_level_standard_error": all_replica_se,
                            "delete_one_replica_level_standard_error": (
                                deleted_replica_se
                            ),
                            "delete_one_se_ratio_to_all": (
                                deleted_replica_se / all_replica_se
                                if all_replica_se > 0.0
                                else np.nan
                            ),
                        }
                    )

    results = pd.DataFrame(result_rows)
    differences = pd.DataFrame(difference_rows)
    replica_values = pd.DataFrame(replica_rows)
    delete_one = pd.DataFrame(delete_one_rows)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_replica_robustness.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "raw_paths": [str(path.resolve().relative_to(ROOT)) for path in raw_paths],
        "seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "blocks_per_replica": block_count,
        "paired_definition": (
            "each zero-field configuration averaged with its global spin inverse"
        ),
        "unpaired_definition": "raw zero-field Wolff trajectories",
        "resampling": (
            "replicas sampled with replacement; common block indices sampled "
            "within each selected replica for the paired and unpaired estimators"
        ),
        "interval_note": (
            "standard errors are hierarchical-bootstrap estimates; the reported "
            "95% interval additionally uses Student-t multiplier with R-1 degrees "
            "of freedom to expose finite-replica small-sample cost"
        ),
    }
    return results, differences, replica_values, delete_one, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", default=DEFAULT_RAW)
    parser.add_argument("--blocks-per-replica", type=int, default=40)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, differences, replica_values, delete_one, metadata = analyze(
        args.raw,
        args.blocks_per_replica,
        args.bootstrap_draws,
        args.seed,
    )
    results.to_csv(
        ROOT
        / "data_processed"
        / "lattice_paired_unpaired_hierarchical.csv",
        index=False,
    )
    differences.to_csv(
        ROOT / "data_processed" / "lattice_paired_unpaired_differences.csv",
        index=False,
    )
    replica_values.to_csv(
        ROOT / "data_processed" / "lattice_replica_level_observables.csv",
        index=False,
    )
    delete_one.to_csv(
        ROOT / "data_processed" / "lattice_replica_delete_one.csv", index=False
    )
    (ROOT / "generated" / "lattice_replica_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    largest = int(results["L"].max())
    selected = results[
        (results["L"] == largest)
        & results["observable"].isin(
            ("delta_betaW", "deltaN_phase_reweighting", "q0_liquid")
        )
    ]
    for _, row in selected.iterrows():
        print(
            f"L={largest} {row['mode']} {row['observable']}: "
            f"{row['value']:.8g} +/- {row['standard_error']:.3g}"
        )


if __name__ == "__main__":
    main()
