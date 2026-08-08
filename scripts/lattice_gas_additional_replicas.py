#!/usr/bin/env python3
"""Generate additional Wolff replicas without touching the baseline archive."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .lattice_gas_phase_switch import (
        K_ISING,
        _wolff_series,
        integrated_autocorrelation_time,
        orient_symmetry_pairs,
    )
except ImportError:
    from lattice_gas_phase_switch import (
        K_ISING,
        _wolff_series,
        integrated_autocorrelation_time,
        orient_symmetry_pairs,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_raw" / "lattice_gas_additional_replicas.npz"


def _run_one(task: tuple[int, int, int, int, int]) -> tuple:
    length, replica, seed, equilibration_steps, production_steps = task
    initial_sign = 1 if replica % 2 == 0 else -1
    arrays = _wolff_series(
        length,
        K_ISING,
        seed,
        equilibration_steps,
        production_steps,
        initial_sign,
    )
    return length, replica, seed, initial_sign, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=25_072_026)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[16, 24, 32, 48, 64]
    )
    parser.add_argument("--replica-start", type=int, default=4)
    parser.add_argument("--replica-stop", type=int, default=12)
    parser.add_argument("--equil-steps", type=int, default=2_000)
    parser.add_argument("--production-steps", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing additional-replica archive deterministically",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing additional-replica archive "
            f"without --overwrite: {args.output}"
        )
    if args.replica_stop <= args.replica_start:
        raise ValueError("replica-stop must exceed replica-start")
    tasks = []
    for length in args.sizes:
        for replica in range(args.replica_start, args.replica_stop):
            seed = int(args.seed + 1009 * length + 104_729 * replica)
            tasks.append(
                (
                    length,
                    replica,
                    seed,
                    args.equil_steps,
                    args.production_steps,
                )
            )

    # Load the compiled kernel once in the parent before spawning workers.
    _wolff_series(4, K_ISING, 1, 2, 2, 1)
    if args.workers == 1:
        results = [_run_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_run_one, tasks))

    raw: dict[str, np.ndarray] = {}
    diagnostics: list[dict] = []
    for length, replica, seed, initial_sign, arrays in sorted(results):
        magnetization, bond_sum, site_zero, site_right, cluster_size = arrays
        prefix = f"L{length}_replica{replica}"
        raw[f"{prefix}_magnetization"] = magnetization
        raw[f"{prefix}_bond_sum"] = bond_sum
        raw[f"{prefix}_site_zero"] = site_zero
        raw[f"{prefix}_site_right"] = site_right
        raw[f"{prefix}_cluster_size"] = cluster_size
        oriented_m, _ = orient_symmetry_pairs(magnetization, site_zero)
        volume = length * length
        cavity_liquid = (
            2.0 * volume - 4.0 * oriented_m + bond_sum
        ) / (8.0 * volume)
        signs = np.sign(magnetization)
        nonzero = signs[signs != 0]
        diagnostics.append(
            {
                "L": length,
                "replica": replica,
                "seed": seed,
                "initial_sign": initial_sign,
                "magnetization_sector_reversals": int(
                    np.count_nonzero(nonzero[1:] != nonzero[:-1])
                ),
                "mean_cluster_fraction": float(np.mean(cluster_size) / volume),
                "mean_abs_magnetization": float(
                    np.mean(np.abs(magnetization)) / volume
                ),
                "tau_int_abs_magnetization": integrated_autocorrelation_time(
                    np.abs(magnetization) / volume
                ),
                "tau_int_pair_empty_liquid": integrated_autocorrelation_time(
                    cavity_liquid
                ),
            }
        )
    np.savez_compressed(args.output, **raw)
    diagnostic_path = (
        ROOT / "data_raw" / "lattice_gas_additional_replicas_diagnostics.csv"
    )
    pd.DataFrame(diagnostics).to_csv(diagnostic_path, index=False)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_gas_additional_replicas.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "output": str(args.output.resolve().relative_to(ROOT)),
        "baseline_archive_modified": False,
        "settings": {
            "base_seed": args.seed,
            "sizes": args.sizes,
            "replica_start": args.replica_start,
            "replica_stop_exclusive": args.replica_stop,
            "equilibration_steps": args.equil_steps,
            "production_steps": args.production_steps,
            "workers": args.workers,
            "overwrite_existing_archive": args.overwrite,
        },
        "seed_formula": "base_seed + 1009*L + 104729*replica",
        "diagnostics": str(diagnostic_path.relative_to(ROOT)),
    }
    (
        ROOT / "generated" / "lattice_gas_additional_replicas_metadata.json"
    ).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(results)} additional replicas to "
        f"{args.output.resolve().relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
