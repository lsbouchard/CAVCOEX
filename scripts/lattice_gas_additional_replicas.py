#!/usr/bin/env python3
"""Generate additional Wolff replicas without touching the baseline archive."""

from __future__ import annotations

import argparse
import hashlib
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


def output_paths(output: Path, root: Path = ROOT) -> tuple[Path, Path, Path]:
    """Keep canonical destinations, or put all custom-output sidecars together."""
    output = output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("--output must have the .npz extension")
    if output == (root / "data_raw" / "lattice_gas_additional_replicas.npz").resolve():
        return (
            output,
            root / "data_raw" / "lattice_gas_additional_replicas_diagnostics.csv",
            root / "generated" / "lattice_gas_additional_replicas_metadata.json",
        )
    return (
        output,
        output.with_name(output.stem + "_diagnostics.csv"),
        output.with_name(output.stem + "_metadata.json"),
    )


def displayed_path(path: Path, root: Path = ROOT) -> str:
    """Use a project-relative path when possible, otherwise an absolute path."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


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
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=".npz archive; custom diagnostics and metadata are written beside it",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing archive and its diagnostics/metadata sidecars",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path, diagnostic_path, metadata_path = output_paths(args.output, ROOT)
    existing = [path for path in (archive_path, diagnostic_path, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing additional-replica products "
            "without --overwrite: " + ", ".join(map(str, existing))
        )
    if args.replica_start < 0 or args.replica_stop <= args.replica_start:
        raise ValueError("replica-start must be nonnegative and replica-stop must exceed it")
    if len(set(args.sizes)) != len(args.sizes) or any(length < 2 for length in args.sizes):
        raise ValueError("sizes must be distinct integers of at least 2")
    if args.equil_steps < 0 or args.production_steps <= 0 or args.workers < 1:
        raise ValueError("equil-steps must be nonnegative; production-steps and workers must be positive")
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
    for path in (archive_path, diagnostic_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive_path, **raw)
    pd.DataFrame(diagnostics).to_csv(diagnostic_path, index=False)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/lattice_gas_additional_replicas.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {
            "scripts/lattice_gas_phase_switch.py": hashlib.sha256(
                Path(__file__).with_name("lattice_gas_phase_switch.py").read_bytes()
            ).hexdigest(),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "output": displayed_path(archive_path, ROOT),
        "output_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
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
        "diagnostics": displayed_path(diagnostic_path, ROOT),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(results)} additional replicas to "
        f"{displayed_path(archive_path, ROOT)}"
    )


if __name__ == "__main__":
    main()
