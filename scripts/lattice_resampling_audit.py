#!/usr/bin/env python3
"""Check block-length sensitivity and replica dispersion without new MC."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from lattice_analysis_common import (
    load_trajectory_archives, contribution_blocks, stratified_block_bootstrap_totals,
)
from lattice_gas_phase_switch import observables_from_contributions
from lattice_pulse_scaling import continuous_pulse_observables

ROOT = Path(__file__).resolve().parents[1]
H = np.linspace(-6.0, 9.0, 151)
NAMES = ("delta_betaW", "delta_rho", "deltaN_liquid", "deltaN_gas",
         "peak_center_h", "peak_magnitude", "fwhm_h")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summary(total, volume):
    observed = observables_from_contributions(total, volume)
    zero = int(np.argmin(np.abs(H)))
    pulse = continuous_pulse_observables(H, observed["deltaN_phase_reweighting"]/volume)
    return np.array([observed[k][zero] for k in NAMES[:4]]
                    + [pulse[k] for k in NAMES[4:]])


def main():
    inputs = [
        ROOT/"data_raw/lattice_gas_phase_switch_trajectories.npz",
        ROOT/"data_raw/lattice_gas_additional_replicas.npz"]
    raw = load_trajectory_archives(inputs)
    rows = []
    for length in (16, 24, 32, 48, 64):
        blocks, _ = contribution_blocks(raw, length, H, 40)
        replica_values = np.array([summary(r.sum(axis=0), length**2) for r in blocks])
        replica_se = replica_values.std(axis=0, ddof=1)/np.sqrt(len(blocks))
        for factor in (1, 2, 4):
            grouped = blocks.reshape(12, 40//factor, factor, len(H), 14).sum(axis=2)
            central = summary(grouped.sum(axis=(0, 1)), length**2)
            samples = np.array([summary(t, length**2) for t in stratified_block_bootstrap_totals(
                grouped, 800, np.random.default_rng(29092026+1009*length+factor))])
            errors = samples.std(axis=0, ddof=1)
            for name, value, error, between in zip(NAMES, central, errors, replica_se):
                rows.append({"L": length, "observable": name,
                    "updates_per_block": 500*factor, "blocks_per_replica":40//factor,
                    "estimate":value, "block_bootstrap_se":error,
                    "replica_mean_se":between, "bootstrap_draws":800})
        print(f"audited block lengths for L={length}", flush=True)
    frame = pd.DataFrame(rows)
    reference = frame[frame.updates_per_block == 500].set_index(["L", "observable"])
    frame["se_ratio_to_500_updates"] = [
        row.block_bootstrap_se/reference.loc[(row.L,row.observable),"block_bootstrap_se"]
        for row in frame.itertuples()]
    frame.to_csv(ROOT/"data_processed/lattice_block_length_audit.csv", index=False)
    metadata = {"generated_utc":datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(__file__),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "method":"stratified time-block bootstrap; all chains retained",
        "block_lengths":[500,1000,2000], "bootstrap_draws":800,
        "iid_old_variance_ratio":2-1/12-1/40,
        "iid_new_variance_ratio":1-1/40,
        "scope":"sampling-error sensitivity, no new Monte Carlo trajectories",
        "minimum_coarsened_se_ratio":float(frame.se_ratio_to_500_updates.min()),
        "maximum_coarsened_se_ratio":float(frame.se_ratio_to_500_updates.max())}
    (ROOT/"generated/lattice_resampling_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
