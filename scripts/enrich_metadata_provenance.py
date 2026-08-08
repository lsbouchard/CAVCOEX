#!/usr/bin/env python3
"""Attach uniform, machine-verifiable provenance to generated metadata.

The numerical generators retain their domain-specific metadata.  This final
release step adds the current script and dependency hashes, hashes every
external input consumed by an analysis, and records the exact command used by
the Makefile.  It deliberately does not alter numerical data or figures.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(relative_path: str) -> str:
    """Return the SHA-256 digest of one release file."""

    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SPECS = {
    "generated/cavity_mc_metadata.json": {
        "script": "scripts/cavity_mc.py",
        "command": (
            "python3 scripts/cavity_mc.py --seed 20260724 --sizes 108 256 "
            "--etas 0.05 0.10 0.20 0.30 0.40 --equil-sweeps 16000 "
            "--tune-sweeps 4000 --production-blocks 50 "
            "--sweeps-per-block 400 --sample-stride 5 "
            "--probes-per-sample 2048"
        ),
    },
    "generated/eos_benchmark_metadata.json": {
        "script": "scripts/cs_benchmark.py",
        "dependencies": ["scripts/hard_sphere_models.py"],
        "command": "python3 scripts/cs_benchmark.py",
    },
    "generated/grand_reweight_metadata.json": {
        "script": "scripts/grand_reweight.py",
        "dependencies": ["scripts/hard_sphere_models.py"],
        "command": "python3 scripts/grand_reweight.py",
    },
    "generated/hard_sphere_model_sensitivity_metadata.json": {
        "script": "scripts/hard_sphere_sensitivity.py",
        "dependencies": ["scripts/hard_sphere_models.py"],
        "inputs": ["data_raw/fernandez2012_coexistence.csv"],
        "command": "python3 scripts/hard_sphere_sensitivity.py",
    },
    "generated/lattice_basin_robustness_metadata.json": {
        "script": "scripts/lattice_basin_robustness.py",
        "dependencies": ["scripts/lattice_analysis_common.py"],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_basin_robustness.py",
    },
    "generated/lattice_gas_additional_replicas_metadata.json": {
        "script": "scripts/lattice_gas_additional_replicas.py",
        "dependencies": ["scripts/lattice_gas_phase_switch.py"],
        "command": (
            "python3 scripts/lattice_gas_additional_replicas.py --seed "
            "25072026 --sizes 16 24 32 48 64 --replica-start 4 --replica-stop 12 "
            "--equil-steps 2000 --production-steps 20000 --workers 1 "
            "--overwrite"
        ),
    },
    "generated/lattice_exact_controls_metadata.json": {
        "script": "scripts/lattice_exact_controls.py",
        "dependencies": ["scripts/lattice_gas_phase_switch.py"],
        "command": "python3 scripts/lattice_exact_controls.py",
    },
    "generated/lattice_gas_phase_switch_metadata.json": {
        "script": "scripts/lattice_gas_phase_switch.py",
        "command": (
            "python3 scripts/lattice_gas_phase_switch.py --seed 25072026 "
            "--sizes 16 24 32 48 64 --replicas 4 --equil-steps 2000 "
            "--production-steps 20000 --blocks-per-replica 40 "
            "--bootstrap-draws 400"
        ),
    },
    "generated/lattice_pulse_scaling_metadata.json": {
        "script": "scripts/lattice_pulse_scaling.py",
        "dependencies": ["scripts/lattice_analysis_common.py"],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
            "data_processed/lattice_scalar_joint_validation.csv",
        ],
        "command": "python3 scripts/lattice_pulse_scaling.py",
    },
    "generated/lattice_pulse_area_sum_rule_metadata.json": {
        "script": "scripts/lattice_pulse_area_sum_rule.py",
        "dependencies": [
            "scripts/lattice_analysis_common.py",
            "scripts/lattice_gas_phase_switch.py",
        ],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_pulse_area_sum_rule.py",
    },
    "generated/lattice_soft_cavity_metadata.json": {
        "script": "scripts/lattice_soft_cavity_family.py",
        "dependencies": [
            "scripts/lattice_analysis_common.py",
            "scripts/lattice_gas_phase_switch.py",
            "scripts/lattice_pulse_scaling.py",
        ],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_soft_cavity_family.py",
    },
    "generated/lattice_pulse_correction_prediction_metadata.json": {
        "script": "scripts/lattice_pulse_correction_prediction.py",
        "dependencies": [
            "scripts/derive_pulse_finite_size_corrections.py",
            "scripts/lattice_analysis_common.py",
            "scripts/lattice_gas_phase_switch.py",
            "scripts/lattice_pulse_scaling.py",
        ],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_pulse_correction_prediction.py",
    },
    "generated/lattice_replica_robustness_metadata.json": {
        "script": "scripts/lattice_replica_robustness.py",
        "dependencies": ["scripts/lattice_analysis_common.py"],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_replica_robustness.py",
    },
    "generated/lattice_reweighting_overlap_metadata.json": {
        "script": "scripts/lattice_reweighting_overlap.py",
        "dependencies": ["scripts/lattice_analysis_common.py"],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_reweighting_overlap.py",
    },
    "generated/lattice_validation_statistics_metadata.json": {
        "script": "scripts/lattice_validation_statistics.py",
        "dependencies": [
            "scripts/lattice_analysis_common.py",
            "scripts/lattice_gas_phase_switch.py",
            "scripts/lattice_pulse_scaling.py",
            "scripts/derive_pulse_finite_size_corrections.py",
        ],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_validation_statistics.py",
    },
    "generated/lattice_free_intercept_joint_quadratic_metadata.json": {
        "script": "scripts/lattice_free_intercept_joint_quadratic.py",
        "dependencies": ["scripts/lattice_validation_statistics.py"],
        "inputs": ["data_processed/lattice_scalar_joint_validation.csv"],
        "command": "python3 scripts/lattice_free_intercept_joint_quadratic.py",
    },
    "generated/lattice_leave_lmax_prediction_metadata.json": {
        "script": "scripts/lattice_leave_lmax_prediction.py",
        "dependencies": [
            "scripts/lattice_analysis_common.py",
            "scripts/lattice_gas_phase_switch.py",
            "scripts/lattice_pulse_correction_prediction.py",
            "scripts/lattice_validation_statistics.py",
        ],
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_leave_lmax_prediction.py",
    },
    "generated/phase_metadata.json": {
        "script": "scripts/phase_conditioned.py",
        "dependencies": ["scripts/hard_sphere_models.py"],
        "inputs": ["data_raw/fernandez2012_coexistence.csv"],
        "command": "python3 scripts/phase_conditioned.py",
    },
    "generated/solid_cavity_mc_metadata.json": {
        "script": "scripts/solid_cavity_mc.py",
        "dependencies": ["scripts/cavity_mc.py"],
        "inputs": ["data_raw/fernandez2012_coexistence.csv"],
        "command": (
            "python3 scripts/solid_cavity_mc.py --seed 92026072 --sizes "
            "108 256 --equil-sweeps 16000 --tune-sweeps 4000 "
            "--production-blocks 100 --sweeps-per-block 400 "
            "--sample-stride 5 --probes-per-sample 2048"
        ),
    },
    "generated/solid_cavity_mc_replica_metadata.json": {
        "script": "scripts/solid_cavity_replica_campaign.py",
        "dependencies": ["scripts/solid_cavity_mc.py", "scripts/cavity_mc.py"],
        "inputs": [
            "data_raw/fernandez2012_coexistence.csv",
            "data_processed/solid_cavity_mc.csv",
            "data_raw/solid_cavity_mc_structure.csv",
        ],
        "command": (
            "python3 scripts/solid_cavity_replica_campaign.py --seed "
            "631026801 --sizes 108 256 --replicas 4 --equil-sweeps 16000 "
            "--tune-sweeps 4000 --production-blocks 100 "
            "--sweeps-per-block 400 --sample-stride 5 "
            "--probes-per-sample 2048 --overwrite"
        ),
    },
    "generated/tonks_metadata.json": {
        "script": "scripts/tonks_validation.py",
        "command": "python3 scripts/tonks_validation.py",
    },
}


def main() -> None:
    refreshed = datetime.now(timezone.utc).isoformat()
    for metadata_path, spec in SPECS.items():
        path = ROOT / metadata_path
        if not path.exists():
            raise FileNotFoundError(path)
        script = str(spec["script"])
        dependencies = [str(item) for item in spec.get("dependencies", [])]
        inputs = [str(item) for item in spec.get("inputs", [])]
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["script"] = script
        metadata["script_sha256"] = sha256(script)
        metadata["dependency_sha256"] = {
            item: sha256(item) for item in dependencies
        }
        metadata["input_sha256"] = {item: sha256(item) for item in inputs}
        metadata["reproduce_command"] = str(spec["command"])
        metadata["provenance_refreshed_utc"] = refreshed
        path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"refreshed provenance in {len(SPECS)} metadata files")


if __name__ == "__main__":
    main()
