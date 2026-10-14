#!/usr/bin/env python3
"""Record release-file hashes without rewriting historical run provenance.

Generation-time metadata belongs to the numerical generators.  This packaging
step writes a separate ``generated/release_provenance.json``: a snapshot of
the currently distributed files and a suggested command. A matching hash is an integrity check, not
evidence that the command was executed.  Legacy top-level hashes (some of
which were refreshed during earlier packaging) are preserved verbatim.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

from make_release_manifest import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def local_dependencies(script: str, root: Path) -> list[str]:
    """Find the static local Python import closure for a packaged entry point."""
    found: set[str] = set()
    pending = [script]
    while pending:
        current = pending.pop()
        tree = ast.parse((root / current).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                relative = f"scripts/{module.rsplit('.', 1)[-1]}.py"
                if relative != script and relative not in found and (root / relative).is_file():
                    found.add(relative)
                    pending.append(relative)
    return sorted(found)


SPECS = {
    "generated/reaud_covariance_stability_metadata.json": {
        "script": "scripts/reaud_covariance_mode_stability.py",
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
            "data_processed/lattice_leave_lmax_prediction_coefficients.csv",
            "data_processed/lattice_leave_lmax_prediction_covariance.npz",
            "data_processed/lattice_scalar_joint_validation.csv",
        ],
        "command": "python3 scripts/reaud_covariance_mode_stability.py",
    },
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
    "generated/hard_sphere_defect_sensitivity_metadata.json": {
        "script": "scripts/hard_sphere_defect_sensitivity.py",
        "inputs": [
            "data_raw/fernandez2012_coexistence.csv",
            "data_raw/hard_sphere_vacancy_scale.csv",
        ],
        "command": "python3 scripts/hard_sphere_defect_sensitivity.py",
    },
    "generated/lattice_resampling_audit_metadata.json": {
        "script": "scripts/lattice_resampling_audit.py",
        "inputs": [
            "data_raw/lattice_gas_phase_switch_trajectories.npz",
            "data_raw/lattice_gas_additional_replicas.npz",
        ],
        "command": "python3 scripts/lattice_resampling_audit.py",
    },
}


def enrich(root: Path = ROOT) -> int:
    """Snapshot current files without changing any generation metadata bytes."""

    def digest(relative_path: str) -> str:
        return sha256_file(root / relative_path)

    refreshed = datetime.now(timezone.utc).isoformat()
    records = {}
    for metadata_path, spec in SPECS.items():
        path = root / metadata_path
        if not path.exists():
            raise FileNotFoundError(path)
        script = str(spec["script"])
        dependencies = sorted(set(local_dependencies(script, root)).union(
            str(item) for item in spec.get("dependencies", [])
        ))
        inputs = [str(item) for item in spec.get("inputs", [])]
        records[metadata_path] = {
            "recorded_utc": refreshed,
            "scope": (
                "Current distributed files only; no execution is asserted. "
                "Existing generation-time and legacy packaging fields are "
                "preserved without modification."
            ),
            "execution_metadata_sha256": digest(metadata_path),
            "script": script,
            "script_sha256": digest(script),
            "dependency_sha256": {
                item: digest(item) for item in dependencies
            },
            "input_sha256": {item: digest(item) for item in inputs},
            "suggested_reproduce_command": str(spec["command"]),
        }
    (root / "generated" / "release_provenance.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(SPECS)


def main() -> None:
    count = enrich()
    print(f"recorded {count} release contexts separately; execution metadata bytes unchanged")


if __name__ == "__main__":
    main()
