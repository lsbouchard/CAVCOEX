import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from solid_cavity_mc import contact_shell_pressure  # noqa: E402
from solid_cavity_replica_campaign import (  # noqa: E402
    REPLICA_SEED_STRIDE,
    ZERO_EVENT_BOUND_COLUMNS,
    hierarchical_summary,
    replica_base_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_replica_seed_streams_are_deterministic_and_disjoint() -> None:
    seeds = [replica_base_seed(631_026_801, index) for index in range(4)]
    assert len(set(seeds)) == 4
    assert np.all(np.diff(seeds) == REPLICA_SEED_STRIDE)


def test_hierarchical_summary_respects_within_and_between_replica_error() -> None:
    values = np.asarray([1.0, 1.2, 0.9, 1.1])
    within = np.asarray([0.08, 0.10, 0.09, 0.11])
    result = hierarchical_summary(values, within)
    expected_within = np.sqrt(np.sum(within**2)) / values.size
    expected_between = np.std(values, ddof=1) / np.sqrt(values.size)
    assert np.isclose(result["mean"], np.mean(values))
    assert np.isclose(result["within_replica_se"], expected_within)
    assert np.isclose(result["between_replica_se"], expected_between)
    assert np.isclose(
        result["hierarchical_se"], max(expected_within, expected_between)
    )
    assert result["max_delete_one_shift_over_hierarchical_se"] < 1.0


def test_contact_shell_pressure_uses_unordered_pair_normalization() -> None:
    positions = np.asarray([[1.0, 1.0, 1.0], [2.01, 1.0, 1.0]])
    length = 10.0
    pressure, contact_g, pairs = contact_shell_pressure(
        positions, length, shell_width=0.02
    )
    rho = 2.0 / length**3
    eta = np.pi * rho / 6.0
    shell_volume = 4.0 * np.pi / 3.0 * (1.02**3 - 1.0)
    expected_g = 1.0 / (0.5 * 2.0 * rho * shell_volume)
    assert pairs == 1
    assert np.isclose(contact_g, expected_g)
    assert np.isclose(pressure, rho * (1.0 + 4.0 * eta * expected_g))


def test_production_replica_archive_closes_independent_seed_gap() -> None:
    processed = pd.read_csv(
        ROOT / "data_processed" / "solid_cavity_mc_additional_replicas.csv"
    )
    states = processed[["N", "replica", "seed", "equil_seed"]].drop_duplicates()
    assert set(states["N"]) == {108, 256}
    assert states.groupby("N")["replica"].nunique().to_dict() == {108: 4, 256: 4}
    assert states["seed"].nunique() == 8
    assert states["equil_seed"].nunique() == 8
    assert not processed["cross_basin_normalization"].any()
    assert not processed["N4_phase_switch_data"].any()
    assert set(processed["phase_scope"]) == {
        "defect_free_commensurate_FCC_solid_basin"
    }
    assert processed["crystal_structure_gate"].all()
    exact = processed[np.isclose(processed["R_over_sigma"], 0.50)]
    assert exact["exact_control_pass"].all()
    target = processed[np.isclose(processed["R_over_sigma"], 0.80)]
    assert target["empty_events"].min() >= 100
    assert target["positive_blocks"].min() >= 75
    assert target["half_difference_z"].max() < 2.0
    assert target["replica_audited_reliable"].all()
    assert set(target["reliability_status"]) == {"replica_audited_estimate"}
    assert not set(ZERO_EVENT_BOUND_COLUMNS).intersection(processed.columns)
    assert not processed["reliability_status"].str.contains(
        "probability", case=False
    ).any()
    zero_event = processed[processed["empty_events"] == 0]
    assert not zero_event.empty
    assert set(zero_event["reliability_status"]) == {
        "raw_exposure_only_zero_events"
    }


def test_replica_comparisons_are_consistent_with_original_trajectory() -> None:
    comparison = pd.read_csv(
        ROOT / "data_processed" / "solid_cavity_mc_replica_comparison.csv"
    )
    rare = comparison[comparison["metric"].isin(["P_empty_R0p80", "betaW_R0p80"])]
    structure = comparison[
        comparison["metric"].isin(
            [
                "global_Q6",
                "max_FCC_Bragg",
                "lindemann_ratio",
                "beta_pressure_contact_shell_delta_0p02",
            ]
        )
    ]
    assert rare["additional_minus_baseline_z"].abs().max() < 1.0
    assert structure["additional_minus_baseline_z"].abs().max() < 2.0
    assert comparison["max_delete_one_shift_over_hierarchical_se"].max() < 1.0
    assert comparison["cochran_q_per_dof"].max() < 1.1
    pressure = structure[
        structure["metric"] == "beta_pressure_contact_shell_delta_0p02"
    ]
    assert pressure["additional_minus_published_z"].isna().all()
    assert pressure["interpretation"].str.contains(
        "no significance test", regex=False
    ).all()


def test_replica_metadata_records_full_provenance_and_current_hashes() -> None:
    metadata = json.loads(
        (ROOT / "generated" / "solid_cavity_mc_replica_metadata.json").read_text()
    )
    assert metadata["scope"]["ensemble"] == "canonical NVT"
    assert "No phase switch" in metadata["scope"]["not_direct_coexistence"]
    assert metadata["replicas_per_size"] == 4
    assert len(metadata["seed_records"]) == 8
    assert metadata["quality_summary"]["independent_state_trajectories"] == 8
    assert metadata["quality_summary"]["all_crystal_gates_pass"]
    assert metadata["quality_summary"]["all_exact_R0p5_controls_pass"]
    assert metadata["quality_summary"]["all_R0p80_replica_audited"]
    assert not metadata["classification"]["probability_floor_used"]
    assert "no rule-of-three" in metadata["classification"]["zero_event_reporting"]
    assert metadata["script_sha256"] == _sha256(
        ROOT / "scripts" / "solid_cavity_replica_campaign.py"
    )
    assert metadata["solid_cavity_mc_script_sha256"] == _sha256(
        ROOT / "scripts" / "solid_cavity_mc.py"
    )
    assert metadata["baseline_inputs"]["processed_sha256"] == _sha256(
        ROOT / "data_processed" / "solid_cavity_mc.csv"
    )
    assert metadata["baseline_inputs"]["structure_sha256"] == _sha256(
        ROOT / "data_raw" / "solid_cavity_mc_structure.csv"
    )
