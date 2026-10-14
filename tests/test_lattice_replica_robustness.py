import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIZES = {16, 24, 32, 48, 64}
REPLICAS = set(range(12))
RAW_ARCHIVES = (
    "data_raw/lattice_gas_phase_switch_trajectories.npz",
    "data_raw/lattice_gas_additional_replicas.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_replica_ids(path: Path) -> dict[int, set[int]]:
    pattern = re.compile(r"^L(\d+)_replica(\d+)_magnetization$")
    ids = {length: set() for length in SIZES}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            match = pattern.match(key)
            if match:
                ids[int(match.group(1))].add(int(match.group(2)))
    return ids


def _value(
    frame: pd.DataFrame, mode: str, observable: str, column: str = "value"
) -> float:
    row = frame[
        (frame["L"] == 64)
        & (frame["mode"] == mode)
        & (frame["observable"] == observable)
    ]
    assert len(row) == 1
    return float(row.iloc[0][column])


def test_largest_size_paired_and_unpaired_targets() -> None:
    frame = pd.read_csv(
        ROOT
        / "data_processed"
        / "lattice_paired_unpaired_hierarchical.csv"
    )
    assert abs(_value(frame, "paired", "delta_betaW") - 4.3001694) < 1.0e-6
    assert abs(_value(frame, "unpaired", "delta_betaW") - 4.3008351) < 1.0e-6
    assert (
        abs(_value(frame, "paired", "deltaN_phase_reweighting") + 1816.4438)
        < 1.0e-3
    )
    assert (
        abs(_value(frame, "unpaired", "deltaN_phase_reweighting") + 1819.1428)
        < 1.0e-3
    )
    assert abs(_value(frame, "unpaired", "q0_liquid") - 0.50075417) < 1.0e-8


def test_block_errors_are_positive_and_pairing_reduces_variance() -> None:
    frame = pd.read_csv(
        ROOT
        / "data_processed"
        / "lattice_paired_unpaired_hierarchical.csv"
    )
    paired_error = _value(
        frame, "paired", "deltaN_phase_reweighting", "standard_error"
    )
    unpaired_error = _value(
        frame, "unpaired", "deltaN_phase_reweighting", "standard_error"
    )
    assert paired_error > 0.0
    assert unpaired_error > 0.0
    assert paired_error < unpaired_error
    assert set(frame["L"]) == SIZES
    assert set(frame["replicas"]) == {12}
    assert np.all(frame["blocks_per_replica"] == 40)
    paired_q_error = _value(frame, "paired", "q0_liquid", "standard_error")
    assert paired_q_error == 0.0


def test_paired_and_unpaired_discrepancies_are_standardized_consistently() -> None:
    differences = pd.read_csv(
        ROOT / "data_processed" / "lattice_paired_unpaired_differences.csv"
    )
    selected = differences[
        (differences["L"] == 64)
        & differences["observable"].isin(
            ["delta_betaW", "deltaN_phase_reweighting"]
        )
    ]
    np.testing.assert_allclose(
        selected["absolute_difference_over_unpaired_se"],
        selected["paired_minus_unpaired"].abs()/selected["unpaired_standard_error"],
    )


def test_fixed_bond_comparison_has_finite_uncertainties() -> None:
    frame = pd.read_csv(
        ROOT
        / "data_processed"
        / "lattice_paired_unpaired_hierarchical.csv"
    )
    differences = frame[
        (frame["mode"] == "paired")
        & frame["observable"].isin(
            [
                "P_fixed_minus_spatial_liquid",
                "P_fixed_minus_spatial_gas",
                "delta_betaW_fixed_minus_spatial",
            ]
        )
    ].copy()
    assert set(differences["L"]) == SIZES
    assert len(differences) == 3 * len(SIZES)
    standardized = differences["value"].abs() / differences["standard_error"]
    assert np.all(np.isfinite(standardized))


def test_all_size_delete_one_replica_diagnostics_are_complete() -> None:
    delete_one = pd.read_csv(
        ROOT / "data_processed" / "lattice_replica_delete_one.csv"
    )
    observables = ["delta_betaW", "deltaN_phase_reweighting"]
    for length in sorted(SIZES):
        for mode in ("paired", "unpaired"):
            for observable in observables:
                selected_delete = delete_one[
                    (delete_one["L"] == length)
                    & (delete_one["mode"] == mode)
                    & (delete_one["observable"] == observable)
                ]
                assert len(selected_delete) == 12
                assert set(selected_delete["omitted_replica"]) == REPLICAS
                normalized_shifts = (
                    selected_delete["shift_from_all_replicas"].abs()
                    / selected_delete["all_replica_level_standard_error"]
                )
                assert np.all(np.isfinite(normalized_shifts))
                ratios = selected_delete["delete_one_se_ratio_to_all"]
                np.testing.assert_allclose(
                    ratios,
                    selected_delete["delete_one_replica_level_standard_error"]
                    / selected_delete["all_replica_level_standard_error"],
                    rtol=1e-12,
                )


def test_additional_replica_archive_has_distinct_deterministic_seeds() -> None:
    diagnostics = pd.read_csv(
        ROOT / "data_raw" / "lattice_gas_additional_replicas_diagnostics.csv"
    )
    assert len(diagnostics) == 40
    assert set(diagnostics["L"]) == SIZES
    assert (
        diagnostics.groupby("L")["replica"]
        .apply(set)
        .eq(set(range(4, 12)))
        .all()
    )
    assert diagnostics["seed"].is_unique


def test_merged_outputs_record_both_archives_and_all_replica_ids() -> None:
    baseline_ids = _archive_replica_ids(ROOT / RAW_ARCHIVES[0])
    additional_ids = _archive_replica_ids(ROOT / RAW_ARCHIVES[1])
    assert all(ids == set(range(4)) for ids in baseline_ids.values())
    assert all(ids == set(range(4, 12)) for ids in additional_ids.values())

    metadata = json.loads(
        (
            ROOT / "generated" / "lattice_replica_robustness_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["raw_paths"] == list(RAW_ARCHIVES)
    assert metadata["input_sha256"] == {
        relative_path: _sha256(ROOT / relative_path)
        for relative_path in RAW_ARCHIVES
    }

    replica_values = pd.read_csv(
        ROOT / "data_processed" / "lattice_replica_level_observables.csv"
    )
    assert set(replica_values["L"]) == SIZES
    observed_ids = replica_values.groupby(["L", "mode", "observable"])[
        "replica"
    ].apply(set)
    assert observed_ids.eq(REPLICAS).all()
