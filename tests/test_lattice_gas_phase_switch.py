import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIZES = {16, 24, 32, 48, 64}
REPLICAS = set(range(12))


def test_two_phase_wolff_campaign_has_common_normalization() -> None:
    baseline = pd.read_csv(
        ROOT / "data_raw" / "lattice_gas_phase_switch_diagnostics.csv"
    )
    additional = pd.read_csv(
        ROOT / "data_raw" / "lattice_gas_additional_replicas_diagnostics.csv"
    )
    diagnostics = pd.concat((baseline, additional), ignore_index=True)

    assert set(baseline["L"]) == SIZES
    assert set(additional["L"]) == SIZES
    assert baseline.groupby("L")["replica"].apply(set).eq(set(range(4))).all()
    assert (
        additional.groupby("L")["replica"]
        .apply(set)
        .eq(set(range(4, 12)))
        .all()
    )
    assert diagnostics.groupby("L")["replica"].apply(set).eq(REPLICAS).all()
    assert not diagnostics.duplicated(["L", "replica"]).any()
    assert diagnostics["seed"].is_unique
    assert int(diagnostics["magnetization_sector_reversals"].min()) > 10_000
    assert np.all(diagnostics["mean_cluster_fraction"].between(0.0, 1.0))


def test_retired_four_replica_products_are_absent() -> None:
    retired = (
        ROOT / "data_processed" / "lattice_gas_phase_switch.csv",
        ROOT / "generated" / "lattice_gas_phase_switch_summary.csv",
        ROOT / "figures" / "lattice_gas_phase_switch.pdf",
    )
    assert not [path for path in retired if path.exists()]


def test_baseline_metadata_is_retained_as_production_provenance() -> None:
    metadata = json.loads(
        (
            ROOT / "generated" / "lattice_gas_phase_switch_metadata.json"
        ).read_text(encoding="utf-8")
    )
    provenance = metadata["provenance"]
    assert provenance["raw_trajectory_path"] == (
        "data_raw/lattice_gas_phase_switch_trajectories.npz"
    )
    assert provenance["diagnostics_path"] == (
        "data_raw/lattice_gas_phase_switch_diagnostics.csv"
    )
    assert metadata["settings"]["replicas"] == 4
    assert "trajectory production only" in provenance["analysis_note"]
    assert not {"processed_path", "summary_path", "figure_path"}.intersection(
        provenance
    )


def test_merged_phase_switch_outputs_use_twelve_replicas_at_every_size() -> None:
    hierarchical = pd.read_csv(
        ROOT
        / "data_processed"
        / "lattice_paired_unpaired_hierarchical.csv"
    )
    pulse = pd.read_csv(
        ROOT / "data_processed" / "lattice_peak_center_amplitude.csv"
    )
    width = pd.read_csv(
        ROOT / "data_processed" / "lattice_width_hierarchical.csv"
    )

    for frame in (hierarchical, pulse, width):
        assert set(frame["L"]) == SIZES
        assert set(frame["replicas"]) == {12}


def test_largest_size_recovers_exact_pure_phase_parameters() -> None:
    hierarchical = pd.read_csv(
        ROOT
        / "data_processed"
        / "lattice_paired_unpaired_hierarchical.csv"
    )
    paired = hierarchical[
        (hierarchical["L"] == 64) & (hierarchical["mode"] == "paired")
    ].set_index("observable")

    exact = {
        "delta_rho": 0.911319377877496,
        "delta_betaW": 4.299960245352132,
    }
    for observable, exact_value in exact.items():
        row = paired.loc[observable]
        standardized_error = abs(float(row["value"]) - exact_value) / float(
            row["standard_error"]
        )
        assert standardized_error < 3.0


def test_merged_pulse_decomposition_closes_numerically() -> None:
    curves = pd.read_csv(
        ROOT / "data_processed" / "lattice_pulse_curves.csv"
    )
    assert set(curves["L"]) == SIZES
    residual = curves["decomposition_residual_over_V"].to_numpy()
    assert np.max(np.abs(residual)) < 1.0e-12
