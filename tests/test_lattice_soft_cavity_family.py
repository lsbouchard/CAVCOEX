from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lattice_analysis_common import (  # noqa: E402
    contribution_blocks,
    load_trajectory_archives,
)
from lattice_gas_phase_switch import (  # noqa: E402
    K_ISING,
    exact_pair_cavity_parameters,
)
from lattice_soft_cavity_family import (  # noqa: E402
    DEFAULT_RAW,
    H_GRID,
    SOFT_LAMBDAS,
    archived_exact_plot_inputs,
    coupling_coordinate,
    exact_staging_features,
    exact_soft_pair_parameters,
    replica_soft_contributions,
    soft_pair_weights,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_soft_pair_weights_have_identity_and_hard_cavity_endpoints() -> None:
    p00_liquid = np.array([0.03, 0.09, 0.21])
    p00_gas = np.array([0.68, 0.54, 0.42])

    identity_liquid, identity_gas = soft_pair_weights(
        p00_liquid, p00_gas, 0.0
    )
    assert np.allclose(identity_liquid, 1.0, rtol=0.0, atol=1.0e-15)
    assert np.allclose(identity_gas, 1.0, rtol=0.0, atol=1.0e-15)

    hard_liquid, hard_gas = soft_pair_weights(
        p00_liquid, p00_gas, float("inf")
    )
    assert np.array_equal(hard_liquid, p00_liquid)
    assert np.array_equal(hard_gas, p00_gas)
    with np.testing.assert_raises(ValueError):
        soft_pair_weights(p00_liquid, p00_gas, -0.1)


def test_exact_soft_family_is_monotone_and_recovers_hard_pair() -> None:
    contrasts = np.array(
        [exact_soft_pair_parameters(K_ISING, value)[2] for value in SOFT_LAMBDAS]
    )
    coordinates = np.array([coupling_coordinate(value) for value in SOFT_LAMBDAS])
    assert np.all(np.diff(contrasts) > 0.0)
    assert np.all(np.diff(coordinates) > 0.0)
    assert coordinates[-1] == 1.0
    assert np.allclose(
        exact_soft_pair_parameters(K_ISING, float("inf")),
        exact_pair_cavity_parameters(K_ISING),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_hard_soft_contributions_match_production_hard_cavity_schema() -> None:
    trajectories = load_trajectory_archives(DEFAULT_RAW)
    h_grid = np.array([-1.0, 0.0, 2.0])
    one_replica = {(16, 0): trajectories[(16, 0)]}
    standard, replicas = contribution_blocks(
        one_replica, 16, h_grid, 40, mode="paired"
    )
    staged = replica_soft_contributions(
        trajectories[(16, 0)], 16**2, h_grid, float("inf"), 40
    )
    assert replicas == [0]
    # The first twelve columns define every thermodynamic observable.  The
    # final two are fixed-site diagnostics, intentionally absent in the
    # spatially averaged soft family and therefore filled with zeros.
    assert np.allclose(staged[..., :12], standard[0, ..., :12], rtol=2e-13, atol=2e-10)
    assert np.count_nonzero(staged[..., 12:]) == 0


def test_soft_cavity_artifacts_are_complete_and_self_consistent() -> None:
    feature_path = ROOT / "data_processed" / "lattice_soft_cavity_features.csv"
    curve_path = ROOT / "data_processed" / "lattice_soft_cavity_curves.csv"
    figure_path = ROOT / "figures" / "lattice_soft_cavity_family.pdf"
    metadata_path = ROOT / "generated" / "lattice_soft_cavity_metadata.json"
    features = pd.read_csv(feature_path, dtype={"lambda_label": str})
    curves = pd.read_csv(curve_path, dtype={"lambda_label": str})
    metadata = json.loads(metadata_path.read_text())

    expected_labels = ["0.25", "0.5", "1", "2", "infinity"]
    assert list(features["lambda_label"]) == expected_labels
    assert len(curves) == len(expected_labels) * len(H_GRID)
    assert set(curves["lambda_label"]) == set(expected_labels)
    assert np.all(features["L"] == 64)
    assert np.all(features["replicas"] == 12)
    assert np.all(features["blocks_per_replica"] == 40)
    assert np.all(features["bootstrap_draws"] == metadata["bootstrap_draws"])
    assert np.all(features["bootstrap_draws"] >= 2_000)

    measured_columns = [
        "measured_delta_betaW",
        "measured_peak_center_h",
        "measured_peak_magnitude",
        "measured_fwhm_h",
    ]
    error_columns = [
        "se_delta_betaW",
        "se_peak_center_h",
        "se_peak_magnitude",
        "se_fwhm_h",
    ]
    z_columns = [
        "difference_z_delta_betaW",
        "difference_z_peak_center_h",
        "difference_z_peak_magnitude",
        "difference_z_fwhm_h",
    ]
    assert np.all(np.isfinite(features[measured_columns + error_columns + z_columns]))
    assert np.all(features[error_columns].to_numpy() > 0.0)
    assert np.all(np.diff(features["exact_delta_betaW"]) > 0.0)
    assert np.all(np.diff(features["measured_delta_betaW"]) > 0.0)
    hard_exact = exact_pair_cavity_parameters(K_ISING)[2]
    assert np.isclose(
        features.iloc[-1]["exact_delta_betaW"], hard_exact, rtol=0.0, atol=1e-12
    )
    assert metadata["replica_ids"] == list(range(12))
    assert "not thermodynamic-limit extrapolation tests" in metadata["comparison_scope"]
    assert metadata["runtime_seconds"] > 0.0
    assert figure_path.stat().st_size > 10_000

    recorded_hashes = dict(metadata["output_sha256"])
    render = metadata.get("figure_render", {})
    if render:
        assert set(render["output_sha256"]) == {"figures/lattice_soft_cavity_family.pdf"}
        for relative_path, expected_hash in render["input_sha256"].items():
            if relative_path.startswith("data_processed/"):
                assert metadata["output_sha256"][relative_path] == expected_hash
            assert _sha256(ROOT / relative_path) == expected_hash
        assert render["script_sha256"] == _sha256(ROOT / "scripts/lattice_soft_cavity_family.py")
    recorded_hashes.update(render.get("output_sha256", {}))
    for relative_path, expected_hash in recorded_hashes.items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_exact_display_curves_reproduce_saved_feature_values() -> None:
    features = pd.read_csv(ROOT / "data_processed/lattice_soft_cavity_features.csv")
    values = exact_staging_features(features["staging_coordinate"].to_numpy(),
                                    archived_exact_plot_inputs())
    for column, curve in values.items():
        assert np.allclose(curve, features[column], rtol=2e-13, atol=2e-13)
