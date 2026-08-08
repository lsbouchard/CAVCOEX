from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lattice_reweighting_overlap import support_summary


ROOT = Path(__file__).resolve().parents[1]


def test_support_summary_is_scale_invariant_and_exact_for_uniform_weights() -> None:
    weights = np.full(17, 3.5)
    first = support_summary(weights)
    second = support_summary(11.0 * weights)
    assert first["units"] == second["units"]
    for key in (
        "kish_count",
        "kish_fraction",
        "entropy_count",
        "perplexity_fraction",
        "maximum_normalized_weight",
    ):
        assert np.isclose(first[key], second[key], rtol=1.0e-14, atol=1.0e-14)
    assert first["kish_count"] == 17.0
    assert np.isclose(first["kish_fraction"], 1.0)
    assert np.isclose(first["perplexity_fraction"], 1.0)
    assert np.isclose(first["maximum_normalized_weight"], 1.0 / 17.0)


def test_orbit_and_replica_block_support_outputs_are_well_formed() -> None:
    curves = pd.read_csv(
        ROOT / "data_processed" / "lattice_reweighting_overlap.csv"
    )
    assert set(curves["analysis_unit"]) == {
        "antithetic_orbit",
        "replica_time_block",
    }
    assert set(curves["estimator"]) == {
        "unconstrained_orbit",
        "constrained_orbit",
        "liquid_cavity_branch",
        "gas_cavity_branch",
    }
    assert set(curves["replicas"]) == {12}
    assert (curves["kish_fraction"] > 0.0).all()
    assert (curves["kish_fraction"] <= 1.0 + 1.0e-12).all()
    assert (curves["perplexity_fraction"] > 0.0).all()
    assert (curves["perplexity_fraction"] <= 1.0 + 1.0e-12).all()
    assert (curves["maximum_normalized_weight"] > 0.0).all()
    assert (curves["replicas_above_one_percent"] == 12).all()


def test_spin_inversion_orbits_have_uniform_unconstrained_weight_at_zero_field() -> None:
    curves = pd.read_csv(
        ROOT / "data_processed" / "lattice_reweighting_overlap.csv"
    )
    selected = curves[
        (curves["analysis_unit"] == "antithetic_orbit")
        & (curves["estimator"] == "unconstrained_orbit")
        & np.isclose(curves["h"], 0.0)
    ]
    assert len(selected) == 5
    assert np.allclose(selected["kish_fraction"], 1.0, atol=1.0e-12)
    assert np.allclose(selected["perplexity_fraction"], 1.0, atol=1.0e-12)


def test_displayed_range_has_orbit_support_and_block_aggregation_is_not_dominated() -> None:
    curves = pd.read_csv(
        ROOT / "data_processed" / "lattice_reweighting_overlap.csv"
    )
    orbit = curves[curves["analysis_unit"] == "antithetic_orbit"]
    blocks = curves[curves["analysis_unit"] == "replica_time_block"]
    assert orbit["kish_fraction"].min() > 0.05
    assert blocks["kish_fraction"].min() > 0.90
    assert blocks["maximum_normalized_weight"].max() < 0.02


def test_support_interval_sensitivity_uses_all_three_thresholds() -> None:
    frame = pd.read_csv(
        ROOT / "data_processed" / "lattice_support_interval_sensitivity.csv"
    )
    assert set(np.round(frame["support_threshold"], 2)) == {0.05, 0.10, 0.20}
    assert set(frame["observable"]) == {
        "peak_center_h",
        "peak_magnitude",
        "fwhm_h",
    }
    estimated = frame[frame["status"] == "estimated"]
    assert len(estimated) >= 30
    assert np.isfinite(estimated["estimate"]).all()
