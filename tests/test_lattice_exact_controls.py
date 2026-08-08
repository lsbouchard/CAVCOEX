import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lattice_exact_controls import (  # noqa: E402
    K_ISING,
    enumerate_microstates,
    exact_observables_at_h,
    independent_direct_sum_error,
    periodic_bonds,
    scalar_lattice_gas_direct_sum,
)


EXACT_PATH = ROOT / "data_processed" / "lattice_exact_finite_size.csv"
INTERFACE_PATH = (
    ROOT / "data_processed" / "lattice_interface_weight_estimate.csv"
)
METADATA_PATH = ROOT / "generated" / "lattice_exact_controls_metadata.json"


def test_periodic_bond_convention_including_l2_parallel_edges() -> None:
    bonds_l2 = periodic_bonds(2)
    assert bonds_l2.shape == (8, 2)
    unordered = [tuple(sorted(pair)) for pair in bonds_l2.tolist()]
    unique, counts = np.unique(np.asarray(unordered), axis=0, return_counts=True)
    assert len(unique) == 4
    assert np.array_equal(counts, np.full(4, 2))

    bonds_l3 = periodic_bonds(3)
    assert bonds_l3.shape == (18, 2)
    assert len({tuple(sorted(pair)) for pair in bonds_l3.tolist()}) == 18


def test_spin_inversion_exchanges_basins_and_cavity_complements() -> None:
    states = enumerate_microstates(4)
    complement = ((1 << states.volume) - 1) - states.codes
    complement = complement.astype(np.int64)
    assert np.all(states.number + states.number[complement] == states.volume)
    assert np.array_equal(
        states.magnetization, -states.magnetization[complement]
    )
    assert np.array_equal(states.bond_sum, states.bond_sum[complement])
    assert np.array_equal(states.liquid, ~states.liquid[complement])
    assert np.array_equal(
        states.spatial_empty, states.spatial_full[complement]
    )
    assert np.array_equal(states.fixed_empty, states.fixed_full[complement])


def test_exact_outputs_normalize_and_close_at_machine_precision() -> None:
    frame = pd.read_csv(EXACT_PATH)
    assert set(frame["L"]) == {2, 3, 4}
    assert np.all(frame.groupby("L")["configurations"].first() == [16, 512, 65536])
    assert float(np.max(np.abs(frame["probability_sum"] - 1.0))) < 5.0e-13
    assert float(np.max(np.abs(frame["q0_sum"] - 1.0))) < 2.0e-12
    assert float(np.max(np.abs(frame["qC_sum"] - 1.0))) < 2.0e-12
    assert float(np.max(np.abs(frame["decomposition_residual"]))) < 5.0e-12
    assert float(np.max(np.abs(frame["covariance_identity_residual"]))) < 1.0e-12
    assert float(np.max(np.abs(frame["log_odds_shift_residual"]))) < 1.0e-12
    assert float(np.max(np.abs(frame["fixed_minus_spatial_empty"]))) < 5.0e-12


def test_covariance_identity_and_five_point_work_derivative() -> None:
    length = 3
    h_value = 0.7
    step = 1.0e-3
    center = exact_observables_at_h(length, h_value)
    assert abs(center["deltaN"] - center["deltaN_from_covariance"]) < 1.0e-13
    values = [
        exact_observables_at_h(length, h_value + offset * step)["betaW"]
        for offset in (-2, -1, 1, 2)
    ]
    derivative = (
        values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]
    ) / (12.0 * step)
    assert abs(derivative - center["d_betaW_dh_analytic"]) < 1.0e-9


def test_independent_scalar_lattice_gas_direct_sum() -> None:
    # This path explicitly sums exp(beta*epsilon*n_bonds+x*N) and counts
    # empty bonds in scalar loops; it does not reuse the vectorized Ising sum.
    vectorized = exact_observables_at_h(3, 1.25)
    scalar = scalar_lattice_gas_direct_sum(3, 1.25)
    keys = (
        "q0_liquid",
        "qC_liquid",
        "P_spatial_empty",
        "P_fixed_empty",
        "delta_betaW",
        "deltaN",
        "deltaN_intraphase",
        "deltaN_phase_reweighting",
    )
    assert max(abs(vectorized[key] - scalar[key]) for key in keys) < 1.0e-12
    assert independent_direct_sum_error() < 1.0e-12


def test_production_helpers_match_exact_sufficient_statistics() -> None:
    frame = pd.read_csv(EXACT_PATH)
    assert (
        float(frame["production_helper_max_relative_contribution_error"].max())
        < 1.0e-10
    )
    assert (
        float(frame["production_helper_max_abs_observable_error"].max())
        < 1.0e-10
    )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["analysis_version"] == "lattice-exact-controls-v1"
    assert metadata["enumeration"]["configuration_counts"] == {
        "2": 16,
        "3": 512,
        "4": 65536,
    }
    assert "scripts/lattice_gas_phase_switch.py" in metadata["dependency_sha256"]


def test_exact_interface_tension_and_prefactor_estimates() -> None:
    frame = pd.read_csv(INTERFACE_PATH)
    expected_tau = 0.2280631670946952
    assert np.isclose(
        2.0 * K_ISING + np.log(np.tanh(K_ISING)),
        expected_tau,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.allclose(frame["beta_tau_axial"], expected_tau, rtol=0.0, atol=1.0e-15)
    assert set(frame["L"]) == {16, 24, 32, 48, 64}
    assert set(frame["prefactor_label"].astype(str)) == {"1", "L", "V"}

    for length, group in frame.groupby("L"):
        expected_bare = np.exp(-2.0 * expected_tau * length)
        assert np.allclose(
            group["bare_two_interface_factor"],
            expected_bare,
            rtol=1.0e-12,
            atol=0.0,
        )
        by_label = group.set_index(group["prefactor_label"].astype(str))
        expected_prefactors = {"1": 1.0, "L": float(length), "V": float(length**2)}
        for label, prefactor in expected_prefactors.items():
            row = by_label.loc[label]
            relative = prefactor * expected_bare
            probability = relative / (1.0 + relative)
            assert np.isclose(row["prefactor_value"], prefactor)
            assert np.isclose(row["relative_interface_weight_estimate"], relative)
            assert np.isclose(row["interface_probability_scale_estimate"], probability)
            assert np.isclose(
                row["worst_scale_normalized_response_estimate"],
                min(1.0, 2.0 * probability),
            )

    for _, group in frame.groupby("prefactor_label"):
        ordered = group.sort_values("L")
        assert np.all(np.diff(ordered["relative_interface_weight_estimate"]) < 0.0)
    assert frame["estimate_status"].str.contains("not a rigorous bound").all()
