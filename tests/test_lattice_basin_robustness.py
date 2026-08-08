from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_predeclared_interfacial_basin_is_negligible_in_observed_sample() -> None:
    frame = pd.read_csv(
        ROOT / "data_processed" / "lattice_basin_robustness.csv"
    )
    assert set(frame["magnetization_fraction_cutoff"]) == {0.05, 0.10, 0.15}
    assert set(frame["raw_configurations"]) == {240_000}
    assert int(frame.loc[frame["L"] >= 32, "interfacial_observations"].sum()) == 0
    assert frame["observed_q0_interface"].max() < 7.0e-5
    assert frame["absolute_interface_contribution_over_V"].max() < 5.0e-6


def test_zero_observations_are_not_mislabeled_as_conditional_estimates() -> None:
    frame = pd.read_csv(
        ROOT / "data_processed" / "lattice_basin_robustness.csv"
    )
    zero = frame[frame["interfacial_observations"] == 0]
    assert not zero["conditional_interface_estimable"].any()
    assert (zero["observed_total_interface_contribution"] == 0.0).all()
