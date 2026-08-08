from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_equal_weight_reweighting_coefficient_matches_density_identity() -> None:
    raw = pd.read_csv(
        ROOT / "data_raw" / "fernandez2012_coexistence.csv",
        dtype={"N": str},
    )
    summary = pd.read_csv(
        ROOT / "generated" / "phase_equal_weight_summary.csv"
    )
    finite = raw[raw["N"] != "infinity"].copy()
    assert len(summary) == len(finite) == 7
    for row in summary.itertuples(index=False):
        expected = (row.rho_s - row.rho_f) ** 2 / (
            2.0 * (row.rho_s + row.rho_f)
        )
        assert np.isclose(row.deltaN_reweight_over_V, expected, rtol=2.0e-14)


def test_fwhm_coefficient_matches_closed_form_for_every_source_size() -> None:
    summary = pd.read_csv(
        ROOT / "generated" / "phase_equal_weight_summary.csv"
    )
    widths = pd.read_csv(ROOT / "generated" / "coexistence_widths.csv")
    merged = summary.merge(widths, on=["N_fluid_reference", "V"])
    for row in merged.itertuples(index=False):
        delta_w = np.log(row.rho_f / row.rho_s)
        expected = (
            2.0 * np.arccosh(2.0 + np.cosh(0.5 * delta_w))
            / abs(row.rho_s - row.rho_f)
        )
        assert np.isclose(row.V_times_fwhm_mu, expected, rtol=2.0e-13)
