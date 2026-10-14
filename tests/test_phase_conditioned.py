from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


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


def test_pulse_legend_is_outside_data_axes(tmp_path, monkeypatch) -> None:
    import phase_conditioned as phase

    raw = pd.read_csv(ROOT / "data_raw/fernandez2012_coexistence.csv", dtype={"N": str})
    detailed = pd.read_csv(ROOT / "data_processed/phase_conditioned_cavity.csv")
    widths = pd.read_csv(ROOT / "generated/coexistence_widths.csv")
    close = phase.plt.close
    captured = []

    def check_and_close(figure):
        if hasattr(figure, "axes"):
            figure.canvas.draw()
            axis = figure.axes[0]
            legend = axis.get_legend()
            assert legend is not None
            assert not legend.get_window_extent().overlaps(axis.get_window_extent())
            captured.append(True)
        close(figure)

    monkeypatch.setattr(phase.plt, "close", check_and_close)
    phase.plot_collapse(detailed, raw, widths, tmp_path / "pulse.pdf")
    assert captured == [True]
