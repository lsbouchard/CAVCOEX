import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hard_sphere_defect_sensitivity import (  # noqa: E402
    build_outputs,
    complex_step_jacobian,
    density_coefficient_covariance_sensitivity,
    metric_vector,
    observables,
    source_standard_errors,
)


def test_vacancy_work_shift_is_exact() -> None:
    v_fluid = 1.06448
    v_solid = 0.96405
    vacancy_fraction = 2.6e-4
    baseline = observables(v_fluid, v_solid, 0.0)
    perturbed = observables(v_fluid, v_solid, vacancy_fraction)
    assert np.isclose(
        perturbed["beta_delta_w"] - baseline["beta_delta_w"],
        -np.log1p(-vacancy_fraction),
        rtol=2.0e-14,
    )


def test_density_only_coefficient_matches_closed_form() -> None:
    v_fluid = 1.06448
    v_solid = 0.96405
    vacancy_fraction = 2.6e-4
    values = observables(v_fluid, v_solid, vacancy_fraction)
    rho_fluid = 1.0 / v_fluid
    rho_solid = (1.0 - vacancy_fraction) / v_solid
    expected = (rho_solid - rho_fluid) ** 2 / (
        2.0 * (rho_solid + rho_fluid)
    )
    assert np.isclose(values["reweight_coefficient"], expected, rtol=1.0e-15)


def test_published_scenario_separates_source_error_from_defect_scale() -> None:
    raw = pd.read_csv(
        ROOT / "data_raw" / "fernandez2012_coexistence.csv", dtype={"N": str}
    )
    source = raw[raw["N"] == "infinity"].iloc[0]
    scenarios, comparison = build_outputs(source, 2.6e-4)
    coefficient = comparison.set_index("metric").loc[
        "reweight_coefficient"
    ]
    assert len(scenarios) == 2
    assert np.isclose(
        coefficient.relative_shift_percent,
        -0.5368298005,
        rtol=2.0e-9,
    )
    assert coefficient.absolute_shift_over_baseline_source_standard_error > 2.7
    assert "not a confidence interval" in coefficient.interpretation


def test_generated_artifact_records_fixed_site_assumption() -> None:
    artifact = pd.read_csv(
        ROOT / "data_processed" / "hard_sphere_defect_sensitivity.csv"
    )
    perturbed = artifact[
        artifact["scenario"] == "fixed_site_density_vacancy_scale"
    ].iloc[0]
    assert np.isclose(perturbed.vacancy_fraction, 2.6e-4)
    assert "fixed lattice-site density" in perturbed.defect_scope
    assert "total uncertainty" in perturbed.defect_scope


def test_covariance_sensitivity_matches_independent_jacobian_and_sharp_bounds() -> None:
    vf, vs, sf, ss = 1.06448, 0.96405, 0.00010, 0.00003
    result = density_coefficient_covariance_sensitivity(
        vf, vs, sf, ss, np.asarray([-1.0, 0.0, 1.0])
    )
    metrics = ("reweight_coefficient",)
    jacobian = complex_step_jacobian(
        lambda fluid, solid: metric_vector(fluid, solid, 0.0, metrics),
        np.asarray([vf, vs]),
    )[0]
    np.testing.assert_allclose(
        result.loc[0, ["d_coefficient_d_v_fluid", "d_coefficient_d_v_solid"]],
        jacobian,
        rtol=1.0e-14,
    )
    a, b = jacobian * np.asarray([sf, ss])
    errors = result.se_reweight_coefficient_source.to_numpy()
    np.testing.assert_allclose(
        errors,
        [abs(a - b), np.hypot(a, b), abs(a + b)],
        rtol=1.0e-14,
    )
    assert errors[0] == pytest.approx(abs(a) + abs(b))
    assert errors[-1] == pytest.approx(abs(abs(a) - abs(b)))
    assert errors[1] == pytest.approx(
        source_standard_errors(vf, vs, sf, ss, 0.0, metrics)[0]
    )


@pytest.mark.parametrize("correlation", [-1.001, 1.001, np.nan])
def test_covariance_sensitivity_rejects_invalid_correlation(correlation: float) -> None:
    with pytest.raises(ValueError):
        density_coefficient_covariance_sensitivity(1.1, 0.9, 0.01, 0.01, correlation)


def test_covariance_sensitivity_vanishes_at_zero_density_contrast() -> None:
    result = density_coefficient_covariance_sensitivity(
        1.0, 1.0, 0.01, 0.02, np.linspace(-1.0, 1.0, 11)
    )
    assert np.all(result.reweight_coefficient == 0.0)
    assert np.all(result.se_reweight_coefficient_source == 0.0)
