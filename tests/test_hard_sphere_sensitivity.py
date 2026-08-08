import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hard_sphere_models import (  # noqa: E402
    BANNERMAN_2010_COEFFICIENTS,
    ETA_CLOSE_PACKED,
    SPEEDY_1998_COEFFICIENTS,
    speedy_solid_s0,
    speedy_solid_z,
)
from hard_sphere_sensitivity import (  # noqa: E402
    SOLID_MODELS,
    model_observables,
)
from phase_conditioned import finite_size_observables  # noqa: E402


def test_speedy_solid_s0_matches_numerical_pressure_derivative() -> None:
    for coefficients in (
        SPEEDY_1998_COEFFICIENTS,
        BANNERMAN_2010_COEFFICIENTS,
    ):
        for eta in (0.525, 0.545, 0.60):
            step = 1.0e-6
            rho = eta
            pressure_plus = (rho + step) * speedy_solid_z(
                rho + step, coefficients
            )
            pressure_minus = (rho - step) * speedy_solid_z(
                rho - step, coefficients
            )
            derivative = (pressure_plus - pressure_minus) / (2.0 * step)
            expected = 1.0 / derivative
            assert np.isclose(
                speedy_solid_s0(eta, coefficients), expected, rtol=2.0e-9
            )
    assert ETA_CLOSE_PACKED > 0.74


def test_density_only_term_is_identical_across_solid_models() -> None:
    arguments = {
        "rho_f": np.asarray([0.94]),
        "rho_s": np.asarray([1.04]),
        "pressure": np.asarray([11.57]),
        "volume": 500.0,
        "h_values": np.linspace(-10.0, 10.0, 21),
    }
    observables = [
        model_observables(**arguments, model_key=model.key)
        for model in SOLID_MODELS
    ]
    for candidate in observables[1:]:
        assert np.array_equal(
            candidate["scaled_reweight"], observables[0]["scaled_reweight"]
        )
        assert np.array_equal(candidate["reweight"], observables[0]["reweight"])


def test_pressure_anchored_model_reproduces_existing_baseline() -> None:
    rho_f = np.asarray([0.929])
    rho_s = np.asarray([1.034])
    pressure = np.asarray([11.44])
    h_values = np.asarray([-3.0, 0.0, 4.0])
    volume = 900.0
    reference = finite_size_observables(
        rho_f, rho_s, pressure, volume, h_values
    )
    sensitivity = model_observables(
        rho_f,
        rho_s,
        pressure,
        volume,
        h_values,
        "pressure_anchored_free_volume",
    )
    assert np.allclose(
        sensitivity["intraphase"], reference["intraphase"], rtol=2.0e-15
    )
    assert np.allclose(sensitivity["total"], reference["total"], rtol=2.0e-15)


def test_generated_sensitivity_artifacts_preserve_invariant() -> None:
    curves = pd.read_csv(
        ROOT / "data_processed" / "hard_sphere_model_sensitivity.csv"
    )
    envelope = pd.read_csv(
        ROOT / "data_processed" / "hard_sphere_model_envelope.csv"
    )
    crossings = pd.read_csv(
        ROOT / "data_processed" / "hard_sphere_model_crossings.csv"
    )
    assert curves["model"].nunique() == len(SOLID_MODELS) == 3
    assert len(crossings) == len(SOLID_MODELS)
    assert envelope["density_coefficient_model_range"].max() < 5.0e-15
    assert np.allclose(
        curves["deltaN_total"],
        curves["deltaN_intraphase"] + curves["deltaN_reweight"],
        rtol=0.0,
        atol=2.0e-13,
    )
