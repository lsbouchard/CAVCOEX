from pathlib import Path
import json
import math
import sys

import numpy as np
import pytest
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from derive_pulse_finite_size_corrections import (  # noqa: E402
    PulseParameters,
    finite_volume_response,
    first_order_response,
    fwhm_corrections,
    peak_corrections,
    render_json,
    render_tex,
)


PARAMETER_SETS = (
    PulseParameters(0.70, 0.30, 0.08, 1.20, -0.20, 0.12),
    PulseParameters(0.45, -0.25, 0.04, -1.00, 0.18, -0.07),
    PulseParameters(-0.60, 0.20, -0.03, 0.90, 0.15, 0.05),
    PulseParameters(-0.35, -0.10, 0.02, -1.40, -0.12, 0.04),
)


def direct_response_derivative(
    h: float, volume: float, parameters: PulseParameters
) -> float:
    """Derivative of the unexpanded finite-V formula, evaluated independently."""

    d = parameters.delta_rho
    prefactor = d + (parameters.d0 + parameters.dchi * h) / volume
    y_value = d * h + (
        parameters.d0 * h + 0.5 * parameters.dchi * h * h
    ) / volume
    a_value = parameters.a0 + (parameters.a1 - parameters.dn * h) / volume

    def local_logistic(x: float) -> float:
        if x >= 0.0:
            return 1.0 / (1.0 + math.exp(-x))
        exponential = math.exp(x)
        return exponential / (1.0 + exponential)

    left = local_logistic(y_value - a_value)
    right = local_logistic(y_value)
    left_prime = left * (1.0 - left)
    right_prime = right * (1.0 - right)
    kernel_value = left - right
    kernel_y = left_prime - right_prime
    kernel_a = -left_prime
    y_derivative = prefactor
    a_derivative = -parameters.dn / volume
    return (
        parameters.dchi * kernel_value / volume
        + prefactor
        * (kernel_y * y_derivative + kernel_a * a_derivative)
    )


def direct_observables(
    parameters: PulseParameters, volume: float
) -> dict[str, float]:
    """Extract the exact finite-V peak and FWHM without analytic corrections."""

    peak_theory = peak_corrections(parameters)
    width_theory = fwhm_corrections(parameters)
    leading_width = width_theory["fwhm_h_0"]
    branch_values = (
        width_theory["root_minus_h_0"],
        width_theory["root_plus_h_0"],
    )
    lower = min(branch_values) - leading_width
    upper = max(branch_values) + leading_width
    grid = np.linspace(lower, upper, 6001)
    response = np.array(
        [finite_volume_response(float(h), volume, parameters) for h in grid]
    )
    derivative = np.array(
        [direct_response_derivative(float(h), volume, parameters) for h in grid]
    )
    derivative_changes = np.where(derivative[:-1] * derivative[1:] <= 0.0)[0]
    assert len(derivative_changes) > 0
    peak_index = min(
        derivative_changes,
        key=lambda index: abs(
            0.5 * (grid[index] + grid[index + 1])
            - peak_theory["center_h_0"]
        ),
    )
    center = brentq(
        lambda h: direct_response_derivative(h, volume, parameters),
        float(grid[peak_index]),
        float(grid[peak_index + 1]),
        xtol=1.0e-14,
    )
    signed_peak = finite_volume_response(center, volume, parameters)
    half_signed_peak = 0.5 * signed_peak
    offsets = response - half_signed_peak
    center_index = int(np.searchsorted(grid, center))

    left_changes = np.where(
        offsets[:center_index] * offsets[1 : center_index + 1] <= 0.0
    )[0]
    right_changes = np.where(
        offsets[center_index:-1] * offsets[center_index + 1 :] <= 0.0
    )[0]
    assert len(left_changes) > 0 and len(right_changes) > 0
    left_index = int(left_changes[-1])
    right_index = int(center_index + right_changes[0])
    half_height_function = (
        lambda h: finite_volume_response(h, volume, parameters)
        - half_signed_peak
    )
    left = brentq(
        half_height_function,
        float(grid[left_index]),
        float(grid[left_index + 1]),
        xtol=1.0e-13,
    )
    right = brentq(
        half_height_function,
        float(grid[right_index]),
        float(grid[right_index + 1]),
        xtol=1.0e-13,
    )
    assert left < center < right
    assert math.copysign(1.0, signed_peak) == math.copysign(
        1.0, peak_theory["signed_peak_0"]
    )
    return {
        "center_h": center,
        "signed_peak": signed_peak,
        "magnitude_peak": abs(signed_peak),
        "fwhm_h": right - left,
    }


@pytest.mark.parametrize("parameters", PARAMETER_SETS)
def test_full_curve_remainder_is_second_order(
    parameters: PulseParameters,
) -> None:
    peak = peak_corrections(parameters)
    width = fwhm_corrections(parameters)
    fields = (
        peak["center_h_0"],
        width["root_minus_h_0"],
        width["root_plus_h_0"],
        peak["center_h_0"] + 0.37 * width["fwhm_h_0"],
    )
    volumes = (128.0, 256.0, 512.0, 1024.0)
    for h in fields:
        errors = [
            abs(
                finite_volume_response(h, volume, parameters)
                - first_order_response(h, volume, parameters)
            )
            for volume in volumes
        ]
        assert errors[-1] < errors[0] / 25.0
        scaled = [volume * volume * error for volume, error in zip(volumes, errors)]
        assert max(scaled) <= 1.12 * max(scaled[:2]) + 1.0e-12


@pytest.mark.parametrize("parameters", PARAMETER_SETS)
def test_implicit_observable_corrections_match_direct_extraction(
    parameters: PulseParameters,
) -> None:
    peak = peak_corrections(parameters)
    width = fwhm_corrections(parameters)
    coefficients = {
        "center_h": (peak["center_h_0"], peak["center_h_1"]),
        "signed_peak": (peak["signed_peak_0"], peak["signed_peak_1"]),
        "magnitude_peak": (
            peak["magnitude_peak_0"],
            peak["magnitude_peak_1"],
        ),
        "fwhm_h": (width["fwhm_h_0"], width["fwhm_h_1"]),
    }
    volumes = (400.0, 800.0, 1600.0, 3200.0)
    observed = [direct_observables(parameters, volume) for volume in volumes]

    for name, (leading, first_order) in coefficients.items():
        estimated_coefficients = [
            volume * (result[name] - leading)
            for volume, result in zip(volumes, observed)
        ]
        coefficient_errors = [
            abs(estimate - first_order) for estimate in estimated_coefficients
        ]
        scale = max(1.0, abs(first_order))
        assert coefficient_errors[-1] < 3.0e-4 * scale
        assert coefficient_errors[-1] < coefficient_errors[0] / 4.0 + 1.0e-8


def test_signed_and_magnitude_conventions_and_negative_density_gap() -> None:
    for parameters in PARAMETER_SETS:
        peak = peak_corrections(parameters)
        expected_sign = -math.copysign(
            1.0, parameters.delta_rho * parameters.a0
        )
        assert math.copysign(1.0, peak["signed_peak_0"]) == expected_sign
        assert peak["magnitude_peak_0"] == abs(peak["signed_peak_0"])
        assert peak["magnitude_peak_1"] == pytest.approx(
            math.copysign(1.0, peak["signed_peak_0"])
            * peak["signed_peak_1"]
        )
        width = fwhm_corrections(parameters)
        physical_leading_width = abs(
            width["root_plus_h_0"] - width["root_minus_h_0"]
        )
        assert width["fwhm_h_0"] == pytest.approx(physical_leading_width)


def test_degenerate_pulse_is_rejected_for_implicit_observables() -> None:
    parameters = PulseParameters(0.8, 0.1, 0.0, 0.0, 0.2, -0.1)
    assert math.isfinite(finite_volume_response(0.3, 100.0, parameters))
    with pytest.raises(ValueError, match="no leading pulse"):
        peak_corrections(parameters)
    with pytest.raises(ValueError, match="no leading pulse"):
        fwhm_corrections(parameters)


def test_generated_formula_artifacts_are_current_and_machine_readable() -> None:
    tex_path = ROOT / "generated" / "pulse_correction_formulas.tex"
    json_path = ROOT / "generated" / "pulse_correction_formulas.json"
    assert tex_path.read_text(encoding="utf-8") == render_tex()
    assert json_path.read_text(encoding="utf-8") == render_json()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "R_1(h)" in payload["curve"]
    assert "magnitude_1" in payload["peak"]
    assert "width_1" in payload["fwhm"]
    assert any("a1-dn*h" in item for item in payload["conventions"])
