"""Independent exact algebra/provenance controls, without MC or a new EOS."""

import csv
import importlib.util
import math
from decimal import Decimal as D
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tracer_radius_response_control.py"
SPEC = importlib.util.spec_from_file_location("tracer_radius_response_control", MODULE_PATH)
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def test_published_infinite_row_matches_independent_decimal_arithmetic():
    with control.DEFAULT_INPUT.open(newline="", encoding="utf-8") as handle:
        row = next(row for row in csv.DictReader(handle) if row["N"] == "infinity")
    expected = D(3) * D(row["p_equal_weight"]) * (D(row["v_fluid"]) - D(row["v_solid"]))
    assert expected == D("3.486738783")
    result = control.evaluate_source()
    assert result["tracer_sigma_derivative_log_K"] == pytest.approx(float(expected), abs=1e-13)
    assert result["cavity_sigma_derivative_work_contrast"] == pytest.approx(float(-2 * expected), abs=2e-13)
    assert result["source_doi"] == "10.1103/PhysRevLett.108.165701"
    assert result["source_row"] == "infinity"
    assert result["suggested_display_tracer_coefficient"] == "3.49"
    assert result["suggested_display_cavity_slope"] == "-6.97"
    assert result["assigned_combined_standard_error"] is None


def test_covariance_free_first_order_error_bounds_are_sharp():
    result = control.evaluate_source()
    lower, upper = result["linearized_tracer_standard_error_bounds"]
    # Decimal arithmetic uses the source's actual marginal errors; no
    # independent-error assumption appears in this reference calculation.
    contributions = [D("0.00030129"), D("0.00347181"), D("0.001041543")]
    exact_upper = sum(contributions)
    exact_lower = 2 * max(contributions) - exact_upper
    assert lower == pytest.approx(float(exact_lower), abs=1e-15)
    assert upper == pytest.approx(float(exact_upper), abs=1e-15)
    # Each extremum is attained by a rank-one PSD covariance.  Signs
    # include the negative solid-volume gradient.
    gradient = result["gradient_pressure_fluid_solid"]
    errors = [.001, .0001, .00003]
    for signs, expected in [((1, 1, -1), upper), ((-1, 1, 1), lower)]:
        vector = [sign * error for sign, error in zip(signs, errors)]
        covariance = [[a * b for b in vector] for a in vector]
        variance = sum(gradient[i] * covariance[i][j] * gradient[j]
                       for i in range(3) for j in range(3))
        assert math.sqrt(variance) == pytest.approx(expected, abs=1e-15)


def test_zero_marginal_errors_and_equal_density_control():
    assert control.matched_tracer_coefficients(10., 1., 1.) == (0., -0.)
    low, high, _, _ = control.linearized_standard_error_bounds(10., 1.1, 1., 0., 0., 0.)
    assert low == high == 0.
    # Reversing the phase names reverses the contrast/susceptibility signs.
    a = control.matched_tracer_coefficients(10., 1.1, 1.)
    b = control.matched_tracer_coefficients(10., 1., 1.1)
    assert a == tuple(-x for x in b)


def test_manuscript_displays_reproducible_rounded_values():
    result = control.evaluate_source()
    root = MODULE_PATH.parents[1]
    if not (root / "manuscript.tex").exists() and not (root / "supplement.tex").exists():
        pytest.skip("Document-binding control requires the full publication bundle; lean code omits both TeX sources")
    main = (root / "manuscript.tex").read_text(encoding="utf-8")
    # The technical material is now in the unified manuscript's appendices.
    # Check the numerical values, not the obsolete wording of their meaning.
    assert f"${result['suggested_display_tracer_coefficient']}$" in main
    technical = main if r"\appendix" in main else (
        root / "supplement.tex"
    ).read_text(encoding="utf-8")
    assert rf"\simeq{result['suggested_display_tracer_coefficient']}" in technical
    assert rf"\simeq{result['suggested_display_cavity_slope']}" in technical


@pytest.mark.parametrize("inputs", [(0., 1., 1.), (-1., 1., 1.),
                                   (1., float("nan"), 1.), (1., 1., float("inf"))])
def test_invalid_reduced_inputs_are_rejected(inputs):
    with pytest.raises(ValueError):
        control.matched_tracer_coefficients(*inputs)


def test_missing_or_duplicate_limit_rows_and_wrong_source_are_rejected(tmp_path):
    with control.DEFAULT_INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    source_row = next(row for row in rows if row["N"] == "infinity")
    for subset in (rows[:-1], rows + [source_row], [dict(source_row, doi="wrong")]):
        path = tmp_path / "source.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
        with pytest.raises(ValueError):
            control.evaluate_source(path)
