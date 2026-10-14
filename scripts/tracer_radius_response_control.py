"""Small, independent calculation of matched-size hard-tracer response.

Source: Fernandez et al., Phys. Rev. Lett. 108, 165701 (2012), Table I,
DOI 10.1103/PhysRevLett.108.165701.  Read the thermodynamic-limit row of
the supplied transcription; do not refit or average finite-size rows.

Inputs are reduced pressure beta*p*sigma^3 and per-particle specific
volumes v_f/sigma^3, v_s/sigma^3.  The dimensionless tracer susceptibility
is sigma*d_(d_t)log(K)=3*(beta*p*sigma^3)*(v_f-v_s)/sigma^3.
The cavity-work contrast slope sigma*a'(sigma) is minus twice that value.
This is the established Evans (2001) first-order fractionation coefficient.

Only marginal source standard errors are available.  The triangle bounds
reported here are sharp bounds for a FIRST-ORDER propagated standard error
over all positive-semidefinite covariance matrices with those marginals.
They are not confidence intervals or an assigned combined standard error.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SOURCE_DOI = "10.1103/PhysRevLett.108.165701"
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data_raw" / "fernandez2012_coexistence.csv"


def matched_tracer_coefficients(pressure, fluid_specific_volume, solid_specific_volume):
    """Return susceptibility and cavity slope from positive reduced inputs."""
    values = (pressure, fluid_specific_volume, solid_specific_volume)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Pressure and specific volumes must be positive and finite")
    coefficient = 3 * pressure * (fluid_specific_volume - solid_specific_volume)
    return coefficient, -2 * coefficient


def linearized_standard_error_bounds(pressure, fluid_volume, solid_volume,
                                     se_pressure, se_fluid_volume, se_solid_volume):
    """Sharp covariance-free bounds, not an independence approximation.

    Gradient order is (pressure, fluid volume, solid volume).  A linearized
    standard error is the length of a sum of vectors whose lengths are
    |gradient_i|*se_i.  The triangle and reverse-triangle bounds are sharp.
    """
    matched_tracer_coefficients(pressure, fluid_volume, solid_volume)
    errors = (se_pressure, se_fluid_volume, se_solid_volume)
    if not all(math.isfinite(error) and error >= 0 for error in errors):
        raise ValueError("Marginal standard errors must be nonnegative and finite")
    gradient = (3 * (fluid_volume - solid_volume), 3 * pressure, -3 * pressure)
    magnitudes = tuple(abs(g) * error for g, error in zip(gradient, errors))
    upper = sum(magnitudes)
    lower = max(0., 2 * max(magnitudes) - upper)
    return lower, upper, gradient, magnitudes


def evaluate_source(path=DEFAULT_INPUT):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["N"].strip() == "infinity"]
    if len(rows) != 1:
        raise ValueError("Exactly one thermodynamic-limit row is required")
    row = rows[0]
    if row["doi"].strip() != SOURCE_DOI:
        raise ValueError("Unexpected source DOI")
    pressure = float(row["p_equal_weight"])
    fluid_volume, solid_volume = float(row["v_fluid"]), float(row["v_solid"])
    coefficient, cavity_slope = matched_tracer_coefficients(pressure, fluid_volume, solid_volume)
    lower, upper, gradient, contributions = linearized_standard_error_bounds(
        pressure, fluid_volume, solid_volume, float(row["se_p_equal_weight"]),
        float(row["se_v_fluid"]), float(row["se_v_solid"]))
    return {
        "source_doi": SOURCE_DOI,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_row": "infinity",
        "inputs_reduced": {"beta_p_sigma3": pressure,
                           "v_fluid_over_sigma3": fluid_volume,
                           "v_solid_over_sigma3": solid_volume},
        "tracer_sigma_derivative_log_K": coefficient,
        "cavity_sigma_derivative_work_contrast": cavity_slope,
        "linearized_tracer_standard_error_bounds": [lower, upper],
        "linearized_cavity_standard_error_bounds": [2 * lower, 2 * upper],
        "gradient_pressure_fluid_solid": list(gradient),
        "absolute_marginal_error_contributions": list(contributions),
        "assigned_combined_standard_error": None,
        "source_covariance_available": False,
        "suggested_display_tracer_coefficient": f"{coefficient:.2f}",
        "suggested_display_cavity_slope": f"{cavity_slope:.2f}",
        "interpretation": "First-order size-fractionation coefficient; not a new fractionation law",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    arguments = parser.parse_args()
    print(json.dumps(evaluate_source(arguments.input), indent=2, sort_keys=True))
