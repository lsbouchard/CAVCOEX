"""Analytic water/ice application of equilibrium cavity reweighting.

Inputs must come from a specified water model or measurement, with consistent
energy units. This module supplies no water-model data, dynamics, hysteresis,
or absolute nucleation-rate prediction. Temperature-window formulas are
leading two-phase limits; cluster-profile conditioning is a finite-law identity.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def phase_probability_tilt(log_odds, work_contrast, thermal_energy):
    """Return reference and constrained ice probabilities from exact odds.

    ``work_contrast`` is ice minus liquid exclusion work. For a hard cavity
    each work is -kBT log of its phase-conditional compatibility probability.
    """
    energy = np.asarray(thermal_energy, dtype=float)
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("thermal_energy must be finite and positive")
    y = np.asarray(log_odds, dtype=float)
    contrast = np.asarray(work_contrast, dtype=float)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(contrast)):
        raise ValueError("log odds and work contrast must be finite")
    return expit(y), expit(y - contrast / energy)


def thermal_response_per_molecule(z, a, *, volume_gap, latent_heat):
    """Leading probability, volume, and enthalpy reweighting responses.

    z=N*latent_heat*(beta-beta_eq), a=work_contrast/(kB*T_m).
    volume_gap is ice minus liquid volume per molecule; latent_heat is
    liquid minus ice enthalpy per molecule. No density-ordering assumption
    is imposed. Within-phase responses are not included in these terms.
    """
    if not np.isfinite(latent_heat) or latent_heat <= 0:
        raise ValueError("latent_heat must be finite and positive")
    if not np.isfinite(volume_gap):
        raise ValueError("volume_gap must be finite")
    q0, qc = phase_probability_tilt(z, a, 1.0)
    dq = qc - q0
    return dq, volume_gap * dq, -latent_heat * dq


def equal_weight_temperature_shift(*, melting_temperature, work_contrast,
                                   molecule_count, latent_heat):
    """Leading shift -T_m*work_contrast/(N*latent_heat), in temperature units.

    This is a finite-sample equal-weight shift, not a bulk melting-point shift.
    The work contrast must use the same fixed-pressure phase definitions.
    """
    if (isinstance(molecule_count, (bool, np.bool_))
            or not isinstance(molecule_count, (int, np.integer))
            or molecule_count <= 0):
        raise ValueError("molecule_count must be a positive integer")
    if not np.isfinite(melting_temperature) or melting_temperature <= 0:
        raise ValueError("melting_temperature must be finite and positive")
    if not np.isfinite(latent_heat) or latent_heat <= 0:
        raise ValueError("latent_heat must be finite and positive")
    if not np.isfinite(work_contrast):
        raise ValueError("work_contrast must be finite")
    return -melting_temperature * work_contrast / (molecule_count * latent_heat)


def cluster_profile_reweight(reference_mass, compatibility, *, thermal_energy):
    """Exactly reweight a discrete, fixed-bin ice-cluster probability law.

    Return constrained mass, constrained -kBT log(mass) profile, and the
    unconditioned cavity work. Input probabilities must already be unbiased
    (umbrella weights removed). Zero-mass bins get infinite profile height.
    A barrier must be located anew on the resulting full profile.
    """
    mass = np.asarray(reference_mass, dtype=float)
    p = np.asarray(compatibility, dtype=float)
    if mass.ndim != 1 or mass.size == 0 or p.shape != mass.shape:
        raise ValueError("inputs must be nonempty one-dimensional arrays of equal shape")
    if not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("reference mass must be finite and nonnegative")
    if not np.isclose(mass.sum(), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("reference mass must sum to one")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("compatibility must lie in [0,1]")
    if not np.isfinite(thermal_energy) or thermal_energy <= 0:
        raise ValueError("thermal_energy must be finite and positive")
    weight = mass * p
    normalization = float(weight.sum())
    if normalization <= 0:
        raise ValueError("the constrained normalization must be positive")
    constrained_mass = weight / normalization
    profile = np.full(mass.shape, np.inf)
    positive = constrained_mass > 0
    profile[positive] = -thermal_energy * np.log(constrained_mass[positive])
    return constrained_mass, profile, -thermal_energy * np.log(normalization)
