import numpy as np
from scipy.special import logsumexp

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hard_sphere_models import spt_cavity_work, spt_density_derivatives  # noqa: E402


def test_exact_ensemble_mixture_from_synthetic_partition_functions() -> None:
    rng = np.random.default_rng(712341)
    n = np.arange(0, 31)
    log_q = rng.normal(size=n.size) - 0.09 * n**2
    canonical_constraint_probability = rng.uniform(0.02, 1.0, size=n.size)
    x = 0.61

    log_xi_0 = logsumexp(x * n + log_q)
    log_xi_c = logsumexp(
        x * n + log_q + np.log(canonical_constraint_probability)
    )
    direct = np.exp(log_xi_c - log_xi_0)

    particle_probability = np.exp(x * n + log_q - log_xi_0)
    mixture = np.dot(particle_probability, canonical_constraint_probability)
    np.testing.assert_allclose(direct, mixture, rtol=3.0e-15, atol=3.0e-15)


def test_size_biased_distribution_is_normalized_and_has_fano_mean() -> None:
    rng = np.random.default_rng(9917)
    p0 = rng.uniform(size=50)
    p0 /= p0.sum()
    n0 = np.arange(p0.size)
    mean0 = float(np.dot(p0, n0))

    n_conditioned = np.arange(p0.size - 1)
    p_conditioned = (n_conditioned + 1.0) * p0[1:] / mean0
    np.testing.assert_allclose(p_conditioned.sum(), 1.0, atol=2.0e-15)

    mean_conditioned = float(np.dot(p_conditioned, n_conditioned))
    variance0 = float(np.dot(p0, (n0 - mean0) ** 2))
    np.testing.assert_allclose(
        mean_conditioned - mean0,
        variance0 / mean0 - 1.0,
        rtol=3.0e-15,
        atol=3.0e-15,
    )


def test_half_diameter_cavity_is_exact_and_has_zero_mixing_coefficient() -> None:
    for eta in (0.05, 0.2, 0.4, 0.49):
        np.testing.assert_allclose(
            spt_cavity_work(eta, 0.5), -np.log1p(-eta), atol=2.0e-15
        )
        first, second = spt_density_derivatives(eta, 0.5)
        # The five-point numerical derivatives recover w''-(w')^2=0.
        np.testing.assert_allclose(second - first**2, 0.0, atol=2.0e-8)


def test_two_phase_number_decomposition_closes_to_machine_precision() -> None:
    rng = np.random.default_rng(63190)
    q0 = rng.dirichlet([2.0, 3.0])
    work = rng.uniform(0.1, 2.0, size=2)
    mean_n0 = np.asarray([91.0, 109.0])
    delta_n_phase = np.asarray([-0.8, -1.1])
    q_c_unnormalized = q0 * np.exp(-work)
    q_c = q_c_unnormalized / q_c_unnormalized.sum()

    direct = np.dot(q_c, mean_n0 + delta_n_phase) - np.dot(q0, mean_n0)
    decomposed = np.dot(q_c, delta_n_phase) + np.dot(q_c - q0, mean_n0)
    np.testing.assert_allclose(direct, decomposed, rtol=8.0e-15, atol=2.0e-14)


def test_arbitrary_weight_hard_sphere_phase_bias_matches_size_bias() -> None:
    """GNZ phase reweighting and unrestricted size bias have one coefficient."""

    for q_s in (0.13, 0.5, 0.81):
        q_f = 1.0 - q_s
        rho_f = 0.9391
        rho_s = 1.0373
        rho_bar = q_f * rho_f + q_s * rho_s

        p_f = rho_f  # The common factor 1/zeta cancels.
        p_s = rho_s
        q_s_constrained = q_s * p_s / (q_f * p_f + q_s * p_s)
        phase_reweighting = (q_s_constrained - q_s) * (rho_s - rho_f)
        size_bias = q_f * q_s * (rho_s - rho_f) ** 2 / rho_bar

        np.testing.assert_allclose(
            phase_reweighting, size_bias, rtol=3.0e-15, atol=2.0e-17
        )
