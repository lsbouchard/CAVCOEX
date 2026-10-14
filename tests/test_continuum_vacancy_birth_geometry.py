"""Independent proposal geometry controls, including post-deletion refits.

Only generic, uniquely fitted registries are expected to be translation
covariant. Degenerate deterministic chart choices are allowed in MH; none
of these registry rules defines an energy or occupancy constraint.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pytest
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import (
    cube_mixture_density, fcc_reference, fcc_vacancy_centers,
    fitted_fcc_occupancy, vacancy_birth_density,
)


def independently_vectorized_cube_density(points, centers, length, width, epsilon):
    if len(centers) == 0:
        return np.full(len(points), 1 / length**3)
    differences = points[:, None, :] - centers[None, :, :]
    differences -= length * np.rint(differences / length)
    covers = (abs(differences) < width).all(axis=2).sum(axis=1)
    return epsilon / length**3 + (1 - epsilon) * covers / (len(centers) * (2 * width)**3)


def test_cube_marginal_sums_all_overlapping_periodic_components_and_normalizes():
    length, width, epsilon = 3., .7, .2
    centers = np.array([[.1, .1, .1], [.3, .3, .3], [2.9, 2.9, 2.9]])
    point = np.array([.2, .2, .2])
    assert cube_mixture_density(point, centers, length, width, epsilon) == pytest.approx(
        epsilon / length**3 + (1 - epsilon) / (2 * width)**3)
    points = length * qmc.Sobol(3, scramble=True, seed=615451).random_base2(17)
    density = independently_vectorized_cube_density(points, centers, length, width, epsilon)
    assert abs(length**3 * density.mean() - 1) < .003
    observed = np.array([cube_mixture_density(p, centers, length, width, epsilon)
                         for p in points[::128]])
    np.testing.assert_allclose(observed, density[::128], atol=1e-14, rtol=0)


def test_perfect_single_vacancy_birth_and_reverse_death_have_analytic_density():
    length = (32 * .96405)**(1 / 3)
    reference = fcc_reference(32, length)
    removed = 11
    point = reference[removed].copy()
    residual = np.delete(reference, removed, axis=0)
    width, epsilon = .05, .2
    expected = epsilon / length**3 + (1 - epsilon) / (2 * width)**3
    assert len(fcc_vacancy_centers(residual, 31, length, reference)) == 1
    assert vacancy_birth_density(point, residual, 31, length, reference,
                                 width, epsilon) == pytest.approx(expected)
    assert vacancy_birth_density(point, reference, 32, length, reference,
                                 width, epsilon, removed) == pytest.approx(expected)
    # The pre-deletion host has no empty sites and must revert to FULL uniform.
    assert vacancy_birth_density(point, reference, 32, length, reference,
                                 width, epsilon) == pytest.approx(1 / length**3)


def test_generic_registry_permutation_and_translation_invariance_of_proposal_density():
    rng = np.random.default_rng(615452)
    length = (32 * .96405)**(1 / 3)
    reference = fcc_reference(32, length)
    positions = np.delete(reference, [5, 17], axis=0)
    positions = (positions + rng.normal(0, .03, positions.shape)) % length
    vacancies = fcc_vacancy_centers(positions, len(positions), length, reference)
    assert len(vacancies) == 2
    points = np.vstack((vacancies + rng.uniform(-.08, .08, vacancies.shape),
                        rng.uniform(0, length, (80, 3)))) % length
    shift = np.array([1.193, -.579, 2.639])
    width, epsilon = .12, .2
    expected = np.array([vacancy_birth_density(p, positions, len(positions),
                                             length, reference, width, epsilon)
                         for p in points])
    permutation = rng.permutation(len(positions))
    permuted = np.array([vacancy_birth_density(p, positions[permutation], len(positions),
                                             length, reference, width, epsilon)
                         for p in points])
    translated = np.array([vacancy_birth_density((p + shift) % length,
                                               (positions + shift) % length,
                                               len(positions), length, reference, width, epsilon)
                          for p in points])
    np.testing.assert_allclose(permuted, expected, atol=1e-13, rtol=0)
    np.testing.assert_allclose(translated, expected, atol=1e-13, rtol=0)
    assert (expected >= epsilon / length**3).all()


def test_ignore_recomputes_every_registry_coordinate_and_handles_last_swap_order():
    rng = np.random.default_rng(615453)
    length = (32 * .96405)**(1 / 3)
    reference = fcc_reference(32, length)
    positions = (reference[:9] + rng.normal(0, .11, (9, 3))) % length
    query = rng.uniform(0, length, (30, 3))
    width, epsilon = .3, .2
    for tag in range(len(positions)):
        residual = np.delete(positions, tag, axis=0)
        # Actual accepted death replaces the deleted slot by the final particle.
        swapped = positions.copy()
        swapped[tag] = positions[-1]
        swapped = swapped[:-1]
        for p in np.vstack((query, positions[tag])):
            ignored = vacancy_birth_density(p, positions, len(positions),
                                            length, reference, width, epsilon, tag)
            explicit = vacancy_birth_density(p, residual, len(residual),
                                             length, reference, width, epsilon)
            reordered = vacancy_birth_density(p, swapped, len(swapped),
                                              length, reference, width, epsilon)
            assert ignored == pytest.approx(explicit, abs=1e-13)
            assert ignored == pytest.approx(reordered, abs=1e-13)


def test_uniform_fallbacks_and_degenerate_chart_are_finite_and_deterministic():
    length = 3.
    reference = fcc_reference(4, length)
    positions = np.array([[.2, .4, .6], [1.1, 1.3, 1.5]])
    point = np.array([.25, .35, .45])
    for n, ignore in ((0, -1), (1, -1), (2, 0), (2, 1)):
        assert vacancy_birth_density(point, positions, n, length, reference,
                                     .12, .2, ignore) == 1 / length**3
    assert cube_mixture_density(point, np.empty((0, 3)), length, .12, .2) == 1 / length**3
    with pytest.raises(ValueError, match="deleted index"):
        vacancy_birth_density(point, positions, 2, length, reference, .12, .2, 2)

    # Both parity fits tie in the mathematical configuration below, despite
    # nonvanishing axial modes. A single registry cannot be equivariant on it:
    # its translation symmetry exchanges the two possible FCC parity classes.
    length = (32 * .96405)**(1 / 3)
    reference = fcc_reference(32, length)
    delta = np.full(3, length / 4)
    symmetric = (np.arange(4)[:, None] * delta[None, :]) % length
    modes = np.exp(8j * np.pi / length * symmetric).sum(axis=0)
    np.testing.assert_allclose(abs(modes), 4, atol=1e-14, rtol=0)
    differences = symmetric[:, None] - symmetric[None]
    differences -= length * np.rint(differences / length)
    separation = np.linalg.norm(differences, axis=2)
    assert separation[np.triu_indices(4, 1)].min() > 1
    first = fitted_fcc_occupancy(symmetric, 4, length, reference)
    second = fitted_fcc_occupancy(symmetric.copy(), 4, length, reference)
    for observed, expected in zip(first, second):
        np.testing.assert_array_equal(observed, expected)
    value = vacancy_birth_density(symmetric[0], symmetric, 4, length, reference, .12, .2)
    assert np.isfinite(value) and value > 0
