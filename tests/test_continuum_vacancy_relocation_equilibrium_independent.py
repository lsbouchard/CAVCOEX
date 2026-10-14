"""Actual-kernel binding of independent relocation/partition controls.

The low-N caps here are deliberately truncating and are compared only with
their own partition integrals. They are not dense-phase or GNZ evidence.
"""
import math
import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import continuum_replica_exchange as kernel
import test_continuum_number_umbrella_equilibrium as quadrature
from test_continuum_vacancy_birth_equilibrium import VECTORS, verify_fourier_caches
from test_residual_vacancy_relocation_reference import reference_cube_density


def test_actual_proposal_density_integrates_all_periodic_overlapping_cubes():
    length, width, epsilon = 1., .2, .2
    centers = np.array([[.95, .5, .5], [.05, .5, .5], [.05, .6, .5]])
    cuts = [np.unique(np.r_[0., length, (centers[:, axis] - width) % length,
                            (centers[:, axis] + width) % length]) for axis in range(3)]
    integral = 0.
    for intervals in product(*(list(zip(axis[:-1], axis[1:])) for axis in cuts)):
        point = np.array([.5 * (a + b) for a, b in intervals])
        actual = kernel.cube_mixture_density(point, centers, length, width, epsilon)
        expected = reference_cube_density(point, centers, length, width, epsilon)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
        integral += math.prod(b - a for a, b in intervals) * actual
    np.testing.assert_allclose(integral, 1., atol=2e-14, rtol=0)
    for point in (np.array([0., .55, .5]), np.array([1., .55, .5])):
        expected = epsilon + (1 - epsilon) / (2 * width) ** 3
        np.testing.assert_allclose(kernel.cube_mixture_density(point, centers, length, width, epsilon), expected)
    assert kernel.cube_mixture_density(np.zeros(3), np.empty((0, 3)), length, width, epsilon) == 1.


def test_enabled_relocation_matches_independent_cavity_pair_order_number_partition_integrals(monkeypatch):
    actual = quadrature.run_chunk
    counter = np.zeros((4, 2), dtype=np.int64)

    def enabled(*args, **kwargs):
        kwargs['vacancy_relocation_probability'] = 1.
        kwargs['vacancy_relocation_counts'] = counter
        return actual(*args, **kwargs)

    monkeypatch.setattr(quadrature, 'run_chunk', enabled)
    # This unchanged independent reference integrates all ordered coordinates
    # with exactly one 1/N!, nonzero soft/order/number penalties, and physical
    # hard/cavity supports. It verifies all six pairwise product-number laws,
    # Fourier order means, saved support and independently rebuilt caches.
    quadrature.test_number_and_order_umbrellas_match_independent_nine_dimensional_product_integrals()
    assert np.all(counter[:, 0] > 10000)
    assert np.all(counter[:, 1] > 500)
    assert np.all(counter[:, 1] < counter[:, 0])


def test_relocation_accepts_and_rejects_without_number_changes_or_stale_fourier_cache():
    length, cap, sweeps = 3., 2, 3000
    positions = np.array([[[.4, .4, .4], [1.9, 1.9, 1.9]]])
    initial = positions.copy()
    counts = np.array([2], dtype=np.int64)
    centers = np.zeros((1, 3))
    diameter, radius, activity = np.ones(1), np.array([.5]), np.array([1000.])
    identity = np.zeros(1, dtype=np.int64)
    reference = kernel.fcc_reference(32, length)
    counter = np.zeros((1, 2), dtype=np.int64)
    kernel.seed_rng(190101)
    samples, _, _, configurations = kernel.run_chunk(
        positions, counts, centers, diameter, radius, activity, identity,
        length, sweeps, 0, 1, reference, np.array([1e-300]), np.array([np.inf]),
        np.array([2.5]), np.array([.5]), np.empty((0, 2), dtype=np.int64),
        np.empty(0, dtype=np.int64), np.array([0], dtype=np.int64),
        False, True, .30, .2, 1, 1,
        number_stiffness=np.array([1.3]), number_preferred=np.array([1.1]),
        vacancy_relocation_probability=1., vacancy_relocation_counts=counter,
    )
    # Large activity blocks every numerical death and cap blocks births;
    # sub-ulp ordinary displacements leave coordinates exactly unchanged.
    assert np.all(samples[:, 0, 0] == cap) and counts[0] == cap
    assert counter[0, 0] == sweeps
    assert 100 < counter[0, 1] < sweeps - 100
    before = np.concatenate((initial[None], configurations[:-1]), axis=0)
    changed = np.any(configurations != before, axis=(1, 2, 3))
    assert np.count_nonzero(changed) == counter[0, 1]
    verify_fourier_caches(samples, configurations, length, 2 * VECTORS)
    separation = configurations[:, 0, 0] - configurations[:, 0, 1]
    separation -= length * np.rint(separation / length)
    assert np.all(np.linalg.norm(separation, axis=1) >= 1.)
    assert np.all(samples[:, 0, 6] >= .5)
