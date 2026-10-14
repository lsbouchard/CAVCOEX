"""Independent exhaustive/random controls of the unused fast FCC prototype."""

import itertools
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from continuum_replica_exchange import fcc_reference, fitted_fcc_occupancy
from continuum_fcc_nearest_prototype import (
    canonical_fcc_site_index, fitted_fcc_occupancy_fast,
    is_canonical_fcc_reference, nearest_canonical_fcc_site,
)


def independent_full_scan(point, translation, length, reference, parity_shift=0., fit_metric=False):
    if fit_metric:
        differences = point - reference - translation
        differences[:, 0] -= parity_shift
    else:
        centers = (reference + translation) % length
        differences = point - centers
    differences -= length * np.rint(differences / length)
    squared = ((differences[:, 0]**2 + differences[:, 1]**2)
               + differences[:, 2]**2)
    site = np.argmin(squared)
    return site, squared[site]


def independent_original_full_fit(positions, n, length, reference, ignore=-1):
    """Retain a complete scan oracle even after production adopts fast lookup."""
    cells = int(round((len(reference) / 4.)**(1 / 3)))
    translation = np.zeros(3)
    if n - (1 if ignore >= 0 else 0):
        for axis in range(3):
            re, im = 0., 0.
            for particle in range(n):
                if particle != ignore:
                    phase = 4 * np.pi * cells / length * positions[particle, axis]
                    re += math.cos(phase)
                    im += math.sin(phase)
            translation[axis] = length / (4 * np.pi * cells) * math.atan2(im, re)
    best_fit, best_parity = np.inf, 0
    for parity in range(2):
        fit = 0.
        for particle in range(n):
            if particle != ignore:
                _, distance = independent_full_scan(positions[particle], translation,
                                                     length, reference,
                                                     parity * length / (2 * cells), True)
                fit += distance
        if fit < best_fit:
            best_fit, best_parity = fit, parity
    translation[0] += best_parity * length / (2 * cells)
    centers = (reference + translation) % length
    occupancy = np.zeros(len(reference), dtype=np.int64)
    for particle in range(n):
        if particle != ignore:
            site, _ = independent_full_scan(positions[particle], translation, length, reference)
            occupancy[site] += 1
    return centers, occupancy, best_fit


@pytest.mark.parametrize("cells", [1, 2, 3])
def test_half_cell_index_mapping_is_exact_periodic_and_canonical(cells):
    length = 2. * cells
    reference = fcc_reference(4 * cells**3, length)
    for k in itertools.product(range(-2, 2 * cells + 2), repeat=3):
        if sum(k) % 2 == 0:
            index = canonical_fcc_site_index(*k, cells)
            np.testing.assert_array_equal(reference[index], np.array(k) % (2 * cells))


@pytest.mark.parametrize("cells", [1, 2, 3])
def test_exhaustive_half_grid_ties_match_lowest_full_scan_site(cells):
    # Integer box length makes these exact binary half-grid ties, including
    # period boundaries and two/three simultaneous coordinate ties.
    length = 2. * cells
    reference = fcc_reference(4 * cells**3, length)
    translation = np.zeros(3)
    values = np.arange(-1, 2 * cells + 1.25, .5)
    for coordinates in itertools.product(values, repeat=3):
        point = np.array(coordinates)
        expected_site, expected_distance = independent_full_scan(point, translation, length, reference)
        observed_site, observed_distance = nearest_canonical_fcc_site(
            point, translation, length, cells, reference)
        assert observed_site == expected_site
        assert observed_distance == expected_distance


def test_even_rounded_triple_still_needs_multiple_half_grid_ties():
    reference = fcc_reference(32, 4.)
    point = np.array([1.5, 1.5, .5])
    rounded = tuple(np.rint(point).astype(int))
    assert sum(rounded) % 2 == 0
    rounded_site = canonical_fcc_site_index(*rounded, 2)
    observed_site, _ = nearest_canonical_fcc_site(point, np.zeros(3), 4., 2, reference)
    assert observed_site == 3
    assert observed_site < rounded_site


@pytest.mark.parametrize("cells", [1, 2, 3])
def test_odd_repair_axis_and_sign_ties_match_full_scan(cells):
    rng = np.random.default_rng(786543 + cells)
    length = 3.136721
    reference = fcc_reference(4 * cells**3, length)
    for trial in range(800):
        translation = rng.uniform(-length, length, 3)
        integer = rng.integers(-2, 2 * cells + 2, 3)
        # Deliberate equal-magnitude residuals, away from the half-grid
        # fallback, with zero-sign ties included.
        residual = rng.choice([-.25, 0., .25], 3)
        point = (integer + residual) * length / (2 * cells) + translation
        for shift, fit_metric in ((0., False), (0., True), (length / (2 * cells), True)):
            expected = independent_full_scan(point, translation, length, reference, shift, fit_metric)
            observed = nearest_canonical_fcc_site(point, translation, length, cells,
                                                  reference, shift, fit_metric)
            assert observed == expected


@pytest.mark.parametrize("cells", [1, 2, 3])
def test_random_translations_and_both_distance_conventions_match_full_scan(cells):
    rng = np.random.default_rng(786513 + cells)
    length = 3.136721
    reference = fcc_reference(4 * cells**3, length)
    for trial in range(1400):
        translation = rng.uniform(-length, length, 3)
        point = rng.uniform(-length, 2 * length, 3)
        # Include nearly/exactly tied grid points after nonbinary scaling.
        if trial % 3 == 0:
            point = (rng.integers(-2, 2 * cells + 2, 3) + .5) * length / (2 * cells) + translation
        for shift, fit_metric in ((0., False), (0., True), (length / (2 * cells), True)):
            expected_site, expected_distance = independent_full_scan(
                point, translation, length, reference, shift, fit_metric)
            observed_site, observed_distance = nearest_canonical_fcc_site(
                point, translation, length, cells, reference, shift, fit_metric)
            assert observed_site == expected_site
            assert observed_distance == expected_distance


@pytest.mark.parametrize("cells", [1, 2, 3])
def test_full_fast_fit_matches_original_registry_occupancy_and_scores(cells):
    rng = np.random.default_rng(786523 + cells)
    length = 3.136721
    reference = fcc_reference(4 * cells**3, length)
    for trial in range(80):
        n = rng.integers(0, len(reference) + 1)
        if trial % 2:
            positions = (reference + rng.normal(0, .04, reference.shape)
                         + rng.uniform(0, length, 3)) % length
            rng.shuffle(positions)
        else:
            positions = rng.uniform(0, length, reference.shape)
        ignore = rng.integers(n) if n and trial % 3 == 0 else -1
        expected = independent_original_full_fit(positions, n, length, reference, ignore)
        current = fitted_fcc_occupancy(positions, n, length, reference, ignore)
        observed = fitted_fcc_occupancy_fast(positions, n, length, reference, ignore)
        for actual in (current, observed):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            assert actual[2] == expected[2]


def test_reference_guard_rejects_reordered_perturbed_and_noncanonical_arrays():
    reference = fcc_reference(32, 3.136721)
    assert is_canonical_fcc_reference(reference, 3.136721)
    assert not is_canonical_fcc_reference(reference[::-1].copy(), 3.136721)
    perturbed = reference.copy()
    perturbed[1, 1] = np.nextafter(perturbed[1, 1], np.inf)
    assert not is_canonical_fcc_reference(perturbed, 3.136721)
    assert not is_canonical_fcc_reference(np.zeros_like(reference), 3.136721)
    assert not is_canonical_fcc_reference(reference, -1.)
