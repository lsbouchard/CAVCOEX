"""Independent constant-cost canonical-FCC nearest-site prototype.

This file is not imported by the sampler.  The compiled helpers assume the
canonical coordinates/order of ``fcc_reference(4*c**3, L)``; callers must
check that assumption once before using them in a production kernel.
"""

import math

import numpy as np
from numba import njit


def is_canonical_fcc_reference(reference, length):
    """Exact once-per-run guard; arbitrary user-supplied references need a scan."""
    if not np.isfinite(length) or length <= 0 or reference.ndim != 2 or reference.shape[1] != 3:
        return False
    cells = int(round((len(reference) / 4.)**(1 / 3)))
    if cells <= 0 or 4 * cells**3 != len(reference):
        return False
    basis = np.array([[0., 0., 0.], [0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])
    expected = np.array([(np.array([i, j, k]) + b) * length / cells
                         for i in range(cells) for j in range(cells)
                         for k in range(cells) for b in basis])
    return np.array_equal(reference, expected)


@njit(cache=True, inline="always")
def wrapped_fcc_site_index(kx, ky, kz, cells):
    """Index arithmetic after half-cell coordinates have already been wrapped."""
    return 4 * (((kx >> 1) * cells + (ky >> 1)) * cells + (kz >> 1)) + 2 * (kx & 1) + (ky & 1)


@njit(cache=True, inline="always")
def canonical_fcc_site_index(kx, ky, kz, cells):
    """Map an even-parity half-cell triple to canonical FCC enumeration."""
    period = 2 * cells
    return wrapped_fcc_site_index(kx % period, ky % period, kz % period, cells)


@njit(cache=True, inline="always")
def candidate_distance(point, translation, length, reference, site, parity_shift, fit_metric):
    """Match the two original full-scan arithmetic conventions explicitly."""
    squared = 0.
    for axis in range(3):
        if fit_metric:
            delta = point[axis] - reference[site, axis] - translation[axis]
            if axis == 0:
                delta -= parity_shift
        else:
            center = (reference[site, axis] + translation[axis]) % length
            delta = point[axis] - center
        delta -= length * np.rint(delta / length)
        squared += delta * delta
    return squared


@njit(cache=True)
def nearest_canonical_fcc_site_with_ties(point, translation, length, cells, reference,
                                        parity_shift=0., fit_metric=False):
    """Rare-boundary fallback, including all tied rounded integer triples.

    The unconstrained nearest integer triple is repaired by one +/-axis
    change if odd. At half-integer boundaries, all tied rounded triples
    are included before parity repair. Candidate distances are evaluated
    on the actual reference coordinates, with equal distances resolved by
    the lowest canonical site index. A tiny near-tie envelope also protects
    against roundoff in the grid-coordinate transformation.
    """
    base = np.empty(3, dtype=np.int64)
    alternatives = np.zeros(3, dtype=np.int64)
    tie_axes = np.empty(3, dtype=np.int64)
    tie_count = 0
    for axis in range(3):
        coordinate = point[axis] - translation[axis]
        if axis == 0:
            coordinate -= parity_shift
        scaled = (2 * cells / length) * coordinate
        base[axis] = int(np.rint(scaled))
        residual = scaled - base[axis]
        tolerance = 16 * np.finfo(np.float64).eps * max(1., abs(scaled))
        if abs(abs(residual) - .5) <= tolerance:
            alternatives[axis] = 1 if residual >= 0 else -1
            tie_axes[tie_count] = axis
            tie_count += 1

    best_site = len(reference)
    best_distance = np.inf
    candidate = np.empty(3, dtype=np.int64)
    for mask in range(1 << tie_count):
        candidate[:] = base
        for bit in range(tie_count):
            if mask & (1 << bit):
                axis = tie_axes[bit]
                candidate[axis] += alternatives[axis]
        if (candidate[0] + candidate[1] + candidate[2]) % 2 == 0:
            site = canonical_fcc_site_index(candidate[0], candidate[1], candidate[2], cells)
            distance = candidate_distance(point, translation, length, reference, site,
                                          parity_shift, fit_metric)
            if distance < best_distance or (distance == best_distance and site < best_site):
                best_site, best_distance = site, distance
        else:
            for axis in range(3):
                for sign in (-1, 1):
                    candidate[axis] += sign
                    site = canonical_fcc_site_index(candidate[0], candidate[1], candidate[2], cells)
                    distance = candidate_distance(point, translation, length, reference, site,
                                                  parity_shift, fit_metric)
                    if distance < best_distance or (distance == best_distance and site < best_site):
                        best_site, best_distance = site, distance
                    candidate[axis] -= sign
    return best_site, best_distance


@njit(cache=True)
def nearest_canonical_fcc_site(point, translation, length, cells, reference,
                               parity_shift=0., fit_metric=False):
    """Allocation-free generic path, with the verified half-grid fallback.

    Odd-parity repair costs ``1-2*abs(residual_axis)``. Thus only axes with
    maximal residual magnitude need be compared, including equal/near-equal
    axes and both signs at a zero residual. Wrapping is performed once;
    one-step repairs use branches rather than repeated integer division.
    """
    scale = 2 * cells / length
    sx = scale * (point[0] - translation[0] - parity_shift)
    sy = scale * (point[1] - translation[1])
    sz = scale * (point[2] - translation[2])
    kx, ky, kz = int(np.rint(sx)), int(np.rint(sy)), int(np.rint(sz))
    ex, ey, ez = sx - kx, sy - ky, sz - kz
    tolerance = 16 * np.finfo(np.float64).eps * max(1., abs(sx), abs(sy), abs(sz))
    if (abs(abs(ex) - .5) <= tolerance or abs(abs(ey) - .5) <= tolerance
            or abs(abs(ez) - .5) <= tolerance):
        return nearest_canonical_fcc_site_with_ties(point, translation, length, cells,
                                                    reference, parity_shift, fit_metric)
    period = 2 * cells
    kx, ky, kz = kx % period, ky % period, kz % period
    if (kx + ky + kz) % 2 == 0:
        site = wrapped_fcc_site_index(kx, ky, kz, cells)
        return site, candidate_distance(point, translation, length, reference, site,
                                        parity_shift, fit_metric)
    largest = max(abs(ex), abs(ey), abs(ez))
    best_site, best_distance = len(reference), np.inf
    for axis in range(3):
        residual = ex if axis == 0 else ey if axis == 1 else ez
        if abs(residual) < largest - tolerance:
            continue
        for sign in (-1, 1):
            if residual * sign < -tolerance:
                continue
            nx, ny, nz = kx, ky, kz
            value = (kx if axis == 0 else ky if axis == 1 else kz) + sign
            if value < 0:
                value += period
            elif value >= period:
                value -= period
            if axis == 0:
                nx = value
            elif axis == 1:
                ny = value
            else:
                nz = value
            site = wrapped_fcc_site_index(nx, ny, nz, cells)
            distance = candidate_distance(point, translation, length, reference, site,
                                          parity_shift, fit_metric)
            if distance < best_distance or (distance == best_distance and site < best_site):
                best_site, best_distance = site, distance
    return best_site, best_distance


@njit(cache=True)
def fitted_fcc_occupancy_fast(positions, n, length, reference, ignore=-1):
    """Prototype of the full original fit, replacing only nearest-site scans."""
    if n < 0 or n > positions.shape[0] or ignore < -1 or ignore >= n:
        raise ValueError("Registry particle count or deleted index is invalid")
    cells = int(round((len(reference) / 4.)**(1 / 3)))
    translation = np.zeros(3)
    if n - (1 if ignore >= 0 else 0):
        for axis in range(3):
            re, im = 0., 0.
            for particle in range(n):
                if particle == ignore:
                    continue
                phase = 4 * np.pi * cells / length * positions[particle, axis]
                re += math.cos(phase)
                im += math.sin(phase)
            translation[axis] = length / (4 * np.pi * cells) * math.atan2(im, re)
    best_fit, best_parity = np.inf, 0
    for parity in range(2):
        shift = parity * length / (2 * cells)
        fit = 0.
        for particle in range(n):
            if particle != ignore:
                _, distance = nearest_canonical_fcc_site(positions[particle], translation,
                                                         length, cells, reference, shift, True)
                fit += distance
        if fit < best_fit:
            best_fit, best_parity = fit, parity
    translation[0] += best_parity * length / (2 * cells)
    centers = (reference + translation) % length
    occupied = np.zeros(len(reference), dtype=np.int64)
    for particle in range(n):
        if particle != ignore:
            site, _ = nearest_canonical_fcc_site(positions[particle], translation,
                                                length, cells, reference)
            occupied[site] += 1
    return centers, occupied, best_fit
