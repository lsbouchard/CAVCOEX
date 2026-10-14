"""Independent finite-graph checks of the cavity-relative pair measure."""

import itertools

import numpy as np
import pytest


@pytest.mark.parametrize("activity", [.2, 1., 7.3])
def test_relative_pair_profile_and_its_number_integral(activity):
    # The ring's site-count measure plays the role of volume. All allowed
    # subsets are enumerated; no sampler or insertion-law implementation is
    # imported. Adjacent sites exclude one another.
    volume = 8
    hosts = []
    for bits in itertools.product((0, 1), repeat=volume):
        if not any(bits[i] and bits[(i + 1) % volume] for i in range(volume)):
            hosts.append(tuple(i for i, present in enumerate(bits) if present))
    weights = np.array([activity**len(host) for host in hosts])
    probability = weights / weights.sum()
    numbers = np.array([len(host) for host in hosts])
    mean_number = probability @ numbers
    pair_measure = np.zeros(volume)
    cavity_measure = np.zeros(volume)
    cavity_partition = 0.
    for host, weight, prob in zip(hosts, weights, probability):
        for first in host:
            for second in host:
                if first != second:
                    pair_measure[(second - first) % volume] += prob / volume
        for center in range(volume):
            if any((particle - center) % volume in (0, 1, volume - 1)
                   for particle in host):
                continue
            cavity_partition += weight / volume
            for particle in host:
                cavity_measure[(particle - center) % volume] += weight / volume
    cavity_measure /= cavity_partition
    density = mean_number / volume
    assert cavity_measure == pytest.approx(pair_measure / density, abs=3e-14)
    pair_correlation = pair_measure / density**2
    assert cavity_measure == pytest.approx(density * pair_correlation, abs=3e-14)
    for subset in ((0,), (1, 7), (2, 3, 5), tuple(range(volume))):
        assert cavity_measure[list(subset)].sum() == pytest.approx(
            pair_measure[list(subset)].sum() / density, abs=3e-14)
    assert cavity_measure.sum() == pytest.approx(
        probability @ (numbers * (numbers - 1)) / mean_number, abs=3e-14)
    variance = probability @ numbers**2 - mean_number**2
    assert cavity_measure.sum() - mean_number == pytest.approx(
        variance / mean_number - 1, abs=3e-14)


def test_extensive_mixture_term_does_not_remove_finite_deletion_background():
    # A sharply bimodal number law makes the -1 term exact and explicit.
    # A small box need not yet have a positive total cavity response even
    # though its phase-reweighting contribution is positive.
    densities = np.array([.75, 1.])
    weights = np.array([.5, .5])
    mean_density = weights @ densities
    coefficient = np.prod(weights) * np.diff(densities)[0]**2 / mean_density
    for volume in (32, 400):
        numbers = volume * densities
        mean_number = weights @ numbers
        variance = weights @ (numbers - mean_number)**2
        total_response = variance / mean_number - 1
        assert total_response == pytest.approx(volume * coefficient - 1)
    assert 32 * coefficient - 1 < 0 < 400 * coefficient - 1
