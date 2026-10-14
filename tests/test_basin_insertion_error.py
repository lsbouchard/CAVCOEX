"""Independent finite-state control of the phase insertion/removal bridge.

The hard-core lattice gas on a finite graph has the same insertion identity
as continuum hard spheres.  Exhaustive independent-set enumeration checks
the signed basin-boundary term without using the project's sampler.
"""

import itertools

import numpy as np
import pytest


@pytest.mark.parametrize("activity", [0.2, 1.0, 7.3])
def test_exact_basin_insertion_error_and_absolute_bound(activity):
    vertex_count = 6
    edges = tuple((i, (i + 1) % vertex_count) for i in range(vertex_count))
    states = [
        np.asarray(bits, dtype=int)
        for bits in itertools.product((0, 1), repeat=vertex_count)
        if all(not (bits[i] and bits[j]) for i, j in edges)
    ]
    numbers = np.asarray([state.sum() for state in states], dtype=float)
    weights = activity**numbers
    probabilities = weights / weights.sum()

    def label(state):
        return int(state.sum() >= 2)

    labels = np.asarray([label(state) for state in states])
    availability = np.asarray(
        [
            sum(
                state[i] == 0
                and state[(i - 1) % vertex_count] == 0
                and state[(i + 1) % vertex_count] == 0
                for i in range(vertex_count)
            )
            / vertex_count
            for state in states
        ]
    )
    mean_number = probabilities @ numbers
    error_terms = []
    basin_probabilities = []
    for basin in (0, 1):
        indicator = (labels == basin).astype(float)
        q = probabilities @ indicator
        conditional_density = (probabilities @ (numbers * indicator)) / (
            vertex_count * q
        )
        conditional_insertion = (probabilities @ (availability * indicator)) / q
        signed_boundary = []
        absolute_boundary = []
        for state in states:
            differences = []
            for i in np.flatnonzero(state):
                deleted = state.copy()
                deleted[i] = 0
                differences.append(int(label(deleted) == basin) - int(label(state) == basin))
            signed_boundary.append(sum(differences))
            absolute_boundary.append(sum(abs(value) for value in differences))
        epsilon = (probabilities @ np.asarray(signed_boundary)) / (
            activity * vertex_count * q
        )
        delta = (probabilities @ np.asarray(absolute_boundary)) / mean_number
        np.testing.assert_allclose(
            conditional_insertion,
            conditional_density / activity + epsilon,
            rtol=3e-14,
            atol=3e-14,
        )
        assert abs(epsilon) <= mean_number * delta / (activity * vertex_count * q) + 1e-14
        conditional_number = (probabilities @ (numbers * indicator)) / q
        conditional_variance = (probabilities @ (numbers**2 * indicator)) / q - conditional_number**2
        direct_cavity_number = (
            probabilities @ (numbers * availability * indicator)
        ) / (probabilities @ (availability * indicator))
        signed_boundary = np.asarray(signed_boundary)
        measurable_derivative = probabilities @ (
            signed_boundary * (numbers - conditional_number - 1)
        )
        predicted_intraphase_response = (
            q * (conditional_variance - conditional_number) + measurable_derivative
        ) / (q * conditional_number + probabilities @ signed_boundary)
        np.testing.assert_allclose(
            direct_cavity_number - conditional_number,
            predicted_intraphase_response,
            rtol=3e-14,
            atol=3e-14,
        )
        error_terms.append(epsilon)
        basin_probabilities.append(q)

    # Exhaustiveness cancels the basin-boundary terms and recovers unrestricted GNZ.
    np.testing.assert_allclose(np.dot(basin_probabilities, error_terms), 0, atol=1e-14)
    np.testing.assert_allclose(
        probabilities @ availability,
        mean_number / (activity * vertex_count),
        rtol=3e-14,
        atol=3e-14,
    )

    # The same law holds for local structure, including a cavity-center-dependent
    # observable; particle-number size biasing alone cannot establish this.
    observables = (
        lambda center, state: float(state[0] * state[2]),
        lambda center, state: float(state[(center + 2) % vertex_count]),
    )
    insertion_probability = probabilities @ availability
    for observable in observables:
        direct_terms = []
        deletion_terms = []
        for state in states:
            direct_terms.append(
                sum(
                    observable(center, state)
                    for center in range(vertex_count)
                    if state[center] == 0
                    and state[(center - 1) % vertex_count] == 0
                    and state[(center + 1) % vertex_count] == 0
                )
                / vertex_count
            )
            deleted_sum = 0.0
            for particle in np.flatnonzero(state):
                deleted = state.copy()
                deleted[particle] = 0
                deleted_sum += observable(particle, deleted)
            deletion_terms.append(deleted_sum)
        np.testing.assert_allclose(
            (probabilities @ np.asarray(direct_terms)) / insertion_probability,
            (probabilities @ np.asarray(deletion_terms)) / mean_number,
            rtol=3e-14,
            atol=3e-14,
        )
