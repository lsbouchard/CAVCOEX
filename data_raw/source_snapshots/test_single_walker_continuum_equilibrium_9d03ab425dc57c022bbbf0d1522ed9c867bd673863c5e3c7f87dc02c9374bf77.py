"""Independent small continuum target controls for the single-host engine.

The cap is deliberately N<=3, and therefore does NOT meet the physical packing
bound. All expected laws are the explicit capped partition integrals. No
unrestricted reduced-Palm/GNZ number identity or dense-coexistence claim is used.
"""
import math
import os
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
from scipy.stats import qmc

from scripts import continuum_simulated_tempering as tempering

TEST_EXECUTION_SOURCE_BYTES = Path(__file__).read_bytes()


VECTORS = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                    [2, 0, 0], [0, 2, 0], [0, 0, 2]])


def _independent_partition_table(length, activity, cap, cells, penalty, kappa, preferred, nu, nstar):
    """9D Sobol integration, with exactly known n=0,1 and ideal sectors."""
    assert cap == 3
    volume, vectors = length ** 3, cells * VECTORS
    points = ((qmc.Sobol(9, scramble=True, seed=260929401).random_base2(17)
               .reshape((-1, 3, 3)) - .5) * length)
    clear = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        difference = points[:, first] - points[:, second]
        difference -= length * np.rint(difference / length)
        clear.append(np.linalg.norm(difference, axis=1) >= 1.)
    clear = np.array(clear).T
    cavity_clear = np.linalg.norm(points, axis=2) >= 1.
    sector = np.zeros((5, cap + 1))
    order_numerator = np.zeros(5)
    for node in range(5):
        sector[node, 0] = math.exp(-kappa[node] * preferred[node] ** 2
                                   - nu[node] * nstar[node] ** 2)
        accessible = volume - 4 * np.pi / 3 if node == 1 else volume
        sector[node, 1] = activity * accessible * math.exp(
            -kappa[node] * (1 - preferred[node]) ** 2
            -nu[node] * (1 - nstar[node]) ** 2)
        order_numerator[node] = sector[node, 1]
    sums = np.exp(2j * np.pi / length * (points[:, 0] @ vectors.T))
    for n in (2, 3):
        sums += np.exp(2j * np.pi / length * (points[:, n - 1] @ vectors.T))
        order = np.abs(sums).mean(axis=1) / n
        hard_mask = clear[:, 0] if n == 2 else clear.all(axis=1)
        overlaps = (~clear[:, 0]).astype(float) if n == 2 else (~clear).sum(axis=1)
        factor = (activity * volume) ** n / math.factorial(n)
        for node in range(5):
            if np.isinf(penalty[node]):
                pair = hard_mask.astype(float)
            else:
                pair = np.exp(-penalty[node] * overlaps)
            support = cavity_clear[:, :n].all(axis=1) if node == 1 else np.ones(len(points))
            density = (pair * support * np.exp(-kappa[node] * (order - preferred[node]) ** 2
                                               - nu[node] * (n - nstar[node]) ** 2))
            sector[node, n] = factor * density.mean()
            order_numerator[node] += factor * (order * density).mean()
    # The ideal number law is analytic, not an empirical integration result.
    np.testing.assert_allclose(sector[4], [(activity * volume) ** n / math.factorial(n)
                                         for n in range(cap + 1)], atol=1e-14, rtol=0)
    return sector, order_numerator / sector.sum(axis=1)


def _optional_export(result, coordinate_centers, positions, counts, centers, sectors,
                     expected_joint, target, log_weights, seed):
    """Export only when explicitly requested; never silently overwrite a file."""
    destination = os.environ.get("CAVCOEX_ST_CONTROL_EXPORT")
    if not destination:
        return
    path = Path(destination).resolve()
    sidecar = path.with_suffix(".json")
    if not path.is_relative_to(tempering.ROOT) or path.suffix != ".npz":
        raise ValueError("Control export must be an explicit .npz path within the project")
    if path.exists() or sidecar.exists():
        raise ValueError("Independent control export may not overwrite an existing artifact")
    trace, local, moves, configurations, clocks, labels, node, cycle = result
    sources = {name: tempering.capture_source(source, content) for name, source, content in (
        ("driver", tempering.__file__, tempering.EXECUTION_SOURCE_BYTES),
        ("kernel", tempering.kernel.__file__, tempering.kernel.EXECUTION_SOURCE_BYTES),
        ("graph_validation", tempering.replica_validation.__file__, tempering.VALIDATION_SOURCE_BYTES),
        ("independent_control", __file__, TEST_EXECUTION_SOURCE_BYTES),
    )}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=tempering.SCHEMA, samples=trace, log_weights=log_weights,
        sites=32, cap=3, coordinate_configurations=configurations,
        coordinate_clock_steps=clocks, coordinate_labels=labels, coordinate_centers=coordinate_centers,
        local_accepts=local, label_move_counts=moves, final_positions=positions,
        final_counts=counts, final_centers=centers, final_label=node, final_cycle=cycle,
        final_clock=155000, control_partition_sectors=sectors,
        control_expected_joint=expected_joint, **target)
    metadata = {
        "schema": tempering.SCHEMA,
        "settings": {"production_steps": 150000, "calibration_steps": 5000,
                     "coordinate_stride": 7, "seed": seed, "graph_from": None,
                     "weights_from": None},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "columns": tempering.COLUMNS,
        "source_snapshots": sources,
        "input_sha256": {},
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "calibration_cutoff_clock": 5000,
        "sample_cadence_clock_steps": 1,
        "weights_frozen_before_production": True,
        "deletion_label_threshold": .45,
        "control_scope": "Independent low-N implementation regression, not continuum coexistence evidence",
        "cap_scope": "Deliberately truncated N<=3 toy; compare only with its explicit partition integrals, not unrestricted Palm/GNZ identities",
        "partition_quadrature": {"dimension": 9, "sobol_points": 2 ** 17,
                                  "scramble_seed": 260929401},
        "sampling_stage": "After local update, before lazy label hop; every full-clock dwell retained",
        "reproduce_argv": ["env", "CAVCOEX_ST_CONTROL_EXPORT=" + str(path),
                            "python3", "-m", "pytest", "-q", __file__,
                            "-k", "actual_single_host"],
    }
    sidecar.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def test_actual_single_host_matches_capped_joint_and_conditional_partition_laws():
    length, activity, cap, cells = 3., .06, 3, 2
    penalty = np.array([np.inf, np.inf, .7, 8., 0.])
    kappa = np.array([0., 0., 2.5, 0., 0.])
    preferred = np.array([0., 0., .7, 0., 0.])
    nu = np.array([0., 0., .8, .5, 0.])
    nstar = np.array([0., 0., 2.2, .6, 0.])
    sectors, expected_order = _independent_partition_table(
        length, activity, cap, cells, penalty, kappa, preferred, nu, nstar)
    z = sectors.sum(axis=1)
    conditional_number = sectors / z[:, None]
    # Deliberately nonuniform node frequencies, despite an exactly normalized
    # input table: arbitrary finite weights cannot alter a conditional law.
    node_weight = np.array([.7, 1.3, 2.1, .8, 1.7])
    log_weights = np.log(node_weight) - np.log(z)
    joint_expected = node_weight[:, None] / node_weight.sum() * conditional_number
    edges = np.array([[0, 2], [2, 3], [3, 4], [4, 1], [0, 1], [2, 4]], dtype=np.int64)
    neighbors, edge_table, degrees = tempering.adjacency(edges, 5)
    assert set(degrees) == {2, 3}  # The actual engine must use its degree correction.
    positions = np.zeros((1, cap, 3))
    counts = np.zeros(1, dtype=np.int64)
    centers = np.zeros((1, 3))
    diameter = np.ones(5)
    radius = np.array([0., 1., 0., 0., 0.])
    fields = np.full(5, math.log(activity))
    displacement = np.full(5, .8)
    physical = np.array([True, True, False, False, False])
    reference = tempering.kernel.fcc_reference(4 * cells ** 3, length)
    cdf = tempering.capped_poisson_cdf(math.log(activity * length ** 3), cap)
    arguments = (positions, counts, centers)
    tail = (reference, length, diameter, radius, fields, penalty, kappa, preferred,
            nu, nstar, displacement, physical, 0, 1, 4, neighbors, edge_table,
            degrees, log_weights, cdf)
    tempering.kernel.seed_rng(260929402)
    warm = tempering.run_tempering_chunk(*arguments, 4, 0, 0, 5000, *tail,
                                         coordinate_stride=7, biased_birth=True,
                                         birth_width=.3, birth_uniform_fraction=.2)
    node, cycle = warm[-2:]
    coordinate_centers = np.empty((len(np.arange(7 - (5000 % 7) - 1, 150000, 7)), 3))
    result = tempering.run_tempering_chunk(*arguments, node, cycle, 5000, 150000, *tail,
                                           coordinate_stride=7, biased_birth=True,
                                           birth_width=.3, birth_uniform_fraction=.2,
                                           coordinate_centers=coordinate_centers)
    trace, local, moves, configurations, clocks, coordinate_labels, _, _ = result
    assert trace.shape == (150000, 12)
    assert np.array_equal(trace[:, 0], np.arange(5001, 155001))
    assert np.array_equal(trace[:, 1], np.rint(trace[:, 1]))
    assert np.array_equal(trace[:, 2], np.rint(trace[:, 2]))
    assert np.isfinite(trace).all()
    labels, numbers = trace[:, 1].astype(int), trace[:, 2].astype(int)
    joint = np.bincount(labels * (cap + 1) + numbers, minlength=5 * (cap + 1))
    joint = joint.reshape((5, cap + 1)) / len(trace)
    np.testing.assert_allclose(joint, joint_expected, rtol=0, atol=.012)
    np.testing.assert_allclose(joint.sum(axis=1), node_weight / node_weight.sum(),
                               rtol=0, atol=.015)
    for label in range(5):
        selected = labels == label
        assert selected.sum() > 10000
        observed = np.bincount(numbers[selected], minlength=cap + 1) / selected.sum()
        np.testing.assert_allclose(observed, conditional_number[label], rtol=0, atol=.025)
        np.testing.assert_allclose(trace[selected, 3].mean(), expected_order[label],
                                   rtol=0, atol=.025)
    # The selected physical cap is intentionally populated, exercising rejection
    # of a reverse tagged restoration from a cavity host at N=K.
    assert np.count_nonzero((labels == 1) & (numbers == cap)) > 500
    assert local[3] > 10000
    tagged_edge = np.flatnonzero(np.all(edges == [0, 1], axis=1))[0]
    assert moves[tagged_edge, 1] > 1000
    assert (moves[:, 1] > 1000).all()
    assert np.all(trace[(labels == 0) | (labels == 1), 4] == 0)
    assert np.all(trace[labels == 1, 7] >= 1.)
    # At the ideal node, each conditional center-to-host void law is also known.
    for n in (1, 2, 3):
        selected = (labels == 4) & (numbers == n)
        probability = 1 - (1 - 4 * np.pi * .7 ** 3 / (3 * length ** 3)) ** n
        np.testing.assert_allclose(np.mean(trace[selected, 7] < .7), probability,
                                   rtol=0, atol=.02)
    events = trace[:, 11].astype(bool)
    assert events.sum() > 20000
    assert np.all(labels[events] == 4)
    cycles = trace[:, 10].astype(int)
    assert np.array_equal(np.diff(cycles), events[1:].astype(int))
    assert np.all(clocks % 7 == 0)
    rows = clocks.astype(int) - 5001
    assert np.array_equal(coordinate_labels, labels[rows])
    # Independent Fourier reconstruction, every saved active host, all labels.
    vectors = cells * VECTORS
    physical_counts_seen = set()
    for index, row in enumerate(rows):
        n = numbers[row]
        active = configurations[index, :n]
        assert np.isnan(configurations[index, n:]).all()
        assert np.isfinite(active).all() and np.all((active >= 0) & (active < length))
        b = np.abs(np.exp(2j * np.pi / length * (active @ vectors.T)).mean(axis=0)).mean() if n else 0.
        np.testing.assert_allclose(b, trace[row, 3], atol=1e-10, rtol=0)
        difference = active - coordinate_centers[index]
        difference -= length * np.rint(difference / length)
        minimum_distance = np.linalg.norm(difference, axis=1).min() if n else length
        np.testing.assert_allclose(minimum_distance, trace[row, 7], atol=1e-10, rtol=0)
        if coordinate_labels[index] in (0, 1):
            physical_counts_seen.add(n)
            baseline = int(b > .45)
            deletion_shift = 0
            for tag in range(n):
                remainder = np.delete(active, tag, axis=0)
                after_b = np.abs(np.exp(2j * np.pi / length * (remainder @ vectors.T))
                                    .mean(axis=0)).mean() if n > 1 else 0.
                deletion_shift += int(after_b > .45) - baseline
            assert trace[row, 8] == deletion_shift
            if coordinate_labels[index] == 1:
                assert minimum_distance >= 1.
            for first in range(n):
                for second in range(first):
                    difference = active[first] - active[second]
                    difference -= length * np.rint(difference / length)
                    assert np.linalg.norm(difference) >= 1.
    assert physical_counts_seen == {0, 1, 2, 3}
    _optional_export(result, coordinate_centers, positions, counts, centers, sectors,
                     joint_expected,
                     dict(volume=length ** 3, diameter=diameter, radius=radius, activity=fields,
                          pair_penalty=penalty, umbrella_stiffness=kappa, umbrella_preferred=preferred,
                          number_stiffness=nu, number_preferred=nstar, displacement=displacement,
                          exchange_edges=edges, base_node=0, cavity_node=1, ideal_node=4,
                          physical_nodes=np.array([0, 1], dtype=np.int64)),
                     log_weights, 260929402)


def test_designated_heatbath_erases_incoming_host_memory_and_shared_rng_is_reproducible():
    length, cap = 3., 3
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    neighbors, edge_table, degrees = tempering.adjacency(edges, 3)
    reference = tempering.kernel.fcc_reference(32, length)
    fields = np.full(3, math.log(.06))
    cdf = tempering.capped_poisson_cdf(fields[2] + math.log(length ** 3), cap)
    tail = (reference, length, np.ones(3), np.array([0., 1., 0.]), fields,
            np.array([np.inf, np.inf, 0.]), np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3), np.full(3, .8),
            np.array([True, True, False]), 0, 1, 2, neighbors, edge_table, degrees,
            np.array([0., .3, -.4]), cdf)
    outputs, states = [], []
    for initial_n, initial_coordinate in ((0, 0.), (3, 1.23)):
        positions = np.full((1, cap, 3), initial_coordinate)
        counts = np.array([initial_n], dtype=np.int64)
        centers = np.full((1, 3), initial_coordinate)
        tempering.kernel.seed_rng(260929403)
        result = tempering.run_tempering_chunk(positions, counts, centers, 2, 0, 0, 2000,
                                               *tail, coordinate_stride=7, biased_birth=True,
                                               birth_width=.3, birth_uniform_fraction=.2)
        outputs.append(result)
        n = counts[0]
        states.append((positions[0, :n].copy(), counts.copy(), centers.copy()))
    assert outputs[0][0][0, 11] == 1  # True full refresh before any first local observation.
    for index in range(6):
        np.testing.assert_array_equal(outputs[0][index], outputs[1][index])
    assert outputs[0][-2:] == outputs[1][-2:]
    for first, second in zip(states[0], states[1]):
        np.testing.assert_array_equal(first, second)
