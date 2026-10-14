import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cavity_mc import (  # noqa: E402
    advance_sweeps,
    box_length,
    fcc_positions,
    minimum_pair_distance,
    production_blocks,
    summarize_block_probability,
)


def test_fcc_initializer_is_nonoverlapping_at_dense_fluid_limit() -> None:
    for n_particles in (108, 256):
        length = box_length(n_particles, 0.40)
        positions = fcc_positions(n_particles, length)
        assert positions.shape == (n_particles, 3)
        assert minimum_pair_distance(positions, length) > 1.0


def test_single_particle_mc_preserves_hard_sphere_constraint() -> None:
    n_particles = 108
    length = box_length(n_particles, 0.30)
    positions = fcc_positions(n_particles, length)
    n_cells = int(np.floor(length))
    accepted, attempted = advance_sweeps(
        positions,
        length,
        n_cells,
        0.18,
        50,
        918_273,
    )
    assert 0 < accepted < attempted
    assert minimum_pair_distance(positions, length) >= 1.0 - 2.0e-12


def test_random_probe_recovers_exact_half_diameter_cavity() -> None:
    n_particles = 108
    eta = 0.20
    length = box_length(n_particles, eta)
    positions = fcc_positions(n_particles, length)
    n_cells = int(np.floor(length))
    hits, _, _ = production_blocks(
        positions,
        length,
        n_cells,
        0.25,
        12,
        10,
        5,
        5_000,
        np.asarray([0.50, 0.75, 1.00]) ** 2,
        827_361,
    )
    trials_per_block = 10_000
    estimate = summarize_block_probability(hits[:, 0], trials_per_block)
    exact = 1.0 - eta
    assert abs(estimate["P_empty"] - exact) <= 4.0 * estimate["se_P_empty"]


def test_rare_event_gate_never_labels_zero_or_low_counts_reliable() -> None:
    zero = summarize_block_probability(np.zeros(25, dtype=int), 10_000)
    assert not zero["reliable_direct"]
    assert "zero_events" in zero["reliability_status"]
    assert np.isnan(zero["betaW_N"])
    assert not any("upper95" in key or "bound" in key for key in zero)

    sparse = np.zeros(25, dtype=int)
    sparse[:10] = 3
    low = summarize_block_probability(sparse, 10_000)
    assert low["empty_events"] == 30
    assert not low["reliable_direct"]
    assert "too_few_events" in low["reliability_status"]

    many_precise = np.full(25, 40, dtype=int)
    rare_but_resolved = summarize_block_probability(many_precise, 10_000_000)
    assert rare_but_resolved["P_empty"] < 1.0e-4
    assert rare_but_resolved["reliable_direct"]


def test_multiscale_blocking_never_understates_single_block_error() -> None:
    hits = np.asarray(
        [20] * 10 + [45] * 10 + [25] * 10 + [50] * 10,
        dtype=int,
    )
    summary = summarize_block_probability(hits, 10_000)
    assert summary["se_P_empty"] >= summary["se_P_empty_single_block"]
    assert summary["selected_block_factor"] in (1, 2, 5, 10)
    assert summary["effective_blocks_for_ci"] >= 4


def test_constant_but_different_chain_halves_are_not_stationary():
    hits = np.array([40]*25 + [80]*25)
    summary = summarize_block_probability(hits, 10000)
    assert np.isinf(summary["half_difference_z"])
    assert "block_half_difference" in summary["reliability_status"]
