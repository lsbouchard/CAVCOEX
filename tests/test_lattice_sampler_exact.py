"""Independent equilibrium tests for the sampler, not just estimator algebra."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lattice_exact_controls import enumerate_microstates
from lattice_gas_phase_switch import K_ISING, _wolff_series


def test_wolff_equilibrium_matches_fully_enumerated_four_by_four_lattice():
    states = enumerate_microstates(4)
    log_weight = K_ISING * states.bond_sum
    weight = np.exp(log_weight - log_weight.max())
    weight /= weight.sum()
    magnetization, bonds, _, _, _ = _wolff_series(4, K_ISING, 29092026, 2000, 50000, 1)
    for sampled, exact_values in ((np.abs(magnetization), np.abs(states.magnetization)),
                                  (bonds, states.bond_sum)):
        exact = float(weight @ exact_values)
        block_means = sampled.reshape(100, 500).mean(axis=1)
        se = block_means.std(ddof=1)/np.sqrt(len(block_means))
        assert abs(float(sampled.mean()) - exact) < 6.0 * se


def test_current_wolff_kernel_replays_saved_trajectory_prefix():
    root = Path(__file__).resolve().parents[1]
    length, replica = 16, 0
    seed = 25072026 + 1009*length + 104729*replica
    actual = _wolff_series(length, K_ISING, seed, 2000, 64, 1)
    fields = ("magnetization", "bond_sum", "site_zero", "site_right", "cluster_size")
    with np.load(root/"data_raw/lattice_gas_phase_switch_trajectories.npz") as stored:
        for field, value in zip(fields, actual):
            np.testing.assert_array_equal(value, stored[f"L{length}_replica{replica}_{field}"][:64])
