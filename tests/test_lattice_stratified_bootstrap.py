"""Sampling-design tests independent of desired scientific outcomes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lattice_analysis_common import stratified_block_bootstrap_totals
from lattice_pulse_correction_prediction import resample_total
from lattice_validation_statistics import (
    _hierarchical_totals_common_modes, _resample_subset_total,
)


def test_strata_are_retained_exactly_once():
    # Constant but distinct chains have zero within-chain sampling noise.
    # A nested replica/block bootstrap incorrectly adds between-chain noise.
    blocks = np.broadcast_to(np.array([1.0, 7.0, 12.0])[:, None, None], (3, 8, 1))
    draws = np.asarray(list(stratified_block_bootstrap_totals(
        blocks, 100, np.random.default_rng(919))))
    np.testing.assert_array_equal(draws, np.full((100, 1), 160.0))
    np.testing.assert_array_equal(resample_total(blocks, np.random.default_rng(20)), [160.0])
    np.testing.assert_array_equal(
        _resample_subset_total(blocks, [0, 2], np.random.default_rng(21)), [104.0])


def test_bootstrap_variance_matches_exact_conditional_sampling_variance():
    rng = np.random.default_rng(2001)
    blocks = rng.normal(size=(12, 40, 2))
    exact_covariance = sum(np.cov(chain, rowvar=False, ddof=0) * 40
                           for chain in blocks)
    draws = np.asarray(list(stratified_block_bootstrap_totals(
        blocks, 12000, np.random.default_rng(2002))))
    estimated = np.cov(draws, rowvar=False, ddof=1)
    np.testing.assert_allclose(np.diag(estimated), np.diag(exact_covariance), rtol=0.04)
    assert abs(estimated[0, 1] - exact_covariance[0, 1]) < 0.035 * exact_covariance[0, 0]


def test_paired_modes_use_the_same_block_draw():
    base = np.arange(3*5*2, dtype=float).reshape(3, 5, 2)
    for totals in _hierarchical_totals_common_modes(
        {"a": base, "b": 3.0 * base}, 30, np.random.default_rng(817)):
        np.testing.assert_allclose(totals["b"], 3.0 * totals["a"], rtol=0, atol=0)
