"""Independent conditional-score checks for the actual label Gibbs helper."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pytest
from scipy.special import logsumexp

from scripts import continuum_simulated_tempering as tempering


def _target():
    return (np.log([.06, .06, .11, .02, .004]), np.array([0., 1., 0., .7, 0.]),
            np.array([np.inf, np.inf, .7, 8., 0.]), np.array([0., 0., 2.5, 0., 0.]),
            np.array([0., 0., .7, 0., 0.]), np.array([0., 0., .8, .5, 0.]),
            np.array([0., 0., 2.2, .6, 0.]), np.array([2., 1.5, -.8, 3.1, .2]))


def _independent_probabilities(n, order, overlaps, minimum, arrays):
    x, radius, pair, stiffness, preferred, number, nstar, weights = arrays
    log_masses = []
    for node in range(len(x)):
        supported = ((n == 0 or minimum >= radius[node])
                     and (overlaps == 0 or np.isfinite(pair[node])))
        if supported:
            pair_cost = 0. if np.isinf(pair[node]) else pair[node] * overlaps
            cost = (pair_cost + stiffness[node] * (order - preferred[node]) ** 2
                    + number[node] * (n - nstar[node]) ** 2)
            log_masses.append(weights[node] + x[node] * n - cost)
        else:
            log_masses.append(-np.inf)
    masses = np.array(log_masses)
    return np.exp(masses - logsumexp(masses))


@pytest.mark.parametrize("n,order,overlaps,minimum", [
    (0, 0., 0, -1.),  # Empty-host cavity support must ignore the sentinel.
    (1, 1., 0, .2),
    (2, .37, 1, 1.3),
    (3, .82, 0, 1.4),
])
def test_actual_gibbs_helper_matches_independent_supported_costs(n, order, overlaps, minimum):
    target = _target()
    observed = tempering.conditional_label_probabilities(n, order, overlaps, minimum, *target)
    expected = _independent_probabilities(n, order, overlaps, minimum, target)
    np.testing.assert_allclose(observed, expected, atol=2e-15, rtol=2e-14)
    assert np.isfinite(observed).all() and (observed >= 0).all()
    np.testing.assert_allclose(observed.sum(), 1., atol=2e-15, rtol=0)
    assert np.array_equal(observed == 0, expected == 0)


def test_actual_gibbs_helper_is_gauge_invariant_and_stable_at_large_scores():
    target = _target()
    baseline = tempering.conditional_label_probabilities(2, .4, 0, 1.2, *target)
    shifted = (*target[:-1], target[-1] + 10000.)
    changed = tempering.conditional_label_probabilities(2, .4, 0, 1.2, *shifted)
    np.testing.assert_allclose(changed, baseline, atol=3e-13, rtol=2e-12)
    dominant = (*target[:-1], np.array([1000., -1000., 0., -800., 500.]))
    observed = tempering.conditional_label_probabilities(2, .4, 0, 1.2, *dominant)
    expected = _independent_probabilities(2, .4, 0, 1.2, dominant)
    np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0)
    assert np.isfinite(observed).all() and observed.sum() == 1.


def test_actual_gibbs_helper_rejects_all_invalid_scores():
    target = _target()
    invalid = (target[0], target[1], np.full(5, np.inf), *target[3:])
    with pytest.raises(ValueError, match="No supported"):
        tempering.conditional_label_probabilities(2, .4, 1, 1.2, *invalid)


def test_actual_categorical_draw_never_selects_zero_support_and_samples_its_law():
    probability = np.array([0., .2, 0., .8, 0.])
    tempering.kernel.seed_rng(260930512)
    counts = np.bincount([tempering.draw_conditional_label(probability)
                          for _ in range(30000)], minlength=5)
    assert counts[0] == counts[2] == counts[4] == 0
    np.testing.assert_allclose(counts / counts.sum(), probability, atol=.012, rtol=0)
    # The helper's rounding fallback must still return the last supported
    # label, not a zero-support trailing array entry.
    near_one = np.array([0., np.nextafter(1., 0.), 0.])
    assert all(tempering.draw_conditional_label(near_one) == 1 for _ in range(100))
