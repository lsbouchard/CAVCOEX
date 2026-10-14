"""Independent statistical calibration controls for covariance-split auditing."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reaud_covariance_mode_stability import split_covariance_test


def test_covariance_estimation_and_calibration_sets_cannot_overlap():
    draws = np.random.default_rng(17).normal(size=(100, 3))
    with pytest.raises(ValueError, match="disjoint"):
        split_covariance_test(np.zeros(3), draws, np.arange(60), np.arange(40, 100), 1e-8)


def test_independent_gaussian_calibration_recovers_chi_square_moments():
    # The quadratic distance of a 3-dimensional normal vector has mean 3
    # and 95% quantile 7.8147 when its covariance is known. Independent
    # estimation of a well-sampled covariance should approach those values.
    draws = np.random.default_rng(81811).normal(size=(40000, 3))
    result = split_covariance_test(np.zeros(3), draws,
                                  np.arange(20000), np.arange(20000, 40000), 1e-8)
    assert result["rank"] == 3
    assert abs(result["calibration_mean_Q"] - 3.) < .06
    assert abs(result["calibration_q95"] - 7.814727903) < .2


def test_split_statistics_are_invariant_to_observable_units():
    draws = np.random.default_rng(4301).normal(size=(4000, 3))
    central = np.asarray([.2, -.5, .7])
    fitting, calibration = np.arange(2000), np.arange(2000, 4000)
    baseline = split_covariance_test(central, draws, fitting, calibration, 1e-8)
    units = np.asarray([1e-8, 37., 1e9])
    rescaled = split_covariance_test(central * units, draws * units, fitting, calibration, 1e-8)
    np.testing.assert_allclose(rescaled["Q"], baseline["Q"], rtol=2e-14)
    np.testing.assert_allclose(rescaled["calibration_mean_Q"], baseline["calibration_mean_Q"], rtol=2e-14)
    assert rescaled["disjoint_calibration_p_value"] == baseline["disjoint_calibration_p_value"]
