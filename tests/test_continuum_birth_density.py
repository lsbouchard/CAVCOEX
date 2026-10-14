"""Independent geometric checks of the normalized continuum birth density."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import qmc

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from continuum_replica_exchange import extrapolation_birth_density


def independently_vectorized_density(points,positions,length,width,fraction):
    n = len(positions)
    if n < 2:
        return np.full(len(points),1/length**3)
    centers = np.array([2*positions[i]-positions[j] for i in range(n) for j in range(n) if i != j])
    difference = points[:,None,:]-centers[None,:,:]
    difference -= length*np.rint(difference/length)
    covers = (abs(difference)<width).all(axis=2).sum(axis=1)
    return fraction/length**3+(1-fraction)*covers/(n*(n-1)*(2*width)**3)


def test_marginal_density_counts_all_covering_pairs_and_reverse_deleted_host():
    positions = np.array([[.2,.2,.2],[1.2,1.2,1.2],[2.2,2.2,2.2]])
    density = extrapolation_birth_density.py_func
    before = density(positions[0],positions,3,3.,.15,.2)
    after = density(positions[0],positions,3,3.,.15,.2,0)
    assert before == pytest.approx(.2/27+.8*2/(6*.3**3))
    assert after == pytest.approx(.2/27+.8/.3**3)
    assert after == density(positions[0],positions[1:],2,3.,.15,.2)
    assert after != before


def test_density_permutation_invariance_translation_covariance_and_positive_support():
    rng = np.random.default_rng(615349)
    positions = rng.uniform(0,3.,(8,3))
    points = rng.uniform(0,3.,(100,3))
    shift = np.array([1.19,-.57,2.63])
    density = extrapolation_birth_density.py_func
    expected = independently_vectorized_density(points,positions,3.,.4,.2)
    observed = np.array([density(p,positions,8,3.,.4,.2) for p in points])
    translated = np.array([density((p+shift)%3,(positions+shift)%3,8,3.,.4,.2) for p in points])
    permuted = np.array([density(p,positions[::-1],8,3.,.4,.2) for p in points])
    np.testing.assert_allclose(observed,expected,atol=1e-15,rtol=0)
    np.testing.assert_allclose(translated,expected,atol=1e-15,rtol=0)
    np.testing.assert_allclose(permuted,expected,atol=1e-15,rtol=0)
    assert (observed>=.2/27).all()


def test_periodic_cube_density_integrates_to_one_independently():
    positions = np.array([[.1,2.9,.2],[2.8,.2,2.8],[1.1,1.3,.6]])
    points = 3*qmc.Sobol(3,scramble=True,seed=615351).random_base2(16)
    values = independently_vectorized_density(points,positions,3.,.5,.2)
    assert abs(27*values.mean()-1)<.006
    # Compare the actual scalar implementation on a nontrivial subgrid.
    scalar = np.array([extrapolation_birth_density.py_func(p,positions,3,3.,.5,.2)
                       for p in points[::64]])
    np.testing.assert_allclose(scalar,values[::64],atol=1e-15,rtol=0)


def test_zero_one_particle_and_pure_uniform_limits():
    positions = np.array([[.2,.4,.6],[1.1,1.3,1.5]])
    density = extrapolation_birth_density.py_func
    for number,ignore,fraction in ((0,-1,.2),(1,-1,.2),(2,0,.2),(2,-1,1.)):
        assert density(positions[0],positions,number,3.,.12,fraction,ignore) == 1/27
    with pytest.raises(ValueError,match="deleted index"):
        density(positions[0],positions,2,3.,.12,.2,2)
