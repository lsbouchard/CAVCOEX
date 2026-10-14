"""Independent Fourier-sum controls for variable-N umbrella cache proposals.

The reference calculation uses complex exponentials, not the sampler's
trigonometric helper.  Empty/singleton configurations, rejected proposals,
current-N normalization, and shared scratch-buffer reuse are exercised.
"""

import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import bragg_proposal_into, bragg_value


VECTORS = np.array(
    [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
     [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float,
)


def independent_modes(points, length, cells):
    points = np.asarray(points).reshape((-1, 3))
    return np.exp(2j * np.pi * cells / length * (points @ VECTORS.T)).sum(axis=0)


def assert_cache(re, im, points, length, cells):
    expected = independent_modes(points, length, cells)
    np.testing.assert_allclose(re + 1j * im, expected, atol=2e-12, rtol=0)
    expected_order = abs(expected).mean() / len(points) if len(points) else 0.
    np.testing.assert_allclose(bragg_value(re, im, len(points)), expected_order,
                               atol=2e-13, rtol=0)


def test_random_variable_number_cache_proposals_match_independent_modes():
    rng = np.random.default_rng(188505)
    length, cells = 3.7, 2
    points = []
    re, im = np.zeros(7), np.zeros(7)
    proposal_re, proposal_im = np.empty(7), np.empty(7)
    unused = np.full(3, np.nan)
    # Explicitly traverse N=0 -> 1 -> 0 before the random sequence.
    actions = ["birth", "death"] + rng.choice(
        ["birth", "death", "move"], size=400).tolist()
    for index, action in enumerate(actions):
        if not points:
            action = "birth"
        if action == "birth":
            new_point = rng.uniform(0, length, size=3)
            old_point, old_sign, new_sign = unused, 0., 1.
            proposed_points = points + [new_point]
        else:
            particle = rng.integers(len(points))
            old_point, old_sign = points[particle], -1.
            proposed_points = points.copy()
            if action == "death":
                new_point, new_sign = unused, 0.
                proposed_points.pop(particle)
            else:
                new_point = rng.uniform(0, length, size=3)
                new_sign = 1.
                proposed_points[particle] = new_point
        source_re, source_im = re.copy(), im.copy()
        bragg_proposal_into(re, im, old_point, new_point, old_sign, new_sign,
                            length, cells, proposal_re, proposal_im)
        assert_cache(proposal_re, proposal_im, proposed_points, length, cells)
        np.testing.assert_array_equal(re, source_re)
        np.testing.assert_array_equal(im, source_im)
        # Rejection must leave both the source cache and coordinates intact.
        if index >= 2 and rng.random() < .2:
            assert_cache(re, im, points, length, cells)
        else:
            re[:], im[:] = proposal_re, proposal_im
            points = proposed_points
            assert_cache(re, im, points, length, cells)


@njit
def reuse_scratch_after_accepted_row_copy(modes_re, modes_im, points, length, cells):
    """Exercise the same Numba row assignments used in the production kernel."""
    scratch_re, scratch_im = np.empty(7), np.empty(7)
    bragg_proposal_into(modes_re[0], modes_im[0], points[0], points[0],
                        0., 1., length, cells, scratch_re, scratch_im)
    modes_re[0] = scratch_re
    modes_im[0] = scratch_im
    bragg_proposal_into(modes_re[1], modes_im[1], points[1], points[1],
                        0., 1., length, cells, scratch_re, scratch_im)
    modes_re[1] = scratch_re
    modes_im[1] = scratch_im
    # A discarded proposal must not overwrite either accepted row.
    bragg_proposal_into(modes_re[0], modes_im[0], points[0], points[2],
                        -1., 1., length, cells, scratch_re, scratch_im)
    # A swap must move real and imaginary modes together.
    temporary = modes_re[0].copy()
    modes_re[0] = modes_re[1]
    modes_re[1] = temporary
    temporary = modes_im[0].copy()
    modes_im[0] = modes_im[1]
    modes_im[1] = temporary


def test_numba_row_copies_do_not_alias_reused_proposal_buffers():
    length, cells = 3.7, 2
    points = np.array([[.13, 1.7, 2.3], [2.1, .7, .92], [.4, .8, 1.1]])
    modes_re, modes_im = np.zeros((2, 7)), np.zeros((2, 7))
    reuse_scratch_after_accepted_row_copy(modes_re, modes_im, points, length, cells)
    assert_cache(modes_re[0], modes_im[0], [points[1]], length, cells)
    assert_cache(modes_re[1], modes_im[1], [points[0]], length, cells)


def test_cache_proposal_and_reverse_restore_original_configuration():
    length, cells = 3.7, 2
    points = np.array([[.13, 1.7, 2.3], [2.1, .7, .92], [.4, .8, 1.1]])
    modes = independent_modes(points, length, cells)
    re, im = modes.real.copy(), modes.imag.copy()
    next_re, next_im = np.empty(7), np.empty(7)
    back_re, back_im = np.empty(7), np.empty(7)
    moved = np.array([3.6, .11, 1.4])
    bragg_proposal_into(re, im, points[1], moved, -1., 1.,
                        length, cells, next_re, next_im)
    bragg_proposal_into(next_re, next_im, moved, points[1], -1., 1.,
                        length, cells, back_re, back_im)
    np.testing.assert_allclose(back_re + 1j * back_im, modes, atol=2e-14, rtol=0)
