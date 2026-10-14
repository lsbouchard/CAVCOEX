"""Count-only controls for the finite-cap empty-state minorization proof.

These tests do not sample a Markov chain or estimate a mixing rate. They
exhaustively check the combinatorial draining argument, allowing arbitrary
replica permutations and either outcome of a marked particle/cavity move.
"""

import itertools
import math

import pytest


def marked_counts(counts, cap):
    """Accepted marked update on base0/cavitylast, or identity if unavailable."""
    n, m = counts[0], counts[-1]
    if n == 0 or m + 1 > cap:
        return counts
    return (m + 1,) + counts[1:-1] + (n - 1,)


def test_marked_exchange_can_increase_maximum_but_preserves_total_number():
    before = (1, 4)
    after = marked_counts(before, 5)
    assert after == (5, 0)
    assert max(after) > max(before)
    assert sum(after) == sum(before)


def test_maximum_based_drain_bound_is_not_universal_with_marked_exchange():
    # Positive-probability marked acceptances plus suitable ordinary swaps
    # retain particles beyond ceil(K/m)=5 death-only sweeps.  Every active
    # host loses one particle before each permutation/marked update.
    counts, cap = (5, 5, 5), 5
    permutations = ((0, 1, 2), (1, 2, 0), (1, 2, 0), (0, 1, 2), (2, 1, 0))
    for permutation in permutations:
        drained = tuple(max(n - 1, 0) for n in counts)
        rearranged = tuple(drained[i] for i in permutation)
        counts = marked_counts(rearranged, cap)
    assert counts == (1, 0, 1)
    assert sum(counts) > 0


@pytest.mark.parametrize("replicas,cap,attempts", [(2, 4, 1), (3, 3, 1), (3, 4, 2), (4, 2, 3)])
def test_total_number_bound_drains_all_states_despite_swaps_and_marked_moves(replicas, cap, attempts):
    permutations = tuple(itertools.permutations(range(replicas)))
    reachable = set(itertools.product(range(cap + 1), repeat=replicas))
    steps = math.ceil(replicas * cap / attempts)
    for step in range(steps):
        subsequent = set()
        for counts in reachable:
            drained = tuple(max(n - attempts, 0) for n in counts)
            assert sum(drained) <= max(sum(counts) - attempts, 0)
            for permutation in permutations:
                rearranged = tuple(drained[i] for i in permutation)
                marked = marked_counts(rearranged, cap)
                assert sum(marked) == sum(rearranged)
                assert all(0 <= n <= cap for n in marked)
                subsequent.add(rearranged)  # Rejected or inactive marked move.
                subsequent.add(marked)      # Accepted marked move.
        reachable = subsequent
        assert all(sum(counts) <= max(replicas * cap - (step + 1) * attempts, 0)
                   for counts in reachable)
    assert reachable == {(0,) * replicas}
