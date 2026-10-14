"""Pure finite controls for the new soft-pair/hard normalizer bound.

No sampler imports or Monte Carlo. All partition and TV comparisons are exact
Fractions on a finite point space. The graph is a probability control, not a
fluid--solid model. The common cap may intentionally truncate the hard law.
"""
import itertools
import math
from fractions import Fraction as F

import pytest


VERTICES = tuple(range(4))
EDGES = {(0,1),(1,2),(2,3),(0,3),(0,2)}


def hosts(cap, vertices=VERTICES):
    for mask in range(1 << len(vertices)):
        host = frozenset(vertices[index] for index in range(len(vertices)) if mask & (1 << index))
        if len(host) <= cap:
            yield host


def edges_in(host, edges=EDGES):
    return sum(set(edge) <= host for edge in edges)


def subsets(host):
    values = tuple(host)
    for mask in range(1 << len(values)):
        yield frozenset(values[index] for index in range(len(values)) if mask & (1 << index))


def cavity_allowed(host, center, cavity):
    if not cavity:
        return True
    excluded = {center} | {second if first == center else first
                           for first,second in EDGES if center in (first,second)}
    return not host.intersection(excluded)


def laws(activity, attenuation, cap, cavity):
    """attenuation=exp(-gamma), with gamma times unordered overlap count."""
    soft, hard, hard_sector = {}, {}, [F(0)] * (cap + 1)
    for host in hosts(cap):
        overlaps = edges_in(host)
        for center in VERTICES:
            if cavity_allowed(host,center,cavity):
                # Unordered subsets already contain their sector normalization.
                reference = activity ** len(host) / len(VERTICES)
                soft[(host,center)] = reference * attenuation ** overlaps
                hard[(host,center)] = reference if overlaps == 0 else F(0)
                if overlaps == 0:
                    hard_sector[len(host)] += reference
    return soft, hard, hard_sector


def exponential_polynomial(value, cap):
    return sum((value ** count / math.factorial(count) for count in range(cap + 1)), F(0))


def exponential_upper(value, terms=30):
    value = F(value)
    assert 0 <= value < terms + 2
    polynomial = exponential_polynomial(value,terms)
    next_term = value ** (terms + 1) / math.factorial(terms + 1)
    return polynomial + next_term / (1 - value / (terms + 2))


@pytest.mark.parametrize('attenuation', [F(0),F(1,100),F(1,3),F(1)])
def test_vertex_cover_pointwise_majorant(attenuation):
    for host in hosts(4):
        overlaps = edges_in(host)
        cover_sum = sum((attenuation ** len(removed) for removed in subsets(host)
                         if edges_in(host - removed) == 0), F(0))
        assert attenuation ** overlaps <= cover_sum
        # Selecting one endpoint from every induced edge yields a cover of size
        # at most the number of edges, including the empty cover when P=0.
        selected = {first for first,second in EDGES if {first,second} <= host}
        assert len(selected) <= overlaps and edges_in(host - selected) == 0


@pytest.mark.parametrize('cap', [0,1,2,3,4])
@pytest.mark.parametrize('cavity', [False,True])
@pytest.mark.parametrize('activity,attenuation', [(F(1,10),F(1,3)),(F(2),F(1,100)),(F(10),F(1,1000))])
def test_exact_capped_convolution_and_total_variation_bound(cap,cavity,activity,attenuation):
    soft,hard,sector = laws(activity,attenuation,cap,cavity)
    soft_normalizer,hard_normalizer = sum(soft.values()),sum(hard.values())
    value = activity * len(VERTICES) * attenuation
    convolution = sum((sector[remaining] * exponential_polynomial(value,cap - remaining)
                       for remaining in range(cap + 1)), F(0))
    envelope = exponential_polynomial(value,cap)
    assert 0 < hard_normalizer <= soft_normalizer <= convolution <= hard_normalizer * envelope
    # Conditioning the soft law on P=0 is exactly the hard law, with or without
    # a truncating common cap and with the same normalized cavity-center measure.
    soft_hard_mass = sum((weight for (host,_),weight in soft.items() if edges_in(host) == 0),F(0))
    assert soft_hard_mass == hard_normalizer
    tv = sum((abs(soft[state] / soft_normalizer - hard[state] / hard_normalizer)
              for state in soft),F(0)) / 2
    assert tv == 1 - hard_normalizer / soft_normalizer
    assert tv <= 1 - 1 / envelope


def test_subset_binomial_counts_cancel_the_common_sector_factorial():
    for remaining in range(6):
        for removed in range(6 - remaining):
            total = remaining + removed
            assert F(math.comb(total,removed),math.factorial(total)) == F(
                1,math.factorial(remaining) * math.factorial(removed))
    # A squared factorial or an additional crystal-fragment factor would change
    # this exact full-host counting identity; neither belongs in the target.


def test_same_cap_and_zero_number_umbrella_are_essential():
    attenuation = F(1,10000)
    # Different caps: even an edge-free host gains a top sector not in hard K=1.
    activity,volume = F(2),2
    hard_small_cap = 1 + activity * volume
    soft_larger_cap = (1 + activity) ** volume
    assert soft_larger_cap > hard_small_cap * exponential_upper(activity * volume * attenuation)
    # A common number umbrella favoring N=2 invalidates the deletion convolution.
    # Two vertices form one overlap edge; hard hosts have only N=0 or1.
    activity,attenuation = F(1),F(1,100)
    number_weights = {0:F(1,100) ** 4,1:F(1,100),2:F(1)}
    hard = number_weights[0] + 2 * activity * number_weights[1]
    soft = hard + activity ** 2 * attenuation * number_weights[2]
    assert soft > hard * exponential_upper(2 * activity * attenuation)


def test_pair_cost_coefficient_cannot_silently_count_half_an_overlap():
    claimed_attenuation = F(1,10000)
    actual_attenuation = F(1,100)  # gamma/2 rather than gamma per unordered pair
    hard = F(3)                  # V=2, zeta=1, one possible overlap edge
    soft = hard + actual_attenuation
    assert soft > hard * exponential_upper(2 * claimed_attenuation)


def test_duplicate_ideal_targets_have_equal_quota_adjusted_normalizers_not_raw_weights():
    activity,cap,volume = F(3,5),3,4
    ideal = exponential_polynomial(activity * volume,cap)
    quota_first,quota_second = F(1,50),F(1,400)
    weight_first,weight_second = quota_first / ideal,quota_second / ideal
    assert weight_first != weight_second
    assert weight_first / quota_first == weight_second / quota_second
    # At a desired normalizing root: b1-log tau1 == b2-log tau2 exactly.
    assert weight_first / weight_second == quota_first / quota_second


def test_dense_gamma_twenty_five_bound_is_small_without_any_cap_inactivity_assumption():
    bound = math.exp(16 + math.log(30.8496) - 25)
    assert .00380714 < bound < .00380715
    assert -math.expm1(-bound) < .003800
    assert 9.07 > bound  # observed adjusted efficiency gap is not a root difference
