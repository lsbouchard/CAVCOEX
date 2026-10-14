"""Independent equilibrium/support controls for the continuum expansion."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import (
    activity_probe_branch, burn_in_displacement, fcc_reference, initialize, make_compact_ladder, make_ladder, make_soft_ladder, run_chunk, seed_rng, supports,
)


def test_physical_endpoints_and_support():
    volume, sigma, radius, activity, endpoint = make_ladder()
    assert sigma[endpoint] == 1
    assert radius[endpoint] == 0
    assert radius[-1] == 1
    assert np.all(activity[endpoint:] == 16)
    positions, counts, centers, reference = initialize(32, volume**(1/3), sigma, radius, 80, 123)
    assert len(reference) == 32
    for j in range(len(sigma)):
        assert supports(positions[j], counts[j], centers[j], sigma[j], radius[j], volume**(1/3))


def test_pair_and_cavity_constraints_independent_of_particle_labels():
    positions = np.array([[.2, .2, .2], [2., 2., 2.], [3., 3., 3.]])
    center = np.array([1., 1., 1.])
    assert supports(positions, 3, center, 1., .4, 4.)
    assert supports(positions[::-1].copy(), 3, center, 1., .4, 4.)
    assert not supports(positions, 3, positions[0], 1., .1, 4.)
    positions[1] = positions[0] + .1
    assert not supports(positions, 3, center, 1., 0., 4.)


def test_ideal_gas_particle_number_and_nested_cavity_ratio():
    """A nontrivial analytic control including GCMC and replica exchanges."""
    volume = 20.
    length = volume**(1/3)
    rho = .4
    radius = np.array([0., .5])
    sigma = np.zeros(2)
    x = np.full(2, math.log(rho))
    positions = np.zeros((2, 40, 3))
    rng = np.random.default_rng(157)
    positions[:, :8] = rng.uniform(0., length, (2, 8, 3))
    centers = np.zeros((2, 3))
    # Remove any initial points in the cavity without changing the target.
    counts = np.array([8, 0], dtype=np.int64)
    reference = fcc_reference(4, length)
    ids = np.arange(2)
    seed_rng(99318)
    run_chunk(positions, counts, centers, sigma, radius, x, ids, length, 2000, 0, 10, reference, np.ones(2),np.full(2,np.inf))
    samples, _, swaps, _ = run_chunk(positions, counts, centers, sigma, radius, x, ids, length, 20000, 0, 10, reference, np.ones(2),np.full(2,np.inf))
    n0 = samples[:, 0, 0]
    nc = samples[:, 1, 0]
    cavity_volume = 4*math.pi/3*.5**3
    assert abs(n0.mean()-rho*volume) < .3
    assert abs(n0.var()-rho*volume) < .8
    assert abs(nc.mean()-rho*(volume-cavity_volume)) < .3
    assert abs(samples[:, 0, 3].mean()-math.exp(-rho*cavity_volume)) < .04
    assert np.min(swaps[:, 1]/swaps[:, 0]) > .6
    assert not samples[:, :, 7].any()


def test_ladder_has_no_hidden_basin_energy():
    _, sigma, radius, activity, endpoint = make_ladder(chemical_potential=14.7)
    assert activity[0] == pytest.approx(math.log(1/1.06448))
    assert np.all(activity[endpoint:] == 14.7)
    assert np.all(sigma[endpoint:] == 1)
    assert np.all(np.diff(radius) >= 0)


def test_soft_pair_gas_matches_exact_zero_one_two_particle_partition_sum():
    """The N<=2 soft-pair law has an elementary exact continuum integral."""
    volume = 27.
    length = 3.
    activity = .06
    penalties = np.array([.4,2.7])
    sigma = np.ones(2)
    radius = np.zeros(2)
    x = np.full(2,math.log(activity))
    positions = np.zeros((2,2,3))
    positions[:,0] = .7
    counts = np.ones(2,dtype=np.int64)
    centers = np.zeros((2,3))
    ids = np.arange(2)
    reference = fcc_reference(4,length)
    seed_rng(402196)
    run_chunk(positions,counts,centers,sigma,radius,x,ids,length,2000,1,2,reference,np.ones(2),penalties)
    sample,_,swaps,_ = run_chunk(positions,counts,centers,sigma,radius,x,ids,length,30000,1,2,reference,np.ones(2),penalties)
    excluded = 4*math.pi/3
    for j,penalty in enumerate(penalties):
        q = np.array([1.,activity*volume,
                      .5*(activity*volume)**2*(1-(1-math.exp(-penalty))*excluded/volume)])
        expected = q/q.sum()
        observed = np.bincount(sample[:,j,0].astype(int),minlength=3)/len(sample)
        assert np.max(abs(observed-expected)) < .025
    assert swaps[0,1]/swaps[0,0] > .8


def test_perfect_fcc_occupancy_is_invariant_under_arbitrary_translation():
    volume,sigma,radius,activity,endpoint = make_ladder()
    length = volume**(1/3)
    reference = fcc_reference(32,length)
    for shift in ((.3,.1,.1),(.17,.38,.24),(.42,.36,.33)):
        # Keep N exactly fixed in this geometry control; no phase/weight result.
        translated = (reference+length/2*np.array(shift))%length
        positions = translated[None,:,:].copy()
        count = np.array([32],dtype=np.int64)
        center = np.zeros((1,3))
        seed_rng(5113)
        samples,_,_,_ = run_chunk(positions,count,center,np.ones(1),np.zeros(1),np.array([100.]),np.zeros(1,dtype=np.int64),length,20,0,10,reference,np.zeros(1),np.full(1,np.inf))
        assert np.all(samples[:,:,4:6] == 0)
        assert np.max(samples[:,:,9]) < 1e-12
        assert np.allclose(samples[:,:,1],1.)


def test_soft_path_ends_at_identical_physical_grand_canonical_target():
    volume,sigma,radius,x,endpoint,penalty = make_soft_ladder(chemical_potential=15.4)
    assert np.all(sigma == 1)
    assert penalty[0] == 0
    assert np.all(np.isinf(penalty[endpoint:]))
    assert np.all(x[endpoint:] == 15.4)
    assert radius[endpoint] == 0 and radius[-1] == 1


@pytest.mark.parametrize("invalid", ["soft", "negative_infinity", "bias", "activity", "diameter", "base_radius", "cavity_radius"])
def test_palm_transfer_rejects_incompatible_targets_before_any_moves(invalid):
    """The marked-exchange ratio is valid only for its matched endpoint pair."""
    positions = np.zeros((2, 4, 3))
    counts = np.zeros(2, dtype=np.int64)
    centers = np.zeros((2, 3))
    diameter = np.ones(2)
    radius = np.array([0., 1.])
    activity = np.zeros(2)
    penalties = np.full(2, np.inf)
    stiffness = np.zeros(2)
    if invalid == "soft":
        penalties[1] = 2.
    elif invalid == "negative_infinity":
        penalties[1] = -np.inf
    elif invalid == "bias":
        stiffness[0] = .1
    elif invalid == "activity":
        activity[1] = .1
    elif invalid == "diameter":
        diameter[0] = .9
    elif invalid == "base_radius":
        radius[0] = .1
    elif invalid == "cavity_radius":
        radius[1] = .9
    # Direct Python execution tests the same initial guard without compiling
    # six signatures or allocating a production trajectory.
    with pytest.raises(ValueError, match="Marked transfer requires"):
        run_chunk.py_func(positions, counts, centers, diameter, radius, activity,
                         np.arange(2), 3., 0, 0, 1, fcc_reference(4, 3.),
                         np.ones(2), penalties, stiffness,
                         cavity_nodes=np.array([0, 1]), palm_transfer=True)


def test_activity_probe_leg_is_a_connected_sorted_graph_without_duplicate_base():
    activities,nodes,edges = activity_probe_branch(14.8,8.,17.,46,117,224)
    assert np.all(np.diff(activities)>0)
    assert not np.isclose(activities,14.8,rtol=0,atol=1e-12).any()
    mapping = dict(zip(nodes,activities))
    mapping[117] = 14.8
    assert len(edges) == len(nodes)
    assert set(edges.flat) == set(mapping)
    assert all(mapping[a]<mapping[b] and mapping[b]-mapping[a] <= .200000000001 for a,b in edges)
    assert sum(117 in edge for edge in edges) == 2
    # Preserve the earlier upward-only layout exactly.
    activities,nodes,edges = activity_probe_branch(14.8,14.8,17.,12,117,224)
    assert len(nodes) == 11 and tuple(edges[0]) == (117,224)


@pytest.mark.parametrize("arguments", [(14.8,np.nan,17.,46),(14.8,8.,np.inf,46),
                                         (14.8,15.,17.,46),(14.8,8.,12.,46),
                                         (14.8,14.8,14.8,46),(14.8,8.,17.,1)])
def test_invalid_probe_interval_rejected(arguments):
    with pytest.raises(ValueError,match="Probe interval"):
        activity_probe_branch(*arguments,117,224)


@pytest.mark.parametrize("sites,length", [(0,3.),(-4,3.),(4,0.),(4,np.nan)])
def test_invalid_fcc_reference_rejected(sites,length):
    with pytest.raises(ValueError,match="positive site count"):
        fcc_reference(sites,length)


def test_compact_graph_has_two_common_normalization_routes_and_unbiased_cavity_path():
    volume,diameter,radius,activity,base,penalty,stiffness,preferred,nodes,edges = make_compact_ladder(chemical_potential=14.8)
    assert len(diameter) == 81 and base == 62
    assert tuple(nodes) == (62,63)
    assert np.all(np.isinf(penalty[nodes])) and np.all(stiffness[nodes] == 0)
    assert np.all(activity[nodes] == 14.8)
    assert radius[nodes[0]] == 0 and radius[nodes[-1]] == 1
    assert penalty[0] == penalty[-1] == 0
    assert stiffness[0] == stiffness[-1] == 0
    assert tuple(edges[-1]) == (80,0)
    # Traverse without using the cavity dead end: both auxiliary routes
    # must reach the same physical base and the same ideal reservoir.
    adjacency = {j:[] for j in range(len(diameter))}
    for left,right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    frontier = [0]
    visited = set()
    while frontier:
        j = frontier.pop()
        if j in visited:
            continue
        visited.add(j)
        frontier.extend(adjacency[j])
    assert visited == set(range(len(diameter)))
    assert len(adjacency[base]) == 3 and len(adjacency[0]) == 2


@pytest.mark.parametrize("order_penalty",[2.,4.,15.9])
def test_compact_low_penalty_order_bridge_preserves_common_physical_target(order_penalty):
    _,diameter,radius,activity,base,penalty,stiffness,preferred,nodes,edges = make_compact_ladder(
        chemical_potential=14.8,order_penalty=order_penalty)
    assert np.all(diameter == 1)
    assert np.all(penalty[17:39] == order_penalty)
    assert np.all(stiffness[21:39] == 120)
    assert preferred[21] == .2 and preferred[38] == .88
    assert np.all(np.diff(penalty[:base+1]) >= 0)
    assert np.all(np.isposinf(penalty[nodes]))
    assert np.all(stiffness[nodes] == 0) and np.all(activity[nodes] == 14.8)
    assert tuple(radius[nodes]) == (0.,1.)
    assert tuple(edges[-1]) == (len(diameter)-1,0)
    assert np.max(penalty[1:17]-penalty[:16]) <= order_penalty/16+1e-14


@pytest.mark.parametrize("order_penalty",[0.,-1.,16.,np.nan,np.inf])
def test_compact_invalid_order_bridge_penalty_rejected(order_penalty):
    with pytest.raises(ValueError,match="Order-bridge penalty"):
        make_compact_ladder(order_penalty=order_penalty)


@pytest.mark.parametrize("order_penalty",[2.,8.,15.9])
def test_optional_low_order_return_preserves_old_graph_and_shared_physical_nodes(order_penalty):
    original = make_compact_ladder(chemical_potential=14.8,order_penalty=order_penalty)
    augmented = make_compact_ladder(chemical_potential=14.8,order_penalty=order_penalty,
                                   low_order_return=True)
    _,old_diameter,_,_,old_base,_,_,_,old_cavity,old_edges = original
    _,diameter,radius,activity,base,penalty,stiffness,preferred,cavity,edges = augmented
    begin = len(old_diameter)
    for i in (1,2,3,5,6,7):
        np.testing.assert_array_equal(augmented[i][:begin],original[i])
    np.testing.assert_array_equal(edges[:len(old_edges)],old_edges)
    np.testing.assert_array_equal(cavity,old_cavity)
    assert base == old_base
    assert tuple(edges[len(old_edges)]) == (20,begin)
    assert tuple(edges[-1]) == (len(diameter)-1,base)
    assert penalty[20] == order_penalty and stiffness[20] == 120 and preferred[20] == .2
    assert np.all(preferred[begin:] == .2) and np.all(radius[begin:] == 0)
    assert np.all(diameter[begin:] == 1)
    assert np.all(np.diff(penalty[begin:]) >= 0)
    assert np.all(activity[cavity] == 14.8) and np.all(stiffness[cavity] == 0)
    assert np.all(np.isposinf(penalty[cavity]))
    assert np.all(activity[-12:] == 14.8)
    # A complete branch connects the low-B soft gateway to the unbiased
    # physical node; no fictitious physical weight is supplied.
    branch = edges[len(old_edges):]
    assert len(branch) == len(diameter)-begin+1
    assert np.array_equal(branch[1:-1,0],np.arange(begin,len(diameter)-1))


def test_frozen_displacement_retains_loaded_proposals_without_mutation():
    proposal = np.array([.001,.02,2.])
    original = proposal.copy()
    accept = np.zeros((3,8),dtype=int)
    accept[:,0] = 10
    accept[:,1] = [0,3,10]
    result = burn_in_displacement(proposal,accept,3.,frozen=True)
    assert result is proposal
    np.testing.assert_array_equal(result,original)


def test_unfrozen_displacement_matches_original_burn_rule_and_bounds():
    proposal = np.array([.001,.02,2.])
    accept = np.zeros((3,8),dtype=int)
    accept[:,0] = [0,10,10]
    accept[:,1] = [0,3,10]
    result = burn_in_displacement(proposal,accept,3.)
    expected = np.clip(proposal*np.exp(.5*(np.array([0.,.3,1.])-.3)),.01,1.5)
    np.testing.assert_array_equal(result,expected)
    np.testing.assert_array_equal(proposal,np.array([.001,.02,2.]))
