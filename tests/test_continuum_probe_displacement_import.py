"""Probe proposal matching is explicit under identical or regridded targets."""

import numpy as np
import pytest

from scripts.continuum_replica_exchange import frozen_probe_displacements


def test_identical_probe_graph_copies_every_proposal_scale_without_aliasing():
    current = np.full(7,.12)
    old = np.arange(7)*.03+.01
    nodes = np.array([4,5,6])
    activities = np.array([6.,12.,17.])
    result,matched = frozen_probe_displacements(current,nodes,activities,old,nodes,activities)
    np.testing.assert_array_equal(result[nodes],old[nodes])
    np.testing.assert_array_equal(result[:4],current[:4])
    np.testing.assert_array_equal(matched,nodes)
    result[4] = 42
    assert old[4] != 42 and current[4] == .12


def test_regridded_graph_copies_only_identical_activities_not_node_indices():
    current = np.full(8,.12)
    old = np.arange(7)*.03+.01
    result,matched = frozen_probe_displacements(current,np.array([4,5,6,7]),
                                                np.array([5.,6.,12.,18.]),old,
                                                np.array([4,5,6]),np.array([6.,12.,17.]))
    np.testing.assert_array_equal(matched,[5,6])
    assert result[5] == old[4] and result[6] == old[5]
    assert result[4] == result[7] == .12


def test_added_probes_without_a_previous_leg_keep_defaults_and_report_no_matches():
    result,matched = frozen_probe_displacements(np.full(5,.12),np.array([3,4]),
                                                np.array([6.,17.]),np.full(3,.2),
                                                np.zeros(0,dtype=int),np.zeros(0))
    np.testing.assert_array_equal(result,np.full(5,.12))
    assert len(matched) == 0


@pytest.mark.parametrize("nodes,activities", [
    (np.array([3,3]),np.array([6.,7.])),(np.array([3,4]),np.array([6.,6.])),
    (np.array([3,4]),np.array([6.,np.nan])),(np.array([-1,4]),np.array([6.,7.])),
    (np.array([3,8]),np.array([6.,7.])),(np.array([3.,4.]),np.array([6.,7.])),
])
def test_invalid_probe_mapping_rejected(nodes,activities):
    with pytest.raises(ValueError,match="Probe proposal mapping"):
        frozen_probe_displacements(np.full(5,.12),nodes,activities,np.full(5,.2),
                                   np.array([3,4]),np.array([6.,7.]))


def test_numerically_ambiguous_old_activities_rejected():
    with pytest.raises(ValueError,match="ambiguous"):
        frozen_probe_displacements(np.full(5,.12),np.array([3]),np.array([6.]),
                                   np.full(5,.2),np.array([3,4]),np.array([6.,6.+5e-13]))
