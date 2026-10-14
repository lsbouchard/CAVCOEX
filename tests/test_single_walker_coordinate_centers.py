"""Diagnostic centers retain the exact RNG path and reconstruct hard cavities."""
import math

import numpy as np
import pytest

from scripts import continuum_simulated_tempering as st

def run(centers_out,steps=5000):
    length,cap,clock,stride = 3.,3,13,7
    neighbors,edge_table,degrees = st.adjacency(np.array([[0,1],[1,2],[2,0]]),3)
    positions,counts,centers = np.zeros((1,cap,3)),np.zeros(1,dtype=np.int64),np.zeros((1,3))
    fields = np.full(3,math.log(.06))
    st.kernel.seed_rng(260929411)
    result = st.run_tempering_chunk(positions,counts,centers,2,0,clock,steps,
        st.kernel.fcc_reference(32,length),length,np.ones(3),np.array([0.,1.,0.]),
        fields,np.array([np.inf,np.inf,0.]),np.zeros(3),np.zeros(3),
        np.zeros(3),np.zeros(3),np.full(3,.8),np.array([True,True,False]),
        0,1,2,neighbors,edge_table,degrees,np.array([0.,.3,-.4]),
        st.capped_poisson_cdf(fields[2]+math.log(length**3),cap),
        coordinate_stride=stride,birth_width=.3,coordinate_centers=centers_out)
    return result,positions,counts,centers


def test_optional_coordinate_centers_do_not_change_path_and_reconstruct_cavity():
    count = len(np.arange(7-(13%7)-1,5000,7))
    buffer = np.empty((count,3))
    plain,*plain_state = run(None)
    observed,*observed_state = run(buffer)
    for left,right in zip(plain[:6],observed[:6]):
        np.testing.assert_array_equal(left,right)
    assert plain[-2:] == observed[-2:]
    for left,right in zip(plain_state,observed_state):
        np.testing.assert_array_equal(left,right)
    trace,_,_,coordinates,clocks,labels,_,_ = observed
    rows = clocks.astype(int)-14
    assert (labels==1).sum()>25
    for index,row in enumerate(rows):
        n = int(trace[row,2])
        active = coordinates[index,:n]
        difference = active-buffer[index]
        difference -= 3.*np.rint(difference/3.)
        minimum = np.linalg.norm(difference,axis=1).min() if n else 3.
        assert minimum == pytest.approx(trace[row,7],abs=1e-12)
        if labels[index]==1:
            assert minimum>=1.


def test_wrong_coordinate_center_shape_rejected():
    with pytest.raises(ValueError,match='snapshot count'):
        run(np.empty((1,3)),steps=10)
