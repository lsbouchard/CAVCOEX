#!/usr/bin/env python3
"""Unrestricted grand-canonical continuum hard spheres with staged cavities.

Each replica has target exp(x*N-U)/N! times its cavity indicator, with a
uniform auxiliary cavity center dq/V.  U contains the pair-overlap penalty
and optional structural and particle-number umbrellas.  A diameter/activity tempering ladder
connects the physical endpoint to an ideal gas; a nested cavity-radius ladder
connects that endpoint to an inserted particle.  No site-occupancy constraint
or prescribed basin probability enters any target.  FCC coordinates provide
occupancy diagnostics, and the optional auxiliary order umbrella uses FCC
reciprocal-mode amplitudes.  Its energy enters every MH kernel but vanishes
at every physical and cavity endpoint.

The finite N cap is explicit and is monitored.  Production is a finite-box
calculation, not a thermodynamic-limit coexistence measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
# Pytest and module-mode CLI may import this file under different names.
# Numba caches serialize those module names, so both import roots must exist.
if str(ROOT/"scripts") not in sys.path:
    sys.path.insert(0,str(ROOT/"scripts"))
EXECUTION_SOURCE_BYTES = Path(__file__).read_bytes()
EXECUTION_SOURCE_SHA256 = hashlib.sha256(EXECUTION_SOURCE_BYTES).hexdigest()


def excess_mu_cs(eta: float) -> float:
    return (8 * eta - 9 * eta**2 + 3 * eta**3) / (1 - eta) ** 3


def make_ladder(sites=32, specific_volume=.96405, chemical_potential=16.0,
                diameter_stages=64, cavity_stages=32):
    """Auxiliary activities interpolate CS values; the endpoint x is exact input."""
    volume = sites * specific_volume
    rho = 1 / 1.06448
    # Hard-support exchange deteriorates rapidly at large diameter; concentrate
    # stages there instead of leaving an untraversable uniform coarse ladder.
    diameters = 1 - (1 - np.linspace(0, 1, diameter_stages))**2
    eta = np.pi / 6 * rho * diameters**3
    activity = np.log(rho) + np.array([excess_mu_cs(v) for v in eta])
    activity += (chemical_potential - activity[-1]) * diameters**3
    radius = np.concatenate((np.zeros(diameter_stages),
                             np.linspace(0, 1, cavity_stages + 1)[1:]))
    diameter = np.concatenate((diameters, np.ones(cavity_stages)))
    activity = np.concatenate((activity, np.full(cavity_stages, chemical_potential)))
    return volume, diameter, radius, activity, diameter_stages - 1


def make_hard_ladder(sites=32,specific_volume=.96405,chemical_potential=16.,cavity_stages=32):
    """Only unbiased physical hard targets; optional activity leg added later."""
    size = cavity_stages+1
    return (sites*specific_volume,np.ones(size),np.linspace(0,1,size),
            np.full(size,chemical_potential),0,np.full(size,np.inf),
            np.zeros(size),np.zeros(size))


def make_soft_ladder(sites=32, specific_volume=.96405, chemical_potential=16., cavity_stages=32):
    """Penetrable pair penalties provide a shorter exact auxiliary path."""
    penalties = np.concatenate((np.linspace(0,10,51),np.arange(10.5,20.5,.5),np.array([22.,25.,30.,np.inf])))
    rho = 1/1.06448
    finite_penalties = np.where(np.isfinite(penalties),penalties,1e12)
    fraction = finite_penalties/(finite_penalties+4)
    activity = np.log(rho)+(chemical_potential-np.log(rho))*fraction
    # At large penalties the pair model approaches the physical hard target.
    blend = np.clip((finite_penalties-8)/10,0,1)
    activity = (1-blend)*activity+blend*chemical_potential
    activity[-1] = chemical_potential
    endpoint = len(penalties)-1
    return (sites*specific_volume, np.ones(len(penalties)+cavity_stages),
            np.concatenate((np.zeros(len(penalties)),np.linspace(0,1,cavity_stages+1)[1:])),
            np.concatenate((activity,np.full(cavity_stages,chemical_potential))),
            endpoint, np.concatenate((penalties,np.full(cavity_stages,np.inf))))


def make_umbrella_ladder(sites=32,specific_volume=.96405,chemical_potential=15.5,cavity_stages=32):
    """A continuous auxiliary order bridge; all physical nodes have zero bias."""
    penalties = np.concatenate((np.linspace(0,8,41),np.full(6,8.),np.full(34,8.),
                                np.arange(8.5,20.5,.5),np.full(10,20.),np.array([25.,30.,np.inf])))
    stiffness = np.concatenate((np.zeros(41),np.array([5.,10.,25.,50.,100.,200.]),
                                np.full(34,200.),np.full(24,200.),
                                np.array([150.,100.,75.,50.,25.,10.,5.,2.,1.,0.]),np.zeros(3)))
    preferred = np.concatenate((np.full(47,.2),np.linspace(.2,.86,34),np.full(37,.86)))
    if len(penalties) != len(stiffness) or len(penalties) != len(preferred):
        raise RuntimeError("Order bridge length mismatch")
    finite = np.where(np.isfinite(penalties),penalties,1e12)
    x = np.log(1/1.06448)+(chemical_potential-np.log(1/1.06448))*finite/(finite+4)
    blend = np.clip((finite-8)/10,0,1)
    x = (1-blend)*x+blend*chemical_potential
    x[-1] = chemical_potential
    endpoint = len(penalties)-1
    return (sites*specific_volume,np.ones(endpoint+1+cavity_stages),
            np.concatenate((np.zeros(endpoint+1),np.linspace(0,1,cavity_stages+1)[1:])),
            np.concatenate((x,np.full(cavity_stages,chemical_potential))),endpoint,
            np.concatenate((penalties,np.full(cavity_stages,np.inf))),
            np.concatenate((stiffness,np.zeros(cavity_stages))),
            np.concatenate((preferred,np.zeros(cavity_stages))))


def make_loop_ladder(sites=32,specific_volume=.96405,chemical_potential=15.5,cavity_stages=32):
    volume,dia,radius,x,endpoint,penalty,stiffness,preferred = make_umbrella_ladder(sites,specific_volume,chemical_potential,cavity_stages)
    cavity_nodes = np.arange(endpoint,len(dia),dtype=np.int64)
    branch_start = len(dia)
    _,sd,sr,sx,se,sp = make_soft_ladder(sites,specific_volume,chemical_potential,0)
    # A second, unbiased soft-fluid route shares both the hard physical node
    # and the ideal reservoir with the ordered umbrella route.
    return_indices = np.arange(se-1,-1,-1)
    dia = np.concatenate((dia,sd[return_indices]))
    radius = np.concatenate((radius,sr[return_indices]))
    x = np.concatenate((x,sx[return_indices]))
    penalty = np.concatenate((penalty,sp[return_indices]))
    stiffness = np.concatenate((stiffness,np.zeros(len(return_indices))))
    preferred = np.concatenate((preferred,np.zeros(len(return_indices))))
    edges = [(i,i+1) for i in range(branch_start-1)]
    edges += [(endpoint,branch_start)]
    edges += [(i,i+1) for i in range(branch_start,len(dia)-1)]
    edges += [(len(dia)-1,0)]
    return volume,dia,radius,x,endpoint,penalty,stiffness,preferred,cavity_nodes,np.array(edges,dtype=np.int64)


def make_compact_ladder(sites=32,specific_volume=.96405,chemical_potential=15.5,cavity_stages=1,
                        order_penalty=8.,low_order_return=False):
    """Short two-route soft/order graph; all physical nodes remain unbiased.

    The order route can create a crystal from a disordered initial state;
    the second unbiased soft route independently feeds fluid number sectors.
    These are auxiliary product targets, not prescribed physical phase odds.
    """
    if not np.isfinite(order_penalty) or not 0 < order_penalty < 16:
        raise ValueError("Order-bridge penalty must be finite and lie between zero and 16")
    hardening_count = int(math.ceil((16-order_penalty)/.75))
    penalty = np.concatenate((np.linspace(0,order_penalty,17),np.full(4,order_penalty),
                              np.full(18,order_penalty),
                              np.linspace(min(order_penalty+.75,16),16,hardening_count),
                              np.full(10,16.),[20.,25.,np.inf]))
    stiffness = np.concatenate((np.zeros(17),[10.,30.,60.,120.],np.full(18,120.),
                                np.full(hardening_count,120.),[90.,60.,40.,25.,15.,10.,5.,2.,1.,0.],np.zeros(3)))
    preferred = np.concatenate((np.full(21,.2),np.linspace(.2,.88,18),np.full(hardening_count+13,.88)))
    if not len(penalty) == len(stiffness) == len(preferred):
        raise ValueError("Compact target schedule lengths differ")
    finite = np.where(np.isfinite(penalty),penalty,1e12)
    log_density = np.log(1/1.06448)
    activity = log_density+(chemical_potential-log_density)*finite/(finite+4)
    blend = np.clip((finite-8)/8,0,1)
    activity = (1-blend)*activity+blend*chemical_potential
    endpoint = len(penalty)-1
    activity[-1] = chemical_potential
    diameter = np.ones(endpoint+1+cavity_stages)
    radius = np.concatenate((np.zeros(endpoint+1),np.linspace(0,1,cavity_stages+1)[1:]))
    activity = np.concatenate((activity,np.full(cavity_stages,chemical_potential)))
    penalty = np.concatenate((penalty,np.full(cavity_stages,np.inf)))
    stiffness = np.concatenate((stiffness,np.zeros(cavity_stages)))
    preferred = np.concatenate((preferred,np.zeros(cavity_stages)))
    cavity_nodes = np.arange(endpoint,len(diameter),dtype=np.int64)
    begin = len(diameter)
    return_penalty = np.array([25.,20.,16.,14.,12.,11.,10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,0.])
    return_activity = log_density+(chemical_potential-log_density)*return_penalty/(return_penalty+4)
    blend = np.clip((return_penalty-8)/8,0,1)
    return_activity = (1-blend)*return_activity+blend*chemical_potential
    diameter = np.concatenate((diameter,np.ones(len(return_penalty))))
    radius = np.concatenate((radius,np.zeros(len(return_penalty))))
    activity = np.concatenate((activity,return_activity))
    penalty = np.concatenate((penalty,return_penalty))
    stiffness = np.concatenate((stiffness,np.zeros(len(return_penalty))))
    preferred = np.concatenate((preferred,np.zeros(len(return_penalty))))
    edges = [(i,i+1) for i in range(begin-1)]+[(endpoint,begin)]
    edges += [(i,i+1) for i in range(begin,len(diameter)-1)]+[(len(diameter)-1,0)]
    if low_order_return:
        # The existing gamma=order_penalty, kappa=120, B*=.2 node is a
        # shared disordered gateway.  This extra branch hardens its low-B
        # configurations before removing the auxiliary bias at the SAME
        # physical endpoint.  It never prescribes a physical phase weight.
        low_begin = len(diameter)
        low_penalty = np.concatenate((np.linspace(min(order_penalty+.75,16),16,hardening_count),
                                      np.full(10,16.),[20.,25.]))
        low_stiffness = np.concatenate((np.full(hardening_count,120.),
                                       [90.,60.,40.,25.,15.,10.,5.,2.,1.,0.],np.zeros(2)))
        finite = low_penalty
        low_activity = log_density+(chemical_potential-log_density)*finite/(finite+4)
        blend = np.clip((finite-8)/8,0,1)
        low_activity = (1-blend)*low_activity+blend*chemical_potential
        diameter = np.concatenate((diameter,np.ones(len(low_penalty))))
        radius = np.concatenate((radius,np.zeros(len(low_penalty))))
        activity = np.concatenate((activity,low_activity))
        penalty = np.concatenate((penalty,low_penalty))
        stiffness = np.concatenate((stiffness,low_stiffness))
        preferred = np.concatenate((preferred,np.full(len(low_penalty),.2)))
        edges += [(20,low_begin)]
        edges += [(i,i+1) for i in range(low_begin,len(diameter)-1)]
        edges += [(len(diameter)-1,endpoint)]
    return (sites*specific_volume,diameter,radius,activity,endpoint,penalty,stiffness,
            preferred,cavity_nodes,np.array(edges,dtype=np.int64))


def activity_probe_branch(base_activity, minimum, maximum, count, endpoint, begin):
    """Unbiased hard targets on a sorted chemical-activity exchange leg.

    The existing physical base is a node of this leg, not an extra target
    with prescribed odds. Every edge uses the ordinary full-state MH swap.
    """
    if (not np.isfinite([base_activity, minimum, maximum]).all()
            or count < 2 or not minimum <= base_activity <= maximum
            or minimum >= maximum):
        raise ValueError("Probe interval must be finite, nonempty and contain the base activity")
    activities = np.linspace(minimum, maximum, count)
    activities = activities[~np.isclose(activities,base_activity,rtol=0,atol=1e-12)]
    nodes = np.arange(begin,begin+len(activities),dtype=np.int64)
    chain = sorted([(base_activity,endpoint)]+list(zip(activities,nodes)))
    edges = np.array([(chain[i][1],chain[i+1][1]) for i in range(len(chain)-1)],dtype=np.int64)
    return activities,nodes,edges


def add_compact_number_bridge(ladder,strength=4.):
    """Bias auxiliary low-order number relaxation, never a physical marginal.

    The original low-B branch is retained, with nu(N-N*)² added only on
    that branch.  Four soft gateway nodes turn this term on and ten
    late-hardening nodes remove it again.  Both endpoints are existing
    common-normalization targets; no relative phase odds are supplied.
    """
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("Number-umbrella strength must be finite and positive")
    volume,diameter,radius,activity,endpoint,penalty,stiffness,preferred,cavity,edges = ladder
    candidates = edges[(edges[:,0] == 20)&(edges[:,1]>cavity[-1]),1]
    if len(candidates) != 1 or tuple(edges[-1]) != (len(diameter)-1,endpoint):
        raise ValueError("Number bridge requires the compact low-order return graph")
    begin = int(candidates[0])
    last = len(diameter)-1
    entry = len(diameter)
    target_n = volume/1.06448
    nu = np.zeros(len(diameter))
    number_preferred = np.zeros(len(diameter))
    nu[begin:] = strength
    number_preferred[begin:] = target_n
    entry_strength = strength*np.array([.125,.25,.5,1.])
    removal_strength = strength*np.array([.75,.5,.375,.25,.175,.125,.075,.05,.025,0.])
    templates = np.array([20]*len(entry_strength)+[last]*len(removal_strength))
    diameter = np.concatenate((diameter,diameter[templates]))
    radius = np.concatenate((radius,radius[templates]))
    activity = np.concatenate((activity,activity[templates]))
    penalty = np.concatenate((penalty,penalty[templates]))
    stiffness = np.concatenate((stiffness,stiffness[templates]))
    preferred = np.concatenate((preferred,preferred[templates]))
    nu = np.concatenate((nu,entry_strength,removal_strength))
    number_preferred = np.concatenate((number_preferred,np.full(len(templates),target_n)))
    keep = ~(((edges[:,0] == 20)&(edges[:,1] == begin))
             |((edges[:,0] == last)&(edges[:,1] == endpoint)))
    additions = [(20,entry)]+[(j,j+1) for j in range(entry,entry+len(entry_strength)-1)]
    additions += [(entry+len(entry_strength)-1,begin)]
    tail = entry+len(entry_strength)
    additions += [(last,tail)]+[(j,j+1) for j in range(tail,len(diameter)-1)]
    additions += [(len(diameter)-1,endpoint)]
    edges = np.concatenate((edges[keep],np.array(additions,dtype=np.int64)))
    return (volume,diameter,radius,activity,endpoint,penalty,stiffness,preferred,
            cavity,edges,nu,number_preferred)


@njit(cache=True)
def bragg_value(re,im,n):
    if n == 0:
        return 0.
    return np.sum(np.sqrt(re*re+im*im))/(7*n)


@njit(cache=True)
def bragg_delta(re,im,point,length,cells,sign):
    new_re = re.copy()
    new_im = im.copy()
    for v in range(7):
        phase = bragg_phase(v,point,length,cells)
        new_re[v] += sign*math.cos(phase)
        new_im[v] += sign*math.sin(phase)
    return new_re,new_im


@njit(cache=True,inline="always")
def bragg_phase(v,point,length,cells):
    scale = 2*np.pi*cells/length
    px,py,pz = scale*point[0],scale*point[1],scale*point[2]
    if v == 0:
        return px+py+pz
    if v == 1:
        return px+py-pz
    if v == 2:
        return px-py+pz
    if v == 3:
        return -px+py+pz
    if v == 4:
        return 2*px
    if v == 5:
        return 2*py
    return 2*pz


@njit(cache=True)
def bragg_proposal_into(re,im,old_point,new_point,old_sign,new_sign,length,cells,new_re,new_im):
    for v in range(7):
        new_re[v],new_im[v] = re[v],im[v]
        if old_sign:
            phase = bragg_phase(v,old_point,length,cells)
            new_re[v] += old_sign*math.cos(phase)
            new_im[v] += old_sign*math.sin(phase)
        if new_sign:
            phase = bragg_phase(v,new_point,length,cells)
            new_re[v] += new_sign*math.cos(phase)
            new_im[v] += new_sign*math.sin(phase)


def fcc_reference(sites, length):
    if not isinstance(sites,(int,np.integer)) or sites <= 0 or not np.isfinite(length) or length <= 0:
        raise ValueError("FCC reference requires a positive site count and box length")
    cells = int(round((sites / 4) ** (1/3)))
    if 4 * cells**3 != sites:
        raise ValueError("sites must be 4 times an integer cube")
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    return np.array([(np.array([i, j, k]) + b) * length / cells
                     for i in range(cells) for j in range(cells)
                     for k in range(cells) for b in basis])


@njit(cache=True)
def periodic_distance_squared(a, b, length):
    value = 0.
    for axis in range(3):
        d = a[axis] - b[axis]
        d -= length * np.rint(d / length)
        value += d*d
    return value


@njit(cache=True)
def extrapolation_birth_density(point, positions, n, length, width, uniform_fraction, ignore=-1):
    """Full marginal proposal density, including every covering pair cube.

    For reverse death, ignore is the removed particle's index: the density
    is evaluated in the post-deletion host, not the original configuration.
    The uniform component guarantees positive support on the full torus.
    """
    if n < 0 or n > positions.shape[0] or ignore < -1 or ignore >= n:
        raise ValueError("Proposal-density particle count or deleted index is invalid")
    remaining = n-(1 if ignore >= 0 else 0)
    if remaining < 2 or uniform_fraction == 1:
        return 1/length**3
    covering_pairs = 0
    for i in range(n):
        if i == ignore:
            continue
        for j in range(n):
            if j == i or j == ignore:
                continue
            inside = True
            for axis in range(3):
                distance = point[axis]-(2*positions[i,axis]-positions[j,axis])
                distance -= length*np.rint(distance/length)
                if abs(distance) >= width:
                    inside = False
                    break
            if inside:
                covering_pairs += 1
    return (uniform_fraction/length**3
            +(1-uniform_fraction)*covering_pairs/(remaining*(remaining-1)*(2*width)**3))


@njit(cache=True)
def canonical_fcc_reference(reference,length):
    """Exact canonical-coordinate/order guard; evaluated once per MC chunk."""
    if not np.isfinite(length) or length <= 0 or reference.shape[1] != 3:
        return False
    cells = int(round((len(reference)/4.)**(1/3)))
    if cells <= 0 or 4*cells**3 != len(reference):
        return False
    basis = np.array([[0.,0.,0.],[0.,.5,.5],[.5,0.,.5],[.5,.5,0.]])
    for site in range(len(reference)):
        lattice_cell = site//4
        i = lattice_cell//(cells*cells)
        j = (lattice_cell//cells)%cells
        k = lattice_cell%cells
        for axis in range(3):
            coordinate = i if axis == 0 else j if axis == 1 else k
            expected = (coordinate+basis[site%4,axis])*length/cells
            if reference[site,axis] != expected:
                return False
    return True


@njit(cache=True, inline="always")
def wrapped_fcc_site_index(kx, ky, kz, cells):
    """Index arithmetic after half-cell coordinates have already been wrapped."""
    return 4 * (((kx >> 1) * cells + (ky >> 1)) * cells + (kz >> 1)) + 2 * (kx & 1) + (ky & 1)


@njit(cache=True, inline="always")
def canonical_fcc_site_index(kx, ky, kz, cells):
    """Map an even-parity half-cell triple to canonical FCC enumeration."""
    period = 2 * cells
    return wrapped_fcc_site_index(kx % period, ky % period, kz % period, cells)


@njit(cache=True, inline="always")
def candidate_distance(point, translation, length, reference, site, parity_shift, fit_metric):
    """Match the two original full-scan arithmetic conventions explicitly."""
    squared = 0.
    for axis in range(3):
        if fit_metric:
            delta = point[axis] - reference[site, axis] - translation[axis]
            if axis == 0:
                delta -= parity_shift
        else:
            center = (reference[site, axis] + translation[axis]) % length
            delta = point[axis] - center
        delta -= length * np.rint(delta / length)
        squared += delta * delta
    return squared


@njit(cache=True)
def nearest_canonical_fcc_site_with_ties(point, translation, length, cells, reference,
                                        parity_shift=0., fit_metric=False):
    """Rare-boundary fallback, including all tied rounded integer triples.

    The unconstrained nearest integer triple is repaired by one +/-axis
    change if odd. At half-integer boundaries, all tied rounded triples
    are included before parity repair. Candidate distances are evaluated
    on the actual reference coordinates, with equal distances resolved by
    the lowest canonical site index. A tiny near-tie envelope also protects
    against roundoff in the grid-coordinate transformation.
    """
    base = np.empty(3, dtype=np.int64)
    alternatives = np.zeros(3, dtype=np.int64)
    tie_axes = np.empty(3, dtype=np.int64)
    tie_count = 0
    for axis in range(3):
        coordinate = point[axis] - translation[axis]
        if axis == 0:
            coordinate -= parity_shift
        scaled = (2 * cells / length) * coordinate
        base[axis] = int(np.rint(scaled))
        residual = scaled - base[axis]
        tolerance = 16 * np.finfo(np.float64).eps * max(1., abs(scaled))
        if abs(abs(residual) - .5) <= tolerance:
            alternatives[axis] = 1 if residual >= 0 else -1
            tie_axes[tie_count] = axis
            tie_count += 1

    best_site = len(reference)
    best_distance = np.inf
    candidate = np.empty(3, dtype=np.int64)
    for mask in range(1 << tie_count):
        candidate[:] = base
        for bit in range(tie_count):
            if mask & (1 << bit):
                axis = tie_axes[bit]
                candidate[axis] += alternatives[axis]
        if (candidate[0] + candidate[1] + candidate[2]) % 2 == 0:
            site = canonical_fcc_site_index(candidate[0], candidate[1], candidate[2], cells)
            distance = candidate_distance(point, translation, length, reference, site,
                                          parity_shift, fit_metric)
            if distance < best_distance or (distance == best_distance and site < best_site):
                best_site, best_distance = site, distance
        else:
            for axis in range(3):
                for sign in (-1, 1):
                    candidate[axis] += sign
                    site = canonical_fcc_site_index(candidate[0], candidate[1], candidate[2], cells)
                    distance = candidate_distance(point, translation, length, reference, site,
                                                  parity_shift, fit_metric)
                    if distance < best_distance or (distance == best_distance and site < best_site):
                        best_site, best_distance = site, distance
                    candidate[axis] -= sign
    return best_site, best_distance


@njit(cache=True)
def nearest_canonical_fcc_site(point, translation, length, cells, reference,
                               parity_shift=0., fit_metric=False):
    """Allocation-free generic path, with the verified half-grid fallback.

    Odd-parity repair costs ``1-2*abs(residual_axis)``. Thus only axes with
    maximal residual magnitude need be compared, including equal/near-equal
    axes and both signs at a zero residual. Wrapping is performed once;
    one-step repairs use branches rather than repeated integer division.
    """
    scale = 2 * cells / length
    sx = scale * (point[0] - translation[0] - parity_shift)
    sy = scale * (point[1] - translation[1])
    sz = scale * (point[2] - translation[2])
    kx, ky, kz = int(np.rint(sx)), int(np.rint(sy)), int(np.rint(sz))
    ex, ey, ez = sx - kx, sy - ky, sz - kz
    tolerance = 16 * np.finfo(np.float64).eps * max(1., abs(sx), abs(sy), abs(sz))
    if (abs(abs(ex) - .5) <= tolerance or abs(abs(ey) - .5) <= tolerance
            or abs(abs(ez) - .5) <= tolerance):
        return nearest_canonical_fcc_site_with_ties(point, translation, length, cells,
                                                    reference, parity_shift, fit_metric)
    period = 2 * cells
    kx, ky, kz = kx % period, ky % period, kz % period
    if (kx + ky + kz) % 2 == 0:
        site = wrapped_fcc_site_index(kx, ky, kz, cells)
        return site, candidate_distance(point, translation, length, reference, site,
                                        parity_shift, fit_metric)
    largest = max(abs(ex), abs(ey), abs(ez))
    best_site, best_distance = len(reference), np.inf
    for axis in range(3):
        residual = ex if axis == 0 else ey if axis == 1 else ez
        if abs(residual) < largest - tolerance:
            continue
        for sign in (-1, 1):
            if residual * sign < -tolerance:
                continue
            nx, ny, nz = kx, ky, kz
            value = (kx if axis == 0 else ky if axis == 1 else kz) + sign
            if value < 0:
                value += period
            elif value >= period:
                value -= period
            if axis == 0:
                nx = value
            elif axis == 1:
                ny = value
            else:
                nz = value
            site = wrapped_fcc_site_index(nx, ny, nz, cells)
            distance = candidate_distance(point, translation, length, reference, site,
                                          parity_shift, fit_metric)
            if distance < best_distance or (distance == best_distance and site < best_site):
                best_site, best_distance = site, distance
    return best_site, best_distance


@njit(cache=True)
def fitted_fcc_occupancy_fast(positions, n, length, reference, ignore=-1):
    """Fast canonical fit, replacing only the verified nearest-site scans."""
    if n < 0 or n > positions.shape[0] or ignore < -1 or ignore >= n:
        raise ValueError("Registry particle count or deleted index is invalid")
    cells = int(round((len(reference) / 4.)**(1 / 3)))
    translation = np.zeros(3)
    if n - (1 if ignore >= 0 else 0):
        for axis in range(3):
            re, im = 0., 0.
            for particle in range(n):
                if particle == ignore:
                    continue
                phase = 4 * np.pi * cells / length * positions[particle, axis]
                re += math.cos(phase)
                im += math.sin(phase)
            translation[axis] = length / (4 * np.pi * cells) * math.atan2(im, re)
    best_fit, best_parity = np.inf, 0
    for parity in range(2):
        shift = parity * length / (2 * cells)
        fit = 0.
        for particle in range(n):
            if particle != ignore:
                _, distance = nearest_canonical_fcc_site(positions[particle], translation,
                                                         length, cells, reference, shift, True)
                fit += distance
        if fit < best_fit:
            best_fit, best_parity = fit, parity
    translation[0] += best_parity * length / (2 * cells)
    centers = (reference + translation) % length
    occupied = np.zeros(len(reference), dtype=np.int64)
    for particle in range(n):
        if particle != ignore:
            site, _ = nearest_canonical_fcc_site(positions[particle], translation,
                                                length, cells, reference)
            occupied[site] += 1
    return centers, occupied, best_fit



@njit(cache=True)
def fitted_fcc_occupancy_full_scan(positions,n,length,reference,ignore=-1):
    """Deterministic two-parity registry fit and nearest-site assignments.

    Parity ties select the first candidate; nearest-site ties select the
    lowest reference index; atan2(0,0) is zero. These chart conventions need
    not be globally translation covariant: the registry is diagnostic or
    proposal geometry, never a physical constraint or target energy.
    """
    if n < 0 or n > positions.shape[0] or ignore < -1 or ignore >= n:
        raise ValueError("Registry particle count or deleted index is invalid")
    remaining = n-(1 if ignore >= 0 else 0)
    cells = int(round((len(reference)/4.)**(1./3.)))
    if cells <= 0 or 4*cells**3 != len(reference):
        raise ValueError("Registry reference must have 4 times an integer cube sites")
    translation = np.zeros(3)
    if remaining:
        for axis in range(3):
            re = 0.
            im = 0.
            for p in range(n):
                if p == ignore:
                    continue
                phase = 4*np.pi*cells/length*positions[p,axis]
                re += math.cos(phase)
                im += math.sin(phase)
            translation[axis] = length/(4*np.pi*cells)*math.atan2(im,re)
    best_fit = np.inf
    best_parity = 0
    for parity in range(2):
        fit = 0.
        for p in range(n):
            if p == ignore:
                continue
            nearest = np.inf
            for site in range(len(reference)):
                squared = 0.
                for axis in range(3):
                    delta = positions[p,axis]-reference[site,axis]-translation[axis]
                    if axis == 0:
                        delta -= parity*length/(2*cells)
                    delta -= length*np.rint(delta/length)
                    squared += delta*delta
                nearest = min(nearest,squared)
            fit += nearest
        if fit < best_fit:
            best_fit = fit
            best_parity = parity
    translation[0] += best_parity*length/(2*cells)
    centers = np.empty((len(reference),3))
    for site in range(len(reference)):
        for axis in range(3):
            centers[site,axis] = (reference[site,axis]+translation[axis])%length
    occupied = np.zeros(len(reference),dtype=np.int64)
    for p in range(n):
        if p == ignore:
            continue
        nearest = 0
        best_distance = np.inf
        for site in range(len(reference)):
            distance = periodic_distance_squared(positions[p],centers[site],length)
            if distance < best_distance:
                nearest = site
                best_distance = distance
        occupied[nearest] += 1
    return centers,occupied,best_fit


@njit(cache=True)
def _fitted_fcc_dispatch(positions,n,length,reference,ignore,canonical):
    if canonical and len(reference) > 4:
        return fitted_fcc_occupancy_fast(positions,n,length,reference,ignore)
    return fitted_fcc_occupancy_full_scan(positions,n,length,reference,ignore)


@njit(cache=True)
def fitted_fcc_occupancy(positions,n,length,reference,ignore=-1):
    """Guarded exact registry fit, with original scans for arbitrary references."""
    return _fitted_fcc_dispatch(positions,n,length,reference,ignore,
                               canonical_fcc_reference(reference,length))


@njit(cache=True)
def _fcc_vacancy_centers(positions,n,length,reference,ignore,canonical):
    centers,occupied,_ = _fitted_fcc_dispatch(positions,n,length,reference,ignore,canonical)
    return centers[occupied == 0]


@njit(cache=True)
def fcc_vacancy_centers(positions,n,length,reference,ignore=-1):
    return _fcc_vacancy_centers(positions,n,length,reference,ignore,
                               canonical_fcc_reference(reference,length))


@njit(cache=True)
def cube_mixture_density(point,centers,length,width,uniform_fraction):
    """All covering periodic cubes, with a uniform fallback for no centers."""
    if len(centers) == 0:
        return 1/length**3
    covering = 0
    for center in centers:
        inside = True
        for axis in range(3):
            delta = point[axis]-center[axis]
            delta -= length*np.rint(delta/length)
            if abs(delta) >= width:
                inside = False
                break
        if inside:
            covering += 1
    return uniform_fraction/length**3+(1-uniform_fraction)*covering/(len(centers)*(2*width)**3)


@njit(cache=True)
def vacancy_birth_density(point,positions,n,length,reference,width,uniform_fraction,ignore=-1):
    """Registry-guided proposal ONLY; exact reverse refit after tag removal."""
    if n < 0 or n > positions.shape[0] or ignore < -1 or ignore >= n:
        raise ValueError("Vacancy-density particle count or deleted index is invalid")
    if n-(1 if ignore >= 0 else 0) < 2:
        return 1/length**3
    centers = fcc_vacancy_centers(positions,n,length,reference,ignore)
    return cube_mixture_density(point,centers,length,width,uniform_fraction)


@njit(cache=True)
def _vacancy_birth_density(point,positions,n,length,reference,width,uniform_fraction,ignore,canonical):
    """Internal variant reuses the once-per-chunk canonical-reference guard."""
    if n-(1 if ignore >= 0 else 0) < 2:
        return 1/length**3
    centers = _fcc_vacancy_centers(positions,n,length,reference,ignore,canonical)
    return cube_mixture_density(point,centers,length,width,uniform_fraction)


@njit(cache=True)
def residual_vacancy_relocation(positions,n,center,diameter,radius,penalty,length,
        reference,width,uniform_fraction,canonical,stiffness,preferred,
        modes_re,modes_im,cells,new_re,new_im):
    """One uniform-tag fixed-N MH move using an identical residual both ways.

    The vacancy registry is a proposal, not a particle/site constraint. Both
    marginal densities sum every covering periodic cube of the SAME host
    with the tagged particle removed. Activity, N!, and number umbrellas
    cancel because this kernel leaves N unchanged.
    """
    particle = np.random.randint(n)
    proposal_centers = np.empty((0,3))
    if n >= 3:
        proposal_centers = _fcc_vacancy_centers(
            positions,n,length,reference,particle,canonical)
    point = np.empty(3)
    if len(proposal_centers) and np.random.random() >= uniform_fraction:
        site = np.random.randint(len(proposal_centers))
        for axis in range(3):
            point[axis] = (proposal_centers[site,axis]+width*(2*np.random.random()-1))%length
    else:
        for axis in range(3):
            point[axis] = np.random.random()*length
    before = local_pair_energy(positions[particle],positions,n,particle,
                              diameter,penalty,center,radius,length)
    after = local_pair_energy(point,positions,n,particle,
                             diameter,penalty,center,radius,length)
    if not np.isfinite(after):
        return False
    if stiffness != 0:
        bragg_proposal_into(modes_re,modes_im,positions[particle],point,-1.,1.,
                            length,cells,new_re,new_im)
        before += stiffness*(bragg_value(modes_re,modes_im,n)-preferred)**2
        after += stiffness*(bragg_value(new_re,new_im,n)-preferred)**2
    old_density = cube_mixture_density(positions[particle],proposal_centers,
                                      length,width,uniform_fraction)
    new_density = cube_mixture_density(point,proposal_centers,
                                      length,width,uniform_fraction)
    logratio = before-after+math.log(old_density)-math.log(new_density)
    if math.log(np.random.random()) < min(0.,logratio):
        if stiffness == 0:
            bragg_proposal_into(modes_re,modes_im,positions[particle],point,-1.,1.,
                                length,cells,new_re,new_im)
        positions[particle] = point
        modes_re[:] = new_re
        modes_im[:] = new_im
        return True
    return False


@njit(cache=True)
def valid_point(point, positions, n, skip, diameter, center, radius, length):
    if radius > 0 and periodic_distance_squared(point, center, length) < radius**2:
        return False
    for other in range(n):
        if other != skip and periodic_distance_squared(point, positions[other], length) < diameter**2:
            return False
    return True


@njit(cache=True)
def local_pair_energy(point, positions, n, skip, diameter, penalty, center, radius, length):
    if radius > 0 and periodic_distance_squared(point,center,length) < radius**2:
        return np.inf
    overlaps = 0
    for other in range(n):
        if other != skip and periodic_distance_squared(point,positions[other],length) < diameter**2:
            overlaps += 1
    if overlaps == 0:
        return 0.
    return penalty*overlaps


@njit(cache=True)
def total_pair_energy(positions,n,center,diameter,radius,penalty,length):
    overlaps = 0
    for i in range(n):
        if radius > 0 and periodic_distance_squared(positions[i],center,length) < radius**2:
            return np.inf
        for j in range(i):
            if periodic_distance_squared(positions[i],positions[j],length) < diameter**2:
                overlaps += 1
    if overlaps == 0:
        return 0.
    return penalty*overlaps


@njit(cache=True)
def supports(positions, n, center, diameter, radius, length):
    for i in range(n):
        if radius > 0 and periodic_distance_squared(positions[i], center, length) < radius**2:
            return False
        for j in range(i):
            if periodic_distance_squared(positions[i], positions[j], length) < diameter**2:
                return False
    return True


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def disordered_seed(n, length, seed):
    """Compress an initially ideal configuration for an independent start.

    This initializer does not supply statistical weights.  All production
    samples follow the same unrestricted replica targets whatever the start.
    """
    np.random.seed(seed)
    positions = np.random.random((n, 3))*length
    center = np.zeros(3)
    point = np.empty(3)
    diameter = 0.
    sweeps = 0
    while diameter < 1. and sweeps < 300000:
        step = .1+.4*(1-diameter)
        for repeat in range(30):
            for trial in range(n):
                particle = np.random.randint(n)
                for axis in range(3):
                    point[axis] = (positions[particle,axis]+step*(2*np.random.random()-1))%length
                if valid_point(point,positions,n,particle,diameter,center,0.,length):
                    positions[particle] = point
            sweeps += 1
        next_diameter = min(1.,diameter+.002)
        if supports(positions,n,center,next_diameter,0.,length):
            diameter = next_diameter
    return positions, diameter, sweeps


@njit(cache=True)
def run_chunk(positions, counts, centers, diameter, radius, activity, identities,
              length, sweeps, endpoint, sample_stride, reference, displacement,
              pair_penalty, umbrella_stiffness=None, umbrella_preferred=None,
              exchange_edges=None,cavity_nodes=None,snapshot_nodes=None,palm_transfer=False,
              biased_birth=False,birth_width=.12,birth_uniform_fraction=.2,birth_proposal=0,
              birth_scope=0,overlap_samples=None,number_stiffness=None,number_preferred=None,
              vacancy_relocation_probability=0.,vacancy_relocation_counts=None):
    nrep, cap = positions.shape[:2]
    if sample_stride <= 0 or sweeps < 0 or sweeps % sample_stride:
        raise ValueError("Chunk sweeps must be nonnegative and divisible by a positive sample stride")
    if overlap_samples is not None and overlap_samples.shape != (sweeps//sample_stride,nrep):
        raise ValueError("Pair-overlap diagnostic array must match saved frames and replicas")
    volume = length**3
    samples = np.empty((sweeps // sample_stride, nrep, 10))
    accept = np.zeros((nrep, 8), dtype=np.int64)
    if exchange_edges is None:
        edges = np.empty((nrep-1,2),dtype=np.int64)
        for i in range(nrep-1):
            edges[i,0] = i
            edges[i,1] = i+1
    else:
        edges = exchange_edges
    if cavity_nodes is None:
        cavity_index = np.arange(endpoint,nrep)
    else:
        cavity_index = cavity_nodes
    if snapshot_nodes is None:
        snapshot_index = np.array([endpoint,cavity_index[-1]])
    else:
        snapshot_index = snapshot_nodes
    configurations = np.empty((sweeps//sample_stride,len(snapshot_index),cap,3))
    diagnostic_node = np.zeros(nrep,dtype=np.bool_)
    for r in snapshot_index:
        diagnostic_node[r] = True
    next_radius_by_node = radius.copy()
    for i in range(len(cavity_index)-1):
        next_radius_by_node[cavity_index[i]] = radius[cavity_index[i+1]]
    swaps = np.zeros((len(edges), 2), dtype=np.int64)
    point = np.empty(3)
    stiffness = np.zeros(nrep) if umbrella_stiffness is None else umbrella_stiffness
    preferred = np.zeros(nrep) if umbrella_preferred is None else umbrella_preferred
    number_bias = np.zeros(nrep) if number_stiffness is None else number_stiffness
    number_target = np.zeros(nrep) if number_preferred is None else number_preferred
    if number_bias.shape != (nrep,) or number_target.shape != (nrep,):
        raise ValueError("Number-umbrella arrays must match replicas")
    for r in range(nrep):
        if (not np.isfinite(number_bias[r]) or number_bias[r] < 0
                or not np.isfinite(number_target[r]) or not 0 <= number_target[r] <= cap):
            raise ValueError("Number-umbrella stiffness must be nonnegative and target within capacity")
    for r in cavity_index:
        if number_bias[r] != 0:
            raise ValueError("Physical cavity path requires zero number-umbrella stiffness")
    if (not np.isfinite(vacancy_relocation_probability)
            or not 0 <= vacancy_relocation_probability <= 1):
        raise ValueError("Vacancy-relocation probability must be finite and in [0,1]")
    if vacancy_relocation_counts is not None:
        if vacancy_relocation_counts.shape != (nrep,2):
            raise ValueError("Vacancy-relocation counters must have shape (replicas,2)")
        if np.any(vacancy_relocation_counts < 0):
            raise ValueError("Vacancy-relocation counters must be nonnegative")
    if (biased_birth or vacancy_relocation_probability > 0) and (not np.isfinite(birth_width) or not 0 < birth_width < length/2
                         or not np.isfinite(birth_uniform_fraction) or not 0 < birth_uniform_fraction <= 1):
        raise ValueError("Birth mixture needs positive finite width below half-box and uniform fraction in (0,1]")
    if birth_proposal != 0 and birth_proposal != 1:
        raise ValueError("Birth proposal must be extrapolation(0) or registry-vacancy(1)")
    if birth_scope != 0 and birth_scope != 1:
        raise ValueError("Birth scope must be physical(0) or structured(1)")
    birth_biased_node = np.zeros(nrep,dtype=np.bool_)
    relocation_node = np.zeros(nrep,dtype=np.bool_)
    for r in range(nrep):
        physical = pair_penalty[r] == np.inf and stiffness[r] == 0
        structured = pair_penalty[r] >= 8 or stiffness[r] > 0
        # A frozen target-index mask, never a condition on the current
        # configuration.  Full pair and umbrella increments remain in MH.
        birth_biased_node[r] = (biased_birth and diameter[r] == diameter[endpoint]
                               and (physical if birth_scope == 0 else structured))
        relocation_node[r] = diameter[r] == diameter[endpoint] and structured
    if palm_transfer:
        cavity = cavity_index[-1]
        if (pair_penalty[endpoint] != np.inf or pair_penalty[cavity] != np.inf
            or stiffness[endpoint] != 0 or stiffness[cavity] != 0
            or number_bias[endpoint] != 0 or number_bias[cavity] != 0
            or activity[endpoint] != activity[cavity]
            or diameter[endpoint] != diameter[cavity] or diameter[endpoint] <= 0
            or radius[endpoint] != 0 or radius[cavity] != diameter[endpoint]):
            raise ValueError("Marked transfer requires matched unbiased R=0 and R=sigma hard-sphere targets")
    cells_for_bias = int(round((reference.shape[0]/4.)**(1./3.)))
    canonical_registry = canonical_fcc_reference(reference,length) and len(reference) > 4
    modes_re = np.zeros((nrep,7))
    modes_im = np.zeros((nrep,7))
    new_re = np.empty(7)
    new_im = np.empty(7)
    for r in range(nrep):
        for p in range(counts[r]):
            modes_re[r],modes_im[r] = bragg_delta(modes_re[r],modes_im[r],positions[r,p],length,cells_for_bias,1.)
    index = 0
    for sweep in range(sweeps):
        for r in range(nrep):
            n = counts[r]
            step = displacement[r]
            for move in range(max(n, 1)):
                if n == 0:
                    break
                particle = np.random.randint(n)
                for axis in range(3):
                    point[axis] = (positions[r, particle, axis] + step*(2*np.random.random()-1)) % length
                accept[r, 0] += 1
                before = local_pair_energy(positions[r,particle],positions[r],n,particle,diameter[r],pair_penalty[r],centers[r],radius[r],length)
                after = local_pair_energy(point,positions[r],n,particle,diameter[r],pair_penalty[r],centers[r],radius[r],length)
                if not np.isfinite(after):
                    continue
                if stiffness[r] != 0:
                    bragg_proposal_into(modes_re[r],modes_im[r],positions[r,particle],point,-1.,1.,length,cells_for_bias,new_re,new_im)
                    before += stiffness[r]*(bragg_value(modes_re[r],modes_im[r],n)-preferred[r])**2
                    after += stiffness[r]*(bragg_value(new_re,new_im,n)-preferred[r])**2
                if math.log(np.random.random()) < min(0.,before-after):
                    if stiffness[r] == 0:
                        bragg_proposal_into(modes_re[r],modes_im[r],positions[r,particle],point,-1.,1.,length,cells_for_bias,new_re,new_im)
                    positions[r, particle] = point
                    modes_re[r] = new_re
                    modes_im[r] = new_im
                    accept[r, 1] += 1
            # Fixed attempt count: a state-dependent number of birth/death
            # kernels would not in general preserve the grand-canonical law.
            for move in range(max(reference.shape[0] // 4, 1)):
                accept[r, 2] += 1
                if np.random.random() < .5:
                    if n < cap:
                        proposal_centers = np.empty((0,3))
                        if birth_biased_node[r] and birth_proposal == 1 and n >= 2:
                            proposal_centers = _fcc_vacancy_centers(positions[r],n,length,reference,-1,canonical_registry)
                        if birth_biased_node[r] and birth_proposal == 1 and len(proposal_centers) and np.random.random() >= birth_uniform_fraction:
                            site = np.random.randint(len(proposal_centers))
                            for axis in range(3):
                                point[axis] = (proposal_centers[site,axis]+birth_width*(2*np.random.random()-1))%length
                        elif birth_biased_node[r] and birth_proposal == 0 and n >= 2 and np.random.random() >= birth_uniform_fraction:
                            i = np.random.randint(n)
                            j = np.random.randint(n-1)
                            if j >= i:
                                j += 1
                            for axis in range(3):
                                point[axis] = (2*positions[r,i,axis]-positions[r,j,axis]+birth_width*(2*np.random.random()-1))%length
                        else:
                            for axis in range(3):
                                point[axis] = np.random.random()*length
                        energy = local_pair_energy(point,positions[r],n,-1,diameter[r],pair_penalty[r],centers[r],radius[r],length)
                        if not np.isfinite(energy):
                            continue
                        if stiffness[r] != 0:
                            bragg_proposal_into(modes_re[r],modes_im[r],point,point,0.,1.,length,cells_for_bias,new_re,new_im)
                            energy += stiffness[r]*((bragg_value(new_re,new_im,n+1)-preferred[r])**2-(bragg_value(modes_re[r],modes_im[r],n)-preferred[r])**2)
                        density = 1/volume
                        if birth_biased_node[r]:
                            if birth_proposal == 0:
                                density = extrapolation_birth_density(point,positions[r],n,length,birth_width,birth_uniform_fraction)
                            else:
                                density = cube_mixture_density(point,proposal_centers,length,birth_width,birth_uniform_fraction)
                        logratio = activity[r] - math.log((n+1)*density) - energy
                        if number_bias[r] != 0:
                            logratio -= number_bias[r]*(1+2*(n-number_target[r]))
                        if math.log(np.random.random()) < min(0., logratio):
                            if stiffness[r] == 0:
                                bragg_proposal_into(modes_re[r],modes_im[r],point,point,0.,1.,length,cells_for_bias,new_re,new_im)
                            positions[r, n] = point
                            modes_re[r] = new_re
                            modes_im[r] = new_im
                            n += 1
                            accept[r, 3] += 1
                elif n > 0:
                    particle = np.random.randint(n)
                    energy = local_pair_energy(positions[r,particle],positions[r],n,particle,diameter[r],pair_penalty[r],centers[r],radius[r],length)
                    if stiffness[r] != 0:
                        bragg_proposal_into(modes_re[r],modes_im[r],positions[r,particle],point,-1.,0.,length,cells_for_bias,new_re,new_im)
                        energy += stiffness[r]*((bragg_value(modes_re[r],modes_im[r],n)-preferred[r])**2-(bragg_value(new_re,new_im,n-1)-preferred[r])**2)
                    density = 1/volume
                    if birth_biased_node[r]:
                        if birth_proposal == 0:
                            density = extrapolation_birth_density(positions[r,particle],positions[r],n,length,birth_width,birth_uniform_fraction,particle)
                        else:
                            density = _vacancy_birth_density(positions[r,particle],positions[r],n,length,reference,birth_width,birth_uniform_fraction,particle,canonical_registry)
                    if number_bias[r] != 0:
                        energy += number_bias[r]*(2*(n-number_target[r])-1)
                    if math.log(np.random.random()) < min(0., math.log(n*density)-activity[r]+energy):
                        if stiffness[r] == 0:
                            bragg_proposal_into(modes_re[r],modes_im[r],positions[r,particle],point,-1.,0.,length,cells_for_bias,new_re,new_im)
                        positions[r, particle] = positions[r, n-1]
                        modes_re[r] = new_re
                        modes_im[r] = new_im
                        n -= 1
                        accept[r, 3] += 1
            counts[r] = n
            # Default-off performs no random draw and preserves the legacy
            # trajectory bit for bit. Enabled moves keep N and center fixed.
            if (vacancy_relocation_probability > 0 and relocation_node[r] and n > 0
                    and np.random.random() < vacancy_relocation_probability):
                if vacancy_relocation_counts is not None:
                    vacancy_relocation_counts[r,0] += 1
                relocated = residual_vacancy_relocation(positions[r],n,centers[r],
                    diameter[r],radius[r],pair_penalty[r],length,reference,birth_width,
                    birth_uniform_fraction,canonical_registry,stiffness[r],preferred[r],
                    modes_re[r],modes_im[r],cells_for_bias,new_re,new_im)
                if relocated and vacancy_relocation_counts is not None:
                    vacancy_relocation_counts[r,1] += 1
            accept[r, 4] += 1
            for axis in range(3):
                if np.random.random() < .1:
                    point[axis] = np.random.random()*length
                else:
                    point[axis] = (centers[r, axis] + .12*(2*np.random.random()-1)) % length
            okay = True
            for p in range(n):
                if periodic_distance_squared(positions[r, p], point, length) < radius[r]**2:
                    okay = False
                    break
            if okay:
                centers[r] = point
                accept[r, 5] += 1
        for parity in range(2):
            for edge in range(parity,len(edges),2):
                r = edges[edge,0]
                s = edges[edge,1]
                swaps[edge, 0] += 1
                before = total_pair_energy(positions[r],counts[r],centers[r],diameter[r],radius[r],pair_penalty[r],length)+total_pair_energy(positions[s],counts[s],centers[s],diameter[s],radius[s],pair_penalty[s],length)
                after = total_pair_energy(positions[s],counts[s],centers[s],diameter[r],radius[r],pair_penalty[r],length)+total_pair_energy(positions[r],counts[r],centers[r],diameter[s],radius[s],pair_penalty[s],length)
                order_r = bragg_value(modes_re[r],modes_im[r],counts[r])
                order_s = bragg_value(modes_re[s],modes_im[s],counts[s])
                before += stiffness[r]*(order_r-preferred[r])**2+stiffness[s]*(order_s-preferred[s])**2
                after += stiffness[r]*(order_s-preferred[r])**2+stiffness[s]*(order_r-preferred[s])**2
                if number_bias[r] != 0 or number_bias[s] != 0:
                    before += number_bias[r]*(counts[r]-number_target[r])**2+number_bias[s]*(counts[s]-number_target[s])**2
                    after += number_bias[r]*(counts[s]-number_target[r])**2+number_bias[s]*(counts[r]-number_target[s])**2
                logratio = (activity[r]-activity[s])*(counts[s]-counts[r])+before-after
                if np.isfinite(after) and math.log(np.random.random()) < min(0., logratio):
                        temporary = positions[r].copy()
                        positions[r] = positions[s]
                        positions[s] = temporary
                        temp = centers[r].copy()
                        centers[r] = centers[s]
                        centers[s] = temp
                        value = counts[r]
                        counts[r] = counts[s]
                        counts[s] = value
                        value = identities[r]
                        identities[r] = identities[s]
                        identities[s] = value
                        temp_modes = modes_re[r].copy()
                        modes_re[r] = modes_re[s]
                        modes_re[s] = temp_modes
                        temp_modes = modes_im[r].copy()
                        modes_im[r] = modes_im[s]
                        modes_im[s] = temp_modes
                        swaps[edge, 1] += 1
        if palm_transfer:
            base = endpoint
            cavity = cavity_index[-1]
            nbase = counts[base]
            ncavity = counts[cavity]
            accept[base,6] += 1
            if nbase > 0 and ncavity+1 <= cap and np.random.random() < min(1.,nbase/(ncavity+1.)):
                particle = np.random.randint(nbase)
                old_base = positions[base].copy()
                tag = old_base[particle].copy()
                old_base_re = modes_re[base].copy()
                old_base_im = modes_im[base].copy()
                # New base is the old cavity host with its center restored as
                # an ordinary particle.  The base's uniform center is kept.
                bragg_proposal_into(modes_re[cavity],modes_im[cavity],centers[cavity],centers[cavity],0.,1.,length,cells_for_bias,new_re,new_im)
                modes_re[base] = new_re
                modes_im[base] = new_im
                positions[base,:ncavity] = positions[cavity,:ncavity]
                positions[base,ncavity] = centers[cavity]
                # New cavity is the old base with a uniformly tagged particle
                # removed and that tag serving as its cavity center.
                positions[cavity,:nbase] = old_base[:nbase]
                positions[cavity,particle] = old_base[nbase-1]
                centers[cavity] = tag
                bragg_proposal_into(old_base_re,old_base_im,tag,tag,-1.,0.,length,cells_for_bias,new_re,new_im)
                modes_re[cavity] = new_re
                modes_im[cavity] = new_im
                counts[base] = ncavity+1
                counts[cavity] = nbase-1
                identity = identities[base]
                identities[base] = identities[cavity]
                identities[cavity] = identity
                accept[base,7] += 1
        if (sweep+1) % sample_stride == 0:
            for s in range(len(snapshot_index)):
                configurations[index,s] = positions[snapshot_index[s]]
            for r in range(nrep):
                n = counts[r]
                if overlap_samples is not None:
                    overlaps = 0
                    for p in range(n):
                        for other in range(p):
                            if periodic_distance_squared(positions[r,p],positions[r,other],length) < diameter[r]**2:
                                overlaps += 1
                    overlap_samples[index,r] = overlaps
                hole = True
                next_radius = next_radius_by_node[r]
                mind = length**2
                for p in range(n):
                    distance = periodic_distance_squared(positions[r, p], centers[r], length)
                    mind = min(mind, distance)
                    if distance < next_radius**2:
                        hole = False
                samples[index,r,0] = n
                samples[index,r,1] = bragg_value(modes_re[r],modes_im[r],n)
                samples[index,r,2] = identities[r]
                samples[index,r,3] = 1. if hole else 0.
                samples[index,r,6] = math.sqrt(mind)
                samples[index,r,7] = n == cap
                if not diagnostic_node[r]:
                    # Registry/deletion diagnostics are physical-node
                    # observables.  Auxiliary nodes retain mixing traces only.
                    samples[index,r,4] = -1.
                    samples[index,r,5] = -1.
                    samples[index,r,8] = -1.
                    samples[index,r,9] = -1.
                    continue
                order = 0.
                mode_re = np.zeros(7)
                mode_im = np.zeros(7)
                cells = int(round((reference.shape[0]/4.)**(1./3.)))
                vectors = np.array([[cells, cells, cells], [cells, cells, -cells],
                                    [cells, -cells, cells], [-cells, cells, cells],
                                    [2*cells, 0, 0], [0, 2*cells, 0], [0, 0, 2*cells]])
                for v in range(7):
                    re = 0.
                    im = 0.
                    for p in range(n):
                        phase = 2*np.pi/length * np.dot(vectors[v].astype(np.float64), positions[r, p])
                        re += math.cos(phase)
                        im += math.sin(phase)
                    if n:
                        order += math.sqrt(re*re+im*im)/(7*n)
                    mode_re[v] = re
                    mode_im[v] = im
                if abs(order-bragg_value(modes_re[r],modes_im[r],n)) > 1e-8:
                    raise ValueError("Fourier-mode cache disagrees with independent reconstruction")
                deletion_label_change = 0.
                if n > 1:
                    baseline_label = 1. if order > .45 else 0.
                    for p in range(n):
                        deletion_order = 0.
                        for v in range(7):
                            phase = 2*np.pi/length * np.dot(vectors[v].astype(np.float64), positions[r,p])
                            re = mode_re[v] - math.cos(phase)
                            im = mode_im[v] - math.sin(phase)
                            deletion_order += math.sqrt(re*re+im*im)/(7*(n-1))
                        deletion_label_change += (1. if deletion_order > .45 else 0.)-baseline_label
                elif n == 1:
                    deletion_label_change = -1. if order > .45 else 0.
                # Reuse the independently tested deterministic registry fit.
                _,occupied,best_registry_fit = _fitted_fcc_dispatch(positions[r],n,length,reference,-1,canonical_registry)
                samples[index, r, 0] = n
                samples[index, r, 1] = order
                samples[index, r, 2] = identities[r]
                samples[index, r, 3] = 1. if hole else 0.
                samples[index, r, 4] = np.sum(occupied == 0)
                samples[index, r, 5] = np.sum(np.maximum(occupied-1, 0))
                samples[index, r, 6] = math.sqrt(mind)
                samples[index, r, 7] = n == cap
                samples[index, r, 8] = deletion_label_change
                samples[index, r, 9] = math.sqrt(best_registry_fit/max(n,1))/(length/(cells*math.sqrt(2.)))
            index += 1
    return samples, accept, swaps, configurations


def initialize(sites, length, diameter, radius, cap, seed, mode="crystal"):
    rng = np.random.default_rng(seed)
    ref = fcc_reference(sites, length)
    count = np.full(len(diameter), sites, dtype=np.int64)
    positions = np.zeros((len(diameter), cap, 3))
    centers = np.zeros((len(diameter), 3))
    fluid = None
    if mode == "fluid":
        fluid, maximum_diameter, _ = disordered_seed(max(4, sites-round(.09*sites)), length, seed)
        if maximum_diameter < 1.:
            raise RuntimeError("Independent fluid initializer failed to reach the hard endpoint")
    for j in range(len(diameter)):
        positions[j, :sites] = ref
        centers[j] = ref[0]
        if radius[j] > 0:
            positions[j, 0] = positions[j, sites-1]
            count[j] -= 1
        if mode == "ideal" and diameter[j] == 0:
            positions[j, :sites] = rng.uniform(0, length, (sites, 3))
        if mode == "fluid":
            count[j] = len(fluid)
            positions[j, :len(fluid)] = fluid
            centers[j] = fluid[0]
            if radius[j] > 0:
                positions[j, 0] = positions[j, len(fluid)-1]
                count[j] -= 1
        if not supports(positions[j], count[j], centers[j], diameter[j], radius[j], length):
            raise ValueError("Initialization violates hard support")
    return positions, count, centers, ref


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def file_sha256(path,chunk_size=1024*1024):
    """Hash an archive with bounded memory rather than reading it twice."""
    if not isinstance(chunk_size,int) or chunk_size <= 0:
        raise ValueError("Hash chunk size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def burn_in_displacement(displacement,accept,length,frozen=False):
    """Adapt only symmetric proposal scales; frozen starts retain exact inputs."""
    if frozen:
        return displacement
    rates = accept[:,1]/np.maximum(accept[:,0],1)
    return np.clip(displacement*np.exp(.5*(rates-.3)),.01,length/2)


def frozen_probe_displacements(displacement,probe_nodes,probe_activity,
                               previous_steps,previous_nodes,previous_activity):
    """Copy matching physical-probe proposals, leaving new targets explicit.

    Activities rather than array positions identify probes: a requested
    regrid can retain common targets without importing a different target's
    proposal scale.  The returned node list makes partial matching visible.
    """
    for nodes,activity,steps in ((probe_nodes,probe_activity,displacement),
                                 (previous_nodes,previous_activity,previous_steps)):
        if (nodes.ndim != 1 or nodes.dtype.kind not in "iu"
                or activity.shape != nodes.shape or not np.isfinite(activity).all()
                or len(np.unique(nodes)) != len(nodes) or np.any(nodes<0)
                or np.any(nodes>=len(steps)) or len(np.unique(activity)) != len(activity)
                or not np.isfinite(steps).all() or np.any(steps<0)):
            raise ValueError("Probe proposal mapping requires distinct valid nodes, activities and finite nonnegative scales")
    result = displacement.copy()
    matched = []
    for node,activity in zip(probe_nodes,probe_activity):
        candidates = np.flatnonzero(np.isclose(previous_activity,activity,rtol=0,atol=1e-12))
        if len(candidates)>1:
            raise ValueError("Probe proposal mapping has ambiguous matching activities")
        if len(candidates):
            result[node] = previous_steps[previous_nodes[candidates[0]]]
            matched.append(node)
    return result,np.array(matched,dtype=np.int64)


def execution_source_snapshot():
    snapshot = ROOT/"data_raw/source_snapshots"/f"continuum_replica_exchange_{EXECUTION_SOURCE_SHA256[:16]}.py"
    if not snapshot.exists():
        snapshot.write_bytes(EXECUTION_SOURCE_BYTES)
    return snapshot


def write_state_checkpoint(prefix,settings,completed,positions,counts,centers,
                           diameter,radius,activity,pair_penalty,stiffness,preferred,
                           endpoint,volume,displacement,cavity_nodes,exchange_edges,
                           number_stiffness,number_preferred):
    """Atomic state-only recovery, including frozen targets and source snapshot.

    A restarted campaign uses a new RNG seed and reports a new history; this
    file is not a fabricated continuation of already discarded observations.
    """
    path = prefix.with_suffix(".checkpoint.npz")
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle,final_positions=positions,final_counts=counts,
                            final_centers=centers,diameter=diameter,radius=radius,
                            activity=activity,pair_penalty=pair_penalty,
                            umbrella_stiffness=stiffness,umbrella_preferred=preferred,
                            number_stiffness=number_stiffness,number_preferred=number_preferred,
                            endpoint=endpoint,volume=volume,displacement=displacement,
                            cavity_nodes=cavity_nodes,exchange_edges=exchange_edges,
                            completed_sweeps=completed)
    os.replace(temporary,path)
    snapshot = execution_source_snapshot()
    save_json(prefix.with_suffix(".checkpoint.json"),{
        "kind":"state-only recovery checkpoint; no production estimates",
        "completed_sweeps":completed,"settings":{k:str(v) if isinstance(v,Path) else v for k,v in vars(settings).items()},
        "source_sha256":EXECUTION_SOURCE_SHA256,
        "source_snapshot":str(snapshot.relative_to(ROOT)),
        "raw_sha256":file_sha256(path),
        "restart_semantics":"new history with supplied seed; target activities loaded exactly, new burn-in required if adapting",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--mu", type=float, default=16.)
    parser.add_argument("--diameter-stages", type=int, default=64)
    parser.add_argument("--cavity-stages", type=int, default=32)
    parser.add_argument("--equil-sweeps", type=int, default=2000)
    parser.add_argument("--production-sweeps", type=int, default=10000)
    parser.add_argument("--chunk-sweeps", type=int, default=500)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--probe-configuration-stride",type=int,default=1,help="Save probe coordinates every this many diagnostic frames; base/cavity coordinates remain full-cadence")
    parser.add_argument("--overlap-diagnostics",action="store_true",help="Save pair-overlap counts at every diagnostic frame without changing Monte Carlo random draws")
    parser.add_argument("--seed", type=int, default=2026092901)
    parser.add_argument("--cap", type=int, default=80)
    parser.add_argument("--initialization", choices=("crystal", "fluid"), default="crystal")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--auxiliary-from",type=Path,help="Reuse frozen auxiliary activities with an independent fresh initialization; never copy phase weights")
    parser.add_argument("--tempering",choices=("diameter","soft","umbrella","loop","hard","compact"),default="diameter")
    parser.add_argument("--compact-order-penalty",type=float,default=8.,help="Fixed soft pair penalty during the compact order-parameter bridge; lowering it permits faster defect rearrangements")
    parser.add_argument("--compact-low-order-return",action="store_true",help="Add a low-Bragg auxiliary hardening route feeding the same unbiased physical endpoint as the crystalline route")
    parser.add_argument("--compact-number-stiffness",type=float,default=0.,help="Optional auxiliary low-B nu(N-N*)^2 bias, reversibly staged on/off with N*=V/1.06448; physical targets remain unbiased")
    parser.add_argument("--calibrate-activity",action="store_true",help="Adjust auxiliary activities during burn-in only; physical activities stay fixed")
    parser.add_argument("--freeze-displacement",action="store_true",help="Keep loaded/default symmetric displacement proposals fixed during burn-in too, for matched-kernel independent starts")
    parser.add_argument("--probe-mu-max",type=float,help="Append unbiased hard-sphere activity probes from --mu to this value")
    parser.add_argument("--probe-mu-min",type=float,help="Also extend the unbiased hard activity leg below --mu to promote deletion and melting")
    parser.add_argument("--probe-mu-count",type=int,default=12)
    parser.add_argument("--palm-transfer",action="store_true",help="Exact tagged-particle exchange between unconstrained and R=sigma targets")
    parser.add_argument("--biased-birth",action="store_true",help="Use a normalized full-support configuration-dependent birth proposal")
    parser.add_argument("--birth-proposal",choices=("extrapolation","vacancy"),default="extrapolation",help="Normalized biased proposal geometry, used only with --biased-birth")
    parser.add_argument("--birth-scope",choices=("physical","structured"),default="physical",help="Physical: unbiased hard nodes only. Structured: same-diameter targets with pair penalty >=8 or a nonzero order umbrella; low-penalty reservoirs keep uniform births")
    parser.add_argument("--birth-width",type=float,default=.12)
    parser.add_argument("--birth-uniform-fraction",type=float,default=.2)
    parser.add_argument("--output", type=Path, default=ROOT/"data_raw/continuum_switching_pilot")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.sites <= 0 or not np.isfinite(args.mu):
        parser.error("Site count must be positive and physical chemical activity finite")
    if any(v is not None and not np.isfinite(v) for v in (args.probe_mu_min,args.probe_mu_max)):
        parser.error("Probe chemical activities must be finite")
    if args.auxiliary_from and (args.restart or args.calibrate_activity):
        parser.error("A frozen auxiliary schedule requires fresh initialization and no activity calibration")
    if args.birth_proposal == "vacancy" and not args.biased_birth:
        parser.error("Registry-vacancy proposal requires --biased-birth")
    if args.birth_scope != "physical" and not args.biased_birth:
        parser.error("Expanded birth scope requires --biased-birth")
    if not np.isfinite(args.compact_order_penalty) or not 0 < args.compact_order_penalty < 16:
        parser.error("Compact order-bridge penalty must lie strictly between zero and 16")
    if args.tempering != "compact" and args.compact_order_penalty != 8:
        parser.error("A changed compact order penalty requires --tempering compact")
    if args.tempering != "compact" and args.compact_low_order_return:
        parser.error("A low-order return requires --tempering compact")
    if not np.isfinite(args.compact_number_stiffness) or args.compact_number_stiffness < 0:
        parser.error("Number-umbrella strength must be finite and nonnegative")
    if args.compact_number_stiffness and not (args.tempering == "compact" and args.compact_low_order_return):
        parser.error("A number umbrella requires the compact low-order return graph")
    if args.cap < args.sites or args.diameter_stages < 2 or args.cavity_stages < 1:
        parser.error("Invalid cap or ladder length")
    if args.sample_stride <= 0 or args.probe_configuration_stride <= 0 or args.chunk_sweeps <= 0 or args.production_sweeps <= 0 or args.equil_sweeps < 0:
        parser.error("Stride, chunk and production sweeps must be positive; equilibration nonnegative")
    if args.chunk_sweeps % args.sample_stride or args.equil_sweeps % args.chunk_sweeps or args.production_sweeps % args.chunk_sweeps:
        parser.error("Sweep counts must be multiples of chunk/stride")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [args.output.with_suffix(s) for s in (".npz", ".json")]
    checkpoint_paths = [args.output.with_suffix(s) for s in (".checkpoint.npz",".checkpoint.json")]
    if not args.overwrite and any(p.exists() for p in paths+checkpoint_paths):
        parser.error("Output exists; use a distinct prefix")
    start = time.time()
    execution_source_snapshot()
    volume, diameter, radius, activity, endpoint = make_ladder(args.sites, chemical_potential=args.mu, diameter_stages=args.diameter_stages, cavity_stages=args.cavity_stages)
    pair_penalty = np.full(len(diameter),np.inf)
    stiffness = np.zeros(len(diameter))
    preferred = np.zeros(len(diameter))
    if args.tempering == "soft":
        volume,diameter,radius,activity,endpoint,pair_penalty = make_soft_ladder(args.sites,chemical_potential=args.mu,cavity_stages=args.cavity_stages)
        stiffness = np.zeros(len(diameter))
        preferred = np.zeros(len(diameter))
    if args.tempering == "umbrella":
        volume,diameter,radius,activity,endpoint,pair_penalty,stiffness,preferred = make_umbrella_ladder(args.sites,chemical_potential=args.mu,cavity_stages=args.cavity_stages)
    if args.tempering == "hard":
        volume,diameter,radius,activity,endpoint,pair_penalty,stiffness,preferred = make_hard_ladder(args.sites,chemical_potential=args.mu,cavity_stages=args.cavity_stages)
    cavity_nodes = np.arange(endpoint,len(diameter),dtype=np.int64)
    exchange_edges = np.array([(i,i+1) for i in range(len(diameter)-1)],dtype=np.int64)
    if args.tempering == "loop":
        volume,diameter,radius,activity,endpoint,pair_penalty,stiffness,preferred,cavity_nodes,exchange_edges = make_loop_ladder(args.sites,chemical_potential=args.mu,cavity_stages=args.cavity_stages)
    if args.tempering == "compact":
        volume,diameter,radius,activity,endpoint,pair_penalty,stiffness,preferred,cavity_nodes,exchange_edges = make_compact_ladder(args.sites,chemical_potential=args.mu,cavity_stages=args.cavity_stages,order_penalty=args.compact_order_penalty,low_order_return=args.compact_low_order_return)
    number_stiffness = np.zeros(len(diameter))
    number_preferred = np.zeros(len(diameter))
    if args.compact_number_stiffness:
        ladder = (volume,diameter,radius,activity,endpoint,pair_penalty,stiffness,preferred,cavity_nodes,exchange_edges)
        volume,diameter,radius,activity,endpoint,pair_penalty,stiffness,preferred,cavity_nodes,exchange_edges,number_stiffness,number_preferred = add_compact_number_bridge(ladder,args.compact_number_stiffness)
    probe_nodes = np.zeros(0,dtype=np.int64)
    probe_activity = np.zeros(0)
    if args.probe_mu_max is not None or args.probe_mu_min is not None:
        minimum = args.mu if args.probe_mu_min is None else args.probe_mu_min
        maximum = args.mu if args.probe_mu_max is None else args.probe_mu_max
        begin = len(diameter)
        try:
            probes,probe_nodes,probe_edges = activity_probe_branch(args.mu,minimum,maximum,args.probe_mu_count,endpoint,begin)
        except ValueError as error:
            parser.error(str(error))
        probe_activity = probes.copy()
        diameter = np.concatenate((diameter,np.ones(len(probes))))
        radius = np.concatenate((radius,np.zeros(len(probes))))
        activity = np.concatenate((activity,probes))
        pair_penalty = np.concatenate((pair_penalty,np.full(len(probes),np.inf)))
        stiffness = np.concatenate((stiffness,np.zeros(len(probes))))
        preferred = np.concatenate((preferred,np.zeros(len(probes))))
        number_stiffness = np.concatenate((number_stiffness,np.zeros(len(probes))))
        number_preferred = np.concatenate((number_preferred,np.zeros(len(probes))))
        exchange_edges = np.concatenate((exchange_edges,probe_edges))
    snapshot_nodes = np.concatenate((np.array([endpoint,cavity_nodes[-1]],dtype=np.int64),probe_nodes))
    if args.cap < math.ceil(6*volume/np.pi):
        parser.error("cap must exceed the geometric physical hard-sphere packing bound ceil(6V/pi)")
    saved = args.production_sweeps//args.sample_stride
    probe_saved = math.ceil(saved/args.probe_configuration_stride) if len(probe_nodes) else 0
    raw_storage = (saved*(len(diameter)*10+2*args.cap*3)
                   +probe_saved*len(probe_nodes)*args.cap*3)*8
    if args.overlap_diagnostics:
        raw_storage += saved*len(diameter)*8
    if raw_storage > 200*1024**2:
        parser.error("Projected trajectory exceeds 200 MiB; use shorter batches and --restart")
    length = volume**(1/3)
    if (not np.isfinite(args.birth_width) or not 0 < args.birth_width < length/2
            or not np.isfinite(args.birth_uniform_fraction) or not 0 < args.birth_uniform_fraction <= 1):
        parser.error("Birth width must be below half-box; uniform fraction must lie in (0,1]")
    ref = fcc_reference(args.sites,length)
    identities = np.arange(len(diameter))
    displacement = .12 + .4*(1-diameter)
    imported_probe_displacement_nodes = np.zeros(0,dtype=np.int64)
    auxiliary_digest = None
    restart_digest = None
    if args.auxiliary_from:
        sidecar = args.auxiliary_from.with_suffix(".json")
        auxiliary_digest = file_sha256(args.auxiliary_from)
        if not sidecar.exists() or json.loads(sidecar.read_text()).get("raw_sha256") != auxiliary_digest:
            parser.error("Frozen auxiliary schedule requires an archive with a matching SHA-256 sidecar")
        with np.load(args.auxiliary_from,allow_pickle=False) as previous:
            # Optional physical activity probes may be added or removed.
            # Only the shared auxiliary/cavity graph determines this import.
            core_mask = np.ones(len(activity),dtype=bool)
            core_mask[probe_nodes] = False
            core_nodes = np.flatnonzero(core_mask)
            for name,values in (("diameter",diameter),("radius",radius),("pair_penalty",pair_penalty),("umbrella_stiffness",stiffness),("umbrella_preferred",preferred)):
                if (name not in previous or len(previous[name]) <= core_nodes[-1]
                        or not np.array_equal(previous[name][core_nodes],values[core_nodes])):
                    parser.error(f"Frozen auxiliary {name} differs from the requested graph")
            for name,values in (("number_stiffness",number_stiffness),("number_preferred",number_preferred)):
                old = previous[name] if name in previous else np.zeros(len(previous["diameter"]))
                if len(old) <= core_nodes[-1] or not np.array_equal(old[core_nodes],values[core_nodes]):
                    parser.error(f"Frozen auxiliary {name} differs from the requested graph")
            if not np.array_equal(previous["cavity_nodes"],cavity_nodes):
                parser.error("Frozen cavity path differs")
            old_edges = previous["exchange_edges"]
            old_core_edges = old_edges[np.isin(old_edges,core_nodes).all(axis=1)]
            new_core_edges = exchange_edges[np.isin(exchange_edges,core_nodes).all(axis=1)]
            if not np.array_equal(old_core_edges,new_core_edges):
                parser.error("Frozen auxiliary exchange graph differs")
            old_activity = previous["activity"]
            old_steps = previous["displacement"]
            if (old_activity.ndim != 1 or len(old_activity) <= core_nodes[-1] or not np.isfinite(old_activity).all()
                    or old_steps.ndim != 1 or len(old_steps) <= core_nodes[-1] or not np.isfinite(old_steps).all()
                    or np.any(old_steps<0) or float(previous["volume"]) != volume):
                parser.error("Frozen auxiliary activities, volume or proposal scales are invalid")
            mask = np.ones(len(activity),dtype=bool)
            mask[cavity_nodes] = False
            mask[probe_nodes] = False
            auxiliary_nodes = np.flatnonzero(mask)
            activity[auxiliary_nodes] = old_activity[auxiliary_nodes]
            displacement[core_nodes] = old_steps[core_nodes]
            try:
                displacement,imported_probe_displacement_nodes = frozen_probe_displacements(
                    displacement,probe_nodes,probe_activity,old_steps,
                    previous["probe_nodes"] if "probe_nodes" in previous else np.zeros(0,dtype=np.int64),
                    previous["physical_probe_activity"] if "physical_probe_activity" in previous else np.zeros(0))
            except ValueError as error:
                parser.error(str(error))
    if args.restart:
        sidecar = args.restart.with_suffix(".json")
        restart_digest = file_sha256(args.restart)
        if sidecar.exists():
            previous_metadata = json.loads(sidecar.read_text())
            if previous_metadata.get("raw_sha256") != restart_digest:
                parser.error("Restart sidecar does not verify the archive SHA-256")
        previous = np.load(args.restart)
        for name, values in (("diameter",diameter),("radius",radius)):
            if not np.array_equal(previous[name], values):
                parser.error("Restart target ladder differs")
        for name,values in (("pair_penalty",pair_penalty),("umbrella_stiffness",stiffness),("umbrella_preferred",preferred),("cavity_nodes",cavity_nodes),("exchange_edges",exchange_edges)):
            if name not in previous or not np.array_equal(previous[name],values):
                parser.error(f"Restart {name} differs")
        for name,values in (("number_stiffness",number_stiffness),("number_preferred",number_preferred)):
            old = previous[name] if name in previous else np.zeros(len(previous["diameter"]))
            if not np.array_equal(old,values):
                parser.error(f"Restart {name} differs")
        if previous["final_positions"].shape != (len(diameter),args.cap,3) or float(previous["volume"]) != volume:
            parser.error("Restart volume/cap differs")
        positions = previous["final_positions"].copy()
        counts = previous["final_counts"].copy()
        centers = previous["final_centers"].copy()
        displacement = previous["displacement"].copy()
        old_activity = previous["activity"].copy()
        if (positions.dtype.kind not in "fiu" or centers.dtype.kind not in "fiu"
                or displacement.dtype.kind not in "fiu" or old_activity.dtype.kind not in "fiu"):
            parser.error("Restart coordinates, activity and proposal scales must be real numeric arrays")
        positions = positions.astype(np.float64)
        centers = centers.astype(np.float64)
        displacement = displacement.astype(np.float64)
        old_activity = old_activity.astype(np.float64)
        if old_activity.shape != activity.shape or not np.isfinite(old_activity).all():
            parser.error("Restart activity array must have finite values for every replica")
        if counts.shape != (len(diameter),) or counts.dtype.kind not in "iu" or np.any(counts<0) or np.any(counts>args.cap):
            parser.error("Restart counts must be integer and lie between zero and capacity")
        if centers.shape != (len(diameter),3) or not np.all(np.isfinite(centers)):
            parser.error("Restart centers must be finite three-dimensional torus coordinates")
        if displacement.shape != (len(diameter),) or not np.all(np.isfinite(displacement)) or np.any(displacement<0):
            parser.error("Restart displacement must be finite and nonnegative")
        for j,n in enumerate(counts):
            if not np.all(np.isfinite(positions[j,:n])) or np.any(positions[j,:n]<0) or np.any(positions[j,:n]>=length):
                parser.error("Restart active coordinates are outside the finite torus chart")
            if np.any(centers[j]<0) or np.any(centers[j]>=length):
                parser.error("Restart cavity centers are outside the torus chart")
            energy = total_pair_energy(positions[j],int(n),centers[j],diameter[j],radius[j],pair_penalty[j],length)
            if not np.isfinite(energy):
                parser.error("Restart violates a replica's pair or cavity support")
        if np.any(old_activity[cavity_nodes] != args.mu):
            parser.error("Restart physical activity differs")
        if np.any(old_activity[probe_nodes] != probe_activity):
            parser.error("Restart physical probe activities differ")
        activity = old_activity
    else:
        positions, counts, centers, ref = initialize(args.sites, length, diameter, radius, args.cap, args.seed, args.initialization)
    initial_counts = counts.copy()
    cells = int(round((args.sites/4.)**(1/3)))
    wavevectors = cells*np.array([[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],
                                  [2,0,0],[0,2,0],[0,0,2]])
    initial_bragg = np.array([np.abs(np.exp(2j*np.pi*positions[j,:n]@wavevectors.T/length).mean(axis=0)).mean()
                              if n else 0. for j,n in enumerate(counts)])
    all_samples = []
    all_configurations = []
    all_probe_configurations = []
    all_probe_sample_indices = []
    all_overlap_counts = []
    production_saved = 0
    all_accept = np.zeros((len(diameter), 8), dtype=np.int64)
    all_swap = np.zeros((len(exchange_edges), 2), dtype=np.int64)
    seed_rng(args.seed)
    for chunk in range((args.equil_sweeps+args.production_sweeps)//args.chunk_sweeps):
        overlap_counts = (np.empty((args.chunk_sweeps//args.sample_stride,len(diameter)),dtype=np.int64)
                          if args.overlap_diagnostics else None)
        samples, accept, swap, configurations = run_chunk(positions, counts, centers, diameter, radius, activity, identities, length, args.chunk_sweeps, endpoint, args.sample_stride, ref, displacement,pair_penalty,stiffness,preferred,exchange_edges,cavity_nodes,snapshot_nodes,args.palm_transfer,args.biased_birth,args.birth_width,args.birth_uniform_fraction,1 if args.birth_proposal == "vacancy" else 0,1 if args.birth_scope == "structured" else 0,overlap_counts,number_stiffness,number_preferred)
        current = (chunk+1)*args.chunk_sweeps
        if current > args.equil_sweeps:
            all_samples.append(samples)
            if overlap_counts is not None:
                all_overlap_counts.append(overlap_counts)
            # Copy only retained arrays: a view would keep every discarded
            # probe coordinate frame alive and defeat the memory bound.
            all_configurations.append(configurations[:,:2].copy())
            if len(probe_nodes):
                frame_indices = production_saved+np.arange(len(samples),dtype=np.int64)
                selected = np.flatnonzero(frame_indices%args.probe_configuration_stride == 0)
                all_probe_configurations.append(configurations[selected,2:].copy())
                all_probe_sample_indices.append(frame_indices[selected])
            production_saved += len(samples)
            all_accept += accept
            all_swap += swap
        elif current <= args.equil_sweeps:
            displacement = burn_in_displacement(displacement,accept,length,args.freeze_displacement)
            if args.calibrate_activity and current <= args.equil_sweeps//2:
                auxiliary_mean = samples[:,:,0].mean(axis=0)
                auxiliary_variance = samples[:,:,0].var(axis=0)
                target_n = volume/1.06448
                if args.tempering == "compact":
                    # Density-matched auxiliary endpoints assist transport;
                    # the preferred order is a frozen auxiliary parameter,
                    # never a measured or prescribed physical phase weight.
                    target_n = target_n+(args.sites-target_n)*np.clip((preferred-.2)/.66,0,1)*(stiffness>0)
                update = (target_n-auxiliary_mean)/np.maximum(auxiliary_variance,.1*target_n)
                mask = pair_penalty <= 12 if args.tempering in ("soft","umbrella","loop","compact") else np.ones(len(diameter),dtype=bool)
                mask[cavity_nodes] = False
                mask[probe_nodes] = False
                activity += np.where(mask,np.clip(update,-.5,.5),0.)
        endpoint_samples = samples[:,endpoint]
        print(json.dumps({"sweeps":current,"elapsed_s":round(time.time()-start,1),"endpoint_N":round(float(endpoint_samples[:,0].mean()),3),"endpoint_B":round(float(endpoint_samples[:,1].mean()),3),"crystal_fraction":round(float(np.mean(endpoint_samples[:,1]>.45)),3),"swap_min":round(float(np.min(swap[:,1]/np.maximum(swap[:,0],1))),4)}),flush=True)
        write_state_checkpoint(args.output,args,current,positions,counts,centers,
                               diameter,radius,activity,pair_penalty,stiffness,preferred,
                               endpoint,volume,displacement,cavity_nodes,exchange_edges,
                               number_stiffness,number_preferred)
    trajectory = np.concatenate(all_samples)
    combined_configurations = np.concatenate(all_configurations)
    combined_probe_configurations = (np.concatenate(all_probe_configurations) if all_probe_configurations else np.empty((0,0,args.cap,3)))
    probe_sample_indices = (np.concatenate(all_probe_sample_indices) if all_probe_sample_indices else np.zeros(0,dtype=np.int64))
    optional_diagnostics = {"pair_overlap_counts":np.concatenate(all_overlap_counts)} if all_overlap_counts else {}
    np.savez_compressed(paths[0], samples=trajectory, endpoint_configurations=combined_configurations,physical_probe_configurations=combined_probe_configurations,probe_sample_indices=probe_sample_indices,probe_nodes=probe_nodes,physical_probe_activity=probe_activity,palm_move_counts=all_accept[endpoint,6:8],diameter=diameter, radius=radius, activity=activity, pair_penalty=pair_penalty, umbrella_stiffness=stiffness,umbrella_preferred=preferred,number_stiffness=number_stiffness,number_preferred=number_preferred,cavity_nodes=cavity_nodes,exchange_edges=exchange_edges,endpoint=endpoint, volume=volume, accepts=all_accept, swaps=all_swap, final_positions=positions, final_counts=counts, final_centers=centers, displacement=displacement,**optional_diagnostics)
    snapshot = execution_source_snapshot()
    metadata = {"settings":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},"target":"unrestricted grand-canonical hard-sphere physical endpoints; capped auxiliary ensembles; normalized uniform cavity center dq/V","units":"beta=sigma=Lambda=1 at physical endpoint","source_sha256":EXECUTION_SOURCE_SHA256,"source_snapshot":str(snapshot.relative_to(ROOT)),"raw_sha256":file_sha256(paths[0]),"elapsed_s":time.time()-start,"numpy":np.__version__,"python":platform.python_version(),"columns":["N","fcc_bragg_mean","walker_id","next_radius_support","empty_reference_sites","excess_site_occupancy","minimum_center_distance","at_N_cap","sum_delete_one_solid_label_change","nearest_FCC_registry_rms_over_spacing"],"warnings":["FCC coordinates provide occupancy diagnostics; optional reciprocal-mode structural umbrella is zero on all physical cavity nodes; no physical phase weights or occupancy constraints imposed","A short pilot does not certify equilibrium phase weights or macroscopic coexistence"]}
    metadata["diagnostic_nodes"] = snapshot_nodes.tolist()
    metadata["auxiliary_diagnostic_sentinel"] = -1
    metadata["deletion_label_threshold"] = .45
    metadata["initial_counts"] = initial_counts.tolist()
    metadata["initial_bragg"] = initial_bragg.tolist()
    metadata["pair_overlap_diagnostics"] = bool(args.overlap_diagnostics)
    metadata["imported_probe_displacement_nodes"] = imported_probe_displacement_nodes.tolist()
    proposal_scope = ("same-diameter unbiased hard nodes" if args.birth_scope == "physical"
                      else "same-diameter nodes with pair penalty >=8 or nonzero order umbrella; fixed target-index mask")
    metadata["particle_transfer_proposal"] = (f"normalized {args.birth_proposal} cube mixture on {proposal_scope}; full pair/umbrella energy change; reverse density evaluated after tag removal; uniform fallback on degenerate/no-center cases" if args.biased_birth else "uniform birth positions")
    if args.birth_proposal == "vacancy":
        metadata["registry_conventions"] = "two(200)-phase parity fits; minimum total nearest-site squared distance; first parity and lowest site index break ties; zero phase uses atan2(0,0)=0; no global translation-equivariance claim"
    if args.restart:
        metadata["restart_sha256"] = restart_digest
    if args.auxiliary_from:
        metadata["auxiliary_schedule_sha256"] = auxiliary_digest
    save_json(paths[1],metadata)
    # Successful final output supersedes only this campaign's own state files.
    for checkpoint in checkpoint_paths:
        checkpoint.unlink(missing_ok=True)
    print(str(paths[0]),flush=True)


if __name__ == "__main__":
    main()
