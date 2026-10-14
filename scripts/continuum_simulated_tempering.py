#!/usr/bin/env python3
"""Opt-in one-host regenerative tempering; never a replica phase-weight fit.

The joint target is exp(b_j+x_j*N-Phi_j)/N! with dq/V. Arbitrary finite
FROZEN b preserve each conditional node law. Adaptive visits are discarded.
An exact ideal-node heatbath removes incoming phase/number memory; physical
estimates must weight every clock-time dwell, not average cycle fractions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("NUMBA_NUM_THREADS","1")
import numpy as np
from numba import njit

try:
    from . import continuum_replica_exchange as kernel
    from . import continuum_analyze as replica_validation
    from . import continuum_tempering_graph as native_graph
    from . import continuum_normalizer_projection as normalizer_projection
except ImportError:
    import continuum_replica_exchange as kernel
    import continuum_analyze as replica_validation
    import continuum_tempering_graph as native_graph
    import continuum_normalizer_projection as normalizer_projection

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SOURCE_BYTES = Path(__file__).read_bytes()
EXECUTION_SOURCE_SHA256 = hashlib.sha256(EXECUTION_SOURCE_BYTES).hexdigest()
VALIDATION_SOURCE_BYTES = Path(replica_validation.__file__).read_bytes()
NATIVE_GRAPH_SOURCE_BYTES = Path(native_graph.__file__).read_bytes()
SCHEMA = "cavcoex.single_host_regenerative_tempering.v1"
COLUMNS = ["clock_step","label","N","fcc_bragg_mean","pair_overlap_count",
           "empty_reference_sites","excess_site_occupancy","minimum_center_distance",
           "sum_delete_one_solid_label_change","nearest_FCC_registry_rms_over_spacing",
           "regeneration_cycle_id","ideal_heatbath_event"]


def capped_poisson_cdf(log_mean,cap):
    """Exactly normalize the truncated law, never clamp a Poisson draw."""
    if not np.isfinite(log_mean) or not isinstance(cap,(int,np.integer)) or cap < 0:
        raise ValueError("Finite log mean and nonnegative integer cap required")
    logp = np.array([n*log_mean-math.lgamma(n+1) for n in range(cap+1)])
    weights = np.exp(logp-logp.max())
    cdf = np.cumsum(weights/weights.sum())
    cdf[-1] = 1.
    return cdf


def adjacency(edges,nodes):
    edges = np.asarray(edges)
    if (edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu'
            or np.any(edges<0) or np.any(edges>=nodes)
            or np.any(edges[:,0]==edges[:,1])):
        raise ValueError("Edges must join distinct valid integer nodes")
    undirected = [tuple(sorted(map(int,pair))) for pair in edges]
    if len(set(undirected)) != len(undirected):
        raise ValueError("Duplicate exchange edges are not supported")
    neighbors = [[] for _ in range(nodes)]
    for edge,(left,right) in enumerate(edges):
        neighbors[left].append((right,edge))
        neighbors[right].append((left,edge))
    seen,frontier = {0},[0]
    while frontier:
        for other,_ in neighbors[frontier.pop()]:
            if other not in seen:
                seen.add(other)
                frontier.append(other)
    if len(seen) != nodes:
        raise ValueError("The auxiliary graph must be connected")
    degrees = np.array([len(items) for items in neighbors],dtype=np.int64)
    table = np.full((nodes,max(degrees)), -1,dtype=np.int64)
    edge_table = table.copy()
    for node,items in enumerate(neighbors):
        for index,(other,edge) in enumerate(items):
            table[node,index],edge_table[node,index] = other,edge
    return table,edge_table,degrees


def validate_targets(target,cap):
    names = ("diameter","radius","activity","pair_penalty","umbrella_stiffness",
             "umbrella_preferred","number_stiffness","number_preferred","displacement")
    nodes = len(target['activity'])
    if any(np.shape(target[key]) != (nodes,) for key in names):
        raise ValueError("Target and proposal arrays must match node count")
    for key in names:
        values = target[key]
        finite = np.isfinite(values) | ((key=='pair_penalty') & np.isposinf(values))
        if not finite.all():
            raise ValueError(f"Nonfinite target array: {key}")
    if (not np.all(target['diameter']==1.)
            or any(np.any(target[key]<0) for key in ('radius','pair_penalty','umbrella_stiffness',
                                                     'number_stiffness'))
            or np.any(target['displacement']<=0)
            or np.any(target['number_preferred']<0) or np.any(target['number_preferred']>cap)):
        raise ValueError("This driver uses sigma=1, positive displacements and nonnegative costs")
    volume = float(target['volume'])
    if not np.isfinite(volume) or volume<=0 or volume**(1/3)<target['diameter'][0]:
        raise ValueError("Positive periodic box no shorter than the diameter required")
    if cap < math.ceil(6*volume/(np.pi*target['diameter'][0]**3)):
        raise ValueError("Physical particle cap must exceed the geometric packing bound")
    base,cavity,ideal = map(int,(target['base_node'],target['cavity_node'],target['ideal_node']))
    if len({base,cavity,ideal}) != 3 or min(base,cavity,ideal)<0 or max(base,cavity,ideal)>=nodes:
        raise ValueError("Base, cavity and ideal nodes must be distinct valid indices")
    physical = np.asarray(target['physical_nodes'])
    if (physical.ndim != 1 or physical.dtype.kind not in 'iu' or np.any(physical<0)
            or np.any(physical>=nodes) or len(np.unique(physical))!=len(physical)
            or base not in physical or cavity not in physical):
        raise ValueError("Physical node list must include distinct valid base/cavity nodes")
    if (target['radius'][ideal] != 0 or target['pair_penalty'][ideal] != 0
            or target['umbrella_stiffness'][ideal] != 0 or target['number_stiffness'][ideal] != 0):
        raise ValueError("Regeneration requires an exactly unbiased ideal node")
    for physical in target['physical_nodes']:
        if (target['pair_penalty'][physical] != np.inf or target['umbrella_stiffness'][physical] != 0
                or target['number_stiffness'][physical] != 0):
            raise ValueError("Every physical node must be hard and umbrella-free")
    if (target['radius'][base] != 0 or target['radius'][cavity] != target['diameter'][base]
            or target['activity'][base] != target['activity'][cavity]):
        raise ValueError("Tagged edge requires matched activity at R=0 and R=sigma")
    undirected = {tuple(sorted(map(int,pair))) for pair in target['exchange_edges']}
    if tuple(sorted((base,cavity))) not in undirected:
        raise ValueError("The tagged base/cavity edge must exist")
    if 'radius_path_nodes' in target:
        path = np.asarray(target['radius_path_nodes'])
        if (path.ndim!=1 or path.dtype.kind not in 'iu' or len(path)<3
                or np.any(path<0) or np.any(path>=nodes) or len(np.unique(path))!=len(path)
                or path[0]!=base or path[-1]!=cavity or not set(path)<=set(target['physical_nodes'])
                or np.any(np.diff(target['radius'][path])<=0)
                or np.any(target['activity'][path]!=target['activity'][base])
                or any(tuple(sorted((int(left),int(right)))) not in undirected
                       for left,right in zip(path[:-1],path[1:]))):
            raise ValueError('Physical radius path must increase from base to cavity at matched activity')
    return adjacency(target['exchange_edges'],nodes)


@njit(cache=True)
def overlap_count(positions,n,diameter,length):
    count = 0
    for i in range(n):
        for j in range(i):
            if kernel.periodic_distance_squared(positions[i],positions[j],length)<diameter**2:
                count += 1
    return count


@njit(cache=True)
def reduced_log_density(node,n,b,p,min_distance,x,radius,penalty,kappa,preferred,nu,nstar):
    if (n>0 and min_distance<radius[node]) or (penalty[node]==np.inf and p>0):
        return -np.inf
    pair = 0. if penalty[node]==np.inf else penalty[node]*p
    return x[node]*n-pair-kappa[node]*(b-preferred[node])**2-nu[node]*(n-nstar[node])**2


@njit(cache=True)
def conditional_label_probabilities(n,b,p,min_distance,activity,radius,penalty,kappa,
                                    preferred,nu,nstar,log_weights):
    """Exact finite label conditional; no graph-degree proposal factor."""
    scores = np.empty(len(activity))
    maximum = -np.inf
    for node in range(len(activity)):
        scores[node] = log_weights[node] + reduced_log_density(
            node,n,b,p,min_distance,activity,radius,penalty,kappa,preferred,nu,nstar)
        if np.isnan(scores[node]) or scores[node]==np.inf:
            raise ValueError('Conditional label scores must be finite or unsupported')
        maximum = max(maximum,scores[node])
    if not np.isfinite(maximum):
        raise ValueError('No supported finite label score for conditional refresh')
    total = 0.
    for node in range(len(activity)):
        scores[node] = math.exp(scores[node]-maximum)
        total += scores[node]
    for node in range(len(activity)):
        scores[node] /= total
    return scores


@njit(cache=True)
def draw_conditional_label(probabilities):
    draw,cumulative = np.random.random(),0.
    last_supported = -1
    for node in range(len(probabilities)):
        if probabilities[node]>0:
            last_supported = node
            cumulative += probabilities[node]
            if draw<cumulative:
                return node
    # Floating summation may end a few ulps below one; never choose an invalid
    # zero-support trailing node in that rounding case.
    if last_supported<0:
        raise ValueError('No supported label probability')
    return last_supported


@njit(cache=True)
def ideal_heatbath(positions,counts,centers,cdf,length):
    """Full memory-erasing conditional refresh of N, host and center."""
    u = np.random.random()
    n = 0
    while u>cdf[n]:
        n += 1
    counts[0] = n
    for particle in range(n):
        for axis in range(3):
            positions[0,particle,axis] = np.random.random()*length
    for axis in range(3):
        centers[0,axis] = np.random.random()*length


def validate_label_quota(quota,nodes):
    """Positive normalized label probabilities, never physical phase odds."""
    quota = np.asarray(quota,dtype=float)
    if (quota.shape!=(nodes,) or not np.isfinite(quota).all() or np.any(quota<=0)
            or not np.isclose(quota.sum(),1.,rtol=0.,atol=1e-12)):
        raise ValueError('Label quota must be a finite strictly positive normalized node vector')
    return quota.copy()


def resolve_label_quota(nodes,base,cavity,ideal,mode='uniform'):
    """Construct a declared label-efficiency allocation without phase input."""
    if (not isinstance(nodes,(int,np.integer)) or isinstance(nodes,(bool,np.bool_))
            or nodes<1):
        raise ValueError('Positive integer node count required for label quotas')
    if mode=='uniform':
        return np.full(nodes,1/nodes)
    if mode!='endpoint_priority':
        raise ValueError('Recognized label quota mode required')
    labels = (base,cavity,ideal)
    if (nodes<4 or any(not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_))
                      or not 0<=i<nodes for i in labels) or len(set(labels))!=3):
        raise ValueError('Endpoint priority requires three distinct valid labels and another node')
    quota = np.full(nodes,.38/(nodes-3))
    quota[base],quota[cavity],quota[ideal] = .30,.30,.02
    return validate_label_quota(quota,nodes)


@njit(cache=True)
def run_tempering_chunk(positions,counts,centers,node,cycle,clock,steps,reference,length,
        diameter,radius,activity,penalty,kappa,preferred,nu,nstar,displacement,
        physical,base,cavity,ideal,neighbors,edge_table,degrees,log_weights,cdf,
        calibration_gain=0.,calibration_tau=10000.,coordinate_stride=100,
        biased_birth=True,birth_width=.06,birth_uniform_fraction=.2,coordinate_centers=None,
        gibbs_label=False,constant_calibration_gain=False,neighbor_source_labels=None,
        weight_update_labels=None,gibbs_counters=None,vacancy_relocation_probability=0.,
        vacancy_relocation_counts=None,calibration_label_quota=None):
    """One local exact kernel then one lazy label hop per fixed clock step."""
    trace = np.empty((steps,12))
    coordinate_indices = np.arange(coordinate_stride-(clock%coordinate_stride)-1,steps,coordinate_stride)
    if coordinate_centers is not None and coordinate_centers.shape != (len(coordinate_indices),3):
        raise ValueError("Coordinate-center output must match snapshot count and spatial dimension")
    if neighbor_source_labels is not None and neighbor_source_labels.shape!=(steps,):
        raise ValueError('Pre-neighbor label output must match the chunk clock steps')
    if weight_update_labels is not None and weight_update_labels.shape!=(steps,):
        raise ValueError('Weight-update label output must match the chunk clock steps')
    if gibbs_counters is not None and gibbs_counters.shape!=(2,):
        raise ValueError('Gibbs counters require attempts and label changes')
    if (not np.isfinite(vacancy_relocation_probability)
            or not 0 <= vacancy_relocation_probability <= 1):
        raise ValueError('Vacancy-relocation probability must be finite and in [0,1]')
    if vacancy_relocation_counts is not None and vacancy_relocation_counts.shape!=(2,):
        raise ValueError('Vacancy-relocation output needs attempted/accepted counters')
    nonuniform_quota = False
    if calibration_label_quota is not None:
        if calibration_label_quota.shape!=(len(activity),):
            raise ValueError('Label quota must match every node')
        quota_total = 0.
        for q in calibration_label_quota:
            if not np.isfinite(q) or q<=0:
                raise ValueError('Label quota must be finite and strictly positive')
            quota_total += q
            nonuniform_quota = nonuniform_quota or q!=calibration_label_quota[0]
        if abs(quota_total-1.)>1e-12:
            raise ValueError('Label quota must be normalized')
    configurations = np.full((len(coordinate_indices),positions.shape[1],3),np.nan)
    coordinate_labels = np.empty(len(coordinate_indices),dtype=np.int64)
    local_accepts = np.zeros(8,dtype=np.int64)
    edge_accepts = np.zeros((np.max(edge_table)+1,2),dtype=np.int64)
    identities = np.zeros(1,dtype=np.int64)
    one_d,one_r,one_x,one_p,one_k,one_b,one_nu,one_star,one_step = [np.zeros(1) for _ in range(9)]
    empty_edges = np.empty((0,2),dtype=np.int64)
    empty_nodes = np.empty(0,dtype=np.int64)
    physical_snapshot = np.zeros(1,dtype=np.int64)
    relocation_chunk = np.zeros((1,2),dtype=np.int64)
    coordinate_index = 0
    for step in range(steps):
        regeneration = node == ideal
        if regeneration:
            ideal_heatbath(positions,counts,centers,cdf,length)
            cycle += 1
        one_d[0],one_r[0],one_x[0],one_p[0] = diameter[node],radius[node],activity[node],penalty[node]
        one_k[0],one_b[0],one_nu[0],one_star[0] = kappa[node],preferred[node],nu[node],nstar[node]
        one_step[0] = displacement[node]
        stat,accept,_,_ = kernel.run_chunk(positions,counts,centers,one_d,one_r,one_x,
            identities,length,1,0,1,reference,one_step,one_p,one_k,one_b,
            empty_edges,empty_nodes,physical_snapshot if physical[node] else empty_nodes,
            False,biased_birth,birth_width,birth_uniform_fraction,1,1,None,one_nu,one_star,
            vacancy_relocation_probability,relocation_chunk)
        local_accepts += accept[0]
        n,b = counts[0],stat[0,0,1]
        p = overlap_count(positions[0],n,diameter[node],length)
        trace[step,0],trace[step,1],trace[step,2],trace[step,3] = clock+step+1,node,n,b
        trace[step,4],trace[step,5],trace[step,6] = p,stat[0,0,4],stat[0,0,5]
        trace[step,7],trace[step,8],trace[step,9] = stat[0,0,6],stat[0,0,8],stat[0,0,9]
        trace[step,10],trace[step,11] = cycle,regeneration
        if coordinate_index<len(coordinate_indices) and step==coordinate_indices[coordinate_index]:
            configurations[coordinate_index,:n] = positions[0,:n]
            coordinate_labels[coordinate_index] = node
            if coordinate_centers is not None:
                coordinate_centers[coordinate_index] = centers[0]
            coordinate_index += 1
        # Label-only Gibbs precedes the marked hop, so the recorded N/order/
        # center features are still current. It changes neither coordinates
        # nor the physical conditional law, and carries no degree factor.
        if gibbs_label:
            probabilities = conditional_label_probabilities(n,b,p,stat[0,0,6],activity,
                radius,penalty,kappa,preferred,nu,nstar,log_weights)
            other = draw_conditional_label(probabilities)
            if gibbs_counters is not None:
                gibbs_counters[0] += 1
                gibbs_counters[1] += other!=node
            node = other
        if neighbor_source_labels is not None:
            neighbor_source_labels[step] = node
        # Common laziness cancels. Uniform-neighbor degree factors do not.
        if np.random.random()<.5:
            choice = np.random.randint(degrees[node])
            other,edge = neighbors[node,choice],edge_table[node,choice]
            edge_accepts[edge,0] += 1
            logratio = log_weights[other]-log_weights[node]+math.log(degrees[node]/degrees[other])
            tagged = (node==base and other==cavity) or (node==cavity and other==base)
            valid = True
            if tagged:
                if node==base:
                    if n==0:
                        valid = False
                    else:
                        logratio += math.log(n)-activity[node]-math.log(length**3)
                else:
                    if n>=positions.shape[1]:
                        valid = False
                    else:
                        logratio += activity[node]+math.log(length**3)-math.log(n+1)
            else:
                old = reduced_log_density(node,n,b,p,stat[0,0,6],activity,radius,penalty,kappa,preferred,nu,nstar)
                new = reduced_log_density(other,n,b,p,stat[0,0,6],activity,radius,penalty,kappa,preferred,nu,nstar)
                logratio += new-old
                valid = np.isfinite(new)
            if valid and math.log(np.random.random())<min(0.,logratio):
                if tagged and node==base:
                    tag = np.random.randint(n)
                    centers[0] = positions[0,tag]
                    positions[0,tag] = positions[0,n-1]
                    counts[0] = n-1
                elif tagged:
                    positions[0,n] = centers[0]
                    counts[0] = n+1
                    for axis in range(3):
                        centers[0,axis] = np.random.random()*length
                node = other
                edge_accepts[edge,1] += 1
        if calibration_gain>0:
            gain = calibration_gain if constant_calibration_gain else calibration_gain/(1+(clock+step)/calibration_tau)
            if nonuniform_quota:
                for j in range(len(log_weights)):
                    log_weights[j] += gain*calibration_label_quota[j]
            # A uniform quota adds only a common gauge constant. Skipping it
            # preserves the original trajectory and floating-point arithmetic.
            log_weights[node] -= gain
            # Gauge fix and finite bound are calibration-only efficiency choices.
            if (clock+step+1)%1000==0:
                log_weights -= log_weights[ideal]
                for j in range(len(log_weights)):
                    log_weights[j] = min(1e5,max(-1e5,log_weights[j]))
        if weight_update_labels is not None:
            weight_update_labels[step] = node
    if vacancy_relocation_counts is not None:
        vacancy_relocation_counts[:] += relocation_chunk[0]
    return trace,local_accepts,edge_accepts,configurations,coordinate_indices+clock+1,coordinate_labels,node,cycle


class StagedFlatnessCalibration:
    """Finite discarded Wang--Landau stages, not an equilibration certificate.

    Flatness counts exactly the post-composition label whose weight changed.
    Number/order comparisons instead use the paired pre-Gibbs trace label and
    features. Epoch halves consist of actual consecutive completed chunks.
    """
    def __init__(self,nodes,dense_nodes,gain=.3,minimum_gain=.01875,flatness=.5,
                 minimum_visits=100,half_minimum_visits=50,number_tolerance=.25,
                 bragg_tolerance=.10,label_quota=None,normalizer_projection_mode='none'):
        if (not isinstance(nodes,(int,np.integer)) or nodes<1 or not np.isfinite(gain)
                or not np.isfinite(minimum_gain) or not 0<minimum_gain<=gain
                or not np.isfinite(flatness) or not 0<flatness<=1
                or not isinstance(minimum_visits,(int,np.integer)) or minimum_visits<1
                or not isinstance(half_minimum_visits,(int,np.integer)) or half_minimum_visits<1
                or not np.isfinite(number_tolerance) or number_tolerance<0
                or not np.isfinite(bragg_tolerance) or bragg_tolerance<0):
            raise ValueError('Finite positive gains/visits and valid flatness/stability criteria required')
        dense_nodes = np.asarray(dense_nodes)
        if (dense_nodes.ndim!=1 or dense_nodes.dtype.kind not in 'iu'
                or not len(dense_nodes) or np.any(dense_nodes<0) or np.any(dense_nodes>=nodes)
                or len(np.unique(dense_nodes))!=len(dense_nodes)):
            raise ValueError('Dense diagnostic nodes must be distinct valid integer labels')
        self.nodes,self.dense_nodes = int(nodes),dense_nodes.copy()
        if normalizer_projection_mode not in ('none','nested_physical_radius','certified_normalizers'):
            raise ValueError('Unknown discarded normalizer projection mode')
        self.normalizer_projection_mode = normalizer_projection_mode
        self.label_quota = validate_label_quota(np.full(nodes,1/nodes)
            if label_quota is None else label_quota,nodes)
        self.nonuniform_quota = bool(np.any(self.label_quota!=self.label_quota[0]))
        self.flatness_measure = ('minimum(counts/quota)/mean(counts/quota)' if self.nonuniform_quota
                                 else 'minimum(counts)/mean(counts)')
        self.gain,self.initial_gain,self.minimum_gain = float(gain),float(gain),float(minimum_gain)
        self.criteria = {'flatness_minimum_to_mean':float(flatness),
            'minimum_weight_update_visits_per_label':int(minimum_visits),
            'minimum_observations_per_dense_label_per_half':int(half_minimum_visits),
            'maximum_half_mean_N_difference':float(number_tolerance),
            'maximum_half_mean_Bragg_difference':float(bragg_tolerance),
            'dense_diagnostic_nodes':dense_nodes.tolist()}
        self.stage,self.completed,self.history = 0,False,[]
        self.chunks = []

    def observe(self,trace,weight_update_labels,clock,weights,projection_record=None):
        if self.completed:
            raise ValueError('No adaptation observations may enter completed calibration')
        if ((self.normalizer_projection_mode=='none') != (projection_record is None)):
            raise ValueError('Projection history must match its declared staged calibration mode')
        trace,updated = np.asarray(trace),np.asarray(weight_update_labels)
        if (trace.ndim!=2 or trace.shape[1]!=12 or not len(trace) or updated.shape!=(len(trace),)
                or updated.dtype.kind not in 'iu' or np.any(updated<0) or np.any(updated>=self.nodes)
                or np.shape(weights)!=(self.nodes,) or not np.isfinite(weights).all()):
            raise ValueError('Paired finite trace/update-label statistics and weights required')
        observed = trace[:,1].astype(int)
        if (np.any(trace[:,1]!=observed) or np.any(observed<0) or np.any(observed>=self.nodes)
                or not np.isfinite(trace[:,2:4]).all()):
            raise ValueError('Observation labels must pair with finite N/order features')
        self.chunks.append({'steps':len(trace),
            'updated':np.bincount(updated,minlength=self.nodes),
            'count':np.bincount(observed,minlength=self.nodes),
            'N':np.bincount(observed,weights=trace[:,2],minlength=self.nodes),
            'B':np.bincount(observed,weights=trace[:,3],minlength=self.nodes),
            'heatbaths':int(trace[:,11].sum())})
        counts = sum((chunk['updated'] for chunk in self.chunks),np.zeros(self.nodes,dtype=np.int64))
        scaled_counts = counts/self.label_quota if self.nonuniform_quota else counts
        ratio = float(scaled_counts.min()/scaled_counts.mean())
        minimum_met = bool(counts.min()>=self.criteria['minimum_weight_update_visits_per_label'])
        flatness_met = bool(ratio>=self.criteria['flatness_minimum_to_mean'] and minimum_met)
        midpoint = len(self.chunks)//2
        comparisons = []
        stability_met = midpoint>0
        for node in self.dense_nodes:
            halves = []
            for chunks in (self.chunks[:midpoint],self.chunks[midpoint:]):
                count = int(sum(chunk['count'][node] for chunk in chunks))
                halves.append({'observations':count,
                    'mean_N':float(sum(chunk['N'][node] for chunk in chunks)/count) if count else None,
                    'mean_Bragg':float(sum(chunk['B'][node] for chunk in chunks)/count) if count else None})
            sufficient = all(half['observations']>=self.criteria['minimum_observations_per_dense_label_per_half']
                             for half in halves)
            delta_n = abs(halves[0]['mean_N']-halves[1]['mean_N']) if sufficient else None
            delta_b = abs(halves[0]['mean_Bragg']-halves[1]['mean_Bragg']) if sufficient else None
            passed = bool(sufficient and delta_n<=self.criteria['maximum_half_mean_N_difference']
                          and delta_b<=self.criteria['maximum_half_mean_Bragg_difference'])
            stability_met &= passed
            comparisons.append({'node':int(node),'first_half':halves[0],'second_half':halves[1],
                'sufficient_half_observations':sufficient,'mean_N_difference':delta_n,
                'mean_Bragg_difference':delta_b,'passed':passed})
        passed = bool(flatness_met and stability_met)
        row = {'stage':self.stage,'gain':self.gain,'end_clock':int(clock),
            'epoch_steps':int(sum(chunk['steps'] for chunk in self.chunks)),
            'first_half_steps':int(sum(chunk['steps'] for chunk in self.chunks[:midpoint])),
            'second_half_steps':int(sum(chunk['steps'] for chunk in self.chunks[midpoint:])),
            'weight_update_visits':counts.tolist(),'flatness_minimum_to_mean':ratio,
            'minimum_visits_met':minimum_met,'flatness_passed':flatness_met,
            'dense_half_stability_passed':bool(stability_met),'stage_passed':passed,
            'ideal_heatbath_events':sum(chunk['heatbaths'] for chunk in self.chunks),
            'dense_half_profiles':comparisons,'log_weights':np.asarray(weights).tolist()}
        if self.nonuniform_quota:
            row['quota_scaled_weight_update_visits'] = scaled_counts.tolist()
        if projection_record is not None:
            row['normalizer_projection'] = projection_record
        self.history.append(row)
        if passed:
            if self.gain<=self.minimum_gain:
                self.completed = True
            else:
                self.gain = max(self.minimum_gain,self.gain/2)
                self.stage += 1
                self.chunks = []
        return row

    def outcome(self,actual_steps,budget):
        result = {'method':'staged_wang_landau','completed':self.completed,
            'budget_exhausted':bool(not self.completed and actual_steps>=budget),
            'actual_clock_steps':int(actual_steps),'requested_budget':int(budget),
            'initial_gain':self.initial_gain,'final_gain':self.gain,'minimum_gain':self.minimum_gain,
            'label_quota':self.label_quota.tolist(),'flatness_measure':self.flatness_measure,
            'criteria':self.criteria,'stage_history':self.history,
            'scope':'Discarded label-efficiency diagnostics only; neither flatness nor stability certifies physical phase equilibration'}
        if self.normalizer_projection_mode!='none':
            result['normalizer_projection_mode'] = self.normalizer_projection_mode
        return result


def load_graph(path):
    """Validated frozen replica targets are efficiency inputs, not equilibrium evidence."""
    metadata = json.loads(Path(path).with_suffix('.json').read_text())
    if metadata.get('schema') == native_graph.SCHEMA:
        return native_graph.load_native_graph(path)
    data,metadata = replica_validation.load_validated_run(path,metadata['settings']['sample_stride'])
    target = {key:np.asarray(data[key]).copy() for key in ('diameter','radius','activity','pair_penalty',
        'umbrella_stiffness','umbrella_preferred','number_stiffness','number_preferred','displacement','exchange_edges')}
    ideal_nodes = np.flatnonzero((target['pair_penalty']==0)&(target['radius']==0)
        &(target['umbrella_stiffness']==0)&(target['number_stiffness']==0))
    if len(ideal_nodes)==0:
        raise ValueError("Imported graph has no ideal regeneration node")
    target.update(volume=float(data['volume']),base_node=int(data['endpoint']),
        cavity_node=int(data['cavity_nodes'][-1]),ideal_node=int(ideal_nodes[0]),
        physical_nodes=np.unique(np.r_[data['cavity_nodes'],data['probe_nodes']]).astype(int))
    return target,data,metadata


def load_frozen_weights(path,target,sites,cap):
    """Import only integrity-verified weights for the identical frozen graph."""
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix('.json').read_text())
    if (metadata.get('schema') != SCHEMA or metadata.get('weights_frozen_before_production') is not True
            or metadata.get('raw_sha256') != kernel.file_sha256(path)):
        raise ValueError("Weight archive must have verified single-host frozen execution metadata")
    snapshots = metadata.get('source_snapshots',{})
    if not {'driver','kernel','graph_validation'} <= set(snapshots):
        raise ValueError("Weight archive must identify every local executed dependency")
    for entry in snapshots.values():
        if not isinstance(entry,list) or len(entry)!=2:
            raise ValueError("Invalid executed-source provenance entry")
        source_path = (ROOT/entry[0]).resolve()
        if not source_path.is_relative_to(ROOT) or kernel.file_sha256(source_path) != entry[1]:
            raise ValueError("Weight-archive executed source digest is not verified")
    inputs = metadata.get('input_sha256',{})
    if not isinstance(inputs,dict) or not inputs:
        raise ValueError("Weight archive requires imported-input provenance")
    for name,digest in inputs.items():
        if kernel.file_sha256(ROOT/Path(name)) != digest:
            raise ValueError("Weight-archive imported-input digest is not verified")
    with np.load(path,allow_pickle=False) as archive:
        if str(archive['schema']) != SCHEMA:
            raise ValueError("Weight raw schema differs from execution metadata")
        for name in ('diameter','radius','activity','pair_penalty','umbrella_stiffness',
                     'umbrella_preferred','number_stiffness','number_preferred','exchange_edges',
                     'displacement','volume','base_node','cavity_node','ideal_node','physical_nodes'):
            if not np.array_equal(archive[name],target[name]):
                raise ValueError(f"Imported frozen weight graph differs: {name}")
        if not np.array_equal(archive['sites'],sites) or not np.array_equal(archive['cap'],cap):
            raise ValueError("Imported weight site count or particle cap differs")
        if ('radius_path_nodes' in archive)!=('radius_path_nodes' in target):
            raise ValueError('Frozen weights require identical physical radius-path declarations')
        if 'radius_path_nodes' in target and not np.array_equal(archive['radius_path_nodes'],target['radius_path_nodes']):
            raise ValueError('Frozen weights require identical physical radius-path nodes')
        weights = archive['log_weights'].copy()
    if weights.shape != target['activity'].shape or not np.isfinite(weights).all():
        raise ValueError("Imported node log weights must be finite and match every node")
    return weights


def initial_weight_guess(target,samples,overlaps):
    """Finite local bridge-cost guess solely to accelerate discarded calibration.

    Sparse support or a trapped incoming history does not make this an estimate
    of physical phase odds. Graph cycles are deliberately not fit as evidence.
    """
    nodes = len(target['activity'])
    weights = np.full(nodes,np.nan)
    ideal = target['ideal_node']
    weights[ideal] = 0.
    pending = [ideal]
    neighbors,_,degrees = adjacency(target['exchange_edges'],nodes)
    while pending:
        left = pending.pop(0)
        for right in neighbors[left,:degrees[left]]:
            if np.isfinite(weights[right]):
                continue
            if {left,int(right)}=={target['base_node'],target['cavity_node']}:
                mean = max(1.,float(np.mean(samples[:,target['base_node'],0])))
                difference = math.log(mean)-target['activity'][left]-math.log(target['volume'])
                if left==target['cavity_node']:
                    difference = -difference
            else:
                estimates = []
                for source in (left,right):
                    values = []
                    for row,p in zip(samples[:,source],overlaps[:,source]):
                        a = reduced_log_density(left,int(row[0]),row[1],int(p),row[6],
                            target['activity'],target['radius'],target['pair_penalty'],
                            target['umbrella_stiffness'],target['umbrella_preferred'],
                            target['number_stiffness'],target['number_preferred'])
                        b = reduced_log_density(right,int(row[0]),row[1],int(p),row[6],
                            target['activity'],target['radius'],target['pair_penalty'],
                            target['umbrella_stiffness'],target['umbrella_preferred'],
                            target['number_stiffness'],target['number_preferred'])
                        if np.isfinite(a) and np.isfinite(b):
                            values.append(b-a)
                    if values:
                        estimates.append(float(np.mean(values)))
                difference = float(np.mean(estimates)) if estimates else 0.
            weights[right] = np.clip(weights[left]-difference,-1e4,1e4)
            pending.append(int(right))
    return weights


def capture_source(path,content):
    digest = hashlib.sha256(content).hexdigest()
    destination = ROOT/'data_raw'/'source_snapshots'/f'{Path(path).stem}_{digest}.py'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        destination.write_bytes(content)
    elif kernel.file_sha256(destination)!=digest:
        raise ValueError("Content-keyed executed-source snapshot is corrupt")
    return str(destination.relative_to(ROOT)),digest


def projected_memory_budget(production_steps,calibration_steps,chunk_steps,
                            coordinate_stride,cap,nodes,dense_nodes,gibbs_label,nonuniform_label_quota=False,
                            normalizer_projection_nodes=0,certified_projection_linear_rows=0):
    """Conservative implementation budget, not an OS-enforced memory limit.

    The runtime allowance exceeds the measured 1.55-GiB high-water RSS of
    the refined 2.4M-clock campaign. Array copies, Python history objects,
    JSON/checkpoint temporaries and compression are separately budgeted;
    discarded coordinate/trace chunks are never retained.
    """
    counts = (production_steps,calibration_steps,chunk_steps,coordinate_stride,
              cap,nodes,dense_nodes,normalizer_projection_nodes,certified_projection_linear_rows)
    if (any(not isinstance(v,(int,np.integer)) for v in counts)
            or production_steps<=0 or calibration_steps<0 or chunk_steps<=0
            or coordinate_stride<=0 or cap<1 or nodes<1 or not 0<=dense_nodes<=nodes
            or not 0<=normalizer_projection_nodes<=nodes
            or not 0<=certified_projection_linear_rows<=1024
            or (certified_projection_linear_rows and not 2<=normalizer_projection_nodes<=257)):
        raise ValueError('Valid integer memory-budget dimensions required')
    raw = (production_steps*(len(COLUMNS)+int(gibbs_label))*8
           +(production_steps//coordinate_stride+1)*(cap*3+3)*8)
    history = math.ceil(calibration_steps/chunk_steps)*(2048+(80 if nonuniform_label_quota else 64)*nodes+1536*dense_nodes)
    if normalizer_projection_nodes:
        history += math.ceil(calibration_steps/chunk_steps)*(2048+80*nodes+96*normalizer_projection_nodes)
    if certified_projection_linear_rows:
        history += math.ceil(calibration_steps/chunk_steps)*(4096+160*normalizer_projection_nodes+256*certified_projection_linear_rows)
    chunk = chunk_steps*(len(COLUMNS)+2)*8+(chunk_steps//coordinate_stride+1)*(cap*3+3)*8
    runtime = 1664*1024**2
    peak = runtime+4*raw+6*history+4*chunk
    solver_workspace = 0
    if certified_projection_linear_rows:
        solver_workspace = 8*(16*certified_projection_linear_rows*normalizer_projection_nodes+32*normalizer_projection_nodes**2)
        peak += solver_workspace
    result = {'raw_array_bytes':int(raw),'calibration_history_envelope_bytes':int(history),
            'chunk_workspace_bytes':int(chunk),'runtime_allowance_bytes':runtime,
            'projected_peak_bytes':int(peak),'raw_limit_bytes':200*1024**2,
            'peak_limit_bytes':2*1024**3,
            'scope':'Conservative planning guard with measured-runtime allowance; not a measured allocation profile or OS hard limit'}
    if certified_projection_linear_rows:
        result['certified_projection_solver_workspace_bytes'] = int(solver_workspace)
        result['certified_projection_selected_nodes'] = int(normalizer_projection_nodes)
        result['certified_projection_linear_rows'] = int(certified_projection_linear_rows)
    return result


def checkpoint_outcome_bytes(outcome):
    """UTF-8 JSON avoids NumPy's four-byte Unicode checkpoint expansion."""
    return np.bytes_(json.dumps(outcome,allow_nan=False).encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-from',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New prefix, .npz/.json appended')
    parser.add_argument('--seed',type=int,default=2026092930)
    parser.add_argument('--initialization',choices=('fluid','crystal','ideal'),default='ideal')
    parser.add_argument('--calibration-steps',type=int,default=100000)
    parser.add_argument('--production-steps',type=int,default=100000)
    parser.add_argument('--chunk-steps',type=int,default=5000)
    parser.add_argument('--coordinate-stride',type=int,default=100)
    parser.add_argument('--calibration-gain',type=float,default=.2)
    parser.add_argument('--calibration-tau',type=float,default=50000.)
    parser.add_argument('--gibbs-label',action='store_true',help='Exact label refresh before lazy/marked neighbor hop')
    parser.add_argument('--vacancy-relocation-probability',type=float,default=0.,
        help='Optional fixed-N residual-vacancy MH opportunity per structured-node local sweep')
    parser.add_argument('--calibration-method',choices=('one_over_t','staged_wang_landau'),default='one_over_t')
    parser.add_argument('--calibration-label-quota',choices=('uniform','endpoint_priority'),default='uniform',
        help='Discarded label-efficiency allocation, not physical phase probabilities')
    parser.add_argument('--calibration-normalizer-projection',choices=('none','nested_physical_radius','certified_normalizers'),default='none',
        help='Opt-in discarded staged-calibration normalizer consistency regularization; never production')
    parser.add_argument('--calibration-minimum-gain',type=float,default=.01875)
    parser.add_argument('--calibration-flatness',type=float,default=.5)
    parser.add_argument('--calibration-minimum-visits',type=int,default=100)
    parser.add_argument('--calibration-half-minimum-visits',type=int,default=50)
    parser.add_argument('--calibration-number-tolerance',type=float,default=.25)
    parser.add_argument('--calibration-bragg-tolerance',type=float,default=.10)
    parser.add_argument('--weights-from',type=Path)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    if (args.calibration_steps<0 or args.production_steps<=0 or args.chunk_steps<=0
            or args.coordinate_stride<=0 or not np.isfinite(args.calibration_gain)
            or args.calibration_gain<0 or not np.isfinite(args.calibration_tau) or args.calibration_tau<=0):
        parser.error('Positive finite counts/scales and nonnegative calibration required')
    if (not np.isfinite(args.vacancy_relocation_probability)
            or not 0 <= args.vacancy_relocation_probability <= 1):
        parser.error('Vacancy-relocation probability must be finite and in [0,1]')
    if (args.calibration_normalizer_projection!='none'
            and (args.calibration_method!='staged_wang_landau' or args.calibration_steps<=0)):
        parser.error('Normalizer projection requires positive discarded staged calibration')
    paths = [args.output.with_suffix(suffix) for suffix in ('.npz','.json','.checkpoint.npz')]
    if not args.overwrite and any(path.exists() for path in paths):
        parser.error('Output or recoverable checkpoint exists; choose a new prefix')
    graph_path = args.graph_from.resolve()
    if any(path.resolve() in (graph_path,graph_path.with_suffix('.json')) for path in paths):
        parser.error('Output may not overwrite graph input')
    target,data,graph_metadata = load_graph(graph_path)
    sites,cap = graph_metadata['settings']['sites'],graph_metadata['settings']['cap']
    neighbors,edge_table,degrees = validate_targets(target,cap)
    nodes = len(degrees)
    label_quota = resolve_label_quota(nodes,target['base_node'],target['cavity_node'],
        target['ideal_node'],args.calibration_label_quota)
    nonuniform_quota = bool(np.any(label_quota!=label_quota[0]))
    projection_nodes,projection_rows = 0,0
    if args.calibration_normalizer_projection!='none':
        try:
            if args.calibration_normalizer_projection=='certified_normalizers':
                projection_constraints = normalizer_projection.certified_normalizer_constraints(target,label_quota,cap)
                projection_nodes = len(projection_constraints['selected_nodes'])
                projection_rows = (len(projection_constraints['inequality_matrix'])
                                   +len(projection_constraints['equality_matrix']))
            else:
                projection_constraints = normalizer_projection.nested_physical_constraints(target,label_quota,cap)
                projection_nodes = len(projection_constraints['path_nodes'])
        except ValueError as error:
            parser.error(str(error))
    dense = np.unique(np.r_[target['base_node'],target['cavity_node'],
        target['physical_nodes'][target['activity'][target['physical_nodes']]>=12]]).astype(np.int64)
    memory_budget = projected_memory_budget(args.production_steps,args.calibration_steps,
        args.chunk_steps,args.coordinate_stride,cap,nodes,
        len(dense) if args.calibration_method=='staged_wang_landau' else 0,args.gibbs_label,nonuniform_quota,
        projection_nodes,projection_rows)
    if memory_budget['raw_array_bytes']>memory_budget['raw_limit_bytes']:
        parser.error('Projected raw arrays exceed 200 MiB; shorten batches or thin coordinates')
    if memory_budget['projected_peak_bytes']>memory_budget['peak_limit_bytes']:
        parser.error('Projected peak workspace exceeds 2 GiB; shorten the budget or increase the bounded chunk size')
    staged = None
    if args.calibration_method=='staged_wang_landau':
        try:
            staged = StagedFlatnessCalibration(nodes,dense,args.calibration_gain,
                args.calibration_minimum_gain,args.calibration_flatness,args.calibration_minimum_visits,
                args.calibration_half_minimum_visits,args.calibration_number_tolerance,
                args.calibration_bragg_tolerance,label_quota,args.calibration_normalizer_projection)
        except ValueError as error:
            parser.error(str(error))
    if args.weights_from:
        weights = load_frozen_weights(args.weights_from,target,sites,cap)
    else:
        weights = (data['initial_log_weights'].copy() if 'initial_log_weights' in data else
                   initial_weight_guess(target,data['samples'][::10],data['pair_overlap_counts'][::10]))
    if weights.shape!=(nodes,) or not np.isfinite(weights).all():
        parser.error('Node log weights must be a finite node-count vector')
    weights -= weights[target['ideal_node']]
    initial_weights = weights.copy()
    positions = np.zeros((1,cap,3))
    counts = np.zeros(1,dtype=np.int64)
    centers = np.zeros((1,3))
    length = target['volume']**(1/3)
    if args.initialization=='ideal':
        node = target['ideal_node']
    else:
        node = target['base_node']
        positions,counts,centers,_ = kernel.initialize(sites,length,np.ones(1),np.zeros(1),cap,
                                                       args.seed,args.initialization)
    reference = kernel.fcc_reference(sites,length)
    physical = np.zeros(nodes,dtype=np.bool_)
    physical[target['physical_nodes']] = True
    cdf = capped_poisson_cdf(target['activity'][target['ideal_node']]+math.log(target['volume']),cap)
    sources = {name:capture_source(path,content) for name,path,content in (
        ('driver',__file__,EXECUTION_SOURCE_BYTES),
        ('kernel',kernel.__file__,kernel.EXECUTION_SOURCE_BYTES),
        ('graph_validation',replica_validation.__file__,VALIDATION_SOURCE_BYTES),
        ('native_graph',native_graph.__file__,NATIVE_GRAPH_SOURCE_BYTES))}
    if args.calibration_normalizer_projection!='none':
        sources['normalizer_projection'] = capture_source(normalizer_projection.__file__,
            normalizer_projection.EXECUTION_SOURCE_BYTES)
    input_hashes = {replica_validation.display_path(path):kernel.file_sha256(path)
                    for path in (graph_path,graph_path.with_suffix('.json'))}
    if args.weights_from:
        for path in (args.weights_from,args.weights_from.with_suffix('.json')):
            input_hashes[replica_validation.display_path(path.resolve())] = kernel.file_sha256(path)
    kernel.seed_rng(args.seed)
    trajectories,coordinates,coordinate_indices,coordinate_labels,center_chunks,neighbor_labels = [],[],[],[],[],[]
    calibration_visits = np.zeros(nodes,dtype=np.int64)
    calibration_weight_update_visits = np.zeros(nodes,dtype=np.int64)
    local_accepts = np.zeros(8,dtype=np.int64)
    label_moves = np.zeros((len(target['exchange_edges']),2),dtype=np.int64)
    production_gibbs_counts = np.zeros(2,dtype=np.int64)
    production_relocation_counts = np.zeros(2,dtype=np.int64)
    cycle,clock = 0,0
    calibration_clock = 0
    calibration_outcome = None
    start = time.time()
    for stage,total in (('calibration',args.calibration_steps),('production',args.production_steps)):
        completed = 0
        stage_visits = np.zeros(nodes,dtype=np.int64)
        stage_heatbaths = 0
        while completed<total:
            steps = min(args.chunk_steps,total-completed)
            centers_out = np.empty((len(np.arange(args.coordinate_stride-(clock%args.coordinate_stride)-1,
                                                steps,args.coordinate_stride)),3))
            neighbor_out = np.empty(steps,dtype=np.int64) if args.gibbs_label else None
            updates_out = np.empty(steps,dtype=np.int64) if stage=='calibration' else None
            gibbs_counts = np.zeros(2,dtype=np.int64) if args.gibbs_label else None
            relocation_counts = np.zeros(2,dtype=np.int64)
            gain = (staged.gain if staged is not None else args.calibration_gain) if stage=='calibration' else 0.
            result = run_tempering_chunk(positions,counts,centers,node,cycle,clock,steps,reference,length,
                target['diameter'],target['radius'],target['activity'],target['pair_penalty'],
                target['umbrella_stiffness'],target['umbrella_preferred'],target['number_stiffness'],
                target['number_preferred'],target['displacement'],physical,target['base_node'],
                target['cavity_node'],target['ideal_node'],neighbors,edge_table,degrees,weights,cdf,
                gain,args.calibration_tau,args.coordinate_stride,coordinate_centers=centers_out,
                gibbs_label=args.gibbs_label,constant_calibration_gain=staged is not None,
                neighbor_source_labels=neighbor_out,weight_update_labels=updates_out,gibbs_counters=gibbs_counts,
                vacancy_relocation_probability=args.vacancy_relocation_probability,
                vacancy_relocation_counts=relocation_counts,
                calibration_label_quota=label_quota if nonuniform_quota else None)
            trace,accepts,moves,configs,indices,labels,node,cycle = result
            completed += steps
            clock += steps
            stage_visits += np.bincount(trace[:,1].astype(int),minlength=nodes)
            stage_heatbaths += int(trace[:,11].sum())
            if stage=='calibration':
                calibration_visits += np.bincount(trace[:,1].astype(int),minlength=nodes)
                if gain>0:
                    calibration_weight_update_visits += np.bincount(updates_out,minlength=nodes)
                calibration_clock = clock
                projection_record = None
                if args.calibration_normalizer_projection!='none':
                    projection_function = (normalizer_projection.project_certified_normalizer_weights
                        if args.calibration_normalizer_projection=='certified_normalizers' else
                        normalizer_projection.project_physical_normalizer_weights)
                    projected,projection_record = projection_function(weights,label_quota,target,cap)
                    if np.any(np.abs(projected)>1e5+1e-9):
                        raise ValueError('Projected weights exceed the declared calibration clip bound; no post-projection clipping is allowed')
                    weights[:] = projected
                if staged is not None:
                    epoch = staged.observe(trace,updates_out,clock,weights,projection_record)
            else:
                trajectories.append(trace)
                coordinates.append(configs)
                coordinate_indices.append(indices)
                coordinate_labels.append(labels)
                center_chunks.append(centers_out)
                if args.gibbs_label:
                    neighbor_labels.append(neighbor_out)
                    production_gibbs_counts += gibbs_counts
                local_accepts += accepts
                label_moves += moves
                production_relocation_counts += relocation_counts
            paths[2].parent.mkdir(parents=True,exist_ok=True)
            temporary = paths[2].with_suffix('.tmp.npz')
            np.savez_compressed(temporary,positions=positions,counts=counts,centers=centers,
                label=node,cycle=cycle,clock=clock,stage=stage,log_weights=weights,
                calibration_label_quota=label_quota,
                calibration_normalizer_projection=args.calibration_normalizer_projection,
                vacancy_relocation_counts=production_relocation_counts,
                calibration_weight_update_visits=calibration_weight_update_visits,
                calibration_outcome=checkpoint_outcome_bytes(staged.outcome(calibration_clock,args.calibration_steps)
                    if staged is not None else None),**target)
            temporary.replace(paths[2])
            print(json.dumps({'stage':stage,'steps':completed,'label':int(node),'N':int(counts[0]),
                'visited_nodes':int(np.count_nonzero(stage_visits)),
                'physical_visits':int(stage_visits[target['physical_nodes']].sum()),
                'base_visits':int(stage_visits[target['base_node']]),
                'cavity_visits':int(stage_visits[target['cavity_node']]),
                'stage_heatbath_events':stage_heatbaths,
                'total_regenerations':int(cycle),'calibration_gain':gain,
                'vacancy_relocation_attempted_accepted':production_relocation_counts.tolist(),
                'calibration_stage':staged.stage if staged is not None else None,
                'stage_criteria_passed':epoch['stage_passed'] if stage=='calibration' and staged is not None else None,
                'elapsed_s':round(time.time()-start,2)}),flush=True)
            if stage=='calibration' and staged is not None and staged.completed:
                break
        weights -= weights[target['ideal_node']]
        if not np.isfinite(weights).all():
            raise ValueError('Gauge-fixed production weights must remain finite')
        if stage=='calibration':
            calibration_outcome = (staged.outcome(calibration_clock,args.calibration_steps) if staged is not None else
                {'method':'one_over_t','completed':True,'budget_exhausted':False,
                 'actual_clock_steps':calibration_clock,'requested_budget':args.calibration_steps,
                 'label_quota':label_quota.tolist(),
                 'flatness_measure':('minimum(counts/quota)/mean(counts/quota)' if nonuniform_quota
                                      else 'minimum(counts)/mean(counts)'),
                 'stage_history':[],'scope':'Requested discarded adaptation schedule completed; no flatness or equilibrium claim'})
            if args.calibration_normalizer_projection!='none':
                calibration_outcome['normalizer_projection_mode'] = args.calibration_normalizer_projection
    samples = np.concatenate(trajectories)
    protocol_fields = {'neighbor_source_labels':np.concatenate(neighbor_labels),
                       'gibbs_attempt_change_counts':production_gibbs_counts} if args.gibbs_label else {}
    np.savez_compressed(paths[0],schema=SCHEMA,samples=samples,log_weights=weights,sites=sites,cap=cap,
        initial_log_weights=initial_weights,calibration_visits=calibration_visits,
        calibration_label_quota=label_quota,
        calibration_normalizer_projection=args.calibration_normalizer_projection,
        calibration_weight_update_visits=calibration_weight_update_visits,
        coordinate_configurations=np.concatenate(coordinates),coordinate_clock_steps=np.concatenate(coordinate_indices),
        coordinate_labels=np.concatenate(coordinate_labels),coordinate_centers=np.concatenate(center_chunks),
        vacancy_relocation_counts=production_relocation_counts,
        local_accepts=local_accepts,
        label_move_counts=label_moves,final_positions=positions,final_counts=counts,
        final_centers=centers,final_label=node,final_cycle=cycle,final_clock=clock,**target,**protocol_fields)
    metadata = {'schema':SCHEMA,'settings':{k:replica_validation.display_path(v.resolve()) if isinstance(v,Path) else v
                                           for k,v in vars(args).items()},
        'generated_utc':datetime.now(timezone.utc).isoformat(),'columns':COLUMNS,
        'memory_budget':memory_budget,'checkpoint_calibration_outcome_encoding':'utf-8 JSON byte scalar',
        'source_snapshots':sources,'input_sha256':input_hashes,'raw_sha256':kernel.file_sha256(paths[0]),
        'calibration_cutoff_clock':calibration_clock,'sample_cadence_clock_steps':1,
        'calibration_method':args.calibration_method,'calibration_outcome':calibration_outcome,
        'calibration_label_quota':label_quota.tolist(),
        'calibration_label_quota_mode':args.calibration_label_quota,
        'calibration_normalizer_projection':args.calibration_normalizer_projection,
        'calibration_flatness_measure':('minimum(counts/quota)/mean(counts/quota)' if nonuniform_quota
                                       else 'minimum(counts)/mean(counts)'),
        'calibration_weight_update_rule':('b += gain*quota; b[final_label] -= gain; periodic ideal-node gauge'
            if nonuniform_quota else 'legacy uniform decrement-only; periodic ideal-node gauge'),
        'calibration_gauge_interval_clock_steps':1000,
        'calibration_log_weight_clip':[-1e5,1e5],
        'calibration_weight_update_label_stage':'Final label after Gibbs/neighbor composition, exactly the decremented b entry',
        'calibration_observation_label_stage':'Pre-Gibbs local trace label paired with the same N/order features',
        'label_update_protocol':'gibbs_then_lazy_neighbor_tagged' if args.gibbs_label else 'lazy_neighbor_tagged',
        'vacancy_relocation_proposal':'uniform_tag_residual_vacancy_mh_v1',
        'vacancy_relocation_scope':'same-diameter structured nodes (pair_penalty >= 8 or umbrella_stiffness > 0)',
        'vacancy_relocation_timing':'after GCMC/local translations, before fixed-clock observation and label updates',
        'weights_frozen_before_production':True,
        'sampling_stage':('after local kernel, before conditional Gibbs and lazy neighbor/Palm updates; all dwell/rejections retained'
                          if args.gibbs_label else 'after local kernel, before lazy label hop; all dwell/rejections retained'),
        'regeneration':'Every incoming ideal-label step refreshes exact capped-Poisson N and iid uniform host/center before local update',
        'complete_cycle_definition':'Heatbath event row i through row before next event; retain zero-physical-dwell cycles; discard pre-event and unfinished terminal rows',
        'cycle_uncertainty_scope':'Fixed-clock horizon selects a random duration-dependent prefix; whole-cycle inference is asymptotic, not an exact fixed-iid-count finite-sample assertion',
        'target':'exp(b_j+x_j N-Phi_j)/N! dq/V; frozen finite b changes only label marginal',
        'target_origin':graph_metadata.get('target_origin','Validated frozen replica graph used only for auxiliary efficiency'),
        'warnings':['When imported, replica histories supply efficiency graph/weight guesses only; native graphs inherit box parameters, never physical phase odds',
                    'Calibration observations discarded; use completed regeneration-cycle dwell/reward sums, not means of cycle ratios',
                    'A short pilot is not equilibrium/coexistence evidence'],
        'elapsed_s':time.time()-start,'python':sys.version,'numpy':np.__version__,
        'reproduce_argv':['python3','-m','scripts.continuum_simulated_tempering',*sys.argv[1:]]}
    if args.calibration_normalizer_projection!='none':
        metadata['calibration_normalizer_projection_timing'] = (
            'after each discarded chunk, before staged calibration observation; never frozen production')
    paths[1].write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    paths[2].unlink()


if __name__=='__main__':
    main()
