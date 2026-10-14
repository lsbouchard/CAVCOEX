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
except ImportError:
    import continuum_replica_exchange as kernel
    import continuum_analyze as replica_validation

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SOURCE_BYTES = Path(__file__).read_bytes()
EXECUTION_SOURCE_SHA256 = hashlib.sha256(EXECUTION_SOURCE_BYTES).hexdigest()
VALIDATION_SOURCE_BYTES = Path(replica_validation.__file__).read_bytes()
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


@njit(cache=True)
def run_tempering_chunk(positions,counts,centers,node,cycle,clock,steps,reference,length,
        diameter,radius,activity,penalty,kappa,preferred,nu,nstar,displacement,
        physical,base,cavity,ideal,neighbors,edge_table,degrees,log_weights,cdf,
        calibration_gain=0.,calibration_tau=10000.,coordinate_stride=100,
        biased_birth=True,birth_width=.06,birth_uniform_fraction=.2,coordinate_centers=None):
    """One local exact kernel then one lazy label hop per fixed clock step."""
    trace = np.empty((steps,12))
    coordinate_indices = np.arange(coordinate_stride-(clock%coordinate_stride)-1,steps,coordinate_stride)
    if coordinate_centers is not None and coordinate_centers.shape != (len(coordinate_indices),3):
        raise ValueError("Coordinate-center output must match snapshot count and spatial dimension")
    configurations = np.full((len(coordinate_indices),positions.shape[1],3),np.nan)
    coordinate_labels = np.empty(len(coordinate_indices),dtype=np.int64)
    local_accepts = np.zeros(8,dtype=np.int64)
    edge_accepts = np.zeros((np.max(edge_table)+1,2),dtype=np.int64)
    identities = np.zeros(1,dtype=np.int64)
    one_d,one_r,one_x,one_p,one_k,one_b,one_nu,one_star,one_step = [np.zeros(1) for _ in range(9)]
    empty_edges = np.empty((0,2),dtype=np.int64)
    empty_nodes = np.empty(0,dtype=np.int64)
    physical_snapshot = np.zeros(1,dtype=np.int64)
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
            False,biased_birth,birth_width,birth_uniform_fraction,1,1,None,one_nu,one_star)
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
            gain = calibration_gain/(1+(clock+step)/calibration_tau)
            log_weights[node] -= gain
            # Gauge fix and finite bound are calibration-only efficiency choices.
            if (clock+step+1)%1000==0:
                log_weights -= log_weights[ideal]
                for j in range(len(log_weights)):
                    log_weights[j] = min(1e5,max(-1e5,log_weights[j]))
    return trace,local_accepts,edge_accepts,configurations,coordinate_indices+clock+1,coordinate_labels,node,cycle


def load_graph(path):
    """Validated frozen replica targets are efficiency inputs, not equilibrium evidence."""
    metadata = json.loads(Path(path).with_suffix('.json').read_text())
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
    parser.add_argument('--weights-from',type=Path)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    if (args.calibration_steps<0 or args.production_steps<=0 or args.chunk_steps<=0
            or args.coordinate_stride<=0 or not np.isfinite(args.calibration_gain)
            or args.calibration_gain<0 or not np.isfinite(args.calibration_tau) or args.calibration_tau<=0):
        parser.error('Positive finite counts/scales and nonnegative calibration required')
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
    projected = args.production_steps*len(COLUMNS)*8+(args.production_steps//args.coordinate_stride+1)*cap*3*8
    if projected>200*1024**2:
        parser.error('Projected raw arrays exceed 200 MiB; shorten batches or thin coordinates')
    if args.weights_from:
        weights = load_frozen_weights(args.weights_from,target,sites,cap)
    else:
        weights = initial_weight_guess(target,data['samples'][::10],data['pair_overlap_counts'][::10])
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
        ('graph_validation',replica_validation.__file__,VALIDATION_SOURCE_BYTES))}
    input_hashes = {replica_validation.display_path(path):kernel.file_sha256(path)
                    for path in (graph_path,graph_path.with_suffix('.json'))}
    if args.weights_from:
        for path in (args.weights_from,args.weights_from.with_suffix('.json')):
            input_hashes[replica_validation.display_path(path.resolve())] = kernel.file_sha256(path)
    kernel.seed_rng(args.seed)
    trajectories,coordinates,coordinate_indices,coordinate_labels,center_chunks = [],[],[],[],[]
    calibration_visits = np.zeros(nodes,dtype=np.int64)
    local_accepts = np.zeros(8,dtype=np.int64)
    label_moves = np.zeros((len(target['exchange_edges']),2),dtype=np.int64)
    cycle,clock = 0,0
    start = time.time()
    for stage,total in (('calibration',args.calibration_steps),('production',args.production_steps)):
        completed = 0
        while completed<total:
            steps = min(args.chunk_steps,total-completed)
            centers_out = np.empty((len(np.arange(args.coordinate_stride-(clock%args.coordinate_stride)-1,
                                                steps,args.coordinate_stride)),3))
            result = run_tempering_chunk(positions,counts,centers,node,cycle,clock,steps,reference,length,
                target['diameter'],target['radius'],target['activity'],target['pair_penalty'],
                target['umbrella_stiffness'],target['umbrella_preferred'],target['number_stiffness'],
                target['number_preferred'],target['displacement'],physical,target['base_node'],
                target['cavity_node'],target['ideal_node'],neighbors,edge_table,degrees,weights,cdf,
                args.calibration_gain if stage=='calibration' else 0.,args.calibration_tau,args.coordinate_stride,
                coordinate_centers=centers_out)
            trace,accepts,moves,configs,indices,labels,node,cycle = result
            completed += steps
            clock += steps
            if stage=='calibration':
                calibration_visits += np.bincount(trace[:,1].astype(int),minlength=nodes)
            else:
                trajectories.append(trace)
                coordinates.append(configs)
                coordinate_indices.append(indices)
                coordinate_labels.append(labels)
                center_chunks.append(centers_out)
                local_accepts += accepts
                label_moves += moves
            paths[2].parent.mkdir(parents=True,exist_ok=True)
            temporary = paths[2].with_suffix('.tmp.npz')
            np.savez_compressed(temporary,positions=positions,counts=counts,centers=centers,
                label=node,cycle=cycle,clock=clock,stage=stage,log_weights=weights,**target)
            temporary.replace(paths[2])
            print(json.dumps({'stage':stage,'steps':completed,'label':int(node),'N':int(counts[0]),
                'visited_nodes':int(np.count_nonzero(calibration_visits)),
                'regenerations':int(cycle),'elapsed_s':round(time.time()-start,2)}),flush=True)
        weights -= weights[target['ideal_node']]
    samples = np.concatenate(trajectories)
    np.savez_compressed(paths[0],schema=SCHEMA,samples=samples,log_weights=weights,sites=sites,cap=cap,
        initial_log_weights=initial_weights,calibration_visits=calibration_visits,
        coordinate_configurations=np.concatenate(coordinates),coordinate_clock_steps=np.concatenate(coordinate_indices),
        coordinate_labels=np.concatenate(coordinate_labels),coordinate_centers=np.concatenate(center_chunks),
        local_accepts=local_accepts,
        label_move_counts=label_moves,final_positions=positions,final_counts=counts,
        final_centers=centers,final_label=node,final_cycle=cycle,final_clock=clock,**target)
    metadata = {'schema':SCHEMA,'settings':{k:replica_validation.display_path(v.resolve()) if isinstance(v,Path) else v
                                           for k,v in vars(args).items()},
        'generated_utc':datetime.now(timezone.utc).isoformat(),'columns':COLUMNS,
        'source_snapshots':sources,'input_sha256':input_hashes,'raw_sha256':kernel.file_sha256(paths[0]),
        'calibration_cutoff_clock':args.calibration_steps,'sample_cadence_clock_steps':1,
        'weights_frozen_before_production':True,
        'sampling_stage':'after local kernel, before lazy label hop; all dwell/rejections retained',
        'regeneration':'Every incoming ideal-label step refreshes exact capped-Poisson N and iid uniform host/center before local update',
        'complete_cycle_definition':'Heatbath event row i through row before next event; retain zero-physical-dwell cycles; discard pre-event and unfinished terminal rows',
        'cycle_uncertainty_scope':'Fixed-clock horizon selects a random duration-dependent prefix; whole-cycle inference is asymptotic, not an exact fixed-iid-count finite-sample assertion',
        'target':'exp(b_j+x_j N-Phi_j)/N! dq/V; frozen finite b changes only label marginal',
        'warnings':['Incoming replica history only supplies efficiency graph/weight guesses, never physical phase odds',
                    'Calibration observations discarded; use completed regeneration-cycle dwell/reward sums, not means of cycle ratios',
                    'A short pilot is not equilibrium/coexistence evidence'],
        'elapsed_s':time.time()-start,'python':sys.version,'numpy':np.__version__,
        'reproduce_argv':['python3','-m','scripts.continuum_simulated_tempering',*sys.argv[1:]]}
    paths[1].write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    paths[2].unlink()


if __name__=='__main__':
    main()
