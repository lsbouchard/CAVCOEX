#!/usr/bin/env python3
"""Mixing diagnostics and independent reduced-Palm controls for continuum data.

Block resampling uses common indices for every replica/observable, preserving
correlations from exchanges.  Agreement with an identity is a diagnostic, not
a substitute for phase transitions, reservoir round trips, or independent
initialization checks.  The default 500-sweep block is reported, not silently
treated as known to exceed every correlation time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import zipfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/".mplconfig"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_processed/continuum_switching_audit.json"
DEFAULT_SEED = 202609293
PHASE_STATES = ("FF", "FS", "SF", "SS")
DELETION_LABEL_THRESHOLD = .45
EXECUTION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    """Use JSON null, not nonstandard NaN/Infinity, for undefined estimates."""
    if isinstance(value, dict):
        return {key:json_ready(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def archived_array_shape(path, name):
    """Read only an NPY header, never allocate a large coordinate trajectory."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{name}.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1,0):
                shape,_,_ = np.lib.format.read_array_header_1_0(handle)
            elif version == (2,0):
                shape,_,_ = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported coordinate-array header version: {version}")
    return shape


def round_trips(ids, low, high):
    """Count sampled walker round trips; unrecorded visits can only be missed."""
    if low == high:
        return 0,0,0
    nwalkers = int(ids.max())+1
    legs = np.zeros(nwalkers, dtype=int)
    origin = np.full(nwalkers, -1, dtype=int)
    for row in ids.astype(int):
        for edge, replica in ((0,low),(1,high)):
            walker = row[replica]
            if origin[walker] < 0:
                origin[walker] = edge
            elif edge != origin[walker] and legs[walker] % 2 == 0:
                legs[walker] += 1
            elif edge == origin[walker] and legs[walker] % 2 == 1:
                legs[walker] += 1
    return int(np.sum(legs//2)), int(np.sum(legs)), int(np.sum(legs>0))


def phase_transitions(order, low=.30, high=.65):
    """Count complete hysteresis-threshold visits, excluding threshold chatter."""
    transitions = np.zeros(2, dtype=int)
    phase = -1
    for value in order:
        current = 0 if value <= low else 1 if value >= high else -1
        if current >= 0:
            if phase >= 0 and current != phase:
                transitions[phase] += 1
            phase = current
    return transitions.tolist()


def cavity_path(endpoint, nrep, nodes=None):
    """Resolve explicit cavity-stage nodes, with historical linear defaults."""
    nodes = np.arange(endpoint,nrep) if nodes is None else np.asarray(nodes)
    if (nodes.ndim != 1 or len(nodes) < 2 or not np.isfinite(nodes).all()
            or not np.all(nodes == nodes.astype(int))):
        raise ValueError("cavity_nodes must be a one-dimensional integer path with two or more nodes")
    nodes = nodes.astype(int)
    if (nodes[0] != endpoint or np.any(nodes < 0) or np.any(nodes >= nrep)
            or len(np.unique(nodes)) != len(nodes)):
        raise ValueError("cavity_nodes must start at endpoint and contain distinct valid replica indices")
    return nodes


def estimates(samples, endpoint, volume, activity, threshold=DELETION_LABEL_THRESHOLD, cavity_nodes=None):
    """Empirical plug-in statistics, not independently fitted phase weights.

    Palm quantities use only the unconstrained endpoint and its delete-one
    labels; direct quantities use the cavity endpoint. The two sampled
    series remain correlated through exchanges. For a binary basin every
    nonzero deletion-label change has the same sign within a configuration,
    so the absolute stored sum also counts label-changing deletions.
    """
    if threshold != DELETION_LABEL_THRESHOLD:
        raise ValueError("Stored deletion-label changes use threshold 0.45; other thresholds require recomputing delete-one labels from coordinates")
    nodes = cavity_path(endpoint,samples.shape[1],cavity_nodes)
    base = samples[:,endpoint]
    inserted = samples[:,nodes[-1]]
    n = base[:,0]
    b = (base[:,1]>threshold).astype(float)
    deletion_shift = base[:,8]
    deletion_solid = n*b+deletion_shift
    probabilities = samples[:,nodes[:-1],3].mean(axis=0)
    pstage = float(np.prod(probabilities))
    pgnz = float(n.mean()/volume/np.exp(activity))
    q0 = float(b.mean())
    qstage = float(np.mean(inserted[:,1]>threshold))
    qpalm = float(deletion_solid.mean()/n.mean())
    bc = (inserted[:,1]>threshold).astype(float)
    nc = inserted[:,0]
    n_covariance = float(np.mean((n-n.mean())*(nc-nc.mean())))
    n_variance_product = float(n.var()*nc.var())
    phase_covariance = float(np.mean((b-b.mean())*(bc-bc.mean())))
    phase_variance_product = float(b.var()*bc.var())
    epsilon_solid = (float(deletion_shift.mean()/volume/np.exp(activity)/q0)
                     if q0 > 0 else float("nan"))
    # These phase-conditioned insertion works include exact finite-basin changes.
    psolid = pgnz*qpalm/q0 if q0 > 0 else float("nan")
    pfluid = pgnz*(1-qpalm)/(1-q0) if q0 < 1 else float("nan")
    result = {
        "mean_N":float(n.mean()),"variance_N":float(n.var()),
        "mean_N_inserted":float(inserted[:,0].mean()),
        "mean_N_inserted_Palm":float(np.mean(n*(n-1))/n.mean()),
        "response_direct":float(inserted[:,0].mean()-n.mean()),
        "response_Fano":float(n.var()/n.mean()-1),
        "Fano_factor":float(n.var()/n.mean()),
        "cavity_probability_stage":pstage,"cavity_probability_GNZ":pgnz,
        "work_stage":float(-np.log(pstage)) if pstage > 0 else float("inf"),
        "work_GNZ":float(-np.log(pgnz)),
        "q_solid_base":q0,"q_solid_inserted":qstage,"q_solid_Palm":qpalm,
        "p_solid_Palm":psolid,"p_fluid_Palm":pfluid,
        "epsilon_solid":epsilon_solid,
        "mean_delete_solid_label_shift":float(deletion_shift.mean()),
        "stage_minus_GNZ_work":float(np.log(pgnz/pstage)) if pstage > 0 else float("inf"),
        "direct_minus_Palm_N":float(inserted[:,0].mean()-np.mean(n*(n-1))/n.mean()),
        "direct_minus_Palm_q":float(qstage-qpalm),
        "deletion_label_change_probability":float(np.mean(np.abs(deletion_shift))/n.mean()),
        "endpoint_N_covariance":n_covariance,
        "endpoint_N_correlation":float(n_covariance/np.sqrt(n_variance_product)) if n_variance_product > 0 else float("nan"),
        "endpoint_N_sum_variance":float((n+nc).var()),
        "endpoint_N_variance_sum":float(n.var()+nc.var()),
        "joint_phase_covariance":phase_covariance,
        "joint_phase_correlation":float(phase_covariance/np.sqrt(phase_variance_product)) if phase_variance_product > 0 else float("nan"),
    }
    for state,i,j in zip(PHASE_STATES,(0,0,1,1),(0,1,0,1)):
        joint = float(np.mean((b==i)&(bc==j)))
        product = float(np.mean(b==i)*np.mean(bc==j))
        result[f"joint_phase_probability_{state}"] = joint
        result[f"joint_phase_product_{state}"] = product
        result[f"joint_phase_residual_{state}"] = joint-product
    result["joint_phase_product_total_variation"] = .5*sum(
        abs(result[f"joint_phase_residual_{state}"]) for state in PHASE_STATES)
    # Number-size-biased source basin versus the basin after deleting a
    # uniformly selected host particle. These are not classifier errors.
    deletion_fluid_to_solid = float(np.mean((1-b)*deletion_shift)/n.mean())
    deletion_solid_to_fluid = float(-np.mean(b*deletion_shift)/n.mean())
    result.update({
        "deletion_phase_probability_FF":float(np.mean(n*(1-b))/n.mean()-deletion_fluid_to_solid),
        "deletion_phase_probability_FS":deletion_fluid_to_solid,
        "deletion_phase_probability_SF":deletion_solid_to_fluid,
        "deletion_phase_probability_SS":float(np.mean(n*b)/n.mean()-deletion_solid_to_fluid),
    })
    for label, indicator, shift in (("solid",b,deletion_shift),("fluid",1-b,-deletion_shift)):
        q = float(indicator.mean())
        mean = float(np.mean(n*indicator)/q) if q > 0 else float("nan")
        variance = float(np.mean((n-mean)**2*indicator)/q) if q > 0 else float("nan")
        shifted_derivative = float(np.mean(shift*(n-mean-1))) if q > 0 else float("nan")
        normalization = float(np.mean(n*indicator+shift))
        result.update({
            f"mean_N_{label}":mean,
            f"variance_N_{label}":variance,
            f"epsilon_{label}":float(shift.mean()/volume/np.exp(activity)/q) if q > 0 else float("nan"),
            f"epsilon_derivative_{label}":float(shifted_derivative/volume/np.exp(activity)/q) if q > 0 else float("nan"),
            f"intraphase_response_Palm_{label}":float((q*(variance-mean)+shifted_derivative)/normalization) if q > 0 and normalization > 0 else float("nan"),
        })
    return result


def block_bootstrap(samples, endpoint, volume, activity, block_samples, replicates=500,
                    seed=DEFAULT_SEED, cavity_nodes=None):
    if block_samples < 1 or int(block_samples) != block_samples:
        raise ValueError("block_samples must be a positive integer")
    if replicates < 2 or int(replicates) != replicates:
        raise ValueError("bootstrap replicates must be at least two")
    block_samples,replicates = int(block_samples),int(replicates)
    blocks = len(samples)//block_samples
    if blocks < 4:
        return {}, blocks
    # Auxiliary replicas enter the mixing diagnostics, but not these physical
    # estimators. Slice once before resampling so a large exchange graph does
    # not multiply the allocation and memory traffic of every bootstrap draw.
    # Common time indices still preserve all correlations between cavity stages.
    nodes = cavity_path(endpoint,samples.shape[1],cavity_nodes)
    trimmed = samples[:blocks*block_samples,nodes,:]
    local_nodes = np.arange(len(nodes))
    block_array = trimmed.reshape(blocks,block_samples,*trimmed.shape[1:])
    rng = np.random.default_rng(seed)
    output = {}
    for repeat in range(replicates):
        indices = rng.integers(0,blocks,blocks)
        values = estimates(block_array[indices].reshape(trimmed.shape),0,volume,activity,
                           cavity_nodes=local_nodes)
        for name,value in values.items():
            output.setdefault(name,[]).append(value)
    errors = {name:float(np.std(values,ddof=1)) for name,values in output.items()
              if np.isfinite(values).all()}
    return errors, blocks


def event_visits(event, block_samples):
    """Sampled event entries and complete blocks visited, not independent visits."""
    event = np.asarray(event,dtype=bool)
    entries = int(np.count_nonzero(event & ~np.r_[False,event[:-1]]))
    blocks = len(event)//block_samples
    visited = int(event[:blocks*block_samples].reshape(blocks,block_samples).any(axis=1).sum())
    return {"sample_count":int(event.sum()),"sampled_entries":entries,
            "complete_blocks_visited":visited}


def zero_residual_check(residual, se):
    """Report one identity check; neither a small residual nor acceptance certifies mixing."""
    finite = bool(np.isfinite(residual))
    available = se is not None and np.isfinite(se)
    nonzero = finite and not np.isclose(residual,0.,atol=1e-12,rtol=0.)
    zero_error_failure = bool(available and se == 0 and nonzero)
    exceeds = bool(finite and available and nonzero and
                   (zero_error_failure or (se > 0 and abs(residual) > 3*se)))
    status = ("undefined estimate" if not finite else
              "uncertainty unavailable" if not available else
              "residual exceeds three SE" if exceeds else "residual within three SE")
    return {
        "residual":residual,"common_block_SE":se,
        "standardized_residual":float(residual/se) if finite and available and se > 0 else None,
        "zero_SE_with_nonzero_residual":zero_error_failure,
        "exceeds_three_SE":exceeds,"status":status,
    }


def inventory_trace_summary(counts, block_samples):
    """A fixed phase bank can survive rapid exchanges among many replicas."""
    counts = np.asarray(counts,dtype=int)
    half = len(counts)//2
    return {
        "sampled_count_trace":counts.tolist(),"mean":float(counts.mean()),
        "variance":float(counts.var()),"support":np.unique(counts).tolist(),
        "sampled_count_changes":int(np.count_nonzero(np.diff(counts))),
        "first_half_mean":float(counts[:half].mean()) if half else None,
        "second_half_mean":float(counts[half:].mean()),
        "production_quarter_means":[float(part.mean()) if len(part) else None
                                    for part in np.array_split(counts,4)],
        "level_visits":{str(level):event_visits(counts==level,block_samples)
                        for level in np.unique(counts)},
    }


def support_probability_diagnostics(data, values, errors, block_samples):
    """Distinguish a resolved staged product from an unresolved rare event."""
    nodes = data["cavity_nodes"]
    stages = []
    for source,target in zip(nodes[:-1],nodes[1:]):
        events = data["samples"][:,source,3].astype(bool)
        stages.append({
            "source_node":int(source),"target_node":int(target),
            "source_radius":float(data["radius"][source]),
            "target_radius":float(data["radius"][target]),
            "sampled_trials":len(events),"sampled_successes":int(events.sum()),
            "empirical_probability":float(events.mean()),
            "success_visits":event_visits(events,block_samples),
        })
    positive = all(stage["sampled_successes"] > 0 for stage in stages)
    work_se = errors.get("stage_minus_GNZ_work")
    status = ("unresolved: zero sampled support events in at least one factor" if not positive else
              "uncertainty unresolved: insufficient complete blocks or bootstrap draws with zero support" if work_se is None else
              "finite empirical work and common-block uncertainty available")
    return {
        "method":"multistage support product" if len(nodes)>2 else "direct rare-empty-event sampling",
        "has_intermediate_cavity_stages":len(nodes)>2,"stages":stages,
        "all_factors_have_sampled_successes":positive,
        "work_estimate_resolved":bool(positive and np.isfinite(values["work_stage"])),
        "work_difference_uncertainty_available":work_se is not None,
        "status":status,
        "interpretation":"Zero observed support does not establish zero equilibrium probability; unresolved work is not an identity failure or a successful verification. Two endpoint radii provide direct rare-event sampling, not an independently resolved intermediate-stage product. GNZ values never replace missing support measurements; finite estimates alone do not certify equilibrium.",
    }


def boundary_fraction_diagnostics(values, errors):
    """A constant sampled basin label is not exact equilibrium precision."""
    fields = ("q_solid_base","q_solid_inserted","q_solid_Palm")
    return {
        "fractions":{name:{
            "estimate":values[name],"common_block_SE":errors.get(name),
            "at_empirical_boundary":values[name] in (0.,1.),
            "no_sampled_phase_variation":values[name] in (0.,1.),
            "zero_bootstrap_SE_not_equilibrium_precision":values[name] in (0.,1.) and errors.get(name)==0,
        } for name in fields},
        "interpretation":"A boundary fraction with zero bootstrap variance reflects the visited data only, not zero equilibrium uncertainty or absence of the other basin; do not report it as physical 0±0 or 1±0 precision",
    }


def hard_phase_bank_diagnostics(data, block_samples):
    """Global unbiased R=0 inventory, not just a possibly diluted endpoint pair."""
    samples = data["samples"]
    nrep = samples.shape[1]
    penalty = data.get("pair_penalty",np.full(nrep,np.inf))
    stiffness = data.get("umbrella_stiffness",np.zeros(nrep))
    number_stiffness = data.get("number_stiffness",np.zeros(nrep))
    hard = (data["diameter"]==1)&np.isposinf(penalty)&(stiffness==0)&(number_stiffness==0)
    selected = hard&(data["radius"]==0)
    nodes = np.flatnonzero(selected)
    nodes = nodes[np.argsort(data["activity"][nodes])]
    series = samples[:,nodes]
    operational = np.sum(series[:,:,1]>.45,axis=1)
    conservative = np.sum((series[:,:,1]>.65)&(series[:,:,0]>=20),axis=1)
    all_nodes = np.flatnonzero(hard)
    all_series = samples[:,all_nodes]
    return {
        "nodes":nodes.tolist(),"activities":data["activity"][nodes].tolist(),
        "operational_solid_threshold":.45,"ordered_threshold":.65,"ordered_minimum_N":20,
        "operational_solid_inventory":inventory_trace_summary(operational,block_samples),
        "ordered_high_density_inventory":inventory_trace_summary(conservative,block_samples),
        "host_number_inventory":inventory_trace_summary(series[:,:,0].sum(axis=1),block_samples),
        "including_cavity_stages":{
            "nodes":all_nodes.tolist(),
            "operational_solid_inventory":inventory_trace_summary(
                np.sum(all_series[:,:,1]>.45,axis=1),block_samples),
            "ordered_high_density_inventory":inventory_trace_summary(
                np.sum((all_series[:,:,1]>.65)&(all_series[:,:,0]>=20),axis=1),block_samples),
            "interpretation":"Includes all unbiased hard cavity stages so transfers into and out of the R=0 branch are not mistaken for creation or destruction of the total hard-node phase bank",
        },
        "scope":"Saved production samples only; no burn-in history is inferred",
        "interpretation":"Global phase-bank motion across unbiased radius-zero hard-sphere nodes; a nearly product endpoint pair can conceal a conserved bank diluted across many nodes; variable inventory alone is not an equilibrium certificate and independent fluid/crystal starts remain necessary",
    }


def pair_overlap_diagnostics(data):
    """Descriptive soft-pair overlap profiles; absent observations stay absent."""
    if "pair_overlap_counts" not in data:
        return {"available":False,"status":"not recorded in this history"}
    overlaps = data["pair_overlap_counts"]
    penalty = data.get("pair_penalty",np.full(overlaps.shape[1],np.inf))
    profiles = []
    for node in range(overlaps.shape[1]):
        counts = overlaps[:,node]
        levels,frequency = np.unique(counts,return_counts=True)
        profiles.append({
            "node":node,"diameter":float(data["diameter"][node]),
            "activity":float(data["activity"][node]),
            "hard_pair_interaction":bool(np.isposinf(penalty[node])),
            "finite_pair_penalty":float(penalty[node]) if np.isfinite(penalty[node]) else None,
            "number_stiffness":float(data.get("number_stiffness",np.zeros(overlaps.shape[1]))[node]),
            "number_preferred":float(data.get("number_preferred",np.zeros(overlaps.shape[1]))[node]),
            "mean_overlap_count":float(counts.mean()),
            "zero_overlap_fraction":float(np.mean(counts==0)),
            "maximum_overlap_count":int(counts.max()),
            "production_quarter_means":[float(part.mean()) if len(part) else None
                                        for part in np.array_split(counts,4)],
            "sampled_count_distribution":{str(int(level)):int(number)
                                           for level,number in zip(levels,frequency)},
        })
    return {
        "available":True,"profiles":profiles,
        "definition":"Number of host pairs whose periodic separation is strictly below that node's pair diameter; no cavity-center pairs are counted",
        "scope":"Saved diagnostic frames only; distributions and quarter means are descriptive, with no independent-sample or equilibrium claim",
    }


def joint_number_support_diagnostics(base_number, inserted_number, block_samples,
                                     minimum_expected_count=5, minimum_blocks=4):
    """Flag rare joint tails that an empirical covariance bootstrap cannot invent.

    Nine below/at/above-modal-number cells give a bounded-size warning even
    for many occupancy levels. This is a support diagnostic, not a calibrated
    independence test; its absence is not proof of adequate covariance tails.
    """
    series = (np.asarray(base_number), np.asarray(inserted_number))
    modes = []
    categories = []
    sampled_variation = []
    for number in series:
        levels, counts = np.unique(number, return_counts=True)
        mode = levels[np.argmax(counts)]
        modes.append(int(mode))
        sampled_variation.append(bool(len(levels)>1))
        categories.append(np.sign(number-mode).astype(int)+1)
    names = ("below_mode", "at_mode", "above_mode")
    cells = {}
    sparse = []
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            name = f"{first}/{second}"
            expected = float(len(series[0])*np.mean(categories[0]==i)*np.mean(categories[1]==j))
            visits = event_visits((categories[0]==i)&(categories[1]==j),block_samples)
            cells[name] = {**visits,"empirical_product_expected_sample_count":expected}
            if expected > 0 and (expected < minimum_expected_count or
                                 visits["complete_blocks_visited"] < minimum_blocks):
                sparse.append(name)
    return {
        "modal_particle_numbers":modes,"coarse_joint_cells":cells,
        "sparse_positive_product_cells":sparse,
        "number_variation_sampled_at_both_endpoints":all(sampled_variation),
        "sparse_joint_tail_warning":bool(sparse or not all(sampled_variation)),
        "minimum_empirical_product_expected_count":minimum_expected_count,
        "minimum_complete_blocks_visited":minimum_blocks,
        "interpretation":"Nominal covariance/SE ratios are not calibrated rejections when joint tails are poorly represented: empirical resampling cannot generate absent simultaneous number fluctuations. Counts and expected counts are descriptive and correlated, not binomial effective sample sizes. This coarse support check is necessary caution, not a covariance-uncertainty certificate",
    }


def product_target_diagnostics(samples, endpoint, nodes, values, errors, block_samples,
                               minimum_entries=4, minimum_blocks=4):
    """Separate product-target and strict coverage checks from move acceptance.

    The fixed-index equilibrium target is a product across replicas. A pass
    here is necessary evidence, not proof of equilibrium. The coverage rule
    deliberately rejects unvisited joint cells; it is not a hypothesis-test
    p-value, and the four binary residuals are not four independent tests.
    """
    base = samples[:,endpoint]
    inserted = samples[:,nodes[-1]]
    b = (base[:,1]>.45).astype(int)
    bc = (inserted[:,1]>.45).astype(int)
    codes = 2*b+bc
    visits = {state:event_visits(codes==index,block_samples)
              for index,state in enumerate(PHASE_STATES)}
    missing = [state for state in PHASE_STATES if visits[state]["sample_count"] == 0
               and values[f"joint_phase_product_{state}"] > 0]
    insufficient = [state for state in PHASE_STATES
                    if visits[state]["sampled_entries"] < minimum_entries
                    or visits[state]["complete_blocks_visited"] < minimum_blocks]
    both_marginals = bool(0 < b.mean() < 1 and 0 < bc.mean() < 1)
    residual_fields = [f"joint_phase_residual_{state}" for state in PHASE_STATES]
    residual_fields.append("endpoint_N_covariance")
    residual_checks = {name:zero_residual_check(values[name],errors.get(name))
                       for name in residual_fields}
    number_support = joint_number_support_diagnostics(base[:,0],inserted[:,0],
                                                       block_samples,minimum_blocks=minimum_blocks)
    for name,check in residual_checks.items():
        sparse = (number_support["sparse_joint_tail_warning"] if name == "endpoint_N_covariance"
                  else bool(insufficient))
        check["sparse_joint_support_warning"] = sparse
        check["inference_scope"] = (
            "Nominal descriptive residual/SE only: insufficient joint support prevents a calibrated rejection"
            if sparse else "Nominal common-block diagnostic; not a calibrated independence test or equilibrium certificate"
        )
    occupancy = {}
    for endpoint_name,series in (("unconstrained",base),("inserted",inserted)):
        phase = series[:,1]>.45
        occupancy[endpoint_name] = {}
        for phase_name,selected in (("fluid",~phase),("solid",phase)):
            occupancy[endpoint_name][phase_name] = {}
            for name,column in (("N",0),("empty_reference_sites",4),("excess_site_occupancy",5)):
                levels = np.unique(series[selected,column]).astype(int)
                occupancy[endpoint_name][phase_name][name] = {
                    str(level):event_visits(selected&(series[:,column]==level),block_samples)
                    for level in levels}
    failures = []
    if not both_marginals:
        failures.append("Both basins were not visited at both physical endpoints")
    if missing:
        failures.append("Observed joint support omits cells with positive empirical marginal product")
    if insufficient:
        failures.append("Insufficient sampled entries or complete-block visits in joint phase cells")
    if number_support["sparse_joint_tail_warning"]:
        failures.append("Sparse joint number tails limit interpretation of covariance bootstrap uncertainty")
    if any(record["exceeds_three_SE"] for record in residual_checks.values()):
        failures.append("Nominal product-target residual exceeds three common-block standard errors; consult joint-support qualifications before inference")
    if any(record["common_block_SE"] is None for record in residual_checks.values()):
        failures.append("Common-block uncertainty is unavailable for a product-target residual")
    return {
        "state_order":list(PHASE_STATES),"joint_phase_visits":visits,
        "empirical_product_expected_sample_counts":{
            state:len(samples)*values[f"joint_phase_product_{state}"] for state in PHASE_STATES},
        "both_endpoints_visit_both_basins":both_marginals,
        "missing_cells_with_positive_empirical_product":missing,
        "strict_joint_inventory_coverage_failure":bool(missing),
        "insufficient_joint_state_visits":insufficient,
        "minimum_sampled_entries_per_cell":minimum_entries,
        "minimum_complete_blocks_per_cell":minimum_blocks,
        "product_target_residual_checks":residual_checks,
        "joint_number_support":number_support,
        "occupancy_support_and_visits":occupancy,
        "endpoint_N_sum_support":np.unique(base[:,0]+inserted[:,0]).astype(int).tolist(),
        "failure_reasons":failures,
        "passes_all_reported_diagnostics":not failures,
        "interpretation":"Coverage and common-block residual diagnostics, not proof of equilibration; sampled entries are not independent visits; four binary residuals encode one association; MH acceptance cannot override failures",
        "occupancy_interpretation":"Nearest-FCC reference-site occupancies are structural diagnostics, not thermodynamic defect counts",
    }


def load_validated_run(path, block_sweeps):
    """Check one trajectory without retaining other runs in memory."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    # Earlier archives use the same fixed threshold but predate this explicit
    # sidecar field. Their column-8 statistic must never be reinterpreted.
    if metadata.get("deletion_label_threshold",DELETION_LABEL_THRESHOLD) != DELETION_LABEL_THRESHOLD:
        raise ValueError("Archived deletion-label threshold is incompatible with the fixed 0.45 analysis")
    if metadata.get("raw_sha256") != sha256(path):
        raise ValueError(f"Raw-data digest does not match execution metadata: {path}")
    stride = metadata["settings"]["sample_stride"]
    if stride < 1 or block_sweeps < stride or block_sweeps % stride:
        raise ValueError("block_sweeps must be a positive multiple of the recorded sample_stride")
    with np.load(path, allow_pickle=False) as archive:
        required = ("samples", "endpoint", "volume", "activity", "diameter", "radius", "swaps")
        data = {name: archive[name] for name in required}
        for name in ("cavity_nodes", "exchange_edges", "pair_penalty",
                     "umbrella_stiffness", "umbrella_preferred", "palm_move_counts", "accepts",
                     "probe_nodes", "physical_probe_activity", "probe_sample_indices", "pair_overlap_counts",
                     "number_stiffness", "number_preferred", "displacement"):
            if name in archive:
                data[name] = archive[name]
        if "physical_probe_configurations" in archive:
            data["probe_configuration_shape"] = archived_array_shape(path,"physical_probe_configurations")
    samples = data["samples"]
    endpoint = int(data["endpoint"])
    if (samples.ndim != 3 or len(samples) == 0 or samples.shape[2] < 9
            or not 0 <= endpoint < samples.shape[1]):
        raise ValueError("Expected nonempty time-by-replica samples with nine diagnostics and distinct endpoints")
    if (not np.isfinite(samples).all() or np.any(samples[:,:,0] < 0)
            or np.any(samples[:,:,0] != np.rint(samples[:,:,0]))):
        raise ValueError("Samples contain nonfinite diagnostics or noninteger/negative particle counts")
    nrep = samples.shape[1]
    identities = samples[:,:,2]
    if (np.any(identities != np.rint(identities)) or np.any(identities < 0)
            or np.any(identities >= nrep)
            or not np.all(np.sort(identities,axis=1) == np.arange(nrep))):
        raise ValueError("Every sample must contain one instance of each valid walker identity")
    if any(data[name].shape != (nrep,) for name in ("activity", "diameter", "radius")):
        raise ValueError("Ladder arrays do not match the sampled replica count")
    if "displacement" in data and (data["displacement"].shape != (nrep,)
            or not np.isfinite(data["displacement"]).all() or np.any(data["displacement"] <= 0)):
        raise ValueError("Recorded displacement scales must be finite positive replica-length arrays")
    if (any(not np.isfinite(data[name]).all() for name in ("activity","diameter","radius"))
            or np.any(data["diameter"] < 0) or np.any(data["radius"] < 0)):
        raise ValueError("Ladder parameters must be finite, with nonnegative diameter and radius")
    data["cavity_nodes"] = cavity_path(endpoint,nrep,data.get("cavity_nodes"))
    nodes = data["cavity_nodes"]
    if np.any(~np.isin(samples[:,nodes[:-1],3],(0.,1.))):
        raise ValueError("Next-stage support measurements must be binary event indicators")
    if (data["radius"][nodes[0]] != 0 or data["radius"][nodes[-1]] != 1
            or np.any(np.diff(data["radius"][nodes]) <= 0)
            or np.any(data["diameter"][nodes] != 1)
            or np.any(data["activity"][nodes] != data["activity"][endpoint])):
        raise ValueError("Cavity stages must share the physical diameter/activity and increase from radius zero to one")
    # Legacy hard-diameter paths predate these arrays and have no soft pair
    # penalty or umbrella. New graph paths must explicitly be unbiased here.
    penalty = data.get("pair_penalty",np.full(nrep,np.inf))
    stiffness = data.get("umbrella_stiffness",np.zeros(nrep))
    if (penalty.shape != (nrep,) or np.isnan(penalty).any() or np.any(penalty < 0)
            or stiffness.shape != (nrep,) or not np.isfinite(stiffness).all()
            or np.any(stiffness < 0)):
        raise ValueError("Invalid pair-penalty or umbrella-stiffness array")
    if np.any(~np.isposinf(penalty[nodes])) or np.any(stiffness[nodes] != 0):
        raise ValueError("Every cavity-path node must be an unbiased physical hard-sphere ensemble")
    if ("number_stiffness" in data) != ("number_preferred" in data):
        raise ValueError("Number umbrella requires both number_stiffness and number_preferred arrays")
    declared_number_stiffness = metadata["settings"].get("compact_number_stiffness",0.)
    if not np.isfinite(declared_number_stiffness) or declared_number_stiffness < 0:
        raise ValueError("Declared number-umbrella stiffness must be finite and nonnegative")
    if declared_number_stiffness and "number_stiffness" not in data:
        raise ValueError("Execution metadata declares a number umbrella but its target arrays are missing")
    number_stiffness = data.get("number_stiffness",np.zeros(nrep))
    number_preferred = data.get("number_preferred",np.zeros(nrep))
    if (number_stiffness.shape != (nrep,) or number_preferred.shape != (nrep,)
            or not np.isfinite(number_stiffness).all() or not np.isfinite(number_preferred).all()
            or np.any(number_stiffness < 0) or np.any(number_preferred < 0)):
        raise ValueError("Number-umbrella parameters must be finite nonnegative replica-length arrays")
    if np.any(number_stiffness[nodes] != 0):
        raise ValueError("Physical cavity-path nodes must have zero number umbrella strength")
    if declared_number_stiffness and not np.any(number_stiffness > 0):
        raise ValueError("Execution metadata declares a number umbrella but every recorded stiffness is zero")
    # Normalize the legacy absent-array contract before graph comparisons.
    data["number_stiffness"] = number_stiffness
    data["number_preferred"] = number_preferred
    if "pair_overlap_counts" in data:
        overlaps = data["pair_overlap_counts"]
        if (overlaps.shape != samples.shape[:2] or not np.isfinite(overlaps).all()
                or np.any(overlaps < 0) or np.any(overlaps != np.rint(overlaps))
                or np.any(overlaps > samples[:,:,0]*(samples[:,:,0]-1)/2)):
            raise ValueError("Pair-overlap counts must be nonnegative integers per diagnostic frame/replica, bounded by the host-pair count")
        hard_or_zero_diameter = np.isposinf(penalty)|(data["diameter"]==0)
        if np.any(overlaps[:,hard_or_zero_diameter] != 0):
            raise ValueError("Hard or zero-diameter nodes must have zero pair-overlap counts")
    if "umbrella_preferred" in data:
        preferred = data["umbrella_preferred"]
        if preferred.shape != (nrep,) or not np.isfinite(preferred).all():
            raise ValueError("Invalid umbrella-preferred array")
    if "probe_nodes" in data or "physical_probe_activity" in data:
        probes = data.get("probe_nodes")
        activities = data.get("physical_probe_activity")
        if (probes is None or activities is None or probes.ndim != 1
                or not np.isfinite(probes).all() or np.any(probes != probes.astype(int))
                or np.any(probes < 0) or np.any(probes >= nrep)
                or len(np.unique(probes)) != len(probes) or endpoint in probes
                or activities.shape != probes.shape):
            raise ValueError("Physical activity probes require distinct valid nodes and matching activities")
        probes = probes.astype(int)
        if (not np.array_equal(activities,data["activity"][probes])
                or np.any(data["diameter"][probes] != 1) or np.any(data["radius"][probes] != 0)
                or np.any(~np.isposinf(penalty[probes])) or np.any(stiffness[probes] != 0)
                or np.any(number_stiffness[probes] != 0)):
            raise ValueError("Physical activity probes must be unbiased radius-zero hard-sphere ensembles")
        data["probe_nodes"] = probes
        indices = data.get("probe_sample_indices",np.arange(len(samples)))
        if (indices.ndim != 1 or not np.isfinite(indices).all()
                or np.any(indices != indices.astype(int)) or np.any(indices < 0)
                or np.any(indices >= len(samples)) or np.any(np.diff(indices) <= 0)):
            raise ValueError("probe_sample_indices must be distinct increasing diagnostic-frame indices")
        data["probe_sample_indices"] = indices.astype(int)
        if "probe_configuration_shape" in data:
            shape = data["probe_configuration_shape"]
            if len(shape) != 4 or shape[0] != len(indices) or shape[1] != len(probes) or shape[3] != 3:
                raise ValueError("Probe-coordinate axes do not match saved frame indices and probe_nodes")
    elif "probe_sample_indices" in data or "probe_configuration_shape" in data:
        raise ValueError("Probe coordinates or frame indices require explicit probe_nodes and activities")
    edges = data.get("exchange_edges",np.column_stack((np.arange(nrep-1),np.arange(1,nrep))))
    if (edges.ndim != 2 or edges.shape[1] != 2 or len(edges) < 1
            or not np.isfinite(edges).all() or not np.all(edges == edges.astype(int))
            or np.any(edges < 0) or np.any(edges >= nrep)
            or np.any(edges[:,0] == edges[:,1])):
        raise ValueError("exchange_edges must contain pairs of distinct valid replica indices")
    data["exchange_edges"] = edges.astype(int)
    volume = float(data["volume"])
    if data["swaps"].shape != (len(edges), 2) or not np.isfinite(volume) or volume <= 0:
        raise ValueError("Invalid swap-counter dimensions or nonpositive volume")
    if samples[:,endpoint,0].mean() <= 0:
        raise ValueError("The reduced-Palm estimate requires a positive sampled mean particle number")
    if "palm_move_counts" in data:
        counts = data["palm_move_counts"]
        if (counts.shape != (2,) or not np.isfinite(counts).all()
                or np.any(counts != np.rint(counts)) or np.any(counts < 0)
                or counts[1] > counts[0]):
            raise ValueError("Invalid Palm-transfer attempt/accept counters")
        accepts = data.get("accepts")
        if accepts is not None:
            if accepts.ndim != 2 or accepts.shape[0] != nrep or accepts.shape[1] < 8:
                raise ValueError("Palm-transfer endpoint counters require a replica-by-eight counter array")
            if not np.array_equal(counts,accepts[endpoint,6:8]):
                raise ValueError("Palm-transfer counters disagree with the physical endpoint record")
    return data,metadata


def analyze(path, block_sweeps=500, bootstrap=500, seed=DEFAULT_SEED):
    if block_sweeps < 1 or bootstrap < 2 or seed < 0:
        raise ValueError("block sweeps must be positive, bootstrap draws >= 2, and seed >= 0")
    data,metadata = load_validated_run(path,block_sweeps)
    samples = data["samples"]
    endpoint = int(data["endpoint"])
    nodes = data["cavity_nodes"]
    stride = metadata["settings"]["sample_stride"]
    values = estimates(samples,endpoint,float(data["volume"]),float(data["activity"][endpoint]),cavity_nodes=nodes)
    errors, blocks = block_bootstrap(samples,endpoint,float(data["volume"]),float(data["activity"][endpoint]),block_sweeps//stride,bootstrap,seed,cavity_nodes=nodes)
    physical = samples[:,endpoint]
    cavity = samples[:,nodes[-1]]
    swap = data["swaps"]
    acceptance = swap[:,1]/np.maximum(swap[:,0],1)
    nrep = samples.shape[1]
    penalty = data.get("pair_penalty",np.full(nrep,np.inf))
    stiffness = data.get("umbrella_stiffness",np.zeros(nrep))
    ideal_nodes = np.flatnonzero(((data["diameter"]==0)|(penalty==0))
                                &(data["radius"]==0)&(stiffness==0)&(data["number_stiffness"]==0))
    ideal = int(ideal_nodes[0]) if len(ideal_nodes) else None
    diagnostics = {
        "physical_phase_transitions_low_to_high_high_to_low":phase_transitions(physical[:,1]),
        "inserted_phase_transitions_low_to_high_high_to_low":phase_transitions(cavity[:,1]),
        "ideal_reservoir_node":ideal,
        "ideal_reservoir_status":"available" if ideal is not None else "not present in this replica graph",
        "transport_sampling_scope":"Segment-local saved diagnostic frames only. Restarts reset walker labels to slot indices, so segment round trips must not be summed as lineage totals. Visits between saves may be missed; counts are observed lower bounds, not exact continuous-history totals or equilibration certificates",
        "ideal_to_unconstrained_round_trips":round_trips(samples[:,:,2],ideal,endpoint) if ideal is not None else None,
        "ideal_to_inserted_round_trips":round_trips(samples[:,:,2],ideal,int(nodes[-1])) if ideal is not None else None,
        "cavity_nodes":nodes.tolist(),"exchange_edges":data["exchange_edges"].tolist(),
        "physical_N_support":np.unique(physical[:,0]).astype(int).tolist(),
        "inserted_N_support":np.unique(cavity[:,0]).astype(int).tolist(),
        "physical_empty_reference_sites_support":np.unique(physical[:,4]).astype(int).tolist(),
        "physical_excess_site_occupancy_support":np.unique(physical[:,5]).astype(int).tolist(),
        "N_cap_samples":int(samples[:,:,7].sum()),
        "minimum_neighbor_exchange_acceptance":float(acceptance.min()),
        "zero_acceptance_edges":np.where(swap[:,1]==0)[0].tolist(),
        "zero_attempt_edges":np.where(swap[:,0]==0)[0].tolist(),
        "block_sweeps":block_sweeps,"bootstrap_blocks":blocks,
        "bootstrap_block_samples":block_sweeps//stride,
        "bootstrap_draws":bootstrap,"bootstrap_seed":seed,
        "bootstrap_discarded_samples":len(samples) % (block_sweeps//stride),
        "bootstrap_status":"estimated" if blocks >= 4 else "not estimated: fewer than four complete time blocks",
        "bootstrap_missing_SE_fields":sorted(set(values)-set(errors)),
        "half_difference_N":float(physical[:len(samples)//2,0].mean()-physical[len(samples)//2:,0].mean()),
        "half_difference_q":float(np.mean(physical[:len(samples)//2,1]>.45)-np.mean(physical[len(samples)//2:,1]>.45)),
        "product_target":product_target_diagnostics(samples,endpoint,nodes,values,errors,block_sweeps//stride),
        "unbiased_hard_phase_bank":hard_phase_bank_diagnostics(data,block_sweeps//stride),
        "pair_overlap_profiles":pair_overlap_diagnostics(data),
        "support_probability_control":support_probability_diagnostics(data,values,errors,block_sweeps//stride),
        "boundary_fraction_uncertainty":boundary_fraction_diagnostics(values,errors),
        "independent_identity_checks":{
            "checks":{name:zero_residual_check(values[name],errors.get(name)) for name in
                      ("direct_minus_Palm_N","direct_minus_Palm_q","stage_minus_GNZ_work")},
            "interpretation":"Separate exact marginal-law controls with common-block uncertainty; agreement of one does not override failure of another, joint-inventory failure, inadequate visits or initialization disagreement; no overall equilibration certification",
        },
    }
    diagnostics["palm_transfer"] = {"counters_available":"palm_move_counts" in data}
    if "palm_move_counts" in data:
        attempts,accepted = map(int,data["palm_move_counts"])
        diagnostics["palm_transfer"].update({
            "attempts":attempts,"accepted":accepted,
            "acceptance_fraction":accepted/attempts if attempts else None,
            "interpretation":"Move acceptance is independent of product-target inventory adequacy. Accepted Palm transfers provide an additional endpoint connection even when ordinary direct cavity exchanges have zero observed acceptance; zero ordinary acceptance alone does not establish a disconnected augmented sampler",
        })
    if "probe_nodes" in data and len(data["probe_nodes"]):
        physical_nodes = np.r_[endpoint,data["probe_nodes"]]
        physical_nodes = physical_nodes[np.argsort(data["activity"][physical_nodes])]
        low,high = int(physical_nodes[0]),int(physical_nodes[-1])
        profiles = []
        for node in physical_nodes:
            series = samples[:,node]
            basin = series[:,1]>.45
            profiles.append({
                "node":int(node),"activity":float(data["activity"][node]),
                "mean_N":float(series[:,0].mean()),
                "N_support":np.unique(series[:,0]).astype(int).tolist(),
                "mean_FCC_Bragg_amplitude":float(series[:,1].mean()),
                "q_solid":float(basin.mean()),
                "phase_transitions_low_to_high_high_to_low":phase_transitions(series[:,1]),
                "basin_visits":{label:event_visits(event,block_sweeps//stride)
                                for label,event in (("fluid",~basin),("solid",basin))},
            })
        diagnostics["physical_activity_probes"] = {
            "profiles":profiles,"lowest_activity_node":low,"highest_activity_node":high,
            "lowest_activity_is_distinct_from_base":low != endpoint,
            "lowest_activity_to_unconstrained_round_trips":round_trips(samples[:,:,2],low,endpoint),
            "lowest_activity_to_inserted_round_trips":round_trips(samples[:,:,2],low,int(nodes[-1])),
            "lowest_to_highest_activity_round_trips":round_trips(samples[:,:,2],low,high),
            "coordinate_sample_indices":data["probe_sample_indices"].tolist(),
            "coordinates_available":"probe_configuration_shape" in data,
            "coordinate_shape":list(data["probe_configuration_shape"]) if "probe_configuration_shape" in data else None,
            "coordinate_alignment":"Indices select the saved diagnostic frames; statistics and inventory traces above use every diagnostic frame, not only thinned coordinate frames",
            "interpretation":"Transport within the unbiased hard-sphere activity branch is distinct from ideal-node transport; round trips and one-dimensional profiles alone do not establish joint inventory or phase weights",
        }
    return values,errors,diagnostics,data,metadata


def comparison_context(data, metadata):
    """Record the physical target separately from auxiliary graphs/proposals."""
    endpoint = int(data["endpoint"])
    sigma = float(data["diameter"][endpoint])
    volume = float(data["volume"])
    settings = metadata.get("settings",{})
    cap = settings.get("cap")
    # For a cubic periodic cell of side >= sigma, disjoint radius-sigma/2
    # balls imply N*pi*sigma^3/6 <= V, independent of crystal assumptions.
    required_cap = int(np.ceil(6*volume/(np.pi*sigma**3)))
    nontruncating = bool(cap is not None and cap>=required_cap and volume**(1/3)>=sigma)
    graph = {name:np.asarray(data[name]).tolist() for name in
             ("diameter","radius","activity","cavity_nodes","exchange_edges",
              "pair_penalty","umbrella_stiffness","umbrella_preferred",
              "number_stiffness","number_preferred") if name in data}
    return {
        "physical_target":{
            "x":float(data["activity"][endpoint]),"V":volume,"sigma":sigma,
            "R":float(data["radius"][data["cavity_nodes"][-1]]),
        },
        "phase_label":{
            "deletion_label_threshold":DELETION_LABEL_THRESHOLD,
            "reference_sites":settings.get("sites"),
        },
        "cap":cap,"geometric_nontruncating_cap_requirement":required_cap,
        "capacity_proven_nontruncating":nontruncating,
        "capacity_interpretation":"The geometric disjoint-ball bound certifies no physical truncation; auxiliary caps are irrelevant to the matched physical target. Missing capacity metadata is not a certificate",
        "auxiliary_graph_sha256":hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest(),
        "displacement_proposal_scales":data["displacement"].tolist() if "displacement" in data else None,
        "sampler_source_sha256":metadata.get("source_sha256"),
        "raw_sha256":metadata.get("raw_sha256"),
        "restart_sha256":metadata.get("restart_sha256"),"settings":settings,
    }


def compare_initializations(first_name, first, second_name, second):
    """Compare separate histories without pooling incompatible metastable data."""
    left,right = first["comparison_context"],second["comparison_context"]
    ls,rs = left["settings"],right["settings"]
    target_mismatch = [key for key in left["physical_target"]
                       if left["physical_target"][key] != right["physical_target"][key]]
    same_label = left["phase_label"] == right["phase_label"]
    capacity_ok = left["capacity_proven_nontruncating"] and right["capacity_proven_nontruncating"]
    eligible = not target_mismatch and capacity_ok
    independent = (ls.get("seed") is not None and rs.get("seed") is not None
                   and ls["seed"] != rs["seed"] and not ls.get("restart") and not rs.get("restart")
                   and left.get("raw_sha256") != right.get("raw_sha256"))
    changed_settings = {key:[ls.get(key),rs.get(key)] for key in sorted(ls.keys()|rs.keys())
                        if key not in ("output","overwrite") and ls.get(key)!=rs.get(key)}
    same_graph = left["auxiliary_graph_sha256"] == right["auxiliary_graph_sha256"]
    dl,dr = left.get("displacement_proposal_scales"),right.get("displacement_proposal_scales")
    scales_comparable = same_graph and dl is not None and dr is not None and len(dl) == len(dr)
    different_scales = ([node for node,(a,b) in enumerate(zip(dl,dr)) if a != b]
                        if scales_comparable else None)
    boundary = any(record["estimates"][field] in (0.,1.) for record in (first,second)
                   for field in ("q_solid_base","q_solid_inserted"))
    checks = {}
    if eligible:
        for field in ("mean_N","q_solid_base","Fano_factor","mean_N_inserted","q_solid_inserted"):
            if field.startswith("q_") and not same_label:
                continue
            se1,se2 = (record["block_bootstrap_SE"].get(field) for record in (first,second))
            se = (float(np.hypot(se1,se2)) if independent and se1 is not None and se2 is not None
                  else None)
            checks[field] = zero_residual_check(first["estimates"][field]-second["estimates"][field],se)
            checks[field]["boundary_only_uncertainty_present"] = bool(field.startswith("q_") and
                any(record["estimates"][field] in (0.,1.) for record in (first,second)))
    return {
        "run_order":[first_name,second_name],"difference_convention":"first minus second",
        "same_physical_target":not target_mismatch,"target_mismatch_fields":target_mismatch,
        "same_recorded_phase_label":same_label,
        "both_capacities_proven_nontruncating":capacity_ok,"comparison_eligible":eligible,
        "independent_fresh_random_histories_recorded":bool(independent),
        "combined_SE_status":"quadrature of independent within-history block SE where available" if independent else "not reported: distinct fresh random histories are not established by the recorded seeds/restarts/raw hashes",
        "different_initializations":ls.get("initialization")!=rs.get("initialization"),
        "different_auxiliary_graphs":not same_graph,
        "displacement_proposal_comparison":{
            "both_arrays_recorded":dl is not None and dr is not None,
            "corresponding_replica_targets_established":same_graph,
            "all_replica_scales_equal":not different_scales if scales_comparable else None,
            "different_scale_nodes":different_scales,
            "interpretation":"Compares the complete recorded per-replica displacement arrays, including physical activity probes; identical CLI settings do not establish identical tuned proposals. Missing arrays or different replica targets leave this comparison unresolved, without invalidating a separately verified common physical equilibrium target",
        },
        "different_sampler_sources":left.get("sampler_source_sha256")!=right.get("sampler_source_sha256"),
        "different_settings":changed_settings,
        "boundary_only_uncertainty_present":boundary,"observable_difference_checks":checks,
        "any_difference_exceeds_three_nominal_SE":any(c["exceeds_three_SE"] for c in checks.values()),
        "interpretation":"Histories remain separate and are never pooled. Combined SE is the quadrature sum of within-history block SE under independence, not protection against unvisited phases or common bias. Boundary fractions can have zero conditional bootstrap variance without physical precision. Agreement is not proof of equilibration; incompatible physical targets or uncertified capacities are not compared. Basin fractions additionally require matching recorded labels. Auxiliary graphs and proposal settings may differ without changing a verified physical target",
    }


def plot_audit(data, metadata, output):
    samples = data["samples"]
    endpoint = int(data["endpoint"])
    nodes = cavity_path(endpoint,samples.shape[1],data.get("cavity_nodes"))
    inserted = int(nodes[-1])
    time = (np.arange(len(samples))+1)*metadata["settings"]["sample_stride"]
    plt.rcParams.update({"font.size":8.5,"axes.labelsize":8.7,
                         "xtick.labelsize":7.5,"ytick.labelsize":7.5,
                         "pdf.fonttype":42})
    fig, axes = plt.subplots(2,3,figsize=(7.4,5.3),layout="constrained")
    axes[0,0].plot(time,samples[:,endpoint,0],lw=.7,label="Unconstrained")
    axes[0,0].plot(time,samples[:,inserted,0],lw=.7,alpha=.7,label=r"$R=\sigma$")
    axes[0,0].set(xlabel="Production sweep",ylabel="Particle number")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",
               ncol=2,fontsize=8.5,frameon=False)
    axes[0,1].plot(time,samples[:,endpoint,1],lw=.7)
    axes[0,1].plot(time,samples[:,inserted,1],lw=.7,alpha=.7)
    axes[0,1].axhline(.45,color=".5",ls="--",lw=.6)
    axes[0,1].set(xlabel="Production sweep",ylabel="Mean FCC Bragg amplitude")
    phase0 = (samples[:,endpoint,1]>.45).astype(int)
    phasec = (samples[:,inserted,1]>.45).astype(int)
    joint = np.array([[np.mean((phase0==i)&(phasec==j)) for j in range(2)] for i in range(2)])
    product = np.array([[np.mean(phase0==i)*np.mean(phasec==j) for j in range(2)] for i in range(2)])
    residual = joint-product
    scale = max(float(np.max(np.abs(residual))),1e-3)
    axes[0,2].imshow(residual,cmap="RdBu_r",vmin=-scale,vmax=scale,aspect="auto")
    for i in range(2):
        for j in range(2):
            axes[0,2].text(j,i,f"obs {joint[i,j]:.3g}\nprod {product[i,j]:.3g}\nΔ {residual[i,j]:+.2g}",
                           ha="center",va="center",fontsize=7.4,
                           color="white" if abs(residual[i,j]) > .6*scale else "black")
    axes[0,2].set(xticks=[0,1],xticklabels=["F basin","S basin"],
                  yticks=[0,1],yticklabels=["F basin","S basin"],
                  xlabel="Inserted endpoint",ylabel="Unconstrained endpoint")
    axes[0,2].set_title("Joint phase inventory",fontsize=8.3,pad=5)
    # Diameter is constant on a soft-penalty ladder and on the cavity leg;
    # Edge indices also resolve branched or cyclic exchange graphs.
    axes[1,0].plot(np.arange(len(data["swaps"])),data["swaps"][:,1]/np.maximum(data["swaps"][:,0],1),".",ms=3)
    axes[1,0].set(xlabel="Neighbor replica-pair index",ylabel="Neighbor exchange acceptance",ylim=(0,1))
    identities = samples[:,:,2].astype(int)
    for walker in range(0,identities.max()+1,max(1,(identities.max()+1)//8)):
        location = np.argmax(identities==walker,axis=1)
        axes[1,1].plot(time,location,lw=.6)
    axes[1,1].axhline(endpoint,color=".2",ls="--",lw=.6)
    axes[1,1].set(xlabel="Production sweep",ylabel="Replica index of selected walkers")
    ratios = samples[:,nodes[:-1],3].mean(axis=0)
    axes[1,2].plot(data["radius"][nodes[:-1]],ratios,"o-",ms=3,lw=.8)
    axes[1,2].set(xlabel=r"Cavity radius $R/\sigma$",
                  ylabel="Next-stage support probability" if len(nodes)>2 else "Direct support probability",
                  ylim=(0,1.05))
    if np.any(ratios==0):
        axes[1,2].text(.5,.65,"Zero observed support events\nWork unresolved",transform=axes[1,2].transAxes,
                       ha="center",va="center",fontsize=8)
    for panel,ax in zip("abcdef",axes.flat):
        ax.text(0.,1.03,f"({panel})",transform=ax.transAxes,va="bottom",fontsize=9,
                fontweight="bold",clip_on=False)
        ax.tick_params(labelsize=7.5)
    fig.savefig(output)
    plt.close(fig)


def write_analysis(inputs, output, block_sweeps=500, bootstrap=500, seed=DEFAULT_SEED,
                   figure_dir=None):
    inputs = [Path(path).resolve() for path in inputs]
    output = Path(output).resolve()
    if not inputs:
        raise ValueError("At least one input trajectory is required")
    if block_sweeps < 1 or bootstrap < 2 or seed < 0:
        raise ValueError("block sweeps must be positive, bootstrap draws >= 2, and seed >= 0")
    if output.suffix != ".json":
        raise ValueError("Analysis output must have the .json extension")
    if len({path.stem for path in inputs}) != len(inputs):
        raise ValueError("Input run names must be distinct; duplicate stems would overwrite results")
    canonical = output == DEFAULT_OUTPUT.resolve()
    figure_dir = Path(figure_dir).resolve() if figure_dir else (
        ROOT/"figures" if canonical else output.parent/f"{output.stem}_figures")
    metadata_path = (ROOT/"generated/continuum_switching_analysis_metadata.json" if canonical
                     else output.with_suffix(".metadata.json"))
    figures = [figure_dir/f"{path.stem}_audit.pdf" for path in inputs]
    input_files = [path for raw in inputs for path in (raw, raw.with_suffix(".json"))]
    if set(input_files).intersection([output, metadata_path, *figures]):
        raise ValueError("Analysis outputs must not overwrite raw data or execution metadata")
    # Preflight one run at a time; a large batch must not retain all raw arrays.
    for path in inputs:
        data,_ = load_validated_run(path,block_sweeps)
        del data
    result = {}
    output.parent.mkdir(parents=True,exist_ok=True)
    figure_dir.mkdir(parents=True,exist_ok=True)
    metadata_path.parent.mkdir(parents=True,exist_ok=True)
    for path,figure in zip(inputs,figures):
        values,errors,diagnostics,data,metadata = analyze(path,block_sweeps,bootstrap,seed)
        result[path.stem] = {"estimates":values,"block_bootstrap_SE":errors,
                            "mixing":diagnostics,"raw_file":display_path(path),
                            "comparison_context":comparison_context(data,metadata),
                            "independent_start_comparisons":{}}
        plot_audit(data,metadata,figure)
        # Complete traces remain in the JSON artifact, not repeated as an
        # unbounded terminal transcript for every sampled frame and replica.
        print(json.dumps(json_ready({
            "run":path.stem,"samples":len(data["samples"]),
            "mean_N":values["mean_N"],"q_solid_base":values["q_solid_base"],
            "q_solid_inserted":values["q_solid_inserted"],
            "work_stage":values["work_stage"],
            "bootstrap_blocks":diagnostics["bootstrap_blocks"],
            "full_diagnostics":display_path(output),
        }),allow_nan=False),flush=True)
    names = list(result)
    for i,first in enumerate(names):
        for second in names[i+1:]:
            comparison = compare_initializations(first,result[first],second,result[second])
            result[first]["independent_start_comparisons"][second] = comparison
            result[second]["independent_start_comparisons"][first] = comparison
    output.write_text(json.dumps(json_ready(result),indent=2,allow_nan=False)+"\n")
    command = ["python3", "scripts/continuum_analyze.py", *map(display_path,inputs),
               "--block-sweeps", str(block_sweeps), "--bootstrap", str(bootstrap),
               "--seed", str(seed), "--output", display_path(output),
               "--figure-dir", display_path(figure_dir)]
    record = {
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "script":"scripts/continuum_analyze.py","script_sha256":EXECUTION_SOURCE_SHA256,
        "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
        "matplotlib":matplotlib.__version__,
        "settings":{"block_sweeps":block_sweeps,"bootstrap_draws":bootstrap,"seed":seed,
                    "phase_threshold":.45,"phase_visit_thresholds":[.30,.65],
                    "joint_inventory_minimum_sampled_entries":4,
                    "joint_inventory_minimum_complete_blocks":4,
                    "covariance_tail_categories":"below/at/above modal particle number at each endpoint",
                    "covariance_tail_minimum_empirical_product_expected_count":5,
                    "product_residual_warning_standard_errors":3},
        "resampling":"Common time-block indices for all exchanged replicas and observables; independent raw runs are reported separately",
        "input_sha256":{display_path(path):sha256(path) for path in input_files},
        "output_sha256":{display_path(path):sha256(path) for path in (output,*figures)},
        "reproduce_command":shlex.join(command),
    }
    metadata_path.write_text(json.dumps(record,indent=2)+"\n")
    return result,metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",nargs="+",type=Path)
    parser.add_argument("--block-sweeps",type=int,default=500)
    parser.add_argument("--bootstrap",type=int,default=500)
    parser.add_argument("--seed",type=int,default=DEFAULT_SEED)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir",type=Path)
    args = parser.parse_args()
    try:
        write_analysis(args.inputs,args.output,args.block_sweeps,args.bootstrap,args.seed,args.figure_dir)
    except (ValueError,KeyError,FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
