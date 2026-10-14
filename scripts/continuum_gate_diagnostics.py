#!/usr/bin/env python3
"""Saved-frame phase/number-conditioned transport, never a phase-weight fit.

An observed excursion leaves the unbiased physical-node set and either
reaches a low-penalty reservoir before returning, or returns first.  Sparse
sampling can miss crossings, so this diagnoses conductance only at the
recorded cadence.  It does not turn high exchange acceptance into an
equilibration certificate or pool thermodynamic estimates across restarts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

try:
    from . import continuum_lineage_transport as lineage
    from .continuum_lineage_transport import load_segment, stitch_segments
except ImportError:
    import continuum_lineage_transport as lineage
    from continuum_lineage_transport import load_segment, stitch_segments

analysis = lineage.analysis
EXECUTION_SOURCE_SHA256 = analysis.sha256(Path(__file__))


def structural_state(n,b):
    if n >= 20 and b >= .65:
        return "high_Bragg"
    if n >= 20 and b <= .3:
        return "low_Bragg"
    return "intermediate_or_dilute"


def passage_diagnostics(ids,n,b,physical,reservoir,source_groups,stride):
    """Condition completed/censored physical departures on their last state."""
    if (ids.ndim != 2 or n.shape != ids.shape or b.shape != ids.shape
            or len(ids) == 0 or stride <= 0
            or not np.all(np.sort(ids,axis=1)==np.arange(ids.shape[1]))):
        raise ValueError("Trace rows must be nonempty equal-shaped walker permutations")
    frames,replicas = ids.shape
    if np.shape(physical) != (replicas,) or np.shape(reservoir) != (replicas,):
        raise ValueError("Node masks must match the replica count")
    if np.any(physical & reservoir):
        raise ValueError("Reservoir and physical masks must be disjoint")
    locations = np.argsort(ids,axis=1)
    by_walker_n = np.take_along_axis(n,locations,axis=1)
    by_walker_b = np.take_along_axis(b,locations,axis=1)
    pending = [None]*replicas
    last_source = [None]*replicas
    last_physical = [None]*replicas
    excursions = {}
    arrivals = {}
    conversions = {}
    sources = np.full(replicas,"",dtype=object)
    for name,nodes in source_groups.items():
        sources[nodes] = name
    for time in range(frames):
        for walker in range(replicas):
            node = locations[time,walker]
            label = structural_state(by_walker_n[time,walker],by_walker_b[time,walker])
            if sources[node]:
                last_source[walker] = (sources[node],label,time)
            if time and physical[locations[time-1,walker]] and not physical[node]:
                old_n = int(by_walker_n[time-1,walker])
                old_label = structural_state(old_n,by_walker_b[time-1,walker])
                key = f"{old_label}:N{old_n}"
                row = excursions.setdefault(key,{"departures":0,"reservoir_before_return":0,
                    "physical_return_first":0,"censored":0,"successful_transit_sweeps":[]})
                row["departures"] += 1
                pending[walker] = (key,time-1)
            active = pending[walker]
            if active is not None:
                key,start = active
                if reservoir[node]:
                    excursions[key]["reservoir_before_return"] += 1
                    excursions[key]["successful_transit_sweeps"].append((time-start)*stride)
                    pending[walker] = None
                elif physical[node]:
                    excursions[key]["physical_return_first"] += 1
                    pending[walker] = None
            if physical[node]:
                entering = time == 0 or not physical[locations[time-1,walker]]
                if entering and last_source[walker] is not None:
                    source,source_label,source_time = last_source[walker]
                    key = f"{source}:{source_label}->{label}"
                    arrivals[key] = arrivals.get(key,0)+1
                    previous = last_physical[walker]
                    if previous in ("high_Bragg","low_Bragg") and label in ("high_Bragg","low_Bragg") and previous != label:
                        conversion = f"{source}:{previous}->{label}"
                        conversions[conversion] = conversions.get(conversion,0)+1
                    last_source[walker] = None
                last_physical[walker] = label
    for item in pending:
        if item is not None:
            excursions[item[0]]["censored"] += 1
    for row in excursions.values():
        durations = row.pop("successful_transit_sweeps")
        row["successful_transit_median_sweeps"] = float(np.median(durations)) if durations else None
        row["success_fraction_per_observed_departure"] = row["reservoir_before_return"]/row["departures"]
    return {"physical_departure_excursions":excursions,
            "last_recorded_source_to_physical_arrivals":arrivals,
            "observed_changed_physical_Bragg_stratum_after_source_visit":conversions}


def hypothetical_swap_acceptance(target,n,b,p,min_distance,left,right):
    """Cross-target ratio at saved equal-time states, not actual pre-edge draws."""
    if not np.all(target["diameter"] == target["diameter"][0]):
        raise ValueError("Overlap diagnostics require one common host diameter")
    def cost(node,count,bragg,overlap,distance):
        penalty = target["pair_penalty"][node]
        support = (distance >= target["radius"][node])
        if np.isposinf(penalty):
            support &= overlap == 0
            pair = np.zeros_like(overlap,dtype=float)
        else:
            pair = penalty*overlap
        energy = (pair+target["umbrella_stiffness"][node]
                  *(bragg-target["umbrella_preferred"][node])**2
                  +target["number_stiffness"][node]
                  *(count-target["number_preferred"][node])**2)
        return support,target["activity"][node]*count-energy
    own_left,old_left = cost(left,n[:,left],b[:,left],p[:,left],min_distance[:,left])
    own_right,old_right = cost(right,n[:,right],b[:,right],p[:,right],min_distance[:,right])
    if not np.all(own_left & own_right):
        raise ValueError("Recorded configuration violates its own hard support")
    cross_left,new_left = cost(left,n[:,right],b[:,right],p[:,right],min_distance[:,right])
    cross_right,new_right = cost(right,n[:,left],b[:,left],p[:,left],min_distance[:,left])
    valid = cross_left & cross_right
    probabilities = np.zeros(len(n))
    probabilities[valid] = np.exp(np.minimum(0.,(new_left+new_right-old_left-old_right)[valid]))
    return probabilities,valid


def swap_gate_diagnostics(target,n,b,p,min_distance):
    """Descriptive saved-state support/cost overlap in exact N/Bragg strata."""
    result = []
    for left,right in target["exchange_edges"]:
        probabilities,support = hypothetical_swap_acceptance(target,n,b,p,min_distance,left,right)
        strata = {}
        labels = [f"{structural_state(n[k,left],b[k,left])}:N{int(n[k,left])}|"
                  f"{structural_state(n[k,right],b[k,right])}:N{int(n[k,right])}"
                  for k in range(len(n))]
        for label in set(labels):
            selected = np.array([item == label for item in labels])
            strata[label] = {"saved_pairs":int(selected.sum()),
                "both_cross_support_fraction":float(np.mean(support[selected])),
                "mean_hypothetical_acceptance":float(np.mean(probabilities[selected]))}
        result.append({"edge":[int(left),int(right)],"saved_pairs":len(n),
            "mean_hypothetical_acceptance":float(np.mean(probabilities)),
            "both_cross_support_fraction":float(np.mean(support)),"strata":strata})
    return result


def analyze_lineage(paths):
    segments = [load_segment(path) for path in paths]
    linkage = stitch_segments(segments)
    canonical_ids,n_values,b_values,p_values,d_values = [],[],[],[],[]
    target = None
    for segment,record in zip(segments,linkage["segments"]):
        canonical_ids.append(np.asarray(record["local_to_lineage_walker_id"])[segment["ids"]])
        with np.load(segment["path"],allow_pickle=False) as archive:
            samples = archive["samples"]
            n_values.append(samples[:,:,0].copy())
            b_values.append(samples[:,:,1].copy())
            p_values.append(archive["pair_overlap_counts"].copy())
            d_values.append(samples[:,:,6].copy())
            if target is None:
                target = {key:archive[key] for key in ("diameter","radius","pair_penalty",
                    "umbrella_stiffness","umbrella_preferred","number_stiffness","number_preferred",
                    "endpoint","probe_nodes","cavity_nodes","activity","exchange_edges")}
    ids,n,b = map(np.concatenate,(canonical_ids,n_values,b_values))
    count = ids.shape[1]
    physical = np.zeros(count,dtype=bool)
    physical[np.r_[int(target["endpoint"]),target["probe_nodes"],target["cavity_nodes"]]] = True
    reservoir = ((target["pair_penalty"]<=2)&(target["umbrella_stiffness"]==0)
                 &(target["number_stiffness"]==0)&(target["radius"]==0))
    high = np.flatnonzero((target["umbrella_stiffness"]>=40)
                           &(target["umbrella_preferred"]>=.7))
    low = np.flatnonzero((target["umbrella_stiffness"]>=40)
                          &(target["umbrella_preferred"]<=.3)&(target["number_stiffness"]>0))
    result = passage_diagnostics(ids,n,b,physical,reservoir,
        {"high_Bragg_bridge":high,"low_Bragg_number_bridge":low,
         "unbiased_soft_reservoir":np.flatnonzero(reservoir)},segments[0]["settings"]["sample_stride"])
    return {"scope":"Saved-frame conditional transport only; no pooled thermodynamic means or uncertainty estimates",
            "qualification":"Unrecorded source visits and boundary crossings can be missed. Transits are conditioned on observed departures, not unconditional mean first-passage times. Walker labels follow hosts, not fixed particle cohorts. Auxiliary high/low Bragg strata are not thermodynamic solid/fluid classifications. Hypothetical equal-time swap probabilities are descriptive support/cost overlaps, not observed pre-edge conditional acceptance rates.",
            "inputs":[segment["path"] for segment in segments],
            "raw_sha256":[segment["raw_sha256"] for segment in segments],
            "transport_scope":linkage["sampling_scope"],"saved_frames":len(ids),
            "physical_nodes":np.flatnonzero(physical).tolist(),
            "reservoir_nodes":np.flatnonzero(reservoir).tolist(),
            "hypothetical_saved_state_swap_gates":swap_gate_diagnostics(target,n,b,
                np.concatenate(p_values),np.concatenate(d_values)),**result}


def write_diagnostics(inputs,output):
    inputs = [Path(path).resolve() for path in inputs]
    output = Path(output).resolve()
    metadata_path = output.with_suffix('.metadata.json')
    protected = {file for path in inputs for file in (path,path.with_suffix('.json'))}
    if output.suffix != '.json' or {output,metadata_path}.intersection(protected):
        raise ValueError("Output must be a new JSON artifact, not raw data or its sidecar")
    result = analyze_lineage(inputs)
    source_files = {Path(json.loads(path.with_suffix('.json').read_text())['source_snapshot'])
                    for path in inputs}
    source_files = {(lineage.ROOT/path).resolve() for path in source_files}
    if {output,metadata_path}.intersection(source_files):
        raise ValueError("Output may not overwrite executed-source snapshots")
    files = sorted(protected | source_files)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    metadata_path.write_text(json.dumps({
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'script':'scripts/continuum_gate_diagnostics.py',
        'script_sha256':EXECUTION_SOURCE_SHA256,'python':sys.version,
        'dependency_sha256':{
            'scripts/continuum_lineage_transport.py':lineage.EXECUTION_SOURCE_SHA256,
            'scripts/continuum_analyze.py':analysis.EXECUTION_SOURCE_SHA256},
        'input_sha256':{analysis.display_path(path):analysis.sha256(path) for path in files},
        'output_sha256':{analysis.display_path(output):analysis.sha256(output)},
        'reproduce_argv':['python3','-m','scripts.continuum_gate_diagnostics',
            *map(analysis.display_path,inputs),'--output',analysis.display_path(output)],
    },indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",nargs="+",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    try:
        result = write_diagnostics(args.inputs,args.output) if args.output else analyze_lineage(args.inputs)
    except (ValueError,KeyError,FileNotFoundError) as error:
        parser.error(str(error))
    if not args.output:
        print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
