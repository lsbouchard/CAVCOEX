#!/usr/bin/env python3
"""Fixed-clock conditional estimates and complete-regeneration-cycle errors.

This module is for one-host simulated tempering, not simultaneous product
replicas. Rejected proposals and dwell steps are observations. Conditional
means are ratios of total cycle rewards to total cycle dwell, never averages
of the individual cycle ratios. A fixed-clock stopping time selects a random
prefix of complete cycles; reported cycle bootstrap errors are asymptotic,
not an exact finite-sample unbiasedness or equilibration guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'cavcoex.single_host_regenerative_tempering.v1'
DELETION_THRESHOLD = .45
COLUMNS = ('clock_step','label','N','fcc_bragg_mean','pair_overlap_count',
           'empty_reference_sites','excess_site_occupancy','minimum_center_distance',
           'sum_delete_one_solid_label_change','nearest_FCC_registry_rms_over_spacing',
           'regeneration_cycle_id','ideal_heatbath_event')
REWARD_COLUMNS = (
    'base_dwell','cavity_dwell','base_N','base_N2','cavity_N',
    'base_solid','cavity_solid','base_N_solid','base_deletion_shift',
    'base_absolute_deletion_shift',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


EXECUTION_SOURCE_SHA256 = sha256(Path(__file__))


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def json_ready(value):
    if isinstance(value,dict):
        return {key:json_ready(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):
        return None
    return value


def array_shape(path,name):
    with zipfile.ZipFile(path) as archive:
        with archive.open(name+'.npy') as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1,0):
                return np.lib.format.read_array_header_1_0(handle)[0]
            if version == (2,0):
                return np.lib.format.read_array_header_2_0(handle)[0]
            raise ValueError('Unsupported archived NumPy header')


def scalar_integer(value, name, minimum=0):
    array = np.asarray(value)
    if (array.shape != () or not np.isfinite(array) or array != np.rint(array)
            or array < minimum):
        raise ValueError(name+' must be a finite integer scalar at least '+str(minimum))
    return int(array)


def validate_label_update_protocol(data, metadata):
    """Bind compound label clocks to their intermediate labels and counters.

    The stored samples precede both label updates. A next-row label difference
    is therefore not a neighbor-edge transition when a Gibbs draw participates.
    """
    protocol = metadata.get('label_update_protocol','lazy_neighbor_tagged')
    if protocol not in ('lazy_neighbor_tagged','gibbs_then_lazy_neighbor_tagged'):
        raise ValueError('Unrecognized single-host label-update protocol')
    fields = {'neighbor_source_labels','gibbs_attempt_change_counts'}
    present = fields.intersection(data)
    compound = protocol == 'gibbs_then_lazy_neighbor_tagged'
    declared = metadata.get('settings',{}).get('gibbs_label')
    if declared is not None and (not isinstance(declared,bool) or declared != compound):
        raise ValueError('Gibbs setting disagrees with the recorded label-update protocol')
    if (compound and present != fields) or (present and present != fields):
        raise ValueError('Gibbs clocks require aligned pre-neighbor labels and separate counters')
    if not present:
        return protocol
    samples = data['samples']
    source = np.asarray(data['neighbor_source_labels'])
    counts = np.asarray(data['gibbs_attempt_change_counts'])
    if (source.shape != (len(samples),) or not np.isfinite(source).all()
            or np.any(source != np.rint(source)) or np.any(source < 0)
            or np.any(source >= len(data['activity']))
            or counts.shape != (2,) or not np.isfinite(counts).all()
            or np.any(counts != np.rint(counts)) or np.any(counts < 0)
            or counts[1] > counts[0] or counts[0] > len(samples)):
        raise ValueError('Invalid Gibbs intermediate labels or production counters')
    changed = int(np.count_nonzero(source != samples[:,1]))
    if counts[1] != changed:
        raise ValueError('Gibbs label-change counter disagrees with the intermediate trace')
    if compound and counts[0] != len(samples):
        raise ValueError('Enabled Gibbs protocol must refresh the label at every production clock')
    if not compound and (np.any(counts) or changed):
        raise ValueError('Neighbor-only protocol cannot contain Gibbs label updates')
    source = source.astype(int)
    # Gibbs leaves the host, center and number fixed. The intermediate target
    # must support exactly the recorded configuration; all current kernels use
    # a common pair diameter, so the saved overlap count is reusable here.
    if not np.all(data['diameter'] == data['diameter'][0]):
        raise ValueError('Gibbs support replay requires a common pair diameter')
    if (np.any(samples[np.isposinf(data['pair_penalty'][source]),4] != 0)
            or np.any((samples[:,2] > 0)&(samples[:,7] < data['radius'][source]))):
        raise ValueError('A post-Gibbs target violates the recorded host support')
    return protocol


def validate_calibration_record(data, metadata):
    """Check discarded adaptation bookkeeping, without certifying equilibration."""
    nodes,cutoff = len(data['activity']),int(metadata['calibration_cutoff_clock'])
    settings = metadata.get('settings',{})
    method = metadata.get('calibration_method')
    if method is None:
        if 'calibration_outcome' in metadata or 'calibration_method' in settings or 'calibration_weight_update_visits' in data:
            raise ValueError('New calibration bookkeeping requires its declared method')
        return 'legacy_one_over_t'
    if method not in ('one_over_t','staged_wang_landau') or settings.get('calibration_method',method) != method:
        raise ValueError('Unrecognized or inconsistent calibration method')
    outcome = metadata.get('calibration_outcome')
    if not isinstance(outcome,dict) or outcome.get('method') != method:
        raise ValueError('Declared calibration method requires matching outcome statistics')
    actual = scalar_integer(outcome.get('actual_clock_steps',-1),'Actual calibration steps')
    budget = scalar_integer(outcome.get('requested_budget',-1),'Calibration budget')
    if (actual != cutoff or actual > budget or settings.get('calibration_steps',budget) != budget
            or any(not isinstance(outcome.get(key),bool) for key in ('completed','budget_exhausted'))
            or (not outcome['completed'] and actual < budget)
            or outcome['budget_exhausted'] != (not outcome['completed'] and actual >= budget)):
        raise ValueError('Calibration cutoff, budget and outcome flags disagree')

    def counts(value,total,label):
        a = np.asarray(value,dtype=float)
        if (a.shape != (nodes,) or not np.isfinite(a).all() or np.any(a < 0)
                or np.any(a != np.rint(a)) or a.sum() != total):
            raise ValueError('Invalid '+label+' calibration counts')
        return a

    visits = counts(data.get('calibration_visits',[]),actual,'pre-hop observation')
    gain = settings.get('calibration_gain',0.)
    if not np.isfinite(gain) or gain < 0:
        raise ValueError('Calibration gain must be finite and nonnegative')
    expected_updates = actual if method == 'staged_wang_landau' or gain > 0 else 0
    updated = counts(data.get('calibration_weight_update_visits',[]),expected_updates,'weight-update')
    history = outcome.get('stage_history')
    if not isinstance(history,list):
        raise ValueError('Calibration stage history must be explicitly recorded')
    if method == 'one_over_t':
        if history or actual != budget or not outcome['completed']:
            raise ValueError('One-over-t schedule cannot claim staged or incomplete calibration')
        return method

    criteria = outcome.get('criteria',{})
    try:
        flat = float(criteria['flatness_minimum_to_mean'])
        minimum = scalar_integer(criteria['minimum_weight_update_visits_per_label'],'Minimum visits',1)
        half_minimum = scalar_integer(criteria['minimum_observations_per_dense_label_per_half'],'Minimum half visits',1)
        n_tol,b_tol = (float(criteria[key]) for key in ('maximum_half_mean_N_difference','maximum_half_mean_Bragg_difference'))
        dense = np.asarray(criteria['dense_diagnostic_nodes'])
        initial,minimum_gain,final_gain = (float(outcome[key]) for key in ('initial_gain','minimum_gain','final_gain'))
    except (KeyError,TypeError) as error:
        raise ValueError('Staged calibration criteria and gains are incomplete') from error
    if (not np.isfinite([flat,n_tol,b_tol,initial,minimum_gain,final_gain]).all()
            or not 0 < flat <= 1 or min(n_tol,b_tol) < 0 or not 0 < minimum_gain <= initial
            or dense.ndim != 1 or not len(dense) or dense.dtype.kind not in 'iu'
            or len(np.unique(dense)) != len(dense) or np.any(dense < 0) or np.any(dense >= nodes)):
        raise ValueError('Invalid staged calibration criteria')
    previous = None
    stage_start = 0
    completed_counts = np.zeros(nodes)
    completed_dense = np.zeros(len(dense),dtype=int)
    completed_heatbaths = 0
    expected_gain = initial
    for row in history:
        stage = scalar_integer(row.get('stage',-1),'Calibration stage')
        end = scalar_integer(row.get('end_clock',-1),'Calibration stage clock',1)
        epoch = scalar_integer(row.get('epoch_steps',-1),'Calibration epoch steps',1)
        first = scalar_integer(row.get('first_half_steps',-1),'First half steps')
        second = scalar_integer(row.get('second_half_steps',-1),'Second half steps')
        if previous is None:
            if stage != 0:raise ValueError('Calibration history must start at stage zero')
        elif stage != previous['stage']:
            if stage != previous['stage']+1 or not previous['stage_passed']:
                raise ValueError('Calibration stage advances only after passing recorded criteria')
            completed_counts += np.asarray(previous['weight_update_visits'])
            completed_dense += np.array([sum(item[half]['observations'] for half in ('first_half','second_half'))
                                         for item in previous['dense_half_profiles']])
            completed_heatbaths += previous['ideal_heatbath_events']
            stage_start = previous['end_clock']
            expected_gain = max(minimum_gain,expected_gain/2)
        elif previous['stage_passed']:
            raise ValueError('A passed calibration stage cannot receive additional observations')
        if (end > cutoff or end <= (previous['end_clock'] if previous else 0)
                or epoch != end-stage_start or first+second != epoch
                or row.get('gain') != expected_gain):
            raise ValueError('Calibration stage clock, epoch halves or gain are inconsistent')
        tally = counts(row.get('weight_update_visits',[]),epoch,'stage weight-update')
        if previous is not None and stage == previous['stage'] and np.any(tally < np.asarray(previous['weight_update_visits'])):
            raise ValueError('Cumulative stage weight-update counts cannot decrease')
        ratio = float(tally.min()/tally.mean())
        min_met = bool(tally.min() >= minimum)
        flat_met = bool(min_met and ratio >= flat)
        profiles = row.get('dense_half_profiles',[])
        if [item.get('node') for item in profiles] != dense.tolist():
            raise ValueError('Every dense calibration node requires its separate half-epoch profile')
        stable = first > 0
        half_observations = np.zeros(2,dtype=int)
        dense_totals = []
        for profile in profiles:
            halves = [profile.get(key,{}) for key in ('first_half','second_half')]
            sizes = []
            for half,steps in zip(halves,(first,second)):
                size = scalar_integer(half.get('observations',-1),'Dense half observations')
                sizes.append(size)
                if size > steps:raise ValueError('Dense half observations exceed the recorded half length')
                values = [half.get('mean_N'),half.get('mean_Bragg')]
                if size and (None in values or not np.isfinite(values).all()
                             or not 0 <= values[0] <= int(data['cap']) or not 0 <= values[1] <= 1+1e-12):
                    raise ValueError('Invalid paired dense number/order half statistics')
                if not size and values != [None,None]:
                    raise ValueError('Unvisited dense halves cannot have invented means')
            half_observations += sizes
            dense_totals.append(sum(sizes))
            sufficient = min(sizes) >= half_minimum
            dn = abs(halves[0]['mean_N']-halves[1]['mean_N']) if sufficient else None
            db = abs(halves[0]['mean_Bragg']-halves[1]['mean_Bragg']) if sufficient else None
            passed = bool(sufficient and dn <= n_tol and db <= b_tol)
            if (profile.get('sufficient_half_observations') != sufficient
                    or profile.get('mean_N_difference') != dn or profile.get('mean_Bragg_difference') != db
                    or profile.get('passed') != passed):
                raise ValueError('Dense half-stage flags disagree with recorded N/order moments')
            stable &= passed
        if np.any(half_observations > np.array([first,second])):
            raise ValueError('Dense-label half counts cannot exceed the total observation clock')
        stage_passed = bool(flat_met and stable)
        if (row.get('flatness_minimum_to_mean') != ratio or row.get('minimum_visits_met') != min_met
                or row.get('flatness_passed') != flat_met or row.get('dense_half_stability_passed') != stable
                or row.get('stage_passed') != stage_passed):
            raise ValueError('Staged-calibration pass flags disagree with recorded criteria')
        heatbaths = scalar_integer(row.get('ideal_heatbath_events',-1),'Calibration heatbath events')
        weights = np.asarray(row.get('log_weights',[]),dtype=float)
        if heatbaths > epoch or weights.shape != (nodes,) or not np.isfinite(weights).all():
            raise ValueError('Invalid staged heatbath count or weight vector')
        if previous is not None and stage == previous['stage']:
            previous_totals = [sum(item[half]['observations'] for half in ('first_half','second_half'))
                               for item in previous['dense_half_profiles']]
            if heatbaths < previous['ideal_heatbath_events'] or np.any(np.asarray(dense_totals) < previous_totals):
                raise ValueError('Cumulative stage heatbath and dense-observation counts cannot decrease')
        previous = row
    if history:
        completed_counts += np.asarray(history[-1]['weight_update_visits'])
        completed_dense += np.asarray(dense_totals)
        completed_heatbaths += history[-1]['ideal_heatbath_events']
        should_complete = history[-1]['stage_passed'] and expected_gain <= minimum_gain
        next_gain = expected_gain if should_complete or not history[-1]['stage_passed'] else max(minimum_gain,expected_gain/2)
        weights = np.asarray(history[-1]['log_weights'])
        if (history[-1]['end_clock'] != cutoff or outcome['completed'] != should_complete
                or final_gain != next_gain or not np.array_equal(completed_counts,updated)
                or not np.array_equal(completed_dense,visits[dense])
                or completed_heatbaths != visits[int(data['ideal_node'])]
                or not np.allclose(weights-weights[int(data['ideal_node'])],data['log_weights'],rtol=0,atol=1e-10)):
            raise ValueError('Final staged outcome does not match aggregate updates and frozen weights')
    elif cutoff or outcome['completed'] or final_gain != initial:
        raise ValueError('Empty staged history cannot assert completed adaptation')
    return method


def validate_radius_path(data):
    """Validate an optional ordinary-radius path distinct from the tagged edge."""
    if 'radius_path_nodes' not in data:
        return
    path = np.asarray(data['radius_path_nodes'])
    nodes = len(data['activity'])
    if (path.ndim != 1 or not 3 <= len(path) <= 257 or not np.isfinite(path).all()
            or np.any(path != np.rint(path)) or np.any(path < 0) or np.any(path >= nodes)
            or len(np.unique(path)) != len(path)
            or path[0] != data['base_node'] or path[-1] != data['cavity_node']
            or not set(path) <= set(data['physical_nodes'])):
        raise ValueError('Ordinary cavity-radius path must join distinct physical labels from base to cavity')
    path = path.astype(int)
    if (np.any(np.diff(data['radius'][path]) <= 0)
            or np.any(data['activity'][path] != data['activity'][path[0]])):
        raise ValueError('Radius path must increase at fixed physical chemical activity')
    edges = {tuple(sorted(map(int,edge))) for edge in data['exchange_edges']}
    if any(tuple(sorted(map(int,pair))) not in edges for pair in zip(path[:-1],path[1:])):
        raise ValueError('Every consecutive radius-path pair requires its own ordinary edge')


def reconstruct_explicit_native_graph(settings):
    """Independent lightweight replay of an explicitly specified native graph.

    This does not import an executing sampler or treat CS efficiency guesses
    as measured phase odds. The graph's targets are unbiased hard/ideal laws.
    """
    required = {'volume','sites','cap','sigma','base_field','displacement','cavity_radius_stages'}
    if not required <= settings.keys():
        raise ValueError('Explicit native construction requires every scalar box/graph setting')
    volume,sigma,field,step = (float(settings[key]) for key in ('volume','sigma','base_field','displacement'))
    sites = scalar_integer(settings['sites'],'Native reference size',4)
    cap = scalar_integer(settings['cap'],'Native capacity',1)
    stages = scalar_integer(settings['cavity_radius_stages'],'Native radius stages',1)
    if (not np.isfinite([volume,sigma,field,step]).all() or volume <= 0 or sigma != 1
            or not -2 <= field <= 17 or step <= 0 or stages > 256 or cap > 100000):
        raise ValueError('Explicit native graph settings exceed their physical/resource bounds')
    if volume < 1 or volume > cap*np.pi/6:
        raise ValueError('Explicit native geometry must contain a diameter and have an inactive physical cap')

    def fluid_guess(x):
        lo,hi = 1e-12,.8
        for _ in range(100):
            eta = (lo+hi)*.5
            chemical = math.log(6*eta/np.pi)+(8*eta-9*eta**2+3*eta**3)/(1-eta)**3
            if chemical < x:lo = eta
            else:hi = eta
        eta = (lo+hi)*.5
        return 6*eta/np.pi,(1+eta+eta**2-eta**3)/(1-eta)**3

    hard = np.unique(np.r_[np.arange(-2.,8.001,1.),np.arange(8.5,12.001,.5),
                           np.arange(12.25,17.001,.25),field])
    activity = np.r_[-2.,hard,field]
    base = 1+int(np.flatnonzero(hard==field)[0])
    cavity,nodes = len(activity)-1,len(activity)
    radius = np.zeros(nodes)
    radius[cavity] = sigma
    penalty = np.full(nodes,np.inf)
    penalty[0] = 0.
    edges = np.r_[np.column_stack((np.arange(len(hard)),np.arange(1,len(hard)+1))),[[base,cavity]]].astype(int)
    terms = np.array([n*(-2.+math.log(volume))-math.lgamma(n+1) for n in range(cap+1)])
    log_ideal = float(terms.max()+math.log(np.exp(terms-terms.max()).sum()))
    weights = np.empty(nodes)
    weights[0] = -log_ideal
    for j,x in enumerate(hard,1):
        rho,z = fluid_guess(x)
        weights[j] = -rho*z*volume
    rho,z = fluid_guess(field)
    weights[cavity] = weights[base]+field-math.log(rho)
    weights -= weights[0]
    expected = {'volume':volume,'sites':sites,'cap':cap,'diameter':np.full(nodes,sigma),
        'radius':radius,'activity':activity,'pair_penalty':penalty,'umbrella_stiffness':np.zeros(nodes),
        'umbrella_preferred':np.zeros(nodes),'number_stiffness':np.zeros(nodes),'number_preferred':np.zeros(nodes),
        'displacement':np.full(nodes,step),'exchange_edges':edges,'base_node':base,'cavity_node':cavity,
        'ideal_node':0,'physical_nodes':np.arange(1,nodes)}
    if stages > 1:
        radii = sigma*(1-(1-np.arange(1,stages)/stages)**2)
        for key in ('diameter','radius','activity','pair_penalty','umbrella_stiffness','umbrella_preferred',
                    'number_stiffness','number_preferred','displacement'):
            expected[key] = np.r_[expected[key],radii if key=='radius' else np.full(len(radii),expected[key][base])]
        path = np.r_[base,np.arange(nodes,nodes+len(radii)),cavity].astype(int)
        expected['radius_path_nodes'] = path
        expected['exchange_edges'] = np.r_[edges,np.column_stack((path[:-1],path[1:]))]
        expected['physical_nodes'] = np.arange(1,nodes+len(radii))
        eta = rho*np.pi/6
        work = []
        for r in radii/sigma:
            if r <= .5:
                value = -math.log1p(-8*eta*r**3)
            else:
                t = 2*r-1
                value = ((2*t**3-3*t**2+1)*(-math.log1p(-eta))
                         +(t**3-2*t**2+t)*(3*eta/(1-eta))
                         +(-2*t**3+3*t**2)*(field-math.log(rho))
                         +(t**3-t**2)*(3*(z-1)))
            work.append(weights[base]+value)
        weights = np.r_[weights,work]
    return expected,weights


def verify_imported_targets(data, metadata, verified_inputs):
    """Bind the recorded target/proposal arrays to the actual imported files."""
    common = ('volume','diameter','radius','activity','pair_penalty',
              'umbrella_stiffness','umbrella_preferred','number_stiffness',
              'number_preferred','displacement','exchange_edges')
    endpoint_names = ('base_node','cavity_node','ideal_node','physical_nodes')
    for setting in ('graph_from','weights_from'):
        name = metadata.get('settings',{}).get(setting)
        if not name:
            continue
        path = (ROOT/name).resolve()
        record = json.loads(path.with_suffix('.json').read_text())
        if record.get('raw_sha256') != sha256(path):
            raise ValueError('Imported '+setting+' raw/sidecar digest mismatch')
        source_records = dict(record.get('source_snapshots',{}))
        if 'source_snapshot' in record:
            source_records['historical_driver'] = [record['source_snapshot'],record['source_sha256']]
        for relative,digest in source_records.values():
            source = (ROOT/relative).resolve()
            if not source.is_relative_to(ROOT) or sha256(source) != digest:
                raise ValueError('Imported '+setting+' source snapshot digest mismatch')
            verified_inputs.add(source)
        for relative,digest in record.get('input_sha256',{}).items():
            source = (ROOT/relative).resolve()
            if sha256(source) != digest:
                raise ValueError('Imported '+setting+' parameter-origin digest mismatch')
            verified_inputs.add(source)
        native = record.get('schema') == 'cavcoex.native_tempering_graph.v1'
        single = record.get('schema') == SCHEMA
        required_sources = ({'generator','kernel','graph_validation'} if native else
                            {'driver','kernel','graph_validation'} if single else set())
        if not required_sources <= source_records.keys():
            raise ValueError('Imported '+setting+' is missing executed-source provenance')
        if setting == 'weights_from' and (not single or record.get('weights_frozen_before_production') is not True):
            raise ValueError('Imported weights must come from a verified frozen single-host archive')
        with np.load(path,allow_pickle=False) as archive:
            if (native or single) and str(archive['schema']) != record['schema']:
                raise ValueError('Imported raw schema differs from its sidecar')
            if native and 'samples' in archive:
                raise ValueError('A native graph is not a fabricated trajectory archive')
            if native:
                mode = record.get('construction_mode','validated_parameter_archive')
                origins = record.get('input_sha256',{})
                if not isinstance(origins,dict) or mode not in ('explicit_box','validated_parameter_archive'):
                    raise ValueError('Native graph construction mode/provenance is invalid')
                if mode == 'validated_parameter_archive' and not origins:
                    raise ValueError('Imported native box parameters require nonempty input provenance')
                if mode == 'explicit_box':
                    if origins or 'target_validation' not in source_records:
                        raise ValueError('Explicit native construction requires its validator source and no imported parameter histories')
                    rebuilt,guess = reconstruct_explicit_native_graph(record.get('settings',{}))
                    if ('radius_path_nodes' in rebuilt) != ('radius_path_nodes' in archive):
                        raise ValueError('Explicit native radius-path declaration does not reconstruct')
                    for key,value in rebuilt.items():
                        if key not in archive or not np.array_equal(archive[key],value):
                            raise ValueError('Explicit native scalar settings do not reconstruct target: '+key)
                    if (archive['initial_log_weights'].shape != guess.shape
                            or not np.allclose(archive['initial_log_weights'],guess,rtol=0,atol=1e-10)):
                        raise ValueError('Explicit native efficiency weights do not reconstruct from recorded settings')
            for key in common:
                if key in archive:
                    reference = archive[key]
                elif key in ('number_stiffness','number_preferred') and not (native or single):
                    reference = np.zeros_like(data[key])
                else:
                    raise ValueError('Imported target lacks '+key)
                if not np.array_equal(data[key],reference):
                    raise ValueError('Recorded target/proposal differs from imported '+setting+': '+key)
            if native or single:
                for key in (*endpoint_names,'sites','cap'):
                    if not np.array_equal(data[key],archive[key]):
                        raise ValueError('Imported endpoint/size differs: '+key)
                if ('radius_path_nodes' in data) != ('radius_path_nodes' in archive):
                    raise ValueError('Imported physical radius-path declaration differs')
                if 'radius_path_nodes' in data and not np.array_equal(data['radius_path_nodes'],archive['radius_path_nodes']):
                    raise ValueError('Imported physical radius-path labels differ')
                if native:
                    stages = scalar_integer(record.get('settings',{}).get('cavity_radius_stages',1),'Native radius stages',1)
                    if (stages > 256 or (stages > 1) != ('radius_path_nodes' in data)
                            or ('radius_path_nodes' in data and len(data['radius_path_nodes']) != stages+1)):
                        raise ValueError('Native radius-stage count differs from its recorded path')
            else:
                base = int(archive['endpoint'])
                cavity_nodes = archive['cavity_nodes'] if 'cavity_nodes' in archive else np.arange(base,len(data['activity']))
                probes = archive['probe_nodes'] if 'probe_nodes' in archive else np.array([],dtype=int)
                ideal_nodes = np.flatnonzero((data['pair_penalty']==0)&(data['radius']==0)
                    &(data['umbrella_stiffness']==0)&(data['number_stiffness']==0))
                expected = dict(base_node=base,cavity_node=int(cavity_nodes[-1]),ideal_node=int(ideal_nodes[0]),
                                physical_nodes=np.unique(np.r_[cavity_nodes,probes]).astype(int),
                                sites=record['settings']['sites'],cap=record['settings']['cap'])
                if any(not np.array_equal(data[key],value) for key,value in expected.items()):
                    raise ValueError('Imported replica graph endpoint/size differs')
            if setting == 'weights_from':
                weights = archive['log_weights']
                expected = weights-weights[int(data['ideal_node'])]
                if 'initial_log_weights' not in data or not np.array_equal(data['initial_log_weights'],expected):
                    raise ValueError('Recorded initial weights differ from the imported frozen weights')
            elif native and not metadata['settings'].get('weights_from'):
                weights = archive['initial_log_weights']
                expected = weights-weights[int(data['ideal_node'])]
                if 'initial_log_weights' not in data or not np.array_equal(data['initial_log_weights'],expected):
                    raise ValueError('Recorded initial weights differ from the native graph efficiency guesses')


def load_validated_run(path):
    """Do not infer regenerative inference from a replica-shaped archive."""
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata.get('schema') != SCHEMA:
        raise ValueError('Expected the distinct single-host regenerative-tempering schema')
    if metadata.get('raw_sha256') != sha256(path):
        raise ValueError('Raw digest does not match the recorded execution')
    if metadata.get('weights_frozen_before_production') is not True:
        raise ValueError('Production requires explicitly frozen label weights')
    if metadata.get('sample_cadence_clock_steps') != 1:
        raise ValueError('Regenerative rewards require every clock step, not sparse saved frames')
    if tuple(metadata.get('columns',())) != COLUMNS:
        raise ValueError('Recorded column meanings must match the full-clock schema exactly')
    if metadata.get('deletion_label_threshold',DELETION_THRESHOLD) != DELETION_THRESHOLD:
        raise ValueError('Deletion diagnostic and operational phase threshold must both be0.45')
    snapshots = metadata.get('source_snapshots',{})
    if not {'driver','kernel','graph_validation'} <= snapshots.keys():
        raise ValueError('Executed driver, kernel and graph-validation source snapshots are required')
    verified_inputs = {path,path.with_suffix('.json')}
    for name,record in snapshots.items():
        if not isinstance(record,(list,tuple)) or len(record) != 2:
            raise ValueError('Source snapshot records require path and SHA256')
        source = (ROOT/record[0]).resolve()
        if not source.is_relative_to(ROOT) or sha256(source) != record[1]:
            raise ValueError('Executed source snapshot digest is not verified: '+name)
        verified_inputs.add(source)
    for name,digest in metadata.get('input_sha256',{}).items():
        source = (ROOT/name).resolve()
        if sha256(source) != digest:
            raise ValueError('Imported target/weight input digest mismatch: '+name)
        verified_inputs.add(source)
    for setting in ('graph_from','weights_from'):
        value = metadata.get('settings',{}).get(setting)
        if value:
            source = (ROOT/value).resolve()
            if not {source,source.with_suffix('.json')} <= verified_inputs:
                raise ValueError('Imported '+setting+' archive and sidecar must both be hash-verified')
    shape = array_shape(path,'samples')
    if len(shape) != 2 or shape[1] != 12 or shape[0] < 2 or np.prod(shape)*8 > 200*1024**2:
        raise ValueError('Expected bounded two-dimensional full-clock samples, not product replicas')
    required = ('samples','log_weights','volume','base_node','cavity_node','ideal_node',
                'physical_nodes','diameter','radius','activity','pair_penalty',
                'umbrella_stiffness','umbrella_preferred','number_stiffness','number_preferred',
                'exchange_edges','sites','cap','displacement')
    with np.load(path,allow_pickle=False) as archive:
        if str(archive['schema']) != SCHEMA:
            raise ValueError('Raw and sidecar schema disagree')
        data = {name:archive[name] for name in required}
        diagnostic_shapes = {'neighbor_source_labels':(shape[0],),
                             'gibbs_attempt_change_counts':(2,),
                             'calibration_weight_update_visits':(len(data['activity']),),
                             'calibration_visits':(len(data['activity']),)}
        for name in ('coordinate_clock_steps','coordinate_labels','coordinate_centers','initial_log_weights',
                     'neighbor_source_labels','gibbs_attempt_change_counts',
                     'calibration_weight_update_visits','calibration_visits','radius_path_nodes'):
            if name in archive:
                if name == 'radius_path_nodes':
                    path_shape = array_shape(path,name)
                    if len(path_shape) != 1 or not 3 <= path_shape[0] <= 257:
                        raise ValueError('Archived physical radius path must be a bounded vector')
                if name in diagnostic_shapes and array_shape(path,name) != diagnostic_shapes[name]:
                    raise ValueError('Archived label/calibration diagnostic shape is invalid: '+name)
                data[name] = archive[name]
        if 'coordinate_configurations' in archive:
            data['coordinate_shape'] = array_shape(path,'coordinate_configurations')
    cap,sites = scalar_integer(data['cap'],'Capacity',1),scalar_integer(data['sites'],'Reference size',1)
    if np.shape(data['volume']) != ():
        raise ValueError('Volume must be a finite positive scalar')
    volume = float(data['volume'])
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError('Finite positive volume, integer capacity and reference size required')
    vectors = ('log_weights','diameter','radius','activity','pair_penalty',
               'umbrella_stiffness','umbrella_preferred','number_stiffness','number_preferred','displacement')
    if data['activity'].ndim != 1:
        raise ValueError('Activities must form a one-dimensional target vector')
    nodes = len(data['activity'])
    if nodes < 3 or any(np.shape(data[name]) != (nodes,) for name in vectors):
        raise ValueError('Target and frozen-weight vectors must match all labels')
    for name in vectors:
        values = data[name]
        if not (np.isfinite(values)|((name=='pair_penalty')&np.isposinf(values))).all():
            raise ValueError('Target/weight parameters must be finite, except positive hard penalties')
    if 'initial_log_weights' in data and (data['initial_log_weights'].shape != (nodes,)
            or not np.isfinite(data['initial_log_weights']).all()):
        raise ValueError('Initial log weights must be a finite label-count vector')
    if (any(np.any(data[name] < 0) for name in ('radius','pair_penalty','umbrella_stiffness','number_stiffness','number_preferred'))
            or np.any(data['diameter'] <= 0) or np.any(data['displacement'] <= 0)
            or np.any(data['number_preferred'] > cap)):
        raise ValueError('Diameters must be positive and cost/number parameters nonnegative')
    base,cavity,ideal = (scalar_integer(data[name],name) for name in ('base_node','cavity_node','ideal_node'))
    if (len({base,cavity,ideal}) != 3 or min(base,cavity,ideal) < 0 or max(base,cavity,ideal) >= nodes
            ):
        raise ValueError('Base/cavity/ideal labels must be distinct valid integers')
    physical = data['physical_nodes']
    if (physical.ndim != 1 or not np.isfinite(physical).all()
            or np.any(physical != np.rint(physical)) or np.any(physical < 0) or np.any(physical >= nodes)
            or len(np.unique(physical)) != len(physical) or not {base,cavity} <= set(physical)):
        raise ValueError('Physical nodes must be distinct valid labels including both endpoints')
    physical = physical.astype(int)
    data['physical_nodes'] = physical
    if (np.any(~np.isposinf(data['pair_penalty'][physical]))
            or np.any(data['umbrella_stiffness'][physical] != 0)
            or np.any(data['number_stiffness'][physical] != 0)):
        raise ValueError('Every physical label must be hard and free of both auxiliary umbrellas')
    sigma = float(data['diameter'][base])
    if (np.any(data['diameter'] != sigma) or volume**(1/3) < sigma
            or data['radius'][base] != 0 or data['radius'][cavity] != sigma
            or data['diameter'][cavity] != sigma or data['activity'][base] != data['activity'][cavity]):
        raise ValueError('The base and particle-sized cavity require the same physical activity and diameter')
    if any(data[name][ideal] != 0 for name in ('pair_penalty','radius','umbrella_stiffness','number_stiffness')):
        raise ValueError('The regeneration label must be exactly ideal and umbrella-free')
    edges = data['exchange_edges']
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.isfinite(edges).all()
            or np.any(edges != np.rint(edges)) or np.any(edges < 0) or np.any(edges >= nodes)
            or np.any(edges[:,0] == edges[:,1])):
        raise ValueError('Label edges must join distinct valid labels')
    edge_set = {tuple(sorted(map(int,edge))) for edge in edges}
    if len(edge_set) != len(edges) or tuple(sorted((base,cavity))) not in edge_set:
        raise ValueError('Graph must have distinct undirected edges and the tagged endpoint edge')
    validate_radius_path(data)
    visited = {ideal}
    while True:
        expanded = visited|{int(b) for a,b in np.r_[edges,edges[:,::-1]] if int(a) in visited}
        if expanded == visited:
            break
        visited = expanded
    if len(visited) != nodes:
        raise ValueError('The label graph is disconnected')
    verify_imported_targets(data,metadata,verified_inputs)
    samples = data['samples']
    if (np.any(~np.isfinite(samples[:,[0,1,2,3,4,8,10,11]]))
            or np.any(samples[:,1] != np.rint(samples[:,1]))
            or np.any(samples[:,1] < 0) or np.any(samples[:,1] >= nodes)
            or np.any(samples[:,2] > cap)):
        raise ValueError('Invalid finite clock/count/label/deletion diagnostics')
    labels = samples[:,1].astype(int)
    physical_rows = samples[np.isin(labels,physical)]
    empty,excess = physical_rows[:,5],physical_rows[:,6]
    if (not np.isfinite(physical_rows[:,[5,6,9]]).all()
            or np.any(empty != np.rint(empty)) or np.any(excess != np.rint(excess))
            or np.any(empty < 0) or np.any(empty > sites) or np.any(excess < 0)
            or np.any(excess > physical_rows[:,2])
            or np.any(physical_rows[:,2]+empty-excess != sites)
            or np.any(physical_rows[:,9] < 0)
            or np.any(physical_rows[:,8] != np.rint(physical_rows[:,8]))
            or np.any(np.abs(physical_rows[:,8]) > physical_rows[:,2])):
        raise ValueError('Physical FCC occupancy diagnostics must obey N+empty-excess=reference sites')
    singleton = physical_rows[:,2] == 1
    if np.any(np.abs(physical_rows[physical_rows[:,2] == 0,3]) > 1e-12):
        raise ValueError('Empty physical hosts require the defined Bragg value B=0 and zero deletion shift')
    if (np.any(np.abs(physical_rows[singleton,3]-1) > 1e-12)
            or np.any(physical_rows[singleton,8] != -1)):
        raise ValueError('Singleton physical hosts require B=1 and delete-one label shift=-1, since the empty host has B=0')
    overlaps = samples[:,4]
    if (np.any(overlaps != np.rint(overlaps)) or np.any(overlaps < 0)
            or np.any(overlaps > samples[:,2]*(samples[:,2]-1)/2)
            or np.any(overlaps[np.isposinf(data['pair_penalty'][labels])] != 0)
            or np.any((samples[:,2] > 0)&(samples[:,7] < data['radius'][labels])) or np.isnan(samples[:,7]).any()):
        raise ValueError('Saved host violates overlap-count or cavity support')
    cutoff = scalar_integer(metadata.get('calibration_cutoff_clock',-1),'Calibration cutoff')
    production_steps = scalar_integer(metadata['settings']['production_steps'],'Production steps',1)
    if samples[0,0] != cutoff+1 or len(samples) != production_steps:
        raise ValueError('Production clock must begin after the discarded calibration and contain every declared step')
    # Includes full-clock/regeneration-ID and physical delete-one guards.
    complete_cycle_rewards(samples,base,cavity,ideal)
    data['label_update_protocol'] = validate_label_update_protocol(data,metadata)
    data['calibration_method'] = validate_calibration_record(data,metadata)
    coordinate_fields = {'coordinate_shape','coordinate_clock_steps','coordinate_labels','coordinate_centers'}
    if coordinate_fields.intersection(data) and not coordinate_fields <= data.keys():
        raise ValueError('Coordinate snapshots require aligned coordinates, clocks, labels and centers')
    if 'coordinate_shape' in data:
        times = data.get('coordinate_clock_steps')
        coordinate_labels = data.get('coordinate_labels')
        if (times is None or coordinate_labels is None or times.ndim != 1 or not np.isfinite(times).all()
                or np.any(times != np.rint(times)) or np.any(np.diff(times) <= 0)
                or np.any(times < samples[0,0]) or np.any(times > samples[-1,0])
                or coordinate_labels.shape != times.shape
                or data['coordinate_shape'] != (len(times),cap,3)):
            raise ValueError('Sparse coordinate frames must map to distinct recorded production clock steps')
        indices = (times-samples[0,0]).astype(int)
        if not np.array_equal(coordinate_labels,samples[indices,1]):
            raise ValueError('Sparse coordinate labels do not match their full-clock rows')
        centers = data['coordinate_centers']
        if (centers.shape != (len(times),3) or not np.isfinite(centers).all()
                or np.any(centers < 0) or np.any(centers >= volume**(1/3))):
            raise ValueError('Sparse coordinate centers must lie in the recorded periodic cell')
        data['coordinate_sample_indices'] = indices
    data['physical_cap_certified'] = bool(volume**(1/3) >= sigma and cap >= np.ceil(6*volume/(np.pi*sigma**3)))
    return data,metadata,verified_inputs


def complete_cycle_rewards(samples, base, cavity, ideal):
    """Return one reward row per complete [refresh, next refresh) cycle."""
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] < 12 or len(samples) < 2:
        raise ValueError('Single-walker samples require at least two fixed-clock rows and twelve columns')
    clock,label,n,bragg = samples[:,:4].T
    cycle,event = samples[:,10:12].T
    if (not np.isfinite(samples[:,:6]).all()
            or np.any(clock != np.rint(clock)) or np.any(np.diff(clock) != 1)
            or np.any(label != np.rint(label)) or np.any(label < 0)
            or np.any(n != np.rint(n)) or np.any(n < 0)
            or np.any(bragg < 0) or np.any(bragg > 1+1e-12)):
        raise ValueError('All clock steps, including rejected moves, must be retained with valid labels/counts')
    if (not np.isfinite(cycle).all() or np.any(cycle != np.rint(cycle))
            or np.any(cycle < 0) or np.any(~np.isin(event,(0.,1.)))
            or np.any(np.diff(cycle) != event[1:])
            or not np.array_equal(event.astype(bool),label==ideal)):
        raise ValueError('Cycles must increment exactly at the designated ideal refresh event before each recorded local update')
    if len({base,cavity,ideal}) != 3:
        raise ValueError('Physical base, cavity and regeneration ideal nodes must be distinct')
    starts = np.flatnonzero(event)
    bounds = np.column_stack((starts[:-1],starts[1:]))
    rewards = np.zeros((len(bounds),len(REWARD_COLUMNS)))
    for index,(start,stop) in enumerate(bounds):
        block = samples[start:stop]
        b = block[:,1] == base
        c = block[:,1] == cavity
        nb,nc = block[b,2],block[c,2]
        sb,sc = block[b,3] > DELETION_THRESHOLD,block[c,3] > DELETION_THRESHOLD
        deletion = block[b,8]
        if not np.isfinite(deletion).all() or np.any(deletion != np.rint(deletion)) or np.any(np.abs(deletion) > nb):
            raise ValueError('Physical deletion diagnostic must be the integer signed delete-one label sum')
        rewards[index] = (b.sum(),c.sum(),nb.sum(),(nb*nb).sum(),nc.sum(),
                          sb.sum(),sc.sum(),(nb*sb).sum(),deletion.sum(),np.abs(deletion).sum())
    return rewards,bounds,{
        'recorded_clock_steps':len(samples),'regeneration_events':len(starts),
        'complete_cycles':len(bounds),
        'leading_rows_excluded':int(starts[0]) if len(starts) else len(samples),
        'unfinished_trailing_rows_excluded':len(samples)-int(starts[-1]) if len(starts) else 0,
        'complete_cycle_clock_steps':int(np.sum(bounds[:,1]-bounds[:,0])) if len(bounds) else 0,
        'base_contributing_cycles':int(np.count_nonzero(rewards[:,0])),
        'cavity_contributing_cycles':int(np.count_nonzero(rewards[:,1])),
        'base_solid_contributing_cycles':int(np.count_nonzero(rewards[:,5])),
        'base_fluid_contributing_cycles':int(np.count_nonzero(rewards[:,0]-rewards[:,5])),
        'cavity_solid_contributing_cycles':int(np.count_nonzero(rewards[:,6])),
        'cavity_fluid_contributing_cycles':int(np.count_nonzero(rewards[:,1]-rewards[:,6])),
        'zero_physical_dwell_cycles':int(np.count_nonzero(rewards[:,:2].sum(axis=1)==0)),
        'cycle_interval_convention':'refresh event row inclusive to next refresh event row exclusive; initial and unfinished terminal partial cycles omitted',
    }


def estimate_totals(total, log_weight_base, log_weight_cavity, activity, volume,
                    physical_cap_certified=True):
    """Ratios of total clock rewards; cap certification gates Palm identities."""
    db,dc,n,n2,nc,sb,sc,ns,shift,absolute = np.asarray(total,dtype=float)
    ratio = lambda a,b: float(a/b) if b > 0 else float('nan')
    mean_n,mean_c = ratio(n,db),ratio(nc,dc)
    variance = max(0.,ratio(n2,db)-mean_n*mean_n) if db > 0 else float('nan')
    q,qc = ratio(sb,db),ratio(sc,dc)
    work = (float(np.log(db/dc)+log_weight_cavity-log_weight_base)
            if db > 0 and dc > 0 else float('nan'))
    result = {
        'mean_N_base':mean_n,'variance_N_base':variance,'mean_N_cavity':mean_c,
        'q_solid_base':q,'q_solid_cavity':qc,
        'number_response_direct':mean_c-mean_n,'work_from_label_dwell':work,
        'log_partition_ratio_cavity_to_base':-work,
        'mean_signed_deletion_shift':ratio(shift,db),
        'absolute_deletion_error_per_particle':ratio(absolute,n),
    }
    with np.errstate(over='ignore',invalid='ignore'):
        result['partition_ratio_cavity_to_base'] = float(np.exp(-work))
    if physical_cap_certified:
        response = ratio(variance,mean_n)-1
        q_palm = ratio(ns+shift,n)
        work_palm = float(activity-np.log(mean_n/volume)) if mean_n > 0 else float('nan')
        result.update({
            'number_response_Palm':response,'q_solid_cavity_Palm':q_palm,
            'work_Palm':work_palm,'direct_minus_Palm_N':mean_c-mean_n-response,
            'direct_minus_Palm_q':qc-q_palm,'label_minus_Palm_work':work-work_palm,
        })
    return result


def ratio_cycle_standard_error(numerator, denominator):
    """Regenerative ratio CLT: residual reward divided by mean cycle dwell."""
    numerator,denominator = np.asarray(numerator,float),np.asarray(denominator,float)
    if (numerator.shape != denominator.shape or numerator.ndim != 1
            or not np.isfinite(numerator).all() or not np.isfinite(denominator).all()
            or np.any(denominator < 0)):
        raise ValueError('Finite equal-length cycle rewards and nonnegative dwell are required')
    if len(numerator) < 2 or denominator.sum() <= 0:
        return float('nan')
    estimate = numerator.sum()/denominator.sum()
    residual = numerator-estimate*denominator
    return float(np.std(residual,ddof=1)/(np.sqrt(len(residual))*denominator.mean()))


def bootstrap_complete_cycles(rewards, draws, seed, return_covariance=False, **parameters):
    """Resample whole cycles with common weights, retaining zero-dwell cycles."""
    rewards = np.asarray(rewards,float)
    if rewards.ndim != 2 or rewards.shape[1] != len(REWARD_COLUMNS) or not np.isfinite(rewards).all():
        raise ValueError('Expected a finite complete-cycle reward matrix')
    if draws < 2 or seed < 0:
        raise ValueError('At least two bootstrap draws and a nonnegative seed are required')
    point = estimate_totals(rewards.sum(axis=0),**parameters)
    keys = list(point)
    if len(rewards) < 2:
        result = point,{key:None for key in keys},{key:0. for key in keys}
        return (*result,{'columns':keys,'matrix':[[None]*len(keys) for _ in keys]}) if return_covariance else result
    rng = np.random.default_rng(seed)
    values = np.full((draws,len(keys)),np.nan)
    for draw in range(draws):
        indices = rng.integers(len(rewards),size=len(rewards))
        weights = np.bincount(indices,minlength=len(rewards))
        record = estimate_totals(weights@rewards,**parameters)
        values[draw] = [record[key] for key in keys]
    fraction = {key:float(np.mean(np.isfinite(values[:,i]))) for i,key in enumerate(keys)}
    errors = {key:(float(np.std(values[:,i],ddof=1)) if fraction[key] == 1 else None)
              for i,key in enumerate(keys)}
    result = point,errors,fraction
    if not return_covariance:
        return result
    covariance = [[float(np.cov(values[:,i],values[:,j],ddof=1)[0,1])
                   if fraction[a] == fraction[b] == 1 else None
                   for j,b in enumerate(keys)] for i,a in enumerate(keys)]
    return (*result,{'columns':keys,'matrix':covariance,
                    'scope':'Common whole-cycle resamples; no undefined draw is discarded. Any pair with an undefined draw has undefined covariance.'})


def phase_uncertainty_display(point,error,cycles,high_cycles,low_cycles,maximum_dwell_fraction,incomplete_fraction):
    """Suppress misleading empirical zero/rare-basin errors in display fields."""
    reasons = []
    if not np.isfinite(point):reasons.append('No complete-cycle probability estimate')
    elif point in (0.,1.):reasons.append('A basin is unobserved; empirical zero variance is not equilibrium precision')
    if cycles < 20:reasons.append('Fewer than 20 complete cycles')
    if min(high_cycles,low_cycles) < 10:reasons.append('Fewer than 10 contributing cycles for either basin')
    if maximum_dwell_fraction is None or maximum_dwell_fraction > .2:
        reasons.append('Missing dwell or one cycle contributes more than 20% of dwell')
    if incomplete_fraction > .1:reasons.append('More than 10% of the clock is in incomplete cycles')
    if error is None or not np.isfinite(error):reasons.append('Conditional cycle standard error is undefined')
    return {'value':point,'SE_for_display':float(error) if not reasons else None,
            'display_checks_passed':not reasons,'suppression_reasons':reasons,
            'scope':'Screens for displaying a conditional empirical cycle error, not a convergence or equilibrium certificate. Raw empirical errors are retained separately; an exact structural zero requires an independent analytic argument.'}


def physical_node_cycle_profiles(data,bounds):
    """Separate conditional reward ratios at each physical activity/radius.

    Uses O(clock rows + cycles) workspace per label, never a cycles-by-labels
    bootstrap tensor. CLT residuals retain every complete cycle, including
    cycles with zero dwell at this particular node.
    """
    samples = data['samples']
    count = len(bounds)
    rows = samples[bounds[0,0]:bounds[-1,1]] if count else samples[:0]
    cycle = (rows[:,10]-rows[0,10]).astype(int) if len(rows) else np.array([],dtype=int)
    incomplete = 1-len(rows)/len(samples)
    result = []
    for node in data['physical_nodes']:
        mask = rows[:,1] == node
        indices,n,b = cycle[mask],rows[mask,2],rows[mask,3] > DELETION_THRESHOLD
        dense = b&(n>=20)
        dwell = np.bincount(indices,minlength=count).astype(float)
        nsum = np.bincount(indices,weights=n,minlength=count)
        bsum = np.bincount(indices,weights=b.astype(float),minlength=count)
        dense_sum = np.bincount(indices,weights=dense.astype(float),minlength=count)
        visits = float(dwell.sum())
        mean = float(n.sum()/visits) if visits else float('nan')
        q = float(b.sum()/visits) if visits else float('nan')
        variance = max(0.,float((n*n).sum()/visits)-mean*mean) if visits else float('nan')
        q_dense = float(dense.sum()/visits) if visits else float('nan')
        record = {
            'node':int(node),'activity':float(data['activity'][node]),
            'radius':float(data['radius'][node]),'complete_cycle_clock_visits':int(visits),
            'contributing_complete_cycles':int(np.count_nonzero(dwell)),
            'B_high_contributing_cycles':int(np.count_nonzero(bsum)),
            'B_low_contributing_cycles':int(np.count_nonzero(dwell-bsum)),
            'mean_N':mean,'density':mean/float(data['volume']),'variance_N':variance,
            'operational_B_fraction':q,'dense_operational_B_fraction':q_dense,
            'mean_N_residual_CLT_SE':ratio_cycle_standard_error(nsum,dwell),
            'B_fraction_residual_CLT_SE':ratio_cycle_standard_error(bsum,dwell),
            'dense_B_fraction_residual_CLT_SE':ratio_cycle_standard_error(dense_sum,dwell),
            'covariance_N_B':float((n*b).sum()/visits)-mean*q if visits else float('nan'),
            'maximum_cycle_fraction_of_dwell':float(dwell.max()/visits) if visits else None,
            'B_boundary_fraction_uncertainty':q in (0.,1.) if visits else None,
        }
        record['B_fraction_uncertainty_display'] = phase_uncertainty_display(
            q,record['B_fraction_residual_CLT_SE'],count,record['B_high_contributing_cycles'],
            record['B_low_contributing_cycles'],record['maximum_cycle_fraction_of_dwell'],incomplete)
        record['dense_B_fraction_uncertainty_display'] = phase_uncertainty_display(
            q_dense,record['dense_B_fraction_residual_CLT_SE'],count,int(np.count_nonzero(dense_sum)),
            int(np.count_nonzero(dwell-dense_sum)),record['maximum_cycle_fraction_of_dwell'],incomplete)
        result.append(record)
    return result


def analyze(path,draws=1000,seed=2026092940):
    data,metadata,inputs = load_validated_run(path)
    base,cavity,ideal = (int(data[name]) for name in ('base_node','cavity_node','ideal_node'))
    samples = data['samples']
    rewards,bounds,coverage = complete_cycle_rewards(samples,base,cavity,ideal)
    parameters = dict(log_weight_base=float(data['log_weights'][base]),
                      log_weight_cavity=float(data['log_weights'][cavity]),
                      activity=float(data['activity'][base]),volume=float(data['volume']),
                      physical_cap_certified=data['physical_cap_certified'])
    point,errors,fraction,covariance = bootstrap_complete_cycles(rewards,draws,seed,return_covariance=True,**parameters)
    reasons = []
    if len(rewards) < 20:
        reasons.append('Fewer than20 complete regeneration cycles')
    for name,column in (('base',0),('cavity',1)):
        if np.count_nonzero(rewards[:,column]) < 10:
            reasons.append(f'Fewer than10 complete cycles visit the {name} endpoint')
        for basin in ('solid','fluid'):
            if coverage[f'{name}_{basin}_contributing_cycles'] < 10:
                reasons.append(f'Fewer than10 complete cycles visit the {name} operational {basin} basin')
    boundary = {name:(point[name] in (0.,1.) if np.isfinite(point[name]) else None)
                for name in ('q_solid_base','q_solid_cavity')}
    if any(boundary.values()):
        reasons.append('An endpoint has no sampled variation between the two operational basins; zero conditional variance is not equilibrium precision')
    if any(value < 1 for value in fraction.values()):
        reasons.append('Some whole-cycle resamples have undefined conditional ratios; their standard errors are left undefined, not computed after dropping draws')
    excluded_fraction = (coverage['leading_rows_excluded']+coverage['unfinished_trailing_rows_excluded'])/len(samples)
    if excluded_fraction > .1:
        reasons.append('More than10% of the fixed-clock history belongs to incomplete cycles; a long unfinished visit can dominate stopping-time bias and is not captured by completed-cycle resampling')
    quarters = []
    for rows in np.array_split(samples,4):
        record = {}
        for name,node in (('base',base),('cavity',cavity)):
            selected = rows[rows[:,1] == node]
            record[name] = {
                'clock_visits':len(selected),
                'mean_N':float(selected[:,2].mean()) if len(selected) else None,
                'solid_fraction':float(np.mean(selected[:,3] > DELETION_THRESHOLD)) if len(selected) else None,
            }
        quarters.append(record)
    profile = []
    for node in range(len(data['activity'])):
        rows = samples[samples[:,1] == node]
        profile.append({'node':node,'clock_visits':len(rows),
            'mean_N':float(rows[:,2].mean()) if len(rows) else None,
            'mean_Bragg':float(rows[:,3].mean()) if len(rows) else None,
            'N_support':np.unique(rows[:,2]).astype(int).tolist(),
            'empty_reference_site_support':np.unique(rows[:,5]).astype(int).tolist() if node in data['physical_nodes'] else None,
            'excess_reference_site_support':np.unique(rows[:,6]).astype(int).tolist() if node in data['physical_nodes'] else None})
    residual_se = {name:ratio_cycle_standard_error(rewards[:,numerator],rewards[:,denominator])
                   for name,numerator,denominator in (
                       ('mean_N_base',2,0),('mean_N_cavity',4,1),
                       ('q_solid_base',5,0),('q_solid_cavity',6,1))}
    dwell = {}
    for name,column in (('base',0),('cavity',1)):
        totals = rewards[:,column]
        dwell[name] = {'total_complete_cycle_clock_visits':int(totals.sum()),
                       'maximum_cycle_fraction_of_dwell':float(totals.max()/totals.sum()) if totals.sum() else None}
        if totals.sum() and totals.max()/totals.sum() > .2:
            reasons.append(f'One complete cycle contributes more than20% of {name} dwell; endpoint uncertainty is dominated by sparse long visits')
    phase_display = {
        name:phase_uncertainty_display(point['q_solid_'+name],errors['q_solid_'+name],len(rewards),
                                      coverage[name+'_solid_contributing_cycles'],coverage[name+'_fluid_contributing_cycles'],
                                      dwell[name]['maximum_cycle_fraction_of_dwell'],excluded_fraction)
        for name in ('base','cavity')}
    return {
        'scope':'Experimental single-host diagnostic; no equilibrium or coexistence certification',
        'label_update_protocol':data['label_update_protocol'],
        'calibration_method':data['calibration_method'],
        'discarded_calibration_outcome':metadata.get('calibration_outcome'),
        'calibration_scope':'All adaptation observations are discarded. Recorded flatness/stability outcomes check label-efficiency bookkeeping only and do not certify equilibrium or physical phase mixing.',
        'label_transition_scope':'Clock rows precede label updates. Compound Gibbs/neighbor clocks require the separately recorded post-Gibbs neighbor-source labels; next-row differences are not neighbor acceptance counts. Cycle rewards retain every recorded pre-hop dwell.',
        'phase_label_scope':'The reported solid label is the operational Fourier criterion B>0.45, not an independently established thermodynamic phase; low-N and alternative structural order require separate coordinate checks.',
        'raw_file':display_path(path),'raw_sha256':metadata['raw_sha256'],
        'estimates_from_complete_cycle_reward_ratios':point,
        'whole_cycle_bootstrap_SE':errors,'bootstrap_finite_fraction':fraction,
        'whole_cycle_bootstrap_covariance':covariance,
        'partition_ratio_convention':'Z_cavity/Z_base=(D_cavity/D_base)*exp(g_base-g_cavity); beta W=-log(Z_cavity/Z_base). D is summed full-clock dwell over complete cycles, not an acceptance or transition count.',
        'ratio_residual_CLT_SE':residual_se,
        'uncertainty_scope':'Complete-cycle reward/dwell ratios with common whole-cycle resampling. The underlying refreshed cycles are independent at frozen weights, but a fixed-clock stop selects a random duration-dependent prefix. Standard errors are asymptotic and conditional on observed cycle coverage, not exact finite-sample unbiased inference, an effective physical-snapshot count, or proof of equilibration',
        'bootstrap_draws':draws,'bootstrap_seed':seed,'coverage':coverage,
        'adequacy_warnings':reasons,'minimum_reported_complete_cycles':20,
        'minimum_reported_contributing_cycles_per_endpoint':10,
        'adequacy_threshold_scope':'Diagnostic reporting thresholds, not convergence criteria or a calibrated equilibrium test',
        'boundary_fraction_uncertainty':boundary,'dwell_concentration':dwell,
        'phase_probability_uncertainty_display':phase_display,
        'fraction_of_clock_in_incomplete_cycles':excluded_fraction,
        'physical_cap_certified_nontruncating':data['physical_cap_certified'],
        'Palm_identity_scope':'Untruncated hard-sphere identities are evaluated only when the geometric packing bound proves the physical cap inactive',
        'physical_target':{'activity':parameters['activity'],'volume':parameters['volume'],
                           'diameter':float(data['diameter'][base]),
                           'cavity_radius':float(data['radius'][cavity]),'cap':int(data['cap'])},
        'physical_radius_path_nodes':data['radius_path_nodes'].tolist() if 'radius_path_nodes' in data else None,
        'frozen_log_weights':data['log_weights'].tolist(),
        'physical_node_complete_cycle_profiles':physical_node_cycle_profiles(data,bounds),
        'physical_node_profile_scope':'Each activity/radius is a separate conditional target; no activity pooling. Ratios use only complete-cycle clock rewards with zero-dwell cycles retained. B>0.45 is an operational Fourier indicator, not a thermodynamic solid classifier at low N. The separate dense indicator additionally requires N>=20 and has no reused delete-one/Palm diagnostic. Residual CLT errors are asymptotic and subject to the global incomplete-cycle/coverage warnings.',
        'label_clock_profile':profile,'full_clock_production_quarters':quarters,
        'quarter_scope':'Descriptive full-clock quarters, including leading/trailing partial cycles; distinct from the complete-cycle estimator and not independent replicates',
    },inputs


def write_analysis(paths,output,draws=1000,seed=2026092940):
    paths = [Path(path).resolve() for path in paths]
    output = Path(output).resolve()
    sidecar = output.with_suffix('.metadata.json')
    protected = {item for path in paths for item in (path,path.with_suffix('.json'))}
    if (not paths or len({p.stem for p in paths}) != len(paths) or output.suffix != '.json'
            or {output,sidecar}.intersection(protected)):
        raise ValueError('Provide distinct raw inputs and a new JSON analysis output')
    results,inputs = {},set()
    for path in paths:
        result,run_inputs = analyze(path,draws,seed)
        results[path.stem] = result
        inputs.update(run_inputs)
    if {output,sidecar}.intersection(inputs):
        raise ValueError('Analysis output may not overwrite source snapshots or imported target/weight inputs')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(json_ready(results),indent=2,allow_nan=False)+'\n')
    sidecar.write_text(json.dumps({
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'script':'scripts/continuum_single_walker_analyze.py',
        'script_sha256':EXECUTION_SOURCE_SHA256,'python':sys.version,'numpy':np.__version__,
        'input_sha256':{display_path(path):sha256(path) for path in sorted(inputs)},
        'output_sha256':{display_path(output):sha256(output)},
        'reproduce_argv':['python3','scripts/continuum_single_walker_analyze.py',
                          *map(display_path,paths),'--output',display_path(output),
                          '--bootstrap',str(draws),'--seed',str(seed)],
    },indent=2)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=2026092940)
    args = parser.parse_args()
    try:
        result = write_analysis(args.inputs,args.output,args.bootstrap,args.seed)
    except (ValueError,KeyError,FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({name:record['coverage'] for name,record in result.items()}))


if __name__ == '__main__':
    main()
