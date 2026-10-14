#!/usr/bin/env python3
"""Saved-state endpoint/radius conductance and reverse-birth diagnostics.

This is not a sampling engine or an equilibrium estimator. Frozen-host path
probabilities do not include the intervening local updates and cannot be read
as observed full-loop rates. Label quotas enter only through the actual frozen
weights; multiplying a kernel probability by its nominal quota would be wrong.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import continuum_single_walker_analyze as analysis
from . import continuum_single_walker_gates as gates
from . import continuum_compact_hardening_diagnosis as hardening

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BYTES = Path(__file__).read_bytes()
DEPENDENCIES = [(Path(m.__file__).stem, Path(m.__file__).read_bytes())
                for m in (analysis, gates, hardening)]


def normalizer_constraints(data):
    """Necessary efficiency-weight checks, not a convergence certificate."""
    path = np.asarray(data['radius_path_nodes'], dtype=int)
    quota = np.asarray(data['calibration_label_quota'])
    adjusted = data['log_weights'][path] - np.log(quota[path])
    differences = np.diff(adjusted)
    base, cavity = int(data['base_node']), int(data['cavity_node'])
    volume = float(data['volume'])
    sigma = float(data['diameter'][base])
    packing_bound = min(int(data['cap']), math.floor(6 * volume / (math.pi * sigma**3)))
    if packing_bound < 1:
        raise ValueError('Packing upper bound must be positive')
    log_zeta_volume = float(data['activity'][base]) + math.log(volume)
    return {
        'scope': 'Necessary properties of b=log(quota)-log(partition)+constant only; arbitrary frozen b still preserves conditional targets',
        'path_nodes': path.tolist(),
        'quota_adjusted_log_weights': adjusted.tolist(),
        'adjacent_differences': differences.tolist(),
        'monotonicity_violations': int(np.count_nonzero(differences < -1e-10)),
        'largest_negative_adjacent_difference': float(min(0., differences.min())),
        'endpoint_adjusted_span': float(adjusted[-1] - adjusted[0]),
        'endpoint_work_lower_bound': log_zeta_volume - math.log(packing_bound),
        'endpoint_work_upper_bound': float(np.logaddexp(0., log_zeta_volume)),
        'geometric_particle_upper_bound': packing_bound,
        'endpoint_span_within_exact_bounds': bool(
            log_zeta_volume - math.log(packing_bound) <= adjusted[-1] - adjusted[0]
            <= np.logaddexp(0., log_zeta_volume)),
    }


def checkpoint_contrasts(metadata, path, base, cavity):
    history = metadata['calibration_outcome']['stage_history']
    first, last = int(path[1]), int(path[-2])
    contrasts = []
    for stage in history:
        b = np.asarray(stage['log_weights'])
        contrasts.append({'clock': int(stage['end_clock']), 'gain': float(stage['gain']),
            'stage': int(stage['stage']), 'base_minus_first_radius': float(b[base]-b[first]),
            'cavity_minus_last_radius': float(b[cavity]-b[last]),
            'endpoint_span': float(b[cavity]-b[base]),
            'weighted_flatness': float(stage['flatness_minimum_to_mean']),
            'half_stability_passed': bool(stage['dense_half_stability_passed'])})
    final = history[-1]
    final_clock = int(final['end_clock'])
    recent = [x for x in contrasts if x['clock'] > final_clock-120000]
    return {'scope': 'Discarded adaptive checkpoint history, not thermodynamic observations',
        'all_checkpoint_contrasts': contrasts, 'last_120k_checkpoint_contrasts': recent,
        'last_120k_cavity_radius_gap_range': [min(x['cavity_minus_last_radius'] for x in recent),
                                            max(x['cavity_minus_last_radius'] for x in recent)],
        'last_stage_physical_half_profiles': final['dense_half_profiles'],
        'last_stage_gain': float(final['gain']),
        'last_stage_weighted_flatness': float(final['flatness_minimum_to_mean']),
        'last_stage_half_stability_passed': bool(final['dense_half_stability_passed'])}


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {'samples': len(values), 'mean': float(values.mean()) if len(values) else None,
            'minimum': float(values.min()) if len(values) else None,
            'maximum': float(values.max()) if len(values) else None}


def diagnose(data, metadata, raw, maximum_frames=512):
    if not isinstance(maximum_frames, int) or not 1 <= maximum_frames <= 4096:
        raise ValueError('Coordinate diagnostic frame limit must lie in 1..4096')
    base, cavity = int(data['base_node']), int(data['cavity_node'])
    path = np.asarray(data['radius_path_nodes'], dtype=int)
    degrees = gates.neighbor_degrees(data['exchange_edges'], len(data['activity']))
    volume = float(data['volume'])
    length = volume**(1/3)
    cells = int(round((int(data['sites'])/4)**(1/3)))
    basis = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    reference = np.array([(np.array([i,j,k])+b)*length/cells for i in range(cells)
                         for j in range(cells) for k in range(cells) for b in basis])
    width = float(metadata['settings'].get('birth_width', .06))
    epsilon = float(metadata['settings'].get('birth_uniform_fraction', .2))
    rows = data['samples']
    node_profiles = []
    for node in (base, cavity):
        chosen = rows[:,1] == node
        values = rows[chosen]
        condition = gates.conditional_label_diagnostics(data, values, executed_gibbs=True)
        masses = np.asarray(condition['mean_conditional_probability_by_node'])
        ordinary_destinations = []
        for pair in data['exchange_edges']:
            if node not in pair:
                continue
            other = int(pair[1] if pair[0] == node else pair[0])
            probability, support, score, tagged = gates.directional_score(
                data, node, other, values[:,2], values[:,3], values[:,4], values[:,7], degrees)
            ordinary_destinations.append({'destination': other, 'tagged': bool(tagged),
                'destination_radius': float(data['radius'][other]),
                'frozen_weight_difference': float(data['log_weights'][other]-data['log_weights'][node]),
                'degree_correction': float(math.log(degrees[node]/degrees[other])),
                'cross_support_fraction': float(support.mean()),
                'mean_MH_acceptance_if_proposed': float(probability.mean()),
                'expected_accepted_hops_on_recorded_source_path': float(probability.sum()*.5/degrees[node])})
        node_profiles.append({'node':node, 'clocks':len(values), 'N_support':np.unique(values[:,2]).astype(int).tolist(),
            'mean_N':float(values[:,2].mean()), 'mean_Bragg':float(values[:,3].mean()),
            'mean_Gibbs_mass_on_base':float(masses[base]), 'mean_Gibbs_mass_on_cavity':float(masses[cavity]),
            'mean_Gibbs_mass_off_endpoint_pair':float(np.delete(masses,[base,cavity]).sum()),
            'outgoing_edges':ordinary_destinations})
    with np.load(raw, allow_pickle=False) as archive:
        coordinates = archive['coordinate_configurations']
        local_accepts = archive['local_accepts']
        relocation_counts = archive['vacancy_relocation_counts']
    frame_labels = data['coordinate_labels'].astype(int)
    eligible = np.flatnonzero(np.isin(frame_labels, [base,cavity]))
    selected = eligible[np.unique(np.linspace(0, len(eligible)-1, min(maximum_frames,len(eligible)),dtype=int))] if len(eligible) else []
    death_records, path_records = [], []
    for frame in selected:
        index = int(data['coordinate_clock_steps'][frame]-rows[0,0])
        row = rows[index]
        node, n = int(row[1]), int(row[2])
        host = coordinates[frame,:n]
        phases = np.exp(2j*np.pi*cells/length*(host@hardening.VECTORS.T))
        modes = phases.sum(axis=0)
        if n and abs(float(np.abs(modes).mean()/n)-row[3]) > 1e-9:
            raise ValueError('Snapshot Fourier order disagrees with full-clock row')
        for tag in range(n):
            residual = np.delete(host, tag, axis=0)
            density = hardening.reverse_density(host[tag],residual,reference,length,cells,width,epsilon)
            logratio = math.log(n*density)-data['activity'][node]
            death_records.append((node,n,density,density>epsilon/volume*(1+1e-9),math.exp(min(0.,logratio))))
            if node != base:
                continue
            residual_b = float(np.abs(modes-phases[tag]).mean()/(n-1)) if n>1 else 0.
            distances = residual-host[tag]
            distances -= length*np.rint(distances/length)
            distance = float(np.sqrt((distances**2).sum(axis=1).min())) if len(residual) else math.inf
            if distance < data['radius'][cavity]-1e-12:
                raise ValueError('Tag-deleted host lacks endpoint cavity support')
            probabilities = gates.conditional_label_probabilities(data,np.array([n-1.]),
                np.array([residual_b]),np.array([0.]),np.array([distance]))[0]
            log_tag = (data['log_weights'][cavity]-data['log_weights'][base]
                       +math.log(n)-data['activity'][base]-math.log(volume)
                       +math.log(degrees[base]/degrees[cavity]))
            tag_alpha = math.exp(min(0.,log_tag))
            first = int(path[-2])
            shrink_alpha = math.exp(min(0.,data['log_weights'][first]-data['log_weights'][cavity]
                                       +math.log(degrees[cavity]/degrees[first])))
            first_shrink_rate = probabilities[cavity]*.5/degrees[cavity]*shrink_alpha
            two_clock = .5/degrees[base]*tag_alpha*first_shrink_rate
            path_records.append((n,tag_alpha,probabilities[cavity],first_shrink_rate,two_clock,
                                 float(probabilities[path[1:-1]].sum())))
    death_profiles = []
    for node in (base,cavity):
        records = [x for x in death_records if x[0]==node]
        if records:
            death_profiles.append({'node':node,'uniform_floor':epsilon/volume,
                'removed_particle_proposals':len(records),
                'vacancy_cube_cover_fraction':float(np.mean([x[3] for x in records])),
                'reverse_density':summarize([x[2] for x in records]),
                'mean_death_MH_acceptance':float(np.mean([x[4] for x in records])),
                'uniform_floor_death_acceptance_by_N': {str(n):n*epsilon/volume*math.exp(-data['activity'][node])
                                                       for n in sorted({x[1] for x in records})}})
    paired_profiles = []
    for n in sorted({x[0] for x in path_records}):
        records = [x for x in path_records if x[0]==n]
        paired_profiles.append({'base_N':n,'tags_evaluated':len(records),
            'tagged_delete_MH_acceptance':float(np.mean([x[1] for x in records])),
            'post_delete_cavity_Gibbs_mass':float(np.mean([x[2] for x in records])),
            'post_delete_intermediate_radius_Gibbs_mass':float(np.mean([x[5] for x in records])),
            'next_clock_cavity_retention_and_first_shrink_probability':float(np.mean([x[3] for x in records])),
            'two_clock_tag_then_first_shrink_probability':float(np.mean([x[4] for x in records]))})
    return {'scope':'Nonequilibrium frozen-path, discarded-checkpoint and sparse geometry diagnosis only',
        'quota_is_not_extra_kernel_factor':'Nominal quotas do not multiply frozen transition probabilities. They enter through the actual b; Gibbs retention uses reconstructed conditional mass.',
        'exact_normalizer_consistency':normalizer_constraints(data),
        'discarded_checkpoint_contrasts':checkpoint_contrasts(metadata,path,base,cavity),
        'endpoint_conditional_and_edge_profiles':node_profiles,
        'saved_coordinate_frames':len(selected), 'available_physical_coordinate_frames':len(eligible),
        'coordinate_selection':'Deterministic equally spaced subset of physical frames; proposals and frames are correlated, not independent statistical observations',
        'physical_reverse_deletion_profiles':death_profiles,
        'tag_then_radius_first_step':paired_profiles,
        'paired_path_qualification':'Counterfactual two-clock fixed-host calculation: the intervening local updates are suppressed for this diagnosis. It is not an observed full-loop rate; global Gibbs can also jump directly to any supported radius.',
        'physical_number_attempts_per_clock':int(data['sites'])//4,
        'production_local_attempt_accept_counters':local_accepts.tolist(),
        'production_relocation_attempt_accept_counters':relocation_counts.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-frames',type=int,default=512)
    args = parser.parse_args()
    output=args.output.resolve(); sidecar=output.with_suffix('.metadata.json')
    if output.exists() or sidecar.exists():
        parser.error('Diagnostic output prefix must be new')
    data,metadata,inputs=analysis.load_validated_run(args.input)
    result=diagnose(data,metadata,args.input,args.maximum_frames)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    snapshots={}
    for name,source_bytes in [('continuum_endpoint_quota_diagnosis',SOURCE_BYTES)]+DEPENDENCIES:
        digest=hashlib.sha256(source_bytes).hexdigest()
        source=ROOT/'data_raw/source_snapshots'/f'{name}_{digest}.py'
        if source.exists() and source.read_bytes()!=source_bytes:
            raise ValueError('Content-addressed source collision')
        if not source.exists(): source.write_bytes(source_bytes)
        snapshots[analysis.display_path(source)]=digest
    sidecar.write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':hashlib.sha256(SOURCE_BYTES).hexdigest(),'source_snapshots':snapshots,
        'input_sha256':{analysis.display_path(p):analysis.sha256(p) for p in inputs},
        'output_sha256':{analysis.display_path(output):analysis.sha256(output)},
        'reproduce_argv':['python3','-m','scripts.continuum_endpoint_quota_diagnosis',
            analysis.display_path(args.input),'--output',analysis.display_path(output),
            '--maximum-frames',str(args.maximum_frames)]},indent=2)+'\n')
    print(json.dumps({'physical_frames':result['saved_coordinate_frames'],
        'normalizer_monotonicity_violations':result['exact_normalizer_consistency']['monotonicity_violations']}))


if __name__=='__main__': main()
