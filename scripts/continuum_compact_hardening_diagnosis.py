#!/usr/bin/env python3
"""Path-conditioned compact hardening and saved-state deletion diagnostics.

No MC, kernel mutation, thermodynamic phase assignment or equilibrium estimator
is involved. Reference-site multiple assignments in a soft auxiliary host are
not physical crystal defects. Hypothetical deletion acceptance is evaluated
at saved states, not at the actual pre-deletion proposal clock.
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

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_HASH = hashlib.sha256(SOURCE_BYTES).hexdigest()
VECTORS = np.array([[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],[2,0,0],[0,2,0],[0,0,2]])


def fit_registry(positions,reference,length,cells):
    """Independent NumPy version of the deterministic two-parity full scan."""
    positions = np.asarray(positions)
    translation = (np.angle(np.exp(4j*np.pi*cells*positions/length).sum(axis=0))
                   *length/(4*np.pi*cells)) if len(positions) else np.zeros(3)
    candidates = []
    for parity in (0,1):
        shift = translation.copy()
        shift[0] += parity*length/(2*cells)
        centers = (reference+shift)%length
        offsets = positions[:,None,:]-centers[None,:,:]
        offsets -= length*np.rint(offsets/length)
        distances = (offsets**2).sum(axis=2)
        assignments = np.argmin(distances,axis=1) if len(positions) else np.array([],dtype=int)
        score = distances[np.arange(len(positions)),assignments].sum()
        candidates.append((score,centers,assignments))
    # Exact equal scores select the first parity, as the actual proposal does.
    score,centers,assignments = candidates[int(candidates[1][0]<candidates[0][0])]
    occupancy = np.bincount(assignments,minlength=len(reference))
    return centers,occupancy,assignments,float(score)


def reverse_density(point,residual,reference,length,cells,width,uniform_fraction):
    """All-cover marginal density after a complete post-deletion registry fit."""
    if len(residual)<2:
        return 1/length**3
    centers,occupied,_,_ = fit_registry(residual,reference,length,cells)
    empty = centers[occupied==0]
    if not len(empty):
        return 1/length**3
    delta = point-empty
    delta -= length*np.rint(delta/length)
    covers = np.count_nonzero(np.all(np.abs(delta)<width,axis=1))
    return uniform_fraction/length**3+(1-uniform_fraction)*covers/(len(empty)*(2*width)**3)


def death_logratio(n,density,field,pair_energy,order_difference,nu,nstar):
    """Exact MH death log odds; negative order difference favors survival."""
    unbiased_number = math.log(n*density)-field+pair_energy+order_difference
    number_term = nu*(2*(n-nstar)-1)
    return unbiased_number+number_term,unbiased_number,number_term


def saved_deletion_diagnostics(data,metadata,path):
    length = float(data['volume'])**(1/3)
    cells = int(round((int(data['sites'])/4)**(1/3)))
    basis = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    reference = np.array([(np.array([i,j,k])+b)*length/cells
        for i in range(cells) for j in range(cells) for k in range(cells) for b in basis])
    width = metadata['settings'].get('birth_width',.06)
    uniform = metadata['settings'].get('birth_uniform_fraction',.2)
    with np.load(path,allow_pickle=False) as archive:
        coordinates = archive['coordinate_configurations']
    clock_indices = (data['coordinate_clock_steps']-data['samples'][0,0]).astype(int)
    records = []
    snapshot_summary = []
    for frame,(node,index) in enumerate(zip(data['coordinate_labels'].astype(int),clock_indices)):
        gamma = float(data['pair_penalty'][node])
        if not (6<=gamma<=8 and data['umbrella_stiffness'][node]>0 and data['number_stiffness'][node]>0):
            continue
        row,n = data['samples'][index],int(data['samples'][index,2])
        current = coordinates[frame,:n]
        if n<2:
            continue
        phases = np.exp(2j*np.pi*cells/length*(current@VECTORS.T))
        sums = phases.sum(axis=0)
        b = float(np.abs(sums).mean()/n)
        if abs(b-row[3])>1e-9:
            raise ValueError('Saved-state Bragg reconstruction disagrees with trace')
        centers,occupied,assigned,score = fit_registry(current,reference,length,cells)
        offsets = current[:,None,:]-current[None,:,:]
        offsets -= length*np.rint(offsets/length)
        overlap = (offsets**2).sum(axis=2)<data['diameter'][node]**2
        np.fill_diagonal(overlap,False)
        if int(overlap.sum()/2)!=int(row[4]):
            raise ValueError('Saved-state pair overlap reconstruction disagrees with trace')
        snapshot_summary.append({'node':int(node),'clock':int(row[0]),'N':n,'Bragg':b,
            'pair_overlaps':int(row[4]),'empty_reference_sites':int((occupied==0).sum()),
            'multiply_assigned_reference_particles':int((occupied[assigned]>1).sum()),
            'registry_RMS':float(np.sqrt(score/n))})
        kappa,preferred = data['umbrella_stiffness'][node],data['umbrella_preferred'][node]
        for particle in range(n):
            density = reverse_density(current[particle],np.delete(current,particle,axis=0),
                                      reference,length,cells,width,uniform)
            removed_b = float(np.abs(sums-phases[particle]).mean()/(n-1))
            pair_degree = int(overlap[particle].sum())
            order_difference = kappa*((b-preferred)**2-(removed_b-preferred)**2)
            logratio,number_free,number_term = death_logratio(n,density,data['activity'][node],
                gamma*pair_degree,order_difference,data['number_stiffness'][node],data['number_preferred'][node])
            records.append({'node':int(node),'N':n,'overlapping_particle':pair_degree>0,
                'multiply_assigned_reference_particle':bool(occupied[assigned[particle]]>1),
                'local_pair_overlaps':pair_degree,'reverse_proposal_density':float(density),
                'number_log_odds_term':float(number_term),
                'acceptance':float(np.exp(min(0.,logratio))),
                'counterfactual_nu_zero_acceptance':float(np.exp(min(0.,number_free)))})
    profiles = []
    for n in sorted({x['N'] for x in records}):
        for name,select in [('all',lambda r:True),('overlapping',lambda r:r['overlapping_particle']),
            ('multiply_assigned_reference',lambda r:r['multiply_assigned_reference_particle']),
            ('overlapping_and_multiply_assigned',lambda r:r['overlapping_particle'] and r['multiply_assigned_reference_particle'])]:
            selected = [x for x in records if x['N']==n and select(x)]
            if selected:
                profiles.append({'N':n,'subset':name,'saved_state_particle_proposals':len(selected),
                    'mean_death_acceptance':float(np.mean([x['acceptance'] for x in selected])),
                    'counterfactual_nu_zero_mean_death_acceptance':float(np.mean([x['counterfactual_nu_zero_acceptance'] for x in selected])),
                    'number_log_odds_terms':sorted({x['number_log_odds_term'] for x in selected})})
    return {'scope':'Sparse soft-auxiliary snapshots only; reference assignments are not physical defects. Exact saved-state hypothetical deletion odds include complete reverse-density refit. nu=0 comparison changes only the evaluated auxiliary cost; it is not a simulated or equilibrium result.',
            'snapshots':snapshot_summary,'deletion_profiles':profiles,'proposal_width':float(width),'uniform_fraction':float(uniform)}


def diagnose(data,metadata,path):
    rows = data['samples']
    zero = rows[:,4]==0
    conditional = []
    for name,mask in [('P_zero',zero),('P_positive',~zero)]:
        probability = gates.conditional_label_diagnostics(data,rows[mask],
            executed_gibbs=metadata.get('label_update_protocol')=='gibbs_then_lazy_neighbor_tagged') if mask.any() else None
        conditional.append({'support_stratum':name,'clock_rows':int(mask.sum()),
            'base_Gibbs_probability':None if probability is None else float(probability['mean_conditional_probability_by_node'][int(data['base_node'])])})
    profiles = []
    for node in np.unique(rows[:,1]).astype(int):
        selected = rows[rows[:,1]==node]
        profiles.append({'node':int(node),'clocks':len(selected),
            'gamma':None if np.isposinf(data['pair_penalty'][node]) else float(data['pair_penalty'][node]),
            'hard_pair_support':bool(np.isposinf(data['pair_penalty'][node])),
            'kappa':float(data['umbrella_stiffness'][node]),
            'nu':float(data['number_stiffness'][node]),'mean_N':float(selected[:,2].mean()),
            'mean_Bragg':float(selected[:,3].mean()),'mean_overlaps':float(selected[:,4].mean()),
            'minimum_overlaps':int(selected[:,4].min()),'P_zero_frequency':float(np.mean(selected[:,4]==0))})
    stage = metadata['calibration_outcome']['stage_history'][-1]
    return {'scope':'Nonequilibrium conductance/calibration diagnostics; no phase weight or thermodynamic estimator',
            'P_zero_conditional_support':conditional,'node_profiles':profiles,
            'last_discarded_calibration_stage':stage,
            'saved_deletion_diagnostics':saved_deletion_diagnostics(data,metadata,path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    data,metadata,inputs = analysis.load_validated_run(args.input)
    output = args.output.resolve()
    sidecar = output.with_suffix('.metadata.json')
    if output.exists() or sidecar.exists():
        parser.error('Diagnostic output prefix must be new')
    result = diagnose(data,metadata,args.input)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    directory = ROOT/'data_raw/source_snapshots'
    source = directory/f'continuum_compact_hardening_diagnosis_{SOURCE_HASH}.py'
    if not source.exists():
        source.write_bytes(SOURCE_BYTES)
    elif source.read_bytes()!=SOURCE_BYTES:
        raise ValueError('Content-keyed diagnostic source collision')
    sidecar.write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':SOURCE_HASH,'source_snapshot':analysis.display_path(source),
        'dependency_sha256':{analysis.display_path(Path(module.__file__)):analysis.sha256(Path(module.__file__)) for module in (analysis,gates)},
        'input_sha256':{analysis.display_path(p):analysis.sha256(p) for p in inputs},
        'output_sha256':{analysis.display_path(output):analysis.sha256(output)},
        'reproduce_argv':['python3','-m','scripts.continuum_compact_hardening_diagnosis',analysis.display_path(args.input),'--output',analysis.display_path(output)]},indent=2)+'\n')
    print(json.dumps({'P_zero_clocks':int(np.count_nonzero(data['samples'][:,4]==0)),
                      'deletion_snapshots':len(result['saved_deletion_diagnostics']['snapshots'])}))


if __name__=='__main__':
    main()
