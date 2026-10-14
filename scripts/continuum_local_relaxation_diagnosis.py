#!/usr/bin/env python3
"""Saved-path local-relaxation diagnostics, not equilibrium estimates.

Consecutive recorded labels do not include unobserved intermediate Gibbs/hop
labels. Their spell lengths are descriptive, not actual label holding times.
Auxiliary Bragg and reference assignments are not thermodynamic phase labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from . import continuum_single_walker_analyze as analysis
from . import continuum_single_walker_gates as gates
from . import continuum_compact_hardening_diagnosis as hardening
from . import continuum_structural_controls as structure

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BYTES = Path(__file__).read_bytes()
DEPENDENCY_SOURCE_BYTES = [(Path(module.__file__).stem,Path(module.__file__).read_bytes())
    for module in (analysis,gates,hardening,structure)]


def spell_lengths(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError('Labels must be one dimensional')
    if not len(labels):
        return np.zeros(0, dtype=int)
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return np.diff(np.concatenate(([0], boundaries, [len(labels)])))


def spell_summary(labels):
    spells = spell_lengths(labels)
    return {'spells':len(spells), 'mean_clocks':float(spells.mean()) if len(spells) else None,
            'median_clocks':float(np.median(spells)) if len(spells) else None,
            'maximum_clocks':int(spells.max()) if len(spells) else None,
            'one_clock_fraction':float(np.mean(spells == 1)) if len(spells) else None}


def diagnose(data, metadata, raw):
    rows = data['samples']
    labels = rows[:,1].astype(int)
    events = np.flatnonzero(rows[:,11] == 1)
    quarter_profiles = []
    for quarter, indices in enumerate(np.array_split(np.arange(len(rows)), 4)):
        selected = rows[indices]
        conditional = gates.conditional_label_diagnostics(data, selected,
            executed_gibbs=metadata.get('label_update_protocol') == 'gibbs_then_lazy_neighbor_tagged')
        masses = conditional['mean_conditional_probability_by_node']
        quarter_profiles.append({'quarter':quarter+1, 'clocks':len(indices),
            'ideal_refreshes':int(np.count_nonzero(selected[:,11])),
            'mean_N':float(selected[:,2].mean()), 'mean_Bragg':float(selected[:,3].mean()),
            'mean_overlaps':float(selected[:,4].mean()),
            'P_zero_frequency':float(np.mean(selected[:,4] == 0)),
            'base_conditional_label_probability':float(masses[int(data['base_node'])]),
            'cavity_conditional_label_probability':float(masses[int(data['cavity_node'])]),
            'ideal_conditional_label_probability':float(masses[int(data['ideal_node'])])})
    length = float(data['volume'])**(1/3)
    cells = int(round((int(data['sites'])/4)**(1/3)))
    basis = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    reference = np.array([(np.array([i,j,k])+b)*length/cells for i in range(cells)
        for j in range(cells) for k in range(cells) for b in basis])
    with np.load(raw, allow_pickle=False) as archive:
        coordinates = archive['coordinate_configurations']
    profiles = []
    for node in np.unique(data['coordinate_labels']).astype(int):
        gamma = float(data['pair_penalty'][node])
        if gamma < 6 or gamma == float('inf'):
            continue
        records = []
        for frame in np.flatnonzero(data['coordinate_labels'] == node):
            index = int(data['coordinate_clock_steps'][frame] - rows[0,0])
            n = int(rows[index,2])
            current = coordinates[frame,:n]
            _, occupied, _, score = hardening.fit_registry(current,reference,length,cells)
            q6, _, _ = structure.bond_order(current,length,1.4)
            records.append((n,rows[index,3],rows[index,4],np.count_nonzero(occupied == 0),
                np.maximum(occupied-1,0).sum(),np.sqrt(score/max(n,1)),q6))
        values = np.asarray(records)
        profiles.append({'node':int(node), 'gamma':gamma,
            'kappa':float(data['umbrella_stiffness'][node]),
            'nu':float(data['number_stiffness'][node]), 'snapshots':len(values),
            'mean_N':float(values[:,0].mean()), 'mean_Bragg':float(values[:,1].mean()),
            'P_zero_frequency':float(np.mean(values[:,2] == 0)),
            'mean_empty_reference_sites':float(values[:,3].mean()),
            'mean_excess_reference_assignments':float(values[:,4].mean()),
            'mean_registry_RMS':float(values[:,5].mean()), 'mean_global_Q6':float(values[:,6].mean()),
            'maximum_global_Q6':float(values[:,6].max())})
    return {'scope':'Descriptive nonequilibrium saved-path diagnostics; auxiliary structure is not a phase-weight estimate',
        'recorded_pre_Gibbs_label_spells':spell_summary(labels),
        'recorded_post_Gibbs_pre_neighbor_label_spells':spell_summary(data['neighbor_source_labels']),
        'spell_qualification':'Recorded consecutive labels can conceal intermediate Gibbs/neighbor excursions; spell lengths are not exact label holding times',
        'first_refresh_production_clock':int(events[0]+1) if len(events) else None,
        'last_refresh_production_clock':int(events[-1]+1) if len(events) else None,
        'quarter_profiles':quarter_profiles,
        'auxiliary_snapshot_profiles':profiles,
        'auxiliary_structure_qualification':'Reference occupancy and Q6 of biased soft hosts are descriptive only; no physical endpoint was generated by this analysis'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix('.metadata.json')
    if output.exists() or sidecar.exists():
        parser.error('Diagnostic output prefix must be new')
    data, metadata, inputs = analysis.load_validated_run(args.input)
    result = diagnose(data,metadata,args.input)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    captured = [('continuum_local_relaxation_diagnosis',SOURCE_BYTES)]
    captured += DEPENDENCY_SOURCE_BYTES
    snapshots = {}
    for name, source_bytes in captured:
        digest = hashlib.sha256(source_bytes).hexdigest()
        source = ROOT/'data_raw/source_snapshots'/f'{name}_{digest}.py'
        if source.exists() and source.read_bytes() != source_bytes:
            raise ValueError('Content-keyed diagnostic source collision')
        if not source.exists():
            source.write_bytes(source_bytes)
        snapshots[analysis.display_path(source)] = digest
    sidecar.write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':hashlib.sha256(SOURCE_BYTES).hexdigest(), 'source_snapshots':snapshots,
        'input_sha256':{analysis.display_path(p):analysis.sha256(p) for p in inputs},
        'output_sha256':{analysis.display_path(output):analysis.sha256(output)},
        'reproduce_argv':['python3','-m','scripts.continuum_local_relaxation_diagnosis',
            analysis.display_path(args.input),'--output',analysis.display_path(output)]},indent=2)+'\n')
    print(json.dumps({'last_refresh_production_clock':result['last_refresh_production_clock'],
                     'auxiliary_node_profiles':len(result['auxiliary_snapshot_profiles'])}))


if __name__ == '__main__':
    main()
