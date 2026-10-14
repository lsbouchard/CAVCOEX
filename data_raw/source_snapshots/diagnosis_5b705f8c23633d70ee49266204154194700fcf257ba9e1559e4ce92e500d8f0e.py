#!/usr/bin/env python3
"""Read-only single-host regeneration/weight/radius transport diagnosis.

None of these path-conditioned counts or hypothetical weights are equilibrium
estimates. In particular, a tagged deletion followed by an ordinary or Gibbs
radius return is distinguished from an immediate reverse tagged insertion.
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

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_HASH = hashlib.sha256(SOURCE_BYTES).hexdigest()


def radius_excursions(samples, source_labels, final_label, base, cavity, path):
    """Resolve actual tagged deletions and subsequent radius/base entrances.

    Features in a trace row precede Gibbs and neighbor moves. A tag deletes
    exactly one particle before the following local update. Later observed N
    includes intervening local births/deaths and is reported as such.
    """
    labels = samples[:, 1].astype(int)
    destination = np.r_[labels[1:], final_label]
    source_labels = np.asarray(source_labels)
    if source_labels.shape != labels.shape:
        raise ValueError('Pre-neighbor labels must match full-clock rows')
    deleted = np.flatnonzero((source_labels == base) & (destination == cavity))
    inserted = np.flatnonzero((source_labels == cavity) & (destination == base))
    intermediate = set(map(int, path[1:-1]))
    rows = []
    for start in deleted:
        encountered = set()
        outcome, end, mechanism = 'unfinished', None, None
        for t in range(start + 1, len(samples)):
            # The observation is before both label kernels. A reservoir refresh
            # here destroys the incoming host, so stop before its later hops.
            if samples[t, 11] == 1:
                outcome, end = 'ideal_regeneration_before_base', t
                break
            if labels[t] == base:
                outcome, end = 'base_return', t
                mechanism = ('reverse_tagged' if source_labels[t - 1] == cavity
                             else 'ordinary_radius' if source_labels[t - 1] in intermediate
                             else 'other_or_Gibbs')
                break
            if labels[t] in intermediate:
                encountered.add(int(labels[t]))
            if source_labels[t] in intermediate:
                encountered.add(int(source_labels[t]))
            if source_labels[t] == base:
                # The Gibbs stage returned to base before the neighbor stage.
                # Its host has the unchanged current trace-row particle count.
                outcome, end, mechanism = 'base_return', t, 'Gibbs'
                break
        row = {'tag_clock': int(samples[start, 0]),
               'N_before_tag': int(samples[start, 2]),
               'N_immediately_after_tag': int(samples[start, 2]) - 1,
               'outcome': outcome, 'return_mechanism': mechanism,
               'intermediate_radius_nodes_encountered': sorted(encountered)}
        if end is not None:
            row.update(end_clock=int(samples[end, 0]),
                       subsequent_local_clock_steps=int(end - start),
                       observed_N_at_end=int(samples[end, 2]),
                       observed_N_change_since_pre_tag=int(samples[end, 2] - samples[start, 2]))
        rows.append(row)
    return {'scope': 'Exact tagged number change; subsequent observed changes include local kernels. Source labels are post-Gibbs/pre-neighbor, not observation labels.',
            'tagged_deletions': int(len(deleted)), 'tagged_insertions': int(len(inserted)),
            'deletion_excursions': rows}


def conditional_summary(target, rows, chunk_rows=5000):
    result = gates.conditional_label_diagnostics(target, rows, chunk_rows)
    probabilities = np.asarray(result['mean_conditional_probability_by_node'])
    base, cavity, ideal = (int(target[name]) for name in ('base_node', 'cavity_node', 'ideal_node'))
    dense = ((target['activity'] >= 12) & np.isposinf(target['pair_penalty'])
             & (target['umbrella_stiffness'] == 0) & (target['number_stiffness'] == 0))
    return {'clock_rows': len(rows), 'N_support': np.unique(rows[:, 2]).astype(int).tolist(),
            'ideal_refreshes': int(rows[:, 11].sum()),
            'observed_base_dwells': int(np.count_nonzero(rows[:, 1] == base)),
            'observed_cavity_dwells': int(np.count_nonzero(rows[:, 1] == cavity)),
            'mean_Gibbs_ideal_probability': float(probabilities[ideal]),
            'mean_Gibbs_base_probability': float(probabilities[base]),
            'mean_Gibbs_cavity_probability': float(probabilities[cavity]),
            'mean_Gibbs_dense_target_probability': float(probabilities[dense].sum()),
            'conditional_by_number': result['conditional_probability_by_particle_number']}


def weight_secants(target, weights):
    hard = np.flatnonzero((target['radius'] == 0) & np.isposinf(target['pair_penalty'])
                         & (target['umbrella_stiffness'] == 0) & (target['number_stiffness'] == 0))
    hard = hard[np.argsort(target['activity'][hard])]
    if np.any(np.diff(target['activity'][hard]) <= 0):
        raise ValueError('Weight-secant diagnostics require distinct hard R=0 activities')
    secants = -np.diff(weights[hard]) / np.diff(target['activity'][hard])
    return {'hard_nodes': hard.tolist(), 'adjacent_neutral_particle_counts_without_degree': secants.tolist(),
            'concavity_violations': int(np.count_nonzero(np.diff(secants) < -1e-10)),
            'out_of_cap_counts': int(np.count_nonzero((secants < 0) | (secants > int(target['cap'])))),
            'qualification': 'For exact b=-logXi, secants are nondecreasing and in [0,cap]. Violations diagnose efficiency guesses, not the validity of the physical conditional target.'}


def diagnose(target, metadata):
    rows = target['samples']
    boundaries = np.linspace(0, len(rows), 5, dtype=int)
    quarters = [conditional_summary(target, rows[a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])]
    events = np.flatnonzero(rows[:, 11] == 1)
    leading = rows[:events[0]] if len(events) else rows
    after = rows[events[0]:] if len(events) else rows[:0]
    endpoint = np.isin(rows[:, 1], [int(target['base_node']), int(target['cavity_node'])])
    endpoint_indices = np.flatnonzero(endpoint)
    regeneration = {'ideal_refreshes': int(len(events)),
                    'base_cavity_dwells': int(endpoint.sum()),
                    'leading_partial_base_cavity_dwells': int(np.isin(leading[:, 1], [target['base_node'], target['cavity_node']]).sum()),
                    'first_refresh_clock': int(rows[events[0], 0]) if len(events) else None,
                    'last_endpoint_clock': int(rows[endpoint_indices[-1], 0]) if len(endpoint_indices) else None,
                    'N_max_after_first_refresh': int(after[:, 2].max()) if len(after) else None,
                    'base_cavity_dwells_after_first_refresh': int(np.isin(after[:, 1], [target['base_node'], target['cavity_node']]).sum()) if len(after) else None}
    history = metadata['calibration_outcome'].get('stage_history', [])
    last_stage = history[-1]['stage'] if history else None
    snapshots = [entry for entry in history if entry['stage'] == last_stage]
    calibration = {'status': metadata['calibration_outcome'],
                   'frozen_weight_secants': weight_secants(target, target['log_weights'])}
    if snapshots:
        late = snapshots[len(snapshots) // 2:]
        mean_weights = np.mean([entry['log_weights'] for entry in late], axis=0)
        mean_weights -= mean_weights[int(target['ideal_node'])]
        calibration['late_adaptive_snapshot_average_secants'] = weight_secants(target, mean_weights)
        calibration['late_snapshot_count'] = len(late)
        calibration['terminal_minus_late_average_weights'] = (target['log_weights'] - mean_weights).tolist()
        # A hypothetical replacement on the same recorded path is not a new
        # target estimate or forecast of its subsequently sampled host states.
        hypothetical = dict(target, log_weights=mean_weights)
        calibration['same_path_hypothetical_late_average_weights'] = [
            conditional_summary(hypothetical, rows[a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])]
    return {'scope': 'Path-conditioned diagnostics only; no equilibrium estimate, calibration modification or MC run',
            'quarters': quarters, 'regeneration_and_dense_return': regeneration,
            'radius_number_transport': radius_excursions(rows, target['neighbor_source_labels'],
                int(target['final_label']), int(target['base_node']), int(target['cavity_node']), target['radius_path_nodes']),
            'calibration_weight_diagnostics': calibration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, metadata, inputs = analysis.load_validated_run(args.input)
    with np.load(args.input, allow_pickle=False) as archive:
        data['final_label'] = int(archive['final_label'])
    output = args.output.resolve()
    sidecar = output.with_suffix('.metadata.json')
    if output.exists() or sidecar.exists():
        parser.error('Diagnostic output prefix must be new')
    result = diagnose(data, metadata)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    snapshots = ROOT / 'data_raw/source_snapshots'
    snapshots.mkdir(exist_ok=True)
    captured = {}
    for label, source, digest in [('diagnosis', SOURCE_BYTES, SOURCE_HASH),
                                 ('gates', gates.EXECUTION_SOURCE_BYTES, gates.EXECUTION_SOURCE_SHA256),
                                 ('analysis', Path(analysis.__file__).read_bytes(), analysis.sha256(Path(analysis.__file__)))]:
        path = snapshots / f'{label}_{digest}.py'
        if path.exists() and path.read_bytes() != source:
            raise ValueError('Source snapshot hash collision')
        if not path.exists():
            path.write_bytes(source)
        captured[label] = [analysis.display_path(path), digest]
    sidecar.write_text(json.dumps({'generated_utc': datetime.now(timezone.utc).isoformat(),
        'script': 'scripts/continuum_regeneration_diagnosis.py', 'script_sha256': SOURCE_HASH,
        'source_snapshots': captured,
        'input_sha256': {analysis.display_path(path): analysis.sha256(path) for path in inputs},
        'output_sha256': {analysis.display_path(output): analysis.sha256(output)},
        'reproduce_argv': ['python3', '-m', 'scripts.continuum_regeneration_diagnosis',
                          analysis.display_path(args.input), '--output', analysis.display_path(output)]}, indent=2) + '\n')
    print(json.dumps(result['regeneration_and_dense_return']))


if __name__ == '__main__':
    main()
