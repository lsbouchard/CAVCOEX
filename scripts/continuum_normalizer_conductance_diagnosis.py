#!/usr/bin/env python3
"""Read-only weight-consistency and conductance diagnosis of a saved ST path.

Necessary normalizer-root identities are not requirements on arbitrary frozen
weights, which retain the correct conditional targets. Replaying a discarded
checkpoint's weights on frozen saved hosts isolates a weight contrast; it is
not a reconstruction of the hosts observed during that adaptive checkpoint.
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
DEPENDENCIES = [(Path(m.__file__).stem, Path(m.__file__).read_bytes())
                for m in (analysis, gates)]


def target_signature(data, node):
    """Canonical target, excluding preferences with identically zero cost."""
    order = float(data['umbrella_stiffness'][node])
    number = float(data['number_stiffness'][node])
    return (float(data['activity'][node]), float(data['diameter'][node]),
            float(data['radius'][node]), float(data['pair_penalty'][node]),
            order, float(data['umbrella_preferred'][node]) if order else 0.,
            number, float(data['number_preferred'][node]) if number else 0.)


def necessary_normalizer_root_checks(data):
    """Exact target equivalence and zero-umbrella pair-penalty order."""
    adjusted = data['log_weights'] - np.log(data['calibration_label_quota'])
    duplicates, pair_groups = {}, {}
    for node in range(len(adjusted)):
        duplicates.setdefault(target_signature(data, node), []).append(node)
        if data['umbrella_stiffness'][node] == 0 and data['number_stiffness'][node] == 0:
            signature = tuple(float(data[k][node]) for k in ('activity', 'diameter', 'radius'))
            pair_groups.setdefault(signature, []).append(node)
    duplicate_rows = [{'nodes': nodes, 'quota_adjusted_log_weights': adjusted[nodes].tolist(),
                       'span': float(np.ptp(adjusted[nodes]))}
                      for nodes in duplicates.values() if len(nodes) > 1]
    pair_rows = []
    for signature, nodes in pair_groups.items():
        if len(nodes) < 2:
            continue
        violations = []
        for lower in nodes:
            for higher in nodes:
                if data['pair_penalty'][lower] < data['pair_penalty'][higher]:
                    difference = float(adjusted[higher] - adjusted[lower])
                    if difference < -1e-10:
                        violations.append({'lower_penalty_node': lower,
                                           'higher_penalty_node': higher,
                                           'adjusted_higher_minus_lower': difference})
        pair_rows.append({'activity': signature[0], 'diameter': signature[1],
                          'radius': signature[2], 'nodes': nodes,
                          'pair_penalty': [None if np.isposinf(data['pair_penalty'][j])
                                           else float(data['pair_penalty'][j]) for j in nodes],
                          'quota_adjusted_log_weights': adjusted[nodes].tolist(),
                          'violations': violations})
    return {'scope': 'Necessary identities of b=log(quota)-log(Xi)+constant, not target-law or equilibrium checks for arbitrary frozen b',
            'equal_target_identity': 'Identical target labels have identical Xi; zero-stiffness preferences do not enter their target',
            'pair_order_identity': 'At equal activity, diameter and radius with zero order/number umbrellas, increasing a nonnegative pair penalty decreases Xi, so quota-adjusted b must not decrease',
            'unknown_span_qualification': 'A positive soft-to-hard adjusted gap has the correct sign but no known value here; it must not be called a proven calibration error merely because a saved hard-compatible host has equal energies',
            'exact_duplicate_target_groups': duplicate_rows,
            'zero_umbrella_pair_penalty_groups': pair_rows}


def gibbs_retention(data, chunk_rows=5000):
    rows = data['samples']
    labels = rows[:, 1].astype(int)
    sources = data['neighbor_source_labels']
    physical = np.asarray(data['physical_nodes'], dtype=int)
    is_physical = np.zeros(len(data['activity']), dtype=bool)
    is_physical[physical] = True
    ideal = int(data['ideal_node'])
    summaries = []
    boundaries = np.linspace(0, len(rows), 5, dtype=int)
    for start, stop in [(0, len(rows)), *zip(boundaries[:-1], boundaries[1:])]:
        masses = np.zeros(3)
        expected_label_retained = 0.
        expected_physical_to_auxiliary = 0.
        for left in range(start, stop, chunk_rows):
            right = min(stop, left + chunk_rows)
            batch = rows[left:right]
            probability = gates.conditional_label_probabilities(
                data, batch[:, 2], batch[:, 3], batch[:, 4], batch[:, 7])
            physical_mass = probability[:, physical].sum(axis=1)
            masses += [physical_mass.sum(), (1-physical_mass).sum(), probability[:, ideal].sum()]
            expected_label_retained += probability[np.arange(right-left), labels[left:right]].sum()
            mask = is_physical[labels[left:right]]
            expected_physical_to_auxiliary += (1-physical_mass[mask]).sum()
        count = stop-start
        incoming_physical = is_physical[labels[start:stop]]
        outgoing_physical = is_physical[sources[start:stop]]
        summaries.append({'start_row': int(start), 'stop_row_exclusive': int(stop),
            'clock_rows': int(count), 'mean_Gibbs_physical_mass': float(masses[0]/count),
            'mean_Gibbs_auxiliary_mass': float(masses[1]/count),
            'mean_Gibbs_ideal_mass': float(masses[2]/count),
            'expected_same_label_Gibbs_retention': float(expected_label_retained/count),
            'observed_same_label_Gibbs_retention': float(np.mean(labels[start:stop] == sources[start:stop])),
            'physical_pre_Gibbs_clocks': int(incoming_physical.sum()),
            'expected_physical_to_auxiliary_Gibbs_changes': float(expected_physical_to_auxiliary),
            'observed_physical_to_auxiliary_Gibbs_changes': int(np.count_nonzero(incoming_physical & ~outgoing_physical)),
            'observed_auxiliary_to_physical_Gibbs_changes': int(np.count_nonzero(~incoming_physical & outgoing_physical))})
    return {'scope': 'Executed Gibbs conditional probabilities reconstructed on every recorded pre-Gibbs host/center, with actual intermediate labels; these hosts are not equilibrium samples',
            'full_and_quarter_profiles': summaries}


def discarded_checkpoint_replay(data, metadata, maximum_hosts=4096, last_clocks=120000):
    if not isinstance(maximum_hosts, int) or not 1 <= maximum_hosts <= 8192:
        raise ValueError('Counterfactual saved-host bound must lie in 1..8192')
    rows = data['samples']
    indices = np.unique(np.linspace(0, len(rows)-1, min(maximum_hosts, len(rows)), dtype=int))
    hosts = rows[indices]
    physical = np.asarray(data['physical_nodes'], dtype=int)
    ideal, base, cavity = (int(data[k]) for k in ('ideal_node', 'base_node', 'cavity_node'))
    quota = data['calibration_label_quota']
    history = metadata['calibration_outcome']['stage_history']
    final_clock = history[-1]['end_clock']
    records = []
    previous = None
    for stage in history:
        if stage['end_clock'] <= final_clock-last_clocks:
            previous = stage
            continue
        b = np.asarray(stage['log_weights'])
        target = {**data, 'log_weights': b}
        probability = gates.conditional_label_probabilities(
            target, hosts[:, 2], hosts[:, 3], hosts[:, 4], hosts[:, 7])
        physical_mass = probability[:, physical].sum(axis=1)
        same_stage = previous is not None and previous['stage'] == stage['stage']
        previous_counts = (np.asarray(previous['weight_update_visits']) if same_stage
                           else np.zeros_like(np.asarray(stage['weight_update_visits'])))
        counts = np.asarray(stage['weight_update_visits']) - previous_counts
        elapsed = (stage['end_clock']-previous['end_clock'] if previous is not None
                   else stage['end_clock'])
        if np.any(counts < 0) or counts.sum() != elapsed:
            raise ValueError('Per-checkpoint updated-label increments disagree with clock interval')
        pre = stage.get('normalizer_projection', {}).get('pre_projection_log_weights')
        records.append({'clock': int(stage['end_clock']), 'stage': int(stage['stage']),
            'gain': float(stage['gain']), 'interval_clocks': int(elapsed),
            'interval_updated_physical_fraction': float(counts[physical].sum()/elapsed),
            'requested_physical_quota': float(quota[physical].sum()),
            'interval_updated_ideal_fraction': float(counts[ideal]/elapsed),
            'requested_ideal_quota': float(quota[ideal]),
            'counterfactual_mean_Gibbs_physical_mass': float(physical_mass.mean()),
            'counterfactual_mean_Gibbs_auxiliary_mass': float((1-physical_mass).mean()),
            'counterfactual_mean_Gibbs_ideal_mass': float(probability[:, ideal].mean()),
            'base_minus_unbiased_gamma25_node72': float(b[base]-b[72]),
            'base_minus_cavity': float(b[base]-b[cavity]),
            'ideal_duplicate0_minus88_quota_adjusted': float(b[ideal]-np.log(quota[ideal])-b[88]+np.log(quota[88])),
            'projection_physical_vector_L2_change': (float(np.linalg.norm(b[physical]-np.asarray(pre)[physical]))
                                                    if pre is not None else None),
            'weighted_cumulative_flatness': float(stage['flatness_minimum_to_mean']),
            'half_stability_passed': bool(stage['dense_half_stability_passed'])})
        previous = stage
    return {'scope': 'DISCARDED adaptation only. Each checkpoint POST vector is applied to the SAME deterministically selected frozen-production hosts. Those counterfactual probabilities isolate weights; they are not the actual adaptive Gibbs move probabilities or thermodynamic estimates',
            'host_indices': indices.tolist(), 'host_count': len(indices),
            'recent_clocks': last_clocks, 'checkpoint_profiles': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-hosts', type=int, default=4096)
    args = parser.parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix('.metadata.json')
    if output.exists() or sidecar.exists():
        parser.error('Diagnostic output prefix must be new')
    data, metadata, inputs = analysis.load_validated_run(args.input)
    if 'neighbor_source_labels' not in data:
        parser.error('This diagnostic requires executed Gibbs and aligned intermediate labels')
    if len(data['activity']) != 162 or int(data['base_node']) != 70 or int(data['cavity_node']) != 71:
        parser.error('Checkpoint-specific route indices in this diagnostic require the declared 162-node native compact H1 graph')
    result = {'scope': 'Nonequilibrium fixed-weight path and discarded efficiency-weight diagnosis, not physical phase odds',
              'normalizer_root_checks': necessary_normalizer_root_checks(data),
              'executed_Gibbs_retention': gibbs_retention(data),
              'late_discarded_weight_replay': discarded_checkpoint_replay(data, metadata, args.maximum_hosts)}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    snapshots = {}
    for name, source_bytes in [('continuum_normalizer_conductance_diagnosis', SOURCE_BYTES)]+DEPENDENCIES:
        digest = hashlib.sha256(source_bytes).hexdigest()
        source = ROOT/'data_raw/source_snapshots'/f'{name}_{digest}.py'
        if source.exists() and source.read_bytes() != source_bytes:
            raise ValueError('Content-addressed source collision')
        if not source.exists():
            source.write_bytes(source_bytes)
        snapshots[analysis.display_path(source)] = digest
    sidecar.write_text(json.dumps({'generated_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(SOURCE_BYTES).hexdigest(), 'source_snapshots': snapshots,
        'input_sha256': {analysis.display_path(p): analysis.sha256(p) for p in inputs},
        'output_sha256': {analysis.display_path(output): analysis.sha256(output)},
        'reproduce_argv': ['python3', '-m', 'scripts.continuum_normalizer_conductance_diagnosis',
                          analysis.display_path(args.input), '--output', analysis.display_path(output),
                          '--maximum-hosts', str(args.maximum_hosts)]}, indent=2)+'\n')
    print(json.dumps({'rows': len(data['samples']), 'duplicate_groups': len(result['normalizer_root_checks']['exact_duplicate_target_groups']),
                      'late_checkpoints': len(result['late_discarded_weight_replay']['checkpoint_profiles'])}))


if __name__ == '__main__':
    main()
