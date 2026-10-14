#!/usr/bin/env python3
"""Full-clock pre-hop conductance diagnostics, not equilibrium estimates.

Each recorded row is the actual state immediately before the lazy neighbor
proposal.  Consequently its MH score is reconstructible.  Averaging that
score diagnoses the fixed label weights and configuration support; it does
not establish the correct conditional thermodynamic distribution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

try:
    from . import continuum_single_walker_analyze as analysis
except ImportError:
    import continuum_single_walker_analyze as analysis

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SOURCE_BYTES = Path(__file__).read_bytes()
EXECUTION_SOURCE_SHA256 = hashlib.sha256(EXECUTION_SOURCE_BYTES).hexdigest()
DEPENDENCY_SOURCE_SHA256 = analysis.sha256(Path(analysis.__file__))


def neighbor_degrees(edges, nodes):
    edges = np.asarray(edges)
    if (edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu'
            or np.any(edges < 0) or np.any(edges >= nodes)
            or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('Edges must join distinct valid integer nodes')
    canonical = {tuple(sorted(map(int, pair))) for pair in edges}
    if len(canonical) != len(edges):
        raise ValueError('Duplicate undirected edges are not supported')
    degrees = np.bincount(edges.ravel(), minlength=nodes)
    if np.any(degrees == 0):
        raise ValueError('Every target needs a neighbor')
    return degrees


def reduced_density(target, node, n, b, overlaps, distance):
    """Reduced target log density; avoid infinity times zero at hard nodes."""
    penalty = target['pair_penalty'][node]
    support = ((n == 0) | (distance >= target['radius'][node]))
    if np.isposinf(penalty):
        support &= overlaps == 0
        pair = np.zeros_like(n, dtype=float)
    else:
        pair = penalty * overlaps
    energy = (pair + target['umbrella_stiffness'][node]
              * (b - target['umbrella_preferred'][node]) ** 2
              + target['number_stiffness'][node]
              * (n - target['number_preferred'][node]) ** 2)
    return support, target['activity'][node] * n - energy


def directional_score(target, source, destination, n, b, overlaps, distance, degrees):
    """Exact lazy-neighbor acceptance at a recorded pre-hop state."""
    if not np.all(target['diameter'] == target['diameter'][0]):
        raise ValueError('Recorded overlap counts require a common diameter')
    base, cavity = int(target['base_node']), int(target['cavity_node'])
    own, old = reduced_density(target, source, n, b, overlaps, distance)
    if not np.all(own):
        raise ValueError('Pre-hop state violates its current target support')
    logratio = np.full(len(n), target['log_weights'][destination]
                       - target['log_weights'][source]
                       + np.log(degrees[source] / degrees[destination]))
    tagged = {source, destination} == {base, cavity}
    if tagged:
        for node in (base, cavity):
            if (not np.isposinf(target['pair_penalty'][node])
                    or target['umbrella_stiffness'][node] != 0
                    or target['number_stiffness'][node] != 0):
                raise ValueError('Tagged endpoint targets must be unbiased and hard')
        if (target['activity'][base] != target['activity'][cavity]
                or target['radius'][base] != 0
                or target['radius'][cavity] != target['diameter'][base]):
            raise ValueError('Tagged endpoint activities and radii must match')
        if source == base:
            support = n > 0
            logratio[support] += (np.log(n[support]) - target['activity'][source]
                                  - np.log(float(target['volume'])))
        else:
            support = n < int(target['cap'])
            logratio[support] += (target['activity'][source]
                                  + np.log(float(target['volume'])) - np.log(n[support] + 1))
    else:
        support, new = reduced_density(target, destination, n, b, overlaps, distance)
        logratio += new - old
    probability = np.zeros(len(n))
    probability[support] = np.exp(np.minimum(0., logratio[support]))
    return probability, support, logratio, tagged


def bragg_strata(n, b):
    """Descriptive dense low/high reciprocal-amplitude strata, not phases."""
    labels = np.full(len(n), 'intermediate_or_dilute', dtype=object)
    labels[(n >= 20) & (b <= .3)] = 'low_Bragg'
    labels[(n >= 20) & (b >= .65)] = 'high_Bragg'
    return labels


def conditional_label_probabilities(target, n, b, overlaps, distance):
    """Exact label conditional at fixed host/center, with no degree factor."""
    logits = np.empty((len(n), len(target['activity'])))
    for node in range(logits.shape[1]):
        support, reduced = reduced_density(target, node, n, b, overlaps, distance)
        logits[:, node] = target['log_weights'][node] + reduced
        logits[~support, node] = -np.inf
    maximum = logits.max(axis=1)
    if not np.isfinite(maximum).all():
        raise ValueError('Every conditional label draw needs at least one supported target')
    probability = np.exp(logits - maximum[:, None])
    probability /= probability.sum(axis=1)[:, None]
    return probability


def conditional_label_diagnostics(target, samples, chunk_rows=5000):
    """Hypothetical global Gibbs probabilities on the recorded pre-hop path."""
    if not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError('Conditional-label chunks must be positive integers')
    nodes = len(target['activity'])
    total = np.zeros(nodes)
    by_number = {}
    for start in range(0, len(samples), chunk_rows):
        rows = samples[start:start + chunk_rows]
        probability = conditional_label_probabilities(
            target, rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 7])
        total += probability.sum(axis=0)
        for number in np.unique(rows[:, 2]).astype(int):
            mask = rows[:, 2] == number
            entry = by_number.setdefault(number, {'dwells': 0, 'probability_sum': np.zeros(nodes)})
            entry['dwells'] += int(mask.sum())
            entry['probability_sum'] += probability[mask].sum(axis=0)
    profiles = []
    for number, entry in sorted(by_number.items()):
        probabilities = entry['probability_sum'] / entry['dwells']
        profiles.append({'N': int(number), 'pre_hop_dwells': entry['dwells'],
            'mean_conditional_probability_by_node': probabilities.tolist(),
            'ideal_probability': float(probabilities[int(target['ideal_node'])]),
            'base_probability': float(probabilities[int(target['base_node'])]),
            'cavity_probability': float(probabilities[int(target['cavity_node'])])})
    return {'scope': 'Hypothetical Gibbs label refresh BEFORE the recorded neighbor/tagged hop, not an executed kernel',
            'qualification': 'No adjacency degree factor enters an exact conditional label draw. This diagnostic cannot forecast a post-tag Gibbs move using stale N/order/center, and it cannot establish stationarity of the recorded hosts.',
            'mean_conditional_probability_by_node': (total / len(samples)).tolist(),
            'conditional_probability_by_particle_number': profiles}


def summarize_gate(target, samples, final_label=None, neighbor_source_labels=None):
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != 12 or len(samples) < 2:
        raise ValueError('A nonempty full-clock single-host trace is required')
    if np.any(np.diff(samples[:, 0]) != 1):
        raise ValueError('Directional accepted hops require consecutive clock rows')
    nodes = len(target['activity'])
    labels = samples[:, 1].astype(int)
    if np.any(samples[:, 1] != labels) or np.any(labels < 0) or np.any(labels >= nodes):
        raise ValueError('Trace labels must be valid integers')
    degrees = neighbor_degrees(target['exchange_edges'], nodes)
    next_labels = np.r_[labels[1:], -1 if final_label is None else final_label]
    hop_labels = labels if neighbor_source_labels is None else np.asarray(neighbor_source_labels)
    if (hop_labels.shape!=labels.shape or hop_labels.dtype.kind not in 'iu'
            or np.any(hop_labels<0) or np.any(hop_labels>=nodes)):
        raise ValueError('Compound updates require aligned integer pre-neighbor labels')
    adjacency = {tuple(sorted(map(int, pair))) for pair in target['exchange_edges']}
    if any(tuple(sorted((int(left), int(right)))) not in adjacency
           for left, right in zip(hop_labels, next_labels) if right >= 0 and left != right):
        raise ValueError('Recorded label transition is not a graph edge')
    n, b, p, d = (samples[:, index] for index in (2, 3, 4, 7))
    rows = []
    for left, right in target['exchange_edges']:
        for source, destination in ((int(left), int(right)), (int(right), int(left))):
            chosen = hop_labels == source
            count = int(chosen.sum())
            if not count:
                rows.append({'source': source, 'destination': destination, 'pre_hop_dwells': 0,
                             'observed_accepted_hops': 0, 'expected_accepted_hops': 0.,
                             'mean_MH_acceptance_if_proposed': None, 'strata': {}})
                continue
            probability, support, logratio, tagged = directional_score(
                target, source, destination, n[chosen], b[chosen], p[chosen], d[chosen], degrees)
            observed = next_labels[chosen] == destination
            # The final pre-hop row has no following recorded label unless the
            # archive supplies its final label; exclude it only from flux comparison.
            observable = next_labels[chosen] >= 0
            proposal_probability = .5 / degrees[source]
            strata = {}
            structural = bragg_strata(n[chosen], b[chosen])
            keys = np.array([f'{state}:N{int(number)}'
                             for state, number in zip(structural, n[chosen])])
            for key in np.unique(keys):
                mask = keys == key
                strata[str(key)] = {'pre_hop_dwells': int(mask.sum()),
                    'mean_MH_acceptance_if_proposed': float(probability[mask].mean()),
                    'cross_support_fraction': float(support[mask].mean()),
                    'observed_accepted_hops': int(observed[mask].sum()),
                    'expected_accepted_hops': float((probability[mask & observable]
                                                    * proposal_probability).sum())}
            row = {'source': source, 'destination': destination, 'pre_hop_dwells': count,
                'source_degree': int(degrees[source]), 'tagged': tagged,
                'proposal_probability_per_clock_dwell': float(proposal_probability),
                'observed_accepted_hops': int(observed.sum()),
                'expected_accepted_hops': float((probability[observable] * proposal_probability).sum()),
                'mean_MH_acceptance_if_proposed': float(probability.mean()),
                'cross_support_fraction': float(support.mean()),
                'logratio_quantiles': np.quantile(logratio, [0, .25, .5, .75, 1]).tolist(),
                'strata': strata}
            delta_x = target['activity'][destination] - target['activity'][source]
            if (not tagged and delta_x != 0 and target['radius'][source] == 0
                    and target['radius'][destination] == 0
                    and np.isposinf(target['pair_penalty'][source])
                    and np.isposinf(target['pair_penalty'][destination])
                    and target['umbrella_stiffness'][[source, destination]].sum() == 0
                    and target['number_stiffness'][[source, destination]].sum() == 0):
                row['activity_edge_neutral_MH_particle_count'] = float(
                    -(target['log_weights'][destination] - target['log_weights'][source]
                      + np.log(degrees[source] / degrees[destination])) / delta_x)
            rows.append(row)
    profiles = []
    boundaries = np.linspace(0, len(samples), 5, dtype=int)
    for node in range(nodes):
        chosen = labels == node
        values = n[chosen]
        quarters = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            mask = labels[start:stop] == node
            quarters.append({'dwells': int(mask.sum()),
                             'mean_N': float(n[start:stop][mask].mean()) if mask.any() else None,
                             'mean_Bragg': float(b[start:stop][mask].mean()) if mask.any() else None})
        profiles.append({'node': node, 'activity': float(target['activity'][node]),
            'log_weight': float(target['log_weights'][node]), 'dwells': int(chosen.sum()),
            'N_support': np.unique(values).astype(int).tolist(),
            'mean_N': float(values.mean()) if len(values) else None,
            'mean_Bragg': float(b[chosen].mean()) if len(values) else None, 'quarters': quarters})
    return {'scope': 'Full-clock pre-hop label conductance only; no equilibrium or phase-weight estimate',
        'qualification': 'MH probabilities are exact for the recorded pre-proposal states. Expected hop counts condition on this nonequilibrium path, not stationary thermodynamic probabilities. Descriptive high/low Bragg strata do not identify thermodynamic phases. Missing reservoir regeneration invalidates regenerative inference even if local label flux matches its MH expectation.',
        'clock_rows': len(samples), 'ideal_heatbath_events': int(samples[:, 11].sum()),
        'neighbor_source_scope': 'post-Gibbs/pre-neighbor labels' if neighbor_source_labels is not None else 'recorded pre-hop labels',
        'gibbs_label_changes': int(np.count_nonzero(labels!=hop_labels)) if neighbor_source_labels is not None else 0,
        'node_profiles': profiles, 'directional_edges': rows,
        'hypothetical_global_label_conditional': conditional_label_diagnostics(target, samples)}


def write_diagnostics(path, output):
    path, output = Path(path).resolve(), Path(output).resolve()
    sidecar = output.with_suffix('.metadata.json')
    data, metadata, verified_inputs = analysis.load_validated_run(path)
    protected = set(verified_inputs)
    protected.update((ROOT / entry[0]).resolve() for entry in metadata['source_snapshots'].values())
    protected.update((ROOT / name).resolve() for name in metadata.get('input_sha256', {}))
    if output.suffix != '.json' or protected.intersection((output, sidecar)):
        raise ValueError('Gate output must not overwrite raw data, metadata or executed sources')
    if output.exists() or sidecar.exists():
        raise ValueError('Gate output prefix must be new')
    with np.load(path, allow_pickle=False) as archive:
        final_label = int(archive['final_label']) if 'final_label' in archive else None
        counters = archive['label_move_counts'] if 'label_move_counts' in archive else None
        intermediate = archive['neighbor_source_labels'] if 'neighbor_source_labels' in archive else None
    if metadata.get('label_update_protocol')=='gibbs_then_lazy_neighbor_tagged' and intermediate is None:
        raise ValueError('Compound label updates lack their pre-neighbor intermediate labels')
    result = summarize_gate(data, data['samples'], final_label,intermediate)
    result['input'] = analysis.display_path(path)
    if counters is not None:
        by_pair = {(row['source'], row['destination']): row['observed_accepted_hops']
                   for row in result['directional_edges']}
        reconstructed = np.array([by_pair[int(left), int(right)] + by_pair[int(right), int(left)]
                                  for left, right in data['exchange_edges']])
        if final_label is not None and not np.array_equal(reconstructed, counters[:, 1]):
            raise ValueError('Full-clock accepted transitions disagree with recorded edge counters')
        result['full_clock_accepted_hops_match_recorded_edge_counters'] = final_label is not None
        result['recorded_undirected_attempt_accept_counters'] = counters.tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    sidecar.write_text(json.dumps({'generated_utc': datetime.now(timezone.utc).isoformat(),
        'script': 'scripts/continuum_single_walker_gates.py', 'script_sha256': EXECUTION_SOURCE_SHA256,
        'dependency_sha256': {'scripts/continuum_single_walker_analyze.py': DEPENDENCY_SOURCE_SHA256},
        'input_sha256': {analysis.display_path(file): analysis.sha256(file) for file in sorted(protected)},
        'output_sha256': {analysis.display_path(output): analysis.sha256(output)},
        'reproduce_argv': ['python3', '-m', 'scripts.continuum_single_walker_gates',
                           analysis.display_path(path), '--output', analysis.display_path(output)]},
        indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_diagnostics(args.input, args.output)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({'clock_rows': result['clock_rows'],
                      'ideal_heatbath_events': result['ideal_heatbath_events'],
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
