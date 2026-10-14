#!/usr/bin/env python3
"""Make a compact, explicitly diagnostic summary without copying raw histories.

Detailed analyzer inputs and raw trajectories belong to the full audit bundle.
The lean code release carries this summary, its provenance, and all controls;
none of these exploratory histories is asserted to be an equilibrium sample.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'data_processed/continuum_diagnostic_summary.json'
EXECUTION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FIELDS = (
    'mean_N','mean_N_inserted','q_solid_base','q_solid_inserted','Fano_factor',
    'response_direct','response_Fano','direct_minus_Palm_N','direct_minus_Palm_q',
    'endpoint_N_covariance','joint_phase_residual_SS',
    'deletion_label_change_probability','work_stage','work_GNZ',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def inventory_summary(record):
    """Retain moments/coverage while excluding potentially long frame traces."""
    keys = ('support','sampled_count_changes','production_quarter_means',
            'first_half_mean','second_half_mean','mean','standard_deviation')
    return {key:deepcopy(record[key]) for key in keys if key in record}


def compact_record(record):
    context = deepcopy(record['comparison_context'])
    scales = context.pop('displacement_proposal_scales',None)
    context['displacement_proposal_sha256'] = (hashlib.sha256(
        json.dumps(scales,separators=(',',':')).encode()).hexdigest() if scales is not None else None)
    mixing = record['mixing']
    diagnostics = {key:deepcopy(mixing[key]) for key in (
        'physical_phase_transitions_low_to_high_high_to_low',
        'inserted_phase_transitions_low_to_high_high_to_low',
        'ideal_reservoir_node','ideal_to_unconstrained_round_trips',
        'ideal_to_inserted_round_trips','transport_sampling_scope','block_sweeps',
        'bootstrap_blocks','bootstrap_draws','bootstrap_discarded_samples',
        'boundary_fraction_uncertainty','independent_identity_checks','palm_transfer',
    ) if key in mixing}
    bank = mixing.get('unbiased_hard_phase_bank',{})
    diagnostics['unbiased_hard_phase_bank'] = {
        key:inventory_summary(bank[key]) for key in (
            'operational_solid_inventory','ordered_high_density_inventory','host_number_inventory'
        ) if key in bank}
    probes = mixing.get('physical_activity_probes',{})
    diagnostics['physical_activity_transport'] = {
        key:deepcopy(value) for key,value in probes.items() if 'round_trip' in key}
    product = mixing.get('product_target',{})
    diagnostics['product_target'] = {key:deepcopy(product[key]) for key in (
        'state_order','joint_phase_visits','empirical_product_expected_sample_counts',
        'both_endpoints_visit_both_basins','missing_cells_with_positive_empirical_product',
        'strict_joint_inventory_coverage_failure','insufficient_joint_state_visits',
        'product_target_residual_checks','joint_number_support','failure_reasons',
        'passes_all_reported_diagnostics','interpretation',
    ) if key in product}
    return {
        'scope':'Exploratory diagnostic history, not an equilibrium phase-weight measurement',
        'raw_file_in_full_audit_bundle':record['raw_file'],
        'comparison_context':context,
        'estimates':{key:record['estimates'][key] for key in FIELDS if key in record['estimates']},
        'within_history_block_bootstrap_SE':{
            key:record['block_bootstrap_SE'][key] for key in FIELDS if key in record['block_bootstrap_SE']},
        'diagnostics':diagnostics,
    }


def summarize(inputs,output=DEFAULT_OUTPUT,transport_inputs=(),gate_inputs=()):
    inputs = [Path(path).resolve() for path in inputs]
    transport_inputs = [Path(path).resolve() for path in transport_inputs]
    gate_inputs = [Path(path).resolve() for path in gate_inputs]
    output = Path(output).resolve()
    metadata = (ROOT/'generated/continuum_diagnostic_summary_metadata.json'
                if output == DEFAULT_OUTPUT.resolve() else output.with_suffix('.metadata.json'))
    all_inputs = inputs+transport_inputs+gate_inputs
    if output.suffix != '.json' or output in all_inputs or metadata in all_inputs:
        raise ValueError('Summary must be a new JSON output, not an analysis input')
    histories,comparisons = {},{}
    for path in inputs:
        records = json.loads(path.read_text())
        for name,record in records.items():
            if name in histories:
                raise ValueError(f'Duplicate history name across inputs: {name}')
            histories[name] = compact_record(record)
            for other,comparison in record.get('independent_start_comparisons',{}).items():
                key = ' / '.join(sorted((name,other)))
                if key in comparisons and comparisons[key] != comparison:
                    raise ValueError(f'Conflicting comparison for {key}')
                comparisons[key] = comparison
    if not histories:
        raise ValueError('At least one analyzed diagnostic history is required')
    transport = {}
    known = {value['comparison_context']['raw_sha256'] for value in histories.values()}
    for path in transport_inputs:
        record = json.loads(path.read_text())
        if path.stem in transport:
            raise ValueError('Duplicate transport lineage name')
        if not record.get('segments') or 'stitched_transport' not in record:
            raise ValueError('Transport input must be a validated lineage artifact')
        if not {segment['raw_sha256'] for segment in record['segments']} <= known:
            raise ValueError('Transport lineage contains histories absent from diagnostic summary')
        transport[path.stem] = deepcopy(record)
        # Exact identity maps remain in the full diagnostic artifact. The lean
        # summary retains lineage hashes, time offsets and transport counts.
        for segment in transport[path.stem]['segments']:
            segment.pop('local_to_lineage_walker_id',None)
    gates = {}
    for path in gate_inputs:
        record = json.loads(path.read_text())
        if path.stem in gates or not record.get('raw_sha256') or not set(record['raw_sha256']) <= known:
            raise ValueError('Gate artifact must be unique and use only summarized raw histories')
        gates[path.stem] = {key:deepcopy(record[key]) for key in (
            'scope','qualification','raw_sha256','transport_scope','saved_frames',
            'physical_nodes','reservoir_nodes','physical_departure_excursions',
            'last_recorded_source_to_physical_arrivals',
            'observed_changed_physical_Bragg_stratum_after_source_visit') if key in record}
        gates[path.stem]['saved_state_swap_gate_summary'] = [
            {key:row[key] for key in ('edge','saved_pairs','mean_hypothetical_acceptance',
                                      'both_cross_support_fraction')}
            for row in record.get('hypothetical_saved_state_swap_gates',[])]
    result = {
        'scope':'Diagnostic-only; no dense-continuum equilibrium phase weights are asserted',
        'data_distribution':'Detailed analysis inputs, raw trajectories, checkpoints and executed-source snapshots are retained in the full audit reproducibility bundle, not the lean code checkout',
        'uncertainty_scope':'Block standard errors describe variation within each visited history, not unvisited phases, equilibration, or independence of restart segments. Histories are never pooled',
        'histories':histories,'recorded_comparisons':comparisons,
        'lineage_transport':transport,
        'phase_number_gate_diagnostics':gates,
    }
    encoded = json.dumps(result,indent=2,allow_nan=False)+'\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    metadata.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(encoded)
    metadata.write_text(json.dumps({
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'script':'scripts/summarize_continuum_diagnostics.py',
        'script_sha256':EXECUTION_SOURCE_SHA256,'python':sys.version,
        'input_sha256':{display_path(path):sha256(path) for path in all_inputs},
        'input_distribution':'Inputs are in the full audit bundle; omission from the lean checkout is intentional',
        'output_sha256':{display_path(output):sha256(output)},
        'reproduce_argv':['python3','scripts/summarize_continuum_diagnostics.py',
                          *map(display_path,inputs),'--output',display_path(output),
                          *(['--transport',*map(display_path,transport_inputs)] if transport_inputs else []),
                          *(['--gates',*map(display_path,gate_inputs)] if gate_inputs else [])],
    },indent=2)+'\n')
    return result,metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--transport',nargs='+',type=Path,default=[])
    parser.add_argument('--gates',nargs='+',type=Path,default=[])
    args = parser.parse_args()
    try:
        result,metadata = summarize(args.inputs,args.output,args.transport,args.gates)
    except (ValueError,KeyError,FileNotFoundError) as error:
        parser.error(str(error))
    print(f'Summarized {len(result["histories"])} separate diagnostic histories; provenance: {metadata}')


if __name__ == '__main__':
    main()
