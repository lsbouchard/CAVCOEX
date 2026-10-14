"""Lean diagnostics retain scope/provenance without relabeling or pooling."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import summarize_continuum_diagnostics as summary


def record():
    return {'comparison_context':{'raw_sha256':'raw hash','sampler_source_sha256':'source hash',
                                 'settings':{'restart':'parent.checkpoint.npz'},
                                 'displacement_proposal_scales':[.1,.2]},
            'raw_file':'data_raw/diagnostic.npz','estimates':{'mean_N':31.,'work_stage':None},
            'block_bootstrap_SE':{'mean_N':.02},
            'mixing':{'unbiased_hard_phase_bank':{'ordered_high_density_inventory':{
                'sampled_count_trace':[1]*10000,'production_quarter_means':[1,1,1,1],
                'support':[1],'sampled_count_changes':0}},
                      'pair_overlap_profiles':{'long':'omitted'},
                      'product_target':{'failure_reasons':['Sparse joint tails'],
                          'passes_all_reported_diagnostics':False,
                          'occupancy_support_and_visits':{'large':'omitted'}}},
            'independent_start_comparisons':{}}


def test_compact_summary_retains_lineage_and_boundaries_not_traces_or_pooled_means(tmp_path):
    first = record()
    second = copy.deepcopy(first)
    second['estimates']['mean_N'] = 28.
    original = copy.deepcopy(first)
    source = tmp_path/'detail.json'
    source.write_text(json.dumps({'one':first,'two':second}))
    result,metadata_path = summary.summarize([source],tmp_path/'summary.json')
    assert first == original
    assert result['histories']['one']['estimates']['mean_N'] == 31.
    assert result['histories']['two']['estimates']['mean_N'] == 28.
    assert 'equilibrium phase weights' in result['scope']
    assert 'never pooled' in result['uncertainty_scope']
    assert 'sampled_count_trace' not in json.dumps(result)
    assert result['histories']['one']['comparison_context']['settings']['restart'] == 'parent.checkpoint.npz'
    assert result['histories']['one']['comparison_context']['raw_sha256'] == 'raw hash'
    assert result['histories']['one']['estimates']['work_stage'] is None
    assert result['histories']['one']['diagnostics']['product_target'] == {
        'failure_reasons':['Sparse joint tails'],'passes_all_reported_diagnostics':False}
    metadata = json.loads(metadata_path.read_text())
    assert metadata['input_sha256'][str(source)] == summary.sha256(source)
    assert metadata['output_sha256'][str(tmp_path/'summary.json')] == summary.sha256(tmp_path/'summary.json')


def test_compact_summary_cannot_overwrite_inputs(tmp_path):
    source = tmp_path/'detail.json'
    source.write_text(json.dumps({'one':record()}))
    before = source.read_bytes()
    with pytest.raises(ValueError,match='not an analysis input'):
        summary.summarize([source],source)
    assert source.read_bytes() == before


def test_duplicate_history_is_not_silently_replaced(tmp_path):
    source = tmp_path/'detail.json'
    source.write_text(json.dumps({'one':record()}))
    with pytest.raises(ValueError,match='Duplicate history'):
        summary.summarize([source,source],tmp_path/'summary.json')
    assert not (tmp_path/'summary.json').exists()


def test_lineage_transport_retains_counts_hashes_and_scope_without_full_maps(tmp_path):
    source = tmp_path/'detail.json'
    source.write_text(json.dumps({'one':record()}))
    transport = tmp_path/'transport.json'
    original = {'segments':[{'raw_sha256':'raw hash','local_to_lineage_walker_id':[0,1],
                             'first_saved_cumulative_sweep':10}],
                'stitched_transport':{'ideal_to_base':[1,2,1]},
                'scope':'not pooled thermodynamic estimates'}
    transport.write_text(json.dumps(original))
    result,metadata = summary.summarize([source],tmp_path/'summary.json',[transport])
    reduced = result['lineage_transport']['transport']
    assert reduced['stitched_transport'] == original['stitched_transport']
    assert reduced['segments'][0] == {'raw_sha256':'raw hash','first_saved_cumulative_sweep':10}
    assert json.loads(transport.read_text()) == original
    assert str(transport) in json.loads(metadata.read_text())['input_sha256']
    original['segments'][0]['raw_sha256'] = 'unknown'
    transport.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='absent from diagnostic summary'):
        summary.summarize([source],tmp_path/'other.json',[transport])


def test_compact_gate_summary_keeps_conditioning_warning_and_rejects_unrelated_raw(tmp_path):
    source = tmp_path/'detail.json'
    source.write_text(json.dumps({'one':record()}))
    gate = tmp_path/'gate.json'
    value = {'raw_sha256':['raw hash'],'qualification':'Sparse saved-frame ordering only',
             'hypothetical_saved_state_swap_gates':[{'edge':[0,1],'saved_pairs':20,
                'mean_hypothetical_acceptance':.5,'both_cross_support_fraction':1.,
                'strata':{'long':'omitted'}}]}
    gate.write_text(json.dumps(value))
    result,metadata = summary.summarize([source],tmp_path/'summary.json',gate_inputs=[gate])
    compact = result['phase_number_gate_diagnostics']['gate']
    assert compact['qualification'] == value['qualification']
    assert compact['saved_state_swap_gate_summary'][0]['mean_hypothetical_acceptance'] == .5
    assert 'strata' not in json.dumps(compact)
    assert str(gate) in json.loads(metadata.read_text())['input_sha256']
    value['raw_sha256'] = ['unrelated']
    gate.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='summarized raw histories'):
        summary.summarize([source],tmp_path/'other.json',gate_inputs=[gate])
