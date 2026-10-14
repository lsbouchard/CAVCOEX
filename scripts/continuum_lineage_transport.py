#!/usr/bin/env python3
"""Read-only walker-label stitching for a verified frozen restart lineage.

This joins transport labels, never thermodynamic samples or uncertainty estimates.
All counts refer only to saved frames and remain lower bounds on actual visits.
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
    from . import continuum_analyze as analysis
except ImportError:
    import continuum_analyze as analysis

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SOURCE_SHA256 = analysis.sha256(Path(__file__))
# This archived driver was inspected: every process starts identities=arange,
# restores checkpoint states into unchanged slots, and exchanges walker labels
# with the hosts on ordinary exchanges and accepted Palm transfers. Unknown
# future driver versions require the same explicit semantic review.
VERIFIED_SLOT_RESET_SOURCES = {
    "2f098b5ba9da47e6263db669211e5f1d0bf66f7eeb16a66590c8ada7886d72e7",
}
PROPOSAL_SETTINGS = (
    "sites", "cap", "palm_transfer", "biased_birth", "birth_width",
    "birth_uniform_fraction", "birth_proposal", "birth_scope",
)


def final_bragg(positions, counts, volume, sites):
    cells = round((sites/4.)**(1/3))
    modes = cells*np.array([[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],
                           [2,0,0],[0,2,0],[0,0,2]])
    length = volume**(1/3)
    return np.array([np.abs(np.exp(2j*np.pi*positions[j,:n]@modes.T/length)
                            .mean(axis=0)).mean() if n else 0.
                     for j,n in enumerate(counts)])


def load_segment(path):
    """Validate raw/state/source integrity and retain only compact diagnostics."""
    path = Path(path).resolve()
    metadata_path = path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    settings = metadata['settings']
    stride = settings['sample_stride']
    data,metadata = analysis.load_validated_run(path,stride)
    source = metadata.get('source_sha256')
    if source not in VERIFIED_SLOT_RESET_SOURCES:
        raise ValueError('Walker reset/exchange semantics have not been reviewed for this executed source')
    snapshot = (ROOT/metadata['source_snapshot']).resolve()
    if not snapshot.is_relative_to(ROOT) or analysis.sha256(snapshot) != source:
        raise ValueError('Executed-source snapshot digest is not verified')
    samples = data['samples']
    production = settings['production_sweeps']
    if production <= 0 or production % stride or len(samples)*stride != production:
        raise ValueError('Final saved frame must coincide exactly with production end')
    if 'displacement' not in data:
        raise ValueError('Every frozen displacement scale must be recorded')
    with np.load(path,allow_pickle=False) as archive:
        counts = archive['final_counts']
        positions = archive['final_positions']
    nrep = samples.shape[1]
    if (counts.shape != (nrep,) or counts.dtype.kind not in 'iu'
            or positions.shape != (nrep,settings['cap'],3)
            or np.any(counts < 0) or np.any(counts > settings['cap'])
            or not np.array_equal(counts,samples[-1,:,0])):
        raise ValueError('Final checkpoint counts must match every last sampled slot')
    length = float(data['volume'])**(1/3)
    if any(not np.isfinite(positions[j,:n]).all() or np.any(positions[j,:n] < 0)
           or np.any(positions[j,:n] >= length) for j,n in enumerate(counts)):
        raise ValueError('Final active coordinates must lie in the recorded torus')
    bragg = final_bragg(positions,counts,float(data['volume']),settings['sites'])
    if not np.allclose(bragg,samples[-1,:,1],rtol=0,atol=1e-10):
        raise ValueError('Final checkpoint geometry does not reproduce every last sampled Bragg value')
    initial_counts = np.asarray(metadata['initial_counts'])
    initial_bragg = np.asarray(metadata['initial_bragg'])
    if (initial_counts.shape != (nrep,) or initial_bragg.shape != (nrep,)
            or not np.isfinite(initial_counts).all() or not np.isfinite(initial_bragg).all()
            or np.any(initial_counts != np.rint(initial_counts))
            or np.any(initial_counts < 0) or np.any(initial_counts > settings['cap'])
            or np.any(initial_bragg < 0) or np.any(initial_bragg > 1+1e-12)):
        raise ValueError('Initial count/Bragg records must describe every finite state slot')
    penalty = data.get('pair_penalty',np.full(nrep,np.inf))
    order = data.get('umbrella_stiffness',np.zeros(nrep))
    ideal = np.flatnonzero(((data['diameter']==0)|(penalty==0))
                          &(data['radius']==0)&(order==0)&(data['number_stiffness']==0))
    endpoint = int(data['endpoint'])
    physical = np.r_[endpoint,data.get('probe_nodes',np.array([],dtype=int))]
    physical = physical[np.argsort(data['activity'][physical])]
    pairs = {'lowest_hard_to_base':(int(physical[0]),endpoint),
             'lowest_hard_to_cavity':(int(physical[0]),int(data['cavity_nodes'][-1])),
             'lowest_to_highest_hard':(int(physical[0]),int(physical[-1]))}
    if len(ideal):
        pairs.update({'ideal_to_base':(int(ideal[0]),endpoint),
                      'ideal_to_cavity':(int(ideal[0]),int(data['cavity_nodes'][-1]))})
    return {
        'path':str(path),'raw_sha256':metadata['raw_sha256'],
        'source_sha256':source,'source_snapshot':str(snapshot),
        'metadata_sha256':analysis.sha256(metadata_path),
        'settings':settings,'restart_sha256':metadata.get('restart_sha256'),
        'context':analysis.comparison_context(data,metadata),
        'ids':samples[:,:,2].astype(int),'final_counts':counts,
        'final_bragg':bragg,'initial_counts':initial_counts,
        'initial_bragg':initial_bragg,
        'pairs':pairs,
    }


def validate_link(parent, child):
    """A matching path alone is not enough: validate bytes and all state slots."""
    ps,cs = parent['settings'],child['settings']
    restart = cs.get('restart')
    if not restart or (ROOT/Path(restart)).resolve() != Path(parent['path']).resolve():
        raise ValueError('Restart path must identify the immediately preceding archive')
    if child.get('restart_sha256') != parent['raw_sha256']:
        raise ValueError('Restart SHA256 must match the exact parent raw archive')
    if (cs.get('equil_sweeps') != 0 or cs.get('calibrate_activity')
            or not cs.get('freeze_displacement')):
        raise ValueError('Stitched child must have zero burn-in and no target/proposal adaptation')
    if (ps.get('sample_stride') != cs.get('sample_stride')
            or parent['source_sha256'] != child['source_sha256']):
        raise ValueError('Sampling cadence and executed walker semantics must match')
    if any(ps.get(key) != cs.get(key) for key in PROPOSAL_SETTINGS):
        raise ValueError('Frozen proposal parameters must match across restart')
    pc,cc = parent['context'],child['context']
    if (pc['physical_target'] != cc['physical_target']
            or pc['auxiliary_graph_sha256'] != cc['auxiliary_graph_sha256']
            or pc['displacement_proposal_scales'] != cc['displacement_proposal_scales']
            or parent['pairs'] != child['pairs']):
        raise ValueError('Frozen replica targets, graph and every proposal scale must match')
    if (np.shape(parent['final_bragg']) != np.shape(child['initial_bragg'])
            or not np.array_equal(parent['final_counts'],child['initial_counts'])
            or not np.allclose(parent['final_bragg'],child['initial_bragg'],rtol=0,atol=1e-10)):
        raise ValueError('Child initial counts/Bragg must match every parent final state slot')


def stitch_segments(segments):
    """Compose slot-to-parent labels without pooling any physical observables."""
    if not segments or segments[0]['settings'].get('restart'):
        raise ValueError('A complete lineage must begin with a fresh, non-restarted history')
    if len({segment['raw_sha256'] for segment in segments}) != len(segments):
        raise ValueError('Duplicate or cyclic raw histories are not a lineage')
    nrep = segments[0]['ids'].shape[1]
    canonical = np.arange(nrep)
    joined, records, offset = [], [], 0
    for index,segment in enumerate(segments):
        ids = segment['ids']
        if (ids.ndim != 2 or ids.shape[1] != nrep or len(ids)==0
                or not np.all(np.sort(ids,axis=1)==np.arange(nrep))):
            raise ValueError('Every saved row must be a permutation of walker slot IDs')
        if index:
            validate_link(segments[index-1],segment)
            canonical = joined[-1][-1].copy()
        trace = canonical[ids]
        joined.append(trace)
        settings = segment['settings']
        stride,production = settings['sample_stride'],settings['production_sweeps']
        if stride <= 0 or production % stride or len(ids)*stride != production:
            raise ValueError('Every segment must save its exact final production frame')
        records.append({
            'raw_file':analysis.display_path(segment['path']),
            'raw_sha256':segment['raw_sha256'],
            'restart_sha256':segment.get('restart_sha256'),
            'frames':len(ids),'production_sweeps':production,
            'first_saved_cumulative_sweep':offset+stride,
            'last_saved_cumulative_sweep':offset+production,
            'local_to_lineage_walker_id':canonical.tolist(),
            'segment_local_transport':{key:analysis.round_trips(ids,*nodes)
                                       for key,nodes in segment['pairs'].items()},
        })
        offset += production
    ids = np.concatenate(joined)
    return {
        'scope':'Read-only identity transport across hash-verified frozen restarts, not pooled thermodynamic estimates or an equilibration certificate',
        'sampling_scope':'Saved diagnostic frames only, excluding the first history burn-in. Unrecorded excursions may be missed, so counts are lower bounds; exact lineage mapping removes artificial reset boundaries but cannot restore unsaved visits',
        'count_order':['completed_round_trips','completed_one_way_legs','walkers_with_a_leg'],
        'segments':records,'production_sweeps':offset,'saved_frames':len(ids),
        'source_sha256':segments[0]['source_sha256'],
        'transport_nodes':segments[0]['pairs'],
        'stitched_transport':{key:analysis.round_trips(ids,*nodes)
                              for key,nodes in segments[0]['pairs'].items()},
    }


def write_lineage(inputs,output):
    inputs = [Path(path).resolve() for path in inputs]
    output = Path(output).resolve()
    metadata_path = output.with_suffix('.metadata.json')
    if output.suffix != '.json' or {output,metadata_path}.intersection(
            {file for path in inputs for file in (path,path.with_suffix('.json'))}):
        raise ValueError('Output must be a new JSON artifact, not a raw trajectory or sidecar')
    segments = [load_segment(path) for path in inputs]
    result = stitch_segments(segments)
    files = sorted({file for path in inputs for file in (path,path.with_suffix('.json'))}
                   |{Path(segment['source_snapshot']) for segment in segments})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    metadata_path.write_text(json.dumps({
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'script':'scripts/continuum_lineage_transport.py',
        'script_sha256':EXECUTION_SOURCE_SHA256,'python':sys.version,
        'dependency_sha256':{'scripts/continuum_analyze.py':analysis.EXECUTION_SOURCE_SHA256},
        'input_sha256':{analysis.display_path(path):analysis.sha256(path) for path in files},
        'output_sha256':{analysis.display_path(output):analysis.sha256(output)},
        'reproduce_argv':['python3','scripts/continuum_lineage_transport.py',
                          *map(analysis.display_path,inputs),'--output',analysis.display_path(output)],
    },indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path,help='Fresh root followed by its exact restart children')
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    try:
        result = write_lineage(args.inputs,args.output)
    except (ValueError,KeyError,FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result['stitched_transport']))


if __name__ == '__main__':
    main()
