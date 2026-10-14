#!/usr/bin/env python3
"""Run one child command with immutable log and per-child usage measurements.

This lightweight wrapper performs no sampling. It has exactly one child, so
RUSAGE_CHILDREN high-water RSS is unambiguously that command's measured peak.
On macOS ru_maxrss is bytes; on Linux it is KiB. No system-wide settings change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_HASH = hashlib.sha256(SOURCE_BYTES).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-prefix',required=True,type=Path)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:
        parser.error('A child command is required after --')
    log = args.log_prefix.with_suffix('.log')
    metrics = args.log_prefix.with_suffix('.metrics.json')
    if log.exists() or metrics.exists():
        parser.error('Measured log/metrics prefix must be new')
    log.parent.mkdir(parents=True,exist_ok=True)
    snapshot = ROOT/'data_raw/source_snapshots'/f'continuum_measured_run_{SOURCE_HASH}.py'
    if snapshot.exists() and snapshot.read_bytes()!=SOURCE_BYTES:
        parser.error('Content-keyed wrapper snapshot collision')
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(SOURCE_BYTES)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    with log.open('x') as handle:
        child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1)
        for line in child.stdout:
            handle.write(line)
            handle.flush()
            print(line,end='',flush=True)
        code = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {'schema':'cavcoex.single_child_resource_measurement.v1',
        'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'command':command,'cwd':str(Path.cwd()),'child_pid':child.pid,'child_exit_code':code,
        'wall_seconds':time.monotonic()-start,
        'child_user_seconds':after.ru_utime-before.ru_utime,
        'child_system_seconds':after.ru_stime-before.ru_stime,
        'child_peak_RSS_bytes':int(after.ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
        'RSS_native_unit':'bytes' if sys.platform=='darwin' else 'KiB',
        'child_swaps':int(after.ru_nswap-before.ru_nswap),
        'platform':sys.platform,'measurement':'resource.getrusage(RUSAGE_CHILDREN); exactly one child command',
        'source_sha256':SOURCE_HASH,'source_snapshot':str(snapshot.relative_to(ROOT)),
        'log_sha256':digest(log),
        'thread_limits':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS',
            'OMP_NUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS')},
        'reproduce_argv':['python3','-m','scripts.continuum_measured_run',
            '--log-prefix',str(args.log_prefix),'--',*command]}
    metrics.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'measured_child_exit_code':code,'wall_seconds':record['wall_seconds'],
        'child_peak_RSS_bytes':record['child_peak_RSS_bytes'],'child_swaps':record['child_swaps']}),flush=True)
    raise SystemExit(code)


if __name__=='__main__':
    main()
