"""No numerical child: independent resource-wrapper serialization controls."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continuum_measured_run as measured


@pytest.mark.parametrize('platform,native_rss,expected_rss', [
    ('darwin',4096,4096), ('linux',4096,4096*1024)])
@pytest.mark.parametrize('exit_code', [0,7])
def test_single_child_metrics_units_log_hash_and_exit(monkeypatch,tmp_path,platform,native_rss,expected_rss,exit_code):
    prefix=tmp_path/'measurement'
    argv=['measured','--log-prefix',str(prefix),'--','toy-command','argument']
    monkeypatch.setattr(measured.sys,'argv',argv)
    monkeypatch.setattr(measured.sys,'platform',platform)
    monkeypatch.setattr(measured,'ROOT',tmp_path)
    usage=iter([SimpleNamespace(ru_utime=0.,ru_stime=0.,ru_maxrss=0,ru_nswap=0),
                SimpleNamespace(ru_utime=2.5,ru_stime=.75,ru_maxrss=native_rss,ru_nswap=3)])
    monkeypatch.setattr(measured.resource,'getrusage',lambda kind:next(usage))
    spawned=[]
    def popen(command,**kwargs):
        spawned.append((command,kwargs))
        return SimpleNamespace(stdout=io.StringIO('out\nerror\n'),pid=123,wait=lambda:exit_code)
    monkeypatch.setattr(measured.subprocess,'Popen',popen)
    with pytest.raises(SystemExit) as stop:
        measured.main()
    assert stop.value.code==exit_code
    assert len(spawned)==1
    assert spawned[0][0]==['toy-command','argument']
    assert spawned[0][1]['stderr']==measured.subprocess.STDOUT
    log=prefix.with_suffix('.log')
    assert log.read_text()=='out\nerror\n'
    record=json.loads(prefix.with_suffix('.metrics.json').read_text())
    assert record['child_exit_code']==exit_code
    assert record['child_peak_RSS_bytes']==expected_rss
    assert record['child_swaps']==3
    assert record['child_user_seconds']==2.5
    assert record['child_system_seconds']==.75
    assert record['log_sha256']==hashlib.sha256(log.read_bytes()).hexdigest()
    assert record['wall_seconds']>=0
    snapshot=tmp_path/record['source_snapshot']
    assert snapshot.read_bytes()==measured.SOURCE_BYTES
    assert record['source_sha256']==hashlib.sha256(snapshot.read_bytes()).hexdigest()


@pytest.mark.parametrize('suffix',['.log','.metrics.json'])
def test_existing_derivative_never_spawned_or_overwritten(monkeypatch,tmp_path,suffix):
    prefix=tmp_path/'measurement'
    existing=prefix.with_suffix(suffix)
    existing.write_text('preserve')
    monkeypatch.setattr(measured.sys,'argv',['measured','--log-prefix',str(prefix),'--','toy'])
    monkeypatch.setattr(measured.subprocess,'Popen',lambda *a,**k:pytest.fail('Spawned child on existing prefix'))
    with pytest.raises(SystemExit) as stop:
        measured.main()
    assert stop.value.code==2
    assert existing.read_text()=='preserve'


def test_missing_child_command_fails_before_creating_log(monkeypatch,tmp_path):
    prefix=tmp_path/'measurement'
    monkeypatch.setattr(measured.sys,'argv',['measured','--log-prefix',str(prefix),'--'])
    with pytest.raises(SystemExit) as stop:
        measured.main()
    assert stop.value.code==2
    assert not prefix.with_suffix('.log').exists()


def test_digest_reads_chunks_and_matches_sha256(tmp_path):
    path=tmp_path/'bytes'
    content=b'abcdefg'*(1024**2//7+10)
    path.write_bytes(content)
    assert measured.digest(path)==hashlib.sha256(content).hexdigest()
