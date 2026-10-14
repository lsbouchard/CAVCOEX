"""Checks that packaging cannot silently relabel a run or omit required files."""

import hashlib
import configparser
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import enrich_metadata_provenance as provenance
import make_release_manifest as manifest
import build_release_bundles as bundles


def test_release_snapshot_preserves_run_hashes_and_commands(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "generated").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("# current source\n")
    record = {
        "script_sha256": "original-execution-hash",
        "dependency_sha256": {"old.py": "original-dependency-hash"},
        "input_sha256": {"old.csv": "original-data-hash"},
        "reproduce_command": "python run.py --original-setting",
        "generated_utc": "2026-08-06T00:00:00+00:00",
    }
    target = tmp_path / "generated" / "run_metadata.json"
    target.write_text(json.dumps(record))
    original_bytes = target.read_bytes()
    monkeypatch.setattr(provenance, "SPECS", {
        "generated/run_metadata.json": {
            "script": "scripts/run.py", "command": "python scripts/run.py",
        }
    })
    assert provenance.enrich(tmp_path) == 1
    assert target.read_bytes() == original_bytes
    contexts = json.loads((tmp_path / "generated" / "release_provenance.json").read_text())
    context = contexts["generated/run_metadata.json"]
    assert context["execution_metadata_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert context["script_sha256"] == hashlib.sha256(b"# current source\n").hexdigest()
    assert "no execution is asserted" in context["scope"]


def test_manifest_refuses_missing_required_release_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="required release inputs missing"):
        manifest.included_files(tmp_path)


def test_audit_environment_is_required_separately_from_historical_locks():
    assert 'environment-audit.yml' in manifest.TOP_LEVEL
    assert 'environment-audit.yml' in manifest.CODE_TOP_LEVEL
    root = Path(__file__).resolve().parents[1]
    audit = (root/'environment-audit.yml').read_text()
    historical = (root/'environment-lock.yml').read_text()
    assert 'python=3.12.12' in audit and 'python=3.14.6' in historical
    assert 'not tested' in audit and 'not a fully resolved lock' in audit


def test_default_pytest_collection_excludes_archived_test_source_snapshots():
    assert 'pytest.ini' in manifest.TOP_LEVEL
    assert 'pytest.ini' in manifest.CODE_TOP_LEVEL
    parser = configparser.ConfigParser()
    parser.read(Path(__file__).resolve().parents[1]/'pytest.ini')
    assert parser['pytest']['testpaths'].split() == ['tests']


def test_continuum_methods_are_required_and_included_recursively(tmp_path,monkeypatch):
    assert "continuum_methods.tex" in manifest.TOP_LEVEL
    monkeypatch.setattr(bundles,"ROOT",tmp_path)
    supplement = tmp_path/"supplement.tex"
    methods = tmp_path/"continuum_methods.tex"
    supplement.write_text("\\input{continuum_methods}\n")
    methods.write_text("Continuum sampler methods.\n")
    assert bundles.tex_dependencies(supplement) == {supplement,methods}


def test_compile_bundle_includes_only_referenced_figures(tmp_path,monkeypatch):
    monkeypatch.setattr(bundles,'ROOT',tmp_path)
    source = tmp_path/'manuscript.tex'
    source.write_text(r'\includegraphics[width=1in]{used.pdf}')
    used = tmp_path/'used.pdf'
    used.write_bytes(b'used figure')
    (tmp_path/'exploratory.pdf').write_bytes(b'not a manuscript figure')
    assert bundles.tex_dependencies(source) == {source,used}


def test_code_profile_keeps_publication_inputs_but_omits_failed_continuum_raw(tmp_path):
    included = ['data_raw/lattice_gas_phase_switch_samples.npz',
                'data_raw/source_snapshots/lattice_gas_old.py',
                'scripts/continuum_replica_exchange.py',
                'scripts/continuum_single_walker_analyze.py',
                'scripts/continuum_simulated_tempering.py',
                'tests/test_continuum_analyze.py',
                'data_processed/continuum_diagnostic_summary.json',
                'generated/continuum_diagnostic_summary_metadata.json']
    excluded = ['data_raw/continuum_number_vacancy_fluid_pilot.npz',
                'data_raw/continuum_number_vacancy_fluid_pilot.json',
                'data_raw/source_snapshots/continuum_replica_exchange_abc.py',
                'data_raw/source_snapshots/continuum_simulated_tempering_abc.py',
                'data_raw/continuum_single_walker_control.npz',
                'data_raw/controls/continuum_single_host_gibbs_staged_cli_control_20260930.npz',
                'data_raw/controls/continuum_single_host_gibbs_staged_cli_control_20260930.json',
                'data_processed/continuum_dense_pilot_audit.json',
                'figures/continuum_dense_pilot_audit.pdf']
    assert all(not manifest.excluded_from_code(tmp_path/name,tmp_path) for name in included)
    assert all(manifest.excluded_from_code(tmp_path/name,tmp_path) for name in excluded)


def test_lean_capped_control_keeps_only_exact_verified_source_dependencies(tmp_path):
    raw = tmp_path / manifest.LEAN_CONTROL_METADATA[0]
    raw.parent.mkdir(parents=True)
    raw.with_suffix('.npz').write_bytes(b'capped independent validation fixture')
    sources = tmp_path/'data_raw'/'source_snapshots'
    sources.mkdir()
    retained = sources/'continuum_test_hash.py'
    retained.write_text('# exactly recorded source\n')
    unrelated = sources/'continuum_failed_pilot_hash.py'
    unrelated.write_text('# historical pilot only\n')
    record = {'raw_sha256':manifest.sha256_file(raw.with_suffix('.npz')),
              'source_snapshots':{name:[retained.relative_to(tmp_path).as_posix(),manifest.sha256_file(retained)]
                                  for name in ('driver','kernel','graph_validation','independent_control')}}
    raw.write_text(json.dumps(record))
    assert manifest.retained_control_sources(tmp_path) == {retained}
    assert not manifest.excluded_from_code(retained,tmp_path)
    assert not manifest.excluded_from_code(raw.with_suffix('.npz'),tmp_path)
    assert manifest.excluded_from_code(unrelated,tmp_path)
    assert manifest.excluded_from_code(sources/'test_single_walker_old_control.py',tmp_path)
    assert manifest.excluded_from_code(tmp_path/'data_raw'/'continuum_failed_dense.npz',tmp_path)
    retained.write_text('# altered source bytes\n')
    with pytest.raises(ValueError,match='path/digest'):
        manifest.retained_control_sources(tmp_path)


def test_lean_control_cannot_import_absolute_or_outside_snapshot_paths(tmp_path):
    sidecar = tmp_path / manifest.LEAN_CONTROL_METADATA[0]
    sidecar.parent.mkdir(parents=True)
    sidecar.with_suffix('.npz').write_bytes(b'raw')
    outside = tmp_path/'other.py'
    outside.write_text('# not a source snapshot\n')
    for value in ('other.py',str(outside)):
        record = {'raw_sha256':manifest.sha256_file(sidecar.with_suffix('.npz')),
                  'source_snapshots':{name:[value,manifest.sha256_file(outside)]
                                      for name in ('driver','kernel','graph_validation','independent_control')}}
        sidecar.write_text(json.dumps(record))
        with pytest.raises(ValueError,match='path/digest'):
            manifest.retained_control_sources(tmp_path)


def test_lean_control_rejects_snapshot_directory_symlink_escape(tmp_path):
    root = tmp_path/'project'
    (root/'data_raw').mkdir(parents=True)
    elsewhere = tmp_path/'elsewhere'
    elsewhere.mkdir()
    (root/'data_raw'/'source_snapshots').symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError,match='inside the project'):
        manifest.retained_control_sources(root)


def test_release_hash_streams_large_inputs(tmp_path,monkeypatch):
    path = tmp_path/"trajectory.npz"
    contents = b"bounded hash memory\n"*200000
    path.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()
    def no_full_file_read(_):
        raise AssertionError("Large archive must not use Path.read_bytes")
    monkeypatch.setattr(Path,"read_bytes",no_full_file_read)
    assert manifest.sha256_file(path) == expected


def test_manifest_changes_when_curated_input_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "CODE_TOP_LEVEL", ("README.md",))
    monkeypatch.setattr(manifest, "DIRECTORIES", ("scripts",))
    (tmp_path / "scripts").mkdir()
    readme = tmp_path / "README.md"
    readme.write_text("initial\n")
    (tmp_path / "scripts" / "run.py").write_text("pass\n")
    initial = manifest.manifest_text(tmp_path, "code")
    readme.write_text("changed\n")
    changed = manifest.manifest_text(tmp_path, "code")
    assert initial != changed
    assert len(changed.splitlines()) == 2
    # Regenerable caches are never part of a release checksum list.
    cache = tmp_path / "scripts" / "__pycache__"
    cache.mkdir()
    (cache / "run.pyc").write_bytes(b"cache")
    assert manifest.manifest_text(tmp_path, "code") == changed


def test_frozen_mc_source_hashes_have_distributed_sources():
    root = Path(__file__).resolve().parents[1]
    sources = list((root / "scripts").glob("*.py"))
    sources.extend((root / "data_raw" / "source_snapshots").glob("*.py"))
    digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for name, column in (
        ("cavity_mc_blocks.csv", "script_sha256"),
        ("solid_cavity_mc_blocks.csv", "script_sha256"),
        ("solid_cavity_mc_additional_replica_blocks.csv", "campaign_script_sha256"),
    ):
        with (root / "data_raw" / name).open() as handle:
            recorded = {row[column] for row in csv.DictReader(handle)}
        assert recorded <= digests
    for stem in ("lattice_gas_phase_switch", "lattice_gas_additional_replicas"):
        record = json.loads((root / "generated" / f"{stem}_metadata.json").read_text())
        assert record["script_sha256"] in digests


def _verify_resource_measurement(root, sidecar, record):
    """A closed measurement is evidence even when its child exited nonzero."""
    assert record["schema"] == "cavcoex.single_child_resource_measurement.v1"
    assert sidecar.name.endswith(".metrics.json")
    log = sidecar.with_name(sidecar.name.removesuffix(".metrics.json") + ".log")
    assert log.resolve().is_relative_to(root.resolve())
    assert manifest.sha256_file(log) == record["log_sha256"], "resource log digest mismatch"
    source_relative = Path(record["source_snapshot"])
    assert not source_relative.is_absolute()
    snapshot = root / source_relative
    assert snapshot.resolve().is_relative_to((root / "data_raw/source_snapshots").resolve())
    assert snapshot.resolve().is_relative_to(root.resolve())
    assert snapshot.name == f"continuum_measured_run_{record['source_sha256']}.py"
    assert manifest.sha256_file(snapshot) == record["source_sha256"], "resource source digest mismatch"
    started = datetime.fromisoformat(record["started_utc"])
    finished = datetime.fromisoformat(record["finished_utc"])
    assert started.utcoffset() is not None and finished.utcoffset() is not None
    assert finished >= started, "measurement must record a closed time interval"
    # bool is an int subclass, but is not a recorded process status or usage.
    for key in ("child_pid", "child_exit_code", "child_peak_RSS_bytes", "child_swaps"):
        assert type(record[key]) is int, f"invalid measured integer: {key}"
    assert record["child_pid"] > 0
    assert record["child_peak_RSS_bytes"] >= 0 and record["child_swaps"] >= 0
    # Any integer exit code, including signal termination, is valid provenance;
    # there is deliberately no assertion that the measured command succeeded.
    for key in ("wall_seconds", "child_user_seconds", "child_system_seconds"):
        assert type(record[key]) in (int, float) and math.isfinite(record[key])
        assert record[key] >= 0, f"negative measured duration: {key}"
    assert isinstance(record["platform"], str) and record["platform"]
    assert record["RSS_native_unit"] == ("bytes" if record["platform"] == "darwin" else "KiB")
    assert record["measurement"] == "resource.getrusage(RUSAGE_CHILDREN); exactly one child command"
    command = record["command"]
    assert isinstance(command, list) and command
    assert all(isinstance(value, str) and value for value in command)
    assert isinstance(record["cwd"], str) and Path(record["cwd"]).is_absolute()
    replay = record["reproduce_argv"]
    assert isinstance(replay, list) and len(replay) == 6 + len(command)
    assert replay[:4] == ["python3", "-m", "scripts.continuum_measured_run", "--log-prefix"]
    assert isinstance(replay[4], str) and replay[4]
    assert replay[5:] == ["--", *command]
    limits = record["thread_limits"]
    assert set(limits) == {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
                           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"}
    assert all(value is None or (isinstance(value, str) and value.isdecimal() and int(value) > 0)
               for value in limits.values())


def _verify_continuum_sidecar(root, sidecar):
    record = json.loads(sidecar.read_text())
    assert isinstance(record, dict)
    schema = record.get("schema")
    if schema == "cavcoex.single_child_resource_measurement.v1":
        _verify_resource_measurement(root, sidecar, record)
        return
    assert schema in (None, "cavcoex.native_tempering_graph.v1",
                      "cavcoex.single_host_regenerative_tempering.v1"), "unknown continuum sidecar schema"
    assert not sidecar.name.endswith(".metrics.json"), "resource metrics require their declared schema"
    assert manifest.sha256_file(sidecar.with_suffix(".npz")) == record["raw_sha256"]
    # Earliest exploratory pilots predate execution-source preservation.
    # Do not fabricate snapshots for those historical records. Every
    # explicit snapshot assertion in a later sidecar must be verifiable.
    if "source_snapshot" in record:
        snapshot = root / record["source_snapshot"]
        assert snapshot.resolve().is_relative_to(root)
        assert manifest.sha256_file(snapshot) == record["source_sha256"]
    for source_path, digest in record.get("source_snapshots", {}).values():
        snapshot = root / source_path
        assert snapshot.resolve().is_relative_to(root)
        assert manifest.sha256_file(snapshot) == digest


def test_recorded_continuum_source_snapshots_and_raw_digests_are_preserved():
    root = Path(__file__).resolve().parents[1]
    for sidecar in (root / "data_raw").glob("continuum*.json"):
        _verify_continuum_sidecar(root, sidecar)


@pytest.fixture
def resource_sidecar(tmp_path):
    sidecar = tmp_path / "data_raw/continuum_control.metrics.json"
    sidecar.parent.mkdir()
    log = sidecar.with_name("continuum_control.log")
    log.write_text("A faithfully recorded child failure.\n")
    source_bytes = b"# exact wrapper execution source\n"
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    snapshot = tmp_path / f"data_raw/source_snapshots/continuum_measured_run_{source_hash}.py"
    snapshot.parent.mkdir()
    snapshot.write_bytes(source_bytes)
    record = {
        "schema": "cavcoex.single_child_resource_measurement.v1",
        "started_utc": "2026-09-30T12:00:00+00:00", "finished_utc": "2026-09-30T12:00:01+00:00",
        "command": ["python3", "-m", "pytest"], "cwd": str(tmp_path),
        "child_pid": 123, "child_exit_code": 1, "wall_seconds": 1.0,
        "child_user_seconds": 0.4, "child_system_seconds": 0.1,
        "child_peak_RSS_bytes": 123456, "child_swaps": 0, "platform": "darwin", "RSS_native_unit": "bytes",
        "measurement": "resource.getrusage(RUSAGE_CHILDREN); exactly one child command",
        "source_sha256": source_hash, "source_snapshot": snapshot.relative_to(tmp_path).as_posix(),
        "log_sha256": manifest.sha256_file(log),
        "thread_limits": {key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
                                               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "reproduce_argv": ["python3", "-m", "scripts.continuum_measured_run", "--log-prefix",
                           "data_raw/continuum_control", "--", "python3", "-m", "pytest"],
    }
    sidecar.write_text(json.dumps(record))
    return sidecar, record


@pytest.mark.parametrize("exit_code", [0, 1, 7, -15])
def test_resource_sidecar_records_success_failure_and_signal_without_relabeling(tmp_path, resource_sidecar, exit_code):
    sidecar, record = resource_sidecar
    record["child_exit_code"] = exit_code
    sidecar.write_text(json.dumps(record))
    _verify_continuum_sidecar(tmp_path, sidecar)
    assert json.loads(sidecar.read_text())["child_exit_code"] == exit_code
    assert not sidecar.with_suffix(".npz").exists()


@pytest.mark.parametrize("damaged", ["log", "source"])
def test_resource_sidecar_rejects_hash_damage(tmp_path, resource_sidecar, damaged):
    sidecar, record = resource_sidecar
    target = sidecar.with_name("continuum_control.log") if damaged == "log" else tmp_path / record["source_snapshot"]
    target.write_text("tampered bytes\n")
    with pytest.raises(AssertionError, match=f"resource {damaged} digest mismatch"):
        _verify_continuum_sidecar(tmp_path, sidecar)


@pytest.mark.parametrize("key,value", [
    ("schema", "cavcoex.unknown.v1"), ("schema", None),
    ("child_exit_code", None), ("child_exit_code", True), ("child_pid", 0),
    ("child_swaps", -1), ("child_peak_RSS_bytes", -1), ("wall_seconds", float("nan")),
    ("child_user_seconds", -1), ("RSS_native_unit", "KiB"),
    ("finished_utc", "2026-09-30T11:59:59+00:00"), ("command", []),
])
def test_resource_sidecar_rejects_unknown_schema_or_incomplete_measurement(tmp_path, resource_sidecar, key, value):
    sidecar, record = resource_sidecar
    record[key] = value
    sidecar.write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        _verify_continuum_sidecar(tmp_path, sidecar)


def test_raw_sidecar_digest_validation_is_not_bypassed_by_schema_dispatch(tmp_path):
    sidecar = tmp_path / "continuum_raw.json"
    raw = sidecar.with_suffix(".npz")
    raw.write_bytes(b"archived raw input")
    record = {"raw_sha256": manifest.sha256_file(raw)}
    sidecar.write_text(json.dumps(record))
    _verify_continuum_sidecar(tmp_path, sidecar)
    raw.write_bytes(b"damaged raw input")
    with pytest.raises(AssertionError):
        _verify_continuum_sidecar(tmp_path, sidecar)
    record["schema"] = "cavcoex.unknown.v1"
    record["raw_sha256"] = manifest.sha256_file(raw)
    sidecar.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="unknown continuum sidecar schema"):
        _verify_continuum_sidecar(tmp_path, sidecar)
