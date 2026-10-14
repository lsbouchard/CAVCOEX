"""Independent geometric replay controls, not dense switching evidence."""
import hashlib
import sys

import numpy as np
import pytest

from scripts.continuum_replica_exchange import fcc_reference
from scripts.continuum_structural_controls import deletion_bragg
from scripts.continuum_tempering_structure import snapshot_control
from scripts import continuum_tempering_structure as structure


def fixture(cavity=False):
    length = (32 * .96405) ** (1 / 3)
    host = fcc_reference(32, length)
    center = host[0].copy() if cavity else np.array([.1, .2, .3])
    if cavity:
        host = host[1:]
    positions = np.full((80, 3), np.nan)
    positions[:len(host)] = host
    row = np.zeros(12)
    row[2] = len(host)
    row[3], deleted = deletion_bragg(host, length, 2)
    row[8] = np.sum(deleted > .45) - len(host) * (row[3] > .45)
    offsets = host - center
    offsets -= length * np.rint(offsets / length)
    row[7] = np.linalg.norm(offsets, axis=1).min()
    return positions, center, len(host), row, length, 1., float(cavity), 2, 1.4, (.3, .35, .4), 20


@pytest.mark.parametrize("cavity", [False, True])
def test_fcc_support_bragg_deletion_and_bond_order(cavity):
    result = snapshot_control(*fixture(cavity))
    assert result["B"] == pytest.approx(1., abs=2e-14)
    assert result["Q6_global"] > .5
    assert all(label["ordered"] for label in result["Q6_labels"].values())
    assert result["Bragg_signed_delete_one_label_change"] == 0


@pytest.mark.parametrize("problem", ["overlap", "bad_center", "bad_padding", "bad_bragg", "bad_deletion", "bad_center_distance"])
def test_bad_snapshot_or_alignment_rejected(problem):
    args = list(fixture())
    if problem == "overlap":
        args[0][1] = args[0][0]
    elif problem == "bad_center":
        args[1][0] = np.nan
    elif problem == "bad_padding":
        args[0][-1] = 0.
    elif problem == "bad_bragg":
        args[3][3] -= .1
    elif problem == "bad_deletion":
        args[3][8] = 1.
    else:
        args[3][7] += .1
    with pytest.raises(ValueError):
        snapshot_control(*args)


def test_cavity_support_replayed_from_center_not_saved_distance():
    args = list(fixture(True))
    args[1] = args[0][0].copy()
    args[3][7] = 0.
    with pytest.raises(ValueError, match="cavity support"):
        snapshot_control(*args)


@pytest.mark.parametrize("number", [0, 1])
def test_empty_and_one_particle_deleted_label_is_reconstructed(number):
    args = list(fixture())
    args[0][:] = np.nan
    if number:
        args[0][0] = [.8, .9, 1.]
    args[2] = number
    args[3][2] = number
    args[3][3] = float(number)
    args[3][8] = -float(number)
    args[3][7] = (np.linalg.norm(args[0][0] - args[1]) if number else args[4])
    result = snapshot_control(*args)
    assert result["Bragg_signed_delete_one_label_change"] == -number
    assert result["Q6_global"] == 0.


def test_profile_does_not_call_inherited_partial_cycle_ids_complete_cycles(tmp_path, monkeypatch):
    args = fixture()
    positions, center, _, row, length, *_ = args
    samples = np.tile(row, (2, 1))
    samples[:, 0] = [1, 2]
    samples[:, 1] = 1
    samples[:, 10] = 1337
    path = tmp_path / "synthetic_snapshot_control.npz"
    np.savez_compressed(path, coordinate_configurations=np.array([positions, positions]),
                        coordinate_centers=np.array([center, center]))
    data = {
        "coordinate_shape": (2, 80, 3), "coordinate_clock_steps": np.array([1, 2]),
        "coordinate_labels": np.array([1, 1]), "samples": samples,
        "volume": length**3, "diameter": np.ones(2), "base_node": 1,
        "sites": 32, "physical_nodes": np.array([1]), "radius": np.zeros(2),
        "activity": np.array([-2., 14.8]),
    }
    monkeypatch.setattr(structure, "load_validated_run", lambda _: (
        data, {"raw_sha256": structure.sha256(path)}, {path}))
    result, _ = structure.analyze(path)
    profile = result["profile"][0]
    assert profile["distinct_recorded_regeneration_cycle_ids"] == 1
    assert "distinct_regeneration_cycles" not in profile
    assert "incomplete" in result["recorded_cycle_id_scope"]
    assert "not counts of completed" in result["recorded_cycle_id_scope"]


def test_execution_snapshots_preserve_imported_bytes_not_later_path_edits(tmp_path, monkeypatch):
    producer = tmp_path / "producer.py"
    imported = b"# bytes imported by the structural replay\n"
    producer.write_bytes(b"# later on-disk edit\n")
    monkeypatch.setattr(structure, "ROOT", tmp_path)
    monkeypatch.setattr(structure, "EXECUTION_SOURCES", {"producer": (producer, imported)})
    records = structure.preserve_execution_sources()
    digest = hashlib.sha256(imported).hexdigest()
    snapshot = tmp_path / "data_raw" / "source_snapshots" / f"producer_{digest}.py"
    assert records["producer"][1] == digest
    assert snapshot.read_bytes() == imported
    assert snapshot.read_bytes() != producer.read_bytes()
    assert structure.preserve_execution_sources() == records
    snapshot.write_bytes(b"# wrong content under a content-addressed name\n")
    with pytest.raises(ValueError, match="snapshot mismatch"):
        structure.preserve_execution_sources()


def test_source_snapshot_directory_cannot_escape_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    (project / "data_raw").mkdir(parents=True)
    outside.mkdir()
    (project / "data_raw" / "source_snapshots").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(structure, "ROOT", project)
    with pytest.raises(ValueError, match="stay inside"):
        structure.preserve_execution_sources()


@pytest.mark.parametrize("existing", ["output", "sidecar"])
def test_structural_replay_refuses_to_overwrite_existing_evidence(tmp_path, monkeypatch, existing):
    output = tmp_path / "replay.json"
    protected = output if existing == "output" else output.with_suffix(".metadata.json")
    protected.write_bytes(b"preserved earlier execution\n")
    monkeypatch.setattr(structure, "analyze", lambda *args: ({}, set()))
    monkeypatch.setattr(sys, "argv", ["structural_replay", str(tmp_path / "input.npz"),
                                    "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        structure.main()
    assert error.value.code == 2
    assert protected.read_bytes() == b"preserved earlier execution\n"
