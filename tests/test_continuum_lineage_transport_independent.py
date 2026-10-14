"""Independent, small synthetic controls for read-only restart-label stitching.

These tests contain no Monte Carlo kernel or equilibration claim.  In particular,
their arrays are label histories, not pooled thermodynamic measurements.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import continuum_lineage_transport as lineage


def _segment(name, ids, parent=None):
    """Make three-slot synthetic diagnostics, with hashes distinct by name."""
    settings = {
        "restart": None if parent is None else parent["path"],
        "sample_stride": 2,
        "production_sweeps": 2 * len(ids),
        "equil_sweeps": 0,
        "calibrate_activity": False,
        "freeze_displacement": True,
        "sites": 4,
        "cap": 4,
        "palm_transfer": True,
        "biased_birth": True,
        "birth_width": .12,
        "birth_uniform_fraction": .2,
        "birth_proposal": "vacancy",
        "birth_scope": "structured",
    }
    segment = {
        "path": str(Path("/synthetic") / f"{name}.npz"),
        "raw_sha256": hashlib.sha256(name.encode()).hexdigest(),
        "source_sha256": "reviewed_source",
        "restart_sha256": None if parent is None else parent["raw_sha256"],
        "settings": settings,
        "context": {
            "physical_target": {"x": 1., "V": 8., "sigma": 1., "R": 1.},
            "auxiliary_graph_sha256": "same_graph",
            "displacement_proposal_scales": [.2, .1, .1],
        },
        "ids": np.asarray(ids, dtype=int),
        "final_counts": np.array([1, 2, 3]),
        "final_bragg": np.array([1., .4, .7]),
        "initial_counts": np.array([1, 2, 3]),
        "initial_bragg": np.array([1., .4, .7]),
        "pairs": {"endpoint_pair": (0, 2)},
    }
    return segment


def test_recursive_mapping_restores_cross_restart_round_trip_without_pooling():
    # Parent moves original walker 0 from low slot 0 to high slot 2.
    first = _segment("fresh", [[0, 1, 2], [1, 2, 0]])
    # Child local ID 2 is the parent's original walker 0.  Moving child ID 2
    # to slot 0 closes the trip that a naive reset-ID concatenation would miss.
    second = _segment("child", [[2, 0, 1], [0, 1, 2]], first)
    third = _segment("grandchild", [[1, 2, 0]], second)
    result = lineage.stitch_segments([first, second, third])
    assert [record["local_to_lineage_walker_id"] for record in result["segments"]] == [
        [0, 1, 2], [1, 2, 0], [1, 2, 0]
    ]
    # Independent manual histories of the three canonical walkers are:
    # walker 0: low -> high -> low -> high -> middle: three legs, one trip;
    # walker 1: middle -> low -> middle -> low -> high: one leg, no trip;
    # walker 2: high -> middle -> high -> middle -> low: one leg, no trip.
    assert result["stitched_transport"]["endpoint_pair"] == (1, 5, 3)
    # The first child's first frame already completes walker 0's low-high-low
    # trip, even though neither segment individually contains a complete trip.
    short_child = deepcopy(second)
    short_child["ids"] = short_child["ids"][:1]
    short_child["settings"]["production_sweeps"] = 2
    prefix = lineage.stitch_segments([first, short_child])
    assert prefix["stitched_transport"]["endpoint_pair"] == (1, 2, 1)
    assert result["production_sweeps"] == 10
    assert result["saved_frames"] == 5
    assert [(r["first_saved_cumulative_sweep"], r["last_saved_cumulative_sweep"])
            for r in result["segments"]] == [(2, 4), (6, 8), (10, 10)]
    assert "not pooled thermodynamic" in result["scope"]
    assert "lower bounds" in result["sampling_scope"]
    assert not any(key in result for key in ("mean_number", "mean_bragg", "samples"))


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s["settings"].update(restart="/synthetic/not_parent.npz"), "Restart path"),
    (lambda s: s.update(restart_sha256="different_bytes"), "Restart SHA256"),
    (lambda s: s["settings"].update(equil_sweeps=1), "zero burn-in"),
    (lambda s: s["settings"].update(calibrate_activity=True), "no target/proposal adaptation"),
    (lambda s: s["settings"].update(freeze_displacement=False), "no target/proposal adaptation"),
    (lambda s: s["settings"].update(sample_stride=3), "Sampling cadence"),
    (lambda s: s.update(source_sha256="new_unreviewed_semantics"), "walker semantics"),
    (lambda s: s["settings"].update(sites=32), "proposal parameters"),
    (lambda s: s["settings"].update(birth_width=.2), "proposal parameters"),
    (lambda s: s["settings"].update(birth_scope="physical"), "proposal parameters"),
    (lambda s: s["context"]["physical_target"].update(x=1.1), "Frozen replica targets"),
    (lambda s: s["context"].update(auxiliary_graph_sha256="different_graph"), "Frozen replica targets"),
    (lambda s: s["context"].update(displacement_proposal_scales=[.2, .1, .2]), "every proposal scale"),
    (lambda s: s["initial_counts"].__setitem__(1, 1), "every parent final state slot"),
    (lambda s: s["initial_bragg"].__setitem__(1, .3), "every parent final state slot"),
    (lambda s: s.update(initial_bragg=np.array([1.])), "every parent final state slot"),
])
def test_link_rejects_nonidentical_source_state_target_or_proposal(mutation, match):
    parent = _segment("parent", [[0, 1, 2]])
    child = _segment("child", [[0, 1, 2]], parent)
    mutation(child)
    with pytest.raises(ValueError, match=match):
        lineage.validate_link(parent, child)


def test_unused_seed_and_output_changes_do_not_break_exact_slot_restart_mapping():
    parent = _segment("parent", [[0, 1, 2]])
    child = _segment("child", [[0, 1, 2]], parent)
    parent["settings"].update(seed=1, output="old_output")
    child["settings"].update(seed=2, output="new_output")
    lineage.validate_link(parent, child)


@pytest.mark.parametrize("case", ["duplicate", "nonfresh", "duplicate_id", "missing_id", "cadence"])
def test_stitch_rejects_incomplete_or_invalid_label_histories(case):
    first = _segment("parent", [[0, 1, 2], [1, 2, 0]])
    second = _segment("child", [[0, 1, 2]], first)
    if case == "duplicate":
        second["raw_sha256"] = first["raw_sha256"]
    elif case == "nonfresh":
        first["settings"]["restart"] = "/synthetic/missing_ancestor.npz"
    elif case == "duplicate_id":
        second["ids"][0] = [0, 0, 2]
    elif case == "missing_id":
        second["ids"][0] = [0, 1, 3]
    else:
        first["settings"]["production_sweeps"] += 1
    with pytest.raises(ValueError):
        lineage.stitch_segments([first, second])


def _raw_fixture(tmp_path, monkeypatch):
    """A tiny integrity fixture, not an executed physical simulation."""
    source = tmp_path / "source_snapshots" / "synthetic_driver.py"
    source.parent.mkdir()
    source.write_text("# Synthetic fixture source, never executed.\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    # Explicitly substitute the semantic allowlist only within this fixture.
    monkeypatch.setattr(lineage, "ROOT", tmp_path)
    monkeypatch.setattr(lineage, "VERIFIED_SLOT_RESET_SOURCES", {source_sha})
    samples = np.zeros((2, 3, 9))
    samples[:, :, 0:2] = 1.
    samples[:, :, 2] = [[0, 1, 2], [1, 2, 0]]
    samples[:, :, 3] = 1.
    arrays = {
        "samples": samples,
        "endpoint": np.array(1),
        "volume": np.array(8.),
        "activity": np.array([-2., -1., -1.]),
        "diameter": np.array([0., 1., 1.]),
        "radius": np.array([0., 0., 1.]),
        "swaps": np.zeros((2, 2), dtype=int),
        "cavity_nodes": np.array([1, 2]),
        "exchange_edges": np.array([[0, 1], [1, 2]]),
        "pair_penalty": np.array([0., np.inf, np.inf]),
        "umbrella_stiffness": np.zeros(3),
        "umbrella_preferred": np.zeros(3),
        "number_stiffness": np.zeros(3),
        "number_preferred": np.zeros(3),
        "displacement": np.array([.2, .1, .1]),
        "final_counts": np.ones(3, dtype=int),
        "final_positions": np.full((3, 4, 3), .3),
    }
    metadata = {
        "settings": {"sample_stride": 2, "production_sweeps": 4, "sites": 4,
                     "cap": 4, "restart": None},
        "source_sha256": source_sha,
        "source_snapshot": str(source.relative_to(tmp_path)),
        "initial_counts": [1, 1, 1],
        "initial_bragg": [1., 1., 1.],
    }
    path = tmp_path / "synthetic.npz"
    return path, arrays, metadata, source


def _save_raw(path, arrays, metadata):
    np.savez_compressed(path, **arrays)
    metadata = deepcopy(metadata)
    metadata["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(metadata))


def test_loader_retains_only_diagnostics_and_recomputes_all_final_geometry(tmp_path, monkeypatch):
    path, arrays, metadata, _ = _raw_fixture(tmp_path, monkeypatch)
    _save_raw(path, arrays, metadata)
    loaded = lineage.load_segment(path)
    assert np.array_equal(loaded["ids"], arrays["samples"][:, :, 2])
    assert np.allclose(loaded["final_bragg"], 1., rtol=0, atol=1e-14)
    assert "samples" not in loaded and "final_positions" not in loaded
    assert loaded["pairs"]["ideal_to_cavity"] == (0, 2)


@pytest.mark.parametrize("case,match", [
    ("unknown_source", "not been reviewed"),
    ("changed_snapshot", "snapshot digest"),
    ("incomplete_final_frame", "production end"),
    ("final_count_mismatch", "every last sampled slot"),
    ("noninteger_final_counts", "Final checkpoint counts"),
    ("nonfinite_active_coordinate", "recorded torus"),
    ("out_of_torus_active_coordinate", "recorded torus"),
    ("wrong_final_bragg", "last sampled Bragg"),
    ("fractional_walker_id", "valid walker identity"),
    ("raw_digest_mismatch", "Raw-data digest"),
    ("initial_bragg_broadcast", "every finite state slot"),
    ("initial_count_broadcast", "every finite state slot"),
    ("initial_count_fractional", "every finite state slot"),
    ("initial_count_above_cap", "every finite state slot"),
    ("initial_bragg_nonfinite", "every finite state slot"),
    ("initial_bragg_above_one", "every finite state slot"),
])
def test_loader_rejects_corrupted_final_state_source_or_saved_labels(tmp_path, monkeypatch, case, match):
    path, arrays, metadata, source = _raw_fixture(tmp_path, monkeypatch)
    if case == "unknown_source":
        metadata["source_sha256"] = "not_allowlisted"
    elif case == "changed_snapshot":
        source.write_text("# different synthetic source\n")
    elif case == "incomplete_final_frame":
        metadata["settings"]["production_sweeps"] = 5
    elif case == "final_count_mismatch":
        arrays["final_counts"][1] = 2
    elif case == "noninteger_final_counts":
        arrays["final_counts"] = arrays["final_counts"].astype(float)
    elif case == "nonfinite_active_coordinate":
        arrays["final_positions"][1, 0, 0] = np.nan
    elif case == "out_of_torus_active_coordinate":
        arrays["final_positions"][2, 0, 0] = 2.
    elif case == "wrong_final_bragg":
        arrays["samples"][-1, 1, 1] = .9
    elif case == "fractional_walker_id":
        arrays["samples"][-1, 1, 2] = 2.25
    elif case == "initial_bragg_broadcast":
        metadata["initial_bragg"] = [1.]
    elif case == "initial_count_broadcast":
        metadata["initial_counts"] = [1]
    elif case == "initial_count_fractional":
        metadata["initial_counts"][1] = 1.5
    elif case == "initial_count_above_cap":
        metadata["initial_counts"][1] = 5
    elif case == "initial_bragg_nonfinite":
        metadata["initial_bragg"][1] = float("nan")
    elif case == "initial_bragg_above_one":
        metadata["initial_bragg"][1] = 1.1
    _save_raw(path, arrays, metadata)
    if case == "raw_digest_mismatch":
        data = json.loads(path.with_suffix(".json").read_text())
        data["raw_sha256"] = "different_raw_bytes"
        path.with_suffix(".json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        lineage.load_segment(path)


def test_independent_bragg_check_ignores_inactive_storage_but_uses_every_active_host():
    positions = np.full((3, 4, 3), np.nan)
    counts = np.array([0, 1, 2])
    positions[1, 0] = [0., 0., 0.]
    positions[2, :2] = [[0., 0., 0.], [1., 0., 0.]]
    # On L=2 and c=1, four (111) magnitudes vanish while three (200)
    # magnitudes equal one; this deliberately checks the actual seven modes.
    assert np.allclose(lineage.final_bragg(positions, counts, 8., 4),
                       [0., 1., 3/7], rtol=0, atol=1e-14)
