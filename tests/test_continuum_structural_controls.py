"""Geometric controls independent of the sampling history."""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuum_replica_exchange import fcc_reference
from continuum_structural_controls import (
    analyze, bond_order, deletion_bragg, file_digest, number_umbrella_stiffness,
    probe_structure,
)


def test_fcc_bond_order_and_translation_invariance():
    length = (32 * .96405)**(1 / 3)
    positions = fcc_reference(32, length)
    expected = np.sqrt(169 / 512)
    baseline = bond_order(positions, length)
    assert baseline == pytest.approx((expected, expected, 12.), abs=2e-14)
    shifted = (positions + np.array([.471, .832, .195]) * length) % length
    assert bond_order(shifted, length) == pytest.approx(baseline, abs=2e-14)


def test_global_bond_order_is_rotation_invariant():
    # A finite tetrahedron stays far from boundaries, so rotating it does not
    # change its nearest-image bonds. This tests the harmonics, not FCC registry.
    points = np.array([[1., 1., 1.], [1., -1., -1.],
                       [-1., 1., -1.], [-1., -1., 1.]]) * .3
    axis = np.array([1., 2., 3.]) / np.sqrt(14)
    skew = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(.731) * skew + (1 - np.cos(.731)) * (skew @ skew)
    assert bond_order(points @ rotation.T + 5., 10.) == pytest.approx(
        bond_order(points + 5., 10.), abs=2e-14)


def test_empty_and_single_particle_controls():
    assert bond_order(np.empty((0, 3)), 4.) == (0., 0., 0.)
    assert bond_order(np.ones((1, 3)), 4.) == (0., 0., 0.)
    with pytest.raises(ValueError):
        bond_order(np.ones((2, 3)), 2.)
    with pytest.raises(ValueError):
        bond_order(np.ones((2, 3)), np.nan)
    with pytest.raises(ValueError):
        bond_order(np.ones((2, 3)), 4., degree=6.5)


def test_deletion_bragg_matches_separate_coordinate_reconstruction():
    rng = np.random.default_rng(20260929)
    positions = rng.uniform(0, 4., size=(27, 3))
    current, deleted = deletion_bragg(positions, 4., 2)
    for index in range(len(positions)):
        independently_deleted, _ = deletion_bragg(np.delete(positions, index, axis=0), 4., 2)
        assert deleted[index] == pytest.approx(independently_deleted, abs=2e-15)
    shuffled = rng.permutation(len(positions))
    shifted = (positions[shuffled] + [.412, 1.35, -.21]) % 4.
    shifted_current, shifted_deleted = deletion_bragg(shifted, 4., 2)
    assert shifted_current == pytest.approx(current, abs=2e-15)
    assert shifted_deleted == pytest.approx(deleted[shuffled], abs=2e-15)


def test_perfect_fcc_deletions_and_low_number_edges():
    length = (32 * .96405)**(1 / 3)
    current, deleted = deletion_bragg(fcc_reference(32, length), length, 2)
    assert current == pytest.approx(1., abs=2e-15)
    assert deleted == pytest.approx(np.ones(32), abs=2e-15)
    current, deleted = deletion_bragg(np.ones((1, 3)), length, 2)
    assert current == pytest.approx(1.)
    assert deleted.tolist() == [0.]
    current, deleted = deletion_bragg(np.empty((0, 3)), length, 2)
    assert current == 0. and not len(deleted)
    with pytest.raises(ValueError):
        deletion_bragg(np.ones((2, 3)), length, 1.5)


@pytest.mark.parametrize("degree", [5, 6])
def test_bond_order_after_deletion_matches_full_independent_reconstruction(degree):
    rng = np.random.default_rng(2110 + degree)
    positions = rng.uniform(.2, 2.2, size=(12, 3))
    *_, deleted = bond_order(positions, 4., degree=degree, with_deletions=True)
    for index in range(len(positions)):
        expected = bond_order(np.delete(positions, index, axis=0), 4., degree=degree)[0]
        assert deleted[index] == pytest.approx(expected, abs=3e-15)
    *_, deleted_empty = bond_order(np.empty((0, 3)), 4., with_deletions=True)
    assert not len(deleted_empty)
    *_, deleted_single = bond_order(np.zeros((1, 3)), 4., with_deletions=True)
    assert deleted_single.tolist() == [0.]


def test_sparse_probe_frames_preserve_diagnostic_alignment():
    length = (32 * .96405)**(1 / 3)
    rng = np.random.default_rng(115)
    positions = np.zeros((2, 1, 32, 3))
    positions[0, 0] = fcc_reference(32, length)
    positions[1, 0, :27] = rng.uniform(0, length, size=(27, 3))
    samples = np.zeros((5, 2, 10))
    samples[1, 1, :2] = (32, deletion_bragg(positions[0, 0], length, 2)[0])
    samples[3, 1, :2] = (27, deletion_bragg(positions[1, 0, :27], length, 2)[0])
    data = {"physical_probe_configurations": positions,
            "probe_nodes": np.array([1]), "physical_probe_activity": np.array([16.]),
            "probe_sample_indices": np.array([1, 3]),
            "pair_penalty": np.array([np.inf, np.inf]),
            "diameter": np.ones(2),
            "umbrella_stiffness": np.zeros(2), "radius": np.zeros(2),
            "activity": np.array([16., 16.])}
    summary, = probe_structure(data, samples, length, 2, 1, 1.4)
    assert [r["frame"] for r in summary["records"]] == [1, 3]
    assert [r["N"] for r in summary["records"]] == [32, 27]
    assert summary["records"][0]["Q6_global"] == pytest.approx(np.sqrt(169 / 512))
    data["number_stiffness"] = np.array([0., .5])
    data["number_preferred"] = np.array([0., 29.])
    with pytest.raises(ValueError, match="unbiased"):
        probe_structure(data, samples, length, 2, 1, 1.4)
    data["number_stiffness"][1] = 0.
    assert probe_structure(data, samples, length, 2, 1, 1.4) == [summary]
    data["probe_sample_indices"] = np.array([1, 1])
    with pytest.raises(ValueError, match="alignment"):
        probe_structure(data, samples, length, 2, 1, 1.4)
    data["probe_sample_indices"] = np.array([1, 3])
    data["umbrella_stiffness"][1] = 2.
    with pytest.raises(ValueError, match="unbiased"):
        probe_structure(data, samples, length, 2, 1, 1.4)
    data["umbrella_stiffness"][1] = 0.
    data["pair_penalty"][1] = -np.inf
    with pytest.raises(ValueError, match="unbiased"):
        probe_structure(data, samples, length, 2, 1, 1.4)
    data["pair_penalty"][1] = np.inf
    data["diameter"][1] = .9
    with pytest.raises(ValueError, match="unbiased"):
        probe_structure(data, samples, length, 2, 1, 1.4)


def endpoint_fixture(sigma=1.):
    volume = 32 * .96405 * sigma**3
    length = volume**(1 / 3)
    reference = fcc_reference(32, length)
    positions = np.zeros((2, 2, 32, 3))
    positions[:, 0] = reference
    positions[:, 1, :31] = reference[1:]
    samples = np.zeros((2, 2, 10))
    samples[:, 0, :2] = (32, deletion_bragg(reference, length, 2)[0])
    samples[:, 1, :2] = (31, deletion_bragg(reference[1:], length, 2)[0])
    return {
        "samples": samples, "endpoint_configurations": positions,
        "endpoint": np.array(0), "cavity_nodes": np.array([0, 1]),
        "volume": np.array(volume), "diameter": np.full(2, sigma),
        "radius": np.array([0., sigma]), "activity": np.full(2, 16.),
        "pair_penalty": np.full(2, np.inf), "umbrella_stiffness": np.zeros(2),
    }


def save_fixture(path, data, declared_number_stiffness=None):
    np.savez_compressed(path, **data)
    settings = {"sites": 32}
    if declared_number_stiffness is not None:
        settings["compact_number_stiffness"] = declared_number_stiffness
    path.with_suffix(".json").write_text(json.dumps({
        "settings": settings, "raw_sha256": file_digest(path),
    }))


def test_endpoint_analysis_scales_cutoff_with_particle_diameter(tmp_path):
    path = tmp_path / "scaled.npz"
    save_fixture(path, endpoint_fixture(sigma=2.))
    result = analyze(path, stride=1)
    base = result["summaries"]["unconstrained"]["ordered"]
    assert base["Q6_global_mean"] == pytest.approx(np.sqrt(169 / 512))
    assert base["mean_neighbors"] == 12.
    assert result["Q6_phase_control"]["absolute_deletion_label_bound"] == 0.


@pytest.mark.parametrize("problem", ["negative_infinity", "wrong_radius",
                                      "unmatched_activity", "fractional_number",
                                      "number_bias"])
def test_endpoint_analysis_rejects_invalid_physical_records(tmp_path, problem):
    path = tmp_path / "invalid.npz"
    data = endpoint_fixture()
    if problem == "negative_infinity":
        data["pair_penalty"][0] = -np.inf
    elif problem == "wrong_radius":
        data["radius"][1] = .9
    elif problem == "unmatched_activity":
        data["activity"][1] = 15.
    elif problem == "number_bias":
        data["number_stiffness"] = np.array([0., .5])
        data["number_preferred"] = np.full(2, 29.)
    else:
        data["samples"][0, 0, 0] = 31.5
    save_fixture(path, data)
    with pytest.raises(ValueError):
        analyze(path, stride=1)


@pytest.mark.parametrize("problem", ["partial", "shape", "negative_strength",
                                      "nan_strength", "nan_preferred",
                                      "negative_preferred", "exceeds_cap"])
def test_number_umbrella_contract_rejects_malformed_records(problem):
    data = {"number_stiffness": np.zeros(3), "number_preferred": np.full(3, 29.)}
    if problem == "partial":
        del data["number_preferred"]
    elif problem == "shape":
        data["number_stiffness"] = np.zeros(2)
    elif problem == "negative_strength":
        data["number_stiffness"][1] = -.5
    elif problem == "nan_strength":
        data["number_stiffness"][1] = np.nan
    elif problem == "nan_preferred":
        data["number_preferred"][1] = np.nan
    elif problem == "negative_preferred":
        data["number_preferred"][1] = -1.
    else:
        data["number_preferred"][1] = 33.
    with pytest.raises(ValueError, match="Number-umbrella"):
        number_umbrella_stiffness(data, 3, cap=32)


def test_number_umbrella_legacy_and_valid_auxiliary_contracts():
    assert np.array_equal(number_umbrella_stiffness({}, 3), np.zeros(3))
    data = {"number_stiffness": np.array([0., 2., 0.]),
            "number_preferred": np.array([0., 29., 0.])}
    assert np.array_equal(number_umbrella_stiffness(data, 3), data["number_stiffness"])


def test_number_bias_is_forbidden_at_an_interior_physical_cavity_stage(tmp_path):
    data = endpoint_fixture()
    data["samples"] = data["samples"][:, [0, 0, 1]]
    for field in ("diameter", "activity", "pair_penalty", "umbrella_stiffness"):
        data[field] = data[field][[0, 0, 1]]
    data["cavity_nodes"] = np.array([0, 1, 2])
    data["radius"] = np.array([0., .5, 1.])
    data["number_stiffness"] = np.array([0., .5, 0.])
    data["number_preferred"] = np.array([0., 29., 0.])
    path = tmp_path / "biased-stage.npz"
    save_fixture(path, data)
    with pytest.raises(ValueError, match="unbiased"):
        analyze(path, stride=1)
    data["number_stiffness"][1] = 0.
    save_fixture(path, data)
    assert analyze(path, stride=1)["records"]


def test_declared_number_bias_requires_recorded_arrays(tmp_path):
    path = tmp_path / "unrecorded-number-bias.npz"
    data = endpoint_fixture()
    save_fixture(path, data, declared_number_stiffness=2.)
    with pytest.raises(ValueError, match="lacks its recorded"):
        analyze(path, stride=1)
    data["number_stiffness"] = np.zeros(2)
    data["number_preferred"] = np.zeros(2)
    save_fixture(path, data, declared_number_stiffness=2.)
    with pytest.raises(ValueError, match="no recorded nonzero"):
        analyze(path, stride=1)
    data["samples"] = data["samples"][:, [0, 1, 0]]
    for field in ("diameter", "activity", "pair_penalty", "umbrella_stiffness", "radius"):
        data[field] = data[field][[0, 1, 0]]
    data["number_stiffness"] = np.array([0., 0., 2.])
    data["number_preferred"] = np.array([0., 0., 29.])
    save_fixture(path, data, declared_number_stiffness=2.)
    assert analyze(path, stride=1)["records"]
