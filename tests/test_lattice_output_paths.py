"""Custom trajectory runs must not modify canonical release sidecars."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lattice_gas_additional_replicas as driver


def arguments(output):
    return argparse.Namespace(
        output=output, overwrite=False, seed=1, sizes=[4], replica_start=4,
        replica_stop=5, equil_steps=2, production_steps=8, workers=1,
    )


def test_canonical_output_paths_are_preserved(tmp_path):
    output = tmp_path / "data_raw/lattice_gas_additional_replicas.npz"
    assert driver.output_paths(output, tmp_path) == (
        output,
        tmp_path / "data_raw/lattice_gas_additional_replicas_diagnostics.csv",
        tmp_path / "generated/lattice_gas_additional_replicas_metadata.json",
    )


def test_external_output_keeps_sidecars_together_without_touching_release(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "generated").mkdir(parents=True)
    original = root / "generated/lattice_gas_additional_replicas_metadata.json"
    original.write_text("unchanged release metadata")
    output = tmp_path / "outside/new-run.npz"
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "parse_args", lambda: arguments(output))
    # Exercise file handling independently of expensive sampling.
    def kernel(length, coupling, seed, equil, steps, initial):
        sign = np.tile(np.array([-1, 1]), (steps + 1) // 2)[:steps]
        return sign * length**2, np.full(steps, 2*length**2), sign, sign, np.ones(steps)
    monkeypatch.setattr(driver, "_wolff_series", kernel)
    driver.main()
    assert output.is_file()
    assert output.with_name("new-run_diagnostics.csv").is_file()
    record = json.loads(output.with_name("new-run_metadata.json").read_text())
    assert record["output"] == str(output)
    assert original.read_text() == "unchanged release metadata"
    assert not (root / "data_raw").exists()


def test_preexisting_sidecar_is_protected_before_sampling(tmp_path, monkeypatch):
    output = tmp_path / "run.npz"
    sidecar = tmp_path / "run_diagnostics.csv"
    sidecar.write_text("retain me")
    monkeypatch.setattr(driver, "parse_args", lambda: arguments(output))
    with pytest.raises(FileExistsError, match="without --overwrite"):
        driver.main()
    assert not output.exists()
    assert sidecar.read_text() == "retain me"
