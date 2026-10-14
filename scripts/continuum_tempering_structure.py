#!/usr/bin/env python3
"""Independent structural replay of sparse single-host tempering coordinates.

Sparse snapshots are descriptive structural controls, not full-clock cycle
rewards, independent physical draws, or a coexistence certificate. A template
Fourier amplitude is checked against orientation-independent bond order.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .continuum_single_walker_analyze import load_validated_run, sha256, display_path
    from .continuum_structural_controls import bond_order, deletion_bragg
except ImportError:
    from continuum_single_walker_analyze import load_validated_run, sha256, display_path
    from continuum_structural_controls import bond_order, deletion_bragg

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SOURCES = {
    "structural_replay": (Path(__file__).resolve(), Path(__file__).read_bytes()),
    "single_walker_analyzer": (
        Path(load_validated_run.__code__.co_filename).resolve(),
        Path(load_validated_run.__code__.co_filename).read_bytes()),
    "structural_controls": (Path(bond_order.__code__.co_filename).resolve(),
                            Path(bond_order.__code__.co_filename).read_bytes()),
}
SOURCE_SHA256 = hashlib.sha256(EXECUTION_SOURCES["structural_replay"][1]).hexdigest()


def preserve_execution_sources():
    """Preserve startup producer bytes rather than subsequently edited paths."""
    directory = ROOT / "data_raw" / "source_snapshots"
    if not directory.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Structural source snapshots must stay inside the project")
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, (path, payload) in EXECUTION_SOURCES.items():
        digest = hashlib.sha256(payload).hexdigest()
        snapshot = directory / f"{path.stem}_{digest}.py"
        if snapshot.exists():
            if snapshot.is_symlink() or snapshot.read_bytes() != payload:
                raise ValueError("Content-addressed structural producer snapshot mismatch")
        else:
            with snapshot.open("xb") as handle:
                handle.write(payload)
        records[name] = [display_path(snapshot), digest]
    return records


def snapshot_control(positions, center, number, stored, length, sigma, radius,
                     cells, cutoff, thresholds, minimum_particles):
    """Reconstruct support, B, delete-one labels, and global/local Q6."""
    positions = np.asarray(positions, dtype=float)
    center = np.asarray(center, dtype=float)
    stored = np.asarray(stored, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 3
            or number != int(number) or number < 0 or number > len(positions)
            or center.shape != (3,) or not np.isfinite(center).all()
            or np.any(center < 0) or np.any(center >= length)
            or stored.shape != (12,)):
        raise ValueError("Invalid coordinate, center, count, or full-clock row")
    number = int(number)
    current = positions[:number]
    if (not np.isfinite(current).all() or np.any(current < 0)
            or np.any(current >= length) or not np.isnan(positions[number:]).all()):
        raise ValueError("Used coordinates must be in the torus and unused slots padded by NaN")
    offsets = current[:, None, :] - current[None, :, :]
    offsets -= length * np.rint(offsets / length)
    distances = np.linalg.norm(offsets, axis=2)
    if number > 1 and np.any(distances[np.triu_indices(number, 1)] < sigma):
        raise ValueError("Physical snapshot violates hard-sphere support")
    to_center = current - center
    to_center -= length * np.rint(to_center / length)
    minimum = float(np.min(np.linalg.norm(to_center, axis=1))) if number else length
    if minimum < radius:
        raise ValueError("Physical snapshot violates cavity support")
    if abs(minimum - stored[7]) > 1e-9:
        raise ValueError("Center-distance reconstruction disagrees with its full-clock row")
    bragg, deleted_bragg = deletion_bragg(current, length, cells)
    if abs(bragg - stored[3]) > 1e-9:
        raise ValueError("Bragg reconstruction disagrees with its full-clock row")
    deletion_shift = int(np.sum(deleted_bragg > .45) - number * (bragg > .45))
    if deletion_shift != stored[8]:
        raise ValueError("Delete-one Bragg label reconstruction disagrees with its full-clock row")
    global_order, local_order, coordination, deleted_order = bond_order(
        current, length, cutoff, with_deletions=True)
    q_labels = {}
    for threshold in thresholds:
        ordered = bool(number >= minimum_particles and global_order > threshold)
        deleted = (number - 1 >= minimum_particles) & (deleted_order > threshold)
        q_labels[str(threshold)] = {
            "ordered": ordered,
            "signed_delete_one_label_change": int(np.sum(deleted) - number * ordered),
        }
    return {
        "N": number, "B": bragg, "Q6_global": global_order,
        "q6_local_mean": local_order, "mean_neighbors": coordination,
        "minimum_center_distance": minimum,
        "Bragg_ordered_at_0.45": bool(bragg > .45),
        "Bragg_signed_delete_one_label_change": deletion_shift,
        "Q6_labels": q_labels,
    }


def analyze(path, cutoff_over_sigma=1.4, thresholds=(.30, .35, .40),
            minimum_particles=20):
    if (not np.isfinite(cutoff_over_sigma) or cutoff_over_sigma <= 0
            or not thresholds or not np.isfinite(thresholds).all()
            or any(not 0 < threshold < 1 for threshold in thresholds)
            or minimum_particles != int(minimum_particles) or minimum_particles < 1):
        raise ValueError("Positive cutoff/minimum count and Q6 thresholds strictly between zero and one required")
    path = Path(path).resolve()
    data, metadata, inputs = load_validated_run(path)
    if "coordinate_shape" not in data:
        raise ValueError("Single-host structural replay requires aligned coordinate snapshots")
    if np.prod(data["coordinate_shape"]) * 8 > 128 * 1024**2:
        raise MemoryError("Coordinate archive exceeds the 128 MiB structural replay bound")
    samples = data["samples"]
    times = data["coordinate_clock_steps"]
    labels = data["coordinate_labels"].astype(int)
    indices = (times - samples[0, 0]).astype(int)
    volume = float(data["volume"])
    length = volume ** (1 / 3)
    sigma = float(data["diameter"][int(data["base_node"])])
    cells = round((int(data["sites"]) / 4) ** (1 / 3))
    if int(data["sites"]) != 4 * cells**3:
        raise ValueError("Reference site count must form a complete periodic FCC lattice")
    cutoff = cutoff_over_sigma * sigma
    if cutoff >= length / 2:
        raise ValueError("First-shell cutoff must be smaller than half the box length")
    with np.load(path, allow_pickle=False) as archive:
        positions = archive["coordinate_configurations"]
        if "coordinate_centers" not in archive:
            raise ValueError("Independent cavity-support replay requires aligned center snapshots")
        centers = archive["coordinate_centers"]
    if centers.shape != (len(times), 3):
        raise ValueError("Center snapshots must align one-to-one with coordinate frames")
    physical = set(map(int, data["physical_nodes"]))
    records = []
    for frame, (clock, node, row_index) in enumerate(zip(times, labels, indices)):
        if node not in physical:
            continue
        row = samples[row_index]
        record = snapshot_control(
            positions[frame], centers[frame], row[2], row, length, sigma,
            float(data["radius"][node]), cells, cutoff, thresholds, minimum_particles)
        record.update(clock_step=int(clock), node=int(node),
                      regeneration_cycle_id=int(row[10]),
                      empty_reference_sites=int(row[5]),
                      excess_reference_site_occupancy=int(row[6]))
        records.append(record)
    profiles = []
    for node in sorted(physical):
        selected = [record for record in records if record["node"] == node]
        q_profile = {}
        for threshold in thresholds:
            key = str(threshold)
            q_profile[key] = {
                "ordered_snapshots": sum(record["Q6_labels"][key]["ordered"] for record in selected),
                "low_B_ordered_snapshots": sum(
                    not record["Bragg_ordered_at_0.45"] and record["Q6_labels"][key]["ordered"]
                    for record in selected),
                "high_B_disordered_snapshots": sum(
                    record["Bragg_ordered_at_0.45"] and not record["Q6_labels"][key]["ordered"]
                    for record in selected),
            }
        profiles.append({
            "node": node, "activity": float(data["activity"][node]),
            "radius_over_sigma": float(data["radius"][node] / sigma),
            "snapshots": len(selected),
            "distinct_recorded_regeneration_cycle_ids": len({record["regeneration_cycle_id"] for record in selected}),
            "N_support": sorted({record["N"] for record in selected}),
            "Bragg_ordered_snapshots": sum(record["Bragg_ordered_at_0.45"] for record in selected),
            "Q6_threshold_controls": q_profile,
        })
    result = {
        "scope": "Descriptive coordinate-based structural replay, not full-clock physical phase probabilities or a convergence certificate",
        "raw_file": display_path(path), "raw_sha256": metadata["raw_sha256"],
        "cutoff_over_sigma": cutoff_over_sigma, "Q6_thresholds": list(thresholds),
        "Q6_minimum_particles": int(minimum_particles),
        "phase_label_scope": "Operational Q6 thresholds with a minimum host count; they are not universal thermodynamic phase definitions. Fourier B alone can miss rotated or incommensurate order",
        "snapshot_sampling_scope": "Sparse fixed-clock coordinate frames retain their exact full-clock row and regeneration ID. They cannot replace every-clock cycle rewards or be counted as independent physical observations",
        "recorded_cycle_id_scope": "Snapshot cycle IDs include leading and terminal incomplete production cycles. Distinct IDs are not counts of completed regeneration cycles; only the full-clock cycle analyzer determines completion",
        "profile": profiles, "records": records,
    }
    return result, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff-over-sigma", type=float, default=1.4)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.30, .35, .40])
    parser.add_argument("--minimum-particles", type=int, default=20)
    args = parser.parse_args()
    result, inputs = analyze(args.input, args.cutoff_over_sigma, tuple(args.thresholds), args.minimum_particles)
    output = args.output.resolve()
    sidecar = output.with_suffix(".metadata.json")
    if output.suffix != ".json" or {output, sidecar}.intersection(inputs):
        parser.error("Choose a JSON output distinct from every raw/source/input file")
    if output.exists() or sidecar.exists():
        parser.error("Structural replay output prefix must be new")
    snapshots = preserve_execution_sources()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    sidecar.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/continuum_tempering_structure.py", "script_sha256": SOURCE_SHA256,
        "source_snapshot": snapshots["structural_replay"][0],
        "source_snapshots": snapshots,
        "dependency_sha256": {
            display_path(path): hashlib.sha256(payload).hexdigest()
            for name, (path, payload) in EXECUTION_SOURCES.items()
            if name != "structural_replay"},
        "input_sha256": {display_path(path): sha256(path) for path in sorted(inputs)},
        "output_sha256": {display_path(output): sha256(output)},
        "reproduce_argv": ["python3", "-m", "scripts.continuum_tempering_structure",
                           display_path(args.input), "--output", display_path(output),
                           "--cutoff-over-sigma", str(args.cutoff_over_sigma), "--thresholds",
                           *map(str, args.thresholds), "--minimum-particles", str(args.minimum_particles)],
    }, indent=2) + "\n")
    print(json.dumps({"physical_snapshots": len(result["records"]), "physical_nodes": len(result["profile"])}))


if __name__ == "__main__":
    main()
