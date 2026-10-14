#!/usr/bin/env python3
"""Orientation-independent bond-order controls for continuum endpoint samples.

Q6 is a structural diagnostic, never a target energy or an imposed phase
probability. All first-shell bonds use the documented fixed cutoff 1.4 sigma.
Snapshots are processed separately, with no all-frames distance array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.special import sph_harm_y

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_digest(path):
    """Hash a trajectory without allocating its compressed contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deletion_bragg(positions, length, cells):
    """Independently reconstruct B and B after each one-particle deletion.

    No sampler cache or its deletion-label diagnostic is used. This makes
    threshold sensitivity meaningful even when the trajectory's stored
    deletion labels were evaluated at one fixed threshold.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if (not np.isfinite(length) or length <= 0 or not np.isfinite(positions).all()
            or not isinstance(cells, (int, np.integer)) or cells < 1):
        raise ValueError("Require finite coordinates, positive box, and integer cells")
    n = len(positions)
    if not n:
        return 0., np.zeros(0)
    modes = cells * np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1],
                              [-1, 1, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
    contributions = np.exp(2j * np.pi * (positions @ modes.T) / length)
    sums = contributions.sum(axis=0)
    current = float(np.abs(sums).mean() / n)
    deleted = (np.abs(sums[None, :] - contributions).mean(axis=1) / (n - 1)
               if n > 1 else np.zeros(1))
    return current, deleted


def bond_order(positions, length, cutoff=1.4, degree=6, with_deletions=False):
    """Return global Q_l, mean local q_l, and mean first-shell coordination.

    theta is the polar angle and phi the azimuth, as required by sph_harm_y.
    Global Q_l averages complex harmonics before taking their norm; local
    q_l takes the norm separately for each particle before averaging.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if (not np.isfinite(length) or not np.isfinite(cutoff) or length <= 0
            or not 0 < cutoff < length / 2
            or not isinstance(degree, (int, np.integer)) or degree < 0):
        raise ValueError("Require a positive box, cutoff below half-box, and nonnegative degree")
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite coordinates")
    n = len(positions)
    if n < 2:
        result = (0., 0., 0.)
        return result + (np.zeros(n),) if with_deletions else result
    if n * n * 3 * 8 > 128 * 1024**2:
        raise MemoryError("Dense bond-vector calculation exceeds 128 MiB")
    vectors = positions[:, None, :] - positions[None, :, :]
    vectors -= length * np.rint(vectors / length)
    distances = np.linalg.norm(vectors, axis=2)
    mask = (distances > 0) & (distances < cutoff)
    rows, cols = np.nonzero(mask)
    if not len(rows):
        result = (0., 0., 0.)
        return result + (np.zeros(n),) if with_deletions else result
    bonds = vectors[rows, cols]
    radii = distances[rows, cols]
    theta = np.arccos(np.clip(bonds[:, 2] / radii, -1., 1.))
    phi = np.arctan2(bonds[:, 1], bonds[:, 0])
    harmonics = sph_harm_y(degree, np.arange(-degree, degree + 1)[:, None],
                          theta[None, :], phi[None, :])
    prefactor = 4 * np.pi / (2 * degree + 1)
    global_order = np.sqrt(prefactor * np.sum(abs(harmonics.mean(axis=1))**2))
    coordination = np.bincount(rows, minlength=n)
    local_modes = np.zeros((2 * degree + 1, n), dtype=complex)
    for m in range(2 * degree + 1):
        np.add.at(local_modes[m], rows, harmonics[m])
    if with_deletions:
        # Removing particle i removes its outgoing and incoming bonds.
        # Y_lm(-u)=(-1)^l Y_lm(u), so both directions are accounted for
        # without reclassifying any bonds between surviving particles.
        remaining_modes = (harmonics.sum(axis=1)[:, None]
                           - (1 + (-1)**degree) * local_modes)
        remaining_bonds = len(rows) - 2 * coordination
        remaining_modes /= np.maximum(remaining_bonds, 1)[None, :]
        deleted_global = np.sqrt(prefactor * np.sum(abs(remaining_modes)**2, axis=0))
        deleted_global[remaining_bonds == 0] = 0.
    local_modes /= np.maximum(coordination, 1)[None, :]
    local_order = np.sqrt(prefactor * np.sum(abs(local_modes)**2, axis=0))
    result = (float(global_order), float(local_order.mean()), float(coordination.mean()))
    return result + (deleted_global,) if with_deletions else result


def number_umbrella_stiffness(data, replicas, cap=None):
    """Validate optional number-bias fields independently of the sampler.

    Legacy trajectories without either field have zero number bias. A
    partially recorded bias cannot be interpreted as an unbiased target.
    """
    present = ("number_stiffness" in data, "number_preferred" in data)
    if not any(present):
        return np.zeros(replicas)
    if not all(present):
        raise ValueError("Number-umbrella fields must be recorded together")
    strength = np.asarray(data["number_stiffness"], dtype=float)
    preferred = np.asarray(data["number_preferred"], dtype=float)
    if (strength.shape != (replicas,) or preferred.shape != (replicas,)
            or not np.isfinite(strength).all() or not np.isfinite(preferred).all()
            or np.any(strength < 0) or np.any(preferred < 0)
            or (cap is not None and np.any(preferred > cap))):
        raise ValueError("Number-umbrella parameters are invalid")
    return strength


def probe_structure(data, samples, length, cells, stride, cutoff, sigma=1.):
    """Align sparse probe coordinates with their original diagnostic frames."""
    if "physical_probe_configurations" not in data:
        raise ValueError("No physical probe coordinate snapshots")
    positions = data["physical_probe_configurations"]
    nodes = data["probe_nodes"]
    activities = data["physical_probe_activity"]
    indices = (data["probe_sample_indices"] if "probe_sample_indices" in data
               else np.arange(len(samples)))
    if (positions.ndim != 4 or positions.shape[0] != len(indices)
            or positions.shape[1] != len(nodes) or len(activities) != len(nodes)
            or nodes.ndim != 1 or not np.issubdtype(nodes.dtype, np.integer)
            or np.any(nodes < 0) or np.any(nodes >= samples.shape[1])
            or len(np.unique(nodes)) != len(nodes)
            or activities.ndim != 1 or not np.isfinite(activities).all()
            or positions.shape[-1] != 3 or indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(samples))
            or np.any(np.diff(indices) <= 0)):
        raise ValueError("Probe coordinate and diagnostic alignment is invalid")
    number_strength = number_umbrella_stiffness(data, samples.shape[1], positions.shape[2])
    if (not np.isfinite(sigma) or sigma <= 0
            or not np.all(data["pair_penalty"][nodes] == np.inf)
            or np.any(data["umbrella_stiffness"][nodes] != 0)
            or np.any(number_strength[nodes] != 0)
            or np.any(data["radius"][nodes] != 0)
            or np.any(data["diameter"][nodes] != sigma)
            or not np.array_equal(data["activity"][nodes], activities)):
        raise ValueError("Activity probes are not unbiased hard R=0 targets")
    summaries = []
    for slot, (node, activity) in enumerate(zip(nodes, activities)):
        records = []
        for coordinate_frame in range(0, len(indices), stride):
            frame = int(indices[coordinate_frame])
            number = int(samples[frame, node, 0])
            if number != samples[frame, node, 0] or not 0 <= number <= positions.shape[2]:
                raise ValueError("Probe particle count is invalid")
            current = positions[coordinate_frame, slot, :number]
            bragg, _ = deletion_bragg(current, length, cells)
            if abs(bragg - samples[frame, node, 1]) > 1e-8:
                raise ValueError("Probe Bragg reconstruction disagrees with saved diagnostic")
            qglobal, qlocal, coordination = bond_order(current, length, cutoff)
            records.append({"frame": frame, "N": number, "B": bragg,
                            "Q6_global": qglobal, "q6_local_mean": qlocal,
                            "mean_neighbors": coordination})
        ordered = [r for r in records if r["B"] > .65 and r["N"] >= 20]
        quarters = []
        for quarter in range(4):
            selected = [r for r in records if min(4 * r["frame"] // len(samples), 3) == quarter]
            quarters.append({
                "snapshots": len(selected),
                "high_B_fraction": float(np.mean([r["B"] > .65 and r["N"] >= 20
                                                  for r in selected])) if selected else None,
                "high_Q6_fraction": float(np.mean([r["Q6_global"] > .35 and r["N"] >= 20
                                                   for r in selected])) if selected else None,
            })
        summaries.append({
            "node": int(node), "x": float(activity), "snapshots": len(records),
            "high_B_snapshots": len(ordered),
            "high_B_Q6_mean": float(np.mean([r["Q6_global"] for r in ordered])) if ordered else None,
            "low_B_high_Q6_snapshots": sum(r["B"] <= .30 and r["Q6_global"] > .35
                                           for r in records),
            "quarters": quarters, "records": records,
        })
    return summaries


def analyze(path, stride=10, cutoff=1.4, thresholds=(.40, .45, .50), include_probes=False):
    if (not isinstance(stride, (int, np.integer)) or stride < 1
            or not np.isfinite(cutoff) or cutoff <= 0
            or not len(thresholds) or not np.isfinite(thresholds).all()
            or any(not 0 <= threshold <= 1 for threshold in thresholds)):
        raise ValueError("Require positive integer stride, positive cutoff, and thresholds in [0,1]")
    metadata = json.loads(path.with_suffix(".json").read_text())
    digest = file_digest(path)
    if digest != metadata["raw_sha256"]:
        raise ValueError(f"Raw digest differs from execution record: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "endpoint_configurations" not in data:
            raise ValueError(f"No positional endpoint snapshots in {path}")
        declared_number_bias = float(metadata["settings"].get("compact_number_stiffness", 0.))
        if not np.isfinite(declared_number_bias) or declared_number_bias < 0:
            raise ValueError("Declared number-umbrella stiffness is invalid")
        if declared_number_bias > 0 and ("number_stiffness" not in data
                                        or "number_preferred" not in data):
            raise ValueError("Declared number umbrella lacks its recorded parameter arrays")
        samples = data["samples"]
        positions = data["endpoint_configurations"]
        endpoint = int(data["endpoint"])
        if (samples.ndim != 3 or not len(samples) or samples.shape[2] < 2
                or not 0 <= endpoint < samples.shape[1]
                or positions.ndim != 4 or positions.shape[-1] != 3):
            raise ValueError("Endpoint coordinate or diagnostic axes are invalid")
        cavity_nodes = (data["cavity_nodes"] if "cavity_nodes" in data
                        else np.arange(endpoint, len(data["diameter"])))
        if (cavity_nodes.ndim != 1 or len(cavity_nodes) < 2
                or not np.issubdtype(cavity_nodes.dtype, np.integer)
                or np.any(cavity_nodes < 0) or np.any(cavity_nodes >= samples.shape[1])
                or len(np.unique(cavity_nodes)) != len(cavity_nodes)):
            raise ValueError("Cavity path node indices are invalid")
        cavity_endpoint = int(cavity_nodes[-1])
        if int(cavity_nodes[0]) != endpoint:
            raise ValueError("Cavity path does not start at the physical base")
        if positions.shape[:2] != (len(samples), 2):
            raise ValueError("Snapshot and diagnostic axes differ")
        volume = float(data["volume"])
        sigma = float(data["diameter"][endpoint])
        physical = cavity_nodes
        number_strength = number_umbrella_stiffness(data, samples.shape[1], positions.shape[2])
        if declared_number_bias > 0 and not np.any(number_strength > 0):
            raise ValueError("Declared number umbrella has no recorded nonzero auxiliary stiffness")
        if (not np.isfinite(volume) or volume <= 0
                or not np.isfinite(sigma) or sigma <= 0
                or np.any(data["diameter"][physical] != sigma)
                or np.any(data["pair_penalty"][physical] != np.inf)
                or np.any(data["umbrella_stiffness"][physical] != 0)
                or np.any(number_strength[physical] != 0)
                or data["radius"][endpoint] != 0
                or data["radius"][cavity_endpoint] != sigma
                or np.any(np.diff(data["radius"][physical]) <= 0)
                or not np.isfinite(data["activity"][physical]).all()
                or np.any(data["activity"][physical] != data["activity"][endpoint])):
            raise ValueError("Endpoints are not matched unbiased hard R=0 and R=sigma targets")
        length = volume**(1 / 3)
        sites = int(metadata["settings"]["sites"])
        cells = round((sites / 4)**(1 / 3))
        if sites != 4 * cells**3:
            raise ValueError("Reference site count is not a complete periodic FCC lattice")
        records = []
        for frame in range(0, len(samples), stride):
            for target, replica in enumerate((endpoint, cavity_endpoint)):
                number = int(samples[frame, replica, 0])
                if number != samples[frame, replica, 0] or not 0 <= number <= positions.shape[2]:
                    raise ValueError("Endpoint particle count is invalid")
                qglobal, qlocal, coordination, deleted_q6 = bond_order(
                    positions[frame, target, :number], length, cutoff * sigma,
                    with_deletions=True)
                reconstructed, deleted = deletion_bragg(
                    positions[frame, target, :number], length, cells)
                if abs(reconstructed - samples[frame, replica, 1]) > 1e-8:
                    raise ValueError("Independent Bragg reconstruction disagrees with saved diagnostic")
                records.append({
                    "frame": frame, "endpoint": target, "N": number,
                    "fcc_bragg_mean": float(samples[frame, replica, 1]),
                    "Q6_global": qglobal, "q6_local_mean": qlocal,
                    "mean_neighbors": coordination,
                    "delete_solid_counts": [int(np.sum(deleted > threshold))
                                            for threshold in thresholds],
                    "delete_Q6_solid_count": int(np.sum(deleted_q6 > .35)),
                })
        probes = probe_structure(data, samples, length, cells, stride, cutoff * sigma,
                                 sigma) if include_probes else None
    summaries = {}
    for target, name in enumerate(("unconstrained", "particle_cavity")):
        rows = [row for row in records if row["endpoint"] == target]
        summaries[name] = {}
        for basin, low, high in (("disordered", -np.inf, .30),
                                 ("intermediate", .30, .65),
                                 ("ordered", .65, np.inf)):
            selected = [r for r in rows if low < r["fcc_bragg_mean"] <= high]
            summaries[name][basin] = {
                "snapshots": len(selected),
                "Q6_global_mean": float(np.mean([r["Q6_global"] for r in selected])) if selected else None,
                "q6_local_mean": float(np.mean([r["q6_local_mean"] for r in selected])) if selected else None,
                "mean_neighbors": float(np.mean([r["mean_neighbors"] for r in selected])) if selected else None,
            }
    base = [r for r in records if r["endpoint"] == 0]
    cavity = [r for r in records if r["endpoint"] == 1]
    mean_number = float(np.mean([r["N"] for r in base]))
    controls = []
    for index, threshold in enumerate(thresholds):
        labels = np.array([r["fcc_bragg_mean"] > threshold for r in base])
        counts = np.array([r["N"] for r in base])
        deletion_counts = np.array([r["delete_solid_counts"][index] for r in base])
        change = deletion_counts - counts * labels
        controls.append({
            "threshold": float(threshold),
            "base_solid_fraction": float(labels.mean()),
            "direct_cavity_solid_fraction": float(np.mean(
                [r["fcc_bragg_mean"] > threshold for r in cavity])),
            "deletion_reconstructed_cavity_solid_fraction":
                float(deletion_counts.mean() / mean_number) if mean_number > 0 else None,
            "signed_deletion_label_correction":
                float(change.mean() / mean_number) if mean_number > 0 else None,
            "absolute_deletion_label_bound":
                float(np.abs(change).mean() / mean_number) if mean_number > 0 else None,
        })
    q6_labels = np.array([r["Q6_global"] > .35 for r in base])
    q6_counts = np.array([r["delete_Q6_solid_count"] for r in base])
    numbers = np.array([r["N"] for r in base])
    q6_control = {
        "threshold": .35,
        "interpretation": "Independent operational bond-order label, not an exhaustive proof of crystallinity",
        "base_ordered_fraction": float(q6_labels.mean()),
        "direct_cavity_ordered_fraction": float(np.mean([r["Q6_global"] > .35 for r in cavity])),
        "deletion_reconstructed_cavity_ordered_fraction":
            float(q6_counts.mean() / mean_number) if mean_number > 0 else None,
        "absolute_deletion_label_bound":
            float(np.abs(q6_counts - numbers * q6_labels).mean() / mean_number)
            if mean_number > 0 else None,
        "low_B_high_Q6_snapshots": sum(r["fcc_bragg_mean"] <= .30 and r["Q6_global"] > .35
                                       for r in records),
        "high_B_low_Q6_snapshots": sum(r["fcc_bragg_mean"] > .65 and r["Q6_global"] <= .35
                                       for r in records),
    }
    return {"raw_file": str(path), "raw_sha256": digest, "summaries": summaries,
            "threshold_controls": controls,
            "Q6_phase_control": q6_control,
            "probe_structure": probes,
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--snapshot-stride", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=1.4)
    parser.add_argument("--include-probes", action="store_true",
                        help="Also reconstruct order in sparsely saved hard activity probes")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data_processed/continuum_structural_audit.json")
    args = parser.parse_args()
    if args.snapshot_stride <= 0 or args.cutoff <= 0:
        parser.error("Snapshot stride and cutoff must be positive")
    result = {"settings": {"snapshot_stride": args.snapshot_stride,
                           "cutoff_sigma": args.cutoff,
                           "include_probes": args.include_probes,
                           "harmonic_degree": 6},
              "interpretation": "Structural controls; not independent equilibrium phase observations",
              "threshold_interpretation": "Snapshot arithmetic ratios only; no equilibration or uncertainty certification",
              "script_sha256": SOURCE_SHA256,
              "runs": {p.stem: analyze(p, args.snapshot_stride, args.cutoff,
                                      include_probes=args.include_probes)
                       for p in args.inputs}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, run in result["runs"].items():
        print(json.dumps({"run": name, "summaries": run["summaries"]}))


if __name__ == "__main__":
    main()
