#!/usr/bin/env python3
"""Independent Monte Carlo and covariance-split audit of the L=64 curve test.

This script leaves all publication calculations unchanged. It recomputes the
primary bootstrap difference curves, saves them for inspection, tests retained
mode stability in independent halves, and calibrates quadratic statistics
using bootstrap draws not used to estimate the covariance. The retained
correlation eigenvector from the published covariance supplies an additional
fixed-direction variance check; this is a retrospective diagnostic, not a
prospectively specified contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import numpy as np
import pandas as pd

from lattice_analysis_common import contribution_blocks, load_trajectory_archives
from lattice_gas_phase_switch import (K_ISING, exact_pair_cavity_parameters,
                                    exact_spontaneous_magnetization,
                                    observables_from_contributions)
from lattice_leave_lmax_prediction import (
    DEFAULT_RAW, DEFAULT_SEED, H_GRID, PRIMITIVE_NAMES,
    TARGET_LENGTH, TARGETS, TRAINING_LENGTHS, extract_target_curves,
    extract_training_vector, first_order_curve, weighted_sequence_fit)
from lattice_free_intercept_joint_quadratic import (
    load_paired_vectors, joint_quadratic_design, analyze_threshold)
from lattice_pulse_correction_prediction import resample_total
from lattice_validation_statistics import regularized_precision, quadratic_from_whitener
from scipy.linalg import block_diag

ROOT = Path(__file__).resolve().parents[1]


def split_covariance_test(central: np.ndarray, draws: np.ndarray,
                          covariance_indices: np.ndarray,
                          calibration_indices: np.ndarray,
                          threshold: float) -> dict:
    """Fit covariance on one sample and calibrate tails on a disjoint sample."""
    if np.intersect1d(covariance_indices, calibration_indices).size:
        raise ValueError("covariance and calibration samples must be disjoint")
    fitting = draws[covariance_indices]
    calibration = draws[calibration_indices]
    covariance = np.cov(fitting, rowvar=False, ddof=1)
    regularized = regularized_precision(covariance, threshold)
    whitener = regularized["whitener"]
    statistic = float(quadratic_from_whitener(central, whitener))
    calibration_noise = calibration - calibration.mean(axis=0)
    noise_statistics = quadratic_from_whitener(calibration_noise, whitener)
    probability = float((1 + np.count_nonzero(noise_statistics >= statistic)) /
                        (len(noise_statistics) + 1))
    return {
        "relative_threshold": threshold, "rank": regularized["rank"],
        "minimum_retained_relative_eigenvalue":
            regularized["minimum_retained_eigenvalue"] /
            regularized["maximum_eigenvalue"],
        "Q": statistic, "disjoint_calibration_p_value": probability,
        "calibration_mean_Q": float(noise_statistics.mean()),
        "calibration_q95": float(np.quantile(noise_statistics, .95)),
        "fifth_relative_eigenvalue": (float(regularized["eigenvalues"][-5]) /
            regularized["maximum_eigenvalue"] if len(central) >= 5 else np.nan),
    }


def regenerate_primary(draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    trajectories = load_trajectory_archives(DEFAULT_RAW)
    gap = exact_spontaneous_magnetization(K_ISING)
    work = exact_pair_cavity_parameters(K_ISING)[2]
    central_training = np.empty((len(TRAINING_LENGTHS), len(PRIMITIVE_NAMES)))
    bootstrap_training = np.empty((draws, *central_training.shape))
    for index, length in enumerate(TRAINING_LENGTHS):
        blocks, _ = contribution_blocks(trajectories, length, H_GRID, 40, "paired")
        central_observed = observables_from_contributions(blocks.sum(axis=(0, 1)), length**2)
        central_training[index] = extract_training_vector(central_observed, length**2, gap, work)
        rng = np.random.default_rng(seed + 10_003 * length)
        for draw in range(draws):
            observed = observables_from_contributions(resample_total(blocks, rng), length**2)
            bootstrap_training[draw, index] = extract_training_vector(observed, length**2, gap, work)
        print(f"training L={length} complete", flush=True)
    central_training[:, 1] = 0.
    bootstrap_training[:, :, 1] = 0.
    # Freeze the original WLS weights rather than changing the predictor when
    # increasing the number of bootstrap draws. The additional draws then
    # evaluate exactly the same published statistical estimator.
    coefficients = pd.read_csv(ROOT / "data_processed/lattice_leave_lmax_prediction_coefficients.csv")
    finite = coefficients[coefficients.record_type == "finite_volume_training_primitive"]
    limits = np.empty(len(PRIMITIVE_NAMES))
    boot_limits = np.empty((draws, len(PRIMITIVE_NAMES)))
    for quantity_index, quantity in enumerate(PRIMITIVE_NAMES):
        errors = finite[finite.quantity == quantity].sort_values("L").bootstrap_se.to_numpy()
        parameters, draw_parameters = weighted_sequence_fit(
            1. / np.square(TRAINING_LENGTHS), central_training[:, quantity_index],
            bootstrap_training[:, :, quantity_index], errors, 1)
        limits[quantity_index] = parameters[0]
        boot_limits[:, quantity_index] = draw_parameters[:, 0]
    prediction = first_order_curve(H_GRID, TARGET_LENGTH**2, limits[:4], limits[4:], gap, work)[0]
    boot_prediction = first_order_curve(H_GRID, TARGET_LENGTH**2, boot_limits[:, :4], boot_limits[:, 4:], gap, work)
    blocks, _ = contribution_blocks(trajectories, TARGET_LENGTH, H_GRID, 40, "paired")
    target = extract_target_curves(observables_from_contributions(blocks.sum(axis=(0, 1)), TARGET_LENGTH**2), TARGET_LENGTH**2)
    boot_target = np.empty_like(boot_prediction)
    rng = np.random.default_rng(seed + 1_000_003 + 10_003 * TARGET_LENGTH)
    for draw in range(draws):
        boot_target[draw] = extract_target_curves(
            observables_from_contributions(resample_total(blocks, rng), TARGET_LENGTH**2),
            TARGET_LENGTH**2)
    return target - prediction, boot_target - boot_prediction


def scalar_split_controls() -> pd.DataFrame:
    """Check the free quadratic and analytic-intercept tests on disjoint draws."""
    values = load_paired_vectors(ROOT / "data_processed/lattice_scalar_joint_validation.csv")
    noise = values["noise_draws"]
    design, _ = joint_quadratic_design(values["volumes"])
    rng = np.random.default_rng(20260929)
    rows = []
    for split in range(12):
        permutation = rng.permutation(len(noise))
        middle = len(noise) // 2
        for direction, fitting, calibration in (
                ("forward", permutation[:middle], permutation[middle:]),
                ("reverse", permutation[middle:], permutation[:middle])):
            covariance = block_diag(*[
                np.cov(noise[fitting, 3*index:3*index+3], rowvar=False)
                for index in range(5)])
            result = analyze_threshold(values["values"], noise[calibration],
                covariance, values["exact_intercepts"], design, 1e-8)
            rows.append({"split": split, "direction": direction, **result["summary"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=4000, choices=(4000,),
                        help="four thousand draws: original two thousand plus independent extension")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cache-only", action="store_true")
    arguments = parser.parse_args()
    cache = ROOT / "data_processed/reaud_covariance_bootstrap_curves.npz"
    metadata_path = ROOT / "generated/reaud_covariance_stability_metadata.json"
    scientific_inputs = (
        ROOT / "data_processed/lattice_leave_lmax_prediction_coefficients.csv",
        ROOT / "data_processed/lattice_leave_lmax_prediction_covariance.npz",
        ROOT / "data_processed/lattice_scalar_joint_validation.csv")
    raw_digests = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in DEFAULT_RAW}
    input_digests = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in scientific_inputs}
    if arguments.cache_only:
        prior = json.loads(metadata_path.read_text())
        if (prior["raw_sha256"] != raw_digests or
                prior["input_sha256"] != input_digests or
                prior["output_sha256"][str(cache.relative_to(ROOT))] !=
                hashlib.sha256(cache.read_bytes()).hexdigest()):
            raise ValueError("cached bootstrap or its scientific inputs changed; regenerate without --cache-only")
        with np.load(cache) as archive:
            central, boot = archive["central"], archive["bootstrap"]
            arguments.seed = int(archive["seed"])
    else:
        central, boot = regenerate_primary(arguments.draws, arguments.seed)
        np.savez_compressed(cache, central=central, bootstrap=boot,
                            h_grid=H_GRID, target_order=np.asarray(TARGETS),
                            seed=arguments.seed)
    rows = []
    mode_records = []
    all_indices = np.arange(len(boot))
    rng = np.random.default_rng(260929)
    permutations = [rng.permutation(all_indices) for _ in range(12)]
    with np.load(ROOT / "data_processed/lattice_leave_lmax_prediction_covariance.npz") as published:
        for target_index, target in enumerate(TARGETS):
            data = boot[:, target_index]
            for threshold in (1e-6, 1e-8, 1e-9, 1e-10):
                for split, permutation in enumerate([all_indices, *permutations]):
                    fitting = permutation[:len(boot)//2]
                    calibration = permutation[len(boot)//2:]
                    for direction, covariance_indices, calibration_indices in (
                            ("forward", fitting, calibration),
                            ("reverse", calibration, fitting)):
                        record = split_covariance_test(central[target_index], data,
                            covariance_indices, calibration_indices, threshold)
                        rows.append({"target": target, "split": split,
                                     "direction": direction, **record})
            reference_covariance = published[f"covariance_{target}"]
            eig = regularized_precision(reference_covariance, 1e-8)
            scales = np.sqrt(np.diag(reference_covariance))
            vector = eig["retained_eigenvectors"][:, 0] / scales
            projected = data @ vector
            observed = float(central[target_index] @ vector)
            se = float(np.std(projected, ddof=1))
            empirical = (projected - projected.mean()) / se
            own = regularized_precision(np.cov(data, rowvar=False), 1e-12)
            ratios = own["eigenvalues"][::-1] / own["maximum_eigenvalue"]
            replay_covariance = np.cov(data[:2000], rowvar=False)
            covariance_replay_error = float(np.max(np.abs(replay_covariance - reference_covariance)) /
                                           np.max(np.abs(reference_covariance)))
            for subset_name, subset in (("all_4000", projected),
                    ("original_2000", projected[:2000]),
                    ("additional_2000", projected[2000:])):
                subset_se = float(np.std(subset, ddof=1))
                noise = subset - subset.mean()
                mode_records.append({"target": target, "subset": subset_name,
                    "draws": len(subset), "published_mode_rank": int(eig["rank"]),
                    "signed_residual_over_se": observed / subset_se,
                    "bootstrap_bias_over_se": (subset.mean() - observed) / subset_se,
                    "variance_over_published_variance": subset_se**2 / eig["minimum_retained_eigenvalue"],
                    "fixed_contrast_empirical_p_value":
                       (1 + np.count_nonzero(np.abs(noise) >= abs(observed))) / (len(subset) + 1),
                    "full_4000_fifth_relative_eigenvalue": float(ratios[4]),
                    "first_2000_covariance_replay_relative_error": covariance_replay_error})
            rows.append({"target": target, "split": -1, "direction": "published_lowest_mode",
                "relative_threshold": 1e-8, "rank": 1,
                "minimum_retained_relative_eigenvalue": eig["minimum_retained_eigenvalue"] / eig["maximum_eigenvalue"],
                "Q": (observed / se)**2,
                "disjoint_calibration_p_value": (1 + np.count_nonzero(np.abs(empirical) >= abs(observed / se))) / (len(boot) + 1),
                "calibration_mean_Q": float(np.mean(empirical**2)),
                "calibration_q95": float(np.quantile(empirical**2, .95))})
    output = ROOT / "data_processed/reaud_covariance_split_tests.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    scalar_output = ROOT / "data_processed/reaud_scalar_split_tests.csv"
    scalar = scalar_split_controls()
    scalar.to_csv(scalar_output, index=False)
    mode_output = ROOT / "data_processed/reaud_fixed_mode_controls.csv"
    pd.DataFrame(mode_records).to_csv(mode_output, index=False)
    metadata = {"script": str(Path(__file__).relative_to(ROOT)),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "bootstrap_draws": len(boot), "seed": arguments.seed,
                "execution_mode": "cached diagnostic replay" if arguments.cache_only else "bootstrap regeneration and diagnostics",
                "scope": "Independent covariance/calibration subsets; retrospective mode check; no publication analysis overwritten",
                "split_scope": "26 overlapping partitions of one 4000-draw bootstrap, not 26 independent datasets",
                "raw_sha256": raw_digests,
                "input_sha256": input_digests,
                "published_predictor_weights_frozen": True,
                "output_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (cache, output, scalar_output, mode_output)}}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    frame = pd.DataFrame(rows)
    print(frame[frame.split >= 0].groupby(["target", "relative_threshold"]).agg(
        rank_min=("rank", "min"), rank_max=("rank", "max"),
        p_min=("disjoint_calibration_p_value", "min"),
        p_max=("disjoint_calibration_p_value", "max"),
        Q_min=("Q", "min"), Q_max=("Q", "max")).to_string())
    print(frame[frame.split < 0].to_string(index=False))


if __name__ == "__main__":
    main()
