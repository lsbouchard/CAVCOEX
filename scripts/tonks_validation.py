#!/usr/bin/env python3
"""Exact one-dimensional validation of ensemble mixing and its 1/V term."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from hard_sphere_models import (
    tonks_grand_distribution,
    tonks_insertion_probability,
)

TARGET_N = (20, 40, 80, 160, 320)
DENSITIES = (0.10, 0.30, 0.50, 0.70)


def thermodynamic_coefficient(rho: float) -> float:
    """Leading coefficient for the unit-rod particle-sized cavity."""

    w_prime = (2.0 - rho) / (1.0 - rho) ** 2
    w_second = (3.0 - rho) / (1.0 - rho) ** 3
    chi_n = rho * (1.0 - rho) ** 2
    return 0.5 * chi_n * (w_second - w_prime**2)


def tune_x(target_n: int, length: float) -> float:
    return float(
        brentq(
            lambda x: tonks_grand_distribution(x, length).mean - target_n,
            -12.0,
            30.0,
            xtol=2.0e-13,
        )
    )


def make_data() -> pd.DataFrame:
    rows = []
    for rho in DENSITIES:
        coefficient = thermodynamic_coefficient(rho)
        for target_n in TARGET_N:
            length = target_n / rho
            x = tune_x(target_n, length)
            grand = tonks_grand_distribution(x, length)
            insertion = np.asarray(
                [
                    tonks_insertion_probability(int(n), length)
                    for n in grand.n
                ]
            )
            mixture_probability = float(np.dot(grand.probability, insertion))
            identity_probability = grand.mean / length * np.exp(-x)
            canonical_probability = tonks_insertion_probability(target_n, length)
            beta_w_n = -np.log(canonical_probability)
            beta_w_mu = -np.log(mixture_probability)
            rows.append(
                {
                    "N": target_n,
                    "L": length,
                    "rho": rho,
                    "x": x,
                    "betaW_N": beta_w_n,
                    "betaW_mu": beta_w_mu,
                    "L_delta_betaW": length * (beta_w_mu - beta_w_n),
                    "predicted_coefficient": coefficient,
                    "mixture_probability": mixture_probability,
                    "identity_probability": identity_probability,
                    "identity_absolute_error": mixture_probability
                    - identity_probability,
                    "source_type": "analytic_exact",
                    "run_id": "tonks_periodic_v1",
                    "seed": np.nan,
                    "units_convention": "beta=sigma=Lambda=1",
                    "raw_data_path": "none_exact_partition_function",
                    "analysis_version": "1",
                }
            )
    return pd.DataFrame(rows)


def plot_data(data: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.3,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    colors = ["#56B4E9", "#009E73", "#E69F00", "#CC79A7"]
    fig, axis = plt.subplots(figsize=(3.4, 2.55), constrained_layout=True)
    for color, (rho, group) in zip(colors, data.groupby("rho")):
        group = group.sort_values("L")
        coefficient = float(group["predicted_coefficient"].iloc[0])
        axis.plot(
            1.0 / group["L"],
            100.0 * (group["L_delta_betaW"] / coefficient - 1.0),
            "o-",
            markersize=3.5,
            linewidth=0.8,
            color=color,
            label=rf"$\rho\sigma={rho:.1f}$",
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.75)
    axis.set_xlabel(r"$1/L$")
    axis.set_ylabel("deviation from leading coefficient (percent)")
    axis.grid(alpha=0.16, linewidth=0.5)
    axis.legend(frameon=False)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    data = make_data()
    data.to_csv(root / "data_processed" / "tonks_validation.csv", index=False)
    plot_data(data, root / "figures" / "tonks_validation.pdf")
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/tonks_validation.py",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "method": "exact periodic Tonks configurational partition functions",
        "maximum_identity_absolute_error": float(
            np.max(np.abs(data["identity_absolute_error"]))
        ),
    }
    (root / "generated" / "tonks_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "maximum exact-mixture identity error: "
        f"{metadata['maximum_identity_absolute_error']:.3e}"
    )


if __name__ == "__main__":
    main()
