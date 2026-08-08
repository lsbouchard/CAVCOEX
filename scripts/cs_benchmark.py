#!/usr/bin/env python3
"""Reproducible hard-sphere equation-of-state benchmark for manuscript N3.

The output contains analytic EOS predictions only.  It does not contain or
purport to contain independent simulation measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = 2001
ETA_MAX = 0.52
CHECK_SEED = 20260724

# Fernandez et al., PRL 108, 165701 (2012), Table I:
# extrapolated coexisting-fluid specific volume v_f/sigma^3 = 1.06448(10).
V_FLUID = 1.06448
V_FLUID_STD = 0.00010
ETA_FLUID = math.pi / (6.0 * V_FLUID)
ETA_FLUID_STD = ETA_FLUID * V_FLUID_STD / V_FLUID

ETA = sp.symbols("eta", nonnegative=True)
X = sp.symbols("x", nonnegative=True)

# Pieprzyk et al., PCCP 21, 6886-6899 (2019), Eq. (4).
# Decimal strings make the printed coefficients exact SymPy rationals.
MKLM_COEFFICIENTS = {
    1: "4",
    2: "6",
    3: "2.3647684",
    4: "-0.8698551",
    5: "1.1062803",
    6: "-1.1014221",
    7: "0.66605866",
    8: "-0.03633431",
    10: "-0.20965164",
    14: "0.10555569",
    22: "-0.00872380",
}


@dataclass(frozen=True)
class EOSModel:
    key: str
    label: str
    citation: str
    z_expr: sp.Expr
    a_ex_expr: sp.Expr

    @property
    def dz_expr(self) -> sp.Expr:
        return sp.diff(self.z_expr, ETA)

    @property
    def inv_s0_expr(self) -> sp.Expr:
        return sp.factor(self.z_expr + ETA * self.dz_expr)

    @property
    def s0_expr(self) -> sp.Expr:
        return 1 / self.inv_s0_expr

    @property
    def mu_ex_expr(self) -> sp.Expr:
        # beta*mu_ex = beta*a_ex + Z - 1 for a one-component fluid.
        return self.a_ex_expr + self.z_expr - 1

    @property
    def dmu_ex_expr(self) -> sp.Expr:
        return sp.diff(self.mu_ex_expr, ETA)


def _j_integral(n: int, x: sp.Expr) -> sp.Expr:
    """Integral J_n(x) = int_0^x t^(n-1)/(1+t) dt for integer n >= 1."""
    polynomial = sum(
        (-1) ** (n - 1 - k) * x**k / sp.Integer(k) for k in range(1, n)
    )
    return polynomial + (-1) ** (n - 1) * sp.log(1 + x)


def build_models() -> tuple[EOSModel, ...]:
    """Build all EOSs and derive free energies from their pressure relations."""
    z_cs = (1 + ETA + ETA**2 - ETA**3) / (1 - ETA) ** 3
    # Integral of (Z-1)/eta from zero to eta.
    a_cs = ETA * (4 - 3 * ETA) / (1 - ETA) ** 2

    z_pyc = (1 + ETA + ETA**2) / (1 - ETA) ** 3
    a_pyc = -sp.log(1 - ETA) + sp.Rational(3, 2) * (
        (1 - ETA) ** -2 - 1
    )

    x_eta = ETA / (1 - ETA)
    z_mklm = 1 + sum(
        sp.Rational(value) * x_eta**power
        for power, value in MKLM_COEFFICIENTS.items()
    )
    a_mklm = sum(
        sp.Rational(value) * _j_integral(power, x_eta)
        for power, value in MKLM_COEFFICIENTS.items()
    )

    return (
        EOSModel(
            "CS",
            "Carnahan-Starling",
            "Carnahan and Starling (1969), doi:10.1063/1.1672048",
            z_cs,
            a_cs,
        ),
        EOSModel(
            "PY-c",
            "Percus-Yevick (compressibility)",
            "Thiele (1963), doi:10.1063/1.1734272",
            z_pyc,
            a_pyc,
        ),
        EOSModel(
            "mKLM-2019",
            "modified KLM (2019)",
            "Pieprzyk et al. (2019), doi:10.1039/C9CP00903E, Eq. (4)",
            z_mklm,
            a_mklm,
        ),
    )


MODELS = build_models()


def _array_function(expr: sp.Expr):
    return sp.lambdify(ETA, expr, modules="numpy")


def evaluate(model: EOSModel, eta: np.ndarray | float) -> dict[str, np.ndarray]:
    """Evaluate quantities derived consistently from a model's Z(eta)."""
    eta_array = np.asarray(eta, dtype=float)
    z = np.asarray(_array_function(model.z_expr)(eta_array), dtype=float)
    mu_ex = np.asarray(_array_function(model.mu_ex_expr)(eta_array), dtype=float)
    inv_s0 = np.asarray(_array_function(model.inv_s0_expr)(eta_array), dtype=float)
    s0 = 1.0 / inv_s0
    d_w_d_mu = 1.0 - s0
    return {
        "eta": eta_array,
        "Z": z,
        "beta_mu_ex": mu_ex,
        "S0": s0,
        "dW_dmu": d_w_d_mu,
    }


def symbolic_consistency_checks() -> dict[str, object]:
    """Exact algebra plus a Sturm certificate for the empirical mKLM fit."""
    for model in MODELS:
        residual = sp.cancel(
            sp.diff(model.a_ex_expr, ETA) - (model.z_expr - 1) / ETA
        )
        if residual != 0:
            raise AssertionError(f"{model.key}: a_ex is inconsistent with Z")

    cs = MODELS[0]
    pyc = MODELS[1]
    expected_cs_inv_s0 = (
        1 + 4 * ETA + 4 * ETA**2 - 4 * ETA**3 + ETA**4
    ) / (1 - ETA) ** 4
    expected_pyc_inv_s0 = (1 + 2 * ETA) ** 2 / (1 - ETA) ** 4
    if sp.cancel(cs.inv_s0_expr - expected_cs_inv_s0) != 0:
        raise AssertionError("CS differentiation identity failed")
    if sp.cancel(pyc.inv_s0_expr - expected_pyc_inv_s0) != 0:
        raise AssertionError("PY-c differentiation identity failed")

    # For mKLM use x=eta/(1-eta).  Since d/deta=(1+x)^2 d/dx,
    # Z+eta Z' = Z+x(1+x)dZ/dx.  Certify q(x)-1 > 0 on
    # 0 < x <= 13/12 (eta <= 0.52) by exact Sturm root counting.
    z_x = 1 + sum(
        sp.Rational(value) * X**power
        for power, value in MKLM_COEFFICIENTS.items()
    )
    q_x = sp.expand(z_x + X * (1 + X) * sp.diff(z_x, X))
    positive_factor = sp.Poly(sp.cancel((q_x - 1) / X), X)
    x_max = sp.Rational(13, 12)
    sturm_roots = positive_factor.count_roots(0, x_max)
    endpoint_0 = positive_factor.eval(0)
    endpoint_max = positive_factor.eval(x_max)
    if not (sturm_roots == 0 and endpoint_0 > 0 and endpoint_max > 0):
        raise AssertionError("mKLM monotonicity certificate failed")

    return {
        "thermodynamic_integral_identity": "exact for all models",
        "CS_invS_minus_1_numerator": "2*eta*(4-eta)",
        "PYc_invS_minus_1_numerator": "eta*(4-eta)*(eta^2+2)",
        "mKLM_sturm_roots_on_x_0_to_13_over_12": int(sturm_roots),
        "mKLM_positive_factor_at_x_0": float(endpoint_0),
        "mKLM_positive_factor_at_x_13_over_12": float(endpoint_max),
    }


def numerical_checks(eta_grid: np.ndarray) -> dict[str, dict[str, float]]:
    """Dense-grid checks of monotonicity and physical response bounds."""
    result: dict[str, dict[str, float]] = {}
    for model in MODELS:
        values = evaluate(model, eta_grid)
        inv_s0 = 1.0 / values["S0"]
        positive_eta = eta_grid > 0
        total_mu_slope = inv_s0[positive_eta] / eta_grid[positive_eta]
        mu_ex_slope = (inv_s0[positive_eta] - 1.0) / eta_grid[positive_eta]

        if not np.all(np.diff(values["beta_mu_ex"]) > 0):
            raise AssertionError(f"{model.key}: beta*mu_ex is not monotone")
        if not np.all(total_mu_slope > 0):
            raise AssertionError(f"{model.key}: beta*mu is not monotone")
        if not np.all(mu_ex_slope > 0):
            raise AssertionError(f"{model.key}: beta*mu_ex derivative is not positive")
        if not (
            np.all(values["dW_dmu"] >= -1e-14)
            and np.all(values["dW_dmu"] < 1)
        ):
            raise AssertionError(f"{model.key}: dW/dmu is outside [0,1)")

        result[model.key] = {
            "min_total_beta_mu_slope_eta_gt_0": float(np.min(total_mu_slope)),
            "min_beta_mu_ex_slope_eta_gt_0": float(np.min(mu_ex_slope)),
            "min_S0": float(np.min(values["S0"])),
            "max_dW_dmu": float(np.max(values["dW_dmu"])),
        }
    return result


def five_point_derivative(function, x: float, h: float = 2.0e-5) -> float:
    """Fourth-order centered finite-difference derivative."""
    return float(
        (
            function(x - 2 * h)
            - 8 * function(x - h)
            + 8 * function(x + h)
            - function(x + 2 * h)
        )
        / (12 * h)
    )


def random_finite_difference_checks(
    samples: int = 24, seed: int = CHECK_SEED
) -> dict[str, float]:
    """Compare symbolic derivatives with finite differences at random eta."""
    rng = np.random.default_rng(seed)
    etas = rng.uniform(0.02, 0.50, size=samples)
    maxima: dict[str, float] = {}
    for model in MODELS:
        for quantity, expr, derivative in (
            ("Z", model.z_expr, model.dz_expr),
            ("beta_mu_ex", model.mu_ex_expr, model.dmu_ex_expr),
        ):
            function = _array_function(expr)
            exact_function = _array_function(derivative)
            errors = []
            for eta in etas:
                exact = float(exact_function(eta))
                finite = five_point_derivative(function, float(eta))
                errors.append(abs(finite - exact) / max(abs(exact), 1.0))
            key = f"{model.key}:{quantity}"
            maxima[key] = float(max(errors))
            if maxima[key] >= 1.0e-8:
                raise AssertionError(
                    f"{key}: finite-difference relative error {maxima[key]:.3e}"
                )
    return maxima


def write_csv(path: Path, eta_grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "eta", "Z", "beta_mu_ex", "S0", "dW_dmu"))
        for model in MODELS:
            values = evaluate(model, eta_grid)
            for row in zip(
                values["eta"],
                values["Z"],
                values["beta_mu_ex"],
                values["S0"],
                values["dW_dmu"],
            ):
                writer.writerow((model.key, *(f"{float(value):.12g}" for value in row)))


def latex_rows(etas: Iterable[float]) -> list[str]:
    def format_significant(value: float, digits: int = 5) -> str:
        if value == 0.0:
            return "0"
        decimal_places = max(
            0, digits - 1 - int(math.floor(math.log10(abs(value))))
        )
        return f"{value:.{decimal_places}f}"

    rows: list[str] = []
    for eta in etas:
        eta_text = f"{eta:.3f}" if eta in (0.492, 0.50) else f"{eta:.2f}"
        for model in MODELS:
            values = evaluate(model, float(eta))
            rows.append(
                f"{model.key} & {eta_text} & "
                f"{format_significant(float(values['Z']))} & "
                f"{format_significant(float(values['beta_mu_ex']))} & "
                f"{format_significant(float(values['S0']))} & "
                f"{format_significant(float(values['dW_dmu']))} \\\\"
            )
    return rows


def write_latex_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(latex_rows((0.10, 0.30, 0.45, 0.492, 0.50)))
    content = (
        "% Generated by scripts/cs_benchmark.py; EOS predictions, not simulation data.\n"
        "\\begin{tabular}{@{}llrrrr@{}}\n"
        "\\toprule\n"
        "Model & $\\eta$ & $Z$ & $\\beta\\mu_{\\rm ex}$ & $S(0)$"
        " & $1-S(0)$ \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    path.write_text(content, encoding="utf-8")


def write_caption(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    caption = (
        "Equation-of-state predictions for a particle-sized hard-sphere cavity. "
        "The upper panel shows $\\beta W_\\mu(\\sigma)=\\beta\\mu_{\\rm ex}$; "
        "the lower panel shows $S(0)=[Z+\\eta Z']^{-1}$ and "
        "$1-S(0)=\\partial W_\\mu/\\partial\\mu$. Curves are analytic EOS "
        "predictions, not independent simulation measurements. The vertical "
        "line and uncertainty band give $\\eta_f=0.491882(46)$, derived from "
        "the extrapolated coexisting-fluid specific volume of Fernandez "
        "et al. The mKLM fluid EOS is continued above coexistence only to "
        "$\\eta=0.52$, within its documented fit range "
        "($\\rho\\sigma^3\\leq1.02$, or $\\eta\\leq0.5341$); that segment is "
        "a metastable-fluid continuation."
    )
    path.write_text(caption + "\n", encoding="utf-8")


def make_figure(path: Path, eta_grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"CS": "#0072B2", "PY-c": "#D55E00", "mKLM-2019": "#009E73"}
    linestyles = {"CS": "-", "PY-c": "--", "mKLM-2019": "-."}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )
    fig, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(3.55, 4.55),
        sharex=True,
        gridspec_kw={"height_ratios": (1.08, 1.0), "hspace": 0.08},
    )

    for axis in (upper, lower):
        axis.axvspan(
            ETA_FLUID - ETA_FLUID_STD,
            ETA_FLUID + ETA_FLUID_STD,
            color="0.72",
            alpha=0.5,
            linewidth=0,
            zorder=0,
        )
        axis.axvline(ETA_FLUID, color="0.35", linewidth=0.9, zorder=1)
        axis.grid(True, color="0.90", linewidth=0.55)
        axis.set_axisbelow(True)

    for model in MODELS:
        values = evaluate(model, eta_grid)
        upper.plot(
            eta_grid,
            values["beta_mu_ex"],
            color=colors[model.key],
            linestyle=linestyles[model.key],
            linewidth=1.45,
            label=model.label,
        )
        lower.plot(
            eta_grid,
            values["S0"],
            color=colors[model.key],
            linestyle="-",
            linewidth=1.35,
        )
        lower.plot(
            eta_grid,
            values["dW_dmu"],
            color=colors[model.key],
            linestyle="--",
            linewidth=1.35,
        )

    upper.set_ylabel(r"$\beta W_\mu(\sigma)=\beta\mu_{\mathrm{ex}}$")
    upper.set_ylim(0, None)
    upper.legend(loc="upper left", frameon=False, ncol=1)
    upper.text(
        ETA_FLUID - 0.004,
        0.97,
        r"$\eta_f=0.491882(46)$",
        transform=upper.get_xaxis_transform(),
        ha="right",
        va="top",
        color="0.25",
        fontsize=7.6,
    )

    lower.set_xlabel(r"packing fraction $\eta$")
    lower.set_ylabel("response")
    lower.set_xlim(0, ETA_MAX)
    lower.set_ylim(-0.025, 1.025)
    lower.set_xticks(np.arange(0, 0.51, 0.1))
    lower.set_yticks(np.arange(0, 1.01, 0.2))

    quantity_handles = [
        Line2D([0], [0], color="0.15", lw=1.35, ls="-", label=r"$S(0)$"),
        Line2D(
            [0], [0], color="0.15", lw=1.35, ls="--", label=r"$1-S(0)$"
        ),
    ]
    lower.legend(
        handles=quantity_handles,
        loc="center right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
    )

    for label, axis in zip(("a", "b"), (upper, lower)):
        axis.text(
            -0.20,
            1.01,
            f"({label})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.23, right=0.99, top=0.985, bottom=0.115)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def write_metadata(
    path: Path,
    points: int,
    symbolic: dict[str, object],
    numerical: dict[str, dict[str, float]],
    finite_difference: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {
        "artifact_type": "analytic EOS predictions; no independent simulation data",
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": script_hash,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sympy": sp.__version__,
        "matplotlib": matplotlib.__version__,
        "grid": {"eta_min": 0.0, "eta_max": ETA_MAX, "points": points},
        "random_check_seed": CHECK_SEED,
        "coexistence_marker": {
            "eta_fluid": ETA_FLUID,
            "eta_fluid_standard_uncertainty": ETA_FLUID_STD,
            "derived_from_v_fluid_sigma3": V_FLUID,
            "v_fluid_sigma3_standard_uncertainty": V_FLUID_STD,
            "source": "Fernandez et al., Phys. Rev. Lett. 108, 165701 (2012)",
            "doi": "10.1103/PhysRevLett.108.165701",
        },
        "models": [
            {"key": model.key, "name": model.label, "source": model.citation}
            for model in MODELS
        ],
        "checks": {
            "symbolic": symbolic,
            "dense_grid": numerical,
            "random_five_point_finite_difference_max_relative_errors": (
                finite_difference
            ),
            "finite_difference_threshold": 1.0e-8,
        },
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def generate(points: int = DEFAULT_POINTS, root: Path = ROOT) -> None:
    if points < 2000:
        raise ValueError("N3 requires at least 2000 eta points")
    eta_grid = np.linspace(0.0, ETA_MAX, points)
    symbolic = symbolic_consistency_checks()
    numerical = numerical_checks(eta_grid)
    finite_difference = random_finite_difference_checks()

    data_path = root / "data" / "eos_benchmark.csv"
    write_csv(data_path, eta_grid)
    # Manuscript N3 names this compatibility copy explicitly.
    write_csv(root / "generated" / "cs_benchmark.csv", eta_grid)
    write_latex_table(root / "generated" / "cs_table.tex")
    write_caption(root / "generated" / "cs_benchmark_caption.tex")
    make_figure(root / "figures" / "cs_benchmark.pdf", eta_grid)
    write_metadata(
        root / "generated" / "eos_benchmark_metadata.json",
        points,
        symbolic,
        numerical,
        finite_difference,
    )

    print(f"Wrote {data_path.relative_to(root)} ({len(MODELS) * points} rows)")
    print("Wrote generated/cs_benchmark.csv")
    print("Wrote generated/cs_table.tex")
    print("Wrote generated/cs_benchmark_caption.tex")
    print("Wrote figures/cs_benchmark.pdf")
    print("Wrote generated/eos_benchmark_metadata.json")
    print(f"eta_f = {ETA_FLUID:.9f} +/- {ETA_FLUID_STD:.9f}")
    for model in MODELS:
        at_coex = evaluate(model, 0.492)
        print(
            f"{model.key} at eta=0.492: "
            f"Z={float(at_coex['Z']):.9f}, "
            f"beta_mu_ex={float(at_coex['beta_mu_ex']):.9f}, "
            f"S0={float(at_coex['S0']):.9f}, "
            f"dW_dmu={float(at_coex['dW_dmu']):.9f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points",
        type=int,
        default=DEFAULT_POINTS,
        help="number of evenly spaced eta values (minimum: 2000)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run exact, dense-grid, and finite-difference checks without outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eta_grid = np.linspace(0.0, ETA_MAX, args.points)
    if args.check_only:
        if args.points < 2000:
            raise ValueError("N3 requires at least 2000 eta points")
        print(json.dumps(symbolic_consistency_checks(), indent=2))
        print(json.dumps(numerical_checks(eta_grid), indent=2))
        print(json.dumps(random_finite_difference_checks(), indent=2))
    else:
        generate(args.points)


if __name__ == "__main__":
    main()
