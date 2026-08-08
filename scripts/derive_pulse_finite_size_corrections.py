#!/usr/bin/env python3
"""Derive the leading finite-volume correction to the coexistence pulse.

The starting point is

    R_V(h) = (D_V/V) [L(y_V-a_V) - L(y_V)],

with the finite-volume expansions specified in the manuscript review.  This
module evaluates the exact expression defined by those truncated inputs,
provides its analytic expansion through O(1/V), and derives peak and FWHM
corrections by the implicit-function theorem.  Running the script writes the
machine-readable JSON and LaTeX formula fragments used to document the
calculation; ``--check`` verifies that committed fragments are current.

No computer-algebra dependency is required.  The formulas are implemented in
terms of the logistic kernel and its derivatives, which also avoids fragile
expanded exponentials for numerical evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "generated"


@dataclass(frozen=True)
class PulseParameters:
    """Coefficients in the prescribed finite-volume expansions.

    ``delta_rho`` is the leading density gap.  The remaining names correspond
    to

      D_V = V delta_rho + d0 + dchi h,
      y_V = delta_rho h + (d0 h + dchi h^2/2)/V,
      a_V = a0 + (a1 - dn h)/V.
    """

    delta_rho: float
    d0: float
    dchi: float
    a0: float
    a1: float
    dn: float

    def validate_curve(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("all pulse parameters must be finite")
        if self.delta_rho == 0.0:
            raise ValueError("delta_rho must be nonzero")

    def validate_observables(self) -> None:
        self.validate_curve()
        if self.a0 == 0.0:
            raise ValueError(
                "a0 must be nonzero: at a0=0 there is no leading pulse, "
                "so peak and FWHM corrections are singular"
            )


def logistic(x: float) -> float:
    """Numerically stable logistic function."""

    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def logistic_prime(x: float) -> float:
    value = logistic(x)
    return value * (1.0 - value)


def logistic_second(x: float) -> float:
    value = logistic(x)
    prime = value * (1.0 - value)
    return prime * (1.0 - 2.0 * value)


def kernel(z: float, a: float) -> float:
    """G(z,a) = L(z-a) - L(z)."""

    return logistic(z - a) - logistic(z)


def kernel_z(z: float, a: float) -> float:
    return logistic_prime(z - a) - logistic_prime(z)


def kernel_a(z: float, a: float) -> float:
    return -logistic_prime(z - a)


def kernel_zz(z: float, a: float) -> float:
    return logistic_second(z - a) - logistic_second(z)


def kernel_za(z: float, a: float) -> float:
    return -logistic_second(z - a)


def finite_volume_response(
    h: float, volume: float, parameters: PulseParameters
) -> float:
    """Evaluate the finite-V formula with the prescribed truncated inputs."""

    parameters.validate_curve()
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("volume must be finite and positive")
    d = parameters.delta_rho
    prefactor = d + (parameters.d0 + parameters.dchi * h) / volume
    y_value = d * h + (
        parameters.d0 * h + 0.5 * parameters.dchi * h * h
    ) / volume
    a_value = parameters.a0 + (parameters.a1 - parameters.dn * h) / volume
    return prefactor * kernel(y_value, a_value)


def leading_response(h: float, parameters: PulseParameters) -> float:
    parameters.validate_curve()
    return parameters.delta_rho * kernel(
        parameters.delta_rho * h, parameters.a0
    )


def first_order_curve_correction(
    h: float, parameters: PulseParameters
) -> float:
    """Return R_1(h) in R_V(h)=R_0(h)+R_1(h)/V+O(V^-2)."""

    parameters.validate_curve()
    d = parameters.delta_rho
    z = d * h
    u = parameters.d0 + parameters.dchi * h
    v = parameters.d0 * h + 0.5 * parameters.dchi * h * h
    w = parameters.a1 - parameters.dn * h
    return u * kernel(z, parameters.a0) + d * (
        v * kernel_z(z, parameters.a0) + w * kernel_a(z, parameters.a0)
    )


def first_order_response(
    h: float, volume: float, parameters: PulseParameters
) -> float:
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("volume must be finite and positive")
    return leading_response(h, parameters) + first_order_curve_correction(
        h, parameters
    ) / volume


def peak_corrections(parameters: PulseParameters) -> dict[str, float]:
    """Leading and 1/V coefficients of center and signed/magnitude peak."""

    parameters.validate_observables()
    d = parameters.delta_rho
    h0 = parameters.a0 / (2.0 * d)
    t = math.tanh(parameters.a0 / 4.0)
    p = logistic_prime(parameters.a0 / 2.0)
    if p == 0.0:
        raise ValueError(
            "the leading peak curvature underflows to zero; the implicit "
            "correction is numerically ill-conditioned"
        )
    u0 = parameters.d0 + parameters.dchi * h0
    v0 = parameters.d0 * h0 + 0.5 * parameters.dchi * h0 * h0
    w0 = parameters.a1 - parameters.dn * h0

    # This compact expression follows from -R_1'(h0)/R_0''(h0).
    center1 = (
        parameters.dchi / (2.0 * d**3 * p)
        - parameters.dn / (2.0 * d**2 * t)
        + (w0 - 2.0 * v0) / (2.0 * d)
    )
    signed0 = -d * t
    signed1 = -u0 * t - d * p * w0
    signed_peak_sign = 1.0 if signed0 > 0.0 else -1.0
    return {
        "center_h_0": h0,
        "center_h_1": center1,
        "signed_peak_0": signed0,
        "signed_peak_1": signed1,
        "magnitude_peak_0": abs(signed0),
        "magnitude_peak_1": signed_peak_sign * signed1,
        "t": t,
        "p": p,
        "u_at_peak": u0,
        "v_at_peak": v0,
        "w_at_peak": w0,
    }


def fwhm_corrections(parameters: PulseParameters) -> dict[str, float]:
    """Leading and 1/V FWHM coefficients from the two half-height roots.

    Branch ``minus``/``plus`` labels z-a0/2 = -q/+q, not necessarily the
    physical left/right root.  Their physical ordering reverses when
    ``delta_rho`` is negative; the explicit sign factor in ``fwhm_h_1`` makes
    the reported width positive at leading order in either convention.
    """

    parameters.validate_observables()
    d = parameters.delta_rho
    peak = peak_corrections(parameters)
    absolute_x = abs(parameters.a0) / 2.0
    log_two = math.log(2.0)
    log_cosh = (
        absolute_x + math.log1p(math.exp(-2.0 * absolute_x)) - log_two
    )
    larger = max(log_two, log_cosh)
    log_argument = larger + math.log1p(
        math.exp(min(log_two, log_cosh) - larger)
    )
    q = log_argument + math.log1p(
        math.sqrt(1.0 - math.exp(-2.0 * log_argument))
    )
    roots: dict[str, float] = {}
    for label, branch_sign in (("minus", -1.0), ("plus", 1.0)):
        h0 = (parameters.a0 / 2.0 + branch_sign * q) / d
        derivative0 = d * d * kernel_z(d * h0, parameters.a0)
        if derivative0 == 0.0:
            raise ArithmeticError("degenerate half-height root")
        correction_at_root = first_order_curve_correction(h0, parameters)
        h1 = -(
            correction_at_root - 0.5 * peak["signed_peak_1"]
        ) / derivative0
        roots[f"root_{label}_h_0"] = h0
        roots[f"root_{label}_h_1"] = h1

    orientation = 1.0 if d > 0.0 else -1.0
    width0 = 2.0 * q / abs(d)
    width1 = orientation * (
        roots["root_plus_h_1"] - roots["root_minus_h_1"]
    )
    return {
        "q": q,
        **roots,
        "fwhm_h_0": width0,
        "fwhm_h_1": width1,
    }


def all_observable_corrections(
    parameters: PulseParameters,
) -> dict[str, dict[str, float]]:
    return {
        "parameters": asdict(parameters),
        "peak": peak_corrections(parameters),
        "fwhm": fwhm_corrections(parameters),
    }


def formula_payload() -> dict[str, Any]:
    """Return a deterministic machine-readable statement of the derivation."""

    return {
        "schema_version": 1,
        "generated_by": "scripts/derive_pulse_finite_size_corrections.py",
        "expansion_parameter": "epsilon = 1/V",
        "input": {
            "response": "R_V(h)=(D_V/V)*G(y_V,a_V)",
            "kernel": "G(z,a)=L(z-a)-L(z), L(x)=1/(1+exp(-x))",
            "D_V": "V*delta_rho+d0+dchi*h",
            "y_V": "delta_rho*h+(d0*h+dchi*h^2/2)/V",
            "a_V": "a0+(a1-dn*h)/V",
        },
        "definitions": {
            "z": "delta_rho*h",
            "u(h)": "d0+dchi*h",
            "v(h)": "d0*h+dchi*h^2/2",
            "w(h)": "a1-dn*h",
            "G_z": "L'(z-a)-L'(z)",
            "G_a": "-L'(z-a)",
        },
        "curve": {
            "R_0(h)": "delta_rho*G(delta_rho*h,a0)",
            "R_1(h)": (
                "u(h)*G(z,a0)+delta_rho*[v(h)*G_z(z,a0)"
                "+w(h)*G_a(z,a0)]"
            ),
            "expansion": (
                "R_V(h)=R_0(h)+R_1(h)/V+O(V^-2), uniformly on compact h sets"
            ),
        },
        "peak": {
            "t": "tanh(a0/4)",
            "p": "1/[4*cosh(a0/4)^2]",
            "h_0": "a0/(2*delta_rho)",
            "u_0": "u(h_0)",
            "v_0": "v(h_0)",
            "w_0": "w(h_0)",
            "h_1_implicit": "-R_1'(h_0)/R_0''(h_0)",
            "h_1": (
                "dchi/(2*delta_rho^3*p)-dn/(2*delta_rho^2*t)"
                "+(w_0-2*v_0)/(2*delta_rho)"
            ),
            "signed_0": "-delta_rho*t",
            "signed_1": "-u_0*t-delta_rho*p*w_0",
            "magnitude_0": "abs(signed_0)",
            "magnitude_1": "sign(signed_0)*signed_1",
        },
        "fwhm": {
            "q": "acosh(2+cosh(a0/2))",
            "branch_h_s_0": "(a0/2+s*q)/delta_rho, s in {-1,+1}",
            "branch_h_s_1": (
                "-[R_1(branch_h_s_0)-signed_1/2]"
                "/R_0'(branch_h_s_0)"
            ),
            "R_0_prime": "delta_rho^2*G_z(delta_rho*h,a0)",
            "width_0": "2*q/abs(delta_rho)",
            "width_1": (
                "sign(delta_rho)*(branch_h_plus_1-branch_h_minus_1)"
            ),
        },
        "conventions": [
            "The signed leading peak is -delta_rho*tanh(a0/4).",
            "The magnitude correction equals sign(signed_peak_0) times the signed correction.",
            "The sign in w(h)=a1-dn*h follows the stated a_V convention.",
            "FWHM is defined from |R_V|; signed half-height roots are equivalent while the pulse sign is fixed.",
            "The +/- half-height labels refer to the scaled z coordinate; their physical h ordering reverses for delta_rho<0.",
            "Peak and FWHM expansions require delta_rho!=0 and a0!=0.",
            "The curve expansion is local in the scaled field h and uniform on fixed compact h intervals.",
        ],
    }


def render_json() -> str:
    return json.dumps(formula_payload(), indent=2, sort_keys=True) + "\n"


def render_tex() -> str:
    """Render the derivation as a standalone LaTeX input fragment."""

    return r"""% Generated by scripts/derive_pulse_finite_size_corrections.py.
% Do not edit by hand; regenerate with the script.
\providecommand{\PulseKernel}{\mathcal G}
\begin{align}
\PulseKernel(z,a)&=L(z-a)-L(z),
&L(x)&=(1+e^{-x})^{-1},\\
z&=\Delta\rho\,h,
&u(h)&=d_0+d_\chi h,\\
v(h)&=d_0h+\tfrac12d_\chi h^2,
&w(h)&=a_1-d_n\,h.
\end{align}
For
\begin{align}
D_V&=V\Delta\rho+d_0+d_\chi h,\\
y_V&=\Delta\rho\,h+\frac{d_0h+d_\chi h^2/2}{V},\\
a_V&=a_0+\frac{a_1-d_n\,h}{V},
\end{align}
the full scaled-field curve has, uniformly for $h$ in any fixed compact
interval, the expansion
\begin{align}
R_V(h)&=R_0(h)+\frac{R_1(h)}{V}+\Order(V^{-2}),\\
R_0(h)&=\Delta\rho\,\PulseKernel(z,a_0),\\
R_1(h)&=u(h)\PulseKernel(z,a_0)\nonumber\\
&\quad+\Delta\rho\left[v(h)\PulseKernel_z(z,a_0)
+w(h)\PulseKernel_a(z,a_0)\right],
\label{eq:pulse-correction-full-curve}
\end{align}
where
\begin{equation}
\PulseKernel_z=L'(z-a)-L'(z),\qquad
\PulseKernel_a=-L'(z-a).
\end{equation}

For the nondegenerate pulse $\Delta\rho\ne0$ and $a_0\ne0$, define
\begin{equation}
t=\tanh(a_0/4),\qquad
p=\frac{1}{4\cosh^2(a_0/4)},\qquad
h_0=\frac{a_0}{2\Delta\rho},
\end{equation}
and $u_0=u(h_0)$, $v_0=v(h_0)$, and $w_0=w(h_0)$.  The peak
location is
\begin{align}
h_{\rm peak,V}&=h_0+\frac{h_1}{V}+\Order(V^{-2}),\\
h_1&=-\frac{R_1'(h_0)}{R_0''(h_0)}
=\frac{d_\chi}{2(\Delta\rho)^3p}
-\frac{d_n}{2(\Delta\rho)^2t}
+\frac{w_0-2v_0}{2\Delta\rho}.
\label{eq:pulse-correction-center}
\end{align}
The signed peak $S_V$ and its magnitude $M_V=|S_V|$ obey
\begin{align}
S_V&=S_0+\frac{S_1}{V}+\Order(V^{-2}),
&S_0&=-\Delta\rho\,t,\\
S_1&=R_1(h_0)=-u_0t-\Delta\rho\,p w_0,\\
M_V&=|S_0|+\frac{\operatorname{sgn}(S_0)S_1}{V}
+\Order(V^{-2}).
\label{eq:pulse-correction-peak}
\end{align}

For the FWHM, let
\begin{align}
q&=\operatorname{arcosh}\!\left[2+\cosh(a_0/2)\right],\\
h_{s,0}&=\frac{a_0/2+s q}{\Delta\rho},\qquad s=\pm1.
\end{align}
Implicit differentiation of
$R_V(h_{s,V})=S_V/2$ gives
\begin{align}
h_{s,V}&=h_{s,0}+\frac{h_{s,1}}{V}+\Order(V^{-2}),\\
h_{s,1}&=-\frac{R_1(h_{s,0})-S_1/2}
{R_0'(h_{s,0})},\\
R_0'(h)&=(\Delta\rho)^2
\PulseKernel_z(\Delta\rho h,a_0).
\end{align}
Consequently,
\begin{align}
\Delta h_{\rm FWHM,V}
&=\Delta h_0+\frac{\Delta h_1}{V}+\Order(V^{-2}),\\
\Delta h_0&=\frac{2q}{|\Delta\rho|},\\
\Delta h_1&=\operatorname{sgn}(\Delta\rho)
\left(h_{+,1}-h_{-,1}\right).
\label{eq:pulse-correction-fwhm}
\end{align}
Here $s=\pm1$ labels the two branches in the scaled coordinate
$z-a_0/2$; their left--right ordering in $h$ reverses when
$\Delta\rho<0$.  The signed leading peak is
$-\Delta\rho\tanh(a_0/4)$, so converting $S_1$ to a magnitude correction
requires the explicit factor $\operatorname{sgn}(S_0)$.  The sign of the
$d_n$ terms above follows the convention
$a_V=a_0+(a_1-d_n\,h)/V$.
"""


def write_or_check(output_dir: Path, check: bool) -> None:
    artifacts = {
        output_dir / "pulse_correction_formulas.tex": render_tex(),
        output_dir / "pulse_correction_formulas.json": render_json(),
    }
    stale: list[Path] = []
    for path, content in artifacts.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if stale:
        joined = ", ".join(str(path) for path in stale)
        raise SystemExit(f"generated pulse-correction artifacts are stale: {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the generated TeX and JSON fragments",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed outputs instead of rewriting them",
    )
    arguments = parser.parse_args()
    write_or_check(arguments.output_dir, arguments.check)


if __name__ == "__main__":
    main()
