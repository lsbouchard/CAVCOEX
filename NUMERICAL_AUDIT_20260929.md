# Numerical audit, 29 September 2026

This report describes the audited local project and code checkout. These
revisions have not been pushed to GitHub; the previously published repository
and historical submission archives remain unchanged.

This audit reviewed the saved-data analysis and numerical kernels. No production
Monte Carlo trajectories were replaced or extended. Computations used one process
with BLAS, OpenMP, and Numba thread limits set to one. The principal correction is
to the lattice uncertainty estimator, not the sampled statistical-mechanical law.

## Findings and corrections

### 1. Nested resampling counted within-chain Monte Carlo variation twice

The previous lattice bootstrap first sampled independent chains with replacement,
then sampled time blocks within each selected chain. The chains are repeated
simulations of one equilibrium law, not randomly selected physical clusters.
Resampling both levels therefore counts their finite-chain sampling noise twice.
For independent, identically distributed equal-length blocks, the expected old
variance divided by the true variance of the grand mean is
`2 - 1/R - 1/B`; with 12 chains and 40 blocks it is 1.89167. A 20,000-experiment
normal control gave 1.89098. Retaining each chain once and resampling its blocks
gave 0.97427, consistent with the ordinary finite-block bootstrap expectation
`1 - 1/B = 0.975`.

All active lattice resampling now retains chains as fixed strata and resamples
time blocks independently within each chain. Paired observables still use the
same block indices, preserving their covariance. Independent lattice sizes use
independent random streams. Regression tests independently check the exact
conditional resampling variance, constant strata, and shared-index covariance.

This correction propagated through the pulse, area, soft-cavity, fixed/free
intercept, cross-validation, coefficient-prediction, and size-prediction analyses.
The analysis preserves observed central curves; weighted extrapolations can shift
because their statistical weights have changed. The code continues to preserve
legacy output column/file names where renaming would break downstream users.

### 2. The chosen time-block scale was checked rather than assumed

`lattice_resampling_audit.py` tests 500-, 1000-, and 2000-update blocks using
800 draws for seven observables at each of five lattice sizes. Relative to
500-update blocks, coarsened standard errors range from 0.869 to 1.091.
Between-chain standard errors divided by baseline block-bootstrap errors range
from 0.683 to 1.283. There is no systematic increase on coarsening. This supports
retaining the original 500-update blocks; it does not prove absence of every
possible long-time mode. The saved chain diagnostics also report short measured
autocorrelation times (maximum approximately 1.18 updates).

The independent solid-replica analysis does **not** implement the erroneous
nested bootstrap: it takes the larger of within-replica and between-replica
standard errors. That conservative uncertainty rule was retained.

### 3. Scientific inferences sharpened by the corrected uncertainty

- For the paired fixed-exact-limit linear fits, centered-bootstrap probabilities
  are approximately 0.68 (center), 0.97 (magnitude), and 0.004 (width). The width
  therefore resolves curvature omitted from the linear finite-size fit.
- The joint fixed-limit linear model has `Q=301.30` and a bootstrap probability at
  the 2000-draw resolution, `1/2001`. The quadratic model has `Q=11.34`, 9 residual
  degrees of freedom, and probability 0.25. Thus the improved precision supports
  the need for the next finite-size term, not a change of thermodynamic limit.
- The free joint quadratic model has `Q=4.28`, 6 residual degrees of freedom, and
  probability 0.64 at the primary covariance cutoff. The three exact-intercept
  restriction gives probability 0.069. Covariance-cutoff sensitivity remains
  reported rather than presenting this as a threshold-invariant result.
- Free linear intercept deviations from their exact limits are 1.09, 0.47, and
  2.85 standard errors (center, magnitude, width). The largest grid-sensitivity
  shift is 0.00177 standard errors. Fit-variant intercept shifts are respectively
  `3.66e-4`, `3.07e-5`, and `1.05e-3`.
- The signed-area fixed-limit linear and quadratic probabilities are 0.52 and
  0.95. The exact truncated-window area remains compatible with extrapolation.
- Across disjoint-replica folds, the exact finite-volume prediction has maximum
  pointwise standardized residual 1.51 and minimum covariance probability 0.062;
  all simultaneous bands contain the evaluation curves. The first-corrected
  asymptotic curve has maximum pointwise residual 1.56, yet some covariance tests
  reach the 1000-draw resolution. Pointwise agreement is not equivalent to joint
  statistical agreement for these strongly correlated curves.
- The three predicted finite-size slopes differ marginally from direct fits by
  -0.23, -0.29, and -0.92 standard errors. Their correlated joint statistic is
  `Q=24.15`, with probability `1/2001`. Omitting the smallest size or adding the
  next term in the primitive extrapolation gives probabilities 0.18 and 0.20.
- Predicting `L=64` using only `L<=48` still gives maximum pointwise residuals
  0.27 (total) and 0.35 (phase reweighting), with simultaneous probabilities 0.89
  and 0.82. At the primary relative correlation-eigenvalue cutoff `1e-8`, total
  and phase-reweighting covariance probabilities are respectively `1/2001` and
  0.23. The total response resolves a small correlated remainder, even though the
  prediction is accurate pointwise. Its fifth retained eigenvalue is only
  `1.022e-8` times the largest, just above the primary cutoff, and this mode
  contributes approximately `15.49^2` to the statistic. At the coarser `1e-6`
  cutoff the total probability is 0.99; the rejection is consequently
  regularization-dependent, not robust across all declared cutoffs.
  Across all declared fit variants, maximum
  pointwise residual is 0.45 and all simultaneous bands contain the data.
- At `L=64`, paired versus raw phase-reweighting estimates differ by 2.45 raw
  standard errors, not fewer than two. This tracks the raw zero-field phase
  fraction `0.500754 +/- 0.000309`; pairing enforces the exact one-half symmetry.
  The largest delete-one-replica shift remains 0.67 replica-level standard errors,
  with delete-one error ratios between 0.814 and 1.095.

These are diagnostic probabilities from finite bootstrap ensembles, not exact
probabilities with arbitrarily many significant digits. Minimum values represent
the stated Monte Carlo resolution and are never reported as zero.

### 4. The main prediction figure now exposes the correlated residual

The two lower panels show standardized residual projections onto retained
correlation eigenvectors, in descending eigenvalue order. The sign convention
makes the largest-magnitude eigenvector component positive. Their squared values
sum to the reported covariance statistic; a saved-data regression checks this.
The dashed `+/-2` lines are guides, not a simultaneous confidence region. The upper
prediction/evaluation curves and simultaneous bands are unchanged.

A `--plot-only` option redraws the figure from saved curves and covariance arrays.
Its source, inputs, output, and timestamp are recorded separately under
`figure_render`; it does not overwrite the earlier numerical-execution record.
The preceding source is retained in the source snapshots.

### 5. Reweighting-support calculation had an avoidable multi-GB allocation

The previous implementation retained all four weight arrays for 301 fields and
240,000 configurations. Those arrays alone require approximately 2.31 GB before
concatenation and temporaries. The replacement processes one field at a time:
the analogous four arrays require about 7.68 MB, plus a 1.92 MB concatenation and
small summaries. These are array-size estimates, not measured peak resident
memory. They do not establish the cause of any earlier macOS panic.

A dense-versus-fieldwise fixture regression agrees to `2e-14` relative/absolute
tolerance. The regenerated full 12,040-row support table differs from its prior
version by at most `1.31e-12` in numeric columns, consistent with changed
floating-point summation order; scientific conclusions are unchanged.

### 6. A zero-variance stationarity diagnostic missed a deterministic shift

If both halves of a hard-sphere block sequence were internally constant but had
different means, the half-difference standard error was zero and the old code
returned a standardized shift of zero. It now returns infinity, correctly failing
the stationarity gate. All 140 stored run/radius combinations were checked; none
was affected. A synthetic regression covers this edge case.

### 7. Regression tests must test calculations, not demand favorable results

Several tests previously required a desirable scientific result, such as a
probability above a chosen rejection threshold or a small prediction-error ratio.
These were replaced with checks of algebra, covariance, standardization,
probability calculation, valid ranges, and output completeness. Actual residuals,
probabilities, and uncertainty-sensitive conclusions remain in the numerical
outputs and are described above. This distinction prevents a valid scientific
disagreement from being mistaken for a software failure.

## Independent checks and scope coverage

The current Wolff kernel exactly replays a saved 64-sample trajectory prefix,
including magnetization, bond sum, reference spins, and cluster sizes. A separate
4-by-4 equilibrium test compares a 50,000-update Wolff run with complete
enumeration using independent 500-update block errors. Existing exact finite-state,
probability-identity, pulse-derivative, and ensemble controls were retained.

The final full test run passed **135 tests in 19.87 seconds**, with the thread
limits above. No tests were skipped in that run.

Source-level review and the test suite cover:

- Analytic/EOS controls: `tonks_validation.py`, `cs_benchmark.py`,
  `grand_reweight.py`, `hard_sphere_models.py`, `hard_sphere_sensitivity.py`,
  `hard_sphere_defect_sensitivity.py`, and `phase_conditioned.py`.
- Sampling: `cavity_mc.py`, `solid_cavity_mc.py`,
  `solid_cavity_replica_campaign.py`, `lattice_gas_phase_switch.py`, and
  `lattice_gas_additional_replicas.py`.
- Exact and estimator controls: `lattice_exact_controls.py`,
  `lattice_basin_robustness.py`, `lattice_analysis_common.py`,
  `lattice_replica_robustness.py`, and `lattice_reweighting_overlap.py`.
- Scaling and prediction: `derive_pulse_finite_size_corrections.py`,
  `lattice_pulse_scaling.py`, `lattice_pulse_area_sum_rule.py`,
  `lattice_soft_cavity_family.py`, `lattice_validation_statistics.py`,
  `lattice_free_intercept_joint_quadratic.py`,
  `lattice_pulse_correction_prediction.py`, and `lattice_leave_lmax_prediction.py`.
- Derived reporting: `make_numerical_macros.py` and the new
  `lattice_resampling_audit.py`.

Release scripts, environment locking, archival source provenance, custom output
paths, and allocation guards were handled by the coordinated release audit.
The independent theory audit handles hard-sphere literature/source covariance
sensitivity. No new thermodynamic phase-switching hard-sphere simulation is
claimed; the crystalline reference remains phase restricted. Testing and finite
trajectory inspection cannot guarantee the absence of every numerical error.

## Reproduction of the corrected analyses

With the project environment, set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, and `MPLCONFIGDIR=.mplconfig`. Run
the following scripts sequentially with Python:

1. `lattice_validation_statistics.py`
2. `lattice_free_intercept_joint_quadratic.py`
3. `lattice_pulse_scaling.py`
4. `lattice_pulse_area_sum_rule.py`
5. `lattice_soft_cavity_family.py`
6. `lattice_pulse_correction_prediction.py`
7. `lattice_leave_lmax_prediction.py`
8. `lattice_replica_robustness.py`
9. `lattice_resampling_audit.py`
10. `lattice_reweighting_overlap.py`
11. `make_numerical_macros.py`

All reside under `scripts/`. This sequence was executed from the saved raw
archives during this audit. The final eigenmode figure was then rendered with
`python scripts/lattice_leave_lmax_prediction.py --plot-only`. A fresh full run
now renders that figure directly. Numerical timestamps and hashes are retained
in the corresponding metadata; figure-only and release-only provenance are kept
distinct from execution provenance.
