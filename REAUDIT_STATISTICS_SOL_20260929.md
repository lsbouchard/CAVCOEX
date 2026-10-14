# Statistical re-audit — 29 September 2026

The central finite-size conclusions survive this independent check. A new
finding concerns the reproducibility of the hard eigenvalue cutoff used in
the largest-size curve test, rather than the estimated variance of its
small residual.

## Largest-size covariance test

The original predictor and its published weighted-fit errors are held fixed.
The bootstrap is extended from 2000 to 4000 draws, with all 12 chains kept
as fixed strata and their 40 time blocks resampled independently. The first
2000 covariance matrices replay the published matrices to a maximum relative
error below `5×10⁻¹⁵`.

In the original total-response covariance, the fifth correlation eigenvalue
is `1.02224×10⁻⁸` of the largest. In the 4000-draw replay it is
`9.85523×10⁻⁹`. It therefore crosses the declared `10⁻⁸` retention cutoff
as the Monte Carlo estimate of the covariance improves.

The replay is partitioned into disjoint 2000-draw covariance-estimation and
calibration samples, using one sequential split and 12 shuffled splits in
both directions: 26 overlapping partitions of one bootstrap sample, **not
26 independent datasets**. At the `10⁻⁸` cutoff, 18 partitions retain four
total-response modes and give `p=0.985–0.992`; eight retain five and give
the smallest reportable probability, `1/2001`. This discontinuity means that
the originally displayed `p≈0.0005` is conditional on the stated covariance
estimate and cutoff. It should not be described as a cutoff-independent or
bootstrap-rank-stable rejection.

The small-mode discrepancy itself is reproducible. Fixing the fifth
eigenvector from the original covariance, the additional 2000 bootstrap
draws give `|Z|=15.65`. Their projected variance is `0.9789` times the original
estimate, and the estimated bootstrap bias is only `0.164` standard errors.
This is a retrospective fixed-direction diagnostic; the direction was not
specified before examining the original covariance.

As a further sensitivity, retaining modes to `10⁻⁹` gives six total-response
modes in every split and detects the discrepancy at the empirical tail
resolution. Reweighting retains four modes with `p=0.179–0.235`. Retaining
still smaller modes at `10⁻¹⁰` detects residuals in both curves, consistent
with the original sensitivity analysis. These are tests of exact equality
to a truncated finite-size predictor: the derived expansion itself permits
a higher-order remainder, and a rejection does not measure that remainder's
asymptotic power.

## Scalar fits and sampling design

The free quadratic scalar model and its nested analytic-intercept test also
use disjoint covariance-estimation and calibration halves of the archived
2000 scalar bootstrap vectors. Across 24 overlapping half-splits, all 15
modes remain retained. The global free-quadratic fit gives `p=0.623–0.666`;
the three exact-intercept restrictions give `p=0.054–0.085`. These controls
support the reported scalar conclusions and show that reuse of the same
bootstrap sample for covariance and tail calibration is not creating those
conclusions.

Inspection confirms that paired and raw estimators share resampling indices,
as do primitive-coefficient predictions and measured pulse slopes. Spin
inversion is an antithetic variance reduction using an exact symmetry; its
two configurations are not treated as independent data. The corrected
stratified bootstrap avoids the previously repaired double counting of
within-chain sampling variance.

Three additional tests check independent Gaussian quadratic-distance moments,
invariance under changing observable units, and rejection of overlapping
covariance/calibration sets. All three pass. This re-audit adds no raw Monte
Carlo trajectories and changes no previous scientific output.

## Reproduction and files

Run from the manuscript project with its scientific Python environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig python scripts/reaud_covariance_mode_stability.py
python -m pytest -q tests/test_reaud_covariance_mode_stability.py
```

The default command regenerates the bootstrap from archived trajectories;
the 7.1 MB bootstrap cache is **not required as an input**. If the cache is
already present, `--cache-only` repeats the split diagnostics after checking
its digest and the recorded scientific-input digests. The calculation uses
one BLAS thread when invoked as above and does not rerun the long sampling
campaigns.

The outputs are:

- `data_processed/reaud_covariance_split_tests.csv`: correlation-mode ranks,
  quadratic statistics and tail calibration on disjoint samples.
- `data_processed/reaud_fixed_mode_controls.csv`: independent additional-draw
  checks of the original retained lowest modes.
- `data_processed/reaud_scalar_split_tests.csv`: independent covariance and
  calibration controls for the free scalar and exact-intercept tests.
- `data_processed/reaud_covariance_bootstrap_curves.npz`: optional saved
  bootstrap for inexpensive replay.
- `generated/reaud_covariance_stability_metadata.json`: scientific-input,
  raw-data, script and output digests.

The continuum extension needs recurrent fluid–solid round trips and accepted
particle-number/occupancy transitions within independent runs, followed by
uncertainty estimates based on blocks longer than the phase-odds correlation
time. Marginal order-parameter fluctuations or staging acceptances alone do
not establish an equilibrated common phase normalization.
