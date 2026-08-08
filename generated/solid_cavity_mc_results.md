# Solid cavity MC: manuscript-ready numerical summary

The runs use eta_s = 0.54312409 +/- 0.00001690, derived from the published thermodynamic-limit solid specific volume 0.96405 +/- 0.00003 sigma^3 (DOI 10.1103/PhysRevLett.108.165701).

## N = 108

- R/sigma = 0.50: P0 = 0.45689069 +/- 0.00011963, beta W = 0.783311 +/- 0.000262; exact-control z = +0.12.
- R/sigma = 0.75: P0 = 0.0006428833 +/- 1.1e-05, beta W = 7.349547 +/- 0.016870; reliable_direct.
- R/sigma = 1.00 audit: 0 empty events in 16384000 correlated spatial probes; zero_events;too_few_positive_blocks;relative_error_too_large. Only this raw exposure is reported, with no independent-trial confidence bound and no finite work estimate.
- Crystal gates over every production block: min Q6 = 0.3697, min FCC Bragg = 0.8613, max Lindemann ratio = 0.1295.

## N = 256

- R/sigma = 0.50: P0 = 0.45693829 +/- 0.00012435, beta W = 0.783207 +/- 0.000272; exact-control z = +0.50.
- R/sigma = 0.75: P0 = 0.00065869141 +/- 6.7e-06, beta W = 7.325255 +/- 0.010217; reliable_direct.
- R/sigma = 1.00 audit: 0 empty events in 16384000 correlated spatial probes; zero_events;too_few_positive_blocks;relative_error_too_large. Only this raw exposure is reported, with no independent-trial confidence bound and no finite work estimate.
- Crystal gates over every production block: min Q6 = 0.3825, min FCC Bragg = 0.8449, max Lindemann ratio = 0.1350.

## Scope limitation

These are fixed-density, canonical, phase-conditioned FCC measurements. They provide no fluid/solid relative weight, no cross-basin normalization, and no phase-switch data.
