PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
OPENBLAS_NUM_THREADS ?= 1
OMP_NUM_THREADS ?= 1
MKL_NUM_THREADS ?= 1
VECLIB_MAXIMUM_THREADS ?= 1
NUMBA_NUM_THREADS ?= 1
NUMEXPR_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMBA_NUM_THREADS NUMEXPR_NUM_THREADS
export MPLCONFIGDIR := $(CURDIR)/.mplconfig

.DEFAULT_GOAL := reproduce
.NOTPARALLEL:
.PHONY: analytic lattice-analysis numerical-macros provenance figures test reproduce release-manifest check-release

analytic:
	$(PYTHON) scripts/tonks_validation.py
	$(PYTHON) scripts/grand_reweight.py
	$(PYTHON) scripts/phase_conditioned.py
	$(PYTHON) scripts/hard_sphere_sensitivity.py
	$(PYTHON) scripts/hard_sphere_defect_sensitivity.py
	$(PYTHON) scripts/tracer_radius_response_control.py
	$(PYTHON) scripts/cs_benchmark.py
	$(PYTHON) scripts/derive_pulse_finite_size_corrections.py
	$(PYTHON) scripts/lattice_exact_controls.py

lattice-analysis:
	$(PYTHON) scripts/lattice_validation_statistics.py
	$(PYTHON) scripts/lattice_free_intercept_joint_quadratic.py
	$(PYTHON) scripts/lattice_pulse_scaling.py
	$(PYTHON) scripts/lattice_pulse_area_sum_rule.py
	$(PYTHON) scripts/lattice_soft_cavity_family.py
	$(PYTHON) scripts/lattice_pulse_correction_prediction.py
	$(PYTHON) scripts/lattice_leave_lmax_prediction.py
	$(PYTHON) scripts/lattice_replica_robustness.py
	$(PYTHON) scripts/lattice_reweighting_overlap.py
	$(PYTHON) scripts/lattice_basin_robustness.py
	$(PYTHON) scripts/lattice_resampling_audit.py

test:
	$(PYTEST) -q

figures:
	$(PYTHON) scripts/render_publication_figures.py

numerical-macros: lattice-analysis
	$(PYTHON) scripts/make_numerical_macros.py

provenance: numerical-macros
	$(PYTHON) scripts/enrich_metadata_provenance.py

release-manifest:
	$(PYTHON) scripts/make_release_manifest.py --profile code

check-release:
	$(PYTHON) scripts/make_release_manifest.py --profile code --check

reproduce: analytic lattice-analysis numerical-macros provenance figures test release-manifest
