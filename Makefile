PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
export MPLCONFIGDIR := $(CURDIR)/.mplconfig

.PHONY: analytic lattice-analysis test reproduce

analytic:
	$(PYTHON) scripts/tonks_validation.py
	$(PYTHON) scripts/grand_reweight.py
	$(PYTHON) scripts/phase_conditioned.py
	$(PYTHON) scripts/hard_sphere_sensitivity.py
	$(PYTHON) scripts/hard_sphere_defect_sensitivity.py
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

test:
	$(PYTEST) -q

reproduce: analytic lattice-analysis test
