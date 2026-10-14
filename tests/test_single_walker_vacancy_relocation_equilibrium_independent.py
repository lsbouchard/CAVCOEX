"""Enabled single-walker relocation/Gibbs composition vs independent 9D laws.

The intentionally capped N<=3 target is not an unrestricted fluid or a
coexistence calculation. All five expected conditional and joint laws retain
the original independent partition quadrature with exactly one 1/N! factor.
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import numpy as np

from scripts import continuum_simulated_tempering as tempering
import test_single_walker_continuum_equilibrium as reference
from test_single_walker_gibbs_continuum_equilibrium import (
    _enable_gibbs_with_independent_protocol_checks,
)


def test_enabled_relocation_and_gibbs_preserve_capped_joint_conditional_and_support_laws(monkeypatch):
    # A diagnostic test wrapper must not emit an older producer's metadata for
    # this distinct enabled protocol, even if an export variable exists outside.
    monkeypatch.delenv('CAVCOEX_ST_CONTROL_EXPORT', raising=False)
    protocol_chunks = _enable_gibbs_with_independent_protocol_checks(monkeypatch)
    actual = tempering.run_tempering_chunk
    relocation_chunks = []

    def enabled(*args, **kwargs):
        steps = int(args[6])
        penalty, kappa = args[12], args[13]
        counters = np.zeros(2, dtype=np.int64)
        kwargs['vacancy_relocation_probability'] = 1.
        kwargs['vacancy_relocation_counts'] = counters
        result = actual(*args, **kwargs)
        rows = result[0]
        labels = rows[:, 1].astype(int)
        eligible = ((penalty[labels] >= 8) | (kappa[labels] > 0)) & (rows[:, 2] > 0)
        # Incoming-label observations follow the updated local host, before
        # Gibbs/Palm. Therefore these current N features independently count
        # every attempted p=1 relocation at the wrapper's correct stage.
        assert counters[0] == np.count_nonzero(eligible)
        assert steps / 100 < counters[1] < counters[0]
        assert np.all(rows[(labels == 0) | (labels == 1), 4] == 0)
        assert np.all(rows[(labels == 1) & (rows[:, 2] > 0), 7] >= 1.)
        assert counters[0] > steps / 4
        relocation_chunks.append((steps, counters.copy()))
        return result

    monkeypatch.setattr(tempering, 'run_tempering_chunk', enabled)
    # Unchanged reference checks the complete joint label-number law, all
    # conditional number/order laws, cavity/ideal void probabilities, capped
    # Palm boundaries, full-clock regeneration, and all sparse coordinate /
    # center / support / Fourier / singleton-deletion reconstructions. The
    # independent Gibbs wrapper also replays post-observation labels/counters
    # and validates that the neighbor step uses the current host features.
    reference.test_actual_single_host_matches_capped_joint_and_conditional_partition_laws()
    assert [steps for steps, _ in protocol_chunks] == [5000, 150000]
    assert [steps for steps, _ in relocation_chunks] == [5000, 150000]
