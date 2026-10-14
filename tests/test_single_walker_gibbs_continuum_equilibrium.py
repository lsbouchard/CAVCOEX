"""Actual fixed-clock Gibbs/neighbor composition against independent laws.

The original independent reference calculations are reused unchanged; only the
optional engine update is enabled. The cap-three cases compare with their own
truncated partition integrals. Only the analytic unit-torus physical case is
untruncated. No dense equilibrium or phase-coexistence claim is made.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from scripts import continuum_simulated_tempering as tempering
import test_single_walker_continuum_equilibrium as mixed_reference
import test_single_walker_exact_periodic_palm as exact_reference
import test_single_walker_unequal_activity_equilibrium as activity_reference


def _enable_gibbs_with_independent_protocol_checks(monkeypatch):
    actual = tempering.run_tempering_chunk
    checked_chunks = []

    def run(*args, **kwargs):
        steps = int(args[6])
        diameter, radius, physical = args[9], args[10], args[18]
        neighbors, edge_table = args[22], args[23]
        node_count = len(diameter)
        source = np.full(steps, -999, dtype=np.int64)
        final = np.full(steps, -999, dtype=np.int64)
        counters = np.zeros(2, dtype=np.int64)
        result = actual(*args, **kwargs, gibbs_label=True,
                        neighbor_source_labels=source,
                        weight_update_labels=final, gibbs_counters=counters)
        trace, _, edge_counters = result[:3]
        observed = trace[:, 1].astype(int)
        assert (source >= 0).all() and (source < node_count).all()
        assert (final >= 0).all() and (final < node_count).all()
        assert observed[0] == args[3]
        assert np.array_equal(observed[1:], final[:-1])
        assert final[-1] == result[-2]
        assert counters[0] == steps
        assert counters[1] == np.count_nonzero(source != observed)
        assert counters[1] > steps / 10
        # A label-only Gibbs update preserves the recorded current host/center.
        # Validate selected support directly from pre-Gibbs physical features.
        for node in range(node_count):
            selected = source == node
            if physical[node]:
                assert (trace[selected, 4] == 0).all()
            occupied = selected & (trace[:, 2] > 0)
            assert (trace[occupied, 7] >= radius[node]).all()
        # The subsequent neighbor/Palm step is the only source->final change.
        # Next full-clock labels alone are insufficient to reconstruct it.
        reconstructed = np.zeros_like(edge_counters[:, 1])
        for old, new in zip(source, final):
            if old != new:
                locations = np.flatnonzero(neighbors[old] == new)
                assert len(locations) == 1
                reconstructed[edge_table[old, locations[0]]] += 1
        assert np.array_equal(reconstructed, edge_counters[:, 1])
        checked_chunks.append((steps, counters.copy()))
        return result

    monkeypatch.setattr(tempering, "run_tempering_chunk", run)
    return checked_chunks


def test_actual_gibbs_unequal_activity_matches_independent_joint_conditional_work_laws(monkeypatch):
    chunks = _enable_gibbs_with_independent_protocol_checks(monkeypatch)
    activity_reference.test_unequal_activity_hard_path_matches_joint_conditional_and_work_laws()
    assert [steps for steps, _ in chunks] == [5000, 160000]


def test_actual_gibbs_unit_torus_matches_untruncated_palm_fano_singleton_work(monkeypatch):
    chunks = _enable_gibbs_with_independent_protocol_checks(monkeypatch)
    exact_reference.test_exact_periodic_palm_number_fano_singleton_and_cycle_work_laws()
    assert [steps for steps, _ in chunks] == [2000, 80000]


def test_actual_gibbs_soft_order_number_targets_match_independent_partition_integrals(monkeypatch):
    chunks = _enable_gibbs_with_independent_protocol_checks(monkeypatch)
    mixed_reference.test_actual_single_host_matches_capped_joint_and_conditional_partition_laws()
    assert [steps for steps, _ in chunks] == [5000, 150000]
