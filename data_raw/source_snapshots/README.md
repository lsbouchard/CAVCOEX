# Sources referenced by frozen simulation and analysis records

`solid_cavity_mc_40530bf7c3d5e719.py` is the exact source whose SHA-256 is
`40530bf7c3d5e719c045dd05dd28cf1f6b7d8043d0f72af0c5f6f2263c0da77c`.
That digest is recorded in the raw baseline fcc block data and in the
independent-replica metadata. The file was recovered without modification
from the original CAVCOEX commit `f471998` during the September 2026 audit.

The current `scripts/solid_cavity_mc.py` includes later presentation changes.
Those changes do not alter the frozen trajectories or their original source
hash. This snapshot preserves the recorded source; it is not the entry point
for current analysis. To reconstruct the old execution layout, copy this file
as `scripts/solid_cavity_mc.py` in a separate checkout with its dependencies.

`cavity_mc_94c52a06a2aef193.py` similarly preserves the exact source with
SHA-256 `94c52a06a2aef1930d989879848bcbfd7e14f9662dec5143b835911ac1b4e2d5`,
recorded in the raw fluid-campaign blocks and as a dependency of the solid
campaign. It was recovered from the same original public commit. The
September audit corrects an edge case in its stationarity diagnostic; that
change does not relabel the historical raw data. Both snapshot files can be
restored under their original names in a separate checkout to reconstruct
the recorded source layout.

`lattice_gas_additional_replicas_91ecb780004d2f42.py` preserves the old driver
with SHA-256 `91ecb780004d2f428c7406da5600f5d62bf3b7f59ced02fc87150eec4fec0bf5`,
as recorded in the additional-replica metadata. The current driver fixes
custom output paths and sidecar-file handling; the sampling kernel and frozen
trajectories are unchanged.

`lattice_gas_phase_switch_152c5aff92e8618d.py` preserves the baseline sampler
source with SHA-256 `152c5aff92e8618dd9c371aa302687e36ade82b2c79195437e5a60a1fcbc9b67`.
The current version corrects a comment that incorrectly called finite-lattice
ties measure-zero; its sampling kernel is unchanged.

`lattice_leave_lmax_prediction_36fe01d71f3ae566.py` preserves the September
analysis source with SHA-256
`36fe01d71f3ae5662e29620f3331a28efb97d56cbfe2a824757a2f04b0b4241b` before
the figure-only change that displays residual covariance modes. The
statistical analysis record retains that execution hash; the subsequent
figure rendering has its own provenance record.

In the full audit reproducibility bundle,
`continuum_replica_exchange_<digest-prefix>.py` files preserve source bytes
captured at startup for the corresponding later continuum diagnostic runs.
Each run's JSON sidecar records the snapshot path and complete SHA-256. The
release tests verify those links and the original raw-archive digests. A
snapshot is the source for that historical execution, not a claim that the
exploratory dense history reached equilibrium.
The lean code checkout intentionally omits the exploratory raw histories and
their continuum snapshots. The narrow exception is the small independent
capped single-host control in `data_raw/controls/`: its raw/sidecar and only
the exact hash-verified source snapshots named by that sidecar are retained
in both profiles. Those include the independent test harness that produced
the control, not just the simulation functions. The baseline publication
snapshots above also remain included in both profiles.

The four earliest continuum pilots predate startup source capture; exact
executed source identity cannot be recovered from their end-of-run hashes.
They are identified explicitly in `REPRODUCIBILITY.md`. No approximate
reconstruction or retrospective source-hash replacement is supplied.
