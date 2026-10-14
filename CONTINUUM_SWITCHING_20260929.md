# Unrestricted continuum switching calculation

## Status

The unrestricted sampler and its exact physical target are implemented.
Independent partition-integral and detailed-balance controls pass. Dense
pilots have created ordered hard-sphere configurations from an all-fluid
start, but the independent initializations have not yet converged to the same
phase weights. No trapped pilot is treated as an equilibrium coexistence
measurement, and none supplies a new manuscript coefficient.

The main paper's finite-volume identities, asymptotic theorem, and lattice
benchmark do not depend on these exploratory continuum histories. The
Supplement documents the sampling construction and its kernel controls;
the dense extension requires further equilibrium evidence.

## Physical ensemble

The cube has volume `32 × 0.96405 σ³`. Particle coordinates and particle
number are unrestricted, and the cavity center is mobile. Physical replicas
have exact hard-sphere interactions, no structural umbrella, and either zero
or particle-sized center exclusion. Unbiased hard replicas at other chemical
fields probe the finite-volume phase balance. Auxiliary replicas soften the
pair interaction and introduce a removable structural umbrella. They do not
supply phase weights to a physical replica.

The cap of 80 exceeds the rigorous physical packing bound of 59, so it does
not exclude any physical hard-sphere sector. Soft auxiliaries are explicitly
capped product ensembles. All target and proposal parameters are frozen
before production observations; adaptive auxiliary activity selection is
confined to discarded equilibration.

## Exact moves and controls

Birth, death, displacement, cavity-center, and full-configuration exchange
moves retain the appropriate grand-canonical target ratios. Two normalized
configuration-dependent birth proposals retain a full-support uniform
component. Their death acceptance uses the reverse insertion density of the
remaining host, including a fresh FCC registry fit where applicable.

At equal activity, an additional marked move transfers a uniformly tagged
particle from the unconstrained host to the cavity-center role, restores the
previous cavity center as a particle, and exchanges the hosts. Its acceptance
is `min(1, N/(M+1))`. Campbell's marked reference-measure identity includes the
particle factorials; no additional volume, activity, or squared count ratio
is present. Independent exhaustive graph controls and actual continuum
partition quadratures verify this move and the joint product law.

The nearest-FCC helper is an optimization, not a change to the proposal.
Canonical mapping, periodic and half-grid ties, complete registry scores,
occupancies, and ignored-particle refits match an independent full scan.
The original scan remains the fallback for noncanonical references. Actual
small-system equilibrium controls exercise both the fallback and optimized
32-site reference.

## What the pilots establish

- The compact all-fluid initialization is independently reconstructed from
  its executed source. It starts with disordered hosts rather than a seeded
  crystal bank.
- The all-fluid compact pilot creates crystals at unbiased hard nodes with
  chemical fields approximately 16.21--17.00. Independently reconstructed
  global bond order is approximately 0.48 in their high-Bragg snapshots.
  The auxiliary reciprocal-amplitude bias alone does not explain this signal.
- The marked move bypasses the need to observe an empty particle-sized ball
  at a uniformly drawn location. Direct cavity samples can therefore be
  checked against deletion reconstructions even when direct insertion events
  are absent.
- The exact target ratio is not evidence of dense equilibration. Crystal
  starts retain a larger ordered inventory than fluid starts, and the
  observed activity-front location remains initialization dependent.
- Frequent endpoint switches may redistribute existing phase representatives
  without creating or melting them. Global inventories, reservoir passages,
  and independent starts expose this failure mode.
- Marginal deletion identities can pass inside a trapped history. Zero
  direct insertion events leave a work estimate unresolved; they do not
  verify a staged work. A boundary phase-frequency estimate has not measured
  the uncertainty in an unsampled phase.
- A nominal endpoint covariance significance can also be misleading when
  rare joint number sectors are absent. Diagnostics retain the nominal
  statistic but report joint-cell exposure and a sparse-support warning.

## Acceptance criteria for an equilibrium result

Use matching physical targets, a frozen auxiliary graph and proposal, and
separate fresh fluid and crystal initializations. Establish recurrent phase
creation and melting, exchange with the number reservoir, and particle-number
and occupancy transitions. Compare the independent histories without pooling
away their disagreement. Assess production quarters and increasing block
lengths, rotationally invariant structure, the full cavity number law and
deletion observables, and equal-time product probabilities. Report sufficient
support for the moments and rare-event estimates actually used.

After converged activity profiles locate a mixed finite-volume field, freeze
the confirmatory field and basin rules before collecting new histories.
A single small box can test the exact finite-volume cavity identities and
phase bias. It cannot establish the thermodynamic volume scaling by itself.
The finite Fano response includes a tagged-particle `−1` background, so a
positive between-phase contribution need not make its total response
positive at this volume.

## Current route and resource policy

Lowering the penetrable pair penalty during the structural bridge from 8 to
2 permits substantially more coordinate and occupancy rearrangement. A
matched pair at chemical field 14.8 nevertheless gave mean numbers 28.39
and 31.98 and solid fractions 0.022 and 1 for fluid and crystal starts,
respectively. Those observed means describe incompatible pilot histories,
not equilibrium estimates. The early soft bridge had similar profiles, but
the late hardening and umbrella-removal stages remained phase dependent.

The current schedule therefore adds a second low-Bragg hardening route
alongside the original high-Bragg crystal-creation route. Both join the
same unbiased hard endpoint; no physical target or prescribed phase odds
changes. A separate optional trace records actual pair-overlap counts
without changing random draws or moves. A fresh-fluid pilot calibrates
auxiliary activities during discarded burn-in; a distinct crystal start
then uses the identical frozen activities and displacement proposals.

The paired low-Bragg-return pilots isolate a number-relaxation gate.
Although the crystal-start low-Bragg hardening route reaches disordered
configurations with almost no overlaps, it retains about 30.5 particles
instead of the fluid-start route's 28.6. Removing the order stiffness
from 40 to 25 restores crystalline order and about 31.8 particles, and
that exchange edge has only about 2.7% acceptance. The physical base remains
initialization dependent. This motivates a removable auxiliary number
umbrella, not a change to any physical phase probability. Its kernel has
now passed independent capped ideal-gas and continuum partition controls,
including full product laws, cache, support, and marked-transfer checks.
The new paired pilot additionally matches displacement proposals on every
activity probe: a driver omission had left them at defaults in earlier pairs,
even though their physical targets and core proposals matched.

The completed 163-replica number-umbrella pair comprises
`continuum_number_vacancy_fluid_pilot` and
`continuum_number_vacancy_crystal_pilot`, each with 20,000 discarded burn-in
sweeps and 20,000 production sweeps. Direct comparison of the archived
arrays verifies identical physical targets, auxiliary graph parameters,
and displacement scales at **every** node, including all 29 activity probes.
The removable number bias resolves the targeted low-order number trap:
both histories have about 29 particles on that branch, and the former
order-stiffness-40-to-25 bottleneck accepts about 97% of exchanges.

It does not yet resolve the physical phase weights. The unconstrained
means are 28.833(28) and 31.9730(46), and the operational solid fractions
are 0.0895(67) and 0.99950(51), respectively. Parentheses here summarize
within-history common-time-block bootstrap variation (500 draws, 1,000-sweep
blocks); they do not account for the unvisited equilibrium populations.
The ordered hard-node inventory rises through quarter means
4.124, 4.298, 4.452, 4.906 in the fluid history, but falls only from
11.784 to 11.494 in the crystal history. Neither completes an ideal-reservoir
round trip, and the crystal history has no observed lowest-hard-activity
transport leg to either physical endpoint. Along number-bias removal,
the crystal history restores its dense ordered population near strengths
1 and 0.7, whereas the fluid history remains predominantly disordered.
Those exchanges have nonzero, substantial acceptance; this is continued
global relaxation, not an invalid target law or a disconnected edge.
The two records are retained separately as a failed independent-start
equilibration check, never pooled into a coexistence estimate.

## Bounded frozen continuations

Each fresh-start lineage was continued through two additional 25,000-sweep
segments, alternating fluid and crystal lineages. Targets and every proposal
scale were unchanged, and no further adaptation or discarded burn-in was
introduced. Each lineage therefore contains 70,000 observed production
sweeps after its initial 20,000-sweep burn-in. New random seeds do not make
these restart segments independent replicates.

In the final fluid-start segment the mean number is 28.782(40), with solid
fraction 0.2284(66); the crystal-start segment gives 31.9628(46) and
0.99760(81). These are conditional within-history standard errors from
1,000-sweep blocks, not equilibrium uncertainty. The final fluid ordered-bank
quarter means are 5.510, 5.552, 5.581 and 5.544, whereas the crystal values
are 10.589, 10.549, 10.726 and 10.557. Locally steadier traces have therefore
not erased the substantial initialization dependence.

Transport bookkeeping was reconstructed across restart boundaries, rather
than adding segment-local counts. Exact parent hashes, source semantics,
frozen target/proposal arrays and all checkpoint state slots are verified
before composing walker labels. Over all saved production frames, the fluid
lineage completes 11 lowest-hard-activity/base and 9 lowest-hard-activity/cavity
round trips; the crystal lineage completes none of either. Ideal/base round
trips occur in both lineages (14 and 5), yet neither completes a
lowest-to-highest-hard-activity round trip. Saved-frame counts are lower
bounds because excursions between observations are unavailable. They are
transport diagnostics, not equilibrium certificates. The residual
phase-inventory relaxation problem survives successful local moves and
visible endpoint flips; no dense phase weight is adopted from these runs.

Only one numerical campaign runs at a time, with one numerical thread,
bounded coordinate snapshots, sparse activity-probe coordinates, and atomic
checkpoints. Executed source bytes and raw hashes are preserved. Earlier
negative pilots and their original metadata remain available; an analysis
replay does not rewrite an execution record. This work has not been pushed
to GitHub or submitted to PRE.

## Separate regenerative single-host implementation

A distinct simulated-tempering engine now transports one host through the
auxiliary labels, with degree-corrected label moves and the exact tagged
base/cavity jump. The designated ideal label independently refreshes particle
number, all host coordinates and the cavity center. Its analysis uses every
clock-time dwell, including rejections; complete-cycle rewards are summed
before dividing by complete-cycle dwell. It does not reuse replica-product
covariance checks or average individual cycle fractions.

An independent five-label capped continuum control passed explicit
nine-dimensional partition checks, including nonzero order and number
umbrellas, irregular label degrees and nonuniform insertion proposals. The
archived 150,000-step control yielded 38,504 complete cycles, including
31,914 cycles with no physical-endpoint dwell. All were retained. Its
weight-corrected cavity work was approximately 0.204 with a whole-cycle
standard error of 0.019, consistent with the capped quadrature value 0.201.
Independent coordinate replay checked all 6,523 physical snapshots, including
center clearance and delete-one labels. The deliberately small particle cap
invalidates ordinary untruncated Palm formulas, which the analyzer therefore
does not report. This is implementation validation, not fluid–solid evidence.
Further actual-engine controls passed unequal-activity partition checks and
an exactly solvable periodic-cell Palm/Fano test whose physical particle cap
is inactive. The latter checks the unrestricted identities directly, without
mistaking an artificially truncated control for an unrestricted ensemble;
its tiny geometry is not a coexistence or contact-derivative experiment.

One bounded dense exploratory run then used 200,000 calibration steps and
200,000 fixed-weight production steps. External timing recorded 32.25 seconds
wall time, 401,702,912 bytes of peak resident memory (about 383 MiB), and a
308,495,368-byte memory footprint (about 294 MiB). Although calibration visited all 163
labels, production contained **no ideal heatbath and no complete regeneration
cycle**. Consequently all complete-cycle thermodynamic estimates and errors
are left undefined. The 5,025 base and 14,780 cavity clock visits are not
substituted for an equilibrium result. All 45 base and 158 cavity coordinate
snapshots were ordered under the independent operational Q6 thresholds
0.30, 0.35 and 0.40; no physical melting was observed. The strict coordinate
replay checked 1,146 physical snapshots in total. The single-host engine has
passed its small-system controls, but this first dense pilot has not solved
the return-to-reservoir sampling problem.

The small capped control and its exact, hash-verified source dependencies are
retained in the lean release. The producer is explicitly recorded as the
independent test harness. A default-test-discovery issue was also corrected:
`pytest.ini` restricts live collection to `tests/`, so immutable historical
producer snapshots whose names begin with `test_` are preserved as evidence
without accidentally running them as additional current tests. Both release
profiles carry this configuration and its regression check.

## Native hard-activity single-host pilot

A second bounded pilot used a native 42-label graph: one ideal reservoir,
an unbiased hard-sphere activity path from -2 to 17, and the matched cavity
at activity 14.8. There were no order or number umbrellas. Initial fluid
equation-of-state weights were efficiency guesses, not imposed phase odds.
Independent actual-engine tests of unequal-activity targets and exact
small-cell Palm identities passed before this run.

The run used 200,000 calibration and 200,000 frozen production steps.
External timing measured 33.10 seconds wall time, 365,805,568 bytes peak
resident memory (about 349 MiB), and a 268,600,280-byte peak footprint.
Calibration visited every label and refreshed the ideal reservoir 4,848
times. Frozen production visited 25 labels but again had **no ideal refresh
and no complete regeneration cycle**. All complete-cycle estimates, including
the activity-resolved conditional quantities, remain undefined. Values at
different chemical activities are not pooled.

The 266 base and 629 cavity dwells were intermittent: neither endpoint was
visited in the middle two production quarters. Base mean number changed
from approximately 25.10 in the first quarter to 27 in the last, while the
cavity means changed from 24.09 to 26. These are descriptive trajectory
averages, not equilibrium estimates. Independent geometry replay passed all
2,000 saved snapshots. The three base and seven cavity snapshots were below
each operational Q6 ordering threshold (0.30, 0.35, 0.40), in contrast to the
ordered endpoint snapshots of the first single-host pilot. Across all 2,000
native-pilot snapshots, Q6 remained between approximately 0.046 and 0.286;
there was no recorded ordering event even at the lowest threshold. This does not
establish a finite-cell phase crossover: neither history regenerated during
production, and no dense continuum phase probability is adopted from either.
The hard-activity edges have common configuration support. A direct
conditional-label calculation at the observed number 27 assigns approximately
0.9966 of the frozen label probability to the six upper hard-activity labels
and only 2.9e-70 to the ideal reservoir. This identifies accumulated
label-weight imbalance as a concrete efficiency problem; merely replacing
neighbor label proposals with a Gibbs draw would not correct those weights.

## Conditional-label and calibration serialization controls

The optional Gibbs-label update passed three independent actual-continuum
target-law controls before being used in a short CLI serialization check.
Its post-Gibbs/pre-neighbor labels are recorded separately, so compound-clock
label changes are not mistaken for individual neighbor acceptances. The
staged discarded calibration likewise distinguishes observed pre-hop labels
from the actual post-update labels whose weights were decremented.

The serialization check used only 500 discarded and 1,000 frozen steps. It
recorded exactly 500 calibration observations and 500 weight updates, followed
by 1,000 Gibbs draws with 875 label changes. Independent parsing reproduced
every cumulative stage flag and count, including the expected exhausted
budget without flatness; all 100 saved coordinate snapshots passed geometric
replay. No completed production cycle or endpoint dwell occurred, so no
thermodynamic estimate is reported. This is a protocol/serialization check,
not a third dense scientific campaign. It depends on the archived native
graph and therefore remains in the full audit bundle; the lean release keeps
only the separately self-contained capped control and its exact sources.

The coordinated scientific validation gate after these additions passed
796 tests with no skips. The test runner took 118.18 seconds; external timing
measured 120.24 seconds wall time, 1,225,457,664 bytes peak resident memory
(about 1.14 GiB), and no swaps. All six numerical thread limits were one.
The optional control-export variable was absent, and the retained raw control
digest and all 31 approved manuscript/figure/PDF digests were unchanged.
This gate validates code and exact small-system controls; it does not supply
the missing dense equilibrium sampling evidence.

## Native radius-ring pilot with staged calibration

The next bounded exploratory run used the explicitly constructed 56-label
graph at base activity 16, with 16 unbiased cavity-radius stages, the tagged
base/cavity edge, optional conditional-label Gibbs updates, and a deep ideal
reservoir. Independent actual-engine radius-ring controls passed before
launch. The five discarded calibration stages met their specified
label-efficiency checks after 705,000 steps of the 800,000-step budget.
Frozen production then ran for 200,000 steps. External timing measured
83.78 seconds wall time and 568,262,656 bytes peak resident memory (about
542 MiB), with no swaps. Strict replay independently verified target arrays,
startup sources, full-clock and compound-label bookkeeping, calibration
counts and every recorded stage outcome.

Production contained 511 ideal refreshes, distributed across its four
quarters as 207, 202, 7 and 95. This produced 510 complete regeneration
cycles, but **every complete cycle had zero base and full-cavity dwell**.
All 154 base and 167 cavity observations occurred in the discarded leading
partial cycle. The leading and trailing partial cycles account for 24,896
of 200,000 clock steps (12.448%). The longest complete cycle spans 66,462
steps, approximately 38% of the complete-cycle clock total, whereas the
median cycle lasts only one step. A large count of short reservoir cycles
therefore does not establish useful dense-target coverage. All endpoint
thermodynamic estimates and uncertainties remain undefined.

The host did not become progressively denser: maximum particle number in
the four production quarters was 28, 25, 25 and 25. Independent coordinate
replay checked all 1,995 physical snapshots. Among the 1,408 snapshots with
at least 20 particles, Q6 ranged from approximately 0.063 to 0.278; none
passed the lowest operational ordering threshold of 0.30. No dense ordered
phase creation is recorded. These observations distinguish this failure
from the earlier crystal-trapped run: reservoir regeneration occurred, but
the complete cycles did not return to the dense endpoints. Passing the
discarded-calibration efficiency checks did not provide convergence or
adequate physical sampling, and no dense phase weight is adopted.

## Native compact bridge implementation

The next implementation step is a self-contained compact graph that joins
soft-pair, order and number auxiliary paths to the same unbiased hard base
and cavity targets. One number path favors the specified FCC reference
count; the other favors the design value volume/1.06448. Both restraints
are removed before reaching the physical endpoints. The designated ideal
reservoir has a dense capped-Poisson law, while optional physical radius
stages provide a separate route for changing host number. Neither the
auxiliary number targets nor the initial efficiency guesses prescribe
physical phase probabilities.

Independent reconstruction checks reproduce all target arrays and edges,
including at a non-binary volume and non-rounded activity, and verify the
declared soft-CS/Gaussian-Bragg efficiency-model constants. They also retain
the original hard-activity/radius defaults. This is a checked graph
construction, not new dense sampling evidence. New analysis executions
preserve the analyzer's exact startup bytes separately from raw
simulation sources; subsequent source edits cannot silently relabel an
earlier execution hash.

The coordinated complete suite after the compact-graph and independent
reconstruction additions passed 857 tests with no skips. External timing
measured 34.36 seconds wall time and 827,408,384 bytes peak resident memory
(about 789 MiB), with all six numerical thread limits set to one and no
swaps. The native graph was independently bound to its recorded scalar
construction and exact source dependencies before this test run. This gate
does not replace a successful dense sampling experiment.

One bounded compact pilot subsequently used 800,000 discarded calibration
steps and 200,000 frozen steps. It completed in 74.80 seconds with about
385 MiB peak resident memory and no swaps. Calibration stages 0 and 1 passed
at clocks 200,000 and 685,000; stage 2 exhausted the budget at gain 0.075.
Its label flatness passed, but physical number/order half-mean stability
did not. For example, the base half means changed from approximately
30.71 to 29.35 particles and from 0.706 to 0.489 in the Bragg indicator.
Those adaptive observations are discarded, not thermodynamic estimates.

Frozen production had 5,438 ideal refreshes and 5,437 complete cycles, but
**no physical-label dwell anywhere in the history**. All complete-cycle
physical quantities and uncertainties remain undefined. Leading/trailing
partial cycles exclude 2,561 and 20 clocks; the longest complete cycle
lasts 8,214 clocks, about 4.2% of the complete-clock total. This failure is
therefore not explained by one dominant censored excursion. All 2,000 saved
coordinate frames are auxiliary, so there is no physical Q6 sample to
classify. The largest visited soft-pair penalty was approximately 11.58,
reached in only one low-order observation. The ordered route could reach
mean Bragg indicator approximately 0.72 at penalty 7.17, but retained about
five overlapping host pairs there. That is auxiliary ordering, not an
overlap-free physical crystal. Strict source/target/calibration analysis
passed and preserved its own executed source bytes. No dense phase weight
or physical ordering probability is adopted from this pilot.

A subsequent pilot increased only the discarded-calibration ceiling to
2,400,000 clocks before 200,000 frozen clocks. It took 295.33 seconds, with
1,668,661,248 bytes peak resident memory (about 1.55 GiB), a reported peak
footprint of 221,595,352 bytes, and no swaps. Calibration again exhausted
its budget without satisfying all stopping criteria, although it reached
the final gain of 0.01875. Frozen production visited 88 labels and made
18,272 physical-label observations, including 1,223 at the base and 886 at
the full cavity. However, it had **zero ideal refreshes and zero complete
regeneration cycles**. Every complete-cycle thermodynamic estimate and error
therefore remains undefined; the physical dwells are not substituted for
equilibrium estimates.

Independent coordinate replay verified 177 physical snapshots. Their Q6
values were approximately 0.419--0.513, with no excess site occupancy;
base counts were 31--32 and full-cavity counts 30--31. Thus the unrestricted
engine did produce physical crystalline configurations, rather than merely
large auxiliary Fourier amplitudes. Physical visits ceased around frozen
clock 67,500, however, and this does not demonstrate equilibrium switching
or its phase probabilities. Later overlap-free, low-Bragg hosts with 29
particles had physical support, but a representative conditional base-label
probability under the frozen efficiency weights was only about 1.5e-7.
That probability describes the extended-ensemble label update, not a
thermodynamic phase weight. These diagnostics motivate a proposal-efficiency
improvement while retaining the same physical target laws.

## Residual-vacancy proposal validation

The optional residual-vacancy relocation proposal preserves particle number,
uses a uniformly selected host tag and the same residual configuration in
its forward and reverse Metropolis--Hastings densities. It does not change
the physical or auxiliary target laws. Its separate production counters and
scope are checked by the strict loader; the default-off path retains the
legacy protocol.

The subsequent measured full collection passed **913 tests** in 123.05
seconds external wall time, with about 1.235 GiB peak resident memory and no
swaps, under six one-thread numerical limits. A newly added independent
actual-engine test then passed separately: the relocation-enabled,
Gibbs-enabled single-walker wrapper reproduced the capped five-node joint
and conditional reference laws, with its counters, current features,
labels, support, centers and caches checked. That separate run took 12.11
seconds and about 473 MiB peak resident memory, with no swaps. Thus 914
distinct controls passed across the 913-test full collection and the
additional one-test gate; a full 914-test collection has not been reported.
These target-law and bookkeeping controls do not establish dense
equilibration in a subsequent pilot.

The relocation-enabled pilot subsequently exhausted its 1,200,000-clock
discarded-calibration budget at stage 3, then produced 400,000 frozen
observations. External timing measured 112.58 seconds with about 525 MiB
peak resident memory and no swaps. Relocation accepted 3,905 of 209,138
eligible attempts. Production had 11,490 ideal refreshes, but **all 11,489
complete regeneration cycles again had zero endpoint rewards**. All 3,264
physical-label observations (31 base and 44 full-cavity observations) lay
in the leading partial cycle before the first refresh at frozen clock
45,255. No post-refresh configuration had zero host overlaps, so those
later auxiliary states did not satisfy physical hard-pair support.

Independent coordinate replay found 28 physical frames with Q6 between
approximately 0.105 and 0.207, including one base frame and no full-cavity
frame. The pilot therefore provides neither complete-cycle thermodynamic
estimates nor dense ordered-phase evidence. Relocation improved neither of
those required sampling outcomes in this history. The next authorized
implementation changes only an optional auxiliary number-restraint
strength during ordered hardening; it leaves every physical target fixed.
No run or equilibrium result for that variant is asserted here.

The H-variant construction and provenance gate subsequently passed 302
tests with no skips in 1.88 seconds of pytest time (2.19 seconds external
wall time). This targeted, non-simulation collection includes the original
graph's exact default-byte oracle, separate declarative reconstruction,
independent factorial number-sector laws and the unchanged physical/low-order
targets. The timing wrapper could not obtain peak-memory statistics because
the sandbox denied its system-clock query; no peak RSS is claimed for this
gate. The tests establish the recorded target construction and backward
compatibility, not successful equilibration with reduced H.

The subsequent full collection passed 982 tests in 38.18 seconds external
wall time, with approximately 831 MiB peak resident memory and no swaps.
Three new pure local-relaxation diagnostic controls passed separately after
that collection; they were not included in the measured 982-test run.
The H=1 graph was then independently bound to its saved scalar settings,
target arrays and six verified source/input dependencies before sampling.

The bounded H=1 pilot used a fresh ideal initialization and new model-based
efficiency guesses, with a 3,000,000-clock discarded-calibration ceiling
followed by 400,000 frozen clocks. Calibration exhausted its budget.
Frozen production had **zero physical-label observations** and 59 ideal
refreshes, all ending by frozen clock 3,144. Complete-cycle physical
estimates remain undefined. Roughly 87% of the recorded hosts had zero
pair overlaps; the run reached a gamma=25 auxiliary host with 32 particles,
Q6 approximately 0.477 and no excess reference-site assignments. These
descriptive auxiliary configurations are not observations at an unbiased
physical endpoint. A representative conditional base-label probability
was still only about 5e-7 under the frozen efficiency weights.

The next authorized implementation therefore targets sampling **labels**
with nonuniform calibration quotas. It changes discarded adaptation, not
physical phase odds or any frozen conditional target law. No quota-based
pilot or successful equilibrium measurement is asserted here.

The quota schema and independent controls subsequently passed 262 targeted
tests in 1.51 seconds. A real serialization control with 500 discarded and
1,000 frozen clocks was replayed under the strict loader: its source/target
bindings, quota allocations, actual update counts and both saved
gauge-invariant checkpoint increments passed. Its largest floating-point
replay envelope was approximately 1.8e-9, with no possible intervening
clipping. The deliberately short control had no production regeneration
cycle or physical endpoint dwell, so all thermodynamic estimates remain
undefined. Its new versioned derivative preserves the executed analyzer
bytes separately from the immutable simulation sources.

The next complete scientific collection passed **1,075 tests with zero
skips**, taking 66.13 seconds in pytest and 67.69 seconds external wall
time. The timing wrapper subsequently failed to query `kern.clockrate`
under the sandbox, after pytest had completed successfully; therefore its
exit status does not indicate a test failure, and peak resident memory and
swap statistics are unavailable for this run. The earlier targeted
collections and separately executed controls are not added to this full
collection's count. All 31 approved publication artifacts remained
byte-identical at this gate.

## Endpoint-quota pilot and discarded normalizer correction

The authorized endpoint-prioritized pilot used 3,000,000 discarded clocks
and 400,000 frozen clocks. It took 249.80 seconds, with about 424 MiB peak
resident memory and zero swaps measured by a single-child resource wrapper.
Its first calibration stage passed after 1,240,000 clocks; stage 1 exhausted
the remaining budget at gain 0.15. Frozen production visited only the base
(294,562 observations) and full cavity (105,438 observations), with particle
counts locked near 27 and 26, respectively. There were **no ideal refreshes
or complete regeneration cycles**, so all complete-cycle thermodynamic
estimates and uncertainties remain undefined.

Independent physical-coordinate replay gave mean Q6 around 0.15, consistent
with disordered configurations in this diagnostic history. Frequent tagged
endpoint moves did not demonstrate phase switching or ordinary number
equilibration: only one ordinary birth/death was accepted in 3.2 million
production attempts. The nearest smaller-radius label had hard support,
but its frozen acceptance from the full cavity was approximately 2e-54.
The weight gap was roughly 123 log units, while the quota ratio accounted
for only about 4.83. Large compensating excursions during discarded
calibration allowed high cumulative weighted flatness without stable
normalizer estimates; intermediate-radius half-mean tests still failed.

An optional discarded-only projection is consequently being validated.
For the unbiased nested cavity path it constrains the quota-adjusted log
weights to obey exact monotonicity and finite matched-cavity span bounds.
These are necessary properties of the desired normalization root, not
assigned phase probabilities or guessed measured free energies. Frozen
conditional targets remain unchanged. Every correction records its full
PRE weights and POST path values; independent replay checks the count-driven
PRE update and convex optimality of the correction. The initial unscaled
solver failed a preserved 17-node large-gap stress fixture and correctly
aborted. A common positive rescaling preserves the mathematical projection
and is undergoing independent validation. No projected dense pilot or
equilibrium switching claim is made here.

After the numerical rescaling, the coordinated independent validation
collection passed 371 tests in 1.81 seconds. It includes strict pre/post
update replay, simultaneous-constraint KKT checks by a method different
from the producer optimizer, analytic projection oracles and the severe
17-node gap. A subsequently added forced solver-non-success test passed
separately in 0.35 seconds, confirming that failure cannot silently become
an accepted projection. These are implementation checks, not a projected
dense sampling result; actual serialization and integrated validation
remain separate gates.

The real projected serialization control then completed 1,000 discarded
and 1,000 frozen clocks in 8.41 seconds, using about 356 MiB peak resident
memory and zero swaps. Strict replay verified both pre/post checkpoint
updates, exact source/input bindings and the independent joint-projection
KKT conditions; the maximum stationarity residuals were approximately
5.7e-14 and 4.4e-16. Final gauge-fixed frozen weights matched the saved
calibration endpoint. The 1,000 frozen observations were all at the ideal
label, producing 999 complete cycles with zero physical rewards. All
physical estimates and errors correctly remained undefined. This validates
the serialized correction protocol, not dense equilibrium sampling.
